"""Smart retry: real exponential-backoff wrapper for flaky operations.

Claude Code CLI-style resilience: run a subprocess, and on a non-zero
exit, wait ``base_delay * 2**attempt`` (+ up to 20% random jitter, capped
at ``max_delay``) and retry. Stops early on success; fails fast on
command-not-found (exit 127); gives up after ``max_attempts`` or a total
time budget of 10 minutes.

Engine::

    result = run_with_retry(["pytest", "-q"], max_attempts=5,
                            base_delay=1.0, max_delay=60.0)
    result["success"]      # True/False
    result["attempts"]     # how many attempts were actually used
    result["returncode"]   # final exit code
    result["output"]       # merged stdout+stderr of the final attempt
    result["attempts_log"] # one dict per attempt: number, started_at
                           # (ISO-8601), duration_s, returncode,
                           # waited_before_s (backoff that preceded it)

LLM tools (see :func:`register`)::

    RetryRun   — retry a command without a shell (argv list or string)
    RetryShell — retry a complex command line through the shell
                 (shell=True, so pipes/redirects/globs work)

Only stdlib + ``.tools`` are imported (``Tool`` for the registry).
Backoff schedule is 0-indexed: after failed attempt n (1-based), the
wait is ``base_delay * 2**(n-1)``, i.e. the first retry waits
``base_delay``.
"""

from __future__ import annotations

import datetime
import random
import shlex
import subprocess
import time
from typing import Any

from .tools import RISK_CONFIRM, Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOTAL_TIME_CAP = 600.0  # hard cap: 10 minutes across all attempts
JITTER_MAX_PCT = 0.20   # up to +20% random jitter on each backoff wait
NOT_FOUND_RC = 127      # command-not-found: never retried, fail fast

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0
DEFAULT_TIMEOUT = 120.0  # per-attempt process timeout (seconds)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def backoff_delay(attempt_index: int, base_delay: float,
                  max_delay: float) -> float:
    """Seconds to wait before retry number ``attempt_index`` (0-based).

    ``base_delay * 2**attempt_index``, plus 0-20% random jitter, capped
    at ``max_delay``.
    """
    try:
        delay = base_delay * (2.0 ** attempt_index)
    except OverflowError:
        # large max_attempts + tiny max_delay reaches 2**1024 (OverflowError)
        return max_delay
    delay *= 1.0 + random.uniform(0.0, JITTER_MAX_PCT)
    return min(delay, max_delay)


def _normalize_argv(command: Any) -> tuple[list[str] | str, bool]:
    """Return (argv, ok). ``command`` may be a list or a string."""
    if isinstance(command, (list, tuple)):
        argv = [str(c) for c in command]
        if not argv:
            return [], False
        return argv, True
    if isinstance(command, str) and command.strip():
        return command, True
    return [], False


def _run_once(command: Any, shell: bool,
              timeout: float) -> tuple[int, str, str, str]:
    """Run one attempt. Returns (returncode, stdout, stderr, note).

    A missing executable is reported as returncode 127 with a note —
    the caller treats that as command-not-found and never retries it.
    """
    if shell:
        popen_args: Any = command
        popen_kwargs: dict = {"shell": True}
    else:
        argv, ok = _normalize_argv(command)
        if not ok:
            return 2, "", "invalid command", "invalid command"
        if isinstance(argv, str):  # string without shell: shlex-split
            try:
                argv = shlex.split(argv)
            except ValueError as e:  # e.g. unbalanced quote from the model
                return (2, "", f"invalid command: {e}", "invalid command")
        popen_args = argv
        popen_kwargs = {"shell": False}
    try:
        proc = subprocess.run(
            popen_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            **popen_kwargs,
        )
    except FileNotFoundError:
        return (NOT_FOUND_RC, "",
                f"command not found: {command!r}",
                "command not found")
    except subprocess.TimeoutExpired as e:
        # text=True may leave stdout/stderr as str OR bytes depending on
        # how far communicate() got before the timeout — handle both.
        def _as_text(v):
            if v is None:
                return ""
            if isinstance(v, bytes):
                return v.decode("utf-8", "replace")
            return str(v)

        out = _as_text(e.stdout)
        err = _as_text(e.stderr)
        return (-1, out, err + f"\n[attempt timed out after {timeout:g}s]",
                "timeout")
    except OSError as e:
        return (NOT_FOUND_RC, "", f"failed to start process: {e}",
                "spawn failed")
    return proc.returncode, proc.stdout or "", proc.stderr or "", ""


def _render_output(out: str, err: str) -> str:
    if err.strip():
        sep = "" if not out or out.endswith("\n") else "\n"
        return out + sep + "--- stderr ---\n" + err
    return out


def run_with_retry(command: Any, *, shell: bool = False,
                   max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                   base_delay: float = DEFAULT_BASE_DELAY,
                   max_delay: float = DEFAULT_MAX_DELAY,
                   timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Run ``command`` with exponential backoff on failure.

    Real subprocess execution — no mocks. Returns a dict with keys:
    ``success``, ``attempts``, ``returncode``, ``output`` (final
    attempt's merged stdout+stderr), ``elapsed_s``, ``cap_exceeded``
    and ``attempts_log`` (timestamped per-attempt entries).
    """
    max_attempts = max(1, int(max_attempts))
    base_delay = max(0.01, float(base_delay))
    max_delay = max(0.01, float(max_delay))  # strict cap, even if < base_delay
    timeout = max(1.0, float(timeout))

    log: list[dict] = []
    start_total = time.time()
    rc, out, err = -1, "", ""
    cap_exceeded = False
    fail_fast_reason = ""

    for n in range(1, max_attempts + 1):
        started = time.time()
        rc, out, err, note = _run_once(command, shell, timeout)
        dur = time.time() - started
        log.append({
            "attempt": n,
            "started_at": datetime.datetime.fromtimestamp(
                started, tz=datetime.timezone.utc).isoformat(
                    timespec="milliseconds"),
            "duration_s": round(dur, 3),
            "returncode": rc,
            "note": note,
            "waited_before_s": None,  # filled in after the sleep
        })
        if rc == 0:
            break  # success: stop early
        if rc == NOT_FOUND_RC:
            fail_fast_reason = (
                "command not found (exit 127) — failing fast, no retry")
            break
        if n == max_attempts:
            break  # out of attempts
        wait = backoff_delay(n - 1, base_delay, max_delay)
        elapsed = time.time() - start_total
        if elapsed + wait > TOTAL_TIME_CAP:
            cap_exceeded = True
            log.append({
                "attempt": "cap",
                "started_at": datetime.datetime.now(
                    tz=datetime.timezone.utc).isoformat(timespec="seconds"),
                "duration_s": 0.0,
                "returncode": rc,
                "note": (f"total time cap ({TOTAL_TIME_CAP:g}s) would be "
                         f"exceeded — giving up before retry {n + 1}"),
                "waited_before_s": None,
            })
            break
        log[-1]["waited_before_s"] = round(wait, 3)
        time.sleep(wait)

    return {
        "success": rc == 0,
        "attempts": sum(1 for e in log
                        if isinstance(e["attempt"], int)),
        "returncode": rc,
        "output": _render_output(out, err),
        "elapsed_s": round(time.time() - start_total, 3),
        "cap_exceeded": cap_exceeded,
        "fail_fast_reason": fail_fast_reason,
        "attempts_log": log,
    }


def _format_result(result: dict) -> str:
    """Human-readable multi-line report for the tool handlers."""
    lines = []
    if result["success"]:
        lines.append(f"SUCCESS after {result['attempts']} attempt(s) "
                     f"(returncode 0, {result['elapsed_s']}s total)")
    else:
        reason = (result["fail_fast_reason"]
                  or ("total time cap exceeded" if result["cap_exceeded"]
                      else f"all {result['attempts']} attempt(s) failed"))
        lines.append(f"FAILED — {reason} "
                     f"(final returncode {result['returncode']}, "
                     f"{result['elapsed_s']}s total)")
    lines.append("")
    lines.append("[attempt log]")
    for e in result["attempts_log"]:
        if not isinstance(e["attempt"], int):
            lines.append(f"  * {e['note']}")
            continue
        note = f" — {e['note']}" if e["note"] else ""
        lines.append(f"  attempt {e['attempt']} @ {e['started_at']} — "
                     f"rc={e['returncode']} ({e['duration_s']}s){note}")
        if e["waited_before_s"] is not None:
            lines.append(f"    waited {e['waited_before_s']}s before retry")
    lines.append("")
    lines.append("[final output]")
    out = result["output"].strip()
    lines.append(out if out else "(no output)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------


def _coerce_params(kwargs: dict) -> tuple[Any, int, float, float, float,
                                          str | None]:
    """Extract and validate shared tool params; error string or None."""
    command = kwargs.get("command")
    if command is None or (isinstance(command, str)
                           and not command.strip()) or command == []:
        return None, 0, 0.0, 0.0, 0.0, "ERROR: command is required"
    try:
        max_attempts = int(kwargs.get("max_attempts",
                                      DEFAULT_MAX_ATTEMPTS))
        base_delay = float(kwargs.get("base_delay", DEFAULT_BASE_DELAY))
        max_delay = float(kwargs.get("max_delay", DEFAULT_MAX_DELAY))
        timeout = float(kwargs.get("timeout", DEFAULT_TIMEOUT))
    except (TypeError, ValueError):
        return (None, 0, 0.0, 0.0, 0.0,
                "ERROR: max_attempts/base_delay/max_delay/timeout must "
                "be numbers")
    if max_attempts < 1:
        return None, 0, 0.0, 0.0, 0.0, "ERROR: max_attempts must be >= 1"
    if base_delay <= 0 or max_delay <= 0 or timeout <= 0:
        return (None, 0, 0.0, 0.0, 0.0,
                "ERROR: base_delay, max_delay and timeout must be positive")
    return command, max_attempts, base_delay, max_delay, timeout, None


def _handle_retry_run(**kwargs: Any) -> str:
    command, max_attempts, base_delay, max_delay, timeout, err = \
        _coerce_params(kwargs)
    if err:
        return err
    result = run_with_retry(command, shell=False,
                            max_attempts=max_attempts,
                            base_delay=base_delay,
                            max_delay=max_delay, timeout=timeout)
    return _format_result(result)


def _handle_retry_shell(**kwargs: Any) -> str:
    command, max_attempts, base_delay, max_delay, timeout, err = \
        _coerce_params(kwargs)
    if err:
        return err
    if not isinstance(command, str):
        return "ERROR: RetryShell needs a command string (shell=True)"
    result = run_with_retry(command, shell=True,
                            max_attempts=max_attempts,
                            base_delay=base_delay,
                            max_delay=max_delay, timeout=timeout)
    return _format_result(result)


def register(agent: Any) -> None:
    """Wire RetryRun / RetryShell into an agent (duck-typed, no imports)."""
    agent.tools["RetryRun"] = Tool(
        name="RetryRun",
        description=(
            "Run a command with real exponential-backoff retries for "
            "flaky operations (network calls, flaky tests, transient "
            "failures). No shell: pass an argv list, or a string which is "
            "shlex-split. Retries on any non-zero exit with "
            "base_delay*2^attempt (+ up to 20% jitter, capped at "
            "max_delay); stops early on success; fails fast on "
            "command-not-found (exit 127); total budget 10 minutes."),
        parameters={"type": "object", "properties": {
            "command": {"description": "argv list or command string"},
            "max_attempts": {"type": "integer",
                             "description": "max tries, default 3"},
            "base_delay": {"type": "number",
                           "description": "seconds before first retry, "
                                          "default 1.0"},
            "max_delay": {"type": "number",
                          "description": "cap per wait, default 60.0"},
            "timeout": {"type": "number",
                        "description": "per-attempt timeout, default 120"}},
            "required": ["command"]},
        handler=_handle_retry_run,
        risk=RISK_CONFIRM,
    )
    agent.tools["RetryShell"] = Tool(
        name="RetryShell",
        description=(
            "Same as RetryRun but runs the command through the shell "
            "(shell=True), so pipes, redirects, globs and compound "
            "commands work. Exponential backoff with jitter, stop-early "
            "on success, fail-fast on exit 127, 10-minute total budget."),
        parameters={"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "shell command line to retry"},
            "max_attempts": {"type": "integer",
                             "description": "max tries, default 3"},
            "base_delay": {"type": "number",
                           "description": "seconds before first retry, "
                                          "default 1.0"},
            "max_delay": {"type": "number",
                          "description": "cap per wait, default 60.0"},
            "timeout": {"type": "number",
                        "description": "per-attempt timeout, default 120"}},
            "required": ["command"]},
        handler=_handle_retry_shell,
        risk=RISK_CONFIRM,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Real retry proofs. Prints PASS lines, returns exit code."""
    import os
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    tmp = tempfile.mkdtemp(prefix="smartretry_selftest_")
    counter = os.path.join(tmp, "counter")
    # Script: fails the first two attempts, succeeds on the third.
    script = os.path.join(tmp, "flaky.sh")
    with open(script, "w") as f:
        f.write("#!/bin/bash\n"
                f'c=$(cat "{counter}" 2>/dev/null || echo 0)\n'
                'c=$((c + 1))\n'
                f'echo "$c" > "{counter}"\n'
                'if [ "$c" -lt 3 ]; then\n'
                '  echo "flaky attempt $c failed" >&2\n'
                '  exit 1\n'
                'fi\n'
                'echo "flaky attempt $c succeeded"\n'
                'exit 0\n')
    os.chmod(script, 0o755)

    # --- 1. retry-until-success -------------------------------------------
    r = run_with_retry([script], max_attempts=5, base_delay=0.2,
                       max_delay=5.0, timeout=30)
    check("fails-twice-then-succeeds: success", r["success"] is True)
    check("fails-twice-then-succeeds: exactly 3 attempts",
          r["attempts"] == 3 and len(r["attempts_log"]) == 3)
    check("fails-twice-then-succeeds: success output",
          "flaky attempt 3 succeeded" in r["output"])
    check("fails-twice-then-succeeds: returncode 0", r["returncode"] == 0)
    ts = [e["started_at"] for e in r["attempts_log"]]
    check("attempts logged with ISO timestamps",
          all(isinstance(t, str) and "T" in t for t in ts))

    # --- 2. backoff delays actually elapsed --------------------------------
    # Always-failing command: 4 attempts => waits of 0.5, 1.0, 2.0 (+jitter).
    t0 = time.time()
    r2 = run_with_retry(["bash", "-c", "exit 1"], max_attempts=4,
                        base_delay=0.5, max_delay=60.0, timeout=30)
    wall2 = time.time() - t0
    check("always-fails: not success", r2["success"] is False)
    check("always-fails: used all 4 attempts", r2["attempts"] == 4)
    expected_min = 0.5 + 1.0 + 2.0  # jitter only adds, never subtracts
    expected_max = expected_min * 1.20 + 3.0  # jitter + spawn overhead slack
    check(f"backoff delays elapsed ({wall2:.2f}s in "
          f"[{expected_min:.1f}, {expected_max:.1f}])",
          expected_min <= wall2 <= expected_max)
    # started_at gaps must match the exponential schedule too
    epochs = [datetime.datetime.fromisoformat(e["started_at"]).timestamp()
              for e in r2["attempts_log"]]
    gaps = [b - a for a, b in zip(epochs, epochs[1:])]
    check("started_at gaps grow exponentially",
          gaps[0] >= 0.45 and gaps[1] >= 0.9 and gaps[2] >= 1.8
          and gaps[0] < gaps[1] < gaps[2])
    waits = [e["waited_before_s"] for e in r2["attempts_log"][:3]]
    check("waits capped at max_delay",
          all(w is not None and w <= 60.0 for w in waits))

    # --- 3. max_delay cap ---------------------------------------------------
    r3 = run_with_retry(["bash", "-c", "exit 1"], max_attempts=4,
                        base_delay=10.0, max_delay=0.5, timeout=30)
    check("max_delay clamps huge backoff",
          all((e["waited_before_s"] or 0) <= 0.5
              for e in r3["attempts_log"]
              if isinstance(e["attempt"], int)))
    check("max_delay run still elapsed < 3s", r3["elapsed_s"] < 3.0)

    # --- 4. exit 127 fails fast ---------------------------------------------
    t0 = time.time()
    r4 = run_with_retry(["definitely_not_a_real_cmd_xyz_987"],
                        max_attempts=5, base_delay=0.5, timeout=30)
    wall4 = time.time() - t0
    check("127: fail fast, single attempt", r4["attempts"] == 1)
    check("127: not success", r4["success"] is False)
    check("127: returncode 127", r4["returncode"] == 127)
    check("127: fail-fast reason recorded",
          bool(r4["fail_fast_reason"]))
    check(f"127: no retry delay ({wall4:.2f}s < 0.5s)", wall4 < 0.5)
    r4b = run_with_retry("definitely_not_a_real_cmd_xyz_987", shell=True,
                         max_attempts=5, base_delay=0.5, timeout=30)
    check("127 via shell: fail fast, single attempt",
          r4b["attempts"] == 1 and r4b["returncode"] == 127)

    # --- 5. RetryShell with a complex command --------------------------------
    out = _handle_retry_shell(command="echo hello | tr a-z A-Z",
                              max_attempts=2, base_delay=0.1)
    check("RetryShell runs complex command",
          "SUCCESS" in out and "HELLO" in out)
    err_out = _handle_retry_run(command=["bash", "-c", "exit 3"],
                                max_attempts=2, base_delay=0.1)
    check("RetryRun handler reports failure",
          "FAILED" in err_out and "final returncode 3" in err_out)
    check("handlers reject empty command",
          _handle_retry_run(command="   ").startswith("ERROR"))
    check("handlers reject bad params",
          _handle_retry_run(command="true",
                            max_attempts=0).startswith("ERROR"))

    # --- 6. register() -------------------------------------------------------
    class _FakeAgent:
        pass
    fake = _FakeAgent()
    fake.tools = {}
    register(fake)
    check("register adds both tools",
          "RetryRun" in fake.tools and "RetryShell" in fake.tools)
    check("tool schemas valid",
          fake.tools["RetryRun"].openai_schema()["function"]["name"]
          == "RetryRun"
          and fake.tools["RetryShell"].openai_schema()["type"]
          == "function")
    check("tools are confirm-risk",
          fake.tools["RetryRun"].risk == RISK_CONFIRM
          and fake.tools["RetryShell"].risk == RISK_CONFIRM)

    # cleanup
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)

    print("PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
