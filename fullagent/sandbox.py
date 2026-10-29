"""Sandboxed command execution with a graceful degradation chain.

Real isolation, picked from what the machine actually provides:

  1. ``bwrap`` (bubblewrap) — full container-style sandbox: read-only host
     binds (``--ro-bind /usr /usr``), private ``--tmpfs /tmp``,
     ``--unshare-net`` for network isolation.
  2. ``unshare`` — ``unshare -n`` gives a private network namespace
     (no external connectivity) while the command still runs normally.
  3. ``rlimit`` — plain subprocess, but with real kernel resource limits
     applied in the child via ``preexec_fn``: ``RLIMIT_CPU`` (CPU time),
     ``RLIMIT_AS`` (address space), ``RLIMIT_FSIZE`` (max file size).

Backends are *functionally* probed, not just ``which``-ed: a backend only
counts as available if we can actually spawn a trivial command through it.
The tool result always names the backend that was really used — the
isolation level is never faked.

Engine::

    info = sandbox_info()        # {'bwrap': False, 'unshare': True, ...}
    res  = run_sandboxed("echo hello")
    # {'sandbox': 'unshare', 'returncode': 0, 'stdout': 'hello\n', ...}

LLM tools (see :func:`register`)::

    SandboxedRun — run a shell command under the best available sandbox
    SandboxInfo  — report which sandbox backends this machine has

Only stdlib + ``.tools`` are imported (``Tool`` for the registry).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from typing import Any

from .tools import RISK_CONFIRM, RISK_SAFE, Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKEND_BWRAP = "bwrap"
BACKEND_UNSHARE = "unshare"
BACKEND_RLIMIT = "rlimit"

DEFAULT_TIMEOUT = 60.0    # sane default for sandboxed one-shots
MAX_TIMEOUT = 3600.0      # hard cap: 1 hour
MIN_TIMEOUT = 1.0

# Default resource ceilings for the weakest (rlimit) backend — and also
# applied as a safety floor under the other backends.
DEFAULT_CPU_LIMIT = 120          # seconds of CPU time
DEFAULT_MEM_LIMIT_MB = 1024      # address space, MiB
DEFAULT_FSIZE_LIMIT_MB = 100     # single file size, MiB

# ---------------------------------------------------------------------------
# Backend detection (functional probes, cached)
# ---------------------------------------------------------------------------

_detection_cache: dict[str, bool] | None = None


def _probe(argv: list[str]) -> bool:
    """Return True only if `argv` actually executes with rc 0."""
    try:
        r = subprocess.run(argv, capture_output=True, timeout=15)
        return r.returncode == 0
    except Exception:
        return False


def _bwrap_usable() -> bool:
    if shutil.which("bwrap") is None:
        return False
    # Minimal real bwrap invocation: ro-bind /usr, throwaway /tmp, run true.
    return _probe(["bwrap",
                   "--ro-bind", "/usr", "/usr",
                   "--tmpfs", "/tmp",
                   "--proc", "/proc",
                   "--dev", "/dev",
                   "true"])


def _unshare_usable() -> bool:
    if shutil.which("unshare") is None:
        return False
    # -n = new network namespace. Fails without the needed privileges,
    # so the probe (not just which()) decides.
    return _probe(["unshare", "-n", "true"])


def detect_backends() -> dict[str, bool]:
    """Probe which sandbox backends genuinely work on this machine."""
    global _detection_cache
    if _detection_cache is None:
        _detection_cache = {
            BACKEND_BWRAP: _bwrap_usable(),
            BACKEND_UNSHARE: _unshare_usable(),
            # rlimit is always available on POSIX; on Windows preexec_fn
            # does not exist, so there it is genuinely unavailable.
            BACKEND_RLIMIT: os.name == "posix",
        }
    return dict(_detection_cache)


def best_backend() -> str:
    """Strongest backend that actually works here."""
    avail = detect_backends()
    for name in (BACKEND_BWRAP, BACKEND_UNSHARE, BACKEND_RLIMIT):
        if avail.get(name):
            return name
    return BACKEND_RLIMIT  # last resort: plain run, no limits (non-POSIX)


def sandbox_info() -> dict[str, Any]:
    """Human/computer-readable report of real sandbox availability."""
    avail = detect_backends()
    return {
        "bwrap": avail[BACKEND_BWRAP],
        "unshare": avail[BACKEND_UNSHARE],
        "rlimit": avail[BACKEND_RLIMIT],
        "best": best_backend(),
        "notes": {
            "bwrap": ("full container isolation: ro /usr bind, private "
                      "/tmp, optional --unshare-net"),
            "unshare": "network-namespace isolation via `unshare -n`",
            "rlimit": ("no namespace isolation; kernel resource limits "
                       "(CPU/memory/file size) in child process"),
        },
    }


# ---------------------------------------------------------------------------
# Resource limits (preexec_fn, POSIX only)
# ---------------------------------------------------------------------------

def _make_preexec(cpu_limit: int | None,
                  mem_limit_mb: int | None,
                  fsize_limit_mb: int | None):
    """Build a preexec_fn applying real rlimits in the child process."""
    def _preexec() -> None:
        import resource  # local import: absent on non-POSIX
        if cpu_limit:
            v = int(cpu_limit)
            resource.setrlimit(resource.RLIMIT_CPU, (v, v))
        if mem_limit_mb:
            v = int(mem_limit_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (v, v))
        if fsize_limit_mb:
            v = int(fsize_limit_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (v, v))
    return _preexec


# ---------------------------------------------------------------------------
# Command construction per backend
# ---------------------------------------------------------------------------

def _build_bwrap_command(command: str, network: bool) -> list[str]:
    argv = [
        "bwrap",
        "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--tmpfs", "/tmp",
        "--proc", "/proc",
        "--dev", "/dev",
    ]
    for path in ("/bin", "/sbin", "/lib", "/lib64", "/etc"):
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    if not network:
        argv.append("--unshare-net")
    argv += [
        "--setenv", "PATH", "/usr/bin:/bin:/usr/sbin:/sbin",
        "--chdir", "/tmp",
        "--", "sh", "-c", command,
    ]
    return argv


def _build_unshare_command(command: str, network: bool) -> list[str]:
    # unshare only isolates the network namespace; the shell runs the
    # command directly otherwise. When network=True we skip -n entirely
    # (a no-op unshare would add nothing).
    if network:
        return ["sh", "-c", command]
    return ["unshare", "-n", "sh", "-c", command]


def build_command(command: str,
                  network: bool = False,
                  backend: str | None = None) -> tuple[list[str], str]:
    """Return (argv, backend_name) for the strongest usable backend."""
    chosen = backend or best_backend()
    if chosen == BACKEND_BWRAP and detect_backends()[BACKEND_BWRAP]:
        return _build_bwrap_command(command, network), BACKEND_BWRAP
    if chosen == BACKEND_UNSHARE and detect_backends()[BACKEND_UNSHARE]:
        if network:
            return _build_unshare_command(command, network), BACKEND_RLIMIT
        return _build_unshare_command(command, network), BACKEND_UNSHARE
    return ["sh", "-c", command], BACKEND_RLIMIT


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def run_sandboxed(command: str,
                  network: bool = False,
                  timeout: float = DEFAULT_TIMEOUT,
                  cpu_limit: int | None = DEFAULT_CPU_LIMIT,
                  mem_limit_mb: int | None = DEFAULT_MEM_LIMIT_MB,
                  fsize_limit_mb: int | None = DEFAULT_FSIZE_LIMIT_MB,
                  backend: str | None = None) -> dict[str, Any]:
    """Run `command` under the best available sandbox, for real.

    Returns a dict with the actual backend used, returncode, stdout,
    stderr, elapsed time, and whether the wall-clock timeout fired.
    Resource limits are enforced in the child for every backend (the
    rlimit floor applies under bwrap/unshare too).
    """
    started = time.time()
    result: dict[str, Any] = {
        "sandbox": None,
        "command": command,
        "network": bool(network),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "elapsed": 0.0,
        "error": None,
    }
    if not command or not command.strip():
        result["error"] = "empty command"
        return result

    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        result["error"] = f"invalid timeout: {timeout!r}"
        return result
    if timeout != timeout:  # NaN would make the clamp/timeout meaningless
        result["error"] = "invalid timeout: NaN"
        return result
    timeout = max(MIN_TIMEOUT, min(timeout, MAX_TIMEOUT))
    argv, used = build_command(command, network=network, backend=backend)
    result["sandbox"] = used

    preexec = None
    if os.name == "posix":
        preexec = _make_preexec(cpu_limit, mem_limit_mb, fsize_limit_mb)

    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            preexec_fn=preexec,
            # Own process group, so a timeout can kill the whole tree.
            **({"start_new_session": True} if os.name == "posix" else {}),
        )
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # subprocess.run() only kills the direct child; backgrounded
            # grandchildren (`cmd &`, shell wrappers) kept running.
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                for stream in (proc.stdout, proc.stderr):
                    try:
                        stream.close()
                    except Exception:
                        pass
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
                out, err = "", ""
            result["timed_out"] = True
            result["stdout"] = out or ""
            result["stderr"] = err or ""
            result["error"] = (f"timed out after {timeout:g}s "
                               f"(process killed)")
        else:
            result["returncode"] = proc.returncode
            result["stdout"] = out or ""
            result["stderr"] = err or ""
    except OSError as e:
        result["error"] = f"failed to start sandboxed process: {e}"
    finally:
        result["elapsed"] = time.time() - started
    return result


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def _format_result(res: dict[str, Any]) -> str:
    lines = [
        f"sandbox: {res['sandbox']}",
        f"returncode: {res['returncode']}",
        f"timed_out: {res['timed_out']}",
        f"elapsed: {res['elapsed']:.2f}s",
    ]
    if res["error"]:
        lines.append(f"error: {res['error']}")
    if res["stdout"]:
        lines.append("--- stdout ---")
        lines.append(res["stdout"].rstrip("\n"))
    if res["stderr"]:
        lines.append("--- stderr ---")
        lines.append(res["stderr"].rstrip("\n"))
    return "\n".join(lines)


def _handle_sandboxed_run(command: str = "",
                          network: bool = False,
                          timeout: float = DEFAULT_TIMEOUT) -> str:
    if isinstance(network, str):
        # bool("false") is True; models often send stringified booleans
        network = network.strip().lower() in ("1", "true", "yes", "y", "on")
    res = run_sandboxed(command, network=bool(network), timeout=timeout)
    return _format_result(res)


def _handle_sandbox_info() -> str:
    info = sandbox_info()
    lines = ["Sandbox backend availability on this machine:"]
    for name in ("bwrap", "unshare", "rlimit"):
        status = "AVAILABLE" if info[name] else "unavailable"
        lines.append(f"  {name:<8} {status}  ({info['notes'][name]})")
    lines.append(f"best backend in use: {info['best']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Wire SandboxedRun / SandboxInfo into an agent (duck-typed)."""
    agent.tools["SandboxedRun"] = Tool(
        name="SandboxedRun",
        description=(
            "Run a shell command inside the strongest available sandbox "
            "on this machine (bubblewrap if present, else network-namespace "
            "isolation via unshare, else subprocess with kernel resource "
            "limits). The result always reports which sandbox was actually "
            "used. network=false (default) isolates the command from the "
            "network when the backend supports it."),
        parameters={"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "shell command to run sandboxed"},
            "network": {"type": "boolean",
                        "description": ("allow network access (default false; "
                                        "only honored where the backend "
                                        "supports it)")},
            "timeout": {"type": "number",
                        "description": "wall-clock seconds before kill "
                                       "(default 60, max 3600)"}},
            "required": ["command"]},
        handler=_handle_sandboxed_run,
        risk=RISK_CONFIRM,
    )
    agent.tools["SandboxInfo"] = Tool(
        name="SandboxInfo",
        description=(
            "Report which sandbox backends are actually available on this "
            "machine (bwrap / unshare / rlimit) and which one SandboxedRun "
            "will use. No command is executed."),
        parameters={"type": "object", "properties": {}},
        handler=_handle_sandbox_info,
        risk=RISK_SAFE,
    )


# ---------------------------------------------------------------------------
# Self-test: everything below genuinely executes
# ---------------------------------------------------------------------------

def _selftest() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(("ok   " if cond else "FAIL ") + name
              + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            failures.append(name)

    print("== sandbox self-test (all real executions) ==")

    # --- 1. SandboxInfo: real availability ---------------------------------
    info = sandbox_info()
    print(f"     detected: bwrap={info['bwrap']} unshare={info['unshare']} "
          f"rlimit={info['rlimit']} best={info['best']}")
    check("info has all backends",
          all(k in info for k in ("bwrap", "unshare", "rlimit", "best")))
    check("best matches strongest available",
          (info["bwrap"] and info["best"] == "bwrap")
          or (not info["bwrap"] and info["unshare"]
              and info["best"] == "unshare")
          or (not info["bwrap"] and not info["unshare"]
              and info["best"] == "rlimit"))

    # --- 2. SandboxedRun: real echo ----------------------------------------
    res = run_sandboxed("echo hello")
    check("echo hello returncode 0", res["returncode"] == 0,
          f"rc={res['returncode']} err={res['error']}")
    check("echo hello output real", res["stdout"].strip() == "hello",
          repr(res["stdout"]))
    check("backend honestly reported",
          res["sandbox"] == info["best"],
          f"used={res['sandbox']} best={info['best']}")

    # --- 3. Network isolation (only where a namespace backend exists) ------
    if info["bwrap"] or info["unshare"]:
        argv, used = build_command("true", network=False)
        argv_str = " ".join(argv)
        has_flag = ("--unshare-net" in argv) or ("-n" in argv)
        check("isolation flag really passed", has_flag, argv_str)
        if shutil.which("curl"):
            net = run_sandboxed(
                "curl -s -m 8 -o /dev/null -w '%{http_code}' "
                "https://example.com ; echo \" rc=$?\"",
                network=False, timeout=20)
            out = (net["stdout"] + net["stderr"]).strip()
            unreachable = (net["returncode"] != 0
                           or "000" in out
                           or "rc=6" in out or "rc=7" in out
                           or "rc=28" in out)
            check(f"external curl fails inside {used}", unreachable, out)
        else:
            print("     (curl not installed — flag check above stands)")
    else:
        print("     (no namespace backend here — skipping net test)")

    # --- 4. Resource limits: command exceeding limits is really killed ------
    # CPU: an infinite loop with a 2s CPU cap must be killed by the kernel
    # (signal, well before any wall-clock timeout). The differential proves
    # it: the same loop with NO cpu cap hits the 3s wall timeout instead.
    cpu = run_sandboxed("python3 -c 'while True: pass'",
                        timeout=30, cpu_limit=2, mem_limit_mb=None,
                        fsize_limit_mb=None)
    killed_by_cpu = (cpu["returncode"] in (-9, -24, 137, 152)
                     # -9/137 = SIGKILL (kernel sends it 1s after the
                     # ignored SIGXCPU), -24/152 = SIGXCPU directly;
                     # 137/152 = shell's 128+signum reporting of the same
                     and not cpu["timed_out"]
                     and cpu["elapsed"] < 15)
    check("CPU hog killed by RLIMIT_CPU", killed_by_cpu,
          f"rc={cpu['returncode']} timed_out={cpu['timed_out']} "
          f"elapsed={cpu['elapsed']:.1f}s")
    nocap = run_sandboxed("python3 -c 'while True: pass'",
                          timeout=3, cpu_limit=None, mem_limit_mb=None,
                          fsize_limit_mb=None)
    check("same loop without cap hits wall timeout (differential)",
          nocap["timed_out"] and nocap["returncode"] is None,
          f"rc={nocap['returncode']} timed_out={nocap['timed_out']}")
    # Memory: allocate 300MB with a 64MB address-space cap → nonzero rc.
    mem = run_sandboxed(
        "python3 -c \"x = bytearray(300*1024*1024); print(len(x))\"",
        timeout=30, cpu_limit=None, mem_limit_mb=64,
        fsize_limit_mb=None)
    check("memory hog killed by RLIMIT_AS", mem["returncode"] != 0,
          f"rc={mem['returncode']} out={mem['stdout'][:60]!r}")
    # File size: writing 5MB past a 1MB FSIZE cap → file capped at 1MiB.
    cap_file = "/tmp/sb_fsize_cap_test.bin"
    fsz = run_sandboxed(
        f"head -c 5000000 /dev/zero > {cap_file} 2>/dev/null; "
        f"stat -c %s {cap_file}; rm -f {cap_file}",
        timeout=30, cpu_limit=None, mem_limit_mb=None,
        fsize_limit_mb=1)
    check("file write capped at RLIMIT_FSIZE",
          fsz["stdout"].strip() == "1048576",
          f"rc={fsz['returncode']} size={fsz['stdout'].strip()!r}")

    # --- 5. register() ------------------------------------------------------
    class _FakeAgent:
        pass
    fake = _FakeAgent()
    fake.tools = {}
    register(fake)
    check("register adds both tools",
          "SandboxedRun" in fake.tools and "SandboxInfo" in fake.tools)
    check("tool schemas valid",
          fake.tools["SandboxedRun"].openai_schema()["function"]["name"]
          == "SandboxedRun"
          and fake.tools["SandboxInfo"].openai_schema()["type"]
          == "function")
    check("SandboxedRun handler end-to-end",
          "sandbox:" in _handle_sandboxed_run(command="echo hi")
          and "hi" in _handle_sandboxed_run(command="echo hi"))
    check("SandboxInfo handler end-to-end",
          "bwrap" in _handle_sandbox_info()
          and "best backend" in _handle_sandbox_info())

    print("PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
