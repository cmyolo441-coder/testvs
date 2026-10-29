"""Hooks system — Claude Code-style PreToolUse / PostToolUse hooks.

User-configured shell hooks, loaded from ``~/.fullagent/hooks.json``::

    {"PreToolUse": [{"match": "run_command|.*", "command": "echo hi"}],
     "PostToolUse": [{"match": "write_file", "command": "/path/to/lint.sh"}]}

``match`` is a regex matched (search) against the tool name. ``command`` is a
shell command run with the environment::

    HOOK_TOOL        — the tool name being executed
    HOOK_ARGS_JSON   — the tool's args, JSON-encoded
    HOOK_RESULT_JSON — (PostToolUse only) the tool's result, JSON-encoded

Conventions for hook scripts:

* Hook stdout is logged (info) on success.
* ``PreToolUse``: exit code != 0 **blocks** the tool call. The hook's
  stderr (or stdout, if stderr is empty) is returned as the error message.
* ``PostToolUse``: failures only log a warning; the tool result stands.
* Arg modification: a ``PreToolUse`` hook may print a single JSON object
  on the *last line* of its stdout of the form ``{"args": {...}}``; those
  args are merged into the tool call (hook keys win).

Safety:

* Every hook runs with a 10s timeout — a hung hook never hangs the agent.
* This module never raises: any hook-loading or hook-execution error is
  caught and the hook is skipped (fail-open), except an explicit non-zero
  exit from a PreToolUse hook, which blocks.
* Never imports ``.agent`` or ``.tui`` — it is dependency-free.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

_log = logging.getLogger(__name__)

HOOK_TIMEOUT_S = 10
DEFAULT_CONFIG_PATH = (Path(os.environ["FULLAGENT_HOME"]).expanduser() / "hooks.json") if os.environ.get("FULLAGENT_HOME") else Path("~/.fullagent/hooks.json").expanduser()


def _config_path(path: str | os.PathLike | None) -> Path:
    if path is None:
        return Path(os.path.expandvars(DEFAULT_CONFIG_PATH.__str__()))
    return Path(os.path.expandvars(str(path)))


def load_hooks(path: str | os.PathLike | None = None) -> dict:
    """Load the hooks config. Missing/unreadable/invalid file → {}."""
    cfg_path = _config_path(path)
    try:
        raw = cfg_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        _log.warning("hooks: invalid JSON in %s — hooks disabled", cfg_path)
        return {}
    if not isinstance(data, dict):
        _log.warning("hooks: top-level JSON in %s must be an object", cfg_path)
        return {}
    return data


def _iter_matching(hooks: dict, kind: str, tool_name: str) -> list[dict]:
    """Return hook entries of ``kind`` whose ``match`` regex hits tool_name."""
    out: list[dict] = []
    entries = hooks.get(kind, [])
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("match", ".*")
        command = entry.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        # A non-string "match" (null, number, list) must skip only this
        # entry. Previously re.search() raised TypeError, aborting the
        # whole loop and silently disabling every later safety hook.
        if not isinstance(pattern, str):
            _log.warning("hooks: non-string match %r — skipping hook",
                         pattern)
            continue
        try:
            if re.search(pattern, tool_name or ""):
                out.append(entry)
        except re.error:
            _log.warning("hooks: bad regex %r — skipping hook", pattern)
    return out


def _run_hook_command(command: str, env: dict) -> tuple[int | None, str, str]:
    """Run one hook command with a timeout. Never raises.

    Returns ``(rc, stdout, stderr)``. ``rc`` is None when the hook could
    not be run at all (timeout, spawn failure) — that is an infrastructure
    error and must fail OPEN, never count as a block.
    """
    try:
        with tempfile.TemporaryFile(mode="w+") as out_f, \
                tempfile.TemporaryFile(mode="w+") as err_f:
            proc = subprocess.run(
                command,
                shell=True,
                env=env,
                stdout=out_f,
                stderr=err_f,
                timeout=HOOK_TIMEOUT_S,
            )
            out_f.seek(0)
            err_f.seek(0)
            return proc.returncode, out_f.read(), err_f.read()
    except subprocess.TimeoutExpired:
        _log.warning("hooks: command timed out after %ss: %s",
                     HOOK_TIMEOUT_S, command[:200])
        return None, "", f"hook timed out after {HOOK_TIMEOUT_S}s"
    except Exception as e:  # noqa: BLE001 — fail-open
        _log.warning("hooks: command failed to run: %s", e)
        return None, "", f"hook failed to run: {e}"


def _hook_env(tool_name: str, args: dict, result: object = None) -> dict:
    env = dict(os.environ)
    env["HOOK_TOOL"] = tool_name or ""
    try:
        env["HOOK_ARGS_JSON"] = json.dumps(args or {}, default=str)
    except Exception:  # noqa: BLE001
        env["HOOK_ARGS_JSON"] = "{}"
    if result is not None:
        try:
            env["HOOK_RESULT_JSON"] = json.dumps(result, default=str)
        except Exception:  # noqa: BLE001
            env["HOOK_RESULT_JSON"] = "null"
    return env


def _parse_arg_patch(stdout: str) -> dict:
    """Extract {"args": {...}} from the last non-empty stdout line."""
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        data = json.loads(lines[-1])
    except (json.JSONDecodeError, ValueError):
        return {}
    if isinstance(data, dict) and isinstance(data.get("args"), dict):
        return data["args"]
    return {}


def run_pre_tool(tool_name: str, args: dict,
                 path: str | os.PathLike | None = None) -> tuple[bool, dict, str]:
    """Run PreToolUse hooks.

    Returns ``(allowed, new_args, message)``. ``new_args`` is ``args`` with
    any hook-supplied patches merged in (hook keys win). If a hook blocks,
    ``allowed`` is False and ``message`` carries the hook's stderr/stdout.
    Never raises — errors fail open (allowed=True).
    """
    try:
        hooks = load_hooks(path)
        entries = _iter_matching(hooks, "PreToolUse", tool_name)
        new_args = dict(args or {})
        for entry in entries:
            env = _hook_env(tool_name, new_args)
            rc, stdout, stderr = _run_hook_command(entry["command"], env)
            if stdout.strip():
                _log.info("hooks: PreToolUse[%s] stdout: %s",
                          tool_name, stdout.strip()[:2000])
            if rc is None:
                # infrastructure error (timeout/spawn) — fail OPEN, skip hook
                continue
            if rc != 0:
                message = (stderr or stdout).strip() \
                    or f"PreToolUse hook blocked {tool_name}"
                _log.warning("hooks: PreToolUse[%s] blocked: %s",
                             tool_name, message[:2000])
                return False, args, message
            patch = _parse_arg_patch(stdout)
            if patch:
                _log.info("hooks: PreToolUse[%s] patched args: %s",
                          tool_name, sorted(patch))
                new_args.update(patch)
        return True, new_args, ""
    except Exception as e:  # noqa: BLE001 — fail-open
        _log.warning("hooks: PreToolUse error (fail-open): %s", e)
        return True, args, ""


def run_post_tool(tool_name: str, args: dict, result: object = None,
                  path: str | os.PathLike | None = None) -> None:
    """Run PostToolUse hooks. Never raises — failures only log."""
    try:
        hooks = load_hooks(path)
        entries = _iter_matching(hooks, "PostToolUse", tool_name)
        for entry in entries:
            env = _hook_env(tool_name, args, result)
            rc, stdout, stderr = _run_hook_command(entry["command"], env)
            if stdout.strip():
                _log.info("hooks: PostToolUse[%s] stdout: %s",
                          tool_name, stdout.strip()[:2000])
            if rc is None:
                # infrastructure error — already warned, keep going
                continue
            if rc != 0:
                _log.warning("hooks: PostToolUse[%s] failed (rc=%s): %s",
                             tool_name, rc, (stderr or stdout).strip()[:2000])
    except Exception as e:  # noqa: BLE001 — never raise
        _log.warning("hooks: PostToolUse error: %s", e)


def register(agent) -> None:
    """Wire hooks into an Agent instance.

    Sets ``agent.hooks_enabled = True``. Hooks are user config (not LLM
    tools), so no tools are registered. The agent's ``_execute_tool`` should
    call :func:`run_pre_tool` before execution and :func:`run_post_tool`
    after (see wiring snippet in module docs / __main__ self-test).
    """
    agent.hooks_enabled = True


# ---------------------------------------------------------------------------
# Wiring snippet for fullagent/agent.py `_execute_tool`
#
# Place the pre-hook call AFTER the approval/snapshot section (so hooks see
# the final args and don't bypass user approval) but BEFORE the
# `self.log.append("tool.call", ...)` / execution block:
#
#     # Hooks: PreToolUse may block or rewrite args (never raises).
#     if getattr(self, "hooks_enabled", False):
#         allowed, patched, hook_msg = run_pre_tool(ev.name, ev.args)
#         if not allowed:
#             ev.status = "blocked"
#             ev.result = f"ERROR: blocked — {hook_msg}"
#             self.log.append("tool.blocked",
#                             {"name": ev.name, "reason": hook_msg},
#                             causation_id=causation_id)
#             return
#         ev.args = patched
#
# Place the post-hook call AFTER the tool has executed and the result/error
# bookkeeping is settled (after the dead-end / oscillation sections, or at
# the very end of _execute_tool):
#
#     # Hooks: PostToolUse is fire-and-forget (never raises).
#     if getattr(self, "hooks_enabled", False):
#         run_post_tool(ev.name, ev.args, ev.result)
#
# Import at the top of agent.py:
#     from .hooks import run_pre_tool, run_post_tool
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Self-test: blocking hook, arg-modifying hook, missing file."""
    import tempfile as _tf

    cfg = {
        "PreToolUse": [
            {"match": "^write_file$", "command": "echo blocked-by-test >&2; exit 1"},
            {"match": "^run_command$",
             "command": 'echo \'{"args": {"command": "echo patched"}}\''},
        ],
        "PostToolUse": [
            {"match": "^run_command$", "command": "exit 0"},
            {"match": "^write_file$", "command": "exit 3"},
        ],
    }
    with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name

    try:
        # 1. Blocking hook: exit 1 on write_file blocks the call.
        allowed, new_args, msg = run_pre_tool(
            "write_file", {"path": "x.txt", "content": "hi"}, path=cfg_path)
        assert allowed is False, f"expected block, got {allowed}"
        assert "blocked-by-test" in msg, f"message missing hook stderr: {msg!r}"
        assert new_args == {"path": "x.txt", "content": "hi"}, new_args

        # 2. Arg-modifying hook: run_command gets {"command": "echo patched"}.
        allowed, new_args, msg = run_pre_tool(
            "run_command", {"command": "rm -rf /"}, path=cfg_path)
        assert allowed is True, f"expected allow, got {allowed} ({msg})"
        assert new_args["command"] == "echo patched", new_args

        # 3. PostToolUse runs without raising (incl. a failing hook, exit 3).
        run_post_tool("write_file", {"path": "x.txt"}, "ok", path=cfg_path)
        run_post_tool("run_command", {"command": "ls"}, "ok", path=cfg_path)

        # 4. Missing file → fail-open, args untouched.
        allowed, new_args, msg = run_pre_tool(
            "write_file", {"path": "x.txt"}, path="/nonexistent/hooks.json")
        assert allowed is True and new_args == {"path": "x.txt"} and msg == ""
        assert load_hooks("/nonexistent/hooks.json") == {}

        # 5. Timeout hook fails open (does not hang the test).
        cfg2 = {"PreToolUse": [{"match": ".*", "command": "sleep 30"}]}
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f2:
            json.dump(cfg2, f2)
            cfg2_path = f2.name
        allowed, _, _ = run_pre_tool("run_command", {}, path=cfg2_path)
        assert allowed is True, "timed-out hook must fail open"

        # 6. register() attaches the flag.
        class _Fake: pass
        ag = _Fake()
        register(ag)
        assert ag.hooks_enabled is True

        # 7. Env contract: hook receives HOOK_TOOL / HOOK_ARGS_JSON.
        cfg3 = {"PreToolUse": [
            {"match": ".*",
             "command": 'test "$HOOK_TOOL" = "my_tool" && '
                        'echo "$HOOK_ARGS_JSON" | grep -q "k"'}]}
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as f3:
            json.dump(cfg3, f3)
            cfg3_path = f3.name
        allowed, _, _ = run_pre_tool("my_tool", {"k": "v"}, path=cfg3_path)
        assert allowed is True, "env-based hook should succeed"

        print("PASS")
    finally:
        Path(cfg_path).unlink(missing_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    _self_test()
