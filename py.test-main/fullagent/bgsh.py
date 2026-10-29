"""Background bash execution: ``run_in_background`` + ``BashOutput``.

Claude Code CLI feature: launch a shell command that keeps running in the
background and poll it for new output, instead of blocking on ``run_command``.

Engine::

    task_id = launch("pytest -q", "full test suite")   # returns immediately
    poll(task_id)          # {id, description, status, output_so_far, returncode}
    kill(task_id)          # terminate the whole process group
    list_background_tasks()  # snapshot for the TUI widget (exact contract below)

LLM tools (see :func:`register`)::

    BashBG     — launch a command in the background, returns the task id
    BashOutput — new output since the last BashOutput poll + status

TUI CONTRACT (a sibling TUI widget consumes this — keep exact):

    list_background_tasks() -> list[dict] with keys
        id, description, status, started_at (float epoch)

Only stdlib + ``.tools`` are imported (``Tool`` for the registry).

Conventions borrowed from ``tools.run_command``:
text mode, ``encoding="utf-8"``, ``errors="replace"``, line-based capture.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from typing import Any

from .tools import RISK_CONFIRM, RISK_SAFE, Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_KILLED = "killed"

DEFAULT_TIMEOUT = 600.0   # 10 minutes, same as run_command's hard cap
MAX_TIMEOUT = 3600.0      # hard cap: 1 hour

# A chatty background task must never exhaust memory: keep at most this
# much text per stream (the tail) and this many finished tasks.
MAX_OUTPUT_CHARS = 262_144
MAX_FINISHED_TASKS = 32

# ---------------------------------------------------------------------------
# Task engine
# ---------------------------------------------------------------------------


class _BgTask:
    """One background shell command.

    stdout/stderr are captured incrementally into thread-safe chunk lists.
    ``_stdout_seen``/``_stderr_seen`` are char offsets of what BashOutput has
    already delivered, so successive polls are incremental.
    """

    def __init__(self, task_id: str, description: str, timeout: float):
        self.id = task_id
        self.description = description
        self.timeout = timeout
        self.started_at = time.time()
        self.status = STATUS_RUNNING
        self.returncode: int | None = None
        self.proc: subprocess.Popen | None = None
        self.spawn_error: str | None = None
        self._lock = threading.Lock()
        self._stdout: deque[str] = deque()
        self._stderr: deque[str] = deque()
        self._stdout_chars = 0
        self._stderr_chars = 0
        self._stdout_dropped = 0
        self._stderr_dropped = 0
        self._stdout_seen = 0
        self._stderr_seen = 0
        self._kill_requested = False


def _append_capped(task: _BgTask, stream: str, text: str) -> None:
    """Append output, discarding the oldest text past MAX_OUTPUT_CHARS.

    Caller must hold ``task._lock``. ``_stdout_dropped``/``_stderr_dropped``
    count chars evicted from the front so poll offsets stay absolute.
    """
    chunks = task._stdout if stream == "out" else task._stderr
    total = (task._stdout_chars if stream == "out"
             else task._stderr_chars) + len(text)
    chunks.append(text)
    dropped = 0
    while total > MAX_OUTPUT_CHARS and chunks:
        if len(chunks) == 1:
            extra = total - MAX_OUTPUT_CHARS
            chunks[0] = chunks[0][extra:]
            total -= extra
            dropped += extra
            break
        old = chunks.popleft()
        total -= len(old)
        dropped += len(old)
    if stream == "out":
        task._stdout_chars = total
        task._stdout_dropped += dropped
    else:
        task._stderr_chars = total
        task._stderr_dropped += dropped


_TASKS: dict[str, _BgTask] = {}
_TASKS_LOCK = threading.Lock()


def _shell_argv() -> list[str] | None:
    """Resolve a POSIX shell the way run_command does (bash preferred)."""
    for name in ("bash", "sh"):
        exe = shutil.which(name)
        if exe:
            return [exe, "-c"]
    return None


def _get(task_id: Any) -> _BgTask | None:
    if not isinstance(task_id, str):
        return None
    with _TASKS_LOCK:
        return _TASKS.get(task_id)


def _pump_pipe(task: _BgTask, pipe, stream: str) -> None:
    """Drain one pipe line-by-line into the task's capped buffer."""
    try:
        for line in iter(pipe.readline, ""):
            with task._lock:
                _append_capped(task, stream, line)
    finally:
        try:
            pipe.close()
        except Exception:  # noqa: BLE001 — never break the pump
            pass


def _kill_group(task: _BgTask) -> None:
    """Terminate the process group (SIGKILL), falling back to proc.kill()."""
    proc = task.proc
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _watch(task: _BgTask) -> None:
    """Wait for the process, enforce the timeout, then settle the status."""
    assert task.proc is not None
    timed_out = False
    try:
        task.returncode = task.proc.wait(timeout=task.timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(task)
        task.returncode = task.proc.wait()
    with task._lock:
        rc = task.returncode
        if timed_out:
            task.status = STATUS_KILLED
            _append_capped(
                task, "err",
                f"\n[timeout after {task.timeout:g}s — process killed]\n")
        elif task._kill_requested and rc is not None and rc < 0:
            # negative rc => died by signal (our kill); a natural exit that
            # raced kill() keeps its real outcome below.
            task.status = STATUS_KILLED
        elif rc == 0:
            task.status = STATUS_DONE
        else:
            task.status = STATUS_ERROR


def launch(command: str, description: str = "",
           timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run ``command`` in the background; return its task id.

    Raises ValueError on a bad command/timeout. A spawn failure does not
    raise — the task is registered with status ``error`` instead.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    try:
        secs = float(timeout)
    except (TypeError, ValueError):
        raise ValueError(f"timeout must be a number, got {timeout!r}")
    if secs <= 0:
        raise ValueError("timeout must be positive")
    secs = min(secs, MAX_TIMEOUT)
    desc = (description.strip() if isinstance(description, str)
            and description.strip() else command.strip()[:60])

    task = _BgTask(uuid.uuid4().hex[:12], desc, secs)
    argv = _shell_argv()
    if argv is None:
        task.status = STATUS_ERROR
        task.spawn_error = "no POSIX shell available (bash/sh not found)"
    else:
        try:
            task.proc = subprocess.Popen(
                argv + [command],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                start_new_session=True,  # own process group -> killpg works
            )
        except OSError as e:
            task.status = STATUS_ERROR
            task.spawn_error = f"failed to start process: {e}"
        else:
            threading.Thread(target=_pump_pipe,
                             args=(task, task.proc.stdout, "out"),
                             daemon=True, name=f"bgsh-out-{task.id}").start()
            threading.Thread(target=_pump_pipe,
                             args=(task, task.proc.stderr, "err"),
                             daemon=True, name=f"bgsh-err-{task.id}").start()
            threading.Thread(target=_watch, args=(task,),
                             daemon=True, name=f"bgsh-watch-{task.id}").start()
    with _TASKS_LOCK:
        _TASKS[task.id] = task
        _evict_finished_locked()
    return task.id


def _evict_finished_locked() -> None:
    """Cap retained finished tasks so a long session cannot leak them."""
    finished = sorted(
        (t for t in _TASKS.values() if t.status != STATUS_RUNNING),
        key=lambda t: t.started_at)
    for old in finished[:max(0, len(finished) - MAX_FINISHED_TASKS)]:
        _TASKS.pop(old.id, None)


def _render(out: str, err: str) -> str:
    """Merge stdout + stderr into the single output string tools return."""
    if err.strip():
        sep = "" if not out or out.endswith("\n") else "\n"
        return out + sep + "--- stderr ---\n" + err
    return out


def poll(task_id: str) -> dict:
    """Full snapshot: {id, description, status, output_so_far, returncode}."""
    task = _get(task_id)
    if task is None:
        return {"id": task_id, "description": "", "status": STATUS_ERROR,
                "output_so_far":
                    f"ERROR: unknown background task id: {task_id!r}",
                "returncode": None}
    with task._lock:
        out = "".join(task._stdout)
        err = "".join(task._stderr)
        if task._stdout_dropped:
            out = (f"[... {task._stdout_dropped} earlier chars "
                   f"truncated ...]\n" + out)
        if task._stderr_dropped:
            err = (f"[... {task._stderr_dropped} earlier chars "
                   f"truncated ...]\n" + err)
        if task.spawn_error:
            err = task.spawn_error + "\n" + err
        return {"id": task.id, "description": task.description,
                "status": task.status,
                "output_so_far": _render(out, err),
                "returncode": task.returncode}


def _read_new(task: _BgTask) -> tuple[str, str]:
    """New (stdout, stderr) text since the last read; advances the offsets."""
    with task._lock:
        out_full = "".join(task._stdout)
        err_full = "".join(task._stderr)
        # offsets are absolute stream positions; the retained tail starts
        # at _stdout_dropped/_stderr_dropped
        out_start = max(task._stdout_seen, task._stdout_dropped)
        err_start = max(task._stderr_seen, task._stderr_dropped)
        new_out = out_full[out_start - task._stdout_dropped:]
        new_err = err_full[err_start - task._stderr_dropped:]
        task._stdout_seen = task._stdout_dropped + len(out_full)
        task._stderr_seen = task._stderr_dropped + len(err_full)
    return new_out, new_err


def kill(task_id: str) -> bool:
    """Terminate the task's process group. True if a kill was attempted."""
    task = _get(task_id)
    if task is None:
        return False
    with task._lock:
        if task.status != STATUS_RUNNING or task.proc is None:
            return False
        task._kill_requested = True
    _kill_group(task)
    return True


def list_background_tasks() -> list[dict]:
    """TUI contract — list of {id, description, status, started_at}.

    ``started_at`` is a float epoch. Keys are exactly these four.
    """
    with _TASKS_LOCK:
        tasks = sorted(_TASKS.values(), key=lambda t: t.started_at)
    result = []
    for t in tasks:
        with t._lock:
            result.append({"id": t.id, "description": t.description,
                           "status": t.status, "started_at": t.started_at})
    return result


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------


def _handle_bash_bg(**kwargs: Any) -> str:
    command = kwargs.get("command", "")
    description = kwargs.get("description", "")
    timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
    try:
        task_id = launch(command, description, timeout)
    except ValueError as e:
        return f"ERROR: {e}"
    task = _get(task_id)
    desc = task.description if task else str(command)[:60]
    return (f"background task started: {task_id}\n"
            f"description: {desc}\n"
            "Poll it with the BashOutput tool using this task id.")


def _handle_bash_output(**kwargs: Any) -> str:
    task_id = kwargs.get("task_id", "")
    tail_lines = kwargs.get("tail_lines")
    task = _get(task_id)
    if task is None:
        return f"ERROR: unknown background task id: {task_id!r}"
    new_out, new_err = _read_new(task)
    with task._lock:
        full_out = "".join(task._stdout)
        status = task.status
        rc = task.returncode
    if tail_lines is not None:
        try:
            n = int(tail_lines)
        except (TypeError, ValueError):
            return (f"ERROR: tail_lines must be an integer, "
                    f"got {tail_lines!r}")
        n = max(0, min(n, 2000))
        with task._lock:
            err_text = "".join(task._stderr)
            spawn_err = task.spawn_error
        combined = ""
        if spawn_err:
            combined += spawn_err + "\n"
        combined += full_out
        if err_text:
            combined += ("\n" if not combined.endswith("\n") else "") + err_text
        lines = combined.splitlines(keepends=True)
        body = "".join(lines[-n:]) if n else ""
    else:
        body = _render(new_out, new_err)
    if not body.strip():
        body = "(no new output)"
    footer = f"\n[status: {status}"
    if rc is not None:
        footer += f", returncode: {rc}"
    footer += "]"
    return body + footer


def register(agent: Any) -> None:
    """Wire BashBG / BashOutput into an agent (duck-typed, no imports)."""
    agent.tools["BashBG"] = Tool(
        name="BashBG",
        description=(
            "Launch a shell command in the background and return its task "
            "id. Use for long builds, test suites, dev servers, or anything "
            "that would block run_command. Poll it with BashOutput using "
            "the returned task id."),
        parameters={"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "shell command to run in background"},
            "description": {"type": "string",
                            "description": "short label shown in task lists"},
            "timeout": {"type": "number",
                        "description": "seconds before auto-kill, default 600"}},
            "required": ["command"]},
        handler=_handle_bash_bg,
        risk=RISK_CONFIRM,
    )
    agent.tools["BashOutput"] = Tool(
        name="BashOutput",
        description=(
            "Read new output from a background task started by BashBG. "
            "Returns only the output since the last BashOutput poll, plus "
            "the task status. Pass tail_lines to get the last N lines of "
            "the full output instead."),
        parameters={"type": "object", "properties": {
            "task_id": {"type": "string",
                        "description": "task id returned by BashBG"},
            "tail_lines": {"type": "integer",
                           "description": "last N lines of full output"}},
            "required": ["task_id"]},
        handler=_handle_bash_output,
        risk=RISK_SAFE,
    )


# ---------------------------------------------------------------------------
# Wiring snippet for fullagent/agent.py (see module docs / __main__ self-test)
#
# Import at the top of agent.py:
#     from . import bgsh
#
# Call once when the agent's tools are set up (next to other optional
# module registrations):
#     bgsh.register(agent)   # adds BashBG / BashOutput tools
#
# ---------------------------------------------------------------------------
# Snippet for run_command in fullagent/tools.py to gain run_in_background
# (report-only: do NOT edit tools.py):
#
#     def run_command(command: str, timeout: int = 120,
#                     on_output=None, should_cancel=None,
#                     run_in_background: bool = False,
#                     description: str = "") -> str:
#         ...
#         if run_in_background:
#             from . import bgsh
#             task_id = bgsh.launch(command, description or command[:60],
#                                   timeout=timeout)
#             return (f"background task started: {task_id}\n"
#                     "poll with BashOutput using this task id.")
#         command, secs, err = _validate_shell_args(command, timeout)
#         ...  # rest unchanged
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Launch/poll/kill/list smoke test. Prints PASS lines, returns exit code."""
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    def wait_status(tid: str, want_gone: bool = True, limit: float = 15.0):
        deadline = time.time() + limit
        while time.time() < deadline:
            s = poll(tid)["status"]
            if want_gone and s != STATUS_RUNNING:
                return s
            if not want_gone and s == STATUS_RUNNING:
                return s
            time.sleep(0.1)
        return poll(tid)["status"]

    # --- 1. launch + incremental output -----------------------------------
    tid = launch("sleep 2 && echo hello", "selftest echo", timeout=30)
    check("launch returns id", isinstance(tid, str) and bool(tid))
    s1 = poll(tid)
    check("poll while running", s1["status"] == STATUS_RUNNING
          and s1["id"] == tid and s1["returncode"] is None)
    time.sleep(1.0)
    early, _ = _read_new(_TASKS[tid])
    check("incremental: nothing before echo", early == "")
    st = wait_status(tid)
    s2 = poll(tid)
    check("finishes done", st == STATUS_DONE and s2["status"] == STATUS_DONE)
    check("returncode 0", s2["returncode"] == 0)
    check("output_so_far has hello", "hello" in s2["output_so_far"])
    got, _ = _read_new(_TASKS[tid])
    check("incremental: hello delivered once", "hello" in got)
    got2, _ = _read_new(_TASKS[tid])
    check("incremental: second read empty", got2 == "")

    # --- 2. kill ------------------------------------------------------------
    tid2 = launch("sleep 60", "selftest kill", timeout=120)
    pid2 = _TASKS[tid2].proc.pid
    time.sleep(0.5)
    check("poll sees running sleep", poll(tid2)["status"] == STATUS_RUNNING)
    check("kill returns True", kill(tid2) is True)
    st2 = wait_status(tid2)
    check("status killed after kill", st2 == STATUS_KILLED)
    try:
        os.kill(pid2, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True
    check("process group actually dead", not alive)
    check("kill twice returns False", kill(tid2) is False)
    check("kill unknown returns False", kill("nope") is False)

    # --- 3. timeout watchdog ------------------------------------------------
    tid3 = launch("sleep 30", "selftest timeout", timeout=1)
    st3 = wait_status(tid3)
    s3 = poll(tid3)
    check("timeout kills task", st3 == STATUS_KILLED)
    check("timeout note in stderr", "timeout after 1s" in s3["output_so_far"])

    # --- 4. list_background_tasks contract ----------------------------------
    items = list_background_tasks()
    ids = {i["id"] for i in items}
    check("list has all tasks", {tid, tid2, tid3} <= ids)
    check("list keys exact",
          all(set(i.keys()) == {"id", "description", "status", "started_at"}
              for i in items))
    check("started_at float epoch",
          all(isinstance(i["started_at"], float) for i in items))
    check("description kept", any(i["description"] == "selftest echo"
                                  for i in items))

    # --- 5. BashOutput handler: incremental + tail_lines ---------------------
    tid4 = launch("echo one && sleep 1 && echo two", "selftest incremental",
                  timeout=30)
    wait_status(tid4)
    r1 = _handle_bash_output(task_id=tid4)
    check("BashOutput sees both lines", "one" in r1 and "two" in r1
          and "status: done" in r1)
    r2 = _handle_bash_output(task_id=tid4)
    check("BashOutput incremental empty", "no new output" in r2)
    r3 = _handle_bash_output(task_id=tid4, tail_lines=1)
    r3_lines = r3.splitlines()
    # note: "status: done" footer contains "one" as substring — compare lines
    check("BashOutput tail_lines=1",
          "two" in r3_lines and "one" not in r3_lines)
    check("BashOutput unknown id",
          _handle_bash_output(task_id="nope").startswith("ERROR"))
    check("BashBG empty command",
          _handle_bash_bg(command="   ").startswith("ERROR"))

    # --- 6. register() -------------------------------------------------------
    class _FakeAgent:
        pass
    fake = _FakeAgent()
    fake.tools = {}
    register(fake)
    check("register adds both tools",
          "BashBG" in fake.tools and "BashOutput" in fake.tools)
    check("tool schemas valid",
          fake.tools["BashBG"].openai_schema()["function"]["name"] == "BashBG"
          and fake.tools["BashOutput"].openai_schema()["type"] == "function")

    print("PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
