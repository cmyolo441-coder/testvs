"""Process supervisor: supervised subprocesses with auto-restart.

systemd-style supervision for long-lived commands (dev servers, watchers,
tunnels). Supervised processes run as REAL subprocesses; when one exits
unexpectedly the supervisor thread restarts it with exponential backoff.

Engine::

    sid = add("python server.py", restart_policy="always")  # real subprocess
    list_supervised()   # [{id, command, status, pid, uptime, restarts, ...}]
    stop_supervised(sid)  # SIGTERM -> SIGKILL on the process group
    logs(sid, tail=50)    # last N lines of the process's captured output

Persistence: every state change is written through to SQLite at
``~/.fullagent/supervise.db`` (table ``supervised``). When the agent
restarts, entries left in ``running`` state are re-armed — they died with
the old process, so the supervisor schedules them for immediate restart.

Restart policies: ``always`` (any exit), ``on-failure`` (non-zero exit),
``never`` (leave stopped). Backoff between restarts is exponential:
5s, 10s, 20s, ... capped at 60s. After ``max_restarts`` (default 10) the
entry is marked ``failed`` and left alone.

LLM tools (see :func:`register`)::

    SuperviseAdd   — supervise a command, returns the supervision id
    SuperviseList  — all supervised processes with status/pid/uptime/restarts
    SuperviseStop  — SIGTERM->SIGKILL a supervised process, marks it stopped
    SuperviseLogs  — tail of a supervised process's captured output

Only stdlib + ``.tools`` are imported (``Tool`` for the registry).
"""

from __future__ import annotations

import os
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .tools import RISK_CONFIRM, RISK_SAFE, Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICIES = ("always", "on-failure", "never")

STATUS_RUNNING = "running"
STATUS_PENDING = "pending"    # waiting out backoff before a restart
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

# Supervisor timing. Module-level so the __main__ self-test can shrink them
# (same code paths, just faster). Production values per the spec.
_POLL_INTERVAL = 5.0     # supervisor loop tick
_BACKOFF_BASE = 5.0      # first restart delay; doubles each crash
_BACKOFF_CAP = 60.0      # max delay between restarts

_DEFAULT_MAX_RESTARTS = 10
_STOP_GRACE_SECS = 3.0   # SIGTERM -> SIGKILL grace
_LOG_KEEP_LINES = 2000   # per-process log trim on each (re)start


# ---------------------------------------------------------------------------
# Paths / shell
# ---------------------------------------------------------------------------


def _app_base_dir() -> Path:
    from . import config
    return config.APP_DIR


def _db_path() -> Path:
    """SQLite path. FULLAGENT_SUPERVISE_DB overrides for tests."""
    env = os.environ.get("FULLAGENT_SUPERVISE_DB")
    if env:
        return Path(env).expanduser()
    return _app_base_dir() / "supervise.db"


def _log_dir() -> Path:
    return _app_base_dir() / "supervise_logs"


def _shell_argv() -> list[str] | None:
    for name in ("bash", "sh"):
        exe = shutil.which(name)
        if exe:
            return [exe, "-c"]
    return None


# ---------------------------------------------------------------------------
# SQLite persistence (one short-lived connection per op: thread-safe)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS supervised (
    id             TEXT PRIMARY KEY,
    command        TEXT NOT NULL,
    restart_policy TEXT NOT NULL DEFAULT 'on-failure',
    max_restarts   INTEGER NOT NULL DEFAULT 10,
    restarts       INTEGER NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'running',
    pid            INTEGER,
    exit_code      INTEGER,
    created_at     REAL NOT NULL,
    started_at     REAL,
    next_restart_at REAL,
    backoff_s      REAL,
    note           TEXT
)
"""


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    con.commit()
    return con


def _db_insert(**fields: Any) -> None:
    cols = ", ".join(fields)
    placeholders = ", ".join("?" for _ in fields)
    with _connect() as con:
        con.execute(f"INSERT INTO supervised ({cols}) VALUES ({placeholders})",
                    tuple(fields.values()))
        con.commit()


def _db_update(sid: str, **fields: Any) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as con:
        con.execute(f"UPDATE supervised SET {sets} WHERE id = ?",
                    (*fields.values(), sid))
        con.commit()


def _db_update_guarded(sid: str, expect_status: str, **fields: Any) -> bool:
    """UPDATE only if the row is still in ``expect_status``.

    Returns True when the transition happened. This makes the supervisor
    loop race-safe against a concurrent SuperviseStop: a stop always wins,
    the loop never resurrects a stopped entry.
    """
    if not fields:
        return False
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _connect() as con:
        cur = con.execute(
            f"UPDATE supervised SET {sets} WHERE id = ? AND status = ?",
            (*fields.values(), sid, expect_status))
        con.commit()
        return cur.rowcount > 0


def _db_delete(sid: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM supervised WHERE id = ?", (sid,))
        con.commit()


def _row(sid: Any) -> dict | None:
    if not isinstance(sid, str):
        return None
    with _connect() as con:
        cur = con.execute("SELECT * FROM supervised WHERE id = ?", (sid,))
        r = cur.fetchone()
    return dict(r) if r is not None else None


def _all_rows() -> list[dict]:
    with _connect() as con:
        cur = con.execute("SELECT * FROM supervised ORDER BY created_at")
        rows = [dict(r) for r in cur.fetchall()]
    return rows


# ---------------------------------------------------------------------------
# Process plumbing
# ---------------------------------------------------------------------------


def _trim_log(path: Path, keep: int = _LOG_KEEP_LINES) -> None:
    """Keep only the last ``keep`` lines of an oversized log file."""
    try:
        if not path.exists() or path.stat().st_size < 1_000_000:
            return
        with open(path, "rb") as f:
            lines = f.read().splitlines(keepends=True)
        with open(path, "wb") as f:
            f.writelines(lines[-keep:])
    except OSError:
        pass


def _spawn(sid: str, command: str) -> subprocess.Popen:
    """Start the real subprocess; stdout+stderr append to the per-id log."""
    argv = _shell_argv()
    if argv is None:
        raise OSError("no POSIX shell available (bash/sh not found)")
    logdir = _log_dir()
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / f"{sid}.log"
    _trim_log(log_path)
    header = ("--- supervise start %s cmd=%s ---\n"
              % (time.strftime("%Y-%m-%d %H:%M:%S"), command[:200]))
    with open(log_path, "ab") as f:
        f.write(header.encode("utf-8", "replace"))
        f.flush()
        # Child inherits the fd; parent's copy closes at block exit.
        return subprocess.Popen(argv + [command],
                                stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)


def _kill_group_proc(proc: subprocess.Popen) -> None:
    """SIGTERM the whole process group, then SIGKILL after the grace period."""
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.wait(timeout=_STOP_GRACE_SECS)
        return
    except (subprocess.TimeoutExpired, OSError):
        pass
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _fmt_dur(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Supervisor singleton
# ---------------------------------------------------------------------------


class _Supervisor:
    """Daemon thread that watches supervised processes and restarts them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._pid_history: dict[str, list[int]] = {}
        self._restart_times: dict[str, list[float]] = {}
        self._first_start: dict[str, float] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        """Idempotent: (re-)arm DB entries, start the daemon thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._rearm()
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="supervise")
            self._thread.start()

    def stop(self) -> None:
        """Stop the daemon thread (used by the self-test; production never
        calls this — the thread is a daemon)."""
        with self._lock:
            t = self._thread
            self._thread = None
        self._stop_event.set()
        if t is not None and t.is_alive():
            t.join(timeout=10)
        with self._lock:
            self._procs.clear()

    # -- main loop ------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.wait(_POLL_INTERVAL):
            try:
                self._poll_once()
            except Exception:  # noqa: BLE001 — never kill the supervisor
                pass

    def _poll_once(self) -> None:
        now = time.time()
        for row in _all_rows():
            sid = row["id"]
            if row["status"] == STATUS_RUNNING:
                with self._lock:
                    proc = self._procs.get(sid)
                if proc is None:
                    # Lost the handle but the DB says running: treat as a
                    # crash so the policy still applies.
                    self._on_exit(row, -1, now)
                    continue
                rc = proc.poll()
                if rc is not None:
                    self._on_exit(row, rc, now)
            elif row["status"] == STATUS_PENDING:
                nxt = row["next_restart_at"]
                if nxt is None or now >= nxt:
                    self._restart(row)

    # -- state transitions ----------------------------------------------

    def _on_exit(self, row: dict, rc: int, now: float) -> None:
        sid = row["id"]
        with self._lock:
            proc = self._procs.pop(sid, None)
        policy = row["restart_policy"]
        should_restart = (policy == "always"
                          or (policy == "on-failure" and rc != 0))
        if not should_restart:
            # Guarded: a concurrent SuperviseStop wins over this transition.
            _db_update_guarded(sid, STATUS_RUNNING, status=STATUS_STOPPED,
                               pid=None, exit_code=rc, next_restart_at=None,
                               backoff_s=None)
            return
        if row["restarts"] >= row["max_restarts"]:
            _db_update_guarded(sid, STATUS_RUNNING, status=STATUS_FAILED,
                               pid=None, exit_code=rc, next_restart_at=None,
                               backoff_s=None,
                               note=f"gave up after {row['max_restarts']} "
                                    f"restarts")
            return
        restarts = row["restarts"] + 1
        backoff = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** (restarts - 1)))
        _db_update_guarded(sid, STATUS_RUNNING, status=STATUS_PENDING,
                           pid=None, exit_code=rc, restarts=restarts,
                           next_restart_at=now + backoff, backoff_s=backoff)
        _ = proc  # handle released; stdio is a log file, nothing to close

    def _restart(self, row: dict) -> None:
        sid = row["id"]
        with self._lock:
            if sid in self._procs:
                return  # already running; stale pending row
        try:
            proc = _spawn(sid, row["command"])
        except OSError as e:
            _db_update_guarded(sid, STATUS_PENDING, status=STATUS_FAILED,
                               next_restart_at=None,
                               note=f"spawn failed: {e}")
            return
        now = time.time()
        with self._lock:
            self._procs[sid] = proc
            self._pid_history.setdefault(sid, []).append(proc.pid)
            self._restart_times.setdefault(sid, []).append(now)
        ok = _db_update_guarded(sid, STATUS_PENDING, status=STATUS_RUNNING,
                                pid=proc.pid, started_at=now,
                                next_restart_at=None, backoff_s=None,
                                exit_code=None)
        if not ok:
            # Lost a race with SuperviseStop: kill what we just started so
            # the stop is honored.
            with self._lock:
                self._procs.pop(sid, None)
            _kill_group_proc(proc)

    def _rearm(self) -> None:
        """On (re)start: entries marked running died with the old process —
        schedule them for immediate restart. Pending entries keep their
        restart count but fire now instead of at their old timestamp."""
        now = time.time()
        for row in _all_rows():
            if row["status"] == STATUS_RUNNING:
                _db_update(row["id"], status=STATUS_PENDING, pid=None,
                           next_restart_at=now, backoff_s=0.0,
                           note="re-armed after supervisor restart")
            elif row["status"] == STATUS_PENDING:
                _db_update(row["id"], next_restart_at=now)


_SUP = _Supervisor()


def ensure_started() -> None:
    """Start the supervisor thread (idempotent)."""
    _SUP.start()


# ---------------------------------------------------------------------------
# Engine API (used by the tool handlers and the self-test)
# ---------------------------------------------------------------------------


def add(command: str, restart_policy: str = "on-failure",
        max_restarts: int = _DEFAULT_MAX_RESTARTS) -> str:
    """Supervise ``command`` as a real subprocess. Returns the id.

    Raises ValueError on bad arguments. If the process cannot be spawned,
    the entry is recorded as ``failed`` and RuntimeError is raised.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    policy = str(restart_policy).strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"restart_policy must be one of {POLICIES}, "
                         f"got {restart_policy!r}")
    try:
        mr = int(max_restarts)
    except (TypeError, ValueError):
        raise ValueError(f"max_restarts must be an integer, "
                         f"got {max_restarts!r}")
    if mr < 0:
        raise ValueError("max_restarts must be >= 0")

    ensure_started()
    sid = uuid.uuid4().hex[:12]
    now = time.time()
    # Hold the supervisor lock across insert+spawn+registration: otherwise
    # a poll can see a RUNNING row with no live handle and treat the spawn
    # as a crash (scheduling a restart, or marking it stopped).
    with _SUP._lock:
        _db_insert(id=sid, command=command.strip(), restart_policy=policy,
                   max_restarts=mr, restarts=0, status=STATUS_RUNNING,
                   pid=None, exit_code=None, created_at=now, started_at=now,
                   next_restart_at=None, backoff_s=None, note=None)
        try:
            proc = _spawn(sid, command.strip())
        except OSError as e:
            _db_update(sid, status=STATUS_FAILED, note=f"spawn failed: {e}")
            raise RuntimeError(f"could not start process (id {sid}): {e}")
        _SUP._procs[sid] = proc
        _SUP._pid_history.setdefault(sid, []).append(proc.pid)
        _SUP._first_start[sid] = now
    _db_update(sid, pid=proc.pid)
    return sid


def stop_supervised(sid: str) -> tuple[bool, str]:
    """SIGTERM->SIGKILL the supervised process; mark it stopped.

    Cancels any pending backoff restart too. Returns (ok, message)."""
    row = _row(sid)
    if row is None:
        return False, f"ERROR: unknown supervised id: {sid!r}"
    was_stopped = row["status"] == STATUS_STOPPED
    # Write STOPPED *before* killing: the kill can take seconds, and the
    # poller must not see a missing handle for a still-RUNNING row and
    # schedule a restart that would outlive this stop.
    _db_update(sid, status=STATUS_STOPPED, pid=None,
               next_restart_at=None, backoff_s=None)
    with _SUP._lock:
        proc = _SUP._procs.pop(sid, None)
    killed = False
    if proc is not None and proc.poll() is None:
        _kill_group_proc(proc)
        killed = True
    elif row["pid"] and _alive(row["pid"]):
        # Belt and braces: a pid the DB knows but we lost the handle for.
        try:
            os.killpg(os.getpgid(row["pid"]), signal.SIGKILL)
            killed = True
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if killed:
        return True, f"stopped {sid} (process terminated)"
    if was_stopped:
        return True, f"{sid} was already stopped"
    return True, f"stopped {sid}"


def list_supervised() -> list[dict]:
    """All supervised entries with live status/pid/uptime/restarts."""
    now = time.time()
    out = []
    for r in _all_rows():
        uptime = None
        if r["status"] == STATUS_RUNNING and r["started_at"]:
            uptime = _fmt_dur(now - r["started_at"])
        out.append({
            "id": r["id"], "command": r["command"],
            "restart_policy": r["restart_policy"],
            "max_restarts": r["max_restarts"], "restarts": r["restarts"],
            "status": r["status"], "pid": r["pid"],
            "uptime": uptime, "exit_code": r["exit_code"],
            "next_restart_in": (_fmt_dur(r["next_restart_at"] - now)
                                if r["status"] == STATUS_PENDING
                                and r["next_restart_at"] else None),
        })
    return out


def logs(sid: str, tail: int = 50) -> str:
    """Last ``tail`` lines of the supervised process's captured output."""
    row = _row(sid)
    if row is None:
        return f"ERROR: unknown supervised id: {sid!r}"
    try:
        n = int(tail)
    except (TypeError, ValueError):
        return f"ERROR: tail must be an integer, got {tail!r}"
    n = max(1, min(n, 2000))
    path = _log_dir() / f"{sid}.log"
    if not path.exists():
        return "(no output logged yet)"
    try:
        data = path.read_bytes().decode("utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: could not read log: {e}"
    lines = data.splitlines()
    body = "\n".join(lines[-n:]) or "(log is empty)"
    return (f"[{sid} status={row['status']} pid={row['pid'] or '-'} "
            f"restarts={row['restarts']}]\n{body}")


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------


def _handle_add(**kwargs: Any) -> str:
    command = kwargs.get("command", "")
    policy = kwargs.get("restart_policy", "on-failure")
    max_restarts = kwargs.get("max_restarts", _DEFAULT_MAX_RESTARTS)
    try:
        sid = add(command, policy, max_restarts)
    except (ValueError, RuntimeError) as e:
        return f"ERROR: {e}"
    return (f"supervised process started: {sid}\n"
            f"command: {command}\n"
            f"restart_policy: {policy}\n"
            f"max_restarts: {max_restarts}\n"
            "Check it with SuperviseList, read output with SuperviseLogs.")


def _handle_list(**kwargs: Any) -> str:
    items = list_supervised()
    if not items:
        return "(no supervised processes)"
    out = []
    for it in items:
        pid = it["pid"] if it["pid"] is not None else "-"
        uptime = it["uptime"] or "-"
        line = (f"{it['id']}  [{it['status']}] pid={pid} uptime={uptime} "
                f"restarts={it['restarts']}/{it['max_restarts']} "
                f"policy={it['restart_policy']}")
        if it["exit_code"] is not None:
            line += f" exit_code={it['exit_code']}"
        if it["next_restart_in"]:
            line += f" next_restart_in={it['next_restart_in']}"
        line += f" cmd={it['command'][:80]}"
        out.append(line)
    return "\n".join(out)


def _handle_stop(**kwargs: Any) -> str:
    sid = kwargs.get("id", "")
    ok, msg = stop_supervised(sid)
    return msg


def _handle_logs(**kwargs: Any) -> str:
    return logs(kwargs.get("id", ""), kwargs.get("tail", 50))


def register(agent: Any) -> None:
    """Wire SuperviseAdd/List/Stop/Logs into an agent (duck-typed)."""
    ensure_started()
    agent.tools["SuperviseAdd"] = Tool(
        name="SuperviseAdd",
        description=(
            "Supervise a shell command as a real background process with "
            "auto-restart. Returns a supervision id. restart_policy: "
            "'always' (restart on any exit), 'on-failure' (restart only on "
            "non-zero exit), 'never' (leave stopped). Crashed processes "
            "restart with exponential backoff (5s, 10s, 20s... cap 60s); "
            "after max_restarts the entry is marked failed. Use for dev "
            "servers, watchers, tunnels, or anything that must stay up."),
        parameters={"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "shell command to supervise"},
            "restart_policy": {"type": "string",
                               "enum": list(POLICIES),
                               "description": "default 'on-failure'"},
            "max_restarts": {"type": "integer",
                             "description": "give up after this many "
                                            "restarts, default 10"}},
            "required": ["command"]},
        handler=_handle_add,
        risk=RISK_CONFIRM,
    )
    agent.tools["SuperviseList"] = Tool(
        name="SuperviseList",
        description=(
            "List all supervised processes: id, status, pid, uptime, "
            "restart count, policy, and command."),
        parameters={"type": "object", "properties": {}},
        handler=_handle_list,
        risk=RISK_SAFE,
    )
    agent.tools["SuperviseStop"] = Tool(
        name="SuperviseStop",
        description=(
            "Stop a supervised process by id: SIGTERM then SIGKILL on the "
            "whole process group, and cancel any pending restart. Marks "
            "the entry stopped."),
        parameters={"type": "object", "properties": {
            "id": {"type": "string",
                   "description": "supervision id from SuperviseAdd"}} ,
            "required": ["id"]},
        handler=_handle_stop,
        risk=RISK_CONFIRM,
    )
    agent.tools["SuperviseLogs"] = Tool(
        name="SuperviseLogs",
        description=(
            "Show the tail of a supervised process's captured stdout/stderr."),
        parameters={"type": "object", "properties": {
            "id": {"type": "string",
                   "description": "supervision id from SuperviseAdd"},
            "tail": {"type": "integer",
                     "description": "last N lines, default 50"}} ,
            "required": ["id"]},
        handler=_handle_logs,
        risk=RISK_SAFE,
    )


# ---------------------------------------------------------------------------
# Wiring snippet for fullagent/agent.py (Agent._register_feature_modules)
#
# Add "supervise" to the module-name tuple next to the other feature
# modules; each module is imported and register(agent) is called.
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """REAL supervision self-test. Prints PASS lines, returns exit code.

    Shrinks the supervisor timing constants (same code paths, faster):
    poll 0.5s, backoff base 1s. Proves: crash -> restart with a new pid,
    restart counter increments, backoff delays really happen and grow
    exponentially, max_restarts gives up, stop really kills the process,
    clean exits are not restarted under on-failure, and re-arm on startup
    revives entries left running.
    """
    global _POLL_INTERVAL, _BACKOFF_BASE, _BACKOFF_CAP
    _POLL_INTERVAL, _BACKOFF_BASE, _BACKOFF_CAP = 0.5, 1.0, 30.0

    failures: list[str] = []
    test_ids: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    def wait_for(pred, limit: float, step: float = 0.3) -> bool:
        deadline = time.time() + limit
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(step)
        return pred()

    try:
        # --- 1. register() ------------------------------------------------
        class _FakeAgent:
            pass
        fake = _FakeAgent()
        fake.tools = {}
        register(fake)
        check("register adds 4 tools",
              all(t in fake.tools for t in
                  ("SuperviseAdd", "SuperviseList",
                   "SuperviseStop", "SuperviseLogs")))
        check("tool schemas valid",
              fake.tools["SuperviseAdd"].openai_schema()["function"]["name"]
              == "SuperviseAdd"
              and fake.tools["SuperviseLogs"].openai_schema()["type"]
              == "function")
        check("supervisor thread alive",
              _SUP._thread is not None and _SUP._thread.is_alive())

        # --- 2. crash -> restart with new pid, backoff grows -------------
        sid = add("exit 1", restart_policy="on-failure", max_restarts=4)
        test_ids.append(sid)
        # Wait for restarts to actually FIRE (restarts counter increments at
        # schedule time, up to one backoff period before the new process
        # starts — pid/timestamp history only records fired restarts).
        ok = wait_for(
            lambda: len(_SUP._restart_times.get(sid, [])) >= 4, limit=60)
        check("restarts increment on crash", ok)
        row = _row(sid)
        check("row shows restarts=4",
              row is not None and row["restarts"] == 4)
        pids = _SUP._pid_history.get(sid, [])
        check("new pid on every restart",
              len(pids) >= 5 and len(set(pids)) >= 5)
        times = _SUP._restart_times.get(sid, [])
        check("4 restart timestamps recorded", len(times) >= 4)
        first = _SUP._first_start.get(sid)
        check("first restart delayed by real backoff (>=0.8s)",
              bool(times) and first is not None
              and (times[0] - first) >= 0.8)
        if len(times) >= 4:
            d = [times[i + 1] - times[i] for i in range(3)]
            # expected ~2s, ~4s, ~8s (backoff 1,2,4 + detection jitter)
            check(f"backoff grows exponentially {d[0]:.1f}s {d[1]:.1f}s "
                  f"{d[2]:.1f}s",
                  d[1] > d[0] * 1.3 and d[2] > d[1] * 1.3)
        else:
            check("backoff deltas measurable", False)

        # --- 3. max_restarts -> failed ------------------------------------
        ok = wait_for(
            lambda: (_row(sid) or {}).get("status") == STATUS_FAILED,
            limit=20)
        row = _row(sid)
        check("gives up after max_restarts -> failed",
              ok and row is not None and row["restarts"] == 4
              and row["pid"] is None)

        # --- 4. sleep 30: running, then stop kills it for real -------------
        sid2 = add("sleep 30", restart_policy="always")
        test_ids.append(sid2)
        ok = wait_for(
            lambda: (_row(sid2) or {}).get("status") == STATUS_RUNNING
            and (_row(sid2) or {}).get("pid"), limit=5)
        row2 = _row(sid2)
        check("sleep 30 supervised as running",
              ok and row2 is not None and row2["pid"])
        pid2 = row2["pid"] if row2 else None
        check("process really alive (kill -0)", pid2 and _alive(pid2))
        listing = _handle_list()
        check("SuperviseList shows it",
              sid2 in listing and "running" in listing
              and "restarts=0/10" in listing)
        msg = _handle_stop(id=sid2)
        check("SuperviseStop ok", "stopped" in msg.lower()
              and "ERROR" not in msg)
        time.sleep(1.0)
        check("process really dead after stop",
              pid2 is not None and not _alive(pid2))
        row2b = _row(sid2)
        check("status stopped, pid cleared",
              row2b is not None and row2b["status"] == STATUS_STOPPED
              and row2b["pid"] is None)
        r0 = row2b["restarts"]
        time.sleep(2.5)
        row2c = _row(sid2)
        check("no restart after stop",
              row2c is not None and row2c["status"] == STATUS_STOPPED
              and row2c["restarts"] == r0)

        # --- 5. clean exit under on-failure -> stopped, no restart --------
        sid3 = add("exit 0", restart_policy="on-failure")
        test_ids.append(sid3)
        ok = wait_for(
            lambda: (_row(sid3) or {}).get("status") == STATUS_STOPPED,
            limit=5)
        row3 = _row(sid3)
        check("clean exit not restarted",
              ok and row3 is not None and row3["restarts"] == 0
              and row3["exit_code"] == 0)

        # --- 6. always policy restarts even clean exits --------------------
        sid4 = add("exit 0", restart_policy="always", max_restarts=2)
        test_ids.append(sid4)
        ok = wait_for(
            lambda: (_row(sid4) or {}).get("status") == STATUS_FAILED,
            limit=20)
        check("always policy restarts clean exits too", ok)

        # --- 7. logs -------------------------------------------------------
        out = _handle_logs(id=sid, tail=10)
        check("SuperviseLogs returns log output",
              "supervise start" in out and f"[{sid} " in out)
        check("SuperviseLogs unknown id",
              _handle_logs(id="nope").startswith("ERROR"))
        check("SuperviseLogs bad tail",
              _handle_logs(id=sid, tail="x").startswith("ERROR"))

        # --- 8. validation --------------------------------------------------
        check("add empty command",
              _handle_add(command="   ").startswith("ERROR"))
        check("add bad policy",
              _handle_add(command="true",
                          restart_policy="bogus").startswith("ERROR"))
        check("add bad max_restarts",
              _handle_add(command="true",
                          max_restarts=-1).startswith("ERROR"))
        check("stop unknown id",
              _handle_stop(id="nope").startswith("ERROR"))
        check("stop already stopped",
              "stopped" in _handle_stop(id=sid2).lower())

        # --- 9. re-arm: entries left running survive a restart -------------
        _SUP.stop()
        check("supervisor stopped", not _SUP._thread
              or not _SUP._thread.is_alive())
        ghost = uuid.uuid4().hex[:12]
        _db_insert(id=ghost, command="sleep 60", restart_policy="always",
                   max_restarts=10, restarts=2, status=STATUS_RUNNING,
                   pid=424242, exit_code=None, created_at=time.time(),
                   started_at=time.time(), next_restart_at=None,
                   backoff_s=None, note=None)
        test_ids.append(ghost)
        _SUP.start()  # must re-arm the orphaned "running" entry
        ok = wait_for(
            lambda: (_row(ghost) or {}).get("status") == STATUS_RUNNING
            and (_row(ghost) or {}).get("pid") not in (None, 424242),
            limit=5)
        grow = _row(ghost)
        check("re-arm revives orphaned entry with a real new process",
              ok and grow is not None and grow["pid"]
              and _alive(grow["pid"]))
        check("re-arm keeps restart count", grow is not None
              and grow["restarts"] == 2)
    finally:
        _POLL_INTERVAL, _BACKOFF_BASE, _BACKOFF_CAP = 5.0, 5.0, 60.0
        for tid in test_ids:
            try:
                stop_supervised(tid)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            try:
                _db_delete(tid)
            except Exception:  # noqa: BLE001
                pass
            try:
                (_log_dir() / f"{tid}.log").unlink(missing_ok=True)
            except OSError:
                pass
        try:
            _SUP.stop()
        except Exception:  # noqa: BLE001
            pass

    print("PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


def _selftest_thread_efficiency() -> int:
    """Prove the supervisor thread is cheap when idle and still restarts
    crashed processes at the production 5s tick: (1) interval sane,
    (2) idle wake rate <= 1/s, (3) crash -> restart fires within the
    production tick budget."""
    import tempfile

    global _BACKOFF_BASE
    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="supervise_eff_"))
    old_db = os.environ.get("FULLAGENT_SUPERVISE_DB")
    old_backoff = _BACKOFF_BASE
    os.environ["FULLAGENT_SUPERVISE_DB"] = str(tmp / "sup.db")
    tid = ""
    try:
        # 1. production interval is a sane 5s -> 0.2 wakes/s when idle
        check(f"production poll interval sane ({_POLL_INTERVAL}s)",
              _POLL_INTERVAL == 5.0 and _POLL_INTERVAL >= 1.0)

        ensure_started()
        check("supervisor thread alive", _SUP._thread is not None
              and _SUP._thread.is_alive())

        # 2. idle wake budget: count real ticks over 2.2s
        ticks: list[float] = []
        orig = _SUP._poll_once

        def counting() -> None:
            ticks.append(time.monotonic())
            orig()

        _SUP._poll_once = counting  # type: ignore[method-assign]
        time.sleep(2.2)
        _SUP._poll_once = orig  # type: ignore[method-assign]
        check("idle wake rate <= 1/s "
              f"({len(ticks)} ticks in 2.2s)", len(ticks) <= 2)

        # 3. function still triggers at production timing: a crashing
        #    process is detected and restarted within the tick budget.
        _BACKOFF_BASE = 1.0  # keep the test fast; poll stays at 5s
        tid = add("exit 3", restart_policy="on-failure", max_restarts=1)
        deadline = time.time() + 20
        ok = False
        while time.time() < deadline:
            if len(_SUP._restart_times.get(tid, [])) >= 1:
                ok = True
                break
            time.sleep(0.3)
        check("crash -> restart fires at 5s production tick", ok)
    finally:
        _BACKOFF_BASE = old_backoff
        if tid:
            try:
                stop_supervised(tid)
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            try:
                _db_delete(tid)
            except Exception:  # noqa: BLE001
                pass
        try:
            _SUP.stop()
        except Exception:  # noqa: BLE001
            pass
        if old_db is None:
            os.environ.pop("FULLAGENT_SUPERVISE_DB", None)
        else:
            os.environ["FULLAGENT_SUPERVISE_DB"] = old_db

    print("PASS - supervise thread efficiency"
          if not failures else f"{len(failures)} EFFICIENCY FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    eff_rc = _selftest_thread_efficiency()
    rc = _selftest()
    raise SystemExit(eff_rc or rc)
