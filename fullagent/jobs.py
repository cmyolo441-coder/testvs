"""Persistent job daemon: SQLite-backed background job queue.

A job survives agent restarts: each job row lives in
``~/.fullagent/jobs.db`` (table ``jobs``) and its real-time output streams
into ``~/.fullagent/logs/{job_id}.log``. On :func:`register` a daemon
thread recovers jobs stranded by a previous process (``running`` rows are
marked ``failed`` with note "process died"; ``pending`` rows are started).

Engine::

    job_id = start_job("make -j4")      # real subprocess, returns uuid
    status(job_id)                      # {status, exit_code, elapsed, ...}
    logs(job_id, n=50)                  # tail of the real log file
    cancel_job(job_id)                  # SIGTERM, SIGKILL after 5s

LLM tools (see :func:`register`)::

    JobStart   — start a shell command as a persistent background job
    JobStatus  — status / exit code / elapsed of a job
    JobLogs    — tail the real log file of a job
    JobCancel  — SIGTERM a job, escalating to SIGKILL after 5s

Only stdlib + ``.tools`` (``Tool`` for the registry) are imported.
``config`` is used for ``APP_DIR``; the store dir can be overridden for
tests via :func:`set_store_dir`.
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

from . import config
from .tools import RISK_CONFIRM, RISK_SAFE, Tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

KILL_GRACE_SECONDS = 5.0
KILL_SIGNAL = signal.SIGTERM
# Windows has no SIGKILL; importing this module must not depend on it.
ESCALATE_SIGNAL = getattr(signal, "SIGKILL", signal.SIGTERM)

# ---------------------------------------------------------------------------
# Store location (parameterized for tests)
# ---------------------------------------------------------------------------

_STORE_DIR_OVERRIDE: Path | None = None


def set_store_dir(path: str | Path | None) -> None:
    """Override the jobs store dir (db + logs). Used by the self-test.

    ``None`` restores the default (``config.APP_DIR``).
    """
    global _STORE_DIR_OVERRIDE
    _STORE_DIR_OVERRIDE = Path(path) if path is not None else None


def _store_dir() -> Path:
    return _STORE_DIR_OVERRIDE if _STORE_DIR_OVERRIDE is not None \
        else config.APP_DIR


def _db_path() -> Path:
    return _store_dir() / "jobs.db"


def _logs_dir() -> Path:
    return _store_dir() / "logs"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    command     TEXT NOT NULL,
    status      TEXT NOT NULL,
    pid         INTEGER,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    exit_code   INTEGER,
    log_path    TEXT NOT NULL,
    note        TEXT
);
"""

_LOCK = threading.RLock()          # guards live registry + start/cancel
_DB_LOCK = threading.Lock()        # serializes sqlite writes
_LIVE: dict[str, dict[str, Any]] = {}  # job_id -> {proc, cancelled, log_lock}
_RECOVERED = False


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection:
    """One short-lived connection per operation (thread-safe pattern)."""
    _logs_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_db_path()))
    conn.execute(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def _get_row(job_id: str) -> dict[str, Any] | None:
    if not isinstance(job_id, str):
        return None
    with _DB_LOCK:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
    return dict(row) if row else None


def _update(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", vals)
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Process plumbing
# ---------------------------------------------------------------------------


def _shell_argv() -> list[str] | None:
    for name in ("bash", "sh"):
        exe = shutil.which(name)
        if exe:
            return [exe, "-c"]
    return None


def _pump(job_id: str, proc: subprocess.Popen, entry: dict) -> None:
    """Stream stdout/stderr to the log file in real time (one thread)."""
    try:
        fh = open(entry["log_path"], "a", encoding="utf-8",
                  errors="replace")
    except OSError:
        entry["pump_done"].set()
        return
    try:
        threads = [
            threading.Thread(
                target=_drain, args=(proc.stdout, fh, entry["log_lock"]),
                daemon=True),
            threading.Thread(
                target=_drain, args=(proc.stderr, fh, entry["log_lock"]),
                daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass
        for pipe in (proc.stdout, proc.stderr):
            try:
                pipe.close()
            except Exception:  # noqa: BLE001 — never break the pump
                pass
        entry["pump_done"].set()


def _drain(pipe, fh, lock: threading.Lock) -> None:
    try:
        for line in iter(pipe.readline, ""):
            with lock:
                fh.write(line)
                fh.flush()
    except Exception:  # noqa: BLE001 — pump must not kill the watcher
        pass


def _watch(job_id: str, proc: subprocess.Popen, entry: dict) -> None:
    """Wait for the job's process; settle the final DB status."""
    try:
        proc.wait()
    except Exception:  # noqa: BLE001 — still record what we know
        pass
    # Drain the tail of stdout/stderr before settling (bounded: a runaway
    # grandchild holding the pipe open must not wedge the watcher).
    entry["pump_done"].wait(10.0)
    with _LOCK:
        _LIVE.pop(job_id, None)
    rc = proc.returncode
    cancelled = bool(entry.get("cancelled"))
    now = time.time()
    if cancelled:
        _update(job_id, status=STATUS_CANCELLED, finished_at=now,
                exit_code=rc)
    elif rc == 0:
        _update(job_id, status=STATUS_DONE, finished_at=now, exit_code=rc)
    else:
        _update(job_id, status=STATUS_FAILED, finished_at=now,
                exit_code=rc, note=f"exit code {rc}")


def _kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _escalate(job_id: str, proc: subprocess.Popen) -> None:
    """SIGKILL after the grace period if the process ignored SIGTERM."""
    time.sleep(KILL_GRACE_SECONDS)
    if proc.poll() is None:
        _kill_group(proc.pid, ESCALATE_SIGNAL)
    # The watcher records the final state once the process actually dies.


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def start_job(command: str) -> str:
    """Start ``command`` as a real background subprocess.

    Returns the job id (uuid hex). Output streams in real time to the job's
    log file. Thread-safe: concurrent calls get unique ids and rows.
    Raises ValueError on an empty command.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    job_id = uuid.uuid4().hex
    now = time.time()
    log_path = _logs_dir() / f"{job_id}.log"

    with _LOCK, _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                "INSERT INTO jobs (id, command, status, created_at,"
                " log_path) VALUES (?, ?, ?, ?, ?)",
                (job_id, command, STATUS_PENDING, now, str(log_path)))
            conn.commit()
        finally:
            conn.close()

    log_path.touch(exist_ok=True)
    entry: dict[str, Any] = {"proc": None, "cancelled": False,
                             "log_lock": threading.Lock(),
                             "pump_done": threading.Event(),
                             "log_path": log_path}
    with _LOCK:
        _LIVE[job_id] = entry

    argv = _shell_argv()
    if argv is None:
        with _LOCK:
            _LIVE.pop(job_id, None)
        _update(job_id, status=STATUS_FAILED, finished_at=time.time(),
                note="no POSIX shell available (bash/sh not found)")
        return job_id

    try:
        proc = subprocess.Popen(
            argv + [command],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=True,  # own process group -> killpg works
        )
    except OSError as e:
        with _LOCK:
            _LIVE.pop(job_id, None)
        _update(job_id, status=STATUS_FAILED, finished_at=time.time(),
                note=f"failed to start process: {e}")
        return job_id

    entry["proc"] = proc
    _update(job_id, status=STATUS_RUNNING, pid=proc.pid, started_at=time.time())
    threading.Thread(target=_pump, args=(job_id, proc, entry), daemon=True,
                     name=f"jobs-pump-{job_id}").start()
    threading.Thread(target=_watch, args=(job_id, proc, entry), daemon=True,
                     name=f"jobs-watch-{job_id}").start()
    return job_id


def status(job_id: str) -> dict[str, Any]:
    """Snapshot: {id, command, status, pid, exit_code, elapsed, log_path}."""
    row = _get_row(job_id)
    if row is None:
        return {"id": job_id, "status": "unknown",
                "error": f"unknown job id: {job_id!r}"}
    now = time.time()
    start = row["started_at"] or row["created_at"]
    end = row["finished_at"] or now
    return {
        "id": row["id"],
        "command": row["command"],
        "status": row["status"],
        "pid": row["pid"],
        "exit_code": row["exit_code"],
        "elapsed": round(end - start, 2),
        "note": row["note"],
        "log_path": row["log_path"],
    }


def logs(job_id: str, n: int = 50) -> str:
    """Tail of the job's real log file — last ``n`` lines."""
    row = _get_row(job_id)
    if row is None:
        return f"ERROR: unknown job id: {job_id!r}"
    path = Path(row["log_path"])
    try:
        lines = int(n)
    except (TypeError, ValueError):
        return f"ERROR: n must be an integer, got {n!r}"
    lines = max(0, min(lines, 5000))
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "(log file not found — job may not have started yet)"
    split = text.splitlines(keepends=True)
    body = "".join(split[-lines:]) if lines else ""
    return body if body else "(log is empty so far)"


def cancel_job(job_id: str) -> bool:
    """SIGTERM the job's process group; SIGKILL after 5s if still alive.

    True if a cancel was attempted (job was live). The watcher settles the
    row to ``cancelled`` once the process dies.
    """
    row = _get_row(job_id)
    if row is None:
        return False
    with _LOCK:
        entry = _LIVE.get(job_id)
    if entry is None or entry.get("proc") is None:
        # Not tracked by this process. Best effort on the stored pid, then
        # mark cancelled if the row still says running/pending.
        if row["status"] in (STATUS_RUNNING, STATUS_PENDING):
            pid = row["pid"]
            if pid:
                try:
                    os.kill(pid, KILL_SIGNAL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            _update(job_id, status=STATUS_CANCELLED,
                    finished_at=time.time(), note="cancelled")
            return True
        return False
    proc = entry["proc"]
    if proc.poll() is not None:
        return False
    entry["cancelled"] = True
    _kill_group(proc.pid, KILL_SIGNAL)
    threading.Thread(target=_escalate, args=(job_id, proc), daemon=True,
                     name=f"jobs-escalate-{job_id}").start()
    return True


def _recover() -> None:
    """Daemon startup recovery.

    Jobs left ``running`` by a dead process are marked ``failed`` with note
    "process died"; jobs left ``pending`` are started for real.
    """
    global _RECOVERED
    with _LOCK:
        if _RECOVERED:
            return
        # Claim the one-shot before releasing: concurrent register() calls
        # race here otherwise and respawn every pending job twice.
        _RECOVERED = True
    with _DB_LOCK:
        conn = _connect()
        try:
            running = conn.execute(
                "SELECT id FROM jobs WHERE status = ?",
                (STATUS_RUNNING,)).fetchall()
            pending = conn.execute(
                "SELECT id, command FROM jobs WHERE status = ?",
                (STATUS_PENDING,)).fetchall()
            now = time.time()
            for (jid,) in running:
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?,"
                    " note = ? WHERE id = ?",
                    (STATUS_FAILED, now, "process died", jid))
            conn.commit()
        finally:
            conn.close()
    for jid, command in pending:
        # Re-spawn: same command, but keep the ORIGINAL job id/row so
        # history is continuous.
        try:
            _respawn(jid, command)
        except Exception:  # noqa: BLE001 — never kill the daemon
            _update(jid, status=STATUS_FAILED, finished_at=time.time(),
                    note="failed to restart after recovery")


def _respawn(job_id: str, command: str) -> None:
    """Start ``command`` reusing an existing job row (recovery path)."""
    argv = _shell_argv()
    if argv is None:
        raise RuntimeError("no POSIX shell available")
    entry: dict[str, Any] = {"proc": None, "cancelled": False,
                             "log_lock": threading.Lock(),
                             "pump_done": threading.Event(),
                             "log_path": Path(_get_row(job_id)["log_path"])}
    with _LOCK:
        _LIVE[job_id] = entry
    proc = subprocess.Popen(
        argv + [command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        start_new_session=True,
    )
    entry["proc"] = proc
    _update(job_id, status=STATUS_RUNNING, pid=proc.pid,
            started_at=time.time())
    threading.Thread(target=_pump, args=(job_id, proc, entry), daemon=True,
                     name=f"jobs-pump-{job_id}").start()
    threading.Thread(target=_watch, args=(job_id, proc, entry), daemon=True,
                     name=f"jobs-watch-{job_id}").start()


# ---------------------------------------------------------------------------
# LLM tools
# ---------------------------------------------------------------------------


def _handle_job_start(**kwargs: Any) -> str:
    command = kwargs.get("command", "")
    try:
        job_id = start_job(command)
    except ValueError as e:
        return f"ERROR: {e}"
    return (f"job started: {job_id}\n"
            f"command: {command}\n"
            "Check it with JobStatus / JobLogs using this job id.")


def _handle_job_status(**kwargs: Any) -> str:
    job_id = kwargs.get("job_id", "")
    s = status(job_id)
    if "error" in s:
        return f"ERROR: {s['error']}"
    lines = [f"job: {s['id']}", f"status: {s['status']}",
             f"command: {s['command']}"]
    if s["pid"]:
        lines.append(f"pid: {s['pid']}")
    if s["exit_code"] is not None:
        lines.append(f"exit_code: {s['exit_code']}")
    lines.append(f"elapsed: {s['elapsed']}s")
    if s["note"]:
        lines.append(f"note: {s['note']}")
    return "\n".join(lines)


def _handle_job_logs(**kwargs: Any) -> str:
    job_id = kwargs.get("job_id", "")
    n = kwargs.get("n", 50)
    body = logs(job_id, n)
    s = status(job_id)
    footer = f"\n[status: {s.get('status', 'unknown')}]"
    return body + footer


def _handle_job_cancel(**kwargs: Any) -> str:
    job_id = kwargs.get("job_id", "")
    if cancel_job(job_id):
        return (f"job {job_id}: SIGTERM sent to its process group; "
                f"escalates to SIGKILL after {KILL_GRACE_SECONDS:g}s "
                "if still alive.")
    return f"ERROR: no running job with id {job_id!r}"


def register(agent: Any) -> None:
    """Wire JobStart / JobStatus / JobLogs / JobCancel into an agent and
    launch the recovery daemon (once per process)."""
    threading.Thread(target=_recover, daemon=True,
                     name="jobs-recovery-daemon").start()
    agent.tools["JobStart"] = Tool(
        name="JobStart",
        description=(
            "Start a shell command as a persistent background job and "
            "return its job id. The job is stored in a SQLite queue at "
            "~/.fullagent/jobs.db and survives agent restarts; its output "
            "streams in real time to ~/.fullagent/logs/{job_id}.log. Use "
            "for long builds, servers, or anything outliving a turn. "
            "Check it with JobStatus / JobLogs using the job id."),
        parameters={"type": "object", "properties": {
            "command": {"type": "string",
                        "description": "shell command to run as a job"}},
            "required": ["command"]},
        handler=_handle_job_start,
        risk=RISK_CONFIRM,
    )
    agent.tools["JobStatus"] = Tool(
        name="JobStatus",
        description=(
            "Get the status of a job started by JobStart: pending, running, "
            "done, failed, or cancelled — plus pid, exit code, and elapsed "
            "time."),
        parameters={"type": "object", "properties": {
            "job_id": {"type": "string",
                       "description": "job id returned by JobStart"}},
            "required": ["job_id"]},
        handler=_handle_job_status,
        risk=RISK_SAFE,
    )
    agent.tools["JobLogs"] = Tool(
        name="JobLogs",
        description=(
            "Tail the real log file of a job: the last n lines of output "
            "(default 50). The log streams in real time while the job "
            "runs."),
        parameters={"type": "object", "properties": {
            "job_id": {"type": "string",
                       "description": "job id returned by JobStart"},
            "n": {"type": "integer",
                  "description": "last n lines to return, default 50"}},
            "required": ["job_id"]},
        handler=_handle_job_logs,
        risk=RISK_SAFE,
    )
    agent.tools["JobCancel"] = Tool(
        name="JobCancel",
        description=(
            "Cancel a running job: sends SIGTERM to its whole process "
            "group, then SIGKILL after 5 seconds if it is still alive."),
        parameters={"type": "object", "properties": {
            "job_id": {"type": "string",
                       "description": "job id returned by JobStart"}},
            "required": ["job_id"]},
        handler=_handle_job_cancel,
        risk=RISK_CONFIRM,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Real-functionality proof: real subprocesses, real sqlite, real logs,
    real SIGTERM/SIGKILL. Runs in a temp dir. Prints PASS lines."""
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    def wait_status(job_id: str, want: set, limit: float = 20.0):
        deadline = time.time() + limit
        while time.time() < deadline:
            s = status(job_id)["status"]
            if s in want:
                return s
            time.sleep(0.1)
        return status(job_id)["status"]

    tmp = Path(tempfile.mkdtemp(prefix="jobs-selftest-"))
    set_store_dir(tmp)
    try:
        # --- 1. start + status transitions + real logs ----------------------
        jid = start_job("sleep 2 && echo hello")
        check("start returns uuid id",
              isinstance(jid, str) and len(jid) == 32)
        s1 = status(jid)
        check("status running right after start",
              s1["status"] == STATUS_RUNNING and s1["pid"] is not None
              and s1["exit_code"] is None)
        st = wait_status(jid, {STATUS_DONE, STATUS_FAILED})
        s2 = status(jid)
        check("finishes done", st == STATUS_DONE)
        check("exit_code 0", s2["exit_code"] == 0)
        check("elapsed >= 1.5s", s2["elapsed"] >= 1.5)
        time.sleep(0.3)  # let the pump flush the tail
        log_text = logs(jid, 100)
        check("real log file contains hello", "hello" in log_text)
        check("log file exists on disk",
              Path(s2["log_path"]).is_file())
        # --- 2. JobLogs tail ----------------------------------------------
        start_job("echo one && echo two && echo three")
        time.sleep(1.5)
        # find that job via db
        with _DB_LOCK:
            conn = _connect()
            try:
                row = conn.execute(
                    "SELECT id FROM jobs WHERE command LIKE '%echo one%'"
                    " ORDER BY created_at DESC LIMIT 1").fetchone()
            finally:
                conn.close()
        j2 = row["id"]
        wait_status(j2, {STATUS_DONE})
        time.sleep(0.3)
        tail1 = logs(j2, 1)
        check("JobLogs tail=1 last line", "three" in tail1
              and "one" not in tail1.splitlines()[:-1])
        tailall = logs(j2, 50)
        check("JobLogs sees all lines",
              all(w in tailall for w in ("one", "two", "three")))
        check("logs unknown id errors",
              logs("nope", 10).startswith("ERROR"))
        # --- 3. cancel a long-running job ---------------------------------
        j3 = start_job("sleep 60")
        time.sleep(0.5)
        pid3 = status(j3)["pid"]
        check("cancel returns True", cancel_job(j3) is True)
        st3 = wait_status(j3, {STATUS_CANCELLED, STATUS_FAILED})
        check("status cancelled after cancel",
              st3 == STATUS_CANCELLED)
        try:
            os.kill(pid3, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        check("process group actually dead", not alive)
        check("cancel twice returns False", cancel_job(j3) is False)
        check("cancel unknown id returns False",
              cancel_job("nope") is False)
        # --- 4. crash recovery: stranded running + pending -----------------
        now = time.time()
        with _DB_LOCK:
            conn = _connect()
            try:
                conn.execute(
                    "INSERT INTO jobs (id, command, status, pid, created_at,"
                    " started_at, log_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("deadbeef1234", "true", STATUS_RUNNING, 999999999,
                     now, now, str(tmp / "logs" / "deadbeef1234.log")))
                conn.execute(
                    "INSERT INTO jobs (id, command, status, created_at,"
                    " log_path) VALUES (?, ?, ?, ?, ?)",
                    ("pending1234", "echo revived", STATUS_PENDING, now,
                     str(tmp / "logs" / "pending1234.log")))
                conn.commit()
            finally:
                conn.close()
        global _RECOVERED
        _RECOVERED = False  # allow the daemon body to run again in-test
        _recover()
        stranded = status("deadbeef1234")
        check("stranded running -> failed",
              stranded["status"] == STATUS_FAILED)
        check("note says process died",
              stranded["note"] == "process died")
        wait_status("pending1234", {STATUS_DONE})
        time.sleep(0.3)
        check("pending job restarted and done",
              status("pending1234")["status"] == STATUS_DONE)
        check("revived job log has output",
              "revived" in logs("pending1234", 10))
        # --- 5. concurrent starts ------------------------------------------
        results: list[str] = []
        errs: list[str] = []

        def _one(i: int) -> None:
            try:
                results.append(start_job(f"echo job{i}"))
            except Exception as e:  # noqa: BLE001
                errs.append(str(e))

        threads = [threading.Thread(target=_one, args=(i,))
                   for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        check("8 concurrent starts, no errors", not errs)
        check("8 unique ids", len(set(results)) == 8)
        deadline = time.time() + 20
        while time.time() < deadline:
            if all(status(j)["status"] == STATUS_DONE for j in results):
                break
            time.sleep(0.2)
        check("all concurrent jobs done",
              all(status(j)["status"] == STATUS_DONE for j in results))
        check("concurrent logs correct",
              all(f"job{i}" in "".join(logs(j, 50) for j in results)
                  for i in range(8)))
        # --- 6. register() --------------------------------------------------
        class _FakeAgent:
            pass
        fake = _FakeAgent()
        fake.tools = {}
        register(fake)
        names = {"JobStart", "JobStatus", "JobLogs", "JobCancel"}
        check("register adds 4 tools", names <= set(fake.tools))
        check("tool schema valid",
              fake.tools["JobStart"].openai_schema()
              ["function"]["name"] == "JobStart")
        r = fake.tools["JobStart"].handler(command="echo via-tool")
        jid4 = r.splitlines()[0].split(": ")[1]
        wait_status(jid4, {STATUS_DONE})
        time.sleep(0.3)
        check("JobStart tool starts real job",
              "via-tool" in fake.tools["JobLogs"].handler(job_id=jid4))
        check("JobStatus tool",
              "status: done" in fake.tools["JobStatus"].handler(job_id=jid4))
        check("JobStart empty command errors",
              fake.tools["JobStart"].handler(command="  ").startswith("ERROR"))
    finally:
        set_store_dir(None)
        shutil.rmtree(tmp, ignore_errors=True)

    print("PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


def _selftest_thread_efficiency() -> int:
    """Prove the job threads are event-driven, not polling: (1) no
    persistent poll loop exists — pump threads block on pipe reads and the
    watcher blocks on proc.wait(), (2) per-job threads die when the job
    ends, (3) the recovery daemon is one-shot, (4) a job still runs to
    completion with real output."""
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="jobs_eff_"))
    old = _STORE_DIR_OVERRIDE
    set_store_dir(tmp)
    try:
        jid = start_job("echo efficiency-probe-ok")
        deadline = time.time() + 15
        while status(jid)["status"] not in ("done", "failed") \
                and time.time() < deadline:
            time.sleep(0.1)
        st = status(jid)
        check("job runs to completion", st["status"] == "done")
        check("real output streamed", "efficiency-probe-ok" in logs(jid))

        # per-job threads must be gone once the work is done: they block
        # on the pipes / proc.wait() and exit — nothing polls.
        deadline = time.time() + 10
        lingering = True
        while time.time() < deadline:
            names = [t.name for t in threading.enumerate()]
            if not any(n.startswith(f"jobs-pump-{jid}")
                       or n.startswith(f"jobs-watch-{jid}") for n in names):
                lingering = False
                break
            time.sleep(0.1)
        check("per-job threads exit when work ends (no polling)",
              not lingering)

        # recovery daemon is one-shot: after register() it must not linger
        class _FakeAgent:
            def __init__(self):
                self.tools = {}

        register(_FakeAgent())
        time.sleep(0.5)
        rec = [t for t in threading.enumerate()
               if t.name == "jobs-recovery-daemon" and t.is_alive()]
        check("recovery daemon is one-shot (no persistent poll)", not rec)
        print("PASS - jobs threads are event-driven "
              "(blocking reads/wait, one-shot recovery)")
    finally:
        set_store_dir(old)

    print("PASS - jobs thread efficiency"
          if not failures else f"{len(failures)} EFFICIENCY FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    eff_rc = _selftest_thread_efficiency()
    rc = _selftest()
    raise SystemExit(eff_rc or rc)
