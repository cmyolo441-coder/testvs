"""Cron scheduler — real in-process cron with SQLite persistence.

Tools
-----
CronAdd(cron_expr, command) — validate a 5-field cron expression, store it,
    and compute its next run time.
CronList()                  — list all crons with their next run time.
CronRemove(id)              — delete a cron (and its run history).
CronLog(id, limit)          — last N run results for a cron.

A daemon scheduler thread (started in :func:`register`) wakes every 30s,
finds due crons (next_run <= now and enabled), runs each as a REAL
subprocess, records the result in the ``runs`` table, and advances
``next_run``. State lives in ``~/.fullagent/cron.db`` (SQLite), so crons
survive agent restarts: on startup the scheduler recomputes ``next_run``
for every enabled cron relative to now.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from . import config
from .tools import Tool

# Wake interval of the scheduler loop.
POLL_SECONDS = 30
# Hard timeout for every cron command.
RUN_TIMEOUT = 120
# Bytes of subprocess output kept per run.
OUTPUT_TAIL_LIMIT = 8 * 1024
# Default location of the cron database.
DB_PATH = config.APP_DIR / "cron.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crons (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    cron_expr TEXT NOT NULL,
    command   TEXT NOT NULL,
    enabled   INTEGER NOT NULL DEFAULT 1,
    last_run  REAL,
    next_run  REAL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    cron_id    INTEGER NOT NULL,
    started_at REAL NOT NULL,
    exit_code  INTEGER NOT NULL,
    output_tail TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (cron_id) REFERENCES crons(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_runs_cron ON runs(cron_id, started_at DESC);
"""


# ---------------------------------------------------------------------------
# Cron expression parsing
# ---------------------------------------------------------------------------

_FIELD_BOUNDS = (
    (0, 59),   # minute
    (0, 23),   # hour
    (1, 31),   # day of month
    (1, 12),   # month
    (0, 7),    # day of week (0 and 7 are both Sunday)
)


def _parse_field(field: str, lo: int, hi: int) -> set[int]:
    """Parse one cron field into a set of ints.

    Supports ``*``, ``*/n``, ``a-b``, ``a-b/n``, ``a/n`` (a..hi step n),
    single values, and comma-separated lists of any of these.
    """
    field = field.strip()
    if not field:
        raise ValueError("empty field")
    values: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            raise ValueError("empty list item in cron field")
        step = 1
        base = part
        if "/" in part:
            base, step_s = part.split("/", 1)
            base = base.strip()
            try:
                step = int(step_s.strip())
            except ValueError:
                raise ValueError(f"bad step {step_s!r} in {part!r}")
            if step < 1:
                raise ValueError("cron step must be >= 1")
        if base == "*" or base == "":
            # bare "*" — or "*"/step where base became "*"
            if base == "" and "/" not in part:
                raise ValueError(f"bad cron field item {part!r}")
            start, end = lo, hi
        elif "-" in base:
            a_s, b_s = base.split("-", 1)
            try:
                start, end = int(a_s), int(b_s)
            except ValueError:
                raise ValueError(f"bad range {base!r}")
        else:
            try:
                start = int(base)
            except ValueError:
                raise ValueError(f"bad cron field item {base!r}")
            end = hi if "/" in part else start
        if not (lo <= start <= hi and lo <= end <= hi):
            raise ValueError(
                f"value out of range [{lo}-{hi}] in {part!r}")
        if start > end:
            raise ValueError(f"reversed range {base!r}")
        for v in range(start, end + 1, step):
            values.add(v)
    if not values:
        raise ValueError("cron field matched nothing")
    return values


def _parse(expr: str) -> tuple[set[int], set[int], set[int],
                               set[int], set[int], bool, bool]:
    """Parse a 5-field cron expression.

    Returns (minutes, hours, doms, months, dows, dom_restricted,
    dow_restricted). DOW is normalized so 7 == 0 (Sunday).
    """
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(
            f"cron expression needs 5 fields (min hour dom month dow), "
            f"got {len(parts)}: {expr!r}")
    dom_raw, dow_raw = parts[2], parts[4]
    bounds = _FIELD_BOUNDS
    mins = _parse_field(parts[0], *bounds[0])
    hours = _parse_field(parts[1], *bounds[1])
    doms = _parse_field(dom_raw, *bounds[2])
    months = _parse_field(parts[3], *bounds[3])
    dows = _parse_field(dow_raw, *bounds[4])
    # Normalize Sunday: cron allows both 0 and 7.
    if 7 in dows:
        dows.discard(7)
        dows.add(0)
    dom_restricted = dom_raw.strip() != "*"
    dow_restricted = dow_raw.strip() != "*"
    return mins, hours, doms, months, dows, dom_restricted, dow_restricted


def next_run_after(expr: str, after: datetime) -> datetime:
    """Return the first minute > ``after`` matching the cron expression.

    Day-of-month / day-of-week follow standard cron semantics: when both
    are restricted (not ``*``) a day matches if EITHER matches.
    """
    (mins, hours, doms, months, dows,
     dom_restricted, dow_restricted) = _parse(expr)
    t = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = t + timedelta(days=366)
    while t <= limit:
        if (t.minute in mins and t.hour in hours and t.month in months):
            cron_dow = t.isoweekday() % 7  # Sunday == 0
            if dom_restricted and dow_restricted:
                day_ok = (t.day in doms) or (cron_dow in dows)
            elif dom_restricted:
                day_ok = t.day in doms
            elif dow_restricted:
                day_ok = cron_dow in dows
            else:
                day_ok = True
            if day_ok:
                return t
        t += timedelta(minutes=1)
    raise ValueError(f"no run of {expr!r} within a year")


def validate_cron_expr(expr: str) -> str:
    """Validate a cron expression; return normalized form or raise."""
    _parse(expr)  # raises ValueError on anything malformed
    return " ".join(expr.split())


# ---------------------------------------------------------------------------
# Scheduler (SQLite-backed, thread-safe)
# ---------------------------------------------------------------------------

class CronScheduler:
    def __init__(self, db_path: Path | str = DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # One connection for the process; check_same_thread=False with the
        # lock held for every operation makes it thread-safe.
        self._conn = sqlite3.connect(str(self.db_path),
                                     check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        self._stop = threading.Event()
        self._wake = threading.Event()  # poke: instant wake when work arrives
        self._thread: Optional[threading.Thread] = None
        # Survive restarts: recompute next_run from now so stale values
        # (e.g. the machine was down) never fire a flood of backfill runs.
        self.recompute_all()

    # -- low-level helpers -------------------------------------------------
    def _row(self, cron_id: int) -> Optional[sqlite3.Row]:
        cur = self._conn.execute("SELECT * FROM crons WHERE id = ?",
                                 (cron_id,))
        return cur.fetchone()

    # -- public API ---------------------------------------------------------
    def add(self, cron_expr: str, command: str) -> tuple[int, float]:
        expr = validate_cron_expr(cron_expr)
        if not command or not command.strip():
            raise ValueError("command must not be empty")
        now = time.time()
        nxt = next_run_after(expr, datetime.now()).timestamp()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO crons (cron_expr, command, enabled, last_run,"
                " next_run, created_at) VALUES (?, ?, 1, NULL, ?, ?)",
                (expr, command.strip(), nxt, now))
            self._conn.commit()
            cron_id = cur.lastrowid
        self.poke()  # wake the loop so the new cron is checked right away
        return cron_id, nxt

    def list(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM crons ORDER BY next_run")
            return [dict(r) for r in cur.fetchall()]

    def remove(self, cron_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM runs WHERE cron_id = ?",
                                     (cron_id,))
            cur = self._conn.execute("DELETE FROM crons WHERE id = ?",
                                     (cron_id,))
            self._conn.commit()
            removed = cur.rowcount > 0
        if removed:
            self.poke()
        return removed

    def set_enabled(self, cron_id: int, enabled: bool) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE crons SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, cron_id))
            self._conn.commit()
            changed = cur.rowcount > 0
        if changed:
            self.poke()
        return changed

    def runs(self, cron_id: int, limit: int = 5) -> list[dict]:
        limit = max(1, min(50, int(limit)))
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM runs WHERE cron_id = ?"
                " ORDER BY started_at DESC LIMIT ?",
                (cron_id, limit))
            return [dict(r) for r in cur.fetchall()]

    def recompute_all(self) -> int:
        """Recompute next_run for every enabled cron, relative to now."""
        now_dt = datetime.now()
        count = 0
        with self._lock:
            cur = self._conn.execute(
                "SELECT id, cron_expr FROM crons WHERE enabled = 1")
            rows = cur.fetchall()
            for r in rows:
                try:
                    nxt = next_run_after(r["cron_expr"], now_dt).timestamp()
                except ValueError:
                    continue
                self._conn.execute(
                    "UPDATE crons SET next_run = ? WHERE id = ?",
                    (nxt, r["id"]))
                count += 1
            self._conn.commit()
        return count

    # -- execution ----------------------------------------------------------
    def _record_run(self, cron_id: int, started_at: float,
                    exit_code: int, output_tail: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (cron_id, started_at, exit_code,"
                " output_tail) VALUES (?, ?, ?, ?)",
                (cron_id, started_at, exit_code,
                 output_tail[-OUTPUT_TAIL_LIMIT:]))
            self._conn.commit()

    def _run_cron(self, cron: dict) -> None:
        """Run one cron command as a real subprocess; record the result."""
        started_at = time.time()
        try:
            # Own process group: on timeout subprocess.run() killed only
            # the shell, leaving backgrounded children running forever.
            proc = subprocess.Popen(
                cron["command"], shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                **({"start_new_session": True} if os.name == "posix" else {}))
            try:
                stdout, stderr = proc.communicate(timeout=RUN_TIMEOUT)
                exit_code = proc.returncode
                out = (stdout or "") + (stderr or "")
            except subprocess.TimeoutExpired:
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
                    stdout, stderr = proc.communicate(timeout=5)
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
                    stdout, stderr = "", ""
                exit_code = 124
                out = ((stdout or "") + (stderr or "")
                       + f"\n[TIMED OUT after {RUN_TIMEOUT}s]")
        except Exception as e:  # noqa: BLE001 — never let cron die silently
            exit_code = 125
            out = f"[failed to launch: {e}]"
        self._record_run(cron["id"], started_at, exit_code, out)
        # Advance to the next slot strictly after now.
        try:
            nxt = next_run_after(cron["cron_expr"],
                                 datetime.now()).timestamp()
        except ValueError:
            nxt = started_at + 365 * 24 * 3600
        with self._lock:
            self._conn.execute(
                "UPDATE crons SET last_run = ?, next_run = ? WHERE id = ?",
                (started_at, nxt, cron["id"]))
            self._conn.commit()

    def run_due(self, now: Optional[float] = None) -> int:
        """Run every due cron once. Returns number of crons executed.

        The whole sweep holds the lock: a concurrent run_due() (background
        thread racing a manual call) blocks instead of double-firing the
        same cron slot.
        """
        now = time.time() if now is None else now
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM crons WHERE enabled = 1 AND next_run <= ?",
                (now,))
            due = [dict(r) for r in cur.fetchall()]
            for cron in due:
                try:
                    self._run_cron(cron)
                except Exception:  # noqa: BLE001 — one cron must not kill others
                    traceback.print_exc()
        return len(due)

    # -- background thread ---------------------------------------------------
    def _loop(self) -> None:
        # Sleeps on _wake: at most one wake per POLL_SECONDS when idle,
        # instantly when poke()d (add/enable/remove) or stop() is requested.
        # The wait comes FIRST (then clear): a fresh thread never races the
        # caller with an immediate tick, and a stale poke can't trigger
        # work before the first wait.
        while not self._stop.is_set():
            self._wake.clear()
            self._wake.wait(POLL_SECONDS)
            if self._stop.is_set():
                break
            try:
                self.run_due()
            except Exception:  # noqa: BLE001 — scheduler must never die
                traceback.print_exc()

    def poke(self) -> None:
        """Wake the scheduler loop immediately (e.g. after add/enable)."""
        self._wake.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="cronsched",
                                        daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()  # release the sleeper instantly
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def close(self) -> None:
        self.stop()
        with self._lock:
            self._conn.close()


# Module-global scheduler, set by register().
_scheduler: Optional[CronScheduler] = None


def get_scheduler() -> Optional[CronScheduler]:
    return _scheduler


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    dt = datetime.fromtimestamp(ts)
    delta = ts - time.time()
    if delta < 0:
        when = f"{-delta:.0f}s overdue"
    elif delta < 90:
        when = f"in {delta:.0f}s"
    elif delta < 3600:
        when = f"in {delta / 60:.0f}min"
    else:
        when = f"in {delta / 3600:.1f}h"
    return f"{dt:%Y-%m-%d %H:%M} ({when})"


def _sched() -> CronScheduler:
    if _scheduler is None:
        raise RuntimeError("cronsched not registered")
    return _scheduler


def _handle_cron_add(cron_expr: str, command: str) -> str:
    try:
        cron_id, nxt = _sched().add(cron_expr, command)
    except ValueError as e:
        return f"error: {e}"
    return (f"✓ cron #{cron_id} added: `{cron_expr}` → {command}\n"
            f"  next run: {_fmt_ts(nxt)}")


def _handle_cron_list() -> str:
    rows = _sched().list()
    if not rows:
        return "no crons scheduled — use CronAdd to add one"
    lines = []
    for r in rows:
        state = "enabled" if r["enabled"] else "disabled"
        lines.append(
            f"#{r['id']} [{state}] `{r['cron_expr']}` → {r['command']}\n"
            f"   next: {_fmt_ts(r['next_run'])}, "
            f"last: {_fmt_ts(r['last_run'])}")
    return "\n".join(lines)


def _handle_cron_remove(id: int) -> str:
    if _sched().remove(int(id)):
        return f"✓ cron #{id} removed (run history cleared)"
    return f"error: no cron with id {id}"


def _handle_cron_log(id: int, limit: int = 5) -> str:
    sched = _sched()
    row = sched._row(int(id))
    if row is None:
        return f"error: no cron with id {id}"
    runs = sched.runs(int(id), limit)
    if not runs:
        return f"cron #{id} (`{row['cron_expr']}`) has not run yet"
    lines = [f"last {len(runs)} run(s) of cron #{id}:"]
    for r in runs:
        dt = datetime.fromtimestamp(r["started_at"])
        tail = r["output_tail"].strip()
        if len(tail) > 400:
            tail = "…" + tail[-400:]
        lines.append(f"  {dt:%Y-%m-%d %H:%M:%S} exit={r['exit_code']}")
        if tail:
            lines.append("    " + tail.replace("\n", "\n    "))
    return "\n".join(lines)


def register(agent: Any) -> None:
    """Wire CronAdd/CronList/CronRemove/CronLog into the agent and start
    the scheduler thread (duck-typed, no imports)."""
    global _scheduler
    if _scheduler is None:
        _scheduler = CronScheduler(DB_PATH)
    _scheduler.start()
    agent.tools["CronAdd"] = Tool(
        name="CronAdd",
        description=("Schedule a shell command on a cron schedule. "
                     "cron_expr is a standard 5-field cron expression "
                     "(minute hour day-of-month month day-of-week), e.g. "
                     "'*/15 * * * *' or '0 9 * * 1-5'. command is the shell "
                     "command to run; its output is captured and logged."),
        parameters={"type": "object", "properties": {
            "cron_expr": {"type": "string",
                          "description": "5-field cron expression"},
            "command": {"type": "string",
                        "description": "shell command to run"},
        }, "required": ["cron_expr", "command"]},
        handler=_handle_cron_add,
    )
    agent.tools["CronList"] = Tool(
        name="CronList",
        description=("List all scheduled crons with their next run time, "
                     "enabled state, and last run time."),
        parameters={"type": "object", "properties": {}},
        handler=lambda: _handle_cron_list(),
    )
    agent.tools["CronRemove"] = Tool(
        name="CronRemove",
        description="Remove a scheduled cron by id (clears its run history).",
        parameters={"type": "object", "properties": {
            "id": {"type": "integer", "description": "cron id"},
        }, "required": ["id"]},
        handler=_handle_cron_remove,
    )
    agent.tools["CronLog"] = Tool(
        name="CronLog",
        description=("Show the last N run results (start time, exit code, "
                     "output tail) for a cron by id."),
        parameters={"type": "object", "properties": {
            "id": {"type": "integer", "description": "cron id"},
            "limit": {"type": "integer",
                      "description": "how many runs to show (default 5)"},
        }, "required": ["id"]},
        handler=_handle_cron_log,
    )
    agent.cron_scheduler = _scheduler


# ---------------------------------------------------------------------------
# Self-test: proves REAL behavior (real SQLite, real subprocess).
# ---------------------------------------------------------------------------


def _selftest_thread_efficiency() -> None:
    """Prove the scheduler thread is cheap when idle and fast when work
    arrives: (1) POLL_SECONDS is a sane 30s, (2) the loop sleeps on _wake
    with a 30s timeout and wakes instantly on poke, (3) a due cron fires
    within seconds of poke() instead of waiting out the 30s tick."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="cron_eff_"))
    sched = CronScheduler(tmp / "cron.db")
    try:
        # 1. sane interval: 30s -> 0.033 wakes/s when idle
        assert POLL_SECONDS == 30, POLL_SECONDS
        print(f"PASS - scheduler interval sane: {POLL_SECONDS}s "
              f"({1 / POLL_SECONDS:.3f} wakes/s idle)")

        # 2+3. the loop sleeps on _wake: a due cron must fire promptly
        # after poke(), not after 30s.
        fired: list[float] = []
        orig = sched.run_due

        def counting(now=None):
            fired.append(time.monotonic())
            return orig(now)

        sched.run_due = counting  # type: ignore[method-assign]
        sched.start()
        try:
            cid, _ = sched.add("* * * * *", "echo poke-fired-ok")
            with sched._lock:
                sched._conn.execute(
                    "UPDATE crons SET next_run = ? WHERE id = ?",
                    (time.time() - 1, cid))
                sched._conn.commit()
            sched.poke()  # explicit: the loop must wake NOW, not in 30s
            t0 = time.monotonic()
            deadline = t0 + 10
            while not sched.runs(cid) and time.monotonic() < deadline:
                time.sleep(0.1)
            dt = time.monotonic() - t0
            assert sched.runs(cid), "due cron never fired after poke()"
            assert dt < 10, f"cron took {dt:.1f}s despite poke()"
            print(f"PASS - poke wakes scheduler: due cron fired in "
                  f"{dt:.2f}s (not 30s)")
            assert fired, "scheduler loop never ticked"
            print(f"PASS - scheduler loop ticked {len(fired)}x "
                  f"(instant on poke, 30s when idle)")
        finally:
            t0 = time.monotonic()
            sched.stop()
            assert time.monotonic() - t0 < 5.0, "stop() hung"
            print("PASS - scheduler stop() returns promptly")
    finally:
        sched.close()

    print("PASS - cronsched thread efficiency")

if __name__ == "__main__":
    import tempfile

    tmp = tempfile.mkdtemp(prefix="cronsched_selftest_")
    DB_PATH = Path(tmp) / "cron.db"  # isolate the real default db

    class FakeAgent:
        def __init__(self):
            self.session_id = "selftest"
            self.tools = {}

    # --- expression parsing: steps, ranges, lists ----------------------------
    assert _parse_field("*/15", 0, 59) == {0, 15, 30, 45}
    assert _parse_field("1-5", 0, 59) == {1, 2, 3, 4, 5}
    assert _parse_field("1-10/3", 0, 59) == {1, 4, 7, 10}
    assert _parse_field("0,30,59", 0, 59) == {0, 30, 59}
    assert _parse_field("9-17/2", 0, 23) == {9, 11, 13, 15, 17}
    for bad in ("", "*/0", "60", "5-2", "x", "1,2,", "*/", "* * *"):
        try:
            if bad == "* * *":
                _parse(bad)
            else:
                _parse_field(bad, 0, 59)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad cron accepted: {bad!r}")

    # DOW normalization: 0 and 7 both Sunday
    mins, hours, doms, months, dows, _, _ = _parse("* * * * 7")
    assert dows == {0}, dows

    # --- register wires 4 tools + starts the real thread --------------------
    a = FakeAgent()
    register(a)
    for name in ("CronAdd", "CronList", "CronRemove", "CronLog"):
        assert name in a.tools, name
    assert hasattr(a, "cron_scheduler")
    assert _scheduler._thread.is_alive()

    sched = a.cron_scheduler
    assert sched.db_path == DB_PATH
    # Stop the background loop before driving run_due() by hand: a pending
    # wake can otherwise consume the forced-due row first, making n == 0.
    sched.stop()

    # --- CronAdd: every minute, next_run computed correctly ------------------
    out = a.tools["CronAdd"].handler(cron_expr="* * * * *",
                                     command="echo hello-cron-test")
    assert "cron #1 added" in out, out
    rows = sched.list()
    assert len(rows) == 1 and rows[0]["id"] == 1
    nxt = rows[0]["next_run"]
    now = time.time()
    assert 0 < nxt - now <= 61, f"next_run {nxt-now:.1f}s off"
    dt = datetime.fromtimestamp(nxt)
    assert dt.second == 0 and dt.microsecond == 0  # minute-aligned

    # invalid expression is rejected, not stored
    out = a.tools["CronAdd"].handler(cron_expr="not a cron", command="x")
    assert out.startswith("error:"), out
    assert len(sched.list()) == 1

    # --- manually trigger a due run: REAL subprocess must execute -----------
    with sched._lock:
        sched._conn.execute("UPDATE crons SET next_run = ? WHERE id = 1",
                            (time.time() - 1,))
        sched._conn.commit()
    n = sched.run_due()
    assert n == 1, f"expected 1 due cron, ran {n}"

    hist = sched.runs(1)
    assert len(hist) == 1
    assert hist[0]["exit_code"] == 0, hist[0]
    assert "hello-cron-test" in hist[0]["output_tail"], hist[0]["output_tail"]
    # next_run advanced past now after the run
    rows = sched.list()
    assert rows[0]["next_run"] > time.time()
    assert rows[0]["last_run"] is not None

    # a failing command records its real exit code and stderr
    a.tools["CronAdd"].handler(cron_expr="* * * * *",
                               command="echo boom >&2; exit 7")
    with sched._lock:
        sched._conn.execute("UPDATE crons SET next_run = ? WHERE id = 2",
                            (time.time() - 1,))
        sched._conn.commit()
    sched.run_due()
    hist = sched.runs(2)
    assert hist[0]["exit_code"] == 7, hist[0]
    assert "boom" in hist[0]["output_tail"]

    # --- CronLog surfaces the real run --------------------------------------
    out = a.tools["CronLog"].handler(id=1)
    assert "exit=0" in out and "hello-cron-test" in out, out
    out = a.tools["CronLog"].handler(id=999)
    assert out.startswith("error:"), out

    # --- restart survival: fresh scheduler reads the same db ----------------
    sched2 = CronScheduler(DB_PATH)
    assert len(sched2.list()) == 2, "crons lost across restart"
    r = sched2.list()[0]
    assert r["next_run"] > time.time(), "stale next_run not recomputed"
    sched2.close()

    # --- CronList formatting -------------------------------------------------
    out = a.tools["CronList"].handler()
    assert "#1" in out and "#2" in out and "next:" in out, out

    # --- CronRemove -----------------------------------------------------------
    out = a.tools["CronRemove"].handler(id=1)
    assert "removed" in out, out
    assert sched._row(1) is None
    assert len(sched.runs(1)) == 0  # history cleared too
    out = a.tools["CronRemove"].handler(id=1)
    assert out.startswith("error:"), out

    print("PASS cronsched self-test: real sqlite + real subprocess + "
          "restart survival verified")

    _selftest_thread_efficiency()
