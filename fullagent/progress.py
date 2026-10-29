"""Progress bars — real, thread-safe progress tracking for background work.

Background tasks (shell jobs, downloads) report progress
through :class:`ProgressTracker`; the TUI renders active trackers as
fragment lines like::

    syncing files ████████░░ 80% (8/10)

Public API:
    - :class:`ProgressTracker` -- thread-safe container of tracker state.
      ``start(label, total) -> id``, ``update(id, current, message=None)``,
      ``increment(id, delta=1) -> int`` (atomic, for racing threads),
      ``mark_done(id)``, ``prune(max_age_s=300)``,
      ``progress_bar_lines() -> list[str]`` (TUI renderer).
    - Tools ``ProgressStart`` / ``ProgressUpdate`` / ``ProgressDone``
      registered on the agent by :func:`register`.
    - ``agent.progress_tracker`` -- the attached tracker instance.

``python3 -m fullagent.progress`` runs the built-in self-test.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Done trackers older than this are pruned automatically.
PRUNE_AFTER_S = 300.0  # 5 minutes

_FULL = "\u2588"  # █
_EMPTY = "\u2591"  # ░


@dataclass
class _Tracker:
    id: str
    label: str
    total: int
    current: int = 0
    message: str = ""
    started_at: float = field(default_factory=time.monotonic)
    done: bool = False
    done_at: Optional[float] = None


class ProgressTracker:
    """Thread-safe registry of progress trackers.

    All state mutations hold a single :class:`threading.Lock`, so
    background threads can update concurrently without lost updates.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trackers: Dict[str, _Tracker] = {}

    # -- lifecycle ------------------------------------------------------

    def start(self, label: str, total: int) -> str:
        """Begin tracking ``label`` (expects ``total`` units); return its id."""
        label = (label or "").strip() or "task"
        try:
            total = int(total)
        except (TypeError, ValueError, OverflowError):
            total = 0
        total = max(total, 0)
        pid = uuid.uuid4().hex[:8]
        with self._lock:
            self._prune_locked(PRUNE_AFTER_S)
            self._trackers[pid] = _Tracker(id=pid, label=label, total=total)
        return pid

    def update(self, pid: str, current: int,
               message: Optional[str] = None) -> None:
        """Set the absolute position of tracker ``pid`` (raises KeyError
        when unknown)."""
        cur = self._clamp_int(current)
        with self._lock:
            t = self._trackers[pid]  # KeyError: unknown id
            t.current = max(0, cur)
            if t.total > 0:
                t.current = min(t.current, t.total)
            if message is not None:
                t.message = str(message)[:200]

    def increment(self, pid: str, delta: int = 1) -> int:
        """Atomically add ``delta`` to tracker ``pid``; return the new
        value.  Built for racing background threads — no lost updates."""
        try:
            delta = int(delta)
        except (TypeError, ValueError, OverflowError):
            delta = 0
        with self._lock:
            t = self._trackers[pid]  # KeyError: unknown id
            t.current = max(0, t.current + delta)
            if t.total > 0:
                t.current = min(t.current, t.total)
            return t.current

    def mark_done(self, pid: str) -> None:
        """Mark tracker ``pid`` done (raises KeyError when unknown)."""
        with self._lock:
            t = self._trackers[pid]  # KeyError: unknown id
            t.done = True
            t.done_at = time.monotonic()
            if t.total > 0:
                t.current = t.total

    def get(self, pid: str) -> Optional[Dict[str, Any]]:
        """Snapshot of tracker state as a plain dict (None if unknown)."""
        with self._lock:
            t = self._trackers.get(pid)
            if t is None:
                return None
            return {"id": t.id, "label": t.label, "total": t.total,
                    "current": t.current, "message": t.message,
                    "started_at": t.started_at, "done": t.done,
                    "done_at": t.done_at}

    # -- cleanup ----------------------------------------------------------

    def prune(self, max_age_s: float = PRUNE_AFTER_S) -> int:
        """Drop done trackers finished more than ``max_age_s`` ago.
        Returns the number removed."""
        with self._lock:
            return self._prune_locked(max_age_s)

    def _prune_locked(self, max_age_s: float) -> int:
        now = time.monotonic()
        stale = [pid for pid, t in self._trackers.items()
                 if t.done and t.done_at is not None
                 and now - t.done_at > max_age_s]
        for pid in stale:
            del self._trackers[pid]
        return len(stale)

    # -- rendering --------------------------------------------------------

    @staticmethod
    def _clamp_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _render_bar(current: int, total: int, width: int = 10) -> str:
        width = max(width, 1)
        if total <= 0:
            filled = 0
        else:
            filled = int(round(max(0.0, min(1.0, current / total)) * width))
        return _FULL * filled + _EMPTY * (width - filled)

    @staticmethod
    def _percent(current: int, total: int) -> int:
        if total <= 0:
            return 0
        return int(round(max(0.0, min(1.0, current / total)) * 100))

    def progress_bar_lines(self, width: int = 10) -> List[str]:
        """Render one text line per ACTIVE (not done) tracker, e.g.::

            syncing files ████████░░ 80% (8/10)

        Old done trackers are pruned as a side effect (auto-cleanup).
        """
        with self._lock:
            self._prune_locked(PRUNE_AFTER_S)
            active = [t for t in self._trackers.values() if not t.done]
        lines = []
        for t in active:
            pct = self._percent(t.current, t.total)
            bar = self._render_bar(t.current, t.total, width)
            line = f"{t.label} {bar} {pct}% ({t.current}/{t.total})"
            if t.message:
                line += f" — {t.message}"
            lines.append(line)
        return lines


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _make_tools(tracker: ProgressTracker) -> list:
    from .tools import Tool, RISK_SAFE

    def progress_start(label: str, total: int = 0) -> str:
        """Start a progress tracker; returns the tracker id."""
        pid = tracker.start(label, total)
        return f"progress id: {pid}"

    def progress_update(id: str, current: int,
                        message: Optional[str] = None) -> str:
        """Update the absolute position of a progress tracker."""
        try:
            tracker.update(id, current, message)
        except KeyError:
            return f"ERROR: unknown progress id: {id}"
        return "OK"

    def progress_done(id: str) -> str:
        """Mark a progress tracker done (it is hidden and auto-pruned)."""
        try:
            tracker.mark_done(id)
        except KeyError:
            return f"ERROR: unknown progress id: {id}"
        return "OK"

    return [
        Tool("ProgressStart",
             "Start a progress bar for background work. Args: label (str), "
             "total (int, expected units). Returns the tracker id.",
             {"type": "object",
              "properties": {
                  "label": {"type": "string"},
                  "total": {"type": "integer"}},
              "required": ["label"]},
             progress_start, risk=RISK_SAFE),
        Tool("ProgressUpdate",
             "Update a progress bar to an absolute position. Args: id "
             "(tracker id), current (int), message (optional status text).",
             {"type": "object",
              "properties": {
                  "id": {"type": "string"},
                  "current": {"type": "integer"},
                  "message": {"type": "string"}},
              "required": ["id", "current"]},
             progress_update, risk=RISK_SAFE),
        Tool("ProgressDone",
             "Mark a progress bar done; it disappears and is auto-pruned. "
             "Args: id (tracker id).",
             {"type": "object",
              "properties": {"id": {"type": "string"}},
              "required": ["id"]},
             progress_done, risk=RISK_SAFE),
    ]


def register(agent: Any) -> None:
    """Attach ``agent.progress_tracker`` and the Progress* tools."""
    tracker = ProgressTracker()
    agent.progress_tracker = tracker
    tools = getattr(agent, "tools", None)
    if isinstance(tools, dict):
        for tool in _make_tools(tracker):
            tools[tool.name] = tool


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _check(name: str, cond: bool) -> None:
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        raise SystemExit(f"self-test failed: {name}")


if __name__ == "__main__":
    # 1. start / get basics
    tr = ProgressTracker()
    pid = tr.start("syncing files", 10)
    _check("start returns non-empty id", bool(pid))
    snap = tr.get(pid)
    _check("get snapshot matches",
           snap is not None and snap["label"] == "syncing files"
           and snap["total"] == 10 and snap["current"] == 0
           and snap["done"] is False)

    # 2. bar rendering math: 0%, 50%, 100%
    tr.update(pid, 0)
    lines = tr.progress_bar_lines()
    _check("0% renders",
           lines == ["syncing files ░░░░░░░░░░ 0% (0/10)"])
    tr.update(pid, 5)
    lines = tr.progress_bar_lines()
    _check("50% renders",
           lines == ["syncing files █████░░░░░ 50% (5/10)"])
    tr.update(pid, 10)
    lines = tr.progress_bar_lines()
    _check("100% renders",
           lines == ["syncing files ██████████ 100% (10/10)"])

    # 3. message appended
    tr.update(pid, 5, "halfway")
    _check("message shown",
           tr.progress_bar_lines() == ["syncing files █████░░░░░ 50% (5/10) — halfway"])

    # 4. concurrency: 8 threads x 250 atomic increments = 2000, no losses
    cpid = tr.start("concurrent", 100000)
    THREADS, PER_THREAD = 8, 250

    def worker() -> None:
        for _ in range(PER_THREAD):
            tr.increment(cpid, 1)

    ts = [threading.Thread(target=worker) for _ in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    _check("concurrent increments, no lost updates",
           tr.get(cpid)["current"] == THREADS * PER_THREAD)

    # 5. done hides tracker; prune removes it only when old
    tr.mark_done(pid)
    _check("done tracker hidden from bar lines",
           not any(pid in ln for ln in tr.progress_bar_lines()))
    _check("recently-done tracker NOT pruned", tr.get(pid) is not None
           and tr.prune() == 0)
    # Backdate past the 5-min cutoff, then prune.
    with tr._lock:
        tr._trackers[pid].done_at = time.monotonic() - (PRUNE_AFTER_S + 1)
    _check("stale done tracker pruned", tr.prune() == 1 and tr.get(pid) is None)
    _check("active tracker survives prune", tr.get(cpid) is not None)

    # 6. unknown ids raise KeyError
    for op in (lambda: tr.update("nope", 1),
               lambda: tr.increment("nope", 1),
               lambda: tr.mark_done("nope")):
        try:
            op()
            _check("unknown id raises", False)
        except KeyError:
            pass
    _check("unknown id raises", True)

    # 7. register() duck-types onto a fake agent
    from types import SimpleNamespace
    fake = SimpleNamespace(tools={})
    register(fake)
    _check("register attaches progress_tracker",
           isinstance(fake.progress_tracker, ProgressTracker))
    _check("register adds 3 tools",
           {"ProgressStart", "ProgressUpdate", "ProgressDone"}
           <= set(fake.tools))

    # 8. tool handlers end-to-end (no agent needed)
    t_start = fake.tools["ProgressStart"].handler
    t_upd = fake.tools["ProgressUpdate"].handler
    t_done = fake.tools["ProgressDone"].handler
    new_id = t_start(label="download", total=4).split(": ", 1)[1]
    _check("ProgressStart returns id", t_upd(id=new_id, current=2) == "OK")
    _check("ProgressUpdate unknown id errors",
           "ERROR" in t_upd(id="bogus", current=1))
    _check("ProgressDone works", t_done(id=new_id) == "OK")
    _check("ProgressDone unknown id errors",
           "ERROR" in t_done(id="bogus"))

    # 9. edge cases
    z = tr.start("zero total", 0)
    _check("zero total renders without div-by-zero",
           tr.progress_bar_lines()[-1] == "zero total ░░░░░░░░░░ 0% (0/0)")
    tr.update(cpid, -50)
    _check("negative clamped to 0", tr.get(cpid)["current"] == 0)
    tr.update(cpid, 10**9)
    _check("over-total clamped", tr.get(cpid)["current"] == 100000)

    print("\nprogress self-test: all checks passed")
