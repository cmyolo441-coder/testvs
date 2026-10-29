"""Presentation-only TUI panel for background shell jobs.

The optional ``fullagent.bgsh`` provider is imported lazily and treated
strictly as an integration boundary: malformed/missing job rows are
skipped, and the panel never raises into the TUI. This module has no
agent delegation, threading, or execution behavior.
"""
from __future__ import annotations

import time

_RUN_MARKER = "\u25cf"  # ●
_IDLE_MARKER = "\u00b7"  # ·
_KILL_HINT = "use BashOutput <id> to inspect a running shell job"
_STATUS_ALIASES = {
    "started": "running", "in_progress": "running", "active": "running",
    "success": "done", "completed": "done", "finished": "done",
    "failed": "error", "killed": "stopped", "cancelled": "stopped",
}


def _norm_status(raw) -> str:
    status = str(raw or "running").strip().lower()
    return _STATUS_ALIASES.get(status, status)


def _elapsed(started_at, now: float) -> str:
    """Human elapsed time; em dash when the start time is unavailable."""
    try:
        seconds = max(0, int(now - float(started_at)))
    except (TypeError, ValueError, OverflowError):
        return "—"
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _row(task_id, status: str, description: str, started_at,
         now: float) -> tuple[bool, str]:
    normalized = _norm_status(status)
    running = normalized == "running"
    marker = _RUN_MARKER if running else _IDLE_MARKER
    text = str(description or "").replace("\n", " ").strip()
    text = text or "(no description)"
    return running, (f"{marker} {task_id} [{normalized}] {text} "
                     f"({_elapsed(started_at, now)})")


def _bg_shell_rows(now: float) -> list[tuple[bool, str]]:
    """Return sanitized rows from ``bgsh.list_background_tasks()``."""
    try:
        from . import bgsh
        tasks = bgsh.list_background_tasks() or []
    except Exception:
        return []
    rows: list[tuple[bool, str]] = []
    for task in tasks:
        try:
            if isinstance(task, dict):
                task_id, status = task.get("id"), task.get("status")
                description, started = (task.get("description"),
                                        task.get("started_at"))
            else:
                task_id = getattr(task, "id", None)
                status = getattr(task, "status", None)
                description = getattr(task, "description", None)
                started = getattr(task, "started_at", None)
            if task_id is not None:
                rows.append(_row(task_id, status, description, started, now))
        except Exception:
            continue
    return rows


def has_activity(agent=None) -> bool:
    """Whether an optional background shell job is currently running."""
    return any(running for running, _ in _bg_shell_rows(time.time()))


class BackgroundPanel:
    """Render running shell jobs first and keep the last rendered lines."""
    def __init__(self) -> None:
        self.lines: list[str] = []

    def refresh(self, agent=None) -> list[str]:
        rows = _bg_shell_rows(time.time())
        running = [row for is_running, row in rows if is_running]
        finished = [row for is_running, row in rows if not is_running]
        out = [" BACKGROUND TASKS"]
        if not rows:
            out.append("  (none)")
        else:
            out.extend(f"  {row}" for row in running + finished)
            out.append(f"  {_KILL_HINT}")
        self.lines = out
        return out


if __name__ == "__main__":
    import sys
    import types

    now = time.time()
    fake_bgsh = types.ModuleType("bgsh")
    fake_bgsh.list_background_tasks = lambda: [
        {"id": "sh-1", "description": "pytest -q", "status": "running",
         "started_at": now - 12},
        {"id": "sh-2", "description": "curl probe", "status": "done",
         "started_at": now - 90},
        {"id": "sh-3", "description": "tail logs", "status": "error",
         "started_at": now - 5},
    ]
    sys.modules["fullagent.bgsh"] = fake_bgsh
    panel = BackgroundPanel()
    text = "\n".join(panel.refresh())
    assert "sh-1" in text and "[running]" in text
    assert "sh-2" in text and "[done]" in text
    assert "sh-3" in text and "[error]" in text
    assert "● sh-1" in text and "· sh-2" in text
    assert "(12s)" in text and "(1m30s)" in text
    assert "BashOutput" in text and "TaskStop" not in text
    assert panel.lines and has_activity()
    fake_bgsh.list_background_tasks = lambda: []
    assert not has_activity()
    assert "(none)" in "\n".join(panel.refresh())
    print("PASS")
