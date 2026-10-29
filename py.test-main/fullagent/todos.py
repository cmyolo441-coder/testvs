"""Claude Code-style todo tracking tools (TodoWrite / TodoRead).

Persisted per session at ``~/.fullagent/todos/<session_id>.json``.

Public API for the TUI and the agent:
    - :func:`register` -- wire both tools into an agent (duck-typed).
    - :func:`get_todos` -- current todo list as ``list[dict]``.
    - :func:`todo_panel_lines` -- ready-to-render panel lines with
      checkbox glyphs (☐ pending, ◐ in_progress, ☑ completed).
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

from .tools import Tool

TODOS_DIR = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"), "todos")

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
VALID_STATUSES = (STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED)

GLYPHS = {
    STATUS_PENDING: "\u2610",       # ☐
    STATUS_IN_PROGRESS: "\u25d0",   # ◐
    STATUS_COMPLETED: "\u2611",     # ☑
}

_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _safe_session_id(session_id: Optional[str]) -> str:
    sid = (session_id or "default").strip() or "default"
    sid = _FILENAME_RE.sub("_", sid)
    return sid[:64]


def _persist_path(session_id: Optional[str]) -> str:
    return os.path.join(TODOS_DIR, _safe_session_id(session_id) + ".json")


class TodoManager:
    """Holds a session's todo list and persists it to disk."""

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = _safe_session_id(session_id)
        self._lock = threading.Lock()
        self._todos: List[Dict[str, Any]] = []
        self.load()

    # -- persistence ----------------------------------------------------
    def load(self) -> None:
        path = _persist_path(self.session_id)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._todos = [self._normalize(t) for t in data
                               if isinstance(t, dict)]
        except (OSError, ValueError):
            self._todos = []

    def save(self) -> None:
        try:
            os.makedirs(TODOS_DIR, exist_ok=True)
            with open(_persist_path(self.session_id), "w",
                      encoding="utf-8") as fh:
                json.dump(self._todos, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass  # persistence is best-effort; never break a tool call

    # -- mutation -------------------------------------------------------
    @staticmethod
    def _normalize(item: Dict[str, Any]) -> Dict[str, Any]:
        status = str(item.get("status", STATUS_PENDING))
        if status not in VALID_STATUSES:
            status = STATUS_PENDING
        return {
            "content": str(item.get("content", "")),
            "status": status,
            "activeForm": str(item.get("activeForm", "")),
        }

    def set_todos(self, todos: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._todos = [self._normalize(t) for t in todos]
            self.save()

    def get_todos(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(t) for t in self._todos]

    # -- rendering ------------------------------------------------------
    def panel_lines(self) -> List[str]:
        lines: List[str] = []
        with self._lock:
            for i, t in enumerate(self._todos, 1):
                glyph = GLYPHS.get(t["status"], GLYPHS[STATUS_PENDING])
                lines.append(f"{glyph} {i}. {t['content']}")
        return lines


# The manager for module-level helpers; set by register() or on demand.
_manager: Optional[TodoManager] = None
_manager_lock = threading.Lock()


def _get_manager(session_id: Optional[str] = None) -> TodoManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TodoManager(session_id)
        return _manager


def get_todos() -> List[Dict[str, Any]]:
    """Current todo list (module-level manager)."""
    return _get_manager().get_todos()


def todo_panel_lines() -> List[str]:
    """Render-ready panel lines; empty list when there are no todos."""
    return _get_manager().panel_lines()


# ---------------------------------------------------------------------------
# Tool handlers (must return str)
# ---------------------------------------------------------------------------

def _handle_todo_write(todos: Any = None, **kwargs: Any) -> str:
    if not isinstance(todos, list):
        return "Error: 'todos' must be a list of todo objects."
    if not todos:
        return "Error: 'todos' must not be empty."
    for t in todos:
        if not isinstance(t, dict) or not str(t.get("content", "")).strip():
            return "Error: every todo needs a non-empty 'content' string."
    mgr = _get_manager()
    mgr.set_todos(todos)
    return f"Updated todo list: {len(todos)} item(s).\n" + "\n".join(
        mgr.panel_lines())


def _handle_todo_read(**kwargs: Any) -> str:
    mgr = _get_manager()
    lines = mgr.panel_lines()
    if not lines:
        return "No todos yet. Use TodoWrite to create a task list."
    return "\n".join(lines)


def make_todo_write_tool() -> Tool:
    return Tool(
        name="TodoWrite",
        description=(
            "Create or replace the current todo list for this session. "
            "Pass the FULL list each time (this replaces the whole list). "
            "Keep todos small and concrete. Use activeForm as the present-"
            "tense label, e.g. activeForm='Writing tests'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                            "activeForm": {"type": "string"},
                        },
                        "required": ["content", "status", "activeForm"],
                    },
                }
            },
            "required": ["todos"],
        },
        handler=_handle_todo_write,
    )


def make_todo_read_tool() -> Tool:
    return Tool(
        name="TodoRead",
        description=(
            "Read the current todo list for this session, with status for "
            "each item. Use before updating with TodoWrite."
        ),
        parameters={"type": "object", "properties": {}},
        handler=_handle_todo_read,
    )


def register(agent: Any) -> None:
    """Wire TodoWrite/TodoRead into an agent (duck-typed, no imports).

    Stores a per-session TodoManager on ``agent.todo_manager`` and registers
    both tools in ``agent.tools``. The module-level helpers
    (:func:`get_todos`, :func:`todo_panel_lines`) then serve this session.
    """
    global _manager
    session_id = getattr(agent, "session_id", None)
    with _manager_lock:
        _manager = TodoManager(session_id)
    agent.todo_manager = _manager
    agent.tools["TodoWrite"] = make_todo_write_tool()
    agent.tools["TodoRead"] = make_todo_read_tool()


if __name__ == "__main__":
    import tempfile

    # Isolate the persist dir for the self-test.
    tmp = tempfile.mkdtemp(prefix="todos_selftest_")
    TODOS_DIR = tmp

    class FakeAgent:
        def __init__(self):
            self.session_id = "selftest-session"
            self.tools = {}

    a = FakeAgent()
    register(a)
    assert "TodoWrite" in a.tools and "TodoRead" in a.tools
    assert hasattr(a, "todo_manager")

    # Round 1: write a list
    out = a.tools["TodoWrite"].handler(todos=[
        {"content": "Write code", "status": "in_progress",
         "activeForm": "Writing code"},
        {"content": "Run tests", "status": "pending",
         "activeForm": "Running tests"},
    ])
    assert "Updated todo list: 2 item(s)." in out, out
    assert "\u25d0" in out and "\u2610" in out  # ◐ and ☐

    # Round 2: read reflects the write
    out = a.tools["TodoRead"].handler()
    assert "Write code" in out and "Run tests" in out

    # Module-level helpers
    assert len(get_todos()) == 2
    panel = todo_panel_lines()
    assert panel[0] == "\u25d0 1. Write code", panel

    # Whole-list replacement + completed glyph
    a.tools["TodoWrite"].handler(todos=[
        {"content": "Write code", "status": "completed",
         "activeForm": "Writing code"},
    ])
    assert todo_panel_lines()[0] == "\u2611 1. Write code"

    # Persistence: a fresh manager for the same session reloads from disk
    again = TodoManager("selftest-session")
    assert len(again.get_todos()) == 1
    assert again.get_todos()[0]["status"] == "completed"

    # Validation errors still return strings, never raise
    assert a.tools["TodoWrite"].handler(todos=[]).startswith("Error:")
    assert a.tools["TodoWrite"].handler(
        todos=[{"content": ""}]).startswith("Error:")

    # Fallback session id for agents without one
    b = FakeAgent()
    b.session_id = None
    register(b)
    b.tools["TodoWrite"].handler(todos=[
        {"content": "Fallback", "status": "pending",
         "activeForm": "Falling back"}])
    assert os.path.exists(os.path.join(tmp, "default.json"))

    print("todos self-test PASSED")
