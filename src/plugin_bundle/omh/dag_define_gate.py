"""A dag-ulw turn's first `delegate_task` without a node marker goes back to `omh_dag define`, once.

The route hint tells the model to plan the graph and call `omh_dag define`
first, and `define` hands back the `delegate_task` calls, each goal carrying
its `[dag:<run>/<node>]` marker (`dag_run.calls_for`). A model that skips it
and fans out plain children instead runs no graph at all: the ledger binds a
child to a node only through that marker (`delegation_ledger.dag_binding`),
so the run view draws nothing and no node is ever verified or repaired.
Nothing refused that call.

So the call is blocked once, with a message naming `define`, when all of
these hold:

* `pre_llm_call` recorded this session's current turn as one whose primary
  hint, as rendered, is dag-ulw's `start_dag_run` (`note_turn`). A delegated
  child records nothing: its hint never carries an orchestrating engine, and
  its own `define` is refused anyway.
* The call is `delegate_task` spawning children, and no goal holds `[dag:`.
  `list`, `steer` and `stop` act on children already running.
* The call runs in that same turn (the host's `turn_id`), and nothing was
  blocked in it yet.
* This conversation has no dag run. One it cannot read counts as one, for
  `define` would be refused there too and the message would point nowhere.

Why it cannot deadlock or loop: a block dispatches nothing, it happens at
most once per turn, and the message says the same call goes through when
sent again. `omh_dag` itself is never held back, so a refused `define` does
not leave the turn stuck behind this gate either.

Never raises: Hermes logs a hook exception and runs the call, which would
also skip the dispatch ledger the rest of `pre_tool_call` writes. A read
that fails reads as a run that exists, which is None.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
import sqlite3
import threading
from typing import Final

from . import dag_run
from .kanban_board_reader import conversation_session_ids
from .runtime_reader import reading_session_id

DELEGATE_TOOL: Final = "delegate_task"
DAG_MARKER: Final = "[dag:"
# The hint whose turn this gate keeps (`awareness._DIRECT_WORKFLOW_NEXT_ACTIONS`).
DAG_WORKFLOW: Final = "dag-ulw"
DAG_START_ACTION: Final = "start_dag_run"
# `delegate_task` actions that spawn; an absent action is Hermes' default, `spawn`.
_SPAWN_ACTIONS: Final = frozenset({"", "spawn"})
MAX_TRACKED_SESSIONS: Final = 256
DEFINE_FIRST_MESSAGE: Final = (
    "Not dispatched: this turn asked for a dag-ulw run, and this conversation has none yet. Call omh_dag "
    "action=define with the graph first, then dispatch the delegate_task calls it returns; their "
    "[dag:<run>/<node>] markers are what the run view draws. If these children are not part of that run, "
    "send this call again and it goes through."
)

_lock = threading.Lock()
# session -> (the turn that starts a dag-ulw run, whether this gate blocked in it)
_turns: "OrderedDict[str, tuple[str, bool]]" = OrderedDict()


def starts_dag_run(rendered_hint: Mapping[str, object]) -> bool:
    """Whether the route hint a turn rendered selects dag-ulw to start a run."""
    return (
        rendered_hint.get("primary_workflow") == DAG_WORKFLOW
        and rendered_hint.get("primary_next_action") == DAG_START_ACTION
    )


def note_turn(session_id: object, turn_id: object, *, starts_run: bool) -> None:
    """Record whether this session's current turn starts a dag-ulw run; `pre_llm_call` calls this every turn."""
    key = str(session_id or "").strip()
    turn = str(turn_id or "").strip()
    if not key:
        return
    with _lock:
        _ = _turns.pop(key, None)
        if starts_run and turn:
            _turns[key] = (turn, False)
            while len(_turns) > MAX_TRACKED_SESSIONS:
                _ = _turns.popitem(last=False)


def _spawned_goals(tool_input: object) -> list[str]:
    if not isinstance(tool_input, Mapping):
        return []
    if str(tool_input.get("action") or "").strip() not in _SPAWN_ACTIONS:
        return []
    tasks = tool_input.get("tasks")
    goals = [task.get("goal") for task in tasks if isinstance(task, Mapping)] if isinstance(tasks, list) else []
    goals.append(tool_input.get("goal"))  # the legacy single-goal shape Hermes still accepts
    return [goal for goal in goals if isinstance(goal, str) and goal.strip()]


def _conversation_has_run(omh_home: str, hermes_home: str, session: str) -> bool:
    try:
        durable = reading_session_id(hermes_home, session)
        if not durable:
            return True
        segments = {durable, *conversation_session_ids(hermes_home, durable)}
        return any(dag_run.session_runs(Path(omh_home), segment) for segment in segments)
    except (OSError, ValueError, TypeError, RuntimeError, sqlite3.Error):
        return True


def delegate_before_define_directive(
    *,
    tool_name: object,
    tool_input: object,
    session_id: object = "",
    turn_id: object = "",
    omh_home: str = "",
    hermes_home: str = "",
) -> dict[str, str] | None:
    """`block` for a dag-ulw turn's first markerless `delegate_task` before any run exists, else None."""
    if str(tool_name or "") != DELEGATE_TOOL:
        return None
    goals = _spawned_goals(tool_input)
    if not goals or any(DAG_MARKER in goal for goal in goals):
        return None
    key = str(session_id or "").strip()
    turn = str(turn_id or "").strip()
    if not key or not turn or not omh_home or not hermes_home:
        return None
    with _lock:
        if _turns.get(key) != (turn, False):
            return None
    if _conversation_has_run(omh_home, hermes_home, key):
        return None
    with _lock:
        # Two calls of one turn can both reach here; the first one to come back blocks.
        if _turns.get(key) != (turn, False):
            return None
        _turns[key] = (turn, True)
    return {"action": "block", "message": DEFINE_FIRST_MESSAGE}


def reset_dag_define_gate() -> None:
    """Test seam: forget every recorded turn."""
    with _lock:
        _turns.clear()


__all__ = [
    "DEFINE_FIRST_MESSAGE",
    "delegate_before_define_directive",
    "note_turn",
    "reset_dag_define_gate",
    "starts_dag_run",
]
