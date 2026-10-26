"""Read-only projection of Hermes Kanban board lanes for the OMH HUD.

Hermes keeps its board in ``kanban.db`` at the Hermes ROOT (shared across
profiles by design: ``<root>/kanban.db`` for the default board,
``<root>/kanban/boards/<slug>/kanban.db`` for a named one, ``HERMES_KANBAN_HOME``
overriding the root, and the session's current board resolved the way the
native tools resolve it: ``HERMES_KANBAN_BOARD``, then ``<root>/kanban/current``,
then ``default``). Workers are detached processes the gateway dispatcher
spawns; the board itself records no token usage and no served-model
observation, so the native status, the assignee and the per-task overrides
are all the board can state on its own.

The worker's own Hermes session records the rest. A dispatched worker runs
under the assignee profile's ``HERMES_HOME``, so its session lives in that
profile's ``state.db`` with ``source = 'kanban'``, and the model, turns, tool
calls, tokens and cost on that row are the same figures the native
delegate_task rows report. ``_worker_usage`` joins them onto a lane row
through the worker's own stamp when there is one and a bounded, uniqueness-
gated dispatch window when there is not; an unlinked row keeps its empty
model and its ``tokens: None``, never a borrowed figure.

This reader opens every database read-only, never writes one, never probes
the gateway and never calls ``hermes kanban``. Every failure is fail-open: a
missing board, a foreign schema, an absent profile home or a lock lost to the
dispatcher all project as "no rows" or "no usage", never as an exception in
the widget's poll.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .hermes_delegation import (
    COMPLETED_LINGER_SECONDS,
    _conversation_session_ids,
    _finite,
    _iso_utc,
)
from .sqlite_readonly import connect_readonly

LANE_BACKEND = "kanban"
DEFAULT_BOARD = "default"
# Hermes' default profile lives at the Hermes root itself; every named one at
# ``<root>/profiles/<name>`` (``hermes_cli/profiles.py::resolve_profile_env``).
DEFAULT_PROFILE = "default"
# A worker that stopped heartbeating for this long is projected as stale: its
# pid is still recorded, but nothing says it is doing anything.
STALE_HEARTBEAT_SECONDS = 15 * 60
# A ready task nobody has claimed after this long, with no worker or claim
# anywhere on the selection, is the observable shape of "no dispatcher".
DISPATCHER_WAIT_SECONDS = 120
OPEN_STATUSES = ("todo", "scheduled", "ready", "running", "blocked", "review")
FINISHED_STATUSES = ("done", "archived")
REQUIRED_COLUMNS = frozenset(
    {
        "id",
        "title",
        "assignee",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "claim_lock",
        "worker_pid",
        "last_failure_error",
        "last_heartbeat_at",
        "skills",
        "model_override",
        "provider_override",
        "reasoning_effort",
        "session_id",
        "workspace_path",
    }
)
_ROW_LIMIT = 8
_SELECT_LIMIT = 64
_ACTION_LIMIT = 96
_BOARD_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
# Hermes' own profile id rule (``validate_profile_name``), which happens to be
# the board slug's shape too: an assignee that could not name a profile
# directory never becomes a path here.
_PROFILE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
# The board stores whole seconds and state.db stores floats, so a worker's
# session can look like it started a moment before the run row that spawned it.
_WORKER_SESSION_LEAD_SECONDS = 5.0
# The dispatcher spawns the worker the moment it claims the run, and the
# worker writes its session row on its first turn. A session that appears
# minutes later is some other dispatch's.
_WORKER_SESSION_SPAWN_SECONDS = 300.0
# The columns of the worker's own session row the HUD reports, in one place so
# both lookups below read the same shape.
# The dispatcher opens every worker with the literal prompt
# `work kanban task <task id>` (hermes_cli/kanban_db_dispatch.py, _worker_argv),
# and Hermes derives the session title from that first prompt, so the title
# is a field the host wrote from the task id, not text a model produced.
_WORKER_PROMPT_TITLE_PREFIX = "work kanban task "
_USAGE_SELECT = (
    "SELECT id, model, model_config, api_call_count, tool_call_count, "
    "input_tokens, output_tokens, actual_cost_usd, estimated_cost_usd, cost_status "
    "FROM sessions"
)


def _text(value: Any, limit: int = 80) -> str:
    return str(value or "").strip()[:limit]


def _epoch(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def hermes_root(hermes_home: str | Path) -> Path:
    """The Hermes root: ``HERMES_HOME`` with a trailing ``profiles/<name>`` stripped.

    Same rule as Hermes' own ``get_default_hermes_root``: a profile home lives at
    ``<root>/profiles/<name>`` and the board is the root's, shared by every
    profile, so a profile-scoped reader still finds the one board.
    """
    home = Path(hermes_home).expanduser()
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def _board_root(hermes_home: str | Path) -> Path:
    """The directory the board files hang off: ``HERMES_KANBAN_HOME`` or the root."""
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    return Path(override).expanduser() if override else hermes_root(hermes_home)


def _board_exists(root: Path, slug: str) -> bool:
    """Hermes' ``board_exists``: ``default`` always; a named board when its
    directory holds ``board.json`` or ``kanban.db`` and nothing is a symlink."""
    if slug == DEFAULT_BOARD:
        return True
    directory = root / "kanban" / "boards" / slug
    if directory.is_symlink():
        return False
    return any(
        (directory / name).exists() and not (directory / name).is_symlink()
        for name in ("board.json", "kanban.db")
    )


def _existing_slug(root: Path, candidate: str) -> str:
    """``candidate`` lowercased when Hermes would accept it and that board
    exists, else ``""`` -- a malformed or stale slug falls through, the way
    ``get_current_board`` falls through, never to an exception."""
    slug = candidate.strip().lower()
    if not slug or not _BOARD_SLUG_RE.fullmatch(slug):
        return ""
    return slug if _board_exists(root, slug) else ""


def current_board(hermes_home: str | Path) -> str:
    """The board the session's ``kanban_*`` tools act on when none is named.

    Mirrors Hermes' ``get_current_board`` resolution: ``HERMES_KANBAN_BOARD``
    (validated, existing) wins, then the one-line slug in
    ``<root>/kanban/current`` written by ``hermes kanban boards switch`` (only
    while that board exists), else ``default``. Read-only; a symlinked
    ``current`` file is treated as absent, like every other linked board file.
    """
    root = _board_root(hermes_home)
    found = _existing_slug(root, os.environ.get("HERMES_KANBAN_BOARD", ""))
    if found:
        return found
    marker = root / "kanban" / "current"
    if marker.is_symlink() or not marker.is_file():
        return DEFAULT_BOARD
    try:
        return _existing_slug(root, marker.read_text(encoding="utf-8")) or DEFAULT_BOARD
    except (OSError, UnicodeDecodeError):
        return DEFAULT_BOARD


def kanban_db_path(hermes_home: str | Path, board: str = "") -> Path | None:
    """Where the board's ``kanban.db`` lives, or ``None`` when it cannot be trusted.

    An empty ``board`` means the session's current board (``current_board``),
    so a reader with no board argument follows the same switch the native
    tools follow. ``None`` for a board slug Hermes would refuse and for a
    database path (or its directory) that is a symlink -- the same refusal the
    other readers apply to the state they open, since a link under
    ``~/.hermes`` could point the read-only connection at a file the operator
    never meant to expose.
    """
    root = _board_root(hermes_home)
    slug = board.strip() or current_board(hermes_home)
    if slug == DEFAULT_BOARD:
        path = root / "kanban.db"
    elif _BOARD_SLUG_RE.fullmatch(slug):
        path = root / "kanban" / "boards" / slug / "kanban.db"
    else:
        return None
    if path.is_symlink() or path.parent.is_symlink():
        return None
    return path


def conversation_session_ids(hermes_home: str | Path, session_ref: str) -> set[str]:
    """The conversation identities the native delegate reader scopes by.

    The board stamps a task with the originating ``HERMES_SESSION_ID``, which is
    the same durable id ``state.db`` names, so the HUD can own board rows with
    exactly the identity set it already owns delegate_task children with.
    """
    if not session_ref:
        return set()
    state_db = Path(hermes_home).expanduser() / "state.db"
    if not state_db.is_file():
        return set()
    try:
        connection = connect_readonly(state_db, timeout=0.25)
    except sqlite3.Error:
        return set()
    try:
        return _conversation_session_ids(connection, session_ref)
    except sqlite3.Error:
        return set()
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass


def _tasks_columns(connection: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in connection.execute('PRAGMA table_info("tasks")')}


def _has_table(connection: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` declares any column, i.e. this schema has it."""
    return any(connection.execute(f'PRAGMA table_info("{table}")'))


def _query_tasks(db: Path, *, now: float) -> list[dict[str, Any]] | None:
    """One bounded SELECT over the open lanes plus the lingering finished ones.

    Each task's latest run rides along: its ``metadata`` is where the worker
    stamps the session id the usage lookup needs, and its ``started_at``
    bounds the fallback match. A board whose schema has no ``task_runs`` (an
    older install, a foreign file) still projects its lanes -- the run columns
    simply read as absent, and every row is then unlinked.

    ``None`` when the board could not be read at all (no connection, a
    ``tasks`` table without the columns above, a failed query), so a caller
    can tell that apart from a board with no lanes. The SELECT fetches one row
    past ``_SELECT_LIMIT`` so the caller can tell a full selection from one the
    bound cut; the caller drops that row.
    """
    try:
        connection = connect_readonly(db, timeout=0.25)
    except sqlite3.Error:
        return None
    try:
        if not REQUIRED_COLUMNS <= _tasks_columns(connection):
            return None
        # The correlated subquery is bounded by the outer LIMIT and served by
        # Hermes' own ``idx_runs_task`` index on (task_id, started_at).
        has_runs = _has_table(connection, "task_runs")
        run_columns = "runs.metadata, runs.started_at" if has_runs else "NULL, NULL"
        run_join = (
            " LEFT JOIN task_runs runs ON runs.id = ("
            "SELECT id FROM task_runs WHERE task_id = t.id "
            "ORDER BY COALESCE(started_at, 0) DESC, id DESC LIMIT 1)"
        ) if has_runs else ""
        open_marks = ",".join("?" for _ in OPEN_STATUSES)
        finished_marks = ",".join("?" for _ in FINISHED_STATUSES)
        cursor = connection.execute(
            "SELECT t.id, t.title, t.assignee, t.status, t.created_at, t.started_at, "
            "t.completed_at, t.claim_lock, t.worker_pid, t.last_failure_error, "
            "t.last_heartbeat_at, t.skills, t.model_override, t.provider_override, "
            "t.reasoning_effort, t.session_id, t.workspace_path, "
            f"COALESCE(links.parent_count, 0), {run_columns} "
            "FROM tasks t LEFT JOIN ("
            "SELECT child_id, COUNT(*) AS parent_count FROM task_links GROUP BY child_id"
            ") links ON links.child_id = t.id" + run_join + " "
            f"WHERE t.status IN ({open_marks}) "
            f"OR (t.status IN ({finished_marks}) AND t.completed_at >= ?) "
            "ORDER BY CASE WHEN t.status = 'running' THEN 0 ELSE 1 END, t.created_at ASC "
            "LIMIT ?",
            (*OPEN_STATUSES, *FINISHED_STATUSES, now - COMPLETED_LINGER_SECONDS, _SELECT_LIMIT + 1),
        )
        rows = cursor.fetchall()
    except sqlite3.Error:
        # A schema Hermes has since changed, a lock the dispatcher holds: the
        # board projects as empty for this poll rather than failing the HUD.
        return None
    finally:
        try:
            connection.close()
        except sqlite3.Error:
            pass
    keys = (
        "id", "title", "assignee", "status", "created_at", "started_at", "completed_at",
        "claim_lock", "worker_pid", "last_failure_error", "last_heartbeat_at", "skills",
        "model_override", "provider_override", "reasoning_effort", "session_id",
        "workspace_path", "parent_count", "run_metadata", "run_started_at",
    )
    return [dict(zip(keys, row, strict=True)) for row in rows]


def _worker_state_db(root: Path, assignee: str) -> Path | None:
    """The ``state.db`` the assignee's worker writes its own session to.

    ``resolve_profile_env`` puts a named profile at ``<root>/profiles/<name>``
    and the default profile at the root itself, and the dispatcher exports
    that home to the worker, so the worker's session lands in that home's
    ``state.db``. The name is normalized the way Hermes normalizes it
    (``normalize_profile_name``: lowercase, ``default`` case-insensitive) and
    refused outright when it could not name a profile directory at all -- an
    assignee is board text, and board text must never build a path. A profile
    whose directory is gone leaves the root's own ``state.db`` as the only one
    to ask; both lookups below match on recorded fields, so asking the wrong
    home returns nothing rather than another profile's figures. Symlinks are
    refused the way ``kanban_db_path`` refuses them.
    """
    name = _text(assignee, limit=80).lower()
    if name and name != DEFAULT_PROFILE:
        if not _PROFILE_ID_RE.fullmatch(name):
            return None
        profile = root / "profiles" / name
        path = (profile / "state.db") if profile.is_dir() else (root / "state.db")
    else:
        path = root / "state.db"
    if path.is_symlink() or path.parent.is_symlink():
        return None
    return path if path.is_file() else None


def _worker_session_id(run_metadata: Any) -> str:
    """The worker's own session id, as the worker stamped it on its run.

    ``tools/kanban_tools.py::_stamp_worker_session_metadata`` adds
    ``worker_session_id`` to the metadata a worker passes to
    ``kanban_complete`` or ``kanban_request_review``, taken from its own
    ``HERMES_SESSION_ID`` and only for the task it is scoped to, and
    ``kanban_db._end_run`` writes that dict onto the run row. It is therefore
    a trusted link and not a guess -- and it appears when the worker reports,
    not while it works, which is why the window match below exists at all.
    """
    if not isinstance(run_metadata, str) or not run_metadata.strip():
        return ""
    try:
        parsed = json.loads(run_metadata)
    except ValueError:
        return ""
    return _text(parsed.get("worker_session_id"), limit=120) if isinstance(parsed, dict) else ""


def _reasoning_effort(model_config: Any) -> str:
    """The session's reasoning effort, parsed the way the native reader parses it.

    Copied from ``hermes_delegation._query_state_db``: ``model_config`` is
    JSON, and a recorded effort counts only while ``reasoning_config`` says
    reasoning is enabled.
    """
    try:
        parsed = json.loads(model_config or "{}")
    except (TypeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    reasoning = parsed.get("reasoning_config", {})
    if isinstance(reasoning, dict) and reasoning.get("enabled"):
        return _text(reasoning.get("effort", ""), limit=20)
    return ""


def _state_db_connection(
    cache: dict[Path, sqlite3.Connection | None], state_db: Path
) -> sqlite3.Connection | None:
    """One read-only connection per ``state.db`` per poll, opened at most once."""
    if state_db not in cache:
        try:
            cache[state_db] = connect_readonly(state_db, timeout=0.25)
        except sqlite3.Error:
            cache[state_db] = None
    return cache[state_db]


def _close_connections(cache: dict[Path, sqlite3.Connection | None]) -> None:
    for connection in cache.values():
        if connection is None:
            continue
        try:
            connection.close()
        except sqlite3.Error:
            pass


def _usage_row(row: Any, match: str) -> dict[str, Any]:
    """One session row projected into the figures a lane row reports."""
    (
        identity, model, model_config, api_calls, tool_calls,
        input_tokens, output_tokens, actual_cost, estimated_cost, cost_status,
    ) = row
    cost = actual_cost if actual_cost is not None else estimated_cost
    return {
        "session_id": _text(identity, limit=120),
        "match": match,
        "model": _text(model),
        "effort": _reasoning_effort(model_config),
        # The same sum the native delegate rows report (`hermes_delegation`:
        # `int(input_tokens + output_tokens)`), which is also what Hermes' own
        # `usage_totals` calls a session's tokens. The two lanes sit in one
        # column of one dock, so their token figures must mean one thing.
        "tokens": int((_finite(input_tokens) or 0.0) + (_finite(output_tokens) or 0.0)),
        # `api_call_count` is the counter a native row's `turn_count` sums per
        # model out of `session_model_usage`; `message_count` counts stored
        # messages, tool results included, which is not a turn.
        "turn_count": int(_finite(api_calls) or 0.0),
        "tool_count": int(_finite(tool_calls) or 0.0),
        "cost_usd": _finite(cost),
        "cost_status": _text(cost_status, limit=40),
    }


def _worker_usage(
    connection: sqlite3.Connection,
    *,
    task_id: str,
    worker_session_id: str,
    workspace_path: str,
    run_started_at: float | None,
) -> dict[str, Any] | None:
    """What the worker's own session recorded, or ``None`` when it is unlinked.

    Three rules, all matches on recorded fields and never on text the model
    wrote. The stamp the worker put on its run row is exact, so it is tried
    first and it is final: a run that named its own session is answered by
    that session or by nothing, because a stamp that does not resolve here
    says this is the wrong home, and the weaker rule below would then answer
    with some other task's worker. Before a worker reports there is no stamp,
    and Hermes records nothing else on a session that names the task -- a
    kanban-source session carries no task id, and since the source tag
    replaced the cwd stamp
    (``_launch_cwd_for_session`` records one only for ``source = 'cli'``) it
    usually carries no directory either. What remains is the dispatch window:
    the worker's session is opened within minutes of the run being claimed, in
    the assignee's own profile home. That is only an identity while exactly
    one session answers it, so two candidates return nothing rather than a
    coin flip, and a session whose recorded cwd names a different workspace is
    excluded outright (which is what still separates two legacy worker rows
    that do carry one). Between the stamp and the window sits the one field
    the host does write from the task id: the session title Hermes derives
    from the dispatcher's own opening prompt, `work kanban task <id>`. Two
    workers spawned in the same tick share a window but not a title, so an
    exact title match inside the window is an identity on its own; two
    same-title sessions in one window are two attempts at this task and the
    newest is the live one. A host that re-titles the session later falls
    back to the window rule.
    """
    try:
        if worker_session_id:
            row = connection.execute(
                f"{_USAGE_SELECT} WHERE id = ? LIMIT 1", (worker_session_id,)
            ).fetchone()
            return _usage_row(row, "worker_session_id") if row is not None else None
        row = None
        match = ""
        if run_started_at is not None and task_id:
            try:
                titled = connection.execute(
                    f"{_USAGE_SELECT} WHERE source = 'kanban' AND title = ? "
                    "AND started_at >= ? AND started_at <= ? "
                    "ORDER BY started_at DESC, id DESC LIMIT 1",
                    (
                        _WORKER_PROMPT_TITLE_PREFIX + task_id,
                        run_started_at - _WORKER_SESSION_LEAD_SECONDS,
                        run_started_at + _WORKER_SESSION_SPAWN_SECONDS,
                    ),
                ).fetchone()
            except sqlite3.OperationalError:
                # A sessions table without `title` (older host): the window
                # rule below still applies.
                titled = None
            if titled is not None:
                return _usage_row(titled, "worker_prompt_title")
        if run_started_at is not None:
            candidates = connection.execute(
                f"{_USAGE_SELECT} WHERE source = 'kanban' "
                "AND started_at >= ? AND started_at <= ? "
                "AND (cwd IS NULL OR cwd = '' OR cwd = ?) "
                "ORDER BY started_at DESC, id DESC LIMIT 2",
                (
                    run_started_at - _WORKER_SESSION_LEAD_SECONDS,
                    run_started_at + _WORKER_SESSION_SPAWN_SECONDS,
                    workspace_path,
                ),
            ).fetchall()
            if len(candidates) == 1:
                row, match = candidates[0], "dispatch_window"
    except sqlite3.Error:
        # A state.db whose schema moved, a lock held by the worker itself: the
        # lane keeps the board's own facts and reports no usage.
        return None
    return _usage_row(row, match) if row is not None else None


def _task_worker_usage(
    task: dict[str, Any],
    *,
    root: Path,
    connections: dict[Path, sqlite3.Connection | None],
) -> dict[str, Any] | None:
    """The worker session's figures for one task, or ``None`` when unlinked."""
    state_db = _worker_state_db(root, _text(task["assignee"], limit=80))
    if state_db is None:
        return None
    connection = _state_db_connection(connections, state_db)
    if connection is None:
        return None
    # A board with no `task_runs` still dates its claim on the task itself.
    run_started = _epoch(task["run_started_at"])
    if run_started is None:
        run_started = _epoch(task["started_at"])
    return _worker_usage(
        connection,
        task_id=_text(task["id"], limit=80),
        worker_session_id=_worker_session_id(task["run_metadata"]),
        workspace_path=_text(task["workspace_path"], limit=1024),
        run_started_at=run_started,
    )


def _first_skill(value: Any) -> str:
    """The first entry of the JSON skills list Hermes stores, else nothing."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = json.loads(value)
    except ValueError:
        return ""
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, str) and item.strip():
                return _text(item, limit=40)
    return ""


def _short_task_id(value: Any) -> str:
    """Hermes ids are ``t_`` + 8 hex; the HUD's id column is eight cells wide."""
    identity = _text(value, limit=40)
    return identity[2:] if identity.startswith("t_") else identity[:8]


def _lane_state(task: dict[str, Any], *, now: float) -> str:
    """The HUD verdict for a native status.

    Only native ``blocked`` is blocked. A recorded ``last_failure_error`` on a
    ``ready`` / ``review`` / ``todo`` row is a retry the dispatcher requeued
    (Hermes clears the error only on a later success), so that row is queued
    and ``retry_pending`` on the row carries the failure fact.
    """
    status = _text(task["status"], limit=20)
    if status in FINISHED_STATUSES:
        return "done"
    if status == "blocked":
        return "blocked"
    heartbeat = _epoch(task["last_heartbeat_at"])
    if (
        status == "running"
        and _epoch(task["worker_pid"]) is not None
        and heartbeat is not None
        and now - heartbeat > STALE_HEARTBEAT_SECONDS
    ):
        return "stale"
    if status == "running":
        return "running"
    return "queued"


def _lane_row(
    task: dict[str, Any],
    *,
    now: float,
    scope: str,
    board: str,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = _lane_state(task, now=now)
    started = _epoch(task["started_at"])
    completed = _epoch(task["completed_at"])
    heartbeat = _epoch(task["last_heartbeat_at"])
    created = _epoch(task["created_at"]) or 0.0
    # A finished row's elapsed is frozen at completion, a settled one at its
    # last heartbeat: only a running lane keeps aging, so a lingering row is
    # byte-stable and the widget can skip its repaint.
    if state == "running":
        frozen_at = now
    elif completed is not None:
        frozen_at = completed
    elif heartbeat is not None:
        frozen_at = heartbeat
    else:
        frozen_at = started
    # A lane that never started (Hermes creates tasks already `blocked`, and a
    # `ready` one waits for a claim) has no elapsed time: None, never a zero.
    elapsed = max(0.0, frozen_at - started) if started is not None and frozen_at is not None else None
    observed = max(value for value in (created, started, heartbeat, completed) if value is not None)
    model = _text(task["model_override"])
    row = {
        "scope": scope,
        "state": state,
        "task_id": _short_task_id(task["id"]),
        "role": _first_skill(task["skills"]) or "worker",
        "action": _text(task["title"], limit=_ACTION_LIMIT),
        "alias": "",
        "provider": _text(task["provider_override"]),
        "model": model,
        "effort": _text(task["reasoning_effort"], limit=20),
        # The board records no usage: None is "not observed", never a zero.
        "tokens": None,
        "elapsed_seconds": elapsed,
        "observed_at": _iso_utc(observed),
        "category": "",
        "completed": state == "done",
        "delegation_id": "",
        # Nothing on the board observes which model answered a worker's calls.
        "model_attestation": {"verdict": "unknown", "basis": "no_usage_observation"},
        "lane_backend": LANE_BACKEND,
        "assignee": _text(task["assignee"], limit=40),
        "board": board,
        "native_status": _text(task["status"], limit=20),
        "parent_count": int(task["parent_count"] or 0),
        # The dispatcher requeued this row after a failed run; it is waiting
        # for the next tick, not blocked.
        "retry_pending": state == "queued" and bool(_text(task["last_failure_error"])),
    }
    if usage is None:
        return row
    # The task's own pin wins where it has one: the override is what the
    # dispatcher was told to run, and the session only observes what answered.
    # `model_attestation` stays unknown either way -- a session row records
    # the model the worker was configured with, not the model that replied.
    row["model"] = model or usage["model"]
    row["effort"] = row["effort"] or usage["effort"]
    # Filled for every linked lane, including a settled one: a `review` row's
    # figures are the work that produced it, and `native_status` already says
    # the lane is not running. `kanban_request_review` is one of only two
    # calls that stamp the link at all, so gating it out by state would drop
    # the best-linked case on the board.
    row["tokens"] = usage["tokens"]
    row["turn_count"] = usage["turn_count"]
    row["tool_count"] = usage["tool_count"]
    row["usage_session_id"] = usage["session_id"]
    row["usage_match"] = usage["match"]
    if usage["cost_usd"] is not None:
        row["cost_usd"] = usage["cost_usd"]
    # The host's own word for a zero: it is what lets the widget render a
    # confirmed `$0.0000 (included)` instead of falling silent on it.
    if usage["cost_status"]:
        row["cost_status"] = usage["cost_status"]
    return row


def _dispatcher_presence(tasks: list[dict[str, Any]], *, now: float) -> str:
    if any(_epoch(task["worker_pid"]) is not None or _text(task["claim_lock"]) for task in tasks):
        return "observed"
    waiting = any(
        _text(task["status"], limit=20) == "ready"
        and (_epoch(task["created_at"]) or now) <= now - DISPATCHER_WAIT_SECONDS
        for task in tasks
    )
    return "not_observed" if waiting else "unknown"


def read_kanban_lanes(
    hermes_home: str | Path,
    *,
    now: float | None = None,
    session_ids: set[str] | frozenset[str] | None = None,
    board: str = "",
    limit: int = _ROW_LIMIT,
) -> dict[str, Any]:
    """Project the board's open lanes, owned by the conversation when it can be.

    Rows whose ``session_id`` is one of ``session_ids`` are scope ``session``;
    when identities were given and none match, the whole board projects as
    explicitly ``global`` -- the same fallback the native delegate reader makes
    for an unmapped conversation, never a borrowed session label.

    ``read_status`` says how the board read went, so a caller that lists lanes
    can say a read failed instead of rendering it as an empty board:
    ``refused`` (``kanban_db_path`` would not name a path), ``absent`` (no
    file), ``unreadable`` (``_query_tasks`` returned ``None``), else ``read``.
    ``selection_capped`` is True when the board held more matching tasks than
    ``_SELECT_LIMIT``: the counts and rows then cover the first ones only.
    """
    current = float(now) if now is not None else time.time()
    board_name = board.strip() or current_board(hermes_home)
    payload: dict[str, Any] = {
        "rows": [],
        "hidden": 0,
        "active": 0,
        "running": 0,
        "blocked": 0,
        "completed": 0,
        "queued": 0,
        "scope": "none",
        "dispatcher_presence": "unknown",
        "board": board_name,
        "read_status": "read",
        "selection_capped": False,
    }
    db = kanban_db_path(hermes_home, board_name)
    if db is None:
        payload["read_status"] = "refused"
        return payload
    if not db.is_file():
        payload["read_status"] = "absent"
        return payload
    tasks = _query_tasks(db, now=current)
    if tasks is None:
        payload["read_status"] = "unreadable"
        return payload
    payload["selection_capped"] = len(tasks) > _SELECT_LIMIT
    tasks = tasks[:_SELECT_LIMIT]
    if not tasks:
        return payload
    owners = set(session_ids or ())
    owned = [task for task in tasks if _text(task["session_id"], limit=160) in owners] if owners else []
    scope = "session" if owned else "global"
    selected = owned or tasks
    payload["scope"] = scope
    payload["dispatcher_presence"] = _dispatcher_presence(selected, now=current)
    # Counts are the whole selection's; only the rows that will be displayed
    # are built, so the per-task state.db lookup is bounded by the row cap and
    # never by the 64-row selection.
    for task in selected:
        state = _lane_state(task, now=current)
        if state in {"running", "stale"}:
            payload["running"] += 1
        elif state == "blocked":
            payload["blocked"] += 1
        elif state == "done":
            payload["completed"] += 1
        else:
            payload["queued"] += 1
    bound = max(1, int(limit))
    payload["hidden"] = max(0, len(selected) - bound)
    root = hermes_root(hermes_home)
    connections: dict[Path, sqlite3.Connection | None] = {}
    try:
        payload["rows"] = [
            _lane_row(
                task,
                now=current,
                scope=scope,
                board=board_name,
                usage=_task_worker_usage(task, root=root, connections=connections),
            )
            for task in selected[:bound]
        ]
    finally:
        _close_connections(connections)
    payload["active"] = payload["running"] + payload["blocked"]
    return payload
