"""What one piece of work cost, summed from records only.

A person asks in chat "how much did this cost?"; the answer has to cover
everything the work used, not only the turns of the session they are typing
in. This reader anchors "the work" on the calling Hermes conversation and adds
up four kinds of recorded spend:

- the conversation's own session rows: the session plus its compression
  continuations (`kanban_board_reader.conversation_session_ids`, the same
  identity set the HUD owns delegate children with);
- delegated Hermes children: every session Hermes marks with
  ``model_config._delegate_from`` below the conversation, walked the way
  Hermes' own cascade walks them (``hermes_state._collect_delegate_child_ids``:
  ``_delegate_from IN (frontier) OR (parent_session_id IN (frontier) AND
  _delegate_from IS NOT NULL)``), so an orchestrator child's own children and
  a child's compression continuation are included;
- fanout units whose dispatch summary carries this conversation's id in
  ``origin_session_id``. `omh coding fanout dispatch` stamps that field from
  ``HERMES_SESSION_ID``, which Hermes injects into every terminal command's
  environment as the session-db id of the spawning conversation. The same
  stamp on the summary covers the run's Hermes-lane and retarget recovery
  attempts;
- Hermes kanban worker runs on tasks whose ``tasks.session_id`` is in the
  conversation. Hermes stamps that column with the originating session when a
  task is created inside an agent loop (``kanban_tools._handle_create``,
  verified against ``state.db`` before stamping), and a worker that creates a
  follow-up task passes its own task's origin on, so the chain stays
  attributed. The worker's own session is the one it stamped on its run as
  ``task_runs.metadata.worker_session_id`` when it reported
  (``kanban_tools._stamp_worker_session_metadata``); that session, its
  continuations and its own delegates are read from the ``state.db`` of the
  profile the run was claimed for. A worker that ran without stamping its
  session is listed as missing -- no timing window or title is used to guess
  which session it was.

Hermes persists spend per API call as deltas into the calling session's own
row (``conversation_loop`` -> ``queue_token_counts``); the in-memory rollup of
a delegate's cost into its parent (``delegate_tool``) is never written back,
so summing a parent and its children's rows does not count a child twice.

Nothing is estimated. A usage row is priced only when Hermes recorded a cost
status other than ``unknown`` (or a cost source), the same rule
``hermes_delegation._informative_cost_row_sql`` applies; everything else is
reported as tokens with no recorded price, never folded into the dollar
figure and never priced from OMH's own ballpark table. A child or unit that
recorded no usage at all is listed as missing, not counted as zero. Metadata
only: ids, models, counts and amounts -- no prompt, reply, or transcript text
is read.
"""
from __future__ import annotations

import math
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from .fanout_scan import path_mtime
from .kanban_board_reader import (
    DEFAULT_BOARD,
    _board_root,
    _existing_slug,
    _worker_session_id,
    _worker_state_db,
    conversation_session_ids,
    hermes_root,
    kanban_db_path,
)
from .run_records import DISPATCH_SUMMARY_FILE, FANOUT_DISPATCH_SCHEMA_VERSION as _FANOUT_DISPATCH_SCHEMA_VERSION
from .runtime_reader import _read_hud_json
from .sqlite_readonly import connect_readonly

COST_RECEIPT_SCHEMA_VERSION = "omh_cost_receipt/v1"
_FANOUT_DIR_PREFIX = "fanout-"
# Bounds the walk and the listing, not the sum: a receipt that hit either
# says so in `truncated` rather than presenting a partial total as whole.
_MAX_SESSIONS = 2000
_MAX_FANOUT_SUMMARIES = 500
_MAX_MISSING_LISTED = 20
_MAX_KANBAN_RUNS = 500

COST_RECEIPT_CLAIM_BOUNDARY = (
    "Summed from Hermes' session accounting in state.db and OMH fanout dispatch summaries. "
    "Observed cost is only what a record priced; usage without a recorded price is reported "
    "as tokens and is not in the cost. It is not a provider invoice."
)

# Stated on every receipt so "not listed" is never read as "cost nothing".
NOT_COVERED = (
    "fanout units dispatched outside a Hermes session or before OMH recorded the originating session",
    "standalone `omh hermes-child dispatch` runs, which record no originating session",
    "Hermes kanban tasks created outside a Hermes session (CLI or dashboard), which record no originating session",
    "earlier attempts of a fanout unit that was dispatched again (the summary keeps the latest)",
)

_SOURCES = ("session", "delegated_children", "fanout_units", "kanban_workers")
_SOURCE_LABELS = {
    "session": "Hermes session",
    "delegated_children": "Delegated Hermes children",
    "fanout_units": "Fanout units",
    "kanban_workers": "Kanban workers",
}
_SOURCE_NOUNS = {"fanout_units": "unit", "kanban_workers": "worker run"}


def build_cost_receipt(*, hermes_home: str | Path, omh_home: str | Path, session_id: str) -> dict[str, Any]:
    """The receipt for the conversation ``session_id`` belongs to."""
    receipt: dict[str, Any] = {
        "schema_version": COST_RECEIPT_SCHEMA_VERSION,
        "session_id": session_id,
        "claim_boundary": COST_RECEIPT_CLAIM_BOUNDARY,
        "not_covered": list(NOT_COVERED),
    }
    state_db = Path(hermes_home).expanduser() / "state.db"
    if session_id and not state_db.is_file():
        receipt["status"] = "not_observed"
        receipt["reason"] = "Hermes state.db not found"
        receipt["text"] = "No cost receipt: Hermes' session store was not found."
        return receipt
    conversation = conversation_session_ids(hermes_home, session_id) if session_id else set()
    if not conversation:
        # The lineage read answers "no such row" and "could not read" alike;
        # an unreadable store is not a conversation that recorded nothing.
        if session_id and _read_sessions(state_db, {session_id}) is None:
            receipt["status"] = "not_observed"
            receipt["reason"] = "state.db could not be read"
            receipt["text"] = "No cost receipt: Hermes' session store could not be read."
            return receipt
        receipt["status"] = "not_observed"
        receipt["reason"] = "no Hermes session row for this session id"
        receipt["text"] = "No cost receipt: Hermes has no session record for this conversation."
        return receipt
    sessions = _read_sessions(state_db, conversation)
    if sessions is None:
        receipt["status"] = "not_observed"
        receipt["reason"] = "state.db could not be read"
        receipt["text"] = "No cost receipt: Hermes' session store could not be read."
        return receipt
    lineage_ids = {row["id"] for row in sessions["rows"]}
    buckets = {source: _empty_bucket() for source in _SOURCES}
    missing: list[dict[str, str]] = []
    for row in sessions["rows"]:
        source = "session" if row["id"] in conversation else "delegated_children"
        bucket = buckets[source]
        bucket["count"] += 1
        if not row["usage"]:
            if source == "delegated_children":
                missing.append({"source": source, "id": row["id"], "reason": "no usage recorded"})
            continue
        for usage in row["usage"]:
            _add_usage(bucket, usage)
    earliest = min((row["started_at"] for row in sessions["rows"] if row["started_at"] is not None), default=None)
    fanout_truncated = _add_fanout(buckets["fanout_units"], missing, Path(omh_home).expanduser(), lineage_ids, earliest)
    counted = {(_file_identity(state_db), session) for session in lineage_ids}
    kanban = _add_kanban(buckets["kanban_workers"], missing, hermes_home, lineage_ids, counted)
    for bucket in buckets.values():
        _finish_bucket(bucket)
    totals = _totals(buckets.values())
    # Which read hit its bound: the delegate walk (the conversation's own rows
    # are always read first), the fanout listing, or the kanban runs.
    bounded = {
        "delegated_children": sessions["truncated"],
        "fanout_units": fanout_truncated,
        "kanban_workers": kanban["truncated"],
    }
    truncated_sources = [source for source in _SOURCES if bounded.get(source)]
    receipt.update(
        {
            "status": "observed",
            "sessions_counted": len(lineage_ids),
            "sources": buckets,
            "observed_cost_usd": totals["cost_usd"],
            "priced_tokens": totals["priced_tokens"],
            "unpriced_tokens": totals["unpriced_tokens"],
            "missing": missing,
            "kanban": {key: kanban[key] for key in ("boards_read", "boards_unreadable")},
            "truncated": bool(truncated_sources),
            "truncated_sources": truncated_sources,
        }
    )
    receipt["text"] = format_cost_receipt(receipt)
    return receipt


def format_cost_receipt(receipt: Mapping[str, Any]) -> str:
    """Plain-language receipt; every number says whether it was observed."""
    if receipt.get("status") != "observed":
        return str(receipt.get("text") or "No cost receipt.")
    lines = ["Cost of this work, from recorded usage only (nothing estimated):"]
    cost = receipt.get("observed_cost_usd")
    lines.append(
        f"- Observed cost: {_money(cost)}" if cost is not None else "- Observed cost: none recorded"
    )
    unpriced = int(receipt.get("unpriced_tokens") or 0)
    if unpriced:
        lines.append(f"- Usage with no recorded price: {unpriced:,} tokens (not included in the cost above)")
    missing = list(receipt.get("missing") or ())
    if missing:
        lines.append(f"- Missing usage: {len(missing)} record(s) with no usage recorded (not counted as zero)")
    lines.append("By source:")
    for source in _SOURCES:
        bucket = (receipt.get("sources") or {}).get(source) or {}
        line = _bucket_line(source, bucket)
        if source == "kanban_workers":
            line = _kanban_line(line, receipt.get("kanban") or {})
        lines.append(f"- {_SOURCE_LABELS[source]}: {line}")
    if missing:
        lines.append("Missing:")
        for item in missing[:_MAX_MISSING_LISTED]:
            lines.append(f"- {item.get('id')}: {item.get('reason')}")
        if len(missing) > _MAX_MISSING_LISTED:
            lines.append(f"- and {len(missing) - _MAX_MISSING_LISTED} more")
    if receipt.get("truncated"):
        lines.append("This receipt hit its read bound; the totals cover only what was read.")
    lines.append("Not covered: " + "; ".join(receipt.get("not_covered") or NOT_COVERED) + ".")
    return "\n".join(lines)


def _bucket_line(source: str, bucket: Mapping[str, Any]) -> str:
    count = int(bucket.get("count") or 0)
    noun = _SOURCE_NOUNS.get(source, "session")
    head = f"{count} {noun}{'s' if count != 1 else ''}"
    if not count:
        return f"{head} recorded"
    parts = [head]
    if bucket.get("cost_usd") is not None:
        bases = ", ".join(bucket.get("cost_bases") or ())
        parts.append(f"{int(bucket.get('priced_tokens') or 0):,} tokens priced at {_money(bucket['cost_usd'])}"
                     + (f" ({bases})" if bases else ""))
    unpriced = int(bucket.get("unpriced_tokens") or 0)
    if unpriced:
        models = ", ".join(bucket.get("unpriced_models") or ()) or "model not recorded"
        parts.append(f"{unpriced:,} tokens with no recorded price ({models})")
    if bucket.get("cost_usd") is None and not unpriced:
        parts.append("no usage recorded")
    return ", ".join(parts)


def _kanban_line(line: str, kanban: Mapping[str, Any]) -> str:
    """Name the boards read, so "0 worker runs" is never read as "no board"."""
    read = list(kanban.get("boards_read") or ())
    unreadable = list(kanban.get("boards_unreadable") or ())
    if not read and not unreadable:
        return "no kanban board present"
    if read:
        line += f" (boards read: {', '.join(read)})"
    if unreadable:
        line += f"; board(s) that could not be read: {', '.join(unreadable)}"
    return line


def _money(value: Any) -> str:
    return f"${float(value):,.4f}"


def _empty_bucket() -> dict[str, Any]:
    return {
        "count": 0,
        "cost_usd": None,
        "priced_tokens": 0,
        "unpriced_tokens": 0,
        "cost_bases": set(),
        "unpriced_models": set(),
    }


def _finish_bucket(bucket: dict[str, Any]) -> None:
    bucket["cost_bases"] = sorted(bucket["cost_bases"])
    bucket["unpriced_models"] = sorted(bucket["unpriced_models"])
    if bucket["cost_usd"] is not None:
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)


def _totals(buckets: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cost: float | None = None
    priced = unpriced = 0
    for bucket in buckets:
        if bucket.get("cost_usd") is not None:
            cost = (cost or 0.0) + float(bucket["cost_usd"])
        priced += int(bucket.get("priced_tokens") or 0)
        unpriced += int(bucket.get("unpriced_tokens") or 0)
    return {
        "cost_usd": round(cost, 6) if cost is not None else None,
        "priced_tokens": priced,
        "unpriced_tokens": unpriced,
    }


def _add_usage(bucket: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Fold one usage record into a bucket: priced when a record priced it."""
    tokens = int(usage.get("tokens") or 0)
    amount = usage.get("cost_usd")
    if amount is not None:
        bucket["cost_usd"] = (bucket["cost_usd"] or 0.0) + float(amount)
        bucket["priced_tokens"] += tokens
        bucket["cost_bases"].add(str(usage.get("basis") or "recorded"))
        return
    bucket["unpriced_tokens"] += tokens
    if tokens:
        bucket["unpriced_models"].add(str(usage.get("model") or "model not recorded"))


def hermes_usage_record(
    *,
    model: Any,
    tokens: int,
    actual_cost: Any,
    estimated_cost: Any,
    cost_status: Any,
    cost_source: Any,
) -> dict[str, Any]:
    """One Hermes usage row as a record: priced only when Hermes priced it.

    ``unknown`` is Hermes' own no-figure status (``usage_pricing._unknown_cost``
    returns ``amount_usd=None``), and a row with neither status nor source said
    nothing about cost; both leave the tokens unpriced. Any other status --
    ``estimated``, ``actual``, ``included`` and whatever billing word a host
    adds later -- vouches for the amount beside it, zero included.
    """
    status = _text(cost_status)
    source = _text(cost_source)
    record: dict[str, Any] = {"model": _text(model), "tokens": tokens}
    if status == "unknown" or not (status or source):
        return record
    actual = _amount(actual_cost)
    estimated = _amount(estimated_cost)
    record["cost_usd"] = actual if actual else (estimated or 0.0)
    record["basis"] = f"Hermes {status or source}"
    return record


def _read_sessions(state_db: Path, conversation: set[str]) -> dict[str, Any] | None:
    try:
        connection = connect_readonly(state_db, timeout=0.5)
    except sqlite3.Error:
        return None
    try:
        columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("sessions")')}
        ids, truncated = _lineage_ids(connection, conversation, columns)
        usage_by_session = _usage_rows(connection, ids)
        rows = []
        placeholders = ",".join("?" for _ in ids)
        wanted = [
            name
            for name in (
                "id", "model", "started_at", "input_tokens", "output_tokens",
                "actual_cost_usd", "estimated_cost_usd", "cost_status", "cost_source", "api_call_count",
            )
            if name in columns
        ]
        for values in connection.execute(
            f"SELECT {', '.join(wanted)} FROM sessions WHERE id IN ({placeholders}) ORDER BY started_at, id",
            sorted(ids),
        ):
            row = dict(zip(wanted, values))
            session_id = str(row["id"])
            usage = usage_by_session.get(session_id)
            if usage is None:
                usage = _session_row_usage(row)
            rows.append({"id": session_id, "started_at": _amount(row.get("started_at")), "usage": usage})
        return {"rows": rows, "truncated": truncated}
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def _lineage_ids(connection: sqlite3.Connection, conversation: set[str], columns: set[str]) -> tuple[set[str], bool]:
    """The conversation plus every delegate below it, walked as Hermes walks it."""
    found = set(conversation)
    if not {"parent_session_id", "model_config"} <= columns:
        return found, False
    delegate_from = (
        "CASE WHEN json_valid(model_config) THEN json_extract(model_config, '$._delegate_from') END"
    )
    frontier = sorted(conversation)
    while frontier:
        if len(found) >= _MAX_SESSIONS:
            return found, True
        placeholders = ",".join("?" for _ in frontier)
        rows = connection.execute(
            f"SELECT id FROM sessions WHERE {delegate_from} IN ({placeholders}) "
            f"OR (parent_session_id IN ({placeholders}) AND {delegate_from} IS NOT NULL)",
            [*frontier, *frontier],
        ).fetchall()
        frontier = sorted({str(row[0]) for row in rows} - found)
        found.update(frontier)
    return found, False


def _usage_rows(connection: sqlite3.Connection, ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    """Per-model usage rows; a session with none falls back to its own row."""
    columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("session_model_usage")')}
    needed = {"session_id", "model", "input_tokens", "output_tokens", "estimated_cost_usd", "actual_cost_usd"}
    if not needed <= columns:
        return {}
    status = "cost_status" if "cost_status" in columns else "NULL"
    source = "cost_source" if "cost_source" in columns else "NULL"
    placeholders = ",".join("?" for _ in ids)
    usage: dict[str, list[dict[str, Any]]] = {}
    for session_id, model, input_tokens, output_tokens, estimated, actual, cost_status, cost_source in connection.execute(
        f"SELECT session_id, model, input_tokens, output_tokens, estimated_cost_usd, actual_cost_usd, "
        f"{status}, {source} FROM session_model_usage WHERE session_id IN ({placeholders}) "
        "ORDER BY session_id, model",
        sorted(ids),
    ):
        tokens = _count(input_tokens) + _count(output_tokens)
        usage.setdefault(str(session_id), []).append(
            hermes_usage_record(
                model=model, tokens=tokens, actual_cost=actual, estimated_cost=estimated,
                cost_status=cost_status, cost_source=cost_source,
            )
        )
    return usage


def _session_row_usage(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    tokens = _count(row.get("input_tokens")) + _count(row.get("output_tokens"))
    if not tokens and not _count(row.get("api_call_count")):
        return []
    return [
        hermes_usage_record(
            model=row.get("model"), tokens=tokens, actual_cost=row.get("actual_cost_usd"),
            estimated_cost=row.get("estimated_cost_usd"), cost_status=row.get("cost_status"),
            cost_source=row.get("cost_source"),
        )
    ]


def _add_fanout(
    bucket: dict[str, Any],
    missing: list[dict[str, str]],
    omh_home: Path,
    lineage_ids: set[str],
    earliest: float | None,
) -> bool:
    """Add units whose dispatch summary names this conversation; True when bounded."""
    root = omh_home / "coding" / "fanout"
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            return False
        names = sorted(entry.name for entry in root.iterdir() if entry.name.startswith(_FANOUT_DIR_PREFIX))
    except OSError:
        return False
    read = 0
    for name in names:
        summary_path = root / name / DISPATCH_SUMMARY_FILE
        written = path_mtime(summary_path)
        # A summary last written before the conversation began cannot hold a
        # unit it dispatched; skipping it is a bound, not an attribution.
        if written is None or (earliest is not None and written < earliest):
            continue
        if read >= _MAX_FANOUT_SUMMARIES:
            return True
        read += 1
        summary = _read_hud_json(summary_path, root=omh_home)
        if summary.get("schema_version") != _FANOUT_DISPATCH_SCHEMA_VERSION:
            continue
        for unit in summary.get("units") or ():
            if isinstance(unit, dict) and str(unit.get("origin_session_id") or "") in lineage_ids:
                _add_fanout_run(bucket, missing, unit, f"{name}/{unit.get('unit_id', '')}")
        if str(summary.get("origin_session_id") or "") in lineage_ids:
            for decision in (summary.get("failure_recovery") or {}).get("decisions") or ():
                attempt = decision.get("attempt") if isinstance(decision, dict) else None
                if isinstance(attempt, dict):
                    _add_fanout_run(
                        bucket, missing, attempt, f"{name}/{decision.get('unit_id', '')} recovery attempt"
                    )
    return False


def _add_fanout_run(bucket: dict[str, Any], missing: list[dict[str, str]], run: Mapping[str, Any], label: str) -> None:
    """One unit run or recovery attempt; a run that never started cost nothing."""
    if "exit_code" not in run:
        return
    bucket["count"] += 1
    usage = run.get("usage")
    if isinstance(usage, dict):
        # A Hermes-lane attempt carries the child's own state.db usage.
        tokens = _count(usage.get("total_tokens"))
        record = hermes_usage_record(
            model=usage.get("model") or run.get("model"), tokens=tokens, actual_cost=None,
            estimated_cost=usage.get("estimated_cost_usd"), cost_status=usage.get("cost_status"),
            cost_source=usage.get("cost_source"),
        )
    else:
        tokens = _count(run.get("tokens_total")) or (_count(run.get("input_tokens")) + _count(run.get("output_tokens")))
        record = {"model": _text(run.get("model")) or _text(run.get("owner")), "tokens": tokens}
        cost = _amount(run.get("cost_usd"))
        if cost is not None:
            record["cost_usd"] = cost
            record["basis"] = "executor reported"
    if record.get("cost_usd") is None and not record["tokens"]:
        missing.append({"source": "fanout_units", "id": label, "reason": "ran, reported no usage"})
        return
    _add_usage(bucket, record)


def _file_identity(path: Path) -> tuple[int, int] | str:
    """One key per database file, however the path to it was spelled."""
    try:
        info = os.stat(path)
    except OSError:
        return str(path)
    return (info.st_dev, info.st_ino)


def _board_paths(hermes_home: str | Path) -> list[tuple[str, Path]]:
    """Every board database that exists: the default board and each named one.

    A task stamped with this conversation may sit on any board, not only the
    current one, so all of them are read. A named board is admitted the way
    Hermes admits it (``_existing_slug``), and ``kanban_db_path`` refuses a
    symlinked database.
    """
    root = _board_root(hermes_home)
    slugs = [DEFAULT_BOARD]
    boards_dir = root / "kanban" / "boards"
    try:
        if stat.S_ISDIR(boards_dir.lstat().st_mode):
            named = {_existing_slug(root, entry.name) for entry in boards_dir.iterdir()}
            slugs += sorted(slug for slug in named if slug and slug != DEFAULT_BOARD)
    except OSError:
        pass
    found = []
    for slug in slugs:
        path = kanban_db_path(hermes_home, slug)
        if path is not None and path.is_file():
            found.append((slug, path))
    return found


def _kanban_runs(board_db: Path, lineage_ids: set[str]) -> list[dict[str, Any]] | None:
    """Runs on tasks this conversation created; ``None`` when the board is unreadable.

    ``spawned`` is whether the dispatcher recorded a ``spawned`` event for the
    run. ``task_runs.worker_pid`` cannot say it: the dispatcher clears it once
    it reaps the worker.
    """
    try:
        connection = connect_readonly(board_db, timeout=0.5)
    except sqlite3.Error:
        return None
    try:
        task_columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("tasks")')}
        run_columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("task_runs")')}
        event_columns = {str(row[1]) for row in connection.execute('PRAGMA table_info("task_events")')}
        needed_runs = {"id", "task_id", "profile", "status", "ended_at", "metadata", "started_at"}
        if not ({"id", "session_id", "assignee"} <= task_columns and needed_runs <= run_columns):
            return None
        spawned = (
            "EXISTS (SELECT 1 FROM task_events e WHERE e.run_id = r.id AND e.kind = 'spawned')"
            if {"run_id", "kind"} <= event_columns
            else "0"
        )
        placeholders = ",".join("?" for _ in lineage_ids)
        rows = connection.execute(
            f"SELECT r.id, r.task_id, r.profile, t.assignee, r.status, r.ended_at, r.metadata, {spawned} "
            f"FROM task_runs r JOIN tasks t ON t.id = r.task_id WHERE t.session_id IN ({placeholders}) "
            "ORDER BY r.started_at, r.id LIMIT ?",
            [*sorted(lineage_ids), _MAX_KANBAN_RUNS + 1],
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    keys = ("run_id", "task_id", "profile", "assignee", "status", "ended_at", "metadata", "spawned")
    return [dict(zip(keys, row)) for row in rows]


def _add_kanban(
    bucket: dict[str, Any],
    missing: list[dict[str, str]],
    hermes_home: str | Path,
    lineage_ids: set[str],
    counted: set[tuple[Any, str]],
) -> dict[str, Any]:
    """Add worker runs on this conversation's kanban tasks.

    A run with neither a ``spawned`` event nor a stamped session started no
    worker (a spawn that failed, a run Hermes synthesized for a manual
    completion) and cost nothing. ``counted`` holds every (database, session)
    already summed, so a session the delegate walk reached, or one two runs
    both stamped, is added once.
    """
    result: dict[str, Any] = {"boards_read": [], "boards_unreadable": [], "truncated": False}
    root = hermes_root(hermes_home)
    for slug, board_db in _board_paths(hermes_home):
        runs = _kanban_runs(board_db, lineage_ids)
        if runs is None:
            result["boards_unreadable"].append(slug)
            continue
        result["boards_read"].append(slug)
        if len(runs) > _MAX_KANBAN_RUNS:
            runs = runs[:_MAX_KANBAN_RUNS]
            result["truncated"] = True
        for run in runs:
            worker_session = _worker_session_id(run["metadata"])
            if not (run["spawned"] or worker_session):
                continue
            bucket["count"] += 1
            label = f"kanban {slug}/{run['task_id']} run {run['run_id']}"
            if not worker_session:
                reason = (
                    "worker still running; it records its session when it reports"
                    if run["status"] == "running" and run["ended_at"] is None
                    else "worker ran and recorded no session on its run"
                )
            else:
                profile = _text(run["profile"], limit=80) or _text(run["assignee"], limit=80)
                reason = _add_worker_session(
                    bucket, missing, result, _worker_state_db(root, profile), worker_session, counted
                )
            if reason:
                missing.append({"source": "kanban_workers", "id": label, "reason": reason})
    return result


def _add_worker_session(
    bucket: dict[str, Any],
    missing: list[dict[str, str]],
    result: dict[str, Any],
    worker_db: Path | None,
    worker_session: str,
    counted: set[tuple[Any, str]],
) -> str:
    """Fold in the worker session, its continuations and delegates; the missing reason, else ``""``."""
    if worker_db is None:
        return "worker profile has no readable state.db"
    identity = _file_identity(worker_db)
    worker_conversation = conversation_session_ids(worker_db.parent, worker_session)
    if not worker_conversation:
        return f"worker session {worker_session} has no row in its profile's state.db"
    sessions = _read_sessions(worker_db, worker_conversation)
    if sessions is None:
        return "worker profile state.db could not be read"
    result["truncated"] = result["truncated"] or sessions["truncated"]
    for row in sessions["rows"]:
        key = (identity, row["id"])
        if key in counted:
            continue
        counted.add(key)
        if not row["usage"]:
            missing.append({"source": "kanban_workers", "id": row["id"], "reason": "no usage recorded"})
            continue
        for usage in row["usage"]:
            _add_usage(bucket, usage)
    return ""


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _amount(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _text(value: Any, limit: int = 120) -> str:
    return str(value or "").strip()[:limit]


__all__ = [
    "COST_RECEIPT_SCHEMA_VERSION",
    "build_cost_receipt",
    "format_cost_receipt",
    "hermes_usage_record",
]
