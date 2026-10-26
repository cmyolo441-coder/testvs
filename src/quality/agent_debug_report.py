"""``agent_debug_report/v1``: cited findings for one agent session.

The ``agent-debug`` skill diagnoses a stuck, looping or failing agent run, and
until this reader its report was a declared name with nothing behind it: a
plausible narrative looked the same as a diagnosis. This reader gives the
diagnosis a floor of observed rows. It reads one of two sources and derives a
closed set of finding kinds from record fields only:

- Hermes' own session store, ``<hermes_home>/state.db``, opened ``mode=ro``
  through ``hermes_state``, the open the reply lint and the session-usage
  report share. A finding cites ``messages.id`` values.
- A supplied session record: a JSON Lines file, one message object per line
  with the fields a ``messages`` row has (``session_id``, ``role``,
  ``content``, ``tool_call_id``, ``tool_name``, ``tool_calls``,
  ``timestamp``, ``_compressed_summary``). A finding cites line numbers, so
  every reference is ``<record>:<line>``.

The finding kinds:

- ``tool_error``: a tool result whose JSON object records an error in a typed
  field -- a non-zero integer ``exit_code``, ``success`` false, or a non-empty
  string ``error``. No word of the result is matched.
- ``identical_retry_after_error``: the next call of the same tool after a
  ``tool_error`` carried byte-identical arguments (compared by the sha256 of
  the canonical JSON of the assistant row's ``tool_calls`` arguments).
- ``background_without_notify``: a tool result that records a started process
  (an integer ``pid``) with ``notify_on_complete`` not true, so nothing will
  bring its completion back into the turn.
- ``compaction_boundary``: a message row marked ``_compressed_summary``, the
  summary a compaction put in place of the rows before it.

Selection is exact or refused. ``--hermes-session`` takes a full id,
``latest``, or a prefix that names exactly one session; a prefix that names
more than one is an error that lists them, never a guess. An optional turn
range (1-based, counted over user rows that are not compaction summaries)
narrows the rows read to those turns.

Reading is bounded. At most ``max_rows`` rows (state.db) or lines (record)
are read, and at most ``max_row_bytes`` of any one cell or line is taken into
memory: state.db cells are cut inside SQLite with ``substr``, and a record
line longer than the budget is skipped in chunks without being kept. Each row
is reduced on read to the typed fields the kinds need, so no content, prompt,
argument or tool output is retained. A row over the byte budget is listed by
reference under ``budget.oversized_refs`` and was not checked; a session cut
short by the row budget says so in ``budget.row_limit_reached``.
``budget.complete`` is true only when neither happened, and the hypothesis
layer will not treat an absence as evidence unless it is.

Every finding carries a citation -- session id, message ids (or record lines)
with their timestamps, tool_call ids, tool name, error class, exit code,
argument digest -- and nothing else. ``agent_debug_report_errors`` refuses a
report whose finding lacks the citation its kind requires, cites another
session, or carries a key outside the closed citation shape; the builder runs
it on every report it returns. ``agent_debug_reference_errors`` re-reads the
cited rows from the source and refuses a reference that is missing, foreign,
stale or mismatched.

A compaction re-persists rows under new ids, so a tool call is counted once
per distinct ``tool_call_id``, the first row deciding what it was. A field the
source does not have leaves its kind ``unavailable`` rather than silently
clean. The source is never written.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, BinaryIO, Iterator, Mapping

from .hermes_state import (
    HERMES_LATEST_SESSION,
    TOOL_CALL_KEY_SQL,
    hermes_epoch,
    open_state_db_readonly,
    resolve_session_id,
)


AGENT_DEBUG_REPORT_SCHEMA_VERSION = "agent_debug_report/v1"

TOOL_ERROR = "tool_error"
IDENTICAL_RETRY_AFTER_ERROR = "identical_retry_after_error"
BACKGROUND_WITHOUT_NOTIFY = "background_without_notify"
COMPACTION_BOUNDARY = "compaction_boundary"
FINDING_KINDS: tuple[str, ...] = (
    TOOL_ERROR,
    IDENTICAL_RETRY_AFTER_ERROR,
    BACKGROUND_WITHOUT_NOTIFY,
    COMPACTION_BOUNDARY,
)
ERROR_CLASSES: tuple[str, ...] = ("nonzero_exit", "success_false", "error_field")
ARGUMENT_DIGEST_CHARS = 16

SOURCE_STATE_DB = "hermes_state_db"
SOURCE_RECORD = "session_record"
LOCATOR_MESSAGE_ID = "state_db_message_id"
LOCATOR_RECORD_LINE = "record_line"
LOCATORS: dict[str, str] = {SOURCE_STATE_DB: LOCATOR_MESSAGE_ID, SOURCE_RECORD: LOCATOR_RECORD_LINE}

SELECTION_EXACT = "exact"
SELECTION_LATEST = "latest"
SELECTION_UNIQUE_PREFIX = "unique_prefix"
SELECTION_ONLY_SESSION = "only_session"
SELECTIONS: tuple[str, ...] = (SELECTION_EXACT, SELECTION_LATEST, SELECTION_UNIQUE_PREFIX, SELECTION_ONLY_SESSION)

DEFAULT_MAX_ROWS = 20_000
DEFAULT_MAX_ROW_BYTES = 256 * 1024
MAX_LISTED_OVERSIZED = 20
_AMBIGUOUS_LISTED = 5
_SKIP_CHUNK_BYTES = 64 * 1024

CITATION_KEYS: tuple[str, ...] = (
    "session_id",
    "message_ids",
    "timestamps",
    "tool_call_ids",
    "tool_name",
    "error_class",
    "exit_code",
    "arguments_sha256",
)
# What each kind must cite beyond the session id, message ids and timestamps
# every finding carries, and how many rows it spans.
_REQUIRED_CITATION: dict[str, tuple[int, tuple[str, ...]]] = {
    TOOL_ERROR: (1, ("tool_call_ids", "tool_name", "error_class")),
    IDENTICAL_RETRY_AFTER_ERROR: (2, ("tool_call_ids", "tool_name", "error_class", "arguments_sha256")),
    BACKGROUND_WITHOUT_NOTIFY: (1, ("tool_call_ids", "tool_name")),
    COMPACTION_BOUNDARY: (1, ()),
}
_FINDING_KEYS = frozenset({"finding_id", "kind", "citation"})
_SUMMARY_COLUMN = "_compressed_summary"
# The only result fields any kind reads. A row is reduced to these on read.
_TYPED_RESULT_KEYS = ("exit_code", "success", "pid", "notify_on_complete")
# Why a kind can be unavailable: the source lacks the field it is derived from.
# The closed set the readers write and the validator accepts.
UNAVAILABLE_REASONS: dict[str, dict[str, str]] = {
    SOURCE_STATE_DB: {
        IDENTICAL_RETRY_AFTER_ERROR: "messages.tool_calls column not present",
        COMPACTION_BOUNDARY: f"messages.{_SUMMARY_COLUMN} column not present",
    },
    SOURCE_RECORD: {
        IDENTICAL_RETRY_AFTER_ERROR: "no record line carries a tool_calls field",
        COMPACTION_BOUNDARY: f"no record line carries a {_SUMMARY_COLUMN} field",
    },
}

AGENT_DEBUG_REPORT_CLAIM_BOUNDARY = (
    "An agent debug report cites rows one session record persisted, read without writing to it, and "
    "quotes no prompt, reply, argument, or tool output. A finding is an observed record, not a diagnosis: "
    "it does not show why the agent acted as it did, that a retry was wrong, that a compaction lost what "
    "mattered, or that any recovery worked, and it is not execution, review, CI, or merge evidence. A kind "
    "listed as unavailable was not checked, and rows listed as oversized or past the row budget were not read."
)


class AgentDebugReportError(ValueError):
    """The session could not be read or selected, or a report failed validation; the message says which."""


def parse_turn_range(text: str | None) -> tuple[int, int | None] | None:
    """``"3"`` -> (3, 3), ``"2:5"`` -> (2, 5), ``"4:"`` -> (4, None); 1-based and inclusive."""
    if text is None:
        return None
    value = str(text).strip()
    head, sep, tail = value.partition(":")
    try:
        start = int(head)
        end: int | None = int(tail) if tail.strip() else (None if sep else start)
    except ValueError:
        raise AgentDebugReportError(f"turn range {value!r} must be N, N:M, or N: with whole numbers") from None
    if start < 1 or (end is not None and end < start):
        raise AgentDebugReportError(f"turn range {value!r} must start at 1 or later and not end before it starts")
    return start, end


def build_agent_debug_report(
    hermes_home: str | Path | None,
    session_id: str | None,
    *,
    session_record: str | Path | None = None,
    turns: tuple[int, int | None] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_row_bytes: int = DEFAULT_MAX_ROW_BYTES,
) -> dict[str, Any]:
    """Read one session and return a validated ``agent_debug_report/v1``.

    The source is ``<hermes_home>/state.db`` unless ``session_record`` names a
    JSON Lines record. ``session_id`` is an id, ``latest``, or a unique id
    prefix; with a record it may be omitted when the record holds one session.
    """
    if max_rows < 1 or max_row_bytes < 1:
        raise AgentDebugReportError("max_rows and max_row_bytes must be at least 1")
    if session_record is not None:
        read = _read_record(Path(session_record), session_id, turns, max_rows, max_row_bytes)
    else:
        if hermes_home is None:
            raise AgentDebugReportError("a Hermes home or a session record is required")
        read = _read_state_db(hermes_home, session_id, turns, max_rows, max_row_bytes)
    report = _report_from_rows(read, max_rows=max_rows, max_row_bytes=max_row_bytes)
    errors = agent_debug_report_errors(report)
    if errors:
        raise AgentDebugReportError("agent_debug_report/v1 failed validation: " + "; ".join(errors))
    return report


# --- reading ---------------------------------------------------------------


def _read_state_db(
    hermes_home: str | Path,
    session_id: str | None,
    turns: tuple[int, int | None] | None,
    max_rows: int,
    max_row_bytes: int,
) -> dict[str, Any]:
    path, connection = open_state_db_readonly(hermes_home, error=AgentDebugReportError)
    try:
        resolved, selection = _resolve_state_db_session(connection, session_id)
        cursor = connection.execute("SELECT * FROM sessions WHERE id = ?", (resolved,))
        session_row = cursor.fetchone()
        if session_row is None:
            raise AgentDebugReportError(f"no Hermes session {resolved}")
        session_fields = dict(zip((column[0] for column in cursor.description), session_row))
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")}
        has_calls = "tool_calls" in columns
        has_summary = _SUMMARY_COLUMN in columns
        not_summary = f" AND {_SUMMARY_COLUMN} = 0" if has_summary else ""
        user_ids = [
            int(row[0])
            for row in connection.execute(
                f"SELECT id FROM messages WHERE session_id = ? AND role = 'user'{not_summary} ORDER BY id", (resolved,)
            )
        ]
        low, high = _turn_window(user_ids, turns)
        count, max_id = connection.execute(
            "SELECT COUNT(*), MAX(id) FROM messages WHERE session_id = ?", (resolved,)
        ).fetchone()
        where = "session_id = :sid"
        params: dict[str, Any] = {"sid": resolved, "cap": max_row_bytes, "limit": max_rows + 1}
        if low is not None:
            where += " AND id >= :low"
            params["low"] = low
        if high is not None:
            where += " AND id < :high"
            params["high"] = high
        calls_sql = (
            "substr(CAST(tool_calls AS BLOB), 1, :cap), length(CAST(tool_calls AS BLOB))" if has_calls else "NULL, NULL"
        )
        summary_sql = _SUMMARY_COLUMN if has_summary else "NULL"
        cursor = connection.execute(
            f"SELECT id, role, {TOOL_CALL_KEY_SQL}, tool_call_id, tool_name, "
            "substr(CAST(content AS BLOB), 1, :cap), length(CAST(content AS BLOB)), "
            f"{calls_sql}, {summary_sql}, timestamp FROM messages WHERE {where} ORDER BY id LIMIT :limit",
            params,
        )
        # One row at a time off the cursor, reduced before the next arrives, so
        # at most one budget-cut row's cells are held at once. The query asks
        # for one row past the budget only to learn that the limit was reached.
        rows: list[dict[str, Any]] = []
        bytes_read = 0
        row_limit_reached = False
        for message_id, role, call_key, tool_call_id, tool_name, content, content_len, calls, calls_len, summary, stamp in cursor:
            if len(rows) >= max_rows:
                row_limit_reached = True
                break
            bytes_read += len(content or b"") + len(calls or b"")
            content_oversized = int(content_len or 0) > max_row_bytes
            calls_oversized = int(calls_len or 0) > max_row_bytes
            rows.append(
                _reduced_row(
                    ref=int(message_id),
                    session_id=resolved,
                    role=str(role or ""),
                    call_key=str(call_key),
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=None if content_oversized else _text(content),
                    tool_calls=None if calls_oversized else _json_value(_text(calls)),
                    summary=None if summary is None else bool(summary),
                    stamp=stamp,
                    oversized=content_oversized or calls_oversized,
                )
            )
    except sqlite3.Error as exc:
        raise AgentDebugReportError(f"could not read {path}: {exc}") from exc
    finally:
        connection.close()

    reasons = UNAVAILABLE_REASONS[SOURCE_STATE_DB]
    unavailable: list[dict[str, str]] = []
    if not has_calls:
        unavailable.append({"kind": IDENTICAL_RETRY_AFTER_ERROR, "reason": reasons[IDENTICAL_RETRY_AFTER_ERROR]})
    if not has_summary:
        unavailable.append({"kind": COMPACTION_BOUNDARY, "reason": reasons[COMPACTION_BOUNDARY]})
    return {
        "source": {
            "kind": SOURCE_STATE_DB,
            "path": str(path),
            "label": path.name,
            "locator": LOCATOR_MESSAGE_ID,
            "requested_session": None if session_id is None else str(session_id),
            "selection": selection,
            "snapshot": {"session_rows": int(count or 0), "max_message_id": None if max_id is None else int(max_id)},
        },
        "session": {
            "id": resolved,
            "source": session_fields.get("source") or None,
            "started_at": hermes_epoch(session_fields.get("started_at")),
            "ended_at": hermes_epoch(session_fields.get("ended_at")),
            "end_reason": session_fields.get("end_reason") or None,
        },
        "turns": _turns_block(turns, len(user_ids)),
        "rows": rows,
        "rows_read": len(rows),
        "row_limit_reached": row_limit_reached,
        "bytes_read": bytes_read,
        "bytes_skipped": 0,
        "unavailable": unavailable,
    }


def _resolve_state_db_session(connection: sqlite3.Connection, session_id: str | None) -> tuple[str, str]:
    wanted = str(session_id or "").strip()
    if not wanted:
        raise AgentDebugReportError("a session id (or `latest`) is required to read state.db")
    if wanted == HERMES_LATEST_SESSION:
        return resolve_session_id(connection, wanted, None, error=AgentDebugReportError), SELECTION_LATEST
    if connection.execute("SELECT 1 FROM sessions WHERE id = ?", (wanted,)).fetchone() is not None:
        return wanted, SELECTION_EXACT
    matches = [
        str(row[0])
        for row in connection.execute(
            "SELECT id FROM sessions WHERE substr(id, 1, ?) = ? ORDER BY id LIMIT ?",
            (len(wanted), wanted, _AMBIGUOUS_LISTED + 1),
        )
    ]
    return _single_match(wanted, matches), SELECTION_UNIQUE_PREFIX


def _single_match(wanted: str, matches: list[str]) -> str:
    if not matches:
        raise AgentDebugReportError(f"no Hermes session {wanted}")
    if len(matches) > 1:
        listed = ", ".join(matches[:_AMBIGUOUS_LISTED]) + (", ..." if len(matches) > _AMBIGUOUS_LISTED else "")
        raise AgentDebugReportError(
            f"session {wanted!r} is ambiguous: it matches more than one session ({listed}); give the full id"
        )
    return matches[0]


def _read_record(
    path: Path,
    session_id: str | None,
    turns: tuple[int, int | None] | None,
    max_rows: int,
    max_row_bytes: int,
) -> dict[str, Any]:
    try:
        info = path.stat()
    except OSError as exc:
        raise AgentDebugReportError(f"no session record at {path}: {exc.strerror or exc}") from exc
    if not path.is_file():
        raise AgentDebugReportError(f"session record {path} is not a regular file")
    rows: list[dict[str, Any]] = []
    oversized_lines: list[int] = []
    bytes_read = bytes_skipped = lines_read = 0
    row_limit_reached = False
    try:
        with path.open("rb") as handle:
            for line_number, raw, skipped in _bounded_lines(handle, max_row_bytes):
                if lines_read >= max_rows:
                    row_limit_reached = True
                    break
                lines_read += 1
                bytes_read += len(raw)
                bytes_skipped += skipped
                if skipped:
                    oversized_lines.append(line_number)
                    continue
                message = _json_value(raw.decode("utf-8", "replace"))
                if not isinstance(message, dict):
                    continue
                rows.append(
                    _reduced_row(
                        ref=line_number,
                        session_id=message.get("session_id"),
                        role=str(message.get("role") or ""),
                        call_key=str(message.get("tool_call_id") or f"row:{line_number}"),
                        tool_call_id=message.get("tool_call_id"),
                        tool_name=message.get("tool_name"),
                        content=message.get("content") if isinstance(message.get("content"), str) else None,
                        tool_calls=message.get("tool_calls") if "tool_calls" in message else _ABSENT,
                        summary=bool(message[_SUMMARY_COLUMN]) if _SUMMARY_COLUMN in message else None,
                        stamp=message.get("timestamp"),
                        oversized=False,
                    )
                )
    except OSError as exc:
        raise AgentDebugReportError(f"could not read {path}: {exc.strerror or exc}") from exc

    sessions: dict[str, float] = {}
    for row in rows:
        if row["session_id"]:
            stamp = row["timestamp"] if row["timestamp"] is not None else -math.inf
            sessions[row["session_id"]] = max(sessions.get(row["session_id"], -math.inf), stamp)
    resolved, selection = _resolve_record_session(session_id, sessions)
    selected = [row for row in rows if row["session_id"] == resolved]
    user_turn = 0
    windowed: list[dict[str, Any]] = []
    for row in selected:
        if row["role"] == "user" and not row["summary"]:
            user_turn += 1
        if turns is None or (user_turn >= turns[0] and (turns[1] is None or user_turn <= turns[1])):
            windowed.append(row)
    if turns is not None and turns[0] > user_turn:
        raise AgentDebugReportError(f"turn {turns[0]} is past the session's last turn ({user_turn})")
    reasons = UNAVAILABLE_REASONS[SOURCE_RECORD]
    unavailable: list[dict[str, str]] = []
    if not any(row["has_tool_calls_field"] for row in selected):
        unavailable.append({"kind": IDENTICAL_RETRY_AFTER_ERROR, "reason": reasons[IDENTICAL_RETRY_AFTER_ERROR]})
    if not any(row["summary"] is not None for row in selected):
        unavailable.append({"kind": COMPACTION_BOUNDARY, "reason": reasons[COMPACTION_BOUNDARY]})
    stamps = [row["timestamp"] for row in selected if row["timestamp"] is not None]
    return {
        "source": {
            "kind": SOURCE_RECORD,
            "path": str(path),
            "label": path.name,
            "locator": LOCATOR_RECORD_LINE,
            "requested_session": None if session_id is None else str(session_id),
            "selection": selection,
            "snapshot": {"bytes": int(info.st_size), "mtime_ns": int(info.st_mtime_ns)},
        },
        "session": {
            "id": resolved,
            "source": None,
            "started_at": min(stamps) if stamps else None,
            "ended_at": None,
            "end_reason": None,
        },
        "turns": _turns_block(turns, None if row_limit_reached or oversized_lines else user_turn),
        "rows": windowed,
        "rows_read": lines_read,
        "row_limit_reached": row_limit_reached,
        "bytes_read": bytes_read,
        "bytes_skipped": bytes_skipped,
        "oversized_refs": oversized_lines,
        "unavailable": unavailable,
    }


def _resolve_record_session(session_id: str | None, sessions: Mapping[str, float]) -> tuple[str, str]:
    wanted = str(session_id or "").strip()
    if not sessions:
        raise AgentDebugReportError("the session record has no line that names a session_id")
    if not wanted:
        if len(sessions) > 1:
            listed = ", ".join(sorted(sessions)[:_AMBIGUOUS_LISTED])
            raise AgentDebugReportError(
                f"the session record holds more than one session ({listed}); name one with --hermes-session"
            )
        return next(iter(sessions)), SELECTION_ONLY_SESSION
    if wanted == HERMES_LATEST_SESSION:
        latest = max(sessions.values())
        return _single_match(wanted, sorted(sid for sid, stamp in sessions.items() if stamp == latest)), SELECTION_LATEST
    if wanted in sessions:
        return wanted, SELECTION_EXACT
    return _single_match(wanted, sorted(sid for sid in sessions if sid.startswith(wanted))), SELECTION_UNIQUE_PREFIX


def _bounded_lines(handle: BinaryIO, max_row_bytes: int) -> Iterator[tuple[int, bytes, int]]:
    """``(line number, kept bytes, skipped bytes)`` per line; an oversized line keeps nothing.

    A line is oversized when more than ``max_row_bytes`` bytes precede its
    newline. ``readline(limit)`` never returns more than ``max_row_bytes + 1``
    bytes, so such a line is detected by its missing newline, and the rest of
    it is read in fixed chunks and dropped -- the complete line is never held
    in memory.
    """
    line_number = 0
    while True:
        raw = handle.readline(max_row_bytes + 1)
        if not raw:
            return
        line_number += 1
        if raw.endswith(b"\n") or len(raw) <= max_row_bytes:
            yield line_number, raw, 0
            continue
        skipped = len(raw)
        while not raw.endswith(b"\n"):
            raw = handle.readline(_SKIP_CHUNK_BYTES)
            if not raw:
                break
            skipped += len(raw)
        yield line_number, b"", skipped


_ABSENT = object()


def _reduced_row(
    *,
    ref: int,
    session_id: Any,
    role: str,
    call_key: str,
    tool_call_id: Any,
    tool_name: Any,
    content: str | None,
    tool_calls: Any,
    summary: bool | None,
    stamp: Any,
    oversized: bool,
) -> dict[str, Any]:
    """The typed fields the kinds read, and nothing of the row's text."""
    has_tool_calls_field = tool_calls is not _ABSENT
    calls = None if tool_calls is _ABSENT else tool_calls
    return {
        "ref": ref,
        "session_id": str(session_id) if isinstance(session_id, str) and session_id else None,
        "role": role,
        "call_key": call_key,
        "tool_call_id": str(tool_call_id) if isinstance(tool_call_id, str) and tool_call_id else None,
        "tool_name": str(tool_name or ""),
        "result": _typed_result(content) if role == "tool" else None,
        "digests": _argument_digests(calls) if role == "assistant" else {},
        "has_tool_calls_field": has_tool_calls_field,
        "summary": summary,
        "timestamp": hermes_epoch(stamp),
        "oversized": oversized,
    }


def _turn_window(user_ids: list[int], turns: tuple[int, int | None] | None) -> tuple[int | None, int | None]:
    """``[low, high)`` message ids for a 1-based inclusive turn range; ``(None, None)`` for the whole session."""
    if turns is None:
        return None, None
    start, end = turns
    if start > len(user_ids):
        raise AgentDebugReportError(f"turn {start} is past the session's last turn ({len(user_ids)})")
    high = user_ids[end] if end is not None and end < len(user_ids) else None
    return user_ids[start - 1], high


def _turns_block(turns: tuple[int, int | None] | None, turn_count: int | None) -> dict[str, Any]:
    return {
        "start": None if turns is None else turns[0],
        "end": None if turns is None else turns[1],
        "turn_count": turn_count,
    }


def _report_from_rows(read: Mapping[str, Any], *, max_rows: int, max_row_bytes: int) -> dict[str, Any]:
    resolved = read["session"]["id"]
    rows = read["rows"]
    arguments: dict[str, str] = {}
    for row in rows:
        for call_id, digest in row["digests"].items():
            arguments.setdefault(call_id, digest)
    calls = _distinct_calls(row for row in rows if row["role"] == "tool")
    findings: list[dict[str, Any]] = []
    previous_by_tool: dict[str, dict[str, Any]] = {}
    for call in calls:
        tool_call_id = call["tool_call_id"]
        if tool_call_id is None:
            continue
        result = call["result"]
        error_class, exit_code = _error_class(result)
        if error_class is not None:
            findings.append(_finding(TOOL_ERROR, resolved, [call], error_class=error_class, exit_code=exit_code))
        if _started_without_notify(result):
            findings.append(_finding(BACKGROUND_WITHOUT_NOTIFY, resolved, [call]))
        digest = arguments.get(tool_call_id)
        previous = previous_by_tool.get(call["tool_name"])
        if previous is not None and previous["error_class"] is not None and digest is not None and digest == previous["digest"]:
            findings.append(
                _finding(
                    IDENTICAL_RETRY_AFTER_ERROR,
                    resolved,
                    [previous["call"], call],
                    error_class=previous["error_class"],
                    exit_code=previous["exit_code"],
                    arguments_sha256=digest,
                )
            )
        previous_by_tool[call["tool_name"]] = {
            "call": call,
            "error_class": error_class,
            "exit_code": exit_code,
            "digest": digest,
        }
    unavailable = list(read["unavailable"])
    if not any(item["kind"] == COMPACTION_BOUNDARY for item in unavailable):
        for row in rows:
            if row["summary"]:
                findings.append(_finding(COMPACTION_BOUNDARY, resolved, [row]))
    findings.sort(key=lambda item: (item["citation"]["message_ids"][0], FINDING_KINDS.index(item["kind"])))

    oversized_refs = sorted({*(row["ref"] for row in rows if row["oversized"]), *read.get("oversized_refs", ())})
    budget = {
        "max_rows": max_rows,
        "max_row_bytes": max_row_bytes,
        "rows_read": int(read["rows_read"]),
        "bytes_read": int(read["bytes_read"]),
        "bytes_skipped": int(read["bytes_skipped"]),
        "row_limit_reached": bool(read["row_limit_reached"]),
        "oversized_rows": len(oversized_refs),
        "oversized_refs": oversized_refs[:MAX_LISTED_OVERSIZED],
        "complete": not read["row_limit_reached"] and not oversized_refs,
    }
    return {
        "schema_version": AGENT_DEBUG_REPORT_SCHEMA_VERSION,
        "source": dict(read["source"]),
        "session": dict(read["session"]),
        "turns": dict(read["turns"]),
        "budget": budget,
        "counts": {
            "tool_calls": len(calls),
            "tool_calls_without_id": sum(1 for call in calls if call["tool_call_id"] is None),
            "tool_calls_with_arguments": sum(1 for call in calls if call["tool_call_id"] in arguments),
        },
        "checked_kinds": [kind for kind in FINDING_KINDS if kind not in {item["kind"] for item in unavailable}],
        "unavailable": unavailable,
        "findings": findings,
        "finding_counts": {kind: sum(1 for item in findings if item["kind"] == kind) for kind in FINDING_KINDS},
        "observed": True,
        "claim_boundary": AGENT_DEBUG_REPORT_CLAIM_BOUNDARY,
    }


# --- validation ------------------------------------------------------------


def agent_debug_report_errors(report: Mapping[str, Any]) -> list[str]:
    """Every reason ``report`` is not a valid ``agent_debug_report/v1``; empty when it is.

    A finding must name a kind from ``FINDING_KINDS`` and carry a citation
    with exactly ``CITATION_KEYS``: the report's own session id, one message
    id and one timestamp per row the kind spans, and the fields that kind
    requires. A key outside that shape is refused, so text cannot ride along.
    A report may not call its reading complete when its budget block records
    an oversized row or a reached row limit. ``unavailable`` lists kinds with
    the reader's own reason for its source, ``checked_kinds`` is every other
    kind, no finding has an unavailable kind, ``finding_counts`` counts the
    findings by kind, and the claim boundary is the reader's own.
    """
    errors: list[str] = []
    if report.get("schema_version") != AGENT_DEBUG_REPORT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {AGENT_DEBUG_REPORT_SCHEMA_VERSION}")
    session = report.get("session")
    session_id = session.get("id") if isinstance(session, Mapping) else None
    if not isinstance(session_id, str) or not session_id:
        errors.append("session.id is missing")
    source = report.get("source")
    if isinstance(source, Mapping) and "locator" in source:
        if LOCATORS.get(str(source.get("kind"))) != source.get("locator"):
            errors.append("source.locator does not match source.kind")
    budget = report.get("budget")
    if isinstance(budget, Mapping) and "complete" in budget:
        honest = not budget.get("row_limit_reached") and not budget.get("oversized_rows")
        if budget.get("complete") is not honest:
            errors.append("budget.complete must be true exactly when no row was oversized and the row limit was not reached")
    if report.get("claim_boundary") != AGENT_DEBUG_REPORT_CLAIM_BOUNDARY:
        errors.append("claim_boundary must be the agent debug report's own")
    reasons = UNAVAILABLE_REASONS.get(str(source.get("kind")) if isinstance(source, Mapping) else "", {})
    unavailable = report.get("unavailable")
    unavailable_kinds: list[Any] = []
    if not isinstance(unavailable, list):
        errors.append("unavailable must be a list")
    else:
        for item in unavailable:
            if not isinstance(item, Mapping) or set(item) != {"kind", "reason"} or reasons.get(str(item.get("kind"))) != item.get("reason"):
                errors.append("unavailable must list a kind with the reader's own reason for this source, and nothing else")
                continue
            unavailable_kinds.append(item["kind"])
    if report.get("checked_kinds") != [kind for kind in FINDING_KINDS if kind not in unavailable_kinds]:
        errors.append("checked_kinds must be every finding kind the report does not list as unavailable")
    findings = report.get("findings")
    if not isinstance(findings, list):
        return errors + ["findings must be a list"]
    counted = {kind: sum(1 for item in findings if isinstance(item, Mapping) and item.get("kind") == kind) for kind in FINDING_KINDS}
    for kind in unavailable_kinds:
        if counted.get(kind):
            errors.append(f"{kind} is listed unavailable but has findings")
    before_findings = len(errors)
    seen_ids: set[str] = set()
    for index, finding in enumerate(findings):
        label = f"finding {index}"
        if not isinstance(finding, Mapping):
            errors.append(f"{label} is not an object")
            continue
        extra = sorted(set(finding) - _FINDING_KEYS)
        if extra:
            errors.append(f"{label} carries keys outside the finding shape: {', '.join(extra)}")
        kind = finding.get("kind")
        if kind not in _REQUIRED_CITATION:
            errors.append(f"{label} has unknown kind {kind!r}")
            continue
        citation = finding.get("citation")
        if not isinstance(citation, Mapping):
            errors.append(f"{label} ({kind}) has no citation")
            continue
        errors.extend(f"{label} ({kind}) {problem}" for problem in _citation_errors(kind, citation, session_id))
        finding_id = finding.get("finding_id")
        message_ids = citation.get("message_ids")
        first = message_ids[0] if isinstance(message_ids, list) and message_ids else None
        if finding_id != f"{kind}:{first}":
            errors.append(f"{label} ({kind}) finding_id must be {kind}:<first cited message id>")
        elif finding_id in seen_ids:
            errors.append(f"{label} ({kind}) repeats finding_id {finding_id}")
        seen_ids.add(str(finding_id))
    # Checked only over well-formed findings: a malformed one is already
    # refused above, and recounting it would name the same fault twice.
    if len(errors) == before_findings and report.get("finding_counts") != counted:
        errors.append("finding_counts must count the report's findings by kind")
    return errors


def agent_debug_reference_errors(
    report: Mapping[str, Any],
    *,
    hermes_home: str | Path | None = None,
    session_record: str | Path | None = None,
) -> list[str]:
    """Re-read every cited row from the source and name each reference that no longer holds.

    A reference is *missing* when its row is gone, *foreign* when the row
    belongs to another session, *stale* when the row's timestamp changed
    since the report was built, and *mismatched* when the row is not the tool
    call, tool, or compaction summary the finding says it is. Every reference
    is stale when the source moved under the report's snapshot: a record
    file's size or modification time changed, or the session in state.db
    gained or lost rows (its row count or highest message id changed), so a
    row added after the report cannot go unseen. The snapshot is compared
    even when nothing is cited; only the cited rows are read. An invalid
    report is refused before anything is read.
    """
    errors = agent_debug_report_errors(report)
    if errors:
        return errors
    source = report.get("source") or {}
    session_id = str((report.get("session") or {}).get("id"))
    cited = _cited_rows(report)
    if source.get("kind") == SOURCE_RECORD:
        if session_record is None:
            return ["the report was read from a session record; supply that record to check it"]
        max_row_bytes = int((report.get("budget") or {}).get("max_row_bytes") or DEFAULT_MAX_ROW_BYTES)
        current = _record_rows(Path(session_record), source, set(cited), max_row_bytes)
    elif source.get("kind") == SOURCE_STATE_DB:
        if hermes_home is None:
            return ["the report was read from a Hermes state.db; supply its Hermes home to check it"]
        read = _state_db_rows(hermes_home, set(cited), session_id, source.get("snapshot") or {})
        if isinstance(read, str):
            return [read]
        current, moved = read
        if moved:
            errors.append("every reference is stale: the session's rows in state.db changed since the report")
    else:
        return [f"source.kind {source.get('kind')!r} has no reader"]
    if isinstance(current, str):
        return [current]
    for ref, expected_rows in sorted(cited.items()):
        row = current.get(ref)
        for finding_id, expected in expected_rows:
            label = f"{finding_id} reference {ref}"
            if row is None:
                errors.append(f"{label} is missing from the source")
                continue
            if row["session_id"] != session_id:
                errors.append(f"{label} is foreign: it belongs to another session")
                continue
            if row["timestamp"] != expected["timestamp"]:
                errors.append(f"{label} is stale: the row's timestamp changed since the report")
                continue
            if expected["summary"] and not row["summary"]:
                errors.append(f"{label} is mismatched: the row is not a compaction summary")
            if expected["tool_call_id"] is not None and (
                row["role"] != "tool"
                or row["tool_call_id"] != expected["tool_call_id"]
                or row["tool_name"] != expected["tool_name"]
            ):
                errors.append(f"{label} is mismatched: the row is not the cited tool call")
    return errors


def _cited_rows(report: Mapping[str, Any]) -> dict[int, list[tuple[str, dict[str, Any]]]]:
    cited: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for finding in report.get("findings") or ():
        citation = finding["citation"]
        tool_call_ids = citation.get("tool_call_ids") or []
        for index, ref in enumerate(citation["message_ids"]):
            cited.setdefault(int(ref), []).append(
                (
                    str(finding["finding_id"]),
                    {
                        "timestamp": citation["timestamps"][index],
                        "tool_call_id": tool_call_ids[index] if index < len(tool_call_ids) else None,
                        "tool_name": citation.get("tool_name") or "",
                        "summary": finding["kind"] == COMPACTION_BOUNDARY,
                    },
                )
            )
    return cited


def _state_db_rows(
    hermes_home: str | Path, refs: set[int], session_id: str, snapshot: Mapping[str, Any]
) -> tuple[dict[int, dict[str, Any]], bool] | str:
    """The cited rows as they are now, and whether the session's row count or highest id moved."""
    try:
        path, connection = open_state_db_readonly(hermes_home, error=AgentDebugReportError)
    except AgentDebugReportError as exc:
        return str(exc)
    try:
        count, max_id = connection.execute(
            "SELECT COUNT(*), MAX(id) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
        current = {"session_rows": int(count or 0), "max_message_id": None if max_id is None else int(max_id)}
        moved = current != {"session_rows": snapshot.get("session_rows"), "max_message_id": snapshot.get("max_message_id")}
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(messages)")}
        summary_sql = _SUMMARY_COLUMN if _SUMMARY_COLUMN in columns else "NULL"
        ordered = sorted(refs)
        rows: dict[int, dict[str, Any]] = {}
        for start in range(0, len(ordered), 500):
            chunk = ordered[start : start + 500]
            marks = ",".join("?" for _ in chunk)
            for message_id, session_id, role, tool_call_id, tool_name, summary, stamp in connection.execute(
                f"SELECT id, session_id, role, tool_call_id, tool_name, {summary_sql}, timestamp "
                f"FROM messages WHERE id IN ({marks})",
                chunk,
            ):
                rows[int(message_id)] = {
                    "session_id": str(session_id),
                    "role": str(role or ""),
                    "tool_call_id": str(tool_call_id) if tool_call_id else None,
                    "tool_name": str(tool_name or ""),
                    "summary": bool(summary),
                    "timestamp": hermes_epoch(stamp),
                }
    except sqlite3.Error as exc:
        return f"could not read {path}: {exc}"
    finally:
        connection.close()
    return rows, moved


def _record_rows(
    path: Path, source: Mapping[str, Any], refs: set[int], max_row_bytes: int
) -> dict[int, dict[str, Any]] | str:
    try:
        info = path.stat()
    except OSError as exc:
        return f"no session record at {path}: {exc.strerror or exc}"
    snapshot = source.get("snapshot") or {}
    if (int(info.st_size), int(info.st_mtime_ns)) != (snapshot.get("bytes"), snapshot.get("mtime_ns")):
        return "every reference is stale: the session record's size or modification time changed since the report"
    rows: dict[int, dict[str, Any]] = {}
    if not refs:
        return rows
    last = max(refs)
    try:
        with path.open("rb") as handle:
            for line_number, raw, skipped in _bounded_lines(handle, max_row_bytes):
                if line_number > last:
                    break
                if line_number not in refs or skipped:
                    continue
                message = _json_value(raw.decode("utf-8", "replace"))
                if not isinstance(message, dict):
                    continue
                rows[line_number] = {
                    "session_id": str(message.get("session_id") or ""),
                    "role": str(message.get("role") or ""),
                    "tool_call_id": str(message["tool_call_id"]) if message.get("tool_call_id") else None,
                    "tool_name": str(message.get("tool_name") or ""),
                    "summary": bool(message.get(_SUMMARY_COLUMN)),
                    "timestamp": hermes_epoch(message.get("timestamp")),
                }
    except OSError as exc:
        return f"could not read {path}: {exc.strerror or exc}"
    return rows


# --- formatting ------------------------------------------------------------


def format_agent_debug_report(report: Mapping[str, Any]) -> str:
    """Plain-text rendering: the session, one line per finding, unavailable kinds, the budget, the boundary."""
    session = report.get("session") or {}
    source = report.get("source") or {}
    counts = report.get("counts") or {}
    findings = list(report.get("findings") or ())
    by_line = source.get("locator") == LOCATOR_RECORD_LINE
    out = [f"OMH agent debug report: session {session.get('id')} (source {session.get('source') or '(none)'})"]
    out.append(
        f"  started {_stamp(session.get('started_at'))}    ended {_stamp(session.get('ended_at'))}"
        f"    end reason {session.get('end_reason') or '(none)'}"
    )
    turns = report.get("turns") or {}
    selection = f"  selected {source.get('selection') or SELECTION_EXACT} from {source.get('label') or '(unknown)'}"
    if turns.get("start") is not None:
        end = turns.get("end")
        selection += f"    turns {turns['start']}-{end if end is not None else 'last'}"
    else:
        selection += "    turns all"
    if turns.get("turn_count") is not None:
        selection += f" of {turns['turn_count']}"
    out.append(selection)
    out.append(
        f"  tool calls {int(counts.get('tool_calls', 0))} (distinct tool_call_id)    findings {len(findings)}"
    )
    out.append("Findings")
    if not findings:
        out.append("  none of the checked kinds")
    for finding in findings:
        citation = finding.get("citation") or {}
        refs = [str(item) for item in citation.get("message_ids") or ()]
        parts = [
            str(finding.get("finding_id")),
            ("at " + ",".join(f"{source.get('label')}:{ref}" for ref in refs)) if by_line else ("messages " + ",".join(refs)),
            ("stamped " if by_line else "at ") + ",".join(_stamp(item) for item in citation.get("timestamps") or ()),
        ]
        if citation.get("tool_call_ids"):
            parts.append("calls " + ",".join(str(item) for item in citation["tool_call_ids"]))
        if citation.get("tool_name"):
            parts.append(f"tool {citation['tool_name']}")
        if citation.get("error_class"):
            parts.append(str(citation["error_class"]))
        if citation.get("exit_code") is not None:
            parts.append(f"exit {citation['exit_code']}")
        if citation.get("arguments_sha256"):
            parts.append(f"args sha256 {citation['arguments_sha256']}")
        out.append("  " + "  ".join(parts))
    unavailable = list(report.get("unavailable") or ())
    if unavailable:
        out.append("Unavailable")
        out.extend(f"  {item.get('kind')}: {item.get('reason')}" for item in unavailable)
    budget = report.get("budget") or {}
    if budget:
        line = (
            f"  rows read {budget.get('rows_read')} of at most {budget.get('max_rows')}"
            f"    bytes read {budget.get('bytes_read')} (at most {budget.get('max_row_bytes')} per row)"
        )
        if budget.get("bytes_skipped"):
            line += f"    bytes skipped unread {budget['bytes_skipped']}"
        out.append("Budget")
        out.append(line)
        if budget.get("oversized_rows"):
            out.append(
                f"  oversized, not checked: {budget['oversized_rows']} "
                f"({', '.join(str(ref) for ref in budget.get('oversized_refs') or ())})"
            )
        if budget.get("row_limit_reached"):
            out.append("  row limit reached: rows after the limit were not read")
        out.append(f"  complete {'yes' if budget.get('complete') else 'no'}")
    out.append("Boundary")
    out.append(f"  {report.get('claim_boundary', AGENT_DEBUG_REPORT_CLAIM_BOUNDARY)}")
    return "\n".join(out)


# --- derivation helpers ----------------------------------------------------


def _distinct_calls(tool_rows: Any) -> list[dict[str, Any]]:
    """One entry per distinct call key, the first row deciding what the call was."""
    calls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in tool_rows:
        if row["call_key"] in seen:
            continue
        seen.add(row["call_key"])
        calls.append(
            {
                "message_id": row["ref"],
                "timestamp": row["timestamp"],
                "tool_call_id": row["tool_call_id"],
                "tool_name": row["tool_name"],
                "result": row["result"],
            }
        )
    return calls


def _argument_digests(calls: Any) -> dict[str, str]:
    """``tool_call_id -> sha256 prefix`` of each call's canonical arguments, first call deciding."""
    if isinstance(calls, str):
        calls = _json_value(calls)
    if not isinstance(calls, list):
        return {}
    digests: dict[str, str] = {}
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or call["id"] in digests:
            continue
        function = call.get("function")
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(arguments, str):
            continue
        try:
            canonical = json.dumps(json.loads(arguments), sort_keys=True, separators=(",", ":"))
        except ValueError:
            canonical = arguments
        digests[call["id"]] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:ARGUMENT_DIGEST_CHARS]
    return digests


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    return str(value)


def _json_value(content: str | None) -> Any:
    if content is None:
        return None
    try:
        return json.loads(content)
    except ValueError:
        return None


def _typed_result(content: str | None) -> dict[str, Any] | None:
    """The typed result fields the kinds read; ``error`` is kept only as whether it held text."""
    value = _json_value(content)
    if not isinstance(value, dict):
        return None
    typed = {key: value[key] for key in _TYPED_RESULT_KEYS if key in value}
    error = value.get("error")
    typed["error_present"] = isinstance(error, str) and bool(error.strip())
    return typed


def _error_class(result: Mapping[str, Any] | None) -> tuple[str | None, int | None]:
    """The error class a result's typed fields record, and its integer exit code."""
    if result is None:
        return None, None
    exit_code = result.get("exit_code")
    exit_code = exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None
    if exit_code is not None and exit_code != 0:
        return "nonzero_exit", exit_code
    if result.get("success") is False:
        return "success_false", exit_code
    if result.get("error_present") is True:
        return "error_field", exit_code
    return None, exit_code


def _started_without_notify(result: Mapping[str, Any] | None) -> bool:
    if result is None:
        return False
    pid = result.get("pid")
    started = isinstance(pid, int) and not isinstance(pid, bool)
    return started and result.get("notify_on_complete") is not True


def _finding(
    kind: str,
    session_id: str,
    rows: list[Mapping[str, Any]],
    *,
    error_class: str | None = None,
    exit_code: int | None = None,
    arguments_sha256: str | None = None,
) -> dict[str, Any]:
    ids = [int(row["message_id"] if "message_id" in row else row["ref"]) for row in rows]
    tool_call_ids = [str(row["tool_call_id"]) for row in rows if row.get("tool_call_id")] if kind != COMPACTION_BOUNDARY else []
    return {
        "finding_id": f"{kind}:{ids[0]}",
        "kind": kind,
        "citation": {
            "session_id": session_id,
            "message_ids": ids,
            "timestamps": [row.get("timestamp") for row in rows],
            "tool_call_ids": tool_call_ids,
            "tool_name": (rows[0].get("tool_name") or None) if kind != COMPACTION_BOUNDARY else None,
            "error_class": error_class,
            "exit_code": exit_code,
            "arguments_sha256": arguments_sha256,
        },
    }


def _citation_errors(kind: str, citation: Mapping[str, Any], session_id: Any) -> list[str]:
    errors: list[str] = []
    if set(citation) != set(CITATION_KEYS):
        missing = sorted(set(CITATION_KEYS) - set(citation))
        extra = sorted(set(citation) - set(CITATION_KEYS))
        if missing:
            errors.append(f"citation is missing {', '.join(missing)}")
        if extra:
            errors.append(f"citation carries keys outside the citation shape: {', '.join(extra)}")
    if citation.get("session_id") != session_id:
        errors.append("citation names another session")
    rows, required = _REQUIRED_CITATION[kind]
    message_ids = citation.get("message_ids")
    if not (
        isinstance(message_ids, list)
        and len(message_ids) == rows
        and all(isinstance(item, int) and not isinstance(item, bool) for item in message_ids)
    ):
        errors.append(f"citation must cite {rows} message id{'s' if rows != 1 else ''}")
    timestamps = citation.get("timestamps")
    if not (
        isinstance(timestamps, list)
        and len(timestamps) == rows
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in timestamps)
    ):
        errors.append(f"citation must carry {rows} timestamp{'s' if rows != 1 else ''}")
    tool_call_ids = citation.get("tool_call_ids")
    if "tool_call_ids" in required:
        if not (
            isinstance(tool_call_ids, list)
            and len(tool_call_ids) == rows
            and all(isinstance(item, str) and item for item in tool_call_ids)
        ):
            errors.append(f"citation must cite {rows} tool_call id{'s' if rows != 1 else ''}")
    elif tool_call_ids:
        errors.append("citation cites tool_call ids this kind does not span")
    if "tool_name" in required and not (isinstance(citation.get("tool_name"), str) and citation["tool_name"]):
        errors.append("citation must name the tool")
    if "error_class" in required and citation.get("error_class") not in ERROR_CLASSES:
        errors.append(f"citation must carry an error_class from {', '.join(ERROR_CLASSES)}")
    if "arguments_sha256" in required:
        digest = citation.get("arguments_sha256")
        if not (isinstance(digest, str) and len(digest) == ARGUMENT_DIGEST_CHARS and all(c in "0123456789abcdef" for c in digest)):
            errors.append("citation must carry the arguments sha256 prefix")
    exit_code = citation.get("exit_code")
    if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
        errors.append("citation exit_code must be an integer or null")
    return errors


def _stamp(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return "(unknown)"
    return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
