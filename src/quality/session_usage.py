"""OMH utilization per Hermes host surface, read from ``state.db`` read-only.

Hermes tags a session row with the surface that opened it
(``sessions.source``: ``tui``, ``cli``, ``desktop``, ``oneshot``, ...); a row
without a tag groups as ``(none)``. It persists every tool result as a
``messages`` row with ``role = 'tool'``, ``tool_name`` and ``tool_call_id``.
OMH's plugin registers its tools as ``omh_<name>``, and every OMH skill's
``SKILL.md`` carries a ``name`` the catalog rendered, which a ``skill_view``
result carries back. Nothing observed which host surfaces those signals
reached until a person opened the database by hand; this reader groups them
per source.

The database is opened ``mode=ro``; nothing is written. A compaction
re-persists tool rows under new ids, so every count is over distinct
``tool_call_id`` per session, the first row by id deciding what a call was
(a row without one counts once through its own id). Archived and hidden
sessions are included: a session that used OMH and was archived later still
used it. An empty window is an observation, not a failure; a missing or
unreadable database is an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Mapping

from ..skills.catalog import canonical_skill_for_label
from ..skills.catalog_types import OMH_DESCRIPTION_PREFIX
from .hermes_state import (
    NO_SOURCE_LABEL,
    TOOL_CALL_KEY_SQL as _CALL_KEY,
    hermes_epoch,
    open_state_db_readonly,
    session_source_clause,
)


SESSION_USAGE_SCHEMA_VERSION = "session_usage/v1"
OMH_TOOL_NAME_PREFIX = "omh_"
SKILL_VIEW_TOOL_NAME = "skill_view"
_UNPARSED_SKILL_LABEL = "(unparsed)"
# The description prefix is never required: the name decides. It is read only so
# a prefixed body whose name the catalog does not render still counts.
_DESCRIPTION_MARKER = OMH_DESCRIPTION_PREFIX.strip()
_FRONTMATTER_NAME = re.compile(r"^name:[ \t]*[\"']?([A-Za-z0-9][A-Za-z0-9._-]*)", re.MULTILINE)
# A compaction replaces a skill_view result with `[skill_view] name=<x> (N chars) ...`.
_PLACEHOLDER_PREFIX = f"[{SKILL_VIEW_TOOL_NAME}] name="

SESSION_USAGE_CLAIM_BOUNDARY = (
    "Session usage is a read-only observation of Hermes' own session store, grouped by the "
    "host's own sessions.source tag. It counts tool-result rows and skill_view results Hermes "
    "persisted to a session; it does not show that the model consumed them, that a call "
    "succeeded, that a skill's guidance was followed, or that a reply was correct, and it is "
    "not execution, review, CI, or merge evidence."
)

COUNT_KEYS: tuple[str, ...] = (
    "sessions",
    "tool_calls",
    "omh_tool_calls",
    "sessions_with_omh_tool",
    "skill_views",
    "omh_skill_views",
    "sessions_with_omh_skill_view",
    "sessions_with_any_omh",
)

_COUNTING_RULES: dict[str, str] = {
    "tool_calls": "distinct tool_call_id per session over messages with role = 'tool', the first row "
    "by id deciding (a compaction re-persists rows; a row without a tool_call_id counts once by its own id)",
    "omh_tool": f"tool_name starts with {OMH_TOOL_NAME_PREFIX!r}",
    "omh_skill_view": "skill_view result whose name is a label the OMH catalog renders, current, historical "
    "or retired; that covers a SKILL.md load, a reference-file load, the "
    f"{_PLACEHOLDER_PREFIX!r} placeholder a compaction leaves, and a non-JSON result carrying a frontmatter "
    f"name line. Whatever its name, a result also counts when its JSON description starts with "
    f"{OMH_DESCRIPTION_PREFIX!r} or its non-JSON content contains {_DESCRIPTION_MARKER!r}",
    "window_field": "COALESCE(last_activity_at, started_at); with --since, a session without a "
    "parseable stamp is excluded",
    "archived_hidden": "included",
    "null_source": f"grouped as {NO_SOURCE_LABEL}, which --source accepts to select those sessions",
}


class SessionUsageError(ValueError):
    """The database could not be read or the window was malformed; the message says which."""


def build_session_usage(
    hermes_home: str | Path,
    *,
    since: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Count sessions, tool calls, OMH tool calls and OMH skill loads per ``sessions.source``.

    ``since`` is an ISO-8601 timestamp or epoch seconds; sessions whose last
    activity is older are left out. ``source`` keeps only sessions whose tag
    equals it (``NO_SOURCE_LABEL`` keeps the untagged ones); an unknown tag
    yields no rows, which is still an observation.
    """
    since_epoch = None
    if since is not None:
        since_epoch = hermes_epoch(since)
        if since_epoch is None:
            raise SessionUsageError(f"--since must be an ISO-8601 timestamp or epoch seconds: {since}")
    path, connection = open_state_db_readonly(hermes_home, error=SessionUsageError)
    try:
        sessions_select = "SELECT id, source, COALESCE(last_activity_at, started_at) FROM sessions"
        if source is None:
            session_rows = connection.execute(sessions_select).fetchall()
        else:
            clause, params = session_source_clause(source)
            session_rows = connection.execute(f"{sessions_select} WHERE {clause}", params).fetchall()
        tool_rows = connection.execute(
            f"SELECT session_id, tool_name, {_CALL_KEY} FROM messages WHERE role = 'tool' ORDER BY id"
        ).fetchall()
        skill_rows = connection.execute(
            f"SELECT session_id, {_CALL_KEY}, content "
            "FROM messages WHERE role = 'tool' AND tool_name = ? ORDER BY id",
            (SKILL_VIEW_TOOL_NAME,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise SessionUsageError(f"could not read {path}: {exc}") from exc
    finally:
        connection.close()

    source_of: dict[str, str] = {}
    for session_id, tag, stamp in session_rows:
        if since_epoch is not None:
            activity = hermes_epoch(stamp)
            if activity is None or activity < since_epoch:
                continue
        source_of[str(session_id)] = str(tag) if tag else NO_SOURCE_LABEL

    per_session = {session_id: _SessionCounts() for session_id in source_of}
    seen_calls: set[tuple[str, str]] = set()
    for session_id, tool_name, call_key in tool_rows:
        counts = per_session.get(str(session_id))
        if counts is None or (str(session_id), str(call_key)) in seen_calls:
            continue
        seen_calls.add((str(session_id), str(call_key)))
        counts.tool_calls += 1
        name = str(tool_name or "")
        if name.startswith(OMH_TOOL_NAME_PREFIX):
            counts.omh_tool_names[name] = counts.omh_tool_names.get(name, 0) + 1
    seen_views: set[tuple[str, str]] = set()
    for session_id, call_key, content in skill_rows:
        counts = per_session.get(str(session_id))
        if counts is None or (str(session_id), str(call_key)) in seen_views:
            continue
        seen_views.add((str(session_id), str(call_key)))
        counts.skill_views += 1
        skill_name = _omh_skill_name(str(content or ""))
        if skill_name is not None:
            counts.omh_skill_names[skill_name] = counts.omh_skill_names.get(skill_name, 0) + 1

    rows_by_source: dict[str, dict[str, Any]] = {}
    for session_id, tag in source_of.items():
        row = rows_by_source.setdefault(tag, _empty_row(tag))
        _add_session(row, per_session[session_id])
    rows = [_finish_row(rows_by_source[tag]) for tag in sorted(rows_by_source)]
    totals = {key: sum(int(row[key]) for row in rows) for key in COUNT_KEYS}
    return {
        "schema_version": SESSION_USAGE_SCHEMA_VERSION,
        "source": {
            "kind": "hermes_state_db",
            "path": str(path),
            "since": since,
            "since_epoch": since_epoch,
            "source_filter": source,
        },
        "rows": rows,
        "totals": totals,
        "session_count": totals["sessions"],
        "counting": dict(_COUNTING_RULES),
        "observed": True,
        "claim_boundary": SESSION_USAGE_CLAIM_BOUNDARY,
    }


def format_session_usage_summary(payload: Mapping[str, Any]) -> str:
    """Plain-text rendering: header, one line per source, totals, then the boundary."""
    rows = list(payload.get("rows") or ())
    session_count = int(payload.get("session_count", 0))
    out = [f"OMH session usage: {session_count} session{'s' if session_count != 1 else ''} across {len(rows)} source{'s' if len(rows) != 1 else ''}"]
    provenance = payload.get("source") or {}
    window = [f"{key}: {provenance[key]}" for key in ("since", "source_filter") if provenance.get(key) is not None]
    if window:
        out.append("  " + "    ".join(window))
    for row in rows:
        out.append(f"  {row.get('source')}: " + _counts_line(row))
    out.append("Totals")
    out.append("  " + _counts_line(payload.get("totals") or {}))
    out.append("Boundary")
    out.append(f"  {payload.get('claim_boundary', SESSION_USAGE_CLAIM_BOUNDARY)}")
    return "\n".join(out)


def _counts_line(counts: Mapping[str, Any]) -> str:
    return "  ".join(f"{key.replace('_', ' ')} {int(counts.get(key, 0))}" for key in COUNT_KEYS)


@dataclass
class _SessionCounts:
    tool_calls: int = 0
    skill_views: int = 0
    omh_tool_names: dict[str, int] = field(default_factory=dict)
    omh_skill_names: dict[str, int] = field(default_factory=dict)


def _empty_row(tag: str) -> dict[str, Any]:
    row: dict[str, Any] = {"source": tag}
    row.update({key: 0 for key in COUNT_KEYS})
    row["omh_tool_names"] = {}
    row["omh_skill_names"] = {}
    return row


def _add_session(row: dict[str, Any], counts: _SessionCounts) -> None:
    omh_tool_calls = sum(counts.omh_tool_names.values())
    omh_skill_views = sum(counts.omh_skill_names.values())
    row["sessions"] += 1
    row["tool_calls"] += counts.tool_calls
    row["omh_tool_calls"] += omh_tool_calls
    row["sessions_with_omh_tool"] += 1 if omh_tool_calls else 0
    row["skill_views"] += counts.skill_views
    row["omh_skill_views"] += omh_skill_views
    row["sessions_with_omh_skill_view"] += 1 if omh_skill_views else 0
    row["sessions_with_any_omh"] += 1 if (omh_tool_calls or omh_skill_views) else 0
    for name, count in counts.omh_tool_names.items():
        row["omh_tool_names"][name] = row["omh_tool_names"].get(name, 0) + count
    for name, count in counts.omh_skill_names.items():
        row["omh_skill_names"][name] = row["omh_skill_names"].get(name, 0) + count


def _finish_row(row: dict[str, Any]) -> dict[str, Any]:
    row["omh_tool_names"] = dict(sorted(row["omh_tool_names"].items()))
    row["omh_skill_names"] = dict(sorted(row["omh_skill_names"].items()))
    return row


def _omh_skill_name(content: str) -> str | None:
    """The skill name when a ``skill_view`` result is an OMH skill, else ``None``.

    The name decides it: a label the catalog renders, under any era, is an
    OMH skill (``canonical_skill_for_label``), which holds for the ``ulw-*``
    labels and for retired skills alike. That one rule reads every shape a
    result takes: a SKILL.md load (a JSON object with ``description``), a
    reference-file load (``file`` present, no ``description``), the
    ``[skill_view] name=<x> ...`` placeholder a compaction leaves, and
    non-JSON content that still carries the frontmatter ``name:`` line, so a
    host that changes the result shape still counts.

    The description is consulted only when the name is not a catalog label:
    a body that starts with the ``[omh] `` prefix counts under whatever name
    it carries (a skill imported from a local source, or one the catalog no
    longer names). Non-JSON content with no catalog name falls back to that
    same marker.
    """
    try:
        result = json.loads(content)
    except ValueError:
        result = None
    if isinstance(result, dict):
        description = result.get("description")
        name = result.get("name")
        if isinstance(name, str) and canonical_skill_for_label(name):
            return name
        if isinstance(description, str) and description.startswith(OMH_DESCRIPTION_PREFIX):
            return str(name) if isinstance(name, str) and name.strip() else _UNPARSED_SKILL_LABEL
        return None
    if content.startswith(_PLACEHOLDER_PREFIX):
        words = content[len(_PLACEHOLDER_PREFIX):].split(None, 1)
        name = words[0] if words else ""
        return name if canonical_skill_for_label(name) else None
    named = _FRONTMATTER_NAME.search(content)
    if named is not None and canonical_skill_for_label(named.group(1)):
        return named.group(1)
    return _UNPARSED_SKILL_LABEL if _DESCRIPTION_MARKER in content else None
