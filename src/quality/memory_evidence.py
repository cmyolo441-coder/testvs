"""What OMH memory did on this machine, counted from structured records only.

The only memory quality surface was ``omh memory recall-suite``, which runs
synthetic fixtures. Nothing answered "is OMH memory working here, and what
has it done for the user?" from real traffic. This reader counts, over one
window, what four local records already hold:

* Hermes' ``state.db`` (opened ``mode=ro`` through ``open_state_db_readonly``,
  the session-usage open): ``omh_memory`` calls per ``action`` and per result
  ``status``, Hermes-native ``memory`` calls per action and outcome, and the
  sessions that had an OMH memory write.
* The OMH memory store: records per ``source`` and admission state, records
  created inside the window, the last prefetch receipt (the
  ``omh memory status`` view), consolidation briefs per reason family, and
  the native write journal including ``removed_entry_demoted``.
* Hermes' native memory files: headroom per file, at the configured cap.

Every count comes from a field -- a tool name, a call's argument key, a key
of the result JSON, a store file's field -- never from message text. An
action the model wrote maps onto the tool's closed action set and anything
else counts as ``(other)``; a label OMH code wrote (a status, a source)
passes only as a short identifier token. No free text a message or record
carries can reach the payload.

A read that fails is reported as such and is not a zero: the payload names
the source it could not read and ``ok`` turns false. An empty but readable
store is an observation of zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterable, Mapping

from ..paths import OmhPaths
from ..plugin_bundle.omh.hermes_memory import read_hermes_memory
from ..plugin_bundle.omh.memory_provider import JOURNAL_LIMIT
from ..plugin_bundle.omh.memory_session_digest import is_freeing_remove
from ..plugin_bundle.omh.memory_store_io import scan_project_memory_records
from ..plugin_bundle.omh.tools.memory_tool import MEMORY_ACTIONS
from ..workflows.memory import last_prefetch_status
from .hermes_state import NO_SOURCE_LABEL, TOOL_CALL_KEY_SQL as _CALL_KEY, hermes_epoch, open_state_db_readonly


MEMORY_EVIDENCE_SCHEMA_VERSION = "memory_evidence/v1"
DEFAULT_WINDOW_DAYS = 7
OMH_MEMORY_TOOL = "omh_memory"
NATIVE_MEMORY_TOOL = "memory"
# Hermes' generic wrapper: the assistant entry names `tool_call` and carries
# the real call as `{name, arguments}` or `{calls: [{name, arguments}]}`.
# Hermes refuses a `calls` list of more than one entry for local tools, so
# only a one-entry list is a call that ran; a wrapper it refused leaves a tool
# row named `tool_call` carrying `error`, and that call never ran either.
TOOL_CALL_WRAPPER = "tool_call"
NATIVE_MEMORY_ACTIONS: frozenset[str] = frozenset({"add", "replace", "remove", "batch"})
# The (action, status) pairs of an omh_memory result that wrote OMH's store:
# `remembered` admitted a record, `pending_review` staged a candidate (from a
# capture or a demote), and `demoted` stored a Hermes entry's copy
# (`memory_tool._map_capture_payload`, `memory_tool._demote_entry`).
# `corrected` admitted the corrected record and `pending_review` staged it
# (`memory_tool._correct`); `retired` moved a record to the archive
# (`memory_tool._retire`). `already_remembered`/`already_covered` wrote
# nothing. `confirm/confirmed` is left out: it renews a review date and the
# usage counter at most, and a repeat in one session writes nothing, so the
# status cannot say whether anything was written.
OMH_MEMORY_WRITE_RESULTS: frozenset[tuple[str, str]] = frozenset(
    {
        ("capture", "remembered"),
        ("capture", "pending_review"),
        ("demote", "demoted"),
        ("demote", "pending_review"),
        ("correct", "corrected"),
        ("correct", "pending_review"),
        ("retire", "retired"),
    }
)
# Journal files the memory provider appends (`MemoryProvider._write_handoff`,
# `MemoryProvider._append_write_journal`).
CONSOLIDATION_JOURNAL = "consolidation.jsonl"
WRITE_JOURNAL = "write_journal.jsonl"

NO_VALUE_LABEL = "(none)"
OTHER_LABEL = "(other)"
NO_RESULT_LABEL = "(no_result)"
UNPARSED_LABEL = "(unparsed)"
_SAFE_LABEL = re.compile(r"[a-z0-9][a-z0-9_.-]{0,47}")

READ = "read"
PARTIAL = "partial"
UNREADABLE = "unreadable"
MISSING = "missing"

MEMORY_EVIDENCE_CLAIM_BOUNDARY = (
    "Memory evidence counts local records: tool calls and results Hermes persisted to its session store, "
    "OMH memory store files, and Hermes memory file sizes, for one Hermes home and its OMH home. It shows "
    "what was written, served, and freed on this machine; it does not show that a recalled memory was "
    "read, that it improved an answer, or that a fact is correct, and it is not execution, review, CI, "
    "or merge evidence."
)

_COUNTING_RULES: dict[str, str] = {
    "tool_calls": "distinct call id per session, first row by id deciding on each side: the assistant row's "
    "tool_calls entry and the tool row with the same tool_call_id are one call (a compaction re-persists both)",
    "tool_call_wrapper": f"a {TOOL_CALL_WRAPPER!r} entry is unwrapped to its {{name, arguments}} or one-entry "
    "calls list; a calls list of more than one entry is one Hermes refuses for local tools and is not a call, "
    f"and neither is a wrapped call whose tool row is named {TOOL_CALL_WRAPPER!r} and carries an error",
    "window": "messages.timestamp for calls; created_at for records; raised_at for consolidation briefs; "
    "observed_at for write-journal lines. A stamp that does not parse is outside the window",
    "omh_memory_action": "the result JSON's action field, else the call arguments' action key, mapped onto "
    f"the tool's actions ({', '.join(MEMORY_ACTIONS)}); anything else is {OTHER_LABEL}",
    "omh_memory_status": f"the result JSON's status field; {NO_RESULT_LABEL} when no tool row was persisted, "
    f"{UNPARSED_LABEL} when the tool row is not a JSON object",
    "native_memory_action": "the call arguments' action key, or each operations[].action, mapped onto "
    f"{', '.join(sorted(NATIVE_MEMORY_ACTIONS))}; {UNPARSED_LABEL} when the arguments were not persisted or "
    "do not parse",
    "native_memory_outcome": "the result JSON's success boolean",
    "omh_memory_write": "an omh_memory result with (action, status) in "
    + ", ".join(f"{action}/{status}" for action, status in sorted(OMH_MEMORY_WRITE_RESULTS)),
    "records": "admitted record files only; a candidate waiting for review is not a record and is not counted",
    "labels": f"a status, source, state or reason label passes when it is a short lowercase identifier, else {OTHER_LABEL}",
    "reason_family": "a consolidation reason up to its first ':', counted once per brief",
    "journal_lines": "a line that is not a JSON object is counted as unparsed and any unparsed line marks the "
    "store partial: the provider rewrites a journal whole and atomically, so a torn line is damage, not a race",
    "journal_cap": f"the provider keeps the newest {JOURNAL_LIMIT} write-journal lines; journal_at_cap true means "
    "older lines are gone and the window's write-journal counts may be short",
    "directories": "the memory, records and Hermes memories directories are listed, not stat-ed: a directory "
    "that cannot be listed is unreadable, and only one that does not exist is empty",
    "null_source": f"a session without a sessions.source tag groups as {NO_SOURCE_LABEL}",
}


class MemoryEvidenceError(ValueError):
    """The window was malformed; the message says which."""


def build_memory_evidence(
    paths: OmhPaths,
    *,
    days: int | None = DEFAULT_WINDOW_DAYS,
    since: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Count what OMH memory did since ``since`` (ISO-8601 or epoch), else over the last ``days`` days."""
    moment = time.time() if now is None else float(now)
    if since is not None:
        since_epoch = hermes_epoch(since)
        if since_epoch is None:
            raise MemoryEvidenceError(f"--since must be an ISO-8601 timestamp or epoch seconds: {since}")
        days = None
    else:
        if isinstance(days, bool) or not isinstance(days, int) or days < 1:
            raise MemoryEvidenceError(f"--days must be a positive whole number of days: {days}")
        since_epoch = moment - days * 86400
    window = {
        "days": days,
        "since": _iso(since_epoch),
        "until": _iso(moment),
        "since_epoch": since_epoch,
    }
    state_db = _state_db_counts(paths.hermes_home, since_epoch)
    store = _store_counts(paths, since_epoch, moment)
    native = _native_memory_files(paths.hermes_home, moment)
    unread = [
        {"source": name, "read_state": section["read_state"], "error": section.get("error", "")}
        for name, section in (("state_db", state_db), ("omh_store", store), ("hermes_memory_files", native))
        if section["read_state"] != READ
    ]
    payload: dict[str, Any] = {
        "schema_version": MEMORY_EVIDENCE_SCHEMA_VERSION,
        "homes": {"hermes_home": str(paths.hermes_home), "omh_home": str(paths.omh_home)},
        "window": window,
        "state_db": state_db,
        "omh_store": store,
        "hermes_memory_files": native,
        "unread_sources": unread,
        "ok": not unread,
        "counting": dict(_COUNTING_RULES),
        "redaction_policy": "metadata_only",
        "claim_boundary": MEMORY_EVIDENCE_CLAIM_BOUNDARY,
    }
    payload["headline"] = memory_evidence_headline(payload)
    return payload


def memory_evidence_headline(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The headline claims, each computed from a count in ``payload`` and naming it.

    Nothing here is stored beside the counts: change a count and the claim
    changes with it. A claim drawn from a source that was not read says so
    instead of reporting zero. Each claim's scope names the home its count
    was read from.
    """
    window = payload.get("window") or {}
    homes = payload.get("homes") or {}
    span = f"last {window.get('days')} days" if window.get("days") else f"since {window.get('since')}"
    hermes_scope = f"{span}, Hermes home {homes.get('hermes_home')}"
    omh_scope = f"{span}, OMH home {homes.get('omh_home')}"
    state_db = payload.get("state_db") or {}
    store = payload.get("omh_store") or {}
    omh_statuses = ((state_db.get("omh_memory") or {}).get("by_action_status")) or {}
    claims = (
        (
            "facts remembered",
            "omh_memory capture calls whose result status is remembered",
            "state_db.omh_memory.by_action_status.capture.remembered",
            state_db,
            hermes_scope,
            int((omh_statuses.get("capture") or {}).get("remembered", 0)),
        ),
        (
            "Hermes entries OMH stored a verified copy of",
            "omh_memory demote calls whose result status is demoted: OMH holds an intact copy of the entry. "
            "No removal from Hermes memory is implied; Hermes is unchanged until its own memory tool removes "
            "the entry, which the freed line counts",
            "state_db.omh_memory.by_action_status.demote.demoted",
            state_db,
            hermes_scope,
            int((omh_statuses.get("demote") or {}).get("demoted", 0)),
        ),
        (
            "native entries freed after a demotion",
            "native memory removes whose removed entry OMH holds intact (removed_entry_demoted true), "
            "from the OMH home's write journal",
            "omh_store.write_journal.removed_entry_demoted.true",
            store,
            # At its cap the journal has dropped older lines: the count is a floor.
            f"{omh_scope}; the write journal is at its cap, so earlier removes may be missing"
            if (store.get("write_journal") or {}).get("journal_at_cap")
            else omh_scope,
            int(((store.get("write_journal") or {}).get("removed_entry_demoted") or {}).get("true", 0)),
        ),
    )
    headline: list[dict[str, Any]] = []
    for label, meaning, derived_from, section, scope, count in claims:
        observed = section.get("read_state") == READ
        text = f"{count} {label} ({scope})" if observed else f"{label}: not observed, the source was not fully read ({scope})"
        headline.append(
            {
                "claim": text,
                "count": count if observed else None,
                "observed": observed,
                "meaning": meaning,
                "derived_from": derived_from,
            }
        )
    return headline


def format_memory_evidence_summary(payload: Mapping[str, Any]) -> str:
    """Plain-text rendering: homes and window, headline, the four sources, then the boundary."""
    window = payload.get("window") or {}
    homes = payload.get("homes") or {}
    span = f"last {window.get('days')} days" if window.get("days") else f"since {window.get('since')}"
    out = [
        f"OMH memory evidence: {span} ({window.get('since')} to {window.get('until')})",
        f"  hermes home: {homes.get('hermes_home')}",
        f"  omh home: {homes.get('omh_home')}",
        "Headline",
    ]
    out.extend(f"  {row['claim']}" for row in payload.get("headline") or ())
    state_db = payload.get("state_db") or {}
    out.append(f"Hermes session store ({state_db.get('read_state')})")
    if state_db.get("read_state") == READ:
        omh = state_db.get("omh_memory") or {}
        native = state_db.get("native_memory") or {}
        writes = state_db.get("sessions_with_omh_memory_write") or {}
        out.append(f"  omh_memory calls {omh.get('calls', 0)}: {_pairs(omh.get('by_action_status') or {})}")
        out.append(
            f"  native memory calls {native.get('calls', 0)}: operations {_flat(native.get('operations_by_action') or {})}"
            f"; outcomes {_flat(native.get('by_outcome') or {})}"
        )
        out.append(f"  sessions with an OMH memory write {writes.get('total', 0)}: {_flat(writes.get('by_source') or {})}")
    else:
        out.append(f"  {state_db.get('error', '')}")
    store = payload.get("omh_store") or {}
    out.append(f"OMH store ({store.get('read_state')})")
    if store.get("read_state") != UNREADABLE:
        records = store.get("records") or {}
        prefetch = store.get("last_prefetch") or {}
        consolidation = store.get("consolidation_journal") or {}
        journal = store.get("write_journal") or {}
        out.append(
            f"  records {records.get('total', 0)} (created in window {records.get('created_in_window', 0)}): "
            f"by source {_flat(records.get('by_source') or {})}; by admission state {_flat(records.get('by_admission_state') or {})}"
        )
        if prefetch.get("served_at"):
            out.append(f"  last prefetch: {prefetch.get('state')} at {prefetch.get('served_at')}")
        else:
            out.append("  last prefetch: never served" if prefetch.get("state") == "never_served" else f"  last prefetch: {prefetch.get('state')}")
        out.append(
            f"  consolidation briefs in window {consolidation.get('briefs_in_window', 0)}: "
            f"{_flat(consolidation.get('by_reason_family') or {})}"
        )
        out.append(
            f"  native writes in window {journal.get('entries_in_window', 0)}: {_flat(journal.get('by_action') or {})}; "
            f"removed_entry_demoted {_flat(journal.get('removed_entry_demoted') or {})}"
            + (f" (journal at its cap of {journal.get('journal_limit')} lines; older lines are gone)" if journal.get("journal_at_cap") else "")
        )
    if store.get("error"):
        out.append(f"  {store.get('error')}")
    native_files = payload.get("hermes_memory_files") or {}
    out.append(f"Hermes memory files ({native_files.get('read_state')})")
    for row in native_files.get("files") or ():
        out.append(
            f"  {row.get('label')}: {row.get('chars')}/{row.get('cap')} chars ({row.get('cap_source')} cap), "
            f"headroom {row.get('headroom_chars')}, entries {row.get('entry_count')}"
        )
    if native_files.get("error"):
        out.append(f"  {native_files.get('error')}")
    out.append("Boundary")
    out.append(f"  {payload.get('claim_boundary', MEMORY_EVIDENCE_CLAIM_BOUNDARY)}")
    return "\n".join(out)


@dataclass
class _Call:
    session_id: str
    tool: str
    arguments: Any = None
    has_arguments: bool = False
    result: Any = None
    has_result: bool = False
    result_parsed: bool = False


@dataclass
class _Tally:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, key: str, amount: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + amount

    def sorted(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


def _state_db_counts(hermes_home: Path, since_epoch: float) -> dict[str, Any]:
    """Tool-call counts from ``state.db``; a missing or unreadable database is named, not zeroed."""
    try:
        path, connection = open_state_db_readonly(hermes_home, error=MemoryEvidenceError)
    except MemoryEvidenceError as exc:
        missing = _probe(Path(hermes_home).expanduser() / "state.db") == MISSING
        return {"read_state": MISSING if missing else UNREADABLE, "error": str(exc)}
    try:
        source_rows = connection.execute("SELECT id, source FROM sessions").fetchall()
        call_rows = connection.execute(
            "SELECT session_id, tool_calls FROM messages WHERE role = 'assistant' AND tool_calls IS NOT NULL "
            "AND (instr(tool_calls, ?) > 0 OR instr(tool_calls, ?) > 0) AND timestamp >= ? ORDER BY id",
            (NATIVE_MEMORY_TOOL, TOOL_CALL_WRAPPER, since_epoch),
        ).fetchall()
        result_rows = connection.execute(
            f"SELECT session_id, tool_name, {_CALL_KEY}, content FROM messages "
            "WHERE role = 'tool' AND tool_name IN (?, ?, ?) AND timestamp >= ? ORDER BY id",
            (OMH_MEMORY_TOOL, NATIVE_MEMORY_TOOL, TOOL_CALL_WRAPPER, since_epoch),
        ).fetchall()
    except sqlite3.Error as exc:
        return {"read_state": UNREADABLE, "path": str(path), "error": f"could not read {path}: {type(exc).__name__}"}
    finally:
        connection.close()

    calls: dict[tuple[str, str], _Call] = {}
    wrapped: set[tuple[str, str]] = set()
    for session_id, tool_calls in call_rows:
        for call_id, name, arguments, parsed, via_wrapper in _memory_tool_calls(tool_calls):
            key = (str(session_id), call_id)
            if key not in calls:
                calls[key] = _Call(str(session_id), name, arguments=arguments, has_arguments=parsed)
                if via_wrapper:
                    wrapped.add(key)
    seen_results: set[tuple[str, str]] = set()
    refused: set[tuple[str, str]] = set()
    for session_id, tool_name, call_key, content in result_rows:
        key = (str(session_id), str(call_key))
        if tool_name == TOOL_CALL_WRAPPER and key not in wrapped:
            continue
        if key in seen_results:
            continue
        seen_results.add(key)
        result = _json_value(content) if isinstance(content, str) else None
        if tool_name == TOOL_CALL_WRAPPER and isinstance(result, dict) and "error" in result:
            refused.add(key)
            continue
        call = calls.setdefault(key, _Call(str(session_id), str(tool_name)))
        call.has_result = True
        call.result = result if isinstance(result, dict) else None
        call.result_parsed = isinstance(result, dict)
    for key in refused:
        calls.pop(key, None)

    source_of = {str(session_id): (str(tag) if tag else NO_SOURCE_LABEL) for session_id, tag in source_rows}
    omh_by_action_status: dict[str, _Tally] = {}
    omh_calls = 0
    native_calls = 0
    native_operations = _Tally()
    native_outcomes = _Tally()
    write_sessions: set[str] = set()
    for call in calls.values():
        if call.tool == OMH_MEMORY_TOOL:
            omh_calls += 1
            action = _omh_action(call)
            status = _omh_status(call)
            omh_by_action_status.setdefault(action, _Tally()).add(status)
            if (action, status) in OMH_MEMORY_WRITE_RESULTS:
                write_sessions.add(call.session_id)
        elif call.tool == NATIVE_MEMORY_TOOL:
            native_calls += 1
            for action in _native_actions(call):
                native_operations.add(action)
            native_outcomes.add(_native_outcome(call))
    by_source = _Tally()
    for session_id in write_sessions:
        by_source.add(source_of.get(session_id, NO_SOURCE_LABEL))
    return {
        "read_state": READ,
        "path": str(path),
        "omh_memory": {
            "calls": omh_calls,
            "by_action_status": {action: omh_by_action_status[action].sorted() for action in sorted(omh_by_action_status)},
        },
        "native_memory": {
            "calls": native_calls,
            "operations_by_action": native_operations.sorted(),
            "by_outcome": native_outcomes.sorted(),
        },
        "sessions_with_omh_memory_write": {"total": len(write_sessions), "by_source": by_source.sorted()},
    }


def _memory_tool_calls(tool_calls: Any) -> Iterable[tuple[str, str, Any, bool, bool]]:
    """``(call id, tool name, arguments, arguments parsed, via wrapper)`` for each memory call in one assistant row."""
    entries = _json_value(tool_calls)
    if not isinstance(entries, list):
        return
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") if isinstance(entry.get("function"), dict) else {}
        name = function.get("name")
        call_id = entry.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        arguments = _json_value(function.get("arguments"))
        via_wrapper = name == TOOL_CALL_WRAPPER
        if via_wrapper:
            name, arguments = _unwrapped(arguments)
        if name not in (OMH_MEMORY_TOOL, NATIVE_MEMORY_TOOL):
            continue
        yield call_id, str(name), arguments, isinstance(arguments, dict), via_wrapper


def _unwrapped(arguments: Any) -> tuple[Any, Any]:
    """The ``(name, arguments)`` a ``tool_call`` wrapper carries, or ``(None, None)``."""
    if not isinstance(arguments, dict):
        return None, None
    inner = arguments
    if "calls" in arguments:
        calls = arguments.get("calls")
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
            return None, None
        inner = calls[0]
    return inner.get("name"), _json_value(inner.get("arguments"))


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


def _omh_action(call: _Call) -> str:
    if call.result_parsed and "action" in call.result:
        return _closed_label(call.result.get("action"), MEMORY_ACTIONS)
    if call.has_arguments and "action" in call.arguments:
        return _closed_label(call.arguments.get("action"), MEMORY_ACTIONS)
    return UNPARSED_LABEL


def _omh_status(call: _Call) -> str:
    if not call.has_result:
        return NO_RESULT_LABEL
    if not call.result_parsed:
        return UNPARSED_LABEL
    return _label(call.result.get("status"))


def _native_actions(call: _Call) -> list[str]:
    if not call.has_arguments:
        return [UNPARSED_LABEL]
    operations = call.arguments.get("operations")
    if isinstance(operations, list):
        return [
            _closed_label(op.get("action"), NATIVE_MEMORY_ACTIONS) if isinstance(op, dict) else OTHER_LABEL
            for op in operations
        ] or [NO_VALUE_LABEL]
    return [_closed_label(call.arguments.get("action"), NATIVE_MEMORY_ACTIONS)]


def _native_outcome(call: _Call) -> str:
    if not call.has_result:
        return NO_RESULT_LABEL
    if not call.result_parsed:
        return UNPARSED_LABEL
    success = call.result.get("success")
    if success is True:
        return "succeeded"
    if success is False:
        return "failed"
    return NO_VALUE_LABEL


def _store_counts(paths: OmhPaths, since_epoch: float, moment: float) -> dict[str, Any]:
    """Store counts; a store that does not exist yet is an observation of zero.

    Each directory is probed by listing it before any reader touches it.
    The readers test existence with ``Path.exists``/``is_file``, which on
    Python 3.14 answer False for a path they may not enter and on 3.11
    raise, and ``stat`` succeeds on a directory nobody may list; probing
    first makes every version take the same branch. A memory directory that
    cannot be listed is ``unreadable`` -- nothing under it is read; a records
    directory alone that cannot be listed leaves the journals readable and
    the store ``partial``.
    """
    omh_home = Path(paths.omh_home).expanduser()
    memory_dir = omh_home / "memory"
    if _probe_directory(memory_dir) == UNREADABLE:
        return {"read_state": UNREADABLE, "memory_dir": str(memory_dir), "error": f"could not list {memory_dir.name}"}
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    refused: list[dict[str, str]] = []
    if _probe_directory(memory_dir / "records") == UNREADABLE:
        errors.append("could not list records")
    else:
        try:
            records, refused = scan_project_memory_records(omh_home)
        except OSError as exc:
            errors.append(f"could not read records: {type(exc).__name__}")
    by_source = _Tally()
    by_state = _Tally()
    created_in_window = 0
    for record in records:
        by_source.add(_label(record.get("source")))
        admission = record.get("admission") if isinstance(record.get("admission"), dict) else {}
        by_state.add(_label(admission.get("state") or record.get("review_status")))
        if _in_window(record.get("created_at"), since_epoch):
            created_in_window += 1
    refused_reasons = _Tally()
    for row in refused:
        refused_reasons.add(_label(row.get("reason")))
    consolidation_entries, consolidation_bad, consolidation_damaged = _journal(memory_dir / CONSOLIDATION_JOURNAL, errors)
    write_entries, write_bad, write_damaged = _journal(memory_dir / WRITE_JOURNAL, errors)
    prefetch = last_prefetch_status(paths, now=datetime.fromtimestamp(moment, tz=timezone.utc))
    if prefetch.get("state") == "unreadable":
        errors.append("could not read the prefetch receipt")

    briefs = 0
    families = _Tally()
    for entry in consolidation_entries:
        if not _in_window(entry.get("raised_at"), since_epoch):
            continue
        briefs += 1
        reasons = entry.get("reasons") if isinstance(entry.get("reasons"), list) else []
        for family in sorted({str(reason).split(":", 1)[0] for reason in reasons if isinstance(reason, str)}):
            families.add(_label(family))
    writes = 0
    by_action = _Tally()
    demoted = {"true": 0, "false": 0, "absent": 0}
    for entry in write_entries:
        if not _in_window(entry.get("observed_at"), since_epoch):
            continue
        writes += 1
        action = _label(entry.get("action"))
        by_action.add(action)
        if action == "remove":
            # The digest's own rule, imported: one definition of a freeing remove.
            flag = entry.get("removed_entry_demoted")
            demoted["true" if is_freeing_remove(entry) else "false" if flag is False else "absent"] += 1
    if refused_reasons.counts.get("unreadable_file", 0):
        errors.append(f"{refused_reasons.counts['unreadable_file']} record files did not parse")
    for name, damaged in ((CONSOLIDATION_JOURNAL, consolidation_damaged), (WRITE_JOURNAL, write_damaged)):
        if damaged:
            errors.append(f"{name} has unparsed lines")
    section: dict[str, Any] = {
        "read_state": PARTIAL if errors else READ,
        "memory_dir": str(memory_dir),
        "records": {
            "total": len(records),
            "by_source": by_source.sorted(),
            "by_admission_state": by_state.sorted(),
            "created_in_window": created_in_window,
            "refused_files_by_reason": refused_reasons.sorted(),
        },
        "last_prefetch": prefetch,
        "consolidation_journal": {
            "briefs_in_window": briefs,
            "by_reason_family": families.sorted(),
            "unparsed_lines": consolidation_bad,
        },
        "write_journal": {
            "entries_in_window": writes,
            "by_action": by_action.sorted(),
            "removed_entry_demoted": demoted,
            "unparsed_lines": write_bad,
            # The provider keeps only the newest JOURNAL_LIMIT lines: at the cap,
            # older lines -- possibly inside the window -- are already gone.
            "journal_at_cap": len(write_entries) + write_bad >= JOURNAL_LIMIT,
            "journal_limit": JOURNAL_LIMIT,
        },
    }
    if errors:
        section["error"] = "; ".join(errors)
    return section


def _journal(path: Path, errors: list[str]) -> tuple[list[dict[str, Any]], int, bool]:
    """Parsed JSON-object lines, the unparsed count, and whether any line failed to parse.

    The provider rewrites a journal whole, atomically and under its state
    lock (``_append_bounded_json_line``), so a reader never meets a half
    written line: an unparsed line is damage, and the next provider write
    drops it. An absent journal is empty; one that exists and cannot be read
    is an error.
    """
    state = _probe(path)
    if state == MISSING:
        return [], 0, False
    if state == UNREADABLE:
        errors.append(f"could not read {path.name}")
        return [], 0, False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"could not read {path.name}: {type(exc).__name__}")
        return [], 0, False
    entries: list[dict[str, Any]] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        entry = _json_value(line)
        if isinstance(entry, dict):
            entries.append(entry)
        else:
            bad += 1
    return entries, bad, bad > 0


def _native_memory_files(hermes_home: Path, moment: float) -> dict[str, Any]:
    """Headroom per Hermes memory file through the bundle reader ``omh memory status`` uses.

    The reader tests existence with ``Path.exists``, which on Python 3.14
    reads a refused path as absent and on 3.11 raises, so the directory is
    probed by listing it before the reader runs, each file with ``stat``
    after, and any ``OSError`` the reader still raises is an unread source.
    """
    memories_dir = Path(hermes_home).expanduser() / "memories"
    if _probe_directory(memories_dir) == UNREADABLE:
        return {"read_state": UNREADABLE, "files": [], "error": f"could not list {memories_dir.name}"}
    try:
        files = [reading.to_dict() for reading in read_hermes_memory(hermes_home, now=moment)]
    except OSError as exc:
        return {"read_state": UNREADABLE, "files": [], "error": f"could not read {memories_dir.name}: {type(exc).__name__}"}
    errors = [str(row["label"]) for row in files if row.get("error")]
    errors.extend(str(row["label"]) for row in files if not row.get("error") and _probe(Path(str(row["path"]))) == UNREADABLE)
    for row in files:
        if row.get("error"):
            row["error"] = "unreadable"
    section: dict[str, Any] = {"read_state": UNREADABLE if errors else READ, "files": files}
    if errors:
        section["error"] = "could not read " + ", ".join(errors)
    return section


def _probe(path: Path) -> str:
    """``read`` when ``stat`` succeeds, ``missing`` only on FileNotFoundError, else ``unreadable``."""
    try:
        path.stat()
    except FileNotFoundError:
        return MISSING
    except OSError:
        return UNREADABLE
    return READ


def _probe_directory(path: Path) -> str:
    """``read`` when the directory lists, ``missing`` only when it does not exist, else ``unreadable``.

    ``stat`` is not enough here: it succeeds on a directory with mode 0.
    """
    try:
        with os.scandir(path):
            pass
    except FileNotFoundError:
        return MISSING
    except OSError:
        return UNREADABLE
    return READ


def _label(value: Any) -> str:
    if value is None or value == "":
        return NO_VALUE_LABEL
    text = str(value).strip().lower()
    return text if _SAFE_LABEL.fullmatch(text) else OTHER_LABEL


def _closed_label(value: Any, vocabulary: Iterable[str]) -> str:
    """A model-written label kept only when it is one of the tool's own values."""
    if value is None or value == "":
        return NO_VALUE_LABEL
    text = str(value).strip().lower()
    return text if text in vocabulary else OTHER_LABEL


def _in_window(stamp: Any, since_epoch: float) -> bool:
    epoch = hermes_epoch(stamp)
    return epoch is not None and epoch >= since_epoch


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flat(counts: Mapping[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items()) or "none"


def _pairs(by_action_status: Mapping[str, Mapping[str, Any]]) -> str:
    return "; ".join(f"{action} ({_flat(statuses)})" for action, statuses in by_action_status.items()) or "none"
