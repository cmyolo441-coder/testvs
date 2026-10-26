"""Live coding status board projection and renderer (`omh_coding_status_board/v1`).

Answers one question — "what coding work is running right now?" — by merging
three local observation sources into one row per unit and rendering them as an
aligned, dialect-neutral board. Every row carries a label, the runtime that
owns it, the model that runtime was told to use, a status, an elapsed time, and
a token total. Below the units, the same board lists the work Hermes runs on
its own -- `delegate_task` children and kanban board tasks -- through the two
readers `omh_hud` uses, each row with its source and state, and says when
either Hermes database could not be read.

Boundaries, in order of importance:

- Honesty over completeness. A unit's runtime and model are always exact when
  present because omh itself selected them and put them on the command line; a
  Hermes worker's model is the one Hermes recorded. Tokens,
  session refs, and the elapsed time of an unfinished unit are observed or the
  literal `unknown` — never estimated, never derived from a conversation token
  budget (that budget belongs to a different actor and using it here would be a
  category error). A number on the board is a number an executor reported.
- Reporting only. Nothing here is persisted, dispatched, or fed back into a
  route; `build_status_board` reads local artifacts and returns a payload.
  The board is observed activity, not result, verification, review, CI,
  merge-readiness, or merge evidence.
- Deterministic. `now` is a parameter so callers and tests pin it; the clock is
  read only when the caller declines to name a time. Ordering, column widths,
  and truncation are functions of the data alone.
- English by default. Localized copy is opt-in through the explicit `locale`
  argument (fed by `--language` / `OMH_LANG`); the OS locale is never consulted.
- Metadata only. Unit ids, run refs, model ids, and status words are read;
  free-text summaries pass the shared progress sanitizer before they render.

Telemetry parsing is deliberately absent: `unit_telemetry.parse_unit_telemetry`
runs at dispatch time and its values arrive here already persisted in the
dispatch summary, so this module never re-parses executor output.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Mapping

from ..system.local_store import read_json_object_result, utc_now
from ..evidence import PHASE_CODE, status_label
from ..evidence.labels import (
    CONFIDENCE_BLOCKED,
    CONFIDENCE_FAILED,
    CONFIDENCE_NOT_RUN,
    CONFIDENCE_REPORTED_DONE,
    CONFIDENCE_RUNNING,
)
from ..plugin_bundle.omh.hermes_delegation import read_hermes_native_subagents
from ..plugin_bundle.omh.kanban_board_reader import read_kanban_lanes
from ..plugin_bundle.omh.run_records import DISPATCH_STATUS_VOCABULARY, DISPATCH_SUMMARY_FILE, FANOUT_CONTRACT_FILE
from ..system.paths import OmhPaths
from ..workflows.observation_journal import (
    failure_diagnostic_text, project_bound_failure_diagnostic,
    project_run_failure_diagnostic, read_observation_events_result,
)
from .fanout_failure_diagnostics import FailureDiagnostic, is_object_list, is_string_map
from .context_safety import sanitize_user_facing_progress_text
from .executor_progress import project_active_executor_status
from .inflight import read_inflight_markers
from .unit_execution_state import is_stuck
from .routing_observation import (
    authenticate_executor_observation,
    build_routing_observation,
    render_routing_status_rows,
)
from ..catalogs.briefing_vocabulary import status_board_copy
from ..system.display_width import display_width, pad_to_width
from ..catalogs.model_vendor_glyphs import vendor_glyph
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

CODING_STATUS_BOARD_SCHEMA_VERSION: Final[str] = "omh_coding_status_board/v1"

CODING_STATUS_BOARD_CLAIM_BOUNDARY: Final[str] = (
    "Status board rows are metadata-only observed activity. A unit's runtime and model are exact "
    "because omh selected them; a Hermes worker's model is the one Hermes recorded. Tokens, session "
    "refs, and the elapsed time of an unfinished unit are observed or the literal unknown, never "
    "estimated. It is not result, verification, review, CI, merge-readiness, or merge evidence."
)

# The messenger line. The full boundary above is longer than a two-row board of
# roughly 275 characters -- over half of what a person sees on a phone would be
# boilerplate they scroll past to reach two lines of data, which is exactly the
# saturation a chat surface has to avoid. The full text stays in the payload and
# in `--json`, where an auditor reads it; the short form carries the two claims
# a reader of the board actually needs.
CODING_STATUS_BOARD_SHORT_BOUNDARY: Final[str] = (
    "Observed activity only; unknown is never estimated."
)

# Closed status vocabulary, shared with the plugin's running-work reader.
# Anything else an artifact carries collapses to `prepared_not_observed`.
STATUS_VOCABULARY: Final[tuple[str, ...]] = DISPATCH_STATUS_VOCABULARY

UNKNOWN: Final[str] = "unknown"
DEFAULT_LIMIT: Final[int] = 20

SOURCE_INFLIGHT: Final[str] = "inflight_markers"
SOURCE_DISPATCH_SUMMARY: Final[str] = "dispatch_summary"
SOURCE_EXECUTOR_PROGRESS: Final[str] = "executor_progress"
SOURCE_HERMES_DELEGATES: Final[str] = "hermes_delegate_children"
SOURCE_HERMES_KANBAN: Final[str] = "hermes_kanban_board"
# Merge order is also precedence order: an in-flight marker wins over a
# dispatch-summary row for the same unit, and both win over a progress binding.
# The two Hermes sources come last and never merge into a unit: their rows are
# Hermes' own delegate children and board tasks, keyed by Hermes' ids, and the
# board lists them apart under `hermes_workers`.
SOURCE_ORDER: Final[tuple[str, ...]] = (
    SOURCE_INFLIGHT,
    SOURCE_DISPATCH_SUMMARY,
    SOURCE_EXECUTOR_PROGRESS,
    SOURCE_HERMES_DELEGATES,
    SOURCE_HERMES_KANBAN,
)

# The row cap both Hermes readers are called with. It is at or above each
# reader's own selection bound, so the board counts every row a reader selected
# and applies its own `limit` afterwards; rows a reader still reports as hidden
# are added to `worker_count` rather than dropped. The selection itself is
# bounded: a reader that reached its bound says so (`selection_capped`), and
# the board then renders its counts as lower bounds rather than as totals.
_HERMES_READ_LIMIT: Final[int] = 64
# The SOURCE cell, and the Hermes file each reader opens (named in the "not
# available" line by file name only, never by its local path).
_HERMES_SOURCE_LABELS: Final[Mapping[str, str]] = {
    SOURCE_HERMES_DELEGATES: "hermes-delegate",
    SOURCE_HERMES_KANBAN: "hermes-kanban",
}
_HERMES_SOURCE_FILES: Final[Mapping[str, str]] = {
    SOURCE_HERMES_DELEGATES: "state.db",
    SOURCE_HERMES_KANBAN: "kanban.db",
}
# A reader's `read_status` -> the line that says the read failed. `read` has no
# line; any other value, including one this map does not know, gets one, so a
# failed read is never shown as an empty source.
_HERMES_READ_COPY: Final[Mapping[str, str]] = {
    "absent": "hermes_absent",
    "unreadable": "hermes_unreadable",
    "partial": "hermes_partial",
    "refused": "hermes_refused",
}
_HERMES_READ_OK: Final[str] = "read"
# What a person reads for each state the two Hermes readers produce. `done` is
# Hermes' report about its own worker, so it reads `reported done`, the label a
# unit's `completed` gets; a state outside both readers' vocabularies fails
# closed to `not run`, the way `normalize_status` does. `stale` is a kanban
# lane Hermes still calls running whose heartbeat stopped.
_HERMES_STATE_TEXT: Final[Mapping[str, str]] = {
    "running": CONFIDENCE_RUNNING,
    "stale": "stale",
    "queued": "queued",
    "blocked": CONFIDENCE_BLOCKED,
    "failed": CONFIDENCE_FAILED,
    "done": CONFIDENCE_REPORTED_DONE,
}
# States that still hold a worker slot as far as Hermes knows; they sort first,
# as a unit with `status: running` does whether or not it is stuck.
_HERMES_LIVE_STATES: Final[frozenset[str]] = frozenset({"running", "stale"})

_SUMMARY_LIMIT: Final[int] = 120
# Bound for a status word the closed vocabulary refused. It is a word, not a
# message. It reaches the STATUS column too: a payload-only field answers an
# auditor reading `--json` and nobody reading the board, and the board is where
# the collapse is actually noticed.
_STATUS_SOURCE_LIMIT: Final[int] = 80
_MODEL_DEFAULT_LABEL: Final[str] = "executor default"
# `inflight.INFLIGHT_MARKER_STATUSES` value that means the marker was read and
# carries observed fields; the other two mean it carries nothing.
_MARKER_PRESENT: Final[str] = "present"

# Column headers, in the same order `omh coding fanout brief` prints its parts.
# `_fanout_brief_unit_line` in src/commands/coding.py joins:
#     unit_id — owner (model_label) — status — elapsed — tokens — session … — summary
# and builds `model_label` with `model_label_for` below, so the board and the
# brief read identically by construction.
_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("LABEL", "label"),
    ("RUNTIME", "runtime"),
    ("MODEL", "model_label"),
    ("STATUS", "status"),
    ("ELAPSED", "elapsed_text"),
    ("TOKENS", "tokens_text"),
    ("TOOLS", "tool_count_text"),
    ("SESSION", "session_ref"),
)
# The Hermes table keeps the unit table's column order with SOURCE in front.
# Its ID is Hermes' own (a child session's tail, a board task id), and it has
# no SESSION column because the delegate reader carries only that tail.
_HERMES_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("SOURCE", "source"),
    ("ID", "worker_id"),
    ("MODEL", "model_label"),
    ("STATE", "state"),
    ("ELAPSED", "elapsed_text"),
    ("TOKENS", "tokens_text"),
    ("TOOLS", "tool_count_text"),
)

_RICH_MARKDOWN: Final[str] = "rich_markdown"
_LIMITED_MARKDOWN: Final[str] = "limited_markdown"


def build_status_board(paths: OmhPaths, *, limit: int = DEFAULT_LIMIT, now: str = "") -> dict[str, Any]:
    """Project every locally observed coding unit into one board payload."""
    observed_at = now or utc_now()
    effective_limit = limit if limit > 0 else 0

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    sources_used: list[str] = []

    for unit in _inflight_units(paths, observed_at):
        _merge_unit(merged, unit, source=SOURCE_INFLIGHT, sources_used=sources_used)
    for unit in _dispatch_summary_units(paths):
        _merge_unit(merged, unit, source=SOURCE_DISPATCH_SUMMARY, sources_used=sources_used)
    for unit in _progress_units(paths, observed_at):
        _merge_unit(merged, unit, source=SOURCE_EXECUTOR_PROGRESS, sources_used=sources_used)
    hermes = _hermes_workers(paths, observed_at, limit=effective_limit)
    sources_used.extend(
        str(record["source"]) for record in hermes["sources"] if int(record["worker_count"]) > 0
    )

    units = sorted(merged.values(), key=_board_sort_key)
    running_count = sum(1 for unit in units if unit["status"] == "running")
    # Counted apart from `running_count` rather than subtracted out of it:
    # `running_count` is the DISPATCH tally (how many units have a live-looking
    # marker) and existing consumers gate on it, so its meaning stays put. What
    # changes is that the header no longer reports that tally alone, because
    # "2 running" over a board with a unit stuck for half an hour is the exact
    # reading the 2026-09-11 incident acted on.
    stuck_count = sum(1 for unit in units if is_stuck(str(unit.get("unit_state", "") or "")))
    return {
        "schema_version": CODING_STATUS_BOARD_SCHEMA_VERSION,
        "observed_at": observed_at,
        "unit_count": len(units),
        "running_count": running_count,
        "stuck_count": stuck_count,
        "units": units[:effective_limit],
        # `unit_count`, `running_count` and `stuck_count` above stay the fanout
        # tally; Hermes workers carry their own counts here, and the rendered
        # headline adds the two.
        "hermes_workers": hermes,
        "sources_used": [source for source in SOURCE_ORDER if source in sources_used],
        "claim_boundary": CODING_STATUS_BOARD_CLAIM_BOUNDARY,
    }


def render_status_board_text(payload: dict[str, Any], *, locale: str = "") -> str:
    """Aligned ASCII board. English unless `locale` names another language.

    `locale` used to reach a boolean: the renderer asked `_korean_locale` and
    every language that was not Korean rendered as English, including the three
    the chat-copy set already supports. The copy now comes from
    `briefing_vocabulary`, so the answer is per locale rather than per side of
    a boolean.
    """
    units = _payload_units(payload)
    workers = _payload_hermes_rows(payload)
    lines = [_header_line(payload, locale=locale)]
    if not units and not workers:
        lines.append(_empty_line(locale=locale))
    if units:
        widths = _column_widths(units)
        lines.append(_table_row([header for header, _ in _COLUMNS], widths, summary=""))
        for unit in units:
            cells = [_cell(unit, field) for _, field in _COLUMNS]
            lines.append(_table_row(cells, widths, summary=str(unit.get("summary", "") or "")))
            observation = unit.get("routing_observation")
            if isinstance(observation, Mapping):
                lines.extend(render_routing_status_rows(observation))
        truncation = _truncation_line(payload, locale=locale)
        if truncation:
            lines.append(truncation)
    if workers:
        widths = [
            max([display_width(header)] + [display_width(_hermes_cell(worker, field)) for worker in workers])
            for header, field in _HERMES_COLUMNS
        ]
        lines.append(_table_row([header for header, _ in _HERMES_COLUMNS], widths, summary=""))
        for worker in workers:
            cells = [_hermes_cell(worker, field) for _, field in _HERMES_COLUMNS]
            lines.append(_table_row(cells, widths, summary=str(worker.get("summary", "") or "")))
        lines.append(_hermes_truncation_line(payload, locale=locale))
    lines.extend(_hermes_read_lines(payload, locale=locale))
    lines.append(str(payload.get("claim_boundary", "")))
    return "\n".join(line for line in lines if line)


def status_board_messenger_body(
    payload: dict[str, Any], *, render_profile: str = _LIMITED_MARKDOWN, locale: str = ""
) -> str:
    """Messenger-safe body. `rich_markdown` keeps alignment; `limited_markdown` is flat bullets.

    The boundary is the SHORT form and sits OUTSIDE the fence. Inside it, a
    paragraph of prose is monospace-wrapped at the table's width, which reads as
    broken rendering rather than as a caveat; and at full length it was more of
    the message than the data was.

    `locale` reaches the same copy the CLI board uses. Before, this function
    took no locale at all and passed `korean=False` at each of its three call
    sites, so a messenger board rendered in English whatever the caller had
    configured -- not as a decision about messengers, which is why nothing said
    so, but because the argument had never been threaded through.
    """
    if render_profile == _RICH_MARKDOWN:
        return (
            "```\n"
            + _aligned_rows_only(payload, locale=locale)
            + "\n```\n"
            + CODING_STATUS_BOARD_SHORT_BOUNDARY
        )
    units = _payload_units(payload)
    workers = _payload_hermes_rows(payload)
    lines = [_header_line(payload, locale=locale)]
    if not units and not workers:
        lines.append(_empty_line(locale=locale))
    for unit in units:
        lines.append(_bullet_line(unit))
    truncation = _truncation_line(payload, locale=locale)
    if truncation:
        lines.append(truncation)
    for worker in workers:
        lines.append(_hermes_bullet_line(worker))
    if workers:
        lines.append(_hermes_truncation_line(payload, locale=locale))
    lines.extend(_hermes_read_lines(payload, locale=locale))
    lines.append(CODING_STATUS_BOARD_SHORT_BOUNDARY)
    return "\n".join(line for line in lines if line)


def _aligned_rows_only(payload: dict[str, Any], *, locale: str = "") -> str:
    """The aligned table without the trailing boundary paragraph."""
    boundary = str(payload.get("claim_boundary", "") or "")
    text = render_status_board_text(payload, locale=locale)
    if boundary and text.endswith(boundary):
        text = text[: -len(boundary)]
    return text.rstrip("\n")


def model_label_for(model: str, reasoning_effort: str) -> str:
    """The `gpt-5.6-sol xhigh` label shape shared with `omh coding fanout brief`."""
    return " ".join(part for part in (str(model or ""), str(reasoning_effort or "")) if part) or _MODEL_DEFAULT_LABEL


def elapsed_text_for(elapsed_seconds: Any) -> str:
    """Compact human elapsed form: `45s`, `35m`, `2h 5m`, `1d 1h`, or `unknown`."""
    if not isinstance(elapsed_seconds, int) or isinstance(elapsed_seconds, bool) or elapsed_seconds < 0:
        return UNKNOWN
    if elapsed_seconds < 60:
        return f"{elapsed_seconds}s"
    if elapsed_seconds < 3600:
        return f"{elapsed_seconds // 60}m"
    if elapsed_seconds < 86400:
        hours, minutes = divmod(elapsed_seconds // 60, 60)
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    days, hours = divmod(elapsed_seconds // 3600, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def tokens_text_for(tokens_total: Any) -> str:
    """Thousands-separated observed total, or the literal `unknown`."""
    if isinstance(tokens_total, int) and not isinstance(tokens_total, bool) and tokens_total >= 0:
        return f"{tokens_total:,}"
    return UNKNOWN


def tool_count_text_for(tool_count: Any) -> str:
    """Observed tool-call count, or the literal `unknown`.

    Zero is a real, renderable answer -- an executor that made no tool calls
    made no tool calls -- which is exactly why an UNOBSERVED count must not
    render as `0`. The two mean opposite things to a reader deciding whether a
    quiet lane is working or stuck.
    """
    if isinstance(tool_count, int) and not isinstance(tool_count, bool) and tool_count >= 0:
        return str(tool_count)
    return UNKNOWN


def normalize_status(value: Any) -> str:
    status = str(value or "").strip()
    return status if status in STATUS_VOCABULARY else "prepared_not_observed"


def unmapped_status_source(value: Any) -> str:
    """The status word `normalize_status` discarded, or "" when it kept it.

    The downgrade itself is correct and stays: an unknown status word must not
    be read as an observation, and `prepared_not_observed` is the honest floor.
    What was wrong is that the word then vanished, so a board could not
    distinguish "nothing was ever written down" from "an executor reported
    something this vocabulary does not know". A row carries the discarded word
    beside the downgraded status instead of losing it.
    """
    status = str(value or "").strip()
    if not status or status in STATUS_VOCABULARY:
        return ""
    return sanitize_user_facing_progress_text(status, max_chars=_STATUS_SOURCE_LIMIT)


def status_text_for(unit: Mapping[str, Any]) -> str:
    """The STATUS column as a person reads it: the vocabulary word, plus what was refused.

    `prepared_not_observed` alone cannot distinguish "nobody ever watched this
    run" from "an executor said something this vocabulary does not know", and
    those call for different actions. The refused word rides in parentheses on
    the same cell rather than in a column of its own, so a board with nothing
    refused is byte-identical to the one this repo already renders.

    A unit whose `unit_state` is stuck replaces the cell outright. `running`
    here only ever meant "a process was spawned and its marker is still
    there", and rendering that word over a unit that has been repeating one
    error for half an hour is the failure this whole surface exists to stop;
    the stuck word, its reason, and how long output has been still take the
    cell instead. A `running` or absent `unit_state` changes nothing, so a
    board of healthy units renders exactly as it did before.

    Hand-mirrored by `plugin_bundle/omh/status_board_reader._status_text`, which
    cannot import this module; `tests/test_coding_status_board.py` gates the two
    against each other.
    """
    stuck = stuck_state_text(unit)
    if stuck:
        return stuck
    # Every row on this board is a coding unit, so `Code` is the phase when the
    # value does not imply a more specific one (`worktree_failed` implies
    # `Setup`). The wire value stays in `unit["status"]` for anything parsing
    # the payload; only this cell changes.
    status = status_label(str(unit.get("status", "") or UNKNOWN), default_phase=PHASE_CODE)
    source = str(unit.get("unmapped_source_status", "") or "")
    return f"{status} (reported {source})" if source else status


def stuck_state_text(unit: Mapping[str, Any]) -> str:
    """`progress_stalled (no_new_output, 1920s since new output)`, or "".

    Empty for every unit whose observed state is `running`, absent, or not a
    stuck one -- `is_stuck` owns that judgement, so a state added to the
    vocabulary starts rendering here without a second list to update.

    Hand-mirrored by `plugin_bundle/omh/status_board_reader._stuck_state_text`.
    """
    state = str(unit.get("unit_state", "") or "")
    if not is_stuck(state):
        return ""
    parts: list[str] = []
    reason = str(unit.get("state_reason", "") or "")
    if reason:
        parts.append(reason)
    elapsed = str(unit.get("stalled_for_seconds", "") or "")
    if elapsed.isdigit() and int(elapsed) > 0:
        parts.append(f"{elapsed}s since new output")
    return f"{state} ({', '.join(parts)})" if parts else state


def _merge_unit(
    merged: dict[tuple[str, str], dict[str, Any]],
    unit: dict[str, Any],
    *,
    source: str,
    sources_used: list[str],
) -> None:
    key = (str(unit.get("fanout_id", "")), str(unit.get("unit_id", "")))
    if source not in sources_used:
        sources_used.append(source)
    if key in merged:
        return
    merged[key] = {field: value for field, value in unit.items() if field != "fanout_id"}


def _board_sort_key(unit: dict[str, Any]) -> tuple[int, str, str]:
    running_first = 0 if unit.get("status") == "running" else 1
    return (running_first, str(unit.get("label", "")), str(unit.get("unit_id", "")))


def _inflight_units(paths: OmhPaths, observed_at: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for marker in read_inflight_markers(paths):
        if not isinstance(marker, dict):
            continue
        unit_id = str(marker.get("unit_id", "") or "")
        if not unit_id:
            continue
        # `absent` and `unreadable` markers carry no observed fields at all.
        # Rendering one as a running unit would put a row on the board that
        # nothing observed, so they are dropped rather than shown as unknown.
        if str(marker.get("marker_status", _MARKER_PRESENT)) != _MARKER_PRESENT:
            continue
        elapsed = _elapsed_since(str(marker.get("started_at", "") or ""), observed_at)
        units.append(
            _unit_row(
                fanout_id=str(marker.get("fanout_id", "") or ""),
                unit_id=unit_id,
                label=unit_id,
                run_ref=str(marker.get("run_ref", "") or ""),
                runtime=str(marker.get("owner", "") or ""),
                runtime_host=str(marker.get("owner_host", "") or ""),
                model=str(marker.get("model", "") or ""),
                reasoning_effort=str(marker.get("reasoning_effort", "") or ""),
                # `status` stays the dispatch word: a marker exists, so a
                # process was spawned. Whether the WORK is moving is a
                # different question, carried beside it in `unit_state` and
                # rendered by `status_text_for`. Keeping the two apart is the
                # point -- `STATUS_VOCABULARY` is read as a closed enum by the
                # graph roster, and "alive" must never render as "progressing".
                status="running",
                elapsed_seconds=elapsed,
                # A running unit has reported no total yet. Anything we put here
                # would be a guess, so it stays unknown until an executor says.
                tokens_total=UNKNOWN,
                session_ref="",
                # A marker records where the unit runs, not what it has done.
                # The worktree path is deliberately not shown: it is a local
                # filesystem path, not observed progress.
                summary="",
                unit_state=str(marker.get("unit_state", "") or ""),
                state_reason=str(marker.get("state_reason", "") or ""),
                stalled_for_seconds=str(marker.get("stalled_for_seconds", "") or ""),
            )
        )
    return units


def _dispatch_summary_units(paths: OmhPaths) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    root = paths.fanout_contracts_dir
    if not root.is_dir():
        return units
    journal_events, _ = read_observation_events_result(paths)
    for fanout_dir in sorted(root.iterdir()):
        if not fanout_dir.is_dir() or fanout_dir.is_symlink():
            continue
        summary, error = read_json_object_result(fanout_dir / DISPATCH_SUMMARY_FILE)
        if error or not is_string_map(summary):
            continue
        fanout_id = str(summary.get("fanout_id", "") or fanout_dir.name)
        contract_units = _contract_units(fanout_dir)
        entries = summary.get("units", [])
        if not is_object_list(entries):
            continue
        for entry in entries:
            if not is_string_map(entry):
                continue
            unit_id = str(entry.get("unit_id", "") or "")
            if not unit_id:
                continue
            run_ref = str(entry.get("run_ref", ""))
            diagnostic = project_bound_failure_diagnostic(
                entry, fanout_id=fanout_dir.name, unit_id=unit_id, run_ref=run_ref,
            ) if fanout_id == fanout_dir.name else None
            events = [event for event in journal_events if event.get("run_id") == run_ref]
            current = project_run_failure_diagnostic(events, run_id=run_ref)
            if events and current.get("attempt_id") != entry.get("attempt_id"):
                diagnostic = None
            contract_unit = contract_units.get(unit_id, {})
            handoff = contract_unit.get("handoff") if isinstance(contract_unit.get("handoff"), Mapping) else {}
            route = handoff.get("model_route") if isinstance(handoff.get("model_route"), Mapping) else {}
            observation = build_routing_observation(
                route=route,
                session_observation=authenticate_executor_observation({
                    "status": entry.get("status"),
                    "owner": entry.get("owner"),
                    "model": entry.get("model"),
                    "reasoning_effort": entry.get("reasoning_effort"),
                    "elapsed_seconds": _finished_seconds(entry.get("duration_seconds")),
                    "tokens": _reported_tokens(entry),
                    "session_id": entry.get("session_ref"),
                    "run_id": entry.get("run_ref"),
                }),
            )
            units.append(
                _unit_row(
                    fanout_id=fanout_id,
                    unit_id=unit_id,
                    label=str(contract_unit.get("title", "") or unit_id),
                    run_ref=str(entry.get("run_ref", "") or ""),
                    runtime=str(entry.get("owner", "") or ""),
                    runtime_host="",
                    model=str(entry.get("model", "") or ""),
                    reasoning_effort=str(entry.get("reasoning_effort", "") or ""),
                    status=normalize_status(entry.get("status")),
                    unmapped_source_status=unmapped_status_source(entry.get("status")),
                    elapsed_seconds=_finished_seconds(entry.get("duration_seconds")),
                    tokens_total=_reported_tokens(entry),
                    session_ref=str(entry.get("session_ref", "") or ""),
                    summary=str(entry.get("reason", "") or ""),
                    routing_observation=observation,
                    failure_diagnostic=diagnostic,
                )
            )
    return units


def _progress_units(paths: OmhPaths, observed_at: str) -> list[dict[str, Any]]:
    projection = project_active_executor_status(paths, now=observed_at)
    units: list[dict[str, Any]] = []
    for row in projection.get("active_executors", []):
        if not isinstance(row, dict):
            continue
        unit_id = str(row.get("target_id", "") or "")
        if not unit_id:
            continue
        event = row.get("latest_event", {})
        event = event if isinstance(event, dict) else {}
        observation = build_routing_observation(
            session_observation=authenticate_executor_observation({
                **row,
                **event,
                "status": event.get("status") or row.get("state"),
                "owner": row.get("executor_profile"),
                "run_id": row.get("target_id"),
            }),
        )
        units.append(
            _unit_row(
                fanout_id="",
                unit_id=unit_id,
                label=unit_id,
                run_ref=unit_id if str(row.get("target_type", "")) == "run" else "",
                runtime=str(row.get("executor_profile", "") or ""),
                runtime_host="",
                # A progress binding records who is working, not which model was
                # picked; the route lives in the fanout contract, so the board
                # says unknown rather than inventing one.
                model="",
                reasoning_effort="",
                status="running",
                elapsed_seconds=_elapsed_since(str(row.get("latest_observed_at", "") or ""), observed_at),
                tokens_total=UNKNOWN,
                # `_compact_event_projection` folds the binding's observed
                # routing metrics into the event, so the count is already here;
                # it simply had no field to travel in.
                tool_count=event.get("tool_count"),
                session_ref="",
                summary=str(event.get("summary", "") or ""),
                routing_observation=observation,
            )
        )
    return units


def _hermes_workers(paths: OmhPaths, observed_at: str, *, limit: int) -> dict[str, Any]:
    """Hermes' own delegate children and kanban lanes, read by the readers `omh_hud` uses.

    The board belongs to no conversation, so both readers answer in their
    explicit global scope, as `omh_hud` does for an unmapped session. Both
    open Hermes' databases read-only. Each source keeps the reader's
    `read_status`, so a database that is missing or would not answer renders as
    "not available" rather than as a source with nothing running.
    """
    start = _parse_timestamp(observed_at)
    now = start.timestamp() if start is not None else None
    native = read_hermes_native_subagents(
        paths.hermes_home, now=now, limit=_HERMES_READ_LIMIT, omh_home=paths.omh_home
    )
    kanban = read_kanban_lanes(paths.hermes_home, now=now, limit=_HERMES_READ_LIMIT)
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for source, read in ((SOURCE_HERMES_DELEGATES, native), (SOURCE_HERMES_KANBAN, kanban)):
        read_rows = [row for row in read.get("rows", []) if isinstance(row, Mapping)]
        rows.extend(_hermes_worker_row(row, source=source) for row in read_rows)
        hidden = int(read.get("hidden", 0) or 0)
        capped = read.get("selection_capped") is True
        record: dict[str, Any] = {
            "source": source,
            "read_status": str(read.get("read_status", "") or ""),
            "scope": str(read.get("scope", "") or ""),
            "worker_count": len(read_rows) + hidden,
            "selection_capped": capped,
            # A capped selection may have left workers unread, and hidden rows
            # are counted without their states: either way the counts below
            # are floors, not totals.
            "counts_are_lower_bounds": capped or hidden > 0,
        }
        if source == SOURCE_HERMES_KANBAN:
            record["board"] = str(read.get("board", "") or "")
            record["dispatcher_presence"] = str(read.get("dispatcher_presence", "") or "")
        sources.append(record)
    rows.sort(key=_hermes_sort_key)
    return {
        "worker_count": sum(int(record["worker_count"]) for record in sources),
        "running_count": sum(1 for row in rows if row["state"] == "running"),
        # A lane Hermes still calls running after its heartbeat stopped. The
        # headline names it beside the stuck units instead of as running.
        "stale_count": sum(1 for row in rows if row["state"] == "stale"),
        "counts_are_lower_bounds": any(record["counts_are_lower_bounds"] for record in sources),
        "rows": rows[:limit],
        "sources": sources,
    }


def _hermes_worker_row(row: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """One reader row in the board's own field names, observed or `unknown`."""
    model = str(row.get("model", "") or "")
    effort = str(row.get("effort", "") or "")
    state = str(row.get("state", "") or "")
    elapsed = row.get("elapsed_seconds")
    # A lane that never started arrives as None and renders `unknown`. A queued
    # lane is `unknown` even when it ran before: no worker is running, the
    # figure would be an earlier run's, and the HUD leaves that cell blank.
    elapsed_seconds = (
        int(elapsed)
        if state != "queued" and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        and math.isfinite(elapsed) and elapsed >= 0
        else UNKNOWN
    )
    tool_count = _observed_tokens(row.get("tool_count"))
    worker: dict[str, Any] = {
        "source": source,
        "worker_id": str(row.get("task_id", "") or "") or UNKNOWN,
        "state": state or UNKNOWN,
        "model": model or UNKNOWN,
        "reasoning_effort": effort or UNKNOWN,
        "model_label": " ".join(part for part in (model, effort) if part) or UNKNOWN,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_text": elapsed_text_for(elapsed_seconds),
        # The delegate reader sends None for a running child that has not
        # reported yet, the kanban reader until a worker session is linked:
        # both are `unknown`, never a zero.
        "tokens_total": _observed_tokens(row.get("tokens")),
        "tokens_text": tokens_text_for(row.get("tokens")),
        "tool_count": tool_count,
        "tool_count_text": tool_count_text_for(tool_count),
        "summary": sanitize_user_facing_progress_text(str(row.get("action", "") or ""), max_chars=_SUMMARY_LIMIT),
        "scope": str(row.get("scope", "") or ""),
    }
    # Why the state reads as it does, when the reader said: a delegate child
    # with no recorded model usage is failed, and a queued lane the dispatcher
    # requeued after a failed run is waiting for a retry, not idle.
    detail = str(row.get("failure_hint", "") or "")
    if not detail and row.get("retry_pending") is True:
        detail = "retry pending"
    if detail:
        worker["state_detail"] = detail
    if source == SOURCE_HERMES_KANBAN:
        worker["native_status"] = str(row.get("native_status", "") or "")
        worker["assignee"] = str(row.get("assignee", "") or "")
    return worker


def _hermes_sort_key(worker: Mapping[str, Any]) -> tuple[int, int, str]:
    """The unit table's order: live rows first, then by source and id."""
    live_first = 0 if worker.get("state") in _HERMES_LIVE_STATES else 1
    source = str(worker.get("source", ""))
    rank = SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)
    return (live_first, rank, str(worker.get("worker_id", "")))


def _unit_row(
    *,
    fanout_id: str,
    unit_id: str,
    label: str,
    run_ref: str,
    runtime: str,
    runtime_host: str,
    model: str,
    reasoning_effort: str,
    status: str,
    elapsed_seconds: Any,
    tokens_total: Any,
    session_ref: str,
    summary: str,
    unmapped_source_status: str = "",
    unit_state: str = "",
    state_reason: str = "",
    stalled_for_seconds: str = "",
    routing_observation: Mapping[str, object] | None = None,
    failure_diagnostic: FailureDiagnostic | None = None,
    tool_count: Any = None,
) -> dict[str, Any]:
    row: dict[str, object] = {
        "fanout_id": fanout_id,
        "label": label or unit_id or UNKNOWN,
        "unit_id": unit_id,
        "run_ref": run_ref or UNKNOWN,
        "runtime": runtime or UNKNOWN,
        "runtime_host": runtime_host or UNKNOWN,
        "model": model or UNKNOWN,
        "reasoning_effort": reasoning_effort or UNKNOWN,
        "model_label": model_label_for(model, reasoning_effort),
        "status": status,
        "elapsed_seconds": elapsed_seconds if isinstance(elapsed_seconds, int) else UNKNOWN,
        "elapsed_text": elapsed_text_for(elapsed_seconds),
        "tokens_total": tokens_total if isinstance(tokens_total, int) else UNKNOWN,
        "tokens_text": tokens_text_for(tokens_total),
        # How much the executor actually did, as opposed to how long it has been
        # running. A lane at twelve minutes and two tool calls and a lane at
        # twelve minutes and sixty are different situations, and elapsed time
        # alone reports them identically. Observed or the literal `unknown`,
        # like every other count here -- `parse_unit_telemetry` records a
        # missing count as an absent key, never as a zero, and a rendered zero
        # would claim an executor did nothing.
        "tool_count": tool_count if isinstance(tool_count, int) else UNKNOWN,
        "tool_count_text": tool_count_text_for(tool_count),
        "session_ref": session_ref or UNKNOWN,
        "summary": sanitize_user_facing_progress_text(summary, max_chars=_SUMMARY_LIMIT),
    }
    if failure_diagnostic is not None:
        row["failure_diagnostic"] = failure_diagnostic
        row["summary"] = failure_diagnostic_text(failure_diagnostic)[:_SUMMARY_LIMIT]
    # Absent unless a word was actually discarded, so an absent key means the
    # status was accepted verbatim -- never that nothing was checked.
    if unmapped_source_status:
        row["unmapped_source_status"] = unmapped_source_status
    # Absent unless a stdout snapshot was actually assessed. An absent
    # `unit_state` means nothing observed the work, which is exactly what a
    # reader must not confuse with "the work is moving".
    if unit_state:
        row["unit_state"] = unit_state
    if state_reason:
        row["state_reason"] = state_reason
    if stalled_for_seconds:
        row["stalled_for_seconds"] = stalled_for_seconds
    if routing_observation is not None:
        row["routing_observation"] = dict(routing_observation)
        row["routing_status_rows"] = list(render_routing_status_rows(routing_observation))
    return row


def _contract_units(fanout_dir: Path) -> dict[str, dict[str, Any]]:
    contract, error = read_json_object_result(fanout_dir / FANOUT_CONTRACT_FILE)
    if error or not isinstance(contract, dict):
        return {}
    units: dict[str, dict[str, Any]] = {}
    for unit in contract.get("units", []):
        if not isinstance(unit, dict):
            continue
        unit_id = str(unit.get("unit_id", "") or "")
        if unit_id:
            units[unit_id] = unit
    return units


def _unit_titles(fanout_dir: Path) -> dict[str, str]:
    """Compatibility helper retained for plugin-parity tests."""
    return {
        unit_id: str(unit.get("title", "") or unit_id)
        for unit_id, unit in _contract_units(fanout_dir).items()
    }


def _finished_seconds(value: Any) -> Any:
    """Observed wall time of a finished unit, floored to whole seconds."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return UNKNOWN
    if value < 0:
        return UNKNOWN
    return int(value)


def _observed_tokens(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return UNKNOWN
    return value


def _reported_tokens(entry: Mapping[str, Any]) -> Any:
    """Prefer a provider-stated total, else the sum of stated components.

    Neither codex nor claude reports a `total_tokens`, verified against real
    captured output, so reading only `tokens_total` left this column `unknown`
    on every actual run -- the exact defect the board exists to remove.
    `tokens_billable` is the sum of components the CLI itself printed
    (input, output, cache read, cache write, reasoning); it is aggregation of
    stated numbers, never an estimate, and it carries its own
    `tokens_billable_source` label in the record for anyone auditing it.
    """
    total = _observed_tokens(entry.get("tokens_total"))
    if total != UNKNOWN:
        return total
    return _observed_tokens(entry.get("tokens_billable"))


def _elapsed_since(started_at: str, observed_at: str) -> Any:
    """Whole seconds between two observed timestamps, or `unknown` if either is not one."""
    start = _parse_timestamp(started_at)
    end = _parse_timestamp(observed_at)
    if start is None or end is None:
        return UNKNOWN
    seconds = int((end - start).total_seconds())
    return seconds if seconds >= 0 else UNKNOWN


def _parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = strict_fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _payload_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    units = payload.get("units", [])
    return [unit for unit in units if isinstance(unit, dict)] if isinstance(units, list) else []


def _cell(unit: dict[str, Any], field: str) -> str:
    # The STATUS column is the one cell that is not a plain field read: it
    # carries the refused source word when there was one. Doing it here rather
    # than as a precomputed row field keeps `_column_widths`, the aligned table,
    # and the fenced messenger profile in agreement by construction.
    if field == "status":
        return status_text_for(unit)
    # MODEL leads with its vendor's glyph, for the same reason and in the same
    # place: a board of three lanes differs in this column and nowhere else a
    # reader scans, and doing it here means the widths, the aligned table and
    # the bullets cannot disagree about how wide the cell is.
    if field == "model_label":
        label = str(unit.get(field, "") or UNKNOWN)
        return f"{vendor_glyph(label)} {label}"
    return str(unit.get(field, "") or UNKNOWN)


def _column_widths(units: list[dict[str, Any]]) -> list[int]:
    return [
        max([display_width(header)] + [display_width(_cell(unit, field)) for unit in units])
        for header, field in _COLUMNS
    ]


def _table_row(cells: list[str], widths: list[int], *, summary: str) -> str:
    padded = [pad_to_width(cell, width) for cell, width in zip(cells, widths)]
    row = "  ".join(padded).rstrip()
    return f"{row}  {summary}".rstrip() if summary else row


def _bullet_line(unit: dict[str, Any]) -> str:
    parts = [
        str(unit.get("label", "")),
        # Runtime and model are ONE field visually: "codex (gpt-5.6-sol xhigh)".
        # Separating them with a dash as well produced "codex — (gpt-5.6-sol
        # xhigh)", a doubled separator around a parenthetical.
        # Same glyph as the aligned board's MODEL column, in front of the same
        # runtime/model pair, so the two profiles read as one surface.
        f"{vendor_glyph(str(unit.get('model_label', '') or ''))} "
        f"{unit.get('runtime', UNKNOWN)} ({unit.get('model_label', _MODEL_DEFAULT_LABEL)})",
        status_text_for(unit),
        *_figure_phrases(unit),
        f"session {unit.get('session_ref', UNKNOWN)}",
    ]
    line = "- " + " — ".join(parts)
    summary = str(unit.get("summary", "") or "")
    return f"{line} — {summary}" if summary else line


def _figure_phrases(row: Mapping[str, Any]) -> list[str]:
    """Elapsed, tool calls and tokens as a bullet says them, `unknown` spelled out."""
    elapsed_text = str(row.get("elapsed_text", UNKNOWN) or UNKNOWN)
    tokens_text = str(row.get("tokens_text", UNKNOWN) or UNKNOWN)
    tools_text = str(row.get("tool_count_text", UNKNOWN) or UNKNOWN)
    return [
        "elapsed unknown" if elapsed_text == UNKNOWN else elapsed_text,
        "tools unknown" if tools_text == UNKNOWN else f"{tools_text} tool calls",
        "tokens unknown" if tokens_text == UNKNOWN else f"{tokens_text} tokens",
    ]


def _payload_hermes(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    block = payload.get("hermes_workers")
    return block if isinstance(block, Mapping) else {}


def _payload_hermes_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _payload_hermes(payload).get("rows", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _hermes_state_text(worker: Mapping[str, Any]) -> str:
    text = _HERMES_STATE_TEXT.get(str(worker.get("state", "") or ""), CONFIDENCE_NOT_RUN)
    detail = str(worker.get("state_detail", "") or "")
    return f"{text} ({detail})" if detail else text


def _hermes_cell(worker: Mapping[str, Any], field: str) -> str:
    if field == "source":
        return _HERMES_SOURCE_LABELS.get(str(worker.get("source", "") or ""), UNKNOWN)
    if field == "state":
        return _hermes_state_text(worker)
    if field == "model_label":
        label = str(worker.get(field, "") or UNKNOWN)
        return f"{vendor_glyph(label)} {label}"
    return str(worker.get(field, "") or UNKNOWN)


def _hermes_bullet_line(worker: Mapping[str, Any]) -> str:
    """A unit bullet's shape, with the source where a unit names its runtime."""
    label = str(worker.get("model_label", "") or UNKNOWN)
    parts = [
        str(worker.get("worker_id", "") or UNKNOWN),
        f"{vendor_glyph(label)} {_hermes_cell(worker, 'source')} ({label})",
        _hermes_state_text(worker),
        *_figure_phrases(worker),
    ]
    line = "- " + " — ".join(parts)
    summary = str(worker.get("summary", "") or "")
    return f"{line} — {summary}" if summary else line


def _hermes_lower_bound(payload: Mapping[str, Any]) -> bool:
    return _payload_hermes(payload).get("counts_are_lower_bounds") is True


def _count_text(count: int, *, lower_bound: bool, locale: str) -> str:
    """`96`, or `at least 96` when a reader limit may have left workers uncounted."""
    return status_board_copy("at_least", locale=locale).format(count=count) if lower_bound else str(count)


def _hermes_truncation_line(payload: Mapping[str, Any], *, locale: str) -> str:
    total = int(_payload_hermes(payload).get("worker_count", 0) or 0)
    shown = len(_payload_hermes_rows(payload))
    dropped = total - shown
    if dropped <= 0:
        return ""
    floor = _hermes_lower_bound(payload)
    return status_board_copy("hermes_truncation", locale=locale).format(
        shown=shown,
        total=_count_text(total, lower_bound=floor, locale=locale),
        dropped=_count_text(dropped, lower_bound=floor, locale=locale),
    )


def _hermes_source_name(record: Mapping[str, Any], *, locale: str) -> str:
    if str(record.get("source", "") or "") == SOURCE_HERMES_KANBAN:
        return status_board_copy("hermes_kanban_source", locale=locale).format(
            board=str(record.get("board", "") or UNKNOWN)
        )
    return status_board_copy("hermes_delegate_source", locale=locale)


def _hermes_read_lines(payload: Mapping[str, Any], *, locale: str) -> list[str]:
    """One line per Hermes source the board could not read in full or counted only in part."""
    sources = _payload_hermes(payload).get("sources", [])
    lines: list[str] = []
    for record in sources if isinstance(sources, list) else []:
        if not isinstance(record, Mapping):
            continue
        name = _hermes_source_name(record, locale=locale)
        status = str(record.get("read_status", "") or "")
        if status != _HERMES_READ_OK:
            key = _HERMES_READ_COPY.get(status, "hermes_unreadable")
            source = str(record.get("source", "") or "")
            lines.append(
                status_board_copy(key, locale=locale).format(
                    source=name, file=_HERMES_SOURCE_FILES.get(source, UNKNOWN)
                )
            )
        if record.get("counts_are_lower_bounds") is True:
            lines.append(status_board_copy("hermes_lower_bound", locale=locale).format(source=name))
    return lines


def _header_line(payload: dict[str, Any], *, locale: str) -> str:
    running = int(payload.get("running_count", 0) or 0)
    total = int(payload.get("unit_count", 0) or 0)
    # Stuck units are SUBTRACTED from the headline count and named separately.
    # They still carry `status: running` on the wire (a marker exists), but a
    # header that folds them into "N running" is the one line a reader takes
    # at a glance, and it must not say work is moving when it is not.
    stuck = min(int(payload.get("stuck_count", 0) or 0), running)
    moving = running - stuck
    # One board, one headline: Hermes workers count beside the units, and a
    # lane whose heartbeat stopped is named with the stuck units, not as running.
    hermes = _payload_hermes(payload)
    moving += int(hermes.get("running_count", 0) or 0)
    stuck += int(hermes.get("stale_count", 0) or 0)
    total += int(hermes.get("worker_count", 0) or 0)
    # A Hermes reader that hit a limit leaves every Hermes-derived count a
    # floor; the headline says so instead of presenting a partial count as all.
    floor = _hermes_lower_bound(payload)
    observed_at = str(payload.get("observed_at", "") or UNKNOWN)
    stuck_text = (
        status_board_copy("header_stuck_suffix", locale=locale).format(
            stuck=_count_text(stuck, lower_bound=floor, locale=locale)
        )
        if stuck
        else ""
    )
    return status_board_copy("header", locale=locale).format(
        moving=_count_text(moving, lower_bound=floor, locale=locale),
        stuck=stuck_text,
        total=_count_text(total, lower_bound=floor, locale=locale),
        observed_at=observed_at,
    )


def _empty_line(*, locale: str) -> str:
    return status_board_copy("empty", locale=locale)


def _truncation_line(payload: dict[str, Any], *, locale: str) -> str:
    total = int(payload.get("unit_count", 0) or 0)
    shown = len(_payload_units(payload))
    dropped = total - shown
    if dropped <= 0:
        return ""
    return status_board_copy("truncation", locale=locale).format(shown=shown, total=total, dropped=dropped)
