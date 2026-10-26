"""Deterministic unit telemetry read from a spawned coding CLI's stdout (`omh_unit_telemetry/v1`).

`omh coding fanout dispatch` spawns a local agent CLI per unit and keeps only a
bounded stdout tail in memory. When that CLI was asked for a structured stream,
the tail carries the numbers a fanout status board otherwise has to print as
"unknown": how many tokens the run consumed, which provider-side session it
belongs to, and -- for the one owner that reports it -- what the call cost.
This module lifts exactly those out and nothing else.

Relationship to `codex_progress` (read this before "fixing" either module):

    `codex_progress._TOKEN_USAGE_KEYS` strips `input_tokens`, `output_tokens`,
    `total_tokens` and friends out of the text it collects. That is correct
    there and must stay: that module builds VISIBLE narration text, and an
    accounting blob swept into a human-facing summary reads as executor
    activity it never was ("Background process ... usage ..." noise). This
    module reads the same key names for the opposite purpose -- as INTEGERS
    into a metadata-only counter. It emits no free text at all: every value it
    returns is an int or an opaque, whitespace-free identifier. Widening
    `codex_progress` to keep tokens would leak accounting into narration;
    narrowing this module to drop them would delete the only real numbers the
    board has. Neither is a bug in the other.

Boundaries, in order of importance:

- Never fabricate. A value the CLI did not report is an ABSENT KEY, never a
  zero. The renderer prints "unknown" for absent; a fabricated `0` would read
  as a real observation of a run that used no tokens. Nothing here estimates,
  infers, or sums partial counts into a total the CLI did not itself report --
  Claude's result object reports input and output but no total, so
  `tokens_total` is simply absent for that owner.
- Pure and deterministic. `parse_unit_telemetry` performs no file I/O, no clock
  read, no environment read, and no network call; the same `(owner, stdout_text)`
  always yields the same payload. It never raises: a truncated, interleaved, or
  pathological stream returns `parsed=False`.
- Metadata only. Token counts and an opaque session identifier are the whole
  surface. Prompts, assistant text, diffs, file paths, and error text are never
  read into the payload; a candidate session ref containing whitespace or
  unprintable characters is free text, not an identifier, and is rejected.
- Honest absence over invented coverage. Owners with no structured stdout
  surface (`pi-family-runtime` and its pi-family and opencode hosts) report
  `parsed=False`, `source="none"`. That is the required answer, not a gap to
  close later with heuristic text scraping.
- Reporting only. Telemetry is an observation of what a CLI printed. It is not
  billing truth, not provider-side accounting, and never execution-result,
  verification, review, CI, merge-readiness, or merge evidence.
"""

from __future__ import annotations

import json
import math
from .fanout_failure_diagnostics import is_string_map
from typing import Any, Final, Mapping

UNIT_TELEMETRY_SCHEMA_VERSION: Final[str] = "omh_unit_telemetry/v1"

UNIT_TELEMETRY_CLAIM_BOUNDARY: Final[str] = (
    "Unit telemetry is a metadata-only reading of numbers a coding CLI printed on its own stdout. "
    "It is not billing truth, not provider-side accounting, and never execution-result, verification, "
    "review, CI, merge-readiness, or merge evidence. A count the CLI did not report is an absent key, "
    "never a zero."
)

# `source` names where the reported values came from, so it is "none" exactly
# when nothing was observed. `parsed` and `source != "none"` are equivalent by
# construction.
UNIT_TELEMETRY_SOURCES: Final[tuple[str, ...]] = ("codex_json", "claude_json", "none")

# Optional keys, in the fixed order they are appended to the payload. Each one
# appears only when the stream actually reported it.
UNIT_TELEMETRY_VALUE_KEYS: Final[tuple[str, ...]] = (
    "tokens_total",
    "tokens_billable",
    "tokens_billable_source",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "cost_usd",
    "session_ref",
)

# Owners whose structured stdout reports a per-call cost, and the top-level key
# it reports it under. `claude -p --output-format json` returns one `result`
# object carrying `total_cost_usd` alongside `usage` -- confirmed from this
# CLI's own `--help` documentation of `--output-format json`, not from a
# stream captured in this repo (same provenance caveat as the claude-code
# token shape above). Codex reports none: `codex exec`'s `turn.completed`
# usage event has no cost field, and this module never estimates one from
# tokens -- an omitted owner here means "no reported cost", read as an absent
# key, exactly like every other unreported count.
_COST_KEY_BY_SOURCE: Final[dict[str, str]] = {"claude_json": "total_cost_usd"}

# Owners with a structured stdout surface. Every other owner -- pi-family-runtime and
# its pi-family and opencode hosts, hermes, the omx/omc runtimes, generic --
# resolves to no source and reports honest absence.
_STRUCTURED_SOURCE_BY_OWNER: Final[dict[str, str]] = {
    "codex": "codex_json",
    "claude": "claude_json",
    "claude-code": "claude_json",
}

# Container keys that hold a token-count mapping, in priority order when one
# event carries several. Codex's token-count event nests both
# `total_token_usage` (cumulative) and `last_token_usage` (this turn only)
# under the same parent; only the cumulative name is listed, so the per-turn
# mapping is never mistaken for a run total. `usage` is the terminal-event
# spelling for both `codex` (turn.completed) and `claude-code` (result).
#
# Provenance: the codex key names are confirmed inside this repo --
# `codex_progress._TOKEN_USAGE_KEYS` carries `input_tokens`, `output_tokens`,
# `total_tokens`, `cached_input_tokens`, `reasoning_output_tokens`, and
# `codex_progress._RAW_RUNTIME_EVENT_TYPES` carries `turn.completed`. The
# claude-code `result` object shape (`usage.input_tokens`,
# `usage.output_tokens`, `session_id`) is the documented
# `--output-format json` surface and is NOT confirmed from a stream captured in
# this repo; it is supported as documented.
_TOKEN_CONTAINER_KEYS: Final[tuple[str, ...]] = ("total_token_usage", "token_usage", "usage")

# (payload key, keys inside the token container, first match wins).
#
# The cache rows are here because REAL captured output proved that dropping
# them is not an omission but a misreport. `claude -p ... --output-format json`
# returned:
#     "usage": {"input_tokens": 2, "cache_creation_input_tokens": 14441,
#               "cache_read_input_tokens": 15273, "output_tokens": 4}
# Reporting `input_tokens: 2` for a call that consumed roughly 29,700 input
# tokens is wrong by four orders of magnitude. `codex exec --json` reported the
# same categories under different names (`cached_input_tokens`,
# `cache_write_input_tokens`) plus `reasoning_output_tokens`, so the two
# vocabularies are normalized onto one set of payload keys here.
#
# Neither CLI reports a total: codex's `turn.completed.usage` has no
# `total_tokens` and claude's `usage` has none either. The row stays for a
# provider that does report one.
_TOKEN_FIELDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("tokens_total", ("total_tokens",)),
    ("input_tokens", ("input_tokens",)),
    ("output_tokens", ("output_tokens",)),
    ("cache_read_tokens", ("cache_read_input_tokens", "cached_input_tokens")),
    ("cache_write_tokens", ("cache_creation_input_tokens", "cache_write_input_tokens")),
    ("reasoning_tokens", ("reasoning_output_tokens",)),
)

# Components of everything a run actually consumed. Summed into
# `tokens_billable` ONLY when the stream reported at least one of them, and
# always labelled `tokens_billable_source: "summed_reported_components"` so it
# can never be read as a total the provider stated. Summing values the CLI
# printed is aggregation; it is not the estimation this module refuses to do.
_BILLABLE_COMPONENT_KEYS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)

# Session/thread identifier keys per source, in priority order.
_SESSION_KEYS_BY_SOURCE: Final[dict[str, tuple[str, ...]]] = {
    "codex_json": ("session_id", "thread_id", "conversation_id"),
    "claude_json": ("session_id",),
}

# Work caps for pathological input. A dispatch tail is a few kilobytes; these
# bounds only exist so a huge, deeply nested, or single-giant-line stream costs
# a predictable amount of work instead of an unbounded one. The per-line cap is
# deliberately generous: a claude `result` object embeds the assistant's final
# message, so a legitimate telemetry-bearing line can be hundreds of kilobytes,
# and a cap tight enough to "look safe" would silently drop real counts.
MAX_STDOUT_LINES: Final[int] = 20000
MAX_LINE_CHARS: Final[int] = 1000000
MAX_DOCUMENT_CHARS: Final[int] = 2000000
MAX_OBJECT_DEPTH: Final[int] = 6
MAX_MAPPING_KEYS: Final[int] = 200
MAX_SEQUENCE_ITEMS: Final[int] = 50
MAX_SESSION_REF_CHARS: Final[int] = 200


def parse_unit_telemetry(owner: str, stdout_text: str) -> dict[str, object]:
    """Return the token counts and session ref a spawned CLI reported, if any.

    Pure and total: no I/O, no clock, no environment, no network, and no
    exception path. Only the keys the stream actually reported appear in the
    result; a missing count is an absent key, never a zero.
    """
    owner_label = owner.strip()
    payload: dict[str, object] = {
        "schema_version": UNIT_TELEMETRY_SCHEMA_VERSION,
        "owner": owner_label,
        "parsed": False,
        "source": "none",
        "claim_boundary": UNIT_TELEMETRY_CLAIM_BOUNDARY,
    }
    source = _structured_source(owner_label)
    if not source or not stdout_text:
        return payload
    observed = _observed_values(_json_objects(stdout_text), source)
    if not observed:
        return payload
    payload["parsed"] = True
    payload["source"] = source
    for key in UNIT_TELEMETRY_VALUE_KEYS:
        if key in observed:
            payload[key] = observed[key]
    return payload


def native_unit_telemetry(owner: str, event: Mapping[str, object]) -> dict[str, object]:
    """Project only root terminal accounting; session identity has a separate decoder."""
    if event.get('parent_tool_use_id') is not None or event.get('type') != (
            'turn.completed' if owner == 'codex' else 'result'):
        return {}
    usage = event.get('usage')
    allowed = {name for _key, names in _TOKEN_FIELDS for name in names}
    safe_usage = ({key: value for key, value in usage.items()
                   if key in allowed and type(value) is int and 0 <= value <= 2**63 - 1}
                  if is_string_map(usage) else {})
    safe: dict[str, object] = {'usage': safe_usage}
    cost = event.get('total_cost_usd')
    if owner == 'claude-code' and type(cost) in (int, float):
        safe['total_cost_usd'] = cost
    result = parse_unit_telemetry(owner, json.dumps(safe))
    _ = result.pop('session_ref', None)
    return result


def _structured_source(owner: str) -> str:
    normalized = owner.casefold().replace("_", "-")
    return _STRUCTURED_SOURCE_BY_OWNER.get(normalized, "")


def _observed_values(objects: list[dict[str, Any]], source: str) -> dict[str, object]:
    """Fold parsed events into the observed values.

    Token counts: the LAST event carrying a token container wins, replacing the
    whole group. Codex emits cumulative counts repeatedly during a run, so a
    later reading supersedes an earlier one -- and replacing the group rather
    than merging it keeps input/output/total from being stitched together
    across two different readings.

    Session ref: the FIRST one observed wins. A session identifier is assigned
    once at stream start and does not change mid-run.
    """
    observed: dict[str, object] = {}
    session_keys = _SESSION_KEYS_BY_SOURCE.get(source, ())
    cost_key = _COST_KEY_BY_SOURCE.get(source, "")
    for item in objects:
        usage = _token_usage(item)
        if usage:
            for payload_key, _source_keys in _TOKEN_FIELDS:
                observed.pop(payload_key, None)
            observed.pop("tokens_billable", None)
            observed.pop("tokens_billable_source", None)
            observed.update(usage)
            if "tokens_billable" in usage:
                observed["tokens_billable_source"] = "summed_reported_components"
        if cost_key:
            cost = _reported_cost(item.get(cost_key))
            if cost is not None:
                observed["cost_usd"] = cost
        if "session_ref" not in observed:
            session_ref = _session_ref(item, session_keys)
            if session_ref:
                observed["session_ref"] = session_ref
    return observed


def _json_objects(stdout_text: str) -> list[dict[str, Any]]:
    """Parse stdout into JSON objects, ignoring every line that is not one.

    JSONL first (codex, and claude's stream-json), because that is what a
    dispatch tail normally holds. Only when no line parsed does this fall back
    to reading the whole text as one document, which is the pretty-printed
    single-object case for `claude -p --output-format json`.
    """
    objects: list[dict[str, Any]] = []
    for index, raw_line in enumerate(stdout_text.splitlines()):
        if index >= MAX_STDOUT_LINES:
            break
        line = raw_line.strip()
        if not line or len(line) > MAX_LINE_CHARS or not line.startswith("{"):
            continue
        item = _json_object(line)
        if item is not None:
            objects.append(item)
    if objects:
        return objects
    if len(stdout_text) > MAX_DOCUMENT_CHARS:
        return []
    item = _json_object(stdout_text)
    return [] if item is None else [item]


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        item = json.loads(text)
    except (ValueError, RecursionError):
        # Truncated mid-line output and interleaved plain-text logs are the
        # normal case, not an error; deeply nested input exhausts the decoder's
        # recursion budget rather than the caller's.
        return None
    return item if isinstance(item, dict) else None


def _token_usage(item: Mapping[str, Any]) -> dict[str, int]:
    containers: dict[str, dict[str, int]] = {}
    _collect_token_containers(item, containers, 0)
    for name in _TOKEN_CONTAINER_KEYS:
        found = containers.get(name)
        if found:
            return found
    return {}


def _collect_token_containers(value: Any, containers: dict[str, dict[str, int]], depth: int) -> None:
    if depth > MAX_OBJECT_DEPTH:
        return
    if isinstance(value, Mapping):
        for name in _TOKEN_CONTAINER_KEYS:
            nested = value.get(name)
            if name not in containers and isinstance(nested, Mapping):
                fields = _token_fields(nested)
                if fields:
                    containers[name] = fields
        for index, nested in enumerate(value.values()):
            if index >= MAX_MAPPING_KEYS:
                return
            _collect_token_containers(nested, containers, depth + 1)
        return
    if isinstance(value, list):
        for nested in value[:MAX_SEQUENCE_ITEMS]:
            _collect_token_containers(nested, containers, depth + 1)


def _token_fields(usage: Mapping[str, Any]) -> dict[str, int]:
    fields: dict[str, int] = {}
    for payload_key, source_keys in _TOKEN_FIELDS:
        for source_key in source_keys:
            count = _token_count(usage.get(source_key))
            if count is not None:
                fields[payload_key] = count
                break
    billable = [fields[key] for key in _BILLABLE_COMPONENT_KEYS if key in fields]
    if billable:
        fields["tokens_billable"] = sum(billable)
    return fields


def _token_count(value: Any) -> int | None:
    """Accept a reported non-negative integer count only.

    `bool` is excluded explicitly: it is an `int` subclass, and `"total_tokens":
    true` would otherwise be recorded as a run that used one token. Floats are
    rejected rather than rounded -- a rounded count is an estimate, and this
    module reports only what was stated.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _reported_cost(value: Any) -> float | None:
    """Accept a reported non-negative finite cost only -- never an estimate.

    `bool` is excluded for the same reason as `_token_count`: it is an `int`
    subclass and would otherwise read `"total_cost_usd": true` as a real
    number. This module sums or rounds only what a CLI already reported; it
    never derives a cost from token counts.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _session_ref(item: Mapping[str, Any], session_keys: tuple[str, ...]) -> str:
    if not session_keys:
        return ""
    refs: dict[str, str] = {}
    _collect_session_refs(item, session_keys, refs, 0)
    for key in session_keys:
        ref = refs.get(key)
        if ref:
            return ref
    return ""


def _collect_session_refs(value: Any, session_keys: tuple[str, ...], refs: dict[str, str], depth: int) -> None:
    if depth > MAX_OBJECT_DEPTH:
        return
    if isinstance(value, Mapping):
        for key in session_keys:
            if key not in refs:
                ref = _opaque_ref(value.get(key))
                if ref:
                    refs[key] = ref
        for index, nested in enumerate(value.values()):
            if index >= MAX_MAPPING_KEYS:
                return
            _collect_session_refs(nested, session_keys, refs, depth + 1)
        return
    if isinstance(value, list):
        for nested in value[:MAX_SEQUENCE_ITEMS]:
            _collect_session_refs(nested, session_keys, refs, depth + 1)


def _opaque_ref(value: Any) -> str:
    """Return a bounded, whitespace-free identifier, or "" when it is not one.

    An identifier is opaque and short. Anything carrying whitespace or an
    unprintable character is prose or a log fragment that happened to sit under
    a session key, and this module emits no free text.
    """
    if not isinstance(value, str):
        return ""
    ref = value.strip()
    if not ref or len(ref) > MAX_SESSION_REF_CHARS:
        return ""
    for char in ref:
        if char.isspace() or not char.isprintable():
            return ""
    return ref
