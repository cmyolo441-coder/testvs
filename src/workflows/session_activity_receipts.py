"""Host-supplied session activity receipts (`session_activity_receipt/v1`, issue #1404).

A session activity receipt is one bounded summary of one observed interval of
one Hermes session, supplied by the host or by an observer plugin that watched
it. OMH does not produce one. It validates what it is handed, records it once,
and lets the learning and operations workflows read the same record instead of
each scraping a transcript.

What the receipt separates, and why each split is a field of its own:

- Exposed versus activated. "The skill was not used" is ambiguous when nobody
  knows whether the host listed the skill in that session. `skills_exposed`
  and `skills_activated` are separate readings, and exposed-but-unused is
  derived only when both are present (`skill_exposure_summary`).
- Final versus partial. A turn boundary, a compaction, a snapshot, or the
  producer's own process exit is `final: false`. Only a `session_end` boundary
  may be final, so a mid-session receipt can never be reported as the
  session's terminal result.
- Observed versus heuristic versus unavailable. Every counter carries its own
  `availability`. A counter the producer did not observe is `unavailable` with
  a null value -- never zero, never inferred from prose -- and a counter the
  producer derived from text (a tool-error count read off output) is
  `heuristic`, which every consumer keeps apart from an observed one.
- Full session versus floor. A producer loaded mid-session saw only part of
  it. Its `observed_interval.coverage` says so, and under that coverage every
  observed counter must be a `floor`, because an exact total for an interval
  the producer did not fully watch is a claim it cannot make.

What the receipt never carries: prompts, tool arguments or results, secrets,
filesystem paths, or transcripts. Every string on it is either a closed
vocabulary value or an opaque reference, the key set is closed, the arrays are
bounded, and skill names can be folded to digests when they are themselves
sensitive. Nothing here crawls a host database, patches a Hermes hook, or
collects anything on its own: the only input is a payload someone chose to
hand over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    append_sidecar_line,
    append_store_line,
    digest_ref,
    record_fingerprint,
    redacted_ref,
    reference_errors,
)
from ..system.local_store import file_lock, read_jsonl_objects, utc_now
from ..system.metadata_safety import is_sensitive_metadata_text
from ..system.paths import OmhPaths
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


SESSION_ACTIVITY_RECEIPT_SCHEMA_VERSION = "session_activity_receipt/v1"
SESSION_ACTIVITY_ADMISSION_SCHEMA_VERSION = "session_activity_admission/v1"
SESSION_ACTIVITY_INGEST_RESULT_SCHEMA_VERSION = "session_activity_ingest_result/v1"
SESSION_ACTIVITY_QUARANTINE_SCHEMA_VERSION = "session_activity_quarantine/v1"
SESSION_ACTIVITY_EVIDENCE_SCHEMA_VERSION = "session_activity_evidence/v1"
SESSION_ACTIVITY_STORE_VALIDATION_SCHEMA_VERSION = "session_activity_receipt_store_validation/v1"

SESSION_ACTIVITY_RECEIPT_STORE_NAME = "session_activity_receipts.jsonl"
SESSION_ACTIVITY_QUARANTINE_STORE_NAME = "session_activity_quarantine.jsonl"

RECEIPT_PRIVACY = "metadata_only"
CLAIM_BOUNDARY = (
    "A session activity receipt is bounded metadata a host or observer supplied about one observed "
    "interval of one session. It is not automatic host execution, not proof of anything outside the "
    "interval it names, and never execution, verification, review, CI, merge-readiness, or merge evidence."
)

PRODUCER_KINDS = ("hermes_host", "observer_plugin", "wrapper", "operator")
# Only `session_end` may carry `final: true`. A process exit is the producer's
# boundary, not the session's: the session may go on without its observer.
BOUNDARY_KINDS = ("turn", "compaction", "snapshot", "process_exit", "session_end")
FINAL_BOUNDARY_KINDS = ("session_end",)
# How much of the session the producer watched. Anything but `full_session`
# is a floor: the producer arrived after the session started, or cannot say.
COVERAGE_KINDS = ("full_session", "from_producer_load", "unknown")
AVAILABILITIES = ("observed", "heuristic", "unavailable")
MEASUREMENTS = ("exact", "floor")
NAME_FORMS = ("plain", "digest")

METRIC_NAMES = (
    "skills_exposed",
    "skills_activated",
    "tool_calls",
    "tool_errors",
    "subagent_spawns",
    "subagent_completions",
    "compaction_boundaries",
    "model_calls",
    "model_errors",
    "tokens_input",
    "tokens_output",
    "tokens_total",
    "context_peak_tokens",
    "wall_clock_ms",
    "model_latency_ms",
)

# Which readings each named consumer workflow reads off the receipt. The
# receipt is one record; the consumer projections differ only in which
# counters they ask about, so a metric one consumer reads as observed can
# never be a metric another consumer reads as zero.
SESSION_ACTIVITY_CONSUMERS: dict[str, tuple[str, ...]] = {
    "workflow-learning": (
        "skills_exposed",
        "skills_activated",
        "tool_calls",
        "tool_errors",
        "compaction_boundaries",
    ),
    "context-budget-review": (
        "context_peak_tokens",
        "tokens_input",
        "tokens_output",
        "tokens_total",
        "compaction_boundaries",
    ),
    "run-efficiency": ("wall_clock_ms", "model_latency_ms", "tool_calls", "model_calls", "tokens_total"),
    "skill-health": ("skills_exposed", "skills_activated"),
    "achievements": ("skills_activated", "tool_calls", "subagent_spawns", "subagent_completions"),
    "agent-ops-review": (
        "tool_calls",
        "tool_errors",
        "subagent_spawns",
        "subagent_completions",
        "model_calls",
        "model_errors",
    ),
}

RECEIPT_KEYS = (
    "boundary",
    "claim_boundary",
    "evidence_refs",
    "metrics",
    "model_refs",
    "observed_at",
    "observed_interval",
    "privacy",
    "producer",
    "profile_ref",
    "receipt_id",
    "schema_version",
    "session_ref",
    "skills",
)
PRODUCER_KEYS = ("kind", "ref", "version")
BOUNDARY_KEYS = ("kind", "final", "sequence")
INTERVAL_KEYS = ("started_at", "ended_at", "coverage")
READING_KEYS = ("value", "availability", "measurement")
SKILLS_KEYS = (
    "exposed_names",
    "activated_names",
    "exposed_names_truncated",
    "activated_names_truncated",
    "name_form",
)

# Key names raw session material arrives under, on top of the shared list.
# The key set is closed anyway; naming these makes the refusal say why.
RAW_SESSION_KEYS = frozenset(
    {
        "tool_args",
        "tool_arguments",
        "tool_input",
        "tool_inputs",
        "tool_output",
        "tool_outputs",
        "tool_result",
        "tool_results",
        "messages",
        "prompts",
        "path",
        "paths",
        "cwd",
        "file_path",
        "file_paths",
    }
)

# A skill name is a catalog label, never a path: no slash, unlike the wider
# opaque-reference shape evidence handles use.
_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")

MAX_METRIC_VALUE = 10**12
MAX_SEQUENCE = 10**9
MAX_SKILL_NAMES = 64
MAX_MODEL_REFS = 16
MAX_EVIDENCE_REFS = 8
MAX_QUARANTINE_ERRORS = 8

# Everything that identifies *which observation* a receipt records. The
# ingest stamp is excluded: re-sending the same receipt must not become a
# second record because the clock moved.
_OBSERVATION_IDENTITY_KEYS = (
    "boundary",
    "evidence_refs",
    "metrics",
    "model_refs",
    "observed_interval",
    "producer",
    "profile_ref",
    "receipt_id",
    "session_ref",
    "skills",
)

ADMISSION_ACCEPTED = "accepted"
ADMISSION_REJECTED = "rejected"
ADMISSION_QUARANTINED = "quarantined"
ADMISSION_OUTCOMES = (ADMISSION_ACCEPTED, ADMISSION_REJECTED, ADMISSION_QUARANTINED)

REASON_INVALID = "invalid"
REASON_CROSS_PROFILE = "cross_profile"
REASON_STALE_AFTER_FINAL = "stale_after_final"
REASON_STALE_SEQUENCE = "stale_sequence"
REASON_STALE_AGE = "stale_age"
REASON_IDENTITY_CONFLICT = "identity_conflict"
QUARANTINE_REASONS = (
    REASON_CROSS_PROFILE,
    REASON_STALE_AFTER_FINAL,
    REASON_STALE_SEQUENCE,
    REASON_STALE_AGE,
    REASON_IDENTITY_CONFLICT,
)

_LABEL = "session_activity_receipt"


class SessionActivityReceiptError(ValueError):
    """Raised when a supplied payload cannot be admitted as a receipt."""


# --- normalization -------------------------------------------------------


def normalize_session_activity_receipt(payload: Mapping[str, Any], *, redact_skill_names: bool = False) -> dict[str, Any]:
    """Bound a supplied payload into the stored shape without retaining raw material.

    Bounding is not validation: this folds oversized arrays to their caps
    (marking the truncation), folds every skill name that is not an opaque
    identifier into a digest, and fills the fixed constants. Everything it
    cannot fold is left for `validate_session_activity_receipt` to refuse.
    A payload that is not a mapping is returned as an empty record so the
    validator reports the shape fault by name.
    """
    if not isinstance(payload, Mapping):
        return {}
    record: dict[str, Any] = {key: payload[key] for key in RECEIPT_KEYS if key in payload}
    for key in payload:
        if key not in RECEIPT_KEYS:
            record[key] = payload[key]
    record["schema_version"] = SESSION_ACTIVITY_RECEIPT_SCHEMA_VERSION
    record["privacy"] = RECEIPT_PRIVACY
    record["claim_boundary"] = CLAIM_BOUNDARY
    record["skills"] = _normalized_skills(payload.get("skills"), redact_skill_names=redact_skill_names)
    record["model_refs"] = _bounded_ref_list(payload.get("model_refs"), MAX_MODEL_REFS)
    record["evidence_refs"] = _bounded_ref_list(payload.get("evidence_refs"), MAX_EVIDENCE_REFS)
    if "observed_at" not in payload:
        record["observed_at"] = utc_now()
    return record


def _normalized_skills(value: Any, *, redact_skill_names: bool) -> Any:
    if value is None:
        return {
            "exposed_names": [],
            "activated_names": [],
            "exposed_names_truncated": False,
            "activated_names_truncated": False,
            "name_form": "plain",
        }
    if not isinstance(value, Mapping):
        return value
    skills: dict[str, Any] = dict(value)
    name_form = str(value.get("name_form", "plain") or "plain")
    if redact_skill_names:
        name_form = "digest"
    for field in ("exposed_names", "activated_names"):
        names, truncated = _bounded_names(value.get(field), digest=redact_skill_names)
        skills[field] = names
        skills[f"{field}_truncated"] = bool(value.get(f"{field}_truncated", False)) or truncated
    skills["name_form"] = name_form
    return skills


def _bounded_names(value: Any, *, digest: bool) -> tuple[Any, bool]:
    """Bounded, opaque skill names. Non-lists are returned as they came."""
    if value is None:
        return [], False
    if not isinstance(value, list):
        return value, False
    names: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        names.append(digest_ref(text) if digest else _skill_name_or_digest(text))
        if len(names) == MAX_SKILL_NAMES:
            break
    return names, len([item for item in value if str(item or "").strip()]) > len(names)


def _bounded_ref_list(value: Any, limit: int) -> Any:
    if value is None:
        return []
    if not isinstance(value, list):
        return value
    refs: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        refs.append(_opaque_or_digest(text))
        if len(refs) == limit:
            break
    return refs


def _opaque_or_digest(text: str) -> str:
    """The value when it is an opaque reference, a digest handle otherwise.

    A model name or an evidence handle that carries a link, a space, a secret
    shape, or a control character is not retained: the digest is stable
    enough to dedupe on and reveals nothing.
    """
    return redacted_ref(text, field=_LABEL)


def _skill_name_or_digest(text: str) -> str:
    """A skill name when it is one, a digest handle otherwise.

    A name carrying a path separator, whitespace, or a secret shape is not a
    catalog label and is never stored as typed.
    """
    if is_skill_name(text):
        return text
    return digest_ref(text)


def is_skill_name(value: Any) -> bool:
    return isinstance(value, str) and bool(_SKILL_NAME.fullmatch(value)) and not is_sensitive_metadata_text(value)


# --- validation ----------------------------------------------------------


def validate_session_activity_receipt(record: Any) -> list[str]:
    """Every reason a record is not a `session_activity_receipt/v1`.

    Strict on purpose: this runs on the normalized shape, so an unsafe value
    that reaches it was one the adapter could not bound, and a stored record
    that fails it was hand-edited.
    """
    if not isinstance(record, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    errors.extend(_key_set_errors(record, RECEIPT_KEYS, path=_LABEL))
    if record.get("schema_version") != SESSION_ACTIVITY_RECEIPT_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {SESSION_ACTIVITY_RECEIPT_SCHEMA_VERSION}")
    if record.get("privacy") != RECEIPT_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {RECEIPT_PRIVACY}")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the receipt boundary")
    for field in ("receipt_id", "profile_ref", "session_ref", "observed_at"):
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    errors.extend(_stamp_errors(record.get("observed_at"), field="observed_at"))
    errors.extend(_producer_errors(record.get("producer")))
    boundary_errors = _boundary_errors(record.get("boundary"))
    errors.extend(boundary_errors)
    interval_errors = _interval_errors(record.get("observed_interval"))
    errors.extend(interval_errors)
    coverage = _coverage_of(record)
    errors.extend(_metrics_errors(record.get("metrics"), coverage=coverage))
    errors.extend(_skills_errors(record.get("skills")))
    errors.extend(_ref_list_errors(record.get("model_refs"), field="model_refs", limit=MAX_MODEL_REFS))
    errors.extend(_ref_list_errors(record.get("evidence_refs"), field="evidence_refs", limit=MAX_EVIDENCE_REFS))
    return errors


def _key_set_errors(value: Mapping[str, Any], allowed: tuple[str, ...], *, path: str) -> list[str]:
    errors: list[str] = []
    raw = sorted(str(key) for key in value if str(key).lower() in RAW_OR_HIDDEN_KEYS or str(key).lower() in RAW_SESSION_KEYS)
    if raw:
        errors.append(f"{path} must not carry raw session material keys: {raw}")
    extra = sorted(str(key) for key in value if key not in allowed and str(key) not in raw)
    if extra:
        errors.append(f"{path} has unsupported keys: {extra}")
    missing = sorted(set(allowed) - set(value))
    if missing:
        errors.append(f"{path} is missing keys: {missing}")
    return errors


def _producer_errors(value: Any) -> list[str]:
    path = f"{_LABEL} producer"
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]
    errors = _key_set_errors(value, PRODUCER_KEYS, path=path)
    if value.get("kind") not in PRODUCER_KINDS:
        errors.append(f"{path} kind is unsupported: {value.get('kind')!r}")
    errors.extend(reference_errors(value.get("ref"), field="producer ref", label=_LABEL, required=True))
    errors.extend(reference_errors(value.get("version"), field="producer version", label=_LABEL, required=False))
    return errors


def _boundary_errors(value: Any) -> list[str]:
    path = f"{_LABEL} boundary"
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]
    errors = _key_set_errors(value, BOUNDARY_KEYS, path=path)
    kind = value.get("kind")
    if kind not in BOUNDARY_KINDS:
        errors.append(f"{path} kind is unsupported: {kind!r}")
    final = value.get("final")
    if not isinstance(final, bool):
        errors.append(f"{path} final must be a boolean")
    elif final and kind not in FINAL_BOUNDARY_KINDS:
        errors.append(f"{path} final is only allowed on a session_end boundary; a {kind} boundary is partial")
    sequence = value.get("sequence")
    if not _is_int(sequence) or sequence < 0 or sequence > MAX_SEQUENCE:
        errors.append(f"{path} sequence must be an integer from 0 to {MAX_SEQUENCE}")
    return errors


def _interval_errors(value: Any) -> list[str]:
    path = f"{_LABEL} observed_interval"
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]
    errors = _key_set_errors(value, INTERVAL_KEYS, path=path)
    if value.get("coverage") not in COVERAGE_KINDS:
        errors.append(f"{path} coverage is unsupported: {value.get('coverage')!r}")
    for field in ("started_at", "ended_at"):
        errors.extend(reference_errors(value.get(field), field=f"observed_interval {field}", label=_LABEL, required=True))
        errors.extend(_stamp_errors(value.get(field), field=f"observed_interval {field}"))
    started = parse_utc_stamp(value.get("started_at"))
    ended = parse_utc_stamp(value.get("ended_at"))
    if started is not None and ended is not None and ended < started:
        errors.append(f"{path} ended_at must not precede started_at")
    return errors


def _metrics_errors(value: Any, *, coverage: str) -> list[str]:
    path = f"{_LABEL} metrics"
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]
    errors: list[str] = []
    unknown = sorted(str(key) for key in value if key not in METRIC_NAMES)
    if unknown:
        errors.append(f"{path} has unsupported metric names: {unknown}")
    for name in METRIC_NAMES:
        if name in value:
            errors.extend(_reading_errors(value[name], path=f"{path} {name}", coverage=coverage))
    return errors


def _reading_errors(value: Any, *, path: str, coverage: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]
    errors = _key_set_errors(value, READING_KEYS, path=path)
    availability = value.get("availability")
    measurement = value.get("measurement")
    count = value.get("value")
    if availability not in AVAILABILITIES:
        errors.append(f"{path} availability is unsupported: {availability!r}")
        return errors
    if availability == "unavailable":
        if count is not None:
            errors.append(f"{path} is unavailable and must carry a null value, not {count!r}")
        if measurement != "":
            errors.append(f"{path} is unavailable and must carry an empty measurement")
        return errors
    if not _is_int(count) or count < 0 or count > MAX_METRIC_VALUE:
        errors.append(f"{path} value must be an integer from 0 to {MAX_METRIC_VALUE}")
    if measurement not in MEASUREMENTS:
        errors.append(f"{path} measurement is unsupported: {measurement!r}")
    elif measurement == "exact" and coverage != "full_session":
        errors.append(
            f"{path} cannot be exact when observed_interval coverage is {coverage}; "
            "a producer that did not watch the whole session reports a floor"
        )
    return errors


def _skills_errors(value: Any) -> list[str]:
    path = f"{_LABEL} skills"
    if not isinstance(value, Mapping):
        return [f"{path} must be an object"]
    errors = _key_set_errors(value, SKILLS_KEYS, path=path)
    if value.get("name_form") not in NAME_FORMS:
        errors.append(f"{path} name_form is unsupported: {value.get('name_form')!r}")
    for field in ("exposed_names", "activated_names"):
        errors.extend(_skill_name_list_errors(value.get(field), field=f"skills {field}"))
        if not isinstance(value.get(f"{field}_truncated"), bool):
            errors.append(f"{path} {field}_truncated must be a boolean")
    return errors


def _skill_name_list_errors(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        return [f"{_LABEL} {field} must be a list"]
    errors: list[str] = []
    if len(value) > MAX_SKILL_NAMES:
        errors.append(f"{_LABEL} {field} must have at most {MAX_SKILL_NAMES} items")
    for index, item in enumerate(value):
        if not is_skill_name(item):
            errors.append(f"{_LABEL} {field}[{index}] must be a skill name or a digest handle, not a path or secret")
    return errors


def _ref_list_errors(value: Any, *, field: str, limit: int) -> list[str]:
    if not isinstance(value, list):
        return [f"{_LABEL} {field} must be a list"]
    errors: list[str] = []
    if len(value) > limit:
        errors.append(f"{_LABEL} {field} must have at most {limit} items")
    for index, item in enumerate(value):
        errors.extend(reference_errors(item, field=f"{field}[{index}]", label=_LABEL, required=True))
    return errors


def _stamp_errors(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    if parse_utc_stamp(value) is None:
        return [f"{_LABEL} {field} must be an ISO-8601 UTC stamp ending in Z"]
    return []


def parse_utc_stamp(value: Any) -> datetime | None:
    """A `...Z` ISO-8601 stamp as an aware UTC datetime, or None."""
    text = str(value or "")
    if not text.endswith("Z"):
        return None
    try:
        parsed = strict_fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coverage_of(record: Mapping[str, Any]) -> str:
    interval = record.get("observed_interval")
    if not isinstance(interval, Mapping):
        return "unknown"
    coverage = interval.get("coverage")
    return coverage if coverage in COVERAGE_KINDS else "unknown"


# --- readings ------------------------------------------------------------


def metric_reading(receipt: Mapping[str, Any], name: str) -> dict[str, Any]:
    """One counter as the producer reported it, or an explicit `unavailable`.

    Absence is the honest reading, never zero: a receipt that omits a metric
    is a producer that did not observe it.
    """
    metrics = receipt.get("metrics")
    reading = metrics.get(name) if isinstance(metrics, Mapping) else None
    if not isinstance(reading, Mapping) or reading.get("availability") not in AVAILABILITIES:
        return {"value": None, "availability": "unavailable", "measurement": ""}
    if reading.get("availability") == "unavailable":
        return {"value": None, "availability": "unavailable", "measurement": ""}
    return {
        "value": reading.get("value"),
        "availability": str(reading.get("availability")),
        "measurement": str(reading.get("measurement", "")),
    }


def is_final_receipt(receipt: Mapping[str, Any]) -> bool:
    boundary = receipt.get("boundary")
    if not isinstance(boundary, Mapping):
        return False
    return boundary.get("final") is True and boundary.get("kind") in FINAL_BOUNDARY_KINDS


def boundary_sequence(receipt: Mapping[str, Any]) -> int:
    boundary = receipt.get("boundary")
    if not isinstance(boundary, Mapping):
        return -1
    sequence = boundary.get("sequence")
    return sequence if _is_int(sequence) else -1


def skill_exposure_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Exposed-but-unused, derived only when both sides were observed.

    Names win over counters when both lists are complete: the difference of
    two complete name lists is the actual set. Counters are the fallback when
    both were observed but a list is missing or truncated. Anything less is
    `unavailable`, with the reason named, because "not used" is not a finding
    when nobody saw whether it was offered.
    """
    exposed = metric_reading(receipt, "skills_exposed")
    activated = metric_reading(receipt, "skills_activated")
    skills = receipt.get("skills") if isinstance(receipt.get("skills"), Mapping) else {}
    exposed_names = [str(name) for name in skills.get("exposed_names", []) if isinstance(name, str)]
    activated_names = [str(name) for name in skills.get("activated_names", []) if isinstance(name, str)]
    names_complete = (
        bool(exposed_names)
        and skills.get("exposed_names_truncated") is False
        and skills.get("activated_names_truncated") is False
        and activated["availability"] != "unavailable"
    )
    summary: dict[str, Any] = {
        "skills_exposed": exposed,
        "skills_activated": activated,
        "exposed_unused_count": None,
        "exposed_unused_names": [],
        "derivation": "unavailable",
        "reason": "",
    }
    if exposed["availability"] == "unavailable" or activated["availability"] == "unavailable":
        summary["reason"] = "exposure and activation must both be observed before exposed-but-unused can be derived"
        return summary
    if names_complete:
        unused = [name for name in exposed_names if name not in set(activated_names)]
        summary["exposed_unused_count"] = len(unused)
        summary["exposed_unused_names"] = unused
        summary["derivation"] = "names"
        return summary
    difference = int(exposed["value"]) - int(activated["value"])
    if difference < 0:
        summary["reason"] = "activated count exceeds exposed count; the producer's readings disagree"
        return summary
    summary["exposed_unused_count"] = difference
    summary["derivation"] = "counters"
    if exposed["measurement"] == "floor" or activated["measurement"] == "floor":
        summary["reason"] = "derived from floor readings; the true count may be higher"
    return summary


# --- admission -----------------------------------------------------------


def admit_session_activity_receipt(
    payload: Mapping[str, Any],
    *,
    expected_profile_ref: str = "",
    known: Sequence[Mapping[str, Any]] = (),
    now: str = "",
    max_age_seconds: int | None = None,
    redact_skill_names: bool = False,
) -> dict[str, Any]:
    """Validate one supplied payload against the contract and the records already held.

    Returns a `session_activity_admission/v1` mapping whose `outcome` is
    `accepted`, `rejected` (the payload is not a receipt; `errors` says why),
    or `quarantined` (it is a receipt, but not one this store may take:
    `reason` names the cross-profile, staleness, or identity fault). Only an
    accepted admission carries a `receipt`.

    Pure: no file is read or written here. `known` is whatever the caller
    already holds for this store, and `now` is the caller's clock.
    """
    receipt = normalize_session_activity_receipt(payload, redact_skill_names=redact_skill_names)
    errors = validate_session_activity_receipt(receipt)
    if errors:
        return _admission(ADMISSION_REJECTED, REASON_INVALID, errors, None)
    reason = _quarantine_reason(
        receipt,
        expected_profile_ref=expected_profile_ref,
        known=known,
        now=now,
        max_age_seconds=max_age_seconds,
    )
    if reason:
        return _admission(ADMISSION_QUARANTINED, reason, [_quarantine_message(reason)], receipt)
    return _admission(ADMISSION_ACCEPTED, "", [], receipt)


def _admission(outcome: str, reason: str, errors: list[str], receipt: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": SESSION_ACTIVITY_ADMISSION_SCHEMA_VERSION,
        "outcome": outcome,
        "reason": reason,
        "errors": list(errors),
        "receipt": receipt if outcome != ADMISSION_REJECTED else None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _quarantine_reason(
    receipt: Mapping[str, Any],
    *,
    expected_profile_ref: str,
    known: Sequence[Mapping[str, Any]],
    now: str,
    max_age_seconds: int | None,
) -> str:
    if expected_profile_ref and str(receipt.get("profile_ref", "")) != expected_profile_ref:
        return REASON_CROSS_PROFILE
    same_id = [record for record in known if str(record.get("receipt_id", "")) == str(receipt.get("receipt_id", ""))]
    if same_id:
        # An identical re-send is the caller's idempotent path, not a fault;
        # only a different observation under a reused identity is quarantined.
        if all(_observation_fingerprint(record) == _observation_fingerprint(receipt) for record in same_id):
            return ""
        return REASON_IDENTITY_CONFLICT
    session_records = [
        record for record in known if str(record.get("session_ref", "")) == str(receipt.get("session_ref", ""))
    ]
    if any(is_final_receipt(record) for record in session_records):
        return REASON_STALE_AFTER_FINAL
    latest_sequence = max((boundary_sequence(record) for record in session_records), default=-1)
    if boundary_sequence(receipt) < latest_sequence:
        return REASON_STALE_SEQUENCE
    if max_age_seconds is not None and _older_than(receipt, now=now, max_age_seconds=max_age_seconds):
        return REASON_STALE_AGE
    return ""


def _older_than(receipt: Mapping[str, Any], *, now: str, max_age_seconds: int) -> bool:
    interval = receipt.get("observed_interval")
    ended = parse_utc_stamp(interval.get("ended_at")) if isinstance(interval, Mapping) else None
    reference = parse_utc_stamp(now or utc_now())
    if ended is None or reference is None:
        return False
    return (reference - ended).total_seconds() > max_age_seconds


def _quarantine_message(reason: str) -> str:
    messages = {
        REASON_CROSS_PROFILE: "receipt names a profile other than the one this store is bound to",
        REASON_STALE_AFTER_FINAL: "a final receipt for this session is already on record",
        REASON_STALE_SEQUENCE: "a later boundary for this session is already on record",
        REASON_STALE_AGE: "the observed interval ended before the accepted freshness window",
        REASON_IDENTITY_CONFLICT: "receipt_id is already on record with a different observation",
    }
    return messages.get(reason, reason)


# --- store ---------------------------------------------------------------


def session_activity_quarantine_path(store_path: Path) -> Path:
    return store_path.with_name(SESSION_ACTIVITY_QUARANTINE_STORE_NAME)


def ingest_session_activity_receipt(
    paths: OmhPaths,
    payload: Mapping[str, Any],
    *,
    expected_profile_ref: str = "",
    now: str = "",
    max_age_seconds: int | None = None,
    redact_skill_names: bool = False,
) -> dict[str, Any]:
    """Admit one supplied payload against the store and record the outcome.

    Returns a `session_activity_ingest_result/v1` mapping:

        outcome     -- "recorded" | "already_recorded" | "rejected" | "quarantined"
        reason      -- "" or the admission reason
        errors      -- the admission errors
        receipt_id  -- the identity now on record, "" when nothing was recorded
        receipt     -- the record on file for accepted or duplicate payloads

    `already_recorded` is what makes ingestion idempotent by receipt identity:
    the same receipt sent three times is one line. A quarantined payload is
    logged to the quarantine sidecar as bounded metadata (identity handles,
    reason, errors) and never reaches the store. A rejected payload reaches
    nothing on disk.

    The read of the prior records and the append happen under one lock, so
    two concurrent sends of one receipt cannot both record it.
    """
    store_path = paths.runtime_session_activity_receipts_path
    with file_lock(store_path, private=True):
        known, _ = read_jsonl_objects(store_path)
        admission = admit_session_activity_receipt(
            payload,
            expected_profile_ref=expected_profile_ref,
            known=known,
            now=now,
            max_age_seconds=max_age_seconds,
            redact_skill_names=redact_skill_names,
        )
        if admission["outcome"] == ADMISSION_REJECTED:
            return _ingest_result("rejected", admission, receipt=None)
        receipt = admission["receipt"]
        if admission["outcome"] == ADMISSION_QUARANTINED:
            _append_quarantine(store_path, receipt, admission)
            return _ingest_result("quarantined", admission, receipt=None)
        prior = _same_receipt_in(known, str(receipt["receipt_id"]))
        if prior:
            return _ingest_result("already_recorded", admission, receipt=prior)
        append_store_line(store_path, receipt)
    return _ingest_result("recorded", admission, receipt=receipt)


def _ingest_result(outcome: str, admission: Mapping[str, Any], *, receipt: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": SESSION_ACTIVITY_INGEST_RESULT_SCHEMA_VERSION,
        "outcome": outcome,
        "reason": str(admission.get("reason", "")),
        "errors": list(admission.get("errors", [])),
        "receipt_id": str(receipt.get("receipt_id", "")) if receipt else "",
        "receipt": dict(receipt) if receipt else None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _append_quarantine(store_path: Path, receipt: Mapping[str, Any], admission: Mapping[str, Any]) -> None:
    append_sidecar_line(
        session_activity_quarantine_path(store_path),
        {
            "schema_version": SESSION_ACTIVITY_QUARANTINE_SCHEMA_VERSION,
            "quarantined_at": utc_now(),
            "reason": str(admission.get("reason", "")),
            "errors": list(admission.get("errors", []))[:MAX_QUARANTINE_ERRORS],
            "receipt_id": redacted_ref(str(receipt.get("receipt_id", "")), field="receipt_id"),
            "session_ref": redacted_ref(str(receipt.get("session_ref", "")), field="session_ref"),
            "profile_ref": redacted_ref(str(receipt.get("profile_ref", "")), field="profile_ref"),
            "claim_boundary": CLAIM_BOUNDARY,
        },
    )


def _same_receipt_in(records: Sequence[Mapping[str, Any]], receipt_id: str) -> dict[str, Any]:
    matches = [dict(record) for record in records if str(record.get("receipt_id", "")) == receipt_id]
    return matches[-1] if matches else {}


def read_session_activity_receipts(
    paths: OmhPaths,
    *,
    session_ref: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    receipts, _ = read_jsonl_objects(paths.runtime_session_activity_receipts_path)
    if session_ref is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("session_ref", "")) == session_ref]
    if limit is None:
        return receipts
    if limit < 1:
        return []
    return receipts[-limit:]


def read_session_activity_quarantine(paths: OmhPaths) -> list[dict[str, Any]]:
    records, _ = read_jsonl_objects(session_activity_quarantine_path(paths.runtime_session_activity_receipts_path))
    return records


def latest_session_activity_receipt(
    receipts: Sequence[Mapping[str, Any]],
    session_ref: str,
) -> dict[str, Any]:
    """The receipt that speaks for one session: highest boundary sequence, latest on a tie."""
    matches = [dict(receipt) for receipt in receipts if str(receipt.get("session_ref", "")) == str(session_ref)]
    if not matches:
        return {}
    best = matches[0]
    for receipt in matches[1:]:
        if boundary_sequence(receipt) >= boundary_sequence(best):
            best = receipt
    return best


def validate_session_activity_receipt_store(path: Path) -> dict[str, Any]:
    receipts, read_errors = read_jsonl_objects(path)
    errors = list(read_errors)
    seen: dict[str, str] = {}
    for index, receipt in enumerate(receipts, start=1):
        errors.extend(f"{path}:{index}: {error}" for error in validate_session_activity_receipt(receipt))
        receipt_id = str(receipt.get("receipt_id", ""))
        fingerprint = _observation_fingerprint(receipt)
        if receipt_id in seen and seen[receipt_id] != fingerprint:
            errors.append(f"{path}:{index}: {_LABEL} receipt_id is on record with a different observation: {receipt_id}")
        seen.setdefault(receipt_id, fingerprint)
    return {
        "schema_version": SESSION_ACTIVITY_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "ok": not errors,
        "receipt_count": len(receipts),
        "errors": errors,
    }


def _observation_fingerprint(receipt: Mapping[str, Any]) -> str:
    return record_fingerprint(receipt, _OBSERVATION_IDENTITY_KEYS)


# --- consumers -----------------------------------------------------------


def session_activity_evidence(receipt: Mapping[str, Any], consumer: str) -> dict[str, Any]:
    """One consumer workflow's view of one receipt, with every limit preserved.

    Every consumer reads through this projection, so none can read an
    unavailable counter as zero, report a partial boundary as a session
    result, or present a floor as a total. `consumer` must be one of
    `SESSION_ACTIVITY_CONSUMERS`.
    """
    if consumer not in SESSION_ACTIVITY_CONSUMERS:
        raise SessionActivityReceiptError(f"session activity consumer is unsupported: {consumer!r}")
    errors = validate_session_activity_receipt(receipt)
    if errors:
        raise SessionActivityReceiptError(errors[0])
    names = SESSION_ACTIVITY_CONSUMERS[consumer]
    readings = {name: metric_reading(receipt, name) for name in names}
    final = is_final_receipt(receipt)
    coverage = _coverage_of(receipt)
    boundary = receipt["boundary"]
    evidence: dict[str, Any] = {
        "schema_version": SESSION_ACTIVITY_EVIDENCE_SCHEMA_VERSION,
        "consumer": consumer,
        "receipt_id": str(receipt["receipt_id"]),
        "session_ref": str(receipt["session_ref"]),
        "profile_ref": str(receipt["profile_ref"]),
        "producer_kind": str(receipt["producer"]["kind"]),
        "boundary_kind": str(boundary["kind"]),
        "boundary_sequence": int(boundary["sequence"]),
        "session_outcome": "final" if final else "partial",
        "terminal": final,
        "coverage": coverage,
        "observation_floor": coverage != "full_session",
        "metrics": readings,
        "observed_metrics": [name for name in names if readings[name]["availability"] == "observed"],
        "heuristic_metrics": [name for name in names if readings[name]["availability"] == "heuristic"],
        "unavailable_metrics": [name for name in names if readings[name]["availability"] == "unavailable"],
        "evidence_refs": list(receipt["evidence_refs"]),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if "skills_exposed" in names:
        evidence["skill_exposure"] = skill_exposure_summary(receipt)
    return evidence


def attach_session_activity_evidence(
    payload: dict[str, Any],
    receipt: Mapping[str, Any] | None,
    consumer: str,
) -> dict[str, Any]:
    """Add a receipt's evidence to a consumer payload under one fixed key.

    With no receipt the key still lands, as `None`: a consumer that renders
    the key can then say the receipt was not supplied instead of saying
    nothing happened.
    """
    payload["session_activity"] = session_activity_evidence(receipt, consumer) if receipt else None
    return payload
