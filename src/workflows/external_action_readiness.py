"""One truthful answer to "can you do this now?" for one requested outcome.

The defect this closes (#809): OMH could say a connector was installed, that a
host had loaded it, that a probe had reached it, and that an effect had been
recorded through it, but nothing turned those four separate facts into a single
answer to the only question a person asks. Configuration read like capability,
a load read like reachability, and a success from last week read like a success
now.

Two axes, kept apart on purpose, because welding them is the confusion:

* **Evidence tier** -- what class of fact is the strongest thing anyone
  recorded. ``installed`` is a local configuration fact. ``host_observed``
  means a host reported loading the surface. ``usable_observed`` means a host
  reported actually using it. ``used`` means an external-effect receipt records
  a use of *this* outcome. ``stale`` means one of those was true and is now past
  its freshness horizon.
* **State** -- the answer itself: ``ready``, ``blocked``, ``not_observed``,
  ``stale``, ``failed``, with the smallest next action attached to every state
  that is not ``ready``.

Configuration can never reach ``ready`` and the guard is mechanical, not
advisory: :data:`SOURCE_TIERS` says which tiers a source is allowed to claim,
``local_configuration`` may claim only ``installed``, and ``installed`` sits
below the tier level a positive answer requires.

Scoping is by requested OUTCOME, not by connector. Two outcomes over the same
connector routinely have different answers -- the surface may be reachable
while one of the two effects has never succeeded through it -- so surface-level
evidence applies to every outcome that names the surface, and outcome-level
evidence applies only to the outcome it names.

Freshness is derived at READ time from :data:`EXTERNAL_ACTION_STALE_AFTER_SECONDS`,
the way ``coding.action_gate._account_state`` and ``coding.pre_handoff_readiness``
derive theirs. No expiry deadline is written into a record, so shortening the
horizon takes effect on evidence already recorded. ``now`` is a parameter so the
derivation is deterministic under test, and no wall clock is ever placed inside
a compared payload.

Everything here is pure. OMH probes nothing, calls nothing, and reaches no
network: it reads what other surfaces recorded. The store adapters below convert
records those surfaces already write; ``omh_mcp_observation/v1`` and
``omh_evidence_probe/v1`` are deliberately NOT adapted, because neither names a
host or an effect and therefore neither can be scoped to a requested outcome.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    is_unsafe_metadata_line,
    opaque_ref,
    reference_errors,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


EXTERNAL_ACTION_READINESS_SCHEMA_VERSION: Final = "external_action_readiness/v1"
EXTERNAL_ACTION_EVIDENCE_SCHEMA_VERSION: Final = "external_action_evidence/v1"

# Six hours, the horizon `coding.pre_handoff_readiness.READINESS_STALE_AFTER_SECONDS`
# and `coding.action_gate.ACCOUNT_SIGNAL_STALE_AFTER_SECONDS` already give a
# question of this shape -- "is the environment still the one that was
# observed". A second, different number would only make two surfaces disagree
# about the same machine.
EXTERNAL_ACTION_STALE_AFTER_SECONDS: Final = 6 * 60 * 60

# The four tiers a record can assert, weakest to strongest.
EVIDENCE_TIERS: Final = ("installed", "host_observed", "usable_observed", "used")
# The derived fifth. It is never recorded; it is what the strongest tier reads
# as once nothing that was recorded is still fresh.
EVIDENCE_TIER_STALE: Final = "stale"
# The absence of any evidence. Not one of AC2's five: it is what there is when
# nobody recorded anything at all.
EVIDENCE_TIER_NONE: Final = "none"
# The five an answer must tell apart (AC2), plus the empty case.
DISTINGUISHED_EVIDENCE_TIERS: Final = (*EVIDENCE_TIERS, EVIDENCE_TIER_STALE)
REPORTED_EVIDENCE_TIERS: Final = (EVIDENCE_TIER_NONE, *DISTINGUISHED_EVIDENCE_TIERS)

_TIER_LEVELS: Final = {tier: level for level, tier in enumerate(EVIDENCE_TIERS, start=1)}
# The lowest tier that can answer "yes, now". Below it a record describes setup
# or presence, not the surface doing the thing.
_READY_TIER_LEVEL: Final = _TIER_LEVELS["usable_observed"]

# What one record asserts about its tier.
#   observed -- the tier's fact holds.
#   blocked  -- the recording surface reported something in the way.
#   failed   -- an attempt at the outcome was observed not to happen.
EVIDENCE_RESULTS: Final = ("observed", "blocked", "failed")
_NEGATIVE_RESULTS: Final = ("failed", "blocked")

# Where a record came from. Closed, because the tier a source may claim is the
# mechanism behind AC1 and a free-text source would route around it.
EVIDENCE_SOURCES: Final = (
    "local_configuration",
    "plugin_host_observation",
    "mcp_host_session",
    "external_effect_receipt",
)

# Which tiers each source is allowed to claim. `local_configuration` may claim
# only `installed`, which is what makes "configuration alone never produces
# ready now" a property of the schema rather than a rule someone remembers.
SOURCE_TIERS: Final = {
    "local_configuration": ("installed",),
    "plugin_host_observation": ("installed", "host_observed", "usable_observed"),
    "mcp_host_session": ("installed", "host_observed", "usable_observed"),
    "external_effect_receipt": ("used",),
}

EXTERNAL_ACTION_READINESS_STATES: Final = ("ready", "blocked", "not_observed", "stale", "failed")

# Where the reported state came from. `preserved_prior` is AC3: new evidence
# arrived unreadable and the last valid answer was kept rather than overwritten.
STATE_SOURCES: Final = ("derived", "preserved_prior", "no_valid_evidence")
# How much of the supplied evidence survived validation.
EVIDENCE_INTEGRITIES: Final = ("clean", "partial", "unusable")

EXTERNAL_ACTION_EVIDENCE_KEYS: Final = (
    "schema_version",
    "tier",
    "source",
    "result",
    "surface",
    "outcome_id",
    "observed_at",
    "evidence_ref",
)

EXTERNAL_ACTION_READINESS_KEYS: Final = (
    "schema_version",
    "outcome_id",
    "action",
    "surface",
    "state",
    "evidence_tier",
    "state_source",
    "evidence_integrity",
    "reason",
    "next_action",
    "next_step",
    "accepted_count",
    "rejected_count",
    "rejected_evidence",
    "age_seconds",
    "stale_after_seconds",
    "claim_boundary",
)
_REJECTED_EVIDENCE_KEYS: Final = ("index", "reason")

EXTERNAL_ACTION_READINESS_CLAIM_BOUNDARY: Final = (
    "This answer reports only what other surfaces recorded about one requested outcome. OMH ran no "
    "probe, called no connector, and performed no external action; a ready answer says the surface "
    "was observed working recently, not that this action will succeed, and no state here is "
    "execution, verification, review, CI, or merge evidence."
)

NEXT_ACTIONS: Final = {
    "ready": "perform_the_requested_action",
    "blocked": "clear_the_recorded_blocker",
    "not_observed": "observe_the_surface_before_acting",
    "stale": "re_observe_the_surface_before_acting",
    "failed": "read_the_recorded_failure_before_retrying",
}

UNKNOWN_AGE_SECONDS: Final = -1

_MAX_ACTION_CHARS: Final = 160
_MAX_REASON_CHARS: Final = 400
_MAX_STEP_CHARS: Final = 240
_MAX_REJECTED_ROWS: Final = 8
_MAX_REJECTION_REASON_CHARS: Final = 200

# The names each fault is reported under, so an error line says which of the two
# record families it came from.
_LABEL: Final = "external action evidence"
_ANSWER_LABEL: Final = "external action readiness answer"


class ExternalActionReadinessError(ValueError):
    """Raised when an outcome or a piece of evidence cannot be accepted."""


def build_external_action_evidence(
    *,
    tier: str,
    source: str,
    result: str,
    surface: str,
    observed_at: str,
    outcome_id: str = "",
    evidence_ref: str = "",
) -> dict[str, Any]:
    """Mint one evidence record, or refuse.

    `outcome_id` is empty for surface-level evidence -- the surface is
    installed, a host loaded it, a host used it -- and named for evidence about
    one effect, which only the `used` tier can be. Refusing an unnamed `used`
    record is what keeps a success on one outcome from answering for another.
    """
    if source not in EVIDENCE_SOURCES:
        raise ExternalActionReadinessError(f"external action evidence source is unsupported: {source!r}")
    if tier not in EVIDENCE_TIERS:
        raise ExternalActionReadinessError(f"external action evidence tier is unsupported: {tier!r}")
    if tier not in SOURCE_TIERS[source]:
        raise ExternalActionReadinessError(
            f"external action evidence source {source} may not claim tier {tier}"
        )
    if result not in EVIDENCE_RESULTS:
        raise ExternalActionReadinessError(f"external action evidence result is unsupported: {result!r}")
    if tier == "used" and not str(outcome_id or "").strip():
        raise ExternalActionReadinessError(
            "external action evidence at the used tier must name the outcome it observed"
        )
    record = {
        "schema_version": EXTERNAL_ACTION_EVIDENCE_SCHEMA_VERSION,
        "tier": tier,
        "source": source,
        "result": result,
        "surface": _opaque(surface, field="external action evidence surface"),
        "outcome_id": _opaque(outcome_id, field="external action evidence outcome_id") if outcome_id else "",
        "observed_at": _opaque(observed_at, field="external action evidence observed_at"),
        "evidence_ref": _opaque(evidence_ref, field="external action evidence evidence_ref") if evidence_ref else "",
    }
    errors = external_action_evidence_errors(record)
    if errors:
        raise ExternalActionReadinessError(errors[0])
    return record


def external_action_evidence_errors(record: Any) -> list[str]:
    """Every reason one record is not usable evidence. Both directions on keys."""
    if not isinstance(record, Mapping):
        return ["external action evidence must be an object"]
    errors: list[str] = []
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"external action evidence must not carry raw or hidden keys: {forbidden}")
    present = {str(key) for key in record}
    missing = sorted(set(EXTERNAL_ACTION_EVIDENCE_KEYS) - present)
    if missing:
        errors.append(f"external action evidence is missing keys: {missing}")
    unexpected = sorted(present - set(EXTERNAL_ACTION_EVIDENCE_KEYS) - set(forbidden))
    if unexpected:
        errors.append(f"external action evidence has unsupported keys: {unexpected}")
    if record.get("schema_version") != EXTERNAL_ACTION_EVIDENCE_SCHEMA_VERSION:
        errors.append(f"external action evidence schema_version must be {EXTERNAL_ACTION_EVIDENCE_SCHEMA_VERSION}")
    source = record.get("source")
    if source not in EVIDENCE_SOURCES:
        errors.append(f"external action evidence source is unsupported: {source!r}")
    tier = record.get("tier")
    if tier not in EVIDENCE_TIERS:
        errors.append(f"external action evidence tier is unsupported: {tier!r}")
    elif source in EVIDENCE_SOURCES and tier not in SOURCE_TIERS[str(source)]:
        errors.append(f"external action evidence source {source} may not claim tier {tier}")
    if record.get("result") not in EVIDENCE_RESULTS:
        errors.append(f"external action evidence result is unsupported: {record.get('result')!r}")
    if tier == "used" and not str(record.get("outcome_id", "") or ""):
        errors.append("external action evidence at the used tier must name the outcome it observed")
    errors.extend(reference_errors(record.get("surface"), field="surface", label=_LABEL, required=True))
    errors.extend(reference_errors(record.get("observed_at"), field="observed_at", label=_LABEL, required=True))
    errors.extend(reference_errors(record.get("outcome_id"), field="outcome_id", label=_LABEL, required=False))
    errors.extend(reference_errors(record.get("evidence_ref"), field="evidence_ref", label=_LABEL, required=False))
    return errors


def answer_external_action_readiness(
    *,
    outcome_id: str,
    action: str,
    surface: str,
    evidence: Sequence[Mapping[str, Any]] = (),
    prior: Mapping[str, Any] | None = None,
    now: str = "",
) -> dict[str, Any]:
    """The one answer to "can you do this now?" for one requested outcome.

    `evidence` is every record a caller gathered. Three things happen to it, in
    this order, and the order is the point:

    1. Each record is validated. An invalid record is rejected and never enters
       the derivation, so bad evidence structurally cannot raise the answer.
    2. Valid records are scoped. Evidence naming another outcome is ignored --
       it is somebody else's truth, not weaker truth about this one -- and
       surface-level evidence applies only to the named surface.
    3. The answer is derived from the strongest positive evidence that is still
       fresh. When nothing in scope is fresh, the answer is `stale`; a stale
       observation is never allowed to read as `ready`.

    `prior` is the last valid answer for this same outcome. It is used for
    exactly one thing (AC3): when records were supplied and none of them
    survived validation and scoping, the prior state is preserved rather than
    overwritten, and the answer says so through `state_source` and
    `rejected_evidence`. A preserved answer never rises above what the prior
    already held, because a prior is itself a derived answer.

    `now` is a parameter so the derivation is deterministic. An unreadable or
    future `observed_at` reads as older than the horizon, for the reason
    `pre_handoff_readiness._stamp_age_seconds` gives: neither can be shown to be
    fresh, and clamping would turn editing a stored timestamp into a way to
    widen the window.
    """
    safe_outcome_id = _opaque(outcome_id, field="external action outcome_id")
    safe_surface = _opaque(surface, field="external action surface")
    safe_action = _bounded_action(action)

    supplied = list(evidence)
    scoped: list[Mapping[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_count = 0
    for index, record in enumerate(supplied):
        errors = external_action_evidence_errors(record)
        if errors:
            rejected_count += 1
            if len(rejected) < _MAX_REJECTED_ROWS:
                rejected.append({"index": index, "reason": errors[0][:_MAX_REJECTION_REASON_CHARS]})
            continue
        if _applies_to(record, outcome_id=safe_outcome_id, surface=safe_surface):
            scoped.append(record)

    integrity = _integrity(supplied=len(supplied), rejected=rejected_count)
    preserved = _preserved_prior(prior, outcome_id=safe_outcome_id) if rejected_count and not scoped else None

    if preserved is not None:
        state = str(preserved["state"])
        tier = str(preserved["evidence_tier"])
        state_source = "preserved_prior"
        age = UNKNOWN_AGE_SECONDS
    elif rejected_count and not scoped:
        state, tier, state_source, age = "not_observed", EVIDENCE_TIER_NONE, "no_valid_evidence", UNKNOWN_AGE_SECONDS
    else:
        state, tier, age = _derive(scoped, now=now)
        state_source = "derived"

    answer = {
        "schema_version": EXTERNAL_ACTION_READINESS_SCHEMA_VERSION,
        "outcome_id": safe_outcome_id,
        "action": safe_action,
        "surface": safe_surface,
        "state": state,
        "evidence_tier": tier,
        "state_source": state_source,
        "evidence_integrity": integrity,
        "reason": _reason(
            state=state,
            tier=tier,
            state_source=state_source,
            action=safe_action,
            surface=safe_surface,
            rejected_count=rejected_count,
        ),
        "next_action": NEXT_ACTIONS[state],
        "next_step": _next_step(state, surface=safe_surface),
        "accepted_count": len(scoped),
        "rejected_count": rejected_count,
        "rejected_evidence": rejected,
        "age_seconds": age,
        "stale_after_seconds": EXTERNAL_ACTION_STALE_AFTER_SECONDS,
        "claim_boundary": EXTERNAL_ACTION_READINESS_CLAIM_BOUNDARY,
    }
    errors = validate_external_action_readiness_answer(answer)
    if errors:
        raise ExternalActionReadinessError("; ".join(errors))
    return answer


def validate_external_action_readiness_answer(answer: Any) -> list[str]:
    """Both directions: no key of the closed set missing, no key outside it."""
    if not isinstance(answer, Mapping):
        return ["external action readiness answer must be an object"]
    errors: list[str] = []
    present = {str(key) for key in answer}
    missing = sorted(set(EXTERNAL_ACTION_READINESS_KEYS) - present)
    if missing:
        errors.append(f"external action readiness answer is missing keys: {missing}")
    unexpected = sorted(present - set(EXTERNAL_ACTION_READINESS_KEYS))
    if unexpected:
        errors.append(f"external action readiness answer has unsupported keys: {unexpected}")
    if answer.get("schema_version") != EXTERNAL_ACTION_READINESS_SCHEMA_VERSION:
        errors.append(f"schema_version must be {EXTERNAL_ACTION_READINESS_SCHEMA_VERSION}")
    state = answer.get("state")
    if state not in EXTERNAL_ACTION_READINESS_STATES:
        errors.append(f"state is unsupported: {state!r}")
    elif answer.get("next_action") != NEXT_ACTIONS[str(state)]:
        errors.append("next_action must be the action this state calls for")
    if answer.get("evidence_tier") not in REPORTED_EVIDENCE_TIERS:
        errors.append(f"evidence_tier is unsupported: {answer.get('evidence_tier')!r}")
    if answer.get("state_source") not in STATE_SOURCES:
        errors.append(f"state_source is unsupported: {answer.get('state_source')!r}")
    if answer.get("evidence_integrity") not in EVIDENCE_INTEGRITIES:
        errors.append(f"evidence_integrity is unsupported: {answer.get('evidence_integrity')!r}")
    if answer.get("stale_after_seconds") != EXTERNAL_ACTION_STALE_AFTER_SECONDS:
        errors.append(f"stale_after_seconds must be {EXTERNAL_ACTION_STALE_AFTER_SECONDS}")
    if answer.get("claim_boundary") != EXTERNAL_ACTION_READINESS_CLAIM_BOUNDARY:
        errors.append("claim_boundary must state the external action readiness boundary")
    # The invariant AC1 rests on, checked on the finished answer as well as on
    # the way in: no tier below "usable now" may be reported as ready.
    if state == "ready" and _TIER_LEVELS.get(str(answer.get("evidence_tier")), 0) < _READY_TIER_LEVEL:
        errors.append("ready requires fresh usable_observed or used evidence")
    errors.extend(_bounded_text_errors(answer))
    errors.extend(reference_errors(answer.get("outcome_id"), field="outcome_id", label=_ANSWER_LABEL, required=True))
    errors.extend(reference_errors(answer.get("surface"), field="surface", label=_ANSWER_LABEL, required=True))
    errors.extend(_count_errors(answer))
    errors.extend(_rejected_evidence_errors(answer.get("rejected_evidence")))
    return errors


def evidence_from_plugin_host_observation(record: Any) -> dict[str, Any] | None:
    """One `omh_plugin_host_observation/v1` record as evidence, or None.

    A `not_observed` status is the `installed` tier and nothing more: the host
    is configured for OMH and the wrapper watched it do nothing. That is the
    honest local-configuration fact, and AC1 is exactly the rule that it can
    never become an answer of `ready`.
    """
    if not isinstance(record, Mapping):
        return None
    return _host_record_evidence(
        record,
        source="plugin_host_observation",
        usable_events=("tool_call", "hook_call", "status_query"),
        load_events=("plugin_load", "session_end", "plugin_unload"),
    )


def evidence_from_mcp_host_session(record: Any) -> dict[str, Any] | None:
    """One `omh_mcp_host_session/v1` record as evidence, or None.

    Same reading as the plugin host observation, over the MCP host's own event
    vocabulary: a tool call is the host using the bridge, a load or a session
    boundary is the host having it.
    """
    if not isinstance(record, Mapping):
        return None
    return _host_record_evidence(
        record,
        source="mcp_host_session",
        usable_events=("tool_call",),
        load_events=("host_load", "session_start", "session_end", "host_unload"),
    )


def evidence_from_external_effect_receipt(record: Any) -> dict[str, Any] | None:
    """One `external_effect_receipt/v1` record as evidence, or None.

    Only `succeeded` and `failed` say anything about whether the outcome can
    happen. `attempted` and `unknown` are, by that store's own definition,
    observations that could not classify the effect, and an unclassifiable
    observation is not evidence for or against.
    """
    if not isinstance(record, Mapping):
        return None
    results = {"succeeded": "observed", "failed": "failed"}
    result = results.get(str(record.get("observed_result", "")))
    if result is None:
        return None
    return _safe_evidence(
        tier="used",
        source="external_effect_receipt",
        result=result,
        surface=str(record.get("acting_surface", "")),
        outcome_id=str(record.get("effect_id", "")),
        observed_at=str(record.get("observed_at", "")),
        evidence_ref=str(record.get("receipt_id", "")),
    )


def _host_record_evidence(
    record: Mapping[str, Any],
    *,
    source: str,
    usable_events: tuple[str, ...],
    load_events: tuple[str, ...],
) -> dict[str, Any] | None:
    status = str(record.get("status", ""))
    event = str(record.get("event", ""))
    if status == "blocked":
        tier, result = "host_observed", "blocked"
    elif status == "not_observed":
        tier, result = "installed", "observed"
    elif status == "observed" and event in usable_events:
        tier, result = "usable_observed", "observed"
    elif status == "observed" and event in load_events:
        tier, result = "host_observed", "observed"
    else:
        return None
    return _safe_evidence(
        tier=tier,
        source=source,
        result=result,
        surface=str(record.get("host", "")),
        outcome_id="",
        observed_at=str(record.get("observed_at", "") or record.get("recorded_at", "")),
        evidence_ref=str(record.get("session_id", "")),
    )


def _safe_evidence(
    *,
    tier: str,
    source: str,
    result: str,
    surface: str,
    outcome_id: str,
    observed_at: str,
    evidence_ref: str,
) -> dict[str, Any] | None:
    """Adapt one stored record, or drop it when the store cannot be trusted.

    An adapter reads a file that can be hand-edited, so a record carrying an
    unusable identifier is dropped rather than raised: the surface asking "can
    you do this now" must still get an answer, and a dropped record simply is
    not evidence.
    """
    try:
        return build_external_action_evidence(
            tier=tier,
            source=source,
            result=result,
            surface=surface,
            outcome_id=outcome_id,
            observed_at=observed_at,
            evidence_ref=evidence_ref,
        )
    except ExternalActionReadinessError:
        return None


def _applies_to(record: Mapping[str, Any], *, outcome_id: str, surface: str) -> bool:
    """Whether one valid record speaks about this outcome.

    Outcome-level evidence answers for the outcome it names and for no other,
    which is the whole reason a receipt for a different effect cannot satisfy
    this one. Surface-level evidence carries no outcome and answers for every
    outcome that runs over the named surface.
    """
    named = str(record.get("outcome_id", ""))
    if named:
        return named == outcome_id
    return str(record.get("surface", "")) == surface


def _integrity(*, supplied: int, rejected: int) -> str:
    if not rejected:
        return "clean"
    return "unusable" if rejected == supplied else "partial"


def _preserved_prior(prior: Any, *, outcome_id: str) -> dict[str, Any] | None:
    """The prior answer, when it is a valid answer about this same outcome.

    A prior that does not validate is not a weaker prior; it is not an answer,
    and preserving it would be the same defect AC3 exists to close, one level
    up.
    """
    if not isinstance(prior, Mapping):
        return None
    if validate_external_action_readiness_answer(prior):
        return None
    if str(prior.get("outcome_id", "")) != outcome_id:
        return None
    return {"state": prior.get("state"), "evidence_tier": prior.get("evidence_tier")}


def _derive(scoped: Sequence[Mapping[str, Any]], *, now: str) -> tuple[str, str, int]:
    """(state, evidence tier, age of the deciding record) from scoped evidence."""
    if not scoped:
        return "not_observed", EVIDENCE_TIER_NONE, UNKNOWN_AGE_SECONDS
    aged = [(_age_seconds(str(record.get("observed_at", "")), now), record) for record in scoped]
    fresh = [(age, record) for age, record in aged if age <= EXTERNAL_ACTION_STALE_AFTER_SECONDS]
    if not fresh:
        # Something was true here and none of it still is. Reporting the tier it
        # used to reach would be the exact confusion this module removes.
        return "stale", EVIDENCE_TIER_STALE, min(age for age, _ in aged)
    positives = [(age, record) for age, record in fresh if record.get("result") == "observed"]
    strongest = min(positives, key=_positive_rank) if positives else None
    tier = str(strongest[1]["tier"]) if strongest else EVIDENCE_TIER_NONE
    for result in _NEGATIVE_RESULTS:
        matching = [age for age, record in fresh if record.get("result") == result]
        if matching:
            return result, tier, min(matching)
    if strongest is None:
        return "not_observed", EVIDENCE_TIER_NONE, UNKNOWN_AGE_SECONDS
    age = strongest[0]
    if _TIER_LEVELS[str(strongest[1]["tier"])] >= _READY_TIER_LEVEL:
        return "ready", tier, age
    return "not_observed", tier, age


def _positive_rank(entry: tuple[int, Mapping[str, Any]]) -> tuple[int, int]:
    """Strongest tier first, freshest as the tiebreak, so selection is stable."""
    age, record = entry
    return (-_TIER_LEVELS[str(record["tier"])], age)


def _reason(
    *,
    state: str,
    tier: str,
    state_source: str,
    action: str,
    surface: str,
    rejected_count: int,
) -> str:
    if state_source == "preserved_prior":
        return (
            f"{rejected_count} new evidence record(s) about {action} could not be read, so the last "
            f"valid answer is kept: {state}. Nothing was upgraded and nothing was overwritten; the "
            "gap is unreadable evidence, not a change on the surface."
        )[:_MAX_REASON_CHARS]
    if state_source == "no_valid_evidence":
        return (
            f"Every supplied evidence record about {action} was unreadable and there is no earlier "
            "valid answer to keep, so nothing about this outcome has been observed."
        )[:_MAX_REASON_CHARS]
    tail = f" {rejected_count} unreadable record(s) were rejected and did not affect this answer." if rejected_count else ""
    match state:
        case "ready":
            body = (
                f"{surface} was observed {'completing this outcome' if tier == 'used' else 'in use'} "
                f"recently, so {action} can be attempted now."
            )
        case "blocked":
            body = f"A surface recording {surface} reported something in the way, so {action} cannot proceed yet."
        case "failed":
            body = f"An attempt at {action} was observed not to happen; the recorded failure stands until something newer replaces it."
        case "stale":
            hours = EXTERNAL_ACTION_STALE_AFTER_SECONDS // 3600
            body = (
                f"Everything recorded about {action} is older than {hours} hours, so what was true then "
                "is not evidence about now."
            )
        case _:
            body = _NOT_OBSERVED_REASONS.get(tier, _NOT_OBSERVED_REASONS[EVIDENCE_TIER_NONE]).format(
                action=action,
                surface=surface,
            )
    return (body + tail)[:_MAX_REASON_CHARS]


_NOT_OBSERVED_REASONS: Final = {
    EVIDENCE_TIER_NONE: "Nothing has been recorded about {action}, so whether {surface} works now is unobserved.",
    "installed": "{surface} is configured locally and nothing has been observed using it, so {action} is unproven.",
    "host_observed": "A host reported loading {surface} but nothing was observed using it, so {action} is unproven.",
}


def _next_step(state: str, *, surface: str) -> str:
    match state:
        case "ready":
            return f"Go ahead with the requested action over {surface}."
        case "blocked":
            return f"Clear what the recording surface reported about {surface}, then ask again."
        case "failed":
            return f"Read the recorded failure for this outcome before retrying over {surface}."
        case "stale":
            return f"Ask the host to record a fresh observation of {surface}, then ask again."
        case _:
            return f"Have the host record an observation of {surface} actually being used, then ask again."


def _bounded_action(action: str) -> str:
    text = " ".join(str(action or "").split())
    if not text:
        raise ExternalActionReadinessError("external action readiness requires the requested action")
    if len(text) > _MAX_ACTION_CHARS:
        raise ExternalActionReadinessError(
            f"external action readiness action must be at most {_MAX_ACTION_CHARS} characters"
        )
    if is_unsafe_metadata_line(text):
        raise ExternalActionReadinessError(
            "external action readiness action must be one bounded metadata line without secrets, links, or paths"
        )
    return text


def _bounded_text_errors(answer: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, limit in (("action", _MAX_ACTION_CHARS), ("reason", _MAX_REASON_CHARS), ("next_step", _MAX_STEP_CHARS)):
        value = answer.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            errors.append(f"{field} must be a nonempty string of at most {limit} characters")
        elif is_unsafe_metadata_line(value):
            errors.append(f"{field} must not carry secrets, links, paths, or raw text")
    return errors


def _count_errors(answer: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("accepted_count", "rejected_count"):
        value = answer.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{field} must be a non-negative integer")
    age = answer.get("age_seconds")
    if not isinstance(age, int) or isinstance(age, bool) or age < UNKNOWN_AGE_SECONDS:
        errors.append("age_seconds must be an integer, or -1 when no record decided the state")
    return errors


def _rejected_evidence_errors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return ["rejected_evidence must be a list"]
    if len(value) > _MAX_REJECTED_ROWS:
        return [f"rejected_evidence must hold at most {_MAX_REJECTED_ROWS} rows"]
    errors: list[str] = []
    for row in value:
        if not isinstance(row, Mapping) or {str(key) for key in row} != set(_REJECTED_EVIDENCE_KEYS):
            errors.append(f"each rejected_evidence row must carry exactly {', '.join(_REJECTED_EVIDENCE_KEYS)}")
            continue
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            errors.append("rejected_evidence index must be a non-negative integer")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip() or len(reason) > _MAX_REJECTION_REASON_CHARS:
            errors.append(
                f"rejected_evidence reason must be a nonempty string of at most {_MAX_REJECTION_REASON_CHARS} characters"
            )
    return errors


def _age_seconds(stamp: str, now: str) -> int:
    """Age of an ISO-8601 stamp in seconds, or a value past the horizon.

    Unreadable and future stamps both read as past the horizon, the rule
    `pre_handoff_readiness._stamp_age_seconds` sets: neither can be shown to be
    fresh, and clamping a future stamp would make editing a record a way to
    widen the window from disk.
    """
    observed = _parse_stamp(stamp)
    if observed is None:
        return EXTERNAL_ACTION_STALE_AFTER_SECONDS + 1
    reference = _parse_stamp(now) or datetime.now(timezone.utc)
    age = int((reference - observed).total_seconds())
    return EXTERNAL_ACTION_STALE_AFTER_SECONDS + 1 if age < 0 else age


def _parse_stamp(value: str) -> datetime | None:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        return None
    try:
        parsed = strict_fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=ExternalActionReadinessError)
