"""Bounded caller-supplied audience, contact and channel evidence for #1399.

The six historical artifacts retain their validators. This companion records
observations separately from their policies; no provider is queried. Counts
are unique subjects in the declared exposure unit and observation window.
Channel populations may overlap: reconciliation evidence, not summation,
accounts for the aggregate population.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypeGuard

from .lifecycle_growth_artifacts import validate_audience, validate_safety
from .lifecycle_growth_values import (
    CLAIM_BOUNDARY, MAX_REFERENCES, PREPARED_STATUS, artifact_shape_errors,
    count_errors, ref_errors, refs_errors, state_errors,
)

EXPOSURE_SCHEMA: Final = "lifecycle_growth_exposure_evidence/v1"
_REFS: Final = (
    "eligibility_evidence_refs", "exclusion_evidence_refs", "assignment_evidence_refs",
    "population_evidence_refs", "contact_evidence_refs", "overlap_evidence_refs",
    "overlapping_experiment_refs", "channel_refs",
)
_IDENTITIES: Final = (
    "lifecycle_growth_id", "observation_window_ref", "assignment_unit", "exposure_unit",
    "assignment_event_ref", "actual_exposure_event_ref",
)
_COUNTS: Final = ("eligible_count", "assigned_count", "repeated_contact_count")
_STATES: Final = {
    "population_reconciliation_state": ("reconciled", "inconsistent", "unknown"),
    "contact_pressure_state": ("within_budget", "exceeded", "unknown"),
    "overlap_state": ("absent", "detected", "unknown"),
}
_CHANNEL_COUNTS: Final = ("attempted_count", "delivered_count", "reached_count", "failed_count", "unresolved_count")
_CHANNEL_KEYS: Final = frozenset((*_CHANNEL_COUNTS, "channel_ref", "reachability_state", "evidence_refs", "failure_reason_refs"))
_KEYS: Final = frozenset((*_REFS, *_IDENTITIES, *_COUNTS, *_STATES,
    "schema_version", "status", "claim_boundary", "audience", "safety", "analysis_run_ref", "channels"))


def _object(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return _object(value) and all(isinstance(key, str) for key in value)


def _list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def is_exposure_evidence(value: object) -> TypeGuard[Mapping[str, object]]:
    """Identify the companion schema before applying its closed validator."""
    return _mapping(value) and value.get("schema_version") == EXPOSURE_SCHEMA


def validate_exposure_evidence(value: object) -> list[str]:
    """Structural validation uses the existing closed metadata conventions.

    Null counts and empty evidence lists are readable unknowns, never zeros or
    completed checks. Semantic contradictions are decisions, not decode errors.
    """
    if not _mapping(value) or set(value) != set(_KEYS):
        return ["exposure evidence keys are invalid"]
    errors: list[str] = []
    if value["schema_version"] != EXPOSURE_SCHEMA or value["status"] != PREPARED_STATUS or value["claim_boundary"] != CLAIM_BOUNDARY:
        errors.append("exposure evidence envelope is invalid")
    for field in _IDENTITIES:
        errors.extend(ref_errors(value[field], field=field, required=True))
    errors.extend(ref_errors(value["analysis_run_ref"], field="analysis_run_ref", required=False))
    for field in _REFS:
        errors.extend(refs_errors(value[field], field=field, required=False))
    for field in _COUNTS:
        if value[field] is not None:
            errors.extend(count_errors(value[field], field=field))
    for field, states in _STATES.items():
        errors.extend(state_errors(value[field], field=field, allowed=states))
    for name, schema, validator in (
        ("audience", "audience_trigger_policy/v1", validate_audience),
        ("safety", "lifecycle_safety_policy/v1", validate_safety),
    ):
        record = value[name]
        if not _mapping(record):
            errors.append(name + " artifact is invalid")
            continue
        actual, shape_errors = artifact_shape_errors(record)
        if actual != schema or shape_errors or validator(record):
            errors.append(name + " artifact is invalid")
        if record.get("lifecycle_growth_id") != value["lifecycle_growth_id"]:
            errors.append(name + " lifecycle_growth_id differs")
    channels = value["channels"]
    if not _list(channels) or len(channels) > MAX_REFERENCES:
        return errors + ["channels must be a bounded list"]
    for channel in channels:
        if not _mapping(channel) or set(channel) != set(_CHANNEL_KEYS):
            errors.append("channel keys are invalid")
            continue
        errors.extend(ref_errors(channel["channel_ref"], field="channel_ref", required=True))
        errors.extend(state_errors(channel["reachability_state"], field="reachability_state", allowed=("reachable", "unreachable", "unknown")))
        for field in _CHANNEL_COUNTS:
            if channel[field] is not None:
                errors.extend(count_errors(channel[field], field=field))
        for field in ("evidence_refs", "failure_reason_refs"):
            errors.extend(refs_errors(channel[field], field=field, required=False))
    return errors


@dataclass(frozen=True, slots=True)
class ExposureReview:
    reasons: tuple[str, ...]
    assigned_count: int | None = None
    channels: tuple[Mapping[str, object], ...] = ()


def review_exposure_evidence(
    evidence: object, experiment: Mapping[str, object] | None,
    readout: Mapping[str, object] | None,
) -> ExposureReview:
    """Check supplied observations; a missing readout denotes a first launch.

    First launch checks eligibility, reachability and contact safety only.
    Expansion additionally requires assigned and observed channel populations.
    """
    if evidence is None:
        return ExposureReview(("exposure_evidence_missing",))
    if validate_exposure_evidence(evidence):
        return ExposureReview(("exposure_evidence_invalid",))
    assert _mapping(evidence)
    audience, safety = evidence["audience"], evidence["safety"]
    assert _mapping(audience) and _mapping(safety)
    reasons: list[str] = []
    for record in (experiment, readout):
        if record is not None and evidence["lifecycle_growth_id"] != record.get("lifecycle_growth_id"):
            reasons.append("exposure_identity_mismatch")
    if experiment is not None and any(evidence[field] != experiment.get(field) for field in (
        "assignment_unit", "exposure_unit", "assignment_event_ref", "actual_exposure_event_ref",
    )):
        reasons.append("assignment_identity_mismatch")
    if audience["audience_identity_state"] != "known" or audience["canonical_event_state"] != "known":
        reasons.append("audience_unknown")
    for field in ("consent_state", "suppression_state", "frequency_state", "channel_eligibility_state", "quiet_hours_state", "locale_state", "legal_tenant_state"):
        if safety[field] != "eligible":
            reasons.append(field + "_not_eligible")
    for field in ("eligibility_evidence_refs", "exclusion_evidence_refs", "contact_evidence_refs", "overlap_evidence_refs"):
        if not evidence[field]:
            reasons.append(field + "_missing")
    eligible, assigned = _count(evidence["eligible_count"]), _count(evidence["assigned_count"])
    if eligible is None or eligible == 0:
        reasons.append("eligible_population_unknown_or_empty")
    if evidence["repeated_contact_count"] is None:
        reasons.append("contact_pressure_observation_missing")
    if evidence["contact_pressure_state"] != "within_budget":
        reasons.append("contact_pressure_" + str(evidence["contact_pressure_state"]))
    if evidence["overlapping_experiment_refs"]:
        reasons.append("experiment_overlap_detected")
    if evidence["overlap_state"] != "absent":
        reasons.append("experiment_overlap_" + str(evidence["overlap_state"]))
    raw_channels = evidence["channels"]
    assert _list(raw_channels)
    channels = tuple(channel for channel in raw_channels if _mapping(channel))
    channel_refs = evidence["channel_refs"]
    assert _list(channel_refs)
    observed_refs = [channel["channel_ref"] for channel in channels]
    if not channel_refs or len(set(channel_refs)) != len(channel_refs) or len(observed_refs) != len(channel_refs) or set(observed_refs) != set(channel_refs):
        reasons.append("channel_coverage_unknown_or_inconsistent")
    for channel in channels:
        if channel["reachability_state"] != "reachable" or not channel["evidence_refs"]:
            reasons.append("channel_reachability_unknown_or_unreachable")
        counts = [_count(channel[field]) for field in _CHANNEL_COUNTS]
        attempted_channel, delivered, reached, failed, unresolved = counts
        if failed or unresolved:
            reasons.append("channel_partial_delivery")
        if failed and not channel["failure_reason_refs"]:
            reasons.append("channel_failure_reason_missing")
        if any(count is None for count in counts):
            if readout is not None:
                reasons.append("channel_observation_unknown")
            continue
        assert attempted_channel is not None and delivered is not None and reached is not None and failed is not None and unresolved is not None
        if attempted_channel != delivered + failed + unresolved or reached > delivered:
            reasons.append("channel_counts_inconsistent")
        if reached < attempted_channel:
            reasons.append("channel_partial_delivery")
    if readout is None:
        return ExposureReview(tuple(dict.fromkeys(reasons)), assigned, channels)
    if not evidence["assignment_evidence_refs"] or assigned is None:
        reasons.append("assignment_evidence_missing")
    if evidence["population_reconciliation_state"] != "reconciled" or not evidence["population_evidence_refs"]:
        reasons.append("population_reconciliation_unknown_or_inconsistent")
    analysis = readout.get("analysis_status")
    if not _mapping(analysis) or not evidence["analysis_run_ref"] or evidence["analysis_run_ref"] != analysis.get("run_ref"):
        reasons.append("analysis_run_mismatch")
    attempted = _count(readout.get("attempted_count"))
    reached_total = _count(readout.get("displayed_count"))
    if attempted is not None and reached_total is not None and reached_total < attempted:
        reasons.append("channel_partial_delivery")
    if eligible != readout.get("eligible_count") or (eligible is not None and assigned is not None and attempted is not None and not eligible >= assigned >= attempted):
        reasons.append("assigned_population_mismatch")
    for channel_field, readout_field in (("attempted_count", "attempted_count"), ("delivered_count", "delivered_count"), ("reached_count", "displayed_count")):
        counts = [_count(channel[channel_field]) for channel in channels]
        total = _count(readout.get(readout_field))
        if counts and all(count is not None for count in counts) and total is not None:
            known = [count for count in counts if count is not None]
            if not max(known) <= total <= sum(known):
                reasons.append("channel_population_mismatch")
    return ExposureReview(tuple(dict.fromkeys(reasons)), assigned, channels)
