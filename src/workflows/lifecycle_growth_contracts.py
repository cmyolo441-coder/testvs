"""Lifecycle launch preparation and backwards-compatible public builder facade."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from .lifecycle_growth_analysis import (
    IN_FLIGHT_ANALYSIS_STATES, analysis_cancellation_hold_reasons, analysis_run_state,
    build_analysis_cancellation, build_analysis_status, route_lifecycle_analysis_request, select_latest_analysis_run,
)
from .lifecycle_growth_artifacts import (
    build_audience_trigger_policy, build_growth_experiment_plan, build_growth_handoff_disposition,
    build_lifecycle_growth_brief, build_lifecycle_safety_policy,
)
from .lifecycle_growth_configuration import review_configuration
from .lifecycle_growth_evaluation import (
    evaluate_lifecycle_growth, readout_lifecycle_growth,
)
from .lifecycle_growth_exposure import review_exposure_evidence
from .lifecycle_growth_readout import build_growth_measurement_readout
from .lifecycle_growth_safety import (
    build_step_outcome, build_throttle_grouping, derive_throttle_group_identity,
    route_lifecycle_workflow_mutation, workflow_mutation_hold_reasons,
)
from .lifecycle_growth_validation import (
    expected_errors as _expected_errors, experiment_hold_reasons, validate_lifecycle_growth_artifact,
)
from .lifecycle_growth_values import metadata_ref, metadata_refs

LIFECYCLE_GROWTH_READINESS_SCHEMA_VERSION: Final = "lifecycle_growth_readiness/v1"
LIFECYCLE_GROWTH_READOUT_SCHEMA_VERSION: Final = "lifecycle_growth_readout/v1"
LIFECYCLE_GROWTH_ENTRY_SCHEMA_VERSION: Final = "lifecycle_growth_entry/v1"
_LAUNCH_SCHEMAS: Final = {
    "brief": "lifecycle_growth_brief/v1", "audience": "audience_trigger_policy/v1",
    "safety": "lifecycle_safety_policy/v1", "experiment": "growth_experiment_plan/v1",
    "handoff": "growth_handoff_disposition/v1",
}


def prepare_lifecycle_growth(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    """Prepare first launch without a fictional readout; existing runs use all gates."""
    records, errors = _launch_records(artifacts)
    readout = artifacts.get("readout")
    if isinstance(readout, Mapping):
        if readout.get("schema_version") != "growth_measurement_readout/v1":
            errors.append("readout artifact has the wrong schema")
        elif validate_lifecycle_growth_artifact(readout):
            errors.append("readout artifact is invalid")
        else:
            records["readout"] = readout
    elif readout is not None:
        errors.append("readout artifact is invalid")
    if errors:
        return _readiness(errors)
    _require_same_lifecycle_growth_id(records, errors)
    _hold_for_safety(records["safety"], errors)
    _hold_for_audience(records["audience"], errors)
    errors.extend(experiment_hold_reasons(records["experiment"]))
    _hold_for_handoff(records["handoff"], errors)
    evidence = artifacts.get("exposure_evidence")
    errors.extend(review_exposure_evidence(evidence, records["experiment"], records.get("readout")).reasons)
    if isinstance(evidence, Mapping):
        if any(evidence.get(name) != records[name] for name in ("audience", "safety")):
            errors.append("exposure_policy_mismatch")
        if evidence.get("channel_refs") != records["brief"].get("available_surface_refs"):
            errors.append("channel_scope_mismatch")
    records["exposure_evidence"] = evidence if isinstance(evidence, Mapping) else {}
    for key in ("configuration_binding", "audience_review", "evaluation_context", "metric_coverage", "metric_plan_binding"):
        if key in artifacts:
            records[key] = artifacts[key]
    binding = artifacts.get("configuration_binding")
    if binding is not None and readout is None:
        errors.extend(review_configuration(artifacts, evaluation=False))
    _hold_for_analysis(records, errors)
    result = _readiness(errors)
    result["configuration_integrity"] = False  # Preparation is never observed launch integrity.
    return result


def evaluate_lifecycle_growth_entry(audience: Mapping[str, Any], *, event_id_ref: str, prior_event_id_refs: Sequence[str], prior_exit_observed: bool, active_intervention_refs: Sequence[str]) -> dict[str, object]:
    """Apply idempotency, re-entry and overlap to caller-supplied metadata only."""
    event_id = metadata_ref(event_id_ref, field="event_id_ref")
    prior_ids = metadata_refs(prior_event_id_refs, field="prior_event_id_refs", required=False)
    active = metadata_refs(active_intervention_refs, field="active_intervention_refs", required=False)
    errors = _expected_errors(audience, "audience_trigger_policy/v1", "audience")
    reasons: list[str] = errors.copy()
    if event_id in prior_ids:
        reasons.append("duplicate event id")
    if prior_ids and audience.get("reentry_policy") == "never":
        reasons.append("re-entry is disallowed")
    if prior_ids and audience.get("reentry_policy") == "after_exit_only" and not prior_exit_observed:
        reasons.append("re-entry requires observed exit")
    if active:
        reasons.append("overlapping intervention is active")
    return {
        "schema_version": LIFECYCLE_GROWTH_ENTRY_SCHEMA_VERSION, "verdict": "HOLD" if reasons else "ACCEPT",
        "event_id_ref": event_id, "idempotency_key_ref": audience.get("idempotency_key_ref", ""),
        "reason_codes": reasons,
        "claim_boundary": "This entry decision is metadata validation only; it does not enter an audience or execute a journey.",
    }


def _launch_records(artifacts: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    records: dict[str, Mapping[str, Any]] = {}
    errors: list[str] = []
    for name, schema in _LAUNCH_SCHEMAS.items():
        record = artifacts.get(name)
        if not isinstance(record, Mapping):
            errors.append(f"{name} artifact is missing")
            continue
        errors.extend(_expected_errors(record, schema, name))
        records[name] = record
    return records, errors


def _require_same_lifecycle_growth_id(records: Mapping[str, Mapping[str, Any]], errors: list[str]) -> None:
    ids = {record.get("lifecycle_growth_id") for record in records.values()}
    if len(ids) != 1:
        errors.append("artifacts must share lifecycle_growth_id")


def _hold_for_safety(safety: Mapping[str, Any], errors: list[str]) -> None:
    for field in ("consent_state", "suppression_state", "frequency_state", "channel_eligibility_state", "quiet_hours_state", "locale_state", "legal_tenant_state"):
        if safety.get(field) != "eligible":
            errors.append(f"{field} is not eligible")
    errors.extend(workflow_mutation_hold_reasons(safety))


def _hold_for_audience(audience: Mapping[str, Any], errors: list[str]) -> None:
    if audience.get("audience_identity_state") != "known":
        errors.append("audience identity is unknown")
    if audience.get("canonical_event_state") != "known":
        errors.append("canonical event semantics are unknown")


def _hold_for_analysis(records: Mapping[str, Mapping[str, Any]], errors: list[str]) -> None:
    """Expansion and cancellation must reconcile the supplied existing run."""
    readout = records.get("readout")
    if readout is None:
        return
    result = evaluate_lifecycle_growth(records["experiment"], readout,
        exposure_evidence=records.get("exposure_evidence"), audience_review=records.get("audience_review"),
        configuration_binding=records.get("configuration_binding"), evaluation_context=records.get("evaluation_context"),
        metric_coverage=records.get("metric_coverage"), metric_plan_binding=records.get("metric_plan_binding"))
    if result["interpretation_state"] == "HOLD":
        errors.append("observed readout interpretation is on hold")
        reasons = result["evidence_reason_codes"]
        if isinstance(reasons, list):
            errors.extend(reason for reason in reasons if isinstance(reason, str))
    if analysis_run_state(readout) in IN_FLIGHT_ANALYSIS_STATES:
        errors.append("the latest analysis run is still in flight; reconcile it before preparing another")
    errors.extend(analysis_cancellation_hold_reasons(readout.get("analysis_status"), records["handoff"].get("analysis_cancellation")))


def _hold_for_handoff(handoff: Mapping[str, Any], errors: list[str]) -> None:
    if handoff.get("connector_evidence_state") != "observed_available":
        errors.append("connector is not observed available")


def _readiness(hold_reasons: list[str]) -> dict[str, object]:
    return {
        "schema_version": LIFECYCLE_GROWTH_READINESS_SCHEMA_VERSION,
        "verdict": "HOLD" if hold_reasons else "READY", "launch_ready": not hold_reasons,
        "hold_reasons": hold_reasons,
        "claim_boundary": "Readiness is a local validation result, not approval or external-effect evidence.",
    }


__all__ = [
    "analysis_cancellation_hold_reasons", "build_analysis_cancellation", "build_analysis_status",
    "build_audience_trigger_policy", "build_growth_experiment_plan", "build_growth_handoff_disposition",
    "build_growth_measurement_readout", "build_lifecycle_growth_brief", "build_lifecycle_safety_policy",
    "build_step_outcome", "build_throttle_grouping", "derive_throttle_group_identity",
    "evaluate_lifecycle_growth", "evaluate_lifecycle_growth_entry", "prepare_lifecycle_growth",
    "readout_lifecycle_growth", "route_lifecycle_analysis_request", "route_lifecycle_workflow_mutation",
    "select_latest_analysis_run", "validate_lifecycle_growth_artifact",
]
