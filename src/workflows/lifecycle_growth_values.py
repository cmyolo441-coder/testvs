from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..system.append_only_store import RAW_OR_HIDDEN_KEYS, opaque_ref, reference_errors


MAX_REFERENCES: Final = 8
PREPARED_STATUS: Final = "prepared_not_observed"
CLAIM_BOUNDARY: Final = (
    "This prepared lifecycle-growth metadata records bounded references and caller assertions only. "
    "It does not identify people, send messages, schedule work, mutate flags, deliver treatment, "
    "observe outcomes, or establish causality."
)
ARTIFACT_KEYS: Final = {
    "lifecycle_growth_brief/v1": frozenset(
        {
            "schema_version", "status", "lifecycle_growth_id", "lifecycle_stage", "target_behavior_ref",
            "target_segment_ref", "baseline_ref", "available_surface_refs", "experiment_budget_ref",
            "observed_evidence_refs", "hypotheses", "non_goals", "decision_owner", "claim_boundary",
        }
    ),
    "audience_trigger_policy/v1": frozenset(
        {
            "schema_version", "status", "lifecycle_growth_id", "audience_identity_state",
            "audience_identity_ref", "canonical_event_state", "canonical_event_schema_ref",
            "entry_condition_ref", "exit_condition_ref", "exclusion_refs", "idempotency_key_ref",
            "reentry_policy", "collision_policy", "claim_boundary",
        }
    ),
    "lifecycle_safety_policy/v1": frozenset(
        {
            "schema_version", "status", "lifecycle_growth_id", "consent_state", "suppression_state",
            "frequency_state", "suppression_precedence", "preference_policy_ref",
            "global_frequency_budget_ref", "campaign_frequency_budget_ref", "channel_eligibility_state",
            "quiet_hours_state", "locale_state", "legal_tenant_state", "throttle_grouping",
            "workflow_content_state", "mutation_route", "promotion_decision_state",
            "promotion_result_state", "claim_boundary",
        }
    ),
    "growth_experiment_plan/v1": frozenset(
        {
            "schema_version", "status", "lifecycle_growth_id", "treatment_ref", "control_ref",
            "assignment_unit", "exposure_unit", "sticky_assignment_policy", "assignment_event_ref",
            "actual_exposure_event_ref", "primary_metric_ref", "guardrail_metric_refs", "holdout_state",
            "holdout_rationale_ref", "minimum_runtime_days", "data_health_state", "rollback_conditions",
            "pause_condition_refs", "approval_state", "claim_boundary",
        }
    ),
    "growth_measurement_readout/v1": frozenset(
        {
            "schema_version", "status", "lifecycle_growth_id", "eligible_count", "attempted_count",
            "delivered_count", "displayed_count", "acted_count", "outcome_count", "runtime_days_observed",
            "denominator_state", "data_freshness_state", "instrumentation_state", "sample_ratio_state",
            "cross_exposure_state", "overlap_state", "primary_metric_state", "guardrail_state",
            "causal_claim_status", "rollback_state", "provider_evidence_refs",
            "actual_exposure_evidence_refs", "data_evidence_refs", "runtime_evidence_refs",
            "causal_evidence_refs", "step_outcomes", "step_trace_state", "analysis_status", "disposition",
            "claim_boundary",
        }
    ),
    "growth_handoff_disposition/v1": frozenset(
        {
            "schema_version", "status", "lifecycle_growth_id", "proposed_action_refs",
            "proposed_action_kinds", "action_owner", "approver", "connector_evidence_state",
            "connector_evidence_refs", "timing_state", "stop_condition_refs", "analysis_cancellation",
            "claim_boundary",
        }
    ),
}


def metadata_ref(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=ValueError)


def metadata_refs(values: Sequence[str], *, field: str, required: bool) -> list[str]:
    if isinstance(values, str) or len(values) > MAX_REFERENCES:
        raise ValueError(f"{field} must contain at most {MAX_REFERENCES} opaque references")
    result = [metadata_ref(value, field=field) for value in values]
    if required and not result:
        raise ValueError(f"{field} is required")
    return result


def require_state(value: str, *, field: str, allowed: tuple[str, ...]) -> str:
    if value not in allowed:
        raise ValueError(f"{field} is invalid")
    return value


def require_nonnegative_count(value: int, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def artifact_shape_errors(record: Any) -> tuple[str, list[str]]:
    if not isinstance(record, Mapping):
        return "", ["lifecycle growth artifact must be an object"]
    schema = record.get("schema_version")
    if not isinstance(schema, str) or schema not in ARTIFACT_KEYS:
        return "", ["lifecycle growth artifact schema_version is unsupported"]
    errors: list[str] = []
    raw_keys = sorted(str(key) for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if raw_keys:
        errors.append(f"lifecycle growth artifact must not carry raw or hidden keys: {raw_keys}")
    keys = {str(key) for key in record}
    missing = sorted(ARTIFACT_KEYS[schema] - keys)
    unexpected = sorted(keys - ARTIFACT_KEYS[schema])
    if missing:
        errors.append(f"lifecycle growth artifact missing keys: {missing}")
    if unexpected:
        errors.append(f"lifecycle growth artifact has unsupported keys: {unexpected}")
    if record.get("status") != PREPARED_STATUS:
        errors.append("lifecycle growth artifact status must be prepared_not_observed")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("lifecycle growth artifact claim_boundary is invalid")
    return schema, errors


def ref_errors(value: Any, *, field: str, required: bool) -> list[str]:
    return reference_errors(value, field=field, label="lifecycle growth artifact", required=required)


def refs_errors(value: Any, *, field: str, required: bool) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCES:
        return [f"{field} must be a list of at most {MAX_REFERENCES} opaque references"]
    if required and not value:
        return [f"{field} is required"]
    return [error for ref in value for error in ref_errors(ref, field=field, required=True)]


def state_errors(value: Any, *, field: str, allowed: tuple[str, ...]) -> list[str]:
    return [] if value in allowed else [f"{field} is invalid"]


def count_errors(value: Any, *, field: str) -> list[str]:
    return [] if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else [f"{field} must be non-negative"]


def positive_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
