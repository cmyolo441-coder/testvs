"""Explicit semantic-input adapters for lifecycle and discovery artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from .lifecycle_growth_contracts import (
    build_audience_trigger_policy,
    build_growth_experiment_plan,
    build_growth_handoff_disposition,
    build_lifecycle_growth_brief,
    build_lifecycle_safety_policy,
)
from .product_discovery_validation import (
    build_assumption_test_portfolio,
    build_customer_discovery_plan,
    build_discovery_decision_frame,
    build_discovery_evidence_ledger,
    build_initial_gtm_hypothesis,
    channel_feedback_ledger_for_history,
)


class WorkflowArtifactSemanticInputError(ValueError):
    """Raised when a semantic workflow input does not match its closed shape."""


_LIFECYCLE_ROOT_KEYS: Final = frozenset(
    {"lifecycle_growth_id", "brief", "audience", "safety", "experiment", "handoff"}
)
_DISCOVERY_ROOT_KEYS: Final = frozenset(
    {"discovery_id", "frame", "ledger", "plan", "portfolio", "gtm"}
)
_LIFECYCLE_FIELDS: Final = {
    "brief": frozenset(
        {
            "lifecycle_stage", "target_behavior_ref", "target_segment_ref", "baseline_ref",
            "available_surface_refs", "experiment_budget_ref", "observed_evidence_refs", "hypotheses",
            "non_goals", "decision_owner",
        }
    ),
    "audience": frozenset(
        {
            "audience_identity_state", "audience_identity_ref", "canonical_event_state",
            "canonical_event_schema_ref", "entry_condition_ref", "exit_condition_ref", "exclusion_refs",
            "idempotency_key_ref", "reentry_policy", "collision_policy",
        }
    ),
    "safety": frozenset(
        {
            "consent_state", "suppression_state", "frequency_state", "suppression_precedence",
            "preference_policy_ref", "global_frequency_budget_ref", "campaign_frequency_budget_ref",
            "channel_eligibility_state", "quiet_hours_state", "locale_state", "legal_tenant_state",
            "throttle_grouping", "workflow_content_state", "mutation_route", "promotion_decision_state",
            "promotion_result_state",
        }
    ),
    "experiment": frozenset(
        {
            "treatment_ref", "control_ref", "assignment_unit", "exposure_unit", "sticky_assignment_policy",
            "assignment_event_ref", "actual_exposure_event_ref", "primary_metric_ref", "guardrail_metric_refs",
            "holdout_state", "holdout_rationale_ref", "minimum_runtime_days", "data_health_state",
            "rollback_conditions", "pause_condition_refs", "approval_state",
        }
    ),
    "handoff": frozenset(
        {
            "proposed_action_refs", "proposed_action_kinds", "action_owner", "approver",
            "connector_evidence_state", "connector_evidence_refs", "timing_state", "stop_condition_refs",
            "analysis_cancellation",
        }
    ),
}
_CHANNEL_FEEDBACK_FIELDS: Final = frozenset(
    {"discovery_id", "gtm_artifact_id", "initial_channel_ref", "segment_ref", "entries"}
)
_DISCOVERY_FIELDS: Final = {
    "frame": frozenset(
        {
            "problem_ref", "segment_ref", "segment_definition_state", "alternative_refs", "decision_owner_ref",
            "learning_budget_ref", "deadline_at", "kill_criteria_refs",
        }
    ),
    "ledger": frozenset({"entries"}),
    "plan": frozenset({"participant_criteria_ref", "consent_privacy_ref", "bias_control_refs", "human_task_ref"}),
    "portfolio": frozenset({"assumptions"}),
    "gtm": frozenset(
        {
            "beachhead_segment_ref", "buyer_ref", "user_ref", "current_alternative_ref",
            "value_proposition_ref", "pricing_hypothesis_ref", "initial_channel_ref", "first_cohort_ref",
            "learning_metric_refs",
        }
    ),
}


def build_lifecycle_growth_artifacts(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build all lifecycle artifacts from closed semantic groups."""
    root = _exact_mapping(payload, _LIFECYCLE_ROOT_KEYS, "lifecycle growth build input")
    lifecycle_growth_id = _required_string(root, "lifecycle_growth_id")
    brief = _group(root, "brief", _LIFECYCLE_FIELDS)
    audience = _group(root, "audience", _LIFECYCLE_FIELDS)
    safety = _group(root, "safety", _LIFECYCLE_FIELDS)
    experiment = _group(root, "experiment", _LIFECYCLE_FIELDS)
    handoff = _group(root, "handoff", _LIFECYCLE_FIELDS)
    return {
        "brief": build_lifecycle_growth_brief(lifecycle_growth_id=lifecycle_growth_id, **brief),
        "audience": build_audience_trigger_policy(lifecycle_growth_id=lifecycle_growth_id, **audience),
        "safety": build_lifecycle_safety_policy(lifecycle_growth_id=lifecycle_growth_id, **safety),
        "experiment": build_growth_experiment_plan(lifecycle_growth_id=lifecycle_growth_id, **experiment),
        "handoff": build_growth_handoff_disposition(lifecycle_growth_id=lifecycle_growth_id, **handoff),
    }


def build_channel_feedback_ledger_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a companion from its own semantic boundary, never widen legacy build."""
    values = _exact_mapping(payload, _CHANNEL_FEEDBACK_FIELDS, "channel feedback semantic input")
    return channel_feedback_ledger_for_history(values)


def build_product_discovery_package(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the five pre-decision discovery artifacts from semantic groups."""
    root = _exact_mapping(payload, _DISCOVERY_ROOT_KEYS, "product discovery build input")
    discovery_id = _required_string(root, "discovery_id")
    frame = _group(root, "frame", _DISCOVERY_FIELDS)
    ledger = _group(root, "ledger", _DISCOVERY_FIELDS)
    plan = _group(root, "plan", _DISCOVERY_FIELDS)
    portfolio = _group(root, "portfolio", _DISCOVERY_FIELDS)
    gtm = _group(root, "gtm", _DISCOVERY_FIELDS)
    return {
        "frame": build_discovery_decision_frame(discovery_id=discovery_id, **frame),
        "ledger": build_discovery_evidence_ledger(discovery_id=discovery_id, **ledger),
        "plan": build_customer_discovery_plan(discovery_id=discovery_id, **plan),
        "portfolio": build_assumption_test_portfolio(discovery_id=discovery_id, **portfolio),
        "gtm": build_initial_gtm_hypothesis(discovery_id=discovery_id, **gtm),
    }


def _group(
    payload: Mapping[str, Any], name: str, fields_by_group: Mapping[str, frozenset[str]]
) -> dict[str, Any]:
    return _exact_mapping(_required_mapping(payload, name), fields_by_group[name], f"{name} semantic input")


def _exact_mapping(payload: Mapping[str, Any], expected: frozenset[str], label: str) -> dict[str, Any]:
    if set(payload) != set(expected):
        raise WorkflowArtifactSemanticInputError(f"{label} keys are invalid")
    return dict(payload)


def _required_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise WorkflowArtifactSemanticInputError(f"{field} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise WorkflowArtifactSemanticInputError(f"{field} must be a string")
    return value
