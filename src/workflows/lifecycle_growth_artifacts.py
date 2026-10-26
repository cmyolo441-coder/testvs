from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .lifecycle_growth_analysis import (
    analysis_cancellation_errors,
    analysis_cancellation_record,
)
from .lifecycle_growth_safety import (
    MUTATION_ROUTES,
    PROMOTION_DECISION_STATES,
    PROMOTION_RESULT_STATES,
    THROTTLE_GROUPING_INPUT_KEYS,
    WORKFLOW_CONTENT_STATES,
    build_throttle_grouping,
    throttle_grouping_errors,
)
from .lifecycle_growth_values import (
    CLAIM_BOUNDARY,
    PREPARED_STATUS,
    metadata_ref,
    metadata_refs,
    ref_errors,
    refs_errors,
    require_nonnegative_count,
    require_state,
    state_errors,
)


_SAFETY_STATES = ("eligible", "ineligible", "unknown")
_ACTION_KINDS = ("connector", "content", "analytics", "product", "implementation")


def _record(schema_version: str, lifecycle_growth_id: str, **fields: object) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "status": PREPARED_STATUS,
        "lifecycle_growth_id": metadata_ref(lifecycle_growth_id, field="lifecycle_growth_id"),
        **fields,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_lifecycle_growth_brief(*, lifecycle_growth_id: str, lifecycle_stage: str, target_behavior_ref: str, target_segment_ref: str, baseline_ref: str, available_surface_refs: Sequence[str], experiment_budget_ref: str, observed_evidence_refs: Sequence[str], hypotheses: Sequence[str], non_goals: Sequence[str], decision_owner: str) -> dict[str, object]:
    return _record(
        "lifecycle_growth_brief/v1",
        lifecycle_growth_id,
        lifecycle_stage=metadata_ref(lifecycle_stage, field="lifecycle_stage"),
        target_behavior_ref=metadata_ref(target_behavior_ref, field="target_behavior_ref"),
        target_segment_ref=metadata_ref(target_segment_ref, field="target_segment_ref"),
        baseline_ref=metadata_ref(baseline_ref, field="baseline_ref"),
        available_surface_refs=metadata_refs(available_surface_refs, field="available_surface_refs", required=True),
        experiment_budget_ref=metadata_ref(experiment_budget_ref, field="experiment_budget_ref"),
        observed_evidence_refs=metadata_refs(observed_evidence_refs, field="observed_evidence_refs", required=True),
        hypotheses=metadata_refs(hypotheses, field="hypotheses", required=True),
        non_goals=metadata_refs(non_goals, field="non_goals", required=True),
        decision_owner=metadata_ref(decision_owner, field="decision_owner"),
    )


def build_audience_trigger_policy(*, lifecycle_growth_id: str, audience_identity_state: str, audience_identity_ref: str, canonical_event_state: str, canonical_event_schema_ref: str, entry_condition_ref: str, exit_condition_ref: str, exclusion_refs: Sequence[str], idempotency_key_ref: str, reentry_policy: str, collision_policy: str) -> dict[str, object]:
    identity_state = require_state(audience_identity_state, field="audience_identity_state", allowed=("known", "unknown"))
    event_state = require_state(canonical_event_state, field="canonical_event_state", allowed=("known", "unknown"))
    return _record(
        "audience_trigger_policy/v1",
        lifecycle_growth_id,
        audience_identity_state=identity_state,
        audience_identity_ref=metadata_ref(audience_identity_ref, field="audience_identity_ref") if identity_state == "known" else "",
        canonical_event_state=event_state,
        canonical_event_schema_ref=metadata_ref(canonical_event_schema_ref, field="canonical_event_schema_ref") if event_state == "known" else "",
        entry_condition_ref=metadata_ref(entry_condition_ref, field="entry_condition_ref"),
        exit_condition_ref=metadata_ref(exit_condition_ref, field="exit_condition_ref"),
        exclusion_refs=metadata_refs(exclusion_refs, field="exclusion_refs", required=True),
        idempotency_key_ref=metadata_ref(idempotency_key_ref, field="idempotency_key_ref"),
        reentry_policy=require_state(reentry_policy, field="reentry_policy", allowed=("never", "after_exit_only")),
        collision_policy=require_state(collision_policy, field="collision_policy", allowed=("hold_on_overlap", "separate_interventions")),
    )


def build_lifecycle_safety_policy(*, lifecycle_growth_id: str, consent_state: str, suppression_state: str, frequency_state: str, suppression_precedence: str, preference_policy_ref: str, global_frequency_budget_ref: str, campaign_frequency_budget_ref: str, channel_eligibility_state: str, quiet_hours_state: str, locale_state: str, legal_tenant_state: str, throttle_grouping: Mapping[str, str], workflow_content_state: str, mutation_route: str, promotion_decision_state: str, promotion_result_state: str) -> dict[str, object]:
    if not isinstance(throttle_grouping, Mapping) or {str(key) for key in throttle_grouping} != THROTTLE_GROUPING_INPUT_KEYS:
        raise ValueError(f"throttle_grouping must carry exactly {sorted(THROTTLE_GROUPING_INPUT_KEYS)}")
    return _record(
        "lifecycle_safety_policy/v1",
        lifecycle_growth_id,
        consent_state=require_state(consent_state, field="consent_state", allowed=_SAFETY_STATES),
        suppression_state=require_state(suppression_state, field="suppression_state", allowed=_SAFETY_STATES),
        frequency_state=require_state(frequency_state, field="frequency_state", allowed=_SAFETY_STATES),
        suppression_precedence=require_state(suppression_precedence, field="suppression_precedence", allowed=("suppression_overrides_all",)),
        preference_policy_ref=metadata_ref(preference_policy_ref, field="preference_policy_ref"),
        global_frequency_budget_ref=metadata_ref(global_frequency_budget_ref, field="global_frequency_budget_ref"),
        campaign_frequency_budget_ref=metadata_ref(campaign_frequency_budget_ref, field="campaign_frequency_budget_ref"),
        channel_eligibility_state=require_state(channel_eligibility_state, field="channel_eligibility_state", allowed=_SAFETY_STATES),
        quiet_hours_state=require_state(quiet_hours_state, field="quiet_hours_state", allowed=_SAFETY_STATES),
        locale_state=require_state(locale_state, field="locale_state", allowed=_SAFETY_STATES),
        legal_tenant_state=require_state(legal_tenant_state, field="legal_tenant_state", allowed=_SAFETY_STATES),
        throttle_grouping=build_throttle_grouping(**{str(key): str(value) for key, value in throttle_grouping.items()}),
        workflow_content_state=require_state(workflow_content_state, field="workflow_content_state", allowed=WORKFLOW_CONTENT_STATES),
        mutation_route=require_state(mutation_route, field="mutation_route", allowed=MUTATION_ROUTES),
        promotion_decision_state=require_state(promotion_decision_state, field="promotion_decision_state", allowed=PROMOTION_DECISION_STATES),
        promotion_result_state=require_state(promotion_result_state, field="promotion_result_state", allowed=PROMOTION_RESULT_STATES),
    )


def build_growth_experiment_plan(*, lifecycle_growth_id: str, treatment_ref: str, control_ref: str, assignment_unit: str, exposure_unit: str, sticky_assignment_policy: str, assignment_event_ref: str, actual_exposure_event_ref: str, primary_metric_ref: str, guardrail_metric_refs: Sequence[str], holdout_state: str, holdout_rationale_ref: str, minimum_runtime_days: int, data_health_state: str, rollback_conditions: Sequence[str], pause_condition_refs: Sequence[str], approval_state: str) -> dict[str, object]:
    runtime_days = require_nonnegative_count(minimum_runtime_days, field="minimum_runtime_days")
    if runtime_days < 1:
        raise ValueError("minimum_runtime_days must be at least one")
    return _record(
        "growth_experiment_plan/v1",
        lifecycle_growth_id,
        treatment_ref=metadata_ref(treatment_ref, field="treatment_ref"),
        control_ref=metadata_ref(control_ref, field="control_ref"),
        assignment_unit=metadata_ref(assignment_unit, field="assignment_unit"),
        exposure_unit=metadata_ref(exposure_unit, field="exposure_unit"),
        sticky_assignment_policy=require_state(sticky_assignment_policy, field="sticky_assignment_policy", allowed=("sticky",)),
        assignment_event_ref=metadata_ref(assignment_event_ref, field="assignment_event_ref"),
        actual_exposure_event_ref=metadata_ref(actual_exposure_event_ref, field="actual_exposure_event_ref"),
        primary_metric_ref=metadata_ref(primary_metric_ref, field="primary_metric_ref"),
        guardrail_metric_refs=metadata_refs(guardrail_metric_refs, field="guardrail_metric_refs", required=True),
        holdout_state=require_state(holdout_state, field="holdout_state", allowed=("preserved", "unknown")),
        holdout_rationale_ref=metadata_ref(holdout_rationale_ref, field="holdout_rationale_ref") if holdout_state == "preserved" else "",
        minimum_runtime_days=runtime_days,
        data_health_state=require_state(data_health_state, field="data_health_state", allowed=("healthy", "broken", "unknown")),
        rollback_conditions=metadata_refs(rollback_conditions, field="rollback_conditions", required=True),
        pause_condition_refs=metadata_refs(pause_condition_refs, field="pause_condition_refs", required=True),
        approval_state=require_state(approval_state, field="approval_state", allowed=("approved", "pending", "rejected")),
    )


def build_growth_handoff_disposition(*, lifecycle_growth_id: str, proposed_action_refs: Sequence[str], proposed_action_kinds: Sequence[str], action_owner: str, approver: str, connector_evidence_state: str, connector_evidence_refs: Sequence[str], timing_state: str, stop_condition_refs: Sequence[str], analysis_cancellation: Mapping[str, Any]) -> dict[str, object]:
    actions = metadata_refs(proposed_action_refs, field="proposed_action_refs", required=True)
    if len(actions) != len(proposed_action_kinds):
        raise ValueError("proposed_action_kinds must match proposed_action_refs")
    return _record(
        "growth_handoff_disposition/v1",
        lifecycle_growth_id,
        proposed_action_refs=actions,
        proposed_action_kinds=[require_state(kind, field="proposed_action_kinds", allowed=_ACTION_KINDS) for kind in proposed_action_kinds],
        action_owner=metadata_ref(action_owner, field="action_owner"),
        approver=metadata_ref(approver, field="approver"),
        connector_evidence_state=require_state(connector_evidence_state, field="connector_evidence_state", allowed=("observed_available", "caller_asserted_available", "unavailable", "unknown")),
        connector_evidence_refs=metadata_refs(connector_evidence_refs, field="connector_evidence_refs", required=connector_evidence_state == "observed_available"),
        timing_state=require_state(timing_state, field="timing_state", allowed=("not_scheduled", "scheduled_by_external_owner")),
        stop_condition_refs=metadata_refs(stop_condition_refs, field="stop_condition_refs", required=True),
        analysis_cancellation=analysis_cancellation_record(analysis_cancellation),
    )


def validate_brief(record: Mapping[str, Any]) -> list[str]:
    errors = _base_errors(record)
    for field in ("lifecycle_stage", "target_behavior_ref", "target_segment_ref", "baseline_ref", "experiment_budget_ref", "decision_owner"):
        errors.extend(ref_errors(record.get(field), field=field, required=True))
    for field in ("available_surface_refs", "observed_evidence_refs", "hypotheses", "non_goals"):
        errors.extend(refs_errors(record.get(field), field=field, required=True))
    return errors


def validate_audience(record: Mapping[str, Any]) -> list[str]:
    errors = _base_errors(record)
    errors.extend(state_errors(record.get("audience_identity_state"), field="audience_identity_state", allowed=("known", "unknown")))
    errors.extend(state_errors(record.get("canonical_event_state"), field="canonical_event_state", allowed=("known", "unknown")))
    for field in ("entry_condition_ref", "exit_condition_ref", "idempotency_key_ref"):
        errors.extend(ref_errors(record.get(field), field=field, required=True))
    errors.extend(refs_errors(record.get("exclusion_refs"), field="exclusion_refs", required=True))
    errors.extend(ref_errors(record.get("audience_identity_ref"), field="audience_identity_ref", required=record.get("audience_identity_state") == "known"))
    errors.extend(ref_errors(record.get("canonical_event_schema_ref"), field="canonical_event_schema_ref", required=record.get("canonical_event_state") == "known"))
    errors.extend(state_errors(record.get("reentry_policy"), field="reentry_policy", allowed=("never", "after_exit_only")))
    errors.extend(state_errors(record.get("collision_policy"), field="collision_policy", allowed=("hold_on_overlap", "separate_interventions")))
    return errors


def validate_safety(record: Mapping[str, Any]) -> list[str]:
    errors = _base_errors(record)
    for field in ("consent_state", "suppression_state", "frequency_state", "channel_eligibility_state", "quiet_hours_state", "locale_state", "legal_tenant_state"):
        errors.extend(state_errors(record.get(field), field=field, allowed=_SAFETY_STATES))
    errors.extend(state_errors(record.get("suppression_precedence"), field="suppression_precedence", allowed=("suppression_overrides_all",)))
    for field in ("preference_policy_ref", "global_frequency_budget_ref", "campaign_frequency_budget_ref"):
        errors.extend(ref_errors(record.get(field), field=field, required=True))
    errors.extend(throttle_grouping_errors(record.get("throttle_grouping")))
    errors.extend(state_errors(record.get("workflow_content_state"), field="workflow_content_state", allowed=WORKFLOW_CONTENT_STATES))
    errors.extend(state_errors(record.get("mutation_route"), field="mutation_route", allowed=MUTATION_ROUTES))
    errors.extend(state_errors(record.get("promotion_decision_state"), field="promotion_decision_state", allowed=PROMOTION_DECISION_STATES))
    errors.extend(state_errors(record.get("promotion_result_state"), field="promotion_result_state", allowed=PROMOTION_RESULT_STATES))
    return errors


def validate_experiment(record: Mapping[str, Any]) -> list[str]:
    errors = _base_errors(record)
    for field in ("treatment_ref", "control_ref", "assignment_unit", "exposure_unit", "assignment_event_ref", "actual_exposure_event_ref", "primary_metric_ref"):
        errors.extend(ref_errors(record.get(field), field=field, required=True))
    errors.extend(state_errors(record.get("sticky_assignment_policy"), field="sticky_assignment_policy", allowed=("sticky",)))
    errors.extend(refs_errors(record.get("guardrail_metric_refs"), field="guardrail_metric_refs", required=True))
    errors.extend(state_errors(record.get("holdout_state"), field="holdout_state", allowed=("preserved", "unknown")))
    errors.extend(ref_errors(record.get("holdout_rationale_ref"), field="holdout_rationale_ref", required=record.get("holdout_state") == "preserved"))
    errors.extend(state_errors(record.get("data_health_state"), field="data_health_state", allowed=("healthy", "broken", "unknown")))
    errors.extend(refs_errors(record.get("rollback_conditions"), field="rollback_conditions", required=True))
    errors.extend(refs_errors(record.get("pause_condition_refs"), field="pause_condition_refs", required=True))
    errors.extend(state_errors(record.get("approval_state"), field="approval_state", allowed=("approved", "pending", "rejected")))
    days = record.get("minimum_runtime_days")
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        errors.append("minimum_runtime_days must be at least one")
    return errors


def validate_handoff(record: Mapping[str, Any]) -> list[str]:
    errors = _base_errors(record)
    actions = record.get("proposed_action_refs")
    kinds = record.get("proposed_action_kinds")
    errors.extend(refs_errors(actions, field="proposed_action_refs", required=True))
    if not isinstance(actions, list) or not isinstance(kinds, list):
        errors.append("proposed_action_kinds must match proposed_action_refs")
    elif len(kinds) != len(actions):
        errors.append("proposed_action_kinds must match proposed_action_refs")
    elif not all(kind in _ACTION_KINDS for kind in kinds):
        errors.append("proposed_action_kinds is invalid")
    for field in ("action_owner", "approver"):
        errors.extend(ref_errors(record.get(field), field=field, required=True))
    errors.extend(state_errors(record.get("connector_evidence_state"), field="connector_evidence_state", allowed=("observed_available", "caller_asserted_available", "unavailable", "unknown")))
    errors.extend(refs_errors(record.get("connector_evidence_refs"), field="connector_evidence_refs", required=record.get("connector_evidence_state") == "observed_available"))
    errors.extend(state_errors(record.get("timing_state"), field="timing_state", allowed=("not_scheduled", "scheduled_by_external_owner")))
    errors.extend(refs_errors(record.get("stop_condition_refs"), field="stop_condition_refs", required=True))
    errors.extend(analysis_cancellation_errors(record.get("analysis_cancellation")))
    return errors


def _base_errors(record: Mapping[str, Any]) -> list[str]:
    return ref_errors(record.get("lifecycle_growth_id"), field="lifecycle_growth_id", required=True)
