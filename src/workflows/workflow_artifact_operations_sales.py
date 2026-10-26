"""Known JSON adapters for the typed sales-pipeline workflow boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .sales_pipeline_models import (
    AccountFollowUp,
    CriterionEvidence,
    CurrencyConversionBasis,
    ForecastCategoryDefinition,
    NextStep,
    OpportunitySnapshot,
    OutcomeObservation,
    PriorForecastSnapshot,
    ProposedCrmCorrection,
    RenewalSignal,
    SalesPipelineHandoffInput,
    SalesPipelineReviewInput,
    ScenarioDefinition,
    StageDefinition,
)


class SalesWorkflowArtifactInputError(ValueError):
    """Raised when bounded JSON cannot become a known sales model."""


def sales_review_input(payload: Mapping[str, Any]) -> SalesPipelineReviewInput:
    """Adapt one closed JSON record into the existing typed review input."""
    _only_keys(payload, _REVIEW_KEYS, "sales review")
    return SalesPipelineReviewInput(
        source_ref=_string(payload, "source_ref"), snapshot_as_of=_string(payload, "snapshot_as_of"),
        reviewed_at=_string(payload, "reviewed_at"), horizon_end=_string(payload, "horizon_end"),
        cohort_id=_string(payload, "cohort_id"), included_motions=_strings(payload, "included_motions"),
        included_owner_ids=_strings(payload, "included_owner_ids"), reporting_currency=_string(payload, "reporting_currency"),
        amount_semantics=_string(payload, "amount_semantics"),
        stage_definitions=tuple(_stage(row) for row in _rows(payload, "stage_definitions")),
        forecast_definitions=tuple(_forecast(row) for row in _rows(payload, "forecast_definitions")),
        scenario_definitions=tuple(_scenario(row) for row in _rows(payload, "scenario_definitions")),
        opportunities=tuple(_opportunity(row) for row in _rows(payload, "opportunities")),
        decision_owner_id=_string(payload, "decision_owner_id"),
        freshness_limit_days=_integer(payload, "freshness_limit_days", 7),
        concentration_limit_basis_points=_integer(payload, "concentration_limit_basis_points", 5000),
        conversion_bases=tuple(_conversion(row) for row in _rows(payload, "conversion_bases", False)),
        prior_forecasts=tuple(_prior(row) for row in _rows(payload, "prior_forecasts", False)),
        observed_outcomes=tuple(_outcome(row) for row in _rows(payload, "observed_outcomes", False)),
        renewal_signals=tuple(_renewal(row) for row in _rows(payload, "renewal_signals", False)),
    )


def sales_handoff_input(payload: Mapping[str, Any], review: SalesPipelineReviewInput) -> SalesPipelineHandoffInput:
    """Adapt explicit actions only after the real typed review evaluates."""
    _only_keys(payload, {"selected_follow_ups", "proposed_corrections", "decision_owner_id"}, "sales handoff")
    from .sales_pipeline_review import evaluate_sales_pipeline_review

    return SalesPipelineHandoffInput(
        evaluation=evaluate_sales_pipeline_review(review),
        selected_follow_ups=tuple(_follow_up(row) for row in _rows(payload, "selected_follow_ups")),
        proposed_corrections=tuple(_correction(row) for row in _rows(payload, "proposed_corrections")),
        decision_owner_id=_string(payload, "decision_owner_id"),
    )


def _stage(row: Mapping[str, Any]) -> StageDefinition:
    _only_keys(row, {"stage_id", "meaning", "required_exit_criteria", "stall_after_days"}, "stage definition")
    return StageDefinition(_string(row, "stage_id"), _string(row, "meaning"), _strings(row, "required_exit_criteria"), _integer(row, "stall_after_days"))


def _forecast(row: Mapping[str, Any]) -> ForecastCategoryDefinition:
    _only_keys(row, {"category_id", "meaning", "included_scenario_ids"}, "forecast definition")
    return ForecastCategoryDefinition(_string(row, "category_id"), _string(row, "meaning"), _strings(row, "included_scenario_ids"))


def _scenario(row: Mapping[str, Any]) -> ScenarioDefinition:
    _only_keys(row, {"scenario_id", "meaning"}, "scenario definition")
    return ScenarioDefinition(_string(row, "scenario_id"), _string(row, "meaning"))


def _opportunity(row: Mapping[str, Any]) -> OpportunitySnapshot:
    _only_keys(row, _OPPORTUNITY_KEYS, "opportunity")
    next_step = row.get("next_step")
    return OpportunitySnapshot(
        opportunity_id=_string(row, "opportunity_id"), account_id=_string(row, "account_id"), owner_id=_string(row, "owner_id"),
        motion_id=_string(row, "motion_id"), stage_id=_string(row, "stage_id"),
        seller_forecast_category_id=_string(row, "seller_forecast_category_id"),
        seller_probability_basis_points=_optional_integer(row, "seller_probability_basis_points"), amount_minor=_integer(row, "amount_minor"),
        currency=_string(row, "currency"), stage_entered_on=_string(row, "stage_entered_on"), expected_close_on=_string(row, "expected_close_on"),
        previous_stage_id=_string(row, "previous_stage_id", ""), previous_close_on=_string(row, "previous_close_on", ""),
        exit_criteria=tuple(_criterion(item) for item in _rows(row, "exit_criteria", False)), next_step=_next_step(next_step),
        buyer_commitment_state=_string(row, "buyer_commitment_state", "none_observed"),
        buyer_commitment_evidence_refs=_strings(row, "buyer_commitment_evidence_refs", False),
    )


def _criterion(row: Mapping[str, Any]) -> CriterionEvidence:
    _only_keys(row, {"criterion_id", "state", "evidence_refs"}, "criterion evidence")
    return CriterionEvidence(_string(row, "criterion_id"), _string(row, "state"), _strings(row, "evidence_refs", False))


def _next_step(value: Any) -> NextStep | None:
    if value is None:
        return None
    row = _mapping(value, "next_step")
    _only_keys(row, {"action_id", "due_on", "evidence_refs"}, "next_step")
    return NextStep(_string(row, "action_id"), _string(row, "due_on"), _strings(row, "evidence_refs", False))


def _conversion(row: Mapping[str, Any]) -> CurrencyConversionBasis:
    _only_keys(row, {"source_currency", "target_currency", "numerator", "denominator", "evidence_ref", "observed_at"}, "conversion basis")
    return CurrencyConversionBasis(_string(row, "source_currency"), _string(row, "target_currency"), _integer(row, "numerator"), _integer(row, "denominator"), _string(row, "evidence_ref"), _string(row, "observed_at"))


def _prior(row: Mapping[str, Any]) -> PriorForecastSnapshot:
    _only_keys(row, {"source_ref", "snapshot_as_of", "cohort_id", "horizon_end", "opportunity_id", "seller_forecast_category_id", "seller_probability_basis_points", "amount_minor", "currency", "evidence_refs"}, "prior forecast")
    return PriorForecastSnapshot(_string(row, "source_ref"), _string(row, "snapshot_as_of"), _string(row, "cohort_id"), _string(row, "horizon_end"), _string(row, "opportunity_id"), _string(row, "seller_forecast_category_id"), _optional_integer(row, "seller_probability_basis_points"), _integer(row, "amount_minor"), _string(row, "currency"), _strings(row, "evidence_refs"))


def _outcome(row: Mapping[str, Any]) -> OutcomeObservation:
    _only_keys(row, {"source_ref", "observed_at", "cohort_id", "horizon_end", "opportunity_id", "outcome", "reason_id", "amount_minor", "currency", "evidence_refs"}, "outcome observation")
    return OutcomeObservation(_string(row, "source_ref"), _string(row, "observed_at"), _string(row, "cohort_id"), _string(row, "horizon_end"), _string(row, "opportunity_id"), _string(row, "outcome"), _string(row, "reason_id"), _integer(row, "amount_minor"), _string(row, "currency"), _strings(row, "evidence_refs"))


def _renewal(row: Mapping[str, Any]) -> RenewalSignal:
    _only_keys(row, {"source_ref", "account_id", "owner_id", "renewal_on", "health_state", "utilization_state", "support_state", "budget_state", "staffing_state", "evidence_refs", "expansion_hypothesis_id"}, "renewal signal")
    return RenewalSignal(_string(row, "source_ref"), _string(row, "account_id"), _string(row, "owner_id"), _string(row, "renewal_on"), _string(row, "health_state"), _string(row, "utilization_state"), _string(row, "support_state"), _string(row, "budget_state"), _string(row, "staffing_state"), _strings(row, "evidence_refs"), _string(row, "expansion_hypothesis_id", ""))


def _follow_up(row: Mapping[str, Any]) -> AccountFollowUp:
    _only_keys(row, {"opportunity_id", "owner_id", "due_on", "action_id", "exit_criterion_id", "evidence_refs"}, "account follow-up")
    return AccountFollowUp(_string(row, "opportunity_id"), _string(row, "owner_id"), _string(row, "due_on"), _string(row, "action_id"), _string(row, "exit_criterion_id"), _strings(row, "evidence_refs"))


def _correction(row: Mapping[str, Any]) -> ProposedCrmCorrection:
    _only_keys(row, {"object_id", "object_type", "field_id", "proposed_value", "evidence_refs", "owner_id", "approval_state"}, "CRM correction")
    return ProposedCrmCorrection(_string(row, "object_id"), _string(row, "object_type"), _string(row, "field_id"), _string(row, "proposed_value"), _strings(row, "evidence_refs"), _string(row, "owner_id"), _string(row, "approval_state"))


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise SalesWorkflowArtifactInputError(f"{field} must be an object")
    return value


def _rows(payload: Mapping[str, Any], field: str, required: bool = True) -> tuple[Mapping[str, Any], ...]:
    value = payload.get(field)
    if value is None and not required:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SalesWorkflowArtifactInputError(f"{field} must be an array")
    return tuple(_mapping(item, f"{field} item") for item in value)


def _strings(payload: Mapping[str, Any], field: str, required: bool = True) -> tuple[str, ...]:
    value = payload.get(field)
    if value is None and not required:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not all(isinstance(item, str) for item in value):
        raise SalesWorkflowArtifactInputError(f"{field} must be an array of strings")
    return tuple(item for item in value if isinstance(item, str))


def _string(payload: Mapping[str, Any], field: str, default: str | None = None) -> str:
    value = payload.get(field, default)
    if not isinstance(value, str):
        raise SalesWorkflowArtifactInputError(f"{field} must be a string")
    return value


def _integer(payload: Mapping[str, Any], field: str, default: int | None = None) -> int:
    value = payload.get(field, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SalesWorkflowArtifactInputError(f"{field} must be an integer")
    return value


def _optional_integer(payload: Mapping[str, Any], field: str) -> int | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise SalesWorkflowArtifactInputError(f"{field} must be an integer or null")
    return value


def _only_keys(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    if set(payload) - allowed:
        raise SalesWorkflowArtifactInputError(f"{label} has unsupported fields")


_REVIEW_KEYS = {
    "source_ref", "snapshot_as_of", "reviewed_at", "horizon_end", "cohort_id", "included_motions", "included_owner_ids", "reporting_currency", "amount_semantics", "stage_definitions", "forecast_definitions", "scenario_definitions", "opportunities", "decision_owner_id", "freshness_limit_days", "concentration_limit_basis_points", "conversion_bases", "prior_forecasts", "observed_outcomes", "renewal_signals",
}
_OPPORTUNITY_KEYS = {
    "opportunity_id", "account_id", "owner_id", "motion_id", "stage_id", "seller_forecast_category_id", "seller_probability_basis_points", "amount_minor", "currency", "stage_entered_on", "expected_close_on", "previous_stage_id", "previous_close_on", "exit_criteria", "next_step", "buyer_commitment_state", "buyer_commitment_evidence_refs",
}
