from __future__ import annotations

from .sales_pipeline_artifacts import (
    validate_sales_forecast_assessment,
    validate_sales_outcome_learning_annex,
    validate_sales_pipeline_artifact,
    validate_sales_pipeline_handoff,
    validate_sales_pipeline_health,
    validate_sales_pipeline_scope,
    validate_sales_renewal_risk_annex,
)
from .sales_pipeline_forecast import build_sales_forecast_assessment, build_sales_outcome_learning_annex
from .sales_pipeline_gates import parse_timestamp, sales_pipeline_gate_errors
from .sales_pipeline_handoff import prepare_sales_pipeline_handoff
from .sales_pipeline_health import build_sales_pipeline_health
from .sales_pipeline_models import (
    AccountFollowUp,
    CriterionEvidence,
    CurrencyConversionBasis,
    EvaluatedSalesPipelineReview,
    ForecastCategoryDefinition,
    JsonObject,
    NextStep,
    OpportunitySnapshot,
    OutcomeObservation,
    PreparedSalesPipelineReview,
    PriorForecastSnapshot,
    ProposedCrmCorrection,
    RenewalSignal,
    SalesPipelineContractError,
    SalesPipelineHandoffInput,
    SalesPipelineReviewInput,
    ScenarioDefinition,
    StageDefinition,
)
from .sales_pipeline_renewals import build_sales_renewal_risk_annex

_SCOPE_BOUNDARY = (
    "This scope records bounded metadata and opaque identifiers for a supplied snapshot. It retains no raw export and is not "
    "source retrieval, CRM mutation, sync, alerting, calculation, revenue booking, or authoritative finance evidence."
)


def _scope_artifact(request: SalesPipelineReviewInput, errors: tuple[str, ...]) -> JsonObject:
    snapshot = parse_timestamp(request.snapshot_as_of)
    reviewed = parse_timestamp(request.reviewed_at)
    age_seconds = int((reviewed - snapshot).total_seconds()) if snapshot is not None and reviewed is not None else None
    return {
        "schema_version": "sales_pipeline_scope/v1",
        "status": "HOLD" if errors else "READY",
        "source_ref": request.source_ref,
        "as_of": request.snapshot_as_of,
        "reviewed_at": request.reviewed_at,
        "cohort_id": request.cohort_id,
        "horizon_end": request.horizon_end,
        "included_motions": list(request.included_motions),
        "included_owner_ids": list(request.included_owner_ids),
        "reporting_currency": request.reporting_currency,
        "amount_semantics": request.amount_semantics,
        "stage_definitions": [
            {
                "stage_id": item.stage_id,
                "meaning": item.meaning,
                "required_exit_criteria": list(item.required_exit_criteria),
                "stall_after_days": item.stall_after_days,
            }
            for item in request.stage_definitions
        ],
        "forecast_definitions": [
            {
                "category_id": item.category_id,
                "meaning": item.meaning,
                "included_scenario_ids": list(item.included_scenario_ids),
            }
            for item in request.forecast_definitions
        ],
        "scenario_definitions": [
            {"scenario_id": item.scenario_id, "meaning": item.meaning}
            for item in request.scenario_definitions
        ],
        "conversion_bases": [
            {
                "source_currency": item.source_currency,
                "target_currency": item.target_currency,
                "numerator": item.numerator,
                "denominator": item.denominator,
                "evidence_ref": item.evidence_ref,
                "observed_at": item.observed_at,
            }
            for item in request.conversion_bases
        ],
        "freshness": {
            "limit_days": request.freshness_limit_days,
            "age_seconds": age_seconds,
            "status": "invalid" if age_seconds is None else "fresh" if 0 <= age_seconds <= request.freshness_limit_days * 86_400 else "stale",
        },
        "known_data_quality_gaps": list(errors),
        "opportunity_count": len(request.opportunities),
        "decision_owner_id": request.decision_owner_id,
        "raw_export_retained": False,
        "claim_boundary": _SCOPE_BOUNDARY,
    }


def prepare_sales_pipeline_review(request: SalesPipelineReviewInput) -> PreparedSalesPipelineReview:
    errors = sales_pipeline_gate_errors(request)
    scope = _scope_artifact(request, errors)
    shape_errors = validate_sales_pipeline_artifact(scope)
    if shape_errors:
        errors = (*errors, *shape_errors)
    return PreparedSalesPipelineReview("HOLD" if errors else "READY", errors, scope)


def evaluate_sales_pipeline_review(request: SalesPipelineReviewInput) -> EvaluatedSalesPipelineReview:
    preparation = prepare_sales_pipeline_review(request)
    if preparation.status == "HOLD":
        return EvaluatedSalesPipelineReview(
            "HOLD", preparation.errors, preparation.scope, None, None, None, None,
        )
    health = build_sales_pipeline_health(request)
    forecast = build_sales_forecast_assessment(request)
    outcome_annex = (
        build_sales_outcome_learning_annex(request)
        if any(
            item.cohort_id == request.cohort_id and item.horizon_end == request.horizon_end
            for item in request.observed_outcomes
        )
        else None
    )
    renewal_annex = build_sales_renewal_risk_annex(request) if request.renewal_signals else None
    artifacts = (health, forecast, outcome_annex, renewal_annex)
    artifact_errors = tuple(
        error
        for artifact in artifacts
        if artifact is not None
        for error in validate_sales_pipeline_artifact(artifact)
    )
    if artifact_errors:
        return EvaluatedSalesPipelineReview(
            "HOLD", artifact_errors, preparation.scope, None, None, None, None,
        )
    return EvaluatedSalesPipelineReview(
        "READY", (), preparation.scope, health, forecast, outcome_annex, renewal_annex,
    )


__all__ = (
    "AccountFollowUp",
    "CriterionEvidence",
    "CurrencyConversionBasis",
    "EvaluatedSalesPipelineReview",
    "ForecastCategoryDefinition",
    "NextStep",
    "OpportunitySnapshot",
    "OutcomeObservation",
    "PreparedSalesPipelineReview",
    "PriorForecastSnapshot",
    "ProposedCrmCorrection",
    "RenewalSignal",
    "SalesPipelineContractError",
    "SalesPipelineHandoffInput",
    "SalesPipelineReviewInput",
    "ScenarioDefinition",
    "StageDefinition",
    "evaluate_sales_pipeline_review",
    "prepare_sales_pipeline_handoff",
    "prepare_sales_pipeline_review",
    "validate_sales_forecast_assessment",
    "validate_sales_outcome_learning_annex",
    "validate_sales_pipeline_artifact",
    "validate_sales_pipeline_handoff",
    "validate_sales_pipeline_health",
    "validate_sales_pipeline_scope",
    "validate_sales_renewal_risk_annex",
)
