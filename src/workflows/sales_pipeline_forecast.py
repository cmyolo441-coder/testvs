from __future__ import annotations

from collections import Counter, defaultdict

from .sales_pipeline_health import convert_to_reporting_amount, reporting_amount
from .sales_pipeline_models import (
    JsonObject,
    OutcomeObservation,
    PriorForecastSnapshot,
    SalesPipelineReviewInput,
)

_FORECAST_BOUNDARY = (
    "This assessment preserves supplied seller states, rule-derived scenarios, and observed buyer commitments separately. "
    "It is not an authoritative finance forecast, revenue booking, seller commitment, CRM mutation, or connector result."
)


def _calibration(request: SalesPipelineReviewInput) -> JsonObject:
    priors_by_id: defaultdict[str, list[PriorForecastSnapshot]] = defaultdict(list)
    outcomes_by_id: defaultdict[str, list[OutcomeObservation]] = defaultdict(list)
    for prior in request.prior_forecasts:
        if prior.cohort_id == request.cohort_id and prior.horizon_end == request.horizon_end:
            priors_by_id[prior.opportunity_id].append(prior)
    for outcome in request.observed_outcomes:
        if outcome.cohort_id == request.cohort_id and outcome.horizon_end == request.horizon_end:
            outcomes_by_id[outcome.opportunity_id].append(outcome)
    matched_ids = sorted(
        identifier
        for identifier in priors_by_id.keys() & outcomes_by_id.keys()
        if len(priors_by_id[identifier]) == 1
        and len({item.outcome for item in outcomes_by_id[identifier]}) == 1
    )
    if not matched_ids:
        return {
            "availability": "unavailable",
            "basis": "matching_observed_prior_and_outcome_cohort_required",
            "matched_opportunity_ids": [],
            "prior_forecast_amount_minor": None,
            "observed_won_amount_minor": None,
            "absolute_error_minor": None,
            "excluded_reason": "no_unambiguous_matching_cohort",
        }
    prior_amount = sum(
        convert_to_reporting_amount(
            request,
            priors_by_id[identifier][0].amount_minor,
            priors_by_id[identifier][0].currency,
        )
        for identifier in matched_ids
    )
    won_amount = sum(
        convert_to_reporting_amount(
            request,
            outcomes_by_id[identifier][0].amount_minor,
            outcomes_by_id[identifier][0].currency,
        )
        for identifier in matched_ids
        if outcomes_by_id[identifier][0].outcome == "won"
    )
    return {
        "availability": "available",
        "basis": "matching_observed_prior_and_outcome_cohort",
        "matched_opportunity_ids": matched_ids,
        "prior_forecast_amount_minor": prior_amount,
        "observed_won_amount_minor": won_amount,
        "absolute_error_minor": abs(prior_amount - won_amount),
        "excluded_reason": "",
    }


def build_sales_forecast_assessment(request: SalesPipelineReviewInput) -> JsonObject:
    category_definitions = {item.category_id: item for item in request.forecast_definitions}
    seller_states: list[JsonObject] = []
    buyer_commitments: list[JsonObject] = []
    scenario_amounts = {item.scenario_id: 0 for item in request.scenario_definitions}
    weighted_amount = 0
    supplied_probability_count = 0
    excluded_probability_ids: list[str] = []
    for opportunity in request.opportunities:
        amount = reporting_amount(request, opportunity)
        seller_states.append({
            "opportunity_id": opportunity.opportunity_id,
            "category_id": opportunity.seller_forecast_category_id,
            "probability_basis_points": opportunity.seller_probability_basis_points,
            "state_source": "seller_supplied_snapshot",
        })
        buyer_commitments.append({
            "opportunity_id": opportunity.opportunity_id,
            "state": opportunity.buyer_commitment_state,
            "evidence_refs": list(opportunity.buyer_commitment_evidence_refs),
            "state_source": "observed_buyer_evidence" if opportunity.buyer_commitment_evidence_refs else "none_observed",
        })
        for scenario_id in category_definitions[opportunity.seller_forecast_category_id].included_scenario_ids:
            scenario_amounts[scenario_id] += amount
        probability = opportunity.seller_probability_basis_points
        if probability is None:
            excluded_probability_ids.append(opportunity.opportunity_id)
        else:
            supplied_probability_count += 1
            weighted_amount += amount * probability // 10_000
    scenarios = [
        {
            "scenario_id": item.scenario_id,
            "amount_minor": scenario_amounts[item.scenario_id],
            "basis": "rule_derived_from_supplied_category_mapping",
            "buyer_commitment_inferred": False,
        }
        for item in request.scenario_definitions
    ]
    calibration = _calibration(request)
    return {
        "schema_version": "sales_forecast_assessment/v1",
        "status": "prepared_not_observed",
        "source_ref": request.source_ref,
        "as_of": request.snapshot_as_of,
        "reporting_currency": request.reporting_currency,
        "amount_semantics": request.amount_semantics,
        "seller_states": seller_states,
        "scenarios": scenarios,
        "buyer_commitments": buyer_commitments,
        "weighted_seller_scenario": {
            "amount_minor": weighted_amount if supplied_probability_count else None,
            "excluded_opportunity_ids": excluded_probability_ids,
            "basis": "supplied_seller_probabilities_only",
        },
        "calibration": calibration,
        "confidence": "bounded" if not excluded_probability_ids and calibration["availability"] == "available" else "limited",
        "evidence_limits": [
            "stage_does_not_create_probability",
            "stage_does_not_create_buyer_commitment",
            "scenario_is_not_seller_commitment",
            "unavailable_seller_probabilities_are_excluded",
        ],
        "authoritative_finance_forecast": False,
        "claim_boundary": _FORECAST_BOUNDARY,
    }


def build_sales_outcome_learning_annex(request: SalesPipelineReviewInput) -> JsonObject:
    bounded_outcomes = tuple(
        item
        for item in request.observed_outcomes
        if item.cohort_id == request.cohort_id and item.horizon_end == request.horizon_end
    )
    outcome_counts = Counter(item.outcome for item in bounded_outcomes)
    reason_counts = Counter(item.reason_id for item in bounded_outcomes if item.reason_id)
    outcomes_by_id: defaultdict[str, set[str]] = defaultdict(set)
    for item in bounded_outcomes:
        outcomes_by_id[item.opportunity_id].add(item.outcome)
    contradictions = [
        {"opportunity_id": identifier, "observed_outcomes": sorted(outcomes)}
        for identifier, outcomes in sorted(outcomes_by_id.items())
        if len(outcomes) > 1
    ]
    evidence_gaps = [
        {"opportunity_id": item.opportunity_id, "gap": "missing_reason"}
        for item in bounded_outcomes
        if not item.reason_id
    ]
    follow_ups = [
        {"opportunity_id": item["opportunity_id"], "action_id": "resolve_outcome_contradiction"}
        for item in contradictions
    ] + [
        {"opportunity_id": item["opportunity_id"], "action_id": "research_missing_outcome_reason"}
        for item in evidence_gaps
    ]
    return {
        "schema_version": "sales_outcome_learning_annex/v1",
        "status": "prepared_not_observed",
        "source_ref": request.source_ref,
        "cohort_id": request.cohort_id,
        "horizon_end": request.horizon_end,
        "outcome_counts": [{"outcome": key, "count": value} for key, value in sorted(outcome_counts.items())],
        "reason_counts": [{"reason_id": key, "count": value} for key, value in sorted(reason_counts.items())],
        "contradictions": contradictions,
        "evidence_gaps": evidence_gaps,
        "research_follow_ups": follow_ups,
        "causal_claims_established": False,
        "claim_boundary": (
            "This optional annex summarizes supplied observed outcomes by bounded cohort. It does not infer causes, retrieve "
            "customer material, replace feedback triage, or prove that a proposed follow-up was performed."
        ),
    }
