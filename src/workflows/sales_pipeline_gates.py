from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import re
from typing import Final

from .sales_pipeline_models import SalesPipelineReviewInput
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

_TOKEN_PATTERN: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{1,95}$")
_CURRENCY_PATTERN: Final = re.compile(r"^[A-Z]{3}$")
_ALLOWED_CRITERION_STATES: Final = frozenset(("met", "not_met", "unknown"))
_ALLOWED_COMMITMENT_STATES: Final = frozenset(
    ("none_observed", "interest_observed", "next_step_observed", "purchase_commitment_observed")
)
_ALLOWED_OUTCOMES: Final = frozenset(("won", "lost", "unqualified"))
_ALLOWED_SIGNAL_STATES: Final = frozenset(("observed_positive", "observed_neutral", "observed_negative", "unknown"))


def parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def safe_token(value: str) -> bool:
    return bool(_TOKEN_PATTERN.fullmatch(value)) and "://" not in value


def safe_text(value: str) -> bool:
    return bool(value) and len(value) <= 160 and "\n" not in value and "://" not in value and "@" not in value


def _definition_errors(request: SalesPipelineReviewInput) -> list[str]:
    errors: list[str] = []
    if not request.stage_definitions:
        errors.append("undefined_stage_definitions")
    if not request.forecast_definitions:
        errors.append("undefined_forecast_definitions")
    if not request.scenario_definitions:
        errors.append("undefined_scenario_definitions")
    stage_ids = [item.stage_id for item in request.stage_definitions]
    forecast_ids = [item.category_id for item in request.forecast_definitions]
    scenario_ids = [item.scenario_id for item in request.scenario_definitions]
    for label, values in (("stage", stage_ids), ("forecast", forecast_ids), ("scenario", scenario_ids)):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        errors.extend(f"duplicate_{label}_definition:{value}" for value in duplicates)
    for stage in request.stage_definitions:
        if not safe_token(stage.stage_id) or not safe_text(stage.meaning) or stage.stall_after_days < 1:
            errors.append(f"invalid_stage_definition:{stage.stage_id}")
        if (
            not stage.required_exit_criteria
            or any(not safe_token(item) for item in stage.required_exit_criteria)
            or len(set(stage.required_exit_criteria)) != len(stage.required_exit_criteria)
        ):
            errors.append(f"undefined_exit_criteria:{stage.stage_id}")
    known_scenarios = set(scenario_ids)
    for forecast in request.forecast_definitions:
        if not safe_token(forecast.category_id) or not safe_text(forecast.meaning):
            errors.append(f"invalid_forecast_definition:{forecast.category_id}")
        if any(item not in known_scenarios for item in forecast.included_scenario_ids):
            errors.append(f"undefined_scenario_mapping:{forecast.category_id}")
    for scenario in request.scenario_definitions:
        if not safe_token(scenario.scenario_id) or not safe_text(scenario.meaning):
            errors.append(f"invalid_scenario_definition:{scenario.scenario_id}")
    return errors


def _opportunity_errors(request: SalesPipelineReviewInput) -> list[str]:
    errors: list[str] = []
    stage_ids = {item.stage_id for item in request.stage_definitions}
    forecast_ids = {item.category_id for item in request.forecast_definitions}
    opportunity_ids = [item.opportunity_id for item in request.opportunities]
    duplicates = sorted(value for value, count in Counter(opportunity_ids).items() if count > 1)
    errors.extend(f"duplicate_opportunity_id:{value}" for value in duplicates)
    for item in request.opportunities:
        if not all(safe_token(value) for value in (item.opportunity_id, item.account_id, item.motion_id)):
            errors.append(f"unsafe_opportunity_metadata:{item.opportunity_id}")
        if not item.owner_id:
            errors.append(f"missing_owner:{item.opportunity_id}")
        elif not safe_token(item.owner_id) or item.owner_id not in request.included_owner_ids:
            errors.append(f"unknown_owner:{item.opportunity_id}")
        if item.motion_id not in request.included_motions:
            errors.append(f"unknown_motion:{item.opportunity_id}")
        if item.stage_id not in stage_ids:
            errors.append(f"undefined_stage:{item.opportunity_id}")
        if item.previous_stage_id and item.previous_stage_id not in stage_ids:
            errors.append(f"undefined_previous_stage:{item.opportunity_id}")
        if item.seller_forecast_category_id not in forecast_ids:
            errors.append(f"undefined_forecast_category:{item.opportunity_id}")
        probability = item.seller_probability_basis_points
        if probability is not None and not 0 <= probability <= 10_000:
            errors.append(f"unsupported_probability:{item.opportunity_id}")
        if item.amount_minor < 0 or not _CURRENCY_PATTERN.fullmatch(item.currency):
            errors.append(f"invalid_amount_or_currency:{item.opportunity_id}")
        if parse_date(item.stage_entered_on) is None or parse_date(item.expected_close_on) is None:
            errors.append(f"invalid_opportunity_dates:{item.opportunity_id}")
        if item.previous_close_on and parse_date(item.previous_close_on) is None:
            errors.append(f"invalid_previous_close_date:{item.opportunity_id}")
        if item.buyer_commitment_state not in _ALLOWED_COMMITMENT_STATES:
            errors.append(f"invalid_buyer_commitment:{item.opportunity_id}")
        if item.buyer_commitment_state != "none_observed" and not item.buyer_commitment_evidence_refs:
            errors.append(f"buyer_commitment_without_observed_evidence:{item.opportunity_id}")
        if any(not safe_token(ref) for ref in item.buyer_commitment_evidence_refs):
            errors.append(f"unsafe_buyer_commitment_evidence:{item.opportunity_id}")
        criterion_ids = [criterion.criterion_id for criterion in item.exit_criteria]
        if len(set(criterion_ids)) != len(criterion_ids):
            errors.append(f"duplicate_exit_criterion:{item.opportunity_id}")
        for criterion in item.exit_criteria:
            if (
                not safe_token(criterion.criterion_id)
                or criterion.state not in _ALLOWED_CRITERION_STATES
                or any(not safe_token(ref) for ref in criterion.evidence_refs)
            ):
                errors.append(f"invalid_exit_criterion:{item.opportunity_id}")
            if criterion.state == "met" and not criterion.evidence_refs:
                errors.append(f"met_exit_criterion_without_evidence:{item.opportunity_id}")
        if item.next_step is not None:
            if (
                not safe_token(item.next_step.action_id)
                or parse_date(item.next_step.due_on) is None
                or any(not safe_token(ref) for ref in item.next_step.evidence_refs)
            ):
                errors.append(f"invalid_next_step:{item.opportunity_id}")
    return errors


def _conversion_errors(request: SalesPipelineReviewInput) -> list[str]:
    errors: list[str] = []
    reviewed_at = parse_timestamp(request.reviewed_at)
    pairs = [(item.source_currency, item.target_currency) for item in request.conversion_bases]
    bases = {pair: item for pair, item in zip(pairs, request.conversion_bases, strict=True)}
    duplicate_pairs = sorted(pair for pair, count in Counter(pairs).items() if count > 1)
    errors.extend(f"duplicate_conversion_basis:{source}:{target}" for source, target in duplicate_pairs)
    for basis in request.conversion_bases:
        observed_at = parse_timestamp(basis.observed_at)
        if (
            not _CURRENCY_PATTERN.fullmatch(basis.source_currency)
            or basis.target_currency != request.reporting_currency
            or basis.numerator <= 0
            or basis.denominator <= 0
            or not safe_token(basis.evidence_ref)
            or observed_at is None
            or reviewed_at is None
            or observed_at > reviewed_at
        ):
            errors.append(f"invalid_conversion_basis:{basis.source_currency}")
    currencies = {
        item.currency for item in (*request.opportunities, *request.prior_forecasts, *request.observed_outcomes)
        if item.currency != request.reporting_currency
    }
    for currency in sorted(currencies):
        if (currency, request.reporting_currency) not in bases:
            errors.append(f"mixed_currency_without_observed_conversion:{currency}")
    return errors


def _optional_evidence_errors(request: SalesPipelineReviewInput) -> list[str]:
    errors: list[str] = []
    for prior in request.prior_forecasts:
        tokens = (
            prior.source_ref, prior.cohort_id, prior.opportunity_id, prior.seller_forecast_category_id, *prior.evidence_refs,
        )
        if (
            any(not safe_token(token) for token in tokens)
            or not prior.evidence_refs
            or prior.amount_minor < 0
            or not _CURRENCY_PATTERN.fullmatch(prior.currency)
        ):
            errors.append(f"invalid_prior_forecast:{prior.opportunity_id}")
        if parse_timestamp(prior.snapshot_as_of) is None or parse_date(prior.horizon_end) is None:
            errors.append(f"invalid_prior_forecast_dates:{prior.opportunity_id}")
        if prior.seller_probability_basis_points is not None and not 0 <= prior.seller_probability_basis_points <= 10_000:
            errors.append(f"unsupported_prior_probability:{prior.opportunity_id}")
    for outcome in request.observed_outcomes:
        tokens = (outcome.source_ref, outcome.cohort_id, outcome.opportunity_id, *outcome.evidence_refs)
        if (
            any(not safe_token(token) for token in tokens)
            or outcome.outcome not in _ALLOWED_OUTCOMES
            or not outcome.evidence_refs
            or parse_timestamp(outcome.observed_at) is None
            or parse_date(outcome.horizon_end) is None
            or outcome.amount_minor < 0
            or not _CURRENCY_PATTERN.fullmatch(outcome.currency)
            or (outcome.reason_id and not safe_token(outcome.reason_id))
        ):
            errors.append(f"invalid_observed_outcome:{outcome.opportunity_id}")
    for signal in request.renewal_signals:
        states = (signal.health_state, signal.utilization_state, signal.support_state, signal.budget_state, signal.staffing_state)
        tokens = (signal.source_ref, signal.account_id, signal.owner_id, *signal.evidence_refs)
        if any(state not in _ALLOWED_SIGNAL_STATES for state in states) or any(not safe_token(token) for token in tokens):
            errors.append(f"invalid_renewal_signal:{signal.account_id}")
        if (
            parse_date(signal.renewal_on) is None
            or (signal.expansion_hypothesis_id and not safe_token(signal.expansion_hypothesis_id))
        ):
            errors.append(f"invalid_renewal_identity:{signal.account_id}")
    return errors


def sales_pipeline_gate_errors(request: SalesPipelineReviewInput) -> tuple[str, ...]:
    errors: list[str] = []
    snapshot_as_of = parse_timestamp(request.snapshot_as_of)
    reviewed_at = parse_timestamp(request.reviewed_at)
    horizon_end = parse_date(request.horizon_end)
    if not safe_token(request.source_ref):
        errors.append("unsafe_source_ref")
    if snapshot_as_of is None:
        errors.append("invalid_snapshot_as_of")
    if reviewed_at is None:
        errors.append("invalid_reviewed_at")
    if snapshot_as_of is not None and reviewed_at is not None:
        age_seconds = int((reviewed_at - snapshot_as_of).total_seconds())
        if age_seconds < 0:
            errors.append("future_snapshot")
        elif age_seconds > request.freshness_limit_days * 86_400:
            errors.append("stale_snapshot")
    if horizon_end is None or snapshot_as_of is None or horizon_end < snapshot_as_of.date():
        errors.append("invalid_review_horizon")
    if (
        not safe_token(request.cohort_id)
        or not request.included_motions
        or not request.included_owner_ids
        or not request.opportunities
        or len(set(request.included_owner_ids)) != len(request.included_owner_ids)
    ):
        errors.append("invalid_cohort_scope")
    if any(not safe_token(item) for item in (*request.included_motions, *request.included_owner_ids)):
        errors.append("unsafe_scope_metadata")
    if not _CURRENCY_PATTERN.fullmatch(request.reporting_currency):
        errors.append("invalid_reporting_currency")
    if not safe_token(request.amount_semantics):
        errors.append("undefined_amount_semantics")
    if not safe_token(request.decision_owner_id):
        errors.append("missing_decision_owner")
    if request.freshness_limit_days < 1 or not 1 <= request.concentration_limit_basis_points <= 10_000:
        errors.append("invalid_review_thresholds")
    errors.extend(_definition_errors(request))
    errors.extend(_opportunity_errors(request))
    errors.extend(_conversion_errors(request))
    errors.extend(_optional_evidence_errors(request))
    return tuple(dict.fromkeys(errors))
