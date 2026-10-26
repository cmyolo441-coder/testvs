from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

JsonValue: TypeAlias = str | int | float | bool | None | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SalesPipelineContractError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class StageDefinition:
    stage_id: str
    meaning: str
    required_exit_criteria: tuple[str, ...]
    stall_after_days: int


@dataclass(frozen=True, slots=True)
class ForecastCategoryDefinition:
    category_id: str
    meaning: str
    included_scenario_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioDefinition:
    scenario_id: str
    meaning: str


@dataclass(frozen=True, slots=True)
class CurrencyConversionBasis:
    source_currency: str
    target_currency: str
    numerator: int
    denominator: int
    evidence_ref: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class CriterionEvidence:
    criterion_id: str
    state: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NextStep:
    action_id: str
    due_on: str
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OpportunitySnapshot:
    opportunity_id: str
    account_id: str
    owner_id: str
    motion_id: str
    stage_id: str
    seller_forecast_category_id: str
    seller_probability_basis_points: int | None
    amount_minor: int
    currency: str
    stage_entered_on: str
    expected_close_on: str
    previous_stage_id: str = ""
    previous_close_on: str = ""
    exit_criteria: tuple[CriterionEvidence, ...] = ()
    next_step: NextStep | None = None
    buyer_commitment_state: str = "none_observed"
    buyer_commitment_evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PriorForecastSnapshot:
    source_ref: str
    snapshot_as_of: str
    cohort_id: str
    horizon_end: str
    opportunity_id: str
    seller_forecast_category_id: str
    seller_probability_basis_points: int | None
    amount_minor: int
    currency: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OutcomeObservation:
    source_ref: str
    observed_at: str
    cohort_id: str
    horizon_end: str
    opportunity_id: str
    outcome: str
    reason_id: str
    amount_minor: int
    currency: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RenewalSignal:
    source_ref: str
    account_id: str
    owner_id: str
    renewal_on: str
    health_state: str
    utilization_state: str
    support_state: str
    budget_state: str
    staffing_state: str
    evidence_refs: tuple[str, ...]
    expansion_hypothesis_id: str = ""


@dataclass(frozen=True, slots=True)
class SalesPipelineReviewInput:
    source_ref: str
    snapshot_as_of: str
    reviewed_at: str
    horizon_end: str
    cohort_id: str
    included_motions: tuple[str, ...]
    included_owner_ids: tuple[str, ...]
    reporting_currency: str
    amount_semantics: str
    stage_definitions: tuple[StageDefinition, ...]
    forecast_definitions: tuple[ForecastCategoryDefinition, ...]
    scenario_definitions: tuple[ScenarioDefinition, ...]
    opportunities: tuple[OpportunitySnapshot, ...]
    decision_owner_id: str
    freshness_limit_days: int = 7
    concentration_limit_basis_points: int = 5000
    conversion_bases: tuple[CurrencyConversionBasis, ...] = ()
    prior_forecasts: tuple[PriorForecastSnapshot, ...] = ()
    observed_outcomes: tuple[OutcomeObservation, ...] = ()
    renewal_signals: tuple[RenewalSignal, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedSalesPipelineReview:
    status: str
    errors: tuple[str, ...]
    scope: JsonObject


@dataclass(frozen=True, slots=True)
class EvaluatedSalesPipelineReview:
    status: str
    errors: tuple[str, ...]
    scope: JsonObject
    health: JsonObject | None
    forecast: JsonObject | None
    outcome_annex: JsonObject | None
    renewal_annex: JsonObject | None


@dataclass(frozen=True, slots=True)
class AccountFollowUp:
    opportunity_id: str
    owner_id: str
    due_on: str
    action_id: str
    exit_criterion_id: str
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProposedCrmCorrection:
    object_id: str
    object_type: str
    field_id: str
    proposed_value: str
    evidence_refs: tuple[str, ...]
    owner_id: str
    approval_state: str


@dataclass(frozen=True, slots=True)
class SalesPipelineHandoffInput:
    evaluation: EvaluatedSalesPipelineReview
    selected_follow_ups: tuple[AccountFollowUp, ...]
    proposed_corrections: tuple[ProposedCrmCorrection, ...]
    decision_owner_id: str
