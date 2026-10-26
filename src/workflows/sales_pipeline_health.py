from __future__ import annotations

from collections import defaultdict

from .sales_pipeline_gates import parse_date
from .sales_pipeline_models import (
    JsonObject,
    OpportunitySnapshot,
    SalesPipelineContractError,
    SalesPipelineReviewInput,
)

_HEALTH_BOUNDARY = (
    "This metadata-only health review is calculated from the supplied snapshot. It does not retrieve or retain the raw export, "
    "mutate CRM records, create alerts, contact accounts, or prove buyer or seller commitments."
)


def convert_to_reporting_amount(request: SalesPipelineReviewInput, amount_minor: int, currency: str) -> int:
    if currency == request.reporting_currency:
        return amount_minor
    basis = next(
        item
        for item in request.conversion_bases
        if item.source_currency == currency and item.target_currency == request.reporting_currency
    )
    return amount_minor * basis.numerator // basis.denominator


def reporting_amount(request: SalesPipelineReviewInput, opportunity: OpportunitySnapshot) -> int:
    return convert_to_reporting_amount(request, opportunity.amount_minor, opportunity.currency)


def _concentration(request: SalesPipelineReviewInput) -> tuple[JsonObject, list[JsonObject]]:
    account_amounts: defaultdict[str, int] = defaultdict(int)
    owner_amounts: defaultdict[str, int] = defaultdict(int)
    for opportunity in request.opportunities:
        amount = reporting_amount(request, opportunity)
        account_amounts[opportunity.account_id] += amount
        owner_amounts[opportunity.owner_id] += amount
    total = sum(account_amounts.values())

    def share_rows(amounts: dict[str, int] | defaultdict[str, int], id_field: str) -> list[JsonObject]:
        return [
            {id_field: identifier, "amount_minor": amount, "share_basis_points": amount * 10_000 // total if total else 0}
            for identifier, amount in sorted(amounts.items())
        ]

    account_shares = share_rows(account_amounts, "account_id")
    owner_shares = share_rows(owner_amounts, "owner_id")
    exceptions: list[JsonObject] = [
        {"account_id": identifier, "amount_minor": amount, "share_basis_points": amount * 10_000 // total if total else 0}
        for identifier, amount in sorted(account_amounts.items())
        if total and amount * 10_000 // total > request.concentration_limit_basis_points
    ]
    concentration: JsonObject = {
        "total_amount_minor": total,
        "account_shares": account_shares,
        "owner_shares": owner_shares,
        "threshold_basis_points": request.concentration_limit_basis_points,
        "exceptions": exceptions,
    }
    return concentration, exceptions


def build_sales_pipeline_health(request: SalesPipelineReviewInput) -> JsonObject:
    as_of = parse_date(request.snapshot_as_of[:10])
    if as_of is None:
        raise SalesPipelineContractError("sales pipeline health requires gated as-of metadata")
    stages = {item.stage_id: item for item in request.stage_definitions}
    movement: list[JsonObject] = []
    aging: list[JsonObject] = []
    stalls: list[JsonObject] = []
    slips: list[JsonObject] = []
    exit_gaps: list[JsonObject] = []
    next_steps: list[JsonObject] = []
    exceptions: list[JsonObject] = []
    for opportunity in request.opportunities:
        if opportunity.previous_stage_id and opportunity.previous_stage_id != opportunity.stage_id:
            movement.append({
                "opportunity_id": opportunity.opportunity_id,
                "from_stage_id": opportunity.previous_stage_id,
                "to_stage_id": opportunity.stage_id,
            })
        entered = parse_date(opportunity.stage_entered_on)
        if entered is None:
            raise SalesPipelineContractError("sales pipeline health requires gated stage dates")
        age_days = (as_of - entered).days
        age_row: JsonObject = {
            "opportunity_id": opportunity.opportunity_id,
            "stage_id": opportunity.stage_id,
            "age_days": age_days,
        }
        aging.append(age_row)
        if age_days >= stages[opportunity.stage_id].stall_after_days:
            stall = {**age_row, "threshold_days": stages[opportunity.stage_id].stall_after_days}
            stalls.append(stall)
            exceptions.append({"opportunity_id": opportunity.opportunity_id, "code": "stalled"})
        if opportunity.previous_close_on:
            previous_close = parse_date(opportunity.previous_close_on)
            expected_close = parse_date(opportunity.expected_close_on)
            if previous_close is None or expected_close is None:
                raise SalesPipelineContractError("sales pipeline health requires gated close dates")
            slip_days = (expected_close - previous_close).days
            if slip_days > 0:
                slips.append({"opportunity_id": opportunity.opportunity_id, "slip_days": slip_days})
                exceptions.append({"opportunity_id": opportunity.opportunity_id, "code": "slipped"})
        evidence_by_criterion = {item.criterion_id: item for item in opportunity.exit_criteria}
        for criterion_id in stages[opportunity.stage_id].required_exit_criteria:
            evidence = evidence_by_criterion.get(criterion_id)
            if evidence is None or evidence.state != "met":
                exit_gaps.append({
                    "opportunity_id": opportunity.opportunity_id,
                    "criterion_id": criterion_id,
                    "state": "missing" if evidence is None else evidence.state,
                })
                exceptions.append({"opportunity_id": opportunity.opportunity_id, "code": "exit_criterion_gap"})
        next_step = opportunity.next_step
        if next_step is None:
            quality = "missing"
            next_steps.append({"opportunity_id": opportunity.opportunity_id, "quality": quality})
        else:
            due_on = parse_date(next_step.due_on)
            if due_on is None:
                raise SalesPipelineContractError("sales pipeline health requires gated next-step dates")
            quality = "overdue" if due_on < as_of else "unsupported" if not next_step.evidence_refs else "evidence_backed"
            next_steps.append({
                "opportunity_id": opportunity.opportunity_id,
                "action_id": next_step.action_id,
                "due_on": next_step.due_on,
                "quality": quality,
                "evidence_refs": list(next_step.evidence_refs),
            })
        if quality != "evidence_backed":
            exceptions.append({"opportunity_id": opportunity.opportunity_id, "code": f"next_step_{quality}"})
    concentration, concentration_exceptions = _concentration(request)
    exceptions.extend(
        {"opportunity_id": "portfolio", "code": "account_concentration", "account_id": row["account_id"]}
        for row in concentration_exceptions
    )
    return {
        "schema_version": "sales_pipeline_health/v1",
        "status": "prepared_not_observed",
        "source_ref": request.source_ref,
        "as_of": request.snapshot_as_of,
        "opportunity_count": len(request.opportunities),
        "movement": movement,
        "aging": aging,
        "stalls": stalls,
        "slips": slips,
        "exit_criteria_gaps": exit_gaps,
        "next_step_quality": next_steps,
        "concentration": concentration,
        "duplicate_ids": [],
        "missing_owner_ids": [],
        "deal_exceptions": exceptions,
        "claim_boundary": _HEALTH_BOUNDARY,
    }
