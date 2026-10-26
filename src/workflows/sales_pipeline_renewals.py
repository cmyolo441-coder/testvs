from __future__ import annotations

from collections import defaultdict

from .sales_pipeline_gates import parse_date
from .sales_pipeline_models import JsonObject, SalesPipelineContractError, SalesPipelineReviewInput

_SIGNAL_FIELDS = ("health", "utilization", "support", "budget", "staffing")


def build_sales_renewal_risk_annex(request: SalesPipelineReviewInput) -> JsonObject:
    as_of = parse_date(request.snapshot_as_of[:10])
    horizon_end = parse_date(request.horizon_end)
    if as_of is None or horizon_end is None:
        raise SalesPipelineContractError("sales renewal annex requires gated horizon metadata")
    signals: list[JsonObject] = []
    open_risks: list[JsonObject] = []
    evidence_gaps: list[JsonObject] = []
    expansion_hypotheses: list[JsonObject] = []
    observed_states: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for signal in request.renewal_signals:
        renewal_on = parse_date(signal.renewal_on)
        if renewal_on is None:
            raise SalesPipelineContractError("sales renewal annex requires gated renewal dates")
        in_horizon = as_of <= renewal_on <= horizon_end
        state_values = {
            "health": signal.health_state,
            "utilization": signal.utilization_state,
            "support": signal.support_state,
            "budget": signal.budget_state,
            "staffing": signal.staffing_state,
        }
        signals.append({
            "account_id": signal.account_id,
            "owner_id": signal.owner_id,
            "renewal_on": signal.renewal_on,
            "in_review_horizon": in_horizon,
            "states": state_values,
            "evidence_refs": list(signal.evidence_refs),
        })
        for field, state in state_values.items():
            observed_states[(signal.account_id, field)].add(state)
            if state == "unknown":
                evidence_gaps.append({"account_id": signal.account_id, "signal": field})
            elif state == "observed_negative" and in_horizon:
                open_risks.append({"account_id": signal.account_id, "signal": field, "owner_id": signal.owner_id})
        if signal.expansion_hypothesis_id:
            expansion_hypotheses.append({
                "account_id": signal.account_id,
                "hypothesis_id": signal.expansion_hypothesis_id,
                "state": "hypothesis_not_observed",
                "owner_id": signal.owner_id,
            })
    contradictions = [
        {"account_id": account_id, "signal": field, "observed_states": sorted(states)}
        for (account_id, field), states in sorted(observed_states.items())
        if len(states - {"unknown"}) > 1
    ]
    return {
        "schema_version": "sales_renewal_risk_annex/v1",
        "status": "prepared_not_observed",
        "source_ref": request.source_ref,
        "horizon_end": request.horizon_end,
        "signals": signals,
        "open_risks": open_risks,
        "evidence_gaps": evidence_gaps,
        "contradictions": contradictions,
        "expansion_hypotheses": expansion_hypotheses,
        "claim_boundary": (
            "This optional annex preserves supplied renewal signals, gaps, contradictions, risks, and hypotheses. It does not "
            "claim causality, customer intent, expansion, outreach, CRM mutation, or an authoritative finance result."
        ),
    }
