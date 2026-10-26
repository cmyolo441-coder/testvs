from __future__ import annotations

from .sales_pipeline_artifacts import validate_sales_pipeline_artifact
from .sales_pipeline_gates import parse_date, safe_token
from .sales_pipeline_models import JsonObject, SalesPipelineContractError, SalesPipelineHandoffInput

_APPROVAL_STATES = frozenset(("pending", "approved", "rejected"))


def prepare_sales_pipeline_handoff(request: SalesPipelineHandoffInput) -> JsonObject:
    evaluation = request.evaluation
    if evaluation.status != "READY" or evaluation.health is None or evaluation.forecast is None:
        raise SalesPipelineContractError("sales pipeline handoff requires a READY evaluated review")
    if validate_sales_pipeline_artifact(evaluation.health) or validate_sales_pipeline_artifact(evaluation.forecast):
        raise SalesPipelineContractError("sales pipeline handoff requires valid health and forecast artifacts")
    if not safe_token(request.decision_owner_id):
        raise SalesPipelineContractError("sales pipeline handoff requires a bounded decision owner")
    opportunity_ids = {
        str(item["opportunity_id"])
        for item in evaluation.health["aging"]
        if isinstance(item, dict) and isinstance(item.get("opportunity_id"), str)
    }
    follow_ups: list[JsonObject] = []
    for item in request.selected_follow_ups:
        values = (item.opportunity_id, item.owner_id, item.action_id, item.exit_criterion_id, *item.evidence_refs)
        if item.opportunity_id not in opportunity_ids or any(not safe_token(value) for value in values):
            raise SalesPipelineContractError("sales pipeline follow-up metadata is invalid")
        if parse_date(item.due_on) is None or not item.evidence_refs:
            raise SalesPipelineContractError("sales pipeline follow-up requires due date and evidence")
        follow_ups.append({
            "opportunity_id": item.opportunity_id,
            "owner_id": item.owner_id,
            "due_on": item.due_on,
            "action_id": item.action_id,
            "exit_criterion_id": item.exit_criterion_id,
            "evidence_refs": list(item.evidence_refs),
            "state": "proposed_not_delivered",
        })
    corrections: list[JsonObject] = []
    for item in request.proposed_corrections:
        values = (
            item.object_id, item.object_type, item.field_id, item.proposed_value, item.owner_id, *item.evidence_refs,
        )
        if any(not safe_token(value) for value in values) or not item.evidence_refs:
            raise SalesPipelineContractError("sales pipeline CRM correction metadata is invalid")
        if item.approval_state not in _APPROVAL_STATES:
            raise SalesPipelineContractError("sales pipeline CRM correction approval state is invalid")
        corrections.append({
            "object_id": item.object_id,
            "object_type": item.object_type,
            "field_id": item.field_id,
            "proposed_value": item.proposed_value,
            "evidence_refs": list(item.evidence_refs),
            "owner_id": item.owner_id,
            "approval_state": item.approval_state,
        })
    artifact: JsonObject = {
        "schema_version": "sales_pipeline_handoff/v1",
        "status": "prepared_not_observed",
        "source_ref": str(evaluation.scope["source_ref"]),
        "decision_owner_id": request.decision_owner_id,
        "selected_account_follow_ups": follow_ups,
        "proposed_crm_corrections": corrections,
        "connector": {"availability": "unavailable", "mutation_observed": False, "delivery_observed": False},
        "specialist_routes": [
            {"need": "account_discovery_or_qualification", "workflow": "sales-development"},
            {"need": "qualitative_customer_material", "workflow": "feedback-triage"},
            {"need": "generic_calculation", "workflow": "data-analysis"},
            {"need": "authoritative_finance_reporting", "workflow": "finance-analysis"},
        ],
        "claim_boundary": (
            "This handoff contains proposed owned follow-ups and complete proposed CRM corrections. The connector is unavailable, "
            "so it is not delivery, communication, CRM mutation, sync, alerting, approval, or observed execution evidence."
        ),
    }
    errors = validate_sales_pipeline_artifact(artifact)
    if errors:
        raise SalesPipelineContractError("sales pipeline handoff artifact is invalid: " + "; ".join(errors))
    return artifact
