"""Validated, bounded decision context for downstream workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeAlias

from .decision_prototypes import compact_decision_prototype_receipt, validate_decision_prototype
from .product_discovery_artifact_validation import validate_product_discovery_artifact
from .product_discovery_artifacts import audience_is_defined
from .product_discovery_validation import product_brief_consumption, product_brief_consumption_with_feedback


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
DecisionArtifact: TypeAlias = Mapping[str, JsonValue]
DecisionReceiptHandoff: TypeAlias = dict[str, JsonValue]

DECISION_RECEIPT_HANDOFF_SCHEMA_VERSION: Final = "decision_receipt_handoff/v1"
TARGET_WORKFLOWS: Final = ("ralplan", "product-brief")
_CLAIM_BOUNDARY: Final = (
    "Decision receipt handoff is prepared context only. It is not plan acceptance, execution permission, "
    "implementation, product approval, review, CI, merge-readiness, or merge evidence."
)


class DecisionReceiptHandoffError(ValueError):
    """Raised when a source artifact cannot safely cross its workflow boundary."""


def build_decision_receipt_handoff(
    source_artifact: DecisionArtifact, *, target_workflow: str
) -> DecisionReceiptHandoff:
    """Build bounded prepared context from one validated source artifact."""
    if target_workflow not in TARGET_WORKFLOWS:
        raise DecisionReceiptHandoffError("decision receipt handoff target_workflow is unsupported")
    schema_version = source_artifact.get("schema_version")
    if schema_version == "decision_prototype/v1":
        if target_workflow != "ralplan":
            raise DecisionReceiptHandoffError("decision_prototype handoff targets ralplan only")
        errors = validate_decision_prototype(source_artifact)
        if errors:
            raise DecisionReceiptHandoffError(errors[0])
        return _prototype_handoff(source_artifact)
    if schema_version == "discovery_decision_receipt/v1":
        if target_workflow != "product-brief":
            raise DecisionReceiptHandoffError("discovery decision receipt targets product-brief only")
        errors = validate_product_discovery_artifact(source_artifact)
        if errors:
            raise DecisionReceiptHandoffError(errors[0])
        return _product_brief_handoff(source_artifact)
    raise DecisionReceiptHandoffError("decision receipt handoff source artifact schema is unsupported")


def _compact_prototype_receipt(prototype: DecisionArtifact) -> dict[str, JsonValue]:
    receipt: JsonValue = compact_decision_prototype_receipt(prototype)
    return receipt


def _prototype_handoff(prototype: DecisionArtifact) -> DecisionReceiptHandoff:
    receipt = _compact_prototype_receipt(prototype)
    execution_status = receipt["execution_status"]
    supported_option = receipt["supported_option"]
    if not isinstance(execution_status, str) or not isinstance(supported_option, str):
        raise DecisionReceiptHandoffError("compact prototype receipt has invalid decision fields")
    resolved = execution_status == "observed" and bool(supported_option)
    return {
        "schema_version": DECISION_RECEIPT_HANDOFF_SCHEMA_VERSION,
        "target_workflow": "ralplan",
        "prepared_status": "prepared_context",
        "decision_state": "resolved" if resolved else "unresolved",
        "blocked_reason": "" if resolved else _prototype_blocked_reason(execution_status),
        "decision": {
            "source_schema_version": "decision_prototype/v1",
            "decision_id": receipt["decision_id"],
            "context_decision_ref": receipt["context_decision_ref"],
            "execution_status": execution_status,
            "supported_option": supported_option if resolved else "",
            "rejected_options": receipt["rejected_options"] if resolved else [],
            "confidence": receipt["confidence"] if resolved else "",
            "residual_risk": receipt["residual_risk"],
            "evidence_refs": receipt["evidence_refs"],
            "evidence_limits": receipt["evidence_limits"],
        },
        "production_authority": "none",
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _prototype_blocked_reason(execution_status: str) -> str:
    reasons = {
        "available_not_observed": "prototype_not_observed",
        "prepared_not_observed": "prototype_not_observed",
        "timeout": "prototype_timeout",
        "inconclusive": "prototype_inconclusive",
        "observed": "prototype_option_unresolved",
    }
    reason = reasons.get(execution_status)
    if reason is None:
        raise DecisionReceiptHandoffError("validated prototype has an unsupported execution status")
    return reason


def _product_brief_handoff(receipt: DecisionArtifact) -> DecisionReceiptHandoff:
    context: JsonValue = product_brief_consumption(receipt)
    validated = bool(context)
    return {
        "schema_version": DECISION_RECEIPT_HANDOFF_SCHEMA_VERSION,
        "target_workflow": "product-brief",
        "prepared_status": "prepared_context",
        "decision_state": "validated" if validated else "blocked",
        "blocked_reason": "" if validated else _discovery_blocked_reason(receipt),
        "decision": {
            "source_schema_version": "discovery_decision_receipt/v1",
            "discovery_id": receipt["discovery_id"],
            "decision": receipt["decision"],
            "problem_gate": receipt["problem_gate"],
            "segment_definition_state": receipt["segment_definition_state"],
            "solution_work_permitted": receipt["solution_work_permitted"],
            "missing_audience_evidence_refs": receipt["missing_audience_evidence_refs"],
            "rejected_hypothesis_ids": receipt["rejected_hypothesis_ids"],
            "residual_risk_refs": receipt["residual_risk_refs"],
        },
        "product_brief_context": context,
        "production_authority": "none",
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def build_channel_aware_product_brief_handoff(
    receipt: DecisionArtifact, disposition: DecisionArtifact
) -> DecisionReceiptHandoff:
    """Hold supplied channel feedback without changing legacy receipt consumers."""
    handoff = build_decision_receipt_handoff(receipt, target_workflow="product-brief")
    context = product_brief_consumption_with_feedback(receipt, disposition)
    if not context:
        handoff["decision_state"] = "blocked"
        handoff["blocked_reason"] = _discovery_blocked_reason(receipt, disposition=disposition)
        handoff["product_brief_context"] = {}
    return handoff


def _discovery_blocked_reason(receipt: DecisionArtifact, *, disposition: DecisionArtifact | None = None) -> str:
    if disposition is not None:
        if (not validate_product_discovery_artifact(disposition)
                and disposition.get("schema_version") == "channel_feedback_disposition/v1"
                and disposition.get("discovery_id") == receipt["discovery_id"]
                and disposition.get("segment_ref") == receipt["segment_ref"]
                and disposition.get("channel_reachability") == "contradicts"):
            return "discovery_segment_reachability_contradicted"
        return "discovery_channel_feedback_hold"
    if receipt["problem_gate"] == "refuted":
        return "discovery_problem_refuted"
    if not audience_is_defined(str(receipt["segment_definition_state"])):
        return "discovery_audience_undefined"
    return "discovery_evidence_unresolved"
