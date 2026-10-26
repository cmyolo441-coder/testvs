"""Closed-shape validation for the product-discovery artifact family."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..system.append_only_store import RAW_OR_HIDDEN_KEYS
from .product_discovery_artifacts import (
    _artifact,
    build_assumption_test_portfolio,
    build_channel_feedback_ledger,
    build_channel_feedback_disposition,
    build_customer_discovery_plan,
    build_discovery_decision_frame,
    build_discovery_decision_receipt,
    build_discovery_evidence_ledger,
    build_initial_gtm_hypothesis,
)


_COMMON_KEYS = frozenset({"schema_version", "artifact_id", "discovery_id", "status", "claim_boundary"})
_SCHEMA_KEYS = {
    "channel_feedback_ledger/v1": _COMMON_KEYS | {"gtm_artifact_id", "initial_channel_ref", "segment_ref", "entries"},
    "channel_feedback_disposition/v1": _COMMON_KEYS | {"frame_ref", "gtm_artifact_id", "portfolio_ref", "feedback_ledger_ref", "initial_channel_ref", "segment_ref", "evaluated_at", "channel_reachability", "opportunity_direction", "rejected_channel_hypothesis_refs", "held_observations", "proposed_followup", "disposition", "handoff_held", "next_route"},
    "discovery_decision_frame/v1": _COMMON_KEYS | {"problem_ref", "segment_ref", "segment_definition_state", "alternative_refs", "decision_owner_ref", "learning_budget_ref", "deadline_at", "kill_criteria_refs"},
    "discovery_evidence_ledger/v1": _COMMON_KEYS | {"entries"},
    "customer_discovery_plan/v1": _COMMON_KEYS | {"participant_criteria_ref", "interview_focuses", "consent_privacy_ref", "bias_control_refs", "human_task_ref", "evidence_reentry_required"},
    "assumption_test_portfolio/v1": _COMMON_KEYS | {"assumptions"},
    "discovery_decision_receipt/v1": _COMMON_KEYS | {"problem_ref", "segment_ref", "segment_definition_state", "decision", "problem_gate", "solution_work_permitted", "missing_audience_evidence_refs", "precommitted_test_ids", "eligible_evidence_refs", "rejected_hypothesis_ids", "residual_risk_refs", "next_route"},
    "initial_gtm_hypothesis/v1": _COMMON_KEYS | {"beachhead_segment_ref", "buyer_ref", "user_ref", "current_alternative_ref", "value_proposition_ref", "pricing_hypothesis_ref", "initial_channel_ref", "first_cohort_ref", "learning_metric_refs"},
}


def validate_product_discovery_artifact(record: Any) -> list[str]:
    """Return every closed-shape fault for one versioned discovery artifact."""
    if not isinstance(record, Mapping):
        return ["product discovery artifact must be an object"]
    if any(not isinstance(key, str) for key in record):
        return ["product discovery artifact keys must be strings"]
    forbidden = sorted(key for key in record if key.lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        return [f"product discovery artifact must not carry raw or hidden keys: {forbidden}"]
    schema = record.get("schema_version")
    if not isinstance(schema, str) or schema not in _SCHEMA_KEYS:
        return ["product discovery artifact schema_version is unsupported"]
    expected_keys = _SCHEMA_KEYS[schema]
    missing = sorted(expected_keys - set(record))
    extra = sorted(set(record) - expected_keys)
    if missing or extra:
        errors: list[str] = []
        if missing:
            errors.append(f"product discovery artifact is missing keys: {missing}")
        if extra:
            errors.append(f"product discovery artifact has unsupported keys: {extra}")
        return errors
    match schema:
        case "channel_feedback_ledger/v1":
            return _validated(record, lambda: _stored_channel_feedback_ledger(record))
        case "channel_feedback_disposition/v1":
            values = _fields(record)
            for field in ("disposition", "handoff_held", "next_route"):
                values.pop(field)
            return _validated(record, lambda: build_channel_feedback_disposition(**values))
        case "discovery_decision_frame/v1":
            return _validated(record, lambda: build_discovery_decision_frame(**_fields(record)))
        case "discovery_evidence_ledger/v1":
            return _validated(record, lambda: build_discovery_evidence_ledger(**_fields(record)))
        case "customer_discovery_plan/v1":
            values = _fields(record)
            values.pop("interview_focuses")
            values.pop("evidence_reentry_required")
            return _validated(record, lambda: build_customer_discovery_plan(**values))
        case "assumption_test_portfolio/v1":
            return _validated(record, lambda: build_assumption_test_portfolio(**_fields(record)))
        case "discovery_decision_receipt/v1":
            values = _fields(record)
            values.pop("solution_work_permitted")
            values.pop("missing_audience_evidence_refs")
            return _validated(record, lambda: build_discovery_decision_receipt(**values))
        case "initial_gtm_hypothesis/v1":
            return _validated(record, lambda: build_initial_gtm_hypothesis(**_fields(record)))
        case _:
            return ["product discovery artifact schema_version is unsupported"]


def _stored_channel_feedback_ledger(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate historical rows without rewriting their original content digest."""
    expected = build_channel_feedback_ledger(**_fields(record))
    for supplied, normalized in zip(record["entries"], expected["entries"]):
        if "discovery_id" not in supplied:
            normalized.pop("discovery_id")
    fields = _fields(expected)
    discovery_id = fields.pop("discovery_id")
    expected["artifact_id"] = _artifact(expected["schema_version"], discovery_id, fields,
                                        status=expected["status"])["artifact_id"]
    return expected


def _fields(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key not in _COMMON_KEYS - {"discovery_id"}}


def _validated(record: Mapping[str, Any], builder: Callable[[], dict[str, Any]]) -> list[str]:
    try:
        expected = builder()
    except ValueError as exc:
        return [str(exc)]
    if set(record) != set(expected):
        return ["product discovery artifact keys are invalid"]
    return [] if dict(record) == expected else ["product discovery artifact fields are invalid"]
