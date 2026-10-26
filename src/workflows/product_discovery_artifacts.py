"""Closed, metadata-only artifacts for product-discovery validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..system.append_only_store import RAW_OR_HIDDEN_KEYS, opaque_ref
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


DISCOVERY_DECISION_FRAME_SCHEMA_VERSION: Final = "discovery_decision_frame/v1"
DISCOVERY_EVIDENCE_LEDGER_SCHEMA_VERSION: Final = "discovery_evidence_ledger/v1"
CUSTOMER_DISCOVERY_PLAN_SCHEMA_VERSION: Final = "customer_discovery_plan/v1"
ASSUMPTION_TEST_PORTFOLIO_SCHEMA_VERSION: Final = "assumption_test_portfolio/v1"
DISCOVERY_DECISION_RECEIPT_SCHEMA_VERSION: Final = "discovery_decision_receipt/v1"
INITIAL_GTM_HYPOTHESIS_SCHEMA_VERSION: Final = "initial_gtm_hypothesis/v1"
CHANNEL_FEEDBACK_LEDGER_SCHEMA_VERSION: Final = "channel_feedback_ledger/v1"
CHANNEL_FEEDBACK_DISPOSITION_SCHEMA_VERSION: Final = "channel_feedback_disposition/v1"
CHANNEL_FEEDBACK_AFFECTED_TARGETS: Final = ("channel_reachability", "opportunity_direction")
CHANNEL_FEEDBACK_EFFECTS: Final = ("supports", "contradicts", "unknown")
CHANNEL_FEEDBACK_HOLD_REASONS: Final = (
    "channel_feedback_missing", "channel_feedback_stale", "channel_feedback_duplicate",
    "channel_feedback_foreign_segment", "channel_feedback_wrong_channel", "channel_feedback_wrong_test",
    "channel_feedback_contradictory", "channel_feedback_unresolved",
)
CHANNEL_FEEDBACK_ENTRY_KEYS: Final = frozenset({
    "discovery_id", "feedback_id", "test_id", "channel_ref", "criterion_ref", "segment_ref", "affected_target", "effect",
    "source_class", "source_ref", "observed_at", "sample_count", "confidence_limit",
})

EVIDENCE_SOURCE_CLASSES: Final = (
    "external_human",
    "behavioral_data",
    "internal_stakeholder",
    "secondary_research",
    "synthetic",
    "inferred",
)
ASSUMPTION_CATEGORIES: Final = ("value", "usability", "feasibility", "viability", "go_to_market", "ethics")
DECISIONS: Final = ("kill", "pivot", "persevere", "inconclusive")
PROBLEM_GATES: Final = ("validated", "refuted", "inconclusive")

SEGMENT_DEFINITION_STATES: Final = (
    "recruitable",
    "behaviorally_observed",
    "unknown",
    "synthetic_only",
    "non_recruitable",
)
AUDIENCE_DEFINED_STATES: Final = ("recruitable", "behaviorally_observed")
BUILD_BOUNDARY_ROUTES: Final = ("product-brief", "decision-prototype", "coding-handoff")
DISCOVERY_CONTINUATION_ROUTE: Final = "product-discovery-validation"
MISSING_AUDIENCE_EVIDENCE_REFS: Final = {
    "unknown": (
        "audience-recruitable-segment-definition",
        "audience-participant-recruitment-criteria",
    ),
    "synthetic_only": (
        "audience-external-human-or-behavioral-evidence",
    ),
    "non_recruitable": (
        "audience-recruitable-or-behaviorally-observed-segment",
    ),
}

_CLAIM_BOUNDARY: Final = (
    "This metadata-only artifact preserves bounded references and declared or derived discovery state. "
    "It does not recruit, contact, inspect customers, replay transcripts, build a prototype or PRD, execute a test, "
    "or establish product-market fit."
)


def _ref(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return opaque_ref(value, field=field, error=ValueError)


def _refs(values: Sequence[str], field: str, *, minimum: int = 1) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, str):
        raise ValueError(f"{field} must be a list")
    if len(values) < minimum:
        raise ValueError(f"{field} must have at least {minimum} item")
    return [_ref(value, f"{field}[{index}]") for index, value in enumerate(values)]


def _stamp(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return value


def _segment_definition_state(value: Any) -> str:
    if value not in SEGMENT_DEFINITION_STATES:
        raise ValueError("segment_definition_state is unsupported")
    return str(value)


def audience_is_defined(segment_definition_state: str) -> bool:
    """Report whether a framed segment is recruitable or behaviorally observed."""
    return _segment_definition_state(segment_definition_state) in AUDIENCE_DEFINED_STATES


def missing_audience_evidence_refs(segment_definition_state: str) -> list[str]:
    """Name the audience evidence a segment definition still owes."""
    state = _segment_definition_state(segment_definition_state)
    return list(MISSING_AUDIENCE_EVIDENCE_REFS.get(state, ()))


def _artifact(schema_version: str, discovery_id: str, fields: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    safe_discovery_id = _ref(discovery_id, "discovery_id")
    digest = hashlib.sha256(
        json.dumps({"schema_version": schema_version, "discovery_id": safe_discovery_id, **fields}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return {
        "schema_version": schema_version,
        "artifact_id": f"discovery-{digest}",
        "discovery_id": safe_discovery_id,
        "status": status,
        **fields,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _channel_feedback_entry(entry: Mapping[str, Any], *, discovery_id: str,
                            allow_missing: bool = False) -> dict[str, Any]:
    if not isinstance(entry, Mapping) or any(not isinstance(key, str) for key in entry):
        raise ValueError("channel feedback entry must be an object with string keys")
    if any(key.lower() in RAW_OR_HIDDEN_KEYS or key == "raw_transcript" for key in entry):
        raise ValueError("channel feedback must not carry raw or hidden keys")
    if set(entry) - CHANNEL_FEEDBACK_ENTRY_KEYS:
        raise ValueError("channel feedback entry has unsupported keys")
    safe_discovery_id = _ref(discovery_id, "discovery_id")
    if _ref(entry.get("discovery_id", safe_discovery_id), "discovery_id") != safe_discovery_id:
        raise ValueError("channel feedback entry must have the same discovery_id as its ledger")
    normalized = {**entry, "discovery_id": safe_discovery_id}
    if not allow_missing and set(normalized) != CHANNEL_FEEDBACK_ENTRY_KEYS:
        raise ValueError("channel feedback entry is missing keys")
    for field in ("feedback_id", "test_id", "channel_ref", "criterion_ref", "segment_ref", "source_ref"):
        if field in entry:
            normalized[field] = _ref(entry[field], field)
    if "observed_at" in entry:
        normalized["observed_at"] = _stamp(entry["observed_at"], "observed_at")
    for field, choices in (("affected_target", CHANNEL_FEEDBACK_AFFECTED_TARGETS),
                           ("effect", CHANNEL_FEEDBACK_EFFECTS), ("source_class", EVIDENCE_SOURCE_CLASSES),
                           ("confidence_limit", ("bounded", "limited", "unknown"))):
        if field in entry and entry[field] not in choices:
            raise ValueError(f"channel feedback {field} is unsupported")
    if "sample_count" in entry:
        count = entry["sample_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("channel feedback sample_count must be a positive integer")
    return normalized


def build_channel_feedback_ledger(*, discovery_id: str, gtm_artifact_id: str, initial_channel_ref: str,
                                  segment_ref: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build an appendable ledger; incomplete or copied rows belong in refusal history."""
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise ValueError("entries must be a list")
    normalized = [_channel_feedback_entry(entry, discovery_id=discovery_id) for entry in entries]
    if len({entry["feedback_id"] for entry in normalized}) != len(normalized):
        raise ValueError("feedback_id values must be unique")
    if len({(entry["test_id"], entry["source_ref"], entry["affected_target"]) for entry in normalized}) != len(normalized):
        raise ValueError("duplicate channel feedback source for one precommitted test and target")
    result = _artifact(CHANNEL_FEEDBACK_LEDGER_SCHEMA_VERSION, discovery_id, {
        "gtm_artifact_id": _ref(gtm_artifact_id, "gtm_artifact_id"),
        "initial_channel_ref": _ref(initial_channel_ref, "initial_channel_ref"),
        "segment_ref": _ref(segment_ref, "segment_ref"), "entries": normalized,
    }, status="reentered")
    result["claim_boundary"] = _CHANNEL_FEEDBACK_CLAIM_BOUNDARY
    return result


_CHANNEL_FEEDBACK_CLAIM_BOUNDARY: Final = (
    "Supplied observations and local evaluation are not market execution, customer validation, or product outcome evidence. "
    "These metadata-only companions preserve bounded references and do not modify earlier discovery decisions."
)


def build_channel_feedback_disposition(*, discovery_id: str, frame_ref: str, gtm_artifact_id: str,
        portfolio_ref: str, feedback_ledger_ref: str, initial_channel_ref: str, segment_ref: str, evaluated_at: str,
        channel_reachability: str, opportunity_direction: str, rejected_channel_hypothesis_refs: Sequence[str],
        held_observations: Sequence[Mapping[str, Any]], proposed_followup: str) -> dict[str, Any]:
    """Derive closed gate and route fields, never a replacement problem receipt."""
    effects = (channel_reachability, opportunity_direction)
    if any(effect not in CHANNEL_FEEDBACK_EFFECTS for effect in effects):
        raise ValueError("channel feedback effect is unsupported")
    if proposed_followup not in ("none", "bounded_channel_retest", "gtm_pivot", "reevaluate_discovery"):
        raise ValueError("channel feedback proposed_followup is unsupported")
    if not isinstance(held_observations, Sequence) or isinstance(held_observations, (str, bytes)):
        raise ValueError("held_observations must be a list")
    held = []
    for row in held_observations:
        if not isinstance(row, Mapping) or set(row) != {"feedback_ref", "reason"}:
            raise ValueError("held observation keys are invalid")
        if row["reason"] not in CHANNEL_FEEDBACK_HOLD_REASONS:
            raise ValueError("channel feedback hold reason is unsupported")
        held.append({"feedback_ref": _ref(row["feedback_ref"], "feedback_ref"), "reason": row["reason"]})
    rejected = _refs(rejected_channel_hypothesis_refs, "rejected_channel_hypothesis_refs", minimum=0)
    safe_gtm = _ref(gtm_artifact_id, "gtm_artifact_id")
    if rejected and (rejected != [safe_gtm] or channel_reachability != "contradicts"):
        raise ValueError("rejected channel hypothesis must match a contradicted initial GTM hypothesis")
    if proposed_followup in ("gtm_pivot", "reevaluate_discovery") and not rejected:
        raise ValueError("precommitted channel followup requires a rejected hypothesis")
    held_handoff = bool(held) or any(effect != "supports" for effect in effects) or proposed_followup != "none"
    disposition = "inconclusive" if held or "unknown" in effects else "held" if held_handoff else "supported"
    result = _artifact(CHANNEL_FEEDBACK_DISPOSITION_SCHEMA_VERSION, discovery_id, {
        "frame_ref": _ref(frame_ref, "frame_ref"), "gtm_artifact_id": safe_gtm,
        "portfolio_ref": _ref(portfolio_ref, "portfolio_ref"),
        "feedback_ledger_ref": _ref(feedback_ledger_ref, "feedback_ledger_ref"),
        "initial_channel_ref": _ref(initial_channel_ref, "initial_channel_ref"), "segment_ref": _ref(segment_ref, "segment_ref"),
        "evaluated_at": _stamp(evaluated_at, "evaluated_at"),
        "channel_reachability": channel_reachability, "opportunity_direction": opportunity_direction,
        "rejected_channel_hypothesis_refs": rejected, "held_observations": held,
        "proposed_followup": proposed_followup, "disposition": disposition, "handoff_held": held_handoff,
        "next_route": DISCOVERY_CONTINUATION_ROUTE if held_handoff else "product-brief",
    }, status="derived_from_supplied_metadata")
    result["claim_boundary"] = _CHANNEL_FEEDBACK_CLAIM_BOUNDARY
    return result


def build_discovery_decision_frame(*, discovery_id: str, problem_ref: str, segment_ref: str, segment_definition_state: str, alternative_refs: Sequence[str], decision_owner_ref: str, learning_budget_ref: str, deadline_at: str, kill_criteria_refs: Sequence[str]) -> dict[str, Any]:
    """Prepare the bounded problem decision before evidence is accepted.

    ``segment_definition_state`` names the audience explicitly. An unknown,
    synthetic-only, or non-recruitable segment stays framable so evidence work
    can continue, and blocks solution work at the gate rather than here.
    """
    return _artifact(
        DISCOVERY_DECISION_FRAME_SCHEMA_VERSION,
        discovery_id,
        {
            "problem_ref": _ref(problem_ref, "problem_ref"),
            "segment_ref": _ref(segment_ref, "segment_ref"),
            "segment_definition_state": _segment_definition_state(segment_definition_state),
            "alternative_refs": _refs(alternative_refs, "alternative_refs"),
            "decision_owner_ref": _ref(decision_owner_ref, "decision_owner_ref"),
            "learning_budget_ref": _ref(learning_budget_ref, "learning_budget_ref"),
            "deadline_at": _stamp(deadline_at, "deadline_at"),
            "kill_criteria_refs": _refs(kill_criteria_refs, "kill_criteria_refs"),
        },
        status="prepared_not_observed",
    )


def build_discovery_evidence_ledger(*, discovery_id: str, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Prepare durable evidence re-entry rows without raw customer material."""
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        raise ValueError("entries must be a list")
    normalized = [_evidence_entry(entry) for entry in entries]
    if not normalized:
        raise ValueError("entries must have at least 1 item")
    if len({entry["evidence_id"] for entry in normalized}) != len(normalized):
        raise ValueError("evidence_id values must be unique")
    if len({(entry["test_id"], entry["source_ref"]) for entry in normalized}) != len(normalized):
        raise ValueError("duplicate evidence source for one precommitted test")
    return _artifact(DISCOVERY_EVIDENCE_LEDGER_SCHEMA_VERSION, discovery_id, {"entries": normalized}, status="reentered")


def _evidence_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise ValueError("evidence entry must be an object")
    source_class = entry.get("source_class")
    direction = entry.get("direction")
    observation_kind = entry.get("observation_kind")
    reentry = entry.get("reentry")
    confidence_limit = entry.get("confidence_limit")
    sample_count = entry.get("sample_count")
    representative = entry.get("representative")
    if source_class not in EVIDENCE_SOURCE_CLASSES:
        raise ValueError("evidence source_class is unsupported")
    if direction not in ("supports", "contradicts", "unresolved"):
        raise ValueError("evidence direction is unsupported")
    if observation_kind not in ("past_behavior", "current_workaround", "switching_cost", "observed_commitment", "source_pointer", "prototype_completion"):
        raise ValueError("evidence observation_kind is unsupported")
    if reentry not in ("reentered", "not_reentered") or confidence_limit not in ("bounded", "limited", "unknown"):
        raise ValueError("evidence reentry or confidence_limit is unsupported")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise ValueError("evidence sample_count must be a positive integer")
    if not isinstance(representative, bool):
        raise ValueError("evidence representative must be a boolean")
    return {
        "evidence_id": _ref(entry.get("evidence_id", ""), "evidence_id"),
        "test_id": _ref(entry.get("test_id", ""), "test_id"),
        "source_class": source_class,
        "source_ref": _ref(entry.get("source_ref", ""), "source_ref"),
        "criterion_ref": _ref(entry.get("criterion_ref", ""), "criterion_ref"),
        "observed_at": _stamp(entry.get("observed_at", ""), "observed_at"),
        "segment_ref": _ref(entry.get("segment_ref", ""), "segment_ref"),
        "direction": direction,
        "observation_kind": observation_kind,
        "sample_count": sample_count,
        "representative": representative,
        "reentry": reentry,
        "confidence_limit": confidence_limit,
    }


def build_customer_discovery_plan(*, discovery_id: str, participant_criteria_ref: str, consent_privacy_ref: str, bias_control_refs: Sequence[str], human_task_ref: str) -> dict[str, Any]:
    """Prepare human-owned interview re-entry without simulating participants."""
    return _artifact(
        CUSTOMER_DISCOVERY_PLAN_SCHEMA_VERSION,
        discovery_id,
        {
            "participant_criteria_ref": _ref(participant_criteria_ref, "participant_criteria_ref"),
            "interview_focuses": ["past_behavior", "current_workaround", "switching_cost", "observed_commitment"],
            "consent_privacy_ref": _ref(consent_privacy_ref, "consent_privacy_ref"),
            "bias_control_refs": _refs(bias_control_refs, "bias_control_refs"),
            "human_task_ref": _ref(human_task_ref, "human_task_ref"),
            "evidence_reentry_required": True,
        },
        status="prepared_not_observed",
    )


def build_assumption_test_portfolio(*, discovery_id: str, assumptions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Precommit the smallest disconfirming test before accepting observations."""
    if not isinstance(assumptions, Sequence) or isinstance(assumptions, str):
        raise ValueError("assumptions must be a list")
    normalized = [_assumption(assumption) for assumption in assumptions]
    if not normalized:
        raise ValueError("assumptions must have at least 1 item")
    if len({row["test_id"] for row in normalized}) != len(normalized):
        raise ValueError("assumption test_id values must be unique")
    return _artifact(ASSUMPTION_TEST_PORTFOLIO_SCHEMA_VERSION, discovery_id, {"assumptions": normalized}, status="precommitted")


def _assumption(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("assumption must be an object")
    category = value.get("category")
    impact = value.get("decision_impact")
    gap = value.get("evidence_gap")
    sample_target = value.get("sample_target")
    failure_decision = value.get("failure_decision")
    classes = value.get("required_evidence_classes")
    if category not in ASSUMPTION_CATEGORIES:
        raise ValueError("assumption category is unsupported")
    if not isinstance(impact, int) or isinstance(impact, bool) or impact not in range(1, 6):
        raise ValueError("assumption decision_impact must be from 1 to 5")
    if not isinstance(gap, int) or isinstance(gap, bool) or gap not in range(1, 6):
        raise ValueError("assumption evidence_gap must be from 1 to 5")
    if not isinstance(sample_target, int) or isinstance(sample_target, bool) or sample_target < 1:
        raise ValueError("assumption sample_target must be positive")
    if failure_decision not in ("kill", "pivot"):
        raise ValueError("assumption failure_decision must be kill or pivot")
    if not isinstance(classes, Sequence) or isinstance(classes, str):
        raise ValueError("assumption required_evidence_classes must be a list")
    required_classes = list(classes)
    if not required_classes or any(item not in EVIDENCE_SOURCE_CLASSES for item in required_classes):
        raise ValueError("assumption required_evidence_classes are unsupported")
    return {
        "assumption_id": _ref(value.get("assumption_id", ""), "assumption_id"),
        "category": category,
        "decision_impact": impact,
        "evidence_gap": gap,
        "priority": impact * gap,
        "test_id": _ref(value.get("test_id", ""), "test_id"),
        "smallest_test_ref": _ref(value.get("smallest_test_ref", ""), "smallest_test_ref"),
        "success_criterion_ref": _ref(value.get("success_criterion_ref", ""), "success_criterion_ref"),
        "failure_criterion_ref": _ref(value.get("failure_criterion_ref", ""), "failure_criterion_ref"),
        "inconclusive_criterion_ref": _ref(value.get("inconclusive_criterion_ref", ""), "inconclusive_criterion_ref"),
        "failure_decision": failure_decision,
        "scope_segment_ref": _ref(value.get("scope_segment_ref", ""), "scope_segment_ref"),
        "sample_target": sample_target,
        "deadline_at": _stamp(value.get("deadline_at", ""), "deadline_at"),
        "cost_cap_ref": _ref(value.get("cost_cap_ref", ""), "cost_cap_ref"),
        "owner_ref": _ref(value.get("owner_ref", ""), "owner_ref"),
        "required_evidence_classes": required_classes,
        "precommitted_at": _stamp(value.get("precommitted_at", ""), "precommitted_at"),
    }


def build_initial_gtm_hypothesis(*, discovery_id: str, beachhead_segment_ref: str, buyer_ref: str, user_ref: str, current_alternative_ref: str, value_proposition_ref: str, pricing_hypothesis_ref: str, initial_channel_ref: str, first_cohort_ref: str, learning_metric_refs: Sequence[str]) -> dict[str, Any]:
    """Prepare the initial go-to-market hypothesis without inventing market proof."""
    safe_buyer = _ref(buyer_ref, "buyer_ref")
    safe_user = _ref(user_ref, "user_ref")
    if safe_buyer == safe_user:
        raise ValueError("buyer_ref and user_ref must be distinct")
    return _artifact(
        INITIAL_GTM_HYPOTHESIS_SCHEMA_VERSION,
        discovery_id,
        {
            "beachhead_segment_ref": _ref(beachhead_segment_ref, "beachhead_segment_ref"),
            "buyer_ref": safe_buyer,
            "user_ref": safe_user,
            "current_alternative_ref": _ref(current_alternative_ref, "current_alternative_ref"),
            "value_proposition_ref": _ref(value_proposition_ref, "value_proposition_ref"),
            "pricing_hypothesis_ref": _ref(pricing_hypothesis_ref, "pricing_hypothesis_ref"),
            "initial_channel_ref": _ref(initial_channel_ref, "initial_channel_ref"),
            "first_cohort_ref": _ref(first_cohort_ref, "first_cohort_ref"),
            "learning_metric_refs": _refs(learning_metric_refs, "learning_metric_refs"),
        },
        status="prepared_not_observed",
    )


def build_discovery_decision_receipt(*, discovery_id: str, problem_ref: str, segment_ref: str, segment_definition_state: str, decision: str, problem_gate: str, precommitted_test_ids: Sequence[str], eligible_evidence_refs: Sequence[str], rejected_hypothesis_ids: Sequence[str], residual_risk_refs: Sequence[str], next_route: str) -> dict[str, Any]:
    """Record a bounded decision that can be consumed without transcript replay.

    ``solution_work_permitted`` and ``missing_audience_evidence_refs`` are
    derived here, never supplied, so a receipt cannot claim permitted solution
    work while its audience is unknown, synthetic-only, or non-recruitable.
    """
    if decision not in DECISIONS or problem_gate not in PROBLEM_GATES:
        raise ValueError("decision or problem_gate is unsupported")
    if (problem_gate == "validated") != (decision == "persevere"):
        raise ValueError("only a validated problem may persevere")
    audience_defined = audience_is_defined(segment_definition_state)
    if decision == "persevere" and not audience_defined:
        raise ValueError("only an explicit, recruitable or behaviorally observed segment may persevere")
    solution_work_permitted = audience_defined and problem_gate == "validated"
    safe_route = _ref(next_route, "next_route")
    if not solution_work_permitted and safe_route != DISCOVERY_CONTINUATION_ROUTE:
        raise ValueError("blocked solution work must route back to product-discovery-validation")
    return _artifact(
        DISCOVERY_DECISION_RECEIPT_SCHEMA_VERSION,
        discovery_id,
        {
            "problem_ref": _ref(problem_ref, "problem_ref"),
            "segment_ref": _ref(segment_ref, "segment_ref"),
            "segment_definition_state": _segment_definition_state(segment_definition_state),
            "decision": decision,
            "problem_gate": problem_gate,
            "solution_work_permitted": solution_work_permitted,
            "missing_audience_evidence_refs": missing_audience_evidence_refs(segment_definition_state),
            "precommitted_test_ids": _refs(precommitted_test_ids, "precommitted_test_ids"),
            "eligible_evidence_refs": _refs(eligible_evidence_refs, "eligible_evidence_refs", minimum=0),
            "rejected_hypothesis_ids": _refs(rejected_hypothesis_ids, "rejected_hypothesis_ids", minimum=0),
            "residual_risk_refs": _refs(residual_risk_refs, "residual_risk_refs", minimum=0),
            "next_route": safe_route,
        },
        status="derived_from_supplied_metadata",
    )
