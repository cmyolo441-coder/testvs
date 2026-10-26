"""Adapter-observed result validation for one decision prototype."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_prototypes_schema import OBSERVATION_ADAPTER_SCHEMA_VERSION, OBSERVATION_STATES, consumed_budget_errors, line, mapping, opaque


_OBSERVATION_KEYS = frozenset({"adapter_contract", "adapter_id", "state", "measurements", "interpretation", "confidence", "unresolved_questions", "supported_option", "rejected_options", "decision", "cleanup_status", "prototype_code_ref", "actual_workspace", "consumed_budget"})


def observation_errors(value: Any, proposal: Mapping[str, Any]) -> list[str]:
    observation = mapping(value)
    if set(observation) != _OBSERVATION_KEYS or observation.get("adapter_contract") != OBSERVATION_ADAPTER_SCHEMA_VERSION or not opaque(observation.get("adapter_id")) or observation.get("state") not in OBSERVATION_STATES:
        return ["decision_prototype observation adapter contract is invalid"]
    errors: list[str] = []
    measurements = observation.get("measurements")
    if not isinstance(measurements, list) or len(measurements) > 3 or any(not measurement(item) for item in measurements):
        errors.append("decision_prototype measurements are invalid")
    for key in ("interpretation", "prototype_code_ref"):
        if observation.get(key) and not line(observation.get(key), 240):
            errors.append(f"decision_prototype observation {key} is invalid")
    unresolved = observation.get("unresolved_questions")
    if observation.get("confidence") not in {"low", "medium", "high"} or not isinstance(unresolved, list) or len(unresolved) > 3 or not all(line(item, 240) for item in unresolved):
        errors.append("decision_prototype observation confidence or unresolved questions are invalid")
    workspace = mapping(proposal.get("workspace"))
    if observation.get("actual_workspace") != {"identity": workspace.get("identity"), "write_boundary": workspace.get("write_boundary")}:
        errors.append("decision_prototype actual_workspace must match the declared workspace boundary")
    errors.extend(consumed_budget_errors(observation.get("consumed_budget"), mapping(proposal.get("budget")), require_activity=True))
    raw_alternatives = proposal.get("alternatives")
    alternatives = raw_alternatives if isinstance(raw_alternatives, list) and all(isinstance(item, str) for item in raw_alternatives) else []
    selected = observation.get("supported_option")
    rejected = observation.get("rejected_options")
    rejected_values = rejected if isinstance(rejected, list) and all(isinstance(item, str) for item in rejected) else []
    if observation.get("state") == "observed":
        if not isinstance(measurements, list) or not measurements or selected not in alternatives or not isinstance(rejected, list) or len(rejected_values) != len(rejected) or set(rejected_values) != set(alternatives) - {selected}:
            errors.append("decision_prototype observed result requires measurements and declared option conclusions")
    elif selected or rejected:
        errors.append("decision_prototype timeout or inconclusive result must not select an option")
    if observation.get("decision") not in {"keep", "discard"} or observation.get("cleanup_status") not in {"observed", "failed", "not_observed"}:
        errors.append("decision_prototype observation decision or cleanup status is invalid")
    elif observation.get("decision") == "discard" and observation.get("cleanup_status") != "observed":
        errors.append("decision_prototype discarded prototype requires observed cleanup")
    return errors


def measurement(value: Any) -> bool:
    item = mapping(value)
    return set(item) == {"metric", "value", "evidence_ref"} and line(item.get("metric"), 80) and line(item.get("value"), 80) and opaque(item.get("evidence_ref"))
