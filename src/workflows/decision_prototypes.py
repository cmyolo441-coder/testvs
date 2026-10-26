"""Public preparation, observation, and receipt APIs for decision prototypes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from ..coding.isolation import ISOLATION_SCHEMA_VERSION
from .decision_prototypes_observations import observation_errors
from .decision_prototypes_schema import ARTIFACT_KEYS, CLAIM_BOUNDARY, DECISION_PROTOTYPE_SCHEMA_VERSION, PROPOSAL_KEYS, execution_errors, mapping, promotion_errors, proposal_errors


class DecisionPrototypeError(ValueError):
    """Raised when a prototype would exceed its isolated experiment contract."""


def prepare_decision_prototype(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare one declared experiment without invoking an executor."""
    errors = proposal_errors(proposal)
    if errors:
        raise DecisionPrototypeError(errors[0])
    declared = deepcopy(dict(proposal))
    workspace = mapping(declared["workspace"])
    commands = declared["commands"]
    return {
        "schema_version": DECISION_PROTOTYPE_SCHEMA_VERSION,
        **declared,
        "isolation_schema_version": ISOLATION_SCHEMA_VERSION,
        "execution": {
            "status": "available_not_observed" if mapping(declared["executor"])["available"] else "prepared_not_observed",
            "evidence_class": "prepared_not_observed",
            "adapter_id": "",
            "handoff": {
                "status": "prepared_not_observed",
                "commands": deepcopy(commands),
                "expected_observations": [item["expected_observation"] for item in commands],
                "declared_workspace": {"identity": workspace["identity"], "write_boundary": workspace["write_boundary"]},
                "budget": deepcopy(declared["budget"]),
            },
            "consumed_budget": {"time_seconds": 0, "tool_count": 0, "file_count": 0, "command_count": 0},
        },
        "observations": [],
        "promotion": {"production_code_permitted": False, "accepted_plan_ref": "", "accepted_plan_status": "", "implementation_handoff_ref": ""},
        "claim_boundary": CLAIM_BOUNDARY,
    }


def validate_decision_prototype(value: Mapping[str, Any]) -> list[str]:
    """Return contract errors for malformed prepared or observed metadata."""
    if not isinstance(value, Mapping):
        return ["decision_prototype must be an object"]
    if "schema_version" not in value:
        return proposal_errors(value)
    proposal = {key: value.get(key) for key in PROPOSAL_KEYS}
    errors = proposal_errors(proposal)
    if set(value) != ARTIFACT_KEYS:
        errors.append("decision_prototype keys are invalid")
    if value.get("schema_version") != DECISION_PROTOTYPE_SCHEMA_VERSION:
        errors.append("decision_prototype schema_version is invalid")
    if value.get("isolation_schema_version") != ISOLATION_SCHEMA_VERSION:
        errors.append("decision_prototype isolation schema is invalid")
    if value.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("decision_prototype claim boundary is invalid")
    execution = mapping(value.get("execution"))
    errors.extend(execution_errors(execution, proposal))
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) > 1:
        errors.append("decision_prototype observations must contain at most one adapter observation")
    elif observations:
        observation = mapping(observations[0])
        errors.extend(observation_errors(observation, proposal))
        if execution.get("status") != observation.get("state") or execution.get("adapter_id") != observation.get("adapter_id") or execution.get("consumed_budget") != observation.get("consumed_budget"):
            errors.append("decision_prototype execution must match its adapter observation")
    elif execution.get("status") != ("available_not_observed" if mapping(proposal.get("executor")).get("available") else "prepared_not_observed"):
        errors.append("decision_prototype execution must match its unobserved availability")
    errors.extend(promotion_errors(value.get("promotion")))
    return errors


def observe_decision_prototype(prototype: Mapping[str, Any], adapter_observation: Mapping[str, Any]) -> dict[str, Any]:
    """Record exactly one adapter observation; this function never executes commands."""
    errors = validate_decision_prototype(prototype)
    if errors:
        raise DecisionPrototypeError(errors[0])
    proposal = {key: prototype.get(key) for key in PROPOSAL_KEYS}
    if not bool(mapping(proposal.get("executor")).get("available")):
        raise DecisionPrototypeError("decision_prototype cannot observe without an available executor")
    if prototype["observations"]:
        raise DecisionPrototypeError("decision_prototype already has an observation")
    observation = deepcopy(dict(adapter_observation))
    errors = observation_errors(observation, proposal)
    if errors:
        raise DecisionPrototypeError(errors[0])
    result = deepcopy(dict(prototype))
    result["observations"] = [observation]
    result["execution"] = {**result["execution"], "status": observation["state"], "evidence_class": "adapter_observed", "adapter_id": observation["adapter_id"], "consumed_budget": observation["consumed_budget"]}
    return result


def compact_decision_prototype_receipt(prototype: Mapping[str, Any]) -> dict[str, Any]:
    """Return the planning receipt without commands, paths, or transcripts."""
    errors = validate_decision_prototype(prototype)
    if errors:
        raise DecisionPrototypeError(errors[0])
    observation = prototype["observations"][0] if prototype["observations"] else {}
    measurements = observation.get("measurements", [])
    return {
        "schema_version": "decision_prototype_receipt/v1",
        "decision_id": prototype["decision_id"],
        "context_decision_ref": prototype["context_decision_ref"],
        "execution_status": prototype["execution"]["status"],
        "evidence_class": prototype["execution"]["evidence_class"],
        "supported_option": observation.get("supported_option", ""),
        "rejected_options": observation.get("rejected_options", []),
        "confidence": observation.get("confidence", ""),
        "residual_risk": observation.get("unresolved_questions", []),
        "evidence_refs": [row["evidence_ref"] for row in measurements],
        "evidence_limits": ["bounded experiment", "declared measurement only", "not production validation"],
        "prototype_code_ref": observation.get("prototype_code_ref", ""),
        "promotion": deepcopy(prototype["promotion"]),
        "claim_boundary": CLAIM_BOUNDARY,
    }
