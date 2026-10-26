"""Closed schema checks shared by decision-prototype preparation and persistence."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Final

from ..system.metadata_safety import is_body_shaped_metadata_text, is_raw_pii_shaped, is_sensitive_metadata_text, require_opaque_metadata_ref


DECISION_PROTOTYPE_SCHEMA_VERSION: Final[str] = "decision_prototype/v1"
OBSERVATION_ADAPTER_SCHEMA_VERSION: Final[str] = "decision_prototype_observation/v1"
CLAIM_BOUNDARY: Final[str] = "A decision prototype is a bounded, isolated experiment. Its observations are not implementation, product validation, review, CI, merge-readiness, or permission to promote prototype code."
EXPERIMENT_KINDS: Final[tuple[str, ...]] = ("wireframe", "cli_spike", "api_probe", "fixture", "timing_probe", "test_harness", "mocked_interaction")
OBSERVATION_STATES: Final[tuple[str, ...]] = ("observed", "timeout", "inconclusive")
WORKSPACE_KINDS: Final[tuple[str, ...]] = ("scratch_directory", "temporary_worktree")
PROPOSAL_KEYS: Final[frozenset[str]] = frozenset({"decision_id", "context_decision_ref", "question", "alternatives", "hypothesis", "target", "experiment_kind", "budget", "executor", "workspace", "measurement_method", "stop_conditions", "commands"})
ARTIFACT_KEYS: Final[frozenset[str]] = PROPOSAL_KEYS | {"schema_version", "isolation_schema_version", "execution", "observations", "promotion", "claim_boundary"}
BUDGET_KEYS: Final[frozenset[str]] = frozenset({"time_seconds", "tool_count", "file_count", "command_count"})
_DECISION_ID: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_STOP_CONDITIONS: Final[frozenset[str]] = frozenset({"hypothesis_supported", "hypothesis_refuted", "timeout", "inconclusive", "budget_exhausted"})


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def opaque(value: Any) -> bool:
    try:
        _ = require_opaque_metadata_ref(value, field="decision_prototype reference")
    except ValueError:
        return False
    return True


def line(value: Any, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not is_body_shaped_metadata_text(value, limit=limit) and not is_sensitive_metadata_text(value) and not is_raw_pii_shaped(value)


def proposal_errors(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(value) != PROPOSAL_KEYS:
        errors.append("decision_prototype proposal keys are invalid")
    decision_id = value.get("decision_id")
    if not isinstance(decision_id, str) or not _DECISION_ID.fullmatch(decision_id):
        errors.append("decision_prototype decision_id is invalid")
    if not opaque(value.get("context_decision_ref")):
        errors.append("decision_prototype context_decision_ref must be an opaque reference")
    question = value.get("question")
    if not line(question, 240) or str(question).count("?") != 1 or not str(question).rstrip().endswith("?"):
        errors.append("decision_prototype must declare exactly one question")
    for key in ("hypothesis", "measurement_method"):
        if not line(value.get(key), 240):
            errors.append(f"decision_prototype {key} must be bounded safe metadata")
    alternatives = value.get("alternatives")
    if not isinstance(alternatives, list) or not 2 <= len(alternatives) <= 3 or not all(line(item, 80) for item in alternatives) or len(set(alternatives)) != len(alternatives):
        errors.append("decision_prototype alternatives must name two or three unique options")
    target = mapping(value.get("target"))
    if set(target) != {"kind", "ref"} or target.get("kind") not in {"task", "user"} or not opaque(target.get("ref")):
        errors.append("decision_prototype target is invalid")
    if value.get("experiment_kind") not in EXPERIMENT_KINDS:
        errors.append("decision_prototype experiment_kind is invalid")
    errors.extend(budget_errors(value.get("budget")))
    executor = mapping(value.get("executor"))
    limits = executor.get("capability_limits")
    if set(executor) != {"profile", "available", "capability_limits"} or not opaque(executor.get("profile")) or not isinstance(executor.get("available"), bool) or not isinstance(limits, list) or len(limits) > 4 or not all(opaque(item) for item in limits):
        errors.append("decision_prototype executor is invalid")
    workspace = mapping(value.get("workspace"))
    if set(workspace) != {"kind", "identity", "write_boundary"} or workspace.get("kind") not in WORKSPACE_KINDS or not opaque(workspace.get("identity")) or workspace.get("write_boundary") != "declared_workspace_only":
        errors.append("decision_prototype workspace write_boundary must be declared_workspace_only")
    stops = value.get("stop_conditions")
    if not isinstance(stops, list) or not stops or len(stops) > 3 or not all(isinstance(item, str) and item in _STOP_CONDITIONS for item in stops):
        errors.append("decision_prototype stop_conditions are invalid")
    commands = value.get("commands")
    budget = mapping(value.get("budget"))
    if not isinstance(commands, list) or not commands or any(not command(item) for item in commands):
        errors.append("decision_prototype commands are invalid")
    elif isinstance(budget.get("command_count"), int) and not isinstance(budget.get("command_count"), bool) and len(commands) > budget["command_count"]:
        errors.append("decision_prototype budget command_count must cover declared commands")
    return errors


def budget_errors(value: Any) -> list[str]:
    budget = mapping(value)
    limits = {"time_seconds": 3600, "tool_count": 3, "file_count": 10, "command_count": 5}
    if set(budget) != BUDGET_KEYS:
        return ["decision_prototype budget is invalid"]
    return [f"decision_prototype budget {key} must be bounded" for key, limit in limits.items() if isinstance(budget.get(key), bool) or not isinstance(budget.get(key), int) or not 1 <= budget[key] <= limit]


def command(value: Any) -> bool:
    item = mapping(value)
    return set(item) == {"command", "expected_observation"} and line(item.get("command"), 160) and line(item.get("expected_observation"), 80)


def execution_errors(value: Any, proposal: Mapping[str, Any]) -> list[str]:
    execution = mapping(value)
    allowed = {"available_not_observed", "prepared_not_observed", *OBSERVATION_STATES}
    if set(execution) != {"status", "evidence_class", "adapter_id", "handoff", "consumed_budget"} or execution.get("status") not in allowed or not isinstance(execution.get("adapter_id"), str):
        return ["decision_prototype execution is invalid"]
    available = bool(mapping(proposal.get("executor")).get("available"))
    expected_class = "adapter_observed" if execution.get("status") in OBSERVATION_STATES else "prepared_not_observed"
    if execution.get("evidence_class") != expected_class or available != (execution.get("status") != "prepared_not_observed"):
        return ["decision_prototype execution evidence class or availability is invalid"]
    errors = consumed_budget_errors(execution.get("consumed_budget"), mapping(proposal.get("budget")), require_activity=execution.get("status") in OBSERVATION_STATES)
    handoff = mapping(execution.get("handoff"))
    commands = proposal.get("commands")
    workspace = mapping(proposal.get("workspace"))
    expected_workspace = {"identity": workspace.get("identity"), "write_boundary": workspace.get("write_boundary")}
    expected_observations = [item["expected_observation"] for item in commands] if isinstance(commands, list) and all(isinstance(item, Mapping) for item in commands) else []
    if set(handoff) != {"status", "commands", "expected_observations", "declared_workspace", "budget"} or handoff.get("status") != "prepared_not_observed" or handoff.get("commands") != commands or handoff.get("expected_observations") != expected_observations or handoff.get("declared_workspace") != expected_workspace or handoff.get("budget") != proposal.get("budget"):
        errors.append("decision_prototype handoff must exactly match the declared commands, workspace, and budget")
    return errors


def consumed_budget_errors(value: Any, budget: Mapping[str, Any], *, require_activity: bool) -> list[str]:
    consumed = mapping(value)
    if set(consumed) != BUDGET_KEYS or any(isinstance(budget.get(key), bool) or not isinstance(budget.get(key), int) for key in BUDGET_KEYS) or any(isinstance(consumed.get(key), bool) or not isinstance(consumed.get(key), int) or consumed[key] < 0 or consumed[key] > budget[key] for key in BUDGET_KEYS):
        return ["decision_prototype consumed_budget must be bounded by the declared budget"]
    if require_activity and (consumed["time_seconds"] < 1 or consumed["command_count"] < 1):
        return ["decision_prototype adapter observation must consume time and a declared command"]
    return []


def promotion_errors(value: Any) -> list[str]:
    promotion = mapping(value)
    required = {"production_code_permitted", "accepted_plan_ref", "accepted_plan_status", "implementation_handoff_ref"}
    if set(promotion) != required or not isinstance(promotion.get("production_code_permitted"), bool):
        return ["decision_prototype promotion is invalid"]
    if promotion["production_code_permitted"] and (not opaque(promotion.get("accepted_plan_ref")) or promotion.get("accepted_plan_status") != "accepted" or not opaque(promotion.get("implementation_handoff_ref"))):
        return ["decision_prototype production promotion requires accepted_plan_ref, accepted status, and implementation_handoff_ref"]
    return []
