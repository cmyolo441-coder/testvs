from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .executor_local_workflow_selection import (
    JsonValue,
    WorkflowInput,
    WorkflowRecord,
    availability_for,
    bounded_safe,
    candidate_for,
    dispatchability_for,
    environment_name,
    evidence_reference,
    is_workflow,
    observation_time_relation,
)


EXECUTOR_LOCAL_WORKFLOW_SCHEMA_VERSION: Final = "executor_local_workflow/v1"
EXECUTOR_LOCAL_WORKFLOW_FALLBACK: Final = (
    "Keep the parent handoff prompt and dispatch mode unchanged; do not invoke the candidate."
)
EXECUTOR_LOCAL_WORKFLOW_CLAIM_BOUNDARY: Final = (
    "Prepared executor-local workflow metadata is not evidence of installation, loading, invocation, dispatch, "
    "execution, verification, review, CI, merge readiness, or merge."
)

_ROOT_KEYS: Final = frozenset({"schema_version", "profile", "status", "routed_workflow", "candidate", "availability", "dispatchability", "fallback", "claim_boundary"})
_CANDIDATE_KEYS: Final = frozenset({"kind", "skill_id", "invocation", "rationale", "selection_basis"})
_INVOCATION_KEYS: Final = frozenset({"mode", "syntax", "template", "message_placeholder"})
_AVAILABILITY_KEYS: Final = frozenset({"status", "basis", "profile", "skill_id", "scope", "recorded_at", "observed_at", "evidence_ref"})
_DISPATCHABILITY_KEYS: Final = frozenset({"handoff_dispatchable", "candidate_invocation_dispatchable", "reason"})
_OBSERVED_STATUSES: Final = frozenset({"observed_available", "observed_unavailable"})


def build_executor_local_workflow(
    *,
    profile: str,
    routed_workflow: str,
    parent_handoff_dispatchable: bool,
    availability_evidence: WorkflowInput | None = None,
) -> WorkflowRecord | None:
    candidate = candidate_for(profile, routed_workflow)
    if candidate is None:
        return None
    availability = availability_for(profile, routed_workflow, availability_evidence)
    status = str(availability["status"])
    dispatchability = dispatchability_for(profile, status, parent_handoff_dispatchable)
    binding: WorkflowRecord = {
        "schema_version": EXECUTOR_LOCAL_WORKFLOW_SCHEMA_VERSION,
        "profile": profile,
        "status": status,
        "routed_workflow": routed_workflow,
        "candidate": candidate,
        "availability": availability,
        "dispatchability": dispatchability,
        "fallback": EXECUTOR_LOCAL_WORKFLOW_FALLBACK,
        "claim_boundary": EXECUTOR_LOCAL_WORKFLOW_CLAIM_BOUNDARY,
    }
    errors = validate_executor_local_workflow(binding)
    if errors:
        raise ValueError("; ".join(errors))
    return binding


def validate_executor_local_workflow(binding: WorkflowInput) -> list[str]:
    errors = _key_errors("binding", binding, _ROOT_KEYS)
    errors.extend(_literal_errors(binding))
    candidate, availability, dispatchability = binding.get("candidate"), binding.get("availability"), binding.get("dispatchability")
    if isinstance(candidate, Mapping):
        errors.extend(_candidate_errors(binding, candidate))
    else:
        errors.append("candidate must be a mapping")
    if isinstance(availability, Mapping):
        errors.extend(_availability_errors(binding, availability))
    else:
        errors.append("availability must be a mapping")
    if isinstance(dispatchability, Mapping):
        errors.extend(_dispatchability_errors(binding, dispatchability))
    else:
        errors.append("dispatchability must be a mapping")
    return errors


def _literal_errors(binding: WorkflowInput) -> list[str]:
    errors: list[str] = []
    expected = {
        "schema_version": EXECUTOR_LOCAL_WORKFLOW_SCHEMA_VERSION,
        "fallback": EXECUTOR_LOCAL_WORKFLOW_FALLBACK,
        "claim_boundary": EXECUTOR_LOCAL_WORKFLOW_CLAIM_BOUNDARY,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            errors.append(f"{key} must equal its executor_local_workflow/v1 literal")
    profile = binding.get("profile")
    if profile not in {"codex", "hermes", "omx-runtime", "pi-family-runtime", "omc-runtime"}:
        errors.append("profile must be a mapped executor profile")
    workflow = binding.get("routed_workflow")
    if not is_workflow(workflow):
        errors.append("routed_workflow must be a bounded canonical skill id")
    return errors


def _candidate_errors(binding: WorkflowInput, candidate: Mapping[str, JsonValue]) -> list[str]:
    errors = _key_errors("candidate", candidate, _CANDIDATE_KEYS)
    if candidate.get("selection_basis") != "final_guarded_recommended_workflow":
        errors.append("candidate.selection_basis must be final_guarded_recommended_workflow")
    if not bounded_safe(candidate.get("rationale")):
        errors.append("candidate.rationale must be a bounded nonsensitive string")
    if candidate.get("skill_id") != binding.get("routed_workflow"):
        errors.append("candidate.skill_id must equal routed_workflow")
    expected = candidate_for(str(binding.get("profile", "")), str(binding.get("routed_workflow", "")))
    if expected is not None:
        for key in ("kind", "skill_id", "invocation"):
            if candidate.get(key) != expected[key]:
                errors.append(f"candidate.{key} does not match the executor profile mapping")
    invocation = candidate.get("invocation")
    if isinstance(invocation, Mapping):
        errors.extend(_key_errors("candidate.invocation", invocation, _INVOCATION_KEYS))
        template = invocation.get("template")
        placeholder = invocation.get("message_placeholder")
        if isinstance(template, str) and template.count("{message}") != (1 if template else 0):
            errors.append("candidate.invocation.template has an invalid message placeholder count")
        if placeholder != ("{message}" if template else ""):
            errors.append("candidate.invocation.message_placeholder does not match template")
    else:
        errors.append("candidate.invocation must be a mapping")
    return errors


def _availability_errors(binding: WorkflowInput, availability: Mapping[str, JsonValue]) -> list[str]:
    errors = _key_errors("availability", availability, _AVAILABILITY_KEYS)
    status = availability.get("status")
    if binding.get("status") != status:
        errors.append("status must mirror availability.status")
    if availability.get("profile") != binding.get("profile") or availability.get("skill_id") != binding.get("routed_workflow"):
        errors.append("availability profile and skill_id must match the selected candidate")
    if status == "unknown":
        if (
            availability.get("basis"),
            availability.get("scope"),
            availability.get("recorded_at"),
            availability.get("observed_at"),
            availability.get("evidence_ref"),
        ) != ("prepared_mapping", {}, "", "", ""):
            errors.append("unknown availability must contain only prepared mapping metadata")
    elif status in _OBSERVED_STATUSES:
        scope = availability.get("scope")
        if availability.get("basis") != "operator_recorded_snapshot":
            errors.append("observed availability requires operator_recorded_snapshot basis")
        if not isinstance(scope, Mapping) or set(scope) != {"environment"} or not environment_name(scope.get("environment")):
            errors.append("observed availability requires one bounded environment scope")
        if not evidence_reference(availability.get("evidence_ref")) or not observation_time_relation(
            availability.get("recorded_at"),
            availability.get("observed_at"),
        ):
            errors.append("observed availability requires bounded timestamped evidence")
    else:
        errors.append("availability.status must be unknown, observed_available, or observed_unavailable")
    return errors


def _dispatchability_errors(binding: WorkflowInput, dispatchability: Mapping[str, JsonValue]) -> list[str]:
    errors = _key_errors("dispatchability", dispatchability, _DISPATCHABILITY_KEYS)
    handoff = dispatchability.get("handoff_dispatchable")
    candidate = dispatchability.get("candidate_invocation_dispatchable")
    if type(handoff) is not bool or type(candidate) is not bool:
        errors.append("dispatchability flags must be exact booleans")
        return errors
    profile = str(binding.get("profile", ""))
    status = str(binding.get("status", ""))
    if handoff and profile != "codex":
        errors.append("only a Codex parent handoff may be dispatchable")
    expected = dispatchability_for(profile, status, handoff)
    if candidate != expected["candidate_invocation_dispatchable"] or dispatchability.get("reason") != expected["reason"]:
        errors.append("dispatchability does not match status, profile, and parent handoff")
    return errors


def _key_errors(label: str, value: Mapping[str, JsonValue], expected: frozenset[str]) -> list[str]:
    if set(value) == expected:
        return []
    return [f"{label} must contain exactly: {', '.join(sorted(expected))}"]
