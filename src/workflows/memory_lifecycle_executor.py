"""Store-owned execution adapter for report-first memory lifecycle plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..system.paths import OmhPaths
from ._memory_lifecycle_model import LifecycleMutation, LifecyclePlan
from .memory_lifecycle import make_lifecycle_receipt
from .memory_store import OperationPreflight, run_memory_operation


def execute_memory_lifecycle(paths: OmhPaths, plan: LifecyclePlan, *, preflight: OperationPreflight | None = None) -> dict[str, object]:
    """Apply one plan through the shared operation lock and durable step log."""
    steps = lifecycle_operation_steps(plan)
    operation = run_memory_operation(
        paths,
        operation_id=plan.operation_id,
        operation_type=f"memory_lifecycle_{plan.operation_type}",
        steps=steps,
        now=plan.now,
        preflight=preflight,
    )
    if not _same_steps(operation.get("steps"), steps):
        raise ValueError("lifecycle_operation_mismatch")
    if operation.get("state") != "completed":
        return {"operation_id": plan.operation_id, "state": str(operation.get("state", ""))}
    outcomes = [_outcome(step) for step in operation["steps"]]
    receipt = make_lifecycle_receipt(plan, outcomes)
    receipt["created_at"] = str(operation["created_at"])
    receipt["completed_at"] = str(operation["updated_at"])
    return {"receipt": receipt}


def lifecycle_operation_steps(plan: LifecyclePlan) -> list[dict[str, Any]]:
    """Encode lifecycle mutations as validated, replayable store steps."""
    return [_step(mutation, plan.revision) for mutation in plan.mutations]


def _step(mutation: LifecycleMutation, revision: int) -> dict[str, Any]:
    step: dict[str, Any] = {
        "name": mutation.name,
        "action": _action(mutation),
        "target": mutation.target,
        "key": mutation.target_id,
        "scope": mutation.artifact_kind,
        "revision": str(revision),
    }
    if mutation.action == "move":
        if mutation.source is None:
            raise ValueError("lifecycle move source is missing")
        step["source"] = mutation.source
    elif mutation.action == "write":
        if not isinstance(mutation.payload, Mapping):
            raise ValueError("lifecycle write payload is invalid")
        step["payload"] = dict(mutation.payload)
    elif mutation.action == "rewrite_jsonl":
        if not isinstance(mutation.payload, tuple) or not all(isinstance(item, Mapping) for item in mutation.payload):
            raise ValueError("lifecycle JSONL payload is invalid")
        step["payload"] = [dict(item) for item in mutation.payload]
    return step


def _action(mutation: LifecycleMutation) -> str:
    actions = {"write": "write_json", "delete": "delete", "rewrite_jsonl": "rewrite_jsonl", "move": "move"}
    try:
        return actions[mutation.action]
    except KeyError as exc:
        raise ValueError("unsupported lifecycle mutation") from exc


def _same_steps(value: object, expected: list[dict[str, Any]]) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    return all(
        isinstance(stored, Mapping)
        and {key: item for key, item in stored.items() if key not in {"state", "outcome"}} == step
        for stored, step in zip(value, expected, strict=True)
    )


def _outcome(step: Mapping[str, object]) -> dict[str, object]:
    action = str(step["action"])
    outcome = str(step.get("outcome", ""))
    if action == "delete" and outcome == "already_absent":
        return {"target_id": str(step["key"]), "artifact_kind": str(step["scope"]), "outcome": "already_absent", "reason_code": "already_absent"}
    names = {"write_json": "written", "delete": "removed", "rewrite_jsonl": "rewritten", "move": "moved"}
    return {"target_id": str(step["key"]), "artifact_kind": str(step["scope"]), "outcome": names[action], "reason_code": "applied"}
