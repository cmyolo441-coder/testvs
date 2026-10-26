"""Transactional single-artifact legacy reactivation."""

from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..plugin_bundle.omh.memory_governance import canonical_memory_scope
from ..system.local_store import atomic_write_json, read_json_object_result
from ..system.paths import OmhPaths
from ._memory_migration_reactivation_data import (
    artifact_identity,
    find_active,
    find_operation,
    matching_review,
    new_review,
    operation_steps,
    prepare,
    safe_target,
    target_artifact,
)
from .memory_store import apply_memory_operation_step, run_memory_operation

class OperationRunner(Protocol):
    def __call__(self, paths: OmhPaths, *, operation_id: str, operation_type: str, steps: Sequence[Mapping[str, object]], step_writer: Callable[[OmhPaths, dict[str, str]], None], now: datetime | None = None) -> dict[str, Any]: ...


WriteHook = Callable[[str], None]


def reactivate(
    paths: OmhPaths,
    artifact_id: str,
    *,
    review_id: str,
    apply: bool = False,
    artifact_kind: str | None = None,
    operation_runner: OperationRunner | None = None,
    write_hook: WriteHook | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Reactivate one reviewed v1 record, scope item, or block under the store lock."""
    moment = now or datetime.now(timezone.utc)
    source, unsafe = find_active(paths, artifact_id, artifact_kind)
    existing = find_operation(paths, artifact_id)
    if source is None and unsafe:
        return _result(False, "symlink_or_path_escape")
    if source is None and existing is None:
        return _result(False, "artifact_not_found")
    if source is not None and source["kind"] not in {"record", "scope_item", "block"}:
        return _result(False, "unsupported_artifact_kind")
    if source is not None:
        try:
            canonical_memory_scope(source["artifact"]["scope"])
        except (KeyError, ValueError):
            return _result(False, "scope_invalid")
    if existing is not None and not _review_matches_operation(existing, review_id):
        return _result(False, "matching_immutable_review_required")
    if source is not None:
        prior = matching_review(paths, review_id, source)
        if prior is None:
            return _result(False, "matching_immutable_review_required")
        _preflight, reason = prepare(source, "preflight_artifact", "preflight_review", prior, moment)
        if _preflight is None:
            return _result(False, reason)
    if not apply:
        return _result(False, "apply_required")
    operation_id, steps = _operation(source, existing, paths, review_id)
    if existing is not None and existing.get("state") == "completed":
        return _completed(existing, paths, steps)
    def writer(active_paths: OmhPaths, step: dict[str, str]) -> None:
        if step["name"] not in {"replace_artifact", "write_immutable_review"}:
            apply_memory_operation_step(active_paths, step)
            return
        target = safe_target(active_paths, step["target"])
        if step["name"] == "replace_artifact":
            current = target_artifact(active_paths, step["target"], step["scope"], step["key"])
            if current is not None and _is_reactivated(current, step["key"], step["source"]):
                return
            if current is None:
                raise ValueError("artifact_not_found")
            review = matching_review(active_paths, step["revision"], current)
            if review is None:
                raise ValueError("matching_immutable_review_required")
            value, reason = prepare(current, step["key"], step["source"], review, moment)
            if value is None:
                raise ValueError(reason)
            _write(target, value, step["name"], write_hook)
            return
        current = target_artifact(active_paths, step["source"], "", step["revision"])
        if current is None or not _is_reactivated(current, step["revision"], step["key"]):
            raise ValueError("reactivation_artifact_missing")
        stored, error = read_json_object_result(target)
        if not error and isinstance(stored, dict) and stored.get("review_id") == step["key"]:
            return
        review = _review_for_current(active_paths, step["scope"], current, moment)
        _write(target, review, step["name"], write_hook)

    runner = operation_runner or run_memory_operation
    record = runner(paths, operation_id=operation_id, operation_type="memory_reactivation", steps=steps, step_writer=writer, now=moment)
    return _completed(record, paths, steps)


def _operation(source: dict[str, Any] | None, existing: dict[str, Any] | None, paths: OmhPaths, review_id: str) -> tuple[str, list[dict[str, str]]]:
    if existing is not None:
        raw_steps = existing.get("steps", [])
        steps = [{str(key): str(value) for key, value in step.items() if key != "state"} for step in raw_steps if isinstance(step, dict)]
        return str(existing["operation_id"]), steps
    assert source is not None
    target = source["path"].relative_to(paths.memory_dir).as_posix()
    new_id = f"migrated_{secrets.token_hex(12)}"
    new_review = f"review_{secrets.token_hex(12)}"
    review_dir = "block_reviews" if source["kind"] == "block" else "reviews"
    id_key = {"record": "record_id", "scope_item": "item_id", "block": "block_id"}[source["kind"]]
    old_id = str(source["artifact"].get(id_key, source["artifact"].get("label", "")))
    return f"reactivate_{secrets.token_hex(12)}", operation_steps(old_id, new_id, new_review, review_id, target, f"{review_dir}/{new_review}.json")


def _step_values(steps: list[dict[str, str]]) -> tuple[str, str, str]:
    if len(steps) != 2 or steps[0].get("name") != "replace_artifact" or steps[1].get("name") != "write_immutable_review":
        raise ValueError("invalid_reactivation_operation")
    return steps[0]["key"], steps[1]["key"], steps[1]["scope"]


def _review_matches_operation(operation: dict[str, Any], review_id: str) -> bool:
    steps = operation.get("steps")
    return isinstance(steps, list) and len(steps) == 2 and isinstance(steps[1], dict) and steps[1].get("scope") == review_id


def _is_reactivated(source: dict[str, Any], new_id: str, review_id: str) -> bool:
    artifact = source["artifact"]
    admission = artifact.get("admission")
    key = {"record": "record_id", "scope_item": "item_id", "block": "block_id"}[source["kind"]]
    return artifact.get("schema_version", "").endswith("/v2") and artifact.get(key) == new_id and isinstance(admission, dict) and admission.get("review_id") == review_id


def _review_for_current(paths: OmhPaths, prior_id: str, source: dict[str, Any], now: datetime) -> dict[str, object]:
    directories = (paths.memory_dir / "block_reviews", paths.memory_dir / "reviews") if source["kind"] == "block" else (paths.memory_dir / "reviews", paths.memory_dir / "block_reviews")
    for directory in directories:
        prior_path = directory / f"{prior_id}.json"
        if prior_path.is_symlink():
            continue
        prior, error = read_json_object_result(prior_path)
        if not error and isinstance(prior, dict) and prior.get("review_id") == prior_id:
            return new_review(source["artifact"], str(source["artifact"]["admission"]["review_id"]), prior, now)
    raise ValueError("matching_immutable_review_required")


def _write(target: object, value: dict[str, object], name: str, hook: WriteHook | None) -> None:
    if hook is not None:
        hook(name)
    atomic_write_json(target, value, private=True)


def _completed(record: dict[str, Any], paths: OmhPaths, steps: list[dict[str, str]]) -> dict[str, object]:
    new_id, review_id, _prior = _step_values(steps)
    source = target_artifact(paths, steps[0]["target"], steps[0]["scope"], new_id)
    identity = artifact_identity(source) if source is not None else {}
    return {"schema_version": "memory_reactivation/v1", "applied": record.get("state") == "completed", "reason_code": "reactivated" if record.get("state") == "completed" else str(record.get("state", "interrupted")), "operation_id": str(record.get("operation_id", "")), "review_id": review_id, "artifact_identity": identity, "receipt": record.get("receipt", {})}


def _result(applied: bool, reason: str) -> dict[str, object]:
    return {"schema_version": "memory_reactivation/v1", "applied": applied, "reason_code": reason, "artifact_identity": {}}
