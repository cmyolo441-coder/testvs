"""Persistence adapter for validated decision-prototype artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..local_store import atomic_write_json, ensure_dir, read_json_object
from ..system.metadata_safety import require_opaque_metadata_ref
from ..system.paths import OmhPaths
from ..workflows.decision_prototypes import DecisionPrototypeError, validate_decision_prototype


_DECISION_PROTOTYPE_DIRECTORY = "decision-prototypes"


def persist_decision_prototype(paths: OmhPaths, prototype: Mapping[str, Any]) -> dict[str, Any]:
    """Persist one validated metadata artifact under the active OMH runtime home."""
    errors = validate_decision_prototype(prototype)
    if errors:
        raise DecisionPrototypeError(errors[0])
    decision_id = _decision_id(prototype.get("decision_id"))
    record = dict(prototype)
    path = _artifact_path(paths, decision_id)
    ensure_dir(path.parent, private=True)
    atomic_write_json(path, record, private=True)
    return record


def read_decision_prototype(paths: OmhPaths, decision_id: str) -> dict[str, Any] | None:
    """Read one valid persisted artifact without repairing malformed state."""
    record = read_json_object(_artifact_path(paths, _decision_id(decision_id)))
    if record is None or validate_decision_prototype(record):
        return None
    return record


def _artifact_path(paths: OmhPaths, decision_id: str) -> Path:
    return paths.runtime_dir / _DECISION_PROTOTYPE_DIRECTORY / f"{decision_id}.json"


def _decision_id(value: Any) -> str:
    try:
        decision_id = require_opaque_metadata_ref(value, field="decision_prototype decision_id")
    except ValueError as exc:
        raise DecisionPrototypeError(str(exc)) from exc
    if not decision_id.isascii() or not decision_id.replace("-", "").isalnum() or decision_id.startswith("-") or len(decision_id) > 64:
        raise DecisionPrototypeError("decision_prototype decision_id is invalid")
    return decision_id
