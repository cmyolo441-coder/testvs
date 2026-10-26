from __future__ import annotations

from pathlib import Path

from ..system.local_store import atomic_write_json, ensure_dir, file_lock, read_json_object_result
from ..system.paths import OmhPaths

from .design_direction_iterations_validation import validate_design_direction_iteration


def write_design_direction_iteration(paths: OmhPaths, record: dict[str, object]) -> dict[str, object]:
    """Create one local trajectory; identical root preparation is a no-write replay."""
    _validate(record)
    path = _path(paths, str(record["iteration_id"]))
    with file_lock(path, private=True):
        existing, error = read_json_object_result(path)
        if error:
            raise ValueError(error)
        if existing is not None:
            _validate(existing)
            if existing == record:
                return existing
            raise ValueError("design direction iteration already exists with different contents")
        ensure_dir(_directory(paths), private=True)
        atomic_write_json(path, record, private=True)
        return record


def update_design_direction_iteration(paths: OmhPaths, record: dict[str, object]) -> dict[str, object]:
    """Persist a validated trajectory transition under its immutable root identity."""
    _validate(record)
    path = _path(paths, str(record["iteration_id"]))
    with file_lock(path, private=True):
        existing, error = read_json_object_result(path)
        if error:
            raise ValueError(error)
        if existing is None:
            raise FileNotFoundError(str(record["iteration_id"]))
        _validate(existing)
        if existing == record:
            return existing
        if not _extends_current(existing, record):
            raise ValueError("iteration changed since the rendered revision; re-read before mutating")
        atomic_write_json(path, record, private=True)
        return record


def show_design_direction_iteration(paths: OmhPaths, iteration_id: str) -> dict[str, object]:
    record, error = read_json_object_result(_path(paths, iteration_id))
    if error:
        raise ValueError(error)
    if record is None:
        raise FileNotFoundError(iteration_id)
    _validate(record)
    return record


def list_design_direction_iterations(paths: OmhPaths) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(_directory(paths).glob("*.json")):
        record, error = read_json_object_result(path)
        if record is not None and error is None and not validate_design_direction_iteration(record):
            records.append(record)
    return records


def _directory(paths: OmhPaths) -> Path:
    return paths.omh_home / "design-direction-iterations"


def _path(paths: OmhPaths, iteration_id: str) -> Path:
    if not iteration_id.startswith("design-direction-iteration-") or len(iteration_id) > 160 or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-" for character in iteration_id):
        raise ValueError("iteration_id is invalid")
    return _directory(paths) / f"{iteration_id}.json"


def _extends_current(existing: dict[str, object], incoming: dict[str, object]) -> bool:
    if existing["terminal"] != {"outcome": "OPEN", "reason": None, "accepted_revision_digest": None, "accepted_option_ref": None}:
        return False
    prior = existing["snapshots"]
    next_snapshots = incoming["snapshots"]
    if not isinstance(prior, list) or not isinstance(next_snapshots, list):
        return False
    return next_snapshots == prior or (len(next_snapshots) == len(prior) + 1 and next_snapshots[:-1] == prior)


def _validate(record: dict[str, object]) -> None:
    errors = validate_design_direction_iteration(record)
    if errors:
        raise ValueError("; ".join(errors))
