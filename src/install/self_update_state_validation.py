"""Trusted path parsing for persisted staged self-update state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from ..core.errors import OmhError
except ImportError:  # pragma: no cover - direct-source installer smoke.
    from core.errors import OmhError


def parse_state(root: Path, state: dict[str, Any], legacy: Path) -> dict[str, Any]:
    """Parse state paths into trusted generation paths without rewriting state."""
    migration = state.get("migration")
    if not isinstance(migration, dict):
        _invalid("migration")
    active = state.get("active")
    if not isinstance(active, dict):
        _invalid("active generation")
    migration_status = str(migration.get("status", ""))
    _parse_active(root, active, legacy, migration_status)
    _parse_pointer(root, state.get("pointer"), active, migration_status)
    previous = state.get("previous_known_good")
    if previous is not None:
        parse_generation_entry(root, previous, "previous known-good generation")
    parse_marker(root, state.get("activation_in_progress"))
    selected = state.get("recovery_selected")
    if selected is not None and not isinstance(selected, str):
        _invalid("recovery selection")
    return state


def parse_generation_entry(root: Path, entry: object, label: str) -> Path:
    if not isinstance(entry, dict):
        _invalid(label)
    identifier = entry.get("id")
    path = entry.get("path")
    if not isinstance(identifier, str) or not _is_name(identifier):
        _invalid(f"{label} id")
    if not isinstance(path, str):
        _invalid(f"{label} path")
    return require_generation_path(root, path, identifier, label)


def parse_marker(root: Path, marker: object) -> tuple[Path, Path] | None:
    if marker is None:
        return None
    if not isinstance(marker, dict):
        _invalid("activation marker")
    candidate = marker.get("candidate")
    previous = marker.get("previous")
    phase = marker.get("phase")
    if not isinstance(candidate, str) or not isinstance(previous, str):
        _invalid("activation marker")
    if phase not in {"pre_switch", "post_switch"}:
        _invalid("activation marker phase")
    return (
        require_generation_path(root, candidate, None, "activation marker candidate"),
        require_generation_path(root, previous, None, "activation marker previous"),
    )


def require_generation_path(root: Path, value: Path | str, identifier: str | None, label: str) -> Path:
    """Require one lexical, canonical, non-link child of ``root/generations``."""
    if not isinstance(value, (Path, str)):
        _invalid(label)
    path = Path(value)
    generations = root / "generations"
    canonical_generations = root.resolve() / "generations"
    parents = {generations, canonical_generations}
    if identifier is not None:
        if not _is_name(identifier) or path not in {parent / identifier for parent in parents}:
            _invalid(label)
    elif path.parent not in parents or not _is_name(path.name):
        _invalid(label)
    if _is_link(path):
        _invalid(label)
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise OmhError(f"cannot use invalid self-update {label}") from exc
    if resolved.parent != canonical_generations:
        _invalid(label)
    return path


def parse_gc_entries(root: Path, state: dict[str, Any]) -> set[str]:
    active = parse_generation_entry(root, state.get("active"), "active generation")
    previous = state.get("previous_known_good")
    names = {active.name, "bootstrap-legacy"}
    if previous is not None:
        names.add(parse_generation_entry(root, previous, "previous known-good generation").name)
    return names


def _parse_pointer(root: Path, pointer: object, active: dict[str, Any], migration: str) -> None:
    if not isinstance(pointer, dict) or pointer.get("path") != str(root / "current"):
        _invalid("generation pointer")
    target = pointer.get("target")
    if active.get("kind") == "legacy" and migration == "pending":
        if target != "":
            _invalid("generation pointer")
        return
    if not isinstance(target, str) or target.replace("\\", "/") != f"generations/{active.get('id', '')}":
        _invalid("generation pointer")


def _parse_active(root: Path, active: dict[str, Any], legacy: Path, migration: str) -> None:
    if active.get("kind") != "legacy":
        parse_generation_entry(root, active, "active generation")
        return
    if migration != "pending" or active.get("path") != str(legacy) or active.get("id") != legacy.name:
        _invalid("legacy active generation")


def _is_name(value: str) -> bool:
    return value not in {"", ".", ".."} and Path(value).name == value


def _is_link(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(junction and junction())


def same_existing_path(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return left.resolve() == right.resolve()


def _invalid(label: str) -> None:
    raise OmhError(f"cannot use invalid self-update {label}")
