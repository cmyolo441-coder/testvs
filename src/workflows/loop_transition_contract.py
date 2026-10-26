from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .loop_observation_contract import native_goal_ref_errors
from .loop_phase_policy import LOOP_PHASES, PHASE_TARGETS
from .loop_validation_primitives import (
    MAX_EVIDENCE_REFS,
    is_metadata_text,
    is_storage_id,
    is_utc_rfc3339,
    key_errors,
)


LOOP_PHASE_TRANSITION_SCHEMA: Final[str] = "loop_phase_transition/v1"
_TRANSITION_KEYS: Final = frozenset(
    "schema_version transition_id sequence loop_id from_phase to_phase "
    "from_phase_generation to_phase_generation phase_gate transition_kind "
    "cause source_ref evidence_refs observed_at native_goal".split()
)
_TRANSITION_REQUIRED_KEYS: Final = _TRANSITION_KEYS - {
    "from_phase_generation",
    "to_phase_generation",
    "native_goal",
}


def validate_loop_phase_transition(value: object) -> list[str]:
    """Return every contract error in one phase-transition record."""
    if not isinstance(value, Mapping):
        return ["loop phase transition must be an object"]
    errors = key_errors(
        value,
        _TRANSITION_KEYS,
        "loop phase transition",
        required=_TRANSITION_REQUIRED_KEYS,
    )
    if value.get("schema_version") != LOOP_PHASE_TRANSITION_SCHEMA:
        errors.append(f"schema_version must be {LOOP_PHASE_TRANSITION_SCHEMA}")
    for field in ("transition_id", "loop_id"):
        if not is_storage_id(value.get(field)):
            errors.append(f"{field} must be a storage-safe non-empty id")
    sequence = value.get("sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 1
    ):
        errors.append("sequence must be a positive integer")
    elif value.get("transition_id") != f"phase-transition-{sequence}":
        errors.append(f"transition_id must be phase-transition-{sequence}")
    from_phase = value.get("from_phase")
    to_phase = value.get("to_phase")
    gate = value.get("phase_gate")
    if from_phase not in LOOP_PHASES or to_phase not in LOOP_PHASES:
        errors.append(
            f"illegal loop phase transition: {from_phase!r} -> {to_phase!r}"
        )
    elif gate:
        target = PHASE_TARGETS.get(from_phase)
        if target is None or target[1] != to_phase:
            errors.append(
                f"illegal loop phase transition: {from_phase!r} -> {to_phase!r}"
            )
        elif gate != target[2]:
            errors.append(
                f"phase_gate must be {target[2]} for {from_phase} -> {to_phase}"
            )
        errors.extend(_reference_list_errors(value.get("evidence_refs")))
    elif value.get("transition_kind") == "observed_progress":
        errors.append("observed_progress transitions require a phase_gate")
    elif not isinstance(value.get("evidence_refs"), list):
        errors.append("evidence_refs must be a list")
    for field in ("transition_kind", "cause", "source_ref"):
        if not is_metadata_text(value.get(field)):
            errors.append(f"{field} must be bounded single-line metadata")
    if not is_utc_rfc3339(value.get("observed_at")):
        errors.append("observed_at must be a UTC RFC3339 timestamp")
    errors.extend(_generation_errors(value))
    if "native_goal" in value:
        errors.extend(native_goal_ref_errors(value.get("native_goal")))
    return errors


def _generation_errors(value: Mapping[object, object]) -> list[str]:
    has_from = "from_phase_generation" in value
    has_to = "to_phase_generation" in value
    if has_from != has_to:
        return ["phase generation fields must appear together"]
    if not has_from:
        return []
    from_generation = value.get("from_phase_generation")
    to_generation = value.get("to_phase_generation")
    if (
        isinstance(from_generation, bool)
        or not isinstance(from_generation, int)
        or from_generation < 0
    ):
        return ["from_phase_generation must be a non-negative integer"]
    if (
        isinstance(to_generation, bool)
        or not isinstance(to_generation, int)
        or to_generation != from_generation + 1
    ):
        return ["to_phase_generation must increment from_phase_generation"]
    return []


def _reference_list_errors(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["evidence_refs must contain observed-progress evidence"]
    if (
        len(value) > MAX_EVIDENCE_REFS
        or len(set(item for item in value if isinstance(item, str)))
        != len(value)
        or not all(is_metadata_text(item) for item in value)
    ):
        return ["evidence_refs must contain unique bounded metadata refs"]
    return []
