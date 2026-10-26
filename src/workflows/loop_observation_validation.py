from __future__ import annotations

from collections.abc import Mapping

from .loop_phase_policy import PHASE_TARGETS
from .loop_validation_primitives import (
    is_storage_id,
    key_errors,
    observation_reference_errors,
)


TURN_KEYS = frozenset(
    "turn_index session_ref from_phase to_phase phase_gate "
    "turn_ended_evidence_refs phase_gate_evidence_refs".split()
)
STATUS_KEYS = frozenset(
    "activation_status continuation_status session_ref last_turn_index".split()
)


def activation_errors(value: object) -> list[str]:
    """Return activation contract errors."""
    if not isinstance(value, Mapping):
        return ["activation must be an object"]
    errors = key_errors(
        value, frozenset({"status", "evidence_refs"}), "activation"
    )
    if value.get("status") != "observed":
        errors.append("activation.status must be observed")
    errors.extend(
        observation_reference_errors(
            value.get("evidence_refs"), "activation.evidence_refs"
        )
    )
    return errors


def turn_errors(
    value: object,
    session_ref: object,
    *,
    expected_first_turn_index: int | None,
) -> list[str]:
    """Return contiguous same-session observed-turn errors."""
    if not isinstance(value, list) or not value:
        return ["turns must be a non-empty list of observed turns"]
    errors: list[str] = []
    first_index = (
        value[0].get("turn_index") if isinstance(value[0], Mapping) else None
    )
    if (
        isinstance(first_index, bool)
        or not isinstance(first_index, int)
        or first_index < 1
    ):
        errors.append("turns[0].turn_index must be a positive integer")
        first_index = expected_first_turn_index or 1
    if (
        expected_first_turn_index is not None
        and first_index != expected_first_turn_index
    ):
        errors.append(
            "turns[0].turn_index: expected turn_index "
            f"{expected_first_turn_index}, got {first_index}"
        )
    if first_index == 1 and len(value) < 2:
        errors.append("turns must contain at least two observed turns")
    for offset, turn in enumerate(value):
        label = f"turns[{offset}]"
        expected_index = first_index + offset
        if not isinstance(turn, Mapping):
            errors.append(f"{label} must be an object")
            continue
        errors.extend(key_errors(turn, TURN_KEYS, label))
        if turn.get("turn_index") != expected_index:
            errors.append(
                f"{label}.turn_index: expected turn_index "
                f"{expected_index}, got {turn.get('turn_index')}"
            )
        if turn.get("session_ref") != session_ref:
            errors.append(f"{label}.session_ref must match session_ref")
        target = PHASE_TARGETS.get(turn.get("from_phase"))
        if target is None or target[1] != turn.get("to_phase"):
            errors.append(f"{label} has an illegal loop phase transition")
        elif turn.get("phase_gate") != target[2]:
            errors.append(f"{label}.phase_gate must be {target[2]}")
        errors.extend(
            observation_reference_errors(
                turn.get("turn_ended_evidence_refs"),
                f"{label}.turn_ended_evidence_refs",
            )
        )
        errors.extend(
            observation_reference_errors(
                turn.get("phase_gate_evidence_refs"),
                f"{label}.phase_gate_evidence_refs",
            )
        )
    return errors


def status_errors(value: object) -> list[str]:
    """Return derived native-goal status errors."""
    if not isinstance(value, Mapping):
        return ["native_goal_status must be an object"]
    errors = key_errors(value, STATUS_KEYS, "native_goal_status")
    if value.get("activation_status") != "observed":
        errors.append("native_goal_status.activation_status must be observed")
    if value.get("continuation_status") != "observed":
        errors.append(
            "native_goal_status.continuation_status must be observed"
        )
    if not is_storage_id(value.get("session_ref")):
        errors.append("native_goal_status.session_ref must be a storage-safe id")
    last_turn = value.get("last_turn_index")
    if (
        isinstance(last_turn, bool)
        or not isinstance(last_turn, int)
        or last_turn < 2
    ):
        errors.append("native_goal_status.last_turn_index must be at least two")
    return errors
