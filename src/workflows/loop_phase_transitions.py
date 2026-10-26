from __future__ import annotations

from collections.abc import Iterable, Mapping

from .loop_goal_observations import (
    LOOP_GOAL_DRIVER_OBSERVATION_SCHEMA,
    native_goal_status,
    parse_loop_goal_driver_observation,
    validate_loop_goal_driver_observation,
    validate_loop_goal_driver_observation_history,
)
from .loop_phase_policy import LOOP_PHASES, phase_target
from .loop_transition_contract import (
    LOOP_PHASE_TRANSITION_SCHEMA,
    validate_loop_phase_transition,
)
from .loop_transition_history import validate_loop_phase_history


def set_loop_phase(
    cycle: dict[str, object],
    *,
    to_phase: str,
    transition_kind: str,
    cause: str,
    source_ref: str,
    observed_at: str,
    phase_gate: str = "",
    evidence_refs: Iterable[str] = (),
    native_goal: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Record one phase occurrence and make it the current cycle phase."""
    existing_value = cycle.get("phase_transitions", [])
    if not isinstance(existing_value, list):
        raise ValueError("phase_transitions must be a list")
    existing = list(existing_value)
    for expected_sequence, item in enumerate(existing, start=1):
        errors = validate_loop_phase_transition(item)
        if errors:
            raise ValueError(errors[0])
        if not isinstance(item, Mapping) or item.get("sequence") != (
            expected_sequence
        ):
            raise ValueError(
                "phase transition sequence must be contiguous at "
                f"{expected_sequence}"
            )
    generation = cycle.get("phase_generation", 0)
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("phase_generation must be a non-negative integer")
    sequence = len(existing) + 1
    transition: dict[str, object] = {
        "schema_version": LOOP_PHASE_TRANSITION_SCHEMA,
        "transition_id": f"phase-transition-{sequence}",
        "sequence": sequence,
        "loop_id": cycle.get("loop_id", ""),
        "from_phase": cycle.get("phase", ""),
        "to_phase": to_phase,
        "from_phase_generation": generation,
        "to_phase_generation": generation + 1,
        "phase_gate": phase_gate,
        "transition_kind": transition_kind,
        "cause": cause,
        "source_ref": source_ref,
        "evidence_refs": list(evidence_refs),
        "observed_at": observed_at,
    }
    if native_goal is not None:
        transition["native_goal"] = dict(native_goal)
    errors = validate_loop_phase_transition(transition)
    if errors:
        raise ValueError(errors[0])
    existing.append(transition)
    cycle["phase_transitions"] = existing
    cycle["phase_generation"] = generation + 1
    cycle["phase"] = to_phase
    return cycle


def transition_loop_phase(
    cycle: dict[str, object],
    *,
    to_phase: str,
    phase_gate: str,
    transition_kind: str,
    cause: str,
    source_ref: str,
    evidence_refs: Iterable[str],
    observed_at: str,
    native_goal: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Apply one evidence-backed native transition within current authority."""
    current_phase = str(cycle.get("phase", ""))
    action, expected_phase, expected_gate = phase_target(current_phase)
    if expected_phase != to_phase or expected_gate != phase_gate:
        raise ValueError(
            f"illegal loop phase transition: {current_phase!r} -> {to_phase!r}"
        )
    wait_reason = cycle.get("wait_reason")
    if wait_reason not in {None, "none"}:
        raise ValueError(
            f"loop phase cannot advance while wait_reason is {wait_reason}"
        )
    envelope = cycle.get("authority_envelope")
    if isinstance(envelope, Mapping):
        allowed = envelope.get("allowed_actions")
        if isinstance(allowed, list) and action not in allowed:
            raise ValueError(
                f"{action} is outside the current authority envelope"
            )
    return set_loop_phase(
        cycle,
        to_phase=to_phase,
        phase_gate=phase_gate,
        transition_kind=transition_kind,
        cause=cause,
        source_ref=source_ref,
        evidence_refs=evidence_refs,
        observed_at=observed_at,
        native_goal=native_goal,
    )


__all__ = [
    "LOOP_GOAL_DRIVER_OBSERVATION_SCHEMA",
    "LOOP_PHASES",
    "LOOP_PHASE_TRANSITION_SCHEMA",
    "native_goal_status",
    "parse_loop_goal_driver_observation",
    "phase_target",
    "set_loop_phase",
    "transition_loop_phase",
    "validate_loop_goal_driver_observation",
    "validate_loop_goal_driver_observation_history",
    "validate_loop_phase_history",
    "validate_loop_phase_transition",
]
