from __future__ import annotations

from collections.abc import Mapping

from .loop_observation_history import (
    validate_loop_goal_driver_observation_history,
)
from .loop_transition_contract import validate_loop_phase_transition


def validate_loop_phase_history(cycle: Mapping[str, object]) -> list[str]:
    """Validate loop-bound ordering and observation projections as one history."""
    transitions = cycle.get("phase_transitions", [])
    if not isinstance(transitions, list):
        return ["phase_transitions must be a list"]
    errors: list[str] = []
    modern_started = False
    previous: Mapping[object, object] | None = None
    loop_id = cycle.get("loop_id")
    for index, transition in enumerate(transitions):
        label = f"phase_transitions[{index}]"
        row_errors = validate_loop_phase_transition(transition)
        errors.extend(f"{label}.{error}" for error in row_errors)
        if not isinstance(transition, Mapping):
            continue
        if transition.get("loop_id") != loop_id:
            errors.append(f"{label}.loop_id must match the enclosing loop")
        expected_sequence = index + 1
        if transition.get("sequence") != expected_sequence:
            errors.append(
                f"{label}.phase transition sequence must be {expected_sequence}"
            )
        has_generation = "from_phase_generation" in transition
        if modern_started and not has_generation:
            errors.append(
                f"{label}.generation-aware history cannot return to legacy rows"
            )
        if has_generation:
            if (
                not modern_started
                and transition.get("from_phase_generation") != 0
            ):
                errors.append(
                    f"{label}.first generation-aware transition must start at generation 0"
                )
            if previous is not None:
                if previous.get("to_phase") != transition.get("from_phase"):
                    errors.append(
                        f"{label}.phase transition chain is disconnected"
                    )
                if modern_started and previous.get(
                    "to_phase_generation"
                ) != transition.get("from_phase_generation"):
                    errors.append(
                        f"{label}.phase generation chain is disconnected"
                    )
            modern_started = True
        elif previous is not None and not modern_started:
            if previous.get("to_phase") != transition.get("from_phase"):
                errors.append(f"{label}.phase transition chain is disconnected")
        previous = transition
    if transitions and previous is not None:
        if previous.get("to_phase") != cycle.get("phase"):
            errors.append(
                "phase transition history final phase must match cycle phase"
            )
        if modern_started and previous.get("to_phase_generation") != cycle.get(
            "phase_generation"
        ):
            errors.append(
                "phase transition history final generation must match cycle"
            )
    observations = cycle.get("goal_driver_observations", [])
    errors.extend(
        validate_loop_goal_driver_observation_history(
            observations,
            expected_loop_id=str(loop_id or ""),
        )
    )
    if isinstance(observations, list):
        errors.extend(_native_projection_errors(transitions, observations))
    return errors


def _native_projection_errors(
    transitions: list[object],
    observations: list[object],
) -> list[str]:
    errors: list[str] = []
    expected: dict[
        tuple[object, object, object, object], Mapping[object, object]
    ] = {}
    for observation in observations:
        if not isinstance(observation, Mapping):
            continue
        for turn in observation.get("turns", []):
            if not isinstance(turn, Mapping):
                continue
            key = (
                observation.get("observation_id"),
                observation.get("session_ref"),
                observation.get("goal_command_sha256"),
                turn.get("turn_index"),
            )
            expected[key] = turn
    matched: dict[tuple[object, object, object, object], int] = {}
    for transition in transitions:
        if not isinstance(transition, Mapping):
            continue
        native = transition.get("native_goal")
        if not isinstance(native, Mapping):
            continue
        key = (
            native.get("observation_id"),
            native.get("session_ref"),
            native.get("goal_command_sha256"),
            native.get("turn_index"),
        )
        matched[key] = matched.get(key, 0) + 1
        turn = expected.get(key)
        if turn is None:
            errors.append(
                "phase transition native_goal has no matching observation turn"
            )
            continue
        expected_refs = [
            *turn.get("turn_ended_evidence_refs", []),
            *turn.get("phase_gate_evidence_refs", []),
        ]
        if (
            transition.get("from_phase") != turn.get("from_phase")
            or transition.get("to_phase") != turn.get("to_phase")
            or transition.get("phase_gate") != turn.get("phase_gate")
            or transition.get("evidence_refs") != expected_refs
        ):
            errors.append(
                "phase transition native_goal projection does not match its turn"
            )
    for key in expected:
        if matched.get(key) != 1:
            errors.append(
                "phase transition history must contain exactly one "
                f"native_goal projection for {key[0]} turn {key[3]}"
            )
    return errors
