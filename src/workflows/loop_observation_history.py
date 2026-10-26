from __future__ import annotations

from collections.abc import Mapping

from .loop_observation_contract import validate_loop_goal_driver_observation


def validate_loop_goal_driver_observation_history(
    observations: object,
    *,
    expected_loop_id: str,
) -> list[str]:
    """Return cross-record observation stream errors."""
    if not isinstance(observations, list):
        return ["goal_driver_observations must be a list"]
    errors: list[str] = []
    seen_ids: set[object] = set()
    stream: tuple[object, object] | None = None
    expected_turn = 1
    for index, observation in enumerate(observations):
        label = f"goal_driver_observations[{index}]"
        if not isinstance(observation, Mapping):
            errors.append(f"{label} must be an object")
            continue
        observation_id = observation.get("observation_id")
        if observation_id in seen_ids:
            errors.append(f"{label}.observation_id must be unique")
        seen_ids.add(observation_id)
        current_stream = (
            observation.get("session_ref"),
            observation.get("goal_command_sha256"),
        )
        if stream is None:
            stream = current_stream
        elif current_stream != stream:
            errors.append(
                f"{label} must continue the original session and goal command stream"
            )
        row_errors = validate_loop_goal_driver_observation(
            observation,
            expected_loop_id=expected_loop_id,
            expected_first_turn_index=expected_turn,
        )
        errors.extend(f"{label}.{error}" for error in row_errors)
        turns = observation.get("turns")
        if isinstance(turns, list) and turns:
            last = turns[-1]
            if isinstance(last, Mapping):
                last_index = last.get("turn_index")
                if isinstance(last_index, int) and not isinstance(
                    last_index, bool
                ):
                    expected_turn = last_index + 1
        status = observation.get("native_goal_status")
        if isinstance(status, Mapping) and status.get("last_turn_index") != (
            expected_turn - 1
        ):
            errors.append(
                f"{label}.native_goal_status.last_turn_index must equal {expected_turn - 1}"
            )
    return errors


def native_goal_status(cycle: Mapping[str, object]) -> dict[str, object]:
    """Derive activation and continuation from accepted observation history."""
    default: dict[str, object] = {
        "activation_status": "not_observed",
        "continuation_status": "not_observed",
        "session_ref": "",
        "last_turn_index": 0,
    }
    observations = cycle.get("goal_driver_observations", [])
    if not isinstance(observations, list) or not observations:
        return default
    first = observations[0]
    expected_loop_id = str(cycle.get("loop_id", ""))
    if not expected_loop_id and isinstance(first, Mapping):
        expected_loop_id = str(first.get("loop_id", ""))
    if validate_loop_goal_driver_observation_history(
        observations,
        expected_loop_id=expected_loop_id,
    ):
        return default
    latest = observations[-1]
    assert isinstance(latest, Mapping)
    turns = [
        turn
        for observation in observations
        if isinstance(observation, Mapping)
        for turn in observation.get("turns", [])
        if isinstance(turn, Mapping)
    ]
    indexes = [
        turn.get("turn_index")
        for turn in turns
        if isinstance(turn.get("turn_index"), int)
        and not isinstance(turn.get("turn_index"), bool)
    ]
    return {
        "activation_status": "observed",
        "continuation_status": "observed",
        "session_ref": latest.get("session_ref", ""),
        "last_turn_index": max(indexes, default=0),
    }
