from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from .loop_observation_validation import (
    activation_errors,
    status_errors,
    turn_errors,
)
from .loop_validation_primitives import (
    MAX_METADATA_TEXT,
    forbidden_fields,
    is_metadata_text,
    is_storage_id,
    is_utc_rfc3339,
    key_errors,
)


LOOP_GOAL_DRIVER_OBSERVATION_SCHEMA: Final[str] = (
    "loop_goal_driver_observation/v1"
)
_DIGEST = re.compile(r"[0-9a-f]{64}")
_OBSERVATION_KEYS: Final = frozenset(
    "schema_version observation_id loop_id session_ref goal_command_sha256 "
    "observation_source observed_at activation turns summary privacy".split()
)


def parse_loop_goal_driver_observation(
    value: object,
    *,
    expected_loop_id: str = "",
    expected_goal_command_sha256: str = "",
    expected_first_turn_index: int | None = None,
) -> dict[str, object]:
    """Return a closed metadata record or raise with all boundary errors."""
    errors = validate_loop_goal_driver_observation(
        value,
        expected_loop_id=expected_loop_id,
        expected_goal_command_sha256=expected_goal_command_sha256,
        expected_first_turn_index=expected_first_turn_index,
        allow_derived_status=False,
    )
    if errors:
        raise ValueError("; ".join(errors))
    assert isinstance(value, Mapping)
    activation = value["activation"]
    turns = value["turns"]
    assert isinstance(activation, Mapping)
    assert isinstance(turns, list)
    return {
        "schema_version": value["schema_version"],
        "observation_id": value["observation_id"],
        "loop_id": value["loop_id"],
        "session_ref": value["session_ref"],
        "goal_command_sha256": value["goal_command_sha256"],
        "observation_source": value["observation_source"],
        "observed_at": value["observed_at"],
        "activation": {
            "status": activation["status"],
            "evidence_refs": list(activation["evidence_refs"]),
        },
        "turns": [
            {
                "turn_index": turn["turn_index"],
                "session_ref": turn["session_ref"],
                "from_phase": turn["from_phase"],
                "to_phase": turn["to_phase"],
                "phase_gate": turn["phase_gate"],
                "turn_ended_evidence_refs": list(
                    turn["turn_ended_evidence_refs"]
                ),
                "phase_gate_evidence_refs": list(
                    turn["phase_gate_evidence_refs"]
                ),
            }
            for turn in turns
            if isinstance(turn, Mapping)
        ],
        "summary": value["summary"],
        "privacy": value["privacy"],
    }


def validate_loop_goal_driver_observation(
    value: object,
    expected_loop_id: str = "",
    expected_goal_command_sha256: str = "",
    *,
    expected_first_turn_index: int | None = None,
    allow_derived_status: bool = True,
) -> list[str]:
    """Return every metadata-only native-goal observation error."""
    if not isinstance(value, Mapping):
        return ["loop goal driver observation must be an object"]
    allowed = (
        _OBSERVATION_KEYS | {"native_goal_status"}
        if allow_derived_status
        else _OBSERVATION_KEYS
    )
    errors = key_errors(
        value,
        allowed,
        "loop goal driver observation",
        required=_OBSERVATION_KEYS,
    )
    forbidden = sorted(forbidden_fields(value))
    if forbidden:
        errors.append(
            f"loop goal driver observation has forbidden fields: {forbidden}"
        )
    if value.get("schema_version") != LOOP_GOAL_DRIVER_OBSERVATION_SCHEMA:
        errors.append(
            f"schema_version must be {LOOP_GOAL_DRIVER_OBSERVATION_SCHEMA}"
        )
    for field in ("observation_id", "loop_id", "session_ref"):
        if not is_storage_id(value.get(field)):
            errors.append(f"{field} must be a storage-safe non-empty id")
    if expected_loop_id and value.get("loop_id") != expected_loop_id:
        errors.append(f"loop_id must match expected loop_id {expected_loop_id}")
    digest = value.get("goal_command_sha256")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        errors.append("goal_command_sha256 must be a lowercase 64-hex digest")
    if expected_goal_command_sha256 and digest != expected_goal_command_sha256:
        errors.append("goal_command_sha256 must match the prepared goal command")
    if value.get("observation_source") not in {
        "hermes_host",
        "wrapper",
        "operator",
    }:
        errors.append(
            "observation_source must be hermes_host, wrapper, or operator"
        )
    if not is_utc_rfc3339(value.get("observed_at")):
        errors.append("observed_at must be a UTC RFC3339 timestamp")
    if value.get("privacy") != "metadata_only":
        errors.append("privacy must be metadata_only")
    if not is_metadata_text(value.get("summary")):
        errors.append(
            f"summary must be single-line metadata of at most {MAX_METADATA_TEXT} characters"
        )
    errors.extend(activation_errors(value.get("activation")))
    errors.extend(
        turn_errors(
            value.get("turns"),
            value.get("session_ref"),
            expected_first_turn_index=expected_first_turn_index,
        )
    )
    if "native_goal_status" in value:
        errors.extend(status_errors(value.get("native_goal_status")))
    return errors


def native_goal_ref_errors(value: object) -> list[str]:
    """Return closed native-goal transition projection errors."""
    if not isinstance(value, Mapping):
        return ["native_goal must be an object"]
    expected = frozenset(
        {
            "observation_id",
            "session_ref",
            "goal_command_sha256",
            "turn_index",
        }
    )
    errors = key_errors(value, expected, "native_goal")
    for field in ("observation_id", "session_ref"):
        if not is_storage_id(value.get(field)):
            errors.append(f"native_goal.{field} must be a storage-safe id")
    digest = value.get("goal_command_sha256")
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        errors.append(
            "native_goal.goal_command_sha256 must be a lowercase 64-hex digest"
        )
    turn_index = value.get("turn_index")
    if (
        isinstance(turn_index, bool)
        or not isinstance(turn_index, int)
        or turn_index < 1
    ):
        errors.append("native_goal.turn_index must be a positive integer")
    return errors
