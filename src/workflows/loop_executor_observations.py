"""Closed, bounded adapter observations; never checkpoint or phase authority."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import re
from typing import Final, TypedDict

from ..coding.executor_capability_snapshots import JsonValue
from ..system.metadata_safety import is_sensitive_metadata_text
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

OBSERVATION_SCHEMA: Final = "loop_executor_goal_observation/v1"
MAX_OBSERVATIONS: Final = 128


class LoopDriverError(ValueError):
    code: str

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class DriverStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BUDGET_LIMITED = "budget_limited"
    CLOSED = "closed"


class ExecutorObservation(TypedDict):
    schema_version: str
    loop_id: str
    driver_id: str
    owner: str
    session_ref: str
    objective_sha256: str
    sequence: int
    observation_id: str
    observed_at: str
    status: str
    evidence_refs: list[str]


def observation_record(value: ExecutorObservation) -> dict[str, JsonValue]:
    """Serialize a parsed observation into the existing JSON store's value type."""
    return {
        "schema_version": value["schema_version"], "loop_id": value["loop_id"],
        "driver_id": value["driver_id"], "owner": value["owner"], "session_ref": value["session_ref"],
        "objective_sha256": value["objective_sha256"], "sequence": value["sequence"],
        "observation_id": value["observation_id"], "observed_at": value["observed_at"],
        "status": value["status"], "evidence_refs": list(value["evidence_refs"]),
    }


def opaque_ref(value: JsonValue, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}", value):
        raise LoopDriverError(f"driver_invalid_{field}")
    if ".." in value or is_sensitive_metadata_text(value) or is_sensitive_metadata_text(value.upper()):
        raise LoopDriverError(f"driver_invalid_{field}")
    return value


def digest_ref(value: JsonValue) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LoopDriverError("driver_invalid_digest")
    return value


def observed_time(value: JsonValue) -> datetime:
    if not isinstance(value, str) or len(value) > 40:
        raise LoopDriverError("driver_invalid_observed_at")
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LoopDriverError("driver_invalid_observed_at") from exc
    if parsed.tzinfo is None:
        raise LoopDriverError("driver_invalid_observed_at")
    return parsed


def evidence_refs(value: JsonValue) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise LoopDriverError("driver_invalid_evidence_refs")
    refs = [opaque_ref(ref, "evidence_ref") for ref in value]
    if len(set(refs)) != len(refs):
        raise LoopDriverError("driver_duplicate_evidence_refs")
    return refs


@dataclass(frozen=True, slots=True)
class ObservationIdentity:
    loop_id: str
    driver_id: str
    owner: str
    session_ref: str


def parse_executor_observation(value: Mapping[str, JsonValue], identity: ObservationIdentity) -> ExecutorObservation:
    """Parse once at ingestion/read, rejecting payload slots and foreign streams."""
    if set(value) != set(ExecutorObservation.__annotations__) or value.get("schema_version") != OBSERVATION_SCHEMA:
        raise LoopDriverError("driver_invalid_observation_shape")
    for field, expected in (("loop_id", identity.loop_id), ("driver_id", identity.driver_id),
                            ("owner", identity.owner), ("session_ref", identity.session_ref)):
        if value[field] != expected:
            raise LoopDriverError(f"driver_observation_{field}_mismatch")
    sequence = value["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 1 <= sequence <= 2**53 - 1:
        raise LoopDriverError("driver_invalid_sequence")
    raw_status = value["status"]
    if not isinstance(raw_status, str):
        raise LoopDriverError("driver_invalid_status")
    try:
        status = DriverStatus(raw_status)
    except ValueError as exc:
        raise LoopDriverError("driver_invalid_status") from exc
    stamp = value["observed_at"]
    parsed = observed_time(stamp)
    if parsed > datetime.now(timezone.utc):
        raise LoopDriverError("driver_future_observation")
    return ExecutorObservation(
        schema_version=OBSERVATION_SCHEMA, loop_id=identity.loop_id, driver_id=identity.driver_id,
        owner=identity.owner, session_ref=identity.session_ref, objective_sha256=digest_ref(value["objective_sha256"]),
        sequence=sequence, observation_id=opaque_ref(value["observation_id"], "observation_id"),
        observed_at=parsed.isoformat().replace("+00:00", "Z"), status=status.value,
        evidence_refs=evidence_refs(value["evidence_refs"]),
    )


def checked_history(value: JsonValue, identity: ObservationIdentity) -> list[ExecutorObservation]:
    if not isinstance(value, list) or len(value) > MAX_OBSERVATIONS:
        raise LoopDriverError("driver_observation_history_bound")
    history: list[ExecutorObservation] = []
    ids: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise LoopDriverError("driver_invalid_observation_shape")
        entry = parse_executor_observation(raw, identity)
        if entry["observation_id"] in ids:
            raise LoopDriverError("driver_duplicate_observation")
        if history and (entry["sequence"] <= history[-1]["sequence"] or
                        observed_time(entry["observed_at"]) < observed_time(history[-1]["observed_at"])):
            raise LoopDriverError("driver_stale_observation")
        ids.add(entry["observation_id"])
        history.append(entry)
    return history
