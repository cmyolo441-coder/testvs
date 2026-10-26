"""Selected-owner loop driver contract, independent of completion authority."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from typing import Final, TypedDict, assert_never

from ..coding.executor_capability_snapshots import JsonValue, validate_executor_capability_snapshot
from ..system.hashutil import sha256_text
from .loop_executor_observations import (
    DriverStatus, ExecutorObservation, LoopDriverError, ObservationIdentity,
    checked_history, digest_ref, evidence_refs, observation_record, observed_time, opaque_ref,
)

DRIVER_SCHEMA: Final = "loop_driver/v1"


class DriverRecord(TypedDict):
    schema_version: str
    driver_id: str
    kind: str
    owner: str
    capability_snapshot_digest: str | None
    session_ref: str | None
    objective_sha256: str
    observation_state: str
    observed_at: str | None
    advisory_snapshot: ExecutorObservation | None
    fallback_reason: str
    compatibility: str


def driver_record(value: DriverRecord) -> dict[str, JsonValue]:
    """Serialize the typed companion without widening the JSON storage contract."""
    latest = value["advisory_snapshot"]
    return {
        "schema_version": value["schema_version"], "driver_id": value["driver_id"],
        "kind": value["kind"], "owner": value["owner"], "session_ref": value["session_ref"],
        "capability_snapshot_digest": value["capability_snapshot_digest"],
        "objective_sha256": value["objective_sha256"], "observation_state": value["observation_state"],
        "observed_at": value["observed_at"], "advisory_snapshot": observation_record(latest) if latest else None,
        "fallback_reason": value["fallback_reason"], "compatibility": value["compatibility"],
    }


def record_digest(value: Mapping[str, JsonValue]) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def select_driver(objective: str, selection: Mapping[str, JsonValue] | None) -> tuple[DriverRecord, dict[str, JsonValue] | None]:
    """Capability-gate an explicit owner; static routing never enables controls."""
    digest_ref(objective)
    selected = selection or {"executor": "hermes", "work_kind": "non_coding", "capability_snapshot": None, "session_ref": None}
    if set(selected) != {"executor", "work_kind", "capability_snapshot", "session_ref"}:
        raise LoopDriverError("driver_invalid_selection_shape")
    owner = opaque_ref(selected["executor"], "executor")
    work_kind = selected["work_kind"]
    if work_kind not in ("coding", "non_coding"):
        raise LoopDriverError("driver_invalid_work_kind")
    session = None if selected["session_ref"] is None else opaque_ref(selected["session_ref"], "session_ref")
    snapshot = selected["capability_snapshot"]
    if snapshot is not None:
        if not isinstance(snapshot, dict) or validate_executor_capability_snapshot(snapshot):
            raise LoopDriverError("driver_invalid_capability_snapshot")
    capabilities = snapshot.get("capabilities", {}) if snapshot else {}
    goal = capabilities.get("resumable_goal", {}) if isinstance(capabilities, dict) else {}
    scope = goal.get("scope", {}) if isinstance(goal, dict) else {}
    eligible = (owner != "hermes" and work_kind == "coding" and session is not None and
                snapshot is not None and snapshot["executor"] == owner and isinstance(goal, dict) and
                goal.get("status") == "host_observed" and isinstance(scope, dict) and
                scope.get("session_ref") == session)
    if eligible and isinstance(goal, dict):
        if observed_time(goal["observed_at"]) > datetime.now(timezone.utc):
            raise LoopDriverError("driver_future_capability")
    fallback = ""
    if not eligible:
        fallback = "hermes_owned_or_non_coding" if owner == "hermes" or work_kind == "non_coding" else "goal_controls_not_observed_for_selected_session"
    kind = "external_executor_goal" if eligible else "hermes_goal"
    driver_owner = owner if eligible else "hermes"
    driver_session = session if eligible else None
    capability_digest = record_digest(snapshot) if snapshot else None
    identity = sha256_text(json.dumps([kind, driver_owner, driver_session, objective, capability_digest]))
    return DriverRecord(
        schema_version=DRIVER_SCHEMA, driver_id=identity, kind=kind, owner=driver_owner,
        capability_snapshot_digest=capability_digest, session_ref=driver_session, objective_sha256=objective,
        observation_state="prepared", observed_at=None, advisory_snapshot=None,
        fallback_reason=fallback, compatibility="current",
    ), snapshot


def driver_identity(loop_id: str, driver: Mapping[str, JsonValue]) -> ObservationIdentity:
    return ObservationIdentity(loop_id, digest_ref(driver.get("driver_id")),
                               opaque_ref(driver.get("owner"), "owner"),
                               opaque_ref(driver.get("session_ref"), "session_ref"))


def validate_driver_cycle(cycle: Mapping[str, JsonValue]) -> list[str]:
    """Validate persisted v2 companions without changing legacy native history."""
    try:
        driver = cycle.get("driver")
        if not isinstance(driver, dict) or set(driver) != set(DriverRecord.__annotations__):
            raise LoopDriverError("driver_invalid_shape")
        if driver["schema_version"] != DRIVER_SCHEMA or driver["kind"] not in ("hermes_goal", "external_executor_goal"):
            raise LoopDriverError("driver_invalid_kind")
        goal = cycle.get("goal")
        if not isinstance(goal, dict) or driver["objective_sha256"] != goal.get("reframe_hash"):
            raise LoopDriverError("driver_cycle_objective_mismatch")
        digest_ref(driver["driver_id"])
        digest_ref(driver["objective_sha256"])
        snapshot = cycle.get("executor_capability_snapshot")
        if snapshot is not None and (not isinstance(snapshot, dict) or validate_executor_capability_snapshot(snapshot)):
            raise LoopDriverError("driver_invalid_capability_snapshot")
        if driver["capability_snapshot_digest"] != (record_digest(snapshot) if isinstance(snapshot, dict) else None):
            raise LoopDriverError("driver_capability_digest_mismatch")
        if driver["compatibility"] not in ("current", "migrated_native_driver"):
            raise LoopDriverError("driver_invalid_compatibility")
        if driver["fallback_reason"] not in ("", "hermes_owned_or_non_coding", "goal_controls_not_observed_for_selected_session"):
            raise LoopDriverError("driver_invalid_fallback_reason")
        if driver["kind"] == "external_executor_goal":
            expected, _ = select_driver(str(driver["objective_sha256"]), {
                "executor": driver["owner"], "work_kind": "coding", "session_ref": driver["session_ref"],
                "capability_snapshot": snapshot,
            })
            if expected["kind"] != driver["kind"] or expected["driver_id"] != driver["driver_id"]:
                raise LoopDriverError("driver_capability_binding_mismatch")
            history = checked_history(cycle.get("executor_goal_observations", []), driver_identity(str(cycle["loop_id"]), driver))
            latest = history[-1] if history else None
            if driver["advisory_snapshot"] != latest or driver["observed_at"] != (latest["observed_at"] if latest else None):
                raise LoopDriverError("driver_snapshot_history_mismatch")
            if driver["observation_state"] != ("observed" if latest else "prepared"):
                raise LoopDriverError("driver_observation_state_mismatch")
        elif (driver["owner"] != "hermes" or driver["session_ref"] is not None or
              driver["advisory_snapshot"] is not None or driver["observed_at"] is not None or
              driver["observation_state"] != "prepared" or cycle.get("executor_goal_observations") != []):
            raise LoopDriverError("driver_invalid_native_owner")
    except LoopDriverError as exc:
        return [exc.code]
    return []


def validate_driver_history(cycle: Mapping[str, JsonValue]) -> list[str]:
    """Bound old controllers and migration provenance without retaining input payloads."""
    history = cycle.get("driver_history", [])
    if not isinstance(history, list) or len(history) > 8:
        return ["driver_transfer_history_bound"]
    try:
        for entry in history:
            if not isinstance(entry, dict) or set(entry) != {
                "driver", "executor_capability_snapshot", "executor_goal_observations", "reconciliation"
            }:
                raise LoopDriverError("driver_invalid_transfer_history")
            errors = validate_driver_cycle({**entry, "goal": cycle["goal"], "loop_id": cycle["loop_id"]})
            if errors:
                return errors
            previous = entry["reconciliation"]
            old_driver = entry["driver"]
            if not isinstance(previous, dict) or set(previous) != {"driver_id", "status", "observed_at", "evidence_refs"}:
                raise LoopDriverError("driver_invalid_reconciliation")
            if not isinstance(old_driver, dict) or previous["driver_id"] != old_driver["driver_id"] or previous["status"] not in ("stopped", "absent"):
                raise LoopDriverError("driver_invalid_reconciliation")
            if observed_time(previous["observed_at"]) > datetime.now(timezone.utc):
                raise LoopDriverError("driver_future_observation")
            evidence_refs(previous["evidence_refs"])
        migration = cycle.get("driver_migration")
        if migration is not None:
            if not isinstance(migration, dict) or set(migration) != {"source_schema", "source_revision"}:
                raise LoopDriverError("driver_invalid_migration")
            revision = migration["source_revision"]
            if migration["source_schema"] != "loop_cycle/v1" or isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
                raise LoopDriverError("driver_invalid_migration")
    except LoopDriverError as exc:
        return [exc.code]
    return []


def driver_status(cycle: Mapping[str, JsonValue], checkpoint_ready: bool = False) -> dict[str, JsonValue]:
    """One status projection for CLI and wrapper; operational actions are advisory."""
    raw = cycle.get("driver")
    if not isinstance(raw, dict):
        goal = cycle.get("goal")
        objective = str(goal.get("reframe_hash")) if isinstance(goal, dict) else "0" * 64
        native, _ = select_driver(objective, None)
        driver = driver_record(native)
        driver["compatibility"] = "legacy_native_driver"
    else:
        driver = dict(raw)
    warnings: list[str] = []
    action = "observe_native_goal"
    latest = driver["advisory_snapshot"]
    if driver["kind"] == "external_executor_goal":
        if isinstance(latest, dict):
            status = DriverStatus(str(latest["status"]))
            match status:
                case DriverStatus.ACTIVE:
                    action = "observe_progress"
                case DriverStatus.PAUSED:
                    warnings.append("driver_paused")
                    action = "resume_after_confirmation"
                case DriverStatus.BUDGET_LIMITED:
                    warnings.append("driver_budget_limited")
                    action = "review_budget"
                case DriverStatus.CLOSED:
                    action = "observe_progress" if checkpoint_ready else "resume_for_missing_evidence"
                    if not checkpoint_ready:
                        warnings.append("driver_closed_early")
                case unreachable:
                    assert_never(unreachable)
            if latest["objective_sha256"] != driver["objective_sha256"]:
                warnings.insert(0, "driver_objective_mismatch")
                action = "reconcile_objective"
        else:
            warnings.append("driver_missing")
            action = "observe_or_resume_selected_executor"
    else:
        # Native v1 receipts retain their original meaning and phase authority.
        from .loop_observation_history import native_goal_status
        native_status = native_goal_status(cycle)
        if native_status["activation_status"] == "observed":
            driver["session_ref"] = str(native_status["session_ref"])
            driver["observation_state"] = "observed"
            history = cycle.get("goal_driver_observations")
            if isinstance(history, list) and history and isinstance(history[-1], dict):
                driver["observed_at"] = history[-1]["observed_at"]
    stamp = driver["observed_at"]
    driver.update({"warnings": list(warnings), "next_action": action,
                   "age_seconds": max(0, int((datetime.now(timezone.utc) - observed_time(stamp)).total_seconds())) if stamp else None})
    return driver
