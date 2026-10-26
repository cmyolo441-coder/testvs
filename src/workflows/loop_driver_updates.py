"""Locked-cycle driver mutations; no executor dispatch or evidence promotion."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

from ..coding.executor_capability_snapshots import JsonValue
from ..system.local_store import utc_now
from .loop_driver_contract import driver_identity, driver_record, driver_status, select_driver
from .loop_executor_observations import (
    MAX_OBSERVATIONS, LoopDriverError, checked_history, evidence_refs,
    observation_record, observed_time, parse_executor_observation,
)

TRANSFER_KEYS: Final = frozenset({"driver_id", "status", "observed_at", "evidence_refs"})


def observe_external_driver(cycle: dict[str, JsonValue], submitted: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    """Mutate only advisory state inside the caller's existing record lock."""
    driver = cycle.get("driver")
    if not isinstance(driver, dict) or driver.get("kind") != "external_executor_goal":
        raise LoopDriverError("driver_external_observation_requires_external_owner")
    identity = driver_identity(str(cycle["loop_id"]), driver)
    entry = parse_executor_observation(submitted, identity)
    history = checked_history(cycle.get("executor_goal_observations", []), identity)
    for previous in history:
        if previous["observation_id"] == entry["observation_id"]:
            if previous == entry:
                return None
            raise LoopDriverError("driver_conflicting_observation")
    if history and (entry["sequence"] <= history[-1]["sequence"] or
                    observed_time(entry["observed_at"]) < observed_time(history[-1]["observed_at"])):
        raise LoopDriverError("driver_stale_observation")
    if len(history) == MAX_OBSERVATIONS:
        raise LoopDriverError("driver_observation_history_bound")
    cycle["executor_goal_observations"] = [observation_record(item) for item in [*history, entry]]
    driver.update({"observation_state": "observed", "observed_at": entry["observed_at"], "advisory_snapshot": observation_record(entry)})
    cycle["updated_at"] = utc_now()
    return cycle


def migrate_driver(cycle: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    """Explicit idempotent migration; retain revision provenance and native history."""
    if cycle["schema_version"] == "loop_cycle/v2":
        return None
    goal = cycle["goal"]
    if not isinstance(goal, dict):
        raise LoopDriverError("driver_invalid_goal")
    driver, _ = select_driver(str(goal["reframe_hash"]), None)
    driver["compatibility"] = "migrated_native_driver"
    cycle.update({"schema_version": "loop_cycle/v2", "driver": driver_record(driver),
                  "executor_capability_snapshot": None, "executor_goal_observations": [],
                  "driver_migration": {"source_schema": "loop_cycle/v1", "source_revision": cycle.get("record_revision", 0)},
                  "updated_at": utc_now()})
    return cycle


def bind_driver(cycle: dict[str, JsonValue], submitted: Mapping[str, JsonValue]) -> dict[str, JsonValue] | None:
    """Require observed reconciliation before transferring a possible controller."""
    if set(submitted) != {"selection", "previous_driver"}:
        raise LoopDriverError("driver_invalid_binding_shape")
    selection = submitted["selection"]
    goal = cycle["goal"]
    if not isinstance(selection, dict) or not isinstance(goal, dict):
        raise LoopDriverError("driver_invalid_selection_shape")
    replacement, snapshot = select_driver(str(goal["reframe_hash"]), selection)
    current = driver_status(cycle)
    if replacement["driver_id"] == current["driver_id"]:
        return None
    previous = submitted["previous_driver"]
    if not isinstance(previous, dict) or set(previous) != TRANSFER_KEYS:
        raise LoopDriverError("driver_transfer_requires_reconciliation")
    if previous["driver_id"] != current["driver_id"] or previous["status"] not in ("stopped", "absent"):
        raise LoopDriverError("driver_transfer_requires_reconciliation")
    stamp = observed_time(previous["observed_at"])
    if stamp > datetime.now(timezone.utc):
        raise LoopDriverError("driver_future_observation")
    if current["observed_at"] and stamp < observed_time(current["observed_at"]):
        raise LoopDriverError("driver_stale_reconciliation")
    evidence_refs(previous["evidence_refs"])
    archive = cycle.get("driver_history", [])
    if not isinstance(archive, list) or len(archive) >= 8:
        raise LoopDriverError("driver_transfer_history_bound")
    if cycle["schema_version"] == "loop_cycle/v1":
        migrate_driver(cycle)
    cycle["driver_history"] = [*archive, {
        "driver": cycle["driver"], "executor_capability_snapshot": cycle.get("executor_capability_snapshot"),
        "executor_goal_observations": cycle.get("executor_goal_observations", []),
        "reconciliation": dict(previous),
    }]
    cycle.update({"driver": driver_record(replacement), "executor_capability_snapshot": snapshot,
                  "executor_goal_observations": [], "updated_at": utc_now()})
    return cycle
