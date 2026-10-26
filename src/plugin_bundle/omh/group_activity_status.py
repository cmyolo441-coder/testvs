"""Bounded profile-local activity status, independent of producer imports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TypeAlias

Status: TypeAlias = dict[str, str | int]
COUNTERS = ("dropped", "gapped", "rejected", "write_failed")
OUTCOMES = frozenset({"not_observed", "recorded", "already_recorded", "rejected", "quarantined", "write_failed",
    "queue_dropped", "observer_closed", "drain_timeout", "invalid_checkpoint", "aggregation_failed", "room_capacity",
    "cross_profile", "identity_conflict", "stale_event", "stale_after_final", "unknown_kind", "invalid_shape",
    "invalid_time", "invalid_sequence", "invalid_identity"})


def profile_slot(hermes_home: Path) -> str:
    return hashlib.sha256(str(hermes_home.expanduser().resolve()).encode()).hexdigest()


def status_path(omh_home: Path, hermes_home: Path) -> Path:
    return omh_home / "runtime" / f"group-activity-status-{profile_slot(hermes_home)}.json"


def group_activity_status(enabled: bool = False) -> Status:
    return {"readiness": "unavailable" if enabled else "disabled",
            "compatibility": "member_activity_contract_unsupported" if enabled else "not_observed",
            "dropped": 0, "gapped": 0, "rejected": 0, "write_failed": 0, "last_outcome": "not_observed",
            "next_action": "Enable collection only on a host declaring on_room_member_activity; native coverage remains partial."}


def read_group_activity_status(omh_home: Path, hermes_home: Path) -> Status:
    """Read one bounded historical status snapshot, never discover or start a host."""
    try:
        with status_path(omh_home, hermes_home).open("rb") as handle:
            raw = handle.read(4097)
        if len(raw) > 4096:
            return _unreadable()
        data = json.loads(raw)
    except FileNotFoundError:
        return group_activity_status()
    except (OSError, ValueError, UnicodeError):
        return _unreadable()
    keys = {"schema", "readiness", "compatibility", "last_outcome", *COUNTERS}
    if (not isinstance(data, dict) or set(data) != keys
            or data["schema"] != "omh_group_activity_observer_status/v1"
            or data["compatibility"] != "on_room_member_activity/v1"
            or not isinstance(data["readiness"], str) or data["readiness"] not in {"ready", "stopped"}
            or not isinstance(data["last_outcome"], str) or data["last_outcome"] not in OUTCOMES):
        return _unreadable()
    for name in COUNTERS:
        value = data[name]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 10**9:
            return _unreadable()
    return {name: data[name] for name in keys if name != "schema"}


def _unreadable() -> Status:
    return {**group_activity_status(True), "compatibility": "status_unreadable",
            "next_action": "Inspect the profile-local observer status file; no receipt recording is established by this status."}
