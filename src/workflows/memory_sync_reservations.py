"""Explicit local pre-write reservations for cooperating, independently approved hosts.

OMH never calls a provider/callback. Only a fresh ``claimed`` return satisfies
this local gate, once; it is neither write approval nor evidence of submission.
The host owns approval, stable scope/input identities and at most one write per
fresh claim. A host must not write after an exception or a missing return.
Replays of recorded identities hold, even after a host reports no write.

Keep the ledger and lock sidecar for the entire identity lifetime. No expiry,
release, eviction or automatic recovery is provided. Deletion/rollback, changing
the ledger path, dishonest identities and hosts bypassing the gate are outside
the contract. Atomic replacement protects process interruptions, not power-loss
durability (the shared primitive does not fsync). All state is metadata only.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
from typing import Callable

from ..system.local_store import atomic_write_json, file_lock
from ..system.metadata_safety import require_opaque_metadata_ref
from .memory_sync_fidelity_validation import closed_value, is_metadata_list, metadata_mapping, provider_id

SCHEMA_VERSION = "memory_sync_reservations/v1"
MAX_RESERVATIONS = 1024
MAX_INPUTS = 24
MAX_LEDGER_BYTES = 4 * 1024 * 1024
_SCOPE_KEYS = ("provider_id", "provider_mode", "profile_ref", "session_ref", "policy_digest")
_ENTRY_KEYS = {"scope", "attempt_id", "input_digests", "decision", "reason", "host_outcome"}
_SKIP_REASONS = ("skipped_queue", "skipped_timeout")
_OUTCOMES = ("unknown", "written", "not_written")
_DECODE: Callable[[str], object] = json.loads


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("sync identity must be a sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class SyncScope:
    """Exact provider/profile/session/policy partition, with no inherited scope."""

    provider_id: str
    provider_mode: str
    profile_ref: str
    session_ref: str
    policy_digest: str

    def __post_init__(self) -> None:
        _ = provider_id(self.provider_id)
        for name, value in (
            ("provider_mode", self.provider_mode), ("profile_ref", self.profile_ref),
            ("session_ref", self.session_ref),
        ):
            _ = require_opaque_metadata_ref(value, field=name)
        _ = _digest(self.policy_digest)

    def to_dict(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id, "provider_mode": self.provider_mode,
            "profile_ref": self.profile_ref, "session_ref": self.session_ref,
            "policy_digest": self.policy_digest,
        }


@dataclass(frozen=True, slots=True)
class SyncReservation:
    """Local decision only; a host report is unauthenticated metadata, not a receipt."""

    decision: str
    reason: str
    host_outcome: str = "unknown"

    @property
    def authorizes_provider_write(self) -> bool:
        return False

    @property
    def submission_observed(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class _Entry:
    scope: SyncScope
    attempt_id: str
    input_digests: tuple[str, ...]
    decision: str
    reason: str
    host_outcome: str = "unknown"

    def to_dict(self) -> dict[str, object]:
        return {"scope": self.scope.to_dict(), "attempt_id": self.attempt_id,
                "input_digests": list(self.input_digests), "decision": self.decision,
                "reason": self.reason, "host_outcome": self.host_outcome}


def _inputs(value: Sequence[object]) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not 1 <= len(value) <= MAX_INPUTS:
        raise ValueError("sync input identities must contain 1 to 24 digests")
    return tuple(sorted({_digest(item) for item in value}))


def _load(path: Path) -> list[_Entry]:
    try:
        with path.open(encoding="utf-8") as handle:
            text = handle.read(MAX_LEDGER_BYTES + 1)
    except FileNotFoundError:
        return []
    if len(text.encode("utf-8")) > MAX_LEDGER_BYTES:
        raise ValueError("sync reservation ledger exceeds capacity")
    try:
        raw = metadata_mapping(_DECODE(text), "sync ledger")
    except (ValueError, RecursionError):
        raise ValueError("invalid sync reservation ledger") from None
    if set(raw) != {"schema_version", "entries"} or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("invalid sync reservation ledger schema")
    rows = raw["entries"]
    if not is_metadata_list(rows) or len(rows) > MAX_RESERVATIONS:
        raise ValueError("sync reservation ledger exceeds capacity")
    entries: list[_Entry] = []
    for value in rows:
        row = metadata_mapping(value, "sync reservation")
        if set(row) != _ENTRY_KEYS:
            raise ValueError("invalid sync reservation fields")
        binding = metadata_mapping(row["scope"], "sync scope")
        if set(binding) != set(_SCOPE_KEYS):
            raise ValueError("invalid sync reservation scope")
        refs = {name: require_opaque_metadata_ref(binding[name], field=name) for name in _SCOPE_KEYS}
        raw_inputs = row["input_digests"]
        if not is_metadata_list(raw_inputs):
            raise ValueError("sync input identities must contain 1 to 24 digests")
        entry = _Entry(
            SyncScope(**refs), require_opaque_metadata_ref(row["attempt_id"], field="attempt_id"),
            _inputs(raw_inputs),
            closed_value(row["decision"], ("claimed", "held", "skipped"), "decision"),
            closed_value(row["reason"], ("reserved", "duplicate_input", *_SKIP_REASONS), "reason"),
            closed_value(row["host_outcome"], _OUTCOMES, "host_outcome"),
        )
        if (entry.decision, entry.reason) not in (
            ("claimed", "reserved"), ("held", "duplicate_input"),
            *(("skipped", reason) for reason in _SKIP_REASONS),
        ) or (entry.decision != "claimed" and entry.host_outcome != "unknown"):
            raise ValueError("inconsistent sync reservation state")
        scoped = [previous for previous in entries if previous.scope == entry.scope]
        overlap = any(set(entry.input_digests).intersection(previous.input_digests) for previous in scoped)
        if any(previous.attempt_id == entry.attempt_id for previous in scoped) or overlap != (entry.decision == "held"):
            raise ValueError("inconsistent sync reservation identities")
        entries.append(entry)
    return entries


def _save(path: Path, entries: list[_Entry]) -> None:
    atomic_write_json(path, {"schema_version": SCHEMA_VERSION,
                            "entries": [entry.to_dict() for entry in entries]}, private=True)


def claim_sync(
    path: Path, scope: SyncScope, attempt_id: str, input_digests: Sequence[object],
    *, skip_reason: str | None = None,
) -> SyncReservation:
    """Reserve all coalesced input identities atomically, or hold the whole batch.

    New overlap/skip rows consume their declared identities too. Explicit queue
    and timeout skips stay durable. Capacity, invalid storage, lock failure and
    I/O failure raise before any successful return; never interpret them as a
    bypass. This API does not schedule, wait for a provider, approve or execute.
    """
    attempt = require_opaque_metadata_ref(attempt_id, field="attempt_id")
    inputs = _inputs(input_digests)
    if skip_reason is not None:
        _ = closed_value(skip_reason, _SKIP_REASONS, "skip_reason")
    with file_lock(path, private=True) as lock:
        if lock["enforced"] is not True:
            raise RuntimeError("sync reservations require an enforced file lock")
        entries = _load(path)
        scoped = [entry for entry in entries if entry.scope == scope]
        for entry in scoped:
            if entry.attempt_id == attempt:
                same = entry.input_digests == inputs and (
                    skip_reason == entry.reason if entry.decision == "skipped" else skip_reason is None
                )
                return SyncReservation("held", "duplicate_attempt" if same else "conflicting_attempt", entry.host_outcome)
        if len(entries) >= MAX_RESERVATIONS:
            raise ValueError("sync reservation ledger capacity reached; identities cannot be evicted")
        if any(set(inputs).intersection(entry.input_digests) for entry in scoped):
            decision, reason = "held", "duplicate_input"
        elif skip_reason is not None:
            decision, reason = "skipped", skip_reason
        else:
            decision, reason = "claimed", "reserved"
        entries.append(_Entry(scope, attempt, inputs, decision, reason))
        _save(path, entries)
        return SyncReservation(decision, reason)


def report_sync_outcome(
    path: Path, scope: SyncScope, attempt_id: str, input_digests: Sequence[object], outcome: str,
) -> SyncReservation:
    """Record only the host's assertion; never release a reservation or mint a receipt.

    An unknown report can be resolved by this same host's assertion. Contradictory
    terminal reports fail closed. Even reported not-written identities stay held:
    this API has no authority to reconcile uncertain provider state or retry it.
    """
    attempt = require_opaque_metadata_ref(attempt_id, field="attempt_id")
    inputs = _inputs(input_digests)
    _ = closed_value(outcome, _OUTCOMES, "host_outcome")
    with file_lock(path, private=True) as lock:
        if lock["enforced"] is not True:
            raise RuntimeError("sync reservations require an enforced file lock")
        entries = _load(path)
        for index, entry in enumerate(entries):
            if entry.scope != scope or entry.attempt_id != attempt:
                continue
            if entry.input_digests != inputs or entry.decision != "claimed":
                raise ValueError("host report does not match a claimed reservation")
            if entry.host_outcome not in ("unknown", outcome):
                raise ValueError("conflicting host outcome; reservation remains held")
            if entry.host_outcome != outcome:
                entries[index] = replace(entry, host_outcome=outcome)
                _save(path, entries)
            return SyncReservation("held", "host_report_recorded", outcome)
        raise ValueError("host report has no matching scoped reservation")
