"""Provider-neutral fidelity projection from explicitly supplied scoped metadata.

This is not a receipt collector or a synchronization runtime. A provider-class
row is an operator-supplied observation, not an authenticated OMH provider call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import assert_never

from .memory_sync_fidelity_policy import DIMENSIONS, FidelityPolicy, parse_policy
from .memory_sync_fidelity_validation import metadata_mapping
from .memory_sync_fidelity_receipts import (
    FidelityBinding, SyncOutcome, SyncReceipt, parse_binding, parse_receipts,
)


@dataclass(frozen=True, slots=True)
class MemorySyncFidelity:
    binding: FidelityBinding
    policy: FidelityPolicy
    attempts: tuple[SyncReceipt, ...]


def parse_memory_sync_fidelity(raw: object) -> MemorySyncFidelity:
    supplied = metadata_mapping(raw, "input_fidelity")
    required = {"schema_version", "binding"}
    allowed = required | set(DIMENSIONS) | {"attempts"}
    if not required.issubset(supplied) or set(supplied) - allowed:
        raise ValueError("input_fidelity must contain only supported metadata fields")
    if supplied["schema_version"] != "memory_sync_fidelity/v1":
        raise ValueError("unsupported input_fidelity schema")
    return MemorySyncFidelity(parse_binding(supplied["binding"]), parse_policy(supplied), parse_receipts(supplied.get("attempts", [])))


def build_memory_sync_fidelity(value: MemorySyncFidelity, provider_id: str) -> dict[str, object]:
    eligible: list[SyncReceipt] = []
    rejected: list[dict[str, str]] = []
    for row in value.attempts:
        if row.provider_id != provider_id or row.binding != value.binding:
            # Do not propagate correlation identities from a foreign scope.
            rejected.append({"reason": "receipt_binding_mismatch"})
        else:
            eligible.append(row)
    # Input order is not a clock. Equal-time observations cannot choose a winner.
    latest = max((row.clock for row in eligible), default=None)
    current = [row for row in eligible if row.clock == latest]
    receipt = current[0] if len(current) == 1 else None
    readiness = _readiness(receipt)
    omitted_state = "unknown"
    if receipt is not None and receipt.evidence_class == "observed_trial_receipt":
        omitted_state = receipt.omitted_input_entered_provider_state
    return {
        "schema_version": "memory_sync_fidelity/v1",
        "binding": asdict(value.binding), **value.policy.to_dict(),
        "attempts": [row.to_dict() for row in eligible], "rejected_attempts": rejected,
        "synchronization_readiness": readiness,
        "readiness_receipt_id": receipt.receipt_id if receipt is not None else None,
        "evidence_intake": "operator_supplied_not_authenticated",
        "unknown_field_count": value.policy.unknown_field_count + int(readiness == "unknown"),
        "deletion_and_portability_effect": {
            "omitted_input_entered_provider_state": omitted_state,
            "export": "unknown", "provider_side_deletion": "unknown",
        },
    }


def _readiness(row: SyncReceipt | None) -> str:
    if row is None or row.evidence_class != "observed_trial_receipt":
        return "unknown"
    match row.outcome:
        case SyncOutcome.COMPLETE:
            known = row.selected_count is not None and row.unit != "unknown" and row.entered_provider_state == "yes"
            return "complete_observed" if known else "unknown"
        case SyncOutcome.HEAD | SyncOutcome.TAIL | SyncOutcome.BOUNDARY:
            return "partial_observed" if row.entered_provider_state == "yes" else "unknown"
        case SyncOutcome.QUEUE | SyncOutcome.TIMEOUT:
            return "skipped_observed"
        case SyncOutcome.REJECTED:
            return "rejected_observed"
        case SyncOutcome.FAILED:
            return "failed_observed"
        case SyncOutcome.UNKNOWN:
            return "unknown"
    assert_never(row.outcome)
