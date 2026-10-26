"""Parse bounded synchronization observations, not provider write commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
import re
from typing import Final, assert_never

from ..system.metadata_safety import require_opaque_metadata_ref
from .memory_sync_fidelity_validation import (
    EVIDENCE_CLASSES, closed_value as _closed_value, observed_at as _observed_at,
    provider_id as _provider_id, metadata_mapping, is_metadata_list,
)
from .memory_sync_fidelity_policy import UNITS, bounded_count


class SyncOutcome(str, Enum):
    COMPLETE = "complete"
    HEAD = "truncated_head"
    TAIL = "truncated_tail"
    BOUNDARY = "truncated_boundary"
    REJECTED = "rejected"
    QUEUE = "skipped_queue"
    TIMEOUT = "skipped_timeout"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class FidelityBinding:
    provider_mode: str
    profile_ref: str
    session_ref: str
    policy_digest: str
    input_digest: str


@dataclass(frozen=True, slots=True)
class SyncReceipt:
    receipt_id: str
    provider_id: str
    binding: FidelityBinding
    attempt_id: str
    outcome: SyncOutcome
    selected_count: int | None
    omitted_count: int | None
    unit: str
    omitted_position: str
    failure_category: str
    entered_provider_state: str
    omitted_input_entered_provider_state: str
    evidence_class: str
    observed_at: str
    coalesced_into: str | None

    @property
    def clock(self) -> datetime:
        return datetime.fromisoformat(self.observed_at.replace("Z", "+00:00"))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        del result["binding"]
        result.update(asdict(self.binding))
        result["schema_version"] = "memory_sync_receipt/v1"
        result["outcome"] = self.outcome.value
        if self.coalesced_into is None:
            del result["coalesced_into"]
        return result


BINDING_KEYS: Final = frozenset(FidelityBinding.__dataclass_fields__)
_RECEIPT_KEYS: Final = (frozenset(SyncReceipt.__dataclass_fields__) - {"binding", "coalesced_into"}) | BINDING_KEYS | {"schema_version"}
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}")


def parse_binding(raw: object) -> FidelityBinding:
    supplied = metadata_mapping(raw, "binding")
    if frozenset(supplied) != BINDING_KEYS:
        raise ValueError("fidelity binding must contain mode, profile, session, policy and input digests")
    refs = {key: require_opaque_metadata_ref(supplied[key], field=key) for key in BINDING_KEYS}
    for key in ("policy_digest", "input_digest"):
        if not _DIGEST.fullmatch(refs[key]):
            raise ValueError(f"{key} must be a sha256 digest")
    return FidelityBinding(**refs)


def parse_receipts(raw: object) -> tuple[SyncReceipt, ...]:
    if not is_metadata_list(raw) or len(raw) > 24:
        raise ValueError("fidelity attempts must contain at most 24 receipts")
    receipts: list[SyncReceipt] = []
    for value in raw:
        item = metadata_mapping(value, "receipt")
        if not _RECEIPT_KEYS.issubset(item) or set(item) - (_RECEIPT_KEYS | {"coalesced_into"}):
            raise ValueError("sync receipt must contain only the supported bound metadata fields")
        if item["schema_version"] != "memory_sync_receipt/v1":
            raise ValueError("unsupported sync receipt schema")
        coalesced = item.get("coalesced_into")
        observed_at = _observed_at(item["observed_at"])
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", observed_at):
            raise ValueError("sync observed_at must be a complete UTC timestamp")
        receipt = SyncReceipt(
            receipt_id=require_opaque_metadata_ref(item["receipt_id"], field="receipt_id"),
            provider_id=_provider_id(item["provider_id"]),
            binding=parse_binding({key: item[key] for key in BINDING_KEYS}),
            attempt_id=require_opaque_metadata_ref(item["attempt_id"], field="attempt_id"),
            outcome=SyncOutcome(_closed_value(item["outcome"], tuple(outcome.value for outcome in SyncOutcome), "outcome")),
            selected_count=bounded_count(item["selected_count"], "selected_count"),
            omitted_count=bounded_count(item["omitted_count"], "omitted_count"),
            unit=_closed_value(item["unit"], UNITS, "unit"),
            omitted_position=_closed_value(item["omitted_position"], ("none", "head", "tail", "middle", "unknown"), "omitted_position"),
            failure_category=_closed_value(item["failure_category"], ("none", "provider_error", "transport", "policy_rejection", "timeout", "unknown"), "failure_category"),
            entered_provider_state=_closed_value(item["entered_provider_state"], ("yes", "no", "unknown"), "entered_provider_state"),
            omitted_input_entered_provider_state=_closed_value(item["omitted_input_entered_provider_state"], ("yes", "no", "unknown"), "omitted_input_entered_provider_state"),
            evidence_class=_closed_value(item["evidence_class"], EVIDENCE_CLASSES, "evidence_class"),
            observed_at=observed_at,
            coalesced_into=None if coalesced is None else require_opaque_metadata_ref(coalesced, field="coalesced_into"),
        )
        _check_consistency(receipt)
        receipts.append(receipt)
    if len({row.receipt_id for row in receipts}) != len(receipts):
        raise ValueError("sync receipt ids must be unique; this does not prevent provider writes")
    return tuple(receipts)


def _check_consistency(row: SyncReceipt) -> None:
    """Reject contradictory supplied observations before deriving any readiness."""
    if row.coalesced_into is not None and (
        row.coalesced_into == row.receipt_id or row.outcome is not SyncOutcome.QUEUE
    ):
        raise ValueError("coalesced receipt must name another receipt and remain skipped_queue")
    match row.outcome:
        case SyncOutcome.COMPLETE:
            if row.omitted_count != 0 or row.omitted_position != "none" or row.failure_category != "none":
                raise ValueError("complete sync cannot contain omissions or failures")
            return
        case SyncOutcome.HEAD | SyncOutcome.TAIL | SyncOutcome.BOUNDARY:
            position = {SyncOutcome.HEAD: "head", SyncOutcome.TAIL: "tail", SyncOutcome.BOUNDARY: "middle"}[row.outcome]
            if row.omitted_count == 0 or row.omitted_position not in (position, "unknown"):
                raise ValueError("truncated sync requires consistent omission metadata")
            return
        case SyncOutcome.QUEUE | SyncOutcome.TIMEOUT | SyncOutcome.REJECTED:
            if row.entered_provider_state == "yes" or row.selected_count not in (0, None):
                raise ValueError("skipped or rejected sync cannot claim submitted input")
            return
        case SyncOutcome.FAILED | SyncOutcome.UNKNOWN:
            return
    assert_never(row.outcome)
