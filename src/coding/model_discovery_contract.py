"""Public contract and deterministic payload helpers for model discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Final

MODEL_DISCOVERY_SCHEMA_VERSION: Final[str] = "model_discovery/v1"
MODEL_DISCOVERY_OBSERVATION_STATUSES: Final[tuple[str, ...]] = (
    "recommended",
    "observed_before",
    "confirmed_active",
    "inactive",
    "unobserved",
    "truncated",
)
MODEL_DISCOVERY_SOURCE_STATUSES: Final[tuple[str, ...]] = (
    *MODEL_DISCOVERY_OBSERVATION_STATUSES,
    "layout_unverified",
)
MODEL_DISCOVERY_CLAIM_BOUNDARY: Final[str] = (
    "Discovery reports bounded, allowlisted metadata from fixed local roots. "
    "It never reads auth files or emits transcript content, prompts, tool results, "
    "credentials, entitlement, routing, dispatch, execution, review, CI, or merge evidence."
)

DEFAULT_DISCOVERY_LIMITS: Final[dict[str, int | float]] = {
    "max_records_per_source": 2_000,
    "max_record_bytes": 64 * 1_024,
    "max_depth": 8,
    "soft_budget_seconds": 5.0,
}


@dataclass(frozen=True)
class DiscoveryLimits:
    max_records_per_source: int = 2_000
    max_record_bytes: int = 64 * 1_024
    max_depth: int = 8
    soft_budget_seconds: float = 5.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None) -> DiscoveryLimits:
        if values is None:
            return cls()
        return cls(
            max_records_per_source=max(1, int(values.get("max_records_per_source", 2_000))),
            max_record_bytes=max(1, int(values.get("max_record_bytes", 64 * 1_024))),
            max_depth=max(0, int(values.get("max_depth", 8))),
            soft_budget_seconds=max(0.0, float(values.get("soft_budget_seconds", 5.0))),
        )


def source_payload(
    status: str,
    observations: tuple[object, ...] | list[dict[str, str]],
    *,
    scanned_records: int,
    rejected: int,
    supported_root: bool,
    truncated_reasons: tuple[str, ...],
) -> dict[str, object]:
    if status not in MODEL_DISCOVERY_SOURCE_STATUSES:
        raise ValueError("unknown discovery source status")
    canonical = json.dumps(
        {"status": status, "observations": observations},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "status": status,
        "supported_root": supported_root,
        "scanned_records": scanned_records,
        "observation_count": len(observations),
        "rejected": rejected,
        "truncated_reasons": list(truncated_reasons),
        "fingerprint": sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }


def deduplicate(observations: list[dict[str, str]]) -> list[dict[str, str]]:
    unique = {
        (
            entry["source"],
            entry["provider"],
            entry["model_id"],
            entry["variant"],
            entry["timestamp"],
            entry["status"],
        ): entry
        for entry in observations
    }
    return [unique[key] for key in sorted(unique)]
