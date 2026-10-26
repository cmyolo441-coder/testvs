"""Shared posture/fidelity assessment vocabulary and metadata parsers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TypeGuard

from ..system.metadata_safety import require_opaque_metadata_ref
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


# Unknown means unstated; not_observed means a claim lacks an observation.
FIELD_STATUSES: tuple[str, ...] = ("ready", "missing", "risky", "not_observed", "unknown")
EVIDENCE_CLASSES: tuple[str, ...] = (
    "declared_documentation", "declared_package_metadata", "operator_statement",
    "observed_local_runtime", "observed_trial_receipt", "none",
)
_OBSERVED_EVIDENCE_CLASSES = frozenset({"observed_local_runtime", "observed_trial_receipt"})
_PROVIDER_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ASSESSMENT_KEYS = {"status", "evidence_class"}


@dataclass(frozen=True)
class Assessment:
    """One field's status plus the class of evidence that status rests on."""

    status: str
    evidence_class: str

    def to_dict(self) -> dict[str, str]:
        return {"status": self.status, "evidence_class": self.evidence_class}


def provider_id(value: object) -> str:
    if not isinstance(value, str) or not _PROVIDER_ID.fullmatch(value):
        raise ValueError("provider_id must match [a-z0-9][a-z0-9._-]{0,63}")
    return require_opaque_metadata_ref(value, field="provider_id")


def closed_value(value: object, allowed: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field} must be one of {', '.join(allowed)}")
    return value


def _is_mapping(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)


def is_metadata_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def metadata_mapping(value: object, field: str) -> dict[str, object]:
    """Narrow an untrusted JSON mapping before checking its closed field schema."""
    if not _is_mapping(value):
        raise ValueError(f"{field} must be a metadata mapping")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field} must contain string field names")
        result[key] = item
    return result


def assessments(raw: object, dimensions: tuple[str, ...], field: str) -> dict[str, Assessment]:
    supplied = metadata_mapping(raw, field)
    unsupported = sorted(set(supplied) - set(dimensions))
    if unsupported:
        raise ValueError(f"{field} contains unsupported dimensions: {', '.join(unsupported)}")
    assessments: dict[str, Assessment] = {}
    for dimension, value in supplied.items():
        item = metadata_mapping(value, field)
        if set(item) != _ASSESSMENT_KEYS:
            raise ValueError(f"{field} entry must contain only status and evidence_class")
        status = closed_value(item.get("status"), FIELD_STATUSES, f"{field} status")
        evidence_class = closed_value(item.get("evidence_class"), EVIDENCE_CLASSES, f"{field} evidence_class")
        if status == "ready" and evidence_class not in _OBSERVED_EVIDENCE_CLASSES:
            raise ValueError(f"{field} status ready requires observed evidence, not {evidence_class}")
        assessments[dimension] = Assessment(status, evidence_class)
    return assessments


def observed_at(value: object) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("observed trial observed_at must be an ISO-8601 UTC timestamp")
    try:
        _ = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("observed trial observed_at must be an ISO-8601 UTC timestamp") from exc
    return value
