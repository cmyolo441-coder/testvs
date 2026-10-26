from __future__ import annotations

from datetime import datetime
from typing import Any

from .external_effect_receipts import validate_external_effect_receipt
from .production_readiness_values import (
    EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION,
    OBSERVED_CHECK_RESULT_KEYS,
    OBSERVED_CHECK_RESULT_SCHEMA_VERSION,
    OBSERVED_POSTCONDITION_SCHEMA_VERSION,
    _valid_ref,
    readiness_row_id,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

def _matrix_identity_and_time_errors(
    *, scope: str, task_id: str, revision: str, created_at: str, evidence_fresh_after: str
) -> list[str]:
    errors: list[str] = []
    if not scope.strip():
        errors.append("scope is required")
    for field, value in (("task_id", task_id), ("revision", revision)):
        if not _valid_ref(value):
            errors.append(f"{field} must be an opaque reference")
    created = _timestamp(created_at)
    fresh = _timestamp(evidence_fresh_after)
    if created is None:
        errors.append("created_at must be a timezone-aware ISO-8601 timestamp")
    if fresh is None:
        errors.append("evidence_fresh_after must be a timezone-aware ISO-8601 timestamp")
    if created is not None and fresh is not None and fresh > created:
        errors.append("evidence_fresh_after must not be after created_at")
    return errors


def _external_identity_errors(identity: dict[str, Any], *, required: bool) -> list[str]:
    if not identity:
        return ["external observed readiness requires external_identity"] if required else []
    errors: list[str] = []
    if set(identity) != {"effect_id", "run_id"}:
        errors.append("external_identity must contain exactly effect_id and run_id")
    for field in ("effect_id", "run_id"):
        if not _valid_ref(identity.get(field)):
            errors.append(f"external_identity.{field} must be an opaque reference")
    return errors


def _evidence_errors(
    evidence: dict[str, Any],
    *,
    external_identity: dict[str, Any] | None = None,
    scope: str = "",
    category: str = "",
    task_id: str = "",
    revision: str = "",
    observed_before: str = "",
) -> list[str]:
    schema = evidence.get("schema_version")
    if schema == OBSERVED_CHECK_RESULT_SCHEMA_VERSION:
        errors = _observed_check_errors(
            evidence,
            scope=scope,
            category=category,
            task_id=task_id,
            revision=revision,
        )
        errors.extend(_future_timestamp_errors(evidence.get("observed_at"), observed_before, "observed_check_result observed_at"))
        return errors
    if schema == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION:
        return _external_evidence_errors(
            evidence,
            external_identity=external_identity or {},
            task_id=task_id,
            revision=revision,
            observed_before=observed_before,
        )
    return [f"unsupported readiness evidence schema: {schema}"]


def _observed_check_errors(
    evidence: dict[str, Any], *, scope: str, category: str, task_id: str, revision: str
) -> list[str]:
    errors: list[str] = []
    if set(evidence) != OBSERVED_CHECK_RESULT_KEYS:
        errors.append("observed_check_result keys must match observed_check_result/v1")
    result = evidence.get("result")
    if not isinstance(result, str) or result not in ("passed", "failed"):
        errors.append("observed_check_result result must be passed or failed")
    for field in ("check_id", "evidence_ref"):
        if not _valid_ref(evidence.get(field)):
            errors.append(f"observed_check_result {field} must be an opaque reference")
    if _timestamp(evidence.get("observed_at")) is None:
        errors.append("observed_check_result observed_at must be a timezone-aware ISO-8601 timestamp")
    authority = (
        ("scope", scope),
        ("category", category),
        ("task_id", task_id),
        ("revision", revision),
        ("row_id", readiness_row_id(scope, category, task_id, revision)),
    )
    for field, expected in authority:
        if evidence.get(field) != expected:
            errors.append(f"observed_check_result {field} must match row authority")
    return errors


def _external_evidence_errors(
    evidence: dict[str, Any],
    *,
    external_identity: dict[str, Any],
    task_id: str,
    revision: str,
    observed_before: str,
) -> list[str]:
    errors: list[str] = []
    receipt = evidence.get("receipt")
    postcondition = evidence.get("postcondition")
    if not isinstance(receipt, dict):
        return ["external observed success requires external_effect_receipt/v1"]
    errors.extend(validate_external_effect_receipt(receipt))
    if receipt.get("observed_result") != "succeeded":
        errors.append("external_effect_receipt observed_result must be succeeded")
    if _timestamp(receipt.get("observed_at")) is None:
        errors.append("external_effect_receipt observed_at must be a timezone-aware ISO-8601 timestamp")
    errors.extend(_future_timestamp_errors(receipt.get("observed_at"), observed_before, "external_effect_receipt observed_at"))
    if receipt.get("effect_id") != external_identity.get("effect_id"):
        errors.append("receipt effect_id must match external_identity.effect_id")
    if receipt.get("run_id") != external_identity.get("run_id"):
        errors.append("receipt run_id must match external_identity.run_id")
    supplied_refs = receipt.get("evidence_refs")
    refs = supplied_refs if isinstance(supplied_refs, list) else []
    for label, value in (("task_id", task_id), ("revision", revision)):
        prefix = f"{label.removesuffix('_id')}:"
        marker = f"{prefix}{value}"
        bound_markers = [ref for ref in refs if isinstance(ref, str) and ref.startswith(prefix)]
        if bound_markers != [marker]:
            errors.append(f"receipt evidence_refs must bind exactly one matching {label} marker")
    if not isinstance(postcondition, dict):
        return [*errors, "external observed success requires observed_postcondition/v1"]
    if postcondition.get("schema_version") != OBSERVED_POSTCONDITION_SCHEMA_VERSION:
        errors.append("postcondition schema_version must be observed_postcondition/v1")
    postcondition_result = postcondition.get("result")
    if not isinstance(postcondition_result, str) or postcondition_result not in ("satisfied", "failed"):
        errors.append("postcondition result must be satisfied or failed")
    for field in ("receipt_id", "effect_id", "run_id", "task_id", "revision", "external_ref", "observation_id"):
        if not _valid_ref(postcondition.get(field)):
            errors.append(f"postcondition {field} must be an opaque reference")
    if _timestamp(postcondition.get("observed_at")) is None:
        errors.append("postcondition observed_at must be a timezone-aware ISO-8601 timestamp")
    errors.extend(_future_timestamp_errors(postcondition.get("observed_at"), observed_before, "postcondition observed_at"))
    bindings = (
        ("receipt_id", receipt.get("receipt_id")),
        ("effect_id", external_identity.get("effect_id")),
        ("run_id", external_identity.get("run_id")),
        ("task_id", task_id),
        ("revision", revision),
        ("external_ref", receipt.get("external_ref")),
    )
    for field, expected in bindings:
        if postcondition.get(field) != expected:
            errors.append(f"postcondition {field} must match its bound readiness identity")
    receipt_time = _timestamp(receipt.get("observed_at"))
    postcondition_time = _timestamp(postcondition.get("observed_at"))
    if receipt_time is not None and postcondition_time is not None and postcondition_time < receipt_time:
        errors.append("postcondition observed_at must not precede receipt observed_at")
    return errors


def _future_timestamp_errors(value: object, observed_before: str, label: str) -> list[str]:
    observed = _timestamp(value)
    ceiling = _timestamp(observed_before)
    if observed is not None and ceiling is not None and observed > ceiling:
        return [f"{label} must not be after matrix created_at"]
    return []


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed
