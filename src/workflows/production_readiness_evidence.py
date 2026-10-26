from __future__ import annotations

from typing import Any

from .production_readiness_authentication import _external_authenticity_error
from .production_readiness_structural import _evidence_errors, _timestamp
from .production_readiness_values import (
    EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION,
    OBSERVED_CHECK_RESULT_SCHEMA_VERSION,
    READINESS_CATEGORIES,
    ReadinessTrustContext,
)

def _derive_evidence_state(
    evidence: list[dict[str, Any]],
    *,
    requires_external: bool,
    external_identity: dict[str, Any],
    matrix_schema_version: str,
    matrix_id: str,
    contract_ref: dict[str, Any],
    scope: str,
    category: str,
    task_id: str,
    revision: str,
    fresh_after: str,
    observed_before: str,
    trusted_context: ReadinessTrustContext | None,
) -> str:
    states: list[str] = []
    for item in evidence:
        if _evidence_errors(
            item,
            external_identity=external_identity,
            scope=scope,
            category=category,
            task_id=task_id,
            revision=revision,
            observed_before=observed_before,
        ):
            continue
        schema = item.get("schema_version")
        if requires_external and schema != EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION:
            continue
        if not requires_external and schema == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION:
            continue
        if schema == OBSERVED_CHECK_RESULT_SCHEMA_VERSION:
            observed_at = item.get("observed_at")
        else:
            observed_at = (item.get("postcondition") or {}).get("observed_at")
        if not _fresh(observed_at, fresh_after, observed_before):
            continue
        if schema == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION:
            if _external_authenticity_error(
                item,
                matrix_schema_version=matrix_schema_version,
                matrix_id=matrix_id,
                contract_ref=contract_ref,
                scope=scope,
                category=category,
                task_id=task_id,
                revision=revision,
                created_at=observed_before,
                evidence_fresh_after=fresh_after,
                requires_external_observation=requires_external,
                external_identity=external_identity,
                trusted_context=trusted_context,
            ):
                continue
            receipt = item.get("receipt") or {}
            if not _fresh(receipt.get("observed_at"), fresh_after, observed_before):
                continue
        if schema == OBSERVED_CHECK_RESULT_SCHEMA_VERSION:
            states.append("failed" if item.get("result") == "failed" else "observed")
        else:
            postcondition = item.get("postcondition") or {}
            states.append("failed" if postcondition.get("result") == "failed" else "observed")
    if "failed" in states:
        return "failed"
    if "observed" in states:
        return "observed"
    return "missing"


def _derive_verdict(rows: list[dict[str, Any]]) -> str:
    states = [row.get("evidence_state") for row in rows]
    if "failed" in states:
        return "BLOCK"
    if len(rows) == len(READINESS_CATEGORIES) and states and all(state == "observed" for state in states):
        return "GO"
    return "HOLD"


def _fresh(observed_at: object, fresh_after: str, observed_before: str) -> bool:
    observed = _timestamp(observed_at)
    threshold = _timestamp(fresh_after)
    ceiling = _timestamp(observed_before)
    return observed is not None and threshold is not None and ceiling is not None and threshold <= observed <= ceiling


def _external_receipt_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [
        str(receipt.get("receipt_id", ""))
        for item in evidence
        if item.get("schema_version") == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION
        for receipt in [item.get("receipt")]
        if isinstance(receipt, dict) and str(receipt.get("receipt_id", ""))
    ]


def _external_observation_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [
        str(postcondition.get("observation_id", ""))
        for item in evidence
        if item.get("schema_version") == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION
        for postcondition in [item.get("postcondition")]
        if isinstance(postcondition, dict) and str(postcondition.get("observation_id", ""))
    ]


def _local_check_ids(evidence: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("check_id", ""))
        for item in evidence
        if item.get("schema_version") == OBSERVED_CHECK_RESULT_SCHEMA_VERSION
        and str(item.get("check_id", ""))
    ]


def _local_evidence_refs(evidence: list[dict[str, Any]]) -> list[str]:
    return [
        str(item.get("evidence_ref", ""))
        for item in evidence
        if item.get("schema_version") == OBSERVED_CHECK_RESULT_SCHEMA_VERSION
        and str(item.get("evidence_ref", ""))
    ]
