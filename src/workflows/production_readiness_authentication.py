from __future__ import annotations

from typing import Any, cast

from .production_readiness_json import _canonical_json_snapshot
from .production_readiness_values import (
    EXTERNAL_READINESS_AUTHENTICITY_ALGORITHM,
    EXTERNAL_READINESS_AUTHENTICITY_SCHEMA_VERSION,
    READINESS_MATRIX_SCHEMA_VERSION,
    ReadinessAuthenticationError,
    ReadinessTrustContext,
    _HMAC_RE,
    _canonical_contract_ref,
    _category_requires_external_observation,
    _matrix_id,
)

def authenticate_external_readiness_evidence(
    evidence: dict[str, Any],
    *,
    scope: str,
    category: str,
    task_id: str,
    revision: str,
    created_at: str,
    evidence_fresh_after: str,
    external_identity: dict[str, Any],
    trusted_context: ReadinessTrustContext,
) -> dict[str, Any]:
    """Attach local keyed integrity from a caller-held trusted context."""
    if not _trusted_context_usable(trusted_context):
        raise ValueError("external readiness authentication requires a usable trusted context")
    if not _category_requires_external_observation(category):
        raise ValueError("external readiness authentication requires a registry-owned external category")
    matrix_id = _matrix_id(scope, task_id, revision, created_at)
    payload = _external_authenticity_payload(
        evidence,
        matrix_schema_version=READINESS_MATRIX_SCHEMA_VERSION,
        matrix_id=matrix_id,
        contract_ref=_canonical_contract_ref(),
        context_id=trusted_context.context_id,
        scope=scope,
        category=category,
        task_id=task_id,
        revision=revision,
        created_at=created_at,
        evidence_fresh_after=evidence_fresh_after,
        requires_external_observation=True,
        external_identity=external_identity,
    )
    if payload is None:
        raise ReadinessAuthenticationError(
            "external readiness signed data is not canonical JSON and cannot be authenticated"
        )
    tag = trusted_context._sign(payload)
    return {
        **evidence,
        "authenticity": {
            "schema_version": EXTERNAL_READINESS_AUTHENTICITY_SCHEMA_VERSION,
            "algorithm": EXTERNAL_READINESS_AUTHENTICITY_ALGORITHM,
            "context_id": trusted_context.context_id,
            "hmac_sha256": tag,
        },
    }


def _external_authenticity_error(
    evidence: dict[str, Any],
    *,
    matrix_schema_version: str,
    matrix_id: str,
    contract_ref: dict[str, Any],
    scope: str,
    category: str,
    task_id: str,
    revision: str,
    created_at: str,
    evidence_fresh_after: str,
    requires_external_observation: bool,
    external_identity: dict[str, Any],
    trusted_context: ReadinessTrustContext | None,
) -> str:
    if type(trusted_context) is not ReadinessTrustContext or not _trusted_context_usable(trusted_context):
        return "external readiness authenticity requires a usable trusted context"
    authenticity = evidence.get("authenticity")
    if not isinstance(authenticity, dict) or set(authenticity) != {
        "schema_version",
        "algorithm",
        "context_id",
        "hmac_sha256",
    }:
        return "external readiness authenticity metadata is invalid"
    if (
        authenticity.get("schema_version") != EXTERNAL_READINESS_AUTHENTICITY_SCHEMA_VERSION
        or authenticity.get("algorithm") != EXTERNAL_READINESS_AUTHENTICITY_ALGORITHM
    ):
        return "external readiness authenticity metadata is invalid"
    if authenticity.get("context_id") != trusted_context.context_id:
        return "external readiness authenticity context does not match trusted context"
    candidate = authenticity.get("hmac_sha256")
    if not isinstance(candidate, str) or not _HMAC_RE.fullmatch(candidate):
        return "external readiness authenticity tag does not match trusted context"
    payload = _external_authenticity_payload(
        evidence,
        matrix_schema_version=matrix_schema_version,
        matrix_id=matrix_id,
        contract_ref=contract_ref,
        context_id=trusted_context.context_id,
        scope=scope,
        category=category,
        task_id=task_id,
        revision=revision,
        created_at=created_at,
        evidence_fresh_after=evidence_fresh_after,
        requires_external_observation=requires_external_observation,
        external_identity=external_identity,
    )
    if payload is None:
        return "external readiness signed data is not canonical JSON and is unauthenticated"
    if not trusted_context._verify(payload, candidate):
        return "external readiness authenticity tag does not match trusted context"
    return ""


def _external_authenticity_payload(
    evidence: dict[str, Any],
    *,
    matrix_schema_version: str,
    matrix_id: str,
    contract_ref: dict[str, Any],
    context_id: str,
    scope: str,
    category: str,
    task_id: str,
    revision: str,
    created_at: str,
    evidence_fresh_after: str,
    requires_external_observation: bool,
    external_identity: dict[str, Any],
) -> bytes | None:
    evidence_snapshot = _canonical_json_snapshot(evidence)
    if evidence_snapshot is None:
        return None
    unsigned_evidence = cast(dict[str, Any], evidence_snapshot[0])
    unsigned_evidence.pop("authenticity", None)
    payload = {
        "schema_version": "external_readiness_authenticity_payload/v1",
        "matrix_schema_version": matrix_schema_version,
        "matrix_id": matrix_id,
        "contract_ref": contract_ref,
        "context_id": context_id,
        "scope": scope,
        "category": category,
        "task_id": task_id,
        "revision": revision,
        "created_at": created_at,
        "evidence_fresh_after": evidence_fresh_after,
        "requires_external_observation": requires_external_observation,
        "external_identity": external_identity,
        "evidence": unsigned_evidence,
    }
    snapshot = _canonical_json_snapshot(payload)
    return None if snapshot is None else snapshot[1]


def _trusted_context_usable(context: ReadinessTrustContext | None) -> bool:
    return type(context) is ReadinessTrustContext and context._usable()
