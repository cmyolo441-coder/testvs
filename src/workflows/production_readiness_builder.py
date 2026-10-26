from __future__ import annotations

from typing import Any

from .production_readiness_evidence import (
    _derive_evidence_state,
    _derive_verdict,
    _external_observation_ids,
    _external_receipt_ids,
    _local_check_ids,
    _local_evidence_refs,
)
from .production_readiness_structural import (
    _evidence_errors,
    _external_identity_errors,
    _matrix_identity_and_time_errors,
)
from .production_readiness_validation import parse_readiness_matrix
from .production_readiness_values import (
    EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION,
    READINESS_MATRIX_ROLLBACK_CONTRACT,
    READINESS_MATRIX_SCHEMA_VERSION,
    ReadinessTrustContext,
    _canonical_contract_ref,
    _category_policy_error,
    _category_requires_external_observation,
    _duplicates,
    _matrix_id,
)

def build_readiness_matrix(
    *,
    scope: str,
    task_id: str,
    revision: str,
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    created_at: str,
    evidence_fresh_after: str,
    trusted_context: ReadinessTrustContext | None = None,
) -> dict[str, Any]:
    clean_scope = str(scope).strip()
    clean_task_id = str(task_id).strip()
    clean_revision = str(revision).strip()
    top_errors = _matrix_identity_and_time_errors(
        scope=clean_scope,
        task_id=clean_task_id,
        revision=clean_revision,
        created_at=created_at,
        evidence_fresh_after=evidence_fresh_after,
    )
    if top_errors:
        raise ValueError("; ".join(top_errors))
    matrix_id = _matrix_id(clean_scope, clean_task_id, clean_revision, created_at)
    contract_ref = _canonical_contract_ref()
    normalized_rows: list[dict[str, Any]] = []
    receipt_ids: list[str] = []
    observation_ids: list[str] = []
    check_ids: list[str] = []
    check_evidence_refs: list[str] = []
    for supplied in rows:
        category = str(supplied.get("category", "")).strip()
        policy_requires_external = _category_requires_external_observation(category)
        supplied_requires_external = supplied.get("requires_external_observation")
        identity_errors: list[str] = []
        if not isinstance(supplied_requires_external, bool) or supplied_requires_external != policy_requires_external:
            identity_errors.append(_category_policy_error(category, policy_requires_external))
        requires_external = policy_requires_external
        supplied_identity = supplied.get("external_identity")
        external_identity = supplied_identity if isinstance(supplied_identity, dict) else {}
        identity_errors.extend(_external_identity_errors(external_identity, required=requires_external))
        if not requires_external and external_identity:
            identity_errors.append("non-external readiness rows must not carry external_identity")
        supplied_evidence = supplied.get("evidence")
        evidence = supplied_evidence if isinstance(supplied_evidence, list) else []
        external_items = [
            item
            for item in evidence
            if isinstance(item, dict) and item.get("schema_version") == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION
        ]
        if requires_external and len(external_items) > 1:
            identity_errors.append("external readiness rows accept exactly one external readiness association")
        if not requires_external and external_items:
            identity_errors.append("category policy requires local observed_check_result evidence")
        evidence_errors = [
            error
            for item in evidence
            for error in (
                ["readiness evidence must be an object"]
                if not isinstance(item, dict)
                else _evidence_errors(
                    item,
                    external_identity=external_identity,
                    scope=clean_scope,
                    category=category,
                    task_id=clean_task_id,
                    revision=clean_revision,
                    observed_before=created_at,
                )
            )
        ]
        if identity_errors or evidence_errors:
            raise ValueError(
                f"invalid evidence for {category or '<unknown>'}: {'; '.join([*identity_errors, *evidence_errors])}"
            )
        accepted = [item for item in evidence if isinstance(item, dict)]
        receipt_ids.extend(_external_receipt_ids(accepted))
        observation_ids.extend(_external_observation_ids(accepted))
        check_ids.extend(_local_check_ids(accepted))
        check_evidence_refs.extend(_local_evidence_refs(accepted))
        normalized_rows.append(
            {
                "category": category,
                "requires_external_observation": requires_external,
                "external_identity": dict(external_identity),
                "evidence_state": _derive_evidence_state(
                    accepted,
                    requires_external=requires_external,
                    external_identity=external_identity,
                    task_id=clean_task_id,
                    matrix_schema_version=READINESS_MATRIX_SCHEMA_VERSION,
                    matrix_id=matrix_id,
                    contract_ref=contract_ref,
                    scope=clean_scope,
                    category=category,
                    revision=clean_revision,
                    fresh_after=evidence_fresh_after,
                    observed_before=created_at,
                    trusted_context=trusted_context,
                ),
                "evidence": accepted,
            }
        )
    duplicate_receipts = sorted({receipt_id for receipt_id in receipt_ids if receipt_ids.count(receipt_id) > 1})
    if duplicate_receipts:
        raise ValueError(f"duplicate external readiness receipt associations: {', '.join(duplicate_receipts)}")
    duplicate_observations = _duplicates(observation_ids)
    if duplicate_observations:
        raise ValueError(f"duplicate external readiness postcondition associations: {', '.join(duplicate_observations)}")
    duplicate_checks = _duplicates(check_ids)
    if duplicate_checks:
        raise ValueError(f"duplicate observed_check_result check_id associations: {', '.join(duplicate_checks)}")
    duplicate_check_refs = _duplicates(check_evidence_refs)
    if duplicate_check_refs:
        raise ValueError(
            f"duplicate observed_check_result evidence_ref associations: {', '.join(duplicate_check_refs)}"
        )
    verdict = _derive_verdict(normalized_rows)
    record = {
        "schema_version": READINESS_MATRIX_SCHEMA_VERSION,
        "matrix_id": matrix_id,
        "scope": clean_scope,
        "task_id": clean_task_id,
        "revision": clean_revision,
        "created_at": created_at,
        "evidence_fresh_after": evidence_fresh_after,
        "contract_ref": contract_ref,
        "rows": normalized_rows,
        "verdict": verdict,
        "missing_categories": [
            str(row.get("category", "")) for row in normalized_rows if row.get("evidence_state") != "observed"
        ],
        "rollback_contract": dict(READINESS_MATRIX_ROLLBACK_CONTRACT),
    }
    result = parse_readiness_matrix(record, trusted_context=trusted_context)
    return result.require_artifact().detached_copy()
