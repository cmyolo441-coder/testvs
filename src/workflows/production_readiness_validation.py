from __future__ import annotations

from typing import Any, cast

from . import production_readiness_json as readiness_json
from .production_readiness_authentication import _external_authenticity_error
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
from .production_readiness_values import (
    EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION,
    READINESS_CATEGORIES,
    READINESS_EVIDENCE_STATES,
    READINESS_MATRIX_ROLLBACK_CONTRACT,
    READINESS_MATRIX_SCHEMA_VERSION,
    READINESS_VERDICTS,
    ReadinessTrustContext,
    ReadinessValidationResult,
    ValidatedReadinessArtifact,
    _READINESS_CAPTURE_ERRORS,
    _canonical_contract_ref,
    _category_policy_error,
    _category_requires_external_observation,
    _duplicates,
    _matrix_id,
)

def parse_readiness_matrix(
    value: object, *, trusted_context: ReadinessTrustContext | None = None
) -> ReadinessValidationResult:
    """Capture caller data twice and validate only one matching detached snapshot."""
    if not isinstance(value, dict):
        return ReadinessValidationResult(("readiness_matrix must be an object",), "HOLD", None)
    first = readiness_json._canonical_json_snapshot(value)
    second = readiness_json._canonical_json_snapshot(value)
    if first is None or second is None or first[1] != second[1]:
        return ReadinessValidationResult(_READINESS_CAPTURE_ERRORS, "HOLD", None)
    record = cast(dict[str, Any], first[0])
    errors, derived_verdict = _validate_readiness_snapshot(record, trusted_context=trusted_context)
    if errors:
        return ReadinessValidationResult(tuple(errors), derived_verdict, None)
    artifact = ValidatedReadinessArtifact(first[1], derived_verdict)
    return ReadinessValidationResult((), derived_verdict, artifact)


def validate_readiness_matrix(
    record: dict[str, Any], *, trusted_context: ReadinessTrustContext | None = None
) -> list[str]:
    """Validate a captured artifact; no errors do not promise caller post-return stability."""
    return list(parse_readiness_matrix(record, trusted_context=trusted_context).errors)


def _validate_readiness_snapshot(
    record: dict[str, Any], *, trusted_context: ReadinessTrustContext | None = None
) -> tuple[list[str], str]:
    errors: list[str] = []
    if record.get("schema_version") != READINESS_MATRIX_SCHEMA_VERSION:
        errors.append("schema_version must be readiness_matrix/v1")
    scope = str(record.get("scope", ""))
    task_id = str(record.get("task_id", ""))
    revision = str(record.get("revision", ""))
    created_at = str(record.get("created_at", ""))
    fresh_after = str(record.get("evidence_fresh_after", ""))
    errors.extend(
        _matrix_identity_and_time_errors(
            scope=scope,
            task_id=task_id,
            revision=revision,
            created_at=created_at,
            evidence_fresh_after=fresh_after,
        )
    )
    matrix_id = str(record.get("matrix_id", ""))
    if not matrix_id.strip():
        errors.append("matrix_id is required")
    elif matrix_id != _matrix_id(scope, task_id, revision, created_at):
        errors.append("matrix_id must match canonical scope, task, revision, and created_at identity")
    supplied_contract_ref = record.get("contract_ref")
    contract_ref = supplied_contract_ref if isinstance(supplied_contract_ref, dict) else {}
    if not isinstance(supplied_contract_ref, dict):
        errors.append("contract_ref must be an object")
    elif contract_ref != _canonical_contract_ref():
        errors.append("contract_ref must identify the executable readiness_matrix consumer")
    if record.get("rollback_contract") != READINESS_MATRIX_ROLLBACK_CONTRACT:
        errors.append("rollback_contract must match readiness_matrix_rollback/v1")
    rows = record.get("rows")
    if not isinstance(rows, list):
        errors.append("rows must be a list")
        rows = []
    categories = [str(row.get("category", "")) for row in rows if isinstance(row, dict)]
    duplicates = sorted({category for category in categories if category and categories.count(category) > 1})
    missing = sorted(set(READINESS_CATEGORIES) - set(categories))
    extra = sorted(set(categories) - set(READINESS_CATEGORIES))
    if duplicates:
        errors.append(f"duplicate readiness categories: {', '.join(duplicates)}")
    if missing:
        errors.append(f"missing readiness categories: {', '.join(missing)}")
    if extra:
        errors.append(f"unsupported readiness categories: {', '.join(extra)}")
    derived_rows: list[dict[str, Any]] = []
    receipt_ids: list[str] = []
    observation_ids: list[str] = []
    check_ids: list[str] = []
    check_evidence_refs: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"rows[{index}] must be an object")
            continue
        category = str(row.get("category", ""))
        policy_requires_external = _category_requires_external_observation(category)
        supplied_requires_external = row.get("requires_external_observation")
        if not isinstance(supplied_requires_external, bool):
            errors.append(f"rows[{index}].requires_external_observation must be a boolean")
        policy_matches_row = (
            isinstance(supplied_requires_external, bool)
            and supplied_requires_external == policy_requires_external
        )
        if not policy_matches_row:
            errors.append(f"rows[{index}]: {_category_policy_error(category, policy_requires_external)}")
        requires_external = policy_requires_external
        authentication_external_mode = supplied_requires_external if isinstance(supplied_requires_external, bool) else False
        supplied_identity = row.get("external_identity")
        external_identity = supplied_identity if isinstance(supplied_identity, dict) else {}
        for error in _external_identity_errors(external_identity, required=requires_external):
            errors.append(f"rows[{index}]: {error}")
        if not requires_external and external_identity:
            errors.append(f"rows[{index}]: non-external readiness rows must not carry external_identity")
        evidence = row.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"rows[{index}].evidence must be a list")
            evidence = []
        external_items = [
            item
            for item in evidence
            if isinstance(item, dict) and item.get("schema_version") == EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION
        ]
        authenticated_items = [item for item in evidence if isinstance(item, dict) and "authenticity" in item]
        if requires_external and len(external_items) > 1:
            errors.append(f"rows[{index}] external readiness rows accept exactly one external readiness association")
        if not requires_external and external_items:
            errors.append(f"rows[{index}] category policy requires local observed_check_result evidence")
        for evidence_index, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"rows[{index}].evidence[{evidence_index}] must be an object")
                continue
            for error in _evidence_errors(
                item,
                external_identity=external_identity,
                scope=scope,
                category=category,
                task_id=task_id,
                revision=revision,
                observed_before=created_at,
            ):
                errors.append(f"rows[{index}].evidence[{evidence_index}]: {error}")
        valid_evidence = [item for item in evidence if isinstance(item, dict)]
        receipt_ids.extend(_external_receipt_ids(valid_evidence))
        observation_ids.extend(_external_observation_ids(valid_evidence))
        check_ids.extend(_local_check_ids(valid_evidence))
        check_evidence_refs.extend(_local_evidence_refs(valid_evidence))
        if requires_external and any(
            item.get("schema_version") != EXTERNAL_READINESS_EVIDENCE_SCHEMA_VERSION for item in valid_evidence
        ):
            errors.append(f"rows[{index}] external observed success requires one exactly bound external association")
        derived_state = _derive_evidence_state(
            valid_evidence,
            requires_external=requires_external,
            external_identity=external_identity,
            matrix_schema_version=str(record.get("schema_version", "")),
            matrix_id=matrix_id,
            contract_ref=contract_ref,
            scope=scope,
            category=str(row.get("category", "")),
            task_id=task_id,
            revision=revision,
            fresh_after=fresh_after,
            observed_before=created_at,
            trusted_context=trusted_context,
        )
        if not policy_matches_row:
            derived_state = "missing"
        if row.get("evidence_state") not in READINESS_EVIDENCE_STATES:
            errors.append(f"rows[{index}].evidence_state is unsupported: {row.get('evidence_state')}")
        if authenticated_items and row.get("evidence_state") == "observed" and derived_state != "observed":
            authentication_errors = [
                _external_authenticity_error(
                    item,
                    matrix_schema_version=str(record.get("schema_version", "")),
                    matrix_id=matrix_id,
                    contract_ref=contract_ref,
                    scope=scope,
                    category=category,
                    task_id=task_id,
                    revision=revision,
                    created_at=created_at,
                    evidence_fresh_after=fresh_after,
                    requires_external_observation=authentication_external_mode,
                    external_identity=external_identity,
                    trusted_context=trusted_context,
                )
                for item in authenticated_items
            ]
            errors.extend(error for error in authentication_errors if error)
            errors.append(f"rows[{index}] external observed success requires one exactly bound authenticated association")
        if row.get("evidence_state") != derived_state:
            errors.append(f"rows[{index}].evidence_state must match derived evidence state {derived_state}")
        derived_rows.append({"category": row.get("category", ""), "evidence_state": derived_state})
    duplicate_receipts = sorted({receipt_id for receipt_id in receipt_ids if receipt_ids.count(receipt_id) > 1})
    if duplicate_receipts:
        errors.append(f"duplicate external readiness receipt associations: {', '.join(duplicate_receipts)}")
    duplicate_observations = _duplicates(observation_ids)
    if duplicate_observations:
        errors.append(f"duplicate external readiness postcondition associations: {', '.join(duplicate_observations)}")
    duplicate_checks = _duplicates(check_ids)
    if duplicate_checks:
        errors.append(f"duplicate observed_check_result check_id associations: {', '.join(duplicate_checks)}")
    duplicate_check_refs = _duplicates(check_evidence_refs)
    if duplicate_check_refs:
        errors.append(
            f"duplicate observed_check_result evidence_ref associations: {', '.join(duplicate_check_refs)}"
        )
    derived_verdict = _derive_verdict(derived_rows)
    verdict = record.get("verdict")
    if verdict not in READINESS_VERDICTS:
        errors.append(f"unsupported verdict: {verdict}")
    if verdict != derived_verdict:
        errors.append(f"verdict must match derived verdict {derived_verdict}")
    derived_missing = [str(row.get("category", "")) for row in derived_rows if row.get("evidence_state") != "observed"]
    if record.get("missing_categories") != derived_missing:
        errors.append("missing_categories must match non-observed rows")
    return errors, derived_verdict
