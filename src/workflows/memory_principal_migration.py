"""Reviewed principal migration and owner-filtered export for memory records."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PrincipalMigrationError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason

from ..plugin_bundle.omh.memory_governance import canonical_payload_digest, stable_artifact_identity
from ..plugin_bundle.omh.memory_principals import memory_identity_errors, principal_operation_decision
from ..system.local_store import read_json_object_result
from ..system.paths import OmhPaths
from .memory_principal_assignment import assignment_review_reason, source_review_reason
from .memory_store import run_memory_operation

REPORT_SCHEMA = "memory_principal_migration_report/v1"
PLAN_SCHEMA = "memory_principal_migration_plan/v1"
MIGRATION_SCHEMA = "memory_principal_migration/v1"
EXPORT_SCHEMA = "memory_principal_export/v1"


def build_principal_migration_report(paths: OmhPaths, *, limit: int = 100) -> dict[str, Any]:
    """List identity-unbound records without writing or exposing their content."""
    records: list[dict[str, Any]] = []
    for path in _record_paths(paths):
        value, error = read_json_object_result(path)
        if error or value is None or value.get("schema_version") not in {
            "project_memory_record/v1",
            "project_memory_record/v2",
        }:
            continue
        record_id = value.get("record_id")
        revision = value.get("revision")
        if not isinstance(record_id, str) or not record_id or not isinstance(revision, int):
            continue
        admission: Any = value.get("admission") if isinstance(value.get("admission"), dict) else {}
        records.append({
            "record_id": record_id,
            "revision": revision,
            "schema_version": str(value.get("schema_version")),
            "review_id": str(admission.get("review_id", "")),
            "payload_digest": canonical_payload_digest(value),
            "identity_state": "legacy_unbound",
        })
    bounded = records[: max(0, min(limit, 100))]
    return {
        "schema_version": REPORT_SCHEMA,
        "records": bounded,
        "record_count": len(bounded),
        "omitted_count": max(len(records) - len(bounded), 0),
        "writes_performed": False,
        "redaction_policy": "metadata_only",
        "claim_boundary": "This report assigns no principal and contains no memory content.",
    }


def principal_migration_plan_digest(plan: Mapping[str, Any]) -> str:
    body = {key: value for key, value in plan.items() if key != "plan_digest"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def apply_principal_migration(paths: OmhPaths, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Apply one exact reviewed v2-to-v3 assignment through the operation store."""
    base_fields = {"schema_version", "record_id", "revision", "review_id", "identity", "plan_digest"}
    if set(plan) not in {frozenset(base_fields), frozenset({*base_fields, "assignment_review_id"})}:
        raise PrincipalMigrationError("principal migration plan fields are invalid")
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("plan_digest") != principal_migration_plan_digest(plan):
        raise PrincipalMigrationError("principal migration plan digest is invalid")
    record_id, revision, review_id = plan.get("record_id"), plan.get("revision"), plan.get("review_id")
    if not isinstance(record_id, str) or not isinstance(revision, int) or not isinstance(review_id, str):
        raise PrincipalMigrationError("principal migration identity is invalid")
    operation_id = "principal-migrate-" + str(plan["plan_digest"])[7:31]
    prefix = f"principal-migrations/{operation_id}"
    prior, prior_error = read_json_object_result(paths.memory_dir / prefix / "migration.json")
    if prior_error is None and prior is not None:
        current, current_error = read_json_object_result(paths.memory_dir / "records" / f"{record_id}.json")
        if current_error is None and current is not None and canonical_payload_digest(current) == prior.get("target_digest"):
            return {"schema_version": MIGRATION_SCHEMA, "applied": True, "idempotent": True, "operation_id": operation_id, "state": "completed", "target": prior.get("target", {}), "redaction_policy": "metadata_only"}
        return _migration_refusal("migration_operation_conflict")
    source, error = read_json_object_result(paths.memory_dir / "records" / f"{record_id}.json")
    if error or source is None or source.get("schema_version") != "project_memory_record/v2" or source.get("revision") != revision:
        return _migration_refusal("source_revision_changed")
    admission: Any = source.get("admission") if isinstance(source.get("admission"), dict) else {}
    if admission.get("review_id") != review_id:
        return _migration_refusal("source_review_mismatch")
    review_reason = source_review_reason(paths, source, review_id)
    if review_reason:
        return _migration_refusal(review_reason)
    identity = plan.get("identity")
    if not isinstance(identity, dict) or memory_identity_errors(identity):
        raise PrincipalMigrationError("principal migration identity block is invalid")
    assignment_review_id = plan.get("assignment_review_id")
    if not isinstance(assignment_review_id, str) or not assignment_review_id:
        return _migration_refusal("assignment_review_required")
    assignment_reason = assignment_review_reason(
        paths,
        source,
        review_id,
        identity,
        assignment_review_id,
    )
    if assignment_reason:
        return _migration_refusal(assignment_reason)
    target_review_id = "principal-" + hashlib.sha256(str(plan["plan_digest"]).encode()).hexdigest()[:24]
    target_identity: Any = json.loads(json.dumps(identity))
    target_identity["reviewer"]["review_ref"] = assignment_review_id
    target_identity["audience"]["review_ref"] = assignment_review_id
    subject = target_identity.get("subject_principal")
    target_scope = {"kind": "user", "ref": subject} if isinstance(subject, str) and subject else source.get("scope")
    target: Any = {
        **source,
        "schema_version": "project_memory_record/v3",
        "revision": revision + 1,
        "scope": target_scope,
        "identity": target_identity,
        "admission": {**admission, "review_id": target_review_id},
    }
    target["admission"] = {**target["admission"], "payload_digest": canonical_payload_digest(target)}
    target_review = {
        "schema_version": "project_memory_review_record/v3",
        "review_id": target_review_id,
        "artifact_identity": stable_artifact_identity(target),
        "decision": "approved_manual",
        "reviewer_claim": "reviewed_principal_migration",
        "source_review_id": review_id,
        "assignment_review_id": assignment_review_id,
        "payload_digest": canonical_payload_digest(target),
        "identity": target_identity,
    }
    migration: Any = {
        "schema_version": MIGRATION_SCHEMA,
        "operation_id": operation_id,
        "source": {"record_id": record_id, "revision": revision, "review_id": review_id},
        "target": {"record_id": record_id, "revision": revision + 1, "review_id": target_review_id},
        "source_digest": canonical_payload_digest(source),
        "assignment_review_id": assignment_review_id,
        "target_digest": canonical_payload_digest(target),
        "plan_digest": plan["plan_digest"],
        "rollback_state": "available",
    }
    steps = [
        {"name": "archive_source", "action": "write_json", "target": f"{prefix}/source.json", "payload": source},
        {"name": "write_target_review", "action": "write_json", "target": f"reviews/{target_review_id}.json", "payload": target_review},
        {"name": "write_target", "action": "write_json", "target": f"records/{record_id}.json", "payload": target},
        {"name": "write_reverse_mapping", "action": "write_json", "target": f"{prefix}/migration.json", "payload": migration},
    ]
    operation = run_memory_operation(paths, operation_id=operation_id, operation_type="principal_migration", steps=steps)
    return {
        "schema_version": MIGRATION_SCHEMA,
        "applied": operation.get("state") == "completed",
        "idempotent": False,
        "operation_id": operation_id,
        "state": operation.get("state"),
        "target": migration["target"],
        "receipt": operation.get("receipt", {}),
        "redaction_policy": "metadata_only",
    }


def rollback_principal_migration(paths: OmhPaths, operation_id: str) -> dict[str, Any]:
    """Restore the archived source only while the migrated revision is unchanged."""
    prefix = f"principal-migrations/{operation_id}"
    migration, error = read_json_object_result(paths.memory_dir / prefix / "migration.json")
    source, source_error = read_json_object_result(paths.memory_dir / prefix / "source.json")
    if error or source_error or migration is None or source is None:
        return _migration_refusal("migration_not_found")
    target: Any = migration.get("target") if isinstance(migration.get("target"), dict) else {}
    record_id = str(target.get("record_id", ""))
    current, current_error = read_json_object_result(paths.memory_dir / "records" / f"{record_id}.json")
    rollback, rollback_error = read_json_object_result(paths.memory_dir / prefix / "rollback.json")
    if (
        rollback_error is None
        and rollback is not None
        and current_error is None
        and current is not None
        and canonical_payload_digest(current) == rollback.get("restored_digest")
    ):
        return {"schema_version": "memory_principal_migration_rollback/v1", "applied": True, "idempotent": True, "operation_id": operation_id, "state": "completed"}
    if current_error or current is None or canonical_payload_digest(current) != migration.get("target_digest"):
        return _migration_refusal("rollback_target_changed")
    rollback_id = "principal-rollback-" + hashlib.sha256(operation_id.encode()).hexdigest()[:24]
    steps = [
        {"name": "restore_source", "action": "write_json", "target": f"records/{record_id}.json", "payload": source},
        {"name": "remove_target_review", "action": "delete", "target": f"reviews/{target.get('review_id', '')}.json"},
        {"name": "write_rollback", "action": "write_json", "target": f"{prefix}/rollback.json", "payload": {"schema_version": "memory_principal_migration_rollback/v1", "operation_id": operation_id, "restored_digest": canonical_payload_digest(source)}},
    ]
    operation = run_memory_operation(paths, operation_id=rollback_id, operation_type="principal_migration_rollback", steps=steps)
    return {"schema_version": "memory_principal_migration_rollback/v1", "applied": operation.get("state") == "completed", "idempotent": False, "operation_id": operation_id, "state": operation.get("state"), "receipt": operation.get("receipt", {})}


def export_principal_memory(paths: OmhPaths, context: dict[str, Any]) -> dict[str, Any]:
    """Export only records authorized for one acting principal."""
    records: list[dict[str, Any]] = []
    denied = 0
    for path in _record_paths(paths):
        value, error = read_json_object_result(path)
        if error or value is None:
            continue
        if principal_operation_decision(value, context)["allowed"]:
            records.append(value)
        else:
            denied += 1
    return {"schema_version": EXPORT_SCHEMA, "records": records, "record_count": len(records), "denied_count": denied, "redaction_policy": "reviewed_content"}


def _record_paths(paths: OmhPaths) -> list[Path]:
    directory = paths.memory_dir / "records"
    return sorted(directory.glob("*.json")) if directory.is_dir() and not directory.is_symlink() else []


def _migration_refusal(reason: str) -> dict[str, Any]:
    return {"schema_version": MIGRATION_SCHEMA, "applied": False, "reason_code": reason, "redaction_policy": "metadata_only"}
