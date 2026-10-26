"""Operator-reviewed, digest-bound migration of project scope identities.

Reports never write. Apply and rollback use the existing lifecycle transaction
journal; originals are moved, not edited, and reviews remain immutable.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from ..plugin_bundle.omh.memory_governance import canonical_payload_digest, stable_artifact_identity
from ..plugin_bundle.omh.project_identity import (
    LEGACY_BASENAME_STATE, project_identity_root, resolve_project_identity,
)
from ..system.local_store import atomic_write_json
from ..system.paths import OmhPaths, default_omh_home
from ._memory_lifecycle_model import LifecycleMutation, LifecyclePlan
from ._memory_lifecycle_plans import project_identity_successor
from ._memory_lifecycle_scan import json_files, read_json, safe_token
from .memory_lifecycle_executor import execute_memory_lifecycle
from .memory_store import checked_memory_json_path

REPORT_SCHEMA = "project_identity_migration_report/v1"
RECEIPT_SCHEMA = "project_identity_migration_receipt/v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _root(paths: OmhPaths, root: Path | None) -> Path:
    return root if root is not None else (project_identity_root() or Path.cwd())


def _stores(paths: OmhPaths, root: Path) -> dict[str, OmhPaths]:
    project = root / ".omh"
    user = paths.omh_home if paths.omh_home.resolve() != project.resolve() else default_omh_home()
    stores = {"project": OmhPaths(project, paths.hermes_home)}
    if user.resolve() != project.resolve():
        stores["user"] = OmhPaths(user, paths.hermes_home)
    return stores


def _read(paths: OmhPaths, relative: str) -> dict[str, Any]:
    value, error = read_json(paths.memory_dir, relative)
    if error or value is None:
        raise ValueError("migration_artifact_unreadable")
    return value


def _source_review(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    admission = record.get("admission", {})
    review_id = admission.get("review_id") if isinstance(admission, dict) else None
    if not safe_token(review_id):
        raise ValueError("migration_source_review_invalid")
    review = _read(paths, f"reviews/{review_id}.json")
    if (review.get("schema_version") not in {"project_memory_review_record/v2", "project_memory_review_record/v3"}
            or review.get("decision") not in {"approved_manual", "approved_auto_safe"}
            or review.get("artifact_identity") != stable_artifact_identity(record)
            or review.get("payload_digest") != canonical_payload_digest(record)):
        raise ValueError("migration_source_review_mismatch")
    return review


class ProjectIdentitySourceChangedError(ValueError):
    """The approved snapshot no longer describes this exact record/review pair."""


def _bound_source(paths: OmhPaths, entry: dict[str, Any], relative: str) -> dict[str, Any]:
    source, error = read_json(paths.memory_dir, relative)
    if error or source is None or source.get("revision") != entry["revision"] or _digest(source) != entry["source_digest"]:
        raise ProjectIdentitySourceChangedError("source_changed_since_report")
    review_id = entry.get("review_id")
    if not safe_token(review_id):
        raise ProjectIdentitySourceChangedError("source_changed_since_report")
    review, error = read_json(paths.memory_dir, f"reviews/{review_id}.json")
    if error or review is None or _digest(review) != entry.get("review_digest"):
        raise ProjectIdentitySourceChangedError("source_changed_since_report")
    return source


def _skipped(entry: dict[str, Any]) -> dict[str, str]:
    return {"record_id": entry["record_id"], "store": entry["store"], "reason_code": "source_changed_since_report"}


def build_project_identity_migration_report(paths: OmhPaths, *, root: Path | None = None) -> dict[str, Any]:
    root = _root(paths, root)
    resolution = resolve_project_identity(root)
    entries = []
    if resolution.state == "resolved":
        for store, local in _stores(paths, root).items():
            for relative, error in json_files(local.memory_dir, "records"):
                if error:
                    raise ValueError("migration_store_unreadable")
                record = _read(local, relative)
                scope = record.get("scope", {})
                if not isinstance(scope, dict) or scope.get("kind") != "project" or scope.get("ref") == resolution.identity:
                    continue
                record_id, current_ref = record.get("record_id"), scope.get("ref")
                if not safe_token(record_id) or not safe_token(current_ref):
                    raise ValueError("migration_metadata_invalid")
                # Keep the approved record/review snapshot in each report row;
                # apply must not replace these bindings with a subsequent read.
                admission = record.get("admission", {})
                review_id = admission.get("review_id", "") if isinstance(admission, dict) else ""
                review = _read(local, f"reviews/{review_id}.json") if safe_token(review_id) else {}
                entries.append({"record_id": record_id, "current_ref": current_ref, "proposed_ref": resolution.identity, "store": store,
                                "revision": record.get("revision"), "source_digest": _digest(record),
                                "review_id": review_id if safe_token(review_id) else "", "review_digest": _digest(review)})
    body = {"schema_version": REPORT_SCHEMA, "resolution": asdict(resolution), "entries": entries,
            "compatibility_state": LEGACY_BASENAME_STATE, "redaction_policy": "metadata_only"}
    if resolution.state != "resolved":
        body["guidance"] = "omh memory project-identity init"
    return {**body, "report_digest": _digest(body)}


def _receipt_path(paths: OmhPaths, receipt_id: str) -> Path:
    if not re.fullmatch(r"pim_[0-9a-f]{64}", receipt_id):
        raise ValueError("migration_receipt_invalid")
    return checked_memory_json_path(paths, f"project-identity-migrations/{receipt_id}.json")


def _plan(row: dict[str, Any], source: dict[str, Any], now: datetime, receipt_id: str) -> LifecyclePlan:
    record_id, revision = row["record_id"], row["revision"]
    target, review = project_identity_successor(source, row["proposed_ref"], now, operation_id=f"project-migrate-{receipt_id[4:28]}-{record_id}")
    if _digest(target) != row["target_digest"]:
        raise ValueError("migration_source_changed")
    mutations = (
        LifecycleMutation("preserve_original", "move", f"history/{record_id}.r{revision}.json", f"history:{record_id}:r{revision}", "history", source=f"records/{record_id}.json"),
        LifecycleMutation("write_successor_review", "write", f'reviews/{review["review_id"]}.json', f'review:{review["review_id"]}', "review", payload=review),
        LifecycleMutation("write_successor", "write", f"records/{record_id}.json", f"record:{record_id}:r{revision + 1}", "record", payload=target),
    )
    return LifecyclePlan(f"project-migrate-{receipt_id[4:28]}-{record_id}", "project_identity_migrate", record_id, revision + 1, target["scope"], now, {}, mutations)


def migrate_project_identity(paths: OmhPaths, *, approve: str, root: Path | None = None) -> dict[str, Any]:
    root = _root(paths, root)
    if not re.fullmatch(r"[0-9a-f]{64}", approve):
        raise ValueError("approval_digest_mismatch")
    receipt_id = "pim_" + approve
    destination = _receipt_path(paths, receipt_id)
    stores = _stores(paths, root)
    prior, error = read_json(paths.memory_dir, f"project-identity-migrations/{receipt_id}.json")
    if error not in {None, "already_absent"}:
        raise ValueError("migration_receipt_invalid")
    resolution = resolve_project_identity(root)
    receipt: dict[str, Any]
    if prior is not None:
        if prior.get("schema_version") != RECEIPT_SCHEMA or prior.get("report_digest") != approve or prior.get("project_identity") != resolution.identity or prior.get("state") not in {"prepared", "completed"}:
            raise ValueError("migration_receipt_mismatch")
        receipt = prior
    else:
        report = build_project_identity_migration_report(paths, root=root)
        if report["report_digest"] != approve or resolution.state != "resolved":
            raise ValueError("approval_digest_mismatch")
        now = datetime.now(timezone.utc)
        rows, skipped = [], []
        for entry in report["entries"]:
            local = stores[entry["store"]]
            try:
                source = _bound_source(local, entry, f'records/{entry["record_id"]}.json')
            except ProjectIdentitySourceChangedError:
                skipped.append(_skipped(entry))
                continue
            if source.get("schema_version") not in {"project_memory_record/v2", "project_memory_record/v3"} or source.get("source_class") != "omh_local" or type(source.get("revision")) is not int:
                raise ValueError("migration_requires_reviewed_revision")
            _source_review(local, source)
            successor, review = project_identity_successor(source, resolution.identity, now, operation_id=f'project-migrate-{receipt_id[4:28]}-{entry["record_id"]}')
            if (local.memory_dir / f'history/{entry["record_id"]}.r{source["revision"]}.json').exists() or (local.memory_dir / f'reviews/{review["review_id"]}.json').exists():
                raise ValueError("migration_successor_conflict")
            rows.append({**entry, "target_digest": _digest(successor)})
        receipt = {"schema_version": RECEIPT_SCHEMA, "receipt_id": receipt_id, "report_digest": approve,
                   "project_identity": resolution.identity, "created_at": now.isoformat(), "successors": rows, "skipped": skipped,
                   "rollback_token": receipt_id, "state": "prepared", "redaction_policy": "metadata_only"}
        atomic_write_json(destination, receipt, private=True)
    now = datetime.fromisoformat(receipt["created_at"])
    changed = []
    skipped = list(receipt.get("skipped", []))
    for row in receipt["successors"]:
        local = stores[row["store"]]
        current, _ = read_json(local.memory_dir, f'records/{row["record_id"]}.json')
        target_present = current is not None and _digest(current) == row["target_digest"]
        if receipt["state"] == "completed":
            if not target_present:
                raise ValueError("migration_target_changed")
            continue
        relative = f'history/{row["record_id"]}.r{row["revision"]}.json' if current is None or target_present else f'records/{row["record_id"]}.json'
        try:
            source = _bound_source(local, row, relative)
            _source_review(local, source)

            def preflight(locked: OmhPaths) -> None:
                if resolve_project_identity(root).identity != receipt["project_identity"]:
                    raise ValueError("migration_identity_changed")
                live_source = _bound_source(locked, row, f'records/{row["record_id"]}.json')
                _source_review(locked, live_source)
                if (locked.memory_dir / f'history/{row["record_id"]}.r{row["revision"]}.json').exists():
                    raise ValueError("migration_successor_conflict")

            result = execute_memory_lifecycle(local, _plan(row, source, now, receipt_id), preflight=preflight)
        except ProjectIdentitySourceChangedError:
            skipped.append(_skipped(row))
            continue
        if "receipt" not in result:
            raise ValueError("migration_operation_incomplete")
        changed.append(row)
    skipped_keys = {(row["store"], row["record_id"]) for row in skipped}
    receipt = {**receipt, "state": "completed", "skipped": skipped,
               "successors": [row for row in receipt["successors"] if (row["store"], row["record_id"]) not in skipped_keys]}
    atomic_write_json(destination, receipt, private=True)
    return {**receipt, "successors": changed, "idempotent": not changed}


def rollback_project_identity_migration(paths: OmhPaths, receipt_id: str, *, root: Path | None = None) -> dict[str, Any]:
    destination = _receipt_path(paths, receipt_id)
    receipt: dict[str, Any] = _read(paths, f"project-identity-migrations/{receipt_id}.json")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("receipt_id") != receipt_id or receipt.get("state") not in {"completed", "rolling_back", "rolled_back"}:
        raise ValueError("migration_receipt_invalid")
    stores = _stores(paths, _root(paths, root))
    # Validate every target before retiring any successor.
    for row in receipt["successors"]:
        local = stores[row["store"]]
        current, current_error = read_json(local.memory_dir, f'records/{row["record_id"]}.json')
        if current_error not in {None, "already_absent"}:
            raise ValueError("rollback_target_changed")
        if receipt["state"] in {"rolling_back", "rolled_back"} and (current is None or _digest(current) == row["source_digest"]):
            archived = _read(local, f'archive/project-identity-{row["record_id"]}.r{row["revision"] + 1}.json')
            if _digest(archived) != row["target_digest"]:
                raise ValueError("rollback_target_changed")
            if current is not None:
                continue
        elif current is None or _digest(current) != row["target_digest"]:
            raise ValueError("rollback_target_changed")
        source = _read(local, f'history/{row["record_id"]}.r{row["revision"]}.json')
        if _digest(source) != row["source_digest"]:
            raise ValueError("rollback_source_changed")
    if receipt["state"] != "rolled_back":
        receipt = {**receipt, "state": "rolling_back"}
        atomic_write_json(destination, receipt, private=True)
        for row in receipt["successors"]:
            record_id, revision = row["record_id"], row["revision"]
            mutations = (
                LifecycleMutation("retire_successor", "move", f"archive/project-identity-{record_id}.r{revision + 1}.json", f"archive:{record_id}:r{revision + 1}", "archive", source=f"records/{record_id}.json"),
                LifecycleMutation("restore_original", "move", f"records/{record_id}.json", f"record:{record_id}:r{revision}", "record", source=f"history/{record_id}.r{revision}.json"),
            )
            plan = LifecyclePlan(f"project-rollback-{receipt_id[4:28]}-{record_id}", "project_identity_rollback", record_id, revision, {"kind": "project", "ref": row["current_ref"]}, datetime.fromisoformat(receipt["created_at"]), {}, mutations)
            def preflight(locked: OmhPaths) -> None:
                current = _read(locked, f"records/{record_id}.json")
                source = _read(locked, f"history/{record_id}.r{revision}.json")
                if _digest(current) != row["target_digest"] or _digest(source) != row["source_digest"]:
                    raise ValueError("rollback_target_changed")
                if (locked.memory_dir / f"archive/project-identity-{record_id}.r{revision + 1}.json").exists():
                    raise ValueError("rollback_archive_conflict")

            result = execute_memory_lifecycle(stores[row["store"]], plan, preflight=preflight)
            if "receipt" not in result:
                raise ValueError("rollback_operation_incomplete")
    receipt = {**receipt, "state": "rolled_back"}
    atomic_write_json(destination, receipt, private=True)
    return receipt
