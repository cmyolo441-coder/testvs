"""Phase 1 read-only memory inventory, ledger, and per-artifact reactivation."""

from __future__ import annotations

import re
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..plugin_bundle.omh.memory_governance import canonical_memory_scope
from ..system.local_store import atomic_write_json
from ..system.paths import OmhPaths
from ._memory_migration_reactivation import reactivate
from ._memory_migration_reactivation_data import safe_target
from ._memory_migration_scan import MAX_INVENTORY_ITEMS, MAX_OMISSIONS, build_inventory
from .memory_store import apply_memory_operation_step, run_memory_operation

MEMORY_MIGRATION_INVENTORY_SCHEMA = "memory_migration_inventory/v1"
MEMORY_COPY_LINK_MANIFEST_SCHEMA = "memory_copy_link_manifest/v1"
MAX_LEDGER_ARTIFACTS = 100
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")
_SCHEMA = re.compile(r"^[A-Za-z0-9_.-]{1,120}/v[12]$")
class OperationRunner(Protocol):
    def __call__(self, paths: OmhPaths, *, operation_id: str, operation_type: str, steps: Sequence[Mapping[str, object]], step_writer: Callable[[OmhPaths, dict[str, str]], None], now: datetime | None = None) -> dict[str, Any]: ...


__all__ = [
    "MAX_INVENTORY_ITEMS",
    "MAX_LEDGER_ARTIFACTS",
    "MAX_OMISSIONS",
    "MEMORY_COPY_LINK_MANIFEST_SCHEMA",
    "MEMORY_MIGRATION_INVENTORY_SCHEMA",
    "OperationRunner",
    "build_memory_migration_inventory",
    "reactivate_memory_artifact",
    "validate_memory_copy_link_manifest",
    "write_memory_migration_ledger",
]


def build_memory_migration_inventory(paths: OmhPaths) -> dict[str, object]:
    """Read every known local surface. This dry-run function never writes."""
    return build_inventory(paths)


def write_memory_migration_ledger(
    paths: OmhPaths,
    inventory: Mapping[str, object],
    *,
    ledger_id: str | None = None,
    operation_runner: OperationRunner | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Explicitly persist a bounded metadata-only inventory under the shared lock."""
    if inventory.get("schema_version") != MEMORY_MIGRATION_INVENTORY_SCHEMA:
        raise ValueError("unsupported_inventory_schema")
    identifier = ledger_id or f"migration_{secrets.token_hex(12)}"
    if not _SAFE.fullmatch(identifier):
        raise ValueError("invalid_ledger_id")
    moment = now or datetime.now(timezone.utc)
    ledger = _ledger(inventory, identifier, moment)
    target = f"migrations/{identifier}.json"

    def writer(active_paths: OmhPaths, step: dict[str, str]) -> None:
        if step["name"] != "write_migration_ledger":
            apply_memory_operation_step(active_paths, step)
            return
        path = safe_target(active_paths, step["target"])
        if path.exists():
            from ..system.local_store import read_json_object_result

            current, error = read_json_object_result(path)
            if not error and current == ledger:
                return
        atomic_write_json(path, ledger, private=True)

    runner = operation_runner or run_memory_operation
    operation_id = f"inventory_{identifier}"
    operation = runner(
        paths,
        operation_id=operation_id,
        operation_type="memory_inventory",
        steps=[{"name": "write_migration_ledger", "action": "write", "target": target, "key": identifier, "revision": "1"}],
        step_writer=writer,
        now=moment,
    )
    return {"schema_version": "memory_migration_ledger_write/v1", "ledger_id": identifier, "operation_id": operation_id, "state": str(operation.get("state", "")), "receipt": operation.get("receipt", {}), "redaction_policy": "metadata_only"}


def reactivate_memory_artifact(
    paths: OmhPaths,
    artifact_id: str,
    *,
    review_id: str,
    apply: bool = False,
    artifact_kind: str | None = None,
    operation_runner: OperationRunner | None = None,
    write_hook: Callable[[str], None] | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Apply exactly one v1 artifact's reviewed conversion; report-only by default."""
    if not isinstance(artifact_id, str) or not _SAFE.fullmatch(artifact_id):
        return _reactivation_error("invalid_artifact_id")
    if not isinstance(review_id, str) or not _SAFE.fullmatch(review_id):
        return _reactivation_error("invalid_review_id")
    if artifact_kind not in {None, "record", "scope_item", "block"}:
        return _reactivation_error("unsupported_artifact_kind")
    return reactivate(paths, artifact_id, review_id=review_id, apply=apply, artifact_kind=artifact_kind, operation_runner=operation_runner, write_hook=write_hook, now=now)


def validate_memory_copy_link_manifest(value: object) -> list[str]:
    """Field allowlist for migration copy/link evidence; it forbids content metadata."""
    if not isinstance(value, dict):
        return ["manifest_must_be_object"]
    allowed = {"schema_version", "artifact_identity", "links"}
    errors = ["manifest_has_unsupported_fields"] if set(value) - allowed else []
    if value.get("schema_version") != MEMORY_COPY_LINK_MANIFEST_SCHEMA:
        errors.append("unsupported_manifest_schema")
    errors.extend(_identity_errors(value.get("artifact_identity"), "artifact_identity"))
    links = value.get("links")
    if not isinstance(links, list) or len(links) > 32:
        return errors + ["links_must_be_bounded_list"]
    for link in links:
        if not isinstance(link, dict) or set(link) != {"relation", "artifact_identity"}:
            errors.append("link_has_unsupported_fields")
            continue
        if link["relation"] not in {"copy", "link"}:
            errors.append("invalid_link_relation")
        errors.extend(_identity_errors(link["artifact_identity"], "link_identity"))
    return sorted(set(errors))


def _ledger(inventory: Mapping[str, object], ledger_id: str, now: datetime) -> dict[str, object]:
    return {
        "schema_version": MEMORY_MIGRATION_INVENTORY_SCHEMA,
        "ledger_id": ledger_id,
        "created_at": _stamp(now),
        "inventory_schema_version": MEMORY_MIGRATION_INVENTORY_SCHEMA,
        "counts": _counts(inventory.get("counts")),
        "artifacts": _artifacts(inventory.get("artifacts"))[:MAX_LEDGER_ARTIFACTS],
        "omissions": _rows(inventory.get("omissions"), {"reason_code", "omitted_count"})[:MAX_OMISSIONS],
        "quarantine_proposals": _quarantines(inventory.get("quarantine_proposals"))[:MAX_OMISSIONS],
        "external_exclusions": _rows(inventory.get("external_exclusions"), {"source_class", "reason_code"})[:MAX_OMISSIONS],
        "redaction_policy": "metadata_only",
        "claim_boundary": "This ledger records a dry-run migration inventory; it is not approval, quarantine, or replay evidence.",
    }


def _counts(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, object] = {"total": value["total"]} if isinstance(value.get("total"), int) and not isinstance(value["total"], bool) and value["total"] >= 0 else {}
    for name in ("classification", "source_bucket"):
        source = value.get(name)
        if isinstance(source, dict):
            result[name] = {key: item for key, item in sorted(source.items()) if _SAFE.fullmatch(str(key)) and isinstance(item, int) and not isinstance(item, bool) and item >= 0}
    return result


def _artifacts(value: object) -> list[dict[str, object]]:
    allowed = {"source_bucket", "artifact_kind", "classification", "replay_status", "required_action", "schema_version", "artifact_ref", "scope_status", "omission_reason"}
    rows = _rows(value, allowed)
    for source, result in zip(value if isinstance(value, list) else [], rows, strict=False):
        identity = _ledger_identity(source.get("artifact_identity")) if isinstance(source, dict) else None
        if identity is not None:
            result["artifact_identity"] = identity
    return rows


def _quarantines(value: object) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in value if isinstance(value, list) else []:
        if not isinstance(row, dict) or row.get("action") != "quarantine_proposal" or not _SAFE.fullmatch(str(row.get("reason_code", ""))):
            continue
        artifact = row.get("artifact")
        output: dict[str, object] = {"action": "quarantine_proposal", "reason_code": row["reason_code"]}
        identity = _ledger_identity(artifact)
        if identity is not None:
            output["artifact"] = identity
        elif isinstance(artifact, dict) and artifact.get("artifact_ref") == "identity_unavailable":
            output["artifact"] = {"artifact_ref": "identity_unavailable"}
        rows.append(output)
    return rows


def _rows(value: object, allowed: set[str]) -> list[dict[str, object]]:
    result = []
    for row in value if isinstance(value, list) else []:
        if isinstance(row, dict):
            result.append({key: item for key, item in row.items() if key in allowed and ((isinstance(item, str) and len(item) <= 160 and (_SAFE.fullmatch(item) or key == "schema_version" and _SCHEMA.fullmatch(item))) or (isinstance(item, int) and not isinstance(item, bool) and item >= 0))})
    return result


def _ledger_identity(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict) or set(value) - {"schema_version", "id", "revision", "scope"}:
        return None
    if not _SCHEMA.fullmatch(str(value.get("schema_version", ""))) or not _SAFE.fullmatch(str(value.get("id", ""))):
        return None
    if not isinstance(value.get("revision"), int) or isinstance(value["revision"], bool) or value["revision"] < 1:
        return None
    identity: dict[str, object] = {"schema_version": value["schema_version"], "id": value["id"], "revision": value["revision"]}
    if "scope" in value:
        try:
            identity["scope"] = canonical_memory_scope(value["scope"])
        except ValueError:
            return None
    return identity


def _identity_errors(value: object, label: str) -> list[str]:
    if not isinstance(value, dict) or set(value) - {"schema_version", "id", "id_key", "revision", "scope"}:
        return [f"invalid_{label}"]
    if not _SCHEMA.fullmatch(str(value.get("schema_version", ""))) or not _SAFE.fullmatch(str(value.get("id", ""))) or "id_key" in value and not _SAFE.fullmatch(str(value["id_key"])):
        return [f"invalid_{label}"]
    if not isinstance(value.get("revision"), int) or isinstance(value["revision"], bool) or value["revision"] < 1:
        return [f"invalid_{label}"]
    try:
        canonical_memory_scope(value.get("scope"))
    except ValueError:
        return [f"invalid_{label}"]
    return []


def _reactivation_error(reason: str) -> dict[str, object]:
    return {"schema_version": "memory_reactivation/v1", "applied": False, "reason_code": reason, "artifact_identity": {}}


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
