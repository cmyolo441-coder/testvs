"""Read-only, metadata-only discovery for Phase 1 memory migration."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from ..plugin_bundle.omh.memory_dreaming import profile_states_dir
from ..plugin_bundle.omh.memory_governance import canonical_memory_scope
from ..system.paths import OmhPaths

MAX_INVENTORY_ITEMS = 200
MAX_OMISSIONS = 50
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_SCHEMA = re.compile(r"^[A-Za-z0-9_.-]{1,120}/v[12]$")
_ID_KEYS = {
    "record": "record_id", "scope_item": "item_id", "block": "block_id",
    "candidate": "candidate_id", "review": "review_id", "operation": "operation_id",
    "migration": "ledger_id", "block_link": "block_id",
}
_BUCKETS = (
    ("records", "active_records", "record"),
    ("archive", "archive_history", "record"),
    ("history", "archive_history", "record"),
    ("candidates", "candidates", "candidate"),
    ("reviews", "reviews", "review"),
    ("block_candidates", "candidates", "candidate"),
    ("block_reviews", "reviews", "review"),
    ("block_links", "block_links", "block_link"),
    ("migrations", "migration_ledgers", "migration"),
)


def build_inventory(paths: OmhPaths) -> dict[str, object]:
    """Scan every local migration surface without exposing persisted content."""
    rows: list[dict[str, object]] = []
    for directory, bucket, kind in _BUCKETS:
        for path, problem in _json_files(paths.memory_dir / directory):
            _append_json(rows, bucket, kind, path, problem)
    for tier in ("system", "reference"):
        bucket = f"{tier}_blocks"
        for path, problem in _json_files(paths.memory_dir / "blocks" / tier):
            _append_json(rows, bucket, "block", path, problem)
    for path, problem in _json_files(paths.memory_dir / "scopes"):
        _append_scope(rows, path, problem)
    for path, problem in _json_files(paths.memory_dir, direct=True):
        if path.name == "index.json" or path.name.endswith("index.json"):
            _append_json(rows, "indexes", "index", path, problem)
    _append_journal(rows, paths.memory_dir / "archive" / "retirements.jsonl", "archive_retirements_journal")
    _append_journal(rows, paths.memory_dir / "write_journal.jsonl", "provider_write_journal")
    _append_journal(rows, paths.memory_dir / "consolidation.jsonl", "provider_consolidation_journal")
    # Each profile's latest brief, and a home-wide one an earlier release left.
    for path, problem in [*_json_files(profile_states_dir(paths.omh_home)), *_json_files(paths.memory_dir, direct=True)]:
        if path.name == "consolidation.json":
            _append_json(rows, "provider_consolidation_journal", "journal", path, problem)
    for path, problem in _json_files(paths.memory_operations_dir):
        doc = _document(path, problem)
        if problem or not isinstance(doc, dict) or doc.get("state") != "completed":
            rows.append(_row("incomplete_operations", "operation", doc, problem=problem or "incomplete_operation"))
    rows.sort(key=_sort_key)
    return _inventory(rows)


def _append_scope(rows: list[dict[str, object]], path: Path, problem: str) -> None:
    doc = _document(path, problem)
    if not isinstance(doc, dict):
        rows.append(_row("scope_items", "scope_item", doc, problem=problem or "corrupt_json"))
        return
    items = doc.get("items")
    if not isinstance(items, dict):
        rows.append(_row("scope_items", "scope_item", doc, problem="malformed_scope_items"))
        return
    for item_id in sorted(items, key=str):
        item = items[item_id]
        if not isinstance(item, dict):
            rows.append(_row("scope_items", "scope_item", None, problem="malformed_scope_item"))
            continue
        combined = {**item, "item_id": str(item.get("item_id") or item_id)}
        combined.setdefault("schema_version", doc.get("schema_version"))
        combined.setdefault("scope", doc.get("scope"))
        rows.append(_row("scope_items", "scope_item", combined))


def _append_journal(rows: list[dict[str, object]], path: Path, bucket: str) -> None:
    if path.is_symlink():
        rows.append(_row(bucket, "journal", None, problem="symlink_or_path_escape"))
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    except OSError:
        rows.append(_row(bucket, "journal", None, problem="corrupt_journal"))
        return
    for line in lines:
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            doc = None
        rows.append(_row(bucket, "journal", doc, problem="corrupt_journal" if not isinstance(doc, dict) else ""))


def _append_json(rows: list[dict[str, object]], bucket: str, kind: str, path: Path, problem: str) -> None:
    doc = _document(path, problem)
    rows.append(_row(bucket, kind, doc, problem=problem))
    if bucket.endswith("_blocks") and isinstance(doc, dict) and isinstance(doc.get("source_record_identity"), dict):
        rows.append(_row("block_links", "block_link", doc["source_record_identity"]))


def _document(path: Path, problem: str) -> dict[str, object] | None:
    if problem:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _json_files(root: Path, *, direct: bool = False) -> list[tuple[Path, str]]:
    if root.is_symlink():
        return [(root, "symlink_or_path_escape")]
    if not root.exists() or not root.is_dir():
        return []
    if direct:
        return [(path, "symlink_or_path_escape" if path.is_symlink() else "") for path in sorted(root.glob("*.json"))]
    found: list[tuple[Path, str]] = []
    for base, directories, names in os.walk(root):
        current = Path(base)
        for name in sorted(list(directories)):
            path = current / name
            if path.is_symlink():
                directories.remove(name)
                found.append((path, "symlink_or_path_escape"))
        for name in sorted(names):
            path = current / name
            if path.suffix == ".json":
                found.append((path, "symlink_or_path_escape" if path.is_symlink() else ""))
    return found


def _row(bucket: str, kind: str, doc: dict[str, object] | None, *, problem: str = "") -> dict[str, object]:
    schema = doc.get("schema_version") if isinstance(doc, dict) else None
    classification = _classification(schema, problem)
    identity, scope_state = _identity(doc, kind, schema)
    row: dict[str, object] = {
        "source_bucket": bucket,
        "artifact_kind": kind,
        "classification": classification,
        "replay_status": _replay_status(kind, classification),
        "required_action": _required_action(kind, classification),
    }
    if _safe_schema(schema) or _legacy_schema(schema):
        row["schema_version"] = schema
    if identity:
        row["artifact_identity"] = identity
    else:
        row["artifact_ref"] = "identity_unavailable"
    if scope_state == "invalid":
        row["scope_status"] = "noncanonical"
    if classification == "corrupt":
        row["omission_reason"] = problem or "unsupported_schema"
    return row


def _identity(doc: dict[str, object] | None, kind: str, schema: object) -> tuple[dict[str, object], str]:
    if not isinstance(doc, dict):
        return {}, "missing"
    key = _ID_KEYS.get(kind, "")
    value = doc.get(key)
    if not _safe(value):
        value = next((doc.get(name) for name in ("record_id", "item_id", "block_id", "candidate_id", "review_id") if _safe(doc.get(name))), None)
    if not _safe(value):
        return {}, "missing"
    revision = doc.get("revision", 1)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        return {}, "invalid"
    try:
        scope = canonical_memory_scope(doc["scope"]) if "scope" in doc else None
    except ValueError:
        return {}, "invalid"
    identity: dict[str, object] = {"schema_version": str(schema) if _safe_schema(schema) or _legacy_schema(schema) else "unknown", "id": value, "revision": revision}
    if scope is not None:
        identity["scope"] = scope
    return identity, "valid"


def _classification(schema: object, problem: str) -> str:
    if problem or not isinstance(schema, str):
        return "corrupt"
    if _safe_schema(schema):
        return "v2" if schema.endswith("/v2") else "v1"
    return "legacy" if _legacy_schema(schema) else "corrupt"


def _replay_status(kind: str, classification: str) -> str:
    if kind not in {"record", "scope_item", "block"}:
        return "not_prompt_input"
    return {"v2": "current_policy_rescan_required", "corrupt": "quarantined"}.get(classification, "review_required_legacy")


def _required_action(kind: str, classification: str) -> str:
    if classification == "corrupt":
        return "quarantine_review"
    if kind in {"record", "scope_item", "block"} and classification in {"v1", "legacy"}:
        return "reactivate_per_artifact"
    return "review_only"


def _inventory(rows: list[dict[str, object]]) -> dict[str, object]:
    classifications = Counter(str(row["classification"]) for row in rows)
    sources = Counter(str(row["source_bucket"]) for row in rows)
    quarantines = [
        {"action": "quarantine_proposal", "reason_code": str(row.get("omission_reason", "corrupt_artifact")), "artifact": _artifact_ref(row)}
        for row in rows if row["classification"] == "corrupt"
    ]
    kept = rows[:MAX_INVENTORY_ITEMS]
    omissions = [] if len(rows) <= MAX_INVENTORY_ITEMS else [{"reason_code": "inventory_item_limit", "omitted_count": len(rows) - len(kept)}]
    return {
        "schema_version": "memory_migration_inventory/v1",
        "dry_run": True,
        "artifacts": kept,
        "counts": {"total": len(rows), "classification": dict(sorted(classifications.items())), "source_bucket": dict(sorted(sources.items()))},
        "omissions": omissions[:MAX_OMISSIONS],
        "quarantine_proposals": quarantines[:MAX_OMISSIONS],
        "external_exclusions": [{"source_class": "external", "reason_code": "outside_omh_local_store"}],
        "redaction_policy": "metadata_only",
        "claim_boundary": "Inventory reads OMH-local metadata only; it does not migrate, quarantine, approve, or render memory.",
    }


def _artifact_ref(row: dict[str, object]) -> dict[str, object]:
    identity = row.get("artifact_identity")
    return dict(identity) if isinstance(identity, dict) else {"artifact_ref": "identity_unavailable"}


def _sort_key(row: dict[str, object]) -> tuple[str, str, str]:
    identity = row.get("artifact_identity")
    identifier = str(identity.get("id", "")) if isinstance(identity, dict) else ""
    return str(row["source_bucket"]), str(row["artifact_kind"]), identifier


def _safe(value: object) -> bool:
    return isinstance(value, str) and bool(_SAFE.fullmatch(value))


def _safe_schema(value: object) -> bool:
    return isinstance(value, str) and bool(_SCHEMA.fullmatch(value))


def _legacy_schema(value: object) -> bool:
    return isinstance(value, str) and bool(_SAFE.fullmatch(value)) and value.startswith("legacy_")
