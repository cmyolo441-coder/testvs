"""Validation and payload construction for one legacy-memory reactivation."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..plugin_bundle.omh.memory_governance import (
    MEMORY_CLASSIFIER_VERSION,
    MEMORY_GOVERNANCE_POLICY_VERSION,
    build_retention,
    canonical_memory_scope,
    canonical_payload_digest,
    classify_memory_admission,
    stable_artifact_identity,
)
from ..system.local_store import read_json_object_result
from ..system.paths import OmhPaths
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_TYPES = {"fact", "decision", "lesson", "procedure", "episode"}
_APPROVALS = {"approved_manual", "approved_auto_safe"}


def find_active(paths: OmhPaths, artifact_id: str, kind: str | None = None) -> tuple[dict[str, Any] | None, bool]:
    """Return one active artifact plus whether a matching surface is unsafe."""
    unsafe = False
    for folder, item_kind in (("records", "record"), ("blocks", "block"), ("scopes", "scope_item")):
        if kind and kind != item_kind:
            continue
        for path, bad in _files(paths.memory_dir / folder):
            unsafe = unsafe or bad
            if bad:
                continue
            doc, error = read_json_object_result(path)
            if error or not isinstance(doc, dict):
                continue
            found = _match(path, item_kind, doc, artifact_id)
            if found:
                return found, unsafe
    return None, unsafe


def find_operation(paths: OmhPaths, artifact_id: str) -> dict[str, Any] | None:
    for path, bad in _files(paths.memory_operations_dir):
        if bad:
            continue
        doc, error = read_json_object_result(path)
        steps = doc.get("steps") if isinstance(doc, dict) else None
        if not error and doc.get("operation_type") == "memory_reactivation" and isinstance(steps, list) and any(step.get("scope") == artifact_id for step in steps if isinstance(step, dict)):
            return doc
    return None


def operation_steps(old_id: str, new_id: str, review_id: str, prior_review_id: str, target: str, review_target: str) -> list[dict[str, str]]:
    return [
        {"name": "replace_artifact", "action": "reactivate", "source": review_id, "target": target, "scope": old_id, "key": new_id, "revision": prior_review_id},
        {"name": "write_immutable_review", "action": "reactivate", "source": target, "target": review_target, "scope": prior_review_id, "key": review_id, "revision": new_id},
    ]


def target_artifact(paths: OmhPaths, target: str, old_id: str, new_id: str) -> dict[str, Any] | None:
    path = safe_target(paths, target)
    doc, error = read_json_object_result(path)
    if error or not isinstance(doc, dict):
        return None
    kind = "scope_item" if target.startswith("scopes/") else "block" if target.startswith("blocks/") else "record"
    return _match(path, kind, doc, new_id) or _match(path, kind, doc, old_id)


def safe_target(paths: OmhPaths, relative: str) -> Path:
    if paths.memory_dir.is_symlink() or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or not relative.endswith(".json"):
        raise ValueError("symlink_or_path_escape")
    root = paths.memory_dir.resolve(strict=False)
    path = paths.memory_dir / relative
    resolved = path.resolve(strict=False)
    if path.is_symlink() or (root != resolved and root not in resolved.parents):
        raise ValueError("symlink_or_path_escape")
    return path


def matching_review(paths: OmhPaths, review_id: str, source: dict[str, Any]) -> dict[str, Any] | None:
    expected = legacy_identity(source)
    digest = canonical_payload_digest(source["artifact"])
    for directory in (paths.memory_dir / "reviews", paths.memory_dir / "block_reviews"):
        path = directory / f"{review_id}.json"
        if path.is_symlink():
            continue
        review, error = read_json_object_result(path)
        if error or not isinstance(review, dict):
            continue
        if review.get("schema_version") == "project_memory_review_record/v2" and review.get("review_id") == review_id and review.get("decision") in _APPROVALS and review.get("artifact_identity") == expected and review.get("payload_digest") == digest:
            return review
    return None


def prepare(source: dict[str, Any], new_id: str, new_review_id: str, prior_review: dict[str, Any], now: datetime) -> tuple[dict[str, Any] | None, str]:
    artifact = source["artifact"]
    if not isinstance(artifact.get("schema_version"), str) or not artifact["schema_version"].endswith("/v1"):
        return None, "review_required_legacy"
    try:
        scope = canonical_memory_scope(artifact["scope"])
    except (KeyError, ValueError):
        return None, "scope_invalid"
    revision = artifact.get("revision", 1)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        return None, "revision_invalid"
    safety = classify_memory_admission(_renderable_text(artifact))
    if safety.get("status") != "safe":
        return None, "safety_rescan_required"
    retention, revalidation = _retention(artifact, now)
    if retention is None:
        return None, "retention_invalid"
    state = str(prior_review["decision"])
    admission = {
        "state": state, "review_id": new_review_id, "reviewer_claim": str(prior_review.get("reviewer_claim", "operator")),
        "admitted_at": _stamp(now), "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION,
        "classifier_version": MEMORY_CLASSIFIER_VERSION,
    }
    if source["kind"] == "record":
        result = _record(artifact, new_id, scope, retention, revalidation, admission, safety, now)
    elif source["kind"] == "block":
        result = _block(artifact, new_id, scope, retention, revalidation, admission, now)
    else:
        result = _scope(source, artifact, new_id, scope, retention, revalidation, admission, safety, now)
    if result is None:
        return None, "malformed_legacy_artifact"
    migration_artifact = result if source["kind"] != "scope_item" else result["items"][new_id]
    identity = stable_artifact_identity(migration_artifact)
    admission["artifact_identity"] = identity
    admission["payload_digest"] = canonical_payload_digest(migration_artifact)
    return result, "ready"


def legacy_identity(source: dict[str, Any]) -> dict[str, object]:
    artifact, kind = source["artifact"], source["kind"]
    id_key = {"record": "record_id", "block": "block_id", "scope_item": "item_id"}[kind]
    identifier = artifact.get(id_key, artifact.get("label", ""))
    return {"schema_version": artifact["schema_version"], "id": identifier, "id_key": id_key, "revision": artifact.get("revision", 1), "scope": canonical_memory_scope(artifact["scope"])}


def new_review(artifact: dict[str, Any], review_id: str, prior: dict[str, Any], now: datetime) -> dict[str, object]:
    return {
        "schema_version": "project_memory_review_record/v2", "review_id": review_id,
        "artifact_identity": stable_artifact_identity(artifact), "decision": artifact["admission"]["state"],
        "reviewer_claim": str(prior.get("reviewer_claim", "operator")), "payload_digest": canonical_payload_digest(artifact),
        "policy_version": MEMORY_GOVERNANCE_POLICY_VERSION, "classifier_version": MEMORY_CLASSIFIER_VERSION,
        "reviewed_at": _stamp(now), "decision_reason": "legacy_reactivation", "prior_review_id": prior["review_id"],
    }


def artifact_identity(source: dict[str, Any]) -> dict[str, object]:
    return stable_artifact_identity(source["artifact"])


def _match(path: Path, kind: str, doc: dict[str, Any], artifact_id: str) -> dict[str, Any] | None:
    if kind == "scope_item":
        items = doc.get("items")
        if not isinstance(items, dict):
            return None
        for key, item in items.items():
            if isinstance(item, dict) and (key == artifact_id or item.get("item_id") == artifact_id):
                artifact = {**item, "item_id": str(item.get("item_id") or key), "scope": item.get("scope", doc.get("scope")), "schema_version": item.get("schema_version", doc.get("schema_version"))}
                return {"kind": kind, "path": path, "relative": "", "document": doc, "artifact": artifact, "item_key": key}
        return None
    identifier = doc.get("record_id" if kind == "record" else "block_id")
    if identifier == artifact_id or (kind == "block" and doc.get("label") == artifact_id):
        return {"kind": kind, "path": path, "relative": "", "document": doc, "artifact": doc}
    return None


def _record(data: dict[str, Any], new_id: str, scope: dict[str, object], retention: dict[str, object], revalidation: dict[str, object], admission: dict[str, object], safety: dict[str, object], now: datetime) -> dict[str, object] | None:
    record_type = data.get("record_type", "fact")
    if record_type not in _TYPES or not isinstance(data.get("summary", ""), str):
        return None
    return {"schema_version": "project_memory_record/v2", "record_id": new_id, "candidate_id": "", "revision": 1, "record_type": record_type, "summary": data.get("summary", ""), "scope": scope, "tags": [], "source": "memory_migration", "source_class": "omh_local", "source_ref": "", "admission": admission, "retention": retention, "revalidation": revalidation, "approved_at": _stamp(now), "created_at": _stamp(now), "updated_at": _stamp(now), "ttl": {"ttl_days": retention.get("ttl_days"), "expires_at": str(retention.get("expires_at", ""))}, "staleness": {"stale_after": str(revalidation.get("deadline", "")), "stale_after_days": None}, "safety": safety, "redaction_policy": "metadata_only", "claim_boundary": "Reactivated OMH memory is prepared context only; it is not observed use."}


def _block(data: dict[str, Any], new_id: str, scope: dict[str, object], retention: dict[str, object], revalidation: dict[str, object], admission: dict[str, object], now: datetime) -> dict[str, object] | None:
    label, tier, limit = data.get("label"), data.get("tier"), data.get("limit")
    if not isinstance(label, str) or not _LABEL.fullmatch(label) or tier not in {"system", "reference"} or not isinstance(limit, int) or limit < 1 or not all(isinstance(data.get(key, ""), str) for key in ("description", "value")) or len(data["value"]) > limit:
        return None
    result: dict[str, object] = {"schema_version": "omh_memory_block/v2", "block_id": new_id, "revision": 1, "label": label, "description": data["description"], "value": data["value"], "limit": limit, "tier": tier, "scope": scope, "source_class": "omh_local", "admission": admission, "retention": retention}
    if revalidation:
        result["revalidation"] = revalidation
    return result


def _scope(source: dict[str, Any], data: dict[str, Any], new_id: str, scope: dict[str, object], retention: dict[str, object], revalidation: dict[str, object], admission: dict[str, object], safety: dict[str, object], now: datetime) -> dict[str, object] | None:
    if not isinstance(data.get("key", ""), str) or not isinstance(data.get("summary", data.get("value", "")), str) or not isinstance(data.get("value", ""), str):
        return None
    item = {"schema_version": "omh_memory_scope/v2", "item_id": new_id, "revision": 1, "key": data["key"], "summary": data.get("summary", data["value"]), "value": data["value"], "scope": scope, "source_class": "omh_local", "admission": admission, "retention": retention, "revalidation": revalidation, "safety": safety, "updated_at": _stamp(now)}
    document = dict(source["document"])
    items = dict(document.get("items", {}))
    items.pop(source["item_key"], None)
    items[new_id] = item
    document["items"] = items
    return document


def _retention(data: dict[str, Any], now: datetime) -> tuple[dict[str, object] | None, dict[str, object]]:
    retained = data.get("retention") if isinstance(data.get("retention"), dict) else {}
    retention_class = retained.get("class", "standard")
    ttl_source = retained if "ttl_days" in retained else data.get("ttl", {})
    ttl_days = ttl_source.get("ttl_days") if isinstance(ttl_source, dict) else None
    if ttl_days is not None and (not isinstance(ttl_days, int) or isinstance(ttl_days, bool) or ttl_days < 1):
        return None, {}
    record_type = data.get("record_type", "fact") if data.get("record_type", "fact") in _TYPES else "fact"
    if retention_class not in {"volatile", "standard", "durable"} or (retention_class == "durable" and ttl_days is not None):
        return None, {}
    try:
        retention = build_retention(retention_class, record_type=record_type, admitted_at=now, ttl_days=ttl_days)
    except ValueError:
        return None, {}
    stale = data.get("revalidation") if isinstance(data.get("revalidation"), dict) else data.get("staleness", {})
    deadline = stale.get("deadline", stale.get("stale_after", "")) if isinstance(stale, dict) else ""
    parsed = _time(deadline) if deadline else None
    if deadline and parsed is None:
        return None, {}
    return retention, {"deadline": _stamp(parsed)} if parsed else {}


def _renderable_text(data: dict[str, Any]) -> str:
    return "\n".join(str(data.get(key, "")) for key in ("summary", "value", "description", "label") if isinstance(data.get(key, ""), str))


def _files(root: Path) -> list[tuple[Path, bool]]:
    if root.is_symlink():
        return [(root, True)]
    if not root.exists() or not root.is_dir():
        return []
    found: list[tuple[Path, bool]] = []
    for base, directories, names in os.walk(root):
        current = Path(base)
        for name in list(directories):
            path = current / name
            if path.is_symlink():
                directories.remove(name)
                found.append((path, True))
        found.extend((current / name, (current / name).is_symlink()) for name in names if name.endswith(".json"))
    return sorted(found)


def _time(value: object) -> datetime | None:
    try:
        result = strict_fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
