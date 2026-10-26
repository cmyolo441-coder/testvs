"""Read-only parsers for bounded rejected-decision evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

from ..local_store import read_json_object_result
from ..paths import OmhPaths
from ..plugin_bundle.omh.memory_governance import (
    PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION,
    RETENTION_CLASSES,
    SOURCE_CLASSES,
    canonical_payload_digest,
)
from ..system.metadata_safety import is_sensitive_metadata_text, redact_metadata_text
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


ALLOWED_SCOPE_KINDS = frozenset({"project", "target", "thread", "run"})
SAFE_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s\"'])(?:/|[A-Za-z]:[\\/])(?:[^\s\"']+)")
_CONTENT_HASH_RE = re.compile(r"\b(?:sha(?:1|224|256|384|512)[:=]?)?[a-f0-9]{40,128}\b", re.IGNORECASE)


@dataclass(frozen=True)
class RejectedDecisionEvidence:
    candidate_id: str
    decision_revision: int
    record_type: str
    summary: str
    rejection_reason: str
    scope_kind: str
    scope_ref: str
    tags: tuple[str, ...]
    reviewed_at: str
    source_class: str
    retention: dict[str, object]
    revalidation: dict[str, object] | None
    superseded_by: object | None
    legacy: bool


def read_rejected_decision_evidence(paths: OmhPaths) -> tuple[tuple[RejectedDecisionEvidence, str], ...]:
    reviews_dir, candidates_dir = _managed_dir(paths, "reviews"), _managed_dir(paths, "candidates")
    values: list[tuple[RejectedDecisionEvidence, str]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in _json_files(reviews_dir):
        raw, _error = read_json_object_result(path)
        decision = _parse_v2(raw)
        if decision is not None:
            values.append((decision, metadata_ref(raw.get("review_id")) or path.stem))
            seen.add((decision.scope_ref, decision.candidate_id, decision.decision_revision))
    for path in _json_files(candidates_dir):
        raw, _error = read_json_object_result(path)
        decision = _parse_v1(raw)
        if decision is not None and (decision.scope_ref, decision.candidate_id, decision.decision_revision) not in seen:
            values.append((decision, f"legacy-{decision.candidate_id}"))
    return tuple(values)


def _parse_v2(raw: object) -> RejectedDecisionEvidence | None:
    if not isinstance(raw, dict) or raw.get("schema_version") != PROJECT_MEMORY_REVIEW_RECORD_SCHEMA_VERSION:
        return None
    if raw.get("decision", raw.get("decision_state")) != "rejected" or not metadata_ref(raw.get("review_id")):
        return None
    snapshot = _snapshot(raw)
    if not isinstance(snapshot, dict) or not _bound_to_immutable_review(raw, snapshot):
        return None
    return _evidence(raw, snapshot, legacy=False)


def _parse_v1(raw: object) -> RejectedDecisionEvidence | None:
    if (
        not isinstance(raw, dict)
        or raw.get("schema_version") not in {None, "project_memory_candidate/v1"}
        or raw.get("status") != "rejected"
        or not metadata_ref(raw.get("candidate_id"))
    ):
        return None
    snapshot: dict[str, object] = {
        "record_type": raw.get("record_type"), "summary": raw.get("summary"), "scope": raw.get("scope"),
        "tags": raw.get("tags"), "source_class": "omh_local", "retention": _legacy_retention(raw.get("ttl")),
        "revalidation": _legacy_revalidation(raw.get("staleness")),
        "superseded_by": raw.get("superseded_by") or raw.get("corrected_by"),
    }
    return _evidence(raw, snapshot, legacy=True)


def _snapshot(raw: dict[object, object]) -> dict[str, object] | None:
    for key in ("artifact_snapshot", "artifact", "artifact_metadata"):
        if isinstance(raw.get(key), dict):
            return {str(name): value for name, value in raw[key].items()}
    identity = raw.get("artifact_identity")
    if not isinstance(identity, dict):
        return None
    schema, identifier, id_key = identity.get("schema_version"), identity.get("id"), identity.get("id_key")
    if not isinstance(schema, str) or not metadata_ref(identifier) or id_key not in {"record_id", "candidate_id", "item_id", "block_id"}:
        return None
    return {
        "schema_version": schema, str(id_key): identifier, "revision": identity.get("revision"),
        "record_type": raw.get("record_type"), "summary": raw.get("summary", raw.get("decision_summary")),
        "scope": raw.get("scope", identity.get("scope")), "tags": raw.get("tags"),
        "source_class": raw.get("source_class"), "retention": raw.get("retention"),
        "revalidation": raw.get("revalidation"), "superseded_by": raw.get("superseded_by", raw.get("corrected_by")),
    }


def _bound_to_immutable_review(raw: dict[object, object], snapshot: dict[str, object]) -> bool:
    identity, subject, scope, digest = raw.get("artifact_identity"), _identifier(snapshot), _scope(snapshot.get("scope")), raw.get("payload_digest")
    revision = snapshot.get("revision")
    return (
        isinstance(identity, dict) and subject is not None and scope is not None and isinstance(revision, int)
        and not isinstance(revision, bool) and revision > 0 and identity.get("schema_version") == snapshot.get("schema_version")
        and identity.get("id") == subject and identity.get("revision") == revision
        and identity.get("scope") == {"kind": scope[0], "ref": scope[1]}
        and isinstance(digest, str) and digest == canonical_payload_digest(snapshot)
    )


def _evidence(raw: dict[object, object], snapshot: dict[str, object], *, legacy: bool) -> RejectedDecisionEvidence | None:
    candidate_id = metadata_ref(raw.get("candidate_id")) if legacy else _identifier(snapshot)
    scope, retention, source = _scope(snapshot.get("scope")), snapshot.get("retention"), metadata_ref(snapshot.get("source_class"))
    revision = raw.get("decision_revision", snapshot.get("revision", 1))
    reviewed_at = safe_string(raw.get("reviewed_at", raw.get("decision_at", raw.get("created_at", ""))))
    record_type, summary = metadata_ref(snapshot.get("record_type")), metadata_text(snapshot.get("summary"), limit=500)
    if (
        not candidate_id or scope is None or not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0
        or _timestamp(reviewed_at) is None or not record_type or not summary or not isinstance(retention, dict)
        or retention.get("class") not in RETENTION_CLASSES or source not in SOURCE_CLASSES
    ):
        return None
    revalidation = snapshot.get("revalidation") if isinstance(snapshot.get("revalidation"), dict) else None
    superseded = snapshot.get("superseded_by") or _correction_supersession(snapshot.get("corrected_by"), snapshot.get("correction"))
    return RejectedDecisionEvidence(
        candidate_id, revision, record_type, summary,
        metadata_text(raw.get("decision_reason", raw.get("rejection_reason", raw.get("reason", ""))), limit=300),
        scope[0], scope[1], normalized_tags(snapshot.get("tags")), reviewed_at, source,
        {str(key): value for key, value in retention.items()},
        {str(key): value for key, value in revalidation.items()} if revalidation is not None else None,
        superseded, legacy,
    )


def _managed_dir(paths: OmhPaths, name: str) -> Path:
    root = paths.memory_dir
    if root.is_symlink():
        raise ValueError("rejected-decision memory storage must not be a symlink")
    resolved_root = root.resolve(strict=False)
    if not resolved_root.is_relative_to(paths.omh_home.resolve(strict=False)):
        raise ValueError("rejected-decision memory storage must resolve under OMH home")
    directory = root / name
    if directory.is_symlink():
        raise ValueError(f"rejected-decision {name} storage must not be a symlink")
    if directory.resolve(strict=False).parent != resolved_root:
        raise ValueError(f"rejected-decision {name} storage must resolve under OMH memory")
    return directory


def _json_files(directory: Path) -> tuple[Path, ...]:
    if not directory.exists():
        return ()
    return tuple(path for path in sorted(directory.glob("*.json")) if not path.is_symlink() and path.is_file() and path.resolve().parent == directory.resolve())


def _legacy_retention(raw: object) -> dict[str, object]:
    ttl = raw if isinstance(raw, dict) else {}
    return {"class": "standard", **({"expires_at": ttl["expires_at"]} if isinstance(ttl.get("expires_at"), str) and ttl["expires_at"] else {})}


def _legacy_revalidation(raw: object) -> dict[str, object] | None:
    staleness = raw if isinstance(raw, dict) else {}
    return {"deadline": staleness["stale_after"]} if isinstance(staleness.get("stale_after"), str) and staleness["stale_after"] else None


def _correction_supersession(corrected_by: object, correction: object) -> object | None:
    if corrected_by is not None:
        return corrected_by
    return correction if isinstance(correction, dict) and (correction.get("superseded_by") or correction.get("status") in {"corrected", "superseded"}) else None


def _identifier(snapshot: dict[str, object]) -> str | None:
    return next((value for key in ("record_id", "candidate_id", "item_id", "block_id") if (value := metadata_ref(snapshot.get(key)))), None)


def _scope(raw: object) -> tuple[str, str] | None:
    if not isinstance(raw, dict):
        return None
    kind, ref = safe_string(raw.get("kind")), metadata_ref(raw.get("ref"))
    return (kind, ref) if kind in ALLOWED_SCOPE_KINDS and ref else None


def normalized_tags(values: object) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        return ()
    tags: list[str] = []
    for value in values:
        tag = str(value).strip().lower()
        if tag and SAFE_REF.fullmatch(tag) and not is_sensitive_metadata_text(tag) and tag not in tags:
            tags.append(tag)
    return tuple(tags[:12])


def metadata_text(value: object, *, limit: int) -> str:
    text = safe_string(value)
    return "[redacted]" if _ABSOLUTE_PATH_RE.search(text) or _CONTENT_HASH_RE.search(text) else redact_metadata_text(text, limit=limit)


def metadata_ref(value: object) -> str:
    text = safe_string(value)
    return text if SAFE_REF.fullmatch(text) and not is_sensitive_metadata_text(text) else ""


def safe_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp(value: str) -> datetime | None:
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
