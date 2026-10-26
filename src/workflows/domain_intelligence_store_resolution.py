from __future__ import annotations

from collections import Counter
from pathlib import Path

from .domain_intelligence_contracts import (
    DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
    SAFE_REF,
    SHA256,
    ensure_no_forbidden_keys,
)
from .domain_intelligence_store_security import (
    MAX_DOMAIN_DIAGNOSTICS,
    bounded_json_paths,
    read_bounded_json,
    secure_artifact_path,
)


def resolve_authoritative_artifact(
    directory: Path,
    target_id: str,
    identity_field: str,
    *,
    file_limit: int,
) -> tuple[dict[str, object] | None, str | None]:
    paths, overflow = bounded_json_paths(directory, limit=file_limit)
    if overflow:
        return None, "artifact_file_count_exceeded"
    conflict_reason = f"{identity_field.removesuffix('_id')}_identity_conflict"
    canonical_error: str | None = None
    matches: list[tuple[dict[str, object], Path]] = []
    for path in paths:
        try:
            secure_artifact_path(directory, path.name)
            data = read_bounded_json(path)
        except (OSError, ValueError) as exc:
            if path.stem == target_id:
                canonical_error = str(exc)
            continue
        if data is None:
            if path.stem == target_id:
                canonical_error = "malformed_json"
            continue
        if data.get(identity_field) == target_id:
            matches.append((data, path))
        elif path.stem == target_id:
            canonical_error = conflict_reason
    if len(matches) > 1:
        return None, conflict_reason
    if not matches:
        return None, canonical_error
    data, path = matches[0]
    if path.stem != target_id:
        return None, conflict_reason
    try:
        ensure_no_forbidden_keys(data)
    except ValueError as exc:
        return None, str(exc)
    return data, None


def read_identity_artifacts(
    directory: Path,
    diagnostics: list[dict[str, str]],
    identity_field: str,
    *,
    file_limit: int,
    capacity_limit: int | None = None,
) -> list[tuple[dict[str, object], Path]]:
    paths, overflow = bounded_json_paths(directory, limit=file_limit)
    if overflow:
        _append_diagnostic(diagnostics, directory, "artifact_file_count_exceeded")
        return []
    if capacity_limit is not None and len(paths) > capacity_limit:
        _append_diagnostic(diagnostics, directory, "artifact_file_count_exceeded")
    parsed: list[tuple[dict[str, object], Path, str, str]] = []
    for path in paths:
        try:
            secure_artifact_path(directory, path.name)
            data = read_bounded_json(path)
        except (OSError, ValueError) as exc:
            _append_diagnostic(diagnostics, path, str(exc))
            continue
        if data is None:
            _append_diagnostic(diagnostics, path, "malformed_json")
            continue
        embedded_id = data.get(identity_field)
        if not isinstance(embedded_id, str) or not SAFE_REF.match(embedded_id):
            _append_diagnostic(diagnostics, path, "artifact_identity_mismatch")
            continue
        invalid_reason = _forbidden_key_reason(data)
        parsed.append((data, path, embedded_id, invalid_reason))
    duplicate_ids = {
        value
        for value, count in Counter(item[2] for item in parsed).items()
        if count > 1
    }
    records: list[tuple[dict[str, object], Path]] = []
    for data, path, embedded_id, invalid_reason in parsed:
        conflicting = embedded_id in duplicate_ids
        if conflicting:
            _append_diagnostic(diagnostics, path, "duplicate_embedded_id")
        precedence_reason = _storage_precedence_reason(data, identity_field)
        if precedence_reason:
            _append_diagnostic(diagnostics, path, precedence_reason)
            continue
        if invalid_reason:
            _append_diagnostic(diagnostics, path, invalid_reason)
            continue
        if conflicting:
            continue
        if path.stem != embedded_id:
            _append_diagnostic(diagnostics, path, "artifact_identity_mismatch")
            continue
        if len(records) < file_limit:
            records.append((data, path))
    return records


def read_history_artifacts(
    directory: Path,
    diagnostics: list[dict[str, str]],
    *,
    file_limit: int,
) -> list[tuple[dict[str, object], Path]]:
    paths, overflow = bounded_json_paths(directory, limit=file_limit)
    if overflow:
        _append_diagnostic(diagnostics, directory, "artifact_file_count_exceeded")
        return []
    parsed: list[tuple[dict[str, object], Path, tuple[str, int], str]] = []
    for path in paths:
        try:
            secure_artifact_path(directory, path.name)
            data = read_bounded_json(path)
        except (OSError, ValueError) as exc:
            _append_diagnostic(diagnostics, path, str(exc))
            continue
        if data is None:
            _append_diagnostic(diagnostics, path, "malformed_json")
            continue
        profile_id = data.get("profile_id")
        revision = data.get("revision")
        valid_revision = (
            isinstance(revision, int)
            and not isinstance(revision, bool)
            and revision > 0
        )
        if (
            not isinstance(profile_id, str)
            or not SAFE_REF.match(profile_id)
            or not valid_revision
        ):
            _append_diagnostic(diagnostics, path, "artifact_identity_mismatch")
            continue
        parsed.append((data, path, (profile_id, revision), _forbidden_key_reason(data)))
    duplicate_ids = {
        value
        for value, count in Counter(item[2] for item in parsed).items()
        if count > 1
    }
    records: list[tuple[dict[str, object], Path]] = []
    for data, path, identity, invalid_reason in parsed:
        if identity in duplicate_ids:
            _append_diagnostic(diagnostics, path, "duplicate_embedded_id")
            continue
        if invalid_reason:
            _append_diagnostic(diagnostics, path, invalid_reason)
            continue
        profile_id, revision = identity
        if path.stem != f"{profile_id}_r{revision}":
            _append_diagnostic(diagnostics, path, "artifact_identity_mismatch")
            continue
        if len(records) < file_limit:
            records.append((data, path))
    return records


def diagnostic(path: Path, reason: str) -> dict[str, str]:
    return {"path_name": path.name, "reason": reason}


def _append_diagnostic(
    diagnostics: list[dict[str, str]], path: Path, reason: str
) -> None:
    if len(diagnostics) < MAX_DOMAIN_DIAGNOSTICS:
        diagnostics.append(diagnostic(path, reason))


def _forbidden_key_reason(data: dict[str, object]) -> str:
    try:
        ensure_no_forbidden_keys(data)
    except ValueError as exc:
        return str(exc)
    return ""


def _storage_precedence_reason(data: dict[str, object], identity_field: str) -> str:
    if identity_field != "review_id":
        return ""
    schema = data.get("schema_version")
    if schema is not None and schema != DOMAIN_REVIEW_RECORD_SCHEMA_VERSION:
        return "unsupported_review_schema"
    digest = data.get("payload_digest")
    if digest is not None and (not isinstance(digest, str) or not SHA256.match(digest)):
        return "invalid_review_digest"
    return ""
