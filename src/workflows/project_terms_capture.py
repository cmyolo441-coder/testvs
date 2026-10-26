from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path
import re
import stat

from ..paths import OmhPaths, find_project_root
from ..plugin_bundle.omh.project_identity import require_project_identity, resolve_project_identity
from ..system.binary_io import open_binary
from ..system.local_store import utc_now
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_CANDIDATE_SCHEMA_VERSION,
    REDUCTION_POLICY,
    normalize_confidence,
    normalize_identifier,
    normalize_mappings,
    normalize_provenance,
    normalize_scope,
    normalize_workflow_hints,
    stable_profile_id,
)
from . import domain_intelligence_operation_store as operation_journal
from . import domain_intelligence_store_security as store_security
from .domain_intelligence_store import (
    MAX_DOMAIN_CANDIDATE_FILES,
    bounded_json_paths,
    candidate_path,
    domain_store_lock,
    ensure_candidate_capacity,
    write_candidate,
)
from .domain_intelligence_validation import (
    current_profile_revision,
    validate_candidate_artifact,
)
from .project_terms import (
    MAX_PROJECT_TERMS_SOURCE_BYTES,
    ProjectTermsCaptureInput,
    build_project_terms_capture_inputs,
    parse_project_terms,
)


_CAPTURE_SCHEMA_VERSION = "project_terms_capture/v1"
_SOURCE_NAME = "PROJECT_TERMS.md"
_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)
_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)
_WINDOWS_PATH_CTIME_IS_BIRTHTIME = os.name == "nt"
_PROJECT_TERMS_SOURCE_REF = re.compile(r"^pt_sha256:([0-9a-f]{64})$")
_CAPTURE_OPERATION_SCHEMA_VERSION = "project_terms_candidate_batch_operation/v1"
_CAPTURE_OPERATION_PREFIX = "project_terms_"
_CAPTURE_OPERATION_KEYS = frozenset(
    {"schema_version", "operation_id", "candidates", "claim_boundary", "operation_digest"}
)


def capture_project_terms_file(
    paths: OmhPaths,
    *,
    from_file: str,
    stage: bool = False,
    invocation_cwd: str | Path | None = None,
) -> dict[str, object]:
    """Preview or atomically stage repository-root project terms."""
    root = _canonical_repository_root(invocation_cwd)
    source = _read_repository_project_terms(root, from_file)
    document = parse_project_terms(source)
    inputs = build_project_terms_capture_inputs(document)
    scope = normalize_scope("project", require_project_identity(root))
    domains = _preview_domains(paths, scope, inputs)
    available = _candidate_capacity_available(paths)
    required = len(inputs)
    payload: dict[str, object] = {
        "schema_version": _CAPTURE_SCHEMA_VERSION,
        "state": "prepared_not_observed",
        "reason": "preview_ready",
        "source_path": _SOURCE_NAME,
        "source_sha256": document.source_sha256,
        "scope": scope,
        "domains": domains,
        "capacity": {"available": available, "required": required},
        "mutation_set": [],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if not stage:
        return payload

    candidates = _stage_candidates(paths, scope, inputs)
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    payload.update(
        {
            "reason": "pending_review_staged",
            "domains": [
                {
                    **domain,
                    "base_profile_revision": candidate["base_profile_revision"],
                }
                for domain, candidate in zip(domains, candidates, strict=True)
            ],
            "candidate_ids": candidate_ids,
            "mutation_set": candidate_ids,
        }
    )
    return payload


def project_terms_source_freshness(
    artifact: dict[str, object],
    *,
    invocation_cwd: str | Path | None = None,
) -> dict[str, object]:
    """Derive closed, read-only source freshness for one candidate or profile."""
    root = _canonical_repository_root(invocation_cwd)
    provenance = artifact.get("provenance")
    scope = artifact.get("scope")
    source_ref = provenance.get("source_ref") if isinstance(provenance, dict) else None
    source_class = provenance.get("source_class") if isinstance(provenance, dict) else None
    digest_match = (
        _PROJECT_TERMS_SOURCE_REF.fullmatch(source_ref)
        if isinstance(source_ref, str)
        else None
    )
    tracked = (
        source_class == "omh_local"
        and digest_match is not None
        and isinstance(scope, dict)
        and scope.get("kind") == "project"
        and scope.get("ref") == resolve_project_identity(root).identity
        and bool(scope.get("ref"))
    )
    if not tracked:
        return _freshness_payload(
            state="untracked",
            current_sha256=None,
            candidate_sha256=None,
            reason="not_project_terms_source",
        )

    candidate_sha256 = digest_match.group(1)
    try:
        source = _read_repository_project_terms(root, _SOURCE_NAME)
    except ValueError as exc:
        if str(exc) != "project_terms_source_not_found":
            raise
        return _freshness_payload(
            state="missing",
            current_sha256=None,
            candidate_sha256=candidate_sha256,
            reason="source_file_missing",
        )
    current_sha256 = hashlib.sha256(source).hexdigest()
    if current_sha256 == candidate_sha256:
        return _freshness_payload(
            state="unchanged",
            current_sha256=current_sha256,
            candidate_sha256=candidate_sha256,
            reason="source_matches_candidate",
        )
    return _freshness_payload(
        state="changed",
        current_sha256=current_sha256,
        candidate_sha256=candidate_sha256,
        reason="source_digest_changed",
    )


def _freshness_payload(
    *,
    state: str,
    current_sha256: str | None,
    candidate_sha256: str | None,
    reason: str,
) -> dict[str, object]:
    return {
        "state": state,
        "checked_source_path": _SOURCE_NAME,
        "current_source_sha256": current_sha256,
        "candidate_source_sha256": candidate_sha256,
        "reason": reason,
    }


def _canonical_repository_root(invocation_cwd: str | Path | None) -> Path:
    root = find_project_root(invocation_cwd)
    if root is None:
        raise ValueError("project_terms_repository_root_required")
    canonical = root.resolve(strict=True)
    if canonical != root:
        raise ValueError("project_terms_repository_root_not_canonical")
    return canonical


def _read_repository_project_terms(root: Path, from_file: str) -> bytes:
    if from_file != _SOURCE_NAME:
        raise ValueError("project_terms_source_must_be_repository_root_PROJECT_TERMS.md")
    if _descriptor_relative_reads_supported():
        return _read_repository_project_terms_at(root)
    return _read_repository_project_terms_with_identity_checks(root)


def _descriptor_relative_reads_supported() -> bool:
    return bool(
        _NOFOLLOW_FLAG
        and _DIRECTORY_FLAG
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
    )


def _read_repository_project_terms_at(root: Path) -> bytes:
    root_before = _stat_without_symlinks(root)
    _validate_repository_root(root_before)
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG | _NOFOLLOW_FLAG,
        )
    except OSError as exc:
        raise _root_open_error(exc) from exc
    try:
        root_opened = os.fstat(root_fd)
        if not _same_identity(root_before, root_opened):
            raise ValueError("project_terms_repository_root_changed_while_reading")
        source_before = _stat_source_at(root_fd)
        _validate_source_entry(source_before)
        try:
            source_fd = open_binary(
                _SOURCE_NAME,
                os.O_RDONLY | _CLOEXEC_FLAG | _NONBLOCK_FLAG | _NOFOLLOW_FLAG,
                dir_fd=root_fd,
            )
        except OSError as exc:
            raise _source_open_error(exc) from exc
        try:
            source_opened = os.fstat(source_fd)
            if not _same_identity(source_before, source_opened):
                raise ValueError("project_terms_source_changed_while_reading")
            source = _read_project_terms_descriptor(source_fd, source_opened)
            if not _same_file_snapshot(source_opened, _stat_source_at_after_read(root_fd)):
                raise ValueError("project_terms_source_changed_while_reading")
        finally:
            os.close(source_fd)
        if not _same_identity(root_opened, _stat_without_symlinks(root)):
            raise ValueError("project_terms_repository_root_changed_while_reading")
        return source
    finally:
        os.close(root_fd)


def _read_repository_project_terms_with_identity_checks(root: Path) -> bytes:
    source_path = root / _SOURCE_NAME
    root_before = _stat_without_symlinks(root)
    _validate_repository_root(root_before)
    source_before = _stat_source_path(source_path)
    _validate_source_entry(source_before)
    try:
        source_fd = open_binary(
            source_path,
            os.O_RDONLY | _CLOEXEC_FLAG | _NONBLOCK_FLAG,
        )
    except OSError as exc:
        raise _source_open_error(exc) from exc
    try:
        source_opened = os.fstat(source_fd)
        if not _same_identity(source_before, source_opened):
            raise ValueError("project_terms_source_changed_while_reading")
        source = _read_project_terms_descriptor(source_fd, source_opened)
        if not _same_cross_api_file_snapshot(
            source_opened,
            _stat_source_path_after_read(source_path),
        ):
            raise ValueError("project_terms_source_changed_while_reading")
        if not _same_identity(root_before, _stat_without_symlinks(root)):
            raise ValueError("project_terms_repository_root_changed_while_reading")
        return source
    finally:
        os.close(source_fd)


def _read_project_terms_descriptor(
    source_fd: int,
    source_opened: os.stat_result,
) -> bytes:
    if not stat.S_ISREG(source_opened.st_mode):
        raise ValueError("project_terms_source_must_be_regular_file")
    if source_opened.st_size > MAX_PROJECT_TERMS_SOURCE_BYTES:
        raise ValueError("project_terms_source_too_large")
    chunks: list[bytes] = []
    remaining = MAX_PROJECT_TERMS_SOURCE_BYTES + 1
    while remaining:
        chunk = os.read(source_fd, min(64 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    source = b"".join(chunks)
    source_after = os.fstat(source_fd)
    if len(source) > MAX_PROJECT_TERMS_SOURCE_BYTES:
        raise ValueError("project_terms_source_too_large")
    if (
        not _same_file_snapshot(source_opened, source_after)
        or len(source) != source_after.st_size
    ):
        raise ValueError("project_terms_source_changed_while_reading")
    return source


def _stat_without_symlinks(path: Path) -> os.stat_result:
    try:
        return os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("project_terms_repository_root_changed_while_reading") from exc


def _stat_source_path(path: Path) -> os.stat_result:
    try:
        return os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("project_terms_source_not_found") from exc
    except OSError as exc:
        raise ValueError("project_terms_source_changed_while_reading") from exc


def _stat_source_at(root_fd: int) -> os.stat_result:
    try:
        return os.stat(_SOURCE_NAME, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ValueError("project_terms_source_not_found") from exc
    except OSError as exc:
        raise ValueError("project_terms_source_changed_while_reading") from exc


def _stat_source_path_after_read(path: Path) -> os.stat_result:
    try:
        return _stat_source_path(path)
    except ValueError as exc:
        raise ValueError("project_terms_source_changed_while_reading") from exc


def _stat_source_at_after_read(root_fd: int) -> os.stat_result:
    try:
        return _stat_source_at(root_fd)
    except ValueError as exc:
        raise ValueError("project_terms_source_changed_while_reading") from exc


def _validate_repository_root(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("project_terms_repository_root_must_not_be_symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("project_terms_repository_root_required")


def _validate_source_entry(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("project_terms_source_must_not_be_symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("project_terms_source_must_be_regular_file")


def _root_open_error(exc: OSError) -> ValueError:
    if exc.errno in {errno.ELOOP, errno.EMLINK}:
        return ValueError("project_terms_repository_root_must_not_be_symlink")
    return ValueError("project_terms_repository_root_changed_while_reading")


def _source_open_error(exc: OSError) -> ValueError:
    if isinstance(exc, FileNotFoundError):
        return ValueError("project_terms_source_not_found")
    if exc.errno in {errno.ELOOP, errno.EMLINK}:
        return ValueError("project_terms_source_must_not_be_symlink")
    return ValueError("project_terms_source_changed_while_reading")


def _identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return _identity(left) == _identity(right)


def _file_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        *_identity(metadata),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return _file_snapshot(left) == _file_snapshot(right)


def _same_cross_api_file_snapshot(
    left: os.stat_result,
    right: os.stat_result,
) -> bool:
    left_snapshot = _file_snapshot(left)
    right_snapshot = _file_snapshot(right)
    if _WINDOWS_PATH_CTIME_IS_BIRTHTIME:
        left_snapshot = (*left_snapshot[:-1], 0)
        right_snapshot = (*right_snapshot[:-1], 0)
    return left_snapshot == right_snapshot


def _preview_domains(
    paths: OmhPaths,
    scope: dict[str, object],
    inputs: tuple[ProjectTermsCaptureInput, ...],
) -> list[dict[str, object]]:
    store_root = paths.memory_dir / "domain-intelligence"
    store_is_readable_without_creation = all(
        (store_root / name).is_dir() for name in ("profiles", "reviews", "history")
    )
    domains: list[dict[str, object]] = []
    for capture_input in inputs:
        domain_id = normalize_identifier(capture_input.domain_id, "domain_id")
        profile_id = stable_profile_id(scope, domain_id)
        revision = (
            current_profile_revision(paths, profile_id)
            if store_is_readable_without_creation
            else 0
        )
        domains.append(
            {
                "domain_id": domain_id,
                "profile_id": profile_id,
                "vocabulary_mappings": normalize_mappings(list(capture_input.mappings)),
                "workflow_hints": normalize_workflow_hints(list(capture_input.workflow_hints)),
                "base_profile_revision": revision,
            }
        )
    return domains


def _candidate_capacity_available(paths: OmhPaths) -> int:
    directory = paths.memory_dir / "domain-intelligence" / "candidates"
    if not directory.exists():
        return MAX_DOMAIN_CANDIDATE_FILES
    existing, overflow = bounded_json_paths(
        directory,
        limit=MAX_DOMAIN_CANDIDATE_FILES,
    )
    if overflow:
        return 0
    return max(MAX_DOMAIN_CANDIDATE_FILES - len(existing), 0)


def _stage_candidates(
    paths: OmhPaths,
    scope: dict[str, object],
    inputs: tuple[ProjectTermsCaptureInput, ...],
) -> list[dict[str, object]]:
    # Validate the complete batch before acquiring the mutation lock.
    prepared = [_normalized_capture_input(capture_input) for capture_input in inputs]

    created_at = utc_now()
    with domain_store_lock(paths):
        _recover_project_terms_candidate_batches_locked(paths)
        ensure_candidate_capacity(paths, required=len(prepared))
        candidates: list[dict[str, object]] = []
        reserved_ids: set[str] = set()
        for capture_input in prepared:
            profile_id = stable_profile_id(scope, capture_input.domain_id)
            candidate_id = _new_candidate_id(paths, reserved_ids)
            reserved_ids.add(candidate_id)
            candidate = {
                "schema_version": DOMAIN_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "status": "pending_review",
                "profile_id": profile_id,
                "scope": scope,
                "domain_id": capture_input.domain_id,
                "vocabulary_mappings": normalize_mappings(list(capture_input.mappings)),
                "workflow_hints": normalize_workflow_hints(list(capture_input.workflow_hints)),
                "confidence": normalize_confidence(0.5, 1),
                "provenance": normalize_provenance(
                    capture_input.source_class,
                    capture_input.source_ref,
                    1,
                ),
                "base_profile_revision": current_profile_revision(paths, profile_id),
                "created_at": created_at,
                "updated_at": created_at,
                "redaction_policy": REDUCTION_POLICY,
                "claim_boundary": CLAIM_BOUNDARY,
            }
            validate_candidate_artifact(candidate)
            candidates.append(candidate)
        _write_candidate_batch(paths, candidates)
        return candidates


def _normalized_capture_input(capture_input: ProjectTermsCaptureInput) -> ProjectTermsCaptureInput:
    mappings = normalize_mappings(list(capture_input.mappings))
    hints = normalize_workflow_hints(list(capture_input.workflow_hints))
    normalize_provenance(capture_input.source_class, capture_input.source_ref, 1)
    return ProjectTermsCaptureInput(
        domain_id=normalize_identifier(capture_input.domain_id, "domain_id"),
        mappings=tuple((str(item["phrase"]), str(item["canonical"])) for item in mappings),
        workflow_hints=tuple(hints),
        source_class="omh_local",
        source_ref=capture_input.source_ref,
    )


def _new_candidate_id(paths: OmhPaths, reserved: set[str]) -> str:
    for _attempt in range(16):
        candidate_id = "dicand_" + os.urandom(8).hex()
        if candidate_id in reserved:
            continue
        if not candidate_path(paths, candidate_id).exists():
            return candidate_id
    raise ValueError("candidate_id_generation_exhausted")


def _write_candidate_batch(paths: OmhPaths, candidates: list[dict[str, object]]) -> None:
    operation = _build_capture_operation(candidates)
    try:
        _write_capture_operation(paths, operation)
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            write_candidate(paths, candidate_id, candidate)
        _delete_capture_operation(paths, str(operation["operation_id"]))
    except BaseException as interruption:
        # Preserve the original interruption. If rollback is itself interrupted,
        # the durable operation remains for the next locked reader or writer.
        try:
            _recover_capture_operation(paths, operation)
        except BaseException as recovery_interruption:
            interruption.add_note(
                "candidate batch recovery interrupted: "
                f"{type(recovery_interruption).__name__}: {recovery_interruption}"
            )
        raise


def _build_capture_operation(
    candidates: list[dict[str, object]],
) -> dict[str, object]:
    if not candidates:
        raise ValueError("candidate_batch_empty")
    first_id = str(candidates[0]["candidate_id"])
    operation_id = f"{_CAPTURE_OPERATION_PREFIX}{first_id}"
    operation = operation_journal.seal_operation(
        {
            "schema_version": _CAPTURE_OPERATION_SCHEMA_VERSION,
            "operation_id": operation_id,
            "candidates": candidates,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    _validate_capture_operation(None, operation)
    return operation


def _write_capture_operation(
    paths: OmhPaths,
    operation: dict[str, object],
) -> None:
    operation_journal.write_operation(paths, operation, _validate_capture_operation)


def _delete_capture_operation(paths: OmhPaths, operation_id: str) -> None:
    operation_journal.delete_operation(paths, operation_id, _validate_capture_operation)


def _validate_capture_operation(
    paths: OmhPaths | None,
    operation: dict[str, object],
) -> None:
    if (
        set(operation) != _CAPTURE_OPERATION_KEYS
        or operation.get("schema_version") != _CAPTURE_OPERATION_SCHEMA_VERSION
    ):
        raise ValueError("candidate_batch_operation_schema_mismatch")
    operation_id = operation.get("operation_id")
    if not isinstance(operation_id, str) or not operation_id.startswith(
        _CAPTURE_OPERATION_PREFIX
    ):
        raise ValueError("candidate_batch_operation_identity_mismatch")
    operation_journal.validate_operation_envelope(
        operation,
        operation_id=operation_id,
    )
    candidates = operation.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_batch_operation_candidates")
    candidate_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate_batch_operation_candidates")
        validate_candidate_artifact(candidate)
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in candidate_ids:
            raise ValueError("candidate_batch_operation_duplicate_candidate")
        candidate_ids.add(candidate_id)
        if paths is not None:
            existing = store_security.read_bounded_json(
                candidate_path(paths, candidate_id)
            )
            if existing not in (None, candidate):
                raise ValueError("candidate_batch_recovery_state_conflict")
    if operation_id != f"{_CAPTURE_OPERATION_PREFIX}{candidates[0]['candidate_id']}":
        raise ValueError("candidate_batch_operation_identity_mismatch")


def _recover_project_terms_candidate_batches_locked(paths: OmhPaths) -> None:
    operations = store_security.secure_managed_dir(paths, "operations")
    records, overflow = bounded_json_paths(
        operations,
        limit=store_security.MAX_DOMAIN_ARTIFACT_FILES,
    )
    if overflow:
        raise ValueError("candidate_batch_operation_capacity_exceeded")
    for path in records:
        if not path.stem.startswith(_CAPTURE_OPERATION_PREFIX):
            continue
        operation = operation_journal.load_operation(
            paths,
            path.stem,
            _validate_capture_operation,
        )
        if operation is None:
            continue
        _recover_capture_operation(paths, operation)


def _recover_capture_operation(
    paths: OmhPaths,
    operation: dict[str, object],
) -> None:
    operation_id = str(operation["operation_id"])
    persisted = operation_journal.load_operation(
        paths,
        operation_id,
        _validate_capture_operation,
    )
    if persisted is None:
        return
    if persisted != operation:
        raise ValueError("candidate_batch_operation_state_conflict")
    _validate_capture_operation(paths, persisted)
    candidates = persisted["candidates"]
    assert isinstance(candidates, list)
    for candidate in reversed(candidates):
        assert isinstance(candidate, dict)
        _remove_candidate_for_recovery(
            paths,
            str(candidate["candidate_id"]),
            candidate,
        )
    _delete_capture_operation(paths, operation_id)


def _remove_candidate_for_recovery(
    paths: OmhPaths,
    candidate_id: str,
    candidate: dict[str, object],
) -> None:
    _remove_candidate_for_recovery_impl(paths, candidate_id, candidate)


def _remove_candidate_for_recovery_impl(
    paths: OmhPaths,
    candidate_id: str,
    candidate: dict[str, object],
) -> None:
    filename = f"{candidate_id}.json"
    with store_security.anchored_managed_directory(
        paths,
        "candidates",
    ) as directory_fd:
        existing = store_security.read_bounded_json_at(directory_fd, filename)
        if existing is None:
            return
        if existing != candidate:
            raise ValueError("candidate_batch_recovery_state_conflict")
    store_security.unlink_managed_json(
        paths,
        "candidates",
        filename,
        expected=existing,
    )
