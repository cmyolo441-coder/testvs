"""The OMH memory store on disk: layout, lock, atomic writes, readers, index.

Memory admission (``memory_admission``) writes candidates, records, review
decisions and the index through this module, and both of its callers -- the
``omh_memory`` tool in a Hermes session and the ``omh memory`` CLI through
``omh.workflows.memory`` -- reach the same files with the same bytes. Every
function takes an explicit ``omh_home``; the store is ``<omh_home>/memory``.

The lock sidecars are the names the CLI has always used (``.index.json.lock``,
``.capture.lock``) on the same OS primitive, so a CLI process and a plugin
session against one home still exclude each other. Stdlib only; no import of
the ``omh`` control plane (Hermes loads this directory with its own
interpreter).
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
from typing import Any, Callable, Iterator

from .awareness_delivery import _awareness_delivery_lock, _with_windows_retry
from .memory_governance import PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION, contains_credential_like_material
from .memory_principals import memory_identity_errors
from .memory_recall_support import (
    LEGACY_PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    MEMORY_SUBJECT_PATTERN,
    PROJECT_MEMORY_RECORD_SCHEMA_VERSION,
    _redact_admitted_text,
)

MEMORY_INDEX_SCHEMA_VERSION = "omh_memory_index/v1"
# The store lock's wait before it raises TimeoutError: the CLI's long-standing
# default, so moving the lock into the bundle did not shorten anyone's wait.
STORE_LOCK_TIMEOUT_SECONDS = 10.0
_SAFE_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_PROJECT_MEMORY_RECORD_KEYS = {
    "schema_version",
    "record_id",
    "candidate_id",
    "revision",
    "record_type",
    "summary",
    "scope",
    "tags",
    "source",
    "source_class",
    "source_ref",
    "source_evidence",
    "admission",
    "retention",
    "revalidation",
    "approved_at",
    "created_at",
    "updated_at",
    "ttl",
    "staleness",
    "safety",
    "derived_from",
    "perspective",
    "attention",
    "superseded_by",
    "redaction_policy",
    "claim_boundary",
    "identity",
    "subject",
}
_HANDOFF_CONTEXT_SCOPE_KEYS = {"kind", "ref"}
# Perspective is honcho's peer paradigm reinterpreted deterministically: an
# optional (observer, observed) pair naming whose view a record is and which
# actor it is about. Unscoped records behave exactly as before; a scoped
# record surfaces only through a matching lens, so an executor-specific
# lesson never leaks into another executor's handoff.
_PERSPECTIVE_KEYS = {"observer", "observed"}
UNREADABLE_RECORD_REASONS = ("unsupported_record_schema", "legacy_review_status_missing", "unreadable_file")

StoreLock = Callable[[Path], Any]
Clock = Callable[[], str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def memory_dir(omh_home: Path) -> Path:
    return Path(omh_home) / "memory"


def memory_index_path(omh_home: Path) -> Path:
    return memory_dir(omh_home) / "index.json"


def memory_capture_lock_path(omh_home: Path) -> Path:
    # The lock sidecar is `.<name>.lock`, so this is `memory/.capture.lock`.
    return memory_dir(omh_home) / "capture"


@contextmanager
def memory_store_lock(path: Path) -> Iterator[str]:
    """Exclusive lock on the ``.<name>.lock`` sidecar of ``path``.

    Not re-entrant: an flock is per open file description, so the capture
    lock and the index lock are different files and admission holds the
    first across a call that takes the second.
    """
    with _awareness_delivery_lock(path, timeout_seconds=STORE_LOCK_TIMEOUT_SECONDS) as mechanism:
        yield mechanism


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _with_windows_retry(lambda: path.chmod(0o700))


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace ``path`` whole with private permissions; the store's one write shape."""
    ensure_private_dir(path.parent)
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    created_tmp = False
    try:
        # newline="" keeps written bytes platform-stable ("\n" stays "\n").
        with tmp.open("x", encoding="utf-8", newline="") as handle:
            created_tmp = True
            handle.write(text)
        tmp.chmod(0o600)
        _with_windows_retry(lambda: tmp.replace(path))
        _with_windows_retry(lambda: path.chmod(0o600))
    except OSError:
        if created_tmp and tmp.exists() and not tmp.is_symlink():
            tmp.unlink()
        raise


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def read_json_object_result(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return read_json_object(path), None
    except (OSError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        return None, str(exc)


def memory_candidates_dir(omh_home: Path) -> Path:
    return memory_dir(omh_home) / "candidates"


def memory_records_dir(omh_home: Path) -> Path:
    return memory_dir(omh_home) / "records"


def memory_reviews_dir(omh_home: Path) -> Path:
    return memory_dir(omh_home) / "reviews"


def memory_candidate_path(omh_home: Path, candidate_id: str) -> Path:
    if not _SAFE_REF.match(candidate_id):
        raise ValueError(f"unsafe memory candidate id: {candidate_id!r}")
    path = memory_candidates_dir(omh_home) / f"{candidate_id}.json"
    assert_under_memory_root(omh_home, path)
    return path


def memory_record_path(omh_home: Path, record_id: str) -> Path:
    if not _SAFE_REF.match(record_id):
        raise ValueError(f"unsafe memory record id: {record_id!r}")
    path = memory_records_dir(omh_home) / f"{record_id}.json"
    assert_under_memory_root(omh_home, path)
    return path


def memory_review_path(omh_home: Path, review_id: str) -> Path:
    if not _SAFE_REF.match(review_id):
        raise ValueError(f"unsafe memory review id: {review_id!r}")
    path = memory_reviews_dir(omh_home) / f"{review_id}.json"
    assert_under_memory_root(omh_home, path)
    return path


def assert_under_memory_root(omh_home: Path, path: Path) -> None:
    root = memory_root(omh_home)
    candidate = path.resolve(strict=False)
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"memory write path escapes .omh/memory: {path}")


def memory_root(omh_home: Path) -> Path:
    return memory_dir(omh_home).resolve(strict=False)


def read_project_memory_candidate(omh_home: Path, candidate_id: str) -> dict[str, Any] | None:
    if not _SAFE_REF.match(candidate_id) or contains_credential_like_material(candidate_id):
        raise ValueError("unsafe memory candidate id")
    return read_json_object(memory_candidate_path(omh_home, candidate_id))


def read_project_memory_candidates(omh_home: Path) -> list[dict[str, Any]]:
    return read_memory_json_files(omh_home, memory_candidates_dir(omh_home))


def read_project_memory_records(omh_home: Path) -> list[dict[str, Any]]:
    return scan_project_memory_records(omh_home)[0]


def scan_project_memory_records(omh_home: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Approved records, and the record files this reader cannot answer for.

    Failing closed on an unrecognized record is right. Failing closed *silently*
    is what this exists to stop: a v1 record without an approved review status
    and any record from a newer schema were dropped here with no count, no
    exclusion entry, and nothing in `omh doctor`, so four files on disk read as
    two and the store simply got smaller. The forward case is the one that will
    happen -- the record schema has already moved once, and a shared `~/.omh`
    across two machines on different `omh` versions, a downgrade, or a partly
    finished migration all produce records this build cannot admit.

    The rest of the module already holds this line: the archive attention tier
    leaves the working context NAMED, never silently, and retirement reports a
    corrupt or malformed file per path with its reason. This is the same
    courtesy for the read every other surface goes through.
    """
    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    directory = memory_records_dir(omh_home)
    for record, path_name in read_memory_record_files(omh_home, directory):
        safe_path_name = _redact_admitted_text(path_name)
        if record is None:
            unreadable.append({"path_name": safe_path_name, "reason": "unreadable_file", "schema_version": ""})
            continue
        schema_version = str(record.get("schema_version", "") or "")
        safe_schema_version = _redact_admitted_text(schema_version)
        if schema_version == PROJECT_MEMORY_RECORD_SCHEMA_VERSION:
            records.append(record)
        elif schema_version == PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION:
            if validate_project_memory_record(record):
                unreadable.append({"path_name": safe_path_name, "reason": "unsupported_record_schema", "schema_version": safe_schema_version})
            else:
                records.append(record)
        elif schema_version == LEGACY_PROJECT_MEMORY_RECORD_SCHEMA_VERSION:
            if record.get("review_status") == "approved":
                # Legacy records stay review/status visible, but the evaluator
                # will fail them closed as review_required_legacy before replay.
                records.append(record)
            else:
                unreadable.append({"path_name": safe_path_name, "reason": "legacy_review_status_missing", "schema_version": safe_schema_version})
        else:
            unreadable.append({"path_name": safe_path_name, "reason": "unsupported_record_schema", "schema_version": safe_schema_version})
    return records, unreadable


def read_memory_record_files(omh_home: Path, directory: Path) -> list[tuple[dict[str, Any] | None, str]]:
    """Every record file with its name, `None` for the ones that would not parse.

    `read_memory_json_files` drops an unparseable file and keeps no trace of
    it, which is correct for a reader that only wants the good rows and wrong
    for one that has to say what it skipped.
    """
    if not directory.exists():
        return []
    items: list[tuple[dict[str, Any] | None, str]] = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        assert_under_memory_root(omh_home, path)
        data, _error = read_json_object_result(path)
        items.append((data if isinstance(data, dict) else None, path.name))
    return items


def read_project_memory_reviews(omh_home: Path) -> list[dict[str, Any]]:
    return read_memory_json_files(omh_home, memory_reviews_dir(omh_home))


def read_memory_json_files(omh_home: Path, directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        assert_under_memory_root(omh_home, path)
        # A corrupt store file must cost only itself, not the whole read: a
        # crash mid-write or disk fault used to make every recall, review,
        # and status call raise on the first unreadable file until someone
        # hand-deleted it. Retirement already scans this way.
        data, _error = read_json_object_result(path)
        if isinstance(data, dict):
            items.append(data)
    return items


def project_memory_review_resolver(omh_home: Path) -> dict[str, dict[str, object]]:
    return {
        str(review.get("review_id", "")): review
        for review in read_project_memory_reviews(omh_home)
        if str(review.get("review_id", ""))
    }


def recall_operation_states(omh_home: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    states: dict[str, str] = {}
    for record in records:
        operation_id = record.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id or operation_id in states:
            continue
        if not _SAFE_REF.fullmatch(operation_id):
            states[operation_id] = "invalid"
            continue
        operation, error = read_json_object_result(memory_dir(omh_home) / "operations" / f"{operation_id}.json")
        states[operation_id] = str(operation.get("state", "")) if not error and isinstance(operation, dict) else "unavailable"
    return states


def write_project_memory_candidate_unlocked(omh_home: Path, candidate: dict[str, object]) -> None:
    """Candidate write with NO index rewrite: for callers already holding the store lock."""
    path = memory_candidate_path(omh_home, str(candidate.get("candidate_id", "")))
    atomic_write_json(path, candidate)


def write_project_memory_record(omh_home: Path, record: dict[str, object]) -> None:
    errors = validate_project_memory_record(record)
    if errors:
        raise ValueError("; ".join(errors))
    atomic_write_json(memory_record_path(omh_home, str(record.get("record_id", ""))), record)


def write_project_memory_review_decision(omh_home: Path, review: dict[str, object]) -> dict[str, object]:
    review_id = str(review.get("review_id", ""))
    if not _SAFE_REF.match(review_id):
        raise ValueError(f"unsafe memory review id: {review_id!r}")
    atomic_write_json(memory_review_path(omh_home, review_id), review)
    return review


def write_memory_index(omh_home: Path, *, lock: StoreLock = memory_store_lock, clock: Clock = utc_now) -> None:
    ensure_private_dir(memory_dir(omh_home))
    with lock(memory_index_path(omh_home)):
        write_memory_index_unlocked(omh_home, updated_at=clock())


def write_memory_index_unlocked(omh_home: Path, *, updated_at: str) -> None:
    """Index rewrite for callers already inside the store lock.

    The lock flocks a fresh handle, so it is not reentrant: the retirement and
    approval transactions that already hold the lock must come through here,
    or they wait out the full timeout against themselves.
    """
    root = memory_dir(omh_home)
    ensure_private_dir(root)
    scopes = [path.relative_to(root).as_posix() for path in memory_scope_paths(omh_home)]
    candidates = [path.relative_to(root).as_posix() for path in safe_memory_files(omh_home, memory_candidates_dir(omh_home))]
    records = [path.relative_to(root).as_posix() for path in safe_memory_files(omh_home, memory_records_dir(omh_home))]
    reviews = [path.relative_to(root).as_posix() for path in safe_memory_files(omh_home, memory_reviews_dir(omh_home))]
    atomic_write_json(
        memory_index_path(omh_home),
        {
            "schema_version": MEMORY_INDEX_SCHEMA_VERSION,
            "updated_at": updated_at,
            "scope_files": sorted(scopes),
            "candidate_files": sorted(candidates),
            "record_files": sorted(records),
            "review_files": sorted(reviews),
            "claim_boundary": "OMH local memory only; this index is not Hermes internal memory.",
        },
    )


def safe_memory_files(omh_home: Path, directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    safe_paths: list[Path] = []
    for path in directory.glob("*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        assert_under_memory_root(omh_home, path)
        safe_paths.append(path)
    return sorted(safe_paths)


def memory_scope_paths(omh_home: Path) -> list[Path]:
    scopes_dir = memory_dir(omh_home) / "scopes"
    if not scopes_dir.exists():
        return []
    safe_paths: list[Path] = []
    for path in scopes_dir.rglob("*.json"):
        if path.is_symlink() or not path.is_file():
            continue
        assert_under_memory_root(omh_home, path)
        safe_paths.append(path)
    return sorted(safe_paths)


def validate_project_memory_record(value: Any, *, label: str = "project_memory_record") -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    _validate_allowed_keys(value, _PROJECT_MEMORY_RECORD_KEYS, errors, label)
    schema_version = value.get("schema_version")
    if schema_version not in {PROJECT_MEMORY_RECORD_SCHEMA_VERSION, PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION}:
        errors.append(f"{label}.schema_version is unsupported")
    if schema_version == PRINCIPAL_PROJECT_MEMORY_RECORD_SCHEMA_VERSION:
        errors.extend(f"{label}.{error}" for error in memory_identity_errors(value.get("identity")))
    if not isinstance(value.get("revision"), int) or int(value.get("revision", 0)) <= 0:
        errors.append(f"{label}.revision must be a positive integer")
    admission = value.get("admission")
    if not isinstance(admission, dict) or admission.get("state") not in {"approved_manual", "approved_auto_safe"}:
        errors.append(f"{label}.admission must carry an approved v2 decision")
    if not isinstance(value.get("retention"), dict):
        errors.append(f"{label}.retention must be an object")
    _validate_context_scope(value.get("scope"), errors, f"{label}.scope")
    if "perspective" in value:
        _validate_perspective(value.get("perspective"), errors, f"{label}.perspective", require_observed=True)
    if "subject" in value and not (isinstance(value["subject"], str) and MEMORY_SUBJECT_PATTERN.fullmatch(value["subject"])):
        errors.append(f"{label}.subject must match [a-z0-9][a-z0-9-]{{0,47}}")
    if value.get("redaction_policy") != "metadata_only":
        errors.append(f"{label}.redaction_policy must be metadata_only")
    if _contains_sensitive_text(value):
        errors.append(f"{label} contains sensitive-looking text")
    return errors


def _validate_allowed_keys(value: dict[str, Any], allowed: set[str], errors: list[str], label: str) -> None:
    extra_keys = sorted(set(value) - allowed)
    if extra_keys:
        errors.append(f"{label} has unsupported keys: {extra_keys}")


def _validate_context_scope(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _validate_allowed_keys(value, _HANDOFF_CONTEXT_SCOPE_KEYS, errors, label)
    kind = value.get("kind")
    ref = value.get("ref")
    if not isinstance(kind, str) or not kind:
        errors.append(f"{label}.kind must be a non-empty string")
    if not isinstance(ref, str) or not ref:
        errors.append(f"{label}.ref must be a non-empty string")


def _validate_perspective(value: Any, errors: list[str], label: str, *, require_observed: bool = False) -> None:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return
    _validate_allowed_keys(value, _PERSPECTIVE_KEYS, errors, label)
    for key in ("observer", "observed"):
        actor = value.get(key, "")
        if not isinstance(actor, str) or (actor and not _SAFE_REF.match(actor)):
            errors.append(f"{label}.{key} must be a safe actor label")
    if require_observed and not str(value.get("observed", "") or ""):
        errors.append(f"{label}.observed must name an actor")


def _contains_sensitive_text(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and contains_credential_like_material(key))
            or _contains_sensitive_text(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sensitive_text(item) for item in value)
    if isinstance(value, str):
        return contains_credential_like_material(value)
    return False
