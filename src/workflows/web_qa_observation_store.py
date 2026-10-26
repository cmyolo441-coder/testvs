"""Managed project-local persistence for host-owned web-QA observations.

The store only imports already-sanitized host receipts and local image bytes.  It
never starts a browser, makes a network request, or treats persisted verdicts
as authoritative: every read re-admits the stored plan and receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
import time
from typing import NoReturn

from ..system.local_store import file_lock
from .browser_workflow_learning import BrowserTraceError
from .browser_workflow_learning_store import _root as observed_git_root
from .web_qa_observation import TrustedTraceResolver, WebQaObservationError, build_web_qa_observation
from .web_qa_observation_plan import WebQaObservationPlanError, build_web_qa_observation_plan, parse_normalized_web_qa_observation_plan


WEB_QA_OBSERVATION_STORE_SCHEMA_VERSION = "web_qa_observation_store/v1"
MAX_STORED_METADATA_BYTES = 524_288
MAX_CAPTURE_FILE_BYTES = 25 * 1024 * 1024
_MAX_DEPTH = 24
_RUN_ID = re.compile(r"^web-qa-[a-f0-9]{24}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


class WebQaObservationStoreError(ValueError):
    """Raised for unsafe, malformed, or conflicting managed observation data."""


@dataclass(frozen=True, slots=True)
class ImportedWebQaObservation:
    plan: dict[str, object]
    receipt: dict[str, object]
    observation: dict[str, object]
    captures: list[dict[str, object]]
    timings: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan,
            "receipt": self.receipt,
            "observation": self.observation,
            "captures": self.captures,
            "timings": self.timings,
        }


def prepare_web_qa_observation(
    raw_plan: object,
    project_root: str | Path | None = None,
    *,
    trusted_trace_resolver: TrustedTraceResolver | None = None,
) -> dict[str, object]:
    """Normalize a plan and report whether its terminal identity is stored.

    This is deliberately a prepare-only operation: it creates neither managed
    directories nor files.
    """
    try:
        plan = build_web_qa_observation_plan(raw_plan)
    except WebQaObservationPlanError as exc:
        raise WebQaObservationStoreError(str(exc)) from exc
    root = _git_root(project_root)
    existing = _read_optional(root, str(plan["run_id"]), trusted_trace_resolver=trusted_trace_resolver)
    return {
        "plan": plan,
        "completion_state": "completed" if existing is not None else "not_found",
        "stored_observation": existing.observation if existing is not None else None,
    }


def import_web_qa_observation(
    project_root: str | Path | None,
    plan: object,
    receipt: object,
    capture_files: Mapping[str, str | Path],
    *,
    trusted_trace_resolver: TrustedTraceResolver | None = None,
) -> dict[str, object]:
    """Atomically import one canonical plan/receipt and its verified images.

    ``capture_files`` maps the screenshot SHA-256 named by receipt evidence to
    a host-owned local file. Existing completed identities are read and reused
    only when their canonical plan and receipt are byte-for-byte equivalent.
    """
    root = _git_root(project_root)
    normalized_plan = _normalize_plan(plan)
    run_id = str(normalized_plan["run_id"])
    _run_id(run_id)
    existing = _read_optional(root, run_id, trusted_trace_resolver=trusted_trace_resolver)
    if existing is not None:
        if _canonical(existing.plan) != _canonical(normalized_plan) or _canonical(existing.receipt) != _canonical(receipt):
            raise WebQaObservationStoreError("completed run identity conflicts with a different plan or receipt")
        return existing.as_dict()

    observations = _observations_dir(root)
    _ensure_real_directory(observations)
    lock_target = observations / f".{run_id}.import"
    _safe_child(observations, lock_target)
    lock_sidecar = lock_target.with_name(f".{lock_target.name}.lock")
    _safe_child(observations, lock_sidecar)
    if lock_sidecar.is_symlink():
        raise WebQaObservationStoreError("managed observation lock symlink is refused")
    with file_lock(lock_target, private=True) as lock:
        if lock["enforced"] is not True:
            raise WebQaObservationStoreError("managed observation import requires an enforced per-run file lock")
        existing = _read_optional(root, run_id, trusted_trace_resolver=trusted_trace_resolver)
        if existing is not None:
            if _canonical(existing.plan) != _canonical(normalized_plan) or _canonical(existing.receipt) != _canonical(receipt):
                raise WebQaObservationStoreError("completed run identity conflicts with a different plan or receipt")
            return existing.as_dict()

        validation_started = time.monotonic()
        observation = _admit(normalized_plan, receipt, trusted_trace_resolver)
        expected = _expected_captures(observation)
        supplied = _capture_mapping(capture_files)
        if set(supplied) != set(expected):
            raise WebQaObservationStoreError("capture files must name exactly the observed screenshot digests")
        captures, verified_bytes = _verify_captures(supplied, expected, normalized_plan)
        actual_capture_bytes = sum(len(data) for data in verified_bytes.values())
        reported_artifact_bytes = _execution_integer(receipt, "artifact_bytes")
        if actual_capture_bytes > reported_artifact_bytes:
            raise WebQaObservationStoreError("actual imported capture bytes exceed receipt artifact accounting")
        metadata = {
            "schema_version": WEB_QA_OBSERVATION_STORE_SCHEMA_VERSION,
            "project_identity": _project_identity(root),
            "plan": normalized_plan,
            "receipt": receipt,
            "observation": observation,
            "captures": captures,
            "timings": {
                "validation_ms": max(0, int((time.monotonic() - validation_started) * 1000)),
                "adapter_execution_ms": _adapter_execution_ms(receipt),
                "attempts": _execution_integer(receipt, "attempts"),
                "artifact_bytes": actual_capture_bytes,
                "reported_artifact_bytes": reported_artifact_bytes,
            },
        }
        _validate_metadata(metadata, root, trusted_trace_resolver)
        encoded_metadata = _encoded_metadata(metadata)
        _commit(root, run_id, encoded_metadata, verified_bytes, captures)
        return _record(_object(json.loads(encoded_metadata), "committed metadata")).as_dict()


def read_web_qa_observation(
    project_root: str | Path | None,
    run_id: str,
    *,
    trusted_trace_resolver: TrustedTraceResolver | None = None,
) -> dict[str, object]:
    root = _git_root(project_root)
    _run_id(run_id)
    result = _read_optional(root, run_id, trusted_trace_resolver=trusted_trace_resolver)
    if result is None:
        raise WebQaObservationStoreError("web-QA observation was not found")
    return result.as_dict()


def observation_envelope(
    project_root: str | Path | None,
    run_id: str,
    *,
    trusted_trace_resolver: TrustedTraceResolver | None = None,
) -> dict[str, object]:
    """Resolve stored evidence to the closed comparison input envelope."""
    stored = read_web_qa_observation(project_root, run_id, trusted_trace_resolver=trusted_trace_resolver)
    return {"plan": stored["plan"], "receipt": stored["receipt"]}


def _normalize_plan(plan: object) -> dict[str, object]:
    if type(plan) is not dict:
        raise WebQaObservationStoreError("plan must be an object")
    try:
        if "schema_version" in plan:
            return parse_normalized_web_qa_observation_plan(plan)
        return build_web_qa_observation_plan(plan)
    except WebQaObservationPlanError as exc:
        raise WebQaObservationStoreError(str(exc)) from exc


def _admit(plan: dict[str, object], receipt: object, resolver: TrustedTraceResolver | None) -> dict[str, object]:
    _bounded(receipt)
    _privacy_safe(receipt)
    try:
        observation = build_web_qa_observation(plan, receipt, trusted_trace_resolver=resolver)
    except (WebQaObservationError, TypeError, ValueError) as exc:
        raise WebQaObservationStoreError(f"receipt admission failed: {exc}") from exc
    if _execution_integer(receipt, "artifact_bytes") > _integer(_object(plan["limits"], "plan limits")["max_artifact_bytes"], "plan artifact cap"):
        raise WebQaObservationStoreError("receipt artifact bytes exceed the planned artifact cap")
    return observation


def _expected_captures(observation: dict[str, object]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for cell in _list(observation.get("cells"), "observation cells"):
        channels = _object(cell.get("channels"), "observation channels")
        screenshot = _object(channels.get("screenshot"), "screenshot channel")
        if screenshot.get("status") != "observed":
            continue
        evidence = _object(screenshot.get("evidence"), "screenshot evidence")
        digest = evidence.get("capture_sha256")
        size = evidence.get("byte_size")
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or type(size) is not int or size < 1:
            raise WebQaObservationStoreError("admitted screenshot evidence is invalid")
        previous = expected.setdefault(digest, size)
        if previous != size:
            raise WebQaObservationStoreError("one screenshot digest has conflicting byte sizes")
    return expected


def _capture_mapping(value: Mapping[str, str | Path]) -> dict[str, Path]:
    if not isinstance(value, Mapping):
        raise WebQaObservationStoreError("capture_files must be a digest-to-path mapping")
    result: dict[str, Path] = {}
    for digest, source in value.items():
        if not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or digest in result:
            raise WebQaObservationStoreError("capture_files contains an invalid digest")
        if not isinstance(source, (str, Path)):
            raise WebQaObservationStoreError("capture_files contains an invalid path")
        result[digest] = Path(source).expanduser()
    return result


def _verify_captures(
    supplied: dict[str, Path],
    expected: dict[str, int],
    plan: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    limit = _integer(_object(plan["limits"], "plan limits")["max_artifact_bytes"], "plan artifact cap")
    total = 0
    captures: list[dict[str, object]] = []
    verified: dict[str, bytes] = {}
    for digest, path in sorted(supplied.items()):
        data = _read_source_image(path)
        if len(data) != expected[digest] or hashlib.sha256(data).hexdigest() != digest:
            raise WebQaObservationStoreError("capture file bytes do not match receipt screenshot evidence")
        mime_type = _image_mime(data)
        if not mime_type:
            raise WebQaObservationStoreError("capture file must contain PNG, JPEG, or WebP image bytes")
        total += len(data)
        if total > limit:
            raise WebQaObservationStoreError("capture file bytes exceed the planned artifact cap")
        verified[digest] = data
        captures.append({"sha256": digest, "byte_size": len(data), "mime_type": mime_type, "path": f"captures/{digest}{_extension(mime_type)}"})
    return captures, verified


def _read_source_image(path: Path) -> bytes:
    if path.is_symlink():
        raise WebQaObservationStoreError("capture source symlink is refused")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        raise WebQaObservationStoreError("capture source must be an existing local regular file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 1 or info.st_size > MAX_CAPTURE_FILE_BYTES:
            raise WebQaObservationStoreError("capture source exceeds bounds or is not a regular file")
        chunks: list[bytes] = []
        remaining = MAX_CAPTURE_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(data) > MAX_CAPTURE_FILE_BYTES:
        raise WebQaObservationStoreError("capture source exceeds bounds")
    return data


def _commit(
    root: Path,
    run_id: str,
    encoded_metadata: bytes,
    verified_bytes: dict[str, bytes],
    captures: list[dict[str, object]],
) -> None:
    observations = _observations_dir(root)
    destination = observations / run_id
    _safe_child(observations, destination)
    if destination.exists() or destination.is_symlink():
        raise WebQaObservationStoreError("managed observation directory already exists")
    staging = observations / f".staging-{run_id}-{secrets.token_hex(8)}"
    _safe_child(observations, staging)
    published = False
    try:
        staging.mkdir(mode=0o700)
        captures_dir = staging / "captures"
        captures_dir.mkdir(mode=0o700)
        for capture in captures:
            digest = str(capture["sha256"])
            target = captures_dir / Path(str(capture["path"])).name
            _write_private(target, verified_bytes[digest])
            _verify_capture_bytes(target, capture)
        _write_private(staging / "metadata.json", encoded_metadata)
        if destination.exists() or destination.is_symlink():
            raise WebQaObservationStoreError("managed observation directory already exists")
        os.replace(staging, destination)
        published = True
        destination.chmod(0o700)
        for capture in captures:
            _verify_capture_bytes(destination / str(capture["path"]), capture)
    except Exception:
        if published and destination.exists() and not destination.is_symlink():
            shutil.rmtree(destination, ignore_errors=True)
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _verify_capture_bytes(path: Path, capture: dict[str, object]) -> None:
    data = _read_source_image(path)
    if (
        len(data) != capture["byte_size"]
        or hashlib.sha256(data).hexdigest() != capture["sha256"]
        or _image_mime(data) != capture["mime_type"]
    ):
        raise WebQaObservationStoreError("staged capture bytes do not match verified receipt evidence")


def _read_optional(root: Path, run_id: str, *, trusted_trace_resolver: TrustedTraceResolver | None = None) -> ImportedWebQaObservation | None:
    _run_id(run_id)
    directory = _observations_dir(root) / run_id
    _safe_child(_observations_dir(root), directory)
    if not directory.exists() and not directory.is_symlink():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise WebQaObservationStoreError("managed observation storage symlink escape refused")
    metadata_path = directory / "metadata.json"
    _safe_child(directory, metadata_path)
    metadata = _read_json(metadata_path)
    _validate_metadata(metadata, root, trusted_trace_resolver)
    record = _record(metadata)
    for capture in record.captures:
        relative = str(capture["path"])
        path = directory / relative
        _safe_child(directory, path)
        data = _read_source_image(path)
        if len(data) != capture["byte_size"] or hashlib.sha256(data).hexdigest() != capture["sha256"] or _image_mime(data) != capture["mime_type"]:
            raise WebQaObservationStoreError("managed capture bytes no longer match stored evidence")
    return record


def _record(metadata: dict[str, object]) -> ImportedWebQaObservation:
    timing_data = _object(metadata["timings"], "stored timings")
    return ImportedWebQaObservation(
        plan=_object(metadata["plan"], "stored plan"),
        receipt=_object(metadata["receipt"], "stored receipt"),
        observation=_object(metadata["observation"], "stored observation"),
        captures=_list(metadata["captures"], "stored captures"),
        timings={key: _integer(value, f"stored timing {key}") for key, value in timing_data.items()},
    )


def _validate_metadata(metadata: object, root: Path, resolver: TrustedTraceResolver | None) -> None:
    source = _object(metadata, "stored metadata")
    expected = {"schema_version", "project_identity", "plan", "receipt", "observation", "captures", "timings"}
    if set(source) != expected or source["schema_version"] != WEB_QA_OBSERVATION_STORE_SCHEMA_VERSION:
        raise WebQaObservationStoreError("stored observation metadata is malformed")
    if source["project_identity"] != _project_identity(root):
        raise WebQaObservationStoreError("stored observation belongs to another observed Git root")
    plan = _normalize_plan(source["plan"])
    if _canonical(plan) != _canonical(source["plan"]):
        raise WebQaObservationStoreError("stored plan is noncanonical")
    observation = _admit(plan, source["receipt"], resolver)
    if _canonical(observation) != _canonical(source["observation"]):
        raise WebQaObservationStoreError("stored derived observation does not match admitted evidence")
    expected_captures = _expected_captures(observation)
    captures = _list(source["captures"], "stored captures")
    capture_map: dict[str, dict[str, object]] = {}
    for item in captures:
        capture = _object(item, "stored capture")
        if set(capture) != {"sha256", "byte_size", "mime_type", "path"} or not isinstance(capture.get("sha256"), str):
            raise WebQaObservationStoreError("stored capture metadata is malformed")
        digest = str(capture["sha256"])
        mime = capture.get("mime_type")
        expected_path = f"captures/{digest}{_extension(str(mime))}" if isinstance(mime, str) else ""
        if digest in capture_map or capture.get("byte_size") != expected_captures.get(digest) or capture.get("path") != expected_path:
            raise WebQaObservationStoreError("stored capture metadata does not match receipt evidence")
        capture_map[digest] = capture
    if set(capture_map) != set(expected_captures):
        raise WebQaObservationStoreError("stored captures do not cover screenshot evidence")
    timings = _object(source["timings"], "stored timings")
    timing_keys = {"validation_ms", "adapter_execution_ms", "attempts", "artifact_bytes", "reported_artifact_bytes"}
    if set(timings) != timing_keys:
        raise WebQaObservationStoreError("stored timing metadata is malformed")
    stored_timings = {key: _integer(value, f"stored timing {key}") for key, value in timings.items()}
    if any(value < 0 for value in stored_timings.values()):
        raise WebQaObservationStoreError("stored timing metadata is malformed")
    actual_capture_bytes = sum(_integer(capture["byte_size"], "stored capture byte_size") for capture in captures)
    if (
        stored_timings["adapter_execution_ms"] != _adapter_execution_ms(source["receipt"])
        or stored_timings["attempts"] != _execution_integer(source["receipt"], "attempts")
        or stored_timings["reported_artifact_bytes"] != _execution_integer(source["receipt"], "artifact_bytes")
        or stored_timings["artifact_bytes"] != actual_capture_bytes
        or stored_timings["artifact_bytes"] > stored_timings["reported_artifact_bytes"]
    ):
        raise WebQaObservationStoreError("stored timing metadata does not match receipt execution or imported artifacts")


def _read_json(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise WebQaObservationStoreError("managed observation metadata symlink is refused")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    except OSError as exc:
        raise WebQaObservationStoreError("managed observation metadata was not found") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STORED_METADATA_BYTES:
            raise WebQaObservationStoreError("managed observation metadata exceeds bounds or is unsafe")
        raw = os.read(descriptor, MAX_STORED_METADATA_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_STORED_METADATA_BYTES:
        raise WebQaObservationStoreError("managed observation metadata exceeds bounds")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise WebQaObservationStoreError("managed observation metadata is malformed") from exc
    _bounded(value)
    return _object(value, "managed observation metadata")


def _git_root(cwd: str | Path | None) -> Path:
    try:
        return observed_git_root(cwd)
    except BrowserTraceError as exc:
        raise WebQaObservationStoreError("a real observed Git project root is required") from exc


def _observations_dir(root: Path) -> Path:
    directory = root / ".omh" / "web-visual-qa" / "observations"
    _safe_child(root, directory)
    return directory


def _ensure_real_directory(path: Path) -> None:
    if path.is_symlink():
        raise WebQaObservationStoreError("managed observation storage symlink escape refused")
    try:
        path.mkdir(parents=True, mode=0o700)
    except FileExistsError:
        pass
    if path.is_symlink() or not path.is_dir():
        raise WebQaObservationStoreError("managed observation storage must be a directory")
    path.chmod(0o700)


def _safe_child(parent: Path, child: Path) -> None:
    try:
        relative = child.relative_to(parent)
    except ValueError as exc:
        raise WebQaObservationStoreError("managed observation path traversal or symlink escape refused") from exc
    if ".." in relative.parts:
        raise WebQaObservationStoreError("managed observation path traversal or symlink escape refused")
    # The parent is rooted in the observed Git root (or an admitted child).
    # Prove lexical descent without links instead of comparing two non-strict
    # resolve snapshots across concurrent directory creation/publication.
    current = parent
    for part in ("", *relative.parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            continue
        # Windows junctions are name-surrogate reparse points, not symlinks.
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise WebQaObservationStoreError("managed observation path traversal or symlink escape refused")


def _write_private(path: Path, value: bytes) -> None:
    if path.is_symlink():
        raise WebQaObservationStoreError("managed observation path symlink is refused")
    with path.open("xb") as handle:
        handle.write(value)
    path.chmod(0o600)


def _image_mime(data: bytes) -> str:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _extension(mime: str) -> str:
    if mime == "image/png": return ".png"
    if mime == "image/jpeg": return ".jpg"
    if mime == "image/webp": return ".webp"
    raise WebQaObservationStoreError("stored capture MIME type is unsupported")


def _adapter_execution_ms(receipt: object) -> int:
    execution = _object(_object(receipt, "receipt").get("execution"), "receipt execution")
    commands = _list(execution.get("command_results"), "receipt command results")
    total = 0
    for command in commands:
        duration = _object(command, "receipt command").get("duration_ms")
        if type(duration) is not int or duration < 0:
            raise WebQaObservationStoreError("receipt command duration is invalid")
        total += duration
    return total


def _execution_integer(receipt: object, key: str) -> int:
    value = _object(_object(receipt, "receipt").get("execution"), "receipt execution").get(key)
    result = _integer(value, f"receipt execution {key}")
    if result < 0:
        raise WebQaObservationStoreError(f"receipt execution {key} is invalid")
    return result


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise WebQaObservationStoreError(f"{label} must be an integer")
    return value


def _privacy_safe(value: object) -> None:
    """Reject raw secret-bearing transport material before it reaches disk."""
    forbidden_keys = ("header", "cookie", "credential", "password", "body", "email", "phone", "address")

    def visit(item: object) -> None:
        if type(item) is dict:
            for key, child in item.items():
                lowered = key.lower()
                if key != "forbidden" and any(token in lowered for token in forbidden_keys):
                    raise WebQaObservationStoreError("receipt contains forbidden raw private metadata")
                visit(child)
        elif type(item) is list:
            for child in item:
                visit(child)
        elif isinstance(item, str):
            lowered = item.lower()
            if (
                ("//" in item and ("?" in item or "@" in item))
                or "bearer " in lowered
                or "set-cookie:" in lowered
                or re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", item)
                or re.search(r"\b(?:\+?\d[ -]?){10,15}\b", item)
            ):
                raise WebQaObservationStoreError("receipt contains raw URL query, credential, or PII material")
    visit(value)


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WebQaObservationStoreError("value must contain finite JSON data") from exc


def _encoded_metadata(metadata: dict[str, object]) -> bytes:
    encoded = _canonical(metadata)
    if len(encoded) > MAX_STORED_METADATA_BYTES:
        raise WebQaObservationStoreError("complete observation metadata exceeds byte bound")
    return encoded


def _bounded(value: object, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise WebQaObservationStoreError("managed observation metadata exceeds depth bound")
    if type(value) is dict:
        for key, child in value.items():
            if not isinstance(key, str):
                raise WebQaObservationStoreError("managed observation metadata has an invalid key")
            _bounded(child, depth + 1)
    elif type(value) is list:
        for child in value:
            _bounded(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise WebQaObservationStoreError("managed observation metadata contains non-finite data")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise WebQaObservationStoreError("managed observation metadata contains non-JSON data")


def _project_identity(root: Path) -> str:
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def _run_id(value: str) -> None:
    if not _RUN_ID.fullmatch(value):
        raise WebQaObservationStoreError("run_id is invalid")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant: {value}")


def _object(value: object, label: str) -> dict[str, object]:
    if type(value) is not dict:
        raise WebQaObservationStoreError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[dict[str, object]]:
    if type(value) is not list or not all(type(item) is dict for item in value):
        raise WebQaObservationStoreError(f"{label} must be an object list")
    return value
