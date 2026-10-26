"""Lineage-bound `motion_interaction_capture/v1` import for host-owned web QA.

A host records a browser motion capture (MP4 or WebM) and, optionally, derives a
changed-frame contact sheet from it. This module admits the host's metadata-only
receipt for that capture against one `web_qa_observation_plan/v1` matrix cell and
persists the admitted record under the project's managed web-QA store.

It never launches a browser, records, encodes, or reads media, and it never
stores media bytes or raw frames: every artifact is a digest plus bounded
producer-attested facts. A record proves only that the named artifact was
declared for the named condition; it does not prove visual correctness,
complete interaction coverage, delivery, or PASS.

Every refusal raises `MotionCaptureImportError` whose `reason` is one closed
code from `MOTION_CAPTURE_REFUSAL_REASONS`.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
import re
import secrets
from typing import Final, NoReturn

from ..system.local_store import file_lock
from .browser_workflow_learning import BrowserTraceError
from .browser_workflow_learning_store import observed_checkout_revision
from .web_qa_observation import TrustedTraceResolver
from .web_qa_observation_store import (
    MAX_STORED_METADATA_BYTES,
    WebQaObservationStoreError,
    _bounded,
    _canonical,
    _ensure_real_directory,
    _git_root,
    _normalize_plan,
    _privacy_safe,
    _project_identity,
    _read_json,
    _safe_child,
    _write_private,
    read_web_qa_observation,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


MOTION_INTERACTION_CAPTURE_SCHEMA_VERSION: Final = "motion_interaction_capture/v1"
HOST_MOTION_CAPTURE_RECEIPT_SCHEMA_VERSION: Final = "host_motion_capture_receipt/v1"
MOTION_CAPTURE_STORE_SCHEMA_VERSION: Final = "web_qa_motion_capture_store/v1"
MOTION_GATE_SCHEMA_VERSION: Final = "web_qa_motion_gate/v1"
MOTION_CAPTURE_REFUSAL_REASONS: Final = (
    "malformed_receipt",
    "plan_invalid",
    "path_only_evidence",
    "secret_bearing_metadata",
    "digest_invalid",
    "unsupported_media_type",
    "metadata_limit_exceeded",
    "lineage_mismatch",
    "checkout_revision_mismatch",
    "contact_sheet_lineage_mismatch",
    "stored_record_invalid",
)
MOTION_CAPTURE_CLAIM_BOUNDARY: Final = (
    "The record proves only that the host declared the named recording, and any contact sheet derived "
    "from it, for the named repository, revision, round, route, state, viewport, and browser. OMH did "
    "not capture, decode, or inspect the media."
)
MOTION_CAPTURE_DOES_NOT_PROVE: Final = (
    "visual_correctness",
    "complete_interaction_coverage",
    "delivery",
    "visual_qa_pass",
    "browser_capture_performed_by_omh",
)
RECORDING_MEDIA_TYPES: Final = ("video/mp4", "video/webm")
CONTACT_SHEET_MEDIA_TYPES: Final = ("image/png", "image/jpeg", "image/webp")
SAMPLING_POLICIES: Final = ("every_frame", "fixed_interval", "changed_frames")
MAX_RECORDING_BYTES: Final = 512 * 1024 * 1024
MAX_CONTACT_SHEET_BYTES: Final = 25 * 1024 * 1024
MAX_FRAMES_PER_SECOND: Final = 120
MAX_CONTACT_SHEET_TILES: Final = 64
MAX_RECEIPT_BYTES: Final = 32_768
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]{0,63}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_RUN_ID = re.compile(r"^web-qa-[a-f0-9]{24}$")
_CAPTURE_ID = re.compile(r"^motion-[a-f0-9]{24}$")
_PATH_KEYS = ("path", "uri", "url", "file", "location")


class MotionCaptureImportError(ValueError):
    """A motion capture receipt or stored record was refused for one named reason."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in MOTION_CAPTURE_REFUSAL_REASONS:
            raise ValueError(f"unknown motion capture refusal reason: {reason}")
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


def build_motion_interaction_capture(plan: object, receipt: object) -> dict[str, object]:
    """Admit one host receipt against a plan cell and return the closed record.

    Pure: no I/O. The checkout-revision check belongs to the importer because a
    stored record stays readable after the checkout moves on.
    """
    normalized_plan = _plan(plan)
    source = _receipt_shape(receipt)
    root = _closed(
        source,
        ("schema_version", "receipt_version", "run_id", "cell_id", "lineage", "producer", "requested", "observed", "recording", "contact_sheet"),
        "receipt",
    )
    if root["schema_version"] != HOST_MOTION_CAPTURE_RECEIPT_SCHEMA_VERSION or root["receipt_version"] != 1:
        _refuse("malformed_receipt", "receipt schema is invalid")
    if root["run_id"] != normalized_plan["run_id"]:
        _refuse("lineage_mismatch", "receipt run_id does not name the plan")
    cell = next((item for item in _objects(normalized_plan["matrix"]) if item["cell_id"] == root["cell_id"]), None)
    if cell is None:
        _refuse("lineage_mismatch", "receipt cell_id is not a planned matrix cell")
    subject = _object(normalized_plan["subject"])
    condition = _object(normalized_plan["condition"])
    round_identity = _object(normalized_plan["round"])
    lineage = _closed(root["lineage"], ("repository", "revision", "route_id", "state_id", "viewport_id", "browser_id"), "receipt lineage")
    expected_lineage = {
        "repository": subject["repository"],
        "revision": subject["revision"],
        "route_id": cell["route_id"],
        "state_id": cell["state_id"],
        "viewport_id": cell["viewport_id"],
        "browser_id": cell["browser_id"],
    }
    for key, value in expected_lineage.items():
        if lineage[key] != value:
            _refuse("lineage_mismatch", f"receipt lineage {key} does not match the planned cell")

    producer = _closed(root["producer"], ("producer_id", "version"), "receipt producer")
    _text(producer["producer_id"], _REF, "producer_id")
    _text(producer["version"], _VERSION, "producer version")

    recording = _recording(root["recording"])
    requested = _closed(root["requested"], ("sampling_policy", "max_duration_ms"), "receipt requested")
    _enum(requested["sampling_policy"], SAMPLING_POLICIES, "requested sampling_policy")
    max_run_ms = _int(_object(normalized_plan["limits"])["max_run_seconds"], "plan max_run_seconds") * 1000
    if not 1 <= _int(requested["max_duration_ms"], "requested max_duration_ms") <= max_run_ms:
        _refuse("metadata_limit_exceeded", "requested max_duration_ms exceeds the plan run bound")
    observed = _closed(root["observed"], ("sampling_policy", "started_at", "ended_at"), "receipt observed")
    _enum(observed["sampling_policy"], SAMPLING_POLICIES, "observed sampling_policy")
    started = _utc(observed["started_at"], "observed started_at")
    ended = _utc(observed["ended_at"], "observed ended_at")
    window_ms = int((ended - started).total_seconds() * 1000)
    if window_ms <= 0:
        _refuse("malformed_receipt", "observed capture window must be positive")
    duration_ms = int(recording["duration_ms"])
    if duration_ms > int(requested["max_duration_ms"]) or duration_ms > window_ms:
        _refuse("metadata_limit_exceeded", "recording duration exceeds the requested bound or the observed window")
    contact_sheet = _contact_sheet(root["contact_sheet"], str(recording["sha256"]), duration_ms)

    body: dict[str, object] = {
        "schema_version": MOTION_INTERACTION_CAPTURE_SCHEMA_VERSION,
        "run_id": normalized_plan["run_id"],
        "plan_digest": normalized_plan["plan_digest"],
        "cell_id": cell["cell_id"],
        "lineage": {
            **expected_lineage,
            "locale": condition["locale"],
            "round_id": round_identity["round_id"],
            "round_ordinal": round_identity["ordinal"],
        },
        "producer": dict(producer),
        "requested": dict(requested),
        "observed": dict(observed),
        "recording": recording,
        "contact_sheet": contact_sheet,
        "claim_boundary": MOTION_CAPTURE_CLAIM_BOUNDARY,
        "does_not_prove": list(MOTION_CAPTURE_DOES_NOT_PROVE),
    }
    return {"capture_id": f"motion-{hashlib.sha256(_canonical(body)).hexdigest()[:24]}", **body}


def import_motion_capture(project_root: str | Path | None, plan: object, receipt: object) -> dict[str, object]:
    """Admit and persist one metadata-only motion capture record.

    Re-importing the same receipt returns the stored record unchanged. A
    different artifact digest or capture condition derives a different
    ``capture_id`` and never overwrites earlier evidence.
    """
    root = _root(project_root)
    normalized_plan = _plan(plan)
    record = build_motion_interaction_capture(normalized_plan, receipt)
    try:
        checkout = observed_checkout_revision(root)
    except BrowserTraceError as exc:
        raise MotionCaptureImportError("checkout_revision_mismatch", "the project checkout has no readable HEAD revision") from exc
    if _object(record["lineage"])["revision"] != checkout:
        _refuse("checkout_revision_mismatch", "capture revision does not match the project checkout revision")
    run_id = str(record["run_id"])
    capture_id = str(record["capture_id"])
    directory = _run_directory(root, run_id)
    try:
        _ensure_real_directory(directory.parent)
        _ensure_real_directory(directory)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("stored_record_invalid", str(exc)) from exc
    target = directory / f"{capture_id}.json"
    _child(directory, target)
    with file_lock(target, private=True) as lock:
        if lock["enforced"] is not True:
            _refuse("stored_record_invalid", "motion capture import requires an enforced file lock")
        if target.exists() or target.is_symlink():
            return read_motion_capture(root, run_id, capture_id)
        metadata = {
            "schema_version": MOTION_CAPTURE_STORE_SCHEMA_VERSION,
            "project_identity": _project_identity(root),
            "plan": normalized_plan,
            "receipt": receipt,
            "record": record,
        }
        encoded = _canonical(metadata)
        if len(encoded) > MAX_STORED_METADATA_BYTES:
            _refuse("metadata_limit_exceeded", "motion capture metadata exceeds the stored byte bound")
        staging = directory / f".staging-{capture_id}-{secrets.token_hex(8)}.json"
        _child(directory, staging)
        try:
            _write_private(staging, encoded)
            os.replace(staging, target)
        finally:
            if staging.exists() and not staging.is_symlink():
                staging.unlink()
    return read_motion_capture(root, run_id, capture_id)


def read_motion_capture(project_root: str | Path | None, run_id: str, capture_id: str) -> dict[str, object]:
    """Re-admit one stored record from its stored plan and receipt."""
    root = _root(project_root)
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        _refuse("malformed_receipt", "run_id is invalid")
    if not isinstance(capture_id, str) or not _CAPTURE_ID.fullmatch(capture_id):
        _refuse("malformed_receipt", "capture_id is invalid")
    directory = _run_directory(root, run_id)
    path = directory / f"{capture_id}.json"
    _child(directory, path)
    try:
        metadata = _read_json(path)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("stored_record_invalid", str(exc)) from exc
    if set(metadata) != {"schema_version", "project_identity", "plan", "receipt", "record"} or metadata["schema_version"] != MOTION_CAPTURE_STORE_SCHEMA_VERSION:
        _refuse("stored_record_invalid", "stored motion capture metadata is malformed")
    if metadata["project_identity"] != _project_identity(root):
        _refuse("stored_record_invalid", "stored motion capture belongs to another observed Git root")
    record = build_motion_interaction_capture(metadata["plan"], metadata["receipt"])
    if _canonical(record) != _canonical(metadata["record"]) or record["capture_id"] != capture_id or record["run_id"] != run_id:
        _refuse("stored_record_invalid", "stored motion capture does not re-admit to the same record")
    return record


def list_motion_captures(project_root: str | Path | None, run_id: str) -> list[dict[str, object]]:
    """Re-admit every stored record for one run, ordered by capture_id."""
    root = _root(project_root)
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        _refuse("malformed_receipt", "run_id is invalid")
    directory = _run_directory(root, run_id)
    if not directory.exists() and not directory.is_symlink():
        return []
    if directory.is_symlink() or not directory.is_dir():
        _refuse("stored_record_invalid", "motion capture storage must be a real directory")
    names = sorted(entry.name[:-5] for entry in directory.iterdir() if entry.name.endswith(".json") and _CAPTURE_ID.fullmatch(entry.name[:-5]))
    return [read_motion_capture(root, run_id, name) for name in names]


def project_motion_evidence(
    plan: object,
    observation: dict[str, object],
    records: list[dict[str, object]],
    motion_cell_ids: list[str],
) -> dict[str, object]:
    """Project motion records onto the observation verdict for named cells only.

    A record satisfies a cell only when run, plan digest, repository, revision,
    round, route, state, viewport, and browser all match exactly. The
    projection can only keep or lower the observation verdict: a covered motion
    cell never turns a BLOCK or REVISE into PASS.
    """
    normalized_plan = _plan(plan)
    if observation.get("run_id") != normalized_plan["run_id"] or observation.get("plan_digest") != normalized_plan["plan_digest"]:
        _refuse("lineage_mismatch", "observation does not belong to the plan")
    matrix = {str(cell["cell_id"]): cell for cell in _objects(normalized_plan["matrix"])}
    subject = _object(normalized_plan["subject"])
    round_identity = _object(normalized_plan["round"])
    cells: list[dict[str, object]] = []
    motion_blockers: list[str] = []
    for cell_id in sorted(set(motion_cell_ids)):
        cell = matrix.get(cell_id)
        if cell is None:
            _refuse("lineage_mismatch", "a required motion cell is not a planned matrix cell")
        expected = {
            "repository": subject["repository"],
            "revision": subject["revision"],
            "route_id": cell["route_id"],
            "state_id": cell["state_id"],
            "viewport_id": cell["viewport_id"],
            "browser_id": cell["browser_id"],
            "round_id": round_identity["round_id"],
            "round_ordinal": round_identity["ordinal"],
        }
        cited = sorted(
            str(record["capture_id"])
            for record in records
            if record.get("schema_version") == MOTION_INTERACTION_CAPTURE_SCHEMA_VERSION
            and record.get("run_id") == normalized_plan["run_id"]
            and record.get("plan_digest") == normalized_plan["plan_digest"]
            and all(_object(record.get("lineage")).get(key) == value for key, value in expected.items())
        )
        if cited:
            cells.append({"cell_id": cell_id, "status": "observed", "blocker_id": "", "capture_ids": cited})
        else:
            cells.append({"cell_id": cell_id, "status": "blocked", "blocker_id": "motion_interaction_capture_missing", "capture_ids": []})
            motion_blockers.append(f"{cell_id}:motion_interaction_capture:motion_interaction_capture_missing")
    base_verdict = observation.get("verdict")
    if base_verdict not in ("PASS", "REVISE", "BLOCK"):
        _refuse("stored_record_invalid", "observation verdict is invalid")
    base_blockers = observation.get("blockers")
    if type(base_blockers) is not list:
        _refuse("stored_record_invalid", "observation blockers are invalid")
    return {
        "schema_version": MOTION_GATE_SCHEMA_VERSION,
        "run_id": normalized_plan["run_id"],
        "observation_verdict": base_verdict,
        "verdict": "BLOCK" if motion_blockers else base_verdict,
        "blockers": sorted({*map(str, base_blockers), *motion_blockers}),
        "motion_cells": cells,
        "does_not_prove": list(MOTION_CAPTURE_DOES_NOT_PROVE),
    }


def motion_capture_gate(
    project_root: str | Path | None,
    run_id: str,
    motion_cell_ids: list[str],
    *,
    trusted_trace_resolver: TrustedTraceResolver | None = None,
) -> dict[str, object]:
    """Re-admit a stored observation and its stored motion records, then gate."""
    try:
        stored = read_web_qa_observation(project_root, run_id, trusted_trace_resolver=trusted_trace_resolver)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("stored_record_invalid", str(exc)) from exc
    records = list_motion_captures(project_root, run_id)
    return project_motion_evidence(stored["plan"], _object(stored["observation"]), records, motion_cell_ids)


def _recording(value: object) -> dict[str, object]:
    recording = _closed(value, ("sha256", "byte_size", "media_type", "width", "height", "duration_ms", "frame_count"), "receipt recording")
    _digest(recording["sha256"], "recording sha256")
    _enum_media(recording["media_type"], RECORDING_MEDIA_TYPES, "recording")
    if not 1 <= _int(recording["byte_size"], "recording byte_size") <= MAX_RECORDING_BYTES:
        _refuse("metadata_limit_exceeded", "recording byte_size is outside bounds")
    if not 1 <= _int(recording["width"], "recording width") <= 7680 or not 1 <= _int(recording["height"], "recording height") <= 4320:
        _refuse("metadata_limit_exceeded", "recording dimensions are outside bounds")
    duration_ms = _int(recording["duration_ms"], "recording duration_ms")
    if duration_ms < 1:
        _refuse("metadata_limit_exceeded", "recording duration_ms must be positive")
    frame_limit = max(1, -(-duration_ms * MAX_FRAMES_PER_SECOND // 1000))
    if not 1 <= _int(recording["frame_count"], "recording frame_count") <= frame_limit:
        _refuse("metadata_limit_exceeded", "recording frame_count is outside the duration bound")
    return dict(recording)


def _contact_sheet(value: object, recording_sha256: str, duration_ms: int) -> dict[str, object] | None:
    if value is None:
        return None
    sheet = _closed(value, ("sha256", "byte_size", "media_type", "source_recording_sha256", "tiles"), "receipt contact_sheet")
    _digest(sheet["sha256"], "contact sheet sha256")
    _digest(sheet["source_recording_sha256"], "contact sheet source_recording_sha256")
    _enum_media(sheet["media_type"], CONTACT_SHEET_MEDIA_TYPES, "contact sheet")
    if not 1 <= _int(sheet["byte_size"], "contact sheet byte_size") <= MAX_CONTACT_SHEET_BYTES:
        _refuse("metadata_limit_exceeded", "contact sheet byte_size is outside bounds")
    if sheet["source_recording_sha256"] != recording_sha256:
        _refuse("contact_sheet_lineage_mismatch", "contact sheet is not derived from the named recording")
    if sheet["sha256"] == recording_sha256:
        _refuse("contact_sheet_lineage_mismatch", "contact sheet cannot replace its source recording identity")
    tiles = sheet["tiles"]
    if type(tiles) is not list or not tiles:
        _refuse("malformed_receipt", "contact sheet tiles must be a non-empty list")
    if len(tiles) > MAX_CONTACT_SHEET_TILES:
        _refuse("metadata_limit_exceeded", "contact sheet tiles exceed the tile bound")
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(tiles):
        tile = _closed(raw, ("tile_index", "source_recording_sha256", "start_ms", "end_ms"), "contact sheet tile")
        if _int(tile["tile_index"], "tile_index") != index:
            _refuse("malformed_receipt", "contact sheet tile_index values must be 0..n-1 in order")
        _digest(tile["source_recording_sha256"], "tile source_recording_sha256")
        if tile["source_recording_sha256"] != recording_sha256:
            _refuse("contact_sheet_lineage_mismatch", "a contact sheet tile does not bind to the recording digest")
        start = _int(tile["start_ms"], "tile start_ms")
        end = _int(tile["end_ms"], "tile end_ms")
        if not 0 <= start <= end <= duration_ms:
            _refuse("contact_sheet_lineage_mismatch", "a contact sheet tile claims an interval outside the recording")
        normalized.append(dict(tile))
    return {**sheet, "tiles": normalized}


def _receipt_shape(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _refuse("malformed_receipt", "receipt must be an object")
    try:
        _bounded(value)
        encoded = _canonical(value)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("malformed_receipt", str(exc)) from exc
    if len(encoded) > MAX_RECEIPT_BYTES:
        _refuse("metadata_limit_exceeded", "receipt exceeds the byte bound")
    _no_path_keys(value)
    try:
        _privacy_safe(value)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("secret_bearing_metadata", str(exc)) from exc
    return value


def _no_path_keys(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            lowered = key.lower()
            if any(lowered == token or lowered.endswith(f"_{token}") for token in _PATH_KEYS):
                _refuse("path_only_evidence", "receipt names a file path or URI; artifacts are admitted by digest only")
            _no_path_keys(child)
    elif type(value) is list:
        for child in value:
            _no_path_keys(child)


def _plan(value: object) -> dict[str, object]:
    try:
        return _normalize_plan(value)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("plan_invalid", str(exc)) from exc


def _root(project_root: str | Path | None) -> Path:
    try:
        return _git_root(project_root)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("stored_record_invalid", str(exc)) from exc


def _run_directory(root: Path, run_id: str) -> Path:
    directory = root / ".omh" / "web-visual-qa" / "motion-captures" / run_id
    _child(root, directory)
    return directory


def _child(parent: Path, child: Path) -> None:
    try:
        _safe_child(parent, child)
    except WebQaObservationStoreError as exc:
        raise MotionCaptureImportError("stored_record_invalid", str(exc)) from exc


def _closed(value: object, keys: tuple[str, ...], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != set(keys):
        _refuse("malformed_receipt", f"{label} must contain exactly {', '.join(keys)}")
    return value


def _object(value: object) -> dict[str, object]:
    if type(value) is not dict:
        _refuse("malformed_receipt", "expected an object")
    return value


def _objects(value: object) -> list[dict[str, object]]:
    if type(value) is not list or any(type(item) is not dict for item in value):
        _refuse("plan_invalid", "expected an object list")
    return value


def _int(value: object, label: str) -> int:
    if type(value) is not int:
        _refuse("malformed_receipt", f"{label} must be an integer")
    return value


def _text(value: object, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _refuse("malformed_receipt", f"{label} is invalid")
    return value


def _digest(value: object, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        _refuse("digest_invalid", f"{label} must be a lowercase SHA-256 digest")


def _enum(value: object, choices: tuple[str, ...], label: str) -> None:
    if value not in choices:
        _refuse("malformed_receipt", f"{label} is not one of {', '.join(choices)}")


def _enum_media(value: object, choices: tuple[str, ...], label: str) -> None:
    if value not in choices:
        _refuse("unsupported_media_type", f"{label} media_type must be one of {', '.join(choices)}")


def _utc(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not _UTC.fullmatch(value):
        _refuse("malformed_receipt", f"{label} must be a UTC timestamp")
    try:
        return strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MotionCaptureImportError("malformed_receipt", f"{label} must be a UTC timestamp") from exc


def _refuse(reason: str, detail: str) -> NoReturn:
    raise MotionCaptureImportError(reason, detail)
