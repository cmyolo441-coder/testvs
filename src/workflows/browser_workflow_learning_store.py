from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from ..system.local_store import atomic_write_json, file_lock

from .browser_workflow_learning import BrowserTraceError, BrowserWorkflowTraceReference, JsonObject, JsonValue, MAX_TRACE_BYTES, _lifecycle, _strict_mapping, canonical_origin, parse_browser_workflow_trace, replay_browser_workflow_trace, validate_browser_workflow_trace


def write_browser_workflow_trace(raw: Mapping[str, JsonValue], cwd: str | Path | None = None) -> JsonObject:
    root = _root(cwd); supplied = dict(raw); identity = _root_identity(root)
    supplied["project"] = {"identity": identity}
    source = _strict_mapping(supplied.get("source"), "source")
    origins = supplied.get("origins")
    if not isinstance(origins, list) or not origins: raise BrowserTraceError("origins must include a canonical allowlist")
    source["binding"] = {"project_identity": identity, "origin": canonical_origin(str(origins[0])), "adapter_version": str(supplied.get("adapter_version", "")).strip()}
    supplied["source"] = source
    trace = parse_browser_workflow_trace(supplied, project_identity=identity)
    path = _path(root, str(trace["trace_id"]))
    with file_lock(path, private=True):
        existing = _read(path)
        if existing is not None:
            if existing.get("digest") != trace.get("digest"):
                raise BrowserTraceError("trace id collides with different material", "quarantined")
            return read_browser_workflow_trace(root, str(trace["trace_id"]))
        _write(path, trace)
    return trace


def read_browser_workflow_trace(cwd: str | Path | None, trace_id: str) -> JsonObject:
    root = _root(cwd); trace = _read(_path(root, trace_id))
    if trace is None: raise BrowserTraceError("trace was not found", "not_suitable")
    errors = validate_browser_workflow_trace(trace)
    if trace.get("project") != {"identity": _root_identity(root)}: errors.append("stored trace belongs to another observed Git root")
    if errors: raise BrowserTraceError("stored trace is invalid", "quarantined")
    return trace


def approve_browser_workflow_trace(cwd: str | Path | None, trace_id: str, digest: str) -> JsonObject:
    root = _root(cwd); trace = read_browser_workflow_trace(root, trace_id)
    if trace.get("digest") != digest: raise BrowserTraceError("approval must name the trace's exact digest")
    lifecycle = _strict_mapping(trace["lifecycle"], "lifecycle")
    if lifecycle.get("status") == "approved": return trace
    if lifecycle.get("status") != "pending_approval": raise BrowserTraceError("only a pending trace may be approved")
    path = _path(root, trace_id)
    with file_lock(path, private=True):
        trace = read_browser_workflow_trace(root, trace_id)
        lifecycle = _strict_mapping(trace["lifecycle"], "lifecycle")
        if trace.get("digest") != digest:
            raise BrowserTraceError("approval must name the trace's exact digest")
        if lifecycle.get("status") == "approved": return trace
        if lifecycle.get("status") != "pending_approval":
            raise BrowserTraceError("only a pending trace may be approved")
        trace["lifecycle"] = _next_lifecycle(trace, "approved", digest)
        _write(path, trace)
    return trace


def replay_stored_browser_workflow_trace(cwd: str | Path | None, trace_id: str, observation: Mapping[str, JsonValue]) -> JsonObject:
    root = _root(cwd)
    path = _path(root, trace_id)
    with file_lock(path, private=True):
        trace = read_browser_workflow_trace(root, trace_id)
        lifecycle = _strict_mapping(trace["lifecycle"], "lifecycle")
        result = replay_browser_workflow_trace(trace, observation)
        status = str(result["status"])
        if lifecycle.get("status") == "quarantined" or result.get("reason") == "prior_mismatch":
            return result
        if status == "retry_once":
            trace["lifecycle"] = _next_lifecycle(
                trace, str(lifecycle["status"]), str(lifecycle["approved_digest"]), retry_used=True
            )
            _write(path, trace)
        elif status in {"stale", "quarantined"}:
            digest = str(result.get("fixture_digest", ""))
            stored_history = lifecycle["mismatch_fixture_digests"]
            if not isinstance(stored_history, list):
                raise BrowserTraceError("invalid lifecycle history", "quarantined")
            history = [str(item) for item in stored_history]
            fresh = bool(digest) and digest not in history
            if status == "stale" and not fresh and lifecycle["status"] == "stale":
                return result
            if fresh:
                history.append(digest)
                history.sort()
            next_status = (
                "quarantined"
                if status == "quarantined" or lifecycle["status"] == "stale" and fresh
                else "stale"
            )
            trace["lifecycle"] = _next_lifecycle(trace, next_status, "", history=history)
            _write(path, trace)
            result = {**result, "status": next_status}
    return result


def resolved_browser_workflow_trace_reference(cwd: str | Path | None, trace_id: str) -> JsonObject:
    trace = read_browser_workflow_trace(cwd, trace_id)
    lifecycle = _strict_mapping(trace["lifecycle"], "lifecycle")
    project = _strict_mapping(trace["project"], "project")
    if lifecycle.get("status") != "approved" or lifecycle.get("approved_digest") != trace.get("digest"):
        raise BrowserTraceError("browser trace is not approved", "rejected")
    return BrowserWorkflowTraceReference({"schema_version": "browser_workflow_trace_reference/v1", "trace_id": trace["trace_id"], "digest": trace["digest"], "project_identity": project["identity"], "origins": trace["origins"], "lifecycle_status": "approved"})


def browser_workflow_trace_status(cwd: str | Path | None, trace_id: str) -> JsonObject:
    trace = read_browser_workflow_trace(cwd, trace_id)
    return {"trace_id": trace["trace_id"], "digest": trace["digest"], "project": trace["project"], "origins": trace["origins"], "lifecycle": trace["lifecycle"]}


def resolved_browser_workflow_promotion_reference(cwd: str | Path | None, trace_id: str) -> JsonObject:
    """Mint a read-only promotion input from approved offline fixture proof only."""
    trace = read_browser_workflow_trace(cwd, trace_id)
    lifecycle = _strict_mapping(trace["lifecycle"], "lifecycle")
    project = _strict_mapping(trace["project"], "project")
    fixtures = trace["fixtures"]
    if not isinstance(fixtures, list):
        raise BrowserTraceError("promotion source fixtures are invalid", "quarantined")
    if lifecycle.get("status") != "approved" or lifecycle.get("approved_digest") != trace.get("digest"):
        raise BrowserTraceError("promotion requires an approved browser trace", "rejected")
    results: list[JsonObject] = []
    fixture_digests: JsonObject = {}
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise BrowserTraceError("promotion source fixtures are invalid", "quarantined")
        replay = replay_browser_workflow_trace(trace, {"fixture_id": fixture["fixture_id"]})
        expected_states = {
            "positive": {"replayed"},
            "negative": {"stale", "quarantined"},
            "transient": {"retry_once", "stale"},
        }
        expected = replay.get("status") in expected_states[str(fixture["kind"])]
        if not expected:
            raise BrowserTraceError("promotion requires passing offline fixture replay", "rejected")
        results.append({"fixture_id": fixture["fixture_id"], "fixture_digest": fixture["digest"], "status": replay["status"], "reason": replay["reason"]})
        fixture_digests[str(fixture["fixture_id"])] = fixture["digest"]
    results.sort(key=lambda item: str(item["fixture_id"]))
    replay_digest = hashlib.sha256(json.dumps({"trace_digest": trace["digest"], "fixture_results": results}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    output_schema = trace["output_schema"]
    output_schema_digest = hashlib.sha256(json.dumps(output_schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    return {"schema_version": "browser_workflow_promotion_reference/v1", "project_identity": project["identity"], "trace_id": trace["trace_id"], "trace_digest": trace["digest"], "trace_revision": lifecycle["revision"], "origins": trace["origins"], "output_schema": output_schema, "output_schema_digest": output_schema_digest, "fixture_digests": fixture_digests, "replay_status": "passed", "replay_digest": replay_digest, "lifecycle_status": "approved"}


def _root(cwd: str | Path | None) -> Path:
    import subprocess

    start = Path(cwd or Path.cwd()).expanduser().resolve()
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "--no-optional-locks", "rev-parse", "--show-toplevel"],
            cwd=start, env=environment, capture_output=True, check=True, timeout=5,
        )
        raw = completed.stdout.removesuffix(b"\n")
        root = Path(os.fsdecode(raw)).resolve(strict=True)
        start.relative_to(root)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise BrowserTraceError("a real observed Git root is required", "not_suitable") from exc
    return root


def observed_checkout_revision(root: Path) -> str:
    """Return the full commit id HEAD names in an already-resolved Git root.

    A bounded local identity read, like ``_root``: it writes no index, names no
    remote, and raises ``BrowserTraceError`` when HEAD is unborn or unreadable.
    """
    import subprocess

    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "--no-optional-locks", "rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
            cwd=root, env=environment, capture_output=True, check=True, timeout=5,
        )
        revision = completed.stdout.decode("ascii").strip()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise BrowserTraceError("an observed checkout revision is required", "not_suitable") from exc
    if len(revision) not in (40, 64) or any(char not in "0123456789abcdef" for char in revision):
        raise BrowserTraceError("an observed checkout revision is required", "not_suitable")
    return revision


def _path(root: Path, trace_id: str) -> Path:
    if not trace_id.startswith("bwt-") or len(trace_id) != 28 or any(char not in "0123456789abcdef" for char in trace_id[4:]): raise BrowserTraceError("trace id is invalid", "not_suitable")
    directory = root / ".omh" / "web-visual-qa" / "traces"
    for parent in (root / ".omh", root / ".omh" / "web-visual-qa", directory):
        if parent.is_symlink(): raise BrowserTraceError("trace storage symlink escape refused", "quarantined")
    path = directory / f"{trace_id}.json"
    if path.is_symlink(): raise BrowserTraceError("trace storage symlink escape refused", "quarantined")
    return path


def _read(path: Path) -> JsonObject | None:
    try:
        if not path.exists() and not path.is_symlink(): return None
        if path.is_symlink(): raise BrowserTraceError("stored trace symlink refused", "quarantined")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TRACE_BYTES: raise BrowserTraceError("stored trace exceeds bounds or is unsafe", "quarantined")
            chunks: list[bytes] = []; remaining = MAX_TRACE_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk: break
                chunks.append(chunk); remaining -= len(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        if len(raw) > MAX_TRACE_BYTES: raise BrowserTraceError("stored trace exceeds bounds", "quarantined")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(value, dict): raise ValueError("not object")
        return value
    except BrowserTraceError: raise
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc: raise BrowserTraceError("stored trace is malformed", "quarantined") from exc


def _write(path: Path, value: JsonObject) -> None:
    if validate_browser_workflow_trace(value):
        raise BrowserTraceError("refusing an invalid trace write", "quarantined")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_TRACE_BYTES: raise BrowserTraceError("trace exceeds 256 KiB", "not_suitable")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink(): raise BrowserTraceError("trace storage symlink escape refused", "quarantined")
    atomic_write_json(path, value, private=True)


def _next_lifecycle(trace: JsonObject, status: str, approved: str, *, history: list[str] | None = None, retry_used: bool | None = None) -> JsonObject:
    current = _strict_mapping(trace["lifecycle"], "lifecycle")
    revision = current.get("revision", 0)
    if not isinstance(revision, int) or revision >= 64: raise BrowserTraceError("trace lifecycle revision cap reached", "not_suitable")
    previous_history = current.get("mismatch_fixture_digests", [])
    if not isinstance(previous_history, list):
        raise BrowserTraceError("invalid lifecycle history", "quarantined")
    return _lifecycle(status, approved, revision + 1, bool(current.get("retry_used")) if retry_used is None else retry_used, history if history is not None else [str(item) for item in previous_history])


def _root_identity(root: Path) -> str: return hashlib.sha256(str(root).encode("utf-8")).hexdigest()
def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result: raise ValueError("duplicate JSON key")
        result[key] = value
    return result
def _reject_constant(value: str) -> object: raise ValueError(f"non-finite JSON constant: {value}")
