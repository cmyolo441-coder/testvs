from __future__ import annotations

from json import JSONDecodeError
from pathlib import Path
from typing import Any

from ..local_store import file_lock, read_json_object, utc_now
from ..paths import OmhPaths
from ..skill_pack import routable_skill_names
from ..plugin_bundle.omh.active_workflow_context_state import (
    ALLOWED_TRANSITIONS as ALLOWED_TRANSITIONS,
    session_fingerprint,
    valid_session_binding,
)
from .record_revision import DuplicateMutationReplay, guarded_record_update

SCHEMA_VERSION = 1
LIFECYCLE_OUTCOMES = ("finished", "blocked", "failed", "user_interlude", "question_pending", "cancelled")


class WorkflowStateError(ValueError):
    pass


def validate_workflow_name(workflow: str) -> None:
    if workflow not in routable_skill_names():
        raise WorkflowStateError(f"unknown workflow: {workflow}")


def workflow_state_path(paths: OmhPaths, workflow: str) -> Path:
    validate_workflow_name(workflow)
    return paths.workflow_state_dir / f"{workflow}-state.json"


def read_workflow_state(paths: OmhPaths, workflow: str) -> dict[str, Any] | None:
    path = workflow_state_path(paths, workflow)
    try:
        data = read_json_object(path)
    except JSONDecodeError:
        raise
    except ValueError as exc:
        raise WorkflowStateError(f"state for {workflow} must be a JSON object") from exc
    if not data:
        return None
    if data.get("workflow") not in {None, workflow}:
        raise WorkflowStateError(f"state file {path} belongs to {data.get('workflow')!r}")
    reference = data.get("session_ref", "")
    if not isinstance(reference, str) or (reference and not valid_session_binding(reference)):
        raise WorkflowStateError("invalid workflow session binding")
    if data.get("session_binding", "bound" if reference else "unbound") != ("bound" if reference else "unbound"):
        raise WorkflowStateError("inconsistent workflow session binding")
    return data


def _require_session(stored_ref: str, session_ref: str) -> None:
    """Every mutation of a bound record requires its exact host session."""
    if stored_ref and (not session_ref or stored_ref != session_fingerprint(session_ref)):
        raise WorkflowStateError("matching session_ref required for bound workflow state")


def read_workflow_state_result(paths: OmhPaths, workflow: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        return read_workflow_state(paths, workflow), None
    except (OSError, JSONDecodeError, WorkflowStateError, ValueError) as exc:
        return None, str(exc)


def list_workflow_states(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    states: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if not paths.workflow_state_dir.exists():
        return states, errors
    for path in sorted(paths.workflow_state_dir.glob("*-state.json")):
        workflow = path.name[: -len("-state.json")]
        state, error = read_workflow_state_result(paths, workflow)
        if error:
            errors.append({"path": str(path), "error": error})
        elif state:
            states.append(state)
    return states, errors


def active_workflow_states(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    states, errors = list_workflow_states(paths)
    return [state for state in states if state.get("active")], errors


def _terminal_state(workflow: str, state: dict[str, Any] | None, outcome: str, note: str = "", transition_target: str | None = None) -> dict[str, Any]:
    if outcome not in LIFECYCLE_OUTCOMES:
        raise WorkflowStateError(f"unsupported lifecycle outcome: {outcome}")
    now = utc_now()
    base = state or {"workflow": workflow, "started_at": now}
    result = {
        **base,
        "schema_version": SCHEMA_VERSION,
        "workflow": workflow,
        "active": False,
        "lifecycle_outcome": outcome,
        "updated_at": now,
    }
    if note:
        result["note"] = note
    if transition_target:
        result["transition_target"] = transition_target
    return result


def _workflow_state_lock_anchor(paths: OmhPaths) -> Path:
    """Anchor for the one lock that serializes all workflow-state transitions.

    The active-workflow invariant spans every ``*-state.json`` in the
    directory, so a per-file lock is not enough: the transition check and all
    of the writes it authorizes have to run under this single directory-level
    lock. ``file_lock`` derives ``.workflow-state.lock`` beside this anchor,
    so the anchor path itself is never created or written.
    """
    return paths.workflow_state_dir / "workflow-state"


def _write_workflow_state(paths: OmhPaths, workflow: str, record: dict[str, Any]) -> dict[str, Any]:
    """Write one state file, bumping its record_revision, under its own lock."""
    result = guarded_record_update(
        workflow_state_path(paths, workflow),
        mutate=lambda current: dict(record),
        operation="write_workflow_state",
        lock_name=f"{workflow}-state.json",
        default={},
        private=True,
    )
    return result.record if isinstance(result, DuplicateMutationReplay) else result


def finish_workflow_state(paths: OmhPaths, workflow: str, outcome: str = "finished", note: str = "", *, session_ref: str = "") -> dict[str, Any]:
    validate_workflow_name(workflow)
    with file_lock(_workflow_state_lock_anchor(paths), private=True):
        state = read_workflow_state(paths, workflow)
        _require_session((state or {}).get("session_ref", ""), session_ref)
        result = _terminal_state(workflow, state, outcome, note)
        return _write_workflow_state(paths, workflow, result)


def _transition_allowed(source: str, destination: str) -> bool:
    return destination in ALLOWED_TRANSITIONS.get(source, ())


def start_workflow_state(paths: OmhPaths, workflow: str, note: str = "", *, session_ref: str = "") -> dict[str, Any]:
    validate_workflow_name(workflow)
    # Read the active set and write every state file it authorizes inside one
    # lock. Reading outside the lock let two concurrent starts each observe an
    # empty active set and both become active, or let one overwrite the
    # auto-completion the other had just written.
    with file_lock(_workflow_state_lock_anchor(paths), private=True):
        active, errors = active_workflow_states(paths)
        if errors:
            first = errors[0]
            raise WorkflowStateError(f"cannot start workflow while state is unreadable: {first['path']}: {first['error']}")
        if len(active) > 1:
            raise WorkflowStateError("multiple active workflows require explicit recovery")
        target = read_workflow_state(paths, workflow)
        _require_session((target or {}).get("session_ref", ""), session_ref)
        for current in active:
            _require_session(current.get("session_ref", ""), session_ref)
        binding = {
            "session_binding": "bound" if session_ref else "unbound",
            "activation": {"source": "explicit_api", "observed_by_host": "not_observed"},
        }
        if session_ref:
            binding["session_ref"] = session_fingerprint(session_ref)
        now = utc_now()
        for current in active:
            source = str(current.get("workflow", ""))
            if source == workflow:
                updated = {**current, **binding, "schema_version": SCHEMA_VERSION, "active": True, "updated_at": now}
                if note:
                    updated["note"] = note
                return _write_workflow_state(paths, workflow, updated)
            if not _transition_allowed(source, workflow):
                raise WorkflowStateError(f"cannot start {workflow}; active workflow {source} must finish or be cleared first")
        for current in active:
            source = str(current["workflow"])
            completed = _terminal_state(source, current, "finished", f"auto-completed before starting {workflow}", workflow)
            _write_workflow_state(paths, source, completed)
        state = {
            **binding,
            "schema_version": SCHEMA_VERSION,
            "workflow": workflow,
            "active": True,
            "lifecycle_outcome": None,
            "started_at": now,
            "updated_at": now,
        }
        if note:
            state["note"] = note
        return _write_workflow_state(paths, workflow, state)


def clear_workflow_state(paths: OmhPaths, workflow: str, *, session_ref: str = "") -> bool:
    path = workflow_state_path(paths, workflow)
    with file_lock(_workflow_state_lock_anchor(paths), private=True):
        state = read_workflow_state(paths, workflow)
        _require_session((state or {}).get("session_ref", ""), session_ref)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True
