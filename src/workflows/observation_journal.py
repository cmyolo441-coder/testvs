from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
import secrets
from typing import Any
import unicodedata

from ..coding.fanout_failure_diagnostics import FailureDiagnostic, read_failure_diagnostic
from ..coding.fanout_executor_sessions import bound_session_fields
from ..coding.fanout_capacity import read_capacity_fields
from ..coding.fanout_repair import (
    REPAIR_ATTEMPT_OBSERVED_EVENT,
    REPAIR_ATTEMPT_STARTED_EVENT,
    REPAIR_BLOCKED_REASONS,
    bounded_repair_attempt,
    bounded_repair_checks,
)

from ..system.local_store import append_jsonl_locked, read_jsonl_objects, read_runtime_record, utc_now
from ..plugin_bundle.omh.run_records import (
    CODING_DELEGATION_FILE,
    LIFECYCLE_PROJECTION_SCHEMA_VERSION,
    OBSERVATION_EVENT_SCHEMA_VERSION,
)
from ..system.paths import OmhPaths


OBSERVATION_PRIVACY = "metadata_only"
# `cancelled` is a member because the projection below already produces it:
# `project_run_lifecycle` sets `observation_status` to "cancelled" from the
# `cancelled` event, and `_terminal_status` already ranks it beside blocked and
# failed. Without it here the status could be *projected* but never *recorded*,
# so the one state that says "someone stopped this deliberately" could only
# arrive as a side effect of an event whose own status said something else.
OBSERVATION_STATUSES = ("observed", "blocked", "cancelled", "failed", "not_observed")
CANONICAL_OBSERVATION_EVENTS = (
    "prepared_handoff_created",
    "plan_artifact_created",
    "plan_accepted",
    "plan_revised",
    "plan_cancelled",
    "runtime_start_observed",
    "worktree_creation_observed",
    "executor_dispatch_observed",
    "executor_result_observed",
    # A shape-validation receipt for an executor-reported unit sidecar. It is
    # not verification and therefore does not participate in PROJECTION_ORDER.
    "unit_result_validated",
    "executor_session_observed",
    "capacity_admission_observed",
    "unit_result_missing",
    "unit_result_invalid",
    # Per-unit dispatcher evidence. This is deliberately distinct from the
    # run-level verification_result_observed event and its projection field.
    "unit_verification_observed",
    # A receipt, not a lifecycle rung: the dispatcher ran a reproduction unit's
    # declared command and saw it exit nonzero (`observed`) or zero
    # (`not_observed`: the failure was not observed).
    # It releases the fanout reproduction hold and advances nothing here.
    "reproduction_failure_observed",
    # Receipts, not lifecycle rungs: the fanout repair loop's per-attempt
    # records (`coding/fanout_repair.py`). `started` counts a repair spawn;
    # `observed` carries the verdict of the unit's checks after an attempt
    # (`observed` passed, `failed` still failing, `blocked` budget spent).
    # The attempt count a later dispatch continues from is read off these.
    "repair_attempt_started",
    "repair_attempt_observed",
    "verification_result_observed",
    "review_result_observed",
    "ci_result_observed",
    "merge_gate_observed",
    "merge_observed",
    # A receipt, not a lifecycle rung: it records what a partially failed
    # worktree creation removed or deliberately left behind, so a stopped
    # dispatch says what state the repository is in. It is intentionally
    # absent from `PROJECTION_ORDER` -- cleanup advances nothing.
    "worktree_cleanup",
    "blocked",
    "failed",
    "cancelled",
)
OBSERVATION_EVENT_ALIASES = {
    "coding_handoff_prepared": "prepared_handoff_created",
    "handoff_prepared": "prepared_handoff_created",
    "runtime_start": "runtime_start_observed",
    "worktree_creation": "worktree_creation_observed",
    "worker_dispatch": "executor_dispatch_observed",
    "executor_dispatch": "executor_dispatch_observed",
    "worker_result": "executor_result_observed",
    "executor_result": "executor_result_observed",
    "verification": "verification_result_observed",
    "review": "review_result_observed",
    "ci": "ci_result_observed",
    "merge_readiness": "merge_gate_observed",
    "merge_gate": "merge_gate_observed",
    "merge": "merge_observed",
}
PROJECTION_ORDER = (
    "prepared_not_observed",
    "runtime_start_observed",
    "worktree_creation_observed",
    "dispatch_observed",
    "execution_observed",
    "verification_observed",
    "review_observed",
    "ci_observed",
    "merge_gate_observed",
    "merge_observed",
)


def canonical_observation_event(event: str) -> str:
    value = str(event).strip()
    return OBSERVATION_EVENT_ALIASES.get(value, value)


def _diagnostic_identity(value: object) -> str | None:
    if (isinstance(value, str) and 0 < len(value) <= 2048
            and not any(unicodedata.category(ch).startswith("C") for ch in value)):
        return value
    return None


def project_bound_failure_diagnostic(
    record: Mapping[str, object], *, fanout_id: str, unit_id: str | None, run_ref: str,
) -> FailureDiagnostic | None:
    """Read optional diagnostic data against OUTER identity, never its own claims.

    Missing attempt identity on a unit is legacy/unavailable. Optional workspace
    observations, when present, must agree too. This performs no artifact I/O.
    """
    attempt = _diagnostic_identity(record.get("attempt_id"))
    if attempt is None and (unit_id is not None or "attempt_id" not in record or record["attempt_id"] is not None):
        return None
    for key, expected in (("fanout_id", fanout_id), ("unit_id", unit_id),
                          ("worker_ref", unit_id), ("run_ref", run_ref),
                          ("run_id", run_ref), ("target_id", run_ref)):
        if key in record and record[key] != expected:
            return None
    diagnostic = read_failure_diagnostic(
        record.get("failure_diagnostic"), fanout_id=fanout_id, unit_id=unit_id, attempt_id=attempt,
    )
    if diagnostic is None or diagnostic["run_ref"] != run_ref:
        return None
    owner = record.get("owner", record.get("runtime_profile"))
    if diagnostic["owner"] != owner:
        return None
    for key in ("worktree_ref", "base_sha", "observed_revision"):
        if key in record and record[key] != diagnostic[key]:
            return None
    if "worktree_path" in record and record["worktree_path"] != diagnostic["worktree_ref"]:
        return None
    if diagnostic["phase"] == "worker" and "exit_code" in record:
        if type(record["exit_code"]) is not int or record["exit_code"] != diagnostic["returncode"]:
            return None
    name = record.get("event", record.get("event_type"))
    if name is not None:
        if record.get("status") not in ("failed", "blocked") and name not in (
            "failed", "blocked", "unit_result_missing", "unit_result_invalid",
        ):
            return None
    elif (record.get("process_succeeded") or record.get("status") in ("completed", "already_completed")):
        if diagnostic["phase"] not in ("unit_result", "verification", "dispatcher"):
            return None
    return diagnostic


def observation_failure_diagnostic(record: Mapping[str, object]) -> FailureDiagnostic | None:
    """Validate a fanout observation using its canonical run/unit naming boundary."""
    run_ref = record.get("run_id", record.get("target_id"))
    if not isinstance(run_ref, str):
        return None
    match = re.fullmatch(r"(fanout-[0-9a-f]{12})(?:-(.+))?", run_ref)
    if match is None:
        return None
    return project_bound_failure_diagnostic(
        record, fanout_id=match[1], unit_id=match[2], run_ref=run_ref,
    )


def project_run_failure_diagnostic(
    events: Sequence[Mapping[str, object]], *, run_id: str,
) -> dict[str, object]:
    """Select in journal append order; late old attempts cannot retake ownership.

    Sidecar receipts without attempt metadata do not hide a worker failure.
    A legacy dispatch boundary clears it; no old spill reference is resolved.
    """
    current: str | None = None
    seen: set[str] = set()
    selected: FailureDiagnostic | None = None
    for event in events:
        if event.get("run_id") != run_id:
            continue
        attempt = _diagnostic_identity(event.get("attempt_id"))
        name = canonical_observation_event(str(event.get("event", "")))
        if attempt is not None and attempt != current:
            if attempt in seen:
                continue
            seen.add(attempt)
            current, selected = attempt, None
        elif attempt is None and name in ("executor_dispatch_observed", "runtime_start_observed", "worktree_creation_observed"):
            current, selected = None, None
        if attempt != current:
            continue
        diagnostic = observation_failure_diagnostic(event)
        if diagnostic is not None:
            # Intake follows worker exit; its secondary failure must not hide
            # the more informative process failure from the same attempt.
            if selected is None or diagnostic["phase"] != "unit_result":
                selected = diagnostic
    result: dict[str, object] = {}
    if current is not None:
        result["attempt_id"] = current
    if selected is not None:
        result["failure_diagnostic"] = selected
    return result


def project_run_executor_session(events: Sequence[Mapping[str, object]], *, run_id: str) -> dict[str, object]:
    """Current-attempt receipts in append order; stale replay never retakes ownership."""
    match = re.fullmatch(r'(fanout-[0-9a-f]{12})-(.+)', run_id)
    if match is None:
        return {}
    current: object = None
    seen: set[str] = set()
    result: dict[str, object] = {}
    for event in events:
        if event.get('run_id') != run_id:
            continue
        attempt = event.get('attempt_id')
        name = canonical_observation_event(str(event.get('event', '')))
        if isinstance(attempt, str) and attempt != current:
            if attempt in seen:
                continue
            seen.add(attempt)
            current, result = attempt, {}
        elif attempt is None and name in ('executor_dispatch_observed', 'worktree_creation_observed'):
            current, result = None, {}
        if attempt == current and name == 'executor_session_observed':
            fields = bound_session_fields(event, fanout_id=match[1], unit_id=match[2], run_ref=run_id)
            if fields:
                result = {'attempt_id': current, **fields}
    return result


def failure_diagnostic_text(diagnostic: FailureDiagnostic) -> str:
    """Render only closed diagnostic data, with explicit separate provenance."""
    streams = "; ".join(
        f"{stream['stream']}: {' '.join(stream['text'].splitlines())}"
        for stream in diagnostic["streams"] if stream["text"]
    )
    label = f"{diagnostic['phase']}/{diagnostic['reason']} (exit {diagnostic['returncode']}, {diagnostic['exit_code_source']})"
    return f"{label}; {streams}" if streams else label


def build_observation_event(event: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_observation_event(str(event.get("event", "")))
    observed_at = str(event.get("observed_at") or event.get("updated_at") or utc_now())
    evidence_refs = _evidence_refs(event)
    record: dict[str, Any] = {
        "schema_version": OBSERVATION_EVENT_SCHEMA_VERSION,
        "event_id": str(event.get("event_id") or _event_id(observed_at, event)),
        "target_type": str(event.get("target_type") or ("run" if event.get("run_id") else "runtime")),
        "target_id": str(event.get("target_id") or event.get("run_id") or event.get("session_id") or ""),
        "run_id": str(event.get("run_id") or ""),
        "workflow": str(event.get("workflow") or event.get("skill") or ""),
        "harness": str(event.get("harness") or ""),
        "phase": str(event.get("phase") or ""),
        "event": canonical,
        "status": str(event.get("status") or "observed"),
        "observed_at": observed_at,
        "source": str(event.get("source") or ""),
        "actor": str(event.get("actor") or ""),
        "runtime_profile": str(event.get("runtime_profile") or ""),
        "evidence_refs": evidence_refs,
        "summary": _bounded_summary(event.get("summary", "")),
        "privacy": OBSERVATION_PRIVACY,
    }
    for key in ("plan_artifact", "plan_status", "worktree_ref", "worker_ref"):
        if event.get(key):
            record[key] = str(event[key])
    for key in ("fanout_id", "attempt_id", "invocation_id", "base_sha", "observed_revision"):
        value = _diagnostic_identity(event.get(key))
        if value is not None:
            record[key] = value
    diagnostic = observation_failure_diagnostic(event)
    if diagnostic is not None:
        record["attempt_id"] = diagnostic["attempt_id"]
        record["failure_diagnostic"] = diagnostic
    if canonical == 'executor_session_observed':
        match = re.fullmatch(r'(fanout-[0-9a-f]{12})-(.+)', record['run_id'])
        if match is not None:
            record.update(bound_session_fields(event, fanout_id=match[1], unit_id=match[2], run_ref=record['run_id']))
    if canonical == 'capacity_admission_observed':
        record.update(read_capacity_fields(event))
    if canonical in (REPAIR_ATTEMPT_STARTED_EVENT, REPAIR_ATTEMPT_OBSERVED_EVENT):
        attempt = bounded_repair_attempt(event.get("repair_attempt"))
        checks = bounded_repair_checks(event.get("repair_checks"))
        if attempt is None or checks is None:
            raise ValueError(f"observation_event {canonical} requires a bounded repair_attempt and repair_checks")
        record["repair_attempt"] = attempt
        record["repair_checks"] = checks
        stop_reason = event.get("repair_stop_reason")
        if stop_reason is not None:
            if stop_reason not in REPAIR_BLOCKED_REASONS or canonical != REPAIR_ATTEMPT_OBSERVED_EVENT:
                raise ValueError(f"observation_event {canonical} carries an unsupported repair_stop_reason")
            record["repair_stop_reason"] = stop_reason
    errors = validate_observation_event(record)
    if errors:
        raise ValueError(errors[0])
    return record


def append_observation_event(paths: OmhPaths, event: dict[str, Any]) -> dict[str, Any]:
    record = build_observation_event(event)
    sequence_errors = _validate_observation_event_prerequisites(paths, record, _prior_events_for_record(paths, record))
    if sequence_errors:
        raise ValueError(sequence_errors[0])
    append_jsonl_locked(paths.runtime_journal_events_path, record)
    return record


def read_observation_events_result(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[str]]:
    events, errors = read_jsonl_objects(paths.runtime_journal_events_path)
    for event in events:
        capacity = read_capacity_fields(event)
        event.pop('capacity', None)
        event.pop('capacity_lineage', None)
        event.update(capacity)
        diagnostic = observation_failure_diagnostic(event)
        if "failure_diagnostic" in event:
            del event["failure_diagnostic"]
        if diagnostic is not None:
            event["failure_diagnostic"] = diagnostic
        fields: dict[str, object] = {}
        match = re.fullmatch(r'(fanout-[0-9a-f]{12})-(.+)', str(event.get('run_id', '')))
        if match is not None and event.get('event') == 'executor_session_observed':
            fields = bound_session_fields(event, fanout_id=match[1], unit_id=match[2], run_ref=str(event['run_id']))
        event.pop('executor_session', None)
        event.pop('session_recovery_snapshot', None)
        event.update(fields)
    return events, errors


def read_observation_events(
    paths: OmhPaths,
    *,
    run_id: str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    events, _ = read_observation_events_result(paths)
    if run_id is not None:
        events = [event for event in events if str(event.get("run_id", "")) == run_id]
    if target_type is not None:
        events = [event for event in events if str(event.get("target_type", "")) == target_type]
    if target_id is not None:
        events = [event for event in events if str(event.get("target_id", "")) == target_id]
    return _apply_limit(events, limit)


def validate_observation_event(event: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if event.get("schema_version") != OBSERVATION_EVENT_SCHEMA_VERSION:
        errors.append(f"observation_event schema_version must be {OBSERVATION_EVENT_SCHEMA_VERSION}")
    for key in ("event_id", "target_type", "target_id", "run_id", "event", "status", "observed_at", "privacy"):
        if not isinstance(event.get(key), str):
            errors.append(f"observation_event {key} must be a string")
    if event.get("event") not in CANONICAL_OBSERVATION_EVENTS:
        errors.append(f"observation_event event is unsupported: {event.get('event')!r}")
    if event.get("status") not in OBSERVATION_STATUSES:
        errors.append(f"observation_event status is unsupported: {event.get('status')!r}")
    if event.get("privacy") != OBSERVATION_PRIVACY:
        errors.append("observation_event privacy must be metadata_only")
    if not isinstance(event.get("evidence_refs"), list):
        errors.append("observation_event evidence_refs must be a list")
    else:
        for index, value in enumerate(event.get("evidence_refs", [])):
            if not isinstance(value, str):
                errors.append(f"observation_event evidence_refs[{index}] must be a string")
    if not isinstance(event.get("summary"), str):
        errors.append("observation_event summary must be a string")
    if event.get("target_type") == "run" and not event.get("run_id"):
        errors.append("observation_event target_type=run requires run_id")
    return errors


def validate_observation_journal(paths: OmhPaths, *, run_id: str | None = None) -> dict[str, Any]:
    events, errors = read_observation_events_result(paths)
    filtered: list[tuple[int, dict[str, Any]]] = []
    for index, event in enumerate(events, start=1):
        if run_id and str(event.get("run_id", "")) != run_id:
            continue
        filtered.append((index, event))
        errors.extend(
            f"{paths.runtime_journal_events_path}:{index}: {error}"
            for error in validate_observation_event(event)
        )
    errors.extend(_validate_observation_event_sequence(paths, filtered))
    return {
        "schema_version": "omh_observation_journal_validation/v1",
        "path": str(paths.runtime_journal_events_path),
        "ok": not errors,
        "event_count": len(filtered),
        "errors": errors,
    }


def _prior_events_for_record(paths: OmhPaths, record: dict[str, Any]) -> list[dict[str, Any]]:
    run_id = str(record.get("run_id", ""))
    if not run_id:
        return []
    events, _ = read_observation_events_result(paths)
    return [event for event in events if isinstance(event, dict) and str(event.get("run_id", "")) == run_id]


def _validate_observation_event_sequence(paths: OmhPaths, indexed_events: list[tuple[int, dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    prior_by_run: dict[str, list[dict[str, Any]]] = {}
    for index, event in indexed_events:
        run_id = str(event.get("run_id", ""))
        prior = prior_by_run.setdefault(run_id, [])
        for error in _validate_observation_event_prerequisites(paths, event, prior):
            errors.append(f"{paths.runtime_journal_events_path}:{index}: {error}")
        prior.append(event)
    return errors


def _validate_observation_event_prerequisites(
    paths: OmhPaths,
    event: dict[str, Any],
    prior_events: list[dict[str, Any]],
) -> list[str]:
    run_id = str(event.get("run_id", ""))
    if not run_id or str(event.get("target_type", "")) != "run":
        return []
    event_name = canonical_observation_event(str(event.get("event", "")))
    if event_name not in _ORDERED_LIFECYCLE_EVENTS or str(event.get("status", "observed")) == "not_observed":
        return []
    state = _lifecycle_prerequisite_state(prior_events)
    review_required = _journal_review_required(paths, run_id)
    errors: list[str] = []
    if event_name == "executor_result_observed":
        _require_prerequisites(event_name, state, ["executor_dispatch_observed"], errors)
    elif event_name in {"unit_verification_observed", "verification_result_observed"}:
        _require_prerequisites(
            event_name,
            state,
            ["executor_dispatch_observed", "executor_result_observed"],
            errors,
        )
    elif event_name == "review_result_observed":
        _require_prerequisites(event_name, state, ["verification_result_observed"], errors)
    elif event_name == "ci_result_observed":
        required = ["verification_result_observed"]
        if review_required or state["review_seen"]:
            required.append("review_result_observed")
        _require_prerequisites(event_name, state, required, errors)
    elif event_name == "merge_gate_observed":
        required = [
            "executor_dispatch_observed",
            "executor_result_observed",
            "verification_result_observed",
        ]
        if review_required or state["review_seen"]:
            required.append("review_result_observed")
        if state["review_result_observed"] or state["ci_seen"]:
            required.append("ci_result_observed")
        _require_prerequisites(event_name, state, required, errors)
    elif event_name == "merge_observed":
        required = [
            "executor_dispatch_observed",
            "executor_result_observed",
            "verification_result_observed",
        ]
        if review_required or state["review_seen"]:
            required.append("review_result_observed")
        if state["review_result_observed"] or state["ci_seen"]:
            required.append("ci_result_observed")
        required.append("merge_gate_observed")
        _require_prerequisites(event_name, state, required, errors)
    return errors


_ORDERED_LIFECYCLE_EVENTS = {
    "executor_result_observed",
    "unit_verification_observed",
    "verification_result_observed",
    "review_result_observed",
    "ci_result_observed",
    "merge_gate_observed",
    "merge_observed",
}


def _lifecycle_prerequisite_state(events: list[dict[str, Any]]) -> dict[str, bool]:
    state = {
        "executor_dispatch_observed": False,
        "executor_result_observed": False,
        "verification_result_observed": False,
        "review_result_observed": False,
        "ci_result_observed": False,
        "merge_gate_observed": False,
        "review_seen": False,
        "ci_seen": False,
    }
    for event in events:
        event_name = canonical_observation_event(str(event.get("event", "")))
        status = str(event.get("status", "observed"))
        if event_name == "review_result_observed":
            state["review_seen"] = True
        elif event_name == "ci_result_observed":
            state["ci_seen"] = True
        if status != "observed":
            continue
        if event_name in state:
            state[event_name] = True
    return state


def _require_prerequisites(
    event_name: str,
    state: dict[str, bool],
    required_events: list[str],
    errors: list[str],
) -> None:
    missing = [required for required in required_events if not state.get(required, False)]
    if missing:
        errors.append(f"{event_name} requires {_join_required_events(missing)}")


def _join_required_events(events: list[str]) -> str:
    if len(events) <= 1:
        return events[0] if events else ""
    if len(events) == 2:
        return f"{events[0]} and {events[1]}"
    return f"{', '.join(events[:-1])}, and {events[-1]}"


def _journal_review_required(paths: OmhPaths, run_id: str) -> bool:
    try:
        coding = read_runtime_record(paths.runtime_runs_dir / run_id / CODING_DELEGATION_FILE)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    if not isinstance(coding, dict):
        return False
    handoff = coding.get("executor_handoff")
    review = handoff.get("review") if isinstance(handoff, dict) and isinstance(handoff.get("review"), dict) else {}
    return bool(review.get("required", coding.get("review_required", False)))


def project_run_lifecycle(
    events: list[dict[str, Any]],
    *,
    run_id: str = "",
    workflow: str = "",
    harness: str = "",
    phase: str = "",
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "schema_version": LIFECYCLE_PROJECTION_SCHEMA_VERSION,
        "run_id": run_id,
        "workflow": workflow,
        "harness": harness,
        "phase": phase,
        "prepared_handoff": False,
        "plan_artifact": "",
        "plan_status": "",
        "prompt_dispatched": False,
        "runtime_start_observed": False,
        "worktree_observed": False,
        "execution_observed": False,
        "unit_verification_observed": False,
        "verification_observed": False,
        "review_observed": False,
        "ci_observed": False,
        "merge_gate_observed": False,
        "merge_observed": False,
        "blocked": False,
        "failed": False,
        "cancelled": False,
        "observation_status": "unknown",
        "journal_event_count": 0,
        "latest_event_id": "",
        "latest_event": {},
    }
    for event in events:
        if not isinstance(event, dict):
            continue
        event_run_id = str(event.get("run_id", ""))
        if run_id and event_run_id and event_run_id != run_id:
            continue
        _fold_event(projection, event)
    projection.update(project_run_failure_diagnostic(events, run_id=run_id))
    if projection["journal_event_count"] == 0 and projection["prepared_handoff"]:
        projection["observation_status"] = "prepared_not_observed"
    return projection


def observation_status_from_projection(projection: dict[str, Any], fallback: str = "unknown") -> str:
    status = str(projection.get("observation_status") or "")
    return status if status and status != "unknown" else fallback


def merge_lifecycle_projection(legacy: dict[str, Any], journal: dict[str, Any]) -> dict[str, Any]:
    merged = {**legacy, **journal}
    for key in (
        "prepared_handoff",
        "prompt_dispatched",
        "runtime_start_observed",
        "worktree_observed",
        "execution_observed",
        "unit_verification_observed",
        "verification_observed",
        "review_observed",
        "ci_observed",
        "merge_gate_observed",
        "merge_observed",
        "blocked",
        "failed",
        "cancelled",
    ):
        merged[key] = bool(legacy.get(key)) or bool(journal.get(key))
    for key in ("run_id", "workflow", "harness", "phase", "plan_artifact", "plan_status"):
        merged[key] = journal.get(key) or legacy.get(key, "")
    merged["observation_status"] = _max_status(
        str(legacy.get("observation_status", "unknown")),
        str(journal.get("observation_status", "unknown")),
    )
    merged["journal_event_count"] = int(journal.get("journal_event_count", 0) or 0)
    merged["latest_event_id"] = str(journal.get("latest_event_id", ""))
    merged["latest_event"] = journal.get("latest_event", {})
    return merged


def _fold_event(projection: dict[str, Any], event: dict[str, Any]) -> None:
    status = str(event.get("status", "observed"))
    event_name = canonical_observation_event(str(event.get("event", "")))
    projection["journal_event_count"] = int(projection.get("journal_event_count", 0)) + 1
    projection["latest_event_id"] = str(event.get("event_id", ""))
    projection["latest_event"] = {
        "event": event_name,
        "status": status,
        "summary": str(event.get("summary", "")),
        "observed_at": str(event.get("observed_at", "")),
    }
    if event.get("workflow") and not projection.get("workflow"):
        projection["workflow"] = str(event.get("workflow", ""))
    if event.get("harness") and not projection.get("harness"):
        projection["harness"] = str(event.get("harness", ""))
    if event.get("phase"):
        projection["phase"] = str(event.get("phase", ""))
    if event.get("plan_artifact"):
        projection["plan_artifact"] = str(event.get("plan_artifact", ""))
    if event.get("plan_status"):
        projection["plan_status"] = str(event.get("plan_status", ""))
    if event_name == "plan_accepted":
        projection["plan_status"] = "accepted"
    elif event_name == "plan_revised":
        projection["plan_status"] = "revised"
    elif event_name == "plan_cancelled":
        projection["plan_status"] = "cancelled"
    if status == "blocked":
        projection["blocked"] = True
        projection["observation_status"] = "blocked"
    elif status == "failed":
        projection["failed"] = True
        projection["observation_status"] = "failed"
    elif status == "cancelled":
        # `cancelled` is a member of `OBSERVATION_STATUSES`, so an event can
        # legally carry it. Without this branch the fold ignored the value and
        # fell through to the `!= "observed"` return, and a run someone stopped
        # deliberately projected as a run that had merely not got there yet --
        # the vocabulary member was in the tuple and nothing in the code backed
        # it. The event named `cancelled` was already handled below; a milestone
        # event whose *status* is cancelled was not.
        projection["cancelled"] = True
        projection["observation_status"] = "cancelled"
    if status != "observed":
        return
    if event_name == "prepared_handoff_created":
        projection["prepared_handoff"] = True
        _advance_status(projection, "prepared_not_observed")
    elif event_name == "runtime_start_observed":
        projection["runtime_start_observed"] = True
        _advance_status(projection, "runtime_start_observed")
    elif event_name == "worktree_creation_observed":
        projection["worktree_observed"] = True
        _advance_status(projection, "worktree_creation_observed")
    elif event_name == "executor_dispatch_observed":
        projection["prompt_dispatched"] = True
        _advance_status(projection, "dispatch_observed")
    elif event_name == "executor_result_observed":
        projection["execution_observed"] = True
        _advance_status(projection, "execution_observed")
    elif event_name == "unit_verification_observed":
        # A unit receipt is projected for fanout consumers, but it does not
        # advance or satisfy the existing run-level verification rung.
        projection["unit_verification_observed"] = True
    elif event_name == "verification_result_observed":
        projection["verification_observed"] = True
        _advance_status(projection, "verification_observed")
    elif event_name == "review_result_observed":
        projection["review_observed"] = True
        _advance_status(projection, "review_observed")
    elif event_name == "ci_result_observed":
        projection["ci_observed"] = True
        _advance_status(projection, "ci_observed")
    elif event_name == "merge_gate_observed":
        projection["merge_gate_observed"] = True
        _advance_status(projection, "merge_gate_observed")
    elif event_name == "merge_observed":
        projection["merge_observed"] = True
        _advance_status(projection, "merge_observed")
    elif event_name == "blocked":
        projection["blocked"] = True
        projection["observation_status"] = "blocked"
    elif event_name == "failed":
        projection["failed"] = True
        projection["observation_status"] = "failed"
    elif event_name == "cancelled":
        projection["cancelled"] = True
        projection["observation_status"] = "cancelled"


def _event_id(observed_at: str, event: dict[str, Any]) -> str:
    base = json.dumps(
        {
            "observed_at": observed_at,
            "run_id": event.get("run_id", ""),
            "target_id": event.get("target_id", ""),
            "event": event.get("event", ""),
            "summary": event.get("summary", ""),
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:10]
    return f"{observed_at.replace(':', '').replace('-', '').replace('.', '')}-{digest}-{secrets.token_hex(2)}"


def _evidence_refs(event: dict[str, Any]) -> list[str]:
    refs = event.get("evidence_refs", event.get("evidence_ref", []))
    if isinstance(refs, str):
        refs = [refs]
    if not isinstance(refs, list):
        return []
    return [str(ref) for ref in refs if str(ref)]


def _bounded_summary(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _apply_limit(events: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return events
    if limit < 1:
        return []
    return events[-limit:]


def _advance_status(projection: dict[str, Any], candidate: str) -> None:
    projection["observation_status"] = _max_status(str(projection.get("observation_status", "unknown")), candidate)


def _max_status(current: str, candidate: str) -> str:
    if candidate in {"blocked", "failed", "cancelled"}:
        return candidate
    if current in {"blocked", "failed", "cancelled"}:
        return current
    try:
        return candidate if PROJECTION_ORDER.index(candidate) >= PROJECTION_ORDER.index(current) else current
    except ValueError:
        return candidate if current == "unknown" else current
