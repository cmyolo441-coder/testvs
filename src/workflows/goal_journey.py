from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Final

from ..paths import OmhPaths
from ..runtime.artifacts import list_wrapper_session_records, summarize_delegated_coding_status
from .goal_ledger import build_goal_completion_gate, read_goal_ledger
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


GOAL_JOURNEY_SCHEMA_VERSION = "goal_journey/v1"

# The five states the issue asks a resumed conversation to tell apart, plus
# `cancelled`. Folding a cancelled goal into `blocked` would read as "still
# working on it" for a goal nobody may work on again, so it gets its own name.
GOAL_JOURNEY_STAGES = (
    "intent",
    "preparation",
    "activity",
    "blocked",
    "verified_complete",
    "cancelled",
)

GOAL_JOURNEY_GATE_KINDS = (
    "acceptance_criterion",
    "active_blocker",
    "linked_runtime_run",
    "slop_cleanup_record",
    "goal_status",
)

GOAL_JOURNEY_CRITERION_STATUSES = ("pending", "satisfied")

GOAL_JOURNEY_FRESHNESS_STATES = ("unknown", "fresh", "stale")

# Newest-last tail bound on the checkpoint list. A goal that spans months
# accumulates checkpoints without bound, and this projection is meant to be
# read again in every later conversation. Bounding output never changes a
# verdict: criteria, gates, and the stage are derived from the full checkpoint
# list before this limit is applied.
GOAL_JOURNEY_CHECKPOINT_LIMIT: Final[int] = 20

# One window, named once. Freshness is only computed when the caller passes
# `now`; the default projection reports "unknown" so two reads of an unchanged
# goal produce the identical payload.
GOAL_JOURNEY_FRESH_WINDOW_SECONDS: Final[int] = 7 * 24 * 60 * 60

GOAL_JOURNEY_CLAIM_BOUNDARY: Final[str] = (
    "A goal journey is a read-only projection over local goal, wrapper-session, and runtime "
    "metadata. It reports what was recorded and linked, not what was verified, and it is not "
    "execution, review, CI, or merge evidence."
)

GOAL_JOURNEY_NOT_EVIDENCE = (
    "goal_completion",
    "criterion_verification",
    "handoff_execution",
    "review_execution",
    "ci_execution",
    "merge_execution",
)

_GOAL_JOURNEY_STAGE_SUMMARIES = {
    "intent": "The goal is recorded; no preparation or activity is linked yet.",
    "preparation": "A handoff is prepared for this goal; execution is not observed.",
    "activity": "Work has been recorded against this goal.",
    "blocked": "Progress is blocked; completion cannot be claimed from this state.",
    "verified_complete": "Every required gate carries accepted evidence.",
    "cancelled": "The goal was cancelled; it is terminal and cannot be completed.",
}

_GOAL_ACTIVE_BLOCKED_STATUSES = {"blocked", "failed"}
_GOAL_STATUS_GATE_STATUSES = ("blocked", "failed", "cancelled")


def build_goal_journey(paths: OmhPaths, goal_id: str, *, now: str = "") -> dict[str, Any]:
    """Link one goal to the sessions, plans, handoffs, owners, and runs behind it.

    Read-only: nothing here writes or mutates goal state. Every edge is derived
    from artifacts that already exist -- the ledger names its linked runtime
    runs, a wrapper session names the run it currently owns, and a run names the
    handoff and owner it was prepared for -- rather than from a new writer.

    `now` is a parameter, never a wall-clock read inside the payload, so two
    projections of an unchanged goal compare equal. Omitted, evidence freshness
    reports "unknown" instead of guessing.
    """
    goal = read_goal_ledger(paths, goal_id)
    gate = build_goal_completion_gate(paths, goal_id)
    checkpoints = [item for item in goal["checkpoints"] if isinstance(item, dict)]
    run_ids = [str(run_id) for run_id in goal.get("linked_runtime_runs", [])]
    checks_by_run_id = {str(check.get("run_id", "")): check for check in gate["linked_runtime_checks"]}
    runs = [_journey_run(paths, run_id, checks_by_run_id.get(run_id, {})) for run_id in run_ids]
    session_records = _linked_session_records(paths, set(run_ids))
    sessions = [_journey_session(session) for session in session_records]
    plans = [plan for plan in (_journey_plan(session) for session in session_records) if plan]
    handoffs = _journey_handoffs(session_records, runs)
    owners = _journey_owners(session_records, runs)
    criteria = [_journey_criterion(criterion, checkpoints, now=now) for criterion in goal["acceptance_criteria"]]
    required_gates = _required_gates(goal, criteria, gate)
    blocking_gate_ids = [item["gate_id"] for item in required_gates if not item["evidence_accepted"]]
    stage = _journey_stage(goal, runs, handoffs, blocking_gate_ids)
    return {
        "schema_version": GOAL_JOURNEY_SCHEMA_VERSION,
        "goal_id": str(goal["goal_id"]),
        "goal_status": str(goal["status"]),
        # The stable identity a later conversation resumes on, carried without
        # the objective text the ledger deliberately never stored.
        "objective_hash": str(goal["objective_hash"]),
        "objective_summary": str(goal["objective_summary"]),
        "source": str(goal.get("source", "")),
        "created_at": str(goal.get("created_at", "")),
        "updated_at": str(goal.get("updated_at", "")),
        "stage": stage,
        "stage_summary": _GOAL_JOURNEY_STAGE_SUMMARIES[stage],
        "sessions": sessions,
        "plans": plans,
        "handoffs": handoffs,
        "owners": owners,
        "runs": runs,
        "checkpoints": [_journey_checkpoint(item) for item in checkpoints[-GOAL_JOURNEY_CHECKPOINT_LIMIT:]],
        "checkpoint_history": _checkpoint_history(len(checkpoints)),
        "criteria": criteria,
        "required_gates": required_gates,
        "completion": _journey_completion(gate, required_gates, blocking_gate_ids),
        "resume": _journey_resume(goal, stage, gate, blocking_gate_ids),
        "claim_boundary": GOAL_JOURNEY_CLAIM_BOUNDARY,
        "not_evidence": list(GOAL_JOURNEY_NOT_EVIDENCE),
    }


def validate_goal_journey(payload: dict[str, Any]) -> list[str]:
    """Errors that make a journey payload unsafe to report, empty when it is sound.

    Two of the checks are the issue's acceptance criteria written down as
    invariants rather than as tests only: a criterion may not read as satisfied
    without accepted evidence, and completion may not read as ready while any
    required gate lacks evidence.
    """
    errors: list[str] = []
    if payload.get("schema_version") != GOAL_JOURNEY_SCHEMA_VERSION:
        errors.append("schema_version must be goal_journey/v1")
    if not str(payload.get("goal_id", "")).strip():
        errors.append("goal_id is required")
    if "objective" in payload:
        errors.append("raw objective field is not allowed")
    if not _is_sha256_hex(str(payload.get("objective_hash", ""))):
        errors.append("objective_hash must be a sha256 hex digest")
    if payload.get("stage") not in GOAL_JOURNEY_STAGES:
        errors.append("stage is unsupported")
    for key in ("sessions", "plans", "handoffs", "owners", "runs", "checkpoints", "criteria", "required_gates"):
        if not isinstance(payload.get(key), list):
            errors.append(f"{key} must be a list")
    errors.extend(_criteria_errors(payload.get("criteria")))
    errors.extend(_required_gate_errors(payload.get("required_gates")))
    errors.extend(_completion_errors(payload.get("completion")))
    if str(payload.get("claim_boundary", "")) != GOAL_JOURNEY_CLAIM_BOUNDARY:
        errors.append("claim_boundary must deny that the projection is execution, review, CI, or merge evidence")
    not_evidence = payload.get("not_evidence")
    if not isinstance(not_evidence, list) or not set(GOAL_JOURNEY_NOT_EVIDENCE).issubset(set(not_evidence)):
        errors.append("not_evidence must include all goal journey boundaries")
    return errors


def render_goal_journey_text(journey: dict[str, Any]) -> str:
    """The journey as exact lines for a terminal or a messenger relay.

    Dash lines only. A markdown table is dropped outright by Slack and Telegram,
    which is where a resumed goal is most often read.
    """
    completion = journey.get("completion") if isinstance(journey.get("completion"), dict) else {}
    lines = [
        f"Goal {journey.get('goal_id', '')} — {journey.get('stage', '')}",
        "",
        str(journey.get("objective_summary", "") or "(no objective recorded)"),
        "",
        f"Stage: {journey.get('stage', '')} — {journey.get('stage_summary', '')}",
        f"Completion: {'ready' if completion.get('ready') else 'blocked'} "
        f"({int(completion.get('unsatisfied_required_gates', 0))} required gates lack evidence)",
        f"Next action: {completion.get('next_action', '') or 'unknown'}",
    ]
    lines.extend(_journey_text_block("Criteria", journey.get("criteria"), _criterion_line))
    lines.extend(_journey_text_block("Blocking gates", _blocking_gates(journey), _gate_line))
    lines.extend(_journey_text_block("Sessions", journey.get("sessions"), _session_line))
    lines.extend(_journey_text_block("Handoffs", journey.get("handoffs"), _handoff_line))
    lines.extend(_journey_text_block("Runs", journey.get("runs"), _run_line))
    lines.extend(_journey_text_block("Owners", journey.get("owners"), _owner_line))
    lines.extend(_journey_text_block("Checkpoints", journey.get("checkpoints"), _checkpoint_line))
    resume = journey.get("resume") if isinstance(journey.get("resume"), dict) else {}
    if resume:
        lines.extend(["", f"Resume: {resume.get('continue_command', '')}"])
    lines.extend(["", GOAL_JOURNEY_CLAIM_BOUNDARY])
    return "\n".join(lines).strip()


def _journey_run(paths: OmhPaths, run_id: str, check: dict[str, Any]) -> dict[str, Any]:
    status = _delegated_status(paths, run_id)
    prepared = status.get("prepared", {}) if status else {}
    execution = status.get("execution", {}) if status else {}
    return {
        "run_id": run_id,
        "found": status is not None,
        "workflow": str(prepared.get("workflow", "") or ""),
        "harness": str(prepared.get("harness", "") or ""),
        "owner": str(prepared.get("executor_target", "") or ""),
        "handoff_available": bool(prepared.get("handoff_available", False)),
        "handoff_schema_version": str(prepared.get("handoff_schema_version") or ""),
        "execution_observed": bool(execution.get("observed", False)),
        "next_action": str(status.get("next_action", "")) if status else "record_runtime_evidence",
        "evidence_accepted": bool(check.get("satisfied", False)),
        "summary": str(check.get("summary", "") or "Linked runtime run was not checked."),
    }


def _delegated_status(paths: OmhPaths, run_id: str) -> dict[str, Any] | None:
    # Mirrors `_delegated_runtime_status` in goal_ledger: a linked run the user
    # deleted is a missing edge, not a crashed projection.
    try:
        return summarize_delegated_coding_status(paths, run_id)
    except FileNotFoundError:
        return None


def _linked_session_records(paths: OmhPaths, run_ids: set[str]) -> list[dict[str, Any]]:
    """Wrapper sessions that own one of the goal's linked runtime runs.

    `current_run_id` is the only session-to-run edge the store already keeps, and
    runtime validation refuses two sessions claiming one run, so this is a real
    ownership link rather than a guess.
    """
    matched = [
        session
        for session in list_wrapper_session_records(paths)
        if str(session.get("current_run_id", "")) and str(session.get("current_run_id", "")) in run_ids
    ]
    return sorted(matched, key=lambda session: str(session.get("session_id", "")))


def _journey_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": str(session.get("session_id", "")),
        "source": str(session.get("source", "")),
        "status": str(session.get("status", "")),
        "decision": str(session.get("decision", "")),
        "linked_run_id": str(session.get("current_run_id", "")),
        # Chat identity without the chat: the ledger stores no transcript and
        # neither does this.
        "message_sha256": str(session.get("message_sha256", "")),
        "link_reason": "wrapper session current_run_id matches a linked runtime run",
    }


def _journey_plan(session: dict[str, Any]) -> dict[str, Any]:
    plan = session.get("plan")
    if not isinstance(plan, dict) or not plan:
        return {}
    return {
        "session_id": str(session.get("session_id", "")),
        "linked_run_id": str(session.get("current_run_id", "")),
        "status": str(plan.get("status", "")),
        "recommended_workflow": str(plan.get("recommended_workflow", "")),
        "recommended_harness": str(plan.get("recommended_harness", "")),
        "decision": str(session.get("decision", "")),
    }


def _journey_handoffs(session_records: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    execution_by_run_id = {run["run_id"]: run["execution_observed"] for run in runs}
    entries: list[dict[str, Any]] = []
    for run in runs:
        if not run["handoff_available"]:
            continue
        entries.append(
            {
                "origin": "runtime_run",
                "source_ref": run["run_id"],
                "kind": "prepared_coding_delegation",
                "schema_version": run["handoff_schema_version"],
                "owner": run["owner"],
                "dispatchable": False,
                "execution_observed": run["execution_observed"],
            }
        )
    for session in session_records:
        run_id = str(session.get("current_run_id", ""))
        for kind in ("prompt_handoff", "runtime_handoff"):
            handoff = session.get(kind)
            if not isinstance(handoff, dict) or not handoff:
                continue
            entries.append(
                {
                    "origin": "wrapper_session",
                    "source_ref": str(session.get("session_id", "")),
                    "kind": kind,
                    "schema_version": str(handoff.get("schema_version", "")),
                    "owner": str(handoff.get("selected_executor_profile", "")),
                    "dispatchable": bool(handoff.get("dispatchable", False)),
                    "execution_observed": bool(execution_by_run_id.get(run_id, False)),
                }
            )
    return sorted(entries, key=lambda entry: (entry["origin"], entry["source_ref"], entry["kind"]))


def _journey_owners(session_records: list[dict[str, Any]], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Who was named as the coding owner, and by which artifact.

    Executor-neutral by construction: the owner strings come from whatever the
    run or session recorded, so `codex`, `claude-code`, a runtime profile, and
    `generic` all project identically.
    """
    refs_by_owner: dict[str, set[str]] = {}
    for run in runs:
        if run["found"] and run["owner"]:
            refs_by_owner.setdefault(run["owner"], set()).add(f"run:{run['run_id']}")
    for session in session_records:
        owner = str(session.get("selected_executor_profile") or "")
        if owner:
            refs_by_owner.setdefault(owner, set()).add(f"session:{session.get('session_id', '')}")
    return [
        {"owner": owner, "refs": sorted(refs)}
        for owner, refs in sorted(refs_by_owner.items())
    ]


def _journey_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    evidence_refs = [str(ref) for ref in checkpoint.get("evidence_refs", [])]
    return {
        "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
        "created_at": str(checkpoint.get("created_at", "")),
        "status": str(checkpoint.get("status", "")),
        "summary": str(checkpoint.get("summary", "")),
        "criteria_refs": [str(ref) for ref in checkpoint.get("criteria_refs", [])],
        "evidence_refs": evidence_refs,
        "evidence_accepted": bool(evidence_refs),
        "linked_runtime_run_id": str(checkpoint.get("linked_runtime_run_id", "")),
    }


def _checkpoint_history(total: int) -> dict[str, Any]:
    returned = min(total, GOAL_JOURNEY_CHECKPOINT_LIMIT)
    return {
        "total": total,
        "returned": returned,
        "limit": GOAL_JOURNEY_CHECKPOINT_LIMIT,
        "truncated": total > returned,
        "order": "oldest_to_newest_tail",
    }


def _journey_criterion(criterion: dict[str, Any], checkpoints: list[dict[str, Any]], *, now: str) -> dict[str, Any]:
    criterion_id = str(criterion.get("id", ""))
    accepting = _accepting_checkpoints(criterion_id, checkpoints)
    evidence_refs = [str(ref) for ref in criterion.get("evidence_refs", [])]
    ledger_status = str(criterion.get("status", ""))
    # Three conditions, all required: the ledger says satisfied, the criterion
    # carries evidence refs, and a done checkpoint actually referenced this
    # criterion while carrying evidence. A ledger hand-edited to "satisfied"
    # therefore stays pending here, which is the point of the projection.
    accepted = ledger_status == "satisfied" and bool(evidence_refs) and bool(accepting)
    recorded_at = max((str(item.get("created_at", "")) for item in accepting), default="")
    return {
        "id": criterion_id,
        "summary": str(criterion.get("summary", "")),
        "required": bool(criterion.get("required", True)),
        "ledger_status": ledger_status,
        "journey_status": "satisfied" if accepted else "pending",
        "evidence_accepted": accepted,
        "evidence_refs": evidence_refs,
        "satisfied_by_checkpoints": [str(item.get("checkpoint_id", "")) for item in accepting],
        "evidence_recorded_at": recorded_at,
        **_evidence_freshness(recorded_at, now),
    }


def _accepting_checkpoints(criterion_id: str, checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        checkpoint
        for checkpoint in checkpoints
        if str(checkpoint.get("status", "")) == "done"
        and criterion_id in [str(ref) for ref in checkpoint.get("criteria_refs", [])]
        and checkpoint.get("evidence_refs")
    ]


def _evidence_freshness(recorded_at: str, now: str) -> dict[str, Any]:
    recorded = _parse_time(recorded_at)
    current = _parse_time(now)
    if recorded is None or current is None:
        return {"evidence_age_seconds": None, "evidence_freshness": "unknown"}
    age = int((current - recorded).total_seconds())
    if age < 0:
        # Evidence stamped after `now` says the two clocks disagree, not that
        # the evidence is fresh.
        return {"evidence_age_seconds": None, "evidence_freshness": "unknown"}
    return {
        "evidence_age_seconds": age,
        "evidence_freshness": "fresh" if age <= GOAL_JOURNEY_FRESH_WINDOW_SECONDS else "stale",
    }


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _required_gates(
    goal: dict[str, Any], criteria: list[dict[str, Any]], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Every gate that must carry evidence before completion, and whether it does.

    The kinds mirror what `build_goal_completion_gate` already refuses to pass,
    so the journey can never report a gate as satisfied that the ledger blocks
    on. The criterion entries are stricter than the ledger's own check, which
    only makes the journey block more, never less.
    """
    gates: list[dict[str, Any]] = [
        {
            "gate_id": f"criterion:{criterion['id']}",
            "kind": "acceptance_criterion",
            "summary": criterion["summary"],
            "evidence_accepted": criterion["evidence_accepted"],
            "evidence_refs": criterion["evidence_refs"],
        }
        for criterion in criteria
        if criterion["required"]
    ]
    gates.extend(
        {
            "gate_id": f"blocker:{blocker['id']}",
            "kind": "active_blocker",
            "summary": blocker["summary"],
            "evidence_accepted": False,
            "evidence_refs": [],
        }
        for blocker in gate["active_blockers"]
    )
    gates.extend(
        {
            "gate_id": f"runtime_run:{check['run_id']}",
            "kind": "linked_runtime_run",
            "summary": check["summary"],
            "evidence_accepted": bool(check["satisfied"]),
            "evidence_refs": [],
        }
        for check in gate["linked_runtime_checks"]
    )
    cleanup = gate.get("slop_cleanup_check")
    if isinstance(cleanup, dict):
        gates.append(
            {
                "gate_id": f"slop_cleanup:{cleanup['criterion_id']}",
                "kind": "slop_cleanup_record",
                "summary": cleanup["summary"],
                "evidence_accepted": bool(cleanup["satisfied"]),
                "evidence_refs": [f"run:{cleanup['run_id']}"] if cleanup["run_id"] else [],
            }
        )
    status = str(goal["status"])
    if status in _GOAL_STATUS_GATE_STATUSES:
        gates.append(
            {
                "gate_id": f"goal_status:{status}",
                "kind": "goal_status",
                "summary": f"Goal status is {status}.",
                "evidence_accepted": False,
                "evidence_refs": [],
            }
        )
    return gates


def _journey_completion(
    gate: dict[str, Any], required_gates: list[dict[str, Any]], blocking_gate_ids: list[str]
) -> dict[str, Any]:
    return {
        # Both conditions, never one: the journey may be stricter than the
        # ledger gate, and it must never be looser.
        "ready": bool(gate["ready"]) and not blocking_gate_ids,
        "ledger_gate_ready": bool(gate["ready"]),
        "next_action": str(gate["next_action"]),
        "summary": str(gate["summary"]),
        "required_gates_total": len(required_gates),
        "unsatisfied_required_gates": len(blocking_gate_ids),
        "blocking_gate_ids": list(blocking_gate_ids),
    }


def _journey_resume(
    goal: dict[str, Any], stage: str, gate: dict[str, Any], blocking_gate_ids: list[str]
) -> dict[str, Any]:
    goal_id = str(goal["goal_id"])
    return {
        "goal_id": goal_id,
        "objective_hash": str(goal["objective_hash"]),
        "stage": stage,
        "next_action": str(gate["next_action"]),
        "remaining_required_gates": len(blocking_gate_ids),
        "journey_command": f"omh goal journey --goal {goal_id}",
        "continue_command": f"omh goal continue --goal {goal_id}",
    }


def _journey_stage(
    goal: dict[str, Any], runs: list[dict[str, Any]], handoffs: list[dict[str, Any]], blocking_gate_ids: list[str]
) -> str:
    status = str(goal["status"])
    if status == "cancelled":
        return "cancelled"
    if status == "complete":
        # A ledger that says complete while a required gate still lacks
        # evidence reads as blocked here. Reporting it as verified is the exact
        # overclaim this projection exists to refuse.
        return "verified_complete" if not blocking_gate_ids else "blocked"
    if status in _GOAL_ACTIVE_BLOCKED_STATUSES:
        return "blocked"
    if any(blocker.get("status") == "active" for blocker in goal["blockers"]):
        return "blocked"
    if goal["checkpoints"] or any(run["execution_observed"] for run in runs):
        return "activity"
    if handoffs:
        return "preparation"
    return "intent"


def _criteria_errors(criteria: Any) -> list[str]:
    if not isinstance(criteria, list):
        return []
    errors: list[str] = []
    for index, criterion in enumerate(criteria, start=1):
        if not isinstance(criterion, dict):
            errors.append(f"criteria[{index}] must be an object")
            continue
        if criterion.get("journey_status") not in GOAL_JOURNEY_CRITERION_STATUSES:
            errors.append(f"criteria[{index}].journey_status is invalid")
        if criterion.get("evidence_freshness") not in GOAL_JOURNEY_FRESHNESS_STATES:
            errors.append(f"criteria[{index}].evidence_freshness is invalid")
        if criterion.get("journey_status") == "satisfied" and not _criterion_has_accepted_evidence(criterion):
            errors.append(f"criteria[{index}] is satisfied without accepted evidence")
    return errors


def _criterion_has_accepted_evidence(criterion: dict[str, Any]) -> bool:
    return (
        bool(criterion.get("evidence_accepted"))
        and bool(criterion.get("evidence_refs"))
        and bool(criterion.get("satisfied_by_checkpoints"))
    )


def _required_gate_errors(required_gates: Any) -> list[str]:
    if not isinstance(required_gates, list):
        return []
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(required_gates, start=1):
        if not isinstance(item, dict):
            errors.append(f"required_gates[{index}] must be an object")
            continue
        gate_id = str(item.get("gate_id", ""))
        if not gate_id:
            errors.append(f"required_gates[{index}].gate_id is required")
        elif gate_id in seen:
            errors.append(f"duplicate required gate id: {gate_id}")
        seen.add(gate_id)
        if item.get("kind") not in GOAL_JOURNEY_GATE_KINDS:
            errors.append(f"required_gates[{index}].kind is unsupported")
        if not isinstance(item.get("evidence_accepted"), bool):
            errors.append(f"required_gates[{index}].evidence_accepted must be boolean")
    return errors


def _completion_errors(completion: Any) -> list[str]:
    if not isinstance(completion, dict):
        return ["completion must be an object"]
    errors: list[str] = []
    blocking = completion.get("blocking_gate_ids")
    if not isinstance(blocking, list):
        errors.append("completion.blocking_gate_ids must be a list")
        blocking = []
    if not isinstance(completion.get("ready"), bool):
        errors.append("completion.ready must be boolean")
    if not isinstance(completion.get("ledger_gate_ready"), bool):
        errors.append("completion.ledger_gate_ready must be boolean")
    if completion.get("unsatisfied_required_gates") != len(blocking):
        errors.append("completion.unsatisfied_required_gates must count the blocking gates")
    if completion.get("ready") and blocking:
        errors.append("completion.ready must be false while a required gate lacks evidence")
    if completion.get("ready") and not completion.get("ledger_gate_ready"):
        errors.append("completion.ready must not be true while the ledger completion gate is not ready")
    return errors


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _blocking_gates(journey: dict[str, Any]) -> list[dict[str, Any]]:
    gates = journey.get("required_gates")
    if not isinstance(gates, list):
        return []
    return [gate for gate in gates if isinstance(gate, dict) and not gate.get("evidence_accepted")]


def _journey_text_block(label: str, items: Any, line: Any) -> list[str]:
    if not isinstance(items, list) or not items:
        return ["", f"No {label.lower()} recorded."]
    return ["", f"{label}:", *(f"- {line(item)}" for item in items if isinstance(item, dict))]


def _criterion_line(criterion: dict[str, Any]) -> str:
    evidence = "evidence accepted" if criterion.get("evidence_accepted") else "no accepted evidence"
    return (
        f"{criterion.get('id', '')}: {criterion.get('summary', '')} — "
        f"{criterion.get('journey_status', '')}, {evidence}, "
        f"freshness {criterion.get('evidence_freshness', 'unknown')}"
    )


def _gate_line(gate: dict[str, Any]) -> str:
    return f"{gate.get('gate_id', '')} ({gate.get('kind', '')}): {gate.get('summary', '')}"


def _session_line(session: dict[str, Any]) -> str:
    return (
        f"{session.get('session_id', '')}: {session.get('status', '')}, "
        f"decision {session.get('decision', '')}, run {session.get('linked_run_id', '')}"
    )


def _handoff_line(handoff: dict[str, Any]) -> str:
    observed = "execution observed" if handoff.get("execution_observed") else "execution not observed"
    return (
        f"{handoff.get('origin', '')} {handoff.get('source_ref', '')}: {handoff.get('kind', '')}, "
        f"owner {handoff.get('owner', '') or 'unknown'}, {observed}"
    )


def _run_line(run: dict[str, Any]) -> str:
    evidence = "evidence accepted" if run.get("evidence_accepted") else "evidence missing"
    return (
        f"{run.get('run_id', '')}: owner {run.get('owner', '') or 'unknown'}, "
        f"workflow {run.get('workflow', '') or 'unknown'}, {evidence}"
    )


def _owner_line(owner: dict[str, Any]) -> str:
    return f"{owner.get('owner', '')}: {', '.join(str(ref) for ref in owner.get('refs', []))}"


def _checkpoint_line(checkpoint: dict[str, Any]) -> str:
    evidence = "observed" if checkpoint.get("evidence_accepted") else "prepared"
    return f"{checkpoint.get('checkpoint_id', '')}: {checkpoint.get('summary', '')} — {checkpoint.get('status', '')}, {evidence}"
