from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias, TypedDict, assert_never

from ..workflows.memory import ADVISORY_FRESHNESS_REASONS
from .continuity_state import json_strings as _strings, resume_status_from_evidence

JsonValue: TypeAlias = None | bool | int | float | str | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
WorkspaceReuse: TypeAlias = Literal["allowed", "operator_choice", "blocked", "unknown"]
WorkspaceRequired: TypeAlias = Literal["none", "worktree_recommended", "worktree_required", "unknown"]
WorkspaceCurrent: TypeAlias = Literal["same_workspace", "isolated_worktree", "unobserved", "unknown"]
ResumeStatus: TypeAlias = Literal["not_started", "conversation_safe", "reattach", "blocked"]
MemoryAvailability: TypeAlias = Literal["available", "not_included", "unknown"]
MemoryFreshness: TypeAlias = Literal["not_available", "no_warnings", "warnings_present", "unknown"]
LineDomain: TypeAlias = Literal["workspace", "resume", "memory"]
MemoryLookupStatus: TypeAlias = Literal["present", "absent", "malformed"]

SCHEMA_VERSION: Final = "continuity_briefing/v1"
_MEMORY_SCHEMA_VERSION: Final = "project_memory_recall_pack/v1"
_HANDOFF_KEYS: Final = ("executor_handoff", "runtime_handoff", "prompt_handoff")
_ABSENT_MEMORY_BOUNDARY: Final = "No reviewed project memory was included in this prepared handoff."
_UNKNOWN_MEMORY_BOUNDARY: Final = "Memory continuity is unknown because the available summary was malformed."
_RESUME_NEXT_ACTIONS: Final[Mapping[str, str]] = {"blocked": "review_runtime_evidence", "reattach": "reattach_runtime_evidence"}
_CLAIM_BOUNDARY: Final = (
    "This compact projection reports prepared or observed continuity evidence only; "
    "it does not create a workspace, resume an executor, or prove recalled context was used."
)


class WorkspaceContinuity(TypedDict):
    reuse: WorkspaceReuse
    required: WorkspaceRequired
    current: WorkspaceCurrent


class ResumeContinuity(TypedDict):
    status: ResumeStatus


class MemoryContinuity(TypedDict):
    availability: MemoryAvailability
    included_count: int | None
    excluded_count: int | None
    truncated: bool | None
    freshness: MemoryFreshness
    freshness_warning_count: int | None
    claim_boundary: str


class ContinuityLine(TypedDict):
    domain: LineDomain
    text: str


class ContinuityBriefing(TypedDict):
    schema_version: str
    workspace: WorkspaceContinuity
    resume: ResumeContinuity
    memory: MemoryContinuity
    headline: str
    lines: list[ContinuityLine]
    next_action: str
    claim_boundary: str


def build_continuity_briefing(evidence: Mapping[str, JsonValue]) -> ContinuityBriefing:
    """Project existing handoff evidence into a bounded continuity summary."""
    workspace = _workspace_continuity(evidence)
    memory = _memory_continuity(evidence)
    resume_status = resume_status_from_evidence(evidence)
    workspace_lines: dict[WorkspaceReuse, str] = {
        "allowed": "The workspace policy allows this prepared handoff to continue.",
        "operator_choice": "Choose whether to prepare an isolated workspace before continuing.",
        "blocked": "An isolated workspace is still required before continuing.",
        "unknown": "Workspace reuse is unknown because evidence is missing or malformed.",
    }
    resume_lines: dict[ResumeStatus, str] = {
        "not_started": "This is a fresh handoff; no executor resume is claimed.",
        "conversation_safe": "Terminal evidence allows conversation continuation, not executor resume.",
        "reattach": "Reattach recorded runtime evidence before continuing the conversation.",
        "blocked": "Recorded evidence blocks conversational continuation.",
    }
    memory_lines: dict[MemoryAvailability, str] = {
        "available": (
            f"Reviewed memory included: {memory['included_count']}; excluded: "
            f"{memory['excluded_count']}; freshness warnings: {memory['freshness_warning_count']}."
        ),
        "not_included": "No reviewed project memory was included in this handoff.",
        "unknown": "Memory continuity is unknown because its summary was malformed.",
    }
    next_actions: dict[WorkspaceReuse, str] = {
        "allowed": "continue_prepared_handoff",
        "operator_choice": "choose_workspace_isolation",
        "blocked": "prepare_isolated_workspace",
        "unknown": "review_workspace_evidence",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "workspace": workspace,
        "resume": {"status": resume_status},
        "memory": memory,
        "headline": "Coding continuity is ready to review.",
        "lines": [
            {"domain": "workspace", "text": workspace_lines[workspace["reuse"]]},
            {"domain": "resume", "text": resume_lines[resume_status]},
            {"domain": "memory", "text": memory_lines[memory["availability"]]},
        ],
        "next_action": _RESUME_NEXT_ACTIONS.get(resume_status, next_actions[workspace["reuse"]]),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _workspace_continuity(evidence: Mapping[str, JsonValue]) -> WorkspaceContinuity:
    isolation_plan = _isolation_plan(evidence)
    observation = _mapping(evidence.get("workspace_isolation"))
    runtime_observation = _mapping(evidence.get("runtime_observation"))
    if ("workspace_isolation" in evidence and observation is None) or (
        "runtime_observation" in evidence and runtime_observation is None
    ):
        return {"reuse": "unknown", "required": "unknown", "current": "unknown"}
    worktree_denied = any(
        "worktree_creation" in _strings(runtime_observation.get(key) if runtime_observation else None)
        for key in ("failed_events", "blocked_events")
    )
    observed_events = _strings(runtime_observation.get("observed_events") if runtime_observation else None)
    observed_current = str(observation.get("current", "")) if observation else ""
    observed_status = str(observation.get("status", "")) if observation else ""
    observed_strategy = str(observation.get("strategy", "")) if observation else ""
    plan_strategy = str(isolation_plan.get("strategy", "")) if isolation_plan else ""

    isolated = not worktree_denied and (
        observed_current == "isolated_worktree"
        or "worktree_creation" in observed_events
        or (
            observed_status == "observed"
            and observed_strategy in {"worktree_recommended", "worktree_required"}
        )
    )
    if isolated:
        return {"reuse": "allowed", "required": "none", "current": "isolated_worktree"}

    strategy_states: dict[str, tuple[WorkspaceReuse, WorkspaceRequired]] = {
        "same_workspace_ok": ("allowed", "none"),
        "worktree_recommended": ("operator_choice", "worktree_recommended"),
        "worktree_required": ("blocked", "worktree_required"),
    }
    strategy_state = strategy_states.get(plan_strategy)
    if strategy_state is None:
        return {"reuse": "unknown", "required": "unknown", "current": "unknown"}
    same_workspace = observed_current == "same_workspace" or (
        observed_status == "not_required"
        and (observed_strategy or plan_strategy) == "same_workspace_ok"
    )
    return {
        "reuse": strategy_state[0],
        "required": strategy_state[1],
        "current": "same_workspace" if same_workspace else "unobserved",
    }


def _isolation_plan(evidence: Mapping[str, JsonValue]) -> Mapping[str, JsonValue] | None:
    direct = _mapping(evidence.get("isolation_plan"))
    if direct is not None:
        return direct
    for key in _HANDOFF_KEYS:
        handoff = _mapping(evidence.get(key))
        if handoff is not None:
            plan = _mapping(handoff.get("isolation_plan"))
            if plan is not None:
                return plan
    return None


def _memory_continuity(evidence: Mapping[str, JsonValue]) -> MemoryContinuity:
    status, pack = _memory_pack(evidence)
    match status:
        case "absent":
            return {
                "availability": "not_included",
                "included_count": 0,
                "excluded_count": 0,
                "truncated": None,
                "freshness": "not_available",
                "freshness_warning_count": 0,
                "claim_boundary": _ABSENT_MEMORY_BOUNDARY,
            }
        case "malformed":
            return _unknown_memory()
        case "present":
            pass
        case unreachable:
            assert_never(unreachable)
    if pack is None:
        return _unknown_memory()

    included = _pack_count(pack, "record_count", "included_records")
    excluded = _pack_count(pack, "excluded_count", "excluded_records")
    warning_count = _warning_count(pack)
    boundary = pack.get("claim_boundary")
    raw_pack = "included_records" in pack or "excluded_records" in pack
    truncated_value = pack.get("truncated")
    truncated = truncated_value if raw_pack and isinstance(truncated_value, bool) else None
    valid_truncation = not raw_pack or isinstance(truncated_value, bool)
    if (
        pack.get("schema_version") != _MEMORY_SCHEMA_VERSION
        or included is None
        or excluded is None
        or warning_count is None
        or not valid_truncation
        or not isinstance(boundary, str)
        or not boundary
    ):
        return _unknown_memory()
    return {
        "availability": "available",
        "included_count": included,
        "excluded_count": excluded,
        "truncated": truncated,
        "freshness": "warnings_present" if warning_count else "no_warnings",
        "freshness_warning_count": warning_count,
        "claim_boundary": boundary,
    }


def _memory_pack(
    evidence: Mapping[str, JsonValue],
) -> tuple[MemoryLookupStatus, Mapping[str, JsonValue] | None]:
    containers: list[Mapping[str, JsonValue]] = [evidence]
    for key in _HANDOFF_KEYS:
        if key in evidence:
            handoff = _mapping(evidence.get(key))
            if handoff is None:
                return "malformed", None
            containers.append(handoff)
    work_summary = _mapping(evidence.get("work_summary"))
    handoff_contract = _mapping(work_summary.get("handoff_contract")) if work_summary else None
    if handoff_contract is not None:
        containers.append(handoff_contract)
    for container in containers:
        if "memory_recall_pack" not in container:
            continue
        pack = _mapping(container.get("memory_recall_pack"))
        if pack is None:
            return "malformed", None
        return ("present", pack) if pack else ("absent", None)
    return "absent", None


def _pack_count(
    pack: Mapping[str, JsonValue], count_key: str, list_key: str
) -> int | None:
    count = _nonnegative_int(pack.get(count_key))
    if count is not None:
        return count
    values = pack.get(list_key)
    return len(values) if _is_json_list(values) else None


def _warning_count(pack: Mapping[str, JsonValue]) -> int | None:
    count = _nonnegative_int(pack.get("freshness_warning_count"))
    if count is not None:
        return count
    warnings = pack.get("freshness_warnings")
    if not _is_json_list(warnings):
        return None
    # Advisory notices (a delivered record approaching its review deadline)
    # describe a still-fresh pack; counting them would flip `freshness` to
    # `warnings_present` for a store where nothing is actually unconfirmed.
    # An entry that is not a mapping still counts -- unknown shapes fail
    # toward warning, never toward silence.
    return sum(
        1
        for warning in warnings
        if not (isinstance(warning, Mapping) and str(warning.get("reason_code", "")) in ADVISORY_FRESHNESS_REASONS)
    )


def _unknown_memory() -> MemoryContinuity:
    return {
        "availability": "unknown",
        "included_count": None,
        "excluded_count": None,
        "truncated": None,
        "freshness": "unknown",
        "freshness_warning_count": None,
        "claim_boundary": _UNKNOWN_MEMORY_BOUNDARY,
    }


def _mapping(value: JsonValue | None) -> Mapping[str, JsonValue] | None:
    return value if isinstance(value, Mapping) else None


def _is_json_list(value: JsonValue | None) -> bool:
    return isinstance(value, list)


def _nonnegative_int(value: JsonValue | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
