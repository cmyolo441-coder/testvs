"""Deterministic coordination board projection (`coordination_board/v1`).

Answers one question — "what is moving, blocked, dependency-gated, or ready
next?" — by projecting artifacts OMH already records into one ordered board of
work items. Nothing here creates a second execution ledger: goal ledgers, fanout
contracts and their dispatch summaries, and the runtime observation journal stay
the source of truth, and this module only reads them.

Boundaries, in order of importance:

- A status word is a claim, never evidence. `done`, `completed`, `merge_ready`
  and their siblings move an item out of the ready lanes, but verification,
  review, CI, and merge stay listed in `missing_evidence` until the observation
  journal carries the matching observed event. `evidence_complete` is therefore
  unreachable from a status word alone. See `claims_completion`.
- A satisfied dependency is not evidence either. Fanout's own dependency bar is
  "the owner agent process exited 0"; this board reuses that vocabulary to
  decide gating and keeps it strictly separate from the evidence ladder.
- Deterministic. `now` is a parameter so callers and tests pin it, and
  `board_digest` covers the ordered items alone — never `observed_at` — so
  identical artifacts hash identically no matter when the board was taken.
- Read-only and metadata-only. Ids, owners, status words, lane names, and
  bounded titles are projected; nothing is written, dispatched, or scheduled.
- English only. Localized copy is not offered here; it stays an explicit opt-in
  on the surfaces that already have one.
"""

from __future__ import annotations

import json
from typing import Any, Final

from ..plugin_bundle.omh.run_records import DISPATCH_SUMMARY_FILE, FANOUT_CONTRACT_FILE
from ..hashutil import sha256_text
from ..local_store import read_json_object_result, utc_now
from ..paths import OmhPaths
from .observation_journal import project_run_lifecycle, read_observation_events_result


COORDINATION_BOARD_SCHEMA_VERSION: Final[str] = "coordination_board/v1"

# Lane order is board order: the lanes that need a human decision come first,
# then work that is moving, then work that could start, then work that is only
# recorded, then work whose evidence ladder is closed.
COORDINATION_BOARD_LANES: Final[tuple[str, ...]] = (
    "blocked",
    "dependency_gated",
    "active",
    "next_ready",
    "prepared",
    "evidence_complete",
)

# The evidence ladder. Every item partitions these five kinds into
# `evidence_observed` and `missing_evidence`, so "what is missing" is always a
# complete answer rather than a list that quietly omits what was never checked.
COORDINATION_BOARD_EVIDENCE_KINDS: Final[tuple[str, ...]] = (
    "execution",
    "verification",
    "review",
    "ci",
    "merge",
)

# `project_run_lifecycle` sets each of these only from a canonical observed
# journal event, which is exactly why the board can trust them and cannot
# trust a status word.
_EVIDENCE_PROJECTION_FIELDS: Final[dict[str, str]] = {
    "execution": "execution_observed",
    "verification": "verification_observed",
    "review": "review_observed",
    "ci": "ci_observed",
    "merge": "merge_observed",
}

COORDINATION_BOARD_ITEM_SOURCES: Final[tuple[str, ...]] = (
    "goal_objective",
    "goal_checkpoint",
    "fanout_unit",
)

COORDINATION_BOARD_SOURCES: Final[tuple[str, ...]] = (
    "goal_ledgers",
    "fanout_contracts",
    "runtime_observations",
)

# Generic completion vocabulary, collected from the two ledgers this board
# reads: goal checkpoint statuses and fanout dispatch result statuses. A word
# in this set is a claim that work finished. It is never evidence.
COORDINATION_BOARD_COMPLETION_CLAIMS: Final[tuple[str, ...]] = (
    "already_completed",
    "complete",
    "completed",
    "done",
    "finished",
    "merge_ready",
    "ok",
    "passed",
    "success",
)

# Mirrors `fanout_dispatch._dependency_satisfied` minus `dry_run_planned`.
# A dry run planned a unit and ran nothing, so treating it as a cleared gate
# would show downstream work as startable when no dependency actually finished.
# Diverging only in the conservative direction keeps the board from ever
# under-reporting a gate.
_DEPENDENCY_SATISFYING_STATUSES: Final[tuple[str, ...]] = ("already_completed", "completed")

# Mirrors `fanout_dispatch._dependency_failed` minus `blocked_by_dependency`
# and `not_selected`, which are gating and scheduling answers rather than
# failures and get their own lanes below.
_BLOCKING_UNIT_STATUSES: Final[tuple[str, ...]] = (
    "capability_snapshot_invalid",
    "owner_profile_renamed",
    "route_unresolved",
    "modality_unknown",
    "modality_unsupported",
    "modality_transformation_unobserved",
    "executor_not_ready",
    "failed",
    "unsupported_for_local_dispatch",
    "worktree_failed",
)

_DEPENDENCY_GATED_UNIT_STATUS: Final[str] = "blocked_by_dependency"
_NOT_SELECTED_UNIT_STATUS: Final[str] = "not_selected"
_RUNNING_STATUSES: Final[tuple[str, ...]] = ("running", "in_progress")
_BLOCKED_CHECKPOINT_STATUSES: Final[tuple[str, ...]] = ("blocked", "failed")
_BLOCKED_GOAL_STATUSES: Final[tuple[str, ...]] = ("blocked", "failed")
_TERMINAL_GOAL_STATUSES: Final[tuple[str, ...]] = ("cancelled", "complete")
# `choose` is what a fanout handoff records when no executor was picked; it is
# the fanout spelling of "nobody owns this yet".
_UNOWNED_EXECUTOR_TARGETS: Final[tuple[str, ...]] = ("", "choose", "none", "unknown")

COORDINATION_BOARD_CLAIM_BOUNDARY: Final[str] = (
    "The coordination board is a read-only local projection of recorded artifacts. A generic done, "
    "completed, or merge_ready status word is a claim, never evidence: execution, verification, review, "
    "CI, and merge stay listed as missing until a matching observed runtime record exists. A satisfied "
    "dependency means only that the unit it gates stopped blocking, not that its work ran, was verified, "
    "reviewed, passed CI, or merged."
)

DEFAULT_LIMIT: Final[int] = 20
UNASSIGNED: Final[str] = "unassigned"
_TITLE_LIMIT: Final[int] = 120
_REASON_LIMIT: Final[int] = 160


def build_coordination_board(paths: OmhPaths, *, limit: int = DEFAULT_LIMIT, now: str = "") -> dict[str, Any]:
    """Project every locally recorded coordination item into one ordered board."""
    observed_at = now or utc_now()
    effective_limit = limit if limit > 0 else 0

    events, _errors = read_observation_events_result(paths)
    events_by_run = _events_by_run(events)
    lifecycles = {
        run_id: project_run_lifecycle(run_events, run_id=run_id) for run_id, run_events in events_by_run.items()
    }
    owners = _observed_owners(events_by_run)

    goals = _goal_records(paths)
    fanouts = _fanout_records(paths)
    items: list[dict[str, Any]] = []
    for goal in goals:
        items.extend(_goal_items(goal, lifecycles, owners))
    for fanout in fanouts:
        items.extend(_fanout_items(fanout, lifecycles, owners))
    items.sort(key=_board_sort_key)

    sources_used = set()
    if goals:
        sources_used.add("goal_ledgers")
    if fanouts:
        sources_used.add("fanout_contracts")
    if events_by_run:
        sources_used.add("runtime_observations")

    lane_counts = {lane: sum(1 for item in items if item["lane"] == lane) for lane in COORDINATION_BOARD_LANES}
    return {
        "schema_version": COORDINATION_BOARD_SCHEMA_VERSION,
        # Outside the digest on purpose: a timestamp inside a compared payload
        # turns an equality check into a race.
        "observed_at": observed_at,
        "board_digest": _board_digest(items),
        "item_count": len(items),
        "lane_order": list(COORDINATION_BOARD_LANES),
        "lane_counts": lane_counts,
        "items": items[:effective_limit],
        "sources_used": [source for source in COORDINATION_BOARD_SOURCES if source in sources_used],
        "summary": _board_summary(len(items), lane_counts),
        "claim_boundary": COORDINATION_BOARD_CLAIM_BOUNDARY,
    }


def claims_completion(status: Any) -> bool:
    """True when a recorded status word is a generic completion claim.

    The answer is deliberately never routed into `evidence_observed`: execution,
    verification, review, CI, and merge come only from matching observed
    runtime-journal events. A claim moves an item out of `next_ready` and
    `prepared` — there is nothing left to start — and leaves it `active` until
    the observed ladder closes it.
    """
    return str(status or "").strip().lower() in COORDINATION_BOARD_COMPLETION_CLAIMS


def render_coordination_board_text(payload: dict[str, Any]) -> str:
    """The board as plain lines, grouped by lane, English only."""
    items = _dict_items(payload.get("items"))
    lines = [
        f"Coordination board ({int(payload.get('item_count', 0) or 0)} items) "
        f"— observed at {payload.get('observed_at', '') or 'unknown'}",
        f"Digest {str(payload.get('board_digest', '') or 'unknown')[:12]}",
    ]
    if not items:
        lines.append("No coordinated work recorded.")
    for lane in COORDINATION_BOARD_LANES:
        lane_items = [item for item in items if item.get("lane") == lane]
        if not lane_items:
            continue
        lines.append("")
        lines.append(f"{lane.replace('_', ' ').upper()} ({len(lane_items)})")
        lines.extend(_item_line(item) for item in lane_items)
    truncation = _truncation_line(payload, shown=len(items))
    if truncation:
        lines.extend(["", truncation])
    lines.extend(["", str(payload.get("claim_boundary", "") or "")])
    return "\n".join(lines).strip()


def validate_coordination_board(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != COORDINATION_BOARD_SCHEMA_VERSION:
        errors.append("schema_version must be coordination_board/v1")
    if not str(payload.get("board_digest", "")).strip():
        errors.append("board_digest is required")
    if not str(payload.get("claim_boundary", "")).strip():
        errors.append("claim_boundary is required")
    if not isinstance(payload.get("item_count"), int) or isinstance(payload.get("item_count"), bool):
        errors.append("item_count must be an integer")
    if payload.get("lane_order") != list(COORDINATION_BOARD_LANES):
        errors.append("lane_order must list every board lane in board order")
    lane_counts = payload.get("lane_counts")
    if not isinstance(lane_counts, dict):
        errors.append("lane_counts must be an object")
    elif sorted(lane_counts) != sorted(COORDINATION_BOARD_LANES):
        errors.append("lane_counts must carry every board lane")
    if not isinstance(payload.get("items"), list):
        errors.append("items must be a list")
    else:
        for index, item in enumerate(payload["items"]):
            for error in validate_coordination_board_item(item):
                errors.append(f"items[{index}]: {error}")
    if not isinstance(payload.get("sources_used"), list):
        errors.append("sources_used must be a list")
    elif any(source not in COORDINATION_BOARD_SOURCES for source in payload["sources_used"]):
        errors.append("sources_used carries an unsupported source")
    return errors


def validate_coordination_board_item(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return ["item must be an object"]
    errors: list[str] = []
    if not str(item.get("item_id", "")).strip():
        errors.append("item_id is required")
    if item.get("lane") not in COORDINATION_BOARD_LANES:
        errors.append("lane is unsupported")
    if item.get("source") not in COORDINATION_BOARD_ITEM_SOURCES:
        errors.append("source is unsupported")
    if not isinstance(item.get("claimed_complete"), bool):
        errors.append("claimed_complete must be boolean")
    observed = item.get("evidence_observed")
    missing = item.get("missing_evidence")
    for field, values in (("evidence_observed", observed), ("missing_evidence", missing)):
        if not isinstance(values, list):
            errors.append(f"{field} must be a list")
        elif any(value not in COORDINATION_BOARD_EVIDENCE_KINDS for value in values):
            errors.append(f"{field} carries an unsupported evidence kind")
    if isinstance(observed, list) and isinstance(missing, list):
        if sorted([*observed, *missing]) != sorted(COORDINATION_BOARD_EVIDENCE_KINDS):
            errors.append("evidence_observed and missing_evidence must partition every evidence kind")
        # The mechanical form of the claim boundary: a lane that says the
        # evidence ladder is closed cannot ship with a gap in it.
        if item.get("lane") == "evidence_complete" and missing:
            errors.append("evidence_complete requires no missing evidence")
    for field in ("depends_on", "unmet_dependencies", "blocked_by", "run_refs"):
        if not isinstance(item.get(field), list):
            errors.append(f"{field} must be a list")
    if item.get("lane") == "blocked" and not item.get("blocked_by"):
        errors.append("blocked requires at least one blocked_by reason")
    if item.get("lane") == "dependency_gated" and not item.get("unmet_dependencies"):
        errors.append("dependency_gated requires at least one unmet dependency")
    return errors


def _goal_items(
    goal: dict[str, Any], lifecycles: dict[str, dict[str, Any]], owners: dict[str, str]
) -> list[dict[str, Any]]:
    goal_id = str(goal.get("goal_id", "") or "")
    if not goal_id:
        return []
    goal_status = str(goal.get("status", "") or "unknown")
    work_title = _bounded_text(goal.get("objective_summary", ""), _TITLE_LIMIT) or goal_id
    linked_runs = [str(run) for run in _list_items(goal.get("linked_runtime_runs")) if str(run).strip()]
    goal_blockers = [
        _bounded_text(blocker.get("summary", ""), _REASON_LIMIT) or str(blocker.get("blocker_id", "") or "blocker")
        for blocker in _dict_items(goal.get("blockers"))
        if str(blocker.get("status", "")) == "active"
    ]
    if goal_status in _BLOCKED_GOAL_STATUSES:
        goal_blockers.insert(0, f"goal status {goal_status}")
    goal_open = goal_status not in _TERMINAL_GOAL_STATUSES

    checkpoints = _dict_items(goal.get("checkpoints"))
    if not checkpoints:
        # A goal with acceptance criteria and no checkpoint yet is the most
        # common shape of "work nobody has started". Without this item the
        # board would render it as nothing at all.
        return [
            _item(
                work_id=goal_id,
                work_kind="goal",
                work_title=work_title,
                source="goal_objective",
                local_id="objective",
                title=work_title,
                owner=_first_owner(linked_runs, owners),
                recorded_status=goal_status,
                run_refs=linked_runs,
                lifecycles=lifecycles,
                depends_on=[],
                unmet_dependencies=[],
                blocked_by=list(goal_blockers),
                running=False,
                work_open=goal_open,
            )
        ]

    items: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        checkpoint_id = str(checkpoint.get("checkpoint_id", "") or "")
        if not checkpoint_id:
            continue
        status = str(checkpoint.get("status", "") or "unknown")
        run_refs = [str(checkpoint.get("linked_runtime_run_id", "") or "")]
        run_refs = [run_ref for run_ref in run_refs if run_ref]
        blocked_by = [f"checkpoint status {status}"] if status in _BLOCKED_CHECKPOINT_STATUSES else []
        # A goal-level blocker stops the work that is still open on that goal;
        # a checkpoint that already claims completion has nothing left to stop.
        if not claims_completion(status):
            blocked_by.extend(goal_blockers)
        items.append(
            _item(
                work_id=goal_id,
                work_kind="goal",
                work_title=work_title,
                source="goal_checkpoint",
                local_id=checkpoint_id,
                title=_bounded_text(checkpoint.get("summary", ""), _TITLE_LIMIT) or checkpoint_id,
                owner=_first_owner(run_refs, owners),
                recorded_status=status,
                run_refs=run_refs,
                lifecycles=lifecycles,
                depends_on=[],
                unmet_dependencies=[],
                blocked_by=blocked_by,
                running=status in _RUNNING_STATUSES,
                work_open=goal_open,
            )
        )
    return items


def _fanout_items(
    fanout: dict[str, Any], lifecycles: dict[str, dict[str, Any]], owners: dict[str, str]
) -> list[dict[str, Any]]:
    fanout_id = str(fanout.get("fanout_id", "") or "")
    dispatch = fanout.get("dispatch", {})
    dispatch = dispatch if isinstance(dispatch, dict) else {}
    items: list[dict[str, Any]] = []
    for unit in _dict_items(fanout.get("units")):
        unit_id = str(unit.get("unit_id", "") or "")
        if not unit_id:
            continue
        entry = dispatch.get(unit_id, {})
        entry = entry if isinstance(entry, dict) else {}
        status = str(entry.get("status", "") or unit.get("status", "") or "prepared")
        run_refs = [str(entry.get("run_ref", "") or unit.get("run_ref", "") or "")]
        run_refs = [run_ref for run_ref in run_refs if run_ref]
        declared = [str(dependency) for dependency in _list_items(unit.get("depends_on")) if str(dependency).strip()]
        unmet = [
            f"{fanout_id}/{dependency}"
            for dependency in declared
            if not _dependency_satisfied(dispatch.get(dependency))
        ]
        blocked_by = [f"dispatch status {status}"] if status in _BLOCKING_UNIT_STATUSES else []
        if status == _DEPENDENCY_GATED_UNIT_STATUS and not unmet:
            # The dispatcher recorded the gate; the contract no longer explains
            # it (a dependency was renamed or removed). Reporting the gate with
            # no reason would read as "nothing is waiting".
            unmet = [
                f"{fanout_id}/{dependency}" for dependency in _list_items(entry.get("blocked_on"))
            ] or [f"{fanout_id}/unknown"]
        items.append(
            _item(
                work_id=fanout_id,
                work_kind="fanout",
                work_title=fanout_id,
                source="fanout_unit",
                local_id=unit_id,
                title=_bounded_text(unit.get("title", ""), _TITLE_LIMIT) or unit_id,
                owner=_unit_owner(unit, entry, run_refs, owners),
                recorded_status=status,
                run_refs=run_refs,
                lifecycles=lifecycles,
                depends_on=[f"{fanout_id}/{dependency}" for dependency in declared],
                unmet_dependencies=unmet,
                blocked_by=blocked_by,
                running=status in _RUNNING_STATUSES,
                work_open=status != _NOT_SELECTED_UNIT_STATUS,
            )
        )
    return items


def _item(
    *,
    work_id: str,
    work_kind: str,
    work_title: str,
    source: str,
    local_id: str,
    title: str,
    owner: str,
    recorded_status: str,
    run_refs: list[str],
    lifecycles: dict[str, dict[str, Any]],
    depends_on: list[str],
    unmet_dependencies: list[str],
    blocked_by: list[str],
    running: bool,
    work_open: bool,
) -> dict[str, Any]:
    observed = _observed_evidence(run_refs, lifecycles)
    claimed = claims_completion(recorded_status)
    return {
        "item_id": f"{work_id}/{local_id}",
        "work_id": work_id,
        "work_kind": work_kind,
        "work_title": work_title,
        "source": source,
        "title": title,
        "owner": owner,
        "lane": _lane(
            blocked_by=blocked_by,
            unmet_dependencies=unmet_dependencies,
            observed=observed,
            claimed_complete=claimed,
            running=running,
            work_open=work_open,
        ),
        "recorded_status": _bounded_text(recorded_status, _REASON_LIMIT),
        "claimed_complete": claimed,
        "depends_on": sorted(set(depends_on)),
        "unmet_dependencies": sorted(set(unmet_dependencies)),
        "blocked_by": _unique(blocked_by),
        "run_refs": sorted(set(run_refs)),
        "evidence_observed": observed,
        "missing_evidence": [kind for kind in COORDINATION_BOARD_EVIDENCE_KINDS if kind not in observed],
    }


def _lane(
    *,
    blocked_by: list[str],
    unmet_dependencies: list[str],
    observed: list[str],
    claimed_complete: bool,
    running: bool,
    work_open: bool,
) -> str:
    if blocked_by:
        return "blocked"
    if unmet_dependencies:
        return "dependency_gated"
    if len(observed) == len(COORDINATION_BOARD_EVIDENCE_KINDS):
        return "evidence_complete"
    # `claimed_complete` reaches `active` and stops there. That single term is
    # the acceptance criterion "generic done cannot satisfy verification,
    # review, CI, or merge" expressed as a lane rule: the claim is enough to
    # say nothing is left to start, and never enough to close the ladder.
    if running or observed or claimed_complete:
        return "active"
    if work_open:
        return "next_ready"
    return "prepared"


def _observed_evidence(run_refs: list[str], lifecycles: dict[str, dict[str, Any]]) -> list[str]:
    """Evidence kinds an observed runtime record carries for these runs.

    A kind counts when any linked run recorded it, because the question the
    board answers is "is this evidence missing everywhere?".
    """
    observed: list[str] = []
    for kind in COORDINATION_BOARD_EVIDENCE_KINDS:
        field = _EVIDENCE_PROJECTION_FIELDS[kind]
        if any(lifecycles.get(run_ref, {}).get(field) is True for run_ref in run_refs):
            observed.append(kind)
    return observed


def _dependency_satisfied(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    return str(entry.get("status", "")) in _DEPENDENCY_SATISFYING_STATUSES


def _unit_owner(
    unit: dict[str, Any], entry: dict[str, Any], run_refs: list[str], owners: dict[str, str]
) -> str:
    handoff = unit.get("handoff")
    handoff = handoff if isinstance(handoff, dict) else {}
    for candidate in (unit.get("owner"), handoff.get("executor_target"), entry.get("owner")):
        name = str(candidate or "").strip()
        if name and name.lower() not in _UNOWNED_EXECUTOR_TARGETS:
            return name
    return _first_owner(run_refs, owners)


def _first_owner(run_refs: list[str], owners: dict[str, str]) -> str:
    for run_ref in run_refs:
        owner = owners.get(run_ref, "")
        if owner:
            return owner
    return UNASSIGNED


def _observed_owners(events_by_run: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    """Last observed runtime profile (else actor) per run, from the journal."""
    owners: dict[str, str] = {}
    for run_id, events in events_by_run.items():
        for event in events:
            name = str(event.get("runtime_profile", "") or event.get("actor", "") or "").strip()
            if name:
                owners[run_id] = name
    return owners


def _events_by_run(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group journal events by run id.

    `project_run_lifecycle` folds in any event whose own `run_id` is empty, so
    it must be handed one run's events rather than the whole journal.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        run_id = str(event.get("run_id", "") or "")
        if not run_id:
            continue
        grouped.setdefault(run_id, []).append(event)
    return grouped


def _goal_records(paths: OmhPaths) -> list[dict[str, Any]]:
    """Goal ledgers, skipping any file this board cannot read.

    Deliberately not `list_goal_ledgers`: that helper raises on a malformed
    goal.json, and one corrupt ledger must not take the whole read-only board
    down. `status_board._dispatch_summary_units` skips per artifact for the
    same reason.
    """
    if not paths.goals_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for goal_json in sorted(paths.goals_dir.glob("*/goal.json")):
        record, error = read_json_object_result(goal_json)
        if error or not isinstance(record, dict):
            continue
        records.append(record)
    return records


def _fanout_records(paths: OmhPaths) -> list[dict[str, Any]]:
    root = paths.fanout_contracts_dir
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for fanout_dir in sorted(root.iterdir()):
        if not fanout_dir.is_dir() or fanout_dir.is_symlink():
            continue
        contract, error = read_json_object_result(fanout_dir / FANOUT_CONTRACT_FILE)
        if error or not isinstance(contract, dict):
            continue
        summary, summary_error = read_json_object_result(fanout_dir / DISPATCH_SUMMARY_FILE)
        dispatch: dict[str, Any] = {}
        if not summary_error and isinstance(summary, dict):
            for entry in _dict_items(summary.get("units")):
                unit_id = str(entry.get("unit_id", "") or "")
                if unit_id:
                    dispatch[unit_id] = entry
        records.append(
            {
                "fanout_id": str(contract.get("fanout_id", "") or fanout_dir.name),
                "units": _dict_items(contract.get("units")),
                "dispatch": dispatch,
            }
        )
    return records


def _board_digest(items: list[dict[str, Any]]) -> str:
    """Digest of the ordered items, covering every projected item.

    `observed_at` is excluded so identical artifacts hash identically, and the
    full item list is used rather than the displayed page so a `--limit` change
    is a view change, not an identity change.
    """
    return sha256_text(json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _board_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    lane = str(item.get("lane", ""))
    lane_index = COORDINATION_BOARD_LANES.index(lane) if lane in COORDINATION_BOARD_LANES else len(
        COORDINATION_BOARD_LANES
    )
    return (lane_index, str(item.get("item_id", "")))


def _board_summary(item_count: int, lane_counts: dict[str, int]) -> str:
    if item_count == 0:
        return "No coordinated work recorded."
    parts = [
        f"{lane_counts[lane]} {lane.replace('_', '-')}" for lane in COORDINATION_BOARD_LANES if lane_counts[lane]
    ]
    return f"{item_count} coordination items: {', '.join(parts)}."


def _item_line(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("item_id", "") or "unknown"),
        str(item.get("title", "") or "untitled"),
        f"owner {item.get('owner', UNASSIGNED)}",
        f"recorded {item.get('recorded_status', '') or 'unknown'}",
    ]
    blocked_by = [str(reason) for reason in _list_items(item.get("blocked_by"))]
    if blocked_by:
        parts.append(f"blocked by {'; '.join(blocked_by)}")
    unmet = [str(dependency) for dependency in _list_items(item.get("unmet_dependencies"))]
    if unmet:
        parts.append(f"waiting on {', '.join(unmet)}")
    missing = [str(kind) for kind in _list_items(item.get("missing_evidence"))]
    parts.append(f"missing evidence: {', '.join(missing)}" if missing else "evidence complete")
    return "- " + " — ".join(parts)


def _truncation_line(payload: dict[str, Any], *, shown: int) -> str:
    dropped = int(payload.get("item_count", 0) or 0) - shown
    if dropped <= 0:
        return ""
    return f"Showing {shown} of {payload.get('item_count', 0)} items; {dropped} not shown because of the display limit."


def _bounded_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return unique


def _list_items(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _dict_items(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
