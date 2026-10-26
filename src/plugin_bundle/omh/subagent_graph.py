"""Pure, bounded projection of one recorded local fanout into TUI graph metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Final

from .subagent_graph_contract import (
    GRAPH_CONTRACT_UNIT_LIMIT,
    normalized_graph_units,
)

GRAPH_SCHEMA_VERSION: Final[str] = "subagent_graph/v1"
GRAPH_NODE_LIMIT: Final[int] = 8
GRAPH_METADATA_BYTE_LIMIT: Final[int] = 16_384
GRAPH_CLAIM_BOUNDARY: Final[str] = (
    "Prepared fanout topology and local status records are metadata-only. "
    "They are not execution, verification, review, CI, merge-readiness, or merge evidence."
)

_PREFERENCES: Final[frozenset[str]] = frozenset({"auto", "on", "off"})
_ROSTER_SCHEMA_VERSION: Final[str] = "omh_running_work_board/v1"
_SUCCESS_STATES: Final[frozenset[str]] = frozenset(
    {"completed", "already_completed", "dry_run_planned"}
)
_FAILED_STATES: Final[frozenset[str]] = frozenset(
    {
        "capability_snapshot_invalid",
        "owner_profile_renamed",
        "route_unresolved",
        "modality_unknown",
        "modality_unsupported",
        "modality_transformation_unobserved",
        "failed",
        "blocked_by_dependency",
        "executor_not_ready",
        "unsupported_for_local_dispatch",
        "worktree_failed",
        "not_selected",
        "interrupted",
        "model_choice_required",
    }
)
_FRONTIER_STATES: Final[frozenset[str]] = frozenset({"prepared_not_observed"})
_ROSTER_STATES: Final[frozenset[str]] = frozenset(
    {"running", "prepared_not_observed"} | _SUCCESS_STATES | _FAILED_STATES
)


def project_subagent_graph(
    contract: Mapping[str, object] | None,
    roster: Mapping[str, object],
    *,
    preference: str,
    blockers: Sequence[str] = (),
    node_limit: int = GRAPH_NODE_LIMIT,
) -> dict[str, object]:
    """Return additive `subagent_graph/v1` metadata without I/O or mutation."""
    safe_preference = preference if preference in _PREFERENCES else "auto"
    if safe_preference == "off":
        return _inactive("explicit_off")
    if blockers:
        return _inactive(str(blockers[0]))
    if contract is None:
        return _inactive("empty_graph")

    normalized, error = normalized_graph_units(contract)
    if error:
        return _inactive(error)
    if not normalized:
        return _inactive("empty_graph")
    if len(normalized) == 1:
        return _inactive("single_owner")

    unit_by_id = {unit["unit_id"]: unit for unit in normalized}
    if len(unit_by_id) != len(normalized):
        return _inactive("duplicate_node")
    ids = set(unit_by_id)
    for unit in normalized:
        unit_id = unit["unit_id"]
        dependencies = unit["depends_on"]
        if unit_id in dependencies:
            return _inactive("self_dependency")
        if any(dependency not in ids for dependency in dependencies):
            return _inactive("unknown_dependency")

    order = _topological_order(normalized)
    if not order:
        return _inactive("cyclic_dependency")
    if _has_unordered_overlap(normalized, unit_by_id):
        return _inactive("overlapping_writes_without_edge")

    roster_by_id, roster_error = _paired_roster(contract, roster)
    if roster_error:
        return _inactive(roster_error)

    has_edges = any(unit["depends_on"] for unit in normalized)
    if safe_preference == "auto" and not has_edges:
        return _inactive("no_dependency_edges")

    reason = "explicit_preference" if safe_preference == "on" else "dependency_edges"
    safe_limit = max(1, min(node_limit, GRAPH_NODE_LIMIT))
    visible_ids = order[:safe_limit]
    visible = set(visible_ids)
    states = {
        unit_id: str(roster_by_id.get(unit_id, {}).get("status", "unknown"))
        for unit_id in ids
    }
    effective_states: dict[str, str] = {}
    for unit_id in order:
        raw_state = states[unit_id]
        dependency_states = [
            effective_states.get(str(dependency), "unknown")
            for dependency in unit_by_id[unit_id]["depends_on"]
        ]
        effective_states[unit_id] = (
            "blocked"
            if raw_state in _FRONTIER_STATES
            and any(state in _FAILED_STATES or state == "blocked" for state in dependency_states)
            else raw_state
        )
    blocked_by_by_id: dict[str, list[str]] = {}
    frontier: list[str] = []
    for unit_id in order:
        unit = unit_by_id[unit_id]
        dependencies = list(unit["depends_on"])
        blocked_by = [
            dependency
            for dependency in dependencies
            if effective_states.get(dependency, "unknown") not in _SUCCESS_STATES
        ]
        blocked_by_by_id[unit_id] = blocked_by
        state = effective_states[unit_id]
        in_frontier = not blocked_by and state in _FRONTIER_STATES
        if in_frontier:
            frontier.append(unit_id)
    nodes: list[dict[str, object]] = []
    for unit_id in visible_ids:
        node: dict[str, object] = {
            "node_id": unit_id,
            "state": effective_states[unit_id],
            "blocked_by": blocked_by_by_id[unit_id],
            "in_frontier": unit_id in frontier,
        }
        # `state` above is the DISPATCH word the topology reasons about --
        # a `running` node is one whose marker exists. Whether its work is
        # moving is a separate observation the roster row carries when a
        # stdout snapshot has been assessed (`omh.coding.unit_progress`), and
        # it rides alongside rather than replacing `state`: the frontier,
        # blocked-by, and success/failure arithmetic above is defined on the
        # dispatch vocabulary, and widening that would change the topology.
        # The renderer is what must never show a stuck unit as `running`.
        node.update(_roster_progress_fields(roster_by_id.get(unit_id)))
        nodes.append(node)

    edges = [
        [dependency, unit["unit_id"]]
        for unit in normalized
        for dependency in unit["depends_on"]
        if dependency in visible and unit["unit_id"] in visible
    ]
    payload: dict[str, object] = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "status": "active",
        "reason": reason,
        "nodes": nodes,
        "edges": edges,
        "edge_count": sum(len(unit["depends_on"]) for unit in normalized),
        "frontier": frontier,
        "hidden_nodes": max(0, len(order) - len(visible_ids)),
        "claim_boundary": GRAPH_CLAIM_BOUNDARY,
    }
    if len(json.dumps(payload, sort_keys=True).encode("utf-8")) > GRAPH_METADATA_BYTE_LIMIT:
        return _inactive("graph_metadata_limit_exceeded")
    return payload


def _inactive(reason: str) -> dict[str, object]:
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "status": "inactive",
        "reason": reason,
        "nodes": [],
        "edges": [],
        "edge_count": 0,
        "frontier": [],
        "hidden_nodes": 0,
        "claim_boundary": GRAPH_CLAIM_BOUNDARY,
    }


def _topological_order(units: list[dict[str, object]]) -> list[str]:
    remaining = {str(unit["unit_id"]): set(unit["depends_on"]) for unit in units}
    order: list[str] = []
    while remaining:
        ready = sorted(unit_id for unit_id, dependencies in remaining.items() if not dependencies)
        if not ready:
            return []
        order.extend(ready)
        for unit_id in ready:
            remaining.pop(unit_id)
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return order


def _has_unordered_overlap(
    units: list[dict[str, object]],
    unit_by_id: Mapping[str, dict[str, object]],
) -> bool:
    for index, left in enumerate(units):
        left_scope = set(left["file_scope"])
        if not left_scope:
            continue
        for right in units[index + 1 :]:
            if not left_scope.intersection(right["file_scope"]):
                continue
            left_id = str(left["unit_id"])
            right_id = str(right["unit_id"])
            if not _depends_on(left_id, right_id, unit_by_id) and not _depends_on(
                right_id,
                left_id,
                unit_by_id,
            ):
                return True
    return False


def _depends_on(
    unit_id: str,
    target_id: str,
    unit_by_id: Mapping[str, dict[str, object]],
) -> bool:
    pending = list(unit_by_id[unit_id]["depends_on"])
    seen: set[str] = set()
    while pending:
        dependency = str(pending.pop())
        if dependency == target_id:
            return True
        if dependency in seen or dependency not in unit_by_id:
            continue
        seen.add(dependency)
        pending.extend(unit_by_id[dependency]["depends_on"])
    return False


# Hand-mirror of `omh.coding.unit_execution_state.UNIT_STUCK_STATES`, which
# this bundle cannot import. Parity gated in `tests/test_subagent_graph_roster.py`.
_STUCK_UNIT_STATES: Final[frozenset[str]] = frozenset(
    {
        "awaiting_input",
        "permission_blocked",
        "account_limit",
        "data_missing",
        "progress_stalled",
    }
)


def _roster_progress_fields(unit: Mapping[str, object] | None) -> dict[str, object]:
    """A stuck unit's observed state, reason, and stall age -- or nothing.

    Absent keys mean no stdout snapshot has been assessed for this unit, which
    a reader must not confuse with "the work is moving".
    """
    if not isinstance(unit, Mapping):
        return {}
    state = str(unit.get("unit_state", "") or "")
    if state not in _STUCK_UNIT_STATES:
        return {}
    fields: dict[str, object] = {"unit_state": state}
    reason = str(unit.get("state_reason", "") or "")
    if reason:
        fields["state_reason"] = reason
    elapsed = str(unit.get("stalled_for_seconds", "") or "")
    if elapsed.isdigit() and int(elapsed) > 0:
        fields["stalled_for_seconds"] = int(elapsed)
    return fields


def _paired_roster(
    contract: Mapping[str, object],
    roster: Mapping[str, object],
) -> tuple[dict[str, Mapping[str, object]], str]:
    fanout_id = str(contract.get("fanout_id", ""))
    if roster.get("schema_version") != _ROSTER_SCHEMA_VERSION:
        return {}, "invalid_roster_schema"
    if roster.get("fanout_id") != fanout_id:
        return {}, "unpaired_fanout_roster"
    raw_units = roster.get("units")
    if not fanout_id or not isinstance(raw_units, list):
        return {}, "unpaired_fanout_roster"
    if len(raw_units) > GRAPH_CONTRACT_UNIT_LIMIT:
        return {}, "graph_limit_exceeded"
    contract_ids = {
        str(unit.get("unit_id", ""))
        for unit in contract.get("units", [])
        if isinstance(unit, Mapping)
    }
    if len(raw_units) != len(contract_ids):
        return {}, "unpaired_fanout_roster"
    paired: dict[str, Mapping[str, object]] = {}
    for unit in raw_units:
        if not isinstance(unit, Mapping):
            return {}, "unpaired_fanout_roster"
        unit_id = unit.get("unit_id")
        if not isinstance(unit_id, str):
            return {}, "unpaired_fanout_roster"
        if unit_id in paired:
            return {}, "duplicate_roster_unit"
        if unit_id not in contract_ids or unit.get("fanout_id") != fanout_id:
            return {}, "unpaired_fanout_roster"
        if unit.get("status") not in _ROSTER_STATES:
            return {}, "invalid_roster_status"
        paired[unit_id] = unit
    return (
        (paired, "")
        if set(paired) == contract_ids
        else ({}, "unpaired_fanout_roster")
    )
