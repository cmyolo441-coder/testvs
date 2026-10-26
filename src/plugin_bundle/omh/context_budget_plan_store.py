"""Bounded, session-addressed plan reads for the standalone plugin."""
from __future__ import annotations

import json
from pathlib import Path

from .active_workflow_context_state import session_fingerprint
from .context_budget_plan_capacity import (
    build_capacity_record, metadata_ref, observations, parse_must_keep,
    parse_route_capacity_input, valid_digest,
)
from .context_budget_plan_types import BudgetPlanError, Invalidation, Plan, RouteIdentity
from .runtime_reader import default_omh_home


def context_budget_plan_path(omh_home: str, session_ref: str) -> Path:
    if not session_ref:
        raise BudgetPlanError("session_required")
    home = Path(omh_home).expanduser() if omh_home else default_omh_home()
    return home / "runtime" / "context-budget-plans" / (session_fingerprint(session_ref)[7:] + ".json")


def read_plan(omh_home: str, session_ref: str) -> Plan | None:
    """Reject corrupt/foreign data instead of silently treating it as no plan."""
    path = context_budget_plan_path(omh_home, session_ref)
    if any(parent.is_symlink() for parent in (path, path.parent, path.parent.parent)):
        raise BudgetPlanError("linked_budget_plan")
    try:
        with path.open(encoding="utf-8") as handle:
            raw = handle.read(32769)
    except FileNotFoundError:
        return None
    if len(raw) > 32768:
        raise BudgetPlanError("budget_plan_size_limit")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != "context_budget_plan/v1" or data.get("session_ref") != session_fingerprint(session_ref):
        raise BudgetPlanError("invalid_plan_identity")
    record = data.get("route_capacity")
    if not isinstance(record, dict) or record.get("schema_version") != "route_capacity_record/v1":
        raise BudgetPlanError("invalid_route_capacity_record")
    raw_identity = record.get("route_identity")
    keys = {"executor_profile", "provider", "wire_model", "contract_model_id", "model_family", "catalog_kind", "catalog_fingerprint"}
    if not isinstance(raw_identity, dict) or set(raw_identity) != keys:
        raise BudgetPlanError("invalid_route_identity")
    for key in keys - {"provider", "catalog_fingerprint"}:
        if not isinstance(raw_identity[key], str):
            raise BudgetPlanError("invalid_route_identity")
        metadata_ref(raw_identity[key], optional=key == "contract_model_id")
    provider = raw_identity["provider"]
    if provider is not None:
        if not isinstance(provider, str):
            raise BudgetPlanError("invalid_provider")
        metadata_ref(provider)
    if raw_identity["catalog_fingerprint"] is not None:
        raise BudgetPlanError("unexpected_catalog_fingerprint")
    identity: RouteIdentity = {
        "executor_profile": raw_identity["executor_profile"], "provider": provider,
        "wire_model": raw_identity["wire_model"], "contract_model_id": raw_identity["contract_model_id"],
        "model_family": raw_identity["model_family"], "catalog_kind": raw_identity["catalog_kind"], "catalog_fingerprint": None,
    }
    raw_capacity = record.get("capacity")
    if not isinstance(raw_capacity, dict):
        raise BudgetPlanError("invalid_capacity")
    capacity = parse_route_capacity_input(json.dumps({"schema_version": "route_capacity_input/v1", **raw_capacity}))
    canonical = build_capacity_record(identity, capacity)
    if record != canonical or data.get("active_budget_source") != canonical:
        raise BudgetPlanError("capacity_binding_mismatch")
    pack = parse_must_keep(json.dumps(data.get("must_keep_pack")))
    plan_id, superseded = data.get("plan_id"), data.get("superseded_plan_id")
    if not isinstance(plan_id, str) or not valid_digest(plan_id) or (superseded is not None and (not isinstance(superseded, str) or not valid_digest(superseded))):
        raise BudgetPlanError("invalid_plan_digest")
    history, count = data.get("route_history"), data.get("rebind_count")
    if not isinstance(history, list) or len(history) > 8 or any(not isinstance(item, str) or not valid_digest(item) for item in history):
        raise BudgetPlanError("invalid_route_history")
    if type(count) is not int or not 0 <= count <= 9:
        raise BudgetPlanError("invalid_rebind_count")
    invalidation = data.get("invalidation")
    if not isinstance(invalidation, dict) or set(invalidation) != {"reason", "action"}:
        raise BudgetPlanError("invalid_invalidation")
    reason, action = invalidation["reason"], invalidation["action"]
    if reason not in ("none", "capacity_shrank", "capacity_changed", "capacity_unknown", "rebind_limit"):
        raise BudgetPlanError("invalid_invalidation_reason")
    if action not in ("continue", "checkpoint_required", "overflow_recovery_required", "capacity_unknown_hold", "rebind_loop_hold"):
        raise BudgetPlanError("invalid_recovery_action")
    stale = data.get("stale")
    if type(stale) is not bool or stale != (action != "continue"):
        raise BudgetPlanError("invalid_stale_state")
    parsed_invalidation: Invalidation = {"reason": reason, "action": action}
    return {
        "schema_version": "context_budget_plan/v1", "session_ref": session_fingerprint(session_ref),
        "plan_id": plan_id, "superseded_plan_id": superseded, "route_capacity": canonical,
        "must_keep_pack": pack, "stale": stale, "invalidation": parsed_invalidation,
        "route_history": history, "rebind_count": count, "active_budget_source": canonical, **observations(pack),
    }
