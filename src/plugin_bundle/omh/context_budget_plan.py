"""Shared continuation evaluation; no provider calls or implicit compaction."""
from __future__ import annotations

from .context_budget_plan_capacity import observations
from .context_budget_plan_store import read_plan
from .context_budget_plan_types import BudgetContext, Invalidation, Plan, RouteCapacity


def evaluate_capacity(previous: Plan | None, current: RouteCapacity, retained_tokens: int) -> Invalidation:
    """Compare the full effective policy, retaining any pending recovery debt."""
    usable = current["usable_budget_tokens"]["value"]
    if previous and previous["rebind_count"] >= 9:
        return {"reason": "rebind_limit", "action": "rebind_loop_hold"}
    if usable is None or current["route_identity"]["provider"] is None:
        return {"reason": "capacity_unknown", "action": "capacity_unknown_hold"}
    if retained_tokens > usable:
        return {"reason": "capacity_shrank" if previous else "capacity_changed", "action": "overflow_recovery_required"}
    if previous is None:
        return {"reason": "none", "action": "continue"}
    prior_capacity = previous["route_capacity"]
    if prior_capacity["effective_capacity_digest"] == current["effective_capacity_digest"]:
        return previous["invalidation"]
    prior_usable = prior_capacity["usable_budget_tokens"]["value"]
    reason = "capacity_shrank" if prior_usable is not None and usable < prior_usable else "capacity_changed"
    return {"reason": reason, "action": "checkpoint_required"}


def context_budget_continuation(omh_home: str, session_id: str, model: str) -> BudgetContext | None:
    """Project the selected session record before each model call, read-only.

    Hermes supplies wire model, not provider/capacity. Only exact wire-model
    equality makes the explicitly published session capacity eligible. Provider
    identity remains declared local metadata, never a host observation. Missing
    plan is a no-op; corrupt plan and a mismatched model require bounded recovery.
    """
    if not session_id:
        return None
    error = False
    try:
        plan = read_plan(omh_home, session_id)
    except (OSError, ValueError, UnicodeError, RecursionError):
        plan = None
        error = True
    if plan is None and not error:
        return None
    matches = bool(plan and model and model == plan["route_capacity"]["route_identity"]["wire_model"])
    if plan and matches:
        invalidation = evaluate_capacity(plan, plan["route_capacity"], plan["must_keep_pack"]["estimated_tokens_total"])
    else:
        invalidation: Invalidation = {"reason": "plan_unreadable" if error else "host_route_unbound", "action": "capacity_unknown_hold"}
    pack = plan["must_keep_pack"] if plan else None
    return {
        "schema_version": "context_budget_continuation/v1", "plan_id": plan["plan_id"] if plan else None,
        "must_keep_pack": pack, "stale": invalidation["action"] != "continue", "invalidation": invalidation,
        "usable_budget_tokens": plan["route_capacity"]["usable_budget_tokens"] if plan and matches else {"value": None, "class": "unknown"},
        "active_budget_source": plan["route_capacity"] if plan and matches else None,
        "host_route": {"wire_model_matches": matches, "provider": None, "provider_observation": "unknown"},
        "recovery": {"max_checkpoint_attempts": 1, "completion": "not_observed", "resolution": "explicit_prepare_after_checkpoint_or_capacity_review"},
        "claim_boundary": "Prepared continuation obligation; not provider-call blocking, provider usage, compaction, or billing evidence.",
        **observations(pack),
    }


def render_context_budget(projection: BudgetContext) -> str:
    """Stable bounded suffix: retain pack identity, never copy its contents."""
    pack = projection["must_keep_pack"]
    digest = pack["digest"] if pack else "unknown"
    return (
        f"[OMH Context Budget] action={projection['invalidation']['action']}; "
        f"reason={projection['invalidation']['reason']}; usable_tokens={projection['usable_budget_tokens']['value']}; "
        f"must_keep={digest}. "
        "Before accumulating more context, resolve any hold with one checkpoint/capacity review, "
        "preserve the must-keep pack, then explicitly prepare the plan. "
        "Host provider is unknown. No provider-call blocking, usage, compaction or billing is observed."
    )
