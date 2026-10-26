"""Classify whether an ordinary route may receive advisory domain context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


DISPATCH_ROUTE = "dispatch_route"
PROTECTED_ROUTE = "protected_route"
ELIGIBLE_UNRESOLVED_ROUTE = "eligible_unresolved_route"
_WEAK_DISPATCH_EVIDENCE = "weak_dispatch_evidence"

_UNRESOLVED_ACTIONS = frozenset({"clarify", "fallback"})
_ROUTER_SKILL = "oh-my-hermes"
_PROTECTED_RECOMMENDATION_ACTIONS = frozenset({"answer_directly", "answer_file_lookup"})
_PROTECTED_LEADING_MATCHES = frozenset(
    {"category:maintenance", "category:operator", "metadata:help", "metadata:status", "trigger:show"}
)


@dataclass(frozen=True)
class DomainContextEligibility:
    eligible: bool
    reason: str


def classify_domain_context_eligibility(
    route: Mapping[str, object],
    message: str,
) -> DomainContextEligibility:
    """Return a stable gate decision without changing the route or message.

    The ordinary router remains authoritative. This gate admits only its
    unresolved router-owned clarification shapes; it does not reinterpret
    confidence, recommendations, or candidate handoffs as permission.
    """
    action = str(route.get("action") or "")
    if action == "dispatch":
        return DomainContextEligibility(False, DISPATCH_ROUTE)
    if action not in _UNRESOLVED_ACTIONS:
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if not message.strip() or _has_explicit_invocation_prefix(message):
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if str(route.get("selected_skill") or "") != _ROUTER_SKILL:
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    # A clarify the dispatch-evidence gate made from a confident winner is a
    # dispatch the router declined, not an unresolved router question; it
    # stays as ineligible as the dispatch it replaced.
    if str(route.get("ambiguity_kind") or "") == _WEAK_DISPATCH_EVIDENCE:
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if route.get("explicit") is not False:
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if not str(route.get("clarification") or "").strip():
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if _governance_blocks(route.get("skill_governance")):
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if any(
        route.get(field) is not None
        for field in ("task_card", "workflow_route_plan", "learning_candidate_card")
    ):
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if str(route.get("route_next_action") or "").strip():
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if _has_protected_recommendation_action(route.get("recommendations")):
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    if _has_protected_leading_match(route):
        return DomainContextEligibility(False, PROTECTED_ROUTE)
    return DomainContextEligibility(True, ELIGIBLE_UNRESOLVED_ROUTE)


def _has_explicit_invocation_prefix(message: str) -> bool:
    first = message.strip().split(maxsplit=1)[0].strip(":,").lower()
    return first.startswith(("$", "/", "./", "@"))


def _governance_blocks(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, Mapping):
        return True
    return value.get("status") != "resolved"


def _has_protected_recommendation_action(value: object) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return any(
        isinstance(item, Mapping)
        and str(item.get("next_action") or "") in _PROTECTED_RECOMMENDATION_ACTIONS
        for item in value
    )


def _has_protected_leading_match(route: Mapping[str, object]) -> bool:
    value = route.get("recommendations")
    if not isinstance(value, (list, tuple)):
        return False
    leading = [
        item
        for item in value
        if isinstance(item, Mapping) and item.get("score") == route.get("score")
    ]
    return bool(leading) and all(
        isinstance(item.get("matched"), (list, tuple))
        and any(match in _PROTECTED_LEADING_MATCHES for match in item["matched"])
        for item in leading
    )
