"""Read-only reuse of Hermes' configurable architect and ultrabrain chains."""
import re
from typing import Any

from ..plugin_bundle.omh.hermes_delegation import (
    effective_mixture_category_chains, effective_provider_entitlements,
    load_model_provider_routes, provider_serves_alias, providers_serving_alias,
    resolve_provider_model,
)


def resolve_campaign_routes(*, omh_home=None, hermes_home=None, owner_model="", owner_provider="",
                            owner_effort="", worker_model="", worker_provider="",
                            worker_effort="") -> dict[str, Any]:
    for value in (owner_model, owner_provider, owner_effort, worker_model, worker_provider, worker_effort):
        if not isinstance(value, str) or (value and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value)):
            raise ValueError("campaign_route_must_be_token")
    chains = effective_mixture_category_chains(omh_home, hermes_home)
    aliases, route_status = load_model_provider_routes(omh_home)
    entitlements, entitlement_status, _providers = effective_provider_entitlements(omh_home, hermes_home)
    routes = {}
    for role, category, model, provider, effort in (
        ("root", "architect", owner_model, owner_provider, owner_effort),
        ("worker", "ultrabrain", worker_model, worker_provider, worker_effort),
    ):
        choices = ((model, effort or "high"),) if model else chains[category]
        route: dict[str, Any] = dict(category=category, alias="", wire_model="", provider="",
                     reasoning_effort=effort, source="unavailable", execution="not_observed")
        for index, (alias, chain_effort) in enumerate(choices):
            wire, chosen = resolve_provider_model(alias, provider, aliases)
            candidates = providers_serving_alias(alias, entitlements)
            if not chosen and candidates:
                chosen = candidates[0]
            if chosen and provider_serves_alias(alias, chosen, entitlements) is False:
                continue
            route.update(alias=alias, wire_model=wire, provider=chosen,
                         reasoning_effort=effort or chain_effort,
                         source="user_override" if model or provider or effort else
                         "chain_fallthrough" if index else "chain_head")
            break
        route["configuration_status"] = [route_status, entitlement_status]
        routes[role] = route
    return routes
