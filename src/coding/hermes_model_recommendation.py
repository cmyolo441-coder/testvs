"""The Hermes model recommendation a coding handoff binds its media gate to.

One producer for every surface that prepares a Hermes coding handoff -- the
chat lane and `omh coding delegate` -- so the modality route both of them
scope executor evidence to is the same resolved provider and wire model.
Reads local model discovery only; it is not provider availability, dispatch,
or execution evidence.
"""
from __future__ import annotations

from ..system.paths import OmhPaths
from .model_discovery import discover_local_models
from .model_routing import resolve_model_route


def resolved_hermes_model_recommendation(
    executor_target: str,
    paths: OmhPaths | None,
) -> dict[str, object] | None:
    """The resolved recommendation for a Hermes-owned handoff, or None.

    External owners resolve no recommendation here: their model route is a
    Maestro concern, and a handoff without one keeps its media gate at
    `route_unresolved` rather than borrowing a route it never confirmed.
    """
    if executor_target != "hermes" or paths is None:
        return None
    discovery = discover_local_models(paths.hermes_home.parent)
    observations = discovery.get("observations", [])
    active_models = [
        {
            **observation,
            "model_alias": str(observation.get("model_id", "")).rsplit("/", 1)[-1],
            "provider_family": str(observation.get("provider", "")),
            "compatible_owners": ["hermes"],
        }
        for observation in observations
        if isinstance(observation, dict)
        and observation.get("status") == "confirmed_active"
        and observation.get("model_id")
    ] if isinstance(observations, list) else []
    route = resolve_model_route(
        "hermes",
        role="implementation",
        active_models=active_models,
    )
    recommendation = route.get("recommendation")
    return dict(recommendation) if isinstance(recommendation, dict) else None
