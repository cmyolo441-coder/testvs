"""Typed route metadata for Hermes-child observations."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import TypeAlias

from ..coding.routing_observation import (
    JsonValue,
    authenticate_child_observation,
    authenticate_executor_observation,
    build_routing_observation,
)

JsonScalar: TypeAlias = str | int | float | bool | None


def route(args: argparse.Namespace) -> dict[str, JsonValue]:
    """Build the route projection shared by prepared and observed payloads."""
    return {
        "selected_model": f"{args.provider}/{args.model}",
        "selected_reasoning_effort": args.reasoning,
        "role": "agent_maintainer",
        "executor_profile": "hermes_child",
        "chain": [
            {
                "provider": args.provider,
                "model_id": args.model,
                "reasoning_effort": args.reasoning,
            }
        ],
    }


def result_payload(
    args: argparse.Namespace,
    status: str,
    usage: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    """Build the authenticated terminal observation from bounded usage."""
    session: dict[str, JsonValue] = {"status": status}
    session.update(
        {
            key: usage[key]
            # `cost_status`/`cost_source` travel with the cost they explain:
            # without them a zero cannot be told apart from a provider that
            # reported nothing, and the surface printed a confident $0.00.
            for key in (
                "provider",
                "model",
                "total_tokens",
                "estimated_cost_usd",
                "cost_status",
                "cost_source",
            )
            if key in usage
        }
    )
    if "total_tokens" in session:
        session["tokens"] = session.pop("total_tokens")
    if "estimated_cost_usd" in session:
        session["cost_usd"] = session.pop("estimated_cost_usd")
    if "tokens" not in session:
        token_parts = [
            usage.get(key)
            for key in ("input_tokens", "output_tokens", "reasoning_tokens")
        ]
        observed_parts = [
            value
            for value in token_parts
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        if observed_parts:
            session["tokens"] = sum(observed_parts)
    for source, target in (
        ("turns", "turn"),
        ("tool_calls", "tools"),
        ("cost_usd", "cost_usd"),
    ):
        value = usage.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            session[target] = value
    return build_routing_observation(
        route=route(args),
        child_dispatch=authenticate_child_observation(
            {"status": status, "run_id": args.run_id}
        ),
        session_observation=authenticate_executor_observation(session),
        parent_session_id=args.parent_run_id,
        child_session_id=args.run_id,
        run_id=args.run_id,
    )
