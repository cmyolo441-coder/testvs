"""Advisory host-evaluation contract; no provider client or automatic calls."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, TypedDict


@dataclass(frozen=True)
class ModelRequest:
    claim_id: str
    question: str
    invariant: str
    expected_fact: bool | str
    evidence: str
    input_digest: str


@dataclass(frozen=True)
class ModelAdapter:
    """Host-declared run identity, available even when evaluation times out."""

    evaluate: Callable[[ModelRequest], object]
    provider: str
    model: str
    run_id: str
    adapter: str


class ModelRun(TypedDict):
    state: str
    provider: str
    model: str
    run_id: str
    adapter: str
    claim_id: str
    input_digest: str
    cost_status: str
    cost_usd: float | int | None


def _identifier(value: object) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}", value)
            or any(token in value.lower() for token in (
                "sk-", "token", "secret", "password", "bearer", "ghp_", "github_pat_",
                "xoxb-", "xoxp-", "akia", "aiza", "api_key", "apikey",
            ))):
        raise ValueError("model_provenance")
    return value


def model_provenance(adapter: ModelAdapter, request: ModelRequest) -> ModelRun:
    return {
        "state": "unresolved", "cost_status": "unknown", "cost_usd": None,
        "claim_id": request.claim_id, "input_digest": request.input_digest,
        "provider": _identifier(adapter.provider), "model": _identifier(adapter.model),
        "run_id": _identifier(adapter.run_id), "adapter": _identifier(adapter.adapter),
    }


def model_result(adapter: ModelAdapter, request: ModelRequest) -> ModelRun:
    result = adapter.evaluate(request)
    if not isinstance(result, dict):
        raise ValueError("model_result_shape")
    state = result.get("state")
    if not isinstance(state, str) or state not in {"supported", "stale", "unresolved"}:
        raise ValueError("model_state")
    clean = model_provenance(adapter, request)
    for key in ("provider", "model", "run_id", "adapter", "claim_id", "input_digest"):
        if result.get(key) != clean.get(key):
            raise ValueError("model_provenance_mismatch")
    cost_status, cost = result.get("cost_status"), result.get("cost_usd")
    if not isinstance(cost_status, str) or cost_status not in {"reported", "unknown", "not_incurred"}:
        raise ValueError("model_cost_status")
    if cost_status == "reported":
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or not math.isfinite(cost) or cost < 0:
            raise ValueError("model_cost")
    elif cost is not None:
        raise ValueError("model_cost")
    return {**clean, "state": state, "cost_status": cost_status, "cost_usd": cost}
