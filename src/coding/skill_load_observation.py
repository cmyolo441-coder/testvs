"""Fail-closed, opt-in Hermes loaded-skill inventory observation.

A successful child process, model response, static skill listing, or scheduler
source is not an inventory. ``observed`` is emitted only for the closed,
nonce-bound machine protocol parsed here. Installed Hermes versions that do
not implement that protocol are reported as ``unsupported``.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import os
from threading import Lock
from typing import cast

from .hermes_child_dispatch import require_hermes_child_dispatch_boundary
from .skill_load_process import (
    ProcessOutcome as _ProcessOutcome,
    ResolvedTool as _ResolvedTool,
    SkillLoadProbeRequest,
    inventory_argv as _inventory_argv,
    probe_environment as _probe_environment,
    resolve_executable as _resolve_executable,
    run_inventory_process as _run_inventory_process,
    snapshot_resolved_tool as _snapshot_resolved_tool,
)
from .skill_load_protocol import (
    HERMES_SKILL_INVENTORY_SCHEMA_VERSION,
    HEX_64 as _HEX_64,
    LOAD_STATES,
    PROBE_STATUSES,
    REASON_CODES,
    SKILL_LOAD_OBSERVATION_SCHEMA_VERSION,
    closed_observation as _closed_observation,
    expected_skills_digest,
    normalized_skill_set as _normalized_skill_set,
    set_digest as _set_digest,
    skill_load_observation_is_fresh,
    utc as _utc,
    validate_skill_load_observation,
    validated_inventory_response as _validated_inventory_response,
)

__all__ = (
    "HERMES_SKILL_INVENTORY_SCHEMA_VERSION", "InventoryNonceLedger", "LOAD_STATES",
    "PROBE_STATUSES", "REASON_CODES", "SKILL_LOAD_OBSERVATION_SCHEMA_VERSION",
    "SkillLoadProbeRequest", "expected_skills_digest", "probe_skill_load",
    "skill_load_observation_is_fresh", "validate_skill_load_observation",
    "_ProcessOutcome", "_ResolvedTool", "_inventory_argv", "_resolve_executable",
)


class InventoryNonceLedger:
    """Bounded in-process replay ledger for validated protocol nonces."""

    def __init__(self, *, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError("nonce ledger capacity must be positive")
        self._capacity = capacity
        self._ordered: list[str] = []
        self._seen: set[str] = set()
        self._lock = Lock()

    def contains(self, nonce: str) -> bool:
        with self._lock:
            return nonce in self._seen

    def consume(self, nonce: str) -> bool:
        with self._lock:
            if nonce in self._seen:
                return False
            self._seen.add(nonce)
            self._ordered.append(nonce)
            if len(self._ordered) > self._capacity:
                self._seen.discard(self._ordered.pop(0))
            return True


def probe_skill_load(
    request: SkillLoadProbeRequest,
    *,
    confirmed: bool,
    nonce_ledger: InventoryNonceLedger | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Run the explicit machine inventory protocol once after confirmation."""
    require_hermes_child_dispatch_boundary(
        dispatch_policy="ask_before_dispatch", confirmed=confirmed
    )
    if request.timeout_seconds <= 0:
        raise ValueError("skill-load probe timeout must be positive")
    expected = _normalized_skill_set(request.expected_skills, field="expected_skills")
    nonce = request.nonce or os.urandom(32).hex()
    if not _HEX_64.fullmatch(nonce):
        raise ValueError("skill-load probe nonce must be 64 lowercase hexadecimal characters")
    expected_digest = _set_digest(expected)
    runtime_unavailable = hashlib.sha256(b"runtime-unavailable").hexdigest()
    unavailable_tool = hashlib.sha256(b"unresolved-executable").hexdigest()
    observed_at = _utc(now)
    if nonce_ledger is not None and nonce_ledger.contains(nonce):
        return _closed_observation(
            "probe_error", "inventory_nonce_replayed", nonce, expected_digest,
            unavailable_tool, runtime_unavailable, observed_at,
        )
    env = _probe_environment(request.env)
    try:
        tool = _snapshot_resolved_tool(request.hermes, env)
    except OSError:
        return _closed_observation(
            "probe_error", "inventory_executable_unavailable", nonce, expected_digest,
            unavailable_tool, runtime_unavailable, observed_at,
        )
    tool_fingerprint = tool.fingerprint
    try:
        outcome = _run_inventory_process(
            request, tool.executable, env, nonce, expected_digest, tool_fingerprint
        )
    finally:
        tool.cleanup()
    if outcome.kind == "timeout":
        return _closed_observation(
            "probe_error", "inventory_timeout", nonce, expected_digest,
            tool_fingerprint, runtime_unavailable, observed_at,
        )
    if outcome.kind == "spawn_error":
        return _closed_observation(
            "probe_error", "inventory_process_error", nonce, expected_digest,
            tool_fingerprint, runtime_unavailable, observed_at,
        )
    if outcome.exit_code == 2:
        return _closed_observation(
            "unsupported", "inventory_protocol_unavailable", nonce, expected_digest,
            tool_fingerprint, runtime_unavailable, observed_at,
        )
    if outcome.exit_code != 0:
        return _closed_observation(
            "probe_error", "inventory_process_error", nonce, expected_digest,
            tool_fingerprint, runtime_unavailable, observed_at,
        )
    response, reason = _validated_inventory_response(
        outcome.stdout, nonce=nonce, expected_digest=expected_digest,
        tool_fingerprint=tool_fingerprint, now=observed_at,
    )
    if response is None:
        return _closed_observation(
            "probe_error", reason, nonce, expected_digest, tool_fingerprint,
            runtime_unavailable, observed_at,
        )
    if nonce_ledger is not None and not nonce_ledger.consume(nonce):
        return _closed_observation(
            "probe_error", "inventory_nonce_replayed", nonce, expected_digest,
            tool_fingerprint, runtime_unavailable, observed_at,
        )
    observed_value = response["observed_skills"]
    if not isinstance(observed_value, list):
        raise ValueError("validated inventory lost its observed skill list")
    observed = tuple(cast(list[str], observed_value))
    missing = tuple(sorted(set(expected) - set(observed)))
    unexpected = tuple(sorted(set(observed) - set(expected)))
    state = (
        "not_applicable" if not expected else "none_loaded" if set(expected).isdisjoint(observed)
        else "partially_loaded" if missing else "all_loaded"
    )
    payload: dict[str, object] = {
        "schema_version": SKILL_LOAD_OBSERVATION_SCHEMA_VERSION,
        "probe_status": "observed", "reason_code": "inventory_validated", "nonce": nonce,
        "expected_digest": expected_digest, "inventory_digest": response["inventory_digest"],
        "tool_fingerprint": tool_fingerprint,
        "runtime_fingerprint": response["runtime_fingerprint"],
        "observed_at": response["observed_at"], "expires_at": response["expires_at"],
        "load_state": state, "expected_skills": list(expected),
        "observed_skills": list(observed), "missing_skills": list(missing),
        "unexpected_skills": list(unexpected),
    }
    if validate_skill_load_observation(payload):
        return _closed_observation(
            "probe_error", "inventory_response_malformed", nonce, expected_digest,
            tool_fingerprint, runtime_unavailable, observed_at,
        )
    return payload
