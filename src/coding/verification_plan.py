"""Executor-neutral verification plan compilation and receipt keying.

A contract unit's declared verification becomes a `verification_plan/v1`: a
typed node per check (stable id, tier, safety, dependency edges, timeout,
claim scope) that any executor lane or the dispatcher itself can schedule
against. The plan is pure compilation — no I/O beyond reading lockfiles for
the toolchain digest — so the same unit compiles to the same plan in every
lane.

Receipt keys bind one check to exactly one evidence context: repository /
worktree identity + exact revision + normalized argv + toolchain/config
digest + claim scope. Command text alone is never a key (the issue's
rejected alternative): two spellings of one command share a key only when
every evidence-bearing component matches, and any component change —
including a claim-scope change, so cached evidence can never cross scopes —
produces a different key.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, NewType

from .fanout_contracts import (
    VERIFICATION_CHECK_CLAIM_SCOPES,
    VERIFICATION_CHECK_SAFETIES,
    VERIFICATION_CHECK_TIERS,
    is_verification_check_resource_class,
    verification_command_argv,
)
from .verification_environment import (
    has_secret_execution_environment as has_secret_execution_environment,
    toolchain_digest as toolchain_digest,
    verification_execution_environment as verification_execution_environment,
)

ReceiptKey = NewType("ReceiptKey", str)

VERIFICATION_PLAN_SCHEMA_VERSION = "verification_plan/v1"
VERIFICATION_PLAN_CLAIM_BOUNDARY = (
    "A verification plan compiles the checks one contract unit declares into typed, schedulable nodes. "
    "It is not evidence that any check ran, and a passing targeted check is not full verification, "
    "review, CI, merge-readiness, or merge evidence."
)
# Mirrors the dispatcher's per-command ceiling; a contract may only narrow it.
VERIFICATION_CHECK_DEFAULT_TIMEOUT = 600
UNIT_VERIFICATION_CLAIM_SCOPE = "unit_verification"
DEFAULT_RESOURCE_CLASS = "local_cpu"

@dataclass(frozen=True, slots=True)
class VerificationNode:
    """One schedulable check: stable id plus every field a wave scheduler needs."""

    check_id: str
    declared_id: str
    command: str
    argv: tuple[str, ...]
    env_overrides: tuple[tuple[str, str], ...]
    tier: str
    safety: str
    resource_class: str
    timeout: int
    depends_on: tuple[str, ...]
    claim_scope: str


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    """A unit's compiled checks in declared order."""

    schema_version: str
    fanout_id: str
    unit_id: str
    nodes: tuple[VerificationNode, ...]
    claim_boundary: str

    @property
    def is_serial(self) -> bool:
        """True when the plan can only ever replay the legacy serial loop."""
        return all(
            node.tier == "unit" and node.safety == "stateful" and not node.depends_on
            for node in self.nodes
        )


def _node_metadata(unit: Mapping[str, Any], commands: list[str]) -> list[Mapping[str, Any]]:
    """Per-check metadata aligned with the command list, defaults applied."""
    declared = unit.get("verification_checks")
    if isinstance(declared, (list, tuple)) and declared:
        return [entry if isinstance(entry, Mapping) else {} for entry in declared]
    return [{} for _ in commands]


def compile_verification_plan(
    unit: Mapping[str, Any], *, fanout_id: str, unit_id: str
) -> VerificationPlan | None:
    """Compile one contract unit's declared verification into a typed plan.

    Returns None when the unit declares no runnable command. Metadata-free
    units compile to an all-stateful, dependency-free, unit-tier plan whose
    execution is observationally identical to the legacy serial loop; the
    additive `verification_checks` key opts into tiers, safety, and edges.
    Defaults are applied here (not only at freeze) so a plan compiles the
    same way from a frozen contract or a raw unit mapping.
    """
    declared = unit.get("verification_commands")
    commands = [str(entry) for entry in declared] if isinstance(declared, (list, tuple)) else []
    if not commands:
        return None
    metadata = _node_metadata(unit, commands)
    nodes: list[VerificationNode] = []
    for index, command in enumerate(commands):
        meta = metadata[index] if index < len(metadata) else {}
        env, argv = verification_command_argv(command)
        declared_id = str(meta.get("id") or f"check-{index}")
        raw_tier = str(meta.get("tier") or "unit")
        tier = raw_tier if raw_tier in VERIFICATION_CHECK_TIERS else "unit"
        raw_safety = str(meta.get("safety") or "stateful")
        safety = raw_safety if raw_safety in VERIFICATION_CHECK_SAFETIES else "stateful"
        raw_resource_class = str(meta.get("resource_class") or DEFAULT_RESOURCE_CLASS)
        resource_class = (
            raw_resource_class
            if is_verification_check_resource_class(raw_resource_class)
            else DEFAULT_RESOURCE_CLASS
        )
        expected_scope = "integrated_verification" if tier == "integration" else UNIT_VERIFICATION_CLAIM_SCOPE
        raw_claim_scope = str(meta.get("claim_scope") or expected_scope)
        claim_scope = (
            raw_claim_scope
            if raw_claim_scope in VERIFICATION_CHECK_CLAIM_SCOPES and raw_claim_scope == expected_scope
            else expected_scope
        )
        depends_on = tuple(str(dep) for dep in meta.get("depends_on") or ())
        raw_timeout = meta.get("timeout")
        timeout = raw_timeout if isinstance(raw_timeout, int) and raw_timeout > 0 else VERIFICATION_CHECK_DEFAULT_TIMEOUT
        check_id = hashlib.sha256(
            json.dumps(
                [fanout_id, unit_id, index, argv, sorted(dict(env).items())],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]
        nodes.append(
            VerificationNode(
                check_id=check_id,
                declared_id=declared_id,
                command=command,
                argv=tuple(argv),
                env_overrides=tuple(sorted(env.items())),
                tier=tier,
                safety=safety,
                resource_class=resource_class,
                timeout=timeout,
                depends_on=depends_on,
                claim_scope=claim_scope,
            )
        )
    return VerificationPlan(
        schema_version=VERIFICATION_PLAN_SCHEMA_VERSION,
        fanout_id=fanout_id,
        unit_id=unit_id,
        nodes=tuple(nodes),
        claim_boundary=VERIFICATION_PLAN_CLAIM_BOUNDARY,
    )


def receipt_key(
    node: VerificationNode, *, repo_identity: str, revision: str, toolchain: str
) -> ReceiptKey:
    """The one key one check's evidence may ever be filed under.

    Every component is an invalidator: repository/worktree identity, exact
    revision, normalized argv plus env overrides (never raw command text
    alone), toolchain/config digest, and claim scope. Two consumers computing
    the same key share one receipt; any change computes a different one.
    """
    payload = json.dumps(
        {
            "argv": list(node.argv),
            "claim_scope": node.claim_scope,
            "depends_on": list(node.depends_on),
            "env": [list(pair) for pair in node.env_overrides],
            "resource_class": node.resource_class,
            "safety": node.safety,
            "tier": node.tier,
            "timeout": node.timeout,
            "repo_identity": repo_identity,
            "revision": revision,
            "toolchain": toolchain,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ReceiptKey(hashlib.sha256(payload.encode("utf-8")).hexdigest())
