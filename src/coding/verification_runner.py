"""Tiered execution of a compiled verification plan.

The engine mirrors the dispatcher's unit-level dependency frontier at check
granularity: a node is admitted the moment every node it depends on has
passed, read-only nodes dispatch as a bounded parallel wave (width supplied
by the caller from the existing parallelism policy — never a literal),
stateful nodes serialize on a per-resource-class lock, and integration-tier
nodes hold until the producer-lane fan-in gate opens.

Receipts decide reuse before any process starts: a node whose revision-bound
key already has a passing or failed in-scope receipt shares that receipt and
spawns nothing. A receipt that is missing, stale (any key component moved),
or scope-insufficient is treated as missing evidence and the check runs
fresh. Fail-fast is scoped: a failed node blocks its dependents (recorded
`skipped`, never run) while unrelated nodes run to completion. The aggregate
is explicit — `all_passed` is true only when every node has fresh or
reused in-scope passing evidence; anything else is a HOLD, never a PASS.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
import time

from ..system.local_store import utc_now
from .verification_execution import VerificationExecutionGate
from .verification_outcomes import (
    NodeOutcome,
    PlanRunContext,
    PlanRunResult,
    RunNode,
)
from .verification_plan import (
    ReceiptKey,
    VerificationNode,
    VerificationPlan,
    has_secret_execution_environment,
    receipt_key,
    toolchain_digest,
)
from .verification_receipts import file_receipt, receipt_file_lock, receipt_hit_status

__all__ = [
    "NodeOutcome",
    "PlanRunContext",
    "PlanRunResult",
    "RunNode",
    "run_verification_plan",
]


def _receipt_hit(context: PlanRunContext, node: VerificationNode, key: ReceiptKey) -> NodeOutcome | None:
    """The shared-receipt outcome for one node, or None when evidence is missing."""
    status = receipt_hit_status(
        context.paths, key, claim_scope=node.claim_scope, revision=context.revision
    )
    if status is None:
        return None
    return NodeOutcome(
        node=node,
        status=status,
        detail="" if status == "passed" else f"reused failed receipt {key}",
        reused=True,
        receipt_key=str(key),
        truncation=None,
        deferred=False,
    )


def _produce_node(
    context: PlanRunContext,
    node: VerificationNode,
    key: ReceiptKey | None,
    queued_at: str,
    run_node: RunNode,
) -> NodeOutcome:
    """Run one uncached node and persist its immutable metadata receipt."""
    started_at = utc_now()
    started = time.monotonic()
    status, detail, truncation = run_node(node)
    finished_at = utc_now()
    if key is not None and context.revision is not None:
        file_receipt(
            context.paths,
            key=key,
            node=node,
            revision=context.revision,
            status=status,
            queued_at=queued_at,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.monotonic() - started,
        )
    return NodeOutcome(
        node=node,
        status=status,
        detail=detail,
        reused=False,
        receipt_key=str(key) if key is not None else None,
        truncation=truncation,
        deferred=False,
    )


def _execute_node(
    context: PlanRunContext, node: VerificationNode, key: ReceiptKey | None, run_node: RunNode
) -> NodeOutcome:
    """Run one node under single-flight and file its immutable receipt."""
    queued_at = utc_now()

    def produce() -> NodeOutcome:
        if key is not None:
            with receipt_file_lock(context.paths, key, timeout_seconds=node.timeout + 60):
                hit = _receipt_hit(context, node, key)
                if hit is not None:
                    return hit
                return _produce_node(context, node, key, queued_at, run_node)
        return _produce_node(context, node, key, queued_at, run_node)

    if key is None:
        return produce()
    outcome, _reused = context.single_flight.run(str(key), produce, wait_timeout=node.timeout + 60)
    return outcome


def _dependencies_failed(node: VerificationNode, outcomes: dict[str, NodeOutcome], by_declared: dict[str, str]) -> str | None:
    """The blocking dependency's declared id, or None when none failed."""
    for dep in node.depends_on:
        outcome = outcomes.get(by_declared.get(dep, ""))
        if outcome is not None and outcome.status != "passed":
            return dep
    return None


def _skipped(node: VerificationNode, *, detail: str, deferred: bool) -> NodeOutcome:
    """A recorded, never-run outcome: blocked or fan-in-deferred evidence."""
    return NodeOutcome(
        node=node,
        status="skipped",
        detail=detail,
        reused=False,
        receipt_key=None,
        truncation=None,
        deferred=deferred,
    )


def _deps_passed(node: VerificationNode, outcomes: dict[str, NodeOutcome], by_declared: dict[str, str]) -> bool:
    return all(
        (outcome := outcomes.get(by_declared.get(dep, ""))) is not None and outcome.status == "passed"
        for dep in node.depends_on
    )


def run_verification_plan(
    context: PlanRunContext,
    plan: VerificationPlan,
    *,
    run_node: RunNode,
) -> PlanRunResult:
    """Run one compiled plan and return every node's outcome in declared order."""
    by_declared = {node.declared_id: node.check_id for node in plan.nodes}
    keys: dict[str, ReceiptKey | None] = {}
    for node in plan.nodes:
        keys[node.check_id] = (
            None
            if context.revision is None
            or has_secret_execution_environment(node, environment=context.execution_environment)
            else receipt_key(
                node,
                repo_identity=str(context.worktree.resolve()),
                revision=context.revision,
                toolchain=toolchain_digest(
                    node, worktree=context.worktree, environment=context.execution_environment
                ),
            )
        )
    outcomes: dict[str, NodeOutcome] = {}
    pending: list[VerificationNode] = []
    for node in plan.nodes:
        key = keys[node.check_id]
        hit = _receipt_hit(context, node, key) if key is not None else None
        if hit is not None:
            outcomes[node.check_id] = hit
        else:
            pending.append(node)

    execution_gate = context.execution_gate or VerificationExecutionGate(context.max_workers)
    owns_execution_gate = context.execution_gate is None

    def _task(node: VerificationNode) -> NodeOutcome:
        return _execute_node(context, node, keys[node.check_id], run_node)

    try:
        submitted: dict[Future[NodeOutcome], VerificationNode] = {}
        while pending or submitted:
            progressed = False
            for node in list(pending):
                blocker = _dependencies_failed(node, outcomes, by_declared)
                if blocker is not None:
                    outcomes[node.check_id] = _skipped(
                        node, detail=f"blocked: dependency {blocker} did not pass", deferred=False
                    )
                    pending.remove(node)
                    progressed = True
                    continue
                if not _deps_passed(node, outcomes, by_declared):
                    continue
                if node.tier == "integration" and not context.integration_ready():
                    continue
                submitted[execution_gate.submit(node, lambda node=node: _task(node))] = node
                pending.remove(node)
                progressed = True
            if not submitted:
                # Nothing admissible and nothing running: the remainder is
                # integration-gated behind a closed fan-in, plus anything
                # depending on it. Recorded, never run — missing evidence,
                # never a pass.
                for node in list(pending):
                    blocker = (
                        None
                        if node.tier == "integration"
                        else _dependencies_failed(node, outcomes, by_declared)
                    )
                    outcomes[node.check_id] = _skipped(
                        node,
                        detail=(
                            f"blocked: dependency {blocker} did not pass"
                            if blocker is not None
                            else "deferred until producer lanes fan in"
                        ),
                        deferred=blocker is None,
                    )
                pending.clear()
                break
            if not progressed:
                done, _ = wait(tuple(submitted), return_when=FIRST_COMPLETED)
                for future in done:
                    node = submitted.pop(future)
                    outcomes[node.check_id] = future.result()
    finally:
        if owns_execution_gate:
            execution_gate.shutdown()

    return PlanRunResult(outcomes=tuple(outcomes[node.check_id] for node in plan.nodes))
