"""Executor-neutral post-integration verification surface.

Integration checks are never evidence about a producer worktree. This module
accepts one caller-supplied integrated worktree and its already-verified
revision, refuses to run until producer completion evidence exists, and runs
only integration-tier nodes. Consumers share the normal revision-bound receipt
key, so one integrated full gate is never duplicated for the same target.
"""

from __future__ import annotations

from dataclasses import replace

from .verification_outcomes import NodeOutcome, PlanRunContext, PlanRunResult, RunNode
from .verification_plan import VerificationPlan
from .verification_runner import run_verification_plan


def run_post_integration_verification(
    context: PlanRunContext,
    plan: VerificationPlan,
    *,
    producer_evidence: bool,
    run_node: RunNode,
) -> PlanRunResult:
    """Run one plan's integrated checks only after exact producer evidence.

    A missing revision, missing producer completion evidence, or a plan with no
    integration checks is HOLD rather than a pass. The caller owns binding the
    supplied revision to the integrated worktree before constructing context.
    """
    integration_ids = {node.declared_id for node in plan.nodes if node.tier == "integration"}
    integrated_nodes = tuple(
        replace(node, depends_on=tuple(dep for dep in node.depends_on if dep in integration_ids))
        for node in plan.nodes
        if node.tier == "integration"
    )
    integrated_plan = replace(plan, nodes=integrated_nodes)
    if not integrated_nodes:
        return PlanRunResult(outcomes=())
    if not producer_evidence or context.revision is None:
        return PlanRunResult(
            outcomes=tuple(
                NodeOutcome(
                    node=node,
                    status="skipped",
                    detail="held: missing producer completion or integrated revision evidence",
                    reused=False,
                    receipt_key=None,
                    truncation=None,
                    deferred=True,
                )
                for node in integrated_nodes
            )
        )
    return run_verification_plan(
        replace(context, integration_ready=lambda: True), integrated_plan, run_node=run_node
    )
