"""Cell expansion and terminal-state reduction for paired-run dispatches."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from ..quality.paired_run_model import ArmRole, PairedRunDecision
from .paired_run_dispatch_model import (
    ArmDispatchTarget,
    DispatchBudgets,
    PairedRunDispatchCell,
    PairedRunDispatchPlan,
    PairedRunDispatchPlanError,
    TerminalState,
)
from .paired_run_dispatch_validation import named_limits


def build_cells(
    decision: PairedRunDecision,
    budgets: DispatchBudgets,
    targets: dict[ArmRole, ArmDispatchTarget],
) -> tuple[PairedRunDispatchCell, ...]:
    """Expand the frozen task-by-arm matrix into deterministic isolated cells."""
    executor_limits = named_limits(budgets.executor_concurrency, "executor")
    provider_limits = named_limits(budgets.provider_concurrency, "provider")
    wave_usage: list[tuple[int, dict[str, int], dict[str, int], set[str]]] = []
    cells: list[PairedRunDispatchCell] = []
    for task in decision.tasks:
        for role in (ArmRole.BASELINE, ArmRole.VARIANT):
            target = targets[role]
            wave = _reserve_wave(
                wave_usage,
                budgets.global_concurrency,
                executor_limits[target.executor],
                provider_limits[target.provider],
                target.executor,
                target.provider,
                target.shared_resource_key,
            )
            cells.append(PairedRunDispatchCell(
                task.task_id,
                role,
                task.input_digest,
                decision.execution_revision,
                target.executor,
                target.provider,
                target.model,
                _workspace_id(decision.decision_id, task.task_id, role),
                wave,
                target.shared_resource_key,
            ))
    return tuple(cells)


def record_terminal_state(
    plan: PairedRunDispatchPlan,
    workspace_id: str,
    terminal_state: TerminalState,
) -> PairedRunDispatchPlan:
    """Return a new plan with exactly one previously-pending cell terminal."""
    if not isinstance(terminal_state, TerminalState):
        raise PairedRunDispatchPlanError("terminal_state must be an explicit terminal state")
    cells: list[PairedRunDispatchCell] = []
    found = False
    for cell in plan.cells:
        if cell.workspace_id != workspace_id:
            cells.append(cell)
            continue
        if cell.terminal_state is not None:
            raise PairedRunDispatchPlanError("a terminal cell cannot be rewritten")
        cells.append(replace(cell, terminal_state=terminal_state))
        found = True
    if not found:
        raise PairedRunDispatchPlanError("workspace_id is not in this paired-run plan")
    return replace(plan, cells=tuple(cells))


def _reserve_wave(
    usage: list[tuple[int, dict[str, int], dict[str, int], set[str]]],
    global_limit: int,
    executor_limit: int,
    provider_limit: int,
    executor: str,
    provider: str,
    shared_key: str | None,
) -> int:
    for index, (global_used, executor_used, provider_used, shared_keys) in enumerate(usage):
        if global_used >= global_limit:
            continue
        if executor_used.get(executor, 0) >= executor_limit:
            continue
        if provider_used.get(provider, 0) >= provider_limit:
            continue
        if shared_key is not None and shared_key in shared_keys:
            continue
        _claim_wave(usage, index, executor, provider, shared_key)
        return index
    usage.append((0, {}, {}, set()))
    index = len(usage) - 1
    _claim_wave(usage, index, executor, provider, shared_key)
    return index


def _claim_wave(
    usage: list[tuple[int, dict[str, int], dict[str, int], set[str]]],
    index: int,
    executor: str,
    provider: str,
    shared_key: str | None,
) -> None:
    global_used, executor_used, provider_used, shared_keys = usage[index]
    executor_used[executor] = executor_used.get(executor, 0) + 1
    provider_used[provider] = provider_used.get(provider, 0) + 1
    if shared_key is not None:
        shared_keys.add(shared_key)
    usage[index] = (global_used + 1, executor_used, provider_used, shared_keys)


def _workspace_id(decision_id: str, task_id: str, arm: ArmRole) -> str:
    identity = f"{decision_id}\x00{task_id}\x00{arm.value}".encode("utf-8")
    return "paired-run-" + sha256(identity).hexdigest()[:24]
