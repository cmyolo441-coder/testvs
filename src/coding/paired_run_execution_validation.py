"""Pure receipt and limit checks for paired-run execution."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from .hermes_child_receipts import is_verified_receipt
from .paired_run_dispatch_model import PairedRunDispatchCell, TerminalState
from .paired_run_execution_model import (
    ExecutionState,
    PairedRunExecutionLimits,
    PairedRunExecutionOutcome,
)


def terminal_state(state: ExecutionState) -> TerminalState:
    """Project execution metadata into the pre-existing frozen plan vocabulary."""
    states = {
        ExecutionState.SUCCEEDED: TerminalState.SUCCEEDED,
        ExecutionState.FAILED: TerminalState.FAILED,
        ExecutionState.TIMED_OUT: TerminalState.TIMEOUT,
        ExecutionState.CANCELLED: TerminalState.CANCELLED,
        ExecutionState.RATE_LIMITED: TerminalState.RATE_LIMITED,
        ExecutionState.CRASHED: TerminalState.CRASHED,
        ExecutionState.PARTIAL: TerminalState.PARTIAL,
        ExecutionState.CLEANUP_FAILED: TerminalState.CLEANUP_FAILED,
        # The committed plan predates execution authentication. PARTIAL keeps
        # its all-terminal barrier closed to fan-in while the receipt retains
        # the distinct, non-resumable unauthenticated classification.
        ExecutionState.UNAUTHENTICATED: TerminalState.PARTIAL,
    }
    return states[state]


def normalized_cell(cell: PairedRunDispatchCell) -> PairedRunDispatchCell:
    return replace(cell, terminal_state=None)


def receipt_authenticates(
    outcome: PairedRunExecutionOutcome,
    cell: PairedRunDispatchCell,
) -> bool:
    """Verify the HMAC-sealed receipt binds every identity the receipt can seal."""
    receipt = outcome.receipt
    if receipt is None or not is_verified_receipt(receipt) or receipt.evaluation_binding is None:
        return False
    binding = receipt.evaluation_binding
    if outcome.cell is not None and normalized_cell(outcome.cell) != normalized_cell(cell):
        return False
    if (
        binding.task_id != cell.task_id
        or binding.input_digest != cell.input_digest
        or binding.arm != cell.arm.value
        or binding.executor != cell.executor
        or binding.model != cell.model
        or binding.execution_revision != cell.execution_revision
    ):
        return False
    expected_status = {
        ExecutionState.SUCCEEDED: "completed",
        ExecutionState.FAILED: "failed",
        ExecutionState.TIMED_OUT: "timed_out",
        ExecutionState.CANCELLED: "cancelled",
        ExecutionState.RATE_LIMITED: "failed",
        ExecutionState.CRASHED: "failed",
        ExecutionState.PARTIAL: "completed",
    }.get(outcome.state)
    return expected_status is not None and receipt.status == expected_status


def resume_index(
    cells: tuple[PairedRunDispatchCell, ...],
    prior: tuple[PairedRunExecutionOutcome, ...],
) -> dict[str, PairedRunExecutionOutcome]:
    """Return only unique authenticated terminal receipts bound to exact cells."""
    by_workspace = {cell.workspace_id: cell for cell in cells}
    references = Counter(
        item.receipt.receipt_ref
        for item in prior
        if item.receipt is not None
    )
    valid: dict[str, PairedRunExecutionOutcome] = {}
    for item in prior:
        if item.cell is None or item.cell.workspace_id not in by_workspace:
            continue
        cell = by_workspace[item.cell.workspace_id]
        if (
            item.state in {ExecutionState.CLEANUP_FAILED, ExecutionState.UNAUTHENTICATED}
            or item.cleanup_succeeded is not True
            or item.receipt is None
            or references[item.receipt.receipt_ref] != 1
            or not receipt_authenticates(item, cell)
        ):
            continue
        valid[cell.workspace_id] = replace(
            item, cell=cell, authenticated=True, reused=True
        )
    return valid


def inferred_limits(cells: tuple[PairedRunDispatchCell, ...]) -> PairedRunExecutionLimits:
    """Use the committed launch waves as conservative limits when none are supplied."""
    waves: dict[int, tuple[PairedRunDispatchCell, ...]] = {}
    for cell in cells:
        waves[cell.launch_wave] = (*waves.get(cell.launch_wave, ()), cell)
    global_limit = max((len(wave) for wave in waves.values()), default=1)
    executor: dict[str, int] = {}
    provider: dict[str, int] = {}
    for wave in waves.values():
        for name, count in Counter(cell.executor for cell in wave).items():
            executor[name] = max(executor.get(name, 0), count)
        for name, count in Counter(cell.provider for cell in wave).items():
            provider[name] = max(provider.get(name, 0), count)
    return PairedRunExecutionLimits(global_limit, executor, provider)


def validate_limits(limits: PairedRunExecutionLimits, cells: tuple[PairedRunDispatchCell, ...]) -> None:
    if isinstance(limits.global_concurrency, bool) or limits.global_concurrency < 1:
        raise ValueError("global_concurrency must be positive")
    for label, configured, names in (
        ("executor", limits.executor_concurrency, {cell.executor for cell in cells}),
        ("provider", limits.provider_concurrency, {cell.provider for cell in cells}),
    ):
        if set(configured) != names or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in configured.values()
        ):
            raise ValueError(f"{label} limits must name every planned {label} once")
