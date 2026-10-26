"""Frozen typed values for paired-run dispatch planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from ..quality.paired_run_model import ArmRole


CLAIM_BOUNDARY: Final = (
    "This is a prepared paired-run dispatch plan, not evidence of launch, execution, "
    "provider use, cost, verification, review, CI, merge-readiness, or merge."
)


class PairedRunDispatchPlanError(ValueError):
    """The frozen paired-run decision cannot be planned safely."""


class ApprovalState(StrEnum):
    REQUIRED = "required"
    APPROVED = "approved"


class SharedResourceMode(StrEnum):
    SERIALIZE = "serialize"
    REFUSE = "refuse"


class TerminalState(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RATE_LIMITED = "rate_limited"
    CRASHED = "crashed"
    PARTIAL = "partial"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True, slots=True)
class CostTimeBound:
    """A total cost and elapsed-time bound, expressed in operator units."""

    cost_units: int
    seconds: int


@dataclass(frozen=True, slots=True)
class NamedConcurrencyBudget:
    name: str
    maximum: int


@dataclass(frozen=True, slots=True)
class DispatchBudgets:
    global_concurrency: int
    executor_concurrency: tuple[NamedConcurrencyBudget, ...]
    provider_concurrency: tuple[NamedConcurrencyBudget, ...]
    local_bound: CostTimeBound
    provider_bound: CostTimeBound


@dataclass(frozen=True, slots=True)
class ArmDispatchTarget:
    """The fixed execution target for one arm of the frozen decision."""

    arm: ArmRole
    executor: str
    provider: str
    model: str
    local_estimate: CostTimeBound
    provider_estimate: CostTimeBound
    shared_resource_key: str | None = None


@dataclass(frozen=True, slots=True)
class PairedRunDispatchConfig:
    approval: ApprovalState
    dry_run: bool
    shared_resource_mode: SharedResourceMode
    budgets: DispatchBudgets
    targets: tuple[ArmDispatchTarget, ...]


@dataclass(frozen=True, slots=True)
class PairedRunDispatchCell:
    task_id: str
    arm: ArmRole
    input_digest: str
    execution_revision: str
    executor: str
    provider: str
    model: str
    workspace_id: str
    launch_wave: int
    shared_resource_key: str | None
    terminal_state: TerminalState | None = None


@dataclass(frozen=True, slots=True)
class PairedRunDispatchPlan:
    decision_id: str
    approval: ApprovalState
    dry_run: bool
    cells: tuple[PairedRunDispatchCell, ...]
    claim_boundary: str = CLAIM_BOUNDARY

    @property
    def launch_authorized(self) -> bool:
        """Whether a separate runner may consume cells after explicit approval."""
        return self.approval is ApprovalState.APPROVED and not self.dry_run

    @property
    def launchable_cells(self) -> tuple[PairedRunDispatchCell, ...]:
        """Dry-runs and unapproved plans expose no cells to a future runner."""
        return self.cells if self.launch_authorized else ()

    @property
    def decision_ready(self) -> bool:
        """A comparative decision remains blocked until every exact cell is terminal."""
        return bool(self.cells) and all(cell.terminal_state is not None for cell in self.cells)
