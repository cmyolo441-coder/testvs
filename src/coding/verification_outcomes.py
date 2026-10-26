"""Run context and outcome types for the verification plan engine.

Separated from `verification_runner` so the engine module holds only the
admission logic: these are the typed values the engine, the dispatcher hook,
and tests all exchange. A `NodeOutcome` records what one check resolved to —
ran fresh, reused an immutable receipt, or was recorded skipped (blocked by
a failed dependency, or deferred behind a closed producer fan-in) — and the
aggregate is explicit: `all_passed` is true only when every node holds fresh
or reused in-scope passing evidence; anything else is a HOLD, never a PASS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from .verification_execution import VerificationExecutionGate
from .verification_plan import VerificationNode
from .verification_receipts import SingleFlight

if TYPE_CHECKING:
    from ..system.paths import OmhPaths

# run_node executes one node and returns (status, detail, truncation record),
# the exact contract of the dispatcher's `_run_verification_command`.
RunNode = Callable[[VerificationNode], tuple[str, str, dict[str, Any] | None]]


@dataclass(frozen=True, slots=True)
class PlanRunContext:
    """Everything one plan run needs that is not the plan itself."""

    paths: OmhPaths
    worktree: Path
    revision: str | None
    max_workers: int
    integration_ready: Callable[[], bool]
    single_flight: SingleFlight
    execution_environment: Mapping[str, str] = field(default_factory=dict)
    execution_gate: VerificationExecutionGate | None = None


@dataclass(frozen=True, slots=True)
class NodeOutcome:
    """What one node resolved to, run or reused."""

    node: VerificationNode
    status: str
    detail: str
    reused: bool
    receipt_key: str | None
    truncation: dict[str, Any] | None
    deferred: bool


@dataclass(frozen=True, slots=True)
class PlanRunResult:
    """The aggregate over one plan run, in declared node order."""

    outcomes: tuple[NodeOutcome, ...]

    @property
    def all_passed(self) -> bool:
        return bool(self.outcomes) and all(outcome.status == "passed" for outcome in self.outcomes)

    @property
    def deferred(self) -> bool:
        return any(outcome.deferred for outcome in self.outcomes)

    @property
    def failures(self) -> list[str]:
        return [
            f"{outcome.node.command}: {outcome.detail}"
            for outcome in self.outcomes
            if outcome.status == "failed"
        ]

    @property
    def truncations(self) -> list[dict[str, Any]]:
        return [outcome.truncation for outcome in self.outcomes if outcome.truncation is not None]
