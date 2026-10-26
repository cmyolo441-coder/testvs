"""Explicit Hermes-child adapter for immutable fanout final reviews."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from subprocess import SubprocessError
from threading import Lock

from .final_review_wave import (
    LANE_ORDER,
    ContextProvenance,
    FinalReviewExecutionReservations,
    FinalReviewWave,
    GlobalReviewReservation,
    ImmutableRevision,
    IntegrationReceipt,
    LaneBudgetReservationInput,
    LaneExecutionResult,
    LaneObservation,
    LaneState,
    ProviderReviewReservation,
    ReviewLane,
    ReviewReservation,
    execute_final_review_wave,
    prepare_final_review_wave,
)
from .hermes_child_dispatch import (
    DispatchConfirmationError,
    DispatchRecursionError,
    HermesChildDispatchError,
    HermesChildRequest,
    dispatch_hermes_child,
)
from .final_review_worktree import (
    FinalReviewWorktreeError,
    final_review_worktree_matches,
    isolated_final_review_worktree,
)


_PASS = re.compile(r"\s*<verdict>PASS</verdict>\s*")
_FAIL = re.compile(r"\s*<verdict>FAIL</verdict>\s*")


class FinalReviewLocalEngineError(ValueError):
    """The explicit final-review adapter cannot be configured safely."""


@dataclass(frozen=True, slots=True)
class FinalReviewLocalEngineConfig:
    worktree: Path
    goal_text: str
    provider: str
    model: str
    reasoning: str
    timeout_seconds: float
    hermes: str = "hermes"

    def __post_init__(self) -> None:
        if not self.worktree.is_dir():
            raise FinalReviewLocalEngineError(
                "final review requires an integrated worktree"
            )
        if not self.goal_text.strip():
            raise FinalReviewLocalEngineError(
                "final review requires the fanout goal"
            )
        if not self.provider.strip() or not self.model.strip():
            raise FinalReviewLocalEngineError(
                "final review requires --hermes-provider and --hermes-model"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise FinalReviewLocalEngineError(
                "final review timeout must be positive"
            )


class HermesFinalReviewEngine:
    """Run all four review lenses through independent bounded children."""

    def __init__(self, config: FinalReviewLocalEngineConfig) -> None:
        self._config = config
        self._git_lock = Lock()

    def execute(
        self,
        revision: ImmutableRevision,
        observe: Callable[[LaneObservation], None],
    ) -> FinalReviewWave:
        reservations = tuple(
            LaneBudgetReservationInput(lens, 1, 0)
            for lens in LANE_ORDER
        )
        wave = prepare_final_review_wave(
            f"final-review-{revision.value[:16]}",
            reservations,
        ).integrate(IntegrationReceipt(revision, True))
        completed = execute_final_review_wave(
            wave,
            self._run_lane,
            FinalReviewExecutionReservations(
                GlobalReviewReservation(len(LANE_ORDER), 0),
                {
                    self._config.provider: ProviderReviewReservation(
                        len(LANE_ORDER),
                        0,
                    )
                },
                ReviewReservation(len(LANE_ORDER), 0),
            ),
            provider_for=lambda _lens: self._config.provider,
            # Each child starts from a detached checkout of the integrated
            # revision and a prompt of goal, lens, and revision only; no
            # author session or transcript reaches it.
            context_provenance=ContextProvenance.FRESH_FROM_DIFF,
        )
        for lane in completed.lanes:
            if lane.observed_revision is not None:
                observe(
                    LaneObservation(
                        lane.lens,
                        lane.state,
                        lane.observed_revision,
                        lane.context_provenance,
                    )
                )
        return completed

    def _run_lane(self, lane: ReviewLane) -> LaneExecutionResult:
        revision = lane.bound_revision
        if revision is None:
            return LaneExecutionResult(LaneState.MISSING, None)
        try:
            with isolated_final_review_worktree(
                self._config.worktree,
                revision.value,
                self._git_lock,
            ) as review_worktree:
                result = dispatch_hermes_child(
                    HermesChildRequest(
                        prompt=_review_prompt(
                            self._config.goal_text,
                            revision.value,
                            lane.lens.value,
                        ),
                        model=self._config.model,
                        provider=self._config.provider,
                        reasoning=self._config.reasoning,
                        parent_run_id=f"final-review-{revision.value[:16]}",
                        run_id=(
                            f"final-review-{revision.value[:12]}-"
                            f"{lane.lens.value}"
                        ),
                        timeout_seconds=self._config.timeout_seconds,
                        hermes=self._config.hermes,
                        cwd=review_worktree,
                        allow_parallel=True,
                    ),
                    dispatch_policy="ask_before_dispatch",
                    confirmed=True,
                )
                unchanged = final_review_worktree_matches(
                    review_worktree,
                    revision.value,
                )
        except (
            DispatchConfirmationError,
            DispatchRecursionError,
            HermesChildDispatchError,
            FinalReviewWorktreeError,
            OSError,
            SubprocessError,
            ValueError,
        ):
            return LaneExecutionResult(LaneState.FAILED, revision)
        if not unchanged:
            return LaneExecutionResult(LaneState.BLOCKED, revision)
        return LaneExecutionResult(
            _lane_state(result.status, result.stdout),
            revision,
        )


def _lane_state(status: str, output: str) -> LaneState:
    if status == "timed_out":
        return LaneState.TIMED_OUT
    if status == "cancelled":
        return LaneState.CANCELLED
    if status != "completed":
        return LaneState.FAILED
    if _PASS.fullmatch(output):
        return LaneState.COMPLETED
    if _FAIL.fullmatch(output):
        return LaneState.BLOCKED
    return LaneState.MISSING


def _review_prompt(goal_text: str, revision: str, lens: str) -> str:
    return (
        "Read-only final review. Inspect the current checkout without editing. "
        f"Review lens: {lens}. Integrated tree: {revision}. "
        f"Goal:\n{goal_text}\n"
        "Return exactly <verdict>PASS</verdict> when this lens has no blocking "
        "finding, otherwise return exactly <verdict>FAIL</verdict>."
    )
