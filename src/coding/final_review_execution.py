"""Bounded concurrent execution and deterministic fan-in for final-review waves."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import replace
import threading
from typing import TYPE_CHECKING

from .final_review_execution_models import (
    FinalReviewExecutionReservations,
    LaneExecutionResult,
)
from .final_review_wave_models import (
    LANE_ORDER,
    ContextProvenance,
    ImmutableRevision,
    LaneObservation,
    LaneState,
    ReviewLane,
    ReviewLens,
    WaveStatusProjection,
)
if TYPE_CHECKING:
    from .final_review_wave import FinalReviewWave

LaneRunner = Callable[[ReviewLane], LaneExecutionResult | LaneObservation | None]
ProviderForLane = Callable[[ReviewLens], str]
StatusSink = Callable[[WaveStatusProjection], None]
_TERMINAL_STATES = frozenset(set(LaneState) - {LaneState.PREPARED, LaneState.RUNNING})


def execute_final_review_wave(
    wave: FinalReviewWave,
    lane_runner: LaneRunner,
    reservations: FinalReviewExecutionReservations,
    *,
    provider_for: ProviderForLane,
    context_provenance: ContextProvenance,
    status_sink: StatusSink | None = None,
) -> FinalReviewWave:
    """Run every eligible lane, then reduce terminal evidence in fixed lane order.

    ``context_provenance`` is how this executor builds each reviewer's context.
    It is recorded on every lane it observes; a runner result may declare its
    own. It is metadata about the lane, never evidence that the lane ran.

    A worker holds global, provider, and review reservations simultaneously.
    Every submitted lane is awaited before any terminal state is reduced, so a
    fast failure cannot hide later terminal evidence or change blocker order.
    """
    _emit(wave, status_sink)
    if wave.integration is None or wave.invalidated_by_remediation:
        return wave

    revision = wave.integration.revision
    current = wave
    current_lock = threading.Lock()

    def mark_running(lens: ReviewLens) -> None:
        nonlocal current
        with current_lock:
            current = current.observe(LaneObservation(lens, LaneState.RUNNING, revision, context_provenance))
            projection = current.project_status()
        _emit_projection(projection, status_sink)

    eligible = set(wave.eligible_lanes())
    for lens in LANE_ORDER:
        if lens not in eligible:
            current = current.observe(LaneObservation(lens, LaneState.BLOCKED, revision, context_provenance))

    providers: dict[ReviewLens, str] = {}
    runnable: list[ReviewLens] = []
    for lens in LANE_ORDER:
        if lens not in eligible:
            continue
        provider = provider_for(lens).strip()
        if not provider or provider not in reservations.provider_reservations:
            current = current.observe(LaneObservation(lens, LaneState.MISSING, revision, context_provenance))
        else:
            providers[lens] = provider
            runnable.append(lens)

    if (
        reservations.global_reservation.available_slots == 0
        or reservations.review_reservation.available_slots == 0
    ):
        for lens in runnable:
            current = current.observe(LaneObservation(lens, LaneState.BLOCKED, revision, context_provenance))
        _emit(current, status_sink)
        return current

    runnable = [
        lens
        for lens in runnable
        if reservations.provider_reservations[providers[lens]].available_slots > 0
    ]
    for lens in LANE_ORDER:
        if lens in providers and lens not in runnable:
            current = current.observe(LaneObservation(lens, LaneState.BLOCKED, revision, context_provenance))
    if not runnable:
        _emit(current, status_sink)
        return current

    global_gate = threading.BoundedSemaphore(reservations.global_reservation.available_slots)
    review_gate = threading.BoundedSemaphore(reservations.review_reservation.available_slots)
    provider_gates = {
        name: threading.BoundedSemaphore(reservation.available_slots)
        for name, reservation in reservations.provider_reservations.items()
        if reservation.available_slots > 0
    }
    lanes = {lane.lens: lane for lane in current.lanes}
    with ThreadPoolExecutor(max_workers=len(runnable)) as pool:
        futures: dict[ReviewLens, Future[LaneExecutionResult]] = {
            lens: pool.submit(
                _run_lane,
                lanes[lens],
                lane_runner,
                revision,
                global_gate,
                provider_gates[providers[lens]],
                review_gate,
                mark_running,
            )
            for lens in runnable
        }
        results = {lens: _future_result(future, revision) for lens, future in futures.items()}

    for lens in LANE_ORDER:
        if lens in results:
            observation = results[lens]
            current = current.observe(
                LaneObservation(
                    lens,
                    observation.state,
                    observation.revision,
                    observation.context_provenance or context_provenance,
                )
            )
    _emit(current, status_sink)
    return current


def _run_lane(
    lane: ReviewLane,
    runner: LaneRunner,
    revision: ImmutableRevision,
    global_gate: threading.BoundedSemaphore,
    provider_gate: threading.BoundedSemaphore,
    review_gate: threading.BoundedSemaphore,
    mark_running: Callable[[ReviewLens], None],
) -> LaneExecutionResult:
    with global_gate, provider_gate, review_gate:
        mark_running(lane.lens)
        result = runner(replace(lane, state=LaneState.RUNNING))
    return _terminal_result(lane.lens, result, revision)


def _future_result(future: Future[LaneExecutionResult], revision: ImmutableRevision) -> LaneExecutionResult:
    if future.cancelled():
        return LaneExecutionResult(LaneState.CANCELLED, revision)
    failure = future.exception()
    if isinstance(failure, TimeoutError):
        return LaneExecutionResult(LaneState.TIMED_OUT, revision)
    if isinstance(failure, CancelledError):
        return LaneExecutionResult(LaneState.CANCELLED, revision)
    if failure is not None:
        return LaneExecutionResult(LaneState.FAILED, revision)
    return future.result()


def _terminal_result(
    lens: ReviewLens,
    result: LaneExecutionResult | LaneObservation | None,
    revision: ImmutableRevision,
) -> LaneExecutionResult:
    if result is None:
        return LaneExecutionResult(LaneState.MISSING, revision)
    if isinstance(result, LaneObservation):
        if result.lens is not lens:
            return LaneExecutionResult(LaneState.MISSING, revision)
        result = LaneExecutionResult(result.state, result.revision, result.context_provenance)
    if result.state not in _TERMINAL_STATES or result.revision is None:
        return LaneExecutionResult(LaneState.MISSING, revision)
    return result


def _emit(wave: FinalReviewWave, sink: StatusSink | None) -> None:
    if sink is not None:
        sink(wave.project_status())


def _emit_projection(projection: WaveStatusProjection, sink: StatusSink | None) -> None:
    if sink is not None:
        sink(projection)
