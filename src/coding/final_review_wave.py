"""Stable read-only facade for immutable post-integration final-review waves."""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import final_review_execution as _execution
from . import final_review_execution_models as _execution_models
from .final_review_status import human_lane_status
from .final_review_wave_models import (
    CONTEXT_PROVENANCE_FIELD,
    LANE_ORDER,
    ContextProvenance,
    ImmutableRevision,
    IntegrationReceipt,
    LaneBudgetReservationInput,
    LaneObservation,
    LaneState,
    LaneStatusProjection,
    ReadOnlyCapability,
    ReviewLane,
    ReviewLens,
    WaveAssessment,
    WaveStatusProjection,
    WaveVerdict,
)

_BLOCKING_STATES: frozenset[LaneState] = frozenset(
    {
        LaneState.BLOCKED,
        LaneState.MISSING,
        LaneState.STALE,
        LaneState.FAILED,
        LaneState.TIMED_OUT,
        LaneState.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class FinalReviewWave:
    """Read-only state for four required lanes pinned after integration."""

    wave_id: str
    integration: IntegrationReceipt | None
    lanes: tuple[ReviewLane, ...]
    reservations: tuple[LaneBudgetReservationInput, ...]
    invalidated_by_remediation: bool = False
    replaces_revision: ImmutableRevision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.wave_id, str) or not self.wave_id.strip():
            raise ValueError("wave_id must be a non-empty string")
        object.__setattr__(self, "wave_id", self.wave_id.strip())
        if tuple(lane.lens for lane in self.lanes) != LANE_ORDER:
            raise ValueError("lanes must use the fixed required lens order")
        if tuple(item.lens for item in self.reservations) != LANE_ORDER:
            raise ValueError("reservations must use the fixed required lens order")
        if any(lane.read_only.allows_mutation for lane in self.lanes):
            raise ValueError("final review lanes must remain read-only")
        if self.integration is None:
            if any(lane.bound_revision is not None for lane in self.lanes):
                raise ValueError("lanes cannot bind a revision before integration")
            if any(lane.state is not LaneState.PREPARED for lane in self.lanes):
                raise ValueError("lanes cannot execute before integration")
        elif not self.integration.completed:
            raise ValueError("a final review wave requires completed integration")
        elif any(lane.bound_revision != self.integration.revision for lane in self.lanes):
            raise ValueError("every lane must be pinned to the integrated revision")

    def integrate(self, receipt: IntegrationReceipt) -> FinalReviewWave:
        if self.invalidated_by_remediation:
            raise ValueError("an invalidated wave cannot be integrated")
        if self.integration is not None:
            raise ValueError("a final review wave is already integrated")
        if not receipt.completed:
            raise ValueError("only completed integration can enable review lanes")
        if self.replaces_revision == receipt.revision:
            raise ValueError("remediation requires a new integrated revision")
        return replace(
            self,
            integration=receipt,
            lanes=tuple(replace(lane, bound_revision=receipt.revision) for lane in self.lanes),
        )

    def eligible_lanes(self) -> tuple[ReviewLens, ...]:
        if self.integration is None or self.invalidated_by_remediation:
            return ()
        reservations = {item.lens: item for item in self.reservations}
        return tuple(
            lane.lens
            for lane in self.lanes
            if lane.state is LaneState.PREPARED and reservations[lane.lens].available
        )

    def observe(self, observation: LaneObservation) -> FinalReviewWave:
        if self.integration is None:
            raise ValueError("lanes cannot be observed before integration")
        if self.invalidated_by_remediation:
            raise ValueError("an invalidated wave cannot accept observations")
        index = LANE_ORDER.index(observation.lens)
        state = observation.state if observation.revision == self.integration.revision else LaneState.STALE
        replacement = replace(
            self.lanes[index],
            state=state,
            observed_revision=observation.revision,
            context_provenance=observation.context_provenance,
        )
        return replace(
            self,
            lanes=tuple(replacement if current == index else lane for current, lane in enumerate(self.lanes)),
        )

    def assess(self) -> WaveAssessment:
        if self.invalidated_by_remediation:
            return WaveAssessment(WaveVerdict.BLOCK, None)
        if self.integration is None:
            return WaveAssessment(WaveVerdict.HOLD, LANE_ORDER[0])
        lanes = {lane.lens: lane for lane in self.lanes}
        for lens in LANE_ORDER:
            if lanes[lens].state in _BLOCKING_STATES:
                return WaveAssessment(WaveVerdict.BLOCK, lens)
        for lens in LANE_ORDER:
            if not _may_count_as_independent(lanes[lens]):
                return WaveAssessment(WaveVerdict.BLOCK, lens, CONTEXT_PROVENANCE_FIELD)
        reservations = {item.lens: item for item in self.reservations}
        for lens in LANE_ORDER:
            if lanes[lens].state is LaneState.PREPARED and not reservations[lens].available:
                return WaveAssessment(WaveVerdict.BLOCK, lens)
        for lens in LANE_ORDER:
            if lanes[lens].state is not LaneState.COMPLETED:
                return WaveAssessment(WaveVerdict.HOLD, lens)
        return WaveAssessment(WaveVerdict.PASS, None)

    def project_status(self) -> WaveStatusProjection:
        return WaveStatusProjection(
            self.assess(),
            tuple(
                LaneStatusProjection(
                    lane.lens, lane.state, _execution_status(lane.state), human_lane_status(lane.state)
                )
                for lane in self.lanes
            ),
        )

    def invalidate_for_remediation(self) -> FinalReviewWave:
        return replace(self, invalidated_by_remediation=True)


def prepare_final_review_wave(
    wave_id: str, reservations: tuple[LaneBudgetReservationInput, ...]
) -> FinalReviewWave:
    return FinalReviewWave(
        wave_id,
        None,
        tuple(ReviewLane(lens, LaneState.PREPARED, ReadOnlyCapability(), None) for lens in LANE_ORDER),
        reservations,
    )


def prepare_remediated_wave(
    prior_wave: FinalReviewWave,
    wave_id: str,
    reservations: tuple[LaneBudgetReservationInput, ...],
) -> FinalReviewWave:
    if not prior_wave.invalidated_by_remediation:
        raise ValueError("prior wave must be invalidated before remediation")
    if prior_wave.integration is None:
        raise ValueError("prior wave must have an integrated revision")
    if wave_id == prior_wave.wave_id:
        raise ValueError("remediation requires a new wave_id")
    return replace(
        prepare_final_review_wave(wave_id, reservations),
        replaces_revision=prior_wave.integration.revision,
    )


def _may_count_as_independent(lane: ReviewLane) -> bool:
    """Refuse a lane whose context was inherited, or completed without saying."""
    if lane.context_provenance is ContextProvenance.INHERITED_FROM_AUTHOR:
        return False
    return lane.state is not LaneState.COMPLETED or lane.context_provenance is ContextProvenance.FRESH_FROM_DIFF


def _execution_status(state: LaneState) -> str:
    return "prepared_not_executed" if state is LaneState.PREPARED else state.value


execute_final_review_wave = _execution.execute_final_review_wave
FinalReviewExecutionReservations = _execution_models.FinalReviewExecutionReservations
GlobalReviewReservation = _execution_models.GlobalReviewReservation
LaneExecutionResult = _execution_models.LaneExecutionResult
ProviderReviewReservation = _execution_models.ProviderReviewReservation
ReviewReservation = _execution_models.ReviewReservation
