"""Frozen value types for the final-review-wave facade."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReviewLens(str, Enum):
    REQUIREMENT = "requirement"
    QUALITY = "quality"
    SAFETY = "safety"
    REAL_SURFACE = "real_surface"


class LaneState(str, Enum):
    PREPARED = "prepared"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    MISSING = "missing"
    STALE = "stale"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ContextProvenance(str, Enum):
    """Where a review lane's context came from; only a fresh lane is independent."""

    FRESH_FROM_DIFF = "fresh_from_diff"
    INHERITED_FROM_AUTHOR = "inherited_from_author"


CONTEXT_PROVENANCE_FIELD = "context_provenance"


class WaveVerdict(str, Enum):
    PASS = "PASS"
    HOLD = "HOLD"
    BLOCK = "BLOCK"


LANE_ORDER: tuple[ReviewLens, ...] = (
    ReviewLens.REQUIREMENT,
    ReviewLens.QUALITY,
    ReviewLens.SAFETY,
    ReviewLens.REAL_SURFACE,
)


@dataclass(frozen=True, slots=True)
class ImmutableRevision:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("revision must be a non-empty string")
        object.__setattr__(self, "value", self.value.strip())


@dataclass(frozen=True, slots=True)
class IntegrationReceipt:
    revision: ImmutableRevision
    completed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise ValueError("integration completion must be a boolean")


@dataclass(frozen=True, slots=True)
class ReadOnlyCapability:
    allows_mutation: bool = False

    def __post_init__(self) -> None:
        if self.allows_mutation:
            raise ValueError("final review lanes must be read-only")


@dataclass(frozen=True, slots=True)
class LaneBudgetReservationInput:
    lens: ReviewLens
    limit: int
    reserved: int

    def __post_init__(self) -> None:
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 0:
            raise ValueError("reservation limit must be a non-negative integer")
        if isinstance(self.reserved, bool) or not isinstance(self.reserved, int) or self.reserved < 0:
            raise ValueError("reservation count must be a non-negative integer")
        if self.reserved > self.limit:
            raise ValueError("reservation count cannot exceed its limit")

    @property
    def available(self) -> bool:
        return self.reserved < self.limit


@dataclass(frozen=True, slots=True)
class LaneObservation:
    """One lane's reported state, with the required provenance of its context.

    Declaring provenance is not evidence that the review ran; only the state
    carries that, and a prepared lane stays prepared whatever it declares.
    """

    lens: ReviewLens
    state: LaneState
    revision: ImmutableRevision
    context_provenance: ContextProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "context_provenance", _required_provenance(self.context_provenance)
        )


@dataclass(frozen=True, slots=True)
class ReviewLane:
    lens: ReviewLens
    state: LaneState
    read_only: ReadOnlyCapability
    bound_revision: ImmutableRevision | None
    observed_revision: ImmutableRevision | None = None
    # None means no observation declared a provenance; it never counts as
    # independent review.
    context_provenance: ContextProvenance | None = None


@dataclass(frozen=True, slots=True)
class WaveAssessment:
    verdict: WaveVerdict
    blocking_lens: ReviewLens | None
    refused_field: str | None = None


@dataclass(frozen=True, slots=True)
class LaneStatusProjection:
    lens: ReviewLens
    state: LaneState
    execution_status: str
    status: str = "required"


@dataclass(frozen=True, slots=True)
class WaveStatusProjection:
    assessment: WaveAssessment
    lanes: tuple[LaneStatusProjection, ...]


def _required_provenance(value: object) -> ContextProvenance:
    if isinstance(value, ContextProvenance):
        return value
    if isinstance(value, str):
        for member in ContextProvenance:
            if value == member.value:
                return member
    allowed = " or ".join(member.value for member in ContextProvenance)
    raise ValueError(f"{CONTEXT_PROVENANCE_FIELD} is required: {allowed}")
