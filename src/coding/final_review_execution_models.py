"""Typed execution inputs for concurrent final-review lanes."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .final_review_wave_models import ContextProvenance, ImmutableRevision, LaneState


@dataclass(frozen=True, slots=True)
class _ReviewReservation:
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
    def available_slots(self) -> int:
        return self.limit - self.reserved


@dataclass(frozen=True, slots=True)
class GlobalReviewReservation(_ReviewReservation):
    """Dispatch-wide capacity already reserved for final review."""


@dataclass(frozen=True, slots=True)
class ProviderReviewReservation(_ReviewReservation):
    """Capacity reserved from one named review provider."""


@dataclass(frozen=True, slots=True)
class ReviewReservation(_ReviewReservation):
    """Capacity reserved by the final-review subsystem itself."""


@dataclass(frozen=True, slots=True)
class FinalReviewExecutionReservations:
    """Independent limits all a lane must hold while its runner executes."""

    global_reservation: GlobalReviewReservation
    provider_reservations: Mapping[str, ProviderReviewReservation]
    review_reservation: ReviewReservation

    def __post_init__(self) -> None:
        providers: dict[str, ProviderReviewReservation] = {}
        for provider, reservation in self.provider_reservations.items():
            name = str(provider).strip()
            if not name:
                raise ValueError("provider name must be non-empty")
            if not isinstance(reservation, ProviderReviewReservation):
                raise ValueError("provider reservations must be typed")
            providers[name] = reservation
        object.__setattr__(self, "provider_reservations", MappingProxyType(providers))


@dataclass(frozen=True, slots=True)
class LaneExecutionResult:
    """One runner's terminal observation; no result means missing evidence.

    ``context_provenance`` left as None takes the provenance the wave's
    executor declared for every lane.
    """

    state: LaneState
    revision: ImmutableRevision | None
    context_provenance: ContextProvenance | None = None
