"""Human-readable final-review lane status projection."""

from __future__ import annotations

from .final_review_wave_models import LaneState


def human_lane_status(state: LaneState) -> str:
    """Reduce evidence states to concise operator-facing lane status."""
    if state is LaneState.PREPARED:
        return "required"
    if state is LaneState.RUNNING:
        return "running"
    if state is LaneState.COMPLETED:
        return "completed"
    if state is LaneState.MISSING:
        return "missing"
    return "blocked"
