"""Public facade for the pure metadata-only critical-path health projection.

This module remains the stable import surface.  Typed models, evidence
validation, and metric computation are intentionally isolated so each
responsibility stays auditable and bounded.
"""

from __future__ import annotations

from collections.abc import Sequence

from .critical_path_health_metrics import compute_attribution, compute_critical_path_metrics
from .critical_path_health_models import (
    CRITICAL_PATH_HEALTH_EVENT_SCHEMA_VERSION,
    CRITICAL_PATH_HEALTH_PRIVACY,
    CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
    CriticalPathAttribution,
    CriticalPathEventKind,
    CriticalPathEvidenceGap,
    CriticalPathHealthEvent,
    CriticalPathHealthProjection,
    CriticalPathMetrics,
    CriticalPathTerminalStatus,
)
from .critical_path_health_validation import validate_critical_path_events


__all__ = (
    "CRITICAL_PATH_HEALTH_EVENT_SCHEMA_VERSION",
    "CRITICAL_PATH_HEALTH_PRIVACY",
    "CRITICAL_PATH_HEALTH_SCHEMA_VERSION",
    "CriticalPathAttribution",
    "CriticalPathEventKind",
    "CriticalPathEvidenceGap",
    "CriticalPathHealthEvent",
    "CriticalPathHealthProjection",
    "CriticalPathMetrics",
    "CriticalPathTerminalStatus",
    "compare_critical_path_health",
    "project_critical_path_health",
)


def project_critical_path_health(
    events: Sequence[CriticalPathHealthEvent],
) -> CriticalPathHealthProjection:
    """Project exact metrics only when all supplied lifecycle evidence is coherent."""
    validated = validate_critical_path_events(events)
    gaps = tuple(CriticalPathEvidenceGap(task_id, code) for task_id, code in validated.gaps)
    if gaps:
        return CriticalPathHealthProjection(
            CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
            validated.executor,
            validated.model,
            validated.environment,
            validated.task_revisions,
            None,
            (),
            (),
            gaps,
        )
    return CriticalPathHealthProjection(
        CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
        validated.executor,
        validated.model,
        validated.environment,
        validated.task_revisions,
        compute_critical_path_metrics(validated.attempts),
        compute_attribution(validated.attempts, by="phase"),
        compute_attribution(validated.attempts, by="resource"),
        (),
    )


def compare_critical_path_health(
    baseline: CriticalPathHealthProjection,
    candidate: CriticalPathHealthProjection,
) -> dict[str, int]:
    """Compare complete projections after rejecting incompatible evidence scopes."""
    if (
        baseline.schema_version != CRITICAL_PATH_HEALTH_SCHEMA_VERSION
        or candidate.schema_version != CRITICAL_PATH_HEALTH_SCHEMA_VERSION
        or baseline.metrics is None
        or candidate.metrics is None
        or baseline.evidence_gaps
        or candidate.evidence_gaps
        or baseline.task_revisions != candidate.task_revisions
        or baseline.executor != candidate.executor
        or baseline.model != candidate.model
        or baseline.environment != candidate.environment
    ):
        raise ValueError("incompatible critical path health projections")
    return {
        "wall_clock_delta_ms": candidate.metrics.wall_clock_ms - baseline.metrics.wall_clock_ms,
        "active_delta_ms": candidate.metrics.active_ms - baseline.metrics.active_ms,
        "queue_delta_ms": candidate.metrics.queue_ms - baseline.metrics.queue_ms,
        "critical_path_delta_ms": candidate.metrics.critical_path_ms - baseline.metrics.critical_path_ms,
        "peak_concurrency_delta": candidate.metrics.peak_concurrency - baseline.metrics.peak_concurrency,
        "overlap_savings_delta_ms": candidate.metrics.overlap_savings_ms - baseline.metrics.overlap_savings_ms,
        "repeated_cost_delta_ms": candidate.metrics.repeated_cost_ms - baseline.metrics.repeated_cost_ms,
        "stale_count_delta": candidate.metrics.stale_count - baseline.metrics.stale_count,
        "cleanup_tail_delta_ms": candidate.metrics.cleanup_tail_ms - baseline.metrics.cleanup_tail_ms,
        "reused_task_count_delta": candidate.metrics.reused_task_count - baseline.metrics.reused_task_count,
    }
