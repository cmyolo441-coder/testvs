"""Project observed paired-run lifecycle events into critical-path health."""

from __future__ import annotations

from ..coding.paired_run_health_events import paired_run_health_events_path
from ..system.paths import OmhPaths
from .critical_path_health import project_critical_path_health
from .critical_path_health_direct_events import read_health_events_path
from .critical_path_health_models import (
    CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
    CriticalPathEvidenceGap,
    CriticalPathHealthEvent,
    CriticalPathHealthProjection,
)
from .critical_path_health_source_models import CriticalPathHealthSourceResult


def project_paired_run_critical_path_health(
    paths: OmhPaths,
    decision_id: str,
) -> CriticalPathHealthSourceResult:
    """Project one paired-run journal, preserving every source gap."""
    gaps: set[tuple[str, str]] = set()
    try:
        path = paired_run_health_events_path(paths, decision_id)
    except ValueError:
        gaps.add(("", "paired_run_id_invalid"))
        return _result(decision_id, (), gaps)
    if not path.exists():
        gaps.add(("", "paired_run_health_events_unavailable"))
        return _result(decision_id, (), gaps)
    events = read_health_events_path(path, gaps)
    return _result(decision_id, events, gaps)


def _result(
    decision_id: str,
    events: tuple[CriticalPathHealthEvent, ...],
    source_gaps: set[tuple[str, str]],
) -> CriticalPathHealthSourceResult:
    projected = project_critical_path_health(events)
    gaps = tuple(
        sorted(
            source_gaps
            | {(gap.task_id, gap.code) for gap in projected.evidence_gaps}
        )
    )
    if source_gaps:
        projected = CriticalPathHealthProjection(
            CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
            projected.executor,
            projected.model,
            projected.environment,
            projected.task_revisions,
            None,
            (),
            (),
            tuple(
                CriticalPathEvidenceGap(task_id, code)
                for task_id, code in gaps
            ),
        )
    return CriticalPathHealthSourceResult(
        decision_id,
        events,
        projected,
        gaps,
    )
