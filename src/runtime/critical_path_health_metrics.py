"""DAG and interval metric computation for validated critical-path events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from .critical_path_health_models import Attempt, CriticalPathAttribution, CriticalPathMetrics


def compute_critical_path_metrics(attempts: Sequence[Attempt]) -> CriticalPathMetrics:
    active_ms = sum(attempt.active_ms for attempt in attempts)
    queue_ms = sum(attempt.queue_ms for attempt in attempts)
    wall_clock_ms = max(item.finished_at_ms for item in attempts) - min(item.queued_at_ms for item in attempts)
    by_task = _by_task(attempts)
    active_by_task = {task_id: sum(item.active_ms for item in items) for task_id, items in by_task.items()}
    dependencies = {task_id: items[0].dependencies for task_id, items in by_task.items()}
    cleanup = [item for item in attempts if item.phase == "cleanup"]
    non_cleanup = [item for item in attempts if item.phase != "cleanup"]
    cleanup_tail_ms = max(item.finished_at_ms for item in cleanup) - max(item.finished_at_ms for item in non_cleanup) if cleanup and non_cleanup else 0
    return CriticalPathMetrics(
        wall_clock_ms,
        active_ms,
        queue_ms,
        _critical_path(active_by_task, dependencies),
        _peak_concurrency(attempts),
        max(0, active_ms - wall_clock_ms),
        sum(item.active_ms for item in attempts if item.retry > 0),
        sum(1 for items in by_task.values() if any(item.terminal_status == "stale" for item in items)),
        cleanup_tail_ms,
        sum(1 for items in by_task.values() if any(item.reused for item in items)),
    )


def compute_attribution(
    attempts: Sequence[Attempt], *, by: Literal["phase", "resource"]
) -> tuple[CriticalPathAttribution, ...]:
    buckets: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        name = attempt.phase if by == "phase" else attempt.resource_class
        buckets.setdefault(name, []).append(attempt)
    return tuple(
        CriticalPathAttribution(
            name,
            sum(item.active_ms for item in items),
            sum(item.queue_ms for item in items),
            sum(item.active_ms for item in items if item.retry > 0),
            len({item.task_id for item in items}),
        )
        for name, items in sorted(buckets.items())
    )


def _by_task(attempts: Sequence[Attempt]) -> dict[str, list[Attempt]]:
    by_task: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_task.setdefault(attempt.task_id, []).append(attempt)
    return by_task


def _critical_path(active_by_task: dict[str, int], dependencies: dict[str, tuple[str, ...]]) -> int:
    memo: dict[str, int] = {}

    def duration(task_id: str) -> int:
        if task_id not in memo:
            memo[task_id] = active_by_task[task_id] + max(
                (duration(item) for item in dependencies[task_id]), default=0
            )
        return memo[task_id]

    return max((duration(task_id) for task_id in active_by_task), default=0)


def _peak_concurrency(attempts: Sequence[Attempt]) -> int:
    points: dict[int, list[int]] = {}
    for attempt in attempts:
        if attempt.active_ms > 0:
            points.setdefault(attempt.started_at_ms, [0, 0])[0] += 1
            points.setdefault(attempt.finished_at_ms, [0, 0])[1] += 1
    current = 0
    peak = 0
    for at_ms in sorted(points):
        starts, finishes = points[at_ms]
        current -= finishes
        current += starts
        peak = max(peak, current)
    return peak
