"""Evidence validation for critical-path health projections."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .critical_path_health_models import Attempt, CriticalPathHealthEvent


@dataclass(frozen=True)
class ValidatedCriticalPathEvents:
    executor: str
    model: str
    environment: str
    task_revisions: tuple[tuple[str, str], ...]
    attempts: tuple[Attempt, ...]
    gaps: tuple[tuple[str, str], ...]


def validate_critical_path_events(events: Sequence[CriticalPathHealthEvent]) -> ValidatedCriticalPathEvents:
    first = events[0] if events else None
    executor = first.executor if first is not None else ""
    model = first.model if first is not None else ""
    environment = first.environment if first is not None else ""
    grouped: dict[tuple[str, int], list[CriticalPathHealthEvent]] = {}
    gaps: set[tuple[str, str]] = set()
    if not events:
        gaps.add(("", "no_events"))
    for event in events:
        if event.executor != executor or event.model != model or event.environment != environment:
            gaps.add((event.task_id, "execution_identity_mismatch"))
        grouped.setdefault((event.task_id, event.retry), []).append(event)
    task_revisions = _task_revisions(events, gaps)
    attempts = _attempts_from_events(grouped, gaps)
    _validate_task_consistency(attempts, gaps)
    _validate_dependencies(attempts, gaps)
    return ValidatedCriticalPathEvents(executor, model, environment, task_revisions, attempts, tuple(sorted(gaps)))


def _task_revisions(
    events: Sequence[CriticalPathHealthEvent], gaps: set[tuple[str, str]]
) -> tuple[tuple[str, str], ...]:
    revisions: dict[str, set[str]] = {}
    for event in events:
        revisions.setdefault(event.task_id, set()).add(event.revision)
    values: list[tuple[str, str]] = []
    for task_id, task_values in revisions.items():
        if len(task_values) != 1:
            gaps.add((task_id, "revision_mismatch"))
        else:
            values.append((task_id, next(iter(task_values))))
    return tuple(sorted(values))


def _attempts_from_events(
    grouped: dict[tuple[str, int], list[CriticalPathHealthEvent]], gaps: set[tuple[str, str]]
) -> tuple[Attempt, ...]:
    attempts: list[Attempt] = []
    for (task_id, retry), observed in grouped.items():
        kinds = tuple(event.event for event in observed)
        finish_count = kinds.count("finished")
        if finish_count == 0:
            gaps.add((task_id, "missing_terminal"))
            continue
        if finish_count > 1:
            gaps.add((task_id, "duplicate_terminal"))
            continue
        if kinds != ("queued", "started", "finished"):
            gaps.add((task_id, "invalid_event_order"))
            continue
        queued, started, finished = observed
        if queued.at_ms > started.at_ms or started.at_ms > finished.at_ms:
            gaps.add((task_id, "invalid_event_order"))
            continue
        attempts.append(
            Attempt(
                task_id, retry, queued.revision, queued.dependencies, queued.resource_class, queued.phase,
                queued.at_ms, started.at_ms, finished.at_ms, finished.terminal_status, finished.reused,
            )
        )
    return tuple(attempts)


def _validate_task_consistency(attempts: Sequence[Attempt], gaps: set[tuple[str, str]]) -> None:
    by_task: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_task.setdefault(attempt.task_id, []).append(attempt)
    for task_id, task_attempts in by_task.items():
        first = task_attempts[0]
        for attempt in task_attempts[1:]:
            if (
                attempt.dependencies != first.dependencies
                or attempt.resource_class != first.resource_class
                or attempt.phase != first.phase
            ):
                gaps.add((task_id, "task_metadata_mismatch"))
            if attempt.retry <= first.retry:
                gaps.add((task_id, "invalid_retry_order"))
            first = attempt


def _validate_dependencies(attempts: Sequence[Attempt], gaps: set[tuple[str, str]]) -> None:
    by_task: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_task.setdefault(attempt.task_id, []).append(attempt)
    for task_id, task_attempts in by_task.items():
        for dependency in task_attempts[0].dependencies:
            if dependency not in by_task:
                gaps.add((task_id, "missing_dependency"))
                continue
            dependency_finished_at_ms = max(item.finished_at_ms for item in by_task[dependency])
            task_started_at_ms = min(item.started_at_ms for item in task_attempts)
            if dependency_finished_at_ms > task_started_at_ms:
                gaps.add((task_id, "dependency_order"))
    graph = {task_id: task_attempts[0].dependencies for task_id, task_attempts in by_task.items()}
    if _has_cycle(graph):
        gaps.add(("", "cycle"))


def _has_cycle(graph: dict[str, tuple[str, ...]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> bool:
        if task_id in visiting:
            return True
        if task_id in visited:
            return False
        visiting.add(task_id)
        for dependency in graph[task_id]:
            if dependency in graph and visit(dependency):
                return True
        visiting.remove(task_id)
        visited.add(task_id)
        return False

    return any(visit(task_id) for task_id in graph)
