"""Bounded metadata-only fanout health event emission."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import re
from threading import Lock
import time

from ..runtime.critical_path_health_models import CriticalPathHealthEvent, CriticalPathTerminalStatus
from ..system.local_store import append_jsonl_locked
from ..system.paths import OmhPaths
from .fanout_contracts import FANOUT_ID_PATTERN


_FANOUT_ID = re.compile(FANOUT_ID_PATTERN)


def monotonic_milliseconds() -> int:
    """Return the dispatcher clock in the event schema's millisecond unit."""
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class _Task:
    dependencies: tuple[str, ...]
    resource_class: str
    revision: str
    queued_at_ms: int
    started_at_ms: int | None = None


class FanoutHealthEvents:
    """Best-effort lifecycle recorder with no fabricated timestamp intervals."""

    def __init__(
        self,
        *,
        fanout_id: str,
        revision: str,
        emit: Callable[[CriticalPathHealthEvent], None],
        clock: Callable[[], int],
        executor: str = "fanout_dispatch",
        model: str = "frozen_contract",
        environment: str = "omh",
    ) -> None:
        self._fanout_id = fanout_id
        self._revision = revision
        self._emit = emit
        self._clock = clock
        self._executor = executor
        self._model = model
        self._environment = environment
        self._tasks: dict[tuple[str, int], _Task] = {}
        self._finished: set[tuple[str, int]] = set()
        self._lock = Lock()

    def queued(
        self,
        task_id: str,
        *,
        dependencies: tuple[str, ...],
        resource_class: str,
        retry: int = 0,
        phase: str = "execution",
        revision: str | None = None,
    ) -> None:
        with self._lock:
            key = (task_id, retry)
            if key in self._tasks:
                return
            at_ms = self._clock()
            task = _Task(
                dependencies,
                resource_class,
                self._revision if revision is None else revision,
                at_ms,
            )
            self._tasks[key] = task
        self._record(task_id, "queued", at_ms, task, retry=retry, phase=phase)

    def started(self, task_id: str, *, retry: int = 0, phase: str = "execution") -> None:
        with self._lock:
            key = (task_id, retry)
            task = self._tasks.get(key)
            if task is None or task.started_at_ms is not None:
                return
            at_ms = self._clock()
            if at_ms < task.queued_at_ms:
                return
            started = _Task(task.dependencies, task.resource_class, task.revision, task.queued_at_ms, at_ms)
            self._tasks[key] = started
        self._record(task_id, "started", at_ms, started, retry=retry, phase=phase)

    def finished(
        self,
        task_id: str,
        *,
        terminal_status: CriticalPathTerminalStatus,
        retry: int = 0,
        reused: bool = False,
        phase: str = "execution",
    ) -> None:
        with self._lock:
            key = (task_id, retry)
            task = self._tasks.get(key)
            if task is None or task.started_at_ms is None or key in self._finished:
                return
            at_ms = self._clock()
            if at_ms < task.started_at_ms:
                return
            self._finished.add(key)
        self._record(
            task_id,
            "finished",
            at_ms,
            task,
            retry=retry,
            terminal_status=terminal_status,
            reused=reused,
            phase=phase,
        )

    def _record(
        self,
        task_id: str,
        event: str,
        at_ms: int,
        task: _Task,
        *,
        retry: int,
        phase: str,
        terminal_status: CriticalPathTerminalStatus | str = "",
        reused: bool = False,
    ) -> None:
        try:
            self._emit(
                CriticalPathHealthEvent(
                    task_id=task_id,
                    event=event,
                    at_ms=at_ms,
                    revision=task.revision,
                    executor=self._executor,
                    model=self._model,
                    environment=self._environment,
                    dependencies=task.dependencies,
                    resource_class=task.resource_class,
                    phase=phase,
                    retry=retry,
                    terminal_status=terminal_status,
                    reused=reused,
                )
            )
        except (OSError, TimeoutError, ValueError):
            return


def fanout_health_events_path(paths: OmhPaths, fanout_id: str) -> Path:
    """Contained append-only event journal for one frozen fanout."""
    if _FANOUT_ID.fullmatch(fanout_id) is None:
        raise ValueError("fanout health events require a valid fanout id")
    return paths.fanout_contracts_dir / fanout_id / "critical_path_health_events.jsonl"


def write_fanout_health_event(paths: OmhPaths, fanout_id: str, event: CriticalPathHealthEvent) -> None:
    """Append one pre-validated metadata event for the projector to consume."""
    append_jsonl_locked(fanout_health_events_path(paths, fanout_id), event.to_dict(), private=True)
