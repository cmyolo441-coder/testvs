"""Typed, metadata-only critical-path health event and projection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..system.metadata_safety import require_opaque_metadata_ref


CRITICAL_PATH_HEALTH_EVENT_SCHEMA_VERSION = "critical_path_health_event/v1"
CRITICAL_PATH_HEALTH_SCHEMA_VERSION = "critical_path_health/v1"
CRITICAL_PATH_HEALTH_PRIVACY = "metadata_only"

CriticalPathEventKind = Literal["queued", "started", "finished"]
CriticalPathTerminalStatus = Literal["succeeded", "failed", "cancelled", "skipped", "stale"]

_EVENT_KINDS = frozenset(("queued", "started", "finished"))
_TERMINAL_STATUSES = frozenset(("succeeded", "failed", "cancelled", "skipped", "stale"))
_FORBIDDEN_METADATA_WORDS = ("command", "output", "source", "prompt", "credential", "private", "payload")


@dataclass(frozen=True)
class CriticalPathHealthEvent:
    """One versioned lifecycle observation for one task attempt."""

    task_id: str
    event: CriticalPathEventKind
    at_ms: int
    revision: str
    executor: str
    model: str
    environment: str
    dependencies: tuple[str, ...] = ()
    resource_class: str = "default"
    phase: str = "execution"
    retry: int = 0
    terminal_status: CriticalPathTerminalStatus | Literal[""] = ""
    reused: bool = False
    schema_version: str = CRITICAL_PATH_HEALTH_EVENT_SCHEMA_VERSION
    privacy: str = CRITICAL_PATH_HEALTH_PRIVACY

    def __post_init__(self) -> None:
        if self.schema_version != CRITICAL_PATH_HEALTH_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported critical path health event schema")
        if self.privacy != CRITICAL_PATH_HEALTH_PRIVACY:
            raise ValueError("critical path event privacy must be metadata_only")
        if self.event not in _EVENT_KINDS:
            raise ValueError("critical path event kind is unsupported")
        if not isinstance(self.at_ms, int) or isinstance(self.at_ms, bool) or self.at_ms < 0:
            raise ValueError("critical path event at_ms must be a nonnegative integer")
        if not isinstance(self.retry, int) or isinstance(self.retry, bool) or self.retry < 0:
            raise ValueError("critical path event retry must be a nonnegative integer")
        if not isinstance(self.reused, bool):
            raise ValueError("critical path event reused must be a boolean")
        if not isinstance(self.dependencies, tuple):
            raise ValueError("critical path event dependencies must be a tuple")
        if self.event == "finished":
            if self.terminal_status not in _TERMINAL_STATUSES:
                raise ValueError("finished critical path events require a terminal status")
        elif self.terminal_status:
            raise ValueError("only finished critical path events may carry a terminal status")
        for value in (
            self.task_id,
            self.revision,
            self.executor,
            self.model,
            self.environment,
            self.resource_class,
            self.phase,
            *self.dependencies,
        ):
            _require_private_metadata(value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "privacy": self.privacy,
            "task_id": self.task_id,
            "event": self.event,
            "at_ms": self.at_ms,
            "revision": self.revision,
            "executor": self.executor,
            "model": self.model,
            "environment": self.environment,
            "dependencies": list(self.dependencies),
            "resource_class": self.resource_class,
            "phase": self.phase,
            "retry": self.retry,
            "terminal_status": self.terminal_status,
            "reused": self.reused,
        }


@dataclass(frozen=True)
class CriticalPathEvidenceGap:
    task_id: str
    code: str

    def to_dict(self) -> dict[str, str]:
        return {"task_id": self.task_id, "code": self.code}


@dataclass(frozen=True)
class CriticalPathMetrics:
    wall_clock_ms: int
    active_ms: int
    queue_ms: int
    critical_path_ms: int
    peak_concurrency: int
    overlap_savings_ms: int
    repeated_cost_ms: int
    stale_count: int
    cleanup_tail_ms: int
    reused_task_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "wall_clock_ms": self.wall_clock_ms,
            "active_ms": self.active_ms,
            "queue_ms": self.queue_ms,
            "critical_path_ms": self.critical_path_ms,
            "peak_concurrency": self.peak_concurrency,
            "overlap_savings_ms": self.overlap_savings_ms,
            "repeated_cost_ms": self.repeated_cost_ms,
            "stale_count": self.stale_count,
            "cleanup_tail_ms": self.cleanup_tail_ms,
            "reused_task_count": self.reused_task_count,
        }


@dataclass(frozen=True)
class CriticalPathAttribution:
    name: str
    active_ms: int
    queue_ms: int
    repeated_cost_ms: int
    task_count: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "active_ms": self.active_ms,
            "queue_ms": self.queue_ms,
            "repeated_cost_ms": self.repeated_cost_ms,
            "task_count": self.task_count,
        }


@dataclass(frozen=True)
class CriticalPathHealthProjection:
    schema_version: str
    executor: str
    model: str
    environment: str
    task_revisions: tuple[tuple[str, str], ...]
    metrics: CriticalPathMetrics | None
    phase_attribution: tuple[CriticalPathAttribution, ...]
    resource_attribution: tuple[CriticalPathAttribution, ...]
    evidence_gaps: tuple[CriticalPathEvidenceGap, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "privacy": CRITICAL_PATH_HEALTH_PRIVACY,
            "executor": self.executor,
            "model": self.model,
            "environment": self.environment,
            "task_revisions": [
                {"task_id": task_id, "revision": revision} for task_id, revision in self.task_revisions
            ],
            "metrics": self.metrics.to_dict() if self.metrics is not None else None,
            "phase_attribution": [item.to_dict() for item in self.phase_attribution],
            "resource_attribution": [item.to_dict() for item in self.resource_attribution],
            "evidence_gaps": [gap.to_dict() for gap in self.evidence_gaps],
        }


@dataclass(frozen=True)
class Attempt:
    task_id: str
    retry: int
    revision: str
    dependencies: tuple[str, ...]
    resource_class: str
    phase: str
    queued_at_ms: int
    started_at_ms: int
    finished_at_ms: int
    terminal_status: str
    reused: bool

    @property
    def active_ms(self) -> int:
        return self.finished_at_ms - self.started_at_ms

    @property
    def queue_ms(self) -> int:
        return self.started_at_ms - self.queued_at_ms


def _require_private_metadata(value: str) -> None:
    safe = require_opaque_metadata_ref(value, field="critical path metadata")
    if any(word in safe.lower() for word in _FORBIDDEN_METADATA_WORDS):
        raise ValueError("critical path metadata must not contain a private payload")
