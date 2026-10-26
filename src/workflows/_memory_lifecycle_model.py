from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from ..system.paths import OmhPaths

MAX_MANIFEST_TARGETS = 128


@dataclass(frozen=True)
class LifecycleMutation:
    name: str
    action: str
    target: str
    target_id: str
    artifact_kind: str
    source: str | None = None
    payload: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None


@dataclass(frozen=True)
class LifecyclePlan:
    operation_id: str
    operation_type: str
    record_id: str
    revision: int
    scope: Mapping[str, object]
    now: datetime
    report: Mapping[str, object]
    mutations: tuple[LifecycleMutation, ...]
    preserved: tuple[Mapping[str, object], ...] = ()


LifecycleTransactionExecutor = Callable[[OmhPaths, LifecyclePlan], Mapping[str, object]]
