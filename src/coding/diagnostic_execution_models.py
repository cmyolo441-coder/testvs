"""Typed boundaries and metadata-only values for diagnostic execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from .diagnostic_providers import DIAGNOSTIC_PROVIDER_IDS, DiagnosticCheckOutcome


EXECUTION_STATUSES = (
    "ok", "disabled", "unsupported", "unavailable", "timeout", "cancelled", "crashed", "partial", "stale",
)
OBSERVATION_STATES = ("completed", "timeout", "cancelled", "crashed", "unavailable")
ProviderObservationState = Literal["completed", "timeout", "cancelled", "crashed", "unavailable"]


class ChangedFileResolver(Protocol):
    """Resolves the bounded changed-file scope for an immutable interval."""

    def resolve(self, workspace_id: str, baseline_revision: str, end_revision: str) -> tuple[str, ...]: ...


class RevisionReader(Protocol):
    """Resolves a requested revision, including a moving end ref, to its observed value."""

    def read(self, workspace_id: str, revision: str) -> str: ...


class CancellationSignal(Protocol):
    """A cooperative cancellation boundary injected by the caller."""

    def is_set(self) -> bool: ...


class ProviderRunner(Protocol):
    """Runs one provider over one already-resolved revision and file scope."""

    def run(
        self,
        provider_id: str,
        workspace_id: str,
        revision: str,
        files: tuple[str, ...],
        timeout_ms: int,
        cancelled: CancellationSignal | None,
    ) -> "ProviderObservation": ...


@dataclass(frozen=True)
class ProviderObservation:
    """A runner observation before the provider facade normalizes it."""

    state: ProviderObservationState
    diagnosed_files: tuple[str, ...] = ()
    diagnostics: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.state not in OBSERVATION_STATES:
            raise ValueError(f"diagnostic_execution observation state is unsupported: {self.state!r}")
        object.__setattr__(self, "diagnosed_files", tuple(self.diagnosed_files))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @classmethod
    def completed(
        cls, diagnosed_files: tuple[str, ...], diagnostics: tuple[Mapping[str, object], ...]
    ) -> "ProviderObservation":
        return cls("completed", diagnosed_files, diagnostics)

    @classmethod
    def unavailable(cls) -> "ProviderObservation":
        return cls("unavailable")

    @classmethod
    def crashed(cls) -> "ProviderObservation":
        return cls("crashed")


@dataclass(frozen=True)
class DiagnosticExecutionSettings:
    """Execution-only bounds; provider capabilities retain their own bounds."""

    enabled: bool = True
    max_global_concurrency: int = 4
    max_provider_concurrency: int = 1
    stateful_providers: frozenset[str] = frozenset()
    revalidate_workspace_head: bool = False

    def __post_init__(self) -> None:
        for field, value in (
            ("max_global_concurrency", self.max_global_concurrency),
            ("max_provider_concurrency", self.max_provider_concurrency),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 32:
                raise ValueError(f"diagnostic_execution {field} must be an integer between 1 and 32")
        providers = frozenset(self.stateful_providers)
        if unknown := providers - set(DIAGNOSTIC_PROVIDER_IDS):
            raise ValueError(f"diagnostic_execution stateful providers are not allowlisted: {sorted(unknown)}")
        object.__setattr__(self, "stateful_providers", providers)


@dataclass(frozen=True)
class DiagnosticExecutionRequest:
    """One post-GREEN interval whose diagnostics are an observation, never verification."""

    owner: str
    workspace_id: str
    baseline_revision: str
    end_revision: str
    workspace_path: str = ""


@dataclass(frozen=True)
class ProviderDiagnosticResult:
    """Provider-pair result with normalized deltas and one v1 evidence record."""

    provider_id: str
    status: str
    baseline: DiagnosticCheckOutcome | None
    end: DiagnosticCheckOutcome | None
    evidence: dict[str, object]


@dataclass(frozen=True)
class DiagnosticExecutionResult:
    """The bounded execution report; it cannot carry verification status."""

    status: str
    results: tuple[ProviderDiagnosticResult, ...]
