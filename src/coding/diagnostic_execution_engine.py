"""Bounded post-GREEN execution of allowlisted diagnostic providers."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from subprocess import SubprocessError, TimeoutExpired
from threading import Lock, Semaphore

from .diagnostic_execution_models import (
    CancellationSignal,
    ChangedFileResolver,
    DiagnosticExecutionRequest,
    DiagnosticExecutionResult,
    DiagnosticExecutionSettings,
    ProviderObservation,
    ProviderRunner,
    RevisionReader,
)
from .diagnostic_execution_result import (
    DiagnosticResultContext,
    build_provider_result,
    in_scope_files,
    overall_execution_status,
)
from .diagnostic_providers import (
    DIAGNOSTIC_PROVIDER_IDS,
    DiagnosticProviderConfig,
    ProviderCapability,
)


class DiagnosticExecutionEngine:
    """Runs baseline/end observations with caching, bounds, and HEAD revalidation."""

    def __init__(
        self,
        *,
        config: DiagnosticProviderConfig,
        resolver: ChangedFileResolver,
        revisions: RevisionReader,
        runner: ProviderRunner,
        settings: DiagnosticExecutionSettings | None = None,
        cancellation: CancellationSignal | None = None,
    ) -> None:
        self.config = config
        self.resolver = resolver
        self.revisions = revisions
        self.runner = runner
        self.settings = settings or DiagnosticExecutionSettings()
        self.cancellation = cancellation
        self._global_slots = Semaphore(self.settings.max_global_concurrency)
        self._provider_slots = {provider: Semaphore(self.settings.max_provider_concurrency) for provider in DIAGNOSTIC_PROVIDER_IDS}
        self._stateful_locks = {provider: Lock() for provider in self.settings.stateful_providers}
        self._cache: dict[tuple[str, str, str, tuple[str, ...]], ProviderObservation] = {}
        self._inflight: dict[tuple[str, str, str, tuple[str, ...]], Future[ProviderObservation]] = {}
        self._cache_lock = Lock()

    def execute(self, request: DiagnosticExecutionRequest) -> DiagnosticExecutionResult:
        """Observe every applicable provider at both interval endpoints.

        Provider output is handed only to the committed provider facade and the
        evidence builder. Both reject non-metadata diagnostics by key name.
        """
        if not self.settings.enabled:
            return DiagnosticExecutionResult("disabled", ())
        if self._cancelled():
            return DiagnosticExecutionResult("cancelled", ())
        execution_workspace = request.workspace_path or request.workspace_id
        try:
            baseline = self.revisions.read(
                execution_workspace,
                request.baseline_revision,
            )
            end = self.revisions.read(
                execution_workspace,
                request.end_revision,
            )
            if (
                request.workspace_path
                and self.settings.revalidate_workspace_head
            ):
                workspace_head = self.revisions.read(
                    execution_workspace,
                    "HEAD",
                )
                if workspace_head != end:
                    return DiagnosticExecutionResult("stale", ())
            files = self.resolver.resolve(
                execution_workspace,
                baseline,
                end,
            )
        except (OSError, SubprocessError):
            return DiagnosticExecutionResult("crashed", ())
        selected = self._selected(files)
        if not selected:
            return DiagnosticExecutionResult("unsupported", ())
        with ThreadPoolExecutor(max_workers=min(len(selected), self.settings.max_global_concurrency)) as pool:
            futures = [
                pool.submit(
                    self._observe_pair,
                    request,
                    capability,
                    files,
                    baseline,
                    end,
                    execution_workspace,
                )
                for capability in selected
            ]
            pairs = [future.result() for future in futures]
        try:
            final_end = self.revisions.read(
                execution_workspace,
                request.end_revision,
            )
            if (
                request.workspace_path
                and self.settings.revalidate_workspace_head
            ):
                final_workspace_head = self.revisions.read(
                    execution_workspace,
                    "HEAD",
                )
                if final_workspace_head != final_end:
                    final_end = final_workspace_head
        except (OSError, SubprocessError):
            return DiagnosticExecutionResult("crashed", ())
        context = DiagnosticResultContext(
            request,
            files,
            baseline,
            end,
            final_end,
            self.config.config_identity(),
        )
        results = tuple(
            build_provider_result(context, capability, pair)
            for capability, pair in zip(selected, pairs)
        )
        return DiagnosticExecutionResult(
            overall_execution_status(results),
            results,
        )

    def _selected(self, files: tuple[str, ...]) -> tuple[ProviderCapability, ...]:
        by_id = {capability.provider_id: capability for capability in self.config.capabilities}
        return tuple(
            capability for provider in DIAGNOSTIC_PROVIDER_IDS
            if (capability := by_id.get(provider)) is not None
            and capability.enabled
            and 0
            < len(in_scope_files(files, capability))
            <= capability.max_files_per_check
        )

    def _observe_pair(
        self,
        request: DiagnosticExecutionRequest,
        capability: ProviderCapability,
        files: tuple[str, ...],
        baseline: str,
        end: str,
        execution_workspace: str,
    ) -> tuple[ProviderObservation, ProviderObservation]:
        scope = in_scope_files(files, capability)
        lock = self._stateful_locks.get(capability.provider_id)
        with lock if lock is not None else nullcontext():
            return (
                self._observe(capability, execution_workspace, baseline, scope),
                self._observe(capability, execution_workspace, end, scope),
            )

    def _observe(
        self, capability: ProviderCapability, workspace_id: str, revision: str, files: tuple[str, ...]
    ) -> ProviderObservation:
        if self._cancelled():
            return ProviderObservation("cancelled")
        key = (workspace_id, capability.provider_id, revision, files)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            future = self._inflight.get(key)
            if future is None:
                future = Future()
                self._inflight[key] = future
                owner = True
            else:
                owner = False
        if not owner:
            return future.result()
        try:
            # Provider slot first, global slot last. A global slot is then held
            # only by a runner that is executing. Acquired the other way round,
            # a request queued behind a busy provider held a global slot while
            # it waited, and with two global slots both taken by pyright (one
            # running, one waiting on pyright's own slot) ruff could not start
            # although nothing was running in its place (#1790).
            with self._provider_slots[capability.provider_id], self._global_slots:
                observed = self.runner.run(
                    capability.provider_id, workspace_id, revision, files, capability.max_timeout_ms, self.cancellation
                )
            if not isinstance(observed, ProviderObservation):
                observed = ProviderObservation.crashed()
        except TimeoutExpired:
            observed = ProviderObservation("timeout")
        except (OSError, SubprocessError):
            observed = ProviderObservation.crashed()
        except BaseException as exc:
            # Publish before re-raising: programming errors remain loud, but a
            # same-key waiter must never block forever behind the failed owner.
            with self._cache_lock:
                active = self._inflight.pop(key, None)
                if active is not None:
                    active.set_exception(exc)
            raise
        with self._cache_lock:
            self._cache[key] = observed
            active = self._inflight.pop(key, None)
            if active is not None:
                active.set_result(observed)
        return observed

    def _cancelled(self) -> bool:
        return self.cancellation.is_set() if self.cancellation is not None else False
