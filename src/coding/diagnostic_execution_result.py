"""Normalize provider observations into metadata-only diagnostic evidence."""

from __future__ import annotations

from dataclasses import dataclass

from ..quality.language_diagnostic_evidence import (
    build_language_diagnostic_evidence,
)
from .diagnostic_execution_models import (
    DiagnosticExecutionRequest,
    ProviderDiagnosticResult,
    ProviderObservation,
)
from .diagnostic_providers import (
    DiagnosticCheckOutcome,
    DiagnosticProviderError,
    ProviderCapability,
    build_diagnostic_check_outcome,
)


@dataclass(frozen=True, slots=True)
class DiagnosticResultContext:
    """Shared immutable inputs for one provider result."""

    request: DiagnosticExecutionRequest
    files: tuple[str, ...]
    baseline_revision: str
    observed_end: str
    final_end: str
    config_identity: str


def build_provider_result(
    context: DiagnosticResultContext,
    capability: ProviderCapability,
    pair: tuple[ProviderObservation, ProviderObservation],
) -> ProviderDiagnosticResult:
    """Build one provider result without retaining raw provider output."""
    baseline_observation, end_observation = pair
    provider_files = in_scope_files(context.files, capability)
    invalid = False
    try:
        baseline = _outcome(
            capability,
            context.request.workspace_id,
            context.baseline_revision,
            baseline_observation,
            provider_files,
            context.config_identity,
        )
        end = _outcome(
            capability,
            context.request.workspace_id,
            context.final_end,
            end_observation,
            provider_files,
            context.config_identity,
            observed_revision=context.observed_end,
        )
    except DiagnosticProviderError:
        baseline, end, invalid = None, None, True
    status = (
        "crashed"
        if invalid
        else _status(
            baseline,
            end,
            baseline_observation,
            end_observation,
        )
    )
    introduced, resolved = _deltas(baseline, end)
    evidence = build_language_diagnostic_evidence(
        owner=context.request.owner,
        provider=capability.provider_id,
        workspace_id=context.request.workspace_id,
        baseline_revision=context.baseline_revision,
        end_revision=context.final_end,
        diagnostics_revision=end.diagnostics_revision if end else "",
        check_state=_evidence_state(status),
        config_digest=context.config_identity,
        changed_paths=provider_files,
        introduced=introduced,
        resolved=resolved,
    )
    return ProviderDiagnosticResult(
        capability.provider_id,
        status,
        baseline,
        end,
        evidence,
    )


def in_scope_files(
    files: tuple[str, ...],
    capability: ProviderCapability,
) -> tuple[str, ...]:
    return tuple(
        path
        for path in files
        if path.endswith(capability.file_suffixes)
    )


def overall_execution_status(
    results: tuple[ProviderDiagnosticResult, ...],
) -> str:
    statuses = {result.status for result in results}
    return statuses.pop() if len(statuses) == 1 else "partial"


def _outcome(
    capability: ProviderCapability,
    workspace_id: str,
    revision: str,
    observation: ProviderObservation,
    files: tuple[str, ...],
    config_identity: str,
    observed_revision: str | None = None,
) -> DiagnosticCheckOutcome | None:
    if observation.state == "unavailable":
        return None
    return build_diagnostic_check_outcome(
        workspace_id=workspace_id,
        revision=revision,
        diagnostics_revision=(
            revision
            if observed_revision is None
            else observed_revision
        ),
        provider_id=capability.provider_id,
        terminal_state=observation.state,
        compatibility="provider_selected",
        in_scope_files=files,
        diagnosed_files=observation.diagnosed_files,
        diagnostics=observation.diagnostics,
        config_identity=config_identity,
    )


def _status(
    baseline: DiagnosticCheckOutcome | None,
    end: DiagnosticCheckOutcome | None,
    baseline_observation: ProviderObservation,
    end_observation: ProviderObservation,
) -> str:
    if (
        baseline_observation.state == "unavailable"
        or end_observation.state == "unavailable"
    ):
        return "unavailable"
    if baseline is not None and baseline.outcome != "ok":
        return baseline.outcome
    if end is not None:
        return end.outcome
    return "unavailable"


def _deltas(
    baseline: DiagnosticCheckOutcome | None,
    end: DiagnosticCheckOutcome | None,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    before = (
        {
            tuple(item.as_record().items()): item.as_record()
            for item in baseline.diagnostics
        }
        if baseline
        else {}
    )
    after = (
        {
            tuple(item.as_record().items()): item.as_record()
            for item in end.diagnostics
        }
        if end
        else {}
    )
    return (
        tuple(after[key] for key in sorted(set(after) - set(before))),
        tuple(before[key] for key in sorted(set(before) - set(after))),
    )


def _evidence_state(status: str) -> str:
    if status in ("ok", "stale"):
        return "observed"
    if status in ("timeout", "crashed"):
        return "failed"
    if status == "unsupported":
        return "unsupported"
    return "not_observed"
