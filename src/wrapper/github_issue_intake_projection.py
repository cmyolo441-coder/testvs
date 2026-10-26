"""Privacy-safe persistent projection for GitHub issue intake artifacts."""

from __future__ import annotations

from hashlib import sha256

from .github_issue_intake_types import (
    ARTIFACT_FIELD_ALLOWLIST,
    ConnectorRequest,
    IssueIntakeArtifact,
    IssueIntakeError,
    PersistedConnectorRequest,
    PersistedIssueIntake,
    ObservedIssueResult,
    PersistedObservedResult,
)


def artifact_projection(artifact: IssueIntakeArtifact) -> PersistedIssueIntake:
    """Return metadata, digests, and refs without raw reporter or connector content."""
    request = artifact.connector_request
    observed = artifact.observed_result
    result: PersistedIssueIntake = {
        "schema": artifact.schema,
        "source_boundary": artifact.source_boundary,
        "classification": artifact.classification,
        "field_digests": _field_digests(artifact),
        "evidence_refs": _refs(artifact.observed_evidence),
        "duplicate_status": artifact.duplicate_status,
        "duplicate_refs": list(artifact.duplicate_refs),
        "confirmation_state": artifact.confirmation_state,
        "handoff_state": artifact.handoff_state,
        "intended_mutation": artifact.intended_mutation,
        "approval_authority": artifact.approval_authority,
        "repository": artifact.repository,
        "connector_request": _request_projection(request) if request else None,
        "observed_result": _observed_projection(observed) if observed else None,
        "blocker": artifact.blocker,
    }
    if set(result) - ARTIFACT_FIELD_ALLOWLIST:
        raise IssueIntakeError("privacy-allowlist-escaped")
    return result


def _field_digests(artifact: IssueIntakeArtifact) -> dict[str, str]:
    return {
        "user_visible_problem": _digest(artifact.user_visible_problem),
        "source_summary": _digest(artifact.source_summary),
        "desired_outcome": _digest(artifact.desired_outcome),
        "scope_included": _digest("\x1f".join(artifact.scope_included)),
        "scope_excluded": _digest("\x1f".join(artifact.scope_excluded)),
        "observed_evidence": _digest("\x1f".join(artifact.observed_evidence)),
        "inference": _digest("\x1f".join(artifact.inference)),
    }


def _refs(evidence: tuple[str, ...]) -> list[str]:
    return [f"sha256:{_digest(item)}" for item in evidence[:8]]


def _request_projection(request: ConnectorRequest) -> PersistedConnectorRequest:
    return {
        "repository": request.repository,
        "labels": list(request.labels),
        "idempotency_key": request.idempotency_key,
        "title_digest": _digest(request.title),
        "body_digest": _digest(request.body),
    }


def _observed_projection(observed: ObservedIssueResult) -> PersistedObservedResult:
    return {
        "repository": observed.repository,
        "author": observed.author,
        "labels": list(observed.labels),
        "url": observed.url,
        "idempotency_key": observed.idempotency_key,
        "title_digest": _digest(observed.title),
        "body_digest": _digest(observed.body),
    }


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()
