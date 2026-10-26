"""Typed records and closed vocabularies for GitHub issue intake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

GITHUB_ISSUE_INTAKE_SCHEMA = "github_issue_intake/v1"
CLASSIFICATIONS = ("bug", "feature_request", "documentation_gap", "setup_support", "unclear")
DUPLICATE_STATUSES = ("not_searched", "none_found", "candidates", "confirmed_duplicate")
CONFIRMATION_STATES = ("pending", "confirmed", "maintainer_file_now", "declined")
INTAKE_BLOCKERS = (
    "connector_unavailable",
    "credentials_missing",
    "duplicate_confirmed",
    "security_redirect",
    "missing_evidence",
)
READ_BACK_FIELDS = ("repository", "author", "title", "body", "labels", "url", "idempotency_key")
MAX_INTERVIEW_QUESTIONS = 3
INTERVIEW_TOPICS = ("desired outcome", "scope boundary", "missing evidence")
PUBLIC_REPORTER_AUTHORITY = "public_reporter_scoped"
MAINTAINER_AUTHORITY = "maintainer"
AUTHORITIES = (PUBLIC_REPORTER_AUTHORITY, MAINTAINER_AUTHORITY)
SCOPED_PUBLIC_MUTATION = "create_issue"
SECURITY_SIGNAL_MARKERS = ("security", "vulnerability", "cve", "exploit", "취약점", "보안")
ARTIFACT_FIELD_ALLOWLIST = frozenset(
    {
        "schema", "source_boundary", "classification", "field_digests", "evidence_refs",
        "duplicate_status", "duplicate_refs", "confirmation_state", "handoff_state", "intended_mutation",
        "approval_authority", "repository", "connector_request", "observed_result", "blocker",
    }
)


@dataclass(frozen=True, slots=True)
class _TemplateReceipt:
    """Opaque proof that the checked-in issue form built this request."""

    form_id: str
    required_fields: tuple[str, ...]
    request_digest: str
    idempotency_key: str


class PersistedConnectorRequest(TypedDict):
    repository: str
    labels: list[str]
    idempotency_key: str
    title_digest: str
    body_digest: str


class PersistedObservedResult(TypedDict):
    repository: str
    author: str
    labels: list[str]
    url: str
    idempotency_key: str
    title_digest: str
    body_digest: str


class PersistedIssueIntake(TypedDict):
    schema: str
    source_boundary: str
    classification: str
    field_digests: dict[str, str]
    evidence_refs: list[str]
    duplicate_status: str
    duplicate_refs: list[str]
    confirmation_state: str
    handoff_state: str
    intended_mutation: str
    approval_authority: str
    repository: str
    connector_request: PersistedConnectorRequest | None
    observed_result: PersistedObservedResult | None
    blocker: str


class IssueIntakeError(ValueError):
    """Raised when the intake lifecycle is advanced without its gate."""


@dataclass(frozen=True, slots=True)
class ConnectorRequest:
    """The complete, single-use write held only by the authorized connector."""

    repository: str
    title: str
    body: str
    labels: tuple[str, ...]
    idempotency_key: str
    template_receipt: _TemplateReceipt | None = None


@dataclass(frozen=True, slots=True)
class ObservedIssueResult:
    """Connector read-back, bound to the request that authorized the write."""

    repository: str
    author: str
    title: str
    body: str
    labels: tuple[str, ...]
    url: str
    idempotency_key: str = ""


@dataclass(frozen=True, slots=True)
class AuthenticatedMaintainerObservation:
    """Wrapper-authenticated maintainer transition evidence."""

    actor_id: str
    evidence_ref: str


@dataclass(frozen=True, slots=True)
class ReadBackVerification:
    verified: bool
    checked_fields: tuple[str, ...]
    mismatches: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectionCheck:
    report_type: str
    user_visible_problem: str
    source_summary: str
    smallest_desired_outcome: str
    scope_included: tuple[str, ...]
    scope_excluded: tuple[str, ...]
    observed_evidence: tuple[str, ...]
    inference: tuple[str, ...]
    duplicate_status: str


@dataclass(frozen=True, slots=True)
class IssueIntakeArtifact:
    """Transient content plus a metadata-only persistence projection."""

    source_boundary: str
    classification: str
    user_visible_problem: str
    source_summary: str
    desired_outcome: str
    scope_included: tuple[str, ...]
    scope_excluded: tuple[str, ...]
    observed_evidence: tuple[str, ...]
    inference: tuple[str, ...]
    repository: str
    duplicate_status: str = "not_searched"
    duplicate_refs: tuple[str, ...] = ()
    confirmation_state: str = "pending"
    handoff_state: str = "not_authorized"
    approval_authority: str = PUBLIC_REPORTER_AUTHORITY
    connector_request: ConnectorRequest | None = None
    observed_result: ObservedIssueResult | None = None
    blocker: str = ""
    schema: str = field(default=GITHUB_ISSUE_INTAKE_SCHEMA, init=False)
    intended_mutation: str = field(default=SCOPED_PUBLIC_MUTATION, init=False)

    def to_dict(self) -> PersistedIssueIntake:
        """Project only bounded metadata; raw report and connector text stay transient."""
        from .github_issue_intake_projection import artifact_projection

        return artifact_projection(self)


@dataclass(frozen=True, slots=True)
class ConnectorHandoff:
    """Single connector request plus the state consumed by its dispatch."""

    request: ConnectorRequest
    artifact: IssueIntakeArtifact
