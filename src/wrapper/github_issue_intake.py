"""Safe GitHub issue-intake lifecycle; the core never contacts GitHub."""

from __future__ import annotations

from dataclasses import replace

from .github_issue_intake_templates import build_template_request, template_request_authorized
from .github_issue_intake_types import (
    ARTIFACT_FIELD_ALLOWLIST,
    AUTHORITIES,
    CLASSIFICATIONS,
    CONFIRMATION_STATES,
    DUPLICATE_STATUSES,
    GITHUB_ISSUE_INTAKE_SCHEMA,
    INTAKE_BLOCKERS,
    INTERVIEW_TOPICS,
    MAINTAINER_AUTHORITY,
    MAX_INTERVIEW_QUESTIONS,
    PUBLIC_REPORTER_AUTHORITY,
    READ_BACK_FIELDS,
    SCOPED_PUBLIC_MUTATION,
    SECURITY_SIGNAL_MARKERS,
    AuthenticatedMaintainerObservation,
    ConnectorHandoff,
    ConnectorRequest,
    DirectionCheck,
    IssueIntakeArtifact,
    IssueIntakeError,
    ObservedIssueResult,
    ReadBackVerification,
)

__all__ = (
    "ARTIFACT_FIELD_ALLOWLIST", "AUTHORITIES", "CLASSIFICATIONS", "CONFIRMATION_STATES",
    "DUPLICATE_STATUSES", "GITHUB_ISSUE_INTAKE_SCHEMA", "INTAKE_BLOCKERS", "INTERVIEW_TOPICS",
    "MAINTAINER_AUTHORITY", "MAX_INTERVIEW_QUESTIONS", "PUBLIC_REPORTER_AUTHORITY", "READ_BACK_FIELDS",
    "SCOPED_PUBLIC_MUTATION", "SECURITY_SIGNAL_MARKERS", "AuthenticatedMaintainerObservation",
    "ConnectorHandoff", "ConnectorRequest", "DirectionCheck", "IssueIntakeArtifact", "IssueIntakeError", "ObservedIssueResult",
    "ReadBackVerification", "bounded_interview", "build_template_request", "confirm_creation",
    "connector_handoff", "creation_verified", "direction_check", "direction_check_complete",
    "missing_evidence_request", "mutation_authorized", "outcome_summary", "prepare_issue_intake",
    "record_connector_blocker", "record_duplicate_search", "record_observed_result", "security_redirect_required",
    "verify_read_back",
)


def prepare_issue_intake(
    *, source_boundary: str, classification: str, user_visible_problem: str, source_summary: str,
    desired_outcome: str, scope_included: tuple[str, ...], scope_excluded: tuple[str, ...],
    observed_evidence: tuple[str, ...], inference: tuple[str, ...], repository: str, title: str,
    body: str, labels: tuple[str, ...], approval_authority: str = PUBLIC_REPORTER_AUTHORITY,
    template_fields: tuple[tuple[str, str], ...] = (),
) -> IssueIntakeArtifact:
    """Prepare a transient connector package and a metadata-only artifact projection."""
    if classification not in CLASSIFICATIONS or approval_authority not in AUTHORITIES:
        raise IssueIntakeError("unknown-intake-classification-or-authority")
    if approval_authority == MAINTAINER_AUTHORITY:
        raise IssueIntakeError("maintainer-authority-requires-wrapper-observation")
    if not repository.strip():
        raise IssueIntakeError("explicit-repository-required")
    request = build_template_request(classification, repository, template_fields, authority=approval_authority) if template_fields else None
    artifact = IssueIntakeArtifact(
        source_boundary, classification, user_visible_problem, source_summary, desired_outcome,
        scope_included, scope_excluded, observed_evidence, inference, repository, connector_request=request,
    )
    security_content = (*_artifact_material(artifact), title, body)
    if request:
        security_content = (*security_content, request.title, request.body)
    if security_redirect_required(security_content):
        return replace(artifact, blocker="security_redirect")
    return artifact


def bounded_interview(*, established: tuple[str, ...] = (), desired_outcome: str = "", scope_boundary: str = "", missing_evidence: str = "") -> tuple[str, ...]:
    """Return at most three unresolved decision-changing interview topics."""
    answers = (desired_outcome, scope_boundary, missing_evidence)
    return tuple(topic for topic, answer in zip(INTERVIEW_TOPICS, answers, strict=True) if topic not in established and not answer.strip())[:MAX_INTERVIEW_QUESTIONS]


def direction_check(artifact: IssueIntakeArtifact) -> DirectionCheck:
    """Present only the structured facts needed for an approval decision."""
    return DirectionCheck(artifact.classification, artifact.user_visible_problem, artifact.source_summary,
        artifact.desired_outcome, artifact.scope_included, artifact.scope_excluded, artifact.observed_evidence,
        artifact.inference, artifact.duplicate_status)


def direction_check_complete(artifact: IssueIntakeArtifact) -> bool:
    """Require all public-direction facts before a write can be authorized."""
    return bool(artifact.classification != "unclear" and artifact.user_visible_problem.strip() and artifact.desired_outcome.strip() and artifact.scope_included and artifact.observed_evidence)


def missing_evidence_request(artifact: IssueIntakeArtifact) -> str:
    """Name the facts that stop filing instead of creating a vague issue."""
    gaps = []
    if artifact.classification == "unclear": gaps.append("classification")
    if not artifact.user_visible_problem.strip(): gaps.append("user-visible problem")
    if not artifact.desired_outcome.strip(): gaps.append("desired outcome")
    if not artifact.scope_included: gaps.append("scope boundary")
    if not artifact.observed_evidence: gaps.append("observed evidence")
    return "Missing evidence before filing: " + "; ".join(gaps) + "." if gaps else "No missing evidence; the direction check is ready for confirmation."


def record_duplicate_search(artifact: IssueIntakeArtifact, refs: tuple[str, ...], *, confirmed_duplicate: bool = False) -> IssueIntakeArtifact:
    """Record the required duplicate-search outcome before confirmation."""
    if confirmed_duplicate and not refs: raise IssueIntakeError("confirmed-duplicate-needs-reference")
    status = "confirmed_duplicate" if confirmed_duplicate else "candidates" if refs else "none_found"
    return replace(artifact, duplicate_status=status, duplicate_refs=refs)


def confirm_creation(
    artifact: IssueIntakeArtifact, *, maintainer_file_now: bool = False,
    maintainer_observation: AuthenticatedMaintainerObservation | None = None,
) -> IssueIntakeArtifact:
    """Authorize one immutable request after all public and maintainer gates pass."""
    if maintainer_file_now: raise IssueIntakeError("boolean-maintainer-bypass-is-not-authentication")
    request = artifact.connector_request
    if (
        artifact.blocker
        or artifact.duplicate_status in ("not_searched", "confirmed_duplicate")
        or not direction_check_complete(artifact)
        or request is None
        or not template_request_authorized(request, artifact.classification)
    ):
        raise IssueIntakeError("direction-duplicate-security-and-template-gates-must-pass")
    if artifact.confirmation_state == "declined": raise IssueIntakeError("declined-direction-cannot-be-confirmed")
    if maintainer_observation:
        if not maintainer_observation.actor_id.strip() or not maintainer_observation.evidence_ref.strip():
            raise IssueIntakeError("authenticated-maintainer-identity-and-evidence-required")
        return replace(artifact, confirmation_state="maintainer_file_now", handoff_state="authorized", approval_authority=MAINTAINER_AUTHORITY)
    return replace(artifact, confirmation_state="confirmed", handoff_state="authorized")


def connector_handoff(artifact: IssueIntakeArtifact) -> ConnectorHandoff:
    """Consume authorization once; the external connector must enforce the stable key."""
    if artifact.blocker or artifact.confirmation_state not in ("confirmed", "maintainer_file_now"):
        raise IssueIntakeError("connector-handoff-not-authorized")
    request = artifact.connector_request
    if (
        artifact.duplicate_status == "confirmed_duplicate"
        or artifact.handoff_state != "authorized"
        or request is None
        or not template_request_authorized(request, artifact.classification)
    ):
        raise IssueIntakeError("connector-handoff-blocked")
    return ConnectorHandoff(request, replace(artifact, handoff_state="dispatched"))


def record_connector_blocker(artifact: IssueIntakeArtifact, blocker: str) -> IssueIntakeArtifact:
    """Record a non-successful connector outcome without clearing a security redirect."""
    if blocker not in INTAKE_BLOCKERS: raise IssueIntakeError("unknown-intake-blocker")
    return replace(artifact, blocker="security_redirect" if artifact.blocker == "security_redirect" else blocker, observed_result=None)


def record_observed_result(artifact: IssueIntakeArtifact, observed: ObservedIssueResult) -> IssueIntakeArtifact:
    """Accept read-back only from the one dispatched state and matching request key."""
    request = artifact.connector_request
    if artifact.blocker or artifact.handoff_state != "dispatched" or request is None:
        raise IssueIntakeError("observed-result-requires-dispatched-handoff")
    if observed.idempotency_key != request.idempotency_key:
        raise IssueIntakeError("observed-result-does-not-bind-authorized-request")
    return replace(artifact, observed_result=observed, handoff_state="observed")


def verify_read_back(artifact: IssueIntakeArtifact) -> ReadBackVerification:
    """Compare connector read-back with every field authorized for the one write."""
    request, observed = artifact.connector_request, artifact.observed_result
    if request is None or observed is None: return ReadBackVerification(False, (), ("observed_result",))
    pairs = (("repository", observed.repository, request.repository), ("author", observed.author, "present"),
             ("title", observed.title, request.title), ("body", observed.body, request.body),
             ("labels", set(observed.labels), set(request.labels)), ("url", observed.url, "present"),
             ("idempotency_key", observed.idempotency_key, request.idempotency_key))
    missing = tuple(
        field for field, actual, expected in pairs
        if expected == "present" and isinstance(actual, str) and not actual.strip()
    )
    changed = tuple(field for field, actual, expected in pairs if expected != "present" and actual != expected)
    mismatches = missing + changed
    return ReadBackVerification(not mismatches, READ_BACK_FIELDS, mismatches)


def creation_verified(artifact: IssueIntakeArtifact) -> bool:
    return verify_read_back(artifact).verified


def outcome_summary(artifact: IssueIntakeArtifact) -> str:
    if creation_verified(artifact) and artifact.observed_result: return f"Issue created and read-back verified: {artifact.observed_result.url}"
    if artifact.blocker: return f"Issue package prepared but blocked ({artifact.blocker}). No issue was filed."
    return "Issue package prepared; creation was not observed. No issue was filed."


def mutation_authorized(mutation: str, *, authority: str) -> bool:
    return authority in AUTHORITIES and mutation == SCOPED_PUBLIC_MUTATION


def security_redirect_required(report_markers: tuple[str, ...]) -> bool:
    return any(signal in value.casefold() for value in report_markers for signal in SECURITY_SIGNAL_MARKERS)


def _artifact_material(artifact: IssueIntakeArtifact) -> tuple[str, ...]:
    return (artifact.user_visible_problem, artifact.source_summary, artifact.desired_outcome,
            *artifact.scope_included, *artifact.scope_excluded, *artifact.observed_evidence, *artifact.inference)
