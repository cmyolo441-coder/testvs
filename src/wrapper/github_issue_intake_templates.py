"""Checked-in GitHub issue-form projection for intake connector requests."""

from __future__ import annotations

from hashlib import sha256
from secrets import token_hex

from .github_issue_intake_types import (
    AUTHORITIES,
    ConnectorRequest,
    IssueIntakeError,
    _TemplateReceipt,
)

_TEMPLATE_FIELDS = {
    "bug": (
        ("summary", True), ("user_goal", True), ("reproduce", True),
        ("expected", True), ("affected_surfaces", True), ("observed_evidence", True),
        ("acceptance_criteria", True), ("environment", True), ("logs", False),
    ),
    "feature_request": (
        ("problem", True), ("user_goal", True), ("affected_surfaces", True),
        ("observed_evidence", False), ("proposal", True), ("alternatives", False),
        ("validation", True), ("boundaries", False),
    ),
}
_TEMPLATE_LABELS = {
    "bug": ("bug", "needs-triage"),
    "feature_request": ("enhancement", "needs-triage"),
}
_TEMPLATE_TITLES = {"bug": "[Bug]", "feature_request": "[Feature]"}
_FORM_BY_CLASSIFICATION = {
    "bug": "bug",
    "feature_request": "feature_request",
    "documentation_gap": "feature_request",
    "setup_support": "bug",
}
_FORM_LABELS = {
    "summary": "Summary", "user_goal": "Complete user goal", "reproduce": "Reproduction",
    "expected": "Expected behavior", "affected_surfaces": "Affected surfaces",
    "observed_evidence": "Observed evidence", "acceptance_criteria": "Acceptance criteria",
    "environment": "Environment", "logs": "Logs or output", "problem": "Problem",
    "proposal": "Proposal", "alternatives": "Alternatives considered", "validation": "Success criteria",
    "boundaries": "Boundaries and risks",
}


def build_template_request(
    classification: str,
    repository: str,
    fields: tuple[tuple[str, str], ...],
    *,
    authority: str = "public_reporter_scoped",
) -> ConnectorRequest:
    """Build the exact body and labels required by the checked-in issue form."""
    if authority not in AUTHORITIES:
        raise IssueIntakeError("unknown issue-authority")
    form_id = _FORM_BY_CLASSIFICATION.get(classification)
    if form_id is None:
        raise IssueIntakeError("issue-template-required-for-classification")
    if not repository.strip():
        raise IssueIntakeError("explicit-repository-required")
    values = dict(fields)
    expected = _TEMPLATE_FIELDS[form_id]
    known = {field for field, _required in expected}
    unknown = set(values) - known
    if unknown:
        raise IssueIntakeError("unknown-issue-template-field")
    missing = tuple(field for field, required in expected if required and not values.get(field, "").strip())
    if missing:
        raise IssueIntakeError("missing-required-issue-template-field")
    sections = tuple((field, values[field].strip()) for field, _required in expected if values.get(field, "").strip())
    title_source = sections[0][1]
    title = f"{_TEMPLATE_TITLES[form_id]}: {title_source}"
    body = "\n\n".join(f"## {_FORM_LABELS[field]}\n{value}" for field, value in sections)
    digest = _request_key(repository, title, body, _TEMPLATE_LABELS[form_id])
    idempotency_key = token_hex(32)
    receipt = _TemplateReceipt(
        form_id,
        tuple(field for field, required in expected if required),
        digest,
        idempotency_key,
    )
    return ConnectorRequest(
        repository,
        title,
        body,
        _TEMPLATE_LABELS[form_id],
        idempotency_key,
        receipt,
    )


def request_from_complete_content(
    repository: str,
    title: str,
    body: str,
    labels: tuple[str, ...],
) -> ConnectorRequest:
    """Preserve existing connector-complete artifacts while assigning their stable key."""
    if not title.strip() or not body.strip() or not labels:
        raise IssueIntakeError("incomplete-connector-request")
    return ConnectorRequest(repository, title, body, labels, token_hex(32))


def template_request_authorized(request: ConnectorRequest, classification: str) -> bool:
    """Verify that a complete checked-in form, not arbitrary content, built the request."""
    receipt = request.template_receipt
    form_id = _FORM_BY_CLASSIFICATION.get(classification)
    if receipt is None or form_id is None or receipt.form_id != form_id:
        return False
    expected = _TEMPLATE_FIELDS.get(form_id)
    if expected is None:
        return False
    required_fields = tuple(field for field, required in expected if required)
    expected_digest = _request_key(
        request.repository,
        request.title,
        request.body,
        request.labels,
    )
    return (
        receipt.required_fields == required_fields
        and receipt.request_digest == expected_digest
        and receipt.idempotency_key == request.idempotency_key
    )


def _request_key(repository: str, title: str, body: str, labels: tuple[str, ...]) -> str:
    material = "\x1f".join((repository, title, body, *labels)).encode()
    return sha256(material).hexdigest()
