from __future__ import annotations

from typing import Any

WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION = "work_artifact_copy_manifest/v1"

COPY_CLAIM_BOUNDARY = (
    "Copying this artifact is not dispatch, execution, verification, review, CI, "
    "merge-readiness, or merge evidence."
)

# The stable id set. Order and ids are the contract a caller lists against; an
# entry stays listed with no text rather than disappearing when its source is
# absent, so a picker never changes shape between two reads of the same work.
_ARTIFACT_IDS = (
    "handoff_prompt",
    "acceptance_and_verification",
    "status_brief",
    "evidence_gaps",
    "next_action",
    "issue_pr_followup",
)

_ARTIFACT_LABELS = {
    "handoff_prompt": "Handoff prompt",
    "acceptance_and_verification": "Acceptance and verification",
    "status_brief": "Status brief",
    "evidence_gaps": "Evidence gaps and claim boundary",
    "next_action": "Next action",
    "issue_pr_followup": "Issue/PR follow-up",
}

_ARTIFACT_TYPES = {
    "handoff_prompt": "prompt_text",
    "acceptance_and_verification": "criteria_list",
    "status_brief": "status_text",
    "evidence_gaps": "gap_list",
    "next_action": "action_text",
    "issue_pr_followup": "followup_text",
}


def build_work_artifact_copy_manifest(
    briefing: dict[str, Any],
    *,
    prompt_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """List the copyable blocks of one work item with stable ids and exact text.

    Metadata-only projection of already-persisted wrapper artifacts: it reads a
    ``coding_briefing/v1`` payload plus the session's prompt handoff and copies
    their text verbatim. Nothing here composes new prose, and nothing here
    upgrades a prepared artifact into observed evidence.
    """

    handoff = prompt_handoff if isinstance(prompt_handoff, dict) else {}
    source_schema = str(briefing.get("schema_version", ""))
    texts = {
        "handoff_prompt": (str(handoff.get("prompt_template", "")), str(handoff.get("schema_version", ""))),
        "acceptance_and_verification": (_acceptance_text(briefing), source_schema),
        "status_brief": (_status_text(briefing), source_schema),
        "evidence_gaps": (_evidence_gap_text(briefing), source_schema),
        "next_action": (str(briefing.get("next_action", "")), source_schema),
        "issue_pr_followup": (_followup_text(briefing), source_schema),
    }
    return {
        "schema_version": WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION,
        "session_id": str(briefing.get("session_id", "")),
        "run_id": str(briefing.get("run_id", "")),
        "artifacts": [_entry(artifact_id, *texts[artifact_id]) for artifact_id in _ARTIFACT_IDS],
        "claim_boundary": COPY_CLAIM_BOUNDARY,
    }


def select_work_artifact(manifest: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    """Return the exact text of one listed artifact, or an unavailable answer."""

    for entry in _entries(manifest):
        if str(entry.get("artifact_id", "")) != artifact_id:
            continue
        return {
            "schema_version": WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "label": str(entry.get("label", "")),
            "artifact_type": str(entry.get("artifact_type", "")),
            "source_schema": str(entry.get("source_schema", "")),
            "availability": str(entry.get("availability", "unavailable")),
            "reason": str(entry.get("reason", "")),
            "boundary": "prepared_not_observed",
            "text": str(entry.get("text", "")),
            "claim_boundary": COPY_CLAIM_BOUNDARY,
        }
    return {
        "schema_version": WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "label": "",
        "artifact_type": "",
        "source_schema": "",
        "availability": "unavailable",
        "reason": "unknown_artifact_id",
        "boundary": "prepared_not_observed",
        "text": "",
        "claim_boundary": COPY_CLAIM_BOUNDARY,
    }


def _entry(artifact_id: str, text: str, source_schema: str) -> dict[str, Any]:
    available = bool(text)
    return {
        "artifact_id": artifact_id,
        "label": _ARTIFACT_LABELS[artifact_id],
        "artifact_type": _ARTIFACT_TYPES[artifact_id],
        "source_schema": source_schema if available else "",
        "availability": "available" if available else "unavailable",
        "reason": "" if available else "source_not_recorded",
        "boundary": "prepared_not_observed",
        "text": text,
    }


def _acceptance_text(briefing: dict[str, Any]) -> str:
    work_summary = _object(briefing.get("work_summary"))
    acceptance = _string_list(work_summary.get("acceptance_criteria"))
    verification = _string_list(work_summary.get("verification_expected"))
    lines: list[str] = []
    if acceptance:
        lines.extend(["Acceptance criteria:", *(f"- {item}" for item in acceptance)])
    if verification:
        if lines:
            lines.append("")
        lines.extend(["Verification expected:", *(f"- {item}" for item in verification)])
    return "\n".join(lines)


def _status_text(briefing: dict[str, Any]) -> str:
    return "\n".join(_string_list(briefing.get("user_facing_lines")))


def _evidence_gap_text(briefing: dict[str, Any]) -> str:
    gaps = _string_list(briefing.get("pending_gaps"))
    if not gaps:
        return ""
    claim_boundary = str(briefing.get("claim_boundary", ""))
    lines = ["Pending evidence gaps:", *(f"- {gap}" for gap in gaps)]
    if claim_boundary:
        lines.extend(["", claim_boundary])
    return "\n".join(lines)


def _followup_text(briefing: dict[str, Any]) -> str:
    """The issue/PR block is optional: it exists only when a linked run does."""
    run_id = str(briefing.get("run_id", ""))
    if not run_id:
        return ""
    session_id = str(briefing.get("session_id", ""))
    return "\n".join(
        [
            f"Run: {run_id}",
            f"Session: {session_id}",
            f"Next action: {briefing.get('next_action', '')}",
        ]
    )


def _entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [entry for entry in artifacts if isinstance(entry, dict)]


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]
