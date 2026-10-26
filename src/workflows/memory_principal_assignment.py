"""Immutable reviewed principal assignments for legacy-memory migration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..plugin_bundle.omh.memory_governance import canonical_payload_digest, stable_artifact_identity
from ..plugin_bundle.omh.memory_principals import memory_identity_errors, parse_principal_context
from ..system.local_store import read_json_object_result
from ..system.paths import OmhPaths
from .memory_store import run_memory_operation

ASSIGNMENT_REVIEW_SCHEMA = "memory_principal_assignment_review/v1"
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")


@dataclass(frozen=True, slots=True)
class PrincipalAssignmentError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


def source_review_reason(
    paths: OmhPaths,
    source: dict[str, Any],
    review_id: str,
) -> str:
    """Return empty only for an immutable review bound to this exact source."""
    if not _SAFE_REF.fullmatch(review_id):
        return "source_review_invalid"
    path = paths.memory_dir / "reviews" / f"{review_id}.json"
    if path.is_symlink():
        return "source_review_missing"
    review, error = read_json_object_result(path)
    if error or review is None:
        return "source_review_missing"
    if review.get("schema_version") != "project_memory_review_record/v2" or review.get("review_id") != review_id:
        return "source_review_invalid"
    if review.get("decision") not in {"approved_manual", "approved_auto_safe"}:
        return "source_review_invalid"
    if review.get("artifact_identity") != stable_artifact_identity(source):
        return "source_review_identity_mismatch"
    if review.get("payload_digest") != canonical_payload_digest(source):
        return "source_review_digest_mismatch"
    return ""


def principal_assignment_digest(
    source: dict[str, Any],
    source_review_id: str,
    identity: dict[str, Any],
) -> str:
    """Bind the exact source and behavior-bearing proposed principal policy."""
    reviewer_value = identity.get("reviewer")
    audience_value = identity.get("audience")
    reviewer: dict[str, Any] = reviewer_value if isinstance(reviewer_value, dict) else {}
    audience: dict[str, Any] = audience_value if isinstance(audience_value, dict) else {}
    policy = {
        "schema_version": identity.get("schema_version"),
        "subject_principal": identity.get("subject_principal"),
        "event_actor": identity.get("event_actor"),
        "reviewer_principal": reviewer.get("principal"),
        "executor_perspective": identity.get("executor_perspective"),
        "audience": {
            "schema_version": audience.get("schema_version"),
            "kind": audience.get("kind"),
            "principal_refs": audience.get("principal_refs"),
        },
    }
    body = {
        "source_identity": stable_artifact_identity(source),
        "source_payload_digest": canonical_payload_digest(source),
        "source_review_id": source_review_id,
        "proposed_identity_policy": policy,
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_principal_assignment_review(
    paths: OmhPaths,
    record_id: str,
    revision: int,
    source_review_id: str,
    identity: dict[str, Any],
    reviewer_context: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Persist one explicit assignment review through the existing operation store."""
    source, error = read_json_object_result(paths.memory_dir / "records" / f"{record_id}.json")
    if error or source is None or source.get("schema_version") != "project_memory_record/v2" or source.get("revision") != revision:
        raise PrincipalAssignmentError("source_revision_changed")
    review_reason = source_review_reason(paths, source, source_review_id)
    if review_reason:
        raise PrincipalAssignmentError(review_reason)
    if memory_identity_errors(identity):
        raise PrincipalAssignmentError("proposed_identity_invalid")
    reviewer = parse_principal_context(reviewer_context)
    identity_reviewer_value = identity.get("reviewer")
    identity_reviewer: dict[str, Any] = identity_reviewer_value if isinstance(identity_reviewer_value, dict) else {}
    if reviewer is None or reviewer.get("actor_kind") != "human" or reviewer.get("principal") != identity_reviewer.get("principal"):
        raise PrincipalAssignmentError("assignment_reviewer_mismatch")
    assignment_digest = principal_assignment_digest(source, source_review_id, identity)
    review_id = "principal_assignment_" + assignment_digest[7:31]
    moment = now or datetime.now(timezone.utc)
    reviewed_at = moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    review = {
        "schema_version": ASSIGNMENT_REVIEW_SCHEMA,
        "review_id": review_id,
        "decision": "approved_manual",
        "assignment_digest": assignment_digest,
        "source_identity": stable_artifact_identity(source),
        "source_payload_digest": canonical_payload_digest(source),
        "source_review_id": source_review_id,
        "proposed_identity_digest": assignment_digest,
        "reviewer_principal": reviewer["principal"],
        "reviewed_at": reviewed_at,
        "redaction_policy": "metadata_only",
    }
    operation_id = "principal-assignment-review-" + assignment_digest[7:31]
    operation = run_memory_operation(
        paths,
        operation_id=operation_id,
        operation_type="principal_assignment_review",
        steps=[{"name": "write_assignment_review", "action": "write_json", "target": f"reviews/{review_id}.json", "payload": review}],
        now=moment,
    )
    return {**review, "operation_id": operation_id, "operation_state": operation.get("state")}


def assignment_review_reason(
    paths: OmhPaths,
    source: dict[str, Any],
    source_review_id: str,
    identity: dict[str, Any],
    assignment_review_id: str,
) -> str:
    """Return empty only when a stored assignment review binds this proposal."""
    if not _SAFE_REF.fullmatch(assignment_review_id):
        return "assignment_review_invalid"
    path = paths.memory_dir / "reviews" / f"{assignment_review_id}.json"
    if path.is_symlink():
        return "assignment_review_missing"
    review, error = read_json_object_result(path)
    if error or review is None:
        return "assignment_review_missing"
    expected = principal_assignment_digest(source, source_review_id, identity)
    if (
        review.get("schema_version") != ASSIGNMENT_REVIEW_SCHEMA
        or review.get("review_id") != assignment_review_id
        or review.get("decision") != "approved_manual"
    ):
        return "assignment_review_invalid"
    if review.get("assignment_digest") != expected or review.get("proposed_identity_digest") != expected:
        return "assignment_review_mismatch"
    if review.get("source_identity") != stable_artifact_identity(source) or review.get("source_payload_digest") != canonical_payload_digest(source):
        return "assignment_review_source_mismatch"
    if review.get("source_review_id") != source_review_id:
        return "assignment_review_source_mismatch"
    return ""
