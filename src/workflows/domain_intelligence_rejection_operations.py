from __future__ import annotations

from ..paths import OmhPaths
from ..system.local_store import utc_now
from . import domain_intelligence_operation_store as journal
from . import domain_intelligence_store as store
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
    SAFE_CANDIDATE_ID,
)
from .domain_intelligence_schema import REJECTED_REVIEW_KEYS, validate_review_contract
from .domain_intelligence_validation import validate_candidate_artifact


REJECTION_OPERATION_SCHEMA_VERSION = "domain_intelligence_rejection_operation/v1"
_OPERATION_KEYS = frozenset(
    "schema_version operation_id candidate_id profile_id pending_candidate target_review target_candidate "
    "claim_boundary operation_digest".split()
)


def build_rejection_operation(
    candidate: dict[str, object], *, reviewer_claim: str, reason_code: str
) -> dict[str, object]:
    now = utc_now()
    candidate_id = str(candidate["candidate_id"])
    review = {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "review_id": f"direview_{candidate_id}",
        "candidate_id": candidate_id,
        "profile_id": candidate["profile_id"],
        "revision": None,
        "decision": "rejected",
        "reviewer_claim": reviewer_claim,
        "reason_code": reason_code,
        "reviewed_at": now,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    target = _rejected_candidate(candidate, review)
    operation = journal.seal_operation(
        {
            "schema_version": REJECTION_OPERATION_SCHEMA_VERSION,
            "operation_id": _operation_id(candidate_id),
            "candidate_id": candidate_id,
            "profile_id": candidate["profile_id"],
            "pending_candidate": candidate,
            "target_review": review,
            "target_candidate": target,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    validate_rejection_operation(None, operation)
    return operation


def load_rejection_operation(paths: OmhPaths, candidate_id: str) -> dict[str, object] | None:
    return journal.load_operation(paths, _operation_id(candidate_id), validate_rejection_operation)


def write_rejection_operation(paths: OmhPaths, operation: dict[str, object]) -> None:
    journal.write_operation(paths, operation, validate_rejection_operation)


def validate_rejection_resume_state(paths: OmhPaths, operation: dict[str, object]) -> None:
    validate_rejection_operation(paths, operation)
    journal.require_expected_or_target(
        store.candidate_path(paths, str(operation["candidate_id"])),
        operation["pending_candidate"],
        operation["target_candidate"],
        label="rejection_candidate",
    )
    review = operation["target_review"]
    journal.require_absent_or_exact(
        store.review_path(paths, str(review["review_id"])), review, label="rejection_review"
    )


def write_rejection_review_idempotent(paths: OmhPaths, operation: dict[str, object]) -> None:
    review = operation["target_review"]
    journal.write_absent_or_exact(
        paths,
        store.review_path(paths, str(review["review_id"])), review, label="rejection_review"
    )


def write_rejection_candidate_resumable(paths: OmhPaths, operation: dict[str, object]) -> None:
    journal.write_expected_or_target(
        paths,
        store.candidate_path(paths, str(operation["candidate_id"])),
        operation["pending_candidate"],
        operation["target_candidate"],
        label="rejection_candidate",
    )


def delete_rejection_operation(paths: OmhPaths, candidate_id: str) -> None:
    journal.delete_operation(paths, _operation_id(candidate_id), validate_rejection_operation)


def validate_rejection_operation(paths: OmhPaths | None, operation: dict[str, object]) -> None:
    candidate_id = operation.get("candidate_id")
    if not isinstance(candidate_id, str) or not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("unsafe_candidate_id")
    journal.validate_operation_envelope(operation, operation_id=_operation_id(candidate_id))
    if set(operation) != _OPERATION_KEYS or operation.get("schema_version") != REJECTION_OPERATION_SCHEMA_VERSION:
        raise ValueError("rejection_operation_schema_mismatch")
    pending = operation.get("pending_candidate")
    target = operation.get("target_candidate")
    review = operation.get("target_review")
    if not all(isinstance(item, dict) for item in (pending, target, review)):
        raise ValueError("rejection_operation_artifact_type")
    validate_candidate_artifact(pending)
    validate_candidate_artifact(target)
    validate_review_contract(review, REJECTED_REVIEW_KEYS)
    if pending.get("status") != "pending_review" or target != _rejected_candidate(pending, review):
        raise ValueError("rejection_operation_transition_mismatch")
    expected_review = {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "review_id": f"direview_{candidate_id}",
        "candidate_id": candidate_id,
        "profile_id": pending.get("profile_id"),
        "revision": None,
        "decision": "rejected",
        "reviewer_claim": target.get("reviewed_by"),
        "reason_code": target.get("rejection_reason_code"),
        "reviewed_at": target.get("reviewed_at"),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if review != expected_review:
        raise ValueError("rejection_operation_review_mismatch")
    if operation.get("profile_id") != pending.get("profile_id"):
        raise ValueError("rejection_operation_profile_identity")


def _rejected_candidate(candidate: dict[str, object], review: dict[str, object]) -> dict[str, object]:
    return {
        **candidate,
        "status": "rejected",
        "reviewed_at": review.get("reviewed_at"),
        "reviewed_by": review.get("reviewer_claim"),
        "review_id": review.get("review_id"),
        "rejection_reason_code": review.get("reason_code"),
        "updated_at": review.get("reviewed_at"),
    }


def _operation_id(candidate_id: str) -> str:
    if not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("unsafe_candidate_id")
    return f"reject_{candidate_id}"
