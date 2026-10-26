from __future__ import annotations

from .domain_intelligence_contracts import (
    DEFAULT_REVIEW_REASON_CODE,
    SAFE_CANDIDATE_ID,
    SAFE_PROFILE_ID,
    SHA256,
    ensure_no_forbidden_keys,
    normalize_reason_code,
    normalize_safe_ref,
)
from .domain_intelligence_lineage import ProfileValidationContext
from .domain_intelligence_schema import (
    PROFILE_REVIEW_KEYS,
    REJECTED_REVIEW_KEYS,
    validate_review_contract,
)


def validate_review_artifact_for_status(
    review: dict[str, object],
    *,
    candidates: dict[str, dict[str, object]],
    profiles: dict[tuple[str, int], dict[str, object]],
) -> None:
    ensure_no_forbidden_keys(review)
    decision = review.get("decision")
    if decision == "rejected":
        _validate_rejected_review(review, candidates)
        return
    if decision not in {"approved", "retired"}:
        raise ValueError("invalid_review_decision")

    validate_review_contract(review, PROFILE_REVIEW_KEYS)
    profile_id = review.get("profile_id")
    revision = review.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("invalid_review_revision")
    if not isinstance(profile_id, str) or not SAFE_PROFILE_ID.fullmatch(profile_id):
        raise ValueError("unsafe_review_profile_id")
    matched = profiles.get((profile_id, revision))
    expected_status = "active" if decision == "approved" else "retired"
    if not matched or matched.get("status") != expected_status:
        raise ValueError("orphan_review")

    if decision == "approved":
        expected_candidate = matched.get("candidate_id")
        expected_reviewer = matched.get("approved_by")
        expected_reason = DEFAULT_REVIEW_REASON_CODE
    else:
        expected_candidate = ""
        expected_reviewer = matched.get("retired_by")
        expected_reason = matched.get("retirement_reason_code")
    validate_profile_review(
        review,
        profile_id,
        revision,
        decision,
        matched.get("payload_digest"),
        expected_reviewer,
        expected_reason,
        expected_candidate,
    )


def matching_profile_review(
    context: ProfileValidationContext,
    profile_id: str,
    revision: int,
    decision: str,
    digest: str,
    reviewer: object,
    reason: object,
    candidate_id: object,
) -> dict[str, object] | None:
    review = context.reviews.get(f"direview_{profile_id}_r{revision}")
    if not review:
        return None
    try:
        validate_profile_review(
            review,
            profile_id,
            revision,
            decision,
            digest,
            reviewer,
            reason,
            candidate_id,
        )
    except ValueError:
        return None
    return review


def validate_profile_review(
    review: dict[str, object],
    profile_id: str,
    revision: int,
    decision: str,
    digest: object,
    reviewer: object,
    reason: object,
    candidate_id: object,
) -> None:
    ensure_no_forbidden_keys(review)
    validate_review_contract(review, PROFILE_REVIEW_KEYS)
    if (
        review.get("review_id") != f"direview_{profile_id}_r{revision}"
        or review.get("profile_id") != profile_id
    ):
        raise ValueError("review_identity_mismatch")
    review_revision = review.get("revision")
    if isinstance(review_revision, bool) or review_revision != revision:
        raise ValueError("invalid_review_revision")
    if review.get("decision") != decision or review.get("candidate_id") != candidate_id:
        raise ValueError("review_decision_or_candidate_mismatch")
    review_digest = review.get("payload_digest")
    if not isinstance(review_digest, str) or not SHA256.fullmatch(review_digest):
        raise ValueError("invalid_review_digest")
    if review_digest != digest:
        raise ValueError("review_digest_mismatch")
    if canonical_reviewer_claim(review.get("reviewer_claim")) != reviewer:
        raise ValueError("review_reviewer_mismatch")
    if canonical_reason_code(review.get("reason_code")) != reason:
        raise ValueError("review_reason_mismatch")


def canonical_reviewer_claim(value: object) -> str:
    normalized = normalize_safe_ref(value, "reviewer_claim")
    if value != normalized:
        raise ValueError("reviewer_claim_not_canonical")
    return normalized


def canonical_reason_code(value: object) -> str:
    normalized = normalize_reason_code(value)
    if value != normalized:
        raise ValueError("review_reason_not_canonical")
    return normalized


def _validate_rejected_review(
    review: dict[str, object], candidates: dict[str, dict[str, object]]
) -> None:
    if review.get("decision") != "rejected":
        raise ValueError("invalid_review_decision")
    validate_review_contract(review, REJECTED_REVIEW_KEYS)
    candidate_id = review.get("candidate_id")
    if not isinstance(candidate_id, str) or not SAFE_CANDIDATE_ID.fullmatch(
        candidate_id
    ):
        raise ValueError("unsafe_review_candidate_id")
    matched = candidates.get(candidate_id)
    if not matched or matched.get("status") != "rejected":
        raise ValueError("orphan_review")
    if review.get("review_id") != matched.get("review_id") or review.get(
        "profile_id"
    ) != matched.get("profile_id"):
        raise ValueError("review_identity_mismatch")
    if review.get("revision") is not None:
        raise ValueError("invalid_review_revision")
    reviewer_claim = review.get("reviewer_claim")
    if reviewer_claim != matched.get("reviewed_by"):
        raise ValueError("review_reviewer_mismatch")
    reason_code = review.get("reason_code")
    if reason_code != matched.get("rejection_reason_code"):
        raise ValueError("review_reason_mismatch")
    if review.get("reviewed_at") != matched.get("reviewed_at"):
        raise ValueError("review_timestamp_mismatch")
    canonical_reviewer_claim(reviewer_claim)
    canonical_reason_code(reason_code)
