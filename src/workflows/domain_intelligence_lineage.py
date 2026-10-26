from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..paths import OmhPaths
from .domain_intelligence_store import read_candidates, read_history_profiles, read_reviews


_CONTENT_FIELDS = (
    "profile_id",
    "scope",
    "domain_id",
    "vocabulary_mappings",
    "workflow_hints",
    "confidence",
    "provenance",
)
_SHARED_APPROVAL_FIELDS = _CONTENT_FIELDS + ("base_profile_revision",)
_RETIRED_INHERITED_FIELDS = _CONTENT_FIELDS + (
    "candidate_id",
    "approved_by",
    "approved_at",
    "created_at",
)
_CHAIN_IDENTITY_FIELDS = ("profile_id", "scope", "domain_id")


@dataclass
class ProfileValidationContext:
    history: dict[tuple[str, int], dict[str, object]]
    candidates: dict[str, dict[str, object]]
    reviews: dict[str, dict[str, object]]
    validated: dict[tuple[str, int], dict[str, object]] = field(default_factory=dict)
    validating: set[tuple[str, int]] = field(default_factory=set)


def build_profile_validation_context(
    paths: OmhPaths,
    *,
    history: list[tuple[dict[str, object], Path]] | None = None,
    candidates: list[tuple[dict[str, object], Path]] | None = None,
    reviews: list[tuple[dict[str, object], Path]] | None = None,
) -> ProfileValidationContext:
    history = read_history_profiles(paths, []) if history is None else history
    candidates = read_candidates(paths, []) if candidates is None else candidates
    reviews = read_reviews(paths, []) if reviews is None else reviews
    return ProfileValidationContext(
        history={
            (str(item.get("profile_id")), int(item.get("revision", 0))): item
            for item, _path in history
        },
        candidates={str(item.get("candidate_id")): item for item, _path in candidates},
        reviews={str(item.get("review_id")): item for item, _path in reviews},
    )


def validate_profile_candidate_lineage(
    profile: dict[str, object],
    review: dict[str, object],
    *,
    context: ProfileValidationContext,
    validate_candidate: Callable[[dict[str, object]], None],
    predecessor: dict[str, object] | None,
) -> None:
    status = profile.get("status")
    if status == "active":
        candidate = _read_approved_candidate(profile, context, validate_candidate)
        _validate_active_lineage(profile, review, candidate)
        return
    _validate_retired_lineage(profile, review, predecessor)


def validate_profile_review_lineage(
    profile: dict[str, object],
    review: dict[str, object],
    *,
    predecessor: dict[str, object] | None,
) -> None:
    if profile.get("status") == "active":
        _validate_active_review_lineage(profile, review)
        return
    _validate_retired_lineage(profile, review, predecessor)


def _validate_retired_lineage(
    profile: dict[str, object],
    review: dict[str, object],
    predecessor: dict[str, object] | None,
) -> None:
    if predecessor is None:
        raise ValueError("approved_candidate_lineage_required")
    if any(
        profile.get(field) != predecessor.get(field)
        for field in _RETIRED_INHERITED_FIELDS
    ):
        raise ValueError("approved_candidate_lineage_required")
    if review.get("candidate_id") != "" or review.get("reviewed_at") != profile.get(
        "retired_at"
    ):
        raise ValueError("approved_candidate_lineage_required")


def profile_predecessor(
    profile: dict[str, object],
    context: ProfileValidationContext,
) -> dict[str, object] | None:
    revision = profile.get("revision")
    base_revision = profile.get("base_profile_revision")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or base_revision != revision - 1
    ):
        raise ValueError("approved_candidate_lineage_required")
    if revision == 1:
        return None
    predecessor = context.history.get(
        (str(profile.get("profile_id")), int(base_revision))
    )
    allowed_statuses = (
        {"active", "retired"} if profile.get("status") == "active" else {"active"}
    )
    if (
        predecessor is None
        or predecessor.get("revision") != base_revision
        or predecessor.get("status") not in allowed_statuses
        or any(
            predecessor.get(field) != profile.get(field)
            for field in _CHAIN_IDENTITY_FIELDS
        )
    ):
        raise ValueError("approved_candidate_lineage_required")
    if predecessor.get("base_profile_revision") != base_revision - 1:
        raise ValueError("approved_candidate_lineage_required")
    return predecessor


def _read_approved_candidate(
    profile: dict[str, object],
    context: ProfileValidationContext,
    validate_candidate: Callable[[dict[str, object]], None],
) -> dict[str, object]:
    try:
        candidate = context.candidates[str(profile["candidate_id"])]
        validate_candidate(candidate)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("approved_candidate_lineage_required") from exc
    if candidate.get("status") != "approved":
        raise ValueError("approved_candidate_lineage_required")
    return candidate


def _validate_active_lineage(
    profile: dict[str, object], review: dict[str, object], candidate: dict[str, object]
) -> None:
    if any(
        candidate.get(field) != profile.get(field) for field in _SHARED_APPROVAL_FIELDS
    ):
        raise ValueError("approved_candidate_lineage_required")
    expected = {
        "candidate_id": profile.get("candidate_id"),
        "profile_id": profile.get("profile_id"),
        "revision": profile.get("revision"),
        "review_id": review.get("review_id"),
        "reviewed_by": profile.get("approved_by"),
        "reviewed_at": review.get("reviewed_at"),
    }
    if any(candidate.get(key) != value for key, value in expected.items()):
        raise ValueError("approved_candidate_lineage_required")
    if review.get("decision") != "approved" or review.get(
        "reviewer_claim"
    ) != candidate.get("reviewed_by"):
        raise ValueError("approved_candidate_lineage_required")
    if profile.get("approved_at") != review.get("reviewed_at"):
        raise ValueError("approved_candidate_lineage_required")


def _validate_active_review_lineage(
    profile: dict[str, object], review: dict[str, object]
) -> None:
    if (
        review.get("decision") != "approved"
        or review.get("candidate_id") != profile.get("candidate_id")
        or review.get("reviewer_claim") != profile.get("approved_by")
        or review.get("reviewed_at") != profile.get("approved_at")
    ):
        raise ValueError("approved_candidate_lineage_required")
