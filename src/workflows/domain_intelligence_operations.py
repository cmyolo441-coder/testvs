from __future__ import annotations

from ..paths import OmhPaths
from . import domain_intelligence_operation_store as journal
from . import domain_intelligence_store as store
from .domain_intelligence_artifacts import profile_from_candidate, review_record_for_profile
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DEFAULT_REVIEW_REASON_CODE,
    SAFE_CANDIDATE_ID,
    canonical_profile_digest,
)
from .domain_intelligence_schema import PROFILE_REVIEW_KEYS, validate_profile_contract, validate_review_contract
from .domain_intelligence_validation import validate_candidate_artifact, validate_profile_artifact


APPROVAL_OPERATION_SCHEMA_VERSION = "domain_intelligence_approval_operation/v1"
_OPERATION_KEYS = frozenset(
    "schema_version operation_id candidate_id profile_id base_profile_revision target_revision "
    "pending_candidate prior_profile target_review target_profile target_candidate claim_boundary operation_digest".split()
)
_LINEAGE_FIELDS = tuple(
    "profile_id scope domain_id vocabulary_mappings workflow_hints confidence provenance "
    "base_profile_revision redaction_policy claim_boundary candidate_id".split()
)


def build_approval_operation(
    candidate: dict[str, object], current: dict[str, object] | None, *, reviewer_claim: str
) -> dict[str, object]:
    profile = profile_from_candidate(candidate, current=current, reviewer_claim=reviewer_claim)
    review = review_record_for_profile(candidate, profile, reviewer_claim=reviewer_claim, decision="approved")
    review["reviewed_at"] = profile["approved_at"]
    target_candidate = _approved_candidate(candidate, profile, review)
    operation = journal.seal_operation(
        {
            "schema_version": APPROVAL_OPERATION_SCHEMA_VERSION,
            "operation_id": _operation_id(str(candidate["candidate_id"])),
            "candidate_id": candidate["candidate_id"],
            "profile_id": candidate["profile_id"],
            "base_profile_revision": candidate["base_profile_revision"],
            "target_revision": profile["revision"],
            "pending_candidate": candidate,
            "prior_profile": current,
            "target_review": review,
            "target_profile": profile,
            "target_candidate": target_candidate,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    validate_approval_operation(None, operation)
    return operation


def load_approval_operation(paths: OmhPaths, candidate_id: str) -> dict[str, object] | None:
    return journal.load_operation(paths, _operation_id(candidate_id), validate_approval_operation)


def approval_operation_exists(paths: OmhPaths, candidate_id: str) -> bool:
    return journal.operation_exists(paths, _operation_id(candidate_id), validate_approval_operation)


def finalize_legacy_approval(
    paths: OmhPaths, candidate: dict[str, object], current: dict[str, object] | None, *, reviewer_claim: str
) -> dict[str, object] | None:
    if current is None or current.get("candidate_id") != candidate.get("candidate_id"):
        return None
    review_id = f"direview_{current['profile_id']}_r{current['revision']}"
    review, error = store.read_review(paths, review_id)
    if error or review is None:
        raise ValueError("matching_review_required")
    target = _approved_candidate(candidate, current, review)
    validate_candidate_artifact(target)
    if current.get("approved_by") != reviewer_claim:
        raise ValueError("candidate_already_approved_conflict")
    prior = _legacy_prior_profile(paths, candidate)
    _validate_approval_targets(candidate, current, review, target, prior=prior, legacy_recovery=True)
    store.write_candidate(paths, str(candidate["candidate_id"]), target)
    return {"candidate": target, "profile": current, "review": review}


def write_approval_operation(paths: OmhPaths, operation: dict[str, object]) -> None:
    journal.write_operation(paths, operation, validate_approval_operation)


def validate_approval_resume_state(paths: OmhPaths, operation: dict[str, object]) -> None:
    validate_approval_operation(paths, operation)
    prior = operation["prior_profile"]
    profile = operation["target_profile"]
    candidate = operation["target_candidate"]
    journal.require_expected_or_target(
        store.profile_path(paths, str(operation["profile_id"])), prior, profile, label="approval_profile"
    )
    journal.require_expected_or_target(
        store.candidate_path(paths, str(operation["candidate_id"])),
        operation["pending_candidate"],
        candidate,
        label="approval_candidate",
    )
    if prior is not None:
        journal.require_absent_or_exact(
            store.history_path(paths, str(operation["profile_id"]), int(operation["base_profile_revision"])),
            prior,
            label="approval_history",
        )
    journal.require_absent_or_exact(
        store.review_path(paths, str(operation["target_review"]["review_id"])),
        operation["target_review"],
        label="approval_review",
    )


def write_archive_idempotent(paths: OmhPaths, operation: dict[str, object]) -> None:
    prior = operation["prior_profile"]
    if prior is not None:
        journal.write_absent_or_exact(
            paths,
            store.history_path(paths, str(operation["profile_id"]), int(operation["base_profile_revision"])),
            prior,
            label="approval_history",
        )


def write_review_idempotent(paths: OmhPaths, operation: dict[str, object]) -> None:
    review = operation["target_review"]
    journal.write_absent_or_exact(
        paths,
        store.review_path(paths, str(review["review_id"])), review, label="approval_review"
    )


def write_profile_resumable(paths: OmhPaths, operation: dict[str, object]) -> None:
    journal.write_expected_or_target(
        paths,
        store.profile_path(paths, str(operation["profile_id"])),
        operation["prior_profile"],
        operation["target_profile"],
        label="approval_profile",
    )


def write_candidate_resumable(paths: OmhPaths, operation: dict[str, object]) -> None:
    journal.write_expected_or_target(
        paths,
        store.candidate_path(paths, str(operation["candidate_id"])),
        operation["pending_candidate"],
        operation["target_candidate"],
        label="approval_candidate",
    )


def delete_approval_operation(paths: OmhPaths, candidate_id: str) -> None:
    journal.delete_operation(paths, _operation_id(candidate_id), validate_approval_operation)


def validate_approval_operation(paths: OmhPaths | None, operation: dict[str, object]) -> None:
    candidate_id = operation.get("candidate_id")
    if not isinstance(candidate_id, str) or not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("unsafe_candidate_id")
    operation_id = _operation_id(candidate_id)
    if set(operation) != _OPERATION_KEYS or operation.get("schema_version") != APPROVAL_OPERATION_SCHEMA_VERSION:
        raise ValueError("approval_operation_schema_mismatch")
    try:
        journal.validate_operation_envelope(operation, operation_id=operation_id)
    except ValueError as exc:
        if str(exc) == "decision_operation_digest_mismatch":
            raise ValueError("approval_operation_digest_mismatch") from exc
        raise
    pending = operation.get("pending_candidate")
    target = operation.get("target_candidate")
    profile = operation.get("target_profile")
    review = operation.get("target_review")
    if not all(isinstance(item, dict) for item in (pending, target, profile, review)):
        raise ValueError("approval_operation_artifact_type")
    validate_candidate_artifact(pending)
    validate_candidate_artifact(target)
    prior = operation.get("prior_profile")
    _validate_approval_targets(pending, profile, review, target, prior=prior)
    if operation.get("profile_id") != pending.get("profile_id"):
        raise ValueError("approval_operation_profile_identity")
    if operation.get("base_profile_revision") != pending.get("base_profile_revision"):
        raise ValueError("approval_operation_base_revision")
    if operation.get("target_revision") != profile.get("revision"):
        raise ValueError("approval_operation_target_revision")
    if prior is not None and paths is not None:
        validate_profile_artifact(paths, prior)


def _validate_approval_targets(
    pending: dict[str, object],
    profile: dict[str, object],
    review: dict[str, object],
    target: dict[str, object],
    *,
    prior: object,
    legacy_recovery: bool = False,
) -> None:
    validate_profile_contract(profile)
    validate_review_contract(review, PROFILE_REVIEW_KEYS)
    digest = canonical_profile_digest(profile)
    revision = profile.get("revision")
    expected_target = _approved_candidate(pending, profile, review)
    if pending.get("status") != "pending_review" or target != expected_target:
        raise ValueError("approval_operation_transition_mismatch")
    if any(profile.get(field) != pending.get(field) for field in _LINEAGE_FIELDS):
        raise ValueError("approval_operation_lineage_mismatch")
    expected_revision = int(pending["base_profile_revision"]) + 1
    if profile.get("status") != "active" or revision != expected_revision:
        raise ValueError("approval_operation_revision_mismatch")
    expected_review = {
        "review_id": f"direview_{pending['profile_id']}_r{revision}",
        "candidate_id": pending["candidate_id"],
        "profile_id": pending["profile_id"],
        "revision": revision,
        "decision": "approved",
        "reviewer_claim": profile.get("approved_by"),
        "payload_digest": digest,
        "reviewed_at": profile.get("approved_at"),
        "reason_code": DEFAULT_REVIEW_REASON_CODE,
        "claim_boundary": CLAIM_BOUNDARY,
        "schema_version": review.get("schema_version"),
    }
    if review != expected_review or profile.get("payload_digest") != digest:
        raise ValueError("approval_operation_review_mismatch")
    if prior is None:
        expected_created = profile.get("approved_at")
        if not legacy_recovery and pending.get("base_profile_revision") != 0:
            raise ValueError("approval_operation_prior_profile_required")
    else:
        if not isinstance(prior, dict):
            raise ValueError("approval_operation_prior_profile")
        validate_profile_contract(prior)
        if prior.get("payload_digest") != canonical_profile_digest(prior):
            raise ValueError("approval_operation_prior_profile")
        if prior.get("profile_id") != profile.get("profile_id") or prior.get("revision") != pending.get("base_profile_revision"):
            raise ValueError("approval_operation_prior_profile")
        expected_created = prior.get("created_at")
    if profile.get("created_at") != expected_created or profile.get("updated_at") != profile.get("approved_at"):
        raise ValueError("approval_operation_timestamp_mismatch")


def _approved_candidate(
    candidate: dict[str, object], profile: dict[str, object], review: dict[str, object]
) -> dict[str, object]:
    return {
        **candidate,
        "status": "approved",
        "reviewed_at": review.get("reviewed_at"),
        "reviewed_by": review.get("reviewer_claim"),
        "review_id": review.get("review_id"),
        "revision": profile.get("revision"),
        "updated_at": review.get("reviewed_at"),
    }


def _legacy_prior_profile(paths: OmhPaths, candidate: dict[str, object]) -> dict[str, object] | None:
    revision = int(candidate["base_profile_revision"])
    if revision == 0:
        return None
    diagnostics: list[dict[str, str]] = []
    matches = [
        profile
        for profile, _path in store.read_history_profiles(paths, diagnostics)
        if profile.get("profile_id") == candidate.get("profile_id") and profile.get("revision") == revision
    ]
    if diagnostics or len(matches) != 1:
        raise ValueError("approval_operation_prior_profile_required")
    validate_profile_artifact(paths, matches[0])
    return matches[0]


def _operation_id(candidate_id: str) -> str:
    if not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("unsafe_candidate_id")
    return f"approve_{candidate_id}"
