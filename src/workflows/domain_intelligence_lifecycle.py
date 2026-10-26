from __future__ import annotations

from ..paths import OmhPaths
from .domain_intelligence_capture import capture_domain_candidate as capture_domain_candidate
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
    normalize_identifier,
    normalize_reason_code,
    normalize_safe_ref,
    normalize_scope,
    stable_profile_id,
)
from .domain_intelligence_store import (
    domain_store_lock,
    read_candidate_or_raise,
    read_profile,
)
from .domain_intelligence_operations import (
    approval_operation_exists,
    build_approval_operation,
    delete_approval_operation,
    finalize_legacy_approval,
    load_approval_operation,
    validate_approval_resume_state,
    write_approval_operation,
    write_archive_idempotent,
    write_candidate_resumable,
    write_profile_resumable,
    write_review_idempotent,
)
from .domain_intelligence_transition_fence import require_profile_transition
from .domain_intelligence_rejection_operations import (
    build_rejection_operation, delete_rejection_operation,
    load_rejection_operation,
    validate_rejection_resume_state,
    write_rejection_candidate_resumable,
    write_rejection_operation,
    write_rejection_review_idempotent,
)
from .domain_intelligence_retirement_operations import (
    build_retirement_operation,
    delete_retirement_operation,
    load_retirement_operation,
    validate_retirement_resume_state,
    write_retirement_archive_idempotent,
    write_retirement_operation,
    write_retirement_profile_resumable,
    write_retirement_review_idempotent,
)
from .domain_intelligence_validation import (
    current_profile_for_authority,
    ensure_candidate_pending,
    validate_profile_artifact,
)


def approve_domain_candidate(paths: OmhPaths, candidate_id: str, *, approved_by: str = "operator") -> dict[str, object]:
    reviewer_claim = normalize_safe_ref(approved_by, "reviewer_claim")
    with domain_store_lock(paths):
        operation = load_approval_operation(paths, candidate_id)
        if operation is None:
            candidate = read_candidate_or_raise(paths, candidate_id)
            require_profile_transition(
                paths,
                profile_id=str(candidate["profile_id"]),
                operation_id=f"approve_{candidate_id}",
            )
            ensure_candidate_pending(candidate)
            observed_current = read_profile(paths, str(candidate["profile_id"]))
            legacy = finalize_legacy_approval(
                paths,
                candidate,
                observed_current,
                reviewer_claim=reviewer_claim,
            )
            if legacy is not None:
                return {
                    "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
                    "decision": "approved",
                    **legacy,
                    "claim_boundary": CLAIM_BOUNDARY,
                }
            current = current_profile_for_authority(paths, str(candidate["profile_id"]))
            current_revision = int(current.get("revision", 0)) if current else 0
            if int(candidate.get("base_profile_revision", -1)) != current_revision:
                raise ValueError("stale_candidate")
            operation = build_approval_operation(candidate, current, reviewer_claim=reviewer_claim)
            write_approval_operation(paths, operation)
        else:
            require_profile_transition(
                paths,
                profile_id=str(operation["profile_id"]),
                operation_id=str(operation["operation_id"]),
            )
            if operation["target_profile"].get("approved_by") != reviewer_claim:
                raise ValueError("approval_operation_reviewer_mismatch")
        validate_approval_resume_state(paths, operation)
        write_archive_idempotent(paths, operation)
        write_review_idempotent(paths, operation)
        write_profile_resumable(paths, operation)
        write_candidate_resumable(paths, operation)
        profile = operation["target_profile"]
        review = operation["target_review"]
        decided = operation["target_candidate"]
        delete_approval_operation(paths, candidate_id)
    return {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "decision": "approved",
        "candidate": decided,
        "profile": profile,
        "review": review,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def reject_domain_candidate(
    paths: OmhPaths,
    candidate_id: str,
    *,
    rejected_by: str = "operator",
    reason: str = "",
) -> dict[str, object]:
    reviewer_claim = normalize_safe_ref(rejected_by, "reviewer_claim")
    reason_code = normalize_reason_code(reason)
    with domain_store_lock(paths):
        operation = load_rejection_operation(paths, candidate_id)
        if approval_operation_exists(paths, candidate_id):
            raise ValueError("approval_in_progress")
        if operation is None:
            candidate = read_candidate_or_raise(paths, candidate_id)
            require_profile_transition(
                paths,
                profile_id=str(candidate["profile_id"]),
                operation_id=f"reject_{candidate_id}",
            )
            ensure_candidate_pending(candidate)
            current = current_profile_for_authority(paths, str(candidate["profile_id"]))
            if current is not None and current.get("candidate_id") == candidate_id:
                raise ValueError("candidate_already_approved")
            operation = build_rejection_operation(
                candidate,
                reviewer_claim=reviewer_claim,
                reason_code=reason_code,
            )
            validate_rejection_resume_state(paths, operation)
            write_rejection_operation(paths, operation)
        else:
            require_profile_transition(
                paths,
                profile_id=str(operation["profile_id"]),
                operation_id=str(operation["operation_id"]),
            )
            if (
                operation["target_review"].get("reviewer_claim") != reviewer_claim
                or operation["target_review"].get("reason_code") != reason_code
            ):
                raise ValueError("rejection_operation_decision_mismatch")
        current = current_profile_for_authority(paths, str(operation["pending_candidate"]["profile_id"]))
        if current is not None and current.get("candidate_id") == candidate_id:
            raise ValueError("candidate_already_approved")
        validate_rejection_resume_state(paths, operation)
        write_rejection_review_idempotent(paths, operation)
        write_rejection_candidate_resumable(paths, operation)
        decided = operation["target_candidate"]
        review = operation["target_review"]
        delete_rejection_operation(paths, candidate_id)
    return {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "decision": "rejected",
        "candidate": decided,
        "review": review,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def retire_domain_profile(
    paths: OmhPaths,
    *,
    scope_kind: str,
    scope_ref: str,
    domain_id: str,
    retired_by: str = "operator",
    reason: str = "",
) -> dict[str, object]:
    reviewer_claim = normalize_safe_ref(retired_by, "reviewer_claim")
    reason_code = normalize_reason_code(reason)
    scope = normalize_scope(scope_kind, scope_ref)
    normalized_domain = normalize_identifier(domain_id, "domain_id")
    profile_id = stable_profile_id(scope, normalized_domain)
    with domain_store_lock(paths):
        operation = load_retirement_operation(paths, profile_id)
        if operation is None:
            require_profile_transition(
                paths,
                profile_id=profile_id,
                operation_id=f"retire_{profile_id}",
            )
            current = read_profile(paths, profile_id)
            if current is None:
                raise FileNotFoundError(profile_id)
            validate_profile_artifact(paths, current)
            if current.get("status") == "retired":
                raise ValueError("already_retired")
            operation = build_retirement_operation(
                current,
                reviewer_claim=reviewer_claim,
                reason_code=reason_code,
            )
            validate_retirement_resume_state(paths, operation)
            write_retirement_operation(paths, operation)
        else:
            require_profile_transition(
                paths,
                profile_id=str(operation["profile_id"]),
                operation_id=str(operation["operation_id"]),
            )
            if (
                operation["target_review"].get("reviewer_claim") != reviewer_claim
                or operation["target_review"].get("reason_code") != reason_code
            ):
                raise ValueError("retirement_operation_decision_mismatch")
        validate_retirement_resume_state(paths, operation)
        write_retirement_archive_idempotent(paths, operation)
        write_retirement_review_idempotent(paths, operation)
        write_retirement_profile_resumable(paths, operation)
        retired = operation["target_profile"]
        review = operation["target_review"]
        delete_retirement_operation(paths, profile_id)
    return {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "decision": "retired",
        "profile": retired,
        "review": review,
        "claim_boundary": CLAIM_BOUNDARY,
    }
