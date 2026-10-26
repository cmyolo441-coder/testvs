from __future__ import annotations

from ..paths import OmhPaths
from .domain_intelligence_contracts import (
    DEFAULT_REVIEW_REASON_CODE,
    SAFE_CANDIDATE_ID,
    canonical_profile_digest,
    ensure_no_forbidden_keys,
    normalize_base_profile_revision,
    normalize_identifier,
    normalize_mappings_from_value,
    normalize_scope_from_value,
    normalize_workflow_hints,
    stable_profile_id,
)
from .domain_intelligence_lineage import (
    ProfileValidationContext,
    build_profile_validation_context,
    profile_predecessor,
    validate_profile_candidate_lineage,
    validate_profile_review_lineage,
)
from .domain_intelligence_review_validation import (
    canonical_reason_code,
    canonical_reviewer_claim,
    matching_profile_review,
    validate_review_artifact_for_status as validate_review_artifact_for_status,
)
from .domain_intelligence_schema import validate_candidate_contract
from .domain_intelligence_store import read_profile
from .domain_intelligence_validation_state import (
    ProfileValidationFrame as _ProfileFrame,
    canonical_confidence as _canonical_confidence,
    canonical_provenance as _canonical_provenance,
    profile_key as _profile_key,
    validate_profile_identity as _validate_profile_identity,
)


def ensure_candidate_pending(candidate: dict[str, object]) -> None:
    validate_candidate_artifact(candidate)
    if candidate.get("status") != "pending_review":
        raise ValueError("candidate_not_pending_review")


def validate_candidate_artifact(candidate: dict[str, object]) -> None:
    ensure_no_forbidden_keys(candidate)
    status = validate_candidate_contract(candidate)
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("unsafe_candidate_id")
    scope = normalize_scope_from_value(candidate.get("scope"))
    domain_id = normalize_identifier(candidate.get("domain_id"), "domain_id")
    if candidate.get("domain_id") != domain_id:
        raise ValueError("candidate_domain_id_not_canonical")
    if candidate.get("scope") != scope:
        raise ValueError("candidate_scope_not_normalized")
    if candidate.get("profile_id") != stable_profile_id(scope, domain_id):
        raise ValueError("candidate_profile_identity_mismatch")
    if candidate.get("vocabulary_mappings") != normalize_mappings_from_value(candidate.get("vocabulary_mappings")):
        raise ValueError("candidate_mappings_not_canonical")
    if candidate.get("workflow_hints") != normalize_workflow_hints(candidate.get("workflow_hints")):
        raise ValueError("candidate_workflow_hints_not_canonical")
    confidence = _canonical_confidence(candidate.get("confidence"))
    provenance = _canonical_provenance(candidate.get("provenance"))
    if confidence["observation_count"] != provenance["observation_count"]:
        raise ValueError("observation_count_mismatch")
    normalize_base_profile_revision(candidate.get("base_profile_revision"))
    if status == "approved":
        revision = candidate.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("invalid_revision")
        if candidate.get("review_id") != f"direview_{candidate['profile_id']}_r{revision}":
            raise ValueError("candidate_review_identity_mismatch")
        canonical_reviewer_claim(candidate.get("reviewed_by"))
    elif status == "rejected":
        if candidate.get("review_id") != f"direview_{candidate_id}":
            raise ValueError("candidate_review_identity_mismatch")
        canonical_reviewer_claim(candidate.get("reviewed_by"))
        canonical_reason_code(candidate.get("rejection_reason_code"))


def validate_profile_artifact(
    paths: OmhPaths,
    profile: dict[str, object],
    *,
    context: ProfileValidationContext | None = None,
) -> None:
    _validate_profile_artifact(
        paths,
        profile,
        context=context,
        require_candidate_record=True,
    )


def _validate_profile_artifact(
    paths: OmhPaths,
    profile: dict[str, object],
    *,
    context: ProfileValidationContext | None,
    require_candidate_record: bool,
) -> None:
    context = context or build_profile_validation_context(paths)
    stack = [_ProfileFrame(profile=profile, is_root=True)]
    added_keys: set[tuple[str, int]] = set()
    try:
        while stack:
            frame = stack.pop()
            try:
                if frame.complete:
                    if require_candidate_record:
                        validate_profile_candidate_lineage(
                            frame.profile,
                            frame.review or {},
                            context=context,
                            validate_candidate=validate_candidate_artifact,
                            predecessor=frame.predecessor,
                        )
                    else:
                        validate_profile_review_lineage(
                            frame.profile,
                            frame.review or {},
                            predecessor=frame.predecessor,
                        )
                    profile_key = _profile_key(frame.profile)
                    context.validating.remove(profile_key)
                    added_keys.remove(profile_key)
                    context.validated[profile_key] = frame.profile
                    continue
                status, profile_key = _validate_profile_identity(frame.profile)
                cached = context.validated.get(profile_key)
                if cached is not None:
                    if cached != frame.profile:
                        raise ValueError("approved_candidate_lineage_required")
                    continue
                if profile_key in context.validating:
                    raise ValueError("approved_candidate_lineage_required")
                review = _validate_profile_content(context, frame.profile, status)
                predecessor = profile_predecessor(frame.profile, context)
                context.validating.add(profile_key)
                added_keys.add(profile_key)
                stack.append(
                    _ProfileFrame(
                        profile=frame.profile,
                        is_root=frame.is_root,
                        complete=True,
                        review=review,
                        predecessor=predecessor,
                    )
                )
                if predecessor is not None:
                    stack.append(_ProfileFrame(profile=predecessor, is_root=False))
            except (OSError, TypeError, ValueError) as exc:
                if not frame.is_root:
                    raise ValueError("approved_candidate_lineage_required") from exc
                raise
    finally:
        context.validating.difference_update(added_keys)


def validate_profile_artifact_for_resolution(
    paths: OmhPaths,
    profile: dict[str, object],
    *,
    context: ProfileValidationContext,
) -> None:
    """Validate reviewed profile lineage without consulting candidate storage."""
    _validate_profile_artifact(
        paths,
        profile,
        context=context,
        require_candidate_record=False,
    )


def _validate_profile_content(
    context: ProfileValidationContext,
    profile: dict[str, object],
    status: str,
) -> dict[str, object]:
    profile_id, revision = _profile_key(profile)
    scope = normalize_scope_from_value(profile.get("scope"))
    if profile.get("scope") != scope:
        raise ValueError("scope_not_normalized")
    if profile.get("vocabulary_mappings") != normalize_mappings_from_value(profile.get("vocabulary_mappings")):
        raise ValueError("profile_mappings_not_canonical")
    if profile.get("workflow_hints") != normalize_workflow_hints(profile.get("workflow_hints")):
        raise ValueError("profile_workflow_hints_not_canonical")
    confidence = _canonical_confidence(profile.get("confidence"))
    provenance = _canonical_provenance(profile.get("provenance"))
    if confidence["observation_count"] != provenance["observation_count"]:
        raise ValueError("observation_count_mismatch")
    normalize_base_profile_revision(profile.get("base_profile_revision"))
    candidate_id = profile.get("candidate_id")
    if not isinstance(candidate_id, str) or not SAFE_CANDIDATE_ID.fullmatch(candidate_id):
        raise ValueError("unsafe_candidate_id")
    canonical_reviewer_claim(profile.get("approved_by"))
    digest = canonical_profile_digest(profile)
    if profile.get("payload_digest") != digest:
        raise ValueError("payload_digest_mismatch")
    reviewer = profile.get("approved_by")
    reason = DEFAULT_REVIEW_REASON_CODE
    review_candidate_id = candidate_id
    decision = "approved"
    if status == "retired":
        reviewer = canonical_reviewer_claim(profile.get("retired_by"))
        reason = canonical_reason_code(profile.get("retirement_reason_code"))
        review_candidate_id = ""
        decision = "retired"
    review = matching_profile_review(
        context,
        profile_id,
        revision,
        decision,
        digest,
        reviewer,
        reason,
        review_candidate_id,
    )
    if not review:
        raise ValueError("matching_review_required")
    return review


def current_profile_for_authority(paths: OmhPaths, profile_id: str) -> dict[str, object] | None:
    profile = read_profile(paths, profile_id)
    if profile:
        validate_profile_artifact(paths, profile)
    return profile


def current_profile_revision(paths: OmhPaths, profile_id: str) -> int:
    profile = current_profile_for_authority(paths, profile_id)
    return int(profile["revision"]) if profile else 0
