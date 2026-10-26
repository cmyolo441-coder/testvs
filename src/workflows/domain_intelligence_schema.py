from __future__ import annotations

from datetime import datetime
import re

from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_CANDIDATE_SCHEMA_VERSION,
    DOMAIN_PROFILE_SCHEMA_VERSION,
    DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
    REDUCTION_POLICY,
)


CANDIDATE_BASE_KEYS = frozenset(
    "schema_version candidate_id status profile_id scope domain_id vocabulary_mappings workflow_hints confidence provenance "
    "base_profile_revision created_at updated_at redaction_policy claim_boundary".split()
)
CANDIDATE_KEYS = {
    "pending_review": CANDIDATE_BASE_KEYS,
    "approved": CANDIDATE_BASE_KEYS | {"reviewed_at", "reviewed_by", "review_id", "revision"},
    "rejected": CANDIDATE_BASE_KEYS | {"reviewed_at", "reviewed_by", "review_id", "rejection_reason_code"},
}
PROFILE_ACTIVE_KEYS = frozenset(
    "schema_version profile_id revision status scope domain_id vocabulary_mappings workflow_hints confidence provenance "
    "base_profile_revision candidate_id approved_by approved_at created_at updated_at redaction_policy claim_boundary payload_digest".split()
)
PROFILE_KEYS = {
    "active": PROFILE_ACTIVE_KEYS,
    "retired": PROFILE_ACTIVE_KEYS | {"retired_at", "retired_by", "retirement_reason_code"},
}
PROFILE_REVIEW_KEYS = frozenset(
    "schema_version review_id candidate_id profile_id revision decision reviewer_claim payload_digest reviewed_at reason_code claim_boundary".split()
)
REJECTED_REVIEW_KEYS = frozenset(
    "schema_version review_id candidate_id profile_id revision decision reviewer_claim reviewed_at reason_code claim_boundary".split()
)
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_candidate_contract(candidate: dict[str, object]) -> str:
    if candidate.get("schema_version") != DOMAIN_CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unsupported_candidate_schema")
    status = candidate.get("status")
    if status not in CANDIDATE_KEYS:
        raise ValueError("invalid_candidate_status")
    if set(candidate) != CANDIDATE_KEYS[status]:
        raise ValueError("candidate_schema_mismatch")
    _require_constant(candidate, "claim_boundary", CLAIM_BOUNDARY)
    _require_constant(candidate, "redaction_policy", REDUCTION_POLICY)
    created_at = validate_utc_timestamp(candidate.get("created_at"), "candidate_created_at")
    updated_at = validate_utc_timestamp(candidate.get("updated_at"), "candidate_updated_at")
    if created_at > updated_at:
        raise ValueError("candidate_timestamp_order_invalid")
    if status == "pending_review" and created_at != updated_at:
        raise ValueError("candidate_timestamp_mismatch")
    if status != "pending_review" and validate_utc_timestamp(candidate.get("reviewed_at"), "candidate_reviewed_at") != updated_at:
        raise ValueError("candidate_timestamp_mismatch")
    return status


def validate_profile_contract(profile: dict[str, object]) -> str:
    if profile.get("schema_version") != DOMAIN_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported_profile_schema")
    status = profile.get("status")
    if status not in PROFILE_KEYS:
        raise ValueError("invalid_profile_status")
    if set(profile) != PROFILE_KEYS[status]:
        raise ValueError("profile_schema_mismatch")
    _require_constant(profile, "claim_boundary", CLAIM_BOUNDARY)
    _require_constant(profile, "redaction_policy", REDUCTION_POLICY)
    created_at = validate_utc_timestamp(profile.get("created_at"), "profile_created_at")
    approved_at = validate_utc_timestamp(profile.get("approved_at"), "profile_approved_at")
    updated_at = validate_utc_timestamp(profile.get("updated_at"), "profile_updated_at")
    if created_at > approved_at or approved_at > updated_at:
        raise ValueError("profile_timestamp_order_invalid")
    if status == "active" and approved_at != updated_at:
        raise ValueError("profile_timestamp_mismatch")
    if status == "retired" and validate_utc_timestamp(profile.get("retired_at"), "profile_retired_at") != updated_at:
        raise ValueError("profile_timestamp_mismatch")
    return status


def validate_review_contract(review: dict[str, object], expected_keys: frozenset[str]) -> None:
    if review.get("schema_version") != DOMAIN_REVIEW_RECORD_SCHEMA_VERSION:
        raise ValueError("unsupported_review_schema")
    if set(review) != expected_keys:
        raise ValueError("review_schema_mismatch")
    _require_constant(review, "claim_boundary", CLAIM_BOUNDARY)
    validate_utc_timestamp(review.get("reviewed_at"), "reviewed_at")


def validate_utc_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not _UTC_TIMESTAMP.fullmatch(value):
        raise ValueError(f"invalid_{label}")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError(f"invalid_{label}") from exc
    return value


def _require_constant(value: dict[str, object], key: str, expected: str) -> None:
    if value.get(key) != expected:
        raise ValueError(f"invalid_{key}")
