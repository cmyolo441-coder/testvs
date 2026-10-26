from __future__ import annotations

from ..system.local_store import utc_now
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DEFAULT_REVIEW_REASON_CODE,
    DOMAIN_PROFILE_SCHEMA_VERSION,
    DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
    REDUCTION_POLICY,
    canonical_profile_digest,
)


def profile_from_candidate(
    candidate: dict[str, object],
    *,
    current: dict[str, object] | None,
    reviewer_claim: str,
) -> dict[str, object]:
    now = utc_now()
    revision = current["revision"] + 1 if current else 1
    profile = {
        "schema_version": DOMAIN_PROFILE_SCHEMA_VERSION,
        "profile_id": candidate["profile_id"],
        "revision": revision,
        "status": "active",
        "scope": dict(candidate["scope"]),
        "domain_id": candidate["domain_id"],
        "vocabulary_mappings": list(candidate["vocabulary_mappings"]),
        "workflow_hints": list(candidate["workflow_hints"]),
        "confidence": dict(candidate["confidence"]),
        "provenance": dict(candidate["provenance"]),
        "base_profile_revision": candidate["base_profile_revision"],
        "candidate_id": candidate["candidate_id"],
        "approved_by": reviewer_claim,
        "approved_at": now,
        "created_at": now if current is None else current["created_at"],
        "updated_at": now,
        "redaction_policy": REDUCTION_POLICY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    profile["payload_digest"] = canonical_profile_digest(profile)
    return profile


def review_record_for_profile(
    candidate: dict[str, object] | None,
    profile: dict[str, object],
    *,
    reviewer_claim: str,
    decision: str,
    reason: str = DEFAULT_REVIEW_REASON_CODE,
) -> dict[str, object]:
    if decision == "approved":
        now = profile["approved_at"]
    elif decision == "retired":
        now = profile["retired_at"]
    else:
        raise ValueError("invalid_review_decision")
    profile_id = profile["profile_id"]
    revision = profile["revision"]
    return {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "review_id": f"direview_{profile_id}_r{revision}",
        "candidate_id": candidate["candidate_id"] if candidate else "",
        "profile_id": profile_id,
        "revision": revision,
        "decision": decision,
        "reviewer_claim": reviewer_claim,
        "payload_digest": canonical_profile_digest(profile),
        "reviewed_at": now,
        "reason_code": reason,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def candidate_card(candidate: dict[str, object]) -> dict[str, object]:
    mappings = candidate.get("vocabulary_mappings")
    return {
        "schema_version": "domain_intelligence_review_card/v1",
        "candidate_id": candidate["candidate_id"],
        "status": candidate["status"],
        "profile_id": candidate["profile_id"],
        "scope": candidate.get("scope", {}),
        "domain_id": candidate["domain_id"],
        "mapping_count": len(mappings) if isinstance(mappings, list) else 0,
        "workflow_hints": candidate.get("workflow_hints", []),
        "confidence": candidate.get("confidence", {}),
        "provenance": candidate.get("provenance", {}),
        "base_profile_revision": candidate.get("base_profile_revision", 0),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def profile_projection(profile: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": DOMAIN_PROFILE_SCHEMA_VERSION,
        "profile_id": profile["profile_id"],
        "revision": profile["revision"],
        "status": profile["status"],
        "scope": profile.get("scope", {}),
        "domain_id": profile["domain_id"],
        "vocabulary_mappings": profile.get("vocabulary_mappings", []),
        "workflow_hints": profile.get("workflow_hints", []),
        "confidence": profile.get("confidence", {}),
        "provenance": profile.get("provenance", {}),
        "payload_digest": profile["payload_digest"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
