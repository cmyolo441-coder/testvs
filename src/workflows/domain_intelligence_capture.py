from __future__ import annotations

import os

from ..paths import OmhPaths
from ..system.local_store import utc_now
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_CANDIDATE_SCHEMA_VERSION,
    REDUCTION_POLICY,
    ensure_no_forbidden_keys,
    normalize_confidence,
    normalize_identifier,
    normalize_mappings,
    normalize_provenance,
    normalize_scope,
    normalize_workflow_hints,
    stable_profile_id,
)
from .domain_intelligence_store import (
    domain_store_lock,
    ensure_candidate_capacity,
    write_candidate,
)
from .domain_intelligence_validation import current_profile_revision


def capture_domain_candidate(
    paths: OmhPaths,
    *,
    scope_kind: str,
    scope_ref: str,
    domain_id: str,
    mappings: list[tuple[str, str]],
    workflow_hints: list[str] | None = None,
    source_class: str = "operator_supplied",
    source_ref: str = "",
    observation_count: int = 1,
    confidence: float = 0.5,
) -> dict[str, object]:
    ensure_no_forbidden_keys(locals())
    scope = normalize_scope(scope_kind, scope_ref)
    normalized_domain = normalize_identifier(domain_id, "domain_id")
    normalized_mappings = normalize_mappings(mappings)
    normalized_hints = normalize_workflow_hints(workflow_hints or [])
    provenance = normalize_provenance(source_class, source_ref, observation_count)
    confidence_metadata = normalize_confidence(confidence, observation_count)
    created_at = utc_now()
    profile_id = stable_profile_id(scope, normalized_domain)
    candidate_id = "dicand_" + os.urandom(8).hex()
    with domain_store_lock(paths):
        ensure_candidate_capacity(paths)
        candidate = {
            "schema_version": DOMAIN_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "status": "pending_review",
            "profile_id": profile_id,
            "scope": scope,
            "domain_id": normalized_domain,
            "vocabulary_mappings": normalized_mappings,
            "workflow_hints": normalized_hints,
            "confidence": confidence_metadata,
            "provenance": provenance,
            "base_profile_revision": current_profile_revision(paths, profile_id),
            "created_at": created_at,
            "updated_at": created_at,
            "redaction_policy": REDUCTION_POLICY,
            "claim_boundary": CLAIM_BOUNDARY,
        }
        write_candidate(paths, candidate_id, candidate)
    return {
        "schema_version": "domain_intelligence_capture/v1",
        "candidate": candidate,
        "claim_boundary": CLAIM_BOUNDARY,
    }
