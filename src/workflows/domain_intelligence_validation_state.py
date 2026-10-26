from __future__ import annotations

from dataclasses import dataclass

from .domain_intelligence_contracts import (
    SAFE_PROFILE_ID,
    ensure_no_forbidden_keys,
    normalize_confidence_from_value,
    normalize_identifier,
    normalize_provenance_from_value,
    normalize_scope_from_value,
    stable_profile_id,
)
from .domain_intelligence_schema import validate_profile_contract


@dataclass
class ProfileValidationFrame:
    profile: dict[str, object]
    is_root: bool
    complete: bool = False
    review: dict[str, object] | None = None
    predecessor: dict[str, object] | None = None


def validate_profile_identity(
    profile: dict[str, object],
) -> tuple[str, tuple[str, int]]:
    ensure_no_forbidden_keys(profile)
    status = validate_profile_contract(profile)
    revision = profile.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("invalid_revision")
    scope = normalize_scope_from_value(profile.get("scope"))
    domain_id = normalize_identifier(profile.get("domain_id"), "domain_id")
    if profile.get("domain_id") != domain_id:
        raise ValueError("profile_domain_id_not_canonical")
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not SAFE_PROFILE_ID.fullmatch(profile_id):
        raise ValueError("unsafe_profile_id")
    if profile_id != stable_profile_id(scope, domain_id):
        raise ValueError("profile_identity_mismatch")
    return status, (profile_id, revision)


def profile_key(profile: dict[str, object]) -> tuple[str, int]:
    revision = profile.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("invalid_revision")
    profile_id = profile.get("profile_id")
    if not isinstance(profile_id, str) or not SAFE_PROFILE_ID.fullmatch(profile_id):
        raise ValueError("unsafe_profile_id")
    return profile_id, revision


def canonical_confidence(value: object) -> dict[str, object]:
    normalized = normalize_confidence_from_value(value)
    if value != normalized:
        raise ValueError("confidence_not_canonical")
    return normalized


def canonical_provenance(value: object) -> dict[str, object]:
    normalized = normalize_provenance_from_value(value)
    if value != normalized:
        raise ValueError("provenance_not_canonical")
    return normalized
