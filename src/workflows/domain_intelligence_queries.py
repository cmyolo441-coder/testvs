from __future__ import annotations

from pathlib import Path

from ..system.paths import OmhPaths
from .domain_intelligence_artifacts import candidate_card, profile_projection
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_LIST_SCHEMA_VERSION,
    DOMAIN_REVIEW_QUEUE_SCHEMA_VERSION,
    DOMAIN_STATUS_SCHEMA_VERSION,
    normalize_identifier,
    normalize_scope,
    normalize_scope_from_value,
)
from .domain_intelligence_review_validation import validate_review_artifact_for_status
from .domain_intelligence_store import (
    diagnostic,
    domain_root,
    read_candidates,
    read_history_profiles,
    read_profiles,
    read_reviews,
    store_lock_target,
)
from .domain_intelligence_validation import (
    build_profile_validation_context,
    validate_candidate_artifact,
    validate_profile_artifact,
)
from .domain_intelligence_validation_state import profile_key
from .project_terms_capture import project_terms_source_freshness


def build_domain_review(
    paths: OmhPaths,
    *,
    candidate_id: str | None = None,
    limit: int = 20,
    source_freshness_root: str | Path | None = None,
) -> dict[str, object]:
    diagnostics: list[dict[str, str]] = []
    cards: list[dict[str, object]] = []
    for candidate, path in read_candidates(paths, diagnostics):
        try:
            validate_candidate_artifact(candidate)
        except ValueError as exc:
            diagnostics.append(diagnostic(path, str(exc)))
            continue
        if candidate_id and candidate.get("candidate_id") != candidate_id:
            continue
        if candidate.get("status") != "pending_review":
            continue
        card = candidate_card(candidate)
        if source_freshness_root is not None:
            card["source_freshness"] = project_terms_source_freshness(
                candidate,
                invocation_cwd=source_freshness_root,
            )
        cards.append(card)
        if len(cards) >= max(1, limit):
            break
    return {
        "schema_version": DOMAIN_REVIEW_QUEUE_SCHEMA_VERSION,
        "cards": cards,
        "counts": {
            "pending_review": len(cards),
            "malformed_artifacts": len(diagnostics),
        },
        "diagnostics": diagnostics[:20],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def list_domain_profiles(
    paths: OmhPaths,
    *,
    scope_kind: str | None = None,
    scope_ref: str | None = None,
    domain_id: str | None = None,
    include_retired: bool = False,
    source_freshness_root: str | Path | None = None,
) -> dict[str, object]:
    scope_filter = (
        normalize_scope(scope_kind, scope_ref) if scope_kind or scope_ref else None
    )
    domain_filter = normalize_identifier(domain_id, "domain_id") if domain_id else None
    diagnostics: list[dict[str, str]] = []
    profiles: list[dict[str, object]] = []
    context = build_profile_validation_context(paths)
    for profile, path in read_profiles(paths, diagnostics):
        try:
            validate_profile_artifact(paths, profile, context=context)
        except ValueError as exc:
            diagnostics.append(diagnostic(path, str(exc)))
            continue
        if profile.get("status") != "active" and not include_retired:
            continue
        if scope_filter and profile.get("scope") != scope_filter:
            continue
        if domain_filter and profile.get("domain_id") != domain_filter:
            continue
        projection = profile_projection(profile)
        if source_freshness_root is not None:
            projection["source_freshness"] = project_terms_source_freshness(
                profile,
                invocation_cwd=source_freshness_root,
            )
        profiles.append(projection)
    profiles.sort(key=_profile_sort_key)
    return {
        "schema_version": DOMAIN_LIST_SCHEMA_VERSION,
        "profiles": profiles,
        "counts": {
            "profiles": len(profiles),
            "malformed_artifacts": len(diagnostics),
        },
        "diagnostics": diagnostics[:20],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_domain_status(paths: OmhPaths) -> dict[str, object]:
    diagnostics: list[dict[str, str]] = []
    candidates = read_candidates(paths, diagnostics)
    profiles = read_profiles(paths, diagnostics)
    reviews = read_reviews(paths, diagnostics)
    history = read_history_profiles(paths, diagnostics)
    context = build_profile_validation_context(
        paths,
        history=history,
        candidates=candidates,
        reviews=reviews,
    )
    active = 0
    retired = 0
    valid_profiles: list[dict[str, object]] = []
    for profile, path in profiles:
        try:
            validate_profile_artifact(paths, profile, context=context)
        except ValueError as exc:
            diagnostics.append(diagnostic(path, str(exc)))
            continue
        valid_profiles.append(profile)
        if profile.get("status") == "active":
            active += 1
        elif profile.get("status") == "retired":
            retired += 1
    for profile, path in history:
        try:
            validate_profile_artifact(paths, profile, context=context)
        except ValueError as exc:
            diagnostics.append(diagnostic(path, str(exc)))
            continue
        valid_profiles.append(profile)
    valid_candidates: list[dict[str, object]] = []
    for candidate, path in candidates:
        try:
            validate_candidate_artifact(candidate)
        except ValueError as exc:
            diagnostics.append(diagnostic(path, str(exc)))
            continue
        valid_candidates.append(candidate)
    pending = sum(
        1
        for candidate in valid_candidates
        if candidate.get("status") == "pending_review"
    )
    rejected = sum(
        1 for candidate in valid_candidates if candidate.get("status") == "rejected"
    )
    approved = sum(
        1 for candidate in valid_candidates if candidate.get("status") == "approved"
    )
    candidate_index = {
        str(candidate["candidate_id"]): candidate for candidate in valid_candidates
    }
    profile_index = {
        profile_key(profile): profile
        for profile in valid_profiles
    }
    valid_reviews = 0
    for review, path in reviews:
        try:
            validate_review_artifact_for_status(
                review,
                candidates=candidate_index,
                profiles=profile_index,
            )
        except ValueError as exc:
            diagnostics.append(diagnostic(path, str(exc)))
            continue
        valid_reviews += 1
    lock_target = store_lock_target(paths)
    return {
        "schema_version": DOMAIN_STATUS_SCHEMA_VERSION,
        "store_root": str(domain_root(paths)),
        "lock_target": str(lock_target),
        "lock_file": str(lock_target.with_name(".store.lock")),
        "counts": {
            "candidates": len(valid_candidates),
            "pending_review": pending,
            "approved_candidates": approved,
            "rejected_candidates": rejected,
            "active_profiles": active,
            "retired_profiles": retired,
            "reviews": valid_reviews,
            "malformed_artifacts": len(diagnostics),
        },
        "diagnostics": diagnostics[:20],
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _profile_sort_key(profile: dict[str, object]) -> tuple[str, str, str]:
    scope = normalize_scope_from_value(profile.get("scope"))
    domain_id = normalize_identifier(profile.get("domain_id"), "domain_id")
    kind = scope.get("kind")
    ref = scope.get("ref")
    if not isinstance(kind, str) or not isinstance(ref, str):
        raise ValueError("profile_scope_not_canonical")
    return kind, ref, domain_id
