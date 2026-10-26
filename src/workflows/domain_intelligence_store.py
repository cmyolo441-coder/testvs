from __future__ import annotations

from pathlib import Path

from ..paths import OmhPaths
from .domain_intelligence_contracts import SAFE_REF
from .domain_intelligence_store_security import (
    MAX_DOMAIN_ARTIFACT_BYTES,
    MAX_DOMAIN_ARTIFACT_FILES,
    MAX_DOMAIN_CANDIDATE_FILES,
    MAX_DOMAIN_JSON_DEPTH,
    MAX_DOMAIN_JSON_NODES,
    atomic_write_managed_json,
    bounded_json_paths,
    domain_store_lock,
    ensure_new_artifact_capacity,
    secure_artifact_path,
    secure_domain_root,
    secure_managed_dir,
    secure_store_lock_target,
)
from .domain_intelligence_store_resolution import (
    diagnostic,
    read_history_artifacts,
    read_identity_artifacts,
    resolve_authoritative_artifact,
)

__all__ = (
    "MAX_DOMAIN_ARTIFACT_BYTES",
    "MAX_DOMAIN_ARTIFACT_FILES",
    "MAX_DOMAIN_CANDIDATE_FILES",
    "MAX_DOMAIN_JSON_DEPTH",
    "MAX_DOMAIN_JSON_NODES",
    "atomic_write_managed_json",
    "diagnostic",
    "domain_store_lock",
    "ensure_candidate_capacity",
    "read_history_profiles",
)


def read_candidate_or_raise(paths: OmhPaths, candidate_id: str) -> dict[str, object]:
    if not SAFE_REF.match(candidate_id):
        raise ValueError("unsafe_candidate_id")
    with domain_store_lock(paths):
        _recover_project_terms_candidate_batches(paths)
        candidate, error = resolve_authoritative_artifact(
            candidates_dir(paths),
            candidate_id,
            "candidate_id",
            file_limit=MAX_DOMAIN_ARTIFACT_FILES,
        )
        if error:
            raise ValueError(error)
        if candidate is None:
            raise FileNotFoundError(candidate_id)
        return candidate


def read_profile(paths: OmhPaths, profile_id: str) -> dict[str, object] | None:
    if not SAFE_REF.match(profile_id):
        raise ValueError("unsafe_profile_id")
    profile, error = resolve_authoritative_artifact(
        profiles_dir(paths),
        profile_id,
        "profile_id",
        file_limit=MAX_DOMAIN_ARTIFACT_FILES,
    )
    if error:
        raise ValueError(error)
    return profile


def read_review(paths: OmhPaths, review_id: str) -> tuple[dict[str, object] | None, str | None]:
    if not SAFE_REF.match(review_id):
        return None, "unsafe_review_id"
    try:
        return resolve_authoritative_artifact(
            reviews_dir(paths),
            review_id,
            "review_id",
            file_limit=MAX_DOMAIN_ARTIFACT_FILES,
        )
    except (OSError, ValueError) as exc:
        return None, str(exc)


def archive_profile(paths: OmhPaths, profile: dict[str, object]) -> None:
    profile_id = str(profile.get("profile_id", ""))
    revision = int(profile.get("revision", 0))
    if not SAFE_REF.match(profile_id) or revision < 1:
        raise ValueError("invalid_profile_history_identity")
    target = history_path(paths, profile_id, revision)
    ensure_new_artifact_capacity(
        history_dir(paths),
        target,
        limit=MAX_DOMAIN_ARTIFACT_FILES,
        reason="artifact_capacity_exceeded",
    )
    atomic_write_managed_json(paths, "history", target.name, profile)


def write_candidate(paths: OmhPaths, candidate_id: str, candidate: dict[str, object]) -> None:
    target = candidate_path(paths, candidate_id)
    ensure_new_artifact_capacity(
        candidates_dir(paths),
        target,
        limit=MAX_DOMAIN_CANDIDATE_FILES,
        reason="candidate_capacity_exceeded",
    )
    atomic_write_managed_json(paths, "candidates", target.name, candidate)


def write_profile(paths: OmhPaths, profile_id: str, profile: dict[str, object]) -> None:
    target = profile_path(paths, profile_id)
    ensure_new_artifact_capacity(
        profiles_dir(paths),
        target,
        limit=MAX_DOMAIN_ARTIFACT_FILES,
        reason="artifact_capacity_exceeded",
    )
    atomic_write_managed_json(paths, "profiles", target.name, profile)


def write_review(paths: OmhPaths, review_id: str, review: dict[str, object]) -> None:
    target = review_path(paths, review_id)
    ensure_new_artifact_capacity(
        reviews_dir(paths),
        target,
        limit=MAX_DOMAIN_ARTIFACT_FILES,
        reason="artifact_capacity_exceeded",
    )
    atomic_write_managed_json(paths, "reviews", target.name, review)


def read_candidates(paths: OmhPaths, diagnostics: list[dict[str, str]]) -> list[tuple[dict[str, object], Path]]:
    with domain_store_lock(paths):
        _recover_project_terms_candidate_batches(paths)
        return read_identity_artifacts(
            candidates_dir(paths),
            diagnostics,
            "candidate_id",
            file_limit=MAX_DOMAIN_ARTIFACT_FILES,
            capacity_limit=MAX_DOMAIN_CANDIDATE_FILES,
        )


def _recover_project_terms_candidate_batches(paths: OmhPaths) -> None:
    # Delayed to avoid a module cycle: project-terms staging uses this store.
    from .project_terms_capture import _recover_project_terms_candidate_batches_locked

    _recover_project_terms_candidate_batches_locked(paths)


def read_profiles(paths: OmhPaths, diagnostics: list[dict[str, str]]) -> list[tuple[dict[str, object], Path]]:
    return read_identity_artifacts(
        profiles_dir(paths), diagnostics, "profile_id", file_limit=MAX_DOMAIN_ARTIFACT_FILES
    )


def read_reviews(paths: OmhPaths, diagnostics: list[dict[str, str]]) -> list[tuple[dict[str, object], Path]]:
    return read_identity_artifacts(reviews_dir(paths), diagnostics, "review_id", file_limit=MAX_DOMAIN_ARTIFACT_FILES)


def ensure_candidate_capacity(paths: OmhPaths, *, required: int = 1) -> None:
    if isinstance(required, bool) or not isinstance(required, int) or required < 1:
        raise ValueError("invalid_candidate_capacity_requirement")
    if required > MAX_DOMAIN_CANDIDATE_FILES:
        raise ValueError("candidate_capacity_exceeded")
    directory = candidates_dir(paths)
    existing, overflow = bounded_json_paths(
        directory,
        limit=MAX_DOMAIN_CANDIDATE_FILES - required,
    )
    if overflow or len(existing) + required > MAX_DOMAIN_CANDIDATE_FILES:
        raise ValueError("candidate_capacity_exceeded")


def read_history_profiles(
    paths: OmhPaths,
    diagnostics: list[dict[str, str]],
) -> list[tuple[dict[str, object], Path]]:
    return read_history_artifacts(history_dir(paths), diagnostics, file_limit=MAX_DOMAIN_ARTIFACT_FILES)


def domain_root(paths: OmhPaths) -> Path:
    return secure_domain_root(paths)


def store_lock_target(paths: OmhPaths) -> Path:
    return secure_store_lock_target(paths)


def candidates_dir(paths: OmhPaths) -> Path:
    return secure_managed_dir(paths, "candidates")


def profiles_dir(paths: OmhPaths) -> Path:
    return secure_managed_dir(paths, "profiles")


def reviews_dir(paths: OmhPaths) -> Path:
    return secure_managed_dir(paths, "reviews")


def history_dir(paths: OmhPaths) -> Path:
    return secure_managed_dir(paths, "history")


def candidate_path(paths: OmhPaths, candidate_id: str) -> Path:
    if not SAFE_REF.match(candidate_id):
        raise ValueError("unsafe_candidate_id")
    return secure_artifact_path(candidates_dir(paths), f"{candidate_id}.json")


def profile_path(paths: OmhPaths, profile_id: str) -> Path:
    if not SAFE_REF.match(profile_id):
        raise ValueError("unsafe_profile_id")
    return secure_artifact_path(profiles_dir(paths), f"{profile_id}.json")


def review_path(paths: OmhPaths, review_id: str) -> Path:
    if not SAFE_REF.match(review_id):
        raise ValueError("unsafe_review_id")
    return secure_artifact_path(reviews_dir(paths), f"{review_id}.json")


def history_path(paths: OmhPaths, profile_id: str, revision: int) -> Path:
    if not SAFE_REF.match(profile_id) or revision < 1:
        raise ValueError("unsafe_history_id")
    return secure_artifact_path(history_dir(paths), f"{profile_id}_r{revision}.json")
