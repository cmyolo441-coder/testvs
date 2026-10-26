from __future__ import annotations

import json
from pathlib import Path

from ..paths import OmhPaths
from ..plugin_bundle.omh.project_identity import resolve_project_identity
from .domain_intelligence_lineage import ProfileValidationContext
from .domain_intelligence_store import (
    domain_store_lock,
    read_candidates,
    read_history_profiles,
    read_profiles,
    read_reviews,
)
from .domain_intelligence_validation import validate_profile_artifact


DOMAIN_PROFILE_SOURCE_KIND = "domain_intelligence_profile"
DOMAIN_PROFILE_SUMMARY_MAX_BYTES = 16_384


def build_domain_handoff_projection(
    paths: OmhPaths,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Project validated active domain profiles into handoff-ready metadata.

    This is a read-only view of the existing domain-intelligence store. It does
    not create a store, repair an artifact, or retain source-file bytes.
    """
    project_root = paths.omh_home.parent
    store = paths.memory_dir / "domain-intelligence"
    required = (store / ".store.lock", store / "candidates", store / "profiles", store / "reviews", store / "history")
    if resolve_project_identity(project_root).state != "resolved" or not store.exists():
        return [], []
    if not all(path.exists() for path in required):
        return [], [_exclusion("domain-profile-store", "domain_profile_store_unhealthy")]

    try:
        with domain_store_lock(paths):
            diagnostics: list[dict[str, str]] = []
            candidates = read_candidates(paths, diagnostics)
            profiles = read_profiles(paths, diagnostics)
            reviews = read_reviews(paths, diagnostics)
            history = read_history_profiles(paths, diagnostics)
            return _project_records(
                paths,
                project_root=project_root,
                candidates=candidates,
                profiles=profiles,
                reviews=reviews,
                history=history,
                diagnostics=diagnostics,
            )
    except (OSError, TypeError, ValueError):
        return [], [_exclusion("domain-profile-store", "domain_profile_store_unhealthy")]


def _project_records(
    paths: OmhPaths,
    *,
    project_root: Path,
    candidates: list[tuple[dict[str, object], Path]],
    profiles: list[tuple[dict[str, object], Path]],
    reviews: list[tuple[dict[str, object], Path]],
    history: list[tuple[dict[str, object], Path]],
    diagnostics: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate_values = {str(value.get("candidate_id", "")): value for value, _path in candidates}
    review_values = {str(value.get("review_id", "")): value for value, _path in reviews}
    context = ProfileValidationContext(
        history={
            (str(value.get("profile_id", "")), int(value.get("revision", 0) or 0)): value
            for value, _path in history
        },
        candidates=candidate_values,
        reviews=review_values,
    )
    expected_scope = {"kind": "project", "ref": resolve_project_identity(project_root).identity}
    included: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []

    for diagnostic in diagnostics:
        excluded.append(
            _exclusion(
                Path(str(diagnostic.get("path_name", "artifact"))).stem,
                "domain_profile_malformed",
            )
        )

    for candidate, _path in candidates:
        status = candidate.get("status")
        profile_id = str(candidate.get("profile_id", "")) or str(candidate.get("candidate_id", "domain-profile"))
        if status == "pending_review":
            excluded.append(_exclusion(profile_id, "domain_profile_pending_review"))
        elif status == "rejected":
            excluded.append(_exclusion(profile_id, "domain_profile_rejected"))

    for profile, _path in profiles:
        profile_id = str(profile.get("profile_id", "")) or "domain-profile"
        revision = profile.get("revision")
        item_id = f"{profile_id}:r{revision}" if isinstance(revision, int) else profile_id
        status = profile.get("status")
        if status == "retired":
            excluded.append(_exclusion(item_id, "domain_profile_retired"))
            continue
        if status != "active":
            excluded.append(_exclusion(item_id, "domain_profile_malformed"))
            continue
        scope = profile.get("scope")
        if not isinstance(scope, dict) or {"kind": scope.get("kind"), "ref": scope.get("ref")} != expected_scope:
            excluded.append(_exclusion(item_id, "domain_profile_scope_mismatch"))
            continue
        try:
            validate_profile_artifact(paths, profile, context=context)
        except (OSError, TypeError, ValueError) as exc:
            excluded.append(_exclusion(item_id, _validation_reason(str(exc))))
            continue
        review_id = f"direview_{profile_id}_r{revision}"
        review = review_values.get(review_id)
        if review is None:
            excluded.append(_exclusion(item_id, "domain_profile_review_mismatch"))
            continue
        summary = _profile_summary(profile)
        if len(summary.encode("utf-8")) > DOMAIN_PROFILE_SUMMARY_MAX_BYTES:
            excluded.append(_exclusion(item_id, "domain_profile_over_budget"))
            continue
        included.append(
            {
                "item_id": item_id,
                "key": str(profile.get("domain_id", "")),
                "summary": summary,
                "source": "omh_memory",
                "source_kind": DOMAIN_PROFILE_SOURCE_KIND,
                "truth_level": "approved_context",
                "scope": expected_scope,
                "profile_id": profile_id,
                "profile_revision": int(revision),
                "profile_digest": str(profile.get("payload_digest", "")),
                "review_id": review_id,
                "replay_evaluation": {
                    "schema_version": "omh_memory_replay_evaluation/v1",
                    "artifact_identity": {
                        "kind": DOMAIN_PROFILE_SOURCE_KIND,
                        "profile_id": profile_id,
                    },
                    "revision": int(revision),
                    "admission_mode": "approved_manual",
                    "source_class": "omh_local",
                    "retention_class": "durable",
                    "evaluated_at": str(profile.get("approved_at", "")),
                    "eligible": True,
                    "reason_code": "eligible",
                    "revalidation_evidence": {"review_id": review_id},
                },
            }
        )
    return included, _deduplicated_exclusions(excluded)


def _profile_summary(profile: dict[str, object]) -> str:
    mappings = profile.get("vocabulary_mappings")
    mapping_text = ", ".join(
        f"{str(value.get('phrase', ''))}={str(value.get('canonical', ''))}"
        for value in mappings if isinstance(value, dict)
    ) if isinstance(mappings, list) else ""
    hints = profile.get("workflow_hints")
    hint_text = ", ".join(str(value) for value in hints) if isinstance(hints, list) else ""
    return (
        f"domain {profile.get('domain_id', '')}; mappings: {mapping_text}; "
        f"workflow hints: {hint_text or 'none'}"
    )


def _validation_reason(reason: str) -> str:
    if "payload_digest_mismatch" in reason:
        return "domain_profile_digest_mismatch"
    if "review" in reason or "approved_candidate_lineage_required" in reason:
        return "domain_profile_review_mismatch"
    return "domain_profile_malformed"


def _exclusion(item_id: str, reason: str) -> dict[str, object]:
    return {"item_id": item_id, "source": "omh_memory", "reason": reason}


def _deduplicated_exclusions(values: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            result.append(value)
            seen.add(key)
    return result
