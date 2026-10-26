from __future__ import annotations

from ..paths import OmhPaths
from ..system.local_store import utc_now
from . import domain_intelligence_operation_store as journal
from . import domain_intelligence_store as store
from .domain_intelligence_contracts import (
    CLAIM_BOUNDARY,
    DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
    SAFE_PROFILE_ID,
    canonical_profile_digest,
)
from .domain_intelligence_schema import (
    PROFILE_REVIEW_KEYS,
    validate_profile_contract,
    validate_review_contract,
)
from .domain_intelligence_validation import validate_profile_artifact


RETIREMENT_OPERATION_SCHEMA_VERSION = "domain_intelligence_retirement_operation/v1"
_OPERATION_KEYS = frozenset(
    "schema_version operation_id profile_id prior_profile target_review target_profile "
    "claim_boundary operation_digest".split()
)


def build_retirement_operation(
    profile: dict[str, object], *, reviewer_claim: str, reason_code: str
) -> dict[str, object]:
    now = utc_now()
    target = {
        **profile,
        "revision": int(profile["revision"]) + 1,
        "status": "retired",
        "base_profile_revision": int(profile["revision"]),
        "updated_at": now,
        "retired_at": now,
        "retired_by": reviewer_claim,
        "retirement_reason_code": reason_code,
    }
    target["payload_digest"] = canonical_profile_digest(target)
    review = {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "review_id": f"direview_{profile['profile_id']}_r{target['revision']}",
        "candidate_id": "",
        "profile_id": profile["profile_id"],
        "revision": target["revision"],
        "decision": "retired",
        "reviewer_claim": reviewer_claim,
        "payload_digest": target["payload_digest"],
        "reviewed_at": now,
        "reason_code": reason_code,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    operation = journal.seal_operation(
        {
            "schema_version": RETIREMENT_OPERATION_SCHEMA_VERSION,
            "operation_id": _operation_id(str(profile["profile_id"])),
            "profile_id": profile["profile_id"],
            "prior_profile": profile,
            "target_review": review,
            "target_profile": target,
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    validate_retirement_operation(None, operation)
    return operation


def load_retirement_operation(
    paths: OmhPaths, profile_id: str
) -> dict[str, object] | None:
    return journal.load_operation(
        paths, _operation_id(profile_id), validate_retirement_operation
    )


def write_retirement_operation(paths: OmhPaths, operation: dict[str, object]) -> None:
    journal.write_operation(paths, operation, validate_retirement_operation)


def validate_retirement_resume_state(
    paths: OmhPaths, operation: dict[str, object]
) -> None:
    validate_retirement_operation(paths, operation)
    profile_id = str(operation["profile_id"])
    prior = operation["prior_profile"]
    target = operation["target_profile"]
    journal.require_expected_or_target(
        store.profile_path(paths, profile_id), prior, target, label="retirement_profile"
    )
    journal.require_absent_or_exact(
        store.history_path(paths, profile_id, int(prior["revision"])),
        prior,
        label="retirement_history",
    )
    review = operation["target_review"]
    journal.require_absent_or_exact(
        store.review_path(paths, str(review["review_id"])),
        review,
        label="retirement_review",
    )


def write_retirement_archive_idempotent(
    paths: OmhPaths, operation: dict[str, object]
) -> None:
    prior = operation["prior_profile"]
    journal.write_absent_or_exact(
        paths,
        store.history_path(paths, str(operation["profile_id"]), int(prior["revision"])),
        prior,
        label="retirement_history",
    )


def write_retirement_review_idempotent(
    paths: OmhPaths, operation: dict[str, object]
) -> None:
    review = operation["target_review"]
    journal.write_absent_or_exact(
        paths,
        store.review_path(paths, str(review["review_id"])),
        review,
        label="retirement_review",
    )


def write_retirement_profile_resumable(
    paths: OmhPaths, operation: dict[str, object]
) -> None:
    journal.write_expected_or_target(
        paths,
        store.profile_path(paths, str(operation["profile_id"])),
        operation["prior_profile"],
        operation["target_profile"],
        label="retirement_profile",
    )


def delete_retirement_operation(paths: OmhPaths, profile_id: str) -> None:
    journal.delete_operation(
        paths, _operation_id(profile_id), validate_retirement_operation
    )


def validate_retirement_operation(
    paths: OmhPaths | None, operation: dict[str, object]
) -> None:
    profile_id = operation.get("profile_id")
    if not isinstance(profile_id, str) or not SAFE_PROFILE_ID.fullmatch(profile_id):
        raise ValueError("unsafe_profile_id")
    journal.validate_operation_envelope(
        operation, operation_id=_operation_id(profile_id)
    )
    if (
        set(operation) != _OPERATION_KEYS
        or operation.get("schema_version") != RETIREMENT_OPERATION_SCHEMA_VERSION
    ):
        raise ValueError("retirement_operation_schema_mismatch")
    prior = operation.get("prior_profile")
    target = operation.get("target_profile")
    review = operation.get("target_review")
    if not all(isinstance(item, dict) for item in (prior, target, review)):
        raise ValueError("retirement_operation_artifact_type")
    validate_profile_contract(prior)
    validate_profile_contract(target)
    validate_review_contract(review, PROFILE_REVIEW_KEYS)
    if prior.get("status") != "active" or prior.get(
        "payload_digest"
    ) != canonical_profile_digest(prior):
        raise ValueError("retirement_operation_prior_profile")
    expected_target = {
        **prior,
        "revision": int(prior["revision"]) + 1,
        "status": "retired",
        "base_profile_revision": prior["revision"],
        "updated_at": review.get("reviewed_at"),
        "retired_at": review.get("reviewed_at"),
        "retired_by": review.get("reviewer_claim"),
        "retirement_reason_code": review.get("reason_code"),
    }
    expected_target["payload_digest"] = canonical_profile_digest(expected_target)
    if target != expected_target:
        raise ValueError("retirement_operation_transition_mismatch")
    expected_review = {
        "schema_version": DOMAIN_REVIEW_RECORD_SCHEMA_VERSION,
        "review_id": f"direview_{profile_id}_r{target['revision']}",
        "candidate_id": "",
        "profile_id": profile_id,
        "revision": target["revision"],
        "decision": "retired",
        "reviewer_claim": target["retired_by"],
        "payload_digest": target["payload_digest"],
        "reviewed_at": target["retired_at"],
        "reason_code": target["retirement_reason_code"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if review != expected_review:
        raise ValueError("retirement_operation_review_mismatch")
    if paths is not None:
        validate_profile_artifact(paths, prior)


def _operation_id(profile_id: str) -> str:
    if not SAFE_PROFILE_ID.fullmatch(profile_id):
        raise ValueError("unsafe_profile_id")
    return f"retire_{profile_id}"
