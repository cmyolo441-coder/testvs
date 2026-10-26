"""Closed scalar and persisted-identity validation for memory principals."""

from __future__ import annotations

import re
from typing import Any

from ._governance_safety import contains_credential_like_material

PRINCIPAL_CONTEXT_SCHEMA_VERSION = "memory_principal_context/v1"
MEMORY_IDENTITY_SCHEMA_VERSION = "omh_memory_identity/v1"
MEMORY_AUDIENCE_SCHEMA_VERSION = "omh_memory_audience/v1"
PRINCIPAL_REF_PATTERN = re.compile(r"^principal:v1:[0-9a-f]{64}$")
ACTOR_KINDS = frozenset({"human", "bot", "system", "ambiguous", "unknown"})
BINDING_STATES = frozenset({"validated_local", "host_validated", "unbound"})
PRINCIPAL_CONTEXT_KEYS = frozenset({
    "schema_version", "principal", "profile_ref", "surface_ref", "session_ref",
    "turn_ref", "actor_kind", "identity_evidence_refs", "binding_state",
})
_SAFE_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
_GENERATED_REVIEW_REF = re.compile(
    r"^(?:review_cand_[0-9a-f]{16}|review-mem_[0-9a-f]{16}-r[0-9]+|principal_assignment_[0-9a-f]{24})$"
)


def principal_ref(value: Any) -> bool:
    return isinstance(value, str) and PRINCIPAL_REF_PATTERN.fullmatch(value) is not None


def safe_ref(value: Any) -> bool:
    return isinstance(value, str) and _SAFE_REF.fullmatch(value) is not None


def valid_review_ref(value: Any, *, allow_empty: bool) -> bool:
    if not isinstance(value, str):
        return False
    if not value:
        return allow_empty
    return safe_ref(value) and (
        _GENERATED_REVIEW_REF.fullmatch(value) is not None
        or not contains_credential_like_material(value)
    )


def memory_identity_errors(value: Any) -> list[str]:
    """Validate the closed persisted identity and reviewed audience shape."""
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "subject_principal", "event_actor", "reviewer",
        "executor_perspective", "audience",
    }:
        return ["identity_shape"]
    actor, reviewer, audience = value.get("event_actor"), value.get("reviewer"), value.get("audience")
    if value.get("schema_version") != MEMORY_IDENTITY_SCHEMA_VERSION:
        return ["identity_schema"]
    if not isinstance(actor, dict) or set(actor) != {"principal", "actor_kind"} or actor.get("actor_kind") != "human" or not principal_ref(actor.get("principal")):
        return ["event_actor"]
    if not isinstance(reviewer, dict) or set(reviewer) != {"principal", "review_ref"} or (reviewer.get("principal") is not None and not principal_ref(reviewer.get("principal"))):
        return ["reviewer"]
    reviewer_ref = reviewer.get("review_ref")
    if not valid_review_ref(reviewer_ref, allow_empty=False):
        return ["reviewer_review_ref"]
    if not isinstance(audience, dict) or set(audience) != {"schema_version", "kind", "principal_refs", "review_ref"}:
        return ["audience"]
    refs = audience.get("principal_refs")
    audience_ref = audience.get("review_ref")
    audience_kind = audience.get("kind")
    if audience.get("schema_version") != MEMORY_AUDIENCE_SCHEMA_VERSION or not isinstance(refs, list) or not refs or len(refs) > 64 or any(not principal_ref(item) for item in refs):
        return ["audience"]
    if not isinstance(audience_kind, str) or audience_kind not in {"subject_only", "explicit_principals"}:
        return ["identity_values"]
    if not valid_review_ref(audience_ref, allow_empty=False):
        return ["audience_review_ref"]
    if audience_ref != reviewer_ref:
        return ["review_ref_mismatch"]
    subject = value.get("subject_principal")
    if audience_kind == "subject_only" and (not principal_ref(subject) or refs != [subject]):
        return ["subject_audience"]
    if audience_kind == "explicit_principals" and subject is not None:
        return ["shared_subject"]
    if not safe_ref(value.get("executor_perspective")):
        return ["identity_values"]
    return []
