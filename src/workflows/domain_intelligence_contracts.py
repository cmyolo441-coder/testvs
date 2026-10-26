from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata

from .domain_intelligence_admission import (
    ensure_safe_identifier_content,
    ensure_safe_opaque_ref_content,
    normalize_mappings,  # noqa: F401 - public contract re-export
    normalize_mappings_from_value,
    normalize_workflow_hints,
)


DOMAIN_CANDIDATE_SCHEMA_VERSION = "domain_intelligence_candidate/v1"
DOMAIN_PROFILE_SCHEMA_VERSION = "domain_intelligence_profile/v1"
DOMAIN_REVIEW_RECORD_SCHEMA_VERSION = "domain_intelligence_review_record/v1"
DOMAIN_STATUS_SCHEMA_VERSION = "domain_intelligence_status/v1"
DOMAIN_REVIEW_QUEUE_SCHEMA_VERSION = "domain_intelligence_review_queue/v1"
DOMAIN_LIST_SCHEMA_VERSION = "domain_intelligence_profile_listing/v1"

ALLOWED_SCOPE_KINDS = {"user", "organization", "project"}
ALLOWED_SOURCE_CLASSES = {"operator_supplied", "wrapper_supplied", "omh_local"}
ALLOWED_REVIEW_REASON_CODES = {
    "duplicate",
    "incorrect_scope",
    "insufficient_evidence",
    "operator_request",
    "scope_error",
    "superseded",
}
CLAIM_BOUNDARY = (
    "Domain intelligence is reviewed OMH-local prepared context only. It is a future "
    "routing_prior_not_override and is not routing behavior, execution, review, CI, "
    "merge, authentication, or Hermes internal-memory evidence."
)
REDUCTION_POLICY = "bounded_reviewed_vocabulary_only_no_raw_prompts_or_transcripts"
DEFAULT_REVIEW_REASON_CODE = "operator_request"

SAFE_REF = re.compile(r"^[A-Za-z0-9_.:-]{1,120}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
SAFE_CANDIDATE_ID = re.compile(r"^dicand_[a-f0-9]{16}$")
SAFE_PROFILE_ID = re.compile(r"^dprof_[a-f0-9]{24}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,79}$")
_PROMPTISH_KEYS = {"message", "prompt", "raw", "text", "body", "content", "transcript", "hidden_reasoning"}
_FORBIDDEN_PAYLOAD_KEYS = _PROMPTISH_KEYS | {"messages", "conversation", "log", "logs"}


def stable_profile_id(scope: dict[str, object], domain_id: str) -> str:
    normalized_scope = normalize_scope_from_value(scope)
    payload = {
        "scope": {
            "kind": normalized_scope["kind"],
            "ref": normalized_scope["ref"],
            "ref_authority": "operator_or_wrapper_supplied",
        },
        "domain_id": normalize_identifier(domain_id, "domain_id"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "dprof_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def canonical_profile_digest(profile: dict[str, object]) -> str:
    raw = json.dumps(_digest_payload(profile), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_scope_from_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("scope must be an object")
    if set(value) != {"kind", "ref", "ref_authority", "identity_claim"}:
        raise ValueError("scope_schema_mismatch")
    return normalize_scope(value.get("kind"), value.get("ref"))


def normalize_scope(kind: str | None, ref: str | None) -> dict[str, object]:
    if not isinstance(kind, str):
        raise ValueError("invalid_scope_kind")
    if not isinstance(ref, str):
        raise ValueError("unsafe_scope_ref")
    normalized_kind = kind.strip().lower()
    if normalized_kind not in ALLOWED_SCOPE_KINDS:
        raise ValueError("invalid_scope_kind")
    ensure_safe_opaque_ref_content(ref, "scope_ref")
    normalized_ref = ref.strip()
    if not SAFE_REF.match(normalized_ref):
        raise ValueError("unsafe_scope_ref")
    return {
        "kind": normalized_kind,
        "ref": normalized_ref,
        "ref_authority": "operator_or_wrapper_supplied",
        "identity_claim": "not_authenticated_identity_evidence",
    }


def normalize_identifier(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid_{label}")
    ensure_safe_identifier_content(value, label)
    normalized = unicodedata.normalize("NFKC", value.strip().lower())
    if not _IDENTIFIER.match(normalized):
        raise ValueError(f"invalid_{label}")
    return normalized


def normalize_provenance(source_class: str, source_ref: str, observation_count: int) -> dict[str, object]:
    if not isinstance(source_class, str):
        raise ValueError("invalid_source_class")
    if not isinstance(source_ref, str):
        raise ValueError("invalid_source_ref")
    normalized_class = source_class.strip().lower()
    if normalized_class not in ALLOWED_SOURCE_CLASSES:
        raise ValueError("invalid_source_class")
    ensure_safe_opaque_ref_content(source_ref, "source_ref")
    normalized_ref = source_ref.strip()
    if normalized_ref and not SAFE_REF.match(normalized_ref):
        raise ValueError("unsafe_source_ref")
    if isinstance(observation_count, bool) or not isinstance(observation_count, int) or observation_count < 1 or observation_count > 10000:
        raise ValueError("invalid_observation_count")
    return {
        "source_class": normalized_class,
        "source_ref": normalized_ref,
        "observation_count": observation_count,
        "raw_persisted": False,
    }


def normalize_provenance_from_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("provenance must be an object")
    if set(value) != {"source_class", "source_ref", "observation_count", "raw_persisted"}:
        raise ValueError("provenance_schema_mismatch")
    observation_count = value.get("observation_count", 0)
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise ValueError("invalid_observation_count")
    return normalize_provenance(value.get("source_class", ""), value.get("source_ref", ""), observation_count)


def normalize_confidence(confidence: float, observation_count: int) -> dict[str, object]:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("invalid_confidence_estimate")
    if isinstance(observation_count, bool) or not isinstance(observation_count, int) or not 1 <= observation_count <= 10000:
        raise ValueError("invalid_confidence_observation_count")
    value = float(confidence)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("confidence_out_of_range")
    return {
        "estimate": round(value, 4),
        "evidence_strength": "bounded_operator_review",
        "observation_count": observation_count,
        "routing_authority": "none",
    }


def normalize_confidence_from_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("confidence must be an object")
    if set(value) != {"estimate", "evidence_strength", "observation_count", "routing_authority"}:
        raise ValueError("confidence_schema_mismatch")
    observation_count = value.get("observation_count", 0)
    if isinstance(observation_count, bool) or not isinstance(observation_count, int):
        raise ValueError("invalid_confidence_observation_count")
    estimate = value.get("estimate", -1.0)
    if isinstance(estimate, bool) or not isinstance(estimate, (int, float)):
        raise ValueError("invalid_confidence_estimate")
    normalized = normalize_confidence(estimate, observation_count)
    if not isinstance(value.get("evidence_strength"), str):
        raise ValueError("invalid_confidence_evidence_strength")
    if not isinstance(value.get("routing_authority"), str) or value.get("routing_authority") != "none":
        raise ValueError("invalid_routing_authority")
    return normalized


def normalize_safe_ref(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"unsafe_{label}")
    ensure_safe_opaque_ref_content(value, label)
    normalized = value.strip()
    if not SAFE_REF.match(normalized):
        raise ValueError(f"unsafe_{label}")
    return normalized


def normalize_reason_code(value: object) -> str:
    if value == "":
        return DEFAULT_REVIEW_REASON_CODE
    if not isinstance(value, str):
        raise ValueError("invalid_review_reason_code")
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in ALLOWED_REVIEW_REASON_CODES:
        raise ValueError("invalid_review_reason_code")
    return normalized


def normalize_base_profile_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid_base_profile_revision")
    return value


def ensure_no_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).strip().lower() in _FORBIDDEN_PAYLOAD_KEYS:
                raise ValueError("raw_prompt_transcript_fields_forbidden")
            ensure_no_forbidden_keys(nested)
    elif isinstance(value, list):
        for item in value:
            ensure_no_forbidden_keys(item)


def _digest_payload(profile: dict[str, object]) -> dict[str, object]:
    revision = profile.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("invalid_revision")
    return {
        "schema_version": DOMAIN_PROFILE_SCHEMA_VERSION,
        "profile_id": profile.get("profile_id"),
        "revision": revision,
        "status": profile.get("status"),
        "scope": normalize_scope_from_value(profile.get("scope")),
        "domain_id": normalize_identifier(profile.get("domain_id"), "domain_id"),
        "vocabulary_mappings": normalize_mappings_from_value(profile.get("vocabulary_mappings")),
        "workflow_hints": normalize_workflow_hints(profile.get("workflow_hints", [])),
        "confidence": normalize_confidence_from_value(profile.get("confidence")),
        "provenance": normalize_provenance_from_value(profile.get("provenance")),
        "base_profile_revision": normalize_base_profile_revision(profile.get("base_profile_revision", 0)),
    }
