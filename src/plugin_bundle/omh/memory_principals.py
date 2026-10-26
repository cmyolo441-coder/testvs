"""Opaque acting-principal contracts for OMH-local memory boundaries."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._memory_principal_validation import (
    ACTOR_KINDS,
    BINDING_STATES,
    MEMORY_AUDIENCE_SCHEMA_VERSION,
    MEMORY_IDENTITY_SCHEMA_VERSION,
    PRINCIPAL_CONTEXT_KEYS,
    PRINCIPAL_CONTEXT_SCHEMA_VERSION,
    PRINCIPAL_REF_PATTERN as PRINCIPAL_REF_PATTERN,
    memory_identity_errors,
    principal_ref as _principal,
    safe_ref as _safe,
    valid_review_ref as _valid_review_ref,
)


CHANNEL_REF_PATTERN = re.compile(r"^channel:v1:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PrincipalContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason

def derive_principal_ref(
    omh_home: str | Path,
    *,
    profile_ref: str,
    surface_ref: str,
    host_identity: str,
) -> str:
    """Derive an opaque profile-and-surface-bound principal without retaining input."""
    profile = _safe_ref(profile_ref, "profile_ref")
    surface = _safe_ref(surface_ref, "surface_ref")
    identity = str(host_identity)
    if not identity or len(identity.encode("utf-8")) > 1024:
        raise PrincipalContractError("host_identity must be nonempty and bounded")
    message = f"{profile}\0{surface}\0{identity}".encode("utf-8")
    return "principal:v1:" + hmac.new(_principal_key(omh_home), message, hashlib.sha256).hexdigest()


def derive_channel_ref(omh_home: str | Path, *, profile_ref: str, surface_ref: str, channel_identity: str) -> str:
    """An opaque channel scope ref, keyed like a principal: a group's id can carry a phone number."""
    profile = _safe_ref(profile_ref, "profile_ref")
    surface = _safe_ref(surface_ref, "surface_ref")
    identity = str(channel_identity)
    if not identity or len(identity.encode("utf-8")) > 1024:
        raise PrincipalContractError("channel_identity must be nonempty and bounded")
    message = f"channel\0{profile}\0{surface}\0{identity}".encode("utf-8")
    return "channel:v1:" + hmac.new(_principal_key(omh_home), message, hashlib.sha256).hexdigest()


def _principal_key(omh_home: str | Path) -> bytes:
    """The profile-local HMAC key, created on first use."""
    key_path = Path(omh_home).expanduser() / "memory" / "principal.key"
    key_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        _ = key_path.chmod(0o600)
    else:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(os.urandom(32))
    key_path.chmod(0o600)
    key = key_path.read_bytes()
    if len(key) != 32:
        raise PrincipalContractError("principal key is invalid")
    return key


def build_principal_context(
    omh_home: str | Path,
    *,
    profile_ref: str,
    surface_ref: str,
    session_ref: str,
    turn_ref: str,
    actor_kind: str,
    host_identity: str = "",
    identity_evidence_refs: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build caller-local identity evidence; it is not host authentication proof."""
    kind = str(actor_kind)
    if kind not in ACTOR_KINDS:
        raise PrincipalContractError("unsupported actor_kind")
    principal = (
        derive_principal_ref(
            omh_home,
            profile_ref=profile_ref,
            surface_ref=surface_ref,
            host_identity=host_identity,
        )
        if kind == "human"
        else None
    )
    return {
        "schema_version": PRINCIPAL_CONTEXT_SCHEMA_VERSION,
        "principal": principal,
        "profile_ref": _safe_ref(profile_ref, "profile_ref"),
        "surface_ref": _safe_ref(surface_ref, "surface_ref"),
        "session_ref": _safe_ref(session_ref, "session_ref"),
        "turn_ref": _safe_ref(turn_ref, "turn_ref"),
        "actor_kind": kind,
        "identity_evidence_refs": [_safe_ref(ref, "identity_evidence_ref") for ref in identity_evidence_refs[:8]],
        "binding_state": "validated_local" if kind == "human" else "unbound",
    }


def parse_principal_context(
    value: Any,
    *,
    expected_profile: str = "",
    expected_session: str = "",
    expected_turn: str = "",
) -> dict[str, Any] | None:
    """Parse a closed principal context and fail closed on active-context mismatch."""
    if not isinstance(value, dict) or set(value) != PRINCIPAL_CONTEXT_KEYS:
        return None
    if value.get("schema_version") != PRINCIPAL_CONTEXT_SCHEMA_VERSION:
        return None
    kind = value.get("actor_kind")
    binding = value.get("binding_state")
    principal = value.get("principal")
    if (
        not isinstance(kind, str)
        or kind not in ACTOR_KINDS
        or not isinstance(binding, str)
        or binding not in BINDING_STATES
    ):
        return None
    if kind == "human" and (binding not in {"validated_local", "host_validated"} or not _principal(principal)):
        return None
    if kind != "human" and principal is not None:
        return None
    refs = value.get("identity_evidence_refs")
    if not isinstance(refs, list) or len(refs) > 8 or not all(_safe(ref) for ref in refs):
        return None
    for key in ("profile_ref", "surface_ref", "session_ref", "turn_ref"):
        if not _safe(value.get(key)):
            return None
    if expected_profile and value.get("profile_ref") != expected_profile:
        return None
    if expected_session and value.get("session_ref") != expected_session:
        return None
    if expected_turn and value.get("turn_ref") != expected_turn:
        return None
    return {key: value[key] for key in PRINCIPAL_CONTEXT_KEYS}


def build_memory_identity(
    context: dict[str, Any],
    *,
    scope_kind: str,
    reviewer_principal: str | None = None,
    executor_perspective: str = "hermes",
    audience_principals: list[str] | tuple[str, ...] = (),
    review_ref: str = "",
) -> dict[str, Any]:
    """Bind subject, event actor, reviewer, perspective, and reviewed audience."""
    parsed = parse_principal_context(context)
    if parsed is None or parsed["actor_kind"] != "human":
        raise PrincipalContractError("human principal context required")
    actor = str(parsed["principal"])
    audience = sorted(set(str(item) for item in audience_principals))
    if len(audience) > 64 or any(not _principal(item) for item in audience):
        raise PrincipalContractError("audience principals must be bounded opaque references")
    if scope_kind == "user":
        audience_kind, audience, subject = "subject_only", [actor], actor
    elif audience:
        audience_kind, subject = "explicit_principals", None
    else:
        raise PrincipalContractError("shared project/thread memory requires an explicit principal audience")
    if reviewer_principal is not None and not _principal(reviewer_principal):
        raise PrincipalContractError("reviewer principal must be opaque")
    normalized_review_ref = _review_ref(review_ref, allow_empty=True)
    return {
        "schema_version": MEMORY_IDENTITY_SCHEMA_VERSION,
        "subject_principal": subject,
        "event_actor": {"principal": actor, "actor_kind": "human"},
        "reviewer": {"principal": reviewer_principal, "review_ref": normalized_review_ref},
        "executor_perspective": _safe_ref(executor_perspective, "executor_perspective"),
        "audience": {
            "schema_version": MEMORY_AUDIENCE_SCHEMA_VERSION,
            "kind": audience_kind,
            "principal_refs": audience,
            "review_ref": normalized_review_ref,
        },
    }


def principal_recall_decision(
    record: dict[str, Any],
    context: dict[str, Any] | None,
    *,
    shared_surface: bool,
    channel_scopes: list[dict[str, str]] | tuple[dict[str, str], ...] = (),
) -> dict[str, Any]:
    """Return a metadata-only allow/deny before scope, ranking, or rendering.

    ``channel_scopes`` are the messenger channels the caller's allowlist names
    (`prefetch_scope_allowlist`). A record scoped to exactly one of them is the
    channel's own memory, captured there with no single author, so a shared
    surface admits it identity-unbound: its audience is that channel.
    """
    schema = str(record.get("schema_version", ""))
    if schema != "project_memory_record/v3":
        if shared_surface and record.get("scope") in [scope for scope in channel_scopes if scope.get("kind") == "channel"]:
            return {"allowed": True, "reason_code": "channel_audience"}
        return {"allowed": not shared_surface, "reason_code": "legacy_nonshared" if not shared_surface else "identity_unbound_shared"}
    identity = record.get("identity")
    if not isinstance(identity, dict) or memory_identity_errors(identity):
        return {"allowed": False, "reason_code": "identity_invalid"}
    parsed = parse_principal_context(context)
    if parsed is None:
        return {"allowed": False, "reason_code": "principal_context_missing_or_mismatched"}
    if parsed["actor_kind"] != "human":
        return {"allowed": False, "reason_code": f"actor_{parsed['actor_kind']}"}
    principal = str(parsed["principal"])
    audience = identity.get("audience")
    if not isinstance(audience, dict) or audience.get("schema_version") != MEMORY_AUDIENCE_SCHEMA_VERSION:
        return {"allowed": False, "reason_code": "audience_invalid"}
    kind = audience.get("kind")
    refs = audience.get("principal_refs")
    if not isinstance(refs, list) or len(refs) > 64 or any(not _principal(item) for item in refs):
        return {"allowed": False, "reason_code": "audience_invalid"}
    subject = identity.get("subject_principal")
    if kind == "subject_only":
        allowed = _principal(subject) and subject == principal and refs == [principal]
        return {"allowed": allowed, "reason_code": "principal_match" if allowed else "principal_mismatch"}
    if kind == "explicit_principals":
        allowed = subject is None and principal in refs and bool(audience.get("review_ref"))
        return {"allowed": allowed, "reason_code": "audience_allowed" if allowed else "audience_denied"}
    return {"allowed": False, "reason_code": "audience_invalid"}


def principal_operation_decision(
    record: dict[str, Any],
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Authorize a lifecycle/export operation through the same identity boundary."""
    return principal_recall_decision(record, context, shared_surface=True)


def audience_policy_digest(record: dict[str, Any]) -> str:
    identity = record.get("identity") if isinstance(record.get("identity"), dict) else {}
    audience = identity.get("audience") if isinstance(identity, dict) else {}
    encoded = json.dumps(audience, sort_keys=True, separators=(",", ":")).encode("utf-8") if isinstance(audience, dict) else b""
    return hashlib.sha256(encoded).hexdigest()


def _review_ref(value: Any, *, allow_empty: bool) -> str:
    if not _valid_review_ref(value, allow_empty=allow_empty):
        raise PrincipalContractError("review_ref must be a bounded opaque metadata reference")
    return value


def _safe_ref(value: Any, field: str) -> str:
    text = str(value)
    if not _safe(text):
        raise PrincipalContractError(f"{field} must be an opaque metadata reference")
    return text
