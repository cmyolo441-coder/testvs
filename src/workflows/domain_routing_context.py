from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import unicodedata

from .domain_intelligence_profile_resolution import (
    MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS,
    MAX_DOMAIN_CONTEXT_MATCHES,
    DomainClarificationTarget,
    matches_reviewed_phrase,
    resolve_domain_clarification_target_result,
)
from ..system.local_store import FileLockTimeout


__all__ = (
    "MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS",
    "MAX_DOMAIN_CONTEXT_MATCHES",
    "DomainClarificationTarget",
    "build_domain_routing_context",
    "matches_reviewed_phrase",
    "resolve_domain_routing_context",
)


DOMAIN_ROUTING_CONTEXT_KEY = "domain_routing_context"
DOMAIN_ROUTING_CONTEXT_SCHEMA_VERSION = "domain_routing_context/v1"
DOMAIN_ROUTING_CONTEXT_CLAIM_BOUNDARY = (
    "Reviewed domain context only selects one wrapper clarification question; it is not "
    "routing, plan approval, execution, review, CI, merge, authentication, or Hermes "
    "internal-memory evidence."
)

MAX_WORKFLOW_HINT_CODE_POINTS = 120
MAX_REQUIRED_INPUT_CODE_POINTS = 120
MAX_QUESTION_CODE_POINTS = 240
MAX_CLAIM_BOUNDARY_CODE_POINTS = 320
_SUPPORTED_QUESTION_LOCALES = frozenset({"en", "ko"})
_SNAPSHOT_CHANGE_ERRORS = frozenset(
    {
        "artifact_changed_during_read",
        "domain_health_directory_changed",
        "domain_profile_snapshot_changed",
    }
)
_DOMAIN_ROUTING_RESOLUTION_REASONS = frozenset(
    {
        "applied",
        "canonical_conflict",
        "compatibility_only_workflow_hint",
        "conflicting_workflow_hints",
        "context_projection_invalid",
        "empty_workflow_hints",
        "invalid_binding",
        "invalid_request",
        "input_too_large",
        "match_overflow",
        "missing_binding",
        "missing_question_spec",
        "no_match",
        "non_routable_workflow_hint",
        "profile_store_lock_failed",
        "profile_store_snapshot_changed",
        "profile_store_unhealthy",
        "unknown_workflow_hint",
    }
)


@dataclass(frozen=True)
class DomainRoutingResolution:
    """Private outcome retained only for in-process resolution diagnostics."""

    status: str
    reason: str
    context: dict[str, object] | None = None

    def __post_init__(self) -> None:
        applied = self.status == "applied"
        if self.status not in {"absent", "excluded", "applied"}:
            raise ValueError("invalid_domain_resolution_status")
        if self.reason not in _DOMAIN_ROUTING_RESOLUTION_REASONS:
            raise ValueError("invalid_domain_resolution_reason")
        if applied != (self.context is not None):
            raise ValueError("invalid_domain_resolution_context")


def build_domain_routing_context(
    targets: Sequence[DomainClarificationTarget],
) -> dict[str, object] | None:
    """Build the applied-only public fragment for exactly one valid catalog target."""
    if not isinstance(targets, (list, tuple)) or len(targets) != 1:
        return None
    target = targets[0]
    if not isinstance(target, DomainClarificationTarget) or not _valid_target(target):
        return None

    context: dict[str, object] = {
        "schema_version": DOMAIN_ROUTING_CONTEXT_SCHEMA_VERSION,
        "workflow_hint": target.workflow_hint,
        "required_input": target.required_input,
        "question": {
            "locale": target.question_locale,
            "text": target.question_text,
        },
        "claim_boundary": DOMAIN_ROUTING_CONTEXT_CLAIM_BOUNDARY,
    }
    context["digest"] = _canonical_public_digest(context)
    return {DOMAIN_ROUTING_CONTEXT_KEY: context}


def resolve_domain_routing_context(
    binding: object,
    message: object,
    *,
    locale: str,
) -> dict[str, object] | None:
    """Return only the existing public context projection, never diagnostics."""
    return resolve_domain_routing_context_result(
        binding,
        message,
        locale=locale,
    ).context


def resolve_domain_routing_context_result(
    binding: object,
    message: object,
    *,
    locale: str,
) -> DomainRoutingResolution:
    """Resolve one question while retaining a sanitized in-process outcome."""
    from .domain_intelligence_profile_snapshot import read_validated_domain_profiles_at
    from .domain_project_context import HostProjectBinding

    if not isinstance(message, str):
        return DomainRoutingResolution("absent", "invalid_request")
    if len(message) > MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS:
        return DomainRoutingResolution("excluded", "input_too_large")
    if binding is None:
        return DomainRoutingResolution("absent", "missing_binding")
    if not isinstance(binding, HostProjectBinding):
        return DomainRoutingResolution("absent", "invalid_binding")
    try:
        profiles = read_validated_domain_profiles_at(binding)
    except (OSError, ValueError) as exc:
        return DomainRoutingResolution("excluded", _store_failure_reason(exc))

    target_resolution = resolve_domain_clarification_target_result(
        profiles,
        message,
        project_root=binding.project_root,
        locale=locale,
    )
    target = target_resolution.target
    if target is None:
        status = "absent" if target_resolution.reason == "no_match" else "excluded"
        return DomainRoutingResolution(status, target_resolution.reason)
    context = build_domain_routing_context((target,))
    if context is None:
        return DomainRoutingResolution("excluded", "context_projection_invalid")
    return DomainRoutingResolution("applied", "applied", context)


def _store_failure_reason(exc: OSError | ValueError) -> str:
    if isinstance(exc, FileLockTimeout):
        return "profile_store_lock_failed"
    message = str(exc)
    if message in _SNAPSHOT_CHANGE_ERRORS:
        return "profile_store_snapshot_changed"
    filename = getattr(exc, "filename", None)
    if (
        isinstance(filename, (str, Path))
        and Path(filename).name == ".store.lock"
    ) or message == "shared_lock_unavailable" or message.startswith(
        "domain-intelligence lock path"
    ):
        return "profile_store_lock_failed"
    return "profile_store_unhealthy"


def _valid_target(target: DomainClarificationTarget) -> bool:
    return (
        _valid_public_string(target.workflow_hint, MAX_WORKFLOW_HINT_CODE_POINTS)
        and _valid_public_string(target.required_input, MAX_REQUIRED_INPUT_CODE_POINTS)
        and target.question_locale in _SUPPORTED_QUESTION_LOCALES
        and _valid_public_string(target.question_text, MAX_QUESTION_CODE_POINTS)
        and len(DOMAIN_ROUTING_CONTEXT_CLAIM_BOUNDARY) <= MAX_CLAIM_BOUNDARY_CODE_POINTS
    )


def _valid_public_string(value: object, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value == value.strip()
        and value == unicodedata.normalize("NFKC", value)
        and len(value) <= maximum
    )


def _canonical_public_digest(context: dict[str, object]) -> str:
    preimage = json.dumps(
        context,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()
