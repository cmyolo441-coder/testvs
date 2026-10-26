"""Build and validate ``omh_platform_envelope/v1`` payloads.

Safety rules: ``platform`` is required, one of the 23 registered ids, and no
other top-level context key is accepted. ``source`` must equal the profile's
``transport_source`` -- an adapter cannot speak for a transport it is not.
Every provided ref passes safe-opaque validation and is not a raw phone
number, email address, secret value, or body-shaped blob. Declared limits
use the renderer keys ``max_recommended_chars`` / ``hard_limit_chars``:
integers, ``200 <= recommended < hard``, ``hard >= 400`` (so the headroom
clamp never yields recommended < 200), ``hard`` capped at the verified
profile hard limit or ``CORE_LIMIT_CAP``; effective recommended is
``min(declared, hard - 200)``. A capability override may resolve an unknown
profile field but never contradict a verified one; today every registry
value is unknown. An explicit but unknown ``render_profile`` normalizes down
to ``limited_markdown`` (mirroring ``src/wrapper/contract.py``), recorded in
provenance.
"""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote

from .metadata_safety import (
    is_body_shaped_metadata_text,
    is_raw_pii_shaped,
    is_secret_value_shaped,
    require_opaque_metadata_ref,
)
from .platform_profiles import (
    CAPABILITY_GROUPS,
    CLAIM_BOUNDARY,
    CORE_LIMIT_CAP,
    DEFAULT_RENDER_PROFILE,
    PLATFORM_PROFILES,
    RENDER_PROFILES,
    PlatformProfile,
)

ENVELOPE_SCHEMA_VERSION = "omh_platform_envelope/v1"

_MIN_RECOMMENDED_CHARS = 200  # below this no useful message fits
_LIMIT_HEADROOM_CHARS = 200  # flat recommended-to-hard gap, as in contract.py
_MIN_HARD_LIMIT_CHARS = _MIN_RECOMMENDED_CHARS + _LIMIT_HEADROOM_CHARS
_REF_LIMIT = 160  # single-line bound; longer means body, not reference

_REF_FIELDS = ("conversation_ref", "thread_ref", "user_ref")

#: Top-level keys a platform context may carry; anything else fails closed.
_ALLOWED_CONTEXT_KEYS = frozenset({"platform", "limits", "capabilities", "render_profile", *_REF_FIELDS})

SESSION_SCOPE_CONVERSATION = "conversation"
SESSION_SCOPE_EVENT_FALLBACK = "event_fallback"


class PlatformContextError(ValueError):
    """Raised when a platform context cannot produce a safe envelope."""


def _validated_ref(value: object, *, field: str) -> str:
    try:
        ref = require_opaque_metadata_ref(value, field=field)
    except ValueError as exc:
        raise PlatformContextError(str(exc)) from exc
    if (
        is_raw_pii_shaped(ref)
        or is_secret_value_shaped(ref)
        or is_body_shaped_metadata_text(ref, limit=_REF_LIMIT)
    ):
        raise PlatformContextError(
            f"{field} must be an opaque platform reference, not raw PII, "
            "a secret, or a body"
        )
    return ref


def _is_verified_limit_platform(profile: PlatformProfile) -> bool:
    return profile.transport_source == profile.platform_id


def _resolve_limits(profile: PlatformProfile, declared: object) -> tuple[dict[str, int], str]:
    verified = _is_verified_limit_platform(profile)
    if declared is None:
        return {
            "max_recommended_chars": profile.limits.max_recommended_chars,
            "hard_limit_chars": profile.limits.hard_limit_chars,
        }, "verified" if verified else "conservative_default"
    if not isinstance(declared, Mapping):
        raise PlatformContextError("limits must map max_recommended_chars/hard_limit_chars")
    recommended: Any = declared.get("max_recommended_chars")
    hard: Any = declared.get("hard_limit_chars")
    for name, value in (
        ("max_recommended_chars", recommended),
        ("hard_limit_chars", hard),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise PlatformContextError(f"limits.{name} must be an integer")
    if recommended < _MIN_RECOMMENDED_CHARS:
        raise PlatformContextError(f"limits.max_recommended_chars < {_MIN_RECOMMENDED_CHARS}")
    if recommended >= hard:
        raise PlatformContextError("limits.max_recommended_chars must be below hard_limit_chars")
    if hard < _MIN_HARD_LIMIT_CHARS:
        raise PlatformContextError(f"limits.hard_limit_chars < {_MIN_HARD_LIMIT_CHARS}")
    # Verified platforms cap declarations at the verified hard limit; any
    # other platform may declare up to the core cap.
    cap = profile.limits.hard_limit_chars if verified else CORE_LIMIT_CAP
    if hard > cap:
        raise PlatformContextError(f"limits.hard_limit_chars above {cap} for {profile.platform_id}")
    return {
        "max_recommended_chars": min(recommended, hard - _LIMIT_HEADROOM_CHARS),
        "hard_limit_chars": hard,
    }, "adapter_declared"


def _resolve_capabilities(
    profile: PlatformProfile, overrides: object
) -> tuple[dict[str, dict[str, bool]], dict[str, dict[str, str]]]:
    if overrides is not None and not isinstance(overrides, Mapping):
        raise PlatformContextError("capabilities must be a mapping of groups")
    declared: Mapping[str, Any] = overrides if isinstance(overrides, Mapping) else {}
    for group in declared:
        if group not in CAPABILITY_GROUPS:
            raise PlatformContextError(f"capabilities.{group} is not a known group")
    resolved: dict[str, dict[str, bool]] = {}
    provenance: dict[str, dict[str, str]] = {}
    for group in CAPABILITY_GROUPS:
        group_declared = declared.get(group, {})
        if not isinstance(group_declared, Mapping):
            raise PlatformContextError(f"capabilities.{group} must be a mapping")
        resolved[group] = {}
        provenance[group] = {}
        for name, verified in profile.capabilities[group].items():
            if name not in group_declared:
                resolved[group][name] = verified is True
                provenance[group][name] = (
                    "verified" if verified is not None else "unverified_default_false"
                )
                continue
            override = group_declared[name]
            label = f"capabilities.{group}.{name}"
            if not isinstance(override, bool):
                raise PlatformContextError(f"{label} override must be a boolean")
            if verified is not None and override != verified:
                raise PlatformContextError(
                    f"{label} is verified {verified}; an adapter may not contradict it"
                )
            resolved[group][name] = override
            provenance[group][name] = "verified" if override == verified else "adapter_declared"
        for name in group_declared:
            if name not in profile.capabilities[group]:
                raise PlatformContextError(f"capabilities.{group}.{name} is not a known capability")
    return resolved, provenance


def _resolve_render_profile(profile: PlatformProfile, declared: object) -> tuple[str, str]:
    """Unknown explicit input coerces down to limited, coercion recorded."""
    if declared is None:
        return profile.render_profile, "profile_default"
    if not isinstance(declared, str):
        raise PlatformContextError("render_profile must be a string")
    if declared in RENDER_PROFILES:
        return declared, "adapter_declared"
    return DEFAULT_RENDER_PROFILE, "normalized_unknown_to_limited"


def build_platform_envelope(
    platform_context: Mapping[str, Any], *, source: str
) -> dict[str, Any]:
    """Build an ``omh_platform_envelope/v1`` dict from an adapter's context.

    ``source`` is the transport the adapter speaks for and must equal the
    profile's ``transport_source``. Raises :class:`PlatformContextError` on
    any unsafe, unknown, or contradictory input.
    """
    if not isinstance(platform_context, Mapping):
        raise PlatformContextError("platform_context must be a mapping")
    for key in platform_context:
        if key not in _ALLOWED_CONTEXT_KEYS:
            raise PlatformContextError(f"unknown platform_context key: {key!r}")
    platform = platform_context.get("platform")
    if not isinstance(platform, str) or not platform:
        raise PlatformContextError("platform is required")
    profile = PLATFORM_PROFILES.get(platform)
    if profile is None:
        raise PlatformContextError(f"unknown platform: {platform!r}")
    if not isinstance(source, str) or source != profile.transport_source:
        raise PlatformContextError(
            f"source must be {profile.transport_source!r} for platform "
            f"{profile.platform_id!r}"
        )
    limits, limit_provenance = _resolve_limits(profile, platform_context.get("limits"))
    capabilities, capability_provenance = _resolve_capabilities(
        profile, platform_context.get("capabilities")
    )
    render_profile, render_profile_provenance = _resolve_render_profile(
        profile, platform_context.get("render_profile")
    )
    identity: dict[str, str] = {}
    for field_name in _REF_FIELDS:
        value = platform_context.get(field_name)
        if value is not None:
            identity[field_name] = _validated_ref(value, field=field_name)
    return {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "platform_id": profile.platform_id,
        "transport_source": profile.transport_source,
        "render_profile": render_profile,
        "render_profile_provenance": render_profile_provenance,
        "format_family": profile.format_family,
        "capabilities": capabilities,
        "capability_provenance": capability_provenance,
        "limits": limits,
        "limit_provenance": limit_provenance,
        "identity": identity,
        "session_scope": (
            SESSION_SCOPE_CONVERSATION
            if "conversation_ref" in identity
            else SESSION_SCOPE_EVENT_FALLBACK
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _require_envelope(envelope: Mapping[str, Any]) -> None:
    if (
        not isinstance(envelope, Mapping)
        or envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION
    ):
        raise PlatformContextError(f"expected an {ENVELOPE_SCHEMA_VERSION} envelope")


def compact_session_platform(envelope: Mapping[str, Any]) -> dict[str, str]:
    """Persist only platform identity and scope -- never refs or limits."""
    _require_envelope(envelope)
    return {
        "platform_id": envelope["platform_id"],
        "transport_source": envelope["transport_source"],
        "session_scope": envelope["session_scope"],
    }


def platform_thread_key_scope(envelope: Mapping[str, Any]) -> str:
    """Stable thread-key scope: platform plus conversation/thread identity.

    Each identity segment is percent-encoded (``urllib.parse.quote`` with an
    empty safe set) so the ``:`` segment separator can never collide with a
    ``:`` inside a ref: ``conversation_ref='conv:th_9'`` with no thread and
    ``conversation_ref='conv', thread_ref='th_9'`` produce distinct scopes.
    A literal ``%`` encodes as ``%25``, so refs that already look encoded
    round-trip distinctly as well. Unreserved characters (letters, digits,
    ``-._~``) pass through unchanged, so ordinary refs keep their visible
    keys. The platform id itself is registry-safe and stays plain.
    """
    _require_envelope(envelope)
    parts = [envelope["platform_id"]]
    identity = envelope.get("identity", {})
    for field_name in ("conversation_ref", "thread_ref"):
        if field_name in identity:
            parts.append(quote(identity[field_name], safe=""))
    return ":".join(parts)
