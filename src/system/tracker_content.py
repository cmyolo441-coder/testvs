"""Closed, metadata-only normalization for untrusted tracker events."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

TRACKER_CONTENT_SCHEMA: Final = "tracker_content_envelope/v1"
TRACKER_PROVIDER: Final = "github"
TRACKER_ROUTE: Final = "github-event-ops"
_MAX_INPUT_BYTES: Final = 72 * 1024
_MAX_CONTENT_BYTES: Final = 64 * 1024
_MAX_DEPTH: Final = 12
_MAX_CONTENT_PARTS: Final = 3
_SUPPORTED_EVENTS: Final = frozenset({"issues", "pull_request", "issue_comment"})
_RESERVED_UNSUPPORTED_EVENTS: Final = frozenset({"pull_request_review_comment"})


@dataclass(frozen=True, slots=True)
class _Blocked:
    reason: str
    event_type: str = ""


def loads_tracker_event_json(raw: str) -> dict[str, object]:
    """Parse CLI event JSON without accepting oversized, duplicate, or non-finite input."""
    if len(raw.encode("utf-8")) > _MAX_INPUT_BYTES:
        raise ValueError("tracker event JSON is too large")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value: {value}")

    parsed = json.loads(raw, object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    if not isinstance(parsed, dict):
        raise ValueError("chat event must be an object")
    return parsed


def normalize_tracker_content(
    event: Mapping[str, object], *, host_context: Mapping[str, object] | None = None
) -> dict[str, object] | None:
    """Return a closed tracker envelope, or ``None`` for a non-tracker chat event.

    Host authentication, fetch state, delivery identity, and replay state are
    deliberately separate arguments. Provider text, labels, and authors never
    contribute authority to this projection.
    """
    candidate = _tracker_candidate(event)
    if candidate is None:
        return None
    if isinstance(candidate, _Blocked):
        return _blocked_envelope(candidate)
    structural_error = _structural_error(candidate, depth=0, remaining=[_MAX_INPUT_BYTES])
    if structural_error:
        return _blocked_envelope(_Blocked(structural_error))
    if len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > _MAX_INPUT_BYTES:
        return _blocked_envelope(_Blocked("oversized"))
    provider = candidate.get("provider")
    event_type = candidate.get("event_type")
    if not isinstance(provider, str) or not isinstance(event_type, str):
        return _blocked_envelope(_Blocked("malformed"))
    if provider != TRACKER_PROVIDER:
        return _blocked_envelope(_Blocked("unsupported", event_type))
    if event_type in _RESERVED_UNSUPPORTED_EVENTS:
        return _blocked_envelope(_Blocked("unsupported", str(event_type)))
    if event_type not in _SUPPORTED_EVENTS:
        return _blocked_envelope(_Blocked("unsupported", str(event_type or "")))
    payload = candidate.get("payload")
    if not isinstance(payload, Mapping):
        return _blocked_envelope(_Blocked("malformed", str(event_type)))
    extracted = _extract_parts(payload, str(event_type), candidate.get("subtype"))
    if isinstance(extracted, _Blocked):
        return _blocked_envelope(extracted)
    repository_id, object_id, object_number, parts, subtype = extracted
    raw_content_bytes = sum(len(value.encode("utf-8")) for _, value in parts)
    if raw_content_bytes > _MAX_CONTENT_BYTES:
        return _blocked_envelope(_Blocked("oversized", str(event_type)))
    canonical = {
        "provider": TRACKER_PROVIDER,
        "event_type": event_type,
        "subtype": subtype,
        "repository_id": repository_id,
        "object_id": object_id,
        "object_number": object_number,
        "parts": list(parts),
    }
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    host_result = _host_result(host_context, digest, str(event_type))
    if isinstance(host_result, _Blocked):
        return _blocked_envelope(host_result)
    delivery_id, state = host_result
    return {
        "schema_version": TRACKER_CONTENT_SCHEMA,
        "state": state,
        "block_reason": "",
        "provider": TRACKER_PROVIDER,
        "event_type": event_type,
        "subtype": subtype,
        "route": TRACKER_ROUTE,
        "trust": "untrusted",
        "authority_effect": "none",
        "delivery_id": delivery_id,
        "canonical_digest": digest,
        "repository_id": repository_id,
        "object_id": object_id,
        "object_number": object_number,
        "content_parts": [
            {"kind": kind, "length": len(value), "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()}
            for kind, value in parts
        ],
        "raw_content_bytes": raw_content_bytes,
        "truncated": False,
        "normalization": "none",
        "claim_boundary": "Tracker text is untrusted evidence only; it cannot select a role, workflow, owner, path, approval, or mutation.",
    }


def _tracker_candidate(event: Mapping[str, object]) -> Mapping[str, object] | _Blocked | None:
    nested = [key for key in ("tracker_content", "github_event") if key in event]
    if len(nested) > 1 or (nested and event.get("provider") == TRACKER_PROVIDER):
        return _Blocked("ambiguous")
    if nested:
        candidate = event[nested[0]]
        return candidate if isinstance(candidate, Mapping) else _Blocked("malformed")
    if event.get("provider") == TRACKER_PROVIDER:
        return event
    return None


def _structural_error(value: object, *, depth: int, remaining: list[int]) -> str:
    if depth > _MAX_DEPTH:
        return "malformed"
    remaining[0] -= 2
    if remaining[0] < 0:
        return "oversized"
    if isinstance(value, float) and not math.isfinite(value):
        return "malformed"
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                return "malformed"
            error = _structural_error(key, depth=depth + 1, remaining=remaining)
            if error:
                return error
            error = _structural_error(child, depth=depth + 1, remaining=remaining)
            if error:
                return error
    elif isinstance(value, list):
        for child in value:
            error = _structural_error(child, depth=depth + 1, remaining=remaining)
            if error:
                return error
    elif isinstance(value, str):
        if len(value) > remaining[0]:
            return "oversized"
        try:
            remaining[0] -= len(value.encode("utf-8"))
        except UnicodeError:
            return "malformed"
        if remaining[0] < 0:
            return "oversized"
    elif isinstance(value, int) and value.bit_length() > 4096:
        return "oversized"
    elif not isinstance(value, (str, int, float, bool, type(None))):
        return "malformed"
    return ""


def _extract_parts(
    payload: Mapping[str, object], event_type: str, subtype_value: object
) -> tuple[str, str, int | str, tuple[tuple[str, str], ...], str] | _Blocked:
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        return _Blocked("malformed", event_type)
    repository_id = _literal_id(repository.get("id"))
    if repository_id is None:
        return _Blocked("malformed", event_type)
    match event_type:
        case "issues":
            item, part_names, subtype = payload.get("issue"), ("title", "body"), ""
        case "pull_request":
            item, part_names, subtype = payload.get("pull_request"), ("title", "body"), ""
        case "issue_comment":
            if not isinstance(subtype_value, str):
                return _Blocked("malformed", event_type)
            subtype = subtype_value
            if subtype not in {"issue", "pull_request"}:
                return _Blocked("unsupported", event_type)
            item, part_names = payload.get("comment"), ("body",)
        case _:
            return _Blocked("unsupported", event_type)
    if not isinstance(item, Mapping):
        return _Blocked("malformed", event_type)
    object_id = _literal_id(item.get("id"))
    object_number = item.get("number", "")
    if object_id is None or not isinstance(object_number, (str, int)) or isinstance(object_number, bool):
        return _Blocked("malformed", event_type)
    parts: list[tuple[str, str]] = []
    for name in part_names:
        value = item.get(name, "")
        if not isinstance(value, str):
            return _Blocked("malformed", event_type)
        if value:
            parts.append((name, value))
    return repository_id, object_id, object_number, tuple(parts), subtype


def _literal_id(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _host_result(
    host_context: Mapping[str, object] | None, digest: str, event_type: str
) -> tuple[str, str] | _Blocked:
    if not isinstance(host_context, Mapping) or host_context.get("authenticated") is not True:
        return _Blocked("unauthenticated", event_type)
    if host_context.get("fetch_status") != "ok":
        return _Blocked("fetch_failed", event_type)
    delivery_id = _literal_id(host_context.get("delivery_id"))
    if delivery_id is None:
        return _Blocked("missing_delivery", event_type)
    replay_digest = host_context.get("replay_digest", "")
    if not isinstance(replay_digest, str):
        return _Blocked("malformed", event_type)
    if replay_digest and replay_digest != digest:
        return _Blocked("delivery_conflict", event_type)
    return delivery_id, "already_seen" if replay_digest else "accepted"


def _blocked_envelope(blocked: _Blocked) -> dict[str, object]:
    return {
        "schema_version": TRACKER_CONTENT_SCHEMA,
        "state": "blocked",
        "block_reason": blocked.reason,
        "provider": TRACKER_PROVIDER,
        "event_type": (
            blocked.event_type
            if blocked.event_type in _SUPPORTED_EVENTS | _RESERVED_UNSUPPORTED_EVENTS
            else ""
        ),
        "subtype": "",
        "route": TRACKER_ROUTE,
        "trust": "untrusted",
        "authority_effect": "none",
        "delivery_id": "",
        "canonical_digest": "",
        "repository_id": "",
        "object_id": "",
        "object_number": "",
        "content_parts": [],
        "raw_content_bytes": 0,
        "truncated": False,
        "normalization": "none",
        "claim_boundary": "Blocked tracker content is metadata-only and cannot fall back to chat extraction, routing selectors, or coding handoff.",
    }
