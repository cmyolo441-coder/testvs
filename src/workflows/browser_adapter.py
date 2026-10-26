"""Pure, bounded browser contracts. The injected host owns all execution.

No tool name, HTTP method, label, or origin establishes mutation safety.
``act(read)`` means an adapter's inert semantic read, never page script or
navigation. Other actions need the separate last-mile effect consumer.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Protocol, TypedDict
from urllib.parse import urlsplit

CAPABILITIES_SCHEMA = "browser_adapter_capabilities/v1"
LEASE_SCHEMA = "browser_session_lease/v1"
PAGE_SCHEMA = "browser_page_state/v1"
MUTATION_INTERCEPTION = ("none", "last_mile")
LIMITS = {"leases": 4, "tabs": 4, "actions": 64, "ttl_seconds": 300,
          "capability_seconds": 60, "elements": 32, "state_bytes": 4096}
CHANNELS = {"semantic_state", "screenshot", "console", "network_summary", "accessibility",
            "download", "upload", "mutation_interception"}
MODES = {"headless", "headed", "attached"}
ROLES = {"button", "link", "textbox", "checkbox", "radio", "combobox", "option",
         "heading", "img", "text", "document", "tab", "menuitem", "other"}
TERMINALS = {"released", "expired", "capability_expired", "action_capped", "tab_capped",
             "state_capped", "deadline_exceeded", "capability_invalid", "orphaned"}
CLAIM_BOUNDARY = "Bounded host ownership and observations only; not login validity, page correctness, or external-effect evidence."


class BrowserContractError(ValueError):
    """A closed refusal code; never includes host content."""


class RawCapabilities(TypedDict):
    schema_version: str
    modes: list[str]
    channels: list[str]
    unsupported: list[str]
    mutation_interception: str
    limits: dict[str, int]


class CapabilitySnapshot(RawCapabilities):
    adapter_id: str
    adapter_version: str
    cache_identity: str
    observed_at: float
    expires_at: float


class BrowserAdapter(Protocol):
    """Trusted host injection, not tool-argument supplied callables.

    start/release must serialize by lease_id, enforce deadline autonomously,
    and release must revoke the id and reap *all* its resources, even after
    partial start. observe/act must enforce host DOM revision atomically before
    interaction; OMH's cached revision alone cannot detect live DOM changes.
    All callbacks must return by deadline (including killing host workers).
    Exceptions propagate; the reservation remains unknown and is never retried.
    """
    adapter_id: str
    adapter_version: str

    def capabilities(self) -> RawCapabilities: ...
    def start(self, lease_id: str, scope: dict[str, object], deadline: float) -> dict[str, object]: ...
    def observe(self, lease_id: str, tab_id: str, deadline: float) -> dict[str, object]: ...
    def act(self, lease_id: str, tab_id: str, revision: int, key: str,
            operation: str, deadline: float) -> dict[str, object]: ...
    def release(self, lease_id: str, deadline: float) -> dict[str, object]: ...

    # #1392 extension: preview may ONLY prepare/hold, never speculatively run
    # page code to discover its effects. Every downstream request/redirect must
    # remain held; resume is authorized by a durable attempt, never by GET.
    def preview(self, lease_id: str, handle: str, operation: str) -> dict[str, object]: ...
    def resume(self, lease_id: str, preview_ref: str, attempt_id: str) -> dict[str, object]: ...
    def abort(self, lease_id: str, preview_ref: str) -> dict[str, object]: ...


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def token(value, maximum=128) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum or not re.fullmatch(r"[A-Za-z0-9_.:-]+", value):
        raise BrowserContractError("invalid_token")
    return value


def text(value, maximum=2048) -> str:
    if not isinstance(value, str) or not value or len(value.encode()) > maximum:
        raise BrowserContractError("invalid_text")
    return value


def number(value, maximum, minimum=1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise BrowserContractError("invalid_limit")
    return value


def timestamp(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise BrowserContractError("invalid_time")
    return value


def origin(value: str) -> str:
    text(value, 2048)
    try:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise BrowserContractError("invalid_origin")
        port = parsed.port
    except ValueError as exc:
        raise BrowserContractError("invalid_origin") from exc
    host = parsed.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    suffix = f":{port}" if port and port != {"http": 80, "https": 443}[parsed.scheme] else ""
    result = f"{parsed.scheme}://{host}{suffix}"
    if len(result) > 256:
        raise BrowserContractError("invalid_origin")
    return result


def closed_list(value, allowed, maximum):
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise BrowserContractError("invalid_list")
    if any(not isinstance(v, str) or v not in allowed for v in value) or len(set(value)) != len(value):
        raise BrowserContractError("invalid_list")
    return sorted(value)


def acquisition_request(owner, adapter, request):
    """Hash all identity inputs; raw task/auth/owner never enter the store."""
    owner_ref = digest(text(owner, 256))
    adapter_id, version = token(adapter.adapter_id), token(adapter.adapter_version)
    origins = request.get("origins")
    if not isinstance(origins, list) or not 1 <= len(origins) <= 8:
        raise BrowserContractError("invalid_origins")
    origins = sorted(set(origin(value) for value in origins))
    actions = closed_list(request.get("actions"), {"read", "navigate", "click", "type", "press", "upload", "download"}, 7)
    mode = request.get("mode", "headless")
    if not isinstance(mode, str) or mode not in MODES:
        raise BrowserContractError("unsupported_mode")
    identity = {"owner_ref": owner_ref, "adapter_id": adapter_id, "adapter_version": version,
                "origins": origins, "actions": actions, "mode": mode,
                "auth_boundary_ref": digest(text(request.get("auth_boundary"), 256)),
                "task_ref": digest(text(request.get("task"), 256))}
    return {**identity, "lease_id": digest(identity)}


def capability_snapshot(raw, identity, now: float) -> CapabilitySnapshot:
    if not isinstance(raw, dict) or raw.get("schema_version") != CAPABILITIES_SCHEMA:
        raise BrowserContractError("capability_invalid")
    limits = raw.get("limits")
    if not isinstance(limits, dict) or set(limits) != set(LIMITS):
        raise BrowserContractError("capability_invalid")
    limits = {key: number(limits[key], maximum) for key, maximum in LIMITS.items()}
    interception = raw.get("mutation_interception")
    if interception not in MUTATION_INTERCEPTION:
        raise BrowserContractError("capability_invalid")
    modes = closed_list(raw.get("modes"), MODES, len(MODES))
    channels = closed_list(raw.get("channels"), CHANNELS, len(CHANNELS))
    unsupported = raw.get("unsupported", [])
    if unsupported:
        unsupported = closed_list(unsupported, CHANNELS | MODES, len(CHANNELS | MODES))
    elif unsupported != []:
        raise BrowserContractError("capability_invalid")
    if identity["mode"] not in modes or "semantic_state" not in channels:
        raise BrowserContractError("capability_invalid")
    snapshot: CapabilitySnapshot = {"schema_version": CAPABILITIES_SCHEMA, "adapter_id": identity["adapter_id"],
                "adapter_version": identity["adapter_version"], "cache_identity": digest([
                    identity["adapter_id"], identity["adapter_version"], identity["lease_id"]]),
                "observed_at": timestamp(now), "expires_at": now + limits["capability_seconds"],
                "modes": modes, "channels": channels, "unsupported": unsupported,
                "mutation_interception": interception, "limits": limits}
    if len(json.dumps(snapshot).encode()) > 2048:
        raise BrowserContractError("capability_invalid")
    return snapshot


def page_state(raw, lease, tab_id, previous=None):
    if not isinstance(raw, dict) or raw.get("readback") is not True:
        raise BrowserContractError("readback_unknown")
    url = text(raw.get("url"), 2048)
    observed_origin = origin(url)
    if observed_origin not in lease["origins"]:
        raise BrowserContractError("origin_mismatch")
    host_revision = number(raw.get("revision"), 2**53, 0)
    limits = lease["capabilities"]["limits"]
    elements = raw.get("elements")
    if not isinstance(elements, list) or len(elements) > limits["elements"]:
        raise BrowserContractError("state_capped")
    projected = []
    for element in elements:
        if not isinstance(element, dict):
            raise BrowserContractError("invalid_state")
        role = element.get("role")
        if role not in ROLES:
            raise BrowserContractError("invalid_state")
        projected.append({"role": role, "name_digest": digest(text(element.get("name"))),
                          "key_digest": digest(text(element.get("key"), 256))})
    if len({e["key_digest"] for e in projected}) != len(projected):
        raise BrowserContractError("ambiguous_handle")
    fingerprint = digest([url, host_revision, projected])
    if previous and host_revision < previous["host_revision"]:
        raise BrowserContractError("stale_state")
    revision = previous["revision"] if previous and previous["fingerprint"] == fingerprint else (previous["revision"] + 1 if previous else 1)
    for element in projected:
        element["handle"] = digest([lease["lease_id"], tab_id, revision, element["key_digest"]])
    old = {e["key_digest"] for e in previous["elements"]} if previous else set()
    new = {e["key_digest"] for e in projected}
    result = {"schema_version": PAGE_SCHEMA, "lease_id": lease["lease_id"], "tab_id": tab_id,
              "revision": revision, "host_revision": host_revision, "fingerprint": fingerprint,
              "observed_origin": observed_origin, "observed_url_digest": digest(url),
              "url_redacted": True, "readback": "observed", "elements": projected,
              "delta": {"from_revision": previous["revision"] if previous else 0,
                        "added": len(new - old), "removed": len(old - new),
                        "changed": bool(previous and previous["fingerprint"] != fingerprint)}}
    if len(json.dumps(result).encode()) > limits["state_bytes"]:
        raise BrowserContractError("state_capped")
    return result


def resolve_handle(page, request):
    if type(request.get("revision")) is not int or request["revision"] != page["revision"]:
        raise BrowserContractError("stale_state")
    handle = request.get("handle")
    if handle:
        matches = [e for e in page["elements"] if e["handle"] == handle]
    else:
        role, name = request.get("role"), request.get("name")
        if not isinstance(name, str) or not name or len(name.encode()) > 2048:
            raise BrowserContractError("missing_handle")
        matches = [e for e in page["elements"] if e["role"] == role and e["name_digest"] == digest(name)]
    if len(matches) != 1:
        raise BrowserContractError("ambiguous_handle" if matches else "missing_handle")
    return matches[0]


def pending_effect(raw):
    """Closed held-request metadata seam for #1392; NEVER an authorization."""
    fields = {"schema_version", "held", "method", "origin", "payload_bytes", "payload_digest",
              "payload_shape", "target_role", "target_name_digest", "opens_new_tab", "redirected", "preview_ref"}
    if not isinstance(raw, dict) or set(raw) != fields or raw.get("schema_version") != "browser_pending_effect/v1":
        raise BrowserContractError("invalid_preview")
    if raw["held"] is not True or raw["opens_new_tab"] is not False or raw["redirected"] is not False:
        raise BrowserContractError("unheld_effect")
    if raw["method"] not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
        raise BrowserContractError("invalid_preview")
    if raw["payload_shape"] not in {"empty", "form", "json", "opaque"} or raw["target_role"] not in ROLES:
        raise BrowserContractError("invalid_preview")
    for key in ("payload_digest", "target_name_digest", "preview_ref"):
        if not isinstance(raw[key], str) or not re.fullmatch("[a-f0-9]{64}", raw[key]):
            raise BrowserContractError("invalid_preview")
    number(raw["payload_bytes"], 1024 * 1024, 0)
    if origin(raw["origin"]) != raw["origin"]:
        raise BrowserContractError("invalid_preview")
    return dict(raw)
