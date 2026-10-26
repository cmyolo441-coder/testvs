"""Closed schemas and validation for Hermes skill-load observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Final, cast
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

SKILL_LOAD_OBSERVATION_SCHEMA_VERSION: Final = "skill_load_observation/v1"
HERMES_SKILL_INVENTORY_SCHEMA_VERSION: Final = "hermes_skill_inventory/v1"
PROBE_STATUSES: Final = ("observed", "unsupported", "probe_error")
LOAD_STATES: Final = ("all_loaded", "partially_loaded", "none_loaded", "not_applicable")
REASON_CODES: Final = (
    "inventory_validated", "inventory_protocol_unavailable", "inventory_process_error",
    "inventory_executable_unavailable", "inventory_timeout", "inventory_response_malformed",
    "inventory_nonce_mismatch", "inventory_expected_digest_mismatch",
    "inventory_digest_mismatch", "inventory_response_expired", "inventory_response_stale",
    "inventory_nonce_replayed",
)
MAX_PROTOCOL_SECONDS: Final = 300
SKILL_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
OBSERVATION_FIELDS: Final = frozenset({
    "schema_version", "probe_status", "reason_code", "nonce", "expected_digest",
    "inventory_digest", "tool_fingerprint", "runtime_fingerprint", "observed_at",
    "expires_at", "load_state", "expected_skills", "observed_skills", "missing_skills",
    "unexpected_skills",
})
INVENTORY_FIELDS: Final = frozenset({
    "schema_version", "nonce", "expected_digest", "inventory_digest", "tool_fingerprint",
    "runtime_fingerprint", "observed_at", "expires_at", "observed_skills",
})
INVENTORY_CLAIM_FIELDS: Final = frozenset({
    "inventory_digest", "load_state", "expected_skills", "observed_skills", "missing_skills",
    "unexpected_skills",
})


def expected_skills_digest(skills: Sequence[str]) -> str:
    return set_digest(normalized_skill_set(skills, field="expected_skills"))


def validated_inventory_response(
    raw: bytes, *, nonce: str, expected_digest: str, tool_fingerprint: str, now: datetime,
) -> tuple[dict[str, object] | None, str]:
    try:
        decoded_candidate = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, "inventory_response_malformed"
    if not isinstance(decoded_candidate, dict) or set(decoded_candidate) != INVENTORY_FIELDS:
        return None, "inventory_response_malformed"
    candidate = cast(dict[str, object], decoded_candidate)
    if candidate.get("schema_version") != HERMES_SKILL_INVENTORY_SCHEMA_VERSION:
        return None, "inventory_response_malformed"
    if candidate.get("nonce") != nonce:
        return None, "inventory_nonce_mismatch"
    if candidate.get("expected_digest") != expected_digest:
        return None, "inventory_expected_digest_mismatch"
    if candidate.get("tool_fingerprint") != tool_fingerprint:
        return None, "inventory_response_malformed"
    runtime = candidate.get("runtime_fingerprint")
    if not isinstance(runtime, str) or not HEX_64.fullmatch(runtime):
        return None, "inventory_response_malformed"
    observed_raw = candidate.get("observed_skills")
    if not isinstance(observed_raw, list):
        return None, "inventory_response_malformed"
    try:
        observed = normalized_skill_set(observed_raw, field="observed_skills")
    except ValueError:
        return None, "inventory_response_malformed"
    if observed_raw != list(observed):
        return None, "inventory_response_malformed"
    if candidate.get("inventory_digest") != set_digest(observed):
        return None, "inventory_digest_mismatch"
    try:
        response_observed = parse_time(candidate.get("observed_at"))
        expires = parse_time(candidate.get("expires_at"))
    except ValueError:
        return None, "inventory_response_malformed"
    if response_observed > now + timedelta(seconds=5):
        return None, "inventory_response_malformed"
    if response_observed < now - timedelta(seconds=MAX_PROTOCOL_SECONDS):
        return None, "inventory_response_stale"
    if expires <= now:
        return None, "inventory_response_expired"
    if expires <= response_observed or expires > response_observed + timedelta(seconds=MAX_PROTOCOL_SECONDS):
        return None, "inventory_response_malformed"
    candidate["observed_skills"] = list(observed)
    return candidate, "inventory_validated"


def closed_observation(
    status: str, reason: str, nonce: str, expected_digest: str, tool_fingerprint: str,
    runtime_fingerprint: str, observed_at: datetime,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": SKILL_LOAD_OBSERVATION_SCHEMA_VERSION, "probe_status": status,
        "reason_code": reason, "nonce": nonce, "expected_digest": expected_digest,
        "tool_fingerprint": tool_fingerprint, "runtime_fingerprint": runtime_fingerprint,
        "observed_at": format_time(observed_at),
        "expires_at": format_time(observed_at + timedelta(seconds=MAX_PROTOCOL_SECONDS)),
    }
    errors = validate_skill_load_observation(payload)
    if errors:
        raise ValueError("invalid closed skill-load observation: " + "; ".join(errors))
    return payload


def validate_skill_load_observation(payload: Mapping[str, object] | object) -> list[str]:
    if not isinstance(payload, Mapping):
        return ["payload must be an object"]
    errors: list[str] = []
    extras = sorted(set(payload) - OBSERVATION_FIELDS)
    if extras:
        errors.append(f"unsupported fields: {extras}")
    if payload.get("schema_version") != SKILL_LOAD_OBSERVATION_SCHEMA_VERSION:
        errors.append("schema_version is invalid")
    status = payload.get("probe_status")
    if status not in PROBE_STATUSES:
        errors.append("probe_status is invalid")
    if payload.get("reason_code") not in REASON_CODES:
        errors.append("reason_code is invalid")
    for field in ("nonce", "expected_digest", "tool_fingerprint", "runtime_fingerprint"):
        value = payload.get(field)
        if not isinstance(value, str) or not HEX_64.fullmatch(value):
            errors.append(f"{field} is invalid")
    for field in ("observed_at", "expires_at"):
        try:
            parse_time(payload.get(field))
        except ValueError:
            errors.append(f"{field} is invalid")
    if status == "observed":
        _validate_observed_claims(payload, errors)
    else:
        forbidden = sorted(set(payload) & INVENTORY_CLAIM_FIELDS)
        if forbidden:
            errors.append(f"non-observed probe carries inventory claims: {forbidden}")
        if status == "unsupported" and payload.get("reason_code") != "inventory_protocol_unavailable":
            errors.append("unsupported probe reason is invalid")
        if status == "probe_error" and payload.get("reason_code") in {
            "inventory_validated", "inventory_protocol_unavailable"
        }:
            errors.append("probe_error reason is invalid")
    return errors


def _validate_observed_claims(payload: Mapping[str, object], errors: list[str]) -> None:
    if payload.get("reason_code") != "inventory_validated":
        errors.append("observed probe requires inventory_validated")
    if payload.get("load_state") not in LOAD_STATES:
        errors.append("observed probe requires load_state")
    sets: dict[str, tuple[str, ...]] = {}
    for field in ("expected_skills", "observed_skills", "missing_skills", "unexpected_skills"):
        raw = payload.get(field)
        if not isinstance(raw, list):
            errors.append(f"{field} must be a sorted unique list")
            continue
        try:
            normalized = normalized_skill_set(raw, field=field)
        except ValueError:
            errors.append(f"{field} must be a sorted unique list")
            continue
        if raw != list(normalized):
            errors.append(f"{field} must be a sorted unique list")
        sets[field] = normalized
    digest = payload.get("inventory_digest")
    if not isinstance(digest, str) or not HEX_64.fullmatch(digest):
        errors.append("inventory_digest is invalid")
    if len(sets) != 4:
        return
    expected, observed = sets["expected_skills"], sets["observed_skills"]
    missing = tuple(sorted(set(expected) - set(observed)))
    unexpected = tuple(sorted(set(observed) - set(expected)))
    if sets["missing_skills"] != missing or sets["unexpected_skills"] != unexpected:
        errors.append("inventory set differences are invalid")
    if payload.get("expected_digest") != set_digest(expected):
        errors.append("expected_digest does not bind expected_skills")
    if digest != set_digest(observed):
        errors.append("inventory_digest does not bind observed_skills")
    expected_state = (
        "not_applicable" if not expected else "none_loaded" if set(expected).isdisjoint(observed)
        else "partially_loaded" if missing else "all_loaded"
    )
    if payload.get("load_state") != expected_state:
        errors.append("load_state does not match inventory sets")


def skill_load_observation_is_fresh(
    payload: Mapping[str, object], *, now: datetime | None = None,
) -> bool:
    if validate_skill_load_observation(payload):
        return False
    try:
        observed, expires = parse_time(payload.get("observed_at")), parse_time(payload.get("expires_at"))
    except ValueError:
        return False
    current = utc(now)
    return observed <= current < expires


def normalized_skill_set(values: Sequence[object], *, field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not SKILL_NAME.fullmatch(value):
            raise ValueError(f"{field} contains an invalid skill name")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field} contains duplicate skill names")
    return tuple(sorted(normalized))


def set_digest(values: Sequence[str]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def parse_time(value: object) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamp must be UTC")
    parsed = strict_fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must have timezone")
    return parsed.astimezone(timezone.utc)


def utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current.astimezone(timezone.utc)


def format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
