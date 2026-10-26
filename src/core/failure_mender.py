"""Pure failure classification and no-blind-retry decisions."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Final, Mapping

FAILURE_MENDER_SCHEMA_VERSION: Final = "failure_mender_decision/v1"
FAILURE_KINDS: Final = ("transient", "persistent", "permanent", "external_wait", "unknown")
FAILURE_ACTIONS: Final = ("retry", "replan", "stop", "escalate")
_MAX_REASON_LENGTH: Final = 120

_TRANSIENT_PATTERNS: Final = (
    "timeout", "timed out", "temporarily unavailable", "connection reset",
    "connection refused", "rate limit", "too many requests", "service unavailable",
)
_PERSISTENT_PATTERNS: Final = ("conflict", "stale", "revision", "out of date", "concurrent")
_EXTERNAL_WAIT_PATTERNS: Final = (
    "consent required", "awaiting approval", "credential required",
    "authentication required", "login required",
)
_PERMANENT_PATTERNS: Final = (
    "permission denied", "unauthorized", "forbidden", "invalid", "malformed", "unsupported", "not found",
)


def classify_failure(error: BaseException | str) -> tuple[str, str]:
    """Return ``(kind, reason_code)`` using bounded, deterministic signals."""
    message = str(error).casefold()
    if isinstance(error, (TimeoutError, ConnectionError)) or any(pattern in message for pattern in _TRANSIENT_PATTERNS):
        return "transient", "provider_or_transport_unavailable"
    if any(pattern in message for pattern in _EXTERNAL_WAIT_PATTERNS):
        return "external_wait", "external_authorization_or_approval"
    if isinstance(error, PermissionError) or any(pattern in message for pattern in _PERMANENT_PATTERNS):
        return "permanent", "invalid_or_unauthorized_request"
    if any(pattern in message for pattern in _PERSISTENT_PATTERNS):
        return "persistent", "state_conflict_requires_replan"
    if isinstance(error, ValueError):
        return "permanent", "invalid_input"
    return "unknown", "unclassified_failure"


def decide_failure(
    error: BaseException | str,
    *,
    attempt: int = 0,
    max_retries: int = 2,
    source: str = "",
) -> dict[str, Any]:
    """Build an actionable decision without echoing the failure text."""
    if isinstance(attempt, bool) or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    if isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    kind, reason_code = classify_failure(error)
    action = {
        "transient": "retry" if attempt < max_retries else "escalate",
        "persistent": "replan",
        "permanent": "stop",
        "external_wait": "escalate",
        "unknown": "escalate",
    }[kind]
    text = str(error)
    return {
        "schema_version": FAILURE_MENDER_SCHEMA_VERSION,
        "kind": kind,
        "reason_code": reason_code,
        "action": action,
        "attempt": attempt,
        "max_retries": max_retries,
        "retry_allowed": action == "retry",
        "source": _safe_source(source),
        "failure_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "claim_boundary": "A mender decision is a policy recommendation, not proof that a retry, replan, or escalation occurred.",
    }


def validate_failure_decision(decision: Mapping[str, object] | object) -> list[str]:
    if not isinstance(decision, Mapping):
        return ["decision must be an object"]
    errors: list[str] = []
    if decision.get("schema_version") != FAILURE_MENDER_SCHEMA_VERSION:
        errors.append("schema_version is invalid")
    if decision.get("kind") not in FAILURE_KINDS:
        errors.append("kind is invalid")
    if decision.get("action") not in FAILURE_ACTIONS:
        errors.append("action is invalid")
    for key in ("attempt", "max_retries"):
        value = decision.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"{key} must be a non-negative integer")
    if decision.get("retry_allowed") is not (decision.get("action") == "retry"):
        errors.append("retry_allowed must match action")
    if not re.fullmatch(r"[0-9a-f]{64}", str(decision.get("failure_sha256", ""))):
        errors.append("failure_sha256 must be a sha256 digest")
    if "not" not in str(decision.get("claim_boundary", "")).casefold():
        errors.append("claim_boundary must state a limitation")
    return errors


def build_escalation_request(decision: Mapping[str, object] | object) -> dict[str, Any] | None:
    """Prepare a safe, deterministic Seed handoff for non-retry decisions.

    The request is a proposal only. Callers must review it and invoke their own
    task tracker; this module never creates external issues or retries work.
    """
    if validate_failure_decision(decision):
        raise ValueError("cannot build an escalation request from an invalid decision")
    record = decision
    action = str(record["action"])
    if action not in {"escalate", "replan"}:
        return None
    digest = str(record["failure_sha256"])
    seed_id = f"failure-{digest[:12]}"
    return {
        "schema_version": "failure_escalation_request/v1",
        "seed_id": seed_id,
        "title": f"Investigate {record['reason_code']}",
        "labels": ["failure", "escalation", str(record["kind"])],
        "description": "Investigate the classified failure using the recorded metadata and reproduce it before changing retry behavior.",
        "reason_code": record["reason_code"],
        "source": record["source"],
        "failure_sha256": digest,
        "recommended_action": action,
        "create_command": f"sd create --title 'Investigate {record['reason_code']}' --type bug --labels failure,escalation",
        "claim_boundary": "This is a task proposal; it is not proof that a Seed was created or that escalation occurred.",
    }


def _safe_source(value: str) -> str:
    return " ".join(str(value).split())[:_MAX_REASON_LENGTH]
