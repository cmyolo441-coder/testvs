# allow: SIZE_OK — one lock-scoped transaction owns connector event, gate answer, and receipt recovery.
"""Connector-bound answers for durable decision gates (issue #1370)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final

from ..system.append_only_store import append_store_line, mint_record_id, opaque_ref, redacted_ref
from ..system.local_store import file_lock, read_jsonl_objects, utc_now
from ..system.paths import OmhPaths
from .decision_gates import (
    GATE_CHOICES,
    GATE_STATE_ANSWERED,
    GATE_STATE_OPEN,
    DecisionGateError,
    build_decision_gate_answer,
    gate_is_expired,
    validate_decision_gate,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

CONNECTOR_DECISION_GATE_RECEIPT_SCHEMA_VERSION: Final[str] = "connector_decision_gate_receipt/v1"
CONNECTOR_DECISION_GATE_TRANSACTION_SCHEMA_VERSION: Final[str] = "connector_decision_gate_transaction/v1"
CONNECTOR_DECISION_GATE_RECEIPT_KEYS: Final[tuple[str, ...]] = (
    "actor", "authentication_method", "channel_ref", "choice", "claim_boundary", "connector", "event_id",
    "expected_approver", "expected_resume_digest", "gate_answer_record_ref", "gate_id", "interaction_ref",
    "observed_at", "privacy", "receipt_id", "record_id", "schema_version", "supersedes_gate_ref", "thread_ref",
    "wrapper_expected_revision", "wrapper_session_ref", "payload_digest",
)
CONNECTOR_DECISION_GATE_PAYLOAD_KEYS: Final[tuple[str, ...]] = (
    "gate_id", "expected_resume_digest", "choice", "actor", "connector", "channel_ref", "thread_ref",
    "interaction_ref", "authentication_method", "observed_at", "event_id", "wrapper_session_ref",
    "wrapper_expected_revision",
)
AUTHENTICATION_METHODS: Final[tuple[str, ...]] = ("host_authenticated", "signed_connector_event")
CONNECTOR_DECISION_STATUSES: Final[tuple[str, ...]] = (
    "applied", "already_applied", "expired", "superseded", "unauthorized", "invalid_choice",
    "stale_revision", "unknown_gate", "cancelled", "invalid",
    # Only the plugin tool answers this, on a host whose interpreter has no
    # usable `omh` package; nothing here produces it (#2166).
    "unavailable",
)
CLAIM_BOUNDARY: Final[str] = (
    "A connector decision-gate receipt records authenticated connector metadata bound to one gate answer. "
    "It is not connector credential proof, dispatch, execution, verification, review, CI, merge-readiness, or merge evidence."
)


def wrapper_gate_binding(session_id: str, revision: int) -> dict[str, str]:
    """Derive the sole gate position that may advance one wrapper plan revision."""
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise DecisionGateError("wrapper decision binding requires a non-negative revision")
    safe_session_id = opaque_ref(session_id, field="wrapper decision session", error=DecisionGateError)
    revision_ref = str(revision)
    return {
        "run_id": f"wrapper-{safe_session_id}",
        "subject_class": "tool",
        "subject_ref": safe_session_id,
        "blocked_transition": "plan_accepted",
        "checkpoint_ref": f"wrapper-plan-{safe_session_id}-{revision_ref}",
        "context_revision": f"wrapper-revision-{revision_ref}",
    }


def validate_connector_decision_payload(raw: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Parse the closed connector ingress payload before reading either store."""
    if set(raw) != set(CONNECTOR_DECISION_GATE_PAYLOAD_KEYS):
        return {}, "invalid_payload"
    revision = raw.get("wrapper_expected_revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        return {}, "invalid_payload"
    if raw.get("authentication_method") not in AUTHENTICATION_METHODS or raw.get("choice") not in GATE_CHOICES:
        return {}, "invalid_payload"
    observed = _timestamp(raw.get("observed_at"))
    if observed is None:
        return {}, "invalid_payload"
    cleaned: dict[str, Any] = {"wrapper_expected_revision": revision}
    for key in CONNECTOR_DECISION_GATE_PAYLOAD_KEYS:
        if key == "wrapper_expected_revision":
            continue
        value = raw.get(key)
        if not isinstance(value, str):
            return {}, "invalid_payload"
        try:
            cleaned[key] = opaque_ref(value, field=f"connector decision {key}", error=DecisionGateError)
        except DecisionGateError:
            return {}, "invalid_payload"
    return cleaned, ""


def answer_connector_decision_gate(
    paths: OmhPaths,
    *,
    gate_id: str,
    expected_resume_digest: str,
    choice: str,
    actor: str,
    connector: str,
    channel_ref: str,
    thread_ref: str,
    interaction_ref: str,
    authentication_method: str,
    observed_at: str,
    event_id: str,
    wrapper_session_ref: str,
    wrapper_expected_revision: int,
    trusted_now: str = "",
) -> dict[str, Any]:
    """Append exactly one event-bound answer and its receipt under the gate lock."""
    payload, error = validate_connector_decision_payload(
        {
            "gate_id": gate_id, "expected_resume_digest": expected_resume_digest, "choice": choice, "actor": actor,
            "connector": connector, "channel_ref": channel_ref, "thread_ref": thread_ref,
            "interaction_ref": interaction_ref, "authentication_method": authentication_method,
            "observed_at": observed_at, "event_id": event_id, "wrapper_session_ref": wrapper_session_ref,
            "wrapper_expected_revision": wrapper_expected_revision,
        }
    )
    if error:
        return _result("invalid", error)
    decision_at = trusted_now or utc_now()
    if _timestamp(decision_at) is None:
        return _result("invalid", "invalid_clock")
    path = paths.runtime_decision_gates_path
    with file_lock(path, private=True):
        records, _ = read_jsonl_objects(path)
        digest = _payload_digest(payload)
        receipt = _receipt_for_event(records, payload["event_id"])
        if receipt:
            if str(receipt.get("payload_digest", "")) != digest:
                return _result("invalid", "conflicting_replay")
            return _result("already_applied", "", compact_connector_decision_receipt(receipt))
        transaction = _transaction_for_event(records, payload["event_id"])
        if transaction and str(transaction.get("payload_digest", "")) != digest:
            return _result("invalid", "conflicting_replay")
        gate = _latest_gate(records, payload["gate_id"])
        if not gate:
            return _result("unknown_gate", "unknown_gate")
        binding_error = _gate_binding_error(gate, payload)
        if binding_error:
            return _result("stale_revision", binding_error)
        timestamp_error = _timestamp_error(gate, payload["observed_at"], decision_at)
        if timestamp_error:
            return _result("invalid", timestamp_error)
        if gate_is_expired(gate, decision_at):
            return _result("expired", "expired")
        if payload["actor"] != str(gate.get("approver", "")):
            return _result("unauthorized", "unauthorized")
        if payload["choice"] not in gate.get("choices", []):
            return _result("invalid_choice", "invalid_choice")
        if transaction:
            decision_at = str(transaction.get("decided_at", ""))
        elif _transaction_for_gate(records, payload["gate_id"]):
            return _result("superseded", "pending_event")
        elif str(gate.get("state", "")) != GATE_STATE_OPEN:
            return _result("superseded", "superseded")
        else:
            transaction = _build_transaction(payload, digest, decision_at, str(gate["record_id"]))
            append_store_line(path, transaction)
        if str(gate.get("state", "")) == GATE_STATE_OPEN:
            answer = build_decision_gate_answer(
                gate, actor=payload["actor"], choice=payload["choice"], decided_at=decision_at
            )
            append_store_line(path, answer)
        elif (
            str(gate.get("state", "")) == GATE_STATE_ANSWERED
            and str(gate.get("supersedes_gate_ref", "")) == str(transaction.get("open_gate_record_ref", ""))
            and str(gate.get("decided_at", "")) == decision_at
        ):
            answer = gate
        else:
            return _result("superseded", "superseded")
        receipt = _build_receipt(payload, answer, digest)
        append_store_line(path, receipt)
    return _result("applied", "", compact_connector_decision_receipt(receipt))


def read_connector_decision_receipts(paths: OmhPaths) -> list[dict[str, Any]]:
    """Read connector receipts from the shared append-only gate store."""
    records, _ = read_jsonl_objects(paths.runtime_decision_gates_path)
    return [record for record in records if record.get("schema_version") == CONNECTOR_DECISION_GATE_RECEIPT_SCHEMA_VERSION]


def connector_receipt_for_id(paths: OmhPaths, receipt_id: str) -> dict[str, Any]:
    """Return one valid receipt by identity, or an empty mapping."""
    for receipt in reversed(read_connector_decision_receipts(paths)):
        if receipt.get("receipt_id") == receipt_id and not validate_connector_decision_receipt(receipt):
            return receipt
    return {}


def validate_connector_decision_receipt(record: dict[str, Any]) -> list[str]:
    """Validate through the shared gate-store validator."""
    return validate_decision_gate(record)


def compact_connector_decision_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Render only bounded opaque references for connector-facing output."""
    return {
        "receipt_id": redacted_ref(str(receipt.get("receipt_id", "")), field="connector_receipt"),
        "gate_id": redacted_ref(str(receipt.get("gate_id", "")), field="connector_gate"),
        "choice": str(receipt.get("choice", "")) if receipt.get("choice") in GATE_CHOICES else "",
        "actor": redacted_ref(str(receipt.get("actor", "")), field="connector_actor"),
        "connector": redacted_ref(str(receipt.get("connector", "")), field="connector"),
        "interaction_ref": redacted_ref(str(receipt.get("interaction_ref", "")), field="connector_interaction"),
        "observed_at": redacted_ref(str(receipt.get("observed_at", "")), field="connector_time"),
        "expected_resume_digest": str(receipt.get("expected_resume_digest", "")),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _gate_binding_error(gate: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    try:
        binding = wrapper_gate_binding(str(payload["wrapper_session_ref"]), int(payload["wrapper_expected_revision"]))
    except DecisionGateError:
        return "invalid_wrapper_binding"
    for key, expected in binding.items():
        if str(gate.get(key, "")) != expected:
            return "wrapper_binding_mismatch"
    if str(gate.get("resume_digest", "")) != str(payload["expected_resume_digest"]):
        return "stale_revision"
    return ""


def _timestamp_error(gate: Mapping[str, Any], observed_at: str, trusted_now: str) -> str:
    observed = _timestamp(observed_at)
    opened = _timestamp(gate.get("opened_at"))
    now = _timestamp(trusted_now)
    if observed is None or opened is None or now is None:
        return "invalid_timestamp"
    if observed < opened:
        return "observed_before_open"
    if observed > now:
        return "observed_in_future"
    return ""


def _build_transaction(payload: Mapping[str, Any], digest: str, decided_at: str, open_gate_ref: str) -> dict[str, Any]:
    transaction_id = "connector-transaction-" + hashlib.sha256(digest.encode("utf-8")).hexdigest()[:20]
    return {
        "schema_version": CONNECTOR_DECISION_GATE_TRANSACTION_SCHEMA_VERSION,
        "record_id": mint_record_id(prefix="connector-decision-transaction", identity={"transaction_id": transaction_id}),
        "transaction_id": transaction_id,
        "event_id": str(payload["event_id"]),
        "payload_digest": digest,
        "gate_id": str(payload["gate_id"]),
        "open_gate_record_ref": open_gate_ref,
        "decided_at": decided_at,
        "privacy": "metadata_only",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _build_receipt(payload: Mapping[str, Any], answer: Mapping[str, Any], digest: str) -> dict[str, Any]:
    receipt_id = "connector-receipt-" + hashlib.sha256(digest.encode("utf-8")).hexdigest()[:20]
    return {
        "schema_version": CONNECTOR_DECISION_GATE_RECEIPT_SCHEMA_VERSION,
        "record_id": mint_record_id(prefix="connector-decision", identity={"receipt_id": receipt_id}),
        "receipt_id": receipt_id,
        "gate_id": str(answer["gate_id"]),
        "gate_answer_record_ref": str(answer["record_id"]),
        "expected_resume_digest": str(payload["expected_resume_digest"]),
        "choice": str(payload["choice"]),
        "actor": str(payload["actor"]),
        "expected_approver": str(answer["approver"]),
        "connector": str(payload["connector"]),
        "channel_ref": str(payload["channel_ref"]),
        "thread_ref": str(payload["thread_ref"]),
        "interaction_ref": str(payload["interaction_ref"]),
        "authentication_method": str(payload["authentication_method"]),
        "observed_at": str(payload["observed_at"]),
        "event_id": str(payload["event_id"]),
        "wrapper_session_ref": str(payload["wrapper_session_ref"]),
        "wrapper_expected_revision": int(payload["wrapper_expected_revision"]),
        "supersedes_gate_ref": str(answer["record_id"]),
        "privacy": "metadata_only",
        "claim_boundary": CLAIM_BOUNDARY,
        "payload_digest": digest,
    }


def _receipt_for_event(records: Sequence[Mapping[str, Any]], event_id: str) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("schema_version") == CONNECTOR_DECISION_GATE_RECEIPT_SCHEMA_VERSION and record.get("event_id") == event_id:
            return dict(record)
    return {}


def _transaction_for_event(records: Sequence[Mapping[str, Any]], event_id: str) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("schema_version") == CONNECTOR_DECISION_GATE_TRANSACTION_SCHEMA_VERSION and record.get("event_id") == event_id:
            return dict(record)
    return {}


def _latest_gate(records: Sequence[Mapping[str, Any]], gate_id: str) -> dict[str, Any]:
    gates = [record for record in records if record.get("schema_version") == "decision_gate/v1" and record.get("gate_id") == gate_id]
    return dict(gates[-1]) if gates else {}


def _transaction_for_gate(records: Sequence[Mapping[str, Any]], gate_id: str) -> dict[str, Any]:
    for record in reversed(records):
        if record.get("schema_version") == CONNECTOR_DECISION_GATE_TRANSACTION_SCHEMA_VERSION and record.get("gate_id") == gate_id:
            return dict(record)
    return {}


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _result(status: str, reason_code: str, receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "schema_version": "connector_decision_gate_result/v1",
        "status": status if status in CONNECTOR_DECISION_STATUSES else "invalid",
        "reason_code": reason_code,
        "receipt": dict(receipt or {}),
        "claim_boundary": CLAIM_BOUNDARY,
    }
