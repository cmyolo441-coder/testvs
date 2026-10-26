"""CLI wiring for connector-safe decision-gate answers."""

from __future__ import annotations

import argparse

from ..workflows.decision_gate_receipts import (
    answer_connector_decision_gate,
    read_connector_decision_receipts,
    validate_connector_decision_payload,
)
from ..system.record_revision import record_revision_of
from ..workflows.decision_gates import latest_gate_in, read_decision_gates
from ..wrapper.sessions import consume_connector_decision_receipt, read_wrapper_session
from .common import _paths, _print_json


def apply_connector_decision_answer(paths, *, trusted_now: str = "", **raw_payload):
    """Run the validated receipt write and matching wrapper consumption sequence."""
    payload, error = validate_connector_decision_payload(raw_payload)
    if error:
        return {
            "schema_version": "connector_decision_gate_result/v1",
            "status": "invalid",
            "reason_code": error,
            "receipt": {},
        }
    session = read_wrapper_session(paths, str(payload["wrapper_session_ref"]))
    if not session or str(session.get("status", "")) == "cancelled":
        return {"schema_version": "connector_decision_gate_result/v1", "status": "cancelled", "reason_code": "cancelled", "receipt": {}}
    retry = any(receipt.get("event_id") == payload["event_id"] for receipt in read_connector_decision_receipts(paths))
    if not retry and (
        record_revision_of(session) != int(payload["wrapper_expected_revision"])
        or str(session.get("status", "")) != "plan_presented"
    ):
        gate = latest_gate_in(read_decision_gates(paths), str(payload["gate_id"]))
        if str(gate.get("state", "")) == "answered":
            return {"schema_version": "connector_decision_gate_result/v1", "status": "superseded", "reason_code": "superseded", "receipt": {}}
        return {"schema_version": "connector_decision_gate_result/v1", "status": "stale_revision", "reason_code": "stale_wrapper_session", "receipt": {}}
    existing = answer_connector_decision_gate(paths, **payload, trusted_now=trusted_now)
    if existing["status"] not in {"applied", "already_applied"}:
        return existing
    try:
        wrapper = consume_connector_decision_receipt(
            paths,
            session_id=str(payload["wrapper_session_ref"]),
            receipt_id=str(existing["receipt"]["receipt_id"]),
            now=trusted_now,
        )
    except (FileNotFoundError, ValueError):
        return {**existing, "status": "invalid", "reason_code": "wrapper_refused"}
    return {**existing, "wrapper": wrapper}


def cmd_runtime_decision_gate_answer(args: argparse.Namespace) -> int:
    """Apply one structured connector event to its named gate and wrapper session."""
    _print_json(
        apply_connector_decision_answer(
            _paths(args),
            gate_id=args.gate_id,
            expected_resume_digest=args.expected_resume_digest,
            choice=args.choice,
            actor=args.actor,
            connector=args.connector,
            channel_ref=args.channel_ref,
            thread_ref=args.thread_ref,
            interaction_ref=args.interaction_ref,
            authentication_method=args.authentication_method,
            observed_at=args.observed_at,
            event_id=args.event_id,
            wrapper_session_ref=args.wrapper_session_ref,
            wrapper_expected_revision=args.wrapper_expected_revision,
        )
    )
    return 0


def add_runtime_decision_gate_commands(runtime_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the agent/operator-only structured connector ingress."""
    gate = runtime_sub.add_parser(
        "decision-gate",
        help="Agent/operator surface for authenticated connector decisions; never accepts free-form chat text.",
    )
    gate_sub = gate.add_subparsers(dest="runtime_decision_gate_command", required=True)
    answer = gate_sub.add_parser("answer", help="Append an authenticated connector-bound answer and consume it once.")
    answer.add_argument("--gate-id", required=True)
    answer.add_argument("--expected-resume-digest", required=True)
    answer.add_argument("--choice", choices=("approve", "decline", "defer"), required=True)
    answer.add_argument("--actor", required=True, help="Stable authenticated actor identifier, never a display name.")
    answer.add_argument("--connector", required=True)
    answer.add_argument("--channel-ref", required=True)
    answer.add_argument("--thread-ref", required=True)
    answer.add_argument("--interaction-ref", required=True)
    answer.add_argument("--authentication-method", choices=("host_authenticated", "signed_connector_event"), required=True)
    answer.add_argument("--observed-at", required=True, help="Observed connector decision time in ISO-8601 form.")
    answer.add_argument("--event-id", required=True, help="Stable connector delivery idempotency key.")
    answer.add_argument("--wrapper-session-ref", required=True)
    answer.add_argument("--wrapper-expected-revision", required=True, type=int)
    answer.set_defaults(func=cmd_runtime_decision_gate_answer)
