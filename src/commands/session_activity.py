from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..installer import OmhError
from ..workflows.session_activity_receipts import (
    CLAIM_BOUNDARY,
    SESSION_ACTIVITY_CONSUMERS,
    SessionActivityReceiptError,
    admit_session_activity_receipt,
    ingest_session_activity_receipt,
    read_session_activity_quarantine,
    read_session_activity_receipts,
    session_activity_evidence,
)
from .common import _paths, _print_json


def cmd_runtime_session_receipt_ingest(args: argparse.Namespace) -> int:
    try:
        payload = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OmhError(str(exc)) from exc
    if args.max_age_seconds is not None and args.max_age_seconds < 0:
        raise OmhError("--max-age-seconds must be zero or greater")
    if args.dry_run:
        admission = admit_session_activity_receipt(
            payload,
            expected_profile_ref=args.expected_profile or "",
            known=read_session_activity_receipts(_paths(args)),
            max_age_seconds=args.max_age_seconds,
            redact_skill_names=args.redact_skill_names,
        )
        result = {**admission, "written": False}
        receipt = admission["receipt"] if admission["outcome"] == "accepted" else None
    else:
        result = ingest_session_activity_receipt(
            _paths(args),
            payload,
            expected_profile_ref=args.expected_profile or "",
            max_age_seconds=args.max_age_seconds,
            redact_skill_names=args.redact_skill_names,
        )
        receipt = result["receipt"] if result["outcome"] in ("recorded", "already_recorded") else None
    try:
        result["evidence"] = {
            consumer: session_activity_evidence(receipt, consumer) if receipt else None
            for consumer in (args.consumer or [])
        }
    except SessionActivityReceiptError as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_runtime_session_receipt_list(args: argparse.Namespace) -> int:
    limit = int(args.limit)
    if limit < 1:
        raise OmhError("--limit must be at least 1")
    paths = _paths(args)
    _print_json(
        {
            "schema_version": "session_activity_receipt_list/v1",
            "receipts": read_session_activity_receipts(paths, session_ref=args.session, limit=limit),
            "quarantined": read_session_activity_quarantine(paths)[-limit:],
            "claim_boundary": CLAIM_BOUNDARY,
        }
    )
    return 0


def add_runtime_session_receipt_commands(runtime_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    receipt = runtime_sub.add_parser(
        "session-receipt",
        help="Ingest or list host-supplied session_activity_receipt/v1 payloads. OMH validates what it is handed; it collects nothing.",
    )
    receipt_sub = receipt.add_subparsers(dest="session_receipt_command", required=True)

    ingest = receipt_sub.add_parser(
        "ingest",
        help="Validate one supplied receipt payload, record it once, and project consumer evidence.",
    )
    ingest.add_argument("--input", required=True, help="Path to a session_activity_receipt/v1 JSON payload.")
    ingest.add_argument(
        "--expected-profile",
        default="",
        help="Profile ref this store is bound to; a receipt naming another profile is quarantined.",
    )
    ingest.add_argument(
        "--max-age-seconds",
        type=int,
        default=None,
        help="Quarantine a receipt whose observed interval ended more than this many seconds ago.",
    )
    ingest.add_argument(
        "--redact-skill-names",
        action="store_true",
        help="Store skill names as digest handles instead of plain labels.",
    )
    ingest.add_argument(
        "--consumer",
        action="append",
        choices=sorted(SESSION_ACTIVITY_CONSUMERS),
        help="Project the admitted receipt as evidence for this workflow; repeatable.",
    )
    ingest.add_argument("--dry-run", action="store_true", help="Admit without writing anything.")
    ingest.set_defaults(func=cmd_runtime_session_receipt_ingest)

    listing = receipt_sub.add_parser("list", help="List recorded receipts and quarantined payload handles.")
    listing.add_argument("--session", default=None, help="Only receipts for this session ref.")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true", help="Emit JSON (the default).")
    listing.set_defaults(func=cmd_runtime_session_receipt_list)
