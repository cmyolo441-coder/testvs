from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..installer import OmhError
from ..workflows.realtime_voice_trial_receipts import (
    READINESS_DIMENSIONS,
    READINESS_STATE_KEYS,
    RealtimeVoiceTrialError,
    answer_realtime_voice_readiness,
    validate_realtime_voice_trial_receipt,
)
from .common import _print_json, _wants_json


def _render_text(answer: dict[str, object]) -> str:
    dimensions = answer["dimensions"]
    states = answer["states"]
    lines = [
        f"Verdict: {answer['verdict']} ({answer['test_condition']})",
        f"Trial: {answer['trial_ref']} on connector {answer['connector_ref']} revision {answer['connector_revision']}",
        "",
        "Dimensions:",
    ]
    for name in READINESS_DIMENSIONS:
        dimension = dimensions[name]
        lines.append(f"  {name}: {dimension['state']}")
        for reason in dimension["reasons"]:
            lines.append(f"    - {reason}")
    lines.extend(["", "States:"])
    for key in READINESS_STATE_KEYS:
        lines.append(f"  {key}: {'yes' if states[key] else 'no'}")
    lines.extend(["", "Reasons:"])
    lines.extend(f"  - {reason}" for reason in answer["verdict_reasons"])
    lines.extend(["", str(answer["claim_boundary"])])
    return "\n".join(lines)


def cmd_ops_realtime_voice_readiness(args: argparse.Namespace) -> int:
    try:
        receipt = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OmhError(str(exc)) from exc
    receipt_errors = validate_realtime_voice_trial_receipt(receipt)
    if receipt_errors:
        raise OmhError("; ".join(receipt_errors))
    try:
        answer = answer_realtime_voice_readiness(
            receipt,
            connector_revision=args.connector_revision,
            profile_ref=args.profile,
            now=args.now,
        )
    except RealtimeVoiceTrialError as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(answer)
    else:
        print(_render_text(answer))
    return 0


def add_ops_realtime_voice_readiness_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    readiness = ops_sub.add_parser(
        "realtime-voice-readiness",
        help="Summarize a supplied realtime_voice_trial_receipt/v1 into a per-dimension readiness verdict.",
        description=(
            "Read one realtime voice trial receipt an authorized host, connector, or operator produced and "
            "report turn integrity, latency, fallback, interruption, and spoken tool safety separately. OMH "
            "opens no microphone, call, room, socket, or provider session and runs no part of the trial."
        ),
    )
    readiness.add_argument("--input", required=True, help="Path to a realtime_voice_trial_receipt/v1 JSON file.")
    readiness.add_argument(
        "--connector-revision",
        default="",
        help="Block the verdict unless the receipt observed this exact connector build.",
    )
    readiness.add_argument(
        "--profile",
        default="",
        help="Block the verdict unless the receipt observed this exact profile.",
    )
    readiness.add_argument(
        "--now",
        default="",
        help="ISO-8601 reference time; a trial older than the freshness horizon blocks instead of passing.",
    )
    readiness.add_argument("--json", action="store_true", help="Emit the machine payload instead of plain text.")
    readiness.set_defaults(func=cmd_ops_realtime_voice_readiness)
