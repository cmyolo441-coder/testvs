from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..installer import OmhError
from ..plugin_bundle.omh.cost_receipt import build_cost_receipt
from ..runtime.run_efficiency import build_run_efficiency_report, parse_run_efficiency_input, write_run_efficiency_report
from .common import _paths, _print_json


def cmd_runtime_efficiency_report(args: argparse.Namespace) -> int:
    try:
        raw = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
        receipt = None
        if args.hermes_session:
            # The reader `omh_run_summary` hands a chat: Hermes' own state.db accounting, read-only.
            paths = _paths(args)
            receipt = build_cost_receipt(
                hermes_home=paths.hermes_home, omh_home=paths.omh_home, session_id=args.hermes_session
            )
        report = build_run_efficiency_report(parse_run_efficiency_input(raw), cost_receipt=receipt)
        artifact = write_run_efficiency_report(_paths(args), report) if args.write else {"written": False}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json({**report, "artifact": artifact})
    return 0


def add_runtime_efficiency_command(runtime_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    efficiency = runtime_sub.add_parser(
        "efficiency-report",
        help="Build a local run-efficiency report from supplied context-budget metadata.",
    )
    efficiency.add_argument("--input", required=True, help="Path to a run_efficiency_input/v1 JSON file.")
    efficiency.add_argument(
        "--hermes-session",
        default="",
        help="Hermes session id whose recorded usage (its cost receipt, read from state.db) fills host_usage.",
    )
    efficiency.add_argument("--write", action="store_true", help="Write the deterministic report under the OMH runtime store.")
    efficiency.set_defaults(func=cmd_runtime_efficiency_report)
