from __future__ import annotations

import argparse
from pathlib import Path

from ..installer import OmhError
from ..workflows.mcp_tool_name_compatibility import audit_mcp_tool_name_compatibility
from .common import _print_json


def cmd_harness_mcp_tool_name_compatibility(args: argparse.Namespace) -> int:
    try:
        payload = audit_mcp_tool_name_compatibility(tuple(Path(value) for value in args.snapshot))
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(payload)
    return 0


def add_harness_mcp_tool_name_compatibility_command(harness_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    audit = harness_sub.add_parser(
        "mcp-tool-name-compatibility",
        help="Audit supplied MCP tool-name snapshots before handoff without invoking or registering tools.",
    )
    audit.add_argument(
        "--snapshot", action="append", required=True,
        help="Explicit local metadata snapshot path; repeat for distinct target harnesses. No host discovery.",
    )
    audit.set_defaults(func=cmd_harness_mcp_tool_name_compatibility)
