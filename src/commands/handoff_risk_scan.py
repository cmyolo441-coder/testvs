"""Agent/operator front door for advisory analysis of a final handoff."""
from __future__ import annotations

import argparse
from io import TextIOWrapper
from typing import assert_never
from pathlib import Path
import sys

from ..quality.handoff_risk_model import MAX_BRIEF_BYTES, ScanError, ScanInput, error_report
from ..quality.handoff_risk_scan import scan_handoff
from .common import _print_json, _wants_json


class ScanArguments(argparse.Namespace):
    """The closed argument shape guaranteed by this command's parser."""
    brief_file: str | None = None
    brief_stdin: bool = False
    repo: str | None = None
    protected_branch: list[str] | None = None
    strict: bool = False
    json: bool = False


def cmd_handoff_risk_scan(args: ScanArguments) -> int:
    try:
        brief = None
        if args.brief_file is not None:
            with Path(args.brief_file).open("rb") as stream:
                brief = stream.read(MAX_BRIEF_BYTES + 1)
        elif args.brief_stdin:
            brief = (bytes(sys.stdin.buffer.read(MAX_BRIEF_BYTES + 1)) if isinstance(sys.stdin, TextIOWrapper)
                     else sys.stdin.read(MAX_BRIEF_BYTES + 1).encode("utf-8"))
        report = scan_handoff(ScanInput(
            brief=brief, repo=Path(args.repo) if args.repo is not None else None,
            protected_branches=tuple(args.protected_branch or ("main", "master")),
        ))
    except OSError:
        report = error_report(ScanError("input_unreadable"))
    if _wants_json(args):
        _print_json(report)
    else:
        print(f"OMH handoff risk scan: {report['verdict'] or report['status']}")
        for item in report["findings"]:
            print(f"  {item['severity']}: {item['id']} -> {item['advice_code']}")
        if report["error_category"]:
            print(f"  {report['error_category']}")
        print(report["claim_boundary"])
    match report["status"]:
        case "scan_error":
            return 2
        case "completed":
            return int(args.strict and report["verdict"] == "high_risk")
        case unreachable:
            assert_never(unreachable)


def add_handoff_risk_scan_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser("handoff-risk-scan", help="Agent/operator advisory scan of a composed handoff; never dispatches.")
    brief = parser.add_mutually_exclusive_group()
    _ = brief.add_argument("--brief-file", metavar="PATH")
    _ = brief.add_argument("--brief-stdin", action="store_true")
    _ = parser.add_argument("--repo", metavar="PATH")
    _ = parser.add_argument("--protected-branch", action="append", metavar="NAME", help="Exact branch; repeat to replace main/master defaults.")
    _ = parser.add_argument("--strict", action="store_true", help="Exit 1 on high_risk; scan errors always exit 2.")
    _ = parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=cmd_handoff_risk_scan)
