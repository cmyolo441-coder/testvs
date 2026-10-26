from __future__ import annotations

import argparse

from ..catalogs.design_data import (
    DESIGN_DATA_CONTEXTS,
    DESIGN_DATA_KINDS,
    query_design_data,
)
from ..installer import OmhError
from .common import _print_json, _wants_json


DESIGN_EPILOG = """Data kinds:
  palette  Product-context color palettes as named role tokens.
  font     Display/body font pairings with fallbacks and CJK notes.
  ux       UX guidelines with the contexts they apply to and why they hold.

Examples:
  omh design data --kind palette --context fintech
  omh design data --kind font --context mobile
  omh design data --kind ux --context data-viz --json

Boundary:
  Design reference rows are prepared local data. They inform a DESIGN.md contract;
  the contract still gates implementation. No network call, no model call, no
  rendered UI, and no visual-QA evidence is produced here.
"""


def cmd_design_data(args: argparse.Namespace) -> int:
    try:
        payload = query_design_data(args.kind, args.context)
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(payload)
    else:
        _print_design_data_summary(payload)
    return 0


def _print_design_data_summary(payload: dict[str, object]) -> None:
    kind = str(payload["kind"])
    context = str(payload["context"])
    scope = context or "all contexts"
    print(f"Design reference data: {kind} ({scope})")
    print(f"Rows: {payload['count']}")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        print("")
        print("No rows for this kind and context. Try a different --context.")
        print(f"Available contexts: {', '.join(DESIGN_DATA_CONTEXTS)}")
        return
    for row in rows:
        print("")
        if kind == "palette":
            _print_palette_row(row)
        elif kind == "font":
            _print_font_row(row)
        else:
            _print_ux_row(row)
    print("")
    print(str(payload["evidence_boundary"]))
    print("For machine-readable output, rerun with `--json`.")


def _row_header(row: dict[str, object]) -> str:
    contexts = row.get("contexts", [])
    labels = ", ".join(str(item) for item in contexts) if isinstance(contexts, list) else ""
    return f"{row.get('name', '')} [{labels}]"


def _print_palette_row(row: dict[str, object]) -> None:
    print(f"{_row_header(row)} ({row.get('mode', '')})")
    roles = row.get("roles", {})
    if isinstance(roles, dict):
        for role, value in roles.items():
            print(f"- {role}: {value}")
    print(f"Note: {row.get('note', '')}")


def _print_font_row(row: dict[str, object]) -> None:
    print(_row_header(row))
    print(f"- display: {row.get('display_stack', '')}")
    print(f"- body: {row.get('body_stack', '')}")
    print(f"- cjk: {row.get('cjk_note', '')}")
    print(f"Note: {row.get('note', '')}")


def _print_ux_row(row: dict[str, object]) -> None:
    print(_row_header(row))
    print(f"- guideline: {row.get('guideline', '')}")
    print(f"- why: {row.get('rationale', '')}")


def _add_design_commands(sub) -> None:
    design = sub.add_parser(
        "design",
        help="Query curated local design reference data for palettes, font pairings, and UX guidelines.",
        epilog=DESIGN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    design_sub = design.add_subparsers(dest="design_command", required=True)

    data_cmd = design_sub.add_parser(
        "data",
        help="Print design reference rows for one kind, optionally filtered by product context.",
    )
    data_cmd.add_argument(
        "--kind",
        required=True,
        choices=list(DESIGN_DATA_KINDS),
        help="Which design reference data to read.",
    )
    data_cmd.add_argument(
        "--context",
        default="",
        metavar="CONTEXT",
        help=f"Filter rows by product context. Values: {', '.join(DESIGN_DATA_CONTEXTS)}.",
    )
    data_cmd.add_argument("--json", action="store_true", help="Print the full machine-readable design data payload.")
    data_cmd.set_defaults(func=cmd_design_data)


__all__ = [
    "DESIGN_EPILOG",
    "_add_design_commands",
    "cmd_design_data",
]
