"""CLI surface for the live coding status board (`omh coding status-board`).

Thin adapter only: `omh.coding.status_board` owns the projection, the claim
boundary, and the renderer, so this module resolves paths, validates the limit,
and picks between the text board and the machine payload. Keeping the
projection out of here is what lets the messenger body and the CLI board render
from the same payload instead of drifting into two boards.

Plain text is the default because this command answers a human question ("what
is running right now?"); `--json` is the opt-in for an agent or a script.
"""

from __future__ import annotations

import argparse

from ..coding.status_board import DEFAULT_LIMIT, build_status_board, render_status_board_text
from ..installer import OmhError
from .common import _paths, _print_json, _wants_json


def cmd_coding_status_board(args: argparse.Namespace) -> int:
    payload = build_status_board(_paths(args), limit=_status_board_limit(args))
    if _wants_json(args):
        _print_json(payload)
    else:
        print(render_status_board_text(payload))
    return 0


def _status_board_limit(args: argparse.Namespace) -> int:
    """Row cap for the board, refused rather than coerced when it is not positive.

    `build_status_board` treats any non-positive limit as zero and returns an
    empty `units` list while still reporting a non-zero `unit_count`. Silently
    accepting `--limit 0` would therefore print "0 running of 3 observed" with
    no rows and read as "nothing is running", which is the opposite of what the
    board observed.
    """
    limit = int(getattr(args, "limit", DEFAULT_LIMIT))
    if limit < 1:
        raise OmhError("--limit must be at least 1")
    return limit


def add_coding_status_board_command(coding_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    status_board = coding_sub.add_parser(
        "status-board",
        help=(
            "Show every locally observed coding unit, plus Hermes delegate children and kanban workers "
            "read from Hermes' state.db and kanban.db, as one board (read-only, reporting-only, metadata-only)."
        ),
    )
    status_board.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum rows to display in each table, units and Hermes workers (default: {DEFAULT_LIMIT}).",
    )
    status_board.add_argument("--json", action="store_true", help="Emit the machine payload instead of plain text.")
    status_board.set_defaults(func=cmd_coding_status_board)
