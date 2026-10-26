from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Any

from ..hud import build_hud_payload
from ..install.config_adapter import display_skin_selection, read_config
from ..plugin_bundle.omh.delegation_view import View, render_delegation_view
from ..skin_pack import SkinTheme, default_theme, effective_theme
from ..tui.panels import dag_line
from ..tui.screen import paint_line
from ..tui.style import NO_COLOR, color_depth
from ..tui.terminal import background_from_colorfgbg, colorfgbg_names_dark
from ..tui.theme import build_theme
from ..tui.width import line_width
from .common import _paths, _print_json, _wants_json

# Every glyph and line character the Unicode drawing can use: an output that cannot encode them gets ASCII.
DELEGATION_VIEW_PROBE = "┌┐└┘─│┬┴├┤┼▼◆●◌○◇✓✗▪·…!←→↻↓"


def cmd_hud(args: argparse.Namespace) -> int:
    payload = _hud_payload(args)
    if _wants_json(args):
        _print_json(payload)
    else:
        _print_text(payload, args)
    if not getattr(args, "watch", False):
        return 0
    while True:
        time.sleep(max(0.2, float(args.interval)))
        payload = _hud_payload(args)
        if _wants_json(args):
            _print_json(payload)
        else:
            _print_text(payload, args)


def _print_text(payload: dict[str, Any], args: argparse.Namespace) -> None:
    """The HUD line; with ``--graph``, the delegation view under it, sized to the terminal."""
    lines = [payload["display"]["line"]]
    if getattr(args, "graph", False):
        graph = payload.get("delegation_graph")
        active = isinstance(graph, dict) and graph.get("status") == "active"
        view = render_delegation_view(
            graph if active else None,
            width=shutil.get_terminal_size((100, 24)).columns,
            charset="unicode" if _encodes(DELEGATION_VIEW_PROBE) else "ascii",
        )
        if view.lines and graph.get("scope") == "global":
            lines.append("[global] the newest conversation with delegated work")
        lines += _view_lines(view, args)
    print("\n".join(lines), flush=True)


def _view_lines(view: View, args: argparse.Namespace) -> list[str]:
    """The view in the OMH TUI's colours on a terminal that takes them, else as plain text.

    Colours are measured against the background, and a printed command cannot
    ask the terminal for it as the TUI does (OSC 11): it paints only where
    `COLORFGBG` names a light or a dark one, never a guess that could put a
    light line on a light background.
    """
    depth = color_depth(os.environ, sys.stdout.isatty())
    light = background_from_colorfgbg(os.environ)
    if depth == NO_COLOR or (light is None and not colorfgbg_names_dark(os.environ)):
        return view.text_lines()
    theme = build_theme(_active_skin(args), light)  # None: the palette's own dark background stands in
    return [
        paint_line(line, line_width(line)) for line in (dag_line(row, view.levels, theme, depth) for row in view.lines)
    ]


def _active_skin(args: argparse.Namespace) -> SkinTheme:
    """The palette the OMH TUI opens with (`effective_theme`); the default when Hermes' config cannot be read."""
    paths = _paths(args)
    try:
        hermes_skin = display_skin_selection(read_config(paths.hermes_config_path))
    except (OSError, UnicodeDecodeError):
        return default_theme()
    return effective_theme(paths.omh_home, hermes_skin)[0]


def _encodes(text: str) -> bool:
    """Whether stdout says it can encode ``text``; an output that names no encoding is taken as ASCII."""
    try:
        text.encode(getattr(sys.stdout, "encoding", None) or "ascii")
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _hud_payload(args: argparse.Namespace) -> dict[str, Any]:
    return build_hud_payload(
        _paths(args),
        preset=args.preset,
        limit=args.limit,
        token_metadata=_token_metadata(args),
        graph_preference="on" if getattr(args, "graph", False) else "auto",
    )


def _token_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "tokens_remaining": args.tokens_remaining,
            "token_budget": args.token_budget,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "context_remaining_percent": args.context_remaining_percent,
        }.items()
        if value is not None
    }


def _add_hud_commands(sub) -> None:
    hud = sub.add_parser("hud", help="Print the compact OMH status line that the OMH TUI and plugin surfaces can render.")
    hud.add_argument("--preset", choices=("minimal", "focused", "full"), default="focused")
    hud.add_argument("--limit", type=int, default=3, help="Maximum recent runtime runs to inspect.")
    hud.add_argument("--tokens-remaining", type=float, default=None, help="Optional host-provided token metadata.")
    hud.add_argument("--token-budget", type=float, default=None, help="Optional host-provided token budget metadata.")
    hud.add_argument("--input-tokens", type=float, default=None, help="Optional host-provided input token metadata.")
    hud.add_argument("--output-tokens", type=float, default=None, help="Optional host-provided output token metadata.")
    hud.add_argument("--context-remaining-percent", type=float, default=None, help="Optional host-provided context remaining percentage.")
    hud.add_argument("--watch", action="store_true", help="Keep printing the HUD line until interrupted.")
    hud.add_argument("--interval", type=float, default=2.0, help="Seconds between --watch refreshes.")
    hud.add_argument(
        "--graph",
        action="store_true",
        help="Print the delegation graph under the HUD line: recorded delegations, team parts or a fanout.",
    )
    hud.add_argument("--json", action="store_true", help="Print the full machine-readable HUD payload.")
    hud.set_defaults(func=cmd_hud)
