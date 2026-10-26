from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from ..installer import OmhError
from ..menubar_app import setup_menubar_app, start_menubar_app, stop_menubar_app, uninstall_menubar_app
from ..menubar_status import build_menubar_status_payload, read_process_overlay_file
from .common import _paths, _print_json, _wants_json


_TABLE_VALUE_DISPLAY_LIMIT = 48


def cmd_menubar_status(args: argparse.Namespace) -> int:
    try:
        overlay = read_process_overlay_file(args.overlay) if args.overlay else None
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    payload = build_menubar_status_payload(
        _paths(args),
        limit=args.limit,
        process_overlay=overlay,
        observe_local_processes=bool(args.observe_local_processes),
        now=args.now,
    )
    if _wants_json(args):
        _print_json(payload)
    else:
        print(_format_menubar_status(payload))
    return 0


def cmd_menubar_install(args: argparse.Namespace) -> int:
    try:
        payload = setup_menubar_app(
            _paths(args),
            dry_run=bool(args.dry_run),
            start=not bool(args.no_start),
            force=bool(args.force),
        )
    except RuntimeError as exc:
        raise OmhError(str(exc)) from exc
    _print_menubar_app_payload(args, payload)
    return 0


def cmd_menubar_start(args: argparse.Namespace) -> int:
    _print_menubar_app_payload(args, start_menubar_app(_paths(args)))
    return 0


def cmd_menubar_stop(args: argparse.Namespace) -> int:
    _print_menubar_app_payload(args, stop_menubar_app(_paths(args)))
    return 0


def cmd_menubar_uninstall(args: argparse.Namespace) -> int:
    _print_menubar_app_payload(args, uninstall_menubar_app(_paths(args), dry_run=bool(args.dry_run)))
    return 0


def _add_menubar_commands(sub) -> None:
    menubar = sub.add_parser("menubar", help="Install or inspect the OMH menu bar helper.")
    menubar_sub = menubar.add_subparsers(dest="menubar_command", required=True)

    status = menubar_sub.add_parser("status", help="Show the OMH menu bar status summary.")
    status.add_argument("--limit", type=int, default=3, help="Maximum recent runtime runs to inspect.")
    status.add_argument("--overlay", default=None, help="Optional menubar_process_overlay/v1 JSON from the menu bar app.")
    status.add_argument(
        "--observe-local-processes",
        action="store_true",
        help="Opt in to local Hermes process observation for the native menu bar helper.",
    )
    status.add_argument("--now", default=None, help="Optional ISO timestamp for deterministic overlay TTL evaluation.")
    status.add_argument("--json", action="store_true", help="Print the full menubar_status/v2 payload.")
    status.set_defaults(func=cmd_menubar_status)

    install = menubar_sub.add_parser("install", help="Install and start the native macOS OMH menu bar helper when supported.")
    install.add_argument("--dry-run", action="store_true")
    install.add_argument("--force", action="store_true")
    install.add_argument("--no-start", action="store_true", help="Install the helper without starting the LaunchAgent.")
    install.add_argument("--json", action="store_true", help="Print the full menubar_app/v1 payload.")
    install.set_defaults(func=cmd_menubar_install)

    start = menubar_sub.add_parser("start", help="Start the installed OMH menu bar helper.")
    start.add_argument("--json", action="store_true", help="Print the full menubar_app/v1 payload.")
    start.set_defaults(func=cmd_menubar_start)

    stop = menubar_sub.add_parser("stop", help="Stop the OMH menu bar helper LaunchAgent.")
    stop.add_argument("--json", action="store_true", help="Print the full menubar_app/v1 payload.")
    stop.set_defaults(func=cmd_menubar_stop)

    uninstall = menubar_sub.add_parser("uninstall", help="Stop and remove the OMH menu bar helper.")
    uninstall.add_argument("--dry-run", action="store_true")
    uninstall.add_argument("--json", action="store_true", help="Print the full menubar_app/v1 payload.")
    uninstall.set_defaults(func=cmd_menubar_uninstall)


def _format_menubar_status(payload: Mapping[str, object]) -> str:
    display = _as_mapping(payload.get("display"))
    settings = _as_mapping(payload.get("settings"))
    versions = _as_mapping(payload.get("versions"))
    process_overlay = _as_mapping(payload.get("process_overlay"))
    hermes_sessions = _as_mapping(payload.get("hermes_sessions"))
    hermes_processes = _as_mapping(payload.get("hermes_processes"))

    lines = ["OMH menu bar status", "Summary"]
    headline = _text(display.get("headline")) or "OMH status"
    summary = _text(display.get("summary_line"))
    lines.append(f"  Status: {headline}")
    if summary:
        lines.append(f"  Activity: {summary}")
    if _truthy(hermes_sessions.get("observed")):
        stale = _text(hermes_sessions.get("stale"))
        idle = f" / idle {stale}" if stale and stale != "0" else ""
        lines.append(
            f"  Sessions: live {_text(hermes_sessions.get('live'))}{idle} / "
            f"total {_text(hermes_sessions.get('total'))}"
        )
    else:
        lines.append("  Sessions: not observed")
    if _truthy(hermes_processes.get("observed")):
        agent_count = _text(hermes_processes.get("agent_count"))
        process_count = _text(hermes_processes.get("process_count"))
        lines.append(
            f"  Processes: {agent_count} {'agent' if agent_count == '1' else 'agents'} / "
            f"{process_count} {'process' if process_count == '1' else 'processes'}"
        )
    else:
        lines.append("  Processes: not observed")
    omh_version = _as_mapping(versions.get("omh"))
    if omh_version:
        lines.append(f"  OMH version: {_text(omh_version.get('value')) or 'unknown'}")

    for key in ("omh_connection", "hermes_targets", "coding_handoff", "send_mode"):
        setting = _as_mapping(settings.get(key))
        label = _text(setting.get("label")) if setting else ""
        if label:
            lines.append(f"  {label}")

    cards = _as_sequence(display.get("menu_cards"))
    for card_value in cards:
        card = _as_mapping(card_value)
        if not card:
            continue
        title = _text(card.get("title")) or "Status"
        lines.extend(["", title])
        columns = [_text(column) for column in _as_sequence(card.get("columns"))]
        rows = [_as_mapping(row) for row in _as_sequence(card.get("rows"))]
        table_rows = [row for row in rows if row.get("kind") == "table_row"]
        if columns and table_rows:
            lines.append(f"  {_pad(columns[0], 20)} {columns[1]}")
            for row in table_rows:
                lines.append(
                    f"  {_pad(_text(row.get('left')), 20)} "
                    f"{_truncate(_text(row.get('right')), _TABLE_VALUE_DISPLAY_LIMIT)}"
                )
        else:
            for row in rows:
                label = _text(row.get("label"))
                value = _text(row.get("value"))
                detail = _text(row.get("detail"))
                if not label and not value:
                    continue
                line = f"  {label}: {value}" if label and value else f"  {label or value}"
                if detail:
                    line = f"{line} - {detail}"
                lines.append(line)
        footer = _text(card.get("footer"))
        if footer:
            lines.append(f"  Note: {footer}")

    overlay_status = _text(process_overlay.get("status"))
    if overlay_status:
        lines.extend(["", "Observation"])
        lines.append(f"  Process overlay: {overlay_status.replace('_', ' ')}")
        applied_count = _text(process_overlay.get("applied_count"))
        if applied_count and applied_count != "0":
            lines.append(f"  Observed rows: {applied_count}")
        lines.append("  Boundary: configured targets are not PID evidence unless observed by the helper.")

    lines.extend(["", "For machine-readable output, rerun with `--json`."])
    return "\n".join(lines)


def _print_menubar_app_payload(args: argparse.Namespace, payload: Mapping[str, object]) -> None:
    if _wants_json(args):
        _print_json(payload)
    else:
        print(_format_menubar_app_payload(payload))


def _format_menubar_app_payload(payload: Mapping[str, object]) -> str:
    operation = _text(payload.get("operation")) or "menubar"
    status = _text(payload.get("status")) or "unknown"
    message = _text(payload.get("message"))
    lines = [f"OMH menu bar {operation}", "Summary", f"  Status: {status.replace('_', ' ')}"]
    if message:
        lines.append(f"  Message: {_single_line(message)}")

    if _truthy(payload.get("started")):
        lines.append("  Result: helper started")
    if _truthy(payload.get("stopped")):
        lines.append("  Result: helper stopped")

    removed = [_text(item) for item in _as_sequence(payload.get("removed")) if _text(item)]
    would_remove = [_text(item) for item in _as_sequence(payload.get("would_remove")) if _text(item)]
    if removed:
        lines.append(f"  Removed: {len(removed)} path(s)")
    if would_remove:
        lines.append(f"  Would remove: {len(would_remove)} path(s)")

    launch_agent = _text(payload.get("launch_agent"))
    if launch_agent:
        lines.append(f"  LaunchAgent: {launch_agent}")

    if status in {"missing", "failed", "installed_start_failed"}:
        lines.extend(["", "Next"])
        if status == "missing":
            lines.append("  Run `omh menubar install` to install the helper.")
        elif operation == "start":
            lines.append("  Run `omh doctor`, then `omh menubar install --force` if the helper needs repair.")
        elif operation == "stop":
            lines.append("  The helper may already be stopped. Run `omh menubar status` to inspect the current view.")
        else:
            lines.append("  Run `omh doctor` for the local health report.")

    lines.extend(["", "For machine-readable output, rerun with `--json`."])
    return "\n".join(lines)


def _as_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _truthy(value: object) -> bool:
    return bool(value) and str(value).lower() not in {"false", "0", "none", "null"}


def _truncate(value: str, limit: int) -> str:
    if len(value) > limit:
        return f"{value[: max(0, limit - 3)]}..."
    return value


def _pad(value: str, width: int) -> str:
    return _truncate(value, width).ljust(width)
