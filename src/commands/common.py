from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..ingress import extract_message_text, extract_source_metadata
from ..installer import OmhError
from ..system.tracker_content import loads_tracker_event_json
from ..routing.action_copy import next_action_label, next_action_label_with_id
from ..setup_profiles import read_setup_profile
from ..targets import TARGET_METADATA_KEYS
from .bare_launch import _paths as _paths  # one definition, shared with the bare launch


def _command_hermes_session_id() -> str:
    """The Hermes session this command was started for, which names the memory lens its handoff reads.

    For a command only, never in process: Hermes builds a command's
    environment from the session's own context (`tools/environments/local.py`,
    `_inject_session_context_env`), so a command sees its own session's
    `HERMES_SESSION_ID`, while inside a gateway process the variable belongs
    to whichever agent was built or compressed last.
    """
    return os.environ.get("HERMES_SESSION_ID", "").strip()


def add_revision_guard_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the two optional stale-mutation guard flags to one mutation parser.

    Shared by every CLI surface that reaches a guarded record write, so the
    flags are spelled and defaulted identically: the guarantee the guard
    offers is "the wrapper submits the revision it rendered", and a surface
    that reaches a guarded write without these flags cannot arm it at all.
    Absent flags stay None / "" so they mean "no guard requested" rather than
    revision 0 or an empty mutation id.
    """
    parser.add_argument(
        "--expected-revision",
        type=int,
        default=None,
        help="Reject this mutation when the record_revision no longer matches the rendered state.",
    )
    parser.add_argument(
        "--mutation-id",
        default="",
        help="Client-chosen id that makes a retried mutation replay the original result instead of duplicating it.",
    )


JSON_PRETTY_ENV = "OMH_JSON_PRETTY"
_JSON_PRETTY_TRUE = {"1", "true", "yes", "on"}
_json_pretty_requested = False


def set_json_output_pretty(enabled: bool) -> None:
    """Record the explicit `--pretty` opt-in for human-readable stdout."""
    global _json_pretty_requested
    _json_pretty_requested = bool(enabled)


def json_output_is_pretty() -> bool:
    if _json_pretty_requested:
        return True
    return os.environ.get(JSON_PRETTY_ENV, "").strip().lower() in _JSON_PRETTY_TRUE


def _print_json(data: object) -> None:
    # Agent-facing stdout is compact by default: indentation is a flat multiplier
    # on every supervising-session context window. `--pretty`/OMH_JSON_PRETTY=1
    # restores the human-readable form.
    if json_output_is_pretty():
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    print(json.dumps(data, sort_keys=True, separators=(",", ":")))


def _wants_json(args: argparse.Namespace) -> bool:
    output = os.environ.get("OMH_OUTPUT", "").strip().lower()
    return bool(getattr(args, "json", False)) or output == "json"


def _action_label_with_id(action: str, label: str = "") -> str:
    return next_action_label_with_id(action, label)


def _action_label(action: str, label: str = "") -> str:
    normalized = action.strip()
    if not normalized:
        return ""
    resolved_label = label.strip() or next_action_label(normalized)
    return resolved_label or normalized


def _explicit_source_metadata(args: argparse.Namespace) -> dict[str, str]:
    values = {
        "source_event_id": getattr(args, "source_event_id", ""),
        "channel_ref": getattr(args, "channel_ref", ""),
        "user_ref": getattr(args, "user_ref", ""),
        "render_profile": getattr(args, "render_profile", ""),
    }
    values.update({key: getattr(args, key, "") for key in TARGET_METADATA_KEYS if key != "hermes_home"})
    return {key: str(value) for key, value in values.items() if value}


def _resolved_executor(args: argparse.Namespace, *, default: str) -> str:
    explicit = getattr(args, "executor", None)
    if explicit:
        return str(explicit)
    try:
        profile = read_setup_profile(_paths(args))
    except (OSError, json.JSONDecodeError, ValueError):
        profile = None
    if isinstance(profile, dict):
        executor = str(profile.get("default_executor", ""))
        if executor:
            return executor
    return default


def _chat_input_and_metadata(args: argparse.Namespace) -> tuple[dict[str, object] | str, dict[str, str]]:
    try:
        if args.event_json:
            raw = (
                sys.stdin.read()
                if args.event_json == "-"
                else Path(args.event_json).expanduser().read_text(encoding="utf-8")
            )
            event = loads_tracker_event_json(raw)
            metadata = extract_source_metadata(event)
            metadata.update(_explicit_source_metadata(args))
            return event, metadata
        if args.stdin:
            return sys.stdin.read().strip(), _explicit_source_metadata(args)
        return " ".join(args.message).strip(), _explicit_source_metadata(args)
    except (OSError, json.JSONDecodeError, ValueError):
        raise


def _chat_message(args: argparse.Namespace) -> str:
    try:
        if args.event_json:
            raw = (
                sys.stdin.read()
                if args.event_json == "-"
                else Path(args.event_json).expanduser().read_text(encoding="utf-8")
            )
            return extract_message_text(loads_tracker_event_json(raw))
        if args.stdin:
            return sys.stdin.read().strip()
        return " ".join(args.message).strip()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
