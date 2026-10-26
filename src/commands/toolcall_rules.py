from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..installer import OmhError
from ..plugin_bundle.omh.toolcall_rules import (
    MAX_RULES_FILE_BYTES,
    TOOLCALL_RULES_SCHEMA_VERSION,
    toolcall_rules_path,
    validate_toolcall_rules_document,
)
from .common import _print_json


def cmd_ops_toolcall_rules_validate(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser() if args.path else toolcall_rules_path(args.omh_home or "")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise OmhError(
            f"toolcall rules file not found: {path} (the file's presence is the opt-in; "
            f"create it with schema_version {TOOLCALL_RULES_SCHEMA_VERSION})"
        ) from exc
    except (OSError, ValueError) as exc:
        raise OmhError(f"could not parse toolcall rules file {path}: {exc}") from exc
    errors, accepted = validate_toolcall_rules_document(raw)
    size = path.stat().st_size
    if size > MAX_RULES_FILE_BYTES:
        # The enforcing hook refuses the whole file above this bound, so a
        # "valid" verdict here would certify rules the hook never loads.
        errors.append(
            f"rules file is {size} bytes; the enforcing hook ignores files over "
            f"{MAX_RULES_FILE_BYTES} bytes, so no rule is loaded"
        )
        accepted = 0
    _print_json(
        {
            "schema_version": TOOLCALL_RULES_SCHEMA_VERSION,
            "rules_path": str(path),
            "valid": not errors,
            "accepted_rules": accepted,
            "errors": errors,
            "claim_boundary": (
                "Validation proves the rules file parses; it is not evidence any rule "
                "matched, blocked a call, or improved an outcome."
            ),
        }
    )
    return 0 if not errors else 1


def add_ops_toolcall_rules_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    validate = ops_sub.add_parser(
        "toolcall-rules-validate",
        help=(
            "Validate the user-authored toolcall rules file the plugin enforces at "
            "pre_tool_call; the enforcing hook fails open, so this is where a "
            "silently skipped rule becomes visible."
        ),
    )
    validate.add_argument(
        "--path",
        help="Explicit rules file to validate (default: <omh-home>/rules/toolcall-rules.json).",
    )
    validate.add_argument(
        "--omh-home",
        default="",
        help="OMH home used to resolve the default rules path.",
    )
    validate.set_defaults(func=cmd_ops_toolcall_rules_validate)
