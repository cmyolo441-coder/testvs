from __future__ import annotations

import argparse
from pathlib import Path

from ..installer import OmhError
from ..workflows.prompt_compatibility import audit_prompt_compatibility
from .common import _print_json


def cmd_ops_prompt_compatibility_audit(args: argparse.Namespace) -> int:
    try:
        payload = audit_prompt_compatibility(
            tuple(Path(value) for value in args.path),
            existing_commands=tuple(args.existing_command or ()),
        )
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(payload)
    return 0


def add_ops_prompt_compatibility_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    audit = ops_sub.add_parser(
        "prompt-compatibility-audit",
        help="Audit explicitly named local prompt files without importing or registering slash commands.",
    )
    audit.add_argument("--path", action="append", required=True, help="Explicit local prompt file path to audit; directories are not discovered.")
    audit.add_argument(
        "--existing-command",
        action="append",
        help="Existing slash-command name to check for a normalized candidate collision.",
    )
    audit.set_defaults(func=cmd_ops_prompt_compatibility_audit)
