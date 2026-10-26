"""Shared name-only CLI declarations for fanout child environments."""

from __future__ import annotations

import argparse
import re

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_OWNER_ENV_RE = re.compile(r"^(?P<owner>[a-z0-9-]+):(?P<name>[A-Za-z_][A-Za-z0-9_]{0,127})$")


def add_fanout_environment_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach least-privilege declarations without accepting environment values."""
    parser.add_argument(
        "--owner-env",
        action="append",
        default=None,
        metavar="OWNER:NAME",
        type=_owner_environment_name,
        help="Grant one existing environment variable to one owner (repeatable; names only).",
    )
    parser.add_argument(
        "--project-env",
        action="append",
        default=None,
        metavar="NAME",
        type=_environment_name,
        help="Pass one approved project variable to children and verification (repeatable; names only).",
    )
    parser.add_argument(
        "--verification-env",
        action="append",
        default=None,
        metavar="NAME",
        type=_environment_name,
        help="Grant one existing environment variable to verification only (repeatable; names only).",
    )
    parser.add_argument(
        "--deny-env",
        action="append",
        default=None,
        metavar="NAME",
        type=_environment_name,
        help="Remove one parent-only variable from every child (repeatable; names only).",
    )
    parser.add_argument(
        "--allow-broad-environment",
        action="store_true",
        help="Explicit staged-migration compatibility: inherit the parent environment and mark every receipt.",
    )


def child_environment_policy_from_args(args: argparse.Namespace) -> dict[str, object]:
    """Convert parsed name-only arguments to the dispatch policy shape."""
    owner_capabilities: dict[str, list[str]] = {}
    for entry in getattr(args, "owner_env", None) or ():
        owner, name = entry.split(":", 1)
        owner_capabilities.setdefault(owner, []).append(name)
    return {
        "allow_broad_inheritance": bool(getattr(args, "allow_broad_environment", False)),
        "owner_capabilities": {
            owner: sorted(set(names)) for owner, names in sorted(owner_capabilities.items())
        },
        "project_variables": sorted(set(getattr(args, "project_env", None) or ())),
        "verification_capabilities": sorted(set(getattr(args, "verification_env", None) or ())),
        "denied_names": sorted(set(getattr(args, "deny_env", None) or ())),
    }


def _environment_name(value: str) -> str:
    if _NAME_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("environment variable names must be portable identifiers")
    return value


def _owner_environment_name(value: str) -> str:
    if _OWNER_ENV_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("owner environment grants must be OWNER:NAME")
    return value
