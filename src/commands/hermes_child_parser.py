"""Argument-parser registration for Hermes-child commands."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

HermesChildHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class HermesChildHandlers:
    """CLI handlers supplied by the command module without an import cycle."""

    prepare: HermesChildHandler
    dispatch: HermesChildHandler
    skill_load_probe: HermesChildHandler
    skill_load_status: HermesChildHandler
    status: HermesChildHandler
    cancel: HermesChildHandler


def _add_request_arguments(parser: argparse.ArgumentParser, *, dispatch: bool) -> None:
    parser.add_argument("--prompt-file", default="-", help="Prompt file, or '-' for stdin (default). Prompt text is never accepted on argv.")
    parser.add_argument("--model", required=True, help="Hermes model alias metadata and --model value.")
    parser.add_argument("--provider", required=True, help="Provider alias metadata (not a credential).")
    parser.add_argument("--reasoning", required=True, help="Reasoning alias metadata.")
    parser.add_argument("--parent-run-id", required=True, help="Opaque parent run id.")
    parser.add_argument("--run-id", required=True, help="Opaque isolated child run id.")
    parser.add_argument("--json", action="store_true", help="Emit routing_observation/v1 JSON instead of status rows.")
    if dispatch:
        parser.add_argument("--confirm-dispatch", action="store_true", help="Required explicit approval to start local Hermes.")
        parser.add_argument("--hermes", default="hermes", help="Hermes CLI executable path.")
        parser.add_argument("--cwd", default=None, help="Child working directory.")
        parser.add_argument("--timeout", type=float, default=900.0, help="Hard child timeout in seconds.")
        parser.add_argument("--termination-grace", type=float, default=2.0, help="SIGTERM grace before SIGKILL.")


def configure_hermes_child_parser(
    coding_sub: argparse._SubParsersAction[argparse.ArgumentParser],
    handlers: HermesChildHandlers,
) -> None:
    """Register command handlers after the command module has initialized."""
    child = coding_sub.add_parser(
        "hermes-child",
        help="Agent/maintainer-only explicit isolated Hermes child control (never automatic).",
        description=(
            "AUDIENCE: agent/maintainer. Prepare is non-executing; dispatch calls only the isolated "
            "Hermes child module and requires --confirm-dispatch. Prompts come only from stdin/files."
        ),
    )
    actions = child.add_subparsers(dest="hermes_child_action", required=True)
    prepare = actions.add_parser("prepare", help="Default-safe metadata-only preparation; starts no process.")
    _add_request_arguments(prepare, dispatch=False)
    prepare.set_defaults(func=handlers.prepare)
    dispatch = actions.add_parser("dispatch", help="Explicitly dispatch one bounded local Hermes single-query child (`chat --query-file -`).")
    _add_request_arguments(dispatch, dispatch=True)
    dispatch.set_defaults(func=handlers.dispatch)
    probe = actions.add_parser(
        "skill-load-probe",
        help="Explicitly probe a nonce-bound machine skill inventory; unsupported hosts stay unsupported.",
    )
    probe.add_argument("--confirm-dispatch", action="store_true", help="Required explicit approval to start the local inventory probe.")
    probe.add_argument("--expected-skill", action="append", default=[], help="Expected skill name; repeat for multiple skills. An empty set is valid only after a protocol response.")
    probe.add_argument("--run-id", required=True, help="Opaque isolated probe run id.")
    probe.add_argument("--hermes", default="hermes", help="Hermes CLI executable path.")
    probe.add_argument("--timeout", type=float, default=10.0, help="Hard inventory protocol timeout in seconds.")
    probe.add_argument("--termination-grace", type=float, default=0.25, help="SIGTERM grace before SIGKILL.")
    probe.add_argument("--json", action="store_true", help="Emit skill_load_observation/v1 JSON.")
    probe.set_defaults(func=handlers.skill_load_probe)
    probe_status = actions.add_parser(
        "skill-load-status",
        help="Read a fresh authenticated skill_load_observation/v1 record.",
    )
    probe_status.add_argument("--run-id", required=True)
    probe_status.add_argument("--json", action="store_true")
    probe_status.set_defaults(func=handlers.skill_load_status)
    status = actions.add_parser("status", help="Read the metadata-only routing observation for one child run.")
    status.add_argument("--run-id", required=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=handlers.status)
    cancel = actions.add_parser("cancel", help="Signal an active foreground dispatcher and its isolated process group.")
    cancel.add_argument("--run-id", required=True)
    cancel.add_argument("--json", action="store_true")
    cancel.set_defaults(func=handlers.cancel)
