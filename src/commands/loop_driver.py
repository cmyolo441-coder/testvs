"""Explicit local driver transfer and migration command adapters.

Both handlers go through the same typed Loop operation service the rest of
`omh loop` and the `omh_loop` plugin tool use; only the argparse shape, the
observation file read, and the printed keys stay here.
"""
from __future__ import annotations

import argparse

from ..core.errors import OmhError
from ..workflows.loop_observation_input import read_loop_observation_json
from ..workflows.loop_operations import LoopOperationRequest, run_loop_operation
from .common import _paths, _print_json, add_revision_guard_arguments


def _run(args: argparse.Namespace, action: str, **fields: object) -> dict[str, object]:
    request = LoopOperationRequest(
        action=action,
        loop_id=str(args.loop_id),
        fields=fields,
        expected_revision=args.expected_revision,
        mutation_id=str(getattr(args, "mutation_id", "") or ""),
    )
    return run_loop_operation(_paths(args), request).artifacts


def cmd_loop_driver_bind(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "driver_bind", binding_observation=read_loop_observation_json(args.input)))
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_migrate_driver(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "migrate_driver", apply=args.apply))
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def add_driver_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    binding = sub.add_parser("driver-bind")
    binding.add_argument("--loop", dest="loop_id", required=True)
    binding.add_argument("--input", required=True)
    add_revision_guard_arguments(binding)
    binding.set_defaults(func=cmd_loop_driver_bind)
    migration = sub.add_parser("migrate-driver")
    migration.add_argument("--loop", dest="loop_id", required=True)
    migration.add_argument("--apply", action="store_true")
    add_revision_guard_arguments(migration)
    migration.set_defaults(func=cmd_loop_migrate_driver)
