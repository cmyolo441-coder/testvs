from __future__ import annotations

import argparse
from pathlib import Path

from ..installer import OmhError
from ..workflows.external_rule_import import (
    ExternalRuleImportError,
    apply_rule_import,
    plan_rule_import,
    public_plan,
)
from .common import _paths, _print_json


def cmd_ops_rules_import(args: argparse.Namespace) -> int:
    paths = _paths(args)
    repo_root = Path(args.repo_root).expanduser().resolve()
    try:
        plan = plan_rule_import(
            repo_root,
            skills_dir=paths.skills_dir,
            omh_home=paths.omh_home,
            only_sources=tuple(args.source or ()),
        )
        if args.apply:
            _print_json(apply_rule_import(plan, skills_dir=paths.skills_dir, omh_home=paths.omh_home))
        else:
            _print_json({**public_plan(plan), "applied": False, "hint": "re-run with --apply to write the planned imports"})
    except ExternalRuleImportError as exc:
        raise OmhError(str(exc)) from exc
    except OSError as exc:
        raise OmhError(f"rules import failed: {exc}") from exc
    return 0


def add_ops_rules_import_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    rules_import = ops_sub.add_parser(
        "rules-import",
        help=(
            "Import externally authored agent rules (.cursorrules, .cursor/rules/*.mdc, "
            ".clinerules, .windsurfrules, copilot-instructions) as OMH imported skills; "
            "dry-run by default, --apply writes."
        ),
    )
    rules_import.add_argument("--repo-root", required=True, help="Repository root to discover rule sources under.")
    rules_import.add_argument(
        "--source",
        action="append",
        help="Limit the import to this discovered source path (relative to --repo-root); repeatable.",
    )
    rules_import.add_argument("--apply", action="store_true", help="Write the planned imports (default is dry-run).")
    rules_import.set_defaults(func=cmd_ops_rules_import)
