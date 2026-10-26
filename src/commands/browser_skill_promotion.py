"""CLI leaf registration for project-local skill promotion.

One registration, mounted twice. `omh web-qa promotion` keeps the browser lane's
shipped spelling (`--trace-id`); `omh learning promotion` reaches the same seven
subcommands for any promotion source (`--source-id`). That is the whole of
#1571: the lifecycle was already complete, it was reachable from one place only.
Both mounts call the same functions, so there is no second lane to keep in
agreement with the first.

A parent command owns registration; this module deliberately does not modify
shared parsers.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from ..install.installer import OmhError
from ..workflows.browser_skill_promotion import (
    BrowserSkillPromotionError, approve_browser_skill_lifecycle,
    approve_browser_skill_removal, approve_browser_skill_rollback,
    browser_skill_promotion_status, promote_approved_browser_skill,
    retry_browser_skill_promotion, review_browser_skill_lifecycle,
    review_browser_skill_removal, review_browser_skill_rollback,
)
from ..workflows.browser_skill_promotion_approval import BrowserSkillPromotionApprovalError
from ..workflows.browser_skill_promotion_plan import BrowserSkillPromotionPlanError
from ..workflows.browser_workflow_learning import BrowserTraceError
from ..workflows.skill_promotion_source import SkillPromotionSourceError
from .common import _print_json


# A promoted skill is visible only while its managed `SKILL.md` validates, so
# each of these statuses means "nothing of yours is installed right now".
_PROMOTION_NOT_INSTALLED_STATUSES = frozenset({"stale", "quarantined", "unverified_managed_state"})


def _skill_promotion_exit_code(payload: Any) -> int:
    """0 only when the lane did not end with the entry refused or deactivated.

    `tests/test_exit_code_truthfulness_policy.py` re-derives every mapper under
    `src/commands/` and fails one that maps a failure signal to 0. The signal
    that matters here is a status meaning the entry is not live: `stale` and
    `quarantined` are reached when drift unlinked the managed `SKILL.md`, and
    `unverified_managed_state` when a `SKILL.md` exists that OMH refuses to own.
    A shell reading only the status after `promote` or `status` would otherwise
    be told the skill is installed at the moment it is not.

    `inactive` stays 0 on purpose: it answers "is this promoted?" for a project
    that never promoted it, which is not a promotion that failed. So do
    `removed` and `already_deactivated`, which are what `remove` was asked for.

    Refusals inside the lifecycle raise `OmhError` and never reach here, so this
    grades only the payloads that do come back. Generic failure signals -- a
    refused or interrupted run, or a unit carrying a failure kind -- are never
    success either, so this mapper cannot be passed by ignoring them.
    """
    summary = payload if isinstance(payload, dict) else {}
    if summary.get("refused") or summary.get("interrupted"):
        return 1
    units = summary.get("units")
    if isinstance(units, list) and any(isinstance(unit, dict) and unit.get("failure_kind") for unit in units):
        return 1
    if str(summary.get("status", "")) in _PROMOTION_NOT_INSTALLED_STATUSES:
        return 1
    return 0


def _run(action: Callable[[], dict[str, object]]) -> int:
    try:
        payload = action()
    except (BrowserSkillPromotionError, BrowserSkillPromotionApprovalError, BrowserSkillPromotionPlanError, BrowserTraceError, SkillPromotionSourceError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(payload)
    return _skill_promotion_exit_code(payload)


def add_browser_skill_promotion_commands(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """The `omh web-qa promotion` mount, spelled as it shipped."""
    add_skill_promotion_commands(
        parent,
        source_flag="--trace-id",
        source_help="Approved, replay-passing browser workflow trace id (bwt-<24 hex>).",
        summary="Review and explicitly promote approved browser traces into this project.",
    )


def add_skill_promotion_commands(
    parent: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    source_flag: str = "--source-id",
    source_help: str = (
        "Promotion source id: an approved browser workflow trace (bwt-<24 hex>) "
        "or a reviewed, activated skill draft (sd-<20 hex>)."
    ),
    summary: str = "Review and explicitly promote one reviewed promotion source into this project.",
) -> None:
    promotion = parent.add_parser("promotion", help=summary)
    commands = promotion.add_subparsers(dest="browser_skill_promotion_command", required=True)
    diff = commands.add_parser("diff", help="Render the exact promotion diff and native preflight.")
    _source_skill(diff, source_flag, source_help); diff.add_argument("--operation", choices=("install", "update"), default="install"); diff.set_defaults(func=lambda args: _run(lambda: review_browser_skill_lifecycle(args.project_root, args.source_id, args.skill_name, operation=args.operation)))
    approve = commands.add_parser("approve", help="Persist exact-diff promotion approval.")
    _source_skill(approve, source_flag, source_help); approve.add_argument("--reviewed-diff-digest", required=True); approve.add_argument("--reviewer", required=True); approve.add_argument("--operation", choices=("install", "update"), default="install"); approve.set_defaults(func=lambda args: _run(lambda: approve_browser_skill_lifecycle(args.project_root, args.source_id, args.skill_name, reviewed_diff_digest=args.reviewed_diff_digest, reviewer_identity=args.reviewer, operation=args.operation)))
    promote = commands.add_parser("promote", help="Activate exactly one approved promotion receipt.")
    _project_only(promote); promote.add_argument("--receipt-id", required=True); promote.set_defaults(func=lambda args: _run(lambda: promote_approved_browser_skill(args.project_root, args.receipt_id)))
    status = commands.add_parser("status", help="Validate actual entry/manifest/approval truth and optionally deactivate drift.")
    _project_skill(status); status.add_argument("--no-source-check", action="store_true"); status.set_defaults(func=lambda args: _run(lambda: browser_skill_promotion_status(args.project_root, args.skill_name, check_source=not args.no_source_check)))
    rollback = commands.add_parser("rollback", help="Review or approve a retained generation rollback.")
    _project_skill(rollback); rollback.add_argument("--generation", required=True); rollback.add_argument("--reviewed-diff-digest"); rollback.add_argument("--reviewer"); rollback.set_defaults(func=_rollback)
    remove = commands.add_parser("remove", help="Review or approve removal of only a verified managed SKILL.md.")
    _project_skill(remove); remove.add_argument("--reviewed-diff-digest"); remove.add_argument("--reviewer"); remove.set_defaults(func=_remove)
    retry = commands.add_parser("retry", help="Explicitly resume an incomplete approved pre-entry promotion.")
    _project_only(retry); retry.add_argument("--receipt-id", required=True); retry.set_defaults(func=lambda args: _run(lambda: retry_browser_skill_promotion(args.project_root, args.receipt_id)))


def _rollback(args: argparse.Namespace) -> int:
    if bool(args.reviewed_diff_digest) != bool(args.reviewer):
        raise OmhError("rollback approval requires both --reviewed-diff-digest and --reviewer")
    if args.reviewed_diff_digest:
        return _run(lambda: approve_browser_skill_rollback(args.project_root, args.skill_name, args.generation, reviewed_diff_digest=args.reviewed_diff_digest, reviewer_identity=args.reviewer))
    return _run(lambda: review_browser_skill_rollback(args.project_root, args.skill_name, args.generation))


def _remove(args: argparse.Namespace) -> int:
    if bool(args.reviewed_diff_digest) != bool(args.reviewer):
        raise OmhError("removal approval requires both --reviewed-diff-digest and --reviewer")
    if args.reviewed_diff_digest:
        return _run(lambda: approve_browser_skill_removal(args.project_root, args.skill_name, reviewed_diff_digest=args.reviewed_diff_digest, reviewer_identity=args.reviewer))
    return _run(lambda: review_browser_skill_removal(args.project_root, args.skill_name))


def _project_only(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".")

def _project_skill(parser: argparse.ArgumentParser) -> None:
    _project_only(parser)
    parser.add_argument("--skill-name", required=True)
def _source_skill(parser: argparse.ArgumentParser, source_flag: str, source_help: str) -> None:
    # Both mounts land on `args.source_id`, so the handlers above are one set of
    # calls rather than one per spelling of the flag.
    _project_skill(parser); parser.add_argument(source_flag, dest="source_id", required=True, help=source_help)
