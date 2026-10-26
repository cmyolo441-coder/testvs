from __future__ import annotations

import argparse

from ..goal_ledger import (
    CHECKPOINT_STATUSES,
    GOAL_FAILURE_REASON_CODES,
    SLOP_CLEANUP_OBLIGATIONS,
    build_goal_completion_gate,
    build_goal_continuation,
    build_goal_status_card,
    cancel_goal_ledger,
    complete_goal_ledger,
    create_goal_ledger,
    fail_goal_ledger,
    list_goal_ledgers,
    read_goal_ledger,
    render_goal_status_text,
    record_goal_blocker,
    record_goal_checkpoint,
    slop_cleanup_obligation_criterion,
)
from ..installer import OmhError
from ..quality.working_tree_fingerprint import working_tree_content_fingerprint
from ..system.local_store import utc_now
from ..workflows.coordination_board import (
    DEFAULT_LIMIT as COORDINATION_BOARD_DEFAULT_LIMIT,
    build_coordination_board,
    render_coordination_board_text,
)
from ..workflows.goal_journey import build_goal_journey, render_goal_journey_text
from .common import _paths, _print_json, _wants_json, add_revision_guard_arguments


def cmd_goal_create(args: argparse.Namespace) -> int:
    paths = _paths(args)
    criteria: list = list(args.criterion or [])
    if args.obligation:
        criteria.append(slop_cleanup_obligation_criterion(args.obligation))
    try:
        goal = create_goal_ledger(
            paths,
            args.objective,
            criteria,
            source=args.source,
            goal_id=args.goal_id or None,
            objective_summary=args.summary or None,
            linked_runtime_runs=args.linked_runtime_run or [],
        )
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    _print_json({"goal": goal, "completion_gate": build_goal_completion_gate(paths, goal["goal_id"])})
    return 0


def _mutation_payload(paths, goal_id: str, goal: dict, outcome: dict) -> dict:
    """CLI payload that reports what the persisted record actually says.

    `applied` is re-derived from the stored goal, not from "the call
    returned": a mutation replayed away by a reused mutation_id leaves the
    goal untouched, and printing it as success is how a discarded change
    looks like a successful one.
    """
    return {
        "goal": goal,
        "completion_gate": build_goal_completion_gate(paths, goal_id),
        "applied": bool(outcome.get("applied")),
        "replayed": bool(outcome.get("replayed")),
    }


def cmd_goal_checkpoint(args: argparse.Namespace) -> int:
    paths = _paths(args)
    outcome: dict = {}
    # Read rather than accept: a caller cannot forge the workspace identity a
    # checkpoint claims to cover. A non-authoritative collection is refused so
    # a retry cannot silently reuse a committed-tree-only freshness stamp.
    fingerprint = working_tree_content_fingerprint()
    if not fingerprint.authoritative or fingerprint.fingerprint is None:
        raise OmhError(f"working tree fingerprint is unavailable: {fingerprint.state}")
    try:
        goal = record_goal_checkpoint(
            paths,
            args.goal_id,
            args.summary,
            criteria_refs=args.criterion or [],
            status=args.status,
            evidence_refs=args.evidence_ref or [],
            notes_summary=args.notes_summary or "",
            linked_runtime_run_id=args.linked_runtime_run or "",
            observed_tree=fingerprint.fingerprint,
            expected_revision=args.expected_revision,
            mutation_id=args.mutation_id or None,
            outcome=outcome,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(_mutation_payload(paths, args.goal_id, goal, outcome))
    return 0 if outcome.get("applied") else 1


def cmd_goal_blocker(args: argparse.Namespace) -> int:
    paths = _paths(args)
    outcome: dict = {}
    try:
        goal = record_goal_blocker(
            paths,
            args.goal_id,
            args.summary,
            attempted_recovery=args.attempted_recovery or "",
            missing_authority=args.missing_authority or "",
            evidence_refs=args.evidence_ref or [],
            mark_goal_blocked=args.mark_goal_blocked,
            expected_revision=args.expected_revision,
            mutation_id=args.mutation_id or None,
            outcome=outcome,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(_mutation_payload(paths, args.goal_id, goal, outcome))
    return 0 if outcome.get("applied") else 1


def cmd_goal_complete(args: argparse.Namespace) -> int:
    try:
        result = complete_goal_ledger(
            _paths(args),
            args.goal_id,
            evidence_refs=args.evidence_ref or [],
            expected_revision=args.expected_revision,
            mutation_id=args.mutation_id or None,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0 if result["completed"] else 1


def cmd_goal_cancel(args: argparse.Namespace) -> int:
    paths = _paths(args)
    outcome: dict = {}
    try:
        goal = cancel_goal_ledger(
            paths,
            args.goal_id,
            reason=args.reason or "",
            expected_revision=args.expected_revision,
            mutation_id=args.mutation_id or None,
            outcome=outcome,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    payload = _mutation_payload(paths, args.goal_id, goal, outcome)
    payload["cancelled"] = bool(outcome.get("applied"))
    _print_json(payload)
    return 0 if outcome.get("applied") else 1


def cmd_goal_fail(args: argparse.Namespace) -> int:
    """Terminally fail a goal: a negative but CONCLUSIVE verdict on the objective.

    Distinct from `goal cancel` (an operator decision) and `goal blocker`
    (a recoverable obstacle, resumable once cleared): a failure states that
    the objective cannot be met at all, with a required, closed
    `--reason-code`, and is terminal in the same way `goal cancel` is --
    later checkpoints, blockers, and gates all refuse.
    """
    paths = _paths(args)
    outcome: dict = {}
    try:
        goal = fail_goal_ledger(
            paths,
            args.goal_id,
            args.summary,
            reason_code=args.reason_code,
            evidence_refs=args.evidence_ref or [],
            expected_revision=args.expected_revision,
            mutation_id=args.mutation_id or None,
            outcome=outcome,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    payload = _mutation_payload(paths, args.goal_id, goal, outcome)
    payload["failed"] = bool(outcome.get("applied"))
    _print_json(payload)
    return 0 if outcome.get("applied") else 1


def cmd_goal_status(args: argparse.Namespace) -> int:
    paths = _paths(args)
    wants_text = bool(getattr(args, "text", False))
    if wants_text and not args.goal_id:
        raise OmhError("--text renders one goal; pass --goal GOAL_ID")
    try:
        if args.goal_id:
            card = build_goal_status_card(paths, args.goal_id)
            if wants_text:
                print(render_goal_status_text(card))
                return 0
            _print_json(
                {
                    "goal": read_goal_ledger(paths, args.goal_id),
                    "completion_gate": build_goal_completion_gate(paths, args.goal_id),
                    "status_card": card,
                }
            )
            return 0
        goals = list_goal_ledgers(paths)
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(
        {
            "goals": [
                {
                    "goal_id": goal["goal_id"],
                    "status": goal["status"],
                    "objective_summary": goal["objective_summary"],
                    "criteria_total": len(goal["acceptance_criteria"]),
                    "criteria_satisfied": len([item for item in goal["acceptance_criteria"] if item["status"] == "satisfied"]),
                }
                for goal in goals
            ]
        }
    )
    return 0


def cmd_goal_journey(args: argparse.Namespace) -> int:
    paths = _paths(args)
    try:
        # The wall clock is read here, in the command, and passed in: the
        # projection itself never reads one, so a test can pin freshness and
        # two reads of an unchanged goal stay comparable.
        journey = build_goal_journey(paths, args.goal_id, now=utc_now())
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json({"journey": journey})
        return 0
    print(render_goal_journey_text(journey))
    return 0


def cmd_goal_continue(args: argparse.Namespace) -> int:
    try:
        _print_json({"continuation": build_goal_continuation(_paths(args), args.goal_id)})
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_goal_board(args: argparse.Namespace) -> int:
    """Show every recorded coordination item as one board.

    Plain text is the default because this command answers a human question
    ("what is moving, blocked, or ready next?"); `--json` is the opt-in for an
    agent or a script. The sibling `goal status` keeps JSON as its default
    because wrappers already parse it; this command is new, so it starts on the
    repo's normal side of that line.
    """
    payload = build_coordination_board(_paths(args), limit=_coordination_board_limit(args))
    if _wants_json(args):
        _print_json(payload)
    else:
        print(render_coordination_board_text(payload))
    return 0


def _coordination_board_limit(args: argparse.Namespace) -> int:
    """Row cap, refused rather than coerced when it is not positive.

    A non-positive limit returns an empty `items` list beside a non-zero
    `item_count`, which reads as "nothing is coordinated" — the opposite of
    what the board projected.
    """
    limit = int(getattr(args, "limit", COORDINATION_BOARD_DEFAULT_LIMIT))
    if limit < 1:
        raise OmhError("--limit must be at least 1")
    return limit


def _add_goal_commands(sub) -> None:
    goal = sub.add_parser("goal", help="Manage durable local goal ledgers and completion gates.")
    goal_sub = goal.add_subparsers(dest="goal_command", required=True)

    goal_create = goal_sub.add_parser("create")
    goal_create.add_argument("--goal-id", default="")
    goal_create.add_argument("--objective", required=True)
    goal_create.add_argument("--summary", default="")
    goal_create.add_argument("--criterion", action="append", required=True)
    goal_create.add_argument("--source", default="omh")
    goal_create.add_argument("--linked-runtime-run", action="append")
    goal_create.add_argument(
        "--obligation",
        choices=SLOP_CLEANUP_OBLIGATIONS,
        default="",
        help="Arm a required obligation; slop-cleanup adds AC-SLOP, which an ai-slop-cleaner outcome record satisfies.",
    )
    goal_create.set_defaults(func=cmd_goal_create)

    goal_checkpoint = goal_sub.add_parser("checkpoint")
    goal_checkpoint.add_argument("--goal", dest="goal_id", required=True)
    goal_checkpoint.add_argument("--summary", required=True)
    goal_checkpoint.add_argument("--criterion", action="append")
    goal_checkpoint.add_argument("--status", choices=sorted(CHECKPOINT_STATUSES), default="done")
    goal_checkpoint.add_argument("--evidence-ref", action="append")
    goal_checkpoint.add_argument("--notes-summary", default="")
    goal_checkpoint.add_argument("--linked-runtime-run", default="")
    add_revision_guard_arguments(goal_checkpoint)
    goal_checkpoint.set_defaults(func=cmd_goal_checkpoint)

    goal_blocker = goal_sub.add_parser("blocker")
    goal_blocker.add_argument("--goal", dest="goal_id", required=True)
    goal_blocker.add_argument("--summary", required=True)
    goal_blocker.add_argument("--attempted-recovery", default="")
    goal_blocker.add_argument("--missing-authority", default="")
    goal_blocker.add_argument("--evidence-ref", action="append")
    goal_blocker.add_argument("--mark-goal-blocked", action="store_true")
    add_revision_guard_arguments(goal_blocker)
    goal_blocker.set_defaults(func=cmd_goal_blocker)

    goal_complete = goal_sub.add_parser("complete")
    goal_complete.add_argument("--goal", dest="goal_id", required=True)
    goal_complete.add_argument("--evidence-ref", action="append")
    add_revision_guard_arguments(goal_complete)
    goal_complete.set_defaults(func=cmd_goal_complete)

    goal_cancel = goal_sub.add_parser("cancel")
    goal_cancel.add_argument("--goal", dest="goal_id", required=True)
    goal_cancel.add_argument("--reason", default="")
    add_revision_guard_arguments(goal_cancel)
    goal_cancel.set_defaults(func=cmd_goal_cancel)

    goal_fail = goal_sub.add_parser(
        "fail",
        help="Terminally fail a goal: the objective cannot be met at all (distinct from cancel or blocker).",
    )
    goal_fail.add_argument("--goal", dest="goal_id", required=True)
    goal_fail.add_argument("--summary", required=True)
    goal_fail.add_argument("--reason-code", required=True, choices=sorted(GOAL_FAILURE_REASON_CODES))
    goal_fail.add_argument("--evidence-ref", action="append")
    add_revision_guard_arguments(goal_fail)
    goal_fail.set_defaults(func=cmd_goal_fail)

    goal_status = goal_sub.add_parser("status")
    goal_status.add_argument("--goal", dest="goal_id", default="")
    # JSON stays the default: `goal` is a control-plane group and wrappers parse
    # it, so flipping the default would break them. `--text` is the opt-in for a
    # human or a messenger relay, which is where escaped JSON is unreadable.
    goal_status.add_argument(
        "--text",
        action="store_true",
        help="Render one goal as readable lines instead of the machine payload (requires --goal).",
    )
    goal_status.set_defaults(func=cmd_goal_status)

    goal_continue = goal_sub.add_parser("continue")
    goal_continue.add_argument("--goal", dest="goal_id", required=True)
    goal_continue.set_defaults(func=cmd_goal_continue)

    goal_board = goal_sub.add_parser(
        "board",
        help="Show active, blocked, dependency-gated, and next-ready work as one board (read-only).",
    )
    goal_board.add_argument(
        "--limit",
        type=int,
        default=COORDINATION_BOARD_DEFAULT_LIMIT,
        help=f"Maximum items to display (default: {COORDINATION_BOARD_DEFAULT_LIMIT}).",
    )
    goal_board.add_argument("--json", action="store_true", help="Emit the machine payload instead of plain text.")
    goal_board.set_defaults(func=cmd_goal_board)

    goal_journey = goal_sub.add_parser("journey")
    goal_journey.add_argument("--goal", dest="goal_id", required=True)
    # New surface, so DIRECTION applies without a compatibility exception:
    # plain text is the default user experience and the machine payload is the
    # opt-in. The older `goal` subcommands keep their JSON default because
    # wrappers already parse them.
    goal_journey.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable goal_journey/v1 payload instead of readable lines.",
    )
    goal_journey.set_defaults(func=cmd_goal_journey)
