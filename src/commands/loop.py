"""CLI adapters over the typed Loop operation service.

Every handler here builds one `LoopOperationRequest` and prints the artifacts
the service returns. The lifecycle rules, the field vocabulary, and the error
mapping live in `omh.workflows.loop_operations`, which the `omh_loop` plugin
tool calls through the same entry point, so the two surfaces cannot drift.

What stays here is what is genuinely a command-line concern: argparse shapes,
reading an observation file from a path or stdin, the `OmhError` text these
commands have always raised, and the exact top-level JSON keys they print.
"""
from __future__ import annotations

import argparse

from ..coding.executors import canonical_executor_profile
from ..goal_loop import (
    LOOP_ACTIONS,
    LOOP_DISPATCH_RECOVERY_OUTCOMES,
    LOOP_EXECUTOR_OPTION_IDS,
    LOOP_STICKY_RULE_DEFAULT_GAP,
    LOOP_STICKY_RULE_DEFAULT_MAX_REPEATS,
    LOOP_STICKY_RULE_REPEAT_MODES,
    LOOP_WORKFLOW_PATTERNS,
    PERMISSION_PROFILES,
)
from ..installer import OmhError
from ..workflows.loop_observation_input import read_loop_observation_json
from ..workflows.loop_operations import LoopOperationRequest, run_loop_operation
from .common import _chat_message, _paths, _print_json, add_revision_guard_arguments
from .loop_driver import add_driver_commands


def _run(args: argparse.Namespace, action: str, **fields: object) -> dict[str, object]:
    """Run one service action against the paths this invocation resolved."""
    request = LoopOperationRequest(
        action=action,
        loop_id=str(getattr(args, "loop_id", "") or ""),
        fields=fields,
        expected_revision=getattr(args, "expected_revision", None),
        mutation_id=str(getattr(args, "mutation_id", "") or ""),
    )
    return run_loop_operation(_paths(args), request).artifacts


def cmd_loop_start_card(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "start_card",
                message=_chat_message(args),
                include_goal=args.include_goal,
                source=args.source,
                permission_profile=args.permission_profile,
                default_executor=args.default_executor,
            )
        )
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_assess(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(args, "assess", message=_chat_message(args), include_goal=args.include_goal)
        )
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_start(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "start",
                goal_summary=args.goal_summary,
                goal_reframe=args.goal_reframe,
                success_criteria=args.criterion or [],
                permission_profile=args.permission_profile,
                allowed_executors=args.allowed_executor or [],
                allow_actions=args.allow_action or [],
                forbid_actions=args.forbid_action or [],
                linked_goal_id=args.linked_goal or "",
                source=args.source,
                loop_id=args.loop_id or "",
                allow_unloopable=args.allow_unloopable,
                executor=args.executor,
                work_kind=args.work_kind,
                capability_snapshot=(
                    read_loop_observation_json(args.capability_json) if args.capability_json else None
                ),
                executor_session_ref=args.executor_session_ref or "",
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_status(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "status", loop_id=args.loop_id))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_feedback(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "feedback",
                observed_artifacts=args.observed_artifact or [],
                internal_gap=args.internal_gap or "",
                external_wait=args.external_wait or "",
                context_exhausted=args.context_exhausted,
                budget_exhausted=args.budget_exhausted,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_permit(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "permit",
                allow_actions=args.allow_action or [],
                forbid_actions=args.forbid_action or [],
                allowed_executors=args.allowed_executor or [],
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_tick(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "tick",
                trigger=args.trigger,
                cadence=args.cadence or "",
                worktree_base=args.worktree_base or "",
                worktree_branch=args.worktree_branch or "",
                subagent_role=args.subagent_role or "",
                connector=args.connector or "",
                connector_action=args.connector_action or "",
                workflow_pattern=args.workflow_pattern,
                note=args.note or "",
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_sticky_rule_declare(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "sticky_rule_declare",
                rule_id=args.rule_id,
                text=args.text,
                repeat_mode=args.repeat_mode,
                repeat_gap=args.repeat_gap,
                max_repeats=args.max_repeats,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_run_once(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "run_once"))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_goal_driver_handoff(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "goal_driver_handoff",
                gate_commands=args.gate_command or [],
                max_turns=args.max_turns,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_goal_driver_observe(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "goal_driver_observe",
                driver_observation=read_loop_observation_json(args.observation_json),
            )
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        raise OmhError(f"invalid loop goal driver observation: {exc}") from exc
    return 0


def cmd_loop_queue_list(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "queue_list", include_observed=args.include_observed))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_inspect(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "queue_inspect", queue_id=args.queue_id))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_handoff(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "queue_handoff", queue_id=args.queue_id))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_dispatch(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "queue_dispatch",
                queue_id=args.queue_id,
                executor=args.executor,
                session_ref=args.codex_session_ref or args.session_ref or "",
                thread_ref=args.codex_thread_ref or args.thread_ref or "",
                evidence_refs=args.evidence_ref or [],
                summary=args.summary or "",
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_recover_dispatch(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "queue_recover_dispatch",
                queue_id=args.queue_id,
                prior_attempt_id=args.prior_attempt_id or "",
                prior_outcome=args.prior_outcome,
                outcome_evidence_refs=args.outcome_evidence_ref or [],
                outcome_summary=args.outcome_summary or "",
                executor=args.executor,
                session_ref=args.codex_session_ref or args.session_ref or "",
                thread_ref=args.codex_thread_ref or args.thread_ref or "",
                evidence_refs=args.evidence_ref or [],
                summary=args.summary or "",
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_observe_codex(args: argparse.Namespace) -> int:
    try:
        log_text = ""
        if args.codex_log_text:
            log_text = args.codex_log_text
        elif args.codex_log_jsonl:
            from pathlib import Path

            log_text = Path(args.codex_log_jsonl).read_text(encoding="utf-8")
        _print_json(
            _run(
                args,
                "queue_observe_codex",
                queue_id=args.queue_id,
                codex_log_text=log_text,
                evidence_refs=args.evidence_ref or [],
                codex_log_ref=args.codex_log_ref or "",
                summary=args.summary or "",
                dispatch_attempt_id=args.dispatch_attempt_id or "",
            )
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_narrate(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "queue_narrate", queue_id=args.queue_id))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_observe(args: argparse.Namespace) -> int:
    try:
        _print_json(
            _run(
                args,
                "queue_observe",
                queue_id=args.queue_id,
                evidence_refs=args.evidence_ref or [],
                worktree_evidence_refs=args.worktree_evidence_ref or [],
                subagent_evidence_refs=args.subagent_evidence_ref or [],
                connector_evidence_refs=args.connector_evidence_ref or [],
                summary=args.summary or "",
                dispatch_attempt_id=args.dispatch_attempt_id or "",
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_loop_queue_block(args: argparse.Namespace) -> int:
    try:
        _print_json(_run(args, "queue_block", queue_id=args.queue_id, reason=args.reason))
    except (FileNotFoundError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def _add_loop_commands(sub) -> None:
    loop = sub.add_parser("loop", help="Assess, start, inspect, and advance loopable goal control-plane records.")
    loop_sub = loop.add_subparsers(dest="loop_command", required=True)

    add_driver_commands(loop_sub)
    start = loop_sub.add_parser("start")
    start.add_argument("--executor", type=canonical_executor_profile, default="hermes")
    start.add_argument("--work-kind", choices=("coding", "non_coding"), default="non_coding")
    start.add_argument("--capability-json")
    start.add_argument("--executor-session-ref")
    start.add_argument("--loop-id", default="")
    start.add_argument("--goal-summary", required=True)
    start.add_argument("--goal-reframe", required=True)
    start.add_argument("--criterion", action="append", required=True)
    start.add_argument("--permission-profile", choices=PERMISSION_PROFILES, default="handoff_only")
    start.add_argument("--allowed-executor", action="append")
    start.add_argument("--allow-action", choices=LOOP_ACTIONS, action="append")
    start.add_argument("--forbid-action", choices=LOOP_ACTIONS, action="append")
    start.add_argument("--linked-goal", default="")
    start.add_argument("--source", default="omh")
    start.add_argument(
        "--allow-unloopable",
        action="store_true",
        help="Operator escape hatch: create loop state even when loopability assessment recommends another surface.",
    )
    start.set_defaults(func=cmd_loop_start)

    start_card = loop_sub.add_parser("start-card")
    start_card.add_argument("message", nargs="*", help="Ambitious goal text to shape into a loop start card.")
    start_card.add_argument("--stdin", action="store_true", help="Read the goal text from stdin.")
    start_card.add_argument(
        "--event-json",
        default=None,
        help="Read a Hermes-like JSON event from this path, or '-' for stdin.",
    )
    start_card.add_argument("--include-goal", action="store_true", help="Include the raw goal text in stdout.")
    start_card.add_argument("--source", default="omh")
    start_card.add_argument("--permission-profile", choices=PERMISSION_PROFILES, default="handoff_only")
    start_card.add_argument("--default-executor", type=canonical_executor_profile, choices=LOOP_EXECUTOR_OPTION_IDS, default="choose")
    start_card.set_defaults(func=cmd_loop_start_card)

    assess = loop_sub.add_parser("assess")
    assess.add_argument("message", nargs="*", help="Goal text to classify before starting a loop.")
    assess.add_argument("--stdin", action="store_true", help="Read the goal text from stdin.")
    assess.add_argument(
        "--event-json",
        default=None,
        help="Read a Hermes-like JSON event from this path, or '-' for stdin.",
    )
    assess.add_argument("--include-goal", action="store_true", help="Include the raw goal text in stdout.")
    assess.set_defaults(func=cmd_loop_assess)

    status = loop_sub.add_parser("status")
    status.add_argument("--loop", dest="loop_id", default="")
    status.set_defaults(func=cmd_loop_status)

    feedback = loop_sub.add_parser("feedback")
    feedback.add_argument("--loop", dest="loop_id", required=True)
    feedback.add_argument("--observed-artifact", action="append")
    feedback.add_argument("--internal-gap", default="")
    feedback.add_argument("--external-wait", default="")
    feedback.add_argument("--context-exhausted", action="store_true")
    feedback.add_argument("--budget-exhausted", action="store_true")
    feedback.set_defaults(func=cmd_loop_feedback)

    permit = loop_sub.add_parser("permit")
    permit.add_argument("--loop", dest="loop_id", required=True)
    permit.add_argument("--allow-action", choices=LOOP_ACTIONS, action="append")
    permit.add_argument("--forbid-action", choices=LOOP_ACTIONS, action="append")
    permit.add_argument("--allowed-executor", action="append")
    permit.set_defaults(func=cmd_loop_permit)

    tick = loop_sub.add_parser("tick")
    tick.add_argument("--loop", dest="loop_id", required=True)
    tick.add_argument("--trigger", choices=("manual", "scheduled", "wrapper", "automation"), default="manual")
    tick.add_argument("--cadence", default="")
    tick.add_argument("--worktree-base", default="")
    tick.add_argument("--worktree-branch", default="")
    tick.add_argument("--subagent-role", default="")
    tick.add_argument("--connector", default="")
    tick.add_argument("--connector-action", default="")
    tick.add_argument("--workflow-pattern", choices=LOOP_WORKFLOW_PATTERNS, default="single_step")
    tick.add_argument("--note", default="")
    tick.set_defaults(func=cmd_loop_tick)

    sticky_rule = loop_sub.add_parser("sticky-rule")
    sticky_rule_sub = sticky_rule.add_subparsers(dest="sticky_rule_command", required=True)

    sticky_rule_declare = sticky_rule_sub.add_parser("declare")
    sticky_rule_declare.add_argument("--loop", dest="loop_id", required=True)
    sticky_rule_declare.add_argument("--rule-id", required=True)
    sticky_rule_declare.add_argument("--text", required=True)
    sticky_rule_declare.add_argument("--repeat-mode", choices=LOOP_STICKY_RULE_REPEAT_MODES, default="after_gap")
    sticky_rule_declare.add_argument("--repeat-gap", type=int, default=LOOP_STICKY_RULE_DEFAULT_GAP)
    sticky_rule_declare.add_argument("--max-repeats", type=int, default=LOOP_STICKY_RULE_DEFAULT_MAX_REPEATS)
    add_revision_guard_arguments(sticky_rule_declare)
    sticky_rule_declare.set_defaults(func=cmd_loop_sticky_rule_declare)

    run_once = loop_sub.add_parser("run-once")
    run_once.add_argument("--loop", dest="loop_id", required=True)
    run_once.set_defaults(func=cmd_loop_run_once)

    goal_driver_handoff = loop_sub.add_parser("goal-driver-handoff")
    goal_driver_handoff.add_argument("--loop", dest="loop_id", required=True)
    goal_driver_handoff.add_argument("--gate-command", action="append")
    goal_driver_handoff.add_argument("--max-turns", type=int, default=0)
    goal_driver_handoff.set_defaults(func=cmd_loop_goal_driver_handoff)

    goal_driver_observe = loop_sub.add_parser("goal-driver-observe")
    goal_driver_observe.add_argument("--loop", dest="loop_id", required=True)
    goal_driver_observe.add_argument("--observation-json", required=True)
    add_revision_guard_arguments(goal_driver_observe)
    goal_driver_observe.set_defaults(func=cmd_loop_goal_driver_observe)

    queue = loop_sub.add_parser("queue")
    queue_sub = queue.add_subparsers(dest="queue_command", required=True)

    queue_list = queue_sub.add_parser("list")
    queue_list.add_argument("--loop", dest="loop_id", required=True)
    queue_list.add_argument("--include-observed", action="store_true")
    queue_list.set_defaults(func=cmd_loop_queue_list)

    queue_inspect = queue_sub.add_parser("inspect")
    queue_inspect.add_argument("--loop", dest="loop_id", required=True)
    queue_inspect.add_argument("--queue", dest="queue_id", required=True)
    queue_inspect.set_defaults(func=cmd_loop_queue_inspect)

    queue_handoff = queue_sub.add_parser("handoff")
    queue_handoff.add_argument("--loop", dest="loop_id", required=True)
    queue_handoff.add_argument("--queue", dest="queue_id", required=True)
    queue_handoff.set_defaults(func=cmd_loop_queue_handoff)

    queue_dispatch = queue_sub.add_parser("dispatch")
    queue_dispatch.add_argument("--loop", dest="loop_id", required=True)
    queue_dispatch.add_argument("--queue", dest="queue_id", required=True)
    queue_dispatch.add_argument("--executor", type=canonical_executor_profile, choices=LOOP_EXECUTOR_OPTION_IDS, required=True)
    queue_dispatch.add_argument("--session-ref", default="")
    queue_dispatch.add_argument("--thread-ref", default="")
    queue_dispatch.add_argument("--codex-session-ref", default="")
    queue_dispatch.add_argument("--codex-thread-ref", default="")
    queue_dispatch.add_argument("--evidence-ref", action="append")
    queue_dispatch.add_argument("--summary", default="")
    queue_dispatch.set_defaults(func=cmd_loop_queue_dispatch)

    queue_recover_dispatch = queue_sub.add_parser("recover-dispatch")
    queue_recover_dispatch.add_argument("--loop", dest="loop_id", required=True)
    queue_recover_dispatch.add_argument("--queue", dest="queue_id", required=True)
    queue_recover_dispatch.add_argument("--prior-attempt-id", default="")
    queue_recover_dispatch.add_argument(
        "--prior-outcome",
        choices=LOOP_DISPATCH_RECOVERY_OUTCOMES,
        required=True,
    )
    queue_recover_dispatch.add_argument("--outcome-evidence-ref", action="append", required=True)
    queue_recover_dispatch.add_argument("--outcome-summary", default="")
    queue_recover_dispatch.add_argument("--executor", type=canonical_executor_profile, choices=LOOP_EXECUTOR_OPTION_IDS, required=True)
    queue_recover_dispatch.add_argument("--session-ref", default="")
    queue_recover_dispatch.add_argument("--thread-ref", default="")
    queue_recover_dispatch.add_argument("--codex-session-ref", default="")
    queue_recover_dispatch.add_argument("--codex-thread-ref", default="")
    queue_recover_dispatch.add_argument("--evidence-ref", action="append")
    queue_recover_dispatch.add_argument("--summary", default="")
    queue_recover_dispatch.set_defaults(func=cmd_loop_queue_recover_dispatch)

    queue_observe_codex = queue_sub.add_parser("observe-codex")
    queue_observe_codex.add_argument("--loop", dest="loop_id", required=True)
    queue_observe_codex.add_argument("--queue", dest="queue_id", required=True)
    queue_observe_codex.add_argument("--codex-log-jsonl", default="")
    queue_observe_codex.add_argument("--codex-log-text", default="")
    queue_observe_codex.add_argument("--codex-log-ref", default="")
    queue_observe_codex.add_argument("--evidence-ref", action="append", required=True)
    queue_observe_codex.add_argument("--dispatch-attempt-id", default="")
    queue_observe_codex.add_argument("--summary", default="")
    queue_observe_codex.set_defaults(func=cmd_loop_queue_observe_codex)

    queue_narrate = queue_sub.add_parser("narrate")
    queue_narrate.add_argument("--loop", dest="loop_id", required=True)
    queue_narrate.add_argument("--queue", dest="queue_id", default="")
    queue_narrate.set_defaults(func=cmd_loop_queue_narrate)

    queue_observe = queue_sub.add_parser("observe")
    queue_observe.add_argument("--loop", dest="loop_id", required=True)
    queue_observe.add_argument("--queue", dest="queue_id", required=True)
    queue_observe.add_argument("--evidence-ref", action="append", required=True)
    queue_observe.add_argument("--worktree-evidence-ref", action="append")
    queue_observe.add_argument("--subagent-evidence-ref", action="append")
    queue_observe.add_argument("--connector-evidence-ref", action="append")
    queue_observe.add_argument("--dispatch-attempt-id", default="")
    queue_observe.add_argument("--summary", default="")
    queue_observe.set_defaults(func=cmd_loop_queue_observe)

    queue_block = queue_sub.add_parser("block")
    queue_block.add_argument("--loop", dest="loop_id", required=True)
    queue_block.add_argument("--queue", dest="queue_id", required=True)
    queue_block.add_argument("--reason", required=True)
    queue_block.set_defaults(func=cmd_loop_queue_block)
