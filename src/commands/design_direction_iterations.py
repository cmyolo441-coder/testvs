from __future__ import annotations

import argparse
from pathlib import Path

from ..design_directions import (
    DESIGN_AUDIENCES,
    DESIGN_MODES,
    DESIGN_PLATFORMS,
    DESIGN_PRIMARY_TASKS,
    DESIGN_SURFACES,
    build_design_direction_set,
)
from ..design_direction_iterations import (
    build_design_direction_iteration,
    design_direction_iteration_machine_actions,
    render_design_direction_iteration_html,
    revise_design_direction_iteration,
    select_design_direction_iteration,
    show_design_direction_iteration,
    terminate_design_direction_iteration,
    update_design_direction_iteration,
    write_design_direction_iteration,
)
from ..installer import OmhError
from .common import _paths, _print_json


def cmd_ops_design_direction_iteration_prepare(args: argparse.Namespace) -> int:
    try:
        iteration = build_design_direction_iteration(
            _direction_set(args),
            source_revision_digest=args.source_revision_digest,
            criteria_revision=args.criteria_revision,
            criteria_dimensions=tuple(args.criteria_dimension),
            score_threshold=args.score_threshold,
        )
        stored = write_design_direction_iteration(_paths(args), iteration)
        _emit(stored, args.html)
    except (OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_ops_design_direction_iteration_show(args: argparse.Namespace) -> int:
    try:
        _emit(show_design_direction_iteration(_paths(args), args.iteration_id), args.html)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_ops_design_direction_iteration_revise(args: argparse.Namespace) -> int:
    try:
        current = show_design_direction_iteration(_paths(args), args.iteration_id)
        revised = revise_design_direction_iteration(
            current,
            parent_revision_digest=args.parent_revision_digest,
            feedback_reference=args.feedback_reference,
            feedback_delta=tuple(args.feedback_delta),
            direction_set=_direction_set(args),
            successors=tuple(_successor(value) for value in args.successor),
            criteria_revision=args.criteria_revision,
            criteria_dimensions=tuple(args.criteria_dimension),
            scores=tuple(_score(value) for value in args.score or []),
            model_attempts=tuple(_model_attempt(value) for value in args.model_attempt or []),
        )
        _emit(update_design_direction_iteration(_paths(args), revised), args.html)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_ops_design_direction_iteration_select(args: argparse.Namespace) -> int:
    try:
        current = show_design_direction_iteration(_paths(args), args.iteration_id)
        selected = select_design_direction_iteration(current, option_ref=args.option_ref, remember_this=args.remember_this)
        _emit(update_design_direction_iteration(_paths(args), selected), args.html)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_ops_design_direction_iteration_stop(args: argparse.Namespace) -> int:
    try:
        current = show_design_direction_iteration(_paths(args), args.iteration_id)
        stopped = terminate_design_direction_iteration(current, reason=args.reason)
        _emit(update_design_direction_iteration(_paths(args), stopped), args.html)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    return 0


def _emit(iteration: dict[str, object], html_path: str) -> None:
    preview_path = ""
    if html_path:
        target = Path(html_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_design_direction_iteration_html(iteration), encoding="utf-8", newline="")
        preview_path = str(target)
    _print_json(
        {
            "schema_version": "omh_ops_design_direction_iteration_result/v1",
            "iteration": iteration,
            "preview_path": preview_path,
            "machine_actions": design_direction_iteration_machine_actions(iteration),
        }
    )


def _direction_set(args: argparse.Namespace) -> dict[str, object]:
    return build_design_direction_set(
        surface=args.surface,
        audience=args.audience,
        primary_task=args.primary_task,
        platform=args.platform,
        mode=args.mode,
        context_references=tuple(_context_reference(value) for value in args.context_reference),
        options=tuple(_option(value) for value in args.option),
    )


def _context_reference(value: str) -> tuple[str, str, str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("context_reference must be reference_kind:reference_id:provenance")
    return (parts[0], parts[1], parts[2])


def _option(value: str) -> tuple[str, str, str, str, str, str, tuple[str, ...]]:
    parts = value.split(":")
    if len(parts) != 7:
        raise ValueError("option must be id:hierarchy:palette:typography:layout:signature_element:avoid1|avoid2")
    avoided = tuple(pattern for pattern in parts[6].split("|") if pattern)
    if not avoided:
        raise ValueError("option must name at least one avoid pattern")
    return (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], avoided)


def _successor(value: str) -> tuple[str, tuple[str, ...], str]:
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError("successor must be kind:parent_id|parent_id:target_id")
    return (parts[0], tuple(item for item in parts[1].split("|") if item), parts[2])


def _score(value: str) -> tuple[str, float, str, str, str]:
    parts = value.split(":")
    if len(parts) != 5:
        raise ValueError("score must be dimension:score:evidence_ref:evaluator_id:rubric_revision")
    return (parts[0], float(parts[1]), parts[2], parts[3], parts[4])


def _model_attempt(value: str) -> tuple[str, str, int | None, float | None, float | None, str | None]:
    parts = value.split(":")
    if len(parts) != 6:
        raise ValueError("model_attempt must be purpose:model_id:tokens:cost:latency_ms:failure_ref")
    return (parts[0], parts[1], int(parts[2]) if parts[2] else None, float(parts[3]) if parts[3] else None, float(parts[4]) if parts[4] else None, parts[5] or None)


def add_ops_design_direction_iterations_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    command = ops_sub.add_parser("design-direction-iterations", help="Operate bounded, lineage-bound design direction feedback rounds.")
    sub = command.add_subparsers(dest="design_direction_iteration_command", required=True)
    prepare = sub.add_parser("prepare", help="Persist a root iteration and optionally write a static preview.")
    _add_direction_args(prepare)
    prepare.add_argument("--source-revision-digest", required=True)
    prepare.add_argument("--criteria-revision", required=True)
    prepare.add_argument("--criteria-dimension", action="append", required=True)
    prepare.add_argument("--score-threshold", type=float, required=True)
    prepare.add_argument("--html", default="")
    prepare.set_defaults(func=cmd_ops_design_direction_iteration_prepare)
    revise = sub.add_parser("revise", help="Append one feedback-bound material revision from the rendered parent digest.")
    revise.add_argument("iteration_id")
    revise.add_argument("--parent-revision-digest", required=True)
    revise.add_argument("--feedback-reference", required=True)
    revise.add_argument("--feedback-delta", action="append", required=True)
    revise.add_argument("--successor", action="append", required=True)
    revise.add_argument("--criteria-revision", required=True)
    revise.add_argument("--criteria-dimension", action="append", required=True)
    revise.add_argument("--score", action="append")
    revise.add_argument("--model-attempt", action="append")
    revise.add_argument("--html", default="")
    _add_direction_args(revise)
    revise.set_defaults(func=cmd_ops_design_direction_iteration_revise)
    show = sub.add_parser("show", help="Read one full iteration trajectory without invoking a model or browser.")
    show.add_argument("iteration_id")
    show.add_argument("--html", default="")
    show.set_defaults(func=cmd_ops_design_direction_iteration_show)
    select = sub.add_parser("select", help="Accept one option from the current revision only.")
    select.add_argument("iteration_id")
    select.add_argument("--option-ref", required=True)
    select.add_argument("--remember-this", action="store_true")
    select.add_argument("--html", default="")
    select.set_defaults(func=cmd_ops_design_direction_iteration_select)
    stop = sub.add_parser("stop", help="Record a blocked-evidence, blocked-capability, or cancelled terminal outcome.")
    stop.add_argument("iteration_id")
    stop.add_argument("--reason", choices=("blocked_evidence", "blocked_capability", "cancelled"), required=True)
    stop.add_argument("--html", default="")
    stop.set_defaults(func=cmd_ops_design_direction_iteration_stop)


def _add_direction_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--surface", choices=DESIGN_SURFACES, required=True)
    parser.add_argument("--audience", choices=DESIGN_AUDIENCES, required=True)
    parser.add_argument("--primary-task", choices=DESIGN_PRIMARY_TASKS, required=True)
    parser.add_argument("--platform", choices=DESIGN_PLATFORMS, required=True)
    parser.add_argument("--mode", choices=DESIGN_MODES, required=True)
    parser.add_argument("--context-reference", action="append", required=True)
    parser.add_argument("--option", action="append", required=True)
