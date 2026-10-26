"""Agent/operator read path for one wrapper work-artifact shape."""

from __future__ import annotations

import argparse

from ..installer import OmhError
from ..surfaces.show_shape_models import ShowShapeCapabilities
from ..system.paths import OmhPaths
from ..wrapper import sessions
from ..wrapper.work_artifact_actions import build_work_artifact_show_shape_action
from .common import _paths, _print_json


def cmd_runtime_artifacts_show_shape(args: argparse.Namespace) -> int:
    """Render one session artifact through the existing selected-action facade."""
    paths = _paths(args)
    session_id, status = _selected_session_status(paths, args.session_id, args.artifact_id)
    payload = build_work_artifact_show_shape_action(
        status,
        artifact_id=args.artifact_id,
        lens=args.lens,
        format=args.format,
        # Mermaid is never inferred from a CLI presence or from a caller flag.
        # A stable observed capability hook can widen this later without changing
        # this action path's unavailable result.
        capabilities=ShowShapeCapabilities(),
    )
    # Selection metadata identifies the source session only. It deliberately
    # carries no copied artifact record or source body beyond the requested
    # shape action's established result.
    _print_json({**payload, "selected_session_id": session_id})
    return 0


def _selected_session_status(
    paths: OmhPaths,
    explicit_session_id: str | None,
    artifact_id: str,
) -> tuple[str, dict[str, object]]:
    if explicit_session_id:
        try:
            return explicit_session_id, sessions.build_wrapper_session_status(paths, explicit_session_id)
        except FileNotFoundError as exc:
            raise OmhError(f"wrapper session not found: {explicit_session_id}") from exc
    for session in sorted(sessions.list_wrapper_sessions(paths), key=_session_order, reverse=True):
        session_id = str(session.get("session_id", ""))
        if not session_id:
            continue
        try:
            status = sessions.build_wrapper_session_status(paths, session_id)
        except FileNotFoundError:
            continue
        if _contains_stable_artifact(status, artifact_id):
            return session_id, status
    raise OmhError(f"no current wrapper session contains artifact: {artifact_id}")


def _session_order(session: dict[str, object]) -> tuple[str, str]:
    """Newest update wins; the stable id makes tied updates deterministic."""
    return str(session.get("updated_at", "")), str(session.get("session_id", ""))


def _contains_stable_artifact(status: dict[str, object], artifact_id: str) -> bool:
    selected = build_work_artifact_show_shape_action(status, artifact_id=artifact_id)
    shape = selected.get("shape")
    return isinstance(shape, dict) and shape.get("reason") != "unknown_artifact_id"


def add_runtime_artifacts_show_shape_command(
    artifacts_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    show_shape = artifacts_sub.add_parser(
        "show-shape",
        help="Agent/operator surface: render one session work-artifact shape; never advances the session.",
    )
    show_shape.add_argument(
        "--session-id",
        default=None,
        help="Optional wrapper session override; otherwise selects the latest current session containing the stable artifact.",
    )
    show_shape.add_argument("--artifact-id", required=True, help="Stable artifact id from the selected work-artifact action.")
    show_shape.add_argument("--lens", required=True, choices=("flow", "structure", "change", "state", "ownership"))
    show_shape.add_argument("--format", default="ascii", choices=("ascii", "tree", "diff", "mermaid"))
    show_shape.add_argument("--json", action="store_true", help="Emit the selected-action machine payload.")
    show_shape.set_defaults(func=cmd_runtime_artifacts_show_shape)
