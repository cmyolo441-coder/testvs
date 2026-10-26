"""Bounded, lineage-verifiable feedback rounds for closed design direction sets."""

from .design_direction_iterations_schema import (
    DESIGN_DIRECTION_ITERATION_SCHEMA_VERSION,
    MAX_ACTIVE_OPTIONS,
    MAX_MODEL_ATTEMPTS,
    MAX_REVISION_ROUNDS,
    MAX_SNAPSHOTS,
    SUCCESSOR_KINDS,
    TERMINAL_REASONS,
    build_design_direction_iteration,
    canonical_digest,
)
from .design_direction_iterations_store import (
    list_design_direction_iterations,
    show_design_direction_iteration,
    update_design_direction_iteration,
    write_design_direction_iteration,
)
from .design_direction_iterations_transitions import (
    render_design_direction_iteration_html,
    revise_design_direction_iteration,
    select_design_direction_iteration,
    terminate_design_direction_iteration,
)
from .design_direction_iterations_validation import validate_design_direction_iteration

__all__ = [
    "DESIGN_DIRECTION_ITERATION_SCHEMA_VERSION",
    "MAX_ACTIVE_OPTIONS",
    "MAX_MODEL_ATTEMPTS",
    "MAX_REVISION_ROUNDS",
    "MAX_SNAPSHOTS",
    "SUCCESSOR_KINDS",
    "TERMINAL_REASONS",
    "build_design_direction_iteration",
    "canonical_digest",
    "design_direction_iteration_machine_actions",
    "list_design_direction_iterations",
    "render_design_direction_iteration_html",
    "revise_design_direction_iteration",
    "select_design_direction_iteration",
    "show_design_direction_iteration",
    "terminate_design_direction_iteration",
    "update_design_direction_iteration",
    "validate_design_direction_iteration",
    "write_design_direction_iteration",
]


def design_direction_iteration_machine_actions(
    iteration: dict[str, object],
) -> list[dict[str, object]]:
    """Return the sole action identities consumers may render for this record."""
    terminal = iteration.get("terminal")
    snapshots = iteration.get("snapshots")
    if (
        not isinstance(terminal, dict)
        or not isinstance(snapshots, list)
        or not snapshots
        or not isinstance(snapshots[-1], dict)
    ):
        return []
    iteration_id = str(iteration.get("iteration_id") or "")
    current = snapshots[-1]
    revision_digest = str(current.get("revision_digest") or "")
    if not iteration_id or not revision_digest:
        return []
    if terminal.get("outcome") != "OPEN":
        actions: list[dict[str, object]] = [
            {
                "action": "show_design_direction_iteration",
                "iteration_id": iteration_id,
                "revision_digest": revision_digest,
            }
        ]
        promotion = iteration.get("memory_promotion")
        request = promotion.get("request") if isinstance(promotion, dict) else None
        if isinstance(request, dict) and terminal.get("accepted_revision_digest") == revision_digest:
            actions.append(
                {
                    "action": "request_design_direction_memory_review",
                    "iteration_id": iteration_id,
                    "revision_digest": revision_digest,
                    "memory_promotion_request": dict(request),
                }
            )
        return actions
    return [
        {
            "action": "show_design_direction_iteration",
            "iteration_id": iteration_id,
            "revision_digest": revision_digest,
        },
        {
            "action": "revise_design_direction_iteration",
            "iteration_id": iteration_id,
            "revision_digest": revision_digest,
            "parent_revision_digest": revision_digest,
        },
        {
            "action": "select_design_direction_option",
            "iteration_id": iteration_id,
            "revision_digest": revision_digest,
            "option_refs": list(current.get("option_refs", [])),
        },
    ]
