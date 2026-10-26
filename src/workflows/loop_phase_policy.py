from __future__ import annotations

from typing import Final


LOOP_PHASES: Final[frozenset[str]] = frozenset(
    {
        "interview",
        "plan",
        "research",
        "handoff",
        "execution",
        "feedback",
        "waiting",
        "blocked",
        "complete",
    }
)
PHASE_TARGETS: Final[dict[str, tuple[str, str, str]]] = {
    "interview": ("planning", "plan", "goal_contract_observed"),
    "plan": ("research", "research", "plan_observed"),
    "research": ("executor_handoff", "handoff", "research_evidence_observed"),
    "handoff": ("executor_dispatch", "execution", "handoff_observed"),
    "execution": ("review_fix_loop", "feedback", "verification_observed"),
    "feedback": ("planning", "plan", "feedback_observed"),
}


def phase_target(phase: str) -> tuple[str, str, str]:
    """Return the action, target phase, and gate for one native phase."""
    try:
        return PHASE_TARGETS[phase]
    except KeyError as exc:
        raise ValueError(f"loop phase cannot advance natively: {phase!r}") from exc
