"""Durable attempt-scoped reservations for fanout review lanes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

from ..system.local_store import locked_json_update
from ..system.paths import OmhPaths
from .fanout_artifacts import fanout_review_dispatch_budget_path

REVIEW_DISPATCH_BUDGET_SCHEMA_VERSION: Final = "fanout_review_dispatch_budget/v1"
REVIEW_DISPATCH_BUDGET_CLAIM_BOUNDARY: Final = (
    "A reservation records that an eligible, ready review lane consumed one dispatch allowance for "
    "one explicit goal attempt. It is not review, verification, CI, merge-readiness, or merge evidence."
)

_REVIEW_ROLE_ALIASES: Final[Mapping[str, str]] = {
    "review": "code-review",
    "reviewer": "code-review",
    "code-review": "code-review",
    "code-reviewer": "code-review",
    "hybrid-review": "code-review",
    "manual-qa": "manual-qa",
    "qa": "manual-qa",
    "final-gate": "final-gate",
    "review-gate": "final-gate",
    "hybrid-verification": "final-gate",
}


class ReviewDispatchBudgetError(ValueError):
    """The review dispatch budget configuration or stored state is invalid."""


@dataclass(frozen=True, slots=True)
class ReviewDispatchReservation:
    status: str
    attempt_id: str
    role: str
    limit: int
    reserved: int

    @property
    def granted(self) -> bool:
        return self.status == "reserved"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": REVIEW_DISPATCH_BUDGET_SCHEMA_VERSION,
            "status": self.status,
            "attempt_id": self.attempt_id,
            "role": self.role,
            "limit": self.limit,
            "reserved": self.reserved,
            "claim_boundary": REVIEW_DISPATCH_BUDGET_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True, slots=True)
class ReviewDispatchBudget:
    paths: OmhPaths
    fanout_id: str
    attempt_id: str
    limit: int
    progressed: bool = False

    def __post_init__(self) -> None:
        attempt_id = self.attempt_id.strip()
        if not attempt_id or len(attempt_id) > 128 or any(char in attempt_id for char in "\r\n"):
            raise ReviewDispatchBudgetError(
                "goal_attempt_id must be a non-empty single-line value of at most 128 characters"
            )
        if isinstance(self.limit, bool) or self.limit < 1:
            raise ReviewDispatchBudgetError("review_dispatch_budget must be at least 1")
        object.__setattr__(self, "attempt_id", attempt_id)

    @property
    def path(self) -> Path:
        return fanout_review_dispatch_budget_path(self.paths, self.fanout_id)

    def reserve(self, role: str, unit_id: str) -> ReviewDispatchReservation:
        normalized_role = normalized_review_role(role)
        if normalized_role is None:
            raise ReviewDispatchBudgetError(f"role is not review-budgeted: {role!r}")
        outcome: ReviewDispatchReservation | None = None

        def _reserve(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal outcome
            _validate_state(state, self.fanout_id)
            if not state:
                state.update(
                    {
                        "schema_version": REVIEW_DISPATCH_BUDGET_SCHEMA_VERSION,
                        "fanout_id": self.fanout_id,
                        "current_attempt_id": self.attempt_id,
                        "attempts": {},
                        "claim_boundary": REVIEW_DISPATCH_BUDGET_CLAIM_BOUNDARY,
                    }
                )
            current_attempt = str(state["current_attempt_id"])
            if current_attempt != self.attempt_id and not self.progressed:
                current = state["attempts"].get(current_attempt, {})
                role_state = current.get("roles", {}).get(normalized_role, {})
                outcome = ReviewDispatchReservation(
                    "attempt_not_progressed",
                    self.attempt_id,
                    normalized_role,
                    int(current.get("limit", self.limit)),
                    int(role_state.get("reserved", 0)),
                )
                return state
            if current_attempt != self.attempt_id:
                state["current_attempt_id"] = self.attempt_id
            attempts = state["attempts"]
            attempt = attempts.setdefault(
                self.attempt_id,
                {"limit": self.limit, "progressed": bool(self.progressed), "roles": {}},
            )
            stored_limit = int(attempt.get("limit", self.limit))
            roles = attempt.setdefault("roles", {})
            role_state = roles.setdefault(normalized_role, {"reserved": 0, "unit_ids": []})
            reserved = int(role_state.get("reserved", 0))
            if stored_limit != self.limit:
                outcome = ReviewDispatchReservation(
                    "configuration_mismatch", self.attempt_id, normalized_role, stored_limit, reserved
                )
                return state
            if reserved >= stored_limit:
                outcome = ReviewDispatchReservation(
                    "exhausted", self.attempt_id, normalized_role, stored_limit, reserved
                )
                return state
            role_state["reserved"] = reserved + 1
            role_state.setdefault("unit_ids", []).append(unit_id)
            outcome = ReviewDispatchReservation(
                "reserved", self.attempt_id, normalized_role, stored_limit, reserved + 1
            )
            return state

        locked_json_update(self.path, _reserve, default={}, private=True)
        if outcome is None:
            raise ReviewDispatchBudgetError("review dispatch reservation produced no outcome")
        return outcome


def normalized_review_role(role: str) -> str | None:
    normalized = "-".join(str(role or "").strip().casefold().replace("_", "-").split())
    return _REVIEW_ROLE_ALIASES.get(normalized)


def _validate_state(state: dict[str, Any], fanout_id: str) -> None:
    if not state:
        return
    if state.get("schema_version") != REVIEW_DISPATCH_BUDGET_SCHEMA_VERSION:
        raise ReviewDispatchBudgetError("stored review dispatch budget schema is unsupported")
    if state.get("fanout_id") != fanout_id:
        raise ReviewDispatchBudgetError("stored review dispatch budget fanout_id does not match")
    if not isinstance(state.get("attempts"), dict):
        raise ReviewDispatchBudgetError("stored review dispatch budget attempts must be an object")
