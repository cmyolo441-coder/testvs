"""The fanout repair loop's contract: when a unit is re-dispatched, and when it stops.

A unit that declares `max_repair_attempts` at freeze opts into one behaviour:
when the dispatcher itself ran the unit's declared checks and saw one exit
nonzero, the dispatcher re-dispatches that unit's executor in the SAME
worktree -- no reset, prior commits kept -- with a repair brief naming the
failing checks. It stops on exactly two criteria, both read from record
fields and never from text:

- every declared check is observed passing (the unit reaches `verified`), or
- the declared number of repair attempts has been spent, and the unit is
  recorded `blocked` with the last observed failing check named.

A failure the dispatcher did not observe as a process exit -- a check that
timed out or could not start, a task-linked postcondition that could not be
resolved, a process that crashed or returned no valid result -- is not a repair
trigger: nothing observed says what to repair, so the loop stops and the
unit's existing state speaks for it. A reproduction unit's EXPECTED nonzero
exit rides its own receipt, never a check row, so it can never trigger one.

The attempt count lives in the observation journal, not in a process: every
repair spawn appends `repair_attempt_started`, every observed verdict appends
`repair_attempt_observed`, and `project_unit_repair` reads the count back, so
a later dispatch continues it and never resets it. Everything here is
metadata: commands, exit codes, and closed-vocabulary kinds, never output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from .fanout_contracts import FanoutContractError

FANOUT_UNIT_REPAIR_SCHEMA_VERSION = "fanout_unit_repair/v1"
# A ceiling, not a default, shared with the todo evidence chain and the
# teammate lane: an operator who needs more than three repair passes on one
# unit boundary has a split problem, not a persistence problem.
MAX_REPAIR_ATTEMPTS = 3
REPAIR_ATTEMPT_STARTED_EVENT = "repair_attempt_started"
REPAIR_ATTEMPT_OBSERVED_EVENT = "repair_attempt_observed"
FANOUT_UNIT_REPAIR_CLAIM_BOUNDARY = (
    "A repair record counts dispatcher-started repair attempts and the dispatcher-observed exit status of "
    "the unit's declared checks after each one. It is not evidence that a repair is correct beyond those "
    "checks, and a unit it reports as verified is no more verified than one that passed first time."
)



# Projection states. `pending` is the only one a later dispatch acts on;
# `blocked` and `stopped` are terminal, `passed` is done.
REPAIR_STATE_NONE = "none"
REPAIR_STATE_PASSED = "passed"
REPAIR_STATE_PENDING = "pending"
REPAIR_STATE_BLOCKED = "blocked"
REPAIR_STATE_STOPPED = "stopped"

# The terminal blocked reasons. `repair_budget_exhausted` is spelled exactly as
# the todo evidence chain and the teammate lane spell it. Either one is also
# the unit's `unit_state_reason` when the loop stops blocked; `unit_state`
# itself stays `failed`, so the unit vocabulary's terminal pair is unchanged.
REPAIR_BUDGET_EXHAUSTED = "repair_budget_exhausted"
# The unit's worktree vanished while repairs were owed: continuing it is
# impossible and rebuilding it from base would discard the work being repaired.
REPAIR_WORKTREE_MISSING = "repair_worktree_missing"
REPAIR_BLOCKED_REASONS = (REPAIR_BUDGET_EXHAUSTED, REPAIR_WORKTREE_MISSING)

# The status of a unit a dispatch skipped because another dispatch holds its
# repair loop: two executors in one worktree, or two readers of one journal
# count, is exactly what the loop's lock exists to prevent.
REPAIR_IN_FLIGHT_STATUS = "repair_in_flight"

STOP_CHECKS_PASSED = "checks_passed"
STOP_NOT_REPAIRABLE = "not_repairable"

# The one failure shape that triggers a repair: the dispatcher ran the check
# and saw the process exit nonzero. `_run_verification_command` names that
# `reason="nonzero"` with `exit_code_source="process"`.
_REPAIRABLE_FAILURE_KIND = "nonzero"
_REPAIRABLE_EXIT_CODE_SOURCE = "process"
# Bounds for what one journal event (and the repair brief) carries. Nine is
# eight declared commands plus the task-linked postcondition. A task-linked
# command is a runner plus every selected test path and can be far longer
# than any declared command, so the trigger pairs checks on the FULL command
# and only the journal and brief copies are cut to this length.
_MAX_REPAIR_CHECKS = 9
_MAX_REPAIR_COMMAND_CHARS = 512
_FAILURE_KINDS = ("nonzero", "deadline", "missing_binary", "spawn_error", "denial")


def normalized_max_repair_attempts(unit: Mapping[str, object], index: int) -> int:
    """The declared repair budget, 0 when undeclared; refused when unusable.

    A budget above zero needs something the dispatcher can observe: declared
    verification commands or checks, or a task-linked test runner. A loop with
    nothing to run could only ever stop on its ceiling.
    """
    value = unit.get("max_repair_attempts")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_REPAIR_ATTEMPTS:
        raise FanoutContractError(
            f"unit at index {index} max_repair_attempts must be an integer from 0 to {MAX_REPAIR_ATTEMPTS}"
        )
    if value and not (
        unit.get("verification_commands") or unit.get("verification_checks") or unit.get("task_linked_test_runner")
    ):
        raise FanoutContractError(
            f"unit at index {index} declares max_repair_attempts without a check the dispatcher can run; "
            "declare verification_commands, verification_checks, or task_linked_test_runner"
        )
    return value


def declared_max_repair_attempts(unit: Mapping[str, Any]) -> int:
    """The frozen unit's repair budget; anything but a bounded int reads as 0."""
    value = unit.get("max_repair_attempts")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= MAX_REPAIR_ATTEMPTS:
        return 0
    return value


def observed_check_failure(command: str, reason: str, code: int | None, source: str) -> dict[str, Any]:
    """One failing check as the dispatcher observed it, keyed by its full command."""
    return {
        "command": str(command),
        "exit_code": code if isinstance(code, int) and not isinstance(code, bool) else None,
        "failure_kind": reason if reason in _FAILURE_KINDS else "denial",
        "exit_code_source": source if source in ("process", "not_observed") else "not_observed",
    }


def repair_trigger_checks(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The failing checks that earn a repair attempt, or [] when nothing does.

    Every rung below verification must hold -- the process exited 0 and its
    result validated -- and verification must have run and failed. Rows the
    plan engine recorded `skipped` (blocked behind a failed dependency, or
    deferred behind the fan-in) never ran and say nothing; every other failed
    row must be dispatcher-observed with a captured nonzero exit, paired to
    its capture by command in order. One failure of any other shape makes the
    whole set non-repairable: a repair could not make that check pass.
    """
    if not (
        result.get("status") == "completed"
        and result.get("process_succeeded")
        and result.get("result_schema_valid")
        and result.get("verification_status") == "failed"
    ):
        return []
    captured: dict[str, list[Mapping[str, Any]]] = {}
    for entry in result.get("verification_observed_failures", []) or []:
        if isinstance(entry, Mapping):
            captured.setdefault(str(entry.get("command")), []).append(entry)
    failing: list[dict[str, Any]] = []
    for row in result.get("verification_checks", []) or []:
        if not isinstance(row, Mapping) or row.get("status") in ("passed", "skipped"):
            continue
        queue = captured.get(str(row.get("command", ""))) or []
        entry = queue.pop(0) if queue else None
        if (
            row.get("status") != "failed"
            or row.get("observed_by") != "dispatcher"
            or entry is None
            or entry.get("failure_kind") != _REPAIRABLE_FAILURE_KIND
            or entry.get("exit_code_source") != _REPAIRABLE_EXIT_CODE_SOURCE
        ):
            return []
        failing.append(
            {"command": entry["command"], "exit_code": entry["exit_code"], "failure_kind": entry["failure_kind"]}
        )
    return failing


def _bounded_command(command: str) -> str:
    """The command cut to `_MAX_REPAIR_COMMAND_CHARS` characters and to as many
    UTF-8 bytes, at a character boundary. The byte bound keeps a non-ASCII
    command from outgrowing the brief's prompt ceiling; ASCII is unaffected."""
    text = command[:_MAX_REPAIR_COMMAND_CHARS]
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_REPAIR_COMMAND_CHARS:
        return text
    return encoded[:_MAX_REPAIR_COMMAND_CHARS].decode("utf-8", errors="ignore")


def journal_repair_checks(checks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The bounded copy of a failing-check list the journal and the brief carry."""
    return [
        {
            "command": _bounded_command(str(check.get("command", ""))),
            "exit_code": check.get("exit_code"),
            "failure_kind": str(check.get("failure_kind", "")),
        }
        for check in list(checks)[:_MAX_REPAIR_CHECKS]
    ]


def repair_brief_prompt(*, attempt: int, max_repair_attempts: int, failing_checks: Sequence[Mapping[str, Any]]) -> str:
    """The bounded prompt section a repair dispatch appends. Metadata only.

    No output text rides it: the executor runs each named command itself in
    the worktree it is handed, which holds its own prior commits.
    """
    return "\n[Repair attempt]\n" + json.dumps(
        {
            "repair_attempt": attempt,
            "max_repair_attempts": max_repair_attempts,
            "observed_by": "dispatcher",
            "failing_checks": journal_repair_checks(failing_checks),
            "worktree": "same_worktree_and_branch_prior_commits_kept",
            "authority": "original_goal_scope_and_criteria_unchanged",
        },
        sort_keys=True,
        # The brief rides argv as UTF-8 text; `\uXXXX` escapes would only
        # multiply a non-ASCII command's bytes by up to six.
        ensure_ascii=False,
    )


def bounded_repair_checks(value: object) -> list[dict[str, Any]] | None:
    """The journal-safe copy of a failing-check list, or None when malformed."""
    if not isinstance(value, list) or len(value) > _MAX_REPAIR_CHECKS:
        return None
    checks: list[dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            return None
        command, code, kind = entry.get("command"), entry.get("exit_code"), entry.get("failure_kind")
        if not isinstance(command, str) or not command or len(command) > _MAX_REPAIR_COMMAND_CHARS:
            return None
        if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
            return None
        if kind not in _FAILURE_KINDS:
            return None
        checks.append({"command": command, "exit_code": code, "failure_kind": kind})
    return checks


def bounded_repair_attempt(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_REPAIR_ATTEMPTS:
        return None
    return value


def _check_record(checks: Sequence[Mapping[str, Any]], observed_at: str) -> dict[str, Any] | None:
    """The shared `{command, exit_code, observed_at}` shape for the first failing check."""
    if not checks:
        return None
    first = checks[0]
    return {"command": str(first.get("command", "")), "exit_code": first.get("exit_code"), "observed_at": observed_at}


def project_unit_repair(events: Sequence[Mapping[str, Any]], *, run_id: str, max_repair_attempts: int) -> dict[str, Any]:
    """Fold one run's repair events, in append order, into its loop state.

    `attempts_used` counts every started attempt, so a later dispatch
    continues from it and a duplicate number cannot hide a spent attempt. The
    latest verdict decides the state; a started attempt with no verdict after
    it (an interrupted or killed spawn) reads as still failing on the checks
    that triggered it, so it stays pending with the attempt counted. Each
    attempt names the check whose observed failure triggered it, stamped with
    when the dispatcher saw it.
    """
    attempts: list[dict[str, Any]] = []
    state = REPAIR_STATE_NONE
    blocked_reason = ""
    checks: list[dict[str, Any]] = []
    failure_observed_at = ""
    for event in events:
        if event.get("run_id") != run_id:
            continue
        name = event.get("event")
        attempt = bounded_repair_attempt(event.get("repair_attempt"))
        event_checks = bounded_repair_checks(event.get("repair_checks")) or []
        observed_at = str(event.get("observed_at", "") or "")
        if name == REPAIR_ATTEMPT_STARTED_EVENT and attempt:
            attempts.append({
                "attempt": attempt,
                "started_at": observed_at,
                "check": _check_record(event_checks, failure_observed_at),
            })
            state, checks = "failing", event_checks
        elif name == REPAIR_ATTEMPT_OBSERVED_EVENT:
            status = event.get("status")
            if status == "observed":
                state, checks = REPAIR_STATE_PASSED, []
            elif status == "failed":
                state, checks, failure_observed_at = "failing", event_checks, observed_at
            elif status == "blocked":
                reason = event.get("repair_stop_reason")
                blocked_reason = reason if reason in REPAIR_BLOCKED_REASONS else REPAIR_BUDGET_EXHAUSTED
                state, checks = REPAIR_STATE_BLOCKED, event_checks
                # A budget verdict is itself an observed failure; a missing
                # worktree only carries the last one forward, with its stamp.
                if event_checks and reason != REPAIR_WORKTREE_MISSING:
                    failure_observed_at = observed_at
    attempts_used = len(attempts)
    if state == "failing":
        if not checks:
            state = REPAIR_STATE_STOPPED
        elif attempts_used >= max_repair_attempts:
            state, blocked_reason = REPAIR_STATE_BLOCKED, REPAIR_BUDGET_EXHAUSTED
        else:
            state = REPAIR_STATE_PENDING
    return {
        "state": state,
        "blocked_reason": blocked_reason if state == REPAIR_STATE_BLOCKED else "",
        "attempts_used": attempts_used,
        "attempts": attempts,
        "failing_checks": checks,
        "last_failing_check": _check_record(checks, failure_observed_at),
    }


def repair_record(projection: Mapping[str, Any], *, max_repair_attempts: int) -> dict[str, Any]:
    """The `fanout_unit_repair/v1` block a unit result and the surfaces carry."""
    state = str(projection.get("state", REPAIR_STATE_NONE))
    stop_reason = {
        REPAIR_STATE_PASSED: STOP_CHECKS_PASSED,
        REPAIR_STATE_STOPPED: STOP_NOT_REPAIRABLE,
        REPAIR_STATE_BLOCKED: str(projection.get("blocked_reason") or REPAIR_BUDGET_EXHAUSTED),
    }.get(state, "")
    record: dict[str, Any] = {
        "schema_version": FANOUT_UNIT_REPAIR_SCHEMA_VERSION,
        "max_repair_attempts": max_repair_attempts,
        "attempts_used": int(projection.get("attempts_used", 0) or 0),
        "attempts": [dict(entry) for entry in projection.get("attempts", []) or []],
        "state": state,
        "status": state,
        "stop_reason": stop_reason,
        "last_failing_checks": list(projection.get("failing_checks", []) or []),
        "observed_by": "dispatcher",
        "claim_boundary": FANOUT_UNIT_REPAIR_CLAIM_BOUNDARY,
    }
    if state == REPAIR_STATE_BLOCKED:
        # The shared terminal vocabulary: the exact reason, and always the last
        # failing check the dispatcher observed before the loop was stopped.
        record["blocked_reason"] = stop_reason
        record["last_failing_check"] = projection.get("last_failing_check")
    return record
