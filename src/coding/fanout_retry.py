"""Deterministic retry policy for units dispatched by `omh coding fanout dispatch`.

Two disjoint questions, decided mechanically and in this order:

1. **Is the failure transient?** A rate limit, a provider overload, an HTTP
   5xx, a socket reset or hang-up is the executor's transport failing, not the
   unit's work being wrong. A non-zero exit from a real test or verification
   run is the unit's answer, and re-running it would only produce the same
   answer more slowly. Only the first class is retryable.
2. **Is the unit replay-safe?** Even a transient failure must not be replayed
   once the unit has produced an observed side effect. A unit that wrote files
   into its worktree or emitted a result artifact has to be CONTINUED or
   recovered by hand, never silently re-run from base, because a re-dispatch
   rebuilds the worktree and destroys exactly the work the failure left behind.

Both answers are computed from observed evidence only -- an exit code, the
bounded output tails already captured, and the recovery probe's own measurement
of the worktree. Nothing here calls a model, and nothing here decides that a
retried unit is correct: a retry is another attempt, not evidence.

The backoff is jittered capped exponential so a fanout of N units that all hit
the same provider limit does not retry in lockstep and re-trigger it. The
random source and the clock are injected, so the policy is exercised without a
single sleep in the test suite.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

RETRY_POLICY_SCHEMA_VERSION = "fanout_retry_policy/v1"
RETRY_CLAIM_BOUNDARY = (
    "A retry record states how many times omh started this unit's process and why it stopped trying. "
    "It is not verification, review, CI, or merge evidence, and a unit that succeeded on a later "
    "attempt is no more verified than one that succeeded on the first."
)

# Attempts BEYOND the first, per unit. Deliberately small: each attempt is a
# whole agent-CLI run costing minutes and provider budget, unlike the
# single-request retry ladders this formula comes from. Two retries covers the
# transient window a rate limit or a reset opens without turning one refused
# unit into a quarter-hour of re-spawning.
FANOUT_MAX_RETRIES = 2
# First retry waits ~2s, the second ~4s, before jitter. The cap exists so a
# raised `FANOUT_MAX_RETRIES` cannot grow the wait without bound.
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_MAX_DELAY_SECONDS = 30.0
# Jitter spreads each delay across [75%, 100%] of the computed value. Full
# jitter (0-100%) would let two units collide at ~0s; this floor keeps the
# spread meaningful while guaranteeing the backoff still grows.
RETRY_JITTER_FLOOR = 0.75

# Failure classes. Only `transient_*` is ever retryable; every other class is
# the unit's own answer and is terminal by construction.
CLASS_PROVIDER_LIMIT = "transient_provider_limit"
CLASS_TRANSPORT = "transient_transport"
CLASS_TERMINAL_FAILURE = "terminal_failure"
CLASS_EXECUTOR_MISSING = "executor_missing"
CLASS_UNIT_TIMEOUT = "unit_timeout"

# Deterministic transport shapes, matched case-insensitively over the bounded
# stdout/stderr tails of a FAILED spawn only. Anchored the same way
# `_LIMIT_SHAPED_PATTERNS` in fanout_dispatch is: a bare "500" or "reset"
# would match a line number or a git message and retry a unit whose tests
# simply failed. Provider-limit shapes are NOT repeated here -- the dispatcher
# already classifies those for the limit-signal record and passes the verdict
# in, so the two surfaces cannot drift apart.
_TRANSPORT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("http_500", "status 500"),
    ("http_500", "http 500"),
    ("http_502", "status 502"),
    ("http_502", "http 502"),
    ("http_502", "bad gateway"),
    ("http_503", "status 503"),
    ("http_503", "http 503"),
    ("http_503", "service unavailable"),
    ("http_504", "status 504"),
    ("http_504", "http 504"),
    ("http_504", "gateway timeout"),
    ("provider_overloaded", "overloaded_error"),
    ("provider_overloaded", "server is overloaded"),
    ("provider_overloaded", "model is overloaded"),
    ("provider_error", "internal server error"),
    ("socket_reset", "connection reset by peer"),
    ("socket_reset", "econnreset"),
    ("socket_hang_up", "socket hang up"),
    ("connection_refused", "connection refused"),
    ("connection_refused", "econnrefused"),
    ("connection_closed", "connection closed before"),
    ("upstream_reset", "upstream connect error"),
    ("transport_failed", "fetch failed"),
)

# Replay-safety verdicts, in OMH's evidence vocabulary. `no_observed_side_effect`
# is the ONLY one that permits a re-dispatch; the other two both mean the unit
# has to be continued or recovered by hand.
REPLAY_SAFE = "no_observed_side_effect"
REPLAY_UNSAFE_SIDE_EFFECTS = "observed_side_effects"
REPLAY_UNSAFE_UNMEASURED = "side_effects_unmeasured"


def transport_failure_label(output_tail: str, stderr_tail: str) -> str:
    """The transport shape a failed spawn's tails match, or an empty string."""
    haystack = f"{output_tail}\n{stderr_tail}".lower()
    for label, needle in _TRANSPORT_PATTERNS:
        if needle in haystack:
            return label
    return ""


def classify_unit_failure(
    *,
    exit_code: int,
    output_tail: str,
    stderr_tail: str,
    limit_shaped: str = "",
) -> dict[str, Any]:
    """Split one observed unit failure into retryable transport vs terminal.

    `limit_shaped` is the dispatcher's own limit-signal label for this failure,
    passed in rather than re-derived so the retry verdict and the persisted
    provider-limit evidence can never disagree.

    A timeout and a missing executable get their own terminal classes: both are
    non-zero exits that are emphatically not transport, and naming them stops a
    later reader from assuming the terminal bucket is only test failures.
    """
    if exit_code == 0:
        return {"retryable": False, "failure_class": "", "failure_label": ""}
    if exit_code == 127:
        return {"retryable": False, "failure_class": CLASS_EXECUTOR_MISSING, "failure_label": ""}
    if exit_code == 124:
        # A unit that burned its whole timeout is not a transient transport
        # blip, and it has almost certainly been writing files the entire time.
        return {"retryable": False, "failure_class": CLASS_UNIT_TIMEOUT, "failure_label": ""}
    if limit_shaped:
        return {"retryable": True, "failure_class": CLASS_PROVIDER_LIMIT, "failure_label": limit_shaped}
    label = transport_failure_label(output_tail, stderr_tail)
    if label:
        return {"retryable": True, "failure_class": CLASS_TRANSPORT, "failure_label": label}
    # Everything else -- above all a real non-zero exit from a test or
    # verification run -- is the unit's answer. Retrying it would re-run known
    # work to reach the same verdict.
    return {"retryable": False, "failure_class": CLASS_TERMINAL_FAILURE, "failure_label": ""}


def replay_safety(
    recovery: Mapping[str, Any] | None,
    *,
    artifact_observed: bool = False,
) -> dict[str, Any]:
    """Whether this unit may be re-dispatched, from what was observed of it.

    The predicate is the recovery probe's own measurement, not a guess: it
    already answers "what did this unit leave in its worktree" with three
    distinguishable outcomes, and the middle one -- `capture_failed`, "I could
    not tell" -- is exactly the case that must fail closed. `artifact_observed`
    covers the other side effect a unit can produce without touching the
    worktree: a result sidecar the intake path would read.
    """
    if artifact_observed:
        return {"replay_safe": False, "replay_verdict": REPLAY_UNSAFE_SIDE_EFFECTS, "side_effect": "unit_result_artifact"}
    outcome = str((recovery or {}).get("outcome", "")) if isinstance(recovery, Mapping) else ""
    if outcome == "no_changes":
        return {"replay_safe": True, "replay_verdict": REPLAY_SAFE, "side_effect": "none"}
    if outcome == "recovery_available":
        return {
            "replay_safe": False,
            "replay_verdict": REPLAY_UNSAFE_SIDE_EFFECTS,
            "side_effect": "worktree_changes",
        }
    # No record, or a probe that could not answer: never replay on an
    # unmeasured worktree.
    return {"replay_safe": False, "replay_verdict": REPLAY_UNSAFE_UNMEASURED, "side_effect": "unmeasured"}


def retry_delay_seconds(
    attempt: int,
    *,
    rng: Callable[[], float],
    base: float = RETRY_BASE_DELAY_SECONDS,
    cap: float = RETRY_MAX_DELAY_SECONDS,
) -> float:
    """`min(base * 2^(attempt-1), cap)`, scaled by 75-100% jitter.

    `attempt` is the attempt that just failed, so the first retry waits one
    base delay. `rng` returns a float in [0, 1) -- `random.random` in
    production, a fixed value in tests, which is what makes the whole ladder
    assertable without a clock.
    """
    step = max(1, int(attempt))
    capped = min(base * (2 ** (step - 1)), cap)
    jitter = RETRY_JITTER_FLOOR + (1.0 - RETRY_JITTER_FLOOR) * min(max(rng(), 0.0), 1.0)
    return round(capped * jitter, 6)


def evaluate_unit_retry(
    *,
    attempt: int,
    exit_code: int,
    output_tail: str,
    stderr_tail: str,
    limit_shaped: str = "",
    recovery: Mapping[str, Any] | None = None,
    artifact_observed: bool = False,
    max_retries: int = FANOUT_MAX_RETRIES,
    rng: Callable[[], float],
) -> dict[str, Any]:
    """One retry decision, complete with the reason it went that way.

    The record is the point: a unit that was NOT retried has to say whether it
    was terminal, out of attempts, or blocked by its own side effects, because
    those three lead an operator to three different next actions.
    """
    decision: dict[str, Any] = {
        "attempt": int(attempt),
        "exit_code": int(exit_code),
        "retry": False,
        **classify_unit_failure(
            exit_code=exit_code,
            output_tail=output_tail,
            stderr_tail=stderr_tail,
            limit_shaped=limit_shaped,
        ),
    }
    if not decision["retryable"]:
        decision["decision"] = "terminal"
        return decision
    if int(attempt) > int(max_retries):
        decision["decision"] = "retries_exhausted"
        decision["max_retries"] = int(max_retries)
        return decision
    safety = replay_safety(recovery, artifact_observed=artifact_observed)
    decision.update(safety)
    if not safety["replay_safe"]:
        # The one outcome this whole module exists for: a transient failure
        # that must NOT be replayed. It goes out through the recovery path an
        # operator already reads, as work to continue rather than redo.
        decision["decision"] = "surfaced_for_continuation"
        return decision
    decision["retry"] = True
    decision["decision"] = "retrying"
    decision["delay_seconds"] = retry_delay_seconds(attempt, rng=rng)
    return decision
