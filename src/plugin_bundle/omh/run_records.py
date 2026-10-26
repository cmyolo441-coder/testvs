"""The on-disk run-record format: one definition for both sides of the seam.

The control plane (`omh.*` outside this bundle) WRITES run, dispatch, and
receipt files; the Hermes-side readers in this bundle READ them. The bundle
cannot import `omh.*`, so the bundle owns the format and the writers import it
from here. Before this module each tree spelled every file name and schema
version itself and parity tests caught drift after the fact; now
`tests/test_run_record_format_policy.py` fails on a second spelling instead.

Names and versions only. No IO lives here: locking, journaling, and every write
stay with the writers.
"""

from __future__ import annotations

from typing import Final


# --- File names --------------------------------------------------------------

# `<runtime>/runs/<run_id>/` holds the run record and its optional delegation
# records.
RUN_FILE: Final[str] = "run.json"
CODING_DELEGATION_FILE: Final[str] = "coding_delegation.json"
DELEGATION_FILE: Final[str] = "delegation.json"
WRAPPER_FILE: Final[str] = "wrapper.json"
REVIEW_FILE: Final[str] = "review.json"
CI_FILE: Final[str] = "ci.json"
MERGE_FILE: Final[str] = "merge.json"
# The basename of every append-only event stream: a run's events, the runtime
# observation journal, an executor-progress binding's events, and a wrapper
# session's events. One basename, several directories.
EVENTS_FILE: Final[str] = "events.jsonl"
# `executor_progress/<binding_id>/` holds the binding and its reports beside
# its events.
EXECUTOR_PROGRESS_BINDING_FILE: Final[str] = "binding.json"
EXECUTOR_PROGRESS_REPORTS_FILE: Final[str] = "reports.jsonl"
# `<runtime>/wrapper_sessions/<session_id>/` holds the executor session record.
EXECUTOR_SESSION_FILE: Final[str] = "executor_session.json"
# Runtime-wide, under `<runtime>/journal/` beside the observation journal.
EXTERNAL_EFFECT_RECEIPT_STORE_NAME: Final[str] = "external_effect_receipts.jsonl"
# `<omh_home>/fanout/<fanout_id>/` holds one dispatch's contract and summary.
DISPATCH_SUMMARY_FILE: Final[str] = "dispatch_summary.json"
FANOUT_CONTRACT_FILE: Final[str] = "fanout_contract.json"
CONTRACT_PROVENANCE_FILE: Final[str] = "contract_provenance.json"
# A run's context-budget ledger.
CONTEXT_BUDGET_FILE: Final[str] = "context_budget.json"


# --- Schema versions ---------------------------------------------------------

EXTERNAL_EFFECT_RECEIPT_SCHEMA_VERSION: Final[str] = "external_effect_receipt/v1"
FANOUT_DISPATCH_SCHEMA_VERSION: Final[str] = "fanout_dispatch_summary/v1"
FANOUT_CONTRACT_SCHEMA_VERSION: Final[str] = "fanout_contract/v2"
FANOUT_CONTRACT_PROVENANCE_SCHEMA_VERSION: Final[str] = "fanout_contract_provenance/v1"
INFLIGHT_MARKER_SCHEMA_VERSION: Final[str] = "omh_inflight_marker/v1"
EXECUTOR_PROGRESS_BINDING_SCHEMA_VERSION: Final[str] = "omh_executor_progress_binding/v1"
EXECUTOR_PROGRESS_REPORT_SCHEMA_VERSION: Final[str] = "omh_progress_report/v1"
EXECUTOR_PROGRESS_EVENT_SCHEMA_VERSION: Final[str] = "omh_progress_event/v1"
LIFECYCLE_PROJECTION_SCHEMA_VERSION: Final[str] = "omh_lifecycle_projection/v1"
OBSERVATION_EVENT_SCHEMA_VERSION: Final[str] = "omh_observation_event/v1"
RUN_CONTEXT_BUDGET_SCHEMA_VERSION: Final[str] = "omh_run_context_budget/v1"


# --- Vocabularies ------------------------------------------------------------

# The one addition to the shipped progress vocabulary. It carries the bounded
# raw `source_event` verbatim so an owner word this repo does not map stays
# VISIBLE instead of being rounded to a neighbouring event. It is deliberately
# NOT a terminal or closing event type: an unrecognized word ends nothing.
UNMAPPED_SOURCE_EVENT: Final[str] = "unmapped_source_event"
# The shared executor-progress event vocabulary. The first twelve entries keep
# their original order; the CLI derives `omh runtime progress observe --event`
# choices from it. An event type missing here is silently dropped at the read
# boundary, which is why there is one definition and not a copy per side.
EXECUTOR_PROGRESS_EVENT_TYPES: Final[tuple[str, ...]] = (
    "executor_dispatched",
    "repo_exploration",
    "running_no_diff_observed",
    "diff_started",
    "tests_started",
    "tests_failed",
    "tests_passed",
    "executor_completed",
    "executor_blocked",
    "executor_failed",
    # An observed cancellation. It sits beside the other three end-state words
    # rather than folding into `executor_failed`, because "this ran and did not
    # work" and "someone stopped this" call for different recovery. The lane
    # (`executor_progress.infer_progress_event_type`) is what decides whether a
    # word the normalizer translates is CORROBORATED; a run whose process was
    # never observed to stop cannot reach this event by narration alone.
    "executor_cancelled",
    "reported_change_not_observed",
    "progress_observed",
    UNMAPPED_SOURCE_EVENT,
)
# `pi_family_runtime` is one profile covering every pi-family host CLI (`pi`,
# pi's distribution host, `opencode`) because the binding answers "which lane is working", not "which
# binary was on PATH". A profile missing here rejects every binding, event, and
# report that lane writes.
EXECUTOR_PROGRESS_PROFILES: Final[tuple[str, ...]] = ("codex", "claude_code", "hermes_local", "pi_family_runtime")
EXECUTOR_PROGRESS_BINDING_STATES: Final[tuple[str, ...]] = ("active", "stale", "expired", "closed")

# `cancelled` is an OBSERVED result, not an unobserved one. Recording it still
# requires the same `observed=True` evidence every other terminal result
# requires: OMH records that a cancellation was observed (process termination or
# an authoritative executor result), never that one was requested. A request to
# cancel is not a terminal result and has no member here.
OBSERVED_RESULTS: Final[tuple[str, ...]] = ("completed", "blocked", "failed", "cancelled")
# The observed results that END a target without completing it. `cancelled`
# belongs here because a run someone stopped is over: leaving it out is what
# made a cancelled run read as one that had merely not got there yet.
TERMINAL_JOURNAL_STATUSES: Final[frozenset[str]] = frozenset(
    result for result in OBSERVED_RESULTS if result != "completed"
)

EXTERNAL_EFFECT_CLAIM_BOUNDARY: Final[str] = (
    "An external effect receipt is one acting surface's observation of one external effect. "
    "It is not execution, verification, review, CI, merge-readiness, or merge evidence for any other effect."
)
# The run-summary claims that assert an external effect, and the effect kind
# whose receipt has to back each one. An effect id is `<kind>:<run_id>`.
RECEIPT_BACKED_RUN_CLAIMS: Final[dict[str, str]] = {
    "review_observed": "review",
    "ci_observed": "ci",
    "merge_observed": "merge",
}

# Closed status vocabulary a dispatch-summary row can carry. Anything else
# collapses to `prepared_not_observed`, which is the honest reading of "we
# wrote it down but never watched it run".
DISPATCH_STATUS_VOCABULARY: Final[tuple[str, ...]] = (
    "running",
    "completed",
    "failed",
    "worktree_failed",
    "prepared_not_observed",
)
