"""Mechanical backstop for the anti-polling / no-raw-log-dumping policy.

`build_coding_progress_reporting_policy` declares `timed_polling_rejected` and
`raw_log_dumping_rejected`, but declarative text loses to a supervising agent
under pressure. This module keeps a small local ledger of how many bytes each
run's observe/show surfaces have already emitted into agent context. Past the
budget the surfaces degrade to summary-only output plus artifact pointers, so
the policy becomes mechanically true instead of aspirational.

The ledger is local, metadata-only, and best-effort: a ledger write failure
never blocks an observation surface.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..context_safety import RUN_CONTEXT_BUDGET_BYTES
from ..local_store import ensure_dir, locked_json_update, read_json_object, utc_now
from ..paths import OmhPaths
from ..plugin_bundle.omh.run_records import (
    CONTEXT_BUDGET_FILE as CONTEXT_BUDGET_LEDGER_NAME,
    RUN_CONTEXT_BUDGET_SCHEMA_VERSION,
)


RUN_UNCHANGED_SCHEMA_VERSION = "omh_run_show_unchanged/v1"
PROGRESS_STATUS_UNCHANGED_SCHEMA_VERSION = "omh_progress_status_unchanged/v1"
CAPABILITY_PROJECTION_SUMMARY_ONLY_SCHEMA_VERSION = "omh_capability_projection_summary_only/v1"

# The ledger bucket a task-scoped capability projection spends from. A
# projection is emitted per task, not per run, so the ledger key is the task id
# the caller supplies rather than a run id.
CAPABILITY_PROJECTION_SURFACE = "capability_projection"

# `progress-status` spans every binding rather than one run, so it needs a
# ledger bucket that is not a run id. The double underscores keep it from
# colliding with a real one.
PROGRESS_STATUS_LEDGER_KEY = "__executor_progress_status__"


def run_show_surface(*, full: bool, history_limit: object = None) -> str:
    """Ledger key for a `runtime show` emission.

    The projection varies with the history limit, so the comparison key has to
    as well. One shared key across limits means alternating `--limit 20` and
    `--limit 50` never matches and suppression silently never fires.
    """
    return f"runtime_show:{'full' if full else history_limit}"


# Bookkeeping a progress observation rewrites even when it reports nothing.
# `update_binding_reporter_state` stamps these on every observation, including
# suppressed ones, so leaving them in the fingerprint means one intervening
# `progress-observe` makes an otherwise identical projection look new. A
# supervising agent always observes between status checks, so that defeated the
# unchanged check entirely in real use while unit tests -- which call `show`
# twice in a row -- still passed.
_FINGERPRINT_VOLATILE_KEYS = frozenset(
    {
        "updated_at",
        "last_observed_at",
        "last_observed_signal_hash",
        "last_observed_event_count",
        "last_observed_artifact_sha256",
        # The tally of observations deliberately not reported. Counting
        # suppressions is itself a change in the projection, so leaving this in
        # meant every suppressed observation un-suppressed the next status
        # check -- the check defeating itself.
        "suppressed_duplicate_count",
    }
)


def _without_volatile_bookkeeping(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _without_volatile_bookkeeping(value)
            for key, value in payload.items()
            if key not in _FINGERPRINT_VOLATILE_KEYS
        }
    if isinstance(payload, list):
        return [_without_volatile_bookkeeping(item) for item in payload]
    return payload


def smaller_payload(original: dict[str, Any], compacted: dict[str, Any]) -> dict[str, Any]:
    """Return whichever payload is actually smaller.

    A compact replacement carries fixed overhead -- schema version, claim
    boundary, next action, the full-output hint -- so when the real projection
    is nearly empty the "compact" form is the larger of the two. Measured on a
    live fanout dispatch: `progress-status` with no active binding cost 280
    bytes in full and 854 once "compacted".

    Suppression exists to shrink what reaches agent context. It must never be
    the reason output grew.
    """
    if len(json.dumps(compacted, sort_keys=True, default=str)) < len(
        json.dumps(original, sort_keys=True, default=str)
    ):
        return compacted
    return original


def payload_fingerprint(payload: Any) -> str:
    """Hash what a reader would act on, not every byte of the projection.

    Fingerprint the projection itself, never the emitted envelope: the envelope
    carries the budget, which changes on every call and would make every
    emission look new.

    Real new content still changes the hash -- a new event lengthens the event
    list, and a state change rewrites the run record. Only the per-observation
    bookkeeping above is excluded.
    """
    normalized = _without_volatile_bookkeeping(payload)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def context_budget_ledger_path(paths: OmhPaths) -> Path:
    return paths.runtime_dir / CONTEXT_BUDGET_LEDGER_NAME


def _ledger(paths: OmhPaths) -> dict[str, Any]:
    ledger = read_json_object(context_budget_ledger_path(paths)) or {}
    runs = ledger.get("runs")
    return {"runs": runs if isinstance(runs, dict) else {}}


def _entry(ledger: dict[str, Any], run_id: str) -> dict[str, Any]:
    entry = ledger["runs"].get(run_id)
    if not isinstance(entry, dict):
        return {"emitted_bytes": 0, "call_count": 0, "surfaces": {}, "payload_fingerprints": {}}
    surfaces = entry.get("surfaces")
    fingerprints = entry.get("payload_fingerprints")
    return {
        "emitted_bytes": max(0, int(entry.get("emitted_bytes", 0) or 0)),
        "call_count": max(0, int(entry.get("call_count", 0) or 0)),
        "surfaces": {str(key): int(value) for key, value in surfaces.items()} if isinstance(surfaces, dict) else {},
        "payload_fingerprints": (
            {str(key): str(value) for key, value in fingerprints.items()} if isinstance(fingerprints, dict) else {}
        ),
    }


def run_context_budget(paths: OmhPaths, run_id: str, *, surface: str = "") -> dict[str, Any]:
    """Report how much agent context this run's observe surfaces already spent."""
    entry = _entry(_ledger(paths), run_id)
    emitted = entry["emitted_bytes"]
    exhausted = emitted >= RUN_CONTEXT_BUDGET_BYTES
    return {
        "schema_version": RUN_CONTEXT_BUDGET_SCHEMA_VERSION,
        "run_id": run_id,
        "surface": surface,
        "budget_bytes": RUN_CONTEXT_BUDGET_BYTES,
        "emitted_bytes": emitted,
        "remaining_bytes": max(0, RUN_CONTEXT_BUDGET_BYTES - emitted),
        "observe_call_count": entry["call_count"],
        "surfaces": entry["surfaces"],
        "last_payload_fingerprint": entry["payload_fingerprints"].get(surface, ""),
        "exhausted": exhausted,
        "enforcement": "degrade_to_summary_only_with_artifact_pointers",
        "policy": "timed_polling_rejected; raw_log_dumping_rejected",
    }


def record_context_emission(
    paths: OmhPaths,
    run_id: str,
    *,
    surface: str,
    byte_count: int,
    payload_fingerprint_value: str = "",
) -> dict[str, Any]:
    """Add one observe/show emission to the run's ledger. Best-effort."""
    path = context_budget_ledger_path(paths)

    def update(current: dict[str, Any]) -> dict[str, Any]:
        runs = current.get("runs")
        ledger = {"runs": runs if isinstance(runs, dict) else {}}
        entry = _entry(ledger, run_id)
        entry["emitted_bytes"] += max(0, int(byte_count))
        entry["call_count"] += 1
        entry["surfaces"][surface] = entry["surfaces"].get(surface, 0) + 1
        if payload_fingerprint_value:
            entry["payload_fingerprints"][surface] = payload_fingerprint_value
        entry["updated_at"] = utc_now()
        ledger["runs"][run_id] = entry
        return {
            "schema_version": RUN_CONTEXT_BUDGET_SCHEMA_VERSION,
            "updated_at": entry["updated_at"],
            "runs": ledger["runs"],
        }

    try:
        ensure_dir(paths.runtime_dir)
        locked_json_update(
            path,
            update,
            default={
                "schema_version": RUN_CONTEXT_BUDGET_SCHEMA_VERSION,
                "updated_at": "",
                "runs": {},
            },
        )
    except OSError:
        return run_context_budget(paths, run_id, surface=surface)
    return run_context_budget(paths, run_id, surface=surface)


def public_budget(budget: dict[str, Any]) -> dict[str, Any]:
    """Drop internal bookkeeping before the budget is emitted.

    `last_payload_fingerprint` is a hash of the projection the caller is already
    holding. Emitting it adds bytes, tells the reader nothing, and would make
    `--full` output differ from before this check existed.
    """
    return {key: value for key, value in budget.items() if key != "last_payload_fingerprint"}


def unchanged_run_payload(shown: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Replacement for a projection identical to the one already emitted.

    A supervising agent that polls an unchanged run gets the whole projection
    back every time and narrates it again, which is how one delegated session
    turned twelve fix/test cycles into hundreds of chat messages. Returning the
    lifecycle position plus an explicit "nothing changed" marker removes the
    material to re-narrate without hiding anything: by definition this payload
    carries no information the previous emission did not already carry.

    `--full` bypasses this entirely, so an explicit request still gets
    everything.
    """
    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
    run_id = str(budget.get("run_id", ""))
    budget = public_budget(budget)
    return {
        "schema_version": RUN_UNCHANGED_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "status": str(run.get("status", "")),
            "phase": str(run.get("phase", "")),
            "observation_status": str(run.get("observation_status", "")),
            "updated_at": str(run.get("updated_at", "")),
        },
        "unchanged_since_last_emission": True,
        "delta": {},
        "context_budget": budget,
        "next_action": "wait_for_new_observed_evidence_instead_of_repeating_this_command",
        "full_output_command": f"omh runtime show {run_id} --full",
        "claim_boundary": (
            "Nothing observable changed since the previous emission for this run. This is not "
            "execution, review, CI, merge-readiness, or merge evidence, and repeating the command "
            "will not produce new evidence."
        ),
    }


def unchanged_progress_status_payload(shown: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Replacement for an executor-status projection nothing has moved since.

    "What is running?" is the most repeatable question a supervising agent asks,
    and the answer is the largest single payload in a delegated session -- one
    full binding record per executor. Repeating it unchanged is pure cost, so
    keep only the counts a reader needs to decide whether to look closer.
    """
    active = shown.get("active_executors") if isinstance(shown.get("active_executors"), list) else []
    stale = shown.get("stale_executors") if isinstance(shown.get("stale_executors"), list) else []
    return {
        "schema_version": PROGRESS_STATUS_UNCHANGED_SCHEMA_VERSION,
        "unchanged_since_last_emission": True,
        "active_executor_count": len(active),
        "stale_executor_count": len(stale),
        "delta": {},
        "context_budget": public_budget(budget),
        "next_action": "wait_for_new_observed_evidence_instead_of_repeating_this_command",
        "full_output_command": "omh runtime progress-status --full",
        "claim_boundary": (
            "No executor progress changed since the previous emission. This is not result, "
            "verification, review, CI, merge-readiness, or merge evidence."
        ),
    }


def degrade_run_payload(shown: dict[str, Any], budget: dict[str, Any]) -> dict[str, Any]:
    """Summary-only replacement for a full `show_run` projection."""
    run = shown.get("run") if isinstance(shown.get("run"), dict) else {}
    history = shown.get("history") if isinstance(shown.get("history"), dict) else {}
    journal_events = shown.get("journal_events") if isinstance(shown.get("journal_events"), list) else []
    latest = journal_events[-1] if journal_events and isinstance(journal_events[-1], dict) else {}
    run_id = str(budget.get("run_id", ""))
    budget = public_budget(budget)
    return {
        "schema_version": "omh_run_show_summary_only/v1",
        "run": {
            "run_id": run_id,
            "skill": str(run.get("skill", "")),
            "harness": str(run.get("harness", "")),
            "status": str(run.get("status", "")),
            "phase": str(run.get("phase", "")),
            "observation_status": str(run.get("observation_status", "")),
            "updated_at": str(run.get("updated_at", "")),
        },
        "lifecycle": shown.get("lifecycle", {}),
        "history": history,
        "latest_journal_event": {
            "event": str(latest.get("event", "")),
            "status": str(latest.get("status", "")),
            "observed_at": str(latest.get("observed_at", "")),
        },
        "context_budget": budget,
        "degraded": True,
        "degraded_reason": "run_context_budget_exhausted",
        "next_action": "read_full_history_from_artifacts_instead_of_repeating_this_command",
        "full_history_command": f"omh runtime show {run_id} --full",
        "claim_boundary": (
            "Summary-only projection emitted after this run exhausted its observe-context budget. "
            "It is not execution, review, CI, merge-readiness, or merge evidence; read the listed "
            "artifacts for the full record."
        ),
    }


def degrade_capability_projection_payload(
    projected: dict[str, Any],
    budget: dict[str, Any],
    *,
    dropped: list[dict[str, Any]],
    content_bytes: int,
) -> dict[str, Any]:
    """Summary-only replacement for a task-scoped capability projection.

    Same contract as `degrade_run_payload`: keep the position, drop the
    material, name what was removed, and point at the command that returns it.
    A capability projection has no journal history to shorten, so what it drops
    is capability detail -- which is exactly what a reader must be told, or the
    projection becomes a silent truncation of the authority view.

    The authority block is carried through unchanged. Degrading is a context
    decision; it never narrows or widens what the approved authority granted.
    """
    authority = projected.get("authority") if isinstance(projected.get("authority"), dict) else {}
    task_id = str(projected.get("task_id", ""))
    return {
        "schema_version": CAPABILITY_PROJECTION_SUMMARY_ONLY_SCHEMA_VERSION,
        "task_id": task_id,
        "request_digest": str(projected.get("request_digest", "")),
        "authority": authority,
        "included": [],
        "included_count": 0,
        "offered_count": int(projected.get("offered_count", 0) or 0),
        "exclusions": dropped,
        "exclusion_summary": projected.get("exclusion_summary", {}),
        "exclusion_reason_vocabulary": projected.get("exclusion_reason_vocabulary", []),
        "budget_drop": {
            "dropped_capabilities": [str(entry.get("capability", "")) for entry in dropped],
            "dropped_count": len(dropped),
            "projected_bytes": content_bytes,
            "budget_bytes": int(budget.get("budget_bytes", 0) or 0),
            "remaining_bytes": int(budget.get("remaining_bytes", 0) or 0),
        },
        "context_budget": public_budget(budget),
        "degraded": True,
        "degraded_reason": "capability_projection_context_budget_exhausted",
        "next_action": "widen_the_request_or_start_a_new_task_budget_instead_of_repeating_this_command",
        "full_output_command": f"omh capabilities project --task-id {task_id} --json",
        "claim_boundary": (
            "Summary-only capability projection emitted after this task exhausted its context "
            "budget. Every capability it dropped is named above; nothing was silently withheld. "
            "It is not execution, review, CI, merge-readiness, or merge evidence."
        ),
    }
