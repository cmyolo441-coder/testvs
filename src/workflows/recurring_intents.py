"""Recurring-intent lifecycle records (`hermes_recurring_intent/v1`).

# Why this is a sibling record and not more fields on `prepared_native_blueprint/v1`

There is one recurring-work concept in OMH, not two. A recurring intent is the
*lifecycle envelope* around a prepared native blueprint: it never re-derives
cadence, delivery, or silence, it calls `build_scheduled_ops_blueprint` and
keeps the sub-records that builder produced. The blueprint stays the single
normalizer, and both records live under `.omh/hermes-ops/`.

The blueprint could not carry the lifecycle itself for three reasons:

1. `prepared_native_blueprint/v1` (`hermes_ops_blueprint/v1` before the
   rename) is `projection_only` and write-once.
   `write_hermes_ops_blueprint` refuses to overwrite an existing id and there is
   no update path, because a projection of catalog metadata has nothing to
   update. A recurring intent is revised, activated, and accumulates occurrence
   links, so it needs a mutable record with a revision identity.
2. The blueprint validator refuses any nested `status` outside
   `{prepared, not_observed}` and any `*_status` that claims observation. An
   occurrence has to name an observed runtime run. Putting occurrences inside a
   blueprint would mean weakening exactly the guard that keeps blueprints
   honest, so occurrences live here and the guard itself is reused unchanged
   (`nested_status_boundary_errors`).
3. Blueprint identity is a timestamp plus a random suffix. Linking an
   occurrence to the exact intent it ran under needs a content digest, so a
   changed intent produces a different identifier on its own.

# What "paused" means structurally

`build_recurring_intent` has no parameter that can set an activation state or an
occurrence: a freshly built intent is not merely defaulted to paused, it has no
vocabulary for "an occurrence ran". The validator enforces the same rule for
records that arrive from disk or from a wrapper: with no recorded `activated`
transition, `occurrences` must be empty.

# What activation is

A recorded transition, never a flag flip. OMH runs no scheduler and no
background service. Activation is performed by an approved runtime surface
outside OMH; this record only stores who or what observed it, under which
approval reference, and at which revision. An activated intent is still not
execution evidence — each occurrence points at the runtime run that owns its
evidence.

# Why the failure policy extends v1 instead of becoming a sibling record

The record already carried one fifth of this policy: `overlap_posture`, whose
value was required to be explicit before activation while its handling was left
open. The remaining four decisions (missed run, retry, backfill, failure pause)
had to land in the same digested record, not beside it, for one reason that is
an acceptance criterion rather than a preference: an occurrence links to an
`intent_revision_id`, and the revision digest is what makes that link mean
something. A policy living outside the digest would let an occurrence recorded
under "skip when running, pause after 3 failures" read as evidence for "allow
concurrent, pause after 20" without the revision moving. So the four new
decisions join `REVISION_DIGEST_FIELDS`, and changing any of them produces a new
revision that the older occurrences do not cover.

`overlap_posture` stays at its own top-level path rather than moving under
`failure_policy`. It shipped with a validator branch, a CLI flag, a digest
input, and an activation gate; re-homing it would rewrite all of that to move
bytes. `FAILURE_POLICY_DECISIONS` names the path of each of the five decisions
instead, so the activation gate, the open-decisions list, the digest inputs, and
the human summary all read one table and none of them special-cases overlap.

# Where the boundary sits

The record *declares* the policy. `decide_recurring_occurrence` *projects* it
onto one situation an approved runtime surface reported, and returns what the
policy says plus the occurrence outcome to record. OMH still runs no scheduler:
it never starts, skips, queues, retries, or backfills anything, and a decision
payload is not evidence that the runtime did what the policy said. The one
behavior OMH does own is local and deterministic: a `failed` occurrence that
reaches the declared consecutive-failure threshold pauses this record, because
that is a mutation of local lifecycle metadata, not of anything running.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from ..hashutil import sha256_text
from ..local_store import atomic_write_json, ensure_dir, read_json_object, read_json_object_result, utc_now
from ..paths import OmhPaths
from .hermes_ops import (
    PREPARED_NATIVE_BLUEPRINT_SCHEMA_VERSION,
    PROJECTION_SOURCE_REFS,
    READABLE_BLUEPRINT_SCHEMA_VERSIONS,
    build_scheduled_ops_blueprint,
    nested_status_boundary_errors,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


RECURRING_INTENT_SCHEMA_VERSION = "hermes_recurring_intent/v1"
RECURRING_INTENT_INDEX_SCHEMA_VERSION = "hermes_recurring_intent_index/v1"
RECURRING_INTENT_VALIDATION_SCHEMA_VERSION = "hermes_recurring_intent_validation/v1"
RECURRING_INTENT_PREVIEW_SCHEMA_VERSION = "hermes_recurring_intent_preview/v1"
RECURRING_OCCURRENCE_SCHEMA_VERSION = "hermes_recurring_occurrence/v1"
RECURRING_INTENT_STATUS_CARD_SCHEMA_VERSION = "hermes_recurring_intent_status_card/v1"
RECURRING_OCCURRENCE_DECISION_SCHEMA_VERSION = "hermes_recurring_occurrence_decision/v1"

RECURRING_INTENT_KINDS = ("recurring-intent",)
RECURRING_INTENT_STATUSES = ("prepared", "blocked", "archived")
RECURRING_INTENT_OBSERVATION_STATUSES = ("prepared", "not_observed")
LIFECYCLE_STATES = ("paused", "activated")
OWNER_KINDS = ("unassigned", "human", "team", "runtime_surface", "executor_profile")
ACTIVATION_OBSERVER_KINDS = ("human", "approved_runtime_surface")
OCCURRENCE_OBSERVER_KINDS = ("human", "approved_runtime_surface")
REQUIRED_APPROVAL_KIND = "human_or_approved_runtime_surface"
ACTIVATION_PERFORMED_BY = "approved_runtime_surface_outside_omh"

# The sentinel that separates "nobody chose yet" from "someone chose the safe
# option". Every policy decision starts here and no builder ever replaces it
# with a default: an implicit default that satisfies the activation gate would
# defeat the gate.
UNSET_POSTURE = "unspecified"

OVERLAP_POSTURES = (UNSET_POSTURE, "skip_when_running", "queue_after_running", "allow_concurrent")
MISSED_RUN_POSTURES = (UNSET_POSTURE, "skip_missed_window", "run_once_when_late")
# There is deliberately no unbounded retry posture. Unlimited retry is out of
# scope for this record, so the vocabulary cannot express it.
RETRY_POSTURES = (UNSET_POSTURE, "no_retry", "retry_bounded")
# Same reasoning for backfill: automatic catch-up of every missed window is out
# of scope, so a backfill is either off or a bounded window count.
BACKFILL_POSTURES = (UNSET_POSTURE, "no_backfill", "backfill_bounded_window")
# One real posture: repeated failure always pauses. The decision the user makes
# is the threshold, which is why an explicit posture also demands a count.
FAILURE_PAUSE_POSTURES = (UNSET_POSTURE, "pause_after_consecutive_failures")

RETRY_ATTEMPT_LIMIT = 10
BACKFILL_WINDOW_LIMIT = 10
FAILURE_PAUSE_THRESHOLD_LIMIT = 20

OCCURRENCE_OUTCOMES = ("ran", "failed", "skipped", "missed", "queued")
# Only these two outcomes claim that something ran, so only these two require a
# runtime run reference. A skipped, missed, or queued occurrence exists so that
# a non-run is recorded rather than silent, and it must not carry execution
# evidence it does not have.
EXECUTING_OCCURRENCE_OUTCOMES = ("ran", "failed")

OCCURRENCE_SITUATIONS = ("on_time", "prior_run_active", "missed_window")
OCCURRENCE_DECISIONS = ("start", "start_concurrent", "queue", "skip", "do_not_start")

INTENT_SUMMARY_LIMIT = 240
# The runtime run owns the durable evidence for an occurrence, so this record
# keeps a bounded local index of the most recent links rather than growing one
# JSON file forever -- every mutation reads and rewrites the whole record.
# `occurrences_recorded_total` keeps the count honest after truncation.
RETAINED_OCCURRENCE_LIMIT = 50

RECURRING_INTENT_NOT_EVIDENCE = (
    "host_cron_created",
    "hermes_automation_enabled",
    "omh_scheduler_started",
    "occurrence_executed",
    "runtime_run_started",
    "gateway_delivery_sent",
    "occurrence_skipped_by_runtime",
    "occurrence_retried_by_runtime",
    "occurrence_backfilled_by_runtime",
    "failure_pause_enforced_by_runtime",
    "review",
    "ci",
    "merge_readiness",
    "merge",
)
RECURRING_INTENT_OBSERVED_EVIDENCE_REQUIRED = (
    "An approval reference from a human or an approved runtime surface, recorded before activation.",
    "The approved runtime surface that performed activation, named in the activation transition.",
    "A runtime run reference for every occurrence, owned by the runtime that ran it.",
)
PREPARED_RECURRING_INTENT_IS_NOT = (
    "A Hermes recurring intent is local lifecycle metadata only. OMH runs no scheduler and no background "
    "service, so neither a paused nor an activated intent is host cron creation, scheduler startup, "
    "occurrence execution, gateway delivery, review, CI, merge-readiness, or merge evidence."
)
OCCURRENCE_CLAIM_BOUNDARY = (
    "The linked runtime run owns this execution evidence. This occurrence is only the local link back to the "
    "exact intent revision the run was started under."
)
NON_EXECUTING_OCCURRENCE_CLAIM_BOUNDARY = (
    "Nothing ran for this occurrence. It records that the approved runtime surface applied the declared policy "
    "at a moment it would otherwise have started work, so the non-run is visible instead of silent. It is not "
    "execution evidence and it does not prove the runtime honoured the policy."
)
FAILURE_POLICY_CLAIM_BOUNDARY = (
    "Overlap, missed-run, retry, backfill, and failure-pause handling is policy this record declares and "
    "decide_recurring_occurrence projects onto one situation an approved runtime surface reported. OMH runs no "
    "scheduler: it never starts, skips, queues, retries, or backfills an occurrence, and a decision payload is "
    "not evidence that the runtime did what the policy says. The only thing OMH changes by itself is this local "
    "record, when recorded failures reach the declared threshold and it pauses itself."
)


class FailurePolicyDecision(NamedTuple):
    """One policy decision that must be explicit before activation.

    `path` is a dotted read path into the record, so the five decisions can live
    where they already live (overlap stayed top-level) while the gate, the
    open-decisions list, and the digest all iterate one table.
    """

    decision: str
    path: str
    label: str
    choices: tuple[str, ...]
    question: str


FAILURE_POLICY_DECISIONS = (
    FailurePolicyDecision(
        "overlap",
        "overlap_posture.posture",
        "overlap posture",
        OVERLAP_POSTURES,
        "choose the overlap posture: skip_when_running, queue_after_running, or allow_concurrent",
    ),
    FailurePolicyDecision(
        "missed_run",
        "failure_policy.missed_run.posture",
        "missed-run posture",
        MISSED_RUN_POSTURES,
        "choose what a missed window does: skip_missed_window or run_once_when_late",
    ),
    FailurePolicyDecision(
        "retry",
        "failure_policy.retry.posture",
        "retry posture",
        RETRY_POSTURES,
        "choose the retry posture: no_retry, or retry_bounded with a maximum attempt count",
    ),
    FailurePolicyDecision(
        "backfill",
        "failure_policy.backfill.posture",
        "backfill posture",
        BACKFILL_POSTURES,
        "choose the backfill posture: no_backfill, or backfill_bounded_window with a window count",
    ),
    FailurePolicyDecision(
        "failure_pause",
        "failure_policy.failure_pause.posture",
        "failure-pause threshold",
        FAILURE_PAUSE_POSTURES,
        "choose how many consecutive failures pause this intent",
    ),
)

ACTIVATION_REQUIREMENTS = (
    *(f"explicit_{item.decision}_decision" for item in FAILURE_POLICY_DECISIONS),
    "recorded_approval_reference",
    "recorded_observer_of_the_activating_runtime_surface",
)

# Meaningful fields of the intent, in the order they enter the revision digest.
# Deliberately excluded: intent_id, created_at, updated_at, the revision block
# itself, approval grant state, lifecycle transitions, and occurrences. Times
# never enter the digest -- they are passed in as parameters so the same intent
# always digests to the same revision. Approval grant state is excluded so that
# activating an intent cannot silently orphan the occurrences recorded under it.
REVISION_DIGEST_FIELDS = (
    "title",
    "request_summary",
    "schedule_intent.cadence",
    "schedule_intent.time_hint",
    "schedule_intent.explicit_schedule",
    "delivery_intent.surfaces",
    "delivery_intent.explicit_delivery",
    "silence_policy.mode",
    "silence_policy.explicit_policy",
    "scope.summary",
    "scope.explicit_scope",
    "owner_profile.owner",
    "owner_profile.owner_kind",
    "overlap_posture.posture",
    # The remaining four failure-policy decisions and their bounds. They are in
    # the digest so that changing a policy produces a revision the already
    # recorded occurrences do not cover.
    "failure_policy.missed_run.posture",
    "failure_policy.retry.posture",
    "failure_policy.retry.max_attempts",
    "failure_policy.backfill.posture",
    "failure_policy.backfill.max_windows",
    "failure_policy.failure_pause.posture",
    "failure_policy.failure_pause.consecutive_failure_threshold",
    "required_approval.required",
    "required_approval.approval_kind",
)

_INTENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,159}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def build_recurring_intent(
    request: str,
    *,
    title: str = "",
    schedule: str = "",
    delivery: str = "",
    silence: str = "",
    scope: str = "",
    owner: str = "",
    owner_kind: str = "unassigned",
    overlap_posture: str = UNSET_POSTURE,
    missed_run_posture: str = UNSET_POSTURE,
    retry_posture: str = UNSET_POSTURE,
    retry_max_attempts: int = 0,
    backfill_posture: str = UNSET_POSTURE,
    backfill_max_windows: int = 0,
    failure_pause_posture: str = UNSET_POSTURE,
    failure_pause_threshold: int = 0,
    source: str = "",
    intent_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Prepare a paused recurring intent from a natural-language request.

    There is no activation or occurrence parameter on purpose: the returned
    record cannot express "an occurrence ran" no matter what the caller passes.
    Every failure-policy posture defaults to `unspecified` rather than to a safe
    value, so the activation gate sees "nobody chose" instead of a default that
    nobody read.
    """
    clean_request = " ".join(str(request).split())
    if not clean_request:
        raise ValueError("recurring intent request is required")
    if owner_kind not in OWNER_KINDS:
        raise ValueError(f"unsupported recurring intent owner_kind: {owner_kind}")
    if overlap_posture not in OVERLAP_POSTURES:
        raise ValueError(f"unsupported recurring intent overlap_posture: {overlap_posture}")
    failure_policy = _failure_policy(
        missed_run_posture=missed_run_posture,
        retry_posture=retry_posture,
        retry_max_attempts=retry_max_attempts,
        backfill_posture=backfill_posture,
        backfill_max_windows=backfill_max_windows,
        failure_pause_posture=failure_pause_posture,
        failure_pause_threshold=failure_pause_threshold,
    )
    created = created_at or utc_now()
    hints = {"schedule": schedule.strip(), "delivery": delivery.strip(), "silence": silence.strip()}
    derived = _derived_intents(clean_request, hints, created)
    record: dict[str, Any] = {
        "schema_version": RECURRING_INTENT_SCHEMA_VERSION,
        "intent_id": intent_id or new_recurring_intent_id(clean_request, created),
        "kind": "recurring-intent",
        "title": title.strip() or _default_title(clean_request),
        "status": "prepared",
        "observation_status": "prepared",
        "created_at": created,
        "updated_at": created,
        "source": source.strip(),
        "request_summary": _preview(clean_request, limit=INTENT_SUMMARY_LIMIT),
        "derived_from": {
            "schema_version": PREPARED_NATIVE_BLUEPRINT_SCHEMA_VERSION,
            "authority": "projection_only",
            "normalizer": "build_scheduled_ops_blueprint",
            "source_refs": list(PROJECTION_SOURCE_REFS),
        },
        "derivation_hints": hints,
        "schedule_intent": derived["schedule_intent"],
        "delivery_intent": derived["delivery_intent"],
        "silence_policy": derived["silence_policy"],
        "scope": _scope(clean_request, scope=scope),
        "owner_profile": _owner_profile(owner=owner, owner_kind=owner_kind),
        "overlap_posture": _overlap_posture(overlap_posture),
        "failure_policy": failure_policy,
        "required_approval": {
            "required": True,
            "approval_kind": REQUIRED_APPROVAL_KIND,
            "granted": False,
            "approval_ref": "",
        },
        "lifecycle": {
            "state": "paused",
            "activation_requires": list(ACTIVATION_REQUIREMENTS),
            "activation_performed_by": ACTIVATION_PERFORMED_BY,
            "transitions": [],
        },
        "occurrences": [],
        "occurrences_recorded_total": 0,
        "revision_history": [],
        "not_evidence_until_observed": list(RECURRING_INTENT_NOT_EVIDENCE),
        "observed_evidence_required": list(RECURRING_INTENT_OBSERVED_EVIDENCE_REQUIRED),
        "prepared_is_not": PREPARED_RECURRING_INTENT_IS_NOT,
        "claim_boundary": _claim_boundary(),
    }
    _refresh_derived_blocks(record, revised_at=created)
    errors = validate_recurring_intent(record)
    if errors:
        raise ValueError("; ".join(errors))
    return record


def preview_recurring_intent(record: dict[str, Any]) -> dict[str, Any]:
    """Wrap a prepared intent as a preview a wrapper can show before saving.

    The user described recurring work in chat; this payload is what Hermes reads
    back to them. `save.requires_user_command` is false because confirming the
    summary is the whole user action -- the command below it is the agent-facing
    surface, not something the user types.
    """
    return {
        "schema_version": RECURRING_INTENT_PREVIEW_SCHEMA_VERSION,
        "preview_only": True,
        "saved": False,
        "human_summary": record["human_summary"],
        "open_decisions": list(record["open_decisions"]),
        "save": {
            "requires_user_command": False,
            "user_action": "confirm_or_correct_the_summary",
            "entry_action": "prepare_scheduled_ops_blueprint",
            "agent_surface": "omh ops recurring-intent",
        },
        "intent": record,
        "claim_boundary": dict(record["claim_boundary"]),
    }


def revise_recurring_intent(
    record: dict[str, Any],
    *,
    revised_at: str | None = None,
    title: str | None = None,
    schedule: str | None = None,
    delivery: str | None = None,
    silence: str | None = None,
    scope: str | None = None,
    owner: str | None = None,
    owner_kind: str | None = None,
    overlap_posture: str | None = None,
    missed_run_posture: str | None = None,
    retry_posture: str | None = None,
    retry_max_attempts: int | None = None,
    backfill_posture: str | None = None,
    backfill_max_windows: int | None = None,
    failure_pause_posture: str | None = None,
    failure_pause_threshold: int | None = None,
) -> dict[str, Any]:
    """Return a revised copy with a new revision id, paused again if it was active.

    Revising an activated intent pauses it: the approved runtime surface
    approved a specific revision, and re-activation must record a fresh
    observer against the revision that will actually run.

    This is also the only way out of a failure pause. The safety pause is
    recorded against a revision, so resuming needs a policy revision plus a
    fresh activation rather than a retry of the same failing policy.
    """
    if owner_kind is not None and owner_kind not in OWNER_KINDS:
        raise ValueError(f"unsupported recurring intent owner_kind: {owner_kind}")
    if overlap_posture is not None and overlap_posture not in OVERLAP_POSTURES:
        raise ValueError(f"unsupported recurring intent overlap_posture: {overlap_posture}")
    revised = revised_at or utc_now()
    updated = _clone(record)
    previous_revision = str(updated["revision"]["revision_id"])
    if title is not None:
        clean_title = title.strip()
        if not clean_title:
            raise ValueError("recurring intent title must not be blank")
        updated["title"] = clean_title
    hints = dict(updated.get("derivation_hints", {}))
    for key, value in (("schedule", schedule), ("delivery", delivery), ("silence", silence)):
        if value is not None:
            hints[key] = value.strip()
    if hints != updated.get("derivation_hints"):
        updated["derivation_hints"] = hints
        # The record deliberately does not retain the raw request (metadata-only
        # by default), so a revision re-derives from the bounded request_summary.
        # For a request longer than INTENT_SUMMARY_LIMIT that is the truncated
        # text, which is why the explicit hints exist.
        updated.update(_derived_intents(str(updated["request_summary"]), hints, revised))
    if scope is not None:
        updated["scope"] = _scope(str(updated["request_summary"]), scope=scope)
    if owner is not None or owner_kind is not None:
        profile = updated["owner_profile"]
        updated["owner_profile"] = _owner_profile(
            owner=str(profile["owner"]) if owner is None else owner,
            owner_kind=str(profile["owner_kind"]) if owner_kind is None else owner_kind,
        )
    if overlap_posture is not None:
        updated["overlap_posture"] = _overlap_posture(overlap_posture)
    current_policy = updated["failure_policy"]
    updated["failure_policy"] = _failure_policy(
        missed_run_posture=_kept(missed_run_posture, current_policy["missed_run"]["posture"]),
        retry_posture=_kept(retry_posture, current_policy["retry"]["posture"]),
        retry_max_attempts=_kept(retry_max_attempts, current_policy["retry"]["max_attempts"]),
        backfill_posture=_kept(backfill_posture, current_policy["backfill"]["posture"]),
        backfill_max_windows=_kept(backfill_max_windows, current_policy["backfill"]["max_windows"]),
        failure_pause_posture=_kept(failure_pause_posture, current_policy["failure_pause"]["posture"]),
        failure_pause_threshold=_kept(
            failure_pause_threshold,
            current_policy["failure_pause"]["consecutive_failure_threshold"],
        ),
    )
    updated["updated_at"] = revised
    _refresh_derived_blocks(updated, revised_at=revised)
    if str(updated["revision"]["revision_id"]) == previous_revision:
        raise ValueError("recurring intent revision must change at least one meaningful field")
    updated["revision_history"] = [
        *record.get("revision_history", []),
        {"revision_id": previous_revision, "superseded_at": revised},
    ]
    if str(updated["lifecycle"]["state"]) == "activated":
        updated["lifecycle"]["state"] = "paused"
        updated["lifecycle"]["transitions"] = [
            *updated["lifecycle"]["transitions"],
            {
                "transition": "paused_by_revision",
                "intent_revision_id": str(updated["revision"]["revision_id"]),
                "observer": "",
                "observer_kind": "local_policy",
                "approval_ref": "",
                "activation_surface": "",
                "reason": "intent fields changed; re-activation must record a new observer against this revision",
                "recorded_at": revised,
            },
        ]
        updated["required_approval"] = {
            **updated["required_approval"],
            "granted": False,
            "approval_ref": "",
        }
    _refresh_derived_blocks(updated, revised_at=revised)
    errors = validate_recurring_intent(updated)
    if errors:
        raise ValueError("; ".join(errors))
    return updated


def activate_recurring_intent(
    record: dict[str, Any],
    *,
    observer: str,
    approval_ref: str,
    activation_surface: str,
    observer_kind: str = "approved_runtime_surface",
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Record that an approved runtime surface activated this intent revision.

    OMH does not activate anything here. This appends the transition that says
    who or what observed the activation, so a later occurrence can be read
    against a named observer instead of an anonymous flag.

    Activation is also the failure-policy gate: it refuses while any of the five
    decisions is still `unspecified`, naming each unset one, and it refuses to
    resume a revision that already paused itself after repeated failures.
    """
    if not str(observer).strip():
        raise ValueError("recurring intent activation requires a recorded observer")
    if observer_kind not in ACTIVATION_OBSERVER_KINDS:
        raise ValueError(f"unsupported recurring intent activation observer_kind: {observer_kind}")
    if not str(approval_ref).strip():
        raise ValueError("recurring intent activation requires a recorded approval reference")
    if not str(activation_surface).strip():
        raise ValueError("recurring intent activation requires the approved runtime surface that performed it")
    unset = unset_failure_policy_decisions(record)
    if unset:
        raise ValueError(
            "; ".join(f"recurring intent activation requires an explicit {item.label}" for item in unset)
        )
    if failure_pause_holds_this_revision(record):
        raise ValueError(
            "recurring intent paused itself after repeated failures under this revision: revise the failure "
            "policy and record a fresh approval before re-activating"
        )
    if str(record["lifecycle"]["state"]) == "activated":
        raise ValueError("recurring intent is already activated")
    observed = observed_at or utc_now()
    updated = _clone(record)
    updated["lifecycle"]["state"] = "activated"
    updated["lifecycle"]["transitions"] = [
        *updated["lifecycle"]["transitions"],
        {
            "transition": "activated",
            "intent_revision_id": str(updated["revision"]["revision_id"]),
            "observer": str(observer).strip(),
            "observer_kind": observer_kind,
            "approval_ref": str(approval_ref).strip(),
            "activation_surface": str(activation_surface).strip(),
            "reason": "approved runtime surface reported activation",
            "recorded_at": observed,
        },
    ]
    updated["required_approval"] = {
        **updated["required_approval"],
        "granted": True,
        "approval_ref": str(approval_ref).strip(),
    }
    updated["updated_at"] = observed
    _refresh_derived_blocks(updated, revised_at=str(updated["revision"]["revised_at"]))
    errors = validate_recurring_intent(updated)
    if errors:
        raise ValueError("; ".join(errors))
    return updated


def record_recurring_intent_occurrence(
    record: dict[str, Any],
    *,
    runtime_run_ref: str = "",
    runtime_surface: str,
    observer: str,
    observer_kind: str = "approved_runtime_surface",
    outcome: str = "ran",
    reason: str = "",
    overlapped_run_ref: str = "",
    attempt: int = 1,
    observed_at: str | None = None,
    occurrence_id: str | None = None,
) -> dict[str, Any]:
    """Record one occurrence of this intent against the exact revision it ran under.

    A `ran` or `failed` occurrence links the runtime run that owns its evidence.
    A `skipped`, `missed`, or `queued` occurrence is the opposite: nothing ran,
    so it carries no run reference and instead must state the policy reason it
    did not run. That is what keeps an overlap or a missed window from being
    silent, and it is why a non-executing outcome is refused a run reference
    rather than allowed to borrow one.

    Recording a `failed` occurrence can pause this record. When the consecutive
    failures at the current revision reach the declared threshold, the lifecycle
    moves to `paused`, approval is revoked, and the transition carries the
    reason in plain language.
    """
    if str(record["lifecycle"]["state"]) != "activated":
        raise ValueError(
            "a paused recurring intent cannot record an occurrence: activate it with a recorded observer first"
        )
    if outcome not in OCCURRENCE_OUTCOMES:
        raise ValueError(f"unsupported recurring intent occurrence outcome: {outcome}")
    if observer_kind not in OCCURRENCE_OBSERVER_KINDS:
        raise ValueError(f"unsupported recurring intent occurrence observer_kind: {observer_kind}")
    executed = outcome in EXECUTING_OCCURRENCE_OUTCOMES
    clean_run_ref = str(runtime_run_ref).strip()
    clean_reason = " ".join(str(reason).split())
    if executed and not clean_run_ref:
        raise ValueError("recurring intent occurrence requires a runtime run reference")
    if not executed and clean_run_ref:
        raise ValueError(
            f"a {outcome} recurring intent occurrence did not run, so it must not carry a runtime run reference"
        )
    if not executed and not clean_reason:
        raise ValueError(f"a {outcome} recurring intent occurrence must record why the policy did not run it")
    if not str(runtime_surface).strip():
        raise ValueError("recurring intent occurrence requires the runtime surface that ran it")
    if not str(observer).strip():
        raise ValueError("recurring intent occurrence requires a recorded observer")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("recurring intent occurrence attempt must be a positive integer")
    observed = observed_at or utc_now()
    updated = _clone(record)
    if clean_run_ref and any(
        str(item.get("runtime_run", {}).get("run_ref", "")) == clean_run_ref for item in updated["occurrences"]
    ):
        raise ValueError(f"runtime run {clean_run_ref} is already linked to this recurring intent")
    revision_id = str(updated["revision"]["revision_id"])
    occurrence = {
        "schema_version": RECURRING_OCCURRENCE_SCHEMA_VERSION,
        "occurrence_id": occurrence_id or new_recurring_occurrence_id(revision_id, observed),
        "intent_id": str(updated["intent_id"]),
        "intent_revision_id": revision_id,
        "outcome": outcome,
        "executed": executed,
        "attempt": attempt,
        "reason": clean_reason,
        "overlapped_run_ref": str(overlapped_run_ref).strip(),
        "observer": str(observer).strip(),
        "observer_kind": observer_kind,
        "observed_at": observed,
        "runtime_run": {"surface": str(runtime_surface).strip(), "run_ref": clean_run_ref},
        "evidence_authority": "linked_runtime_run" if executed else "local_policy_decision",
        "claim_boundary": OCCURRENCE_CLAIM_BOUNDARY if executed else NON_EXECUTING_OCCURRENCE_CLAIM_BOUNDARY,
    }
    updated["occurrences"] = [*updated["occurrences"], occurrence][-RETAINED_OCCURRENCE_LIMIT:]
    updated["occurrences_recorded_total"] = int(updated["occurrences_recorded_total"]) + 1
    updated["updated_at"] = observed
    _apply_failure_pause(updated, paused_at=observed)
    _refresh_derived_blocks(updated, revised_at=str(updated["revision"]["revised_at"]))
    errors = validate_recurring_intent(updated)
    if errors:
        raise ValueError("; ".join(errors))
    return updated


def unset_failure_policy_decisions(record: dict[str, Any]) -> list[FailurePolicyDecision]:
    """The policy decisions nobody has chosen yet, in table order.

    `unspecified` and a missing block read the same way, and an explicitly
    chosen value never does -- including when the chosen value happens to be the
    safe one, which is the whole point of the sentinel.
    """
    return [item for item in FAILURE_POLICY_DECISIONS if _policy_value(record, item.path) == UNSET_POSTURE]


def recurring_failure_pause_state(record: dict[str, Any]) -> dict[str, Any]:
    """Consecutive failures at the current revision, against the declared threshold.

    Deterministic and revision-scoped, with no heuristics and no clock:

    - `failed` increments the streak, `ran` resets it to zero.
    - `skipped`, `missed`, and `queued` are transparent: nothing ran, so they
      neither prove recovery nor add a failure.
    - The walk stops at the first occurrence recorded under a different
      revision. Failures under an older policy are not evidence about this one.
    """
    policy = _dict_at(record, "failure_policy", "failure_pause")
    posture = str(policy.get("posture", UNSET_POSTURE))
    threshold = _non_negative_int(policy.get("consecutive_failure_threshold", 0))
    revision_id = str(_dict_at(record, "revision").get("revision_id", ""))
    streak = 0
    counted = 0
    for occurrence in reversed(_list_at(record, "occurrences")):
        if not isinstance(occurrence, dict):
            continue
        if str(occurrence.get("intent_revision_id", "")) != revision_id:
            break
        outcome = str(occurrence.get("outcome", ""))
        if outcome == "failed":
            streak += 1
            counted += 1
            continue
        if outcome == "ran":
            break
        counted += 1
    paused = posture == "pause_after_consecutive_failures" and threshold > 0 and streak >= threshold
    return {
        "posture": posture,
        "threshold": threshold,
        "consecutive_failures": streak,
        "occurrences_inspected": counted,
        "counted_at_revision": revision_id,
        "paused": paused,
        "explanation": _failure_pause_explanation(posture, threshold, streak, revision_id, paused),
    }


def failure_pause_holds_this_revision(record: dict[str, Any]) -> bool:
    """True while a safety pause recorded against the current revision still stands.

    Reads the recorded transition rather than recomputing the streak, so a
    revision that pauses and then loses occurrences to the retention window
    still cannot be re-activated without a policy revision.
    """
    revision_id = str(_dict_at(record, "revision").get("revision_id", ""))
    return any(
        isinstance(item, dict)
        and item.get("transition") == "paused_by_failure_policy"
        and str(item.get("intent_revision_id", "")) == revision_id
        for item in _list_at(_dict_at(record, "lifecycle"), "transitions")
    )


def decide_recurring_occurrence(
    record: dict[str, Any],
    *,
    situation: str,
    now: str,
    active_run_ref: str = "",
    missed_window_count: int = 0,
) -> dict[str, Any]:
    """Project the declared policy onto one situation an approved runtime reported.

    OMH decides nothing about the world here: it reads the policy this revision
    declares and returns what that policy says, which occurrence outcome to
    record, and how many retries or backfill windows the policy allows. The
    runtime surface is what actually starts, skips, queues, or waits.

    `now` is a parameter rather than a clock read so the payload is reproducible
    and safe to compare.
    """
    if situation not in OCCURRENCE_SITUATIONS:
        raise ValueError(f"unsupported recurring occurrence situation: {situation}")
    if not str(now).strip():
        raise ValueError("recurring occurrence decision requires the caller's now timestamp")
    if not isinstance(missed_window_count, int) or isinstance(missed_window_count, bool) or missed_window_count < 0:
        raise ValueError("missed_window_count must be a non-negative integer")
    clean_active_run = str(active_run_ref).strip()
    if situation == "prior_run_active" and not clean_active_run:
        # Without the run reference an overlap decision could not name what it
        # would duplicate, which is exactly the silent case this record exists
        # to prevent.
        raise ValueError("a prior_run_active decision requires the run reference of the occurrence still running")
    pause = recurring_failure_pause_state(record)
    unset = unset_failure_policy_decisions(record)
    outcome = _decision_outcome(record, situation, unset, pause, clean_active_run, missed_window_count)
    return {
        "schema_version": RECURRING_OCCURRENCE_DECISION_SCHEMA_VERSION,
        "intent_id": str(record.get("intent_id", "")),
        "intent_revision_id": str(_dict_at(record, "revision").get("revision_id", "")),
        "lifecycle_state": str(_dict_at(record, "lifecycle").get("state", "")),
        "situation": situation,
        "decision": outcome["decision"],
        "reason": outcome["reason"],
        "record_occurrence_as": outcome["record_occurrence_as"],
        "overlapped_run_ref": clean_active_run if outcome["decision"] != "do_not_start" else "",
        "missed_window_count": missed_window_count,
        "retry_attempts_allowed": _retry_attempts_allowed(record),
        "backfill_windows_allowed": _backfill_windows_allowed(record, situation, outcome["decision"], missed_window_count),
        "applied_policy": applied_recurring_failure_policy(record),
        "unset_policy_decisions": [item.decision for item in unset],
        "failure_pause": pause,
        "decided_at": str(now),
        "authority": "local_policy_projection",
        "enforced_by": ACTIVATION_PERFORMED_BY,
        "claim_boundary": FAILURE_POLICY_CLAIM_BOUNDARY,
    }


def applied_recurring_failure_policy(record: dict[str, Any]) -> dict[str, Any]:
    """The five declared decisions plus their bounds, flat enough to report."""
    retry = _dict_at(record, "failure_policy", "retry")
    backfill = _dict_at(record, "failure_policy", "backfill")
    pause = _dict_at(record, "failure_policy", "failure_pause")
    return {
        "overlap": _policy_value(record, "overlap_posture.posture"),
        "missed_run": _policy_value(record, "failure_policy.missed_run.posture"),
        "retry": str(retry.get("posture", UNSET_POSTURE)),
        "retry_max_attempts": _non_negative_int(retry.get("max_attempts", 0)),
        "backfill": str(backfill.get("posture", UNSET_POSTURE)),
        "backfill_max_windows": _non_negative_int(backfill.get("max_windows", 0)),
        "failure_pause": str(pause.get("posture", UNSET_POSTURE)),
        "consecutive_failure_threshold": _non_negative_int(pause.get("consecutive_failure_threshold", 0)),
    }


def render_recurring_occurrence_decision_text(decision: dict[str, Any]) -> str:
    policy = decision["applied_policy"]
    lines = [
        f"Recurring occurrence decision: {decision['decision']}",
        f"  intent: {decision['intent_id']}",
        f"  revision: {decision['intent_revision_id']}",
        f"  situation: {decision['situation']}",
        f"  reason: {decision['reason']}",
        f"  record this occurrence as: {decision['record_occurrence_as'] or 'nothing to record'}",
        f"  retries allowed: {decision['retry_attempts_allowed']}",
        f"  backfill windows allowed: {decision['backfill_windows_allowed']}",
        "  applied policy: "
        f"overlap={policy['overlap']} missed_run={policy['missed_run']} "
        f"retry={policy['retry']}({policy['retry_max_attempts']}) "
        f"backfill={policy['backfill']}({policy['backfill_max_windows']}) "
        f"failure_pause={policy['failure_pause']}({policy['consecutive_failure_threshold']})",
        f"  failure pause: {decision['failure_pause']['explanation']}",
        "",
        FAILURE_POLICY_CLAIM_BOUNDARY,
    ]
    return "\n".join(lines)


def recurring_intent_revision_id(record: dict[str, Any]) -> str:
    """Deterministic revision id for the intent's meaningful fields.

    No wall clock enters the digest, so the same intent always produces the same
    revision and any meaningful change produces a different one.
    """
    material = {field: _digest_value(record, field) for field in REVISION_DIGEST_FIELDS}
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"rev-{sha256_text(canonical)[:16]}"


def new_recurring_intent_id(request: str, now: datetime | str | None = None) -> str:
    stamp = _stamp(now)
    slug = _slugify(request)[:48] or "recurring-intent"
    return f"{stamp}-recurring-intent-{slug}-{secrets.token_hex(3)}"


def new_recurring_occurrence_id(revision_id: str, now: datetime | str | None = None) -> str:
    return f"{_stamp(now)}-occurrence-{_slugify(revision_id)[:32]}-{secrets.token_hex(3)}"


def validate_recurring_intent(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if record.get("schema_version") != RECURRING_INTENT_SCHEMA_VERSION:
        errors.append("schema_version must be hermes_recurring_intent/v1")
    if record.get("kind") not in RECURRING_INTENT_KINDS:
        errors.append(f"unsupported recurring intent kind: {record.get('kind', '')}")
    intent_id = str(record.get("intent_id", ""))
    if not intent_id.strip():
        errors.append("intent_id is required")
    elif not _valid_intent_id(intent_id):
        errors.append("intent_id must contain only letters, digits, and hyphens, and must not contain path separators")
    if not str(record.get("title", "")).strip():
        errors.append("title is required")
    if record.get("status") not in RECURRING_INTENT_STATUSES:
        errors.append(f"unsupported recurring intent status: {record.get('status', '')}")
    if record.get("observation_status") not in RECURRING_INTENT_OBSERVATION_STATUSES:
        errors.append(f"unsupported recurring intent observation_status: {record.get('observation_status', '')}")
    derived_from = record.get("derived_from")
    if not isinstance(derived_from, dict):
        errors.append("derived_from must be an object")
    else:
        # A saved intent written before the blueprint rename names the legacy
        # id; it is the user's record and stays valid.
        if derived_from.get("schema_version") not in READABLE_BLUEPRINT_SCHEMA_VERSIONS:
            errors.append(f"derived_from.schema_version must be one of {', '.join(READABLE_BLUEPRINT_SCHEMA_VERSIONS)}")
        if derived_from.get("authority") != "projection_only":
            errors.append("derived_from.authority must be projection_only")
    for key in (
        "schedule_intent",
        "delivery_intent",
        "silence_policy",
        "scope",
        "owner_profile",
        "overlap_posture",
        "failure_policy",
        "required_approval",
        "revision",
        "lifecycle",
        "claim_boundary",
        "status_card",
    ):
        if not isinstance(record.get(key), dict):
            errors.append(f"{key} must be an object")
    for key in (
        "occurrences",
        "revision_history",
        "open_decisions",
        "not_evidence_until_observed",
        "observed_evidence_required",
    ):
        if not isinstance(record.get(key), list):
            errors.append(f"{key} must be a list")
    if errors:
        return errors
    if record["owner_profile"].get("owner_kind") not in OWNER_KINDS:
        errors.append(f"unsupported owner_profile.owner_kind: {record['owner_profile'].get('owner_kind', '')}")
    for decision in FAILURE_POLICY_DECISIONS:
        posture = _digest_value(record, decision.path)
        if posture not in decision.choices:
            errors.append(f"unsupported {decision.path}: {'' if posture is None else posture}")
    errors.extend(_failure_policy_bound_errors(record))
    if record["required_approval"].get("required") is not True:
        errors.append("required_approval.required must stay true: activation always needs an approval reference")
    if record["required_approval"].get("approval_kind") != REQUIRED_APPROVAL_KIND:
        errors.append(f"required_approval.approval_kind must be {REQUIRED_APPROVAL_KIND}")
    expected_revision = recurring_intent_revision_id(record)
    if str(record["revision"].get("revision_id", "")) != expected_revision:
        errors.append(
            f"revision.revision_id must be the digest of the intent's meaningful fields ({expected_revision})"
        )
    if not set(RECURRING_INTENT_NOT_EVIDENCE).issubset(set(_string_list(record["not_evidence_until_observed"]))):
        errors.append("not_evidence_until_observed must include the default prepared-vs-observed guard list")
    errors.extend(_lifecycle_errors(record))
    errors.extend(_occurrence_errors(record))
    errors.extend(nested_status_boundary_errors(record))
    return errors


def write_recurring_intent(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    errors = validate_recurring_intent(record)
    if errors:
        raise ValueError("; ".join(errors))
    intent_id = str(record["intent_id"])
    if recurring_intent_exists(paths, intent_id):
        raise ValueError(f"recurring intent already exists: {intent_id}")
    atomic_write_json(_intent_path(paths, intent_id), record, private=True)
    _write_index_cache(paths)
    return record


def update_recurring_intent(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    """Replace a stored intent in place; unlike a blueprint, an intent has a lifecycle."""
    errors = validate_recurring_intent(record)
    if errors:
        raise ValueError("; ".join(errors))
    intent_id = str(record["intent_id"])
    if not recurring_intent_exists(paths, intent_id):
        raise FileNotFoundError(intent_id)
    atomic_write_json(_intent_path(paths, intent_id), record, private=True)
    _write_index_cache(paths)
    return record


def list_recurring_intents(paths: OmhPaths, *, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for intent_path in sorted(paths.recurring_intents_dir.glob("*.json")):
        record, error = read_json_object_result(intent_path)
        if error:
            continue
        if record:
            records.append(record)
    records.sort(key=lambda item: str(item.get("created_at", "")))
    if limit is not None:
        if limit < 1:
            return []
        records = records[-limit:]
    return records


def show_recurring_intent(paths: OmhPaths, intent_id: str) -> dict[str, Any]:
    if not _valid_intent_id(intent_id):
        raise FileNotFoundError(intent_id)
    record = read_json_object(_intent_path(paths, intent_id))
    if record:
        return record
    raise FileNotFoundError(intent_id)


def recurring_intent_exists(paths: OmhPaths, intent_id: str) -> bool:
    return _valid_intent_id(intent_id) and _intent_path(paths, intent_id).exists()


def summarize_recurring_intent(record: dict[str, Any]) -> dict[str, Any]:
    revision_id = str(record.get("revision", {}).get("revision_id", ""))
    occurrences = record.get("occurrences", [])
    occurrences = occurrences if isinstance(occurrences, list) else []
    unset = unset_failure_policy_decisions(record)
    pause = recurring_failure_pause_state(record)
    return {
        "intent_id": str(record.get("intent_id", "")),
        "kind": str(record.get("kind", "")),
        "title": str(record.get("title", "")),
        "status": str(record.get("status", "")),
        "observation_status": str(record.get("observation_status", "")),
        "lifecycle_state": str(record.get("lifecycle", {}).get("state", "")),
        "revision_id": revision_id,
        "updated_at": str(record.get("updated_at", "")),
        "cadence": str(record.get("schedule_intent", {}).get("cadence", "unspecified")),
        "overlap_posture": _policy_value(record, "overlap_posture.posture"),
        "failure_policy_complete": not unset,
        "unset_failure_policy_decisions": [item.decision for item in unset],
        "failure_paused": failure_pause_holds_this_revision(record),
        "consecutive_failures": int(pause["consecutive_failures"]),
        "owner": str(record.get("owner_profile", {}).get("owner", "")),
        "occurrence_count": int(record.get("occurrences_recorded_total", 0) or 0),
        "retained_occurrence_count": len(occurrences),
        "current_revision_occurrence_count": sum(
            1 for item in occurrences if isinstance(item, dict) and str(item.get("intent_revision_id", "")) == revision_id
        ),
        "summary": _preview(str(record.get("request_summary", "")), limit=INTENT_SUMMARY_LIMIT),
    }


def validate_recurring_intent_store(paths: OmhPaths) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    seen: set[str] = set()
    for intent_path in sorted(paths.recurring_intents_dir.glob("*.json")):
        record, error = read_json_object_result(intent_path)
        if error:
            errors.append(f"{intent_path}: {error}")
            continue
        if record:
            records.append(record)
    for record in records:
        intent_id = str(record.get("intent_id", ""))
        if intent_id in seen:
            errors.append(f"duplicate intent_id: {intent_id}")
        seen.add(intent_id)
        for error in validate_recurring_intent(record):
            errors.append(f"{intent_id or '<unknown>'}: {error}")
    index = read_json_object(paths.recurring_intents_index_path)
    if index and index.get("schema_version") != RECURRING_INTENT_INDEX_SCHEMA_VERSION:
        errors.append("recurring intent index cache has unsupported schema_version")
    return {
        "schema_version": RECURRING_INTENT_VALIDATION_SCHEMA_VERSION,
        "ok": not errors,
        "intent_count": len(records),
        "errors": errors,
        "index_authority": "cache_only",
    }


def render_recurring_intent_text(record: dict[str, Any]) -> str:
    schedule = record["schedule_intent"]
    time_hint = str(schedule["time_hint"])
    cadence = str(schedule["cadence"]) + (f" ({time_hint})" if time_hint else "")
    approval = record["required_approval"]
    approval_line = f"granted, ref {approval['approval_ref']}" if approval["granted"] else "required, not granted"
    owner = str(record["owner_profile"]["owner"]) or "unassigned"
    linked = summarize_recurring_intent(record)["current_revision_occurrence_count"]
    policy = applied_recurring_failure_policy(record)
    pause = recurring_failure_pause_state(record)
    lines = [
        str(record["title"]),
        f"  id: {record['intent_id']}",
        f"  revision: {record['revision']['revision_id']}",
        f"  lifecycle: {record['lifecycle']['state']}",
        f"  cadence: {cadence}",
        f"  delivery: {', '.join(_string_list(record['delivery_intent']['surfaces'])) or 'unspecified'}",
        f"  silence: {record['silence_policy']['mode']}",
        f"  scope: {record['scope']['summary']}",
        f"  owner: {owner} ({record['owner_profile']['owner_kind']})",
        f"  overlap posture: {record['overlap_posture']['posture']}",
        f"  missed-run posture: {policy['missed_run']}",
        f"  retry posture: {policy['retry']} (max attempts {policy['retry_max_attempts']})",
        f"  backfill posture: {policy['backfill']} (max windows {policy['backfill_max_windows']})",
        f"  failure pause: {policy['failure_pause']} (threshold {policy['consecutive_failure_threshold']})",
        f"  consecutive failures at this revision: {pause['consecutive_failures']}",
        f"  approval: {approval_line}",
        f"  occurrences linked to this revision: {linked}",
        "",
        record["human_summary"],
    ]
    if record["open_decisions"]:
        lines.append("")
        lines.append("Still needs a decision:")
        lines.extend(f"  - {item}" for item in record["open_decisions"])
    lines.append("")
    lines.append(f"Not observed: {', '.join(record['not_evidence_until_observed'])}")
    lines.append(record["prepared_is_not"])
    return "\n".join(lines)


def render_recurring_intent_list_text(payload: dict[str, Any]) -> str:
    intents = payload.get("intents", [])
    if not intents:
        return "No recurring intents recorded."
    lines = [f"Recurring intents: {payload['count']} of {payload['total_count']}"]
    for item in intents:
        lines.append(
            f"  [{item['lifecycle_state']}] {item['title']} ({item['intent_id']}) "
            f"cadence={item['cadence']} overlap={item['overlap_posture']} "
            f"failure_policy={'complete' if item['failure_policy_complete'] else 'incomplete'} "
            f"occurrences={item['current_revision_occurrence_count']}/{item['occurrence_count']}"
        )
        if item["failure_paused"]:
            lines.append(f"    paused by its failure policy after {item['consecutive_failures']} consecutive failures")
    lines.append("A recurring intent is prepared lifecycle metadata; OMH runs no scheduler.")
    return "\n".join(lines)


def _derived_intents(request: str, hints: dict[str, str], created_at: str) -> dict[str, Any]:
    """Take cadence, delivery, and silence from the blueprint builder, never re-derive them."""
    blueprint = build_scheduled_ops_blueprint(
        request,
        schedule=hints.get("schedule", ""),
        delivery=hints.get("delivery", ""),
        silence=hints.get("silence", ""),
        created_at=created_at,
    )
    return {
        "schedule_intent": dict(blueprint["schedule_intent"]),
        "delivery_intent": dict(blueprint["delivery_intent"]),
        "silence_policy": dict(blueprint["silence_policy"]),
    }


def _scope(request: str, *, scope: str) -> dict[str, object]:
    explicit = scope.strip()
    return {
        "status": "prepared",
        "summary": explicit or _preview(request, limit=160),
        "explicit_scope": explicit,
        "requires_confirmation": not explicit,
    }


def _owner_profile(*, owner: str, owner_kind: str) -> dict[str, object]:
    clean_owner = owner.strip()
    return {
        "status": "prepared",
        "owner": clean_owner,
        "owner_kind": owner_kind,
        "accountable_for": [
            "approving activation",
            "answering for every occurrence recorded under this intent",
        ],
        "requires_confirmation": not clean_owner or owner_kind == "unassigned",
    }


def _overlap_posture(posture: str) -> dict[str, object]:
    return {
        "status": "prepared",
        "posture": posture,
        "explicit": posture != UNSET_POSTURE,
        "requires_confirmation": posture == UNSET_POSTURE,
        "failure_policy_decision": "overlap",
    }


def _failure_policy(
    *,
    missed_run_posture: str,
    retry_posture: str,
    retry_max_attempts: int,
    backfill_posture: str,
    backfill_max_windows: int,
    failure_pause_posture: str,
    failure_pause_threshold: int,
) -> dict[str, object]:
    """The four failure-policy decisions that live outside `overlap_posture`.

    A bound is normalised to zero for a posture that cannot carry one, and
    demanded for a posture that must: choosing `retry_bounded` without a count
    is not a choice.
    """
    if missed_run_posture not in MISSED_RUN_POSTURES:
        raise ValueError(f"unsupported recurring intent missed_run_posture: {missed_run_posture}")
    if retry_posture not in RETRY_POSTURES:
        raise ValueError(f"unsupported recurring intent retry_posture: {retry_posture}")
    if backfill_posture not in BACKFILL_POSTURES:
        raise ValueError(f"unsupported recurring intent backfill_posture: {backfill_posture}")
    if failure_pause_posture not in FAILURE_PAUSE_POSTURES:
        raise ValueError(f"unsupported recurring intent failure_pause_posture: {failure_pause_posture}")
    attempts = _bounded_policy_count(
        retry_posture,
        retry_max_attempts,
        bounded_posture="retry_bounded",
        limit=RETRY_ATTEMPT_LIMIT,
        label="retry_bounded requires a maximum attempt count",
    )
    windows = _bounded_policy_count(
        backfill_posture,
        backfill_max_windows,
        bounded_posture="backfill_bounded_window",
        limit=BACKFILL_WINDOW_LIMIT,
        label="backfill_bounded_window requires a window count",
    )
    threshold = _bounded_policy_count(
        failure_pause_posture,
        failure_pause_threshold,
        bounded_posture="pause_after_consecutive_failures",
        limit=FAILURE_PAUSE_THRESHOLD_LIMIT,
        label="pause_after_consecutive_failures requires a consecutive-failure threshold",
    )
    return {
        "status": "prepared",
        "missed_run": _policy_decision_block(missed_run_posture),
        "retry": {**_policy_decision_block(retry_posture), "max_attempts": attempts},
        "backfill": {**_policy_decision_block(backfill_posture), "max_windows": windows},
        "failure_pause": {
            **_policy_decision_block(failure_pause_posture),
            "consecutive_failure_threshold": threshold,
        },
        "unbounded_retry_supported": False,
        "unbounded_backfill_supported": False,
        "enforced_by": ACTIVATION_PERFORMED_BY,
        "decision_surface": "decide_recurring_occurrence",
        "claim_boundary": FAILURE_POLICY_CLAIM_BOUNDARY,
    }


def _policy_decision_block(posture: str) -> dict[str, object]:
    return {
        "status": "prepared",
        "posture": posture,
        "explicit": posture != UNSET_POSTURE,
        "requires_confirmation": posture == UNSET_POSTURE,
    }


def _bounded_policy_count(posture: str, value: object, *, bounded_posture: str, limit: int, label: str) -> int:
    if posture != bounded_posture:
        return 0
    count = _non_negative_int(value)
    if count < 1 or count > limit:
        raise ValueError(f"{label} between 1 and {limit}")
    return count


def _kept(supplied: object, current: object) -> Any:
    """Revision semantics: `None` means leave this policy field where it is."""
    return current if supplied is None else supplied


def _policy_value(record: dict[str, Any], path: str) -> str:
    value = _digest_value(record, path)
    return UNSET_POSTURE if value in (None, "") else str(value)


def _dict_at(record: Any, *keys: str) -> dict[str, Any]:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def _list_at(record: Any, key: str) -> list[Any]:
    value = record.get(key) if isinstance(record, dict) else None
    return value if isinstance(value, list) else []


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _failure_pause_explanation(posture: str, threshold: int, streak: int, revision_id: str, paused: bool) -> str:
    if posture == UNSET_POSTURE:
        return (
            "No failure-pause threshold has been chosen yet, so this intent cannot be activated and nothing "
            "will pause it automatically."
        )
    if paused:
        return (
            f"This recurring intent paused itself: {streak} consecutive failed occurrences were recorded under "
            f"revision {revision_id}, reaching the declared threshold of {threshold}. Nothing will run again "
            "until the failure policy is revised and an approved runtime surface records a fresh activation."
        )
    return (
        f"{streak} of {threshold} consecutive failures are recorded under revision {revision_id}; this intent "
        f"pauses itself at {threshold}."
    )


def _apply_failure_pause(record: dict[str, Any], *, paused_at: str) -> None:
    """Move an activated record to `paused` once recorded failures reach the threshold."""
    if str(record["lifecycle"]["state"]) != "activated":
        return
    pause = recurring_failure_pause_state(record)
    if not pause["paused"]:
        return
    record["lifecycle"]["state"] = "paused"
    record["lifecycle"]["transitions"] = [
        *record["lifecycle"]["transitions"],
        {
            "transition": "paused_by_failure_policy",
            "intent_revision_id": str(record["revision"]["revision_id"]),
            "observer": "",
            "observer_kind": "local_policy",
            "approval_ref": "",
            "activation_surface": "",
            "reason": pause["explanation"],
            "recorded_at": paused_at,
        },
    ]
    record["required_approval"] = {**record["required_approval"], "granted": False, "approval_ref": ""}


def _decision_outcome(
    record: dict[str, Any],
    situation: str,
    unset: list[FailurePolicyDecision],
    pause: dict[str, Any],
    active_run_ref: str,
    missed_window_count: int,
) -> dict[str, str]:
    """Which way the declared policy points, in refusal-first order."""
    if unset:
        return {
            "decision": "do_not_start",
            "reason": (
                "This recurring intent cannot run: its failure policy is incomplete. Still unset: "
                + ", ".join(item.label for item in unset)
                + "."
            ),
            "record_occurrence_as": "",
        }
    if pause["paused"]:
        return {"decision": "do_not_start", "reason": pause["explanation"], "record_occurrence_as": ""}
    if str(_dict_at(record, "lifecycle").get("state", "")) != "activated":
        return {
            "decision": "do_not_start",
            "reason": (
                "This recurring intent is paused, so no approved runtime surface has been recorded against the "
                "current revision."
            ),
            "record_occurrence_as": "",
        }
    if situation == "prior_run_active":
        return _overlap_decision(_policy_value(record, "overlap_posture.posture"), active_run_ref)
    if situation == "missed_window":
        return _missed_run_decision(
            _policy_value(record, "failure_policy.missed_run.posture"),
            missed_window_count,
        )
    return {
        "decision": "start",
        "reason": "No prior run is active and no window was missed, so the declared policy allows this occurrence.",
        "record_occurrence_as": "ran",
    }


def _overlap_decision(posture: str, active_run_ref: str) -> dict[str, str]:
    if posture == "skip_when_running":
        return {
            "decision": "skip",
            "reason": (
                f"Run {active_run_ref} is still active and the declared overlap posture is skip_when_running, so "
                "this window is skipped rather than duplicated."
            ),
            "record_occurrence_as": "skipped",
        }
    if posture == "queue_after_running":
        return {
            "decision": "queue",
            "reason": (
                f"Run {active_run_ref} is still active and the declared overlap posture is queue_after_running, so "
                "this window waits for that run instead of starting beside it."
            ),
            "record_occurrence_as": "queued",
        }
    return {
        "decision": "start_concurrent",
        "reason": (
            f"The declared overlap posture is allow_concurrent, so this window may start while run "
            f"{active_run_ref} is still active. Record the occurrence naming that run so the concurrency is "
            "visible rather than silent."
        ),
        "record_occurrence_as": "ran",
    }


def _missed_run_decision(posture: str, missed_window_count: int) -> dict[str, str]:
    if posture == "skip_missed_window":
        return {
            "decision": "skip",
            "reason": (
                f"{missed_window_count} window(s) were missed and the declared missed-run posture is "
                "skip_missed_window, so no catch-up work is started."
            ),
            "record_occurrence_as": "missed",
        }
    return {
        "decision": "start",
        "reason": (
            f"{missed_window_count} window(s) were missed and the declared missed-run posture is "
            "run_once_when_late, so the most recent missed window runs once."
        ),
        "record_occurrence_as": "ran",
    }


def _retry_attempts_allowed(record: dict[str, Any]) -> int:
    retry = _dict_at(record, "failure_policy", "retry")
    if str(retry.get("posture", UNSET_POSTURE)) != "retry_bounded":
        return 0
    return _non_negative_int(retry.get("max_attempts", 0))


def _backfill_windows_allowed(
    record: dict[str, Any],
    situation: str,
    decision: str,
    missed_window_count: int,
) -> int:
    """Older missed windows the policy allows on top of the one being decided.

    Backfill only applies when a missed window is actually being run, and never
    exceeds the declared bound -- the record has no vocabulary for catching up
    on everything.
    """
    if situation != "missed_window" or decision != "start":
        return 0
    backfill = _dict_at(record, "failure_policy", "backfill")
    if str(backfill.get("posture", UNSET_POSTURE)) != "backfill_bounded_window":
        return 0
    return min(max(missed_window_count - 1, 0), _non_negative_int(backfill.get("max_windows", 0)))


def _claim_boundary() -> dict[str, object]:
    return {
        "prepared_is_not_observed": True,
        "activated_is_not_executed": True,
        "omh_runs_no_scheduler": True,
        "occurrence_evidence_authority": "linked_runtime_run",
        "failure_policy_decided_here_enforced_elsewhere": True,
        "failure_policy_enforced_by": ACTIVATION_PERFORMED_BY,
        "failure_policy_statement": FAILURE_POLICY_CLAIM_BOUNDARY,
        "statement": PREPARED_RECURRING_INTENT_IS_NOT,
    }


def _refresh_derived_blocks(record: dict[str, Any], *, revised_at: str) -> None:
    """Recompute the blocks that are a function of the rest of the record."""
    record["revision"] = {
        "revision_id": recurring_intent_revision_id(record),
        "digest_algorithm": "sha256",
        "digest_inputs": list(REVISION_DIGEST_FIELDS),
        "revised_at": revised_at,
    }
    record["open_decisions"] = _open_decisions(record)
    record["human_summary"] = _human_summary(record)
    record["claim_boundary"] = _claim_boundary()
    record["status_card"] = _status_card(record)


def _open_decisions(record: dict[str, Any]) -> list[str]:
    decisions: list[str] = []
    if record["schedule_intent"].get("requires_confirmation"):
        decisions.append("confirm the exact schedule and timezone")
    if record["silence_policy"].get("requires_confirmation"):
        decisions.append("confirm whether to report every occurrence or only changes")
    if record["delivery_intent"].get("requires_observed_gateway_evidence"):
        decisions.append("confirm the delivery target and where its observed gateway evidence comes from")
    if record["scope"].get("requires_confirmation"):
        decisions.append("confirm the scope this recurring work is allowed to touch")
    if record["owner_profile"].get("requires_confirmation"):
        decisions.append("name the owner accountable for this recurring work")
    decisions.extend(item.question for item in unset_failure_policy_decisions(record))
    return decisions


def _human_summary(record: dict[str, Any]) -> str:
    schedule = record["schedule_intent"]
    cadence = str(schedule["cadence"])
    when = "On an unconfirmed cadence" if cadence == "unspecified" else cadence.capitalize()
    if schedule["time_hint"]:
        when = f"{when} {schedule['time_hint']}"
    surfaces = ", ".join(_string_list(record["delivery_intent"]["surfaces"])) or "the current Hermes thread"
    reporting = (
        "only when something changed"
        if record["silence_policy"]["mode"] == "only_report_changes"
        else "after every occurrence"
    )
    owner = str(record["owner_profile"]["owner"]) or "nobody yet"
    if str(record["lifecycle"]["state"]) == "activated":
        closing = (
            "It is activated by an approved runtime surface; OMH still runs no scheduler, and each occurrence "
            "counts only when that runtime records a run reference here."
        )
    elif failure_pause_holds_this_revision(record):
        closing = recurring_failure_pause_state(record)["explanation"]
    else:
        closing = (
            "It is paused: OMH runs no scheduler, so nothing happens until an approved runtime surface "
            "activates it with a recorded observer and approval reference."
        )
    return (
        f"{when}: {record['request_summary']} Report to {surfaces}, {reporting}. "
        f"Owner: {owner}. {_failure_policy_sentence(record)} {closing}"
    )


def _failure_policy_sentence(record: dict[str, Any]) -> str:
    """What the user is told will happen when things go wrong, before activation."""
    unset = unset_failure_policy_decisions(record)
    if unset:
        return (
            "Its failure policy is incomplete, so it cannot be activated yet: "
            + ", ".join(item.label for item in unset)
            + " still need a decision."
        )
    policy = applied_recurring_failure_policy(record)
    overlap = {
        "skip_when_running": "skip this window",
        "queue_after_running": "wait for that run to finish",
        "allow_concurrent": "start anyway alongside it",
    }[policy["overlap"]]
    missed = (
        "the missed window is skipped"
        if policy["missed_run"] == "skip_missed_window"
        else "the most recent missed window runs once when it is late"
    )
    retry = (
        "a failed occurrence is not retried"
        if policy["retry"] == "no_retry"
        else f"a failed occurrence is retried up to {policy['retry_max_attempts']} time(s)"
    )
    backfill = (
        "older missed windows are never backfilled"
        if policy["backfill"] == "no_backfill"
        else f"at most {policy['backfill_max_windows']} older missed window(s) are backfilled"
    )
    return (
        f"If a prior run is still active it will {overlap}; if a window is missed, {missed}; {retry}; "
        f"{backfill}; and it pauses itself after {policy['consecutive_failure_threshold']} consecutive failures."
    )


def _status_card(record: dict[str, Any]) -> dict[str, object]:
    activated = str(record["lifecycle"]["state"]) == "activated"
    pause = recurring_failure_pause_state(record)
    failure_paused = failure_pause_holds_this_revision(record)
    if failure_paused:
        headline = "Recurring intent paused by its own failure policy"
        next_steps = ["revise the failure policy, then record a fresh activation with an approval reference"]
    elif activated:
        headline = "Recurring intent activated by an approved runtime surface"
        next_steps = ["record each occurrence with the runtime run reference that owns its evidence"]
    else:
        headline = "Recurring intent prepared and paused"
        next_steps = ["hand activation to an approved runtime surface and record its observer"]
    return {
        "schema_version": RECURRING_INTENT_STATUS_CARD_SCHEMA_VERSION,
        "headline": headline,
        "lifecycle_state": str(record["lifecycle"]["state"]),
        "revision_id": str(record["revision"]["revision_id"]),
        "prepared": [
            "cadence and time hint",
            "delivery target",
            "silence policy",
            "scope",
            "owner profile",
            "overlap posture",
            "missed-run, retry, backfill, and failure-pause policy",
            "required approval",
        ],
        "applied_policy": applied_recurring_failure_policy(record),
        "failure_pause": dict(pause),
        "not_observed": list(RECURRING_INTENT_NOT_EVIDENCE),
        "next": list(record["open_decisions"]) or next_steps,
    }


def _failure_policy_bound_errors(record: dict[str, Any]) -> list[str]:
    """Every bounded count must match the posture that is allowed to carry it."""
    errors: list[str] = []
    for keys, field, bounded_posture, limit in (
        (("failure_policy", "retry"), "max_attempts", "retry_bounded", RETRY_ATTEMPT_LIMIT),
        (("failure_policy", "backfill"), "max_windows", "backfill_bounded_window", BACKFILL_WINDOW_LIMIT),
        (
            ("failure_policy", "failure_pause"),
            "consecutive_failure_threshold",
            "pause_after_consecutive_failures",
            FAILURE_PAUSE_THRESHOLD_LIMIT,
        ),
    ):
        block = _dict_at(record, *keys)
        path = ".".join((*keys, field))
        raw = block.get(field, 0)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            errors.append(f"{path} must be a non-negative integer")
            continue
        if str(block.get("posture", UNSET_POSTURE)) == bounded_posture:
            if raw < 1 or raw > limit:
                errors.append(f"{path} must be between 1 and {limit} for posture {bounded_posture}")
        elif raw != 0:
            errors.append(f"{path} must be 0 unless the posture is {bounded_posture}")
    return errors


def _lifecycle_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    lifecycle = record["lifecycle"]
    state = str(lifecycle.get("state", ""))
    if state not in LIFECYCLE_STATES:
        errors.append(f"unsupported lifecycle.state: {state}")
    transitions = lifecycle.get("transitions")
    if not isinstance(transitions, list):
        return [*errors, "lifecycle.transitions must be a list"]
    activations = [item for item in transitions if isinstance(item, dict) and item.get("transition") == "activated"]
    for index, transition in enumerate(activations):
        path = f"$.lifecycle.transitions[{index}]"
        if not str(transition.get("observer", "")).strip():
            errors.append(f"{path} records an activation without a recorded observer")
        if transition.get("observer_kind") not in ACTIVATION_OBSERVER_KINDS:
            errors.append(f"{path} activation observer_kind must be human or approved_runtime_surface")
        if not str(transition.get("approval_ref", "")).strip():
            errors.append(f"{path} records an activation without a recorded approval reference")
        if not str(transition.get("activation_surface", "")).strip():
            errors.append(f"{path} records an activation without the approved runtime surface that performed it")
    if state == "activated":
        if not activations:
            errors.append("an activated recurring intent must carry a recorded activation transition")
        for decision in unset_failure_policy_decisions(record):
            errors.append(f"an activated recurring intent must carry an explicit {decision.label}")
        if failure_pause_holds_this_revision(record):
            errors.append(
                "an activated recurring intent must not carry a failure pause at its current revision: "
                "revise the failure policy and record a fresh activation instead"
            )
        if record["required_approval"].get("granted") is not True:
            errors.append("an activated recurring intent must record its approval as granted")
    return errors


def _occurrence_errors(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    occurrences = record["occurrences"]
    total = record.get("occurrences_recorded_total")
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        errors.append("occurrences_recorded_total must be a non-negative integer")
    elif total < len(occurrences):
        errors.append("occurrences_recorded_total must not be lower than the retained occurrence count")
    if len(occurrences) > RETAINED_OCCURRENCE_LIMIT:
        errors.append(f"occurrences must retain at most the {RETAINED_OCCURRENCE_LIMIT} most recent links")
    transitions = record["lifecycle"].get("transitions", [])
    ever_activated = any(
        isinstance(item, dict) and item.get("transition") == "activated"
        for item in (transitions if isinstance(transitions, list) else [])
    )
    if occurrences and not ever_activated:
        errors.append(
            "a recurring intent with no recorded activation transition must not carry occurrences: "
            "it cannot claim that any occurrence ran"
        )
    known_revisions = {str(record["revision"].get("revision_id", ""))}
    for item in record["revision_history"]:
        if isinstance(item, dict):
            known_revisions.add(str(item.get("revision_id", "")))
    for index, occurrence in enumerate(occurrences):
        path = f"$.occurrences[{index}]"
        if not isinstance(occurrence, dict):
            errors.append(f"{path} must be an object")
            continue
        if occurrence.get("schema_version") != RECURRING_OCCURRENCE_SCHEMA_VERSION:
            errors.append(f"{path}.schema_version must be {RECURRING_OCCURRENCE_SCHEMA_VERSION}")
        outcome = occurrence.get("outcome")
        if outcome not in OCCURRENCE_OUTCOMES:
            errors.append(f"{path}.outcome must be one of {', '.join(OCCURRENCE_OUTCOMES)}")
        if not str(occurrence.get("observer", "")).strip():
            errors.append(f"{path} must name the observer that reported the run")
        runtime_run = occurrence.get("runtime_run")
        run_ref = str(runtime_run.get("run_ref", "")).strip() if isinstance(runtime_run, dict) else ""
        if outcome in EXECUTING_OCCURRENCE_OUTCOMES:
            if not run_ref:
                errors.append(f"{path} must link a runtime run reference: preparation is never execution")
        elif outcome in OCCURRENCE_OUTCOMES:
            # A non-executing occurrence records a non-run. Borrowing a run
            # reference would let it read as execution evidence, and an empty
            # reason would make the non-run silent.
            if run_ref:
                errors.append(f"{path} is a {outcome} occurrence and must not carry a runtime run reference")
            if not str(occurrence.get("reason", "")).strip():
                errors.append(f"{path} is a {outcome} occurrence and must record why the policy did not run it")
        revision_id = str(occurrence.get("intent_revision_id", ""))
        if revision_id not in known_revisions:
            errors.append(f"{path} names unknown intent revision {revision_id or '<missing>'}")
    return errors


def _digest_value(record: dict[str, Any], field: str) -> object:
    value: object = record
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    return str(value)


def _intent_path(paths: OmhPaths, intent_id: str) -> Path:
    if not _valid_intent_id(intent_id):
        raise ValueError("intent_id must contain only letters, digits, and hyphens, and must not contain path separators")
    path = paths.recurring_intents_dir / f"{intent_id}.json"
    root = paths.recurring_intents_dir.resolve()
    resolved = path.resolve()
    if resolved.parent != root:
        raise ValueError("intent_id escapes recurring intent storage")
    return path


def _write_index_cache(paths: OmhPaths) -> None:
    records = list_recurring_intents(paths)
    ensure_dir(paths.recurring_intents_root, private=True)
    atomic_write_json(
        paths.recurring_intents_index_path,
        {
            "schema_version": RECURRING_INTENT_INDEX_SCHEMA_VERSION,
            "updated_at": utc_now(),
            "authority": "cache_only",
            "intents": [summarize_recurring_intent(record) for record in records],
        },
        private=True,
    )


def _clone(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record))


def _default_title(text: str) -> str:
    return f"Recurring intent: {_preview(text, limit=72)}"


def _stamp(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return _slugify(value)[:32] or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return _stamp(parsed)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slugify(value: str) -> str:
    return (_SLUG_RE.sub("-", value.casefold()).strip("-") or "recurring-intent")[:64].strip("-")


def _valid_intent_id(value: str) -> bool:
    return bool(_INTENT_ID_RE.fullmatch(str(value)))


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [str(value).strip()] if str(value).strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _preview(value: str, *, limit: int) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."
