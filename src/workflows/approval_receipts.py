"""Append-only receipts for consent an operator actually gave (issue #807).

An approval receipt records that a human answered a confirmation ladder and
said yes to one escalation: one action, on one named scope, for one owner, in
one run, under one safety-profile revision. Nothing else.

Why this is its own record family and not a field group on the delegation
--------------------------------------------------------------------------

This repo has twice refused a new record family (#811, #818) for a good reason:
a family joins on run id, and a join is where an artifact and its metadata
desynchronize. A field group on the record that already exists cannot go stale
against itself.

Consent is the case that reasoning does not cover. It is created at a different
time from the delegation (when the operator answers, not when the handoff is
built), by a different actor (the operator, not the router), it has its own
lifetime (it goes stale on a clock the delegation knows nothing about, and it
can be revoked while the delegation is untouched), and it must survive a rebuild
of the delegation it approves -- rebuilding a prepared handoff must not silently
re-grant consent, and it must not silently destroy consent either. A field group
is written by whoever writes the record; it cannot express "written by someone
else, before this record existed, and still true after this record is replaced".
That is what makes a separate append-only family right here and wrong there.

What a receipt binds
--------------------

An approval is scoped to exactly five things, and each of them is a separate
refusal with its own code:

    run_id                    -> run_not_approved
    owner                     -> owner_not_approved
    approved_action           -> action_not_approved
    (scope_class, scope_ref)  -> scope_not_approved
    safety_profile_revision   -> safety_revision_not_approved

Matching is equality on every dimension, including the scope pair. There is no
containment, prefix, or subsumption rule anywhere in this module, which is what
makes widening structurally impossible rather than merely unimplemented: an
approval for `src/a.py` satisfies a request for `src/a.py` and nothing else --
not `src/`, not `src/b.py`, not the same path for another owner.

Time bounding
-------------

Freshness is derived at read time from the stored decision stamp, the idiom
`coding/executor_auth_signals.py` uses for limit signals, rather than stored as
a deadline the reader must trust (the `_governance_overrides` shape). Two
reasons. A stored `expires_at` is a second independently writable field: a
hand-edited store widens the window by editing one number, while a window that
lives in `APPROVAL_TTL_SECONDS` here cannot be widened by anything on disk. And
the honest reason -- the `storage_retention` boundary of
`handoff_safety_contract/v1` is still `declared_not_enforced`. #835 made
artifact lifecycle readable, but its cleanup preview is a dry run and this store
is not even one of the families it projects, so nothing deletes an approval
receipt. Claiming an approval "expires" on disk would be a lie about a file that
outlives every window it names, so expiry is a property every reader recomputes
and never a property the artifact has.

A decision stamped in the future is not fresh either. It cannot be shown to be
older than zero seconds, so it is refused rather than clamped, which is the
other half of "the window cannot be widened from disk".

Three separate things
---------------------

`external_effect_receipts` separates requested / attempted / succeeded / failed
/ unknown so a request can never be read as an outcome. The same discipline is
why this family exists separately from that one: approval, attempt, and observed
outcome are three different claims (#800). An approval receipt proves consent
was given. It never proves the host applied the permission, and it never proves
anything ran -- `CLAIM_BOUNDARY` says so on every record, and
`validate_approval_receipt` refuses a record shaped to assert execution, by key
name as well as by the closed key set.

Store mechanics
---------------

Everything else follows `workflows/external_effect_receipts.py`, which is this
repo's proven append-only receipt store: a JSONL appended under
`local_store.file_lock` (a real OS lock on POSIX and on Windows), sorted-keys
lines, an append that terminates an unterminated tail before writing so a short
write cannot swallow the next record, closed vocabularies everywhere including
at render, minting that is idempotent by decision fingerprint, supersession and
revocation as new linked records rather than edits, and a mint entry point that
returns its failures instead of raising into the producer.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    append_sidecar_line,
    append_store_line,
    closed_vocabulary_value,
    latest_record_in,
    mint_record_id,
    opaque_ref,
    record_fingerprint,
    redacted_ref,
    reference_errors,
    store_errors,
)
from ..system.local_store import file_lock, read_jsonl_objects, utc_now
from ..system.metadata_safety import is_sensitive_metadata_text
from ..system.paths import OmhPaths
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


APPROVAL_RECEIPT_SCHEMA_VERSION: Final[str] = "approval_receipt/v1"
APPROVAL_DECISION_SCHEMA_VERSION: Final[str] = "approval_decision/v1"
APPROVAL_RECEIPT_STORE_VALIDATION_SCHEMA_VERSION: Final[str] = "approval_receipt_store_validation/v1"
APPROVAL_MINT_RESULT_SCHEMA_VERSION: Final[str] = "approval_mint_result/v1"
APPROVAL_MINT_FAILURE_SCHEMA_VERSION: Final[str] = "approval_mint_failure/v1"

APPROVAL_RECEIPT_STORE_NAME: Final[str] = "approval_receipts.jsonl"
APPROVAL_MINT_FAILURE_STORE_NAME: Final[str] = "approval_mint_failures.jsonl"

RECEIPT_PRIVACY: Final[str] = "metadata_only"

# Stated on every record and echoed on every decision. The three claims #800
# keeps apart are approval, attempt, and observed outcome; this family carries
# the first one and is forbidden from carrying either of the others.
CLAIM_BOUNDARY: Final[str] = (
    "An approval receipt is one recorded answer to one confirmation, scoped to one action on one named "
    "scope for one owner in one run under one safety profile revision. It proves consent was given. It is "
    "not dispatch, execution, or evidence that the host applied the permission, and it approves nothing "
    "else."
)

# What an operator can be asked to approve. Every entry is an action that
# escalates past prepare-only, which is the only kind of answer worth storing:
# approving `research` would put a state in the store that nothing ever reads.
# Mirrors `coding.action_gate.MUTATING_ACTIONS` plus `executor_dispatch` and is
# held here rather than imported because the gate will call into this module --
# `tests/test_approval_receipts.py` pins the relationship instead.
APPROVABLE_ACTIONS: Final[tuple[str, ...]] = (
    "executor_dispatch",
    "external_posting",
    "merge",
    "pr_creation",
    "pr_revision",
    "repo_edit",
)

# What kind of thing `scope_ref` names. AC3 of #807 requires the exact added
# path, endpoint, or tool to be named, so the class says how to read the name
# and the name itself is stored verbatim rather than summarized.
SCOPE_CLASSES: Final[tuple[str, ...]] = (
    "executor_profile",
    "filesystem_path",
    "network_endpoint",
    "permission_profile",
    "tool",
)

# The ladders a confirmation can be answered through. Mirrors
# `coding.action_gate.CONFIRMATION_LADDERS`; a receipt names the ladder it came
# from, so there is no shape in which an approval arrived from somewhere that is
# not the confirmation flow.
CONFIRMATION_LADDERS: Final[tuple[str, ...]] = (
    "executor_selection",
    "permission_profile",
    # #800's risky-action confirmation. Registered here because it is registered
    # there: a ladder the gate can arm and this module cannot name would be a
    # confirmation whose answer is unmintable, so the two tuples move together
    # and the pin in `tests/test_approval_receipts.py` is what forces it.
    "risky_action",
    "operator_confirmation",
)

# `granted` -- the operator said yes.
# `denied`  -- the operator said no. Recorded because a refusal is an answer,
#              and an unanswered confirmation must not look like a refused one.
# `revoked` -- consent already given is withdrawn. Appended as its own record
#              linked to the grant; the grant itself is never rewritten.
APPROVAL_DECISIONS: Final[tuple[str, ...]] = ("granted", "denied", "revoked")

# One hour. Chosen against the two bounded-approval precedents in the tree: it
# is well inside the six-hour horizon `executor_auth_signals` gives an *advisory*
# limit signal (consent is stronger than advice, so it must not outlive it), and
# two orders of magnitude below the seven-day cap `_governance_overrides` allows
# a stale-read override (that override tolerates a stale read; this one
# authorises a mutating action on a live run). A run that lets an hour pass
# between the answer and the action has changed enough that the question is
# worth asking again.
APPROVAL_TTL_SECONDS: Final[int] = 60 * 60

# `age_seconds` when the decision stamp cannot be read, or is stamped in the
# future. Both mean the same thing: the receipt cannot be shown to be fresh.
UNKNOWN_AGE_SECONDS: Final[int] = -1

APPROVAL_RECEIPT_KEYS: Final[tuple[str, ...]] = (
    "approval_id",
    "approved_action",
    "claim_boundary",
    "confirmation_ladder",
    "decided_at",
    "decision",
    "owner",
    "privacy",
    "receipt_id",
    "run_id",
    "safety_profile_revision",
    "schema_version",
    "scope_class",
    "scope_ref",
    "supersedes_receipt_ref",
)

# The three keys whose value is a module constant rather than data. Everything
# else on a receipt is rendered, and `compact_approval_receipt` must cover all
# of it -- a key in the tuple and missing from the compactor is a field that is
# stored and never seen, which has cost this repo twice.
APPROVAL_RECEIPT_CONSTANT_KEYS: Final[tuple[str, ...]] = ("claim_boundary", "privacy", "schema_version")

# Identifiers that must be present, through `require_opaque_metadata_ref`.
_REQUIRED_APPROVAL_REFS: Final[tuple[str, ...]] = (
    "approval_id",
    "decided_at",
    "owner",
    "receipt_id",
    "run_id",
    "safety_profile_revision",
)
# The link is empty on the first answer to a confirmation and only then.
_OPTIONAL_APPROVAL_REFS: Final[tuple[str, ...]] = ("supersedes_receipt_ref",)
_APPROVAL_VOCABULARIES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("approved_action", APPROVABLE_ACTIONS),
    ("confirmation_ladder", CONFIRMATION_LADDERS),
    ("decision", APPROVAL_DECISIONS),
    ("scope_class", SCOPE_CLASSES),
)
# Every string field, which on this record is every field: there is no list, no
# nested object, and deliberately no free text. A summary line would be one more
# place for a prompt to arrive, and consent needs no prose to be citable.
_APPROVAL_STRING_FIELDS: Final[tuple[str, ...]] = APPROVAL_RECEIPT_KEYS

# What identifies *which answer* a receipt records. The stamp, the receipt id,
# and the supersede link are excluded: reporting the same answer twice must not
# mint a second receipt just because the clock moved.
_DECISION_IDENTITY_KEYS: Final[tuple[str, ...]] = (
    "approval_id",
    "approved_action",
    "confirmation_ladder",
    "decision",
    "owner",
    "run_id",
    "safety_profile_revision",
    "scope_class",
    "scope_ref",
)

# Refusal codes. The first five are the five dimensions an approval binds; no
# two of them can collapse into one answer, because "this approval does not
# cover you" is useless to a caller that has to decide what to ask for next.
REFUSAL_RUN: Final[str] = "run_not_approved"
REFUSAL_OWNER: Final[str] = "owner_not_approved"
REFUSAL_ACTION: Final[str] = "action_not_approved"
REFUSAL_SCOPE: Final[str] = "scope_not_approved"
REFUSAL_REVISION: Final[str] = "safety_revision_not_approved"
# Lifecycle refusals: an approval that names exactly this request but cannot
# speak for it any more.
REFUSAL_ABSENT: Final[str] = "approval_absent"
REFUSAL_EXPIRED: Final[str] = "approval_expired"
REFUSAL_REVOKED: Final[str] = "approval_revoked"
REFUSAL_SUPERSEDED: Final[str] = "approval_superseded"
REFUSAL_DENIED: Final[str] = "approval_denied"

# The five scope refusals, in the order `approval_satisfies_request_in` reports
# them when a candidate differs on more than one dimension. Run first because a
# receipt borrowed from another run is the failure #807 is named after; owner
# next because it is the other identity dimension; then what was approved, then
# what it was approved over, then the profile it was cleared under.
SCOPE_REFUSAL_CODES: Final[tuple[str, ...]] = (
    REFUSAL_RUN,
    REFUSAL_OWNER,
    REFUSAL_ACTION,
    REFUSAL_SCOPE,
    REFUSAL_REVISION,
)
LIFECYCLE_REFUSAL_CODES: Final[tuple[str, ...]] = (
    REFUSAL_ABSENT,
    REFUSAL_EXPIRED,
    REFUSAL_REVOKED,
    REFUSAL_SUPERSEDED,
    REFUSAL_DENIED,
)
REFUSAL_CODES: Final[tuple[str, ...]] = SCOPE_REFUSAL_CODES + LIFECYCLE_REFUSAL_CODES

# Constant strings, keyed by code. Nothing is interpolated into a reason: a
# refusal line is rendered to operators, and a line built from request values
# would be one more path for caller-controlled text to reach a terminal.
REFUSAL_REASONS: Final[dict[str, str]] = {
    REFUSAL_RUN: "The approval on record belongs to another run; consent is run-bound and never carries over.",
    REFUSAL_OWNER: "The approval on record was given to another owner.",
    REFUSAL_ACTION: "The approval on record names another action; consent never extends to a sibling action.",
    REFUSAL_SCOPE: (
        "The approval on record names another scope; consent covers exactly the scope it named and never a "
        "broader one."
    ),
    REFUSAL_REVISION: "The approval on record was given under another safety profile revision.",
    REFUSAL_ABSENT: "No approval on record names this action, scope, owner, run, and safety profile revision.",
    REFUSAL_EXPIRED: "The approval on record is outside the approval window, measured now from its decision stamp.",
    REFUSAL_REVOKED: "The approval on record was revoked.",
    REFUSAL_SUPERSEDED: "The approval on record was superseded by a later answer to the same confirmation.",
    REFUSAL_DENIED: "The confirmation on record was answered with a denial.",
}

# Key names an execution claim arrives under. The closed key set already refuses
# every one of them, but it refuses them as "unsupported keys", and the whole
# point of this family is that an approval is not an outcome -- so the shape is
# named and refused by name first. Enforcing E of #800: a record asserting that
# something ran is not constructible here.
_EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "applied",
        "attempted",
        "completed",
        "dispatched",
        "effect_applied",
        "enforced",
        "executed",
        "execution",
        "execution_result",
        "exit_code",
        "external_ref",
        "host_applied",
        "observed",
        "observed_at",
        "observed_result",
        "outcome",
        "performed",
        "ran",
        "result",
        "status",
        "succeeded",
        "success",
    }
)

# The store label every error line and every reference fault is reported under.
_LABEL: Final[str] = "approval_receipt"

# A scope must name something inside the workspace. `require_opaque_metadata_ref`
# already refuses a leading `/` or `~` (the first character must be alphanumeric)
# and every backslash, so what is left is the parent traversal and the Windows
# drive prefix, which are the two ways an in-bounds-looking scope reaches out.
_SCOPE_TRAVERSAL: Final[re.Pattern[str]] = re.compile(r"(?:^|/)\.\.(?:/|$)")
_WINDOWS_DRIVE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:")


class ApprovalReceiptError(ValueError):
    """Raised when a receipt cannot be minted from what an operator answered."""


def approval_id(
    *,
    run_id: str,
    owner: str,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
) -> str:
    """Stable identity of one confirmation question.

    The safety-profile revision is deliberately not in it. The revision is part
    of the *answer's* validity, not part of the question: re-answering the same
    question after the profile moved must supersede the earlier answer rather
    than open a second, parallel chain that both claim to be current.
    """
    base = json.dumps(
        {
            "approved_action": str(approved_action),
            "owner": str(owner),
            "run_id": str(run_id),
            "scope_class": str(scope_class),
            "scope_ref": str(scope_ref),
        },
        sort_keys=True,
    )
    return f"approval-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


def build_approval_receipt(
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    confirmation_ladder: str,
    decision: str = "granted",
    decided_at: str = "",
    supersedes_receipt_ref: str = "",
) -> dict[str, Any]:
    """Mint one receipt, or refuse.

    Every refusal here is the invariant this module exists for: a receipt can
    only describe an answer a human gave to a confirmation about a named scope.
    """
    if decision not in APPROVAL_DECISIONS:
        raise ApprovalReceiptError(f"approval_receipt decision is unsupported: {decision!r}")
    if approved_action not in APPROVABLE_ACTIONS:
        raise ApprovalReceiptError(f"approval_receipt approved_action is unsupported: {approved_action!r}")
    if scope_class not in SCOPE_CLASSES:
        raise ApprovalReceiptError(f"approval_receipt scope_class is unsupported: {scope_class!r}")
    if confirmation_ladder not in CONFIRMATION_LADDERS:
        raise ApprovalReceiptError(
            f"approval_receipt confirmation_ladder is unsupported: {confirmation_ladder!r}"
        )
    safe_scope_ref = _scope_ref(scope_ref)
    safe_owner = _opaque(owner, field="approval_receipt owner")
    safe_run_id = _opaque(run_id, field="approval_receipt run_id")
    safe_revision = _opaque(safety_profile_revision, field="approval_receipt safety_profile_revision")
    safe_supersedes = (
        _opaque(supersedes_receipt_ref, field="approval_receipt supersedes_receipt_ref")
        if supersedes_receipt_ref
        else ""
    )
    stamp = _opaque(str(decided_at or utc_now()), field="approval_receipt decided_at")
    identity = approval_id(
        run_id=safe_run_id,
        owner=safe_owner,
        approved_action=approved_action,
        scope_class=scope_class,
        scope_ref=safe_scope_ref,
    )
    record = {
        "schema_version": APPROVAL_RECEIPT_SCHEMA_VERSION,
        "receipt_id": _receipt_id(stamp, identity, decision),
        "approval_id": identity,
        "run_id": safe_run_id,
        "owner": safe_owner,
        "approved_action": approved_action,
        "scope_class": scope_class,
        "scope_ref": safe_scope_ref,
        "safety_profile_revision": safe_revision,
        "confirmation_ladder": confirmation_ladder,
        "decision": decision,
        "decided_at": stamp,
        "supersedes_receipt_ref": safe_supersedes,
        "privacy": RECEIPT_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    errors = validate_approval_receipt(record)
    if errors:
        raise ApprovalReceiptError(errors[0])
    return record


def validate_approval_receipt(record: dict[str, Any]) -> list[str]:
    """Every reason one record is not an approval receipt."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["approval_receipt must be an object"]
    claims_execution = sorted(key for key in record if str(key).lower() in _EXECUTION_CLAIM_KEYS)
    if claims_execution:
        errors.append(
            "approval_receipt must not carry execution-claim keys: "
            f"{claims_execution}; consent is not an attempt and not an outcome"
        )
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"approval_receipt must not carry raw or hidden keys: {forbidden}")
    extra_keys = sorted(set(record) - set(APPROVAL_RECEIPT_KEYS) - set(claims_execution) - set(forbidden))
    if extra_keys:
        errors.append(f"approval_receipt has unsupported keys: {extra_keys}")
    missing = sorted(set(APPROVAL_RECEIPT_KEYS) - set(record))
    if missing:
        errors.append(f"approval_receipt is missing keys: {missing}")
    if record.get("schema_version") != APPROVAL_RECEIPT_SCHEMA_VERSION:
        errors.append(f"approval_receipt schema_version must be {APPROVAL_RECEIPT_SCHEMA_VERSION}")
    for key in _APPROVAL_STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"approval_receipt {key} must be a string")
    for key, vocabulary in _APPROVAL_VOCABULARIES:
        if record.get(key) not in vocabulary:
            errors.append(f"approval_receipt {key} is unsupported: {record.get(key)!r}")
    if record.get("privacy") != RECEIPT_PRIVACY:
        errors.append("approval_receipt privacy must be metadata_only")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("approval_receipt claim_boundary must state the approval boundary")
    for field in _REQUIRED_APPROVAL_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    for field in _OPTIONAL_APPROVAL_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=False))
    errors.extend(_scope_ref_errors(record.get("scope_ref")))
    if str(record.get("receipt_id", "")) and str(record.get("receipt_id", "")) == str(
        record.get("supersedes_receipt_ref", "")
    ):
        errors.append("approval_receipt supersedes_receipt_ref must not name itself")
    expected_identity = approval_id(
        run_id=str(record.get("run_id", "")),
        owner=str(record.get("owner", "")),
        approved_action=str(record.get("approved_action", "")),
        scope_class=str(record.get("scope_class", "")),
        scope_ref=str(record.get("scope_ref", "")),
    )
    if str(record.get("approval_id", "")) != expected_identity:
        # Without this a hand-edited store could move one grant onto another
        # question's chain and have it supersede an approval it never answered.
        errors.append("approval_receipt approval_id does not identify its own action, scope, owner, and run")
    return errors


def append_approval_receipt(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    """Append one validated receipt. Never rewrites, never reorders."""
    errors = validate_approval_receipt(record)
    if errors:
        raise ApprovalReceiptError(errors[0])
    path = paths.runtime_approval_receipts_path
    # `file_lock` rather than a bare `fcntl.flock`: it holds a real OS lock on
    # both POSIX and Windows, and this store's mutual exclusion has to be a
    # property CI can assert on either platform.
    with file_lock(path, private=True):
        append_store_line(path, record)
    return record


def record_approval_receipt(
    paths: OmhPaths,
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    confirmation_ladder: str,
    decision: str = "granted",
    now: str = "",
) -> dict[str, Any]:
    """Mint and append one receipt, or return the receipt that already records it.

    The strict entry point: it raises when the answer cannot become a receipt
    and when the store cannot be written. The confirmation flow calls
    `mint_approval_receipt` instead, because a receipt store outage must not
    fail the flow that already collected the answer.

    Re-answering a confirmation appends a new receipt linked through
    `supersedes_receipt_ref`; nothing on disk is changed. Reporting the same
    answer again while it is still live appends nothing and returns the receipt
    already on record, so one answer reported three times has one receipt. Once
    a grant is outside its window the identical answer mints again -- consent
    re-given after expiry is new consent, not a duplicate report of the old.

    The read of the prior receipt and the append happen under one lock, so two
    concurrent answers to one confirmation cannot both link to the same
    predecessor and fork the chain.
    """
    record, _ = _record_approval(
        paths.runtime_approval_receipts_path,
        approved_action=approved_action,
        scope_class=scope_class,
        scope_ref=scope_ref,
        owner=owner,
        run_id=run_id,
        safety_profile_revision=safety_profile_revision,
        confirmation_ladder=confirmation_ladder,
        decision=decision,
        now=now,
    )
    return record


def mint_approval_receipt(
    paths: OmhPaths,
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    confirmation_ladder: str,
    decision: str = "granted",
    now: str = "",
) -> dict[str, Any]:
    """Record one answered confirmation without ever raising into the flow.

    The confirmation flow records a receipt *after* the operator has already
    answered. Letting the receipt store fail that flow would make an unwritable
    store look like an answer that was never given, so every refusal and every
    write failure is returned instead of raised.

    Returns an `approval_mint_result/v1` mapping:

        schema_version -- APPROVAL_MINT_RESULT_SCHEMA_VERSION
        minted         -- True only when a new receipt reached the store
        outcome        -- "recorded" | "already_recorded" | "refused" | "not_written"
        approval_id    -- the confirmation this call was about
        receipt_id     -- the receipt now on record, "" when there is none
        decision       -- the decision now on record, "" when there is none
        error          -- the refusal or write failure, "" otherwise

    Failures are also appended to `approval_mint_failures.jsonl` beside the
    store, best effort, so an answer that went unrecorded is visible without the
    flow having to invent a channel for it.
    """
    return mint_approval_receipt_at(
        paths.runtime_approval_receipts_path,
        approved_action=approved_action,
        scope_class=scope_class,
        scope_ref=scope_ref,
        owner=owner,
        run_id=run_id,
        safety_profile_revision=safety_profile_revision,
        confirmation_ladder=confirmation_ladder,
        decision=decision,
        now=now,
    )


def mint_approval_receipt_at(
    store_path: Path,
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    confirmation_ladder: str,
    decision: str = "granted",
    now: str = "",
) -> dict[str, Any]:
    """`mint_approval_receipt` against an already-resolved store path.

    Callers that hold a runtime directory rather than an `OmhPaths` use this, so
    every path they touch is derived from the directory they were given and none
    is reconstructed from an environment they are not running in.
    """
    identity = approval_id(
        run_id=str(run_id),
        owner=str(owner),
        approved_action=str(approved_action),
        scope_class=str(scope_class),
        scope_ref=str(scope_ref),
    )
    try:
        record, minted = _record_approval(
            store_path,
            approved_action=approved_action,
            scope_class=scope_class,
            scope_ref=scope_ref,
            owner=owner,
            run_id=run_id,
            safety_profile_revision=safety_profile_revision,
            confirmation_ladder=confirmation_ladder,
            decision=decision,
            now=now,
        )
    except ApprovalReceiptError as exc:
        return _mint_failure(
            store_path,
            outcome="refused",
            error=str(exc),
            approval_id=identity,
            approved_action=approved_action,
            decision=decision,
            run_id=run_id,
        )
    except OSError as exc:
        # Covers FileLockTimeout, which is a TimeoutError and therefore an
        # OSError: an unavailable lock is a store that could not be written.
        return _mint_failure(
            store_path,
            outcome="not_written",
            error=str(exc),
            approval_id=identity,
            approved_action=approved_action,
            decision=decision,
            run_id=run_id,
        )
    return {
        "schema_version": APPROVAL_MINT_RESULT_SCHEMA_VERSION,
        "minted": minted,
        "outcome": "recorded" if minted else "already_recorded",
        "approval_id": str(record.get("approval_id", "")),
        "receipt_id": str(record.get("receipt_id", "")),
        "decision": str(record.get("decision", "")),
        "error": "",
    }


def read_approval_receipts_result(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[str]]:
    return read_jsonl_objects(paths.runtime_approval_receipts_path)


def read_approval_receipts(
    paths: OmhPaths,
    *,
    run_id: str | None = None,
    approval_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """The store, in append order, optionally scoped to one run or one question."""
    receipts, _ = read_approval_receipts_result(paths)
    if run_id is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("run_id", "")) == run_id]
    if approval_id is not None:
        receipts = [receipt for receipt in receipts if str(receipt.get("approval_id", "")) == approval_id]
    if limit is None:
        return receipts
    if limit < 1:
        return []
    return receipts[-limit:]


def latest_receipt_in(
    receipts: Sequence[Mapping[str, Any]],
    approval_id_value: str,
) -> dict[str, Any]:
    """The receipt that speaks for one confirmation: the one selection rule.

    The store is append-only, so store order is arrival order and the last
    record for a question is its current answer. Every surface that judges an
    approval selects through this function, which is what stops the store
    validator and the satisfaction predicate from picking different receipts for
    one question and reaching opposite conclusions about it.
    """
    return dict(latest_record_in(receipts, key="approval_id", value=str(approval_id_value)))


def approval_age_seconds(decided_at: str, now: str = "") -> int:
    """How old a decision is, computed now, or `UNKNOWN_AGE_SECONDS`.

    A stamp that does not parse and a stamp in the future both return
    `UNKNOWN_AGE_SECONDS`: neither can be shown to be inside the window, and
    clamping a future stamp to zero would make "edit the timestamp forward" a
    way to widen the window from disk.
    """
    decided = _parse_stamp(decided_at)
    if decided is None:
        return UNKNOWN_AGE_SECONDS
    reference = _parse_stamp(now) or datetime.now(timezone.utc)
    age = int((reference - decided).total_seconds())
    if age < 0:
        return UNKNOWN_AGE_SECONDS
    return age


def approval_is_expired(receipt: Mapping[str, Any], now: str = "") -> bool:
    """Whether this receipt is outside the approval window, evaluated now."""
    age = approval_age_seconds(str(receipt.get("decided_at", "")), now)
    return age == UNKNOWN_AGE_SECONDS or age > APPROVAL_TTL_SECONDS


def approval_satisfies_request(
    paths: OmhPaths,
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    now: str = "",
) -> dict[str, Any]:
    """Does an unrevoked, unexpired approval cover exactly this request?

    The entry point a gate calls. Returns an `approval_decision/v1` mapping; see
    `approval_satisfies_request_in` for the shape and for how the refusal code
    is chosen.
    """
    return approval_satisfies_request_in(
        read_approval_receipts(paths),
        approved_action=approved_action,
        scope_class=scope_class,
        scope_ref=scope_ref,
        owner=owner,
        run_id=run_id,
        safety_profile_revision=safety_profile_revision,
        now=now,
    )


def approval_satisfies_request_in(
    receipts: Sequence[Mapping[str, Any]],
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    now: str = "",
) -> dict[str, Any]:
    """`approval_satisfies_request` over receipts already in hand.

    An `approval_decision/v1` mapping:

        satisfied              -- True only when a live approval names exactly
                                  this action, scope, owner, run, and revision
        reason_code            -- "" when satisfied, otherwise one of REFUSAL_CODES
        reason                 -- the constant line for that code
        receipt_id             -- the approval this verdict is about, "" when none
        approval_id            -- identity of the confirmation this request asks about
        age_seconds            -- age of that approval, computed now
        expires_after_seconds  -- APPROVAL_TTL_SECONDS, so a caller can cite the window

    An approval that names this request exactly is judged on its lifecycle:
    superseded, then revoked, then denied, then expired. Otherwise the nearest
    live grant explains what it does cover, reporting its first differing
    dimension in `SCOPE_REFUSAL_CODES` order. There is no containment rule
    anywhere in that walk, so nothing widens.
    """
    identity = approval_id(
        run_id=str(run_id),
        owner=str(owner),
        approved_action=str(approved_action),
        scope_class=str(scope_class),
        scope_ref=str(scope_ref),
    )
    request = (
        str(run_id),
        str(owner),
        str(approved_action),
        (str(scope_class), str(scope_ref)),
        str(safety_profile_revision),
    )
    known = [receipt for receipt in receipts if isinstance(receipt, Mapping)]
    exact = [receipt for receipt in known if not _dimension_mismatches(receipt, request)]
    if exact:
        return _lifecycle_verdict(known, exact[-1], identity, request, now)
    candidate, mismatches = _nearest_live_grant(known, request, now)
    if candidate is None:
        return _decision_payload(
            satisfied=False,
            reason_code=REFUSAL_ABSENT,
            receipt={},
            approval_id_value=identity,
            request=request,
            now=now,
        )
    return _decision_payload(
        satisfied=False,
        reason_code=mismatches[0],
        receipt=candidate,
        approval_id_value=identity,
        request=request,
        now=now,
    )


def validate_approval_receipt_store(path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """Validate the approval store, optionally scoped to one run's records.

    Unscoped (`run_id=None`) this is the store's own report: unparseable lines,
    every record's schema, and the integrity of the supersede chain. A line that
    does not parse belongs to no run -- it has no `run_id` to read -- so it is
    the store's fault and is reported here, once, never against a run that
    happened to be validated while it sat there.
    """
    receipts, read_errors = read_jsonl_objects(path)
    receipt_count, errors = store_errors(
        path,
        receipts,
        read_errors,
        run_id=run_id,
        validate_record=validate_approval_receipt,
        label=_LABEL,
    )
    return {
        "schema_version": APPROVAL_RECEIPT_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "run_id": run_id or "",
        "ok": not errors,
        "receipt_count": receipt_count,
        "mint_failure_count": len(read_approval_mint_failures(path)),
        "errors": errors,
    }


def approval_mint_failures_path(store_path: Path) -> Path:
    """Where unrecorded answers are logged, beside the store they failed to reach."""
    return store_path.with_name(APPROVAL_MINT_FAILURE_STORE_NAME)


def read_approval_mint_failures(store_path: Path) -> list[dict[str, Any]]:
    failures, _ = read_jsonl_objects(approval_mint_failures_path(store_path))
    return failures


def compact_approval_receipt(receipt: Mapping[str, Any], *, now: str = "") -> dict[str, Any]:
    """The user-facing shape, with every field through its class's guard.

    Rendering is the last point at which a hand-edited store reaches a human, so
    nothing is copied through verbatim: identifiers fold to a stable
    non-navigable handle when they are not opaque, and closed vocabularies
    render empty outside their vocabulary. `age_seconds` and `expired` are
    derived here rather than stored, which is the whole point -- nothing on disk
    says an approval is still good.
    """
    age = approval_age_seconds(str(receipt.get("decided_at", "")), now)
    return {
        "receipt_id": redacted_approval_ref(str(receipt.get("receipt_id", ""))),
        "approval_id": redacted_approval_ref(str(receipt.get("approval_id", ""))),
        "run_id": redacted_approval_ref(str(receipt.get("run_id", ""))),
        "owner": redacted_approval_ref(str(receipt.get("owner", ""))),
        "approved_action": closed_approval_value(receipt.get("approved_action"), APPROVABLE_ACTIONS),
        "scope_class": closed_approval_value(receipt.get("scope_class"), SCOPE_CLASSES),
        "scope_ref": redacted_approval_ref(str(receipt.get("scope_ref", ""))),
        "safety_profile_revision": redacted_approval_ref(str(receipt.get("safety_profile_revision", ""))),
        "confirmation_ladder": closed_approval_value(receipt.get("confirmation_ladder"), CONFIRMATION_LADDERS),
        "decision": closed_approval_value(receipt.get("decision"), APPROVAL_DECISIONS),
        "decided_at": redacted_approval_ref(str(receipt.get("decided_at", ""))),
        "supersedes_receipt_ref": redacted_approval_ref(str(receipt.get("supersedes_receipt_ref", ""))),
        "age_seconds": age,
        "expired": approval_is_expired(receipt, now),
    }


def redacted_approval_ref(value: str) -> str:
    """Fold any handle into a bounded, non-navigable receipt reference."""
    return redacted_ref(value, field="approval_ref")


def closed_approval_value(value: Any, allowed: tuple[str, ...]) -> str:
    """A closed-vocabulary field renders only what its vocabulary contains.

    Rendering reads whatever is on disk, and a store can be hand-edited. A value
    outside the vocabulary is not a new state, it is a value with no meaning, so
    it renders empty rather than as itself.
    """
    return closed_vocabulary_value(value, allowed)


def _record_approval(
    path: Path,
    *,
    approved_action: str,
    scope_class: str,
    scope_ref: str,
    owner: str,
    run_id: str,
    safety_profile_revision: str,
    confirmation_ladder: str,
    decision: str,
    now: str,
) -> tuple[dict[str, Any], bool]:
    """The receipt now on record, and whether this call is what put it there."""
    with file_lock(path, private=True):
        identity = approval_id(
            run_id=str(run_id),
            owner=str(owner),
            approved_action=str(approved_action),
            scope_class=str(scope_class),
            scope_ref=str(scope_ref),
        )
        receipts, _ = read_jsonl_objects(path)
        prior = latest_receipt_in(receipts, identity)
        record = build_approval_receipt(
            approved_action=approved_action,
            scope_class=scope_class,
            scope_ref=scope_ref,
            owner=owner,
            run_id=run_id,
            safety_profile_revision=safety_profile_revision,
            confirmation_ladder=confirmation_ladder,
            decision=decision,
            # `now` is the clock for this call: it stamps the new receipt and it
            # is the instant the prior receipt's freshness is judged against.
            # Two clocks here -- a real stamp and an injected comparison -- is
            # how a fresh receipt ends up looking expired the moment it is
            # written.
            decided_at=now,
            supersedes_receipt_ref=str(prior.get("receipt_id", "")) if prior else "",
        )
        if prior and _decision_fingerprint(prior) == _decision_fingerprint(record):
            # Identical re-report of an answer that is still live. A grant past
            # its window is *not* still live: consent re-given after expiry is
            # new consent, and returning the stale receipt would hand the caller
            # an approval that refuses one line later.
            if str(prior.get("decision", "")) != "granted" or not approval_is_expired(prior, now):
                return prior, False
        append_store_line(path, record)
    return record, True


def _mint_failure(
    store_path: Path,
    *,
    outcome: str,
    error: str,
    approval_id: str,
    approved_action: str,
    decision: str,
    run_id: str,
) -> dict[str, Any]:
    failure = {
        "schema_version": APPROVAL_MINT_FAILURE_SCHEMA_VERSION,
        "recorded_at": utc_now(),
        "outcome": outcome,
        "approval_id": str(approval_id),
        "approved_action": str(approved_action),
        "decision": str(decision),
        "run_id": str(run_id),
        "error": _bounded_error(error),
    }
    _append_mint_failure(store_path, failure)
    return {
        "schema_version": APPROVAL_MINT_RESULT_SCHEMA_VERSION,
        "minted": False,
        "outcome": outcome,
        "approval_id": str(approval_id),
        "receipt_id": "",
        "decision": "",
        "error": str(error),
    }


def _append_mint_failure(store_path: Path, failure: dict[str, Any]) -> None:
    append_sidecar_line(approval_mint_failures_path(store_path), failure)


def _lifecycle_verdict(
    receipts: Sequence[Mapping[str, Any]],
    exact: Mapping[str, Any],
    identity: str,
    request: tuple[Any, ...],
    now: str,
) -> dict[str, Any]:
    """Judge an approval that names this request exactly.

    Order matters. Superseded first: a later answer to the same confirmation is
    the current one whatever the earlier record said, so an operator who
    re-answered must not still be bound by what they replaced. Then the terminal
    decisions, then the window.
    """
    head = latest_receipt_in(receipts, str(exact.get("approval_id", "")))
    if str(head.get("receipt_id", "")) != str(exact.get("receipt_id", "")):
        return _decision_payload(
            satisfied=False,
            reason_code=REFUSAL_SUPERSEDED,
            receipt=exact,
            approval_id_value=identity,
            request=request,
            now=now,
        )
    decision = str(exact.get("decision", ""))
    if decision == "revoked":
        return _decision_payload(
            satisfied=False,
            reason_code=REFUSAL_REVOKED,
            receipt=exact,
            approval_id_value=identity,
            request=request,
            now=now,
        )
    if decision != "granted":
        return _decision_payload(
            satisfied=False,
            reason_code=REFUSAL_DENIED,
            receipt=exact,
            approval_id_value=identity,
            request=request,
            now=now,
        )
    if approval_is_expired(exact, now):
        return _decision_payload(
            satisfied=False,
            reason_code=REFUSAL_EXPIRED,
            receipt=exact,
            approval_id_value=identity,
            request=request,
            now=now,
        )
    return _decision_payload(
        satisfied=True,
        reason_code="",
        receipt=exact,
        approval_id_value=identity,
        request=request,
        now=now,
    )


def _nearest_live_grant(
    receipts: Sequence[Mapping[str, Any]],
    request: tuple[Any, ...],
    now: str,
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    """The live grant closest to this request, and how it differs.

    Only grants that are still the current answer to their own confirmation are
    considered: a revoked, denied, or superseded record explains nothing about
    what is allowed now. An expired grant stays a candidate on purpose -- "there
    is an approval for another run" is a more useful refusal than "there is no
    approval", and it refuses either way.
    """
    heads = _chain_heads(receipts)
    best: Mapping[str, Any] | None = None
    best_mismatches: tuple[str, ...] = ()
    for receipt in receipts:
        if str(receipt.get("decision", "")) != "granted":
            continue
        if heads.get(str(receipt.get("approval_id", ""))) != str(receipt.get("receipt_id", "")):
            continue
        mismatches = _dimension_mismatches(receipt, request)
        if not mismatches:
            continue
        if best is None or len(mismatches) <= len(best_mismatches):
            best = receipt
            best_mismatches = mismatches
    return best, best_mismatches


def _chain_heads(receipts: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The current answer to every confirmation in one pass over the store.

    The same selection rule as `latest_receipt_in` -- last record wins -- built
    once so judging a store of n receipts does not walk it n times.
    """
    heads: dict[str, str] = {}
    for receipt in receipts:
        heads[str(receipt.get("approval_id", ""))] = str(receipt.get("receipt_id", ""))
    return heads


def _dimension_mismatches(receipt: Mapping[str, Any], request: tuple[Any, ...]) -> tuple[str, ...]:
    """Which of the five bound dimensions this receipt does not match.

    Append order is `SCOPE_REFUSAL_CODES` order, which is the reporting
    precedence. Every comparison is equality, including the scope pair: there is
    no containment rule here, and that absence is what makes widening
    structurally impossible rather than merely unimplemented.
    """
    run_id, owner, action, scope, revision = request
    mismatches: list[str] = []
    if str(receipt.get("run_id", "")) != run_id:
        mismatches.append(REFUSAL_RUN)
    if str(receipt.get("owner", "")) != owner:
        mismatches.append(REFUSAL_OWNER)
    if str(receipt.get("approved_action", "")) != action:
        mismatches.append(REFUSAL_ACTION)
    if (str(receipt.get("scope_class", "")), str(receipt.get("scope_ref", ""))) != scope:
        mismatches.append(REFUSAL_SCOPE)
    if str(receipt.get("safety_profile_revision", "")) != revision:
        mismatches.append(REFUSAL_REVISION)
    return tuple(mismatches)


def _decision_payload(
    *,
    satisfied: bool,
    reason_code: str,
    receipt: Mapping[str, Any],
    approval_id_value: str,
    request: tuple[Any, ...],
    now: str,
) -> dict[str, Any]:
    run_id, owner, action, scope, revision = request
    scope_class, scope_ref = scope
    return {
        "schema_version": APPROVAL_DECISION_SCHEMA_VERSION,
        "satisfied": satisfied,
        "reason_code": reason_code,
        "reason": REFUSAL_REASONS.get(reason_code, ""),
        "receipt_id": redacted_approval_ref(str(receipt.get("receipt_id", ""))),
        "approval_id": redacted_approval_ref(approval_id_value),
        "requested_action": redacted_approval_ref(str(action)),
        "requested_scope_class": redacted_approval_ref(str(scope_class)),
        "requested_scope_ref": redacted_approval_ref(str(scope_ref)),
        "requested_owner": redacted_approval_ref(str(owner)),
        "requested_run_id": redacted_approval_ref(str(run_id)),
        "requested_safety_profile_revision": redacted_approval_ref(str(revision)),
        "decided_at": redacted_approval_ref(str(receipt.get("decided_at", ""))),
        "age_seconds": approval_age_seconds(str(receipt.get("decided_at", "")), now) if receipt else UNKNOWN_AGE_SECONDS,
        "expires_after_seconds": APPROVAL_TTL_SECONDS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _decision_fingerprint(receipt: Mapping[str, Any]) -> str:
    """Which answer a receipt records, ignoring when it was recorded."""
    return record_fingerprint(receipt, _DECISION_IDENTITY_KEYS)


def _parse_stamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = strict_fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _scope_ref_errors(value: Any) -> list[str]:
    errors = reference_errors(value, field="scope_ref", label=_LABEL, required=True)
    if errors or not isinstance(value, str):
        return errors
    if _SCOPE_TRAVERSAL.search(value):
        return ["approval_receipt scope_ref must not escape the workspace"]
    if _WINDOWS_DRIVE.match(value):
        return ["approval_receipt scope_ref must be workspace-relative, not drive-anchored"]
    return []


def _scope_ref(value: str) -> str:
    text = _opaque(value, field="approval_receipt scope_ref")
    errors = _scope_ref_errors(text)
    if errors:
        raise ApprovalReceiptError(errors[0])
    return text


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=ApprovalReceiptError)


def _bounded_error(value: str) -> str:
    """One bounded, secret-free line for the failure log."""
    raw = " ".join(str(value or "").split())
    if not raw:
        return ""
    if is_sensitive_metadata_text(raw):
        return "[redacted]"
    return raw[:200]


def _receipt_id(decided_at: str, approval_id_value: str, decision: str) -> str:
    return mint_record_id(
        prefix="approval-receipt",
        identity={"approval_id": approval_id_value, "decided_at": decided_at, "decision": decision},
    )
