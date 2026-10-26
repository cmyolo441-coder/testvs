"""Append-only decision gates: one question, one answer, one released transition (issue #825).

What was missing
----------------

OMH could already stop work and say why. `coding.action_gate` withholds an
action pending approval, `workflows.approval_receipts` records the consent an
operator gave, and `workflows.blocked_work_records` keeps the decision
explainable after the turn. What none of them holds is the *question*: which
one thing a human is being asked right now, which choices they have, what each
choice costs, and -- the part that survives nothing today -- exactly which
blocked transition their answer releases.

Without that, a pause is durable and a resume is not. The gate is reconstructed
from whatever the next turn happens to remember, so an answer given to one
question can be applied to a different one, and a restart between the question
and the answer loses the binding entirely.

Why this is its own record family
---------------------------------

The repo refused a new family twice (#811, #818) and accepted three (#807,
#808/#806, #826) on a rule worth restating: a family joins on run id, and a join
is where an artifact and its metadata desynchronize, so a field group on the
record that already exists is the default. A separate family has to earn it.

This one earns it on the same dimensions #807 used, and it earns it hardest
against the family it looks most like. An approval receipt records *an answer
that was given*. It has no shape for a question that has not been answered yet:
`APPROVAL_DECISIONS` is `granted`/`denied`/`revoked`, all three of them past
tense, and there is deliberately no `pending` member -- an unanswered
confirmation must not sit in the approval store looking like a weak grant. A
decision gate is the other half: it exists *before* the answer, which is the
whole of what "durable and resumable" means here, and it is written by the
surface that blocked rather than by the operator who answers.

It is also bound to something an approval is not bound to. An approval binds
five dimensions of a *request*. A gate additionally binds the *position* the run
is paused at -- the blocked transition and the checkpoint -- because releasing
the wrong transition on a correct subject is the failure mode #825 AC3 names.

Coexisting with the name that was already here
----------------------------------------------

`decision_gate` was already a live string in this tree, in two unrelated places,
and neither of them is this:

* `catalogs.playbooks` names three wrapper stages whose declared artifact
  contract is `decision_gate/v1` (`decision_gate`, `delivery_decision`,
  `deploy_decision`). That string named a kind of evidence with no schema behind
  it. This module is that schema, arriving under the name those stages have been
  citing all along, which is why the version string here is `decision_gate/v1`
  and not a second spelling of it.
* `workflows.hermes_planning._wrapper_contract` carries a `decision_gate`
  sub-dict -- `required` / `condition` / `do_not_delegate_when`. That is a
  *contract clause*, not a record: it tells a wrapper the conditions under which
  a prepared plan may be delegated, it is rebuilt from the plan on every call,
  it names no subject, no approver, and no answer, and nothing about it survives
  the turn. It is deliberately left alone. `tests/test_decision_gate.py` pins the
  two apart so a later reader cannot collapse them.

What a gate binds, and what releases it
---------------------------------------

A resume is matched on equality across four groups, each with its own refusal
code, and the codes are `approval_receipts`' own rather than a parallel set:

    run_id                              -> run_not_approved
    (subject_class, subject_ref)        -> scope_not_approved
    (safety_profile_revision,
     context_revision)                  -> safety_revision_not_approved
    (blocked_transition, checkpoint_ref)-> transition_not_unblocked

The first three are `SCOPE_REFUSAL_CODES` members, reused because "this answer
does not cover you" already has a vocabulary in this tree and a second one would
be a second answer to the same question. The fourth is this family's own,
because no existing code says "right subject, wrong place in the run".

`resume_digest` is the single value that seals all seven of those fields at
once. It is re-derived at validate, so a hand-edited store cannot move an answer
onto a transition it was never given for, and it is what a caller cites when it
says which resume it is performing.

There is no containment rule anywhere in this module. Matching is equality on
every dimension, which is what makes widening structurally impossible rather
than merely unimplemented.

One open gate per subject
-------------------------

#825 AC1 is that a blocked workflow identifies *one* open gate. That is enforced
in both directions rather than left to a renderer: `open_decision_gate` refuses
to open a second gate while another is open for the same subject, and
`validate_decision_gate_store` faults a store that already holds two, so a
hand-edited or concurrently written store fails `omh runtime validate` instead
of presenting an operator with two questions and no rule for which one their
answer belongs to.

Nothing here can carry a sentence
---------------------------------

The key set is closed, `RAW_OR_HIDDEN_KEYS` is refused by name first, and there
is no free-text field -- not for the question, not for the risk, not for a
choice's consequence. Every one of those is a closed vocabulary id whose prose
is a module constant joined at render, the idiom `approval_receipts.REFUSAL_REASONS`
and `run_lineage.BREAK_REASONS` already use. A gate is rendered to an operator
who is about to authorise something, so the text they read must be text this
repo wrote, never text that arrived with the request.

An answered gate is not evidence
--------------------------------

`CLAIM_BOUNDARY` is on every record and `validate_decision_gate` refuses a record
shaped to assert execution by key name as well as by the closed key set. Choosing
`approve` proves a human chose it. It does not prove the transition was taken,
and it never proves anything ran.

Determinism
-----------

`resume_digest` covers no clock. `opened_at` and `decided_at` are parameters, and
freshness is recomputed by every reader from the stored stamp against a `now`
that is passed in -- the `approval_receipts` idiom, reusing that module's
`APPROVAL_TTL_SECONDS` and `approval_age_seconds` rather than declaring a second
window. Nothing on disk says a decision is still good.

Store mechanics
---------------

All of it is `system.append_only_store`, shared with the four families beside it:
the torn-tail-safe binary append under a real OS lock on both platforms, the
opaque-reference guards, the digest handles, and the chain walk. This module adds
no store mechanics of its own.

What produces and reads these records today
-------------------------------------------

Stated because the vocabularies here are wider than the wiring, and a reader who
took `QUESTION_CODES` for a coverage claim would be wrong. No production surface
mints a gate yet: `coding.action_gate` still withholds an action without opening
one, exactly as `run_lineage` shipped its chain before anything appended to it.
The one production read surface is `runtime.artifacts.validate_runtime`, which
carries this store's integrity -- including the one-open-gate rule -- into
`omh runtime validate`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
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
from ..system.paths import OmhPaths
from .approval_receipts import (
    APPROVAL_TTL_SECONDS,
    REFUSAL_ABSENT,
    REFUSAL_EXPIRED,
    REFUSAL_REVISION,
    REFUSAL_RUN,
    REFUSAL_SCOPE,
    REFUSAL_SUPERSEDED,
    SCOPE_CLASSES,
    UNKNOWN_AGE_SECONDS,
    approval_age_seconds,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


DECISION_GATE_SCHEMA_VERSION: Final[str] = "decision_gate/v1"
DECISION_GATE_RESUME_SCHEMA_VERSION: Final[str] = "decision_gate_resume/v1"
DECISION_GATE_VIEW_SCHEMA_VERSION: Final[str] = "decision_gate_view/v1"
DECISION_GATE_STORE_VALIDATION_SCHEMA_VERSION: Final[str] = "decision_gate_store_validation/v1"

DECISION_GATE_STORE_NAME: Final[str] = "decision_gates.jsonl"

GATE_PRIVACY: Final[str] = "metadata_only"

# Stated on every record and echoed on every resume verdict and every view. The
# claim a gate makes is deliberately small, and the last clause is the one #825
# AC3 rests on: an answer is bound to one transition and releases that one.
CLAIM_BOUNDARY: Final[str] = (
    "A decision gate is one recorded question put to one named approver about one subject, under one "
    "safety profile revision and one context revision, and at most one recorded answer to it. An answered "
    "gate proves a human chose one of the choices they were offered. It is not dispatch, execution, "
    "verification, review, CI, merge-readiness, or merge evidence, and it releases exactly the one blocked "
    "transition its resume digest names and nothing else."
)

# What kind of thing `subject_ref` names. This is `approval_receipts.SCOPE_CLASSES`
# under the name this family uses for it, imported rather than restated: a gate
# asks about the same kinds of thing an approval is scoped to, and two answers to
# "what class is this subject" is the defect a shared vocabulary prevents.
SUBJECT_CLASSES: Final[tuple[str, ...]] = SCOPE_CLASSES

# Where in a run's lifecycle the work is paused. The names are the observation
# journal's own canonical event names, restated here rather than imported for the
# reason `run_lineage.LINEAGE_EVIDENCE_CLASSES` is restated: a store's on-disk
# vocabulary must not change because another module reordered a tuple.
# `tests/test_decision_gate.py` pins this equal to `run_lineage.LINEAGE_TRANSITIONS`,
# so drift is a failing test rather than a silent schema change.
BLOCKED_TRANSITIONS: Final[tuple[str, ...]] = (
    "plan_accepted",
    "handoff_prepared",
    "runtime_start_observed",
    "worktree_creation_observed",
    "executor_dispatch_observed",
    "executor_result_observed",
    "verification_result_observed",
    "review_result_observed",
    "ci_result_observed",
    "merge_gate_observed",
    "merge_observed",
)

# What a gate can ask. Closed, because the question is rendered to the person
# authorising the thing: the words they read must be words this repo wrote.
QUESTION_CODES: Final[tuple[str, ...]] = (
    "accept_plan_direction",
    "approve_scope_widening",
    "choose_executor_owner",
    "confirm_destructive_action",
    "confirm_external_effect",
    "continue_after_revision_change",
)

QUESTION_TEXT: Final[dict[str, str]] = {
    "accept_plan_direction": "Accept this plan direction so the blocked transition can continue?",
    "approve_scope_widening": (
        "Widen the authority envelope to cover this subject so the blocked transition can continue?"
    ),
    "choose_executor_owner": "Confirm this coding owner so the blocked transition can continue?",
    "confirm_destructive_action": (
        "Confirm this action on this subject, knowing it changes state that is not trivially restored?"
    ),
    "confirm_external_effect": "Confirm this effect outside the workspace on this subject?",
    "continue_after_revision_change": "Continue from this checkpoint under the revisions now in force?",
}

# How bad releasing the transition is if the answer turns out to be wrong. Three
# classes rather than a score: an operator acts on "can I undo this", and a
# number invites a threshold nobody agreed on.
RISK_CLASSES: Final[tuple[str, ...]] = ("reversible", "costly_to_reverse", "irreversible")

RISK_TEXT: Final[dict[str, str]] = {
    "reversible": "Releasing this transition has an effect the same surface can undo.",
    "costly_to_reverse": "Releasing this transition has an effect that can be undone only by work outside this run.",
    "irreversible": "Releasing this transition has an effect that cannot be undone once it is applied.",
}

# The answers a gate can offer.
GATE_CHOICES: Final[tuple[str, ...]] = ("approve", "decline", "defer")

# Which choices release the blocked transition. Exactly one does, and the table
# is what makes "rejected gates are non-resumable" a lookup rather than a string
# comparison scattered across readers.
CHOICE_RELEASES: Final[dict[str, bool]] = {"approve": True, "decline": False, "defer": False}

CHOICE_CONSEQUENCES: Final[dict[str, str]] = {
    "approve": (
        "The blocked transition is released once, for this subject at these revisions, and for nothing else."
    ),
    "decline": "The blocked transition stays blocked and the question is closed; asking again needs a new gate.",
    "defer": (
        "The blocked transition stays blocked and the question is closed unanswered; ask again when the "
        "decision can be made."
    ),
}

# `open`     -- the question is on record and nobody has answered it.
# `answered` -- one of the offered choices was chosen, by the approver the gate
#               named, at a recorded time.
#
# There is deliberately no `withdrawn`, `pending`, or `expired` member. The first
# would be a third way to close a question when `decline` already exists; the
# other two are not states a record is in, they are properties a reader derives
# -- `pending` is what `open` means, and expiry is recomputed from the stamp for
# the reason `approval_receipts` gives at length.
GATE_STATES: Final[tuple[str, ...]] = ("open", "answered")
GATE_STATE_OPEN: Final[str] = "open"
GATE_STATE_ANSWERED: Final[str] = "answered"

DECISION_GATE_KEYS: Final[tuple[str, ...]] = (
    "actor",
    "answered_choice",
    "approver",
    "blocked_transition",
    "checkpoint_ref",
    "choices",
    "claim_boundary",
    "context_revision",
    "decided_at",
    "gate_id",
    "opened_at",
    "privacy",
    "question_code",
    "record_id",
    "resume_digest",
    "risk_class",
    "run_id",
    "safety_profile_revision",
    "schema_version",
    "state",
    "subject_class",
    "subject_id",
    "subject_ref",
    "supersedes_gate_ref",
)

# The three keys whose value is a module constant rather than data. Everything
# else on a record is rendered, and `compact_decision_gate` must cover all of it
# -- a key in the tuple and missing from the compactor is a field that is stored
# and never seen, which has cost this repo twice.
DECISION_GATE_CONSTANT_KEYS: Final[tuple[str, ...]] = ("claim_boundary", "privacy", "schema_version")

# Exactly what `resume_digest` seals: the subject, both revisions, and the
# position in the run. Nothing else, and no clock -- a wall clock inside a
# compared value turns an equality check into a race, which this repo has already
# lost Windows CI time to.
RESUME_DIGEST_KEYS: Final[tuple[str, ...]] = (
    "blocked_transition",
    "checkpoint_ref",
    "context_revision",
    "run_id",
    "safety_profile_revision",
    "subject_class",
    "subject_ref",
)

# The fields only an answered record carries. Empty on an open one, and required
# on an answered one, both checked: an answer missing any of them is #825 AC2's
# refusal, and an open record carrying any of them would be an answer nobody gave.
_ANSWER_FIELDS: Final[tuple[str, ...]] = ("actor", "answered_choice", "decided_at")

# Identifiers that must be present, through `require_opaque_metadata_ref`.
_REQUIRED_GATE_REFS: Final[tuple[str, ...]] = (
    "approver",
    "checkpoint_ref",
    "gate_id",
    "opened_at",
    "record_id",
    "run_id",
    "safety_profile_revision",
    "subject_id",
)
# Legitimately absent. The three answer fields are empty while the gate is open,
# a run whose context was never revised has no context revision, and the link is
# empty on the first record for a question and only then.
_OPTIONAL_GATE_REFS: Final[tuple[str, ...]] = (
    "actor",
    "context_revision",
    "decided_at",
    "supersedes_gate_ref",
)
_GATE_VOCABULARIES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("blocked_transition", BLOCKED_TRANSITIONS),
    ("question_code", QUESTION_CODES),
    ("risk_class", RISK_CLASSES),
    ("state", GATE_STATES),
    ("subject_class", SUBJECT_CLASSES),
)
# The one list field. Its members are closed-vocabulary choice ids, never caller
# text, which is why this family has a list at all and still has no free text.
_GATE_LIST_FIELDS: Final[tuple[str, ...]] = ("choices",)

_GATE_STRING_FIELDS: Final[tuple[str, ...]] = tuple(
    key for key in DECISION_GATE_KEYS if key not in _GATE_LIST_FIELDS
)

# This family's own refusal, for the one dimension `approval_receipts` has no
# code for. An approval binds a request; a gate additionally binds the position
# the run is paused at, and "right subject, wrong place in the run" is the
# failure #825 AC3 is named after.
REFUSAL_TRANSITION: Final[str] = "transition_not_unblocked"
# The question is on record and unanswered. Distinct from `approval_absent`,
# which says nothing on record names this request at all: an operator who has
# been asked and has not replied is in a different position from one who was
# never asked, and the two need different next steps.
REFUSAL_PENDING: Final[str] = "decision_pending"
# Answered with a choice that does not release the transition.
REFUSAL_DECLINED: Final[str] = "decision_declined"

# Which dimension a candidate gate differs on, in the order a verdict reports
# them when it differs on more than one. Run first because a gate borrowed from
# another run is the failure this family shares with #807; then the subject, then
# the revisions it was asked under, then where in the run it was asked.
GATE_MISMATCH_CODES: Final[tuple[str, ...]] = (
    REFUSAL_RUN,
    REFUSAL_SCOPE,
    REFUSAL_REVISION,
    REFUSAL_TRANSITION,
)
# The lifecycle refusals, in the precedence `_lifecycle_verdict` applies.
GATE_LIFECYCLE_CODES: Final[tuple[str, ...]] = (
    REFUSAL_SUPERSEDED,
    REFUSAL_PENDING,
    REFUSAL_DECLINED,
    REFUSAL_EXPIRED,
    REFUSAL_ABSENT,
)
GATE_REFUSAL_CODES: Final[tuple[str, ...]] = GATE_MISMATCH_CODES + GATE_LIFECYCLE_CODES

# The three mismatch codes this family did not invent. Pinned against
# `approval_receipts.SCOPE_REFUSAL_CODES` by `tests/test_decision_gate.py`, so a
# rename there is a failing test here rather than a second vocabulary.
REUSED_REFUSAL_CODES: Final[tuple[str, ...]] = (REFUSAL_RUN, REFUSAL_SCOPE, REFUSAL_REVISION)

# Constant strings, keyed by code. Nothing is interpolated into a reason: a
# refusal is rendered to operators, and a line built from request values would be
# one more path for caller-controlled text to reach a terminal. The prose is this
# family's own even where the code is borrowed -- the code says which dimension
# failed, and the sentence has to say it about a gate rather than about a receipt.
GATE_REFUSAL_REASONS: Final[dict[str, str]] = {
    REFUSAL_RUN: "The gate on record belongs to another run; a decision is run-bound and never carries over.",
    REFUSAL_SCOPE: (
        "The gate on record names another subject; an answer covers exactly the subject it was asked about "
        "and never a broader one."
    ),
    REFUSAL_REVISION: (
        "The gate on record was asked under another safety profile or context revision, so its answer speaks "
        "for a state the run has left."
    ),
    REFUSAL_TRANSITION: (
        "The gate on record unblocks another transition or another checkpoint; a resume releases exactly the "
        "transition its digest names."
    ),
    REFUSAL_PENDING: "The gate on record is still open; a question nobody has answered releases nothing.",
    REFUSAL_DECLINED: (
        "The gate on record was answered with a choice that does not release the blocked transition."
    ),
    REFUSAL_EXPIRED: (
        "The answer on record is outside the decision window, measured now from its decision stamp."
    ),
    REFUSAL_SUPERSEDED: "The gate on record was superseded by a later record for the same question.",
    REFUSAL_ABSENT: (
        "No decision gate on record names this subject, these revisions, this blocked transition, and this "
        "checkpoint."
    ),
}

# Key names an execution claim arrives under. The closed key set already refuses
# every one of them, but it refuses them as "unsupported keys", and the point of
# this family is that an answered question is not an outcome -- so the shape is
# named and refused by name first. Mirrors `run_lineage.EXECUTION_CLAIM_KEYS`
# rather than importing across two unrelated record families;
# `tests/test_decision_gate.py` pins them equal so the two cannot drift.
EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
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
_LABEL: Final[str] = "decision_gate"

# A resume digest is a full sha256 hex string and nothing else. Enforced as a
# shape as well as by re-derivation, so a hand-edited store carrying a path, a
# URL, or a truncated handle in the field a resume is matched on is refused
# before anything tries to match on it.
_DIGEST: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

# A subject must name something inside the workspace. The shared
# `require_opaque_metadata_ref` already refuses a leading `/` or `~` and every
# backslash, so what is left is the parent traversal and the Windows drive
# prefix -- the two ways an in-bounds-looking subject reaches out. Held in the
# same shape as `approval_receipts`, which draws the identical line for the
# identical reason.
_SUBJECT_TRAVERSAL: Final[re.Pattern[str]] = re.compile(r"(?:^|/)\.\.(?:/|$)")
_WINDOWS_DRIVE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]:")


class DecisionGateError(ValueError):
    """Raised when a question or an answer cannot become a decision gate record."""


# ---------------------------------------------------------------------------
# Identity and digest
# ---------------------------------------------------------------------------


def gate_subject_id(*, run_id: str, subject_class: str, subject_ref: str) -> str:
    """Stable identity of one subject inside one run.

    The unit #825 AC1's "one open gate" is counted over. Deliberately narrower
    than `gate_id`: two questions about one subject at two different points in a
    run have two gate ids and one subject id, and that is exactly the state the
    open-gate rule refuses.
    """
    base = json.dumps(
        {
            "run_id": str(run_id),
            "subject_class": str(subject_class),
            "subject_ref": str(subject_ref),
        },
        sort_keys=True,
    )
    return f"gate-subject-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


def gate_id(
    *,
    run_id: str,
    subject_class: str,
    subject_ref: str,
    blocked_transition: str,
    checkpoint_ref: str,
) -> str:
    """Stable identity of one question.

    The revisions are deliberately not in it, for the reason
    `approval_receipts.approval_id` leaves the safety profile out of its own
    identity: a revision is part of the *answer's* validity, not part of the
    question, so re-asking after the profile moved must supersede the earlier
    record rather than open a second parallel chain that both claim to be
    current.
    """
    base = json.dumps(
        {
            "blocked_transition": str(blocked_transition),
            "checkpoint_ref": str(checkpoint_ref),
            "run_id": str(run_id),
            "subject_class": str(subject_class),
            "subject_ref": str(subject_ref),
        },
        sort_keys=True,
    )
    return f"gate-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


def resume_digest(
    *,
    run_id: str,
    subject_class: str,
    subject_ref: str,
    blocked_transition: str,
    checkpoint_ref: str,
    safety_profile_revision: str,
    context_revision: str = "",
) -> str:
    """The one value that identifies which blocked transition an answer releases.

    `record_fingerprint` is the shared hasher; nothing here reimplements it. The
    key tuple is what this family contributes, and it covers no clock, so the
    same resume computed a second apart is the same digest.

    A caller performing a resume computes this from what it holds and cites it.
    `validate_decision_gate` re-derives it from the record's own fields, so an
    answer cannot be moved onto a transition it was never given for by editing
    one number on disk.
    """
    return record_fingerprint(
        {
            "blocked_transition": str(blocked_transition),
            "checkpoint_ref": str(checkpoint_ref),
            "context_revision": str(context_revision or ""),
            "run_id": str(run_id),
            "safety_profile_revision": str(safety_profile_revision),
            "subject_class": str(subject_class),
            "subject_ref": str(subject_ref),
        },
        RESUME_DIGEST_KEYS,
    )


def resume_digest_of(record: Mapping[str, Any]) -> str:
    """`resume_digest` over the sealed fields of a record already in hand."""
    return record_fingerprint(record, RESUME_DIGEST_KEYS)


# ---------------------------------------------------------------------------
# Build, validate, append
# ---------------------------------------------------------------------------


def build_decision_gate(
    *,
    run_id: str,
    subject_class: str,
    subject_ref: str,
    blocked_transition: str,
    checkpoint_ref: str,
    question_code: str,
    risk_class: str,
    choices: Sequence[str],
    approver: str,
    safety_profile_revision: str,
    context_revision: str = "",
    opened_at: str = "",
    supersedes_gate_ref: str = "",
) -> dict[str, Any]:
    """Mint one open gate, or refuse.

    There is no `state` argument and no way to reach `answered` from here. An
    answer is built by `build_decision_gate_answer` *from the open record it
    answers*, which is what makes "the answer names the same subject, the same
    revisions, and the same transition as the question" a property of the
    constructor rather than a check somebody has to remember to run.
    """
    if question_code not in QUESTION_CODES:
        raise DecisionGateError(f"decision_gate question_code is unsupported: {question_code!r}")
    if risk_class not in RISK_CLASSES:
        raise DecisionGateError(f"decision_gate risk_class is unsupported: {risk_class!r}")
    if subject_class not in SUBJECT_CLASSES:
        raise DecisionGateError(f"decision_gate subject_class is unsupported: {subject_class!r}")
    if blocked_transition not in BLOCKED_TRANSITIONS:
        raise DecisionGateError(f"decision_gate blocked_transition is unsupported: {blocked_transition!r}")
    offered = _normalized_choices(choices)
    safe_subject_ref = _subject_ref(subject_ref)
    safe_run_id = _opaque(run_id, field="decision_gate run_id")
    safe_checkpoint = _opaque(checkpoint_ref, field="decision_gate checkpoint_ref")
    safe_approver = _opaque(approver, field="decision_gate approver")
    safe_revision = _opaque(safety_profile_revision, field="decision_gate safety_profile_revision")
    safe_context = (
        _opaque(context_revision, field="decision_gate context_revision") if context_revision else ""
    )
    stamp = _opaque(str(opened_at or utc_now()), field="decision_gate opened_at")
    safe_supersedes = (
        _opaque(supersedes_gate_ref, field="decision_gate supersedes_gate_ref") if supersedes_gate_ref else ""
    )
    identity = gate_id(
        run_id=safe_run_id,
        subject_class=subject_class,
        subject_ref=safe_subject_ref,
        blocked_transition=blocked_transition,
        checkpoint_ref=safe_checkpoint,
    )
    record: dict[str, Any] = {
        "schema_version": DECISION_GATE_SCHEMA_VERSION,
        "record_id": _record_id(identity, stamp, GATE_STATE_OPEN),
        "gate_id": identity,
        "subject_id": gate_subject_id(
            run_id=safe_run_id, subject_class=subject_class, subject_ref=safe_subject_ref
        ),
        "run_id": safe_run_id,
        "subject_class": subject_class,
        "subject_ref": safe_subject_ref,
        "blocked_transition": blocked_transition,
        "checkpoint_ref": safe_checkpoint,
        "safety_profile_revision": safe_revision,
        "context_revision": safe_context,
        "question_code": question_code,
        "risk_class": risk_class,
        "choices": offered,
        "approver": safe_approver,
        "opened_at": stamp,
        "state": GATE_STATE_OPEN,
        "answered_choice": "",
        "actor": "",
        "decided_at": "",
        "resume_digest": "",
        "supersedes_gate_ref": safe_supersedes,
        "privacy": GATE_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    record["resume_digest"] = resume_digest_of(record)
    errors = validate_decision_gate(record)
    if errors:
        raise DecisionGateError(errors[0])
    return record


def build_decision_gate_answer(
    gate: Mapping[str, Any],
    *,
    actor: str,
    choice: str,
    decided_at: str = "",
) -> dict[str, Any]:
    """Mint the answer to one open gate, or refuse.

    Everything the answer is bound to is copied from the gate it answers, never
    accepted from the caller. #825 AC3's "same subject, same revision, same
    scope" is therefore not a check that can be skipped: there is no argument
    through which an answer could name a different one.

    Three refusals are #825 AC2 in its enforcing form. An answer must name an
    actor, must name a choice the gate actually offered, and must be stamped;
    and the actor must be the approver the gate named, because an answer from
    somebody the question was not put to is the "unattended or inferred
    approval" #825 puts out of scope.
    """
    if not isinstance(gate, Mapping):
        raise DecisionGateError("decision_gate answer must be built from a gate record")
    if str(gate.get("state", "")) != GATE_STATE_OPEN:
        raise DecisionGateError("decision_gate answer must answer an open gate")
    errors = validate_decision_gate(dict(gate))
    if errors:
        raise DecisionGateError(f"decision_gate answer must answer a valid gate: {errors[0]}")
    offered = [str(item) for item in gate.get("choices", []) if isinstance(item, str)]
    if choice not in offered:
        raise DecisionGateError(f"decision_gate answered_choice was not offered by this gate: {choice!r}")
    safe_actor = _opaque(actor, field="decision_gate actor")
    if safe_actor != str(gate.get("approver", "")):
        raise DecisionGateError(
            "decision_gate actor must be the approver this gate named; an answer from anyone else is not an "
            "answer to this question"
        )
    stamp = _opaque(str(decided_at or utc_now()), field="decision_gate decided_at")
    record = dict(gate)
    record.update(
        {
            "record_id": _record_id(str(gate.get("gate_id", "")), stamp, GATE_STATE_ANSWERED),
            "state": GATE_STATE_ANSWERED,
            "answered_choice": choice,
            "actor": safe_actor,
            "decided_at": stamp,
            "supersedes_gate_ref": str(gate.get("record_id", "")),
        }
    )
    # Re-derived rather than copied. The seal covers only fields the answer
    # inherits verbatim, so this must equal the gate's -- and deriving it again
    # is what proves that, instead of assuming it.
    record["resume_digest"] = resume_digest_of(record)
    answer_errors = validate_decision_gate(record)
    if answer_errors:
        raise DecisionGateError(answer_errors[0])
    return record


def validate_decision_gate(record: dict[str, Any]) -> list[str]:
    """Every reason one shared-store record is not a gate or connector receipt."""
    if isinstance(record, dict) and record.get("schema_version") == "connector_decision_gate_receipt/v1":
        return _connector_receipt_errors(record)
    if isinstance(record, dict) and record.get("schema_version") == "connector_decision_gate_transaction/v1":
        return _connector_transaction_errors(record)
    if not isinstance(record, dict):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    claims_execution = sorted(key for key in record if str(key).lower() in EXECUTION_CLAIM_KEYS)
    if claims_execution:
        errors.append(
            f"{_LABEL} must not carry execution-claim keys: "
            f"{claims_execution}; an answered question is not an attempt and not an outcome"
        )
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    extra_keys = sorted(set(record) - set(DECISION_GATE_KEYS) - set(claims_execution) - set(forbidden))
    if extra_keys:
        errors.append(f"{_LABEL} has unsupported keys: {extra_keys}")
    missing = sorted(set(DECISION_GATE_KEYS) - set(record))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if record.get("schema_version") != DECISION_GATE_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {DECISION_GATE_SCHEMA_VERSION}")
    for key in _GATE_STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"{_LABEL} {key} must be a string")
    for key, vocabulary in _GATE_VOCABULARIES:
        if record.get(key) not in vocabulary:
            errors.append(f"{_LABEL} {key} is unsupported: {record.get(key)!r}")
    if record.get("privacy") != GATE_PRIVACY:
        errors.append(f"{_LABEL} privacy must be metadata_only")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the decision gate boundary")
    for field in _REQUIRED_GATE_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    for field in _OPTIONAL_GATE_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=False))
    errors.extend(_subject_ref_errors(record.get("subject_ref")))
    errors.extend(_choices_errors(record.get("choices")))
    errors.extend(_state_errors(record))
    errors.extend(_identity_errors(record))
    return errors


def append_decision_gate(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    """Append one validated record. Never rewrites, never reorders."""
    errors = validate_decision_gate(record)
    if errors:
        raise DecisionGateError(errors[0])
    path = paths.runtime_decision_gates_path
    with file_lock(path, private=True):
        append_store_line(path, record)
    return record


def open_decision_gate(
    paths: OmhPaths,
    *,
    run_id: str,
    subject_class: str,
    subject_ref: str,
    blocked_transition: str,
    checkpoint_ref: str,
    question_code: str,
    risk_class: str,
    choices: Sequence[str],
    approver: str,
    safety_profile_revision: str,
    context_revision: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Put one question on record, or refuse. #825 AC1's enforcing half.

    The read of the store and the append happen under one lock, so two surfaces
    blocking on one subject at the same moment cannot both believe they opened
    the only gate.

    Re-asking a question that is already open appends nothing and returns the
    record already on file: one block reported three times is one question, not
    three. Asking a *different* question about a subject that already has an open
    gate is refused outright -- that is the state #825 AC1 forbids, and refusing
    it here is why no renderer ever has to choose between two open gates.
    """
    path = paths.runtime_decision_gates_path
    with file_lock(path, private=True):
        records, _ = read_jsonl_objects(path)
        identity = gate_id(
            run_id=str(run_id),
            subject_class=str(subject_class),
            subject_ref=str(subject_ref),
            blocked_transition=str(blocked_transition),
            checkpoint_ref=str(checkpoint_ref),
        )
        subject = gate_subject_id(
            run_id=str(run_id), subject_class=str(subject_class), subject_ref=str(subject_ref)
        )
        for head in open_gates_in(records):
            if str(head.get("gate_id", "")) == identity:
                return dict(head)
            if str(head.get("subject_id", "")) == subject:
                raise DecisionGateError(
                    f"{_LABEL} subject already has an open gate; one subject carries at most one open "
                    "question, so answer or decline the open one before asking another"
                )
        prior = latest_record_in(records, key="gate_id", value=identity)
        record = build_decision_gate(
            run_id=run_id,
            subject_class=subject_class,
            subject_ref=subject_ref,
            blocked_transition=blocked_transition,
            checkpoint_ref=checkpoint_ref,
            question_code=question_code,
            risk_class=risk_class,
            choices=choices,
            approver=approver,
            safety_profile_revision=safety_profile_revision,
            context_revision=context_revision,
            opened_at=now,
            supersedes_gate_ref=str(prior.get("record_id", "")) if prior else "",
        )
        append_store_line(path, record)
    return record


def answer_decision_gate(
    paths: OmhPaths,
    *,
    gate_id_value: str,
    actor: str,
    choice: str,
    now: str = "",
) -> dict[str, Any]:
    """Record one answer to one open question, or refuse. #825 AC2.

    The read of the open record and the append of the answer happen under one
    lock, so two answers to one question cannot both link to the same open
    record and fork the chain -- the second finds the gate already answered and
    is refused.

    Nothing on disk is edited. The answer is a new record linked through
    `supersedes_gate_ref`, so the question as it was asked stays readable beside
    the answer it got.
    """
    path = paths.runtime_decision_gates_path
    with file_lock(path, private=True):
        records, _ = read_jsonl_objects(path)
        head = latest_record_in(records, key="gate_id", value=str(gate_id_value))
        if not head:
            raise DecisionGateError(f"{_LABEL} has no gate on record for this question")
        if str(head.get("state", "")) != GATE_STATE_OPEN:
            raise DecisionGateError(
                f"{_LABEL} is already answered; a recorded answer is never rewritten, so asking again needs "
                "a new gate"
            )
        record = build_decision_gate_answer(head, actor=actor, choice=choice, decided_at=now)
        append_store_line(path, record)
    return record


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_decision_gates_result(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[str]]:
    return read_jsonl_objects(paths.runtime_decision_gates_path)


def read_decision_gates(
    paths: OmhPaths,
    *,
    run_id: str | None = None,
    gate_id_value: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """The store, in append order, optionally scoped to one run or one question."""
    records, _ = read_decision_gates_result(paths)
    records = [record for record in records if record.get("schema_version") == DECISION_GATE_SCHEMA_VERSION]
    if run_id is not None:
        records = [record for record in records if str(record.get("run_id", "")) == run_id]
    if gate_id_value is not None:
        records = [record for record in records if str(record.get("gate_id", "")) == gate_id_value]
    if limit is None:
        return records
    if limit < 1:
        return []
    return records[-limit:]


def latest_gate_in(records: Sequence[Mapping[str, Any]], gate_id_value: str) -> dict[str, Any]:
    """The record that speaks for one question: the one selection rule.

    The store is append-only, so store order is arrival order and the last record
    for a question is its current state. Every surface that judges a gate selects
    through this function, which is what stops the store validator and the resume
    predicate from picking different records for one question and reaching
    opposite conclusions about it.
    """
    return dict(latest_record_in(records, key="gate_id", value=str(gate_id_value)))


def open_gates_in(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every question currently open, in append order.

    Chain heads only: a superseded open record was replaced by whatever came
    after it and is not a question anybody is being asked.
    """
    heads = _chain_heads(records)
    return [
        dict(record)
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("state", "")) == GATE_STATE_OPEN
        and heads.get(str(record.get("gate_id", ""))) == str(record.get("record_id", ""))
    ]


# ---------------------------------------------------------------------------
# #825 AC3: does an answer release this resume?
# ---------------------------------------------------------------------------


def resume_satisfies_gate(
    paths: OmhPaths,
    *,
    run_id: str,
    subject_class: str,
    subject_ref: str,
    blocked_transition: str,
    checkpoint_ref: str,
    safety_profile_revision: str,
    context_revision: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Does an answered, unexpired, matching gate release exactly this transition?

    The entry point a blocked surface calls before continuing. Returns a
    `decision_gate_resume/v1` mapping; see `resume_satisfies_gate_in` for the
    shape and for how the refusal code is chosen.
    """
    return resume_satisfies_gate_in(
        read_decision_gates(paths),
        run_id=run_id,
        subject_class=subject_class,
        subject_ref=subject_ref,
        blocked_transition=blocked_transition,
        checkpoint_ref=checkpoint_ref,
        safety_profile_revision=safety_profile_revision,
        context_revision=context_revision,
        now=now,
    )


def resume_satisfies_gate_in(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    subject_class: str,
    subject_ref: str,
    blocked_transition: str,
    checkpoint_ref: str,
    safety_profile_revision: str,
    context_revision: str = "",
    now: str = "",
) -> dict[str, Any]:
    """`resume_satisfies_gate` over records already in hand.

    A `decision_gate_resume/v1` mapping:

        released              -- True only when a live answer releases exactly
                                 this subject, these revisions, this transition,
                                 and this checkpoint
        reason_code           -- "" when released, otherwise one of
                                 GATE_REFUSAL_CODES
        reason                -- the constant line for that code
        gate_id / record_id   -- the gate this verdict is about, "" when none
        resume_digest         -- the digest this resume was matched on
        answered_choice       -- the choice on record, "" when none
        actor / decided_at    -- who answered and when, "" when nobody has
        age_seconds           -- age of that record, computed now
        expires_after_seconds -- APPROVAL_TTL_SECONDS, so a caller can cite it

    A gate that names this resume exactly is judged on its lifecycle: superseded,
    then pending, then declined, then expired. Otherwise the nearest gate
    explains what it does cover, reporting its first differing dimension in
    `GATE_MISMATCH_CODES` order. There is no containment rule anywhere in that
    walk, so nothing widens.
    """
    request = (
        str(run_id),
        (str(subject_class), str(subject_ref)),
        (str(safety_profile_revision), str(context_revision or "")),
        (str(blocked_transition), str(checkpoint_ref)),
    )
    digest = resume_digest(
        run_id=run_id,
        subject_class=subject_class,
        subject_ref=subject_ref,
        blocked_transition=blocked_transition,
        checkpoint_ref=checkpoint_ref,
        safety_profile_revision=safety_profile_revision,
        context_revision=context_revision,
    )
    known = [record for record in records if isinstance(record, Mapping)]
    exact = [record for record in known if not _dimension_mismatches(record, request)]
    if exact:
        return _lifecycle_verdict(known, exact[-1], digest, now)
    candidate, mismatches = _nearest_gate(known, request)
    if candidate is None:
        return _resume_payload(
            released=False, reason_code=REFUSAL_ABSENT, record={}, digest=digest, now=now
        )
    return _resume_payload(
        released=False, reason_code=mismatches[0], record=candidate, digest=digest, now=now
    )


def gate_age_seconds(record: Mapping[str, Any], now: str = "") -> int:
    """How old the record's own stamp is, computed now, or `UNKNOWN_AGE_SECONDS`.

    An answered record is measured from `decided_at` and an open one from
    `opened_at`: the stamp that made the record what it is. `approval_age_seconds`
    does the arithmetic, so a stamp that does not parse and a stamp in the future
    are refused here exactly as they are there.
    """
    return approval_age_seconds(_decision_stamp(record), now)


def gate_is_expired(record: Mapping[str, Any], now: str = "") -> bool:
    """Whether this record is outside the decision window, evaluated now.

    The window is `approval_receipts.APPROVAL_TTL_SECONDS`, imported rather than
    redeclared. A decision gate authorises the same kind of escalation an
    approval receipt does, so a second, longer window here would be a way to
    outlive the consent rules by asking the question through a different surface.
    """
    age = gate_age_seconds(record, now)
    return age == UNKNOWN_AGE_SECONDS or age > APPROVAL_TTL_SECONDS


# ---------------------------------------------------------------------------
# #825 AC1: what one blocked run is being asked
# ---------------------------------------------------------------------------


def project_blocked_decision_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    now: str = "",
) -> dict[str, Any]:
    """The one question one blocked run is waiting on, and the checkpoint it sits at.

    A projection and not a second store, for the reason `project_decision_history`
    is one: a second store over one decision sequence is a fork by construction,
    and `supersede_chain_errors` already rejects forks.

    `ok` is False when the run holds more than one open gate. That state is
    refused at write and faulted at validate, so reaching it means the store was
    edited or written by an older build -- and a renderer that silently picked
    one of the two would be choosing which question an operator answers.
    """
    scoped = [
        record
        for record in open_gates_in(records)
        if str(record.get("run_id", "")) == str(run_id)
    ]
    gate = scoped[0] if len(scoped) == 1 else {}
    return {
        "schema_version": DECISION_GATE_VIEW_SCHEMA_VERSION,
        "run_id": _redacted(str(run_id)),
        "ok": len(scoped) <= 1,
        "blocked": bool(scoped),
        "open_gate_count": len(scoped),
        "gate": compact_decision_gate(gate, now=now) if gate else {},
        "checkpoint_ref": _redacted(str(gate.get("checkpoint_ref", ""))) if gate else "",
        "blocked_transition": closed_vocabulary_value(gate.get("blocked_transition"), BLOCKED_TRANSITIONS),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def blocked_decision_gate(paths: OmhPaths, *, run_id: str, now: str = "") -> dict[str, Any]:
    """`project_blocked_decision_gate` over the store on disk."""
    records, _ = read_decision_gates_result(paths)
    return project_blocked_decision_gate(records, run_id=run_id, now=now)


# ---------------------------------------------------------------------------
# Validate the store, render
# ---------------------------------------------------------------------------


def validate_decision_gate_store(path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """Validate the gate store, optionally scoped to one run's records.

    Two halves. `store_errors` gives the per-record schema and the shared chain
    walk, told this family's key names so its findings read as gate records
    rather than as receipts it has none of. The second half is #825 AC1's
    enforcing counterpart at rest: two open gates for one subject is a fault of
    the store, and reporting it here is what makes the rule survive a
    hand-edited file rather than only a well-behaved writer.
    """
    records, read_errors = read_jsonl_objects(path)
    gate_count, errors = store_errors(
        path,
        records,
        read_errors,
        run_id=run_id,
        validate_record=validate_decision_gate,
        label=_LABEL,
        id_key="record_id",
        link_key="supersedes_gate_ref",
        noun="gate record",
    )
    open_gates = [
        record
        for record in open_gates_in(records)
        if run_id is None or str(record.get("run_id", "")) == run_id
    ]
    contested = _contested_subjects(open_gates)
    errors.extend(
        f"{path}: {_LABEL} subject {subject} has {count} open gates; one subject carries at most one open question"
        for subject, count in contested
    )
    return {
        "schema_version": DECISION_GATE_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "run_id": run_id or "",
        "ok": not errors,
        "gate_count": gate_count,
        "open_gate_count": len(open_gates),
        "contested_subject_count": len(contested),
        "errors": errors,
    }


def compact_decision_gate(record: Mapping[str, Any], *, now: str = "") -> dict[str, Any]:
    """The user-facing shape, with every field through its class's guard.

    Rendering is the last point at which a hand-edited store reaches a human, so
    nothing is copied through verbatim: identifiers fold to a stable
    non-navigable handle when they are not opaque, closed vocabularies render
    empty outside their vocabulary, and the digest renders only in the one shape
    it may be stored in.

    `question`, `risk`, and each choice's `consequence` are joined here from the
    module's constant tables rather than stored, which is why a gate can be
    rendered at all without a free-text field. `age_seconds` and `expired` are
    derived for the reason `approval_receipts` derives them: nothing on disk says
    a decision is still good.
    """
    question_code = closed_vocabulary_value(record.get("question_code"), QUESTION_CODES)
    risk_class = closed_vocabulary_value(record.get("risk_class"), RISK_CLASSES)
    answered_choice = closed_vocabulary_value(record.get("answered_choice"), GATE_CHOICES)
    return {
        "record_id": _redacted(str(record.get("record_id", ""))),
        "gate_id": _redacted(str(record.get("gate_id", ""))),
        "subject_id": _redacted(str(record.get("subject_id", ""))),
        "run_id": _redacted(str(record.get("run_id", ""))),
        "subject_class": closed_vocabulary_value(record.get("subject_class"), SUBJECT_CLASSES),
        "subject_ref": _redacted(str(record.get("subject_ref", ""))),
        "blocked_transition": closed_vocabulary_value(record.get("blocked_transition"), BLOCKED_TRANSITIONS),
        "checkpoint_ref": _redacted(str(record.get("checkpoint_ref", ""))),
        "safety_profile_revision": _redacted(str(record.get("safety_profile_revision", ""))),
        "context_revision": _redacted(str(record.get("context_revision", ""))),
        "question_code": question_code,
        "question": QUESTION_TEXT.get(question_code, ""),
        "risk_class": risk_class,
        "risk": RISK_TEXT.get(risk_class, ""),
        "choices": [
            {
                "choice": choice,
                "consequence": CHOICE_CONSEQUENCES.get(choice, ""),
                "releases": CHOICE_RELEASES.get(choice, False),
            }
            for choice in _rendered_choices(record.get("choices"))
        ],
        "approver": _redacted(str(record.get("approver", ""))),
        "opened_at": _redacted(str(record.get("opened_at", ""))),
        "state": closed_vocabulary_value(record.get("state"), GATE_STATES),
        "answered_choice": answered_choice,
        "answered_consequence": CHOICE_CONSEQUENCES.get(answered_choice, ""),
        "releases_transition": CHOICE_RELEASES.get(answered_choice, False),
        "actor": _redacted(str(record.get("actor", ""))),
        "decided_at": _redacted(str(record.get("decided_at", ""))),
        "resume_digest": _rendered_digest(record.get("resume_digest")),
        "supersedes_gate_ref": _redacted(str(record.get("supersedes_gate_ref", ""))),
        "age_seconds": gate_age_seconds(record, now),
        "expired": gate_is_expired(record, now),
        "expires_after_seconds": APPROVAL_TTL_SECONDS,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain_heads(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The current record for every question in one pass over the store.

    The same selection rule as `latest_gate_in` -- last record wins -- built once
    so judging a store of n records does not walk it n times.
    """
    heads: dict[str, str] = {}
    for record in records:
        if isinstance(record, Mapping):
            heads[str(record.get("gate_id", ""))] = str(record.get("record_id", ""))
    return heads


def _contested_subjects(open_gates: Sequence[Mapping[str, Any]]) -> list[tuple[str, int]]:
    """Every subject carrying more than one open gate, and how many.

    Sorted by subject so the report is deterministic, and the subject is folded
    through the render guard because the line reaches an operator.
    """
    counts: dict[str, int] = {}
    for record in open_gates:
        subject = str(record.get("subject_id", ""))
        counts[subject] = counts.get(subject, 0) + 1
    return [(_redacted(subject), count) for subject, count in sorted(counts.items()) if count > 1]


def _lifecycle_verdict(
    records: Sequence[Mapping[str, Any]],
    exact: Mapping[str, Any],
    digest: str,
    now: str,
) -> dict[str, Any]:
    """Judge a gate that names this resume exactly.

    Order matters, and the first two checks are worth separating honestly.

    Superseded goes first and, in a well-formed store, never fires: every record
    matching these dimensions carries the same `gate_id` -- the id is derived
    from four of them -- so the last dimension match is also the chain head, and
    an answer that was replaced is simply never the record judged. The check is
    here for the store this family cannot assume it is reading. Edit the head
    record's subject on disk and it stops matching, which would otherwise leave
    an older, replaced answer speaking for a question whose current record says
    something else. `tests/test_decision_gate.py` reaches it that way and no
    other.

    Then pending, because an unanswered question is the state a caller most needs
    told apart from a refusal. Then the choice, then the window.

    The digest is checked last and separately. Every dimension it seals has
    already been compared, so a mismatch here can only mean the stored digest was
    edited to disagree with the record carrying it -- which is a resume for a
    transition nobody was asked about, and is refused as one.
    """
    head = latest_gate_in(records, str(exact.get("gate_id", "")))
    if str(head.get("record_id", "")) != str(exact.get("record_id", "")):
        return _resume_payload(
            released=False, reason_code=REFUSAL_SUPERSEDED, record=exact, digest=digest, now=now
        )
    if str(exact.get("state", "")) != GATE_STATE_ANSWERED:
        return _resume_payload(
            released=False, reason_code=REFUSAL_PENDING, record=exact, digest=digest, now=now
        )
    if not CHOICE_RELEASES.get(str(exact.get("answered_choice", "")), False):
        return _resume_payload(
            released=False, reason_code=REFUSAL_DECLINED, record=exact, digest=digest, now=now
        )
    if gate_is_expired(exact, now):
        return _resume_payload(
            released=False, reason_code=REFUSAL_EXPIRED, record=exact, digest=digest, now=now
        )
    if str(exact.get("resume_digest", "")) != digest:
        return _resume_payload(
            released=False, reason_code=REFUSAL_TRANSITION, record=exact, digest=digest, now=now
        )
    return _resume_payload(released=True, reason_code="", record=exact, digest=digest, now=now)


def _nearest_gate(
    records: Sequence[Mapping[str, Any]],
    request: tuple[Any, ...],
) -> tuple[Mapping[str, Any] | None, tuple[str, ...]]:
    """The gate closest to this resume, and how it differs.

    Only chain heads are considered: a superseded record explains nothing about
    what is being asked now. An answered head that differs stays a candidate on
    purpose -- "there is an answer, for another transition" is a more useful
    refusal than "there is no gate", and it refuses either way.
    """
    heads = _chain_heads(records)
    best: Mapping[str, Any] | None = None
    best_mismatches: tuple[str, ...] = ()
    for record in records:
        if heads.get(str(record.get("gate_id", ""))) != str(record.get("record_id", "")):
            continue
        mismatches = _dimension_mismatches(record, request)
        if not mismatches:
            continue
        if best is None or len(mismatches) <= len(best_mismatches):
            best = record
            best_mismatches = mismatches
    return best, best_mismatches


def _dimension_mismatches(record: Mapping[str, Any], request: tuple[Any, ...]) -> tuple[str, ...]:
    """Which of the four bound dimensions this record does not match.

    Append order is `GATE_MISMATCH_CODES` order, which is the reporting
    precedence. Every comparison is equality, including the pairs: there is no
    containment rule here, and that absence is what makes widening structurally
    impossible rather than merely unimplemented.
    """
    run_id, subject, revisions, position = request
    mismatches: list[str] = []
    if str(record.get("run_id", "")) != run_id:
        mismatches.append(REFUSAL_RUN)
    if (str(record.get("subject_class", "")), str(record.get("subject_ref", ""))) != subject:
        mismatches.append(REFUSAL_SCOPE)
    if (
        str(record.get("safety_profile_revision", "")),
        str(record.get("context_revision", "")),
    ) != revisions:
        mismatches.append(REFUSAL_REVISION)
    if (str(record.get("blocked_transition", "")), str(record.get("checkpoint_ref", ""))) != position:
        mismatches.append(REFUSAL_TRANSITION)
    return tuple(mismatches)


def _resume_payload(
    *,
    released: bool,
    reason_code: str,
    record: Mapping[str, Any],
    digest: str,
    now: str,
) -> dict[str, Any]:
    return {
        "schema_version": DECISION_GATE_RESUME_SCHEMA_VERSION,
        "released": released,
        "reason_code": reason_code,
        "reason": GATE_REFUSAL_REASONS.get(reason_code, ""),
        "gate_id": _redacted(str(record.get("gate_id", ""))),
        "record_id": _redacted(str(record.get("record_id", ""))),
        "resume_digest": _rendered_digest(digest),
        "state": closed_vocabulary_value(record.get("state"), GATE_STATES),
        "answered_choice": closed_vocabulary_value(record.get("answered_choice"), GATE_CHOICES),
        "actor": _redacted(str(record.get("actor", ""))),
        "decided_at": _redacted(str(record.get("decided_at", ""))),
        "age_seconds": gate_age_seconds(record, now) if record else UNKNOWN_AGE_SECONDS,
        "expires_after_seconds": APPROVAL_TTL_SECONDS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _state_errors(record: Mapping[str, Any]) -> list[str]:
    """The open/answered split, checked in both directions.

    #825 AC2 lives here at rest. An answered record missing an actor, a choice,
    or a stamp is refused, and so is one whose choice was never offered or whose
    actor is not the approver the question was put to. The mirror is just as
    load-bearing: an open record carrying any of those three would be an answer
    nobody gave.
    """
    errors: list[str] = []
    state = str(record.get("state", ""))
    if state == GATE_STATE_OPEN:
        present = sorted(field for field in _ANSWER_FIELDS if str(record.get(field, "")))
        if present:
            errors.append(f"{_LABEL} an open gate must carry no answer; it carries {present}")
        errors.extend(_self_link_errors(record))
        return errors
    if state != GATE_STATE_ANSWERED:
        return errors
    absent = sorted(field for field in _ANSWER_FIELDS if not str(record.get(field, "")))
    if absent:
        errors.append(f"{_LABEL} an answered gate must record {absent}")
    choice = str(record.get("answered_choice", ""))
    if choice and choice not in GATE_CHOICES:
        errors.append(f"{_LABEL} answered_choice is unsupported: {choice!r}")
    offered = record.get("choices")
    if choice and isinstance(offered, list) and choice not in offered:
        errors.append(f"{_LABEL} answered_choice was not offered by this gate: {choice!r}")
    actor = str(record.get("actor", ""))
    if actor and actor != str(record.get("approver", "")):
        errors.append(
            f"{_LABEL} actor must be the approver this gate named; an answer from anyone else is not an "
            "answer to this question"
        )
    if not str(record.get("supersedes_gate_ref", "")):
        errors.append(f"{_LABEL} an answered gate must supersede the open record it answers")
    errors.extend(_self_link_errors(record))
    return errors


def _self_link_errors(record: Mapping[str, Any]) -> list[str]:
    """A record cannot replace itself, whatever state it is in."""
    record_id = str(record.get("record_id", ""))
    if record_id and record_id == str(record.get("supersedes_gate_ref", "")):
        return [f"{_LABEL} supersedes_gate_ref must not name itself"]
    return []


def _identity_errors(record: Mapping[str, Any]) -> list[str]:
    """The three derived identifiers, re-derived.

    Without these a hand-edited store could move one question's answer onto
    another question's chain, count one subject's open gate against a different
    subject, or -- the one that matters most -- point an answer's resume digest
    at a transition nobody was asked about.
    """
    errors: list[str] = []
    run_id = str(record.get("run_id", ""))
    subject_class = str(record.get("subject_class", ""))
    subject_ref = str(record.get("subject_ref", ""))
    if str(record.get("gate_id", "")) != gate_id(
        run_id=run_id,
        subject_class=subject_class,
        subject_ref=subject_ref,
        blocked_transition=str(record.get("blocked_transition", "")),
        checkpoint_ref=str(record.get("checkpoint_ref", "")),
    ):
        errors.append(f"{_LABEL} gate_id does not identify its own run, subject, transition, and checkpoint")
    if str(record.get("subject_id", "")) != gate_subject_id(
        run_id=run_id, subject_class=subject_class, subject_ref=subject_ref
    ):
        errors.append(f"{_LABEL} subject_id does not identify its own run and subject")
    digest = record.get("resume_digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        errors.append(f"{_LABEL} resume_digest must be a sha256 hex digest")
    elif digest != resume_digest_of(record):
        errors.append(f"{_LABEL} resume_digest does not name the transition this gate blocks")
    return errors


def _choices_errors(value: Any) -> list[str]:
    """A sorted, duplicate-free set of offered choices that is a real question.

    The last rule is the one worth stating: a gate whose every choice releases
    the transition is not a decision, it is a formality, and a gate with no
    releasing choice can never be resumed and so should never have been asked.
    Both are refused, which is what makes "choices with consequences" a property
    of the record rather than of the wording.
    """
    if not isinstance(value, list):
        return [f"{_LABEL} choices must be a list"]
    errors: list[str] = []
    for index, choice in enumerate(value):
        if not isinstance(choice, str):
            errors.append(f"{_LABEL} choices[{index}] must be a string")
        elif choice not in GATE_CHOICES:
            errors.append(f"{_LABEL} choices[{index}] is unsupported: {choice!r}")
    if errors:
        return errors
    if list(value) != sorted(set(value)):
        errors.append(f"{_LABEL} choices must be sorted and free of duplicates")
        return errors
    if not any(CHOICE_RELEASES.get(choice, False) for choice in value):
        errors.append(f"{_LABEL} choices must offer at least one choice that releases the transition")
    if all(CHOICE_RELEASES.get(choice, False) for choice in value):
        errors.append(f"{_LABEL} choices must offer at least one choice that does not release the transition")
    return errors


def _normalized_choices(values: Sequence[str]) -> list[str]:
    choices = sorted({str(value).strip() for value in values if str(value).strip()})
    errors = _choices_errors(choices)
    if errors:
        raise DecisionGateError(errors[0])
    return choices


def _subject_ref_errors(value: Any) -> list[str]:
    errors = reference_errors(value, field="subject_ref", label=_LABEL, required=True)
    if errors or not isinstance(value, str):
        return errors
    if _SUBJECT_TRAVERSAL.search(value):
        return [f"{_LABEL} subject_ref must not escape the workspace"]
    if _WINDOWS_DRIVE.match(value):
        return [f"{_LABEL} subject_ref must be workspace-relative, not drive-anchored"]
    return []


def _subject_ref(value: str) -> str:
    text = _opaque(value, field="decision_gate subject_ref")
    errors = _subject_ref_errors(text)
    if errors:
        raise DecisionGateError(errors[0])
    return text


def _decision_stamp(record: Mapping[str, Any]) -> str:
    """The stamp a record's freshness is measured from."""
    return str(record.get("decided_at", "") or record.get("opened_at", ""))


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=DecisionGateError)


def _redacted(value: str) -> str:
    """Fold any handle into a bounded, non-navigable gate reference."""
    return redacted_ref(value, field="decision_gate_ref")


def _rendered_digest(value: Any) -> str:
    """A digest renders only in the one shape it may be stored in."""
    text = str(value or "")
    return text if _DIGEST.fullmatch(text) else ""


def _rendered_choices(value: Any) -> list[str]:
    """A stored choice list renders only when it is a list of known choices.

    A hand-edited store can carry a string where the list should be, and
    iterating that would render one row per character.
    """
    if not isinstance(value, list):
        return []
    return [choice for choice in value if isinstance(choice, str) and choice in GATE_CHOICES]


def _connector_receipt_errors(record: dict[str, Any]) -> list[str]:
    """Validate the extra immutable record type stored beside v1 gate records."""
    keys = {
        "actor", "authentication_method", "channel_ref", "choice", "claim_boundary", "connector", "event_id",
        "expected_approver", "expected_resume_digest", "gate_answer_record_ref", "gate_id", "interaction_ref",
        "observed_at", "privacy", "receipt_id", "record_id", "schema_version", "supersedes_gate_ref", "thread_ref",
        "wrapper_expected_revision", "wrapper_session_ref", "payload_digest",
    }
    required = keys - {"payload_digest"}
    errors: list[str] = []
    extra = sorted(set(record) - keys)
    if extra:
        errors.append(f"connector decision receipt has unsupported keys: {extra}")
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"connector decision receipt is missing keys: {missing}")
    hidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if hidden:
        errors.append(f"connector decision receipt must not carry raw or hidden keys: {hidden}")
    if record.get("privacy") != "metadata_only":
        errors.append("connector decision receipt privacy must be metadata_only")
    if record.get("claim_boundary") != (
        "A connector decision-gate receipt records authenticated connector metadata bound to one gate answer. "
        "It is not connector credential proof, dispatch, execution, verification, review, CI, merge-readiness, or merge evidence."
    ):
        errors.append("connector decision receipt claim_boundary is invalid")
    if record.get("choice") not in GATE_CHOICES:
        errors.append("connector decision receipt choice is invalid")
    if record.get("authentication_method") not in {"host_authenticated", "signed_connector_event"}:
        errors.append("connector decision receipt authentication_method is invalid")
    if record.get("actor") != record.get("expected_approver"):
        errors.append("connector decision receipt actor must match expected_approver")
    if not isinstance(record.get("wrapper_expected_revision"), int) or isinstance(record.get("wrapper_expected_revision"), bool) or record.get("wrapper_expected_revision", -1) < 0:
        errors.append("connector decision receipt wrapper_expected_revision must be a non-negative integer")
    for key in required - {"claim_boundary", "privacy", "schema_version", "wrapper_expected_revision"}:
        errors.extend(reference_errors(record.get(key), field=key, label="connector decision receipt", required=True))
    try:
        strict_fromisoformat(str(record.get("observed_at", "")).replace("Z", "+00:00"))
    except ValueError:
        errors.append("connector decision receipt observed_at must be an ISO-8601 timestamp")
    return errors


def _connector_transaction_errors(record: dict[str, Any]) -> list[str]:
    """Validate the durable event-to-answer binding written before an answer."""
    keys = {
        "schema_version", "record_id", "transaction_id", "event_id", "payload_digest", "gate_id",
        "open_gate_record_ref", "decided_at", "privacy", "claim_boundary",
    }
    errors: list[str] = []
    extra = sorted(set(record) - keys)
    missing = sorted(keys - set(record))
    if extra:
        errors.append(f"connector decision transaction has unsupported keys: {extra}")
    if missing:
        errors.append(f"connector decision transaction is missing keys: {missing}")
    if record.get("privacy") != "metadata_only":
        errors.append("connector decision transaction privacy must be metadata_only")
    if record.get("claim_boundary") != (
        "A connector decision-gate receipt records authenticated connector metadata bound to one gate answer. "
        "It is not connector credential proof, dispatch, execution, verification, review, CI, merge-readiness, or merge evidence."
    ):
        errors.append("connector decision transaction claim_boundary is invalid")
    for key in keys - {"schema_version", "privacy", "claim_boundary"}:
        errors.extend(reference_errors(record.get(key), field=key, label="connector decision transaction", required=True))
    if not isinstance(record.get("payload_digest"), str) or not _DIGEST.fullmatch(str(record.get("payload_digest", ""))):
        errors.append("connector decision transaction payload_digest must be a sha256 hex digest")
    try:
        strict_fromisoformat(str(record.get("decided_at", "")).replace("Z", "+00:00"))
    except ValueError:
        errors.append("connector decision transaction decided_at must be an ISO-8601 timestamp")
    return errors


def _record_id(gate_id_value: str, stamp: str, state: str) -> str:
    return mint_record_id(
        prefix="decision-gate",
        identity={"gate_id": gate_id_value, "stamp": stamp, "state": state},
    )
