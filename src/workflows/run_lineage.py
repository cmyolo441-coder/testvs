"""Append-only, hash-chained run lineage checkpoints (issue #826).

What was missing
----------------

A run's history was already recorded, but in pieces that only agree by
convention. `runtime/records.py` builds the run record and its per-transition
records; `workflows/observation_journal.py` records the ordered lifecycle
events; the plan, the handoff, the workspace binding, and the approval each
live in their own artifact. Asking "what state did this run continue from" meant
joining five files by `run_id` and trusting that none of them had been edited
since.

Nothing in the tree made that trust checkable. There was no hash-chained store
anywhere: no record carried its predecessor's digest, so an edit to a run's
early history left every later artifact reading exactly as it did before.

A lineage checkpoint is the missing spine. Each one names its predecessor by
that predecessor's own digest, so history is append-only in a way that a reader
can *verify* rather than assume: change any field of any earlier checkpoint and
its digest stops matching itself, and the next checkpoint's `parent_digest`
stops naming it.

What the chain seals, and what it does not
------------------------------------------

The digest covers every field whose value carries meaning -- the transition, the
evidence class, the owner, the safety and context revisions, the plan and
handoff digests, the artifacts, the state, the sequence, and the parent link.
Three fields are deliberately outside it:

* `recorded_at`, because a wall clock inside a hashed artifact turns an equality
  check into a race, and this repo has already lost Windows CI time to exactly
  that. The honest consequence is stated rather than hidden: the chain seals
  *what* a run continued from, not *when* each step was written down, so a stamp
  can be edited without breaking a link. Nothing reads freshness off this store.
* `record_id`, which carries a random tail so two checkpoints minted in the same
  second are distinguishable. Sealing it would make the digest unreproducible.
* `digest` itself.

Two checkpoints can therefore never collide by accident despite the clock being
out: within one run the `sequence` differs, and across runs the `run_id` and
`lineage_id` do.

Ordering is a property of the chain, not of the appender
--------------------------------------------------------

`_REQUIRED_PREDECESSORS` mirrors the journal's lifecycle prerequisites and adds
the one the journal never had: a dispatch checkpoint requires an observed start
earlier in the same chain. So a chain cannot be built that reaches dispatch
without passing through `runtime_start_observed`.

That is genuinely new, and it is genuinely narrow. It orders a chain; it does
not make a chain mandatory. No claim rung, journal prerequisite, or delegation
surface asks a run for one, so a run that records no lineage at all can still
claim dispatch and execution with no observed start. The `start_evidence`
boundary of `handoff_safety_contract/v1` therefore stays `declared_not_enforced`
and says exactly this in its own words.

A checkpoint is not evidence
----------------------------

`evidence_class` is derived from the transition and re-derived at validate, so a
hand-edited `plan_accepted` checkpoint that claims `merge_observed` is refused
rather than believed. `state` is cross-checked against it in both directions, so
a `prepared` checkpoint cannot carry an observed class and an `observed` one
cannot carry the prepared class. `checkpoint_implies_evidence_class` answers for
the checkpoint's own class and nothing after it, which is #826 AC3: preparing a
run says only that it was prepared.

Even for its own class, a checkpoint states rather than proves. It is a copy of
what the observation journal already held at that transition, so it can point at
the observation standing behind a class and can never be that observation.
`CLAIM_BOUNDARY` says so on every record, and `validate_run_lineage_checkpoint`
refuses a record shaped to assert execution by key name as well as by the closed
key set.

Store mechanics
---------------

All of it is `system.append_only_store`, shared with the four families beside
it: the torn-tail-safe binary append under a real OS lock, the opaque-reference
guards, the digest handles, and the chain walk. The walk is
`supersede_chain_errors` with this family's own key names -- a hash chain and a
supersede chain reject the same four shapes (a duplicate id, a self-link, a link
to a record that does not already exist, and two records claiming one
predecessor), so the mechanics are the same walk and only the vocabulary
differs. This module adds no store mechanics of its own.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    append_store_line,
    closed_vocabulary_value,
    mint_record_id,
    opaque_ref,
    record_fingerprint,
    redacted_ref,
    reference_errors,
    store_errors,
)
from ..system.local_store import file_lock, read_jsonl_objects, utc_now
from ..system.paths import OmhPaths


RUN_LINEAGE_CHECKPOINT_SCHEMA_VERSION: Final[str] = "run_lineage_checkpoint/v1"
RUN_LINEAGE_RECONSTRUCTION_SCHEMA_VERSION: Final[str] = "run_lineage_reconstruction/v1"
RUN_LINEAGE_STORE_VALIDATION_SCHEMA_VERSION: Final[str] = "run_lineage_store_validation/v1"

RUN_LINEAGE_CHECKPOINT_STORE_NAME: Final[str] = "run_lineage_checkpoints.jsonl"

CHECKPOINT_PRIVACY: Final[str] = "metadata_only"

# Stated on every record and echoed on every reconstruction. A checkpoint is the
# fifth thing this tree keeps apart from the four #800 names: it is a *position
# in history*, not an existence, an ownership, a dispatch, or a result.
CLAIM_BOUNDARY: Final[str] = (
    "A run lineage checkpoint is one hash-chained position in one run's transition history: what the "
    "run continued from, under one owner, one safety profile revision, and one context revision. It "
    "states which evidence class the run had reached when the checkpoint was written. It is not that "
    "evidence: a checkpoint is never dispatch, execution, verification, review, CI, merge-readiness, or "
    "merge evidence, and a prepared checkpoint states no observed class at all."
)

# The transitions a chain can record, in the order a run passes through them.
# The names are the observation journal's own canonical event names rather than
# a second vocabulary for the same milestones -- a lineage chain that called
# dispatch something else could not be read against the journal it mirrors.
LINEAGE_TRANSITIONS: Final[tuple[str, ...]] = (
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

# The evidence-class vocabulary, which is `observation_journal.PROJECTION_ORDER`
# restated here rather than imported for the reason the sibling families mirror
# each other's key sets: a store's on-disk vocabulary must not change because
# another module reordered a tuple. `tests/test_run_lineage_checkpoint.py` pins
# the two equal, so drift is a failing test rather than a silent schema change.
LINEAGE_EVIDENCE_CLASSES: Final[tuple[str, ...]] = (
    "prepared_not_observed",
    "runtime_start_observed",
    "worktree_creation_observed",
    "dispatch_observed",
    "execution_observed",
    "verification_observed",
    "review_observed",
    "ci_observed",
    "merge_gate_observed",
    "merge_observed",
)

# Which class each transition records. Total over `LINEAGE_TRANSITIONS` and onto
# every member of `LINEAGE_EVIDENCE_CLASSES`, both of which are pinned: a
# transition with no class could store anything, and a class no transition can
# produce is a vocabulary member nothing backs -- the exact defect
# `observation_journal` documents for its own `cancelled` status.
#
# The class is derived here and never accepted from a caller. That is what makes
# #826 AC3 structural: there is no argument through which a `plan_accepted`
# checkpoint could be given `merge_observed`.
TRANSITION_EVIDENCE_CLASS: Final[dict[str, str]] = {
    "plan_accepted": "prepared_not_observed",
    "handoff_prepared": "prepared_not_observed",
    "runtime_start_observed": "runtime_start_observed",
    "worktree_creation_observed": "worktree_creation_observed",
    "executor_dispatch_observed": "dispatch_observed",
    "executor_result_observed": "execution_observed",
    "verification_result_observed": "verification_observed",
    "review_result_observed": "review_observed",
    "ci_result_observed": "ci_observed",
    "merge_gate_observed": "merge_gate_observed",
    "merge_observed": "merge_observed",
}

# How the run stood when the checkpoint was written. Separate from the evidence
# class because the two answer different questions: the class says how far the
# run got, the state says how it stands there. A run can be blocked at the
# prepared stage and failed after verification.
LINEAGE_STATES: Final[tuple[str, ...]] = ("prepared", "observed", "blocked", "cancelled", "failed")

# Terminal states say nothing about how far the run got, so they are free to
# appear on any transition. The two positional states are not: they are the
# prepared/observed split itself, and pinning them against the class is the
# second structural expression of AC3.
_PREPARED_STATE: Final[str] = "prepared"
_OBSERVED_STATE: Final[str] = "observed"
_PREPARED_EVIDENCE_CLASS: Final[str] = "prepared_not_observed"

# What must already be in the chain before a transition may join it. The first
# four rows mirror `observation_journal._validate_observation_event_prerequisites`
# so the chain and the journal cannot disagree about lifecycle order.
#
# `executor_dispatch_observed -> runtime_start_observed` is the row the journal
# does not have, and it is the whole of what #826 adds to start ordering: a
# chain cannot reach dispatch without passing through an observed start. It
# orders chains; it does not make a chain mandatory, which is why the
# `start_evidence` safety boundary stays declared.
#
# Review and CI require verification only. The journal makes them conditionally
# required of each other from the delegation's review policy, which is a
# per-run fact this store does not read -- requiring it here would refuse a
# legitimate chain on evidence that lives somewhere else.
_REQUIRED_PREDECESSORS: Final[dict[str, tuple[str, ...]]] = {
    "runtime_start_observed": ("handoff_prepared",),
    "worktree_creation_observed": ("runtime_start_observed",),
    "executor_dispatch_observed": ("runtime_start_observed",),
    "executor_result_observed": ("executor_dispatch_observed",),
    "verification_result_observed": ("executor_result_observed",),
    "review_result_observed": ("verification_result_observed",),
    "ci_result_observed": ("verification_result_observed",),
    "merge_gate_observed": ("verification_result_observed",),
    "merge_observed": ("merge_gate_observed",),
}

RUN_LINEAGE_CHECKPOINT_KEYS: Final[tuple[str, ...]] = (
    "artifact_refs",
    "claim_boundary",
    "context_revision",
    "digest",
    "evidence_class",
    "handoff_digest",
    "lineage_id",
    "owner",
    "parent_digest",
    "plan_digest",
    "privacy",
    "record_id",
    "recorded_at",
    "run_id",
    "safety_profile_revision",
    "schema_version",
    "sequence",
    "state",
    "transition",
)

# The three fields outside the seal, for the reasons the module docstring gives.
# Kept as a tuple so `CHECKPOINT_DIGEST_KEYS` is derived rather than restated: a
# key added to the record and forgotten here would silently leave a meaningful
# field unsealed, which is the one defect this family cannot afford.
UNSEALED_CHECKPOINT_KEYS: Final[tuple[str, ...]] = ("digest", "record_id", "recorded_at")

CHECKPOINT_DIGEST_KEYS: Final[tuple[str, ...]] = tuple(
    key for key in RUN_LINEAGE_CHECKPOINT_KEYS if key not in UNSEALED_CHECKPOINT_KEYS
)

# The two keys whose value is a module constant rather than data.
RUN_LINEAGE_CONSTANT_KEYS: Final[tuple[str, ...]] = ("claim_boundary", "privacy")

_REQUIRED_CHECKPOINT_REFS: Final[tuple[str, ...]] = (
    "lineage_id",
    "owner",
    "record_id",
    "recorded_at",
    "run_id",
    "safety_profile_revision",
)
# Legitimately absent. A chain that starts at `handoff_prepared` has no accepted
# plan artifact to name, and a run whose context was never revised has no
# context revision. What each one *is* required for is `_missing_digest_errors`.
_OPTIONAL_CHECKPOINT_REFS: Final[tuple[str, ...]] = ("context_revision", "handoff_digest", "plan_digest")

# Every string field. `sequence` is an integer and `artifact_refs` is a list;
# everything else on this record is a bounded string, and there is deliberately
# no free text -- a note field would be one more place a prompt could arrive.
_CHECKPOINT_STRING_FIELDS: Final[tuple[str, ...]] = tuple(
    key for key in RUN_LINEAGE_CHECKPOINT_KEYS if key not in ("artifact_refs", "sequence")
)

# More than a dozen artifacts on one checkpoint is a directory listing rather
# than the bounded set of things one transition produced.
ARTIFACT_REF_LIMIT: Final[int] = 12

# Key names an execution claim arrives under. The closed key set already refuses
# every one of them, but it refuses them as "unsupported keys", and the point of
# this family is that a position in history is not an outcome -- so the shape is
# named and refused by name first. Mirrors
# `workspace_bindings.EXECUTION_CLAIM_KEYS` rather than importing across two
# unrelated record families; `tests/test_run_lineage_checkpoint.py` pins them
# equal so the two cannot drift.
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

# Why a reconstruction stopped. Closed ids rather than sentences, for the reason
# `workspace_bindings.RECOVERY_ACTIONS` is: localized copy is then a lookup per
# locale instead of a translated constant.
BREAK_INVALID_RECORD: Final[str] = "checkpoint_is_not_a_valid_record"
BREAK_DIGEST_MISMATCH: Final[str] = "digest_does_not_match_the_checkpoint"
BREAK_PARENT_MISMATCH: Final[str] = "parent_digest_does_not_name_the_previous_checkpoint"
BREAK_SEQUENCE: Final[str] = "sequence_does_not_match_the_chain_position"
BREAK_PREREQUISITE: Final[str] = "transition_is_missing_a_required_predecessor"
# There is deliberately no code for "this checkpoint belongs to another run".
# `validate_run_lineage_checkpoint` re-derives `lineage_id` from the record's
# own `run_id`, and a chain is selected by that same `run_id`, so a grafted
# record fails its own schema before the walk can reach it and is reported as
# `BREAK_INVALID_RECORD` with the reason named in words. A second code for it
# would be a branch nothing can enter.
BREAK_CODES: Final[tuple[str, ...]] = (
    BREAK_DIGEST_MISMATCH,
    BREAK_INVALID_RECORD,
    BREAK_PARENT_MISMATCH,
    BREAK_PREREQUISITE,
    BREAK_SEQUENCE,
)

# Constant strings, keyed by code. Nothing is interpolated into a reason: a
# break is rendered to operators, and a line built from stored values would be
# one more path for hand-edited text to reach a terminal.
BREAK_REASONS: Final[dict[str, str]] = {
    BREAK_INVALID_RECORD: (
        "This checkpoint is not a valid run_lineage_checkpoint/v1 record, so nothing after it in the "
        "chain can be trusted to continue from it."
    ),
    BREAK_DIGEST_MISMATCH: (
        "This checkpoint's digest does not match the checkpoint itself, so one of the fields the digest "
        "seals was changed after it was written."
    ),
    BREAK_PARENT_MISMATCH: (
        "This checkpoint's parent_digest does not name the checkpoint immediately before it, so the "
        "chain does not continue from the history it claims to continue from."
    ),
    BREAK_SEQUENCE: (
        "This checkpoint's sequence is not its position in the chain, so a checkpoint was inserted, "
        "removed, or reordered."
    ),
    BREAK_PREREQUISITE: (
        "This checkpoint's transition requires an earlier transition that the chain does not carry, so "
        "the run reached this state without passing through the state before it."
    ),
}

# The store label every error line and every reference fault is reported under.
_LABEL: Final[str] = "run_lineage_checkpoint"

# A digest is a full sha256 hex string and nothing else. Enforced as a shape
# rather than only by re-derivation, so a hand-edited store carrying a path, a
# URL, or a truncated handle in a digest field is refused before anything tries
# to walk it.
_DIGEST: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

# No parent. The genesis checkpoint of a chain, and the only record allowed to
# carry it.
GENESIS_PARENT_DIGEST: Final[str] = ""

_UNKNOWN_SEQUENCE: Final[int] = -1


class RunLineageError(ValueError):
    """Raised when a transition cannot become a run lineage checkpoint."""


# ---------------------------------------------------------------------------
# Identity and digest
# ---------------------------------------------------------------------------


def lineage_id(run_id: str) -> str:
    """Stable identity of one run's chain.

    Derived from the run and nothing else, so a checkpoint found in the store
    can be checked against the run it claims -- which is what stops a
    hand-edited record from moving one run's history onto another run's chain.
    """
    base = json.dumps({"run_id": str(run_id)}, sort_keys=True)
    return f"lineage-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


def checkpoint_digest(record: Mapping[str, Any]) -> str:
    """This checkpoint's own digest, over every field the seal covers.

    `record_fingerprint` is the shared hasher; nothing here reimplements it. The
    key tuple is what this family contributes, and `CHECKPOINT_DIGEST_KEYS` is
    derived from the record's own key set so a new field is sealed by default.
    """
    return record_fingerprint(record, CHECKPOINT_DIGEST_KEYS)


# ---------------------------------------------------------------------------
# Build, validate, append
# ---------------------------------------------------------------------------


def build_run_lineage_checkpoint(
    *,
    run_id: str,
    transition: str,
    owner: str,
    safety_profile_revision: str,
    sequence: int,
    parent_digest: str = GENESIS_PARENT_DIGEST,
    plan_digest: str = "",
    handoff_digest: str = "",
    context_revision: str = "",
    artifact_refs: Sequence[str] = (),
    state: str = "",
    recorded_at: str = "",
) -> dict[str, Any]:
    """Mint one checkpoint, or refuse.

    `sequence` and `parent_digest` are the caller's, not this function's: a
    checkpoint's position and its predecessor are properties of the chain it is
    joining, and a builder that guessed them from an unread store would be the
    one way a chain could fork without anyone noticing.
    `append_run_lineage_checkpoint` reads both under the store's lock.

    `recorded_at` is a parameter and is outside the seal, so two callers minting
    the same checkpoint a second apart produce the same digest.
    """
    if transition not in TRANSITION_EVIDENCE_CLASS:
        raise RunLineageError(f"run_lineage_checkpoint transition is unsupported: {transition!r}")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise RunLineageError(f"run_lineage_checkpoint sequence must be a non-negative integer: {sequence!r}")
    evidence_class = TRANSITION_EVIDENCE_CLASS[transition]
    resolved_state = str(state or _default_state(evidence_class))
    if resolved_state not in LINEAGE_STATES:
        raise RunLineageError(f"run_lineage_checkpoint state is unsupported: {resolved_state!r}")
    safe_run_id = _opaque(run_id, field="run_lineage_checkpoint run_id")
    safe_owner = _opaque(owner, field="run_lineage_checkpoint owner")
    safe_profile = _opaque(safety_profile_revision, field="run_lineage_checkpoint safety_profile_revision")
    safe_plan = _optional_opaque(plan_digest, field="run_lineage_checkpoint plan_digest")
    safe_handoff = _optional_opaque(handoff_digest, field="run_lineage_checkpoint handoff_digest")
    safe_context = _optional_opaque(context_revision, field="run_lineage_checkpoint context_revision")
    stamp = _opaque(str(recorded_at or utc_now()), field="run_lineage_checkpoint recorded_at")
    parent = str(parent_digest or GENESIS_PARENT_DIGEST)
    identity = lineage_id(safe_run_id)
    record: dict[str, Any] = {
        "schema_version": RUN_LINEAGE_CHECKPOINT_SCHEMA_VERSION,
        "record_id": mint_record_id(
            prefix="run-lineage",
            identity={"lineage_id": identity, "sequence": str(sequence), "transition": transition},
        ),
        "lineage_id": identity,
        "run_id": safe_run_id,
        "sequence": sequence,
        "parent_digest": parent,
        "transition": transition,
        "evidence_class": evidence_class,
        "state": resolved_state,
        "owner": safe_owner,
        "safety_profile_revision": safe_profile,
        "context_revision": safe_context,
        "plan_digest": safe_plan,
        "handoff_digest": safe_handoff,
        "artifact_refs": _normalized_artifact_refs(artifact_refs),
        "recorded_at": stamp,
        "privacy": CHECKPOINT_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
        "digest": "",
    }
    record["digest"] = checkpoint_digest(record)
    errors = validate_run_lineage_checkpoint(record)
    if errors:
        raise RunLineageError(errors[0])
    return record


def validate_run_lineage_checkpoint(record: dict[str, Any]) -> list[str]:
    """Every reason one record is not a run lineage checkpoint."""
    if not isinstance(record, dict):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    claims_execution = sorted(key for key in record if str(key).lower() in EXECUTION_CLAIM_KEYS)
    if claims_execution:
        errors.append(
            f"{_LABEL} must not carry execution-claim keys: "
            f"{claims_execution}; a position in history is not a dispatch and not an outcome"
        )
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    extra_keys = sorted(
        set(record) - set(RUN_LINEAGE_CHECKPOINT_KEYS) - set(claims_execution) - set(forbidden)
    )
    if extra_keys:
        errors.append(f"{_LABEL} has unsupported keys: {extra_keys}")
    missing = sorted(set(RUN_LINEAGE_CHECKPOINT_KEYS) - set(record))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if record.get("schema_version") != RUN_LINEAGE_CHECKPOINT_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {RUN_LINEAGE_CHECKPOINT_SCHEMA_VERSION}")
    for key in _CHECKPOINT_STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"{_LABEL} {key} must be a string")
    errors.extend(_sequence_errors(record.get("sequence")))
    errors.extend(_artifact_ref_errors(record.get("artifact_refs")))
    if record.get("transition") not in TRANSITION_EVIDENCE_CLASS:
        errors.append(f"{_LABEL} transition is unsupported: {record.get('transition')!r}")
    if record.get("state") not in LINEAGE_STATES:
        errors.append(f"{_LABEL} state is unsupported: {record.get('state')!r}")
    if record.get("evidence_class") not in LINEAGE_EVIDENCE_CLASSES:
        errors.append(f"{_LABEL} evidence_class is unsupported: {record.get('evidence_class')!r}")
    if record.get("privacy") != CHECKPOINT_PRIVACY:
        errors.append(f"{_LABEL} privacy must be metadata_only")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the lineage boundary")
    for field in _REQUIRED_CHECKPOINT_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    for field in _OPTIONAL_CHECKPOINT_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=False))
    errors.extend(_digest_shape_errors(record))
    errors.extend(_evidence_class_errors(record))
    errors.extend(_missing_digest_errors(record))
    if str(record.get("run_id", "")) and str(record.get("lineage_id", "")) != lineage_id(
        str(record.get("run_id", ""))
    ):
        # Without this a hand-edited store could graft one run's history onto
        # another run's chain, and reconstructing either would read the other.
        errors.append(f"{_LABEL} lineage_id does not identify its own run")
    return errors


def append_run_lineage_checkpoint(
    paths: OmhPaths,
    *,
    run_id: str,
    transition: str,
    owner: str,
    safety_profile_revision: str,
    plan_digest: str = "",
    handoff_digest: str = "",
    context_revision: str = "",
    artifact_refs: Sequence[str] = (),
    state: str = "",
    recorded_at: str = "",
    continue_from: str = "",
) -> dict[str, Any]:
    """Extend one run's chain by one checkpoint, or refuse.

    The read of the chain head and the append of the new link happen under one
    lock, so two concurrent appends produce two links in a line rather than two
    records claiming one predecessor.

    `continue_from` is #826's "require continuation to name the lineage head":
    once a chain exists, a caller must say which head it read, and a head that
    has moved since is refused instead of silently overwritten. A genesis append
    names nothing, because there is nothing to name.

    `plan_digest`, `handoff_digest`, and `context_revision` are inherited from
    the head when the caller leaves them empty. A checkpoint carries the whole
    state a run continued from, and making every caller restate three unchanged
    references is how one of them ends up omitted on the checkpoint that
    mattered.
    """
    path = paths.runtime_run_lineage_checkpoints_path
    with file_lock(path, private=True):
        records, _ = read_jsonl_objects(path)
        chain = run_lineage_chain(records, run_id=run_id)
        head = chain[-1] if chain else {}
        head_digest = str(head.get("digest", ""))
        if head and str(continue_from or "") != head_digest:
            raise RunLineageError(
                f"{_LABEL} continuation must name the lineage head; this chain's head is not the one named"
            )
        _require_predecessors(chain, transition)
        record = build_run_lineage_checkpoint(
            run_id=run_id,
            transition=transition,
            owner=owner,
            safety_profile_revision=safety_profile_revision,
            sequence=len(chain),
            parent_digest=head_digest,
            plan_digest=plan_digest or str(head.get("plan_digest", "")),
            handoff_digest=handoff_digest or str(head.get("handoff_digest", "")),
            context_revision=context_revision or str(head.get("context_revision", "")),
            artifact_refs=artifact_refs,
            state=state,
            recorded_at=recorded_at,
        )
        append_store_line(path, record)
    return record


# ---------------------------------------------------------------------------
# Read and reconstruct
# ---------------------------------------------------------------------------


def read_run_lineage_checkpoints_result(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[str]]:
    return read_jsonl_objects(paths.runtime_run_lineage_checkpoints_path)


def read_run_lineage_checkpoints(
    paths: OmhPaths,
    *,
    run_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """The store, in append order, optionally scoped to one run."""
    records, _ = read_run_lineage_checkpoints_result(paths)
    if run_id is not None:
        records = run_lineage_chain(records, run_id=run_id)
    if limit is None:
        return records
    if limit < 1:
        return []
    return records[-limit:]


def run_lineage_chain(records: Sequence[Mapping[str, Any]], *, run_id: str) -> list[dict[str, Any]]:
    """One run's checkpoints, in append order.

    Append order is chain order: the store is append-only, so the sequence a
    reader sees is the sequence a writer wrote. `reconstruct_run_lineage` then
    checks that claim against each record's own `sequence` rather than trusting
    it, which is what turns a reordered store into a reported break instead of a
    plausible history.
    """
    return [
        dict(record)
        for record in records
        if isinstance(record, Mapping) and str(record.get("run_id", "")) == str(run_id)
    ]


def run_lineage_head(records: Sequence[Mapping[str, Any]], *, run_id: str) -> dict[str, Any]:
    """The checkpoint a continuation must name, or an empty mapping."""
    chain = run_lineage_chain(records, run_id=run_id)
    return chain[-1] if chain else {}


def reconstruct_run_lineage(records: Sequence[Mapping[str, Any]], *, run_id: str) -> dict[str, Any]:
    """#826 AC1: one run's history, rebuilt from the store alone and verified.

    Offline in the sense this repo means it: a pure function of records already
    in hand, reading no clock, no network, and no artifact the checkpoints point
    at. Every link is recomputed -- each record's own digest from its own
    fields, and each `parent_digest` against the digest of the record before it
    -- so an edit anywhere in the history is a reported break rather than a
    history that still reads plausibly.

    The walk stops at the first break and names the checkpoint that broke it.
    Continuing past one would report every later link as broken too, which is
    true and useless: the answer a reader needs is where the history stopped
    being trustworthy, and everything after the first break is a consequence of
    it.
    """
    chain = run_lineage_chain(records, run_id=run_id)
    intact: list[dict[str, Any]] = []
    break_code = ""
    broken: Mapping[str, Any] = {}
    seen_transitions: set[str] = set()
    for index, record in enumerate(chain):
        break_code = _checkpoint_break(record, index=index, intact=intact, seen=seen_transitions)
        if break_code:
            broken = record
            break
        intact.append(record)
        seen_transitions.add(str(record.get("transition", "")))
    head = intact[-1] if intact else {}
    return {
        "schema_version": RUN_LINEAGE_RECONSTRUCTION_SCHEMA_VERSION,
        "run_id": _redacted(str(run_id)),
        "lineage_id": lineage_id(str(run_id)),
        "ok": not break_code,
        "checkpoint_count": len(chain),
        "intact_checkpoint_count": len(intact),
        "transitions": [closed_vocabulary_value(item.get("transition"), LINEAGE_TRANSITIONS) for item in intact],
        "evidence_classes": [
            closed_vocabulary_value(item.get("evidence_class"), LINEAGE_EVIDENCE_CLASSES) for item in intact
        ],
        "head_digest": _rendered_digest(head.get("digest")),
        "head_transition": closed_vocabulary_value(head.get("transition"), LINEAGE_TRANSITIONS),
        "head_evidence_class": closed_vocabulary_value(head.get("evidence_class"), LINEAGE_EVIDENCE_CLASSES),
        "head_state": closed_vocabulary_value(head.get("state"), LINEAGE_STATES),
        "owner": _redacted(str(head.get("owner", ""))),
        "safety_profile_revision": _redacted(str(head.get("safety_profile_revision", ""))),
        "context_revision": _redacted(str(head.get("context_revision", ""))),
        "plan_digest": _redacted(str(head.get("plan_digest", ""))),
        "handoff_digest": _redacted(str(head.get("handoff_digest", ""))),
        "artifact_refs": _rendered_artifact_refs(head.get("artifact_refs")),
        "break_code": break_code,
        "break_reason": BREAK_REASONS.get(break_code, ""),
        "break_record_id": _redacted(str(broken.get("record_id", ""))),
        "break_transition": closed_vocabulary_value(broken.get("transition"), LINEAGE_TRANSITIONS),
        "break_sequence": len(intact) if break_code else _UNKNOWN_SEQUENCE,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def read_run_lineage(paths: OmhPaths, *, run_id: str) -> dict[str, Any]:
    """`reconstruct_run_lineage` over the store on disk."""
    records, _ = read_run_lineage_checkpoints_result(paths)
    return reconstruct_run_lineage(records, run_id=run_id)


def checkpoint_implies_evidence_class(checkpoint: Mapping[str, Any], evidence_class: str) -> bool:
    """#826 AC3: whether one checkpoint states that one evidence class was reached.

    True only for the single class the checkpoint's own transition records. A
    prepared checkpoint -- `plan_accepted` or `handoff_prepared`, the two
    transitions whose class is `prepared_not_observed` -- therefore answers
    False for every observed class in the vocabulary: preparing a run says only
    that it was prepared.

    Stating a class is not being the evidence for it. A checkpoint copies what
    the observation journal already held at that transition, so it can point at
    the observation behind a class and is never that observation itself;
    `CLAIM_BOUNDARY` is on every record and says so.
    """
    if evidence_class not in LINEAGE_EVIDENCE_CLASSES:
        return False
    if not isinstance(checkpoint, Mapping):
        return False
    return str(checkpoint.get("evidence_class", "")) == evidence_class


# ---------------------------------------------------------------------------
# Validate the store, render
# ---------------------------------------------------------------------------


def validate_run_lineage_checkpoint_store(path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """Validate the lineage store, optionally scoped to one run's checkpoints.

    Two halves, and both are load-bearing. `store_errors` gives the per-record
    schema and the shared chain walk -- a duplicate digest, a self-link, a
    parent naming a record that does not already exist, and two checkpoints
    claiming one predecessor are the four shapes a hash chain and a supersede
    chain reject identically. The reconstruction adds what only a per-run walk
    can see: a digest that no longer matches its own record, a parent that names
    an earlier record but not the *previous* one, a sequence that is not its
    position, and a transition whose required predecessor is absent.

    #826 AC2 lands here: changing history or a referenced digest fails
    `omh runtime validate`, not just an in-process helper.
    """
    records, read_errors = read_jsonl_objects(path)
    checkpoint_count, errors = store_errors(
        path,
        records,
        read_errors,
        run_id=run_id,
        validate_record=validate_run_lineage_checkpoint,
        label=_LABEL,
        id_key="digest",
        link_key="parent_digest",
        noun="checkpoint",
    )
    # A run with no checkpoints has no lineage, so it is not counted as one. The
    # scoped call always produces a report -- that is how a caller asks about one
    # run -- and counting an empty one would report a chain that does not exist.
    lineages = [report for report in _reconstructions_in(records, run_id=run_id) if report["checkpoint_count"]]
    broken = [report for report in lineages if not report["ok"]]
    errors.extend(
        f"{path}: {_LABEL} chain break at sequence {report['break_sequence']} "
        f"({report['break_record_id']}): {report['break_reason']}"
        for report in broken
    )
    return {
        "schema_version": RUN_LINEAGE_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "run_id": run_id or "",
        "ok": not errors,
        "checkpoint_count": checkpoint_count,
        "lineage_count": len(lineages),
        "broken_lineage_count": len(broken),
        "errors": errors,
    }


def compact_run_lineage_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """The user-facing shape, with every field through its class's guard.

    Rendering is the last point at which a hand-edited store reaches a human, so
    nothing is copied through verbatim: identifiers fold to a stable
    non-navigable handle when they are not opaque, closed vocabularies render
    empty outside their vocabulary, and a digest renders only in the one shape a
    digest may be stored in.
    """
    return {
        "record_id": _redacted(str(checkpoint.get("record_id", ""))),
        "lineage_id": _redacted(str(checkpoint.get("lineage_id", ""))),
        "run_id": _redacted(str(checkpoint.get("run_id", ""))),
        "sequence": checkpoint.get("sequence") if isinstance(checkpoint.get("sequence"), int) else _UNKNOWN_SEQUENCE,
        "parent_digest": _rendered_digest(checkpoint.get("parent_digest")),
        "digest": _rendered_digest(checkpoint.get("digest")),
        "transition": closed_vocabulary_value(checkpoint.get("transition"), LINEAGE_TRANSITIONS),
        "evidence_class": closed_vocabulary_value(checkpoint.get("evidence_class"), LINEAGE_EVIDENCE_CLASSES),
        "state": closed_vocabulary_value(checkpoint.get("state"), LINEAGE_STATES),
        "owner": _redacted(str(checkpoint.get("owner", ""))),
        "safety_profile_revision": _redacted(str(checkpoint.get("safety_profile_revision", ""))),
        "context_revision": _redacted(str(checkpoint.get("context_revision", ""))),
        "plan_digest": _redacted(str(checkpoint.get("plan_digest", ""))),
        "handoff_digest": _redacted(str(checkpoint.get("handoff_digest", ""))),
        "artifact_refs": _rendered_artifact_refs(checkpoint.get("artifact_refs")),
        "recorded_at": _redacted(str(checkpoint.get("recorded_at", ""))),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_state(evidence_class: str) -> str:
    return _PREPARED_STATE if evidence_class == _PREPARED_EVIDENCE_CLASS else _OBSERVED_STATE


def _require_predecessors(chain: Sequence[Mapping[str, Any]], transition: str) -> None:
    """Refuse a transition the chain has not earned yet.

    Checked against the transitions already in the chain rather than against the
    immediately previous one: a run that re-verifies after a fix appends
    verification twice, and an ordering rule that forbade that would refuse a
    real history to keep a tidy one.
    """
    required = _REQUIRED_PREDECESSORS.get(transition, ())
    seen = {str(record.get("transition", "")) for record in chain}
    missing = [name for name in required if name not in seen]
    if missing:
        raise RunLineageError(
            f"{_LABEL} {transition} requires an earlier {', '.join(missing)} checkpoint in this chain"
        )


def _checkpoint_break(
    record: Mapping[str, Any],
    *,
    index: int,
    intact: Sequence[Mapping[str, Any]],
    seen: set[str],
) -> str:
    """The first reason this checkpoint does not continue the chain, or "".

    Order matters, and the digest goes first. Editing any sealed field makes a
    record fail its own schema *and* fail its digest, and of the two answers
    only the second one tells a reader what happened: the record is well-formed
    and someone changed it. A record whose digest is not even digest-shaped, or
    that is malformed for some unrelated reason, falls through to the general
    verdict.
    """
    errors = validate_run_lineage_checkpoint(dict(record))
    digest = record.get("digest")
    if isinstance(digest, str) and _DIGEST.fullmatch(digest) and digest != checkpoint_digest(record):
        return BREAK_DIGEST_MISMATCH
    if errors:
        return BREAK_INVALID_RECORD
    if record.get("sequence") != index:
        return BREAK_SEQUENCE
    expected_parent = str(intact[-1].get("digest", "")) if intact else GENESIS_PARENT_DIGEST
    if str(record.get("parent_digest", "")) != expected_parent:
        return BREAK_PARENT_MISMATCH
    required = _REQUIRED_PREDECESSORS.get(str(record.get("transition", "")), ())
    if any(name not in seen for name in required):
        return BREAK_PREREQUISITE
    return ""


def _reconstructions_in(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str | None,
) -> list[dict[str, Any]]:
    """One reconstruction per run present in the store, or just the scoped one.

    Run ids are collected in append order and deduplicated, so a store report is
    deterministic and a run with no parseable id contributes no lineage rather
    than an empty one named "".
    """
    if run_id is not None:
        return [reconstruct_run_lineage(records, run_id=run_id)]
    ordered: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        value = str(record.get("run_id", ""))
        if value and value not in ordered:
            ordered.append(value)
    return [reconstruct_run_lineage(records, run_id=value) for value in ordered]


def _sequence_errors(value: Any) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int):
        return [f"{_LABEL} sequence must be an integer"]
    if value < 0:
        return [f"{_LABEL} sequence must not be negative"]
    return []


def _artifact_ref_errors(value: Any) -> list[str]:
    """A bounded, sorted, duplicate-free list of opaque references.

    Sorted and deduplicated because the list is inside the digest seal: two
    records naming the same artifacts in a different order would otherwise seal
    to different digests and read as different history.
    """
    if not isinstance(value, list):
        return [f"{_LABEL} artifact_refs must be a list"]
    if len(value) > ARTIFACT_REF_LIMIT:
        return [f"{_LABEL} artifact_refs must name at most {ARTIFACT_REF_LIMIT} artifacts"]
    errors: list[str] = []
    for index, item in enumerate(value):
        errors.extend(
            reference_errors(item, field=f"artifact_refs[{index}]", label=_LABEL, required=True)
        )
    if errors:
        return errors
    if list(value) != sorted(set(value)):
        errors.append(f"{_LABEL} artifact_refs must be sorted and free of duplicates")
    return errors


def _digest_shape_errors(record: Mapping[str, Any]) -> list[str]:
    """Both digest fields, their shapes, and the one genesis rule.

    A chain has exactly one record with no parent, and it is the record at
    position zero. Without this rule a hand-edited store could carry a second
    parentless checkpoint mid-chain, and a reader walking from the genesis would
    stop before reaching the real head.
    """
    errors: list[str] = []
    digest = record.get("digest")
    parent = record.get("parent_digest")
    if not isinstance(digest, str) or not _DIGEST.fullmatch(digest):
        errors.append(f"{_LABEL} digest must be a sha256 hex digest")
    elif digest != checkpoint_digest(record):
        errors.append(f"{_LABEL} digest does not match the checkpoint it seals")
    if not isinstance(parent, str) or (parent and not _DIGEST.fullmatch(parent)):
        errors.append(f"{_LABEL} parent_digest must be a sha256 hex digest or empty at the genesis checkpoint")
    if isinstance(digest, str) and isinstance(parent, str) and digest and digest == parent:
        errors.append(f"{_LABEL} parent_digest must not name the checkpoint itself")
    sequence = record.get("sequence")
    if isinstance(sequence, int) and not isinstance(sequence, bool) and isinstance(parent, str):
        if (sequence == 0) != (parent == GENESIS_PARENT_DIGEST):
            errors.append(
                f"{_LABEL} only the checkpoint at sequence 0 has no parent_digest, and it must have none"
            )
    return errors


def _evidence_class_errors(record: Mapping[str, Any]) -> list[str]:
    """The evidence class re-derived, and the state pinned against it.

    #826 AC3, structurally. The class is a function of the transition, so a
    stored class that disagrees with its transition is refused rather than read;
    and the two positional states are pinned to it in both directions, so a
    prepared checkpoint cannot carry an observed class and an observed one
    cannot carry the prepared class.
    """
    errors: list[str] = []
    transition = str(record.get("transition", ""))
    evidence_class = str(record.get("evidence_class", ""))
    state = str(record.get("state", ""))
    expected = TRANSITION_EVIDENCE_CLASS.get(transition)
    if expected is not None and evidence_class != expected:
        errors.append(
            f"{_LABEL} evidence_class must be {expected} for transition {transition}; a checkpoint never "
            "states a class its own transition did not reach"
        )
    if state == _PREPARED_STATE and evidence_class and evidence_class != _PREPARED_EVIDENCE_CLASS:
        errors.append(f"{_LABEL} a prepared checkpoint must carry evidence_class {_PREPARED_EVIDENCE_CLASS}")
    if state == _OBSERVED_STATE and evidence_class == _PREPARED_EVIDENCE_CLASS:
        errors.append(f"{_LABEL} an observed checkpoint must not carry evidence_class {_PREPARED_EVIDENCE_CLASS}")
    return errors


def _missing_digest_errors(record: Mapping[str, Any]) -> list[str]:
    """The references a transition cannot be recorded without.

    A plan cannot be accepted without naming which plan, and nothing from the
    handoff onward can be recorded without naming which handoff. Everything
    else is optional, because a chain that starts at `handoff_prepared` has no
    accepted plan artifact and a run whose context was never revised has no
    context revision.
    """
    errors: list[str] = []
    transition = str(record.get("transition", ""))
    if transition not in TRANSITION_EVIDENCE_CLASS:
        return errors
    if transition == "plan_accepted" and not str(record.get("plan_digest", "")):
        errors.append(f"{_LABEL} plan_digest is required on a plan_accepted checkpoint")
    if LINEAGE_TRANSITIONS.index(transition) >= LINEAGE_TRANSITIONS.index("handoff_prepared"):
        if not str(record.get("handoff_digest", "")):
            errors.append(f"{_LABEL} handoff_digest is required from the handoff_prepared checkpoint onward")
    return errors


def _normalized_artifact_refs(values: Iterable[str]) -> list[str]:
    """Sorted, deduplicated, guarded artifact references."""
    refs = sorted({str(value).strip() for value in values if str(value).strip()})
    if len(refs) > ARTIFACT_REF_LIMIT:
        raise RunLineageError(f"{_LABEL} artifact_refs must name at most {ARTIFACT_REF_LIMIT} artifacts")
    return [_opaque(ref, field="run_lineage_checkpoint artifact_refs") for ref in refs]


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=RunLineageError)


def _optional_opaque(value: str, *, field: str) -> str:
    text = str(value or "").strip()
    return _opaque(text, field=field) if text else ""


def _redacted(value: str) -> str:
    """Fold any handle into a bounded, non-navigable lineage reference."""
    return redacted_ref(value, field="run_lineage_ref")


def _rendered_digest(value: Any) -> str:
    """A digest renders only in the one shape it may be stored in."""
    text = str(value or "")
    return text if _DIGEST.fullmatch(text) else ""


def _rendered_artifact_refs(value: Any) -> list[str]:
    """A stored artifact list renders only when it is a list.

    A hand-edited store can carry a string where the list should be, and
    iterating that would render one entry per character.
    """
    if not isinstance(value, list):
        return []
    return [_redacted(str(item)) for item in value]
