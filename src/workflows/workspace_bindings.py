"""Append-only workspace bindings: one workspace, one active handoff (issue #820).

What was missing
----------------

`coding/isolation.py` decides *whether* a handoff should run in its own
worktree, and `wrapper/worktree_binding.py` builds a recipe for opening one
executor in one path. Neither consults anything about the other handoffs. Two
`build_worktree_executor_binding` calls naming the same directory both
succeeded, and two handoffs on one branch were equally invisible, so the first
sign of a collision was a dirty tree or a lost commit.

This store is the missing exclusivity: an acquisition that either reserves a
canonical workspace for exactly one handoff, or explains who already holds it
and what to do about it.

Why this is its own record family
---------------------------------

The tree refused a new family twice (#811, #818) and accepted one twice (#807,
#808), on a rule worth restating: a family joins on run id, and a join is where
an artifact and its metadata desynchronize, so a field group on the record that
already exists is the default.

A binding earns a family on the dimension the rule is really about -- whose
question it answers. Every other record in this tree describes one run: what it
asked for, what a gate decided about it, what an operator consented to. A
binding describes a *resource*, and its whole value is that it is readable by a
handoff that has nothing to do with the one that wrote it. A field group on a
delegation is invisible to the next delegation by construction, which is
exactly the shape that let two handoffs claim one directory. It also outlives
the run that took it, because a released binding and an abandoned one must be
distinguishable after the turn that created either is over.

Exclusivity, and what it costs
------------------------------

Acquisition is serialized on the store's own OS lock -- `local_store.file_lock`,
real on POSIX and on Windows -- so the read of the active set and the append of
the new claim cannot interleave. Two concurrent acquisitions on one workspace
therefore produce one held binding and one refusal, not two winners.

That is exclusivity over *this store*, and the honest limit is that it is
exclusivity over nothing else. Nothing stops a person running `git worktree
add` twice, and nothing stops an executor already running from writing wherever
it likes. #820's own out-of-scope list excludes creating, deleting, or merging
worktrees, and the `workspace` boundary of `handoff_safety_contract/v1` still
reports `declared_not_enforced` because no OMH-side change can constrain a
running process. What the guard removes is the case OMH *does* own: enabling a
second workspace-bound handoff on a workspace or branch that is already spoken
for.

Two spellings of one directory are one workspace
------------------------------------------------

Identity is the canonical path -- expanded, user-resolved, symlink-resolved,
through the same `paths.expand_path` every other path-taking surface uses --
and never the string the caller typed. `~/w`, `$HOME/w`, `/tmp/w/../w`, and the
absolute path all collide, which is the entire point: a guard that could be
defeated by typing the path differently would be decoration.

The canonical path is then digested. No host path byte reaches the store, and
`_WORKSPACE_REF` refuses a stored `workspace_ref` that is not a digest handle,
so `privacy: "metadata_only"` is structural here rather than a habit -- the
same discipline `blocked_work_records._PATH_SHAPED` applies for the same
reason. The cost is that a store cannot be read back into a path; a caller that
wants to know whether *its* workspace is bound digests its own path and asks.

Staleness blocks, and never releases
------------------------------------

Freshness is derived at read time from the head record's stamp, the idiom
`action_gate._account_state` uses for account signals and `approval_receipts`
uses for consent, rather than stored as a deadline a reader must trust. A
stored expiry is a second independently writable field: hand-editing one number
would widen the window, while `BINDING_STALE_AFTER_SECONDS` cannot be widened
by anything on disk.

A stale binding is refused, not reclaimed. Auto-release would be the background
lease service #820 puts out of scope, and it would be wrong on its own terms: a
binding older than the horizon means nobody reported back, which is not the
same as nobody running. So the refusal names a different reason code from a
fresh conflict and carries a different recovery -- release it explicitly, an
act with an owner -- and reuse stays blocked until someone does.

A binding is not dispatch
-------------------------

The four claims #800 keeps apart are existence, ownership, dispatch, and
result. A binding is the second one and nothing else: it does not say the
directory exists (nothing here stats the filesystem), it does not say an
executor started, and it does not say any work finished. `CLAIM_BOUNDARY` says
so on every record, and `validate_workspace_binding` refuses a record shaped to
assert execution by key name as well as by the closed key set.

Store mechanics
---------------

All of it is `system.append_only_store`, shared with the three families beside
it: the torn-tail-safe binary append under a real OS lock, the
opaque-reference guards, the supersede-chain walk, the digest handles, and the
best-effort mint-failure sidecar. This module adds no store mechanics of its
own; it passes its own key names to the shared walk so its records are not
described as receipts they are not.
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
    digest_ref,
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
from ..system.paths import OmhPaths, expand_path
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


WORKSPACE_BINDING_SCHEMA_VERSION: Final[str] = "workspace_binding_guard/v1"
WORKSPACE_BINDING_DECISION_SCHEMA_VERSION: Final[str] = "workspace_binding_decision/v1"
WORKSPACE_BINDING_STORE_VALIDATION_SCHEMA_VERSION: Final[str] = "workspace_binding_store_validation/v1"
WORKSPACE_BINDING_MINT_FAILURE_SCHEMA_VERSION: Final[str] = "workspace_binding_mint_failure/v1"

WORKSPACE_BINDING_STORE_NAME: Final[str] = "workspace_bindings.jsonl"
WORKSPACE_BINDING_MINT_FAILURE_STORE_NAME: Final[str] = "workspace_binding_mint_failures.jsonl"

BINDING_PRIVACY: Final[str] = "metadata_only"

# Stated on every record and echoed on every decision. The claim a binding
# makes is deliberately one of the four #800 keeps apart: ownership. It is not
# existence, not dispatch, and not result.
CLAIM_BOUNDARY: Final[str] = (
    "A workspace binding is one reservation of one canonical workspace and one branch for one handoff, "
    "under one owner and one base revision. It records who holds the workspace. It is not evidence that "
    "the workspace exists, that any executor was dispatched into it, or that any work ran or finished."
)

# `held`     -- the binding is active and the workspace is spoken for.
# `released` -- the binding is over. Appended as its own record linked to the
#               claim it ends; the held record is never rewritten.
BINDING_STATES: Final[tuple[str, ...]] = ("held", "released")

# What must happen before a binding may be released, declared when it is taken.
#
# `observed_terminal_state` -- the handoff reached a terminal state something
#     observed, and the binding may also be released explicitly.
# `explicit_safe_release`   -- only a deliberate release ends it. A terminal
#     observation is not enough, which is what a caller declares when the
#     workspace needs an operator's eyes before anyone else may take it.
RELEASE_CONDITIONS: Final[tuple[str, ...]] = ("observed_terminal_state", "explicit_safe_release")

# Why a binding ended, named on the release record. Same vocabulary as the
# conditions because a release cites the kind of event that ended it; which of
# the two a given binding will accept is `_PERMITTED_RELEASE_REASONS`.
RELEASE_REASONS: Final[tuple[str, ...]] = RELEASE_CONDITIONS

# An explicit safe release always works. It is the recovery path a stale
# binding points at, so a condition that could refuse it would leave an
# abandoned workspace with no way out short of editing the store by hand.
_PERMITTED_RELEASE_REASONS: Final[dict[str, tuple[str, ...]]] = {
    "observed_terminal_state": ("observed_terminal_state", "explicit_safe_release"),
    "explicit_safe_release": ("explicit_safe_release",),
}

# Six hours, the horizon `action_gate.ACCOUNT_SIGNAL_STALE_AFTER_SECONDS` and
# `executor_auth_signals.LIMIT_SIGNAL_STALE_AFTER_SECONDS` already use for a
# locally recorded state that stops being current state and becomes history.
# Deliberately far above `approval_receipts.APPROVAL_TTL_SECONDS` (one hour): an
# approval authorises one action and a coding handoff holds its workspace for as
# long as the work takes, so an hour would call ordinary work abandoned.
BINDING_STALE_AFTER_SECONDS: Final[int] = 6 * 60 * 60

# `age_seconds` when a stamp cannot be read, or is stamped in the future. Both
# mean the binding cannot be shown to be fresh, so both read as stale.
UNKNOWN_AGE_SECONDS: Final[int] = -1

WORKSPACE_BINDING_KEYS: Final[tuple[str, ...]] = (
    "base_revision",
    "binding_id",
    "branch_ref",
    "claim_boundary",
    "handoff_ref",
    "owner",
    "privacy",
    "recorded_at",
    "record_id",
    "release_condition",
    "release_reason",
    "repository_ref",
    "run_id",
    "safety_profile_revision",
    "schema_version",
    "state",
    "supersedes_binding_ref",
    "workspace_ref",
)

# The three keys whose value is a module constant rather than data. Everything
# else is rendered, and `compact_workspace_binding` must cover all of it -- a
# key in the tuple and missing from the compactor is a field that is stored and
# never seen, which has cost this repo twice.
WORKSPACE_BINDING_CONSTANT_KEYS: Final[tuple[str, ...]] = ("claim_boundary", "privacy", "schema_version")

# Identifiers that must be present, through `require_opaque_metadata_ref`.
_REQUIRED_BINDING_REFS: Final[tuple[str, ...]] = (
    "base_revision",
    "binding_id",
    "branch_ref",
    "handoff_ref",
    "owner",
    "record_id",
    "recorded_at",
    "repository_ref",
    "safety_profile_revision",
    "workspace_ref",
)
# `run_id` is legitimately absent: the guard runs *before* a workspace-bound
# handoff is enabled, and on the delegation lane that is before a run exists.
# The link is empty on the first record of a chain and only then.
_OPTIONAL_BINDING_REFS: Final[tuple[str, ...]] = ("run_id", "supersedes_binding_ref")
_BINDING_VOCABULARIES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("release_condition", RELEASE_CONDITIONS),
    ("state", BINDING_STATES),
)
# Closed vocabularies that may also be empty, which is a state and not a value.
_OPTIONAL_BINDING_VOCABULARIES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("release_reason", RELEASE_REASONS),
)
# Every string field, which on this record is every field: there is no list, no
# nested object, and deliberately no free text. A note field would be one more
# place for a prompt to arrive, and a reservation needs no prose to be citable.
_BINDING_STRING_FIELDS: Final[tuple[str, ...]] = WORKSPACE_BINDING_KEYS

# What the *terms* of a binding are: everything a re-acquisition must match for
# the claim to be the same claim. These are exactly the fields a caller names
# when it asks for a workspace, which is what lets a request and a stored record
# be fingerprinted against the same key set -- comparing two different key sets
# would never match, and every re-acquisition would read as a change of terms.
#
# `run_id`, `safety_profile_revision`, and `release_condition` are deliberately
# out. They describe the record, not the reservation: which run wrote it, which
# profile it was cleared under, and what will end it. The reservation itself is
# a repository, a workspace, a branch, a base revision, a handoff, and an owner.
_BINDING_TERMS_KEYS: Final[tuple[str, ...]] = (
    "base_revision",
    "binding_id",
    "branch_ref",
    "handoff_ref",
    "owner",
    "repository_ref",
    "workspace_ref",
)

# What a call did. `bound` on the decision says whether the requesting handoff
# holds the binding afterwards; this says how it got there.
OUTCOME_ACQUIRED: Final[str] = "acquired"
OUTCOME_ALREADY_HELD: Final[str] = "already_held"
OUTCOME_RELEASED: Final[str] = "released"
OUTCOME_ALREADY_RELEASED: Final[str] = "already_released"
OUTCOME_UNBOUND: Final[str] = "unbound"
OUTCOME_REFUSED: Final[str] = "refused"
OUTCOME_NOT_WRITTEN: Final[str] = "not_written"
BINDING_OUTCOMES: Final[tuple[str, ...]] = (
    OUTCOME_ACQUIRED,
    OUTCOME_ALREADY_HELD,
    OUTCOME_ALREADY_RELEASED,
    OUTCOME_NOT_WRITTEN,
    OUTCOME_REFUSED,
    OUTCOME_RELEASED,
    OUTCOME_UNBOUND,
)

# Refusal codes. The workspace pair and the branch pair are separate codes
# because "this workspace is taken" and "this branch is taken somewhere else"
# have different answers, and a caller that has to decide what to do next
# cannot act on a merged one. Fresh and stale are separate for the same reason:
# one is waited out, the other is released.
REFUSAL_WORKSPACE_BOUND: Final[str] = "workspace_already_bound"
REFUSAL_WORKSPACE_STALE: Final[str] = "workspace_binding_stale"
REFUSAL_BRANCH_BOUND: Final[str] = "branch_already_bound"
REFUSAL_BRANCH_STALE: Final[str] = "branch_binding_stale"
REFUSAL_TERMS_CHANGED: Final[str] = "binding_terms_changed"
REFUSAL_NOT_HELD: Final[str] = "binding_not_held"
REFUSAL_CONDITION_UNMET: Final[str] = "release_condition_unmet"
REFUSAL_MALFORMED: Final[str] = "binding_request_refused"
REFUSAL_NOT_WRITTEN: Final[str] = "binding_not_written"
REFUSAL_CODES: Final[tuple[str, ...]] = (
    REFUSAL_BRANCH_BOUND,
    REFUSAL_BRANCH_STALE,
    REFUSAL_CONDITION_UNMET,
    REFUSAL_MALFORMED,
    REFUSAL_NOT_HELD,
    REFUSAL_NOT_WRITTEN,
    REFUSAL_TERMS_CHANGED,
    REFUSAL_WORKSPACE_BOUND,
    REFUSAL_WORKSPACE_STALE,
)

# Constant strings, keyed by code. Nothing is interpolated into a reason: a
# refusal line is rendered to operators, and a line built from request values
# would be one more path for caller-controlled text to reach a terminal.
REFUSAL_REASONS: Final[dict[str, str]] = {
    REFUSAL_WORKSPACE_BOUND: (
        "This workspace is already bound to another active handoff; two handoffs cannot share one checkout."
    ),
    REFUSAL_WORKSPACE_STALE: (
        "This workspace is bound to a handoff that has not been heard from inside the freshness window. "
        "A binding that old is not reclaimed automatically, because nobody reporting back is not the same "
        "as nobody running."
    ),
    REFUSAL_BRANCH_BOUND: (
        "This branch is already bound to another active workspace in this repository; two checkouts cannot "
        "share one branch."
    ),
    REFUSAL_BRANCH_STALE: (
        "This branch is bound to a workspace that has not been heard from inside the freshness window. "
        "A binding that old is not reclaimed automatically."
    ),
    REFUSAL_TERMS_CHANGED: (
        "This handoff already holds this workspace on other terms; a binding is never rewritten in place."
    ),
    REFUSAL_NOT_HELD: "No active binding on record names this workspace for this handoff and owner.",
    REFUSAL_CONDITION_UNMET: (
        "The binding on record was taken under a condition that only an explicit safe release satisfies."
    ),
    REFUSAL_MALFORMED: "The binding request could not become a record.",
    REFUSAL_NOT_WRITTEN: "The binding store could not be written, so nothing was reserved or released.",
}

# What an operator can do about it. A closed id, never a sentence, for the
# reason `blocked_work_records.RECOVERY_ACTIONS` is: localized copy is then a
# lookup per locale rather than a translated constant.
RECOVERY_WAIT: Final[str] = "wait_for_owner_release"
RECOVERY_RELEASE_STALE: Final[str] = "release_stale_binding_explicitly"
RECOVERY_REACQUIRE: Final[str] = "release_then_reacquire"
RECOVERY_ACQUIRE_FIRST: Final[str] = "acquire_before_releasing"
RECOVERY_EXPLICIT_RELEASE: Final[str] = "release_with_an_explicit_safe_release"
RECOVERY_CORRECT_REQUEST: Final[str] = "correct_the_binding_request"
RECOVERY_RETRY_WRITE: Final[str] = "retry_when_the_store_is_writable"
RECOVERY_ACTIONS: Final[tuple[str, ...]] = (
    RECOVERY_ACQUIRE_FIRST,
    RECOVERY_CORRECT_REQUEST,
    RECOVERY_EXPLICIT_RELEASE,
    RECOVERY_REACQUIRE,
    RECOVERY_RELEASE_STALE,
    RECOVERY_RETRY_WRITE,
    RECOVERY_WAIT,
)

# One recovery per refusal. Total over `REFUSAL_CODES`, which
# `tests/test_workspace_binding_guard.py` pins: a refusal with no recovery is
# the shape that makes a guard unusable, and #820 AC3 is precisely about the
# refusal that has to come with the way out.
REFUSAL_RECOVERY: Final[dict[str, str]] = {
    REFUSAL_WORKSPACE_BOUND: RECOVERY_WAIT,
    REFUSAL_WORKSPACE_STALE: RECOVERY_RELEASE_STALE,
    REFUSAL_BRANCH_BOUND: RECOVERY_WAIT,
    REFUSAL_BRANCH_STALE: RECOVERY_RELEASE_STALE,
    REFUSAL_TERMS_CHANGED: RECOVERY_REACQUIRE,
    REFUSAL_NOT_HELD: RECOVERY_ACQUIRE_FIRST,
    REFUSAL_CONDITION_UNMET: RECOVERY_EXPLICIT_RELEASE,
    REFUSAL_MALFORMED: RECOVERY_CORRECT_REQUEST,
    REFUSAL_NOT_WRITTEN: RECOVERY_RETRY_WRITE,
}

# The guidance itself, keyed by recovery id. Constant, uninterpolated, and
# written for the person reading the refusal rather than for the caller.
RECOVERY_GUIDANCE: Final[dict[str, str]] = {
    RECOVERY_WAIT: (
        "Wait for the handoff that holds it to release the binding, or bind a different workspace or "
        "branch. Nothing here creates, deletes, or moves a worktree or a branch."
    ),
    RECOVERY_RELEASE_STALE: (
        "Confirm the holding handoff is really finished, then release the binding explicitly with reason "
        "explicit_safe_release. Nothing releases it on a timer, so the confirmation is a person's."
    ),
    RECOVERY_REACQUIRE: (
        "Release the binding this handoff already holds, then acquire it again on the new branch or base "
        "revision. A binding's terms are never rewritten in place."
    ),
    RECOVERY_ACQUIRE_FIRST: "Acquire the binding before releasing it; there is nothing on record to release.",
    RECOVERY_EXPLICIT_RELEASE: (
        "Release this binding with reason explicit_safe_release. It was taken under a condition that an "
        "observed terminal state alone does not satisfy."
    ),
    RECOVERY_CORRECT_REQUEST: (
        "Correct the binding request and try again; the reported error names the field that could not be "
        "stored."
    ),
    RECOVERY_RETRY_WRITE: (
        "Retry once the binding store is writable. Nothing was reserved or released, so no workspace "
        "changed hands."
    ),
}

# Key names an execution claim arrives under. The closed key set already refuses
# every one of them, but it refuses them as "unsupported keys", and the whole
# point of this family is that a reservation is not a dispatch -- so the shape
# is named and refused by name first. Mirrors
# `approval_receipts._EXECUTION_CLAIM_KEYS` rather than importing a private name
# across modules; `tests/test_workspace_binding_guard.py` pins the two equal.
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
_LABEL: Final[str] = "workspace_binding"

# A stored workspace reference is a digest handle and nothing else. This is what
# makes "no host path byte reaches the store" a property of the file rather than
# a property of the one function that happens to write it: a hand-edited record
# carrying `/Users/someone/work` is refused here, at validate, at render, and at
# every read that goes through them.
_WORKSPACE_REF: Final[re.Pattern[str]] = re.compile(r"^ref-[0-9a-f]{12}$")


class WorkspaceBindingError(ValueError):
    """Raised when a reservation cannot become a workspace binding record."""


# ---------------------------------------------------------------------------
# Identity: one directory, however it was spelled
# ---------------------------------------------------------------------------


def canonical_workspace_path(workspace_path: str | Path) -> str:
    """The one spelling of a directory this guard knows it by.

    Expanded, user-resolved, and symlink-resolved through `paths.expand_path`,
    which is the same normalization every other path-taking surface in the tree
    uses. An empty value is refused rather than normalized: `Path("")` resolves
    to the current directory, so accepting it would silently bind whatever
    directory the caller happened to be standing in.
    """
    text = str(workspace_path or "").strip()
    if not text:
        raise WorkspaceBindingError("workspace_binding workspace_path is required")
    return str(expand_path(text))


def workspace_ref(workspace_path: str | Path) -> str:
    """The stored handle for one canonical workspace.

    A digest of the canonical path, so two spellings of one directory produce
    one handle and no path byte is stored. Not reversible into a path, which is
    the trade: a caller asks about a workspace by digesting its own path.
    """
    return digest_ref(canonical_workspace_path(workspace_path))


def binding_id(*, repository_ref: str, workspace_ref_value: str) -> str:
    """Stable identity of one workspace claim, in one repository.

    The branch, the handoff, and the owner are deliberately not in it. They are
    properties of *who holds* the workspace, not of which workspace is being
    held: folding them in would open a second parallel chain per branch, and
    both chains would claim to be current for one directory.
    """
    base = json.dumps(
        {"repository_ref": str(repository_ref), "workspace_ref": str(workspace_ref_value)},
        sort_keys=True,
    )
    return f"binding-{hashlib.sha256(base.encode('utf-8')).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Build, validate, append
# ---------------------------------------------------------------------------


def build_workspace_binding(
    *,
    repository_ref: str,
    workspace_path: str | Path,
    branch_ref: str,
    base_revision: str,
    handoff_ref: str,
    owner: str,
    safety_profile_revision: str,
    release_condition: str = "observed_terminal_state",
    state: str = "held",
    release_reason: str = "",
    run_id: str = "",
    recorded_at: str = "",
    supersedes_binding_ref: str = "",
) -> dict[str, Any]:
    """Mint one binding record, or refuse.

    Every refusal here is the invariant this module exists for: a record can
    only describe one canonical workspace, reserved for one handoff, under one
    owner, on one branch, at one base revision.
    """
    if state not in BINDING_STATES:
        raise WorkspaceBindingError(f"workspace_binding state is unsupported: {state!r}")
    if release_condition not in RELEASE_CONDITIONS:
        raise WorkspaceBindingError(
            f"workspace_binding release_condition is unsupported: {release_condition!r}"
        )
    if release_reason and release_reason not in RELEASE_REASONS:
        raise WorkspaceBindingError(f"workspace_binding release_reason is unsupported: {release_reason!r}")
    workspace = workspace_ref(workspace_path)
    safe_repository = _opaque(repository_ref, field="workspace_binding repository_ref")
    safe_branch = _opaque(branch_ref, field="workspace_binding branch_ref")
    safe_revision = _opaque(base_revision, field="workspace_binding base_revision")
    safe_handoff = _opaque(handoff_ref, field="workspace_binding handoff_ref")
    safe_owner = _opaque(owner, field="workspace_binding owner")
    safe_profile = _opaque(safety_profile_revision, field="workspace_binding safety_profile_revision")
    safe_run_id = _opaque(run_id, field="workspace_binding run_id") if run_id else ""
    safe_supersedes = (
        _opaque(supersedes_binding_ref, field="workspace_binding supersedes_binding_ref")
        if supersedes_binding_ref
        else ""
    )
    stamp = _opaque(str(recorded_at or utc_now()), field="workspace_binding recorded_at")
    identity = binding_id(repository_ref=safe_repository, workspace_ref_value=workspace)
    record = {
        "schema_version": WORKSPACE_BINDING_SCHEMA_VERSION,
        "record_id": _record_id(stamp, identity, state),
        "binding_id": identity,
        "repository_ref": safe_repository,
        "workspace_ref": workspace,
        "branch_ref": safe_branch,
        "base_revision": safe_revision,
        "handoff_ref": safe_handoff,
        "owner": safe_owner,
        "run_id": safe_run_id,
        "safety_profile_revision": safe_profile,
        "state": state,
        "release_condition": release_condition,
        "release_reason": release_reason if state == "released" else "",
        "recorded_at": stamp,
        "supersedes_binding_ref": safe_supersedes,
        "privacy": BINDING_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    errors = validate_workspace_binding(record)
    if errors:
        raise WorkspaceBindingError(errors[0])
    return record


def validate_workspace_binding(record: dict[str, Any]) -> list[str]:
    """Every reason one record is not a workspace binding."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["workspace_binding must be an object"]
    claims_execution = sorted(key for key in record if str(key).lower() in EXECUTION_CLAIM_KEYS)
    if claims_execution:
        errors.append(
            "workspace_binding must not carry execution-claim keys: "
            f"{claims_execution}; a reservation is not a dispatch and not an outcome"
        )
    forbidden = sorted(key for key in record if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"workspace_binding must not carry raw or hidden keys: {forbidden}")
    extra_keys = sorted(set(record) - set(WORKSPACE_BINDING_KEYS) - set(claims_execution) - set(forbidden))
    if extra_keys:
        errors.append(f"workspace_binding has unsupported keys: {extra_keys}")
    missing = sorted(set(WORKSPACE_BINDING_KEYS) - set(record))
    if missing:
        errors.append(f"workspace_binding is missing keys: {missing}")
    if record.get("schema_version") != WORKSPACE_BINDING_SCHEMA_VERSION:
        errors.append(f"workspace_binding schema_version must be {WORKSPACE_BINDING_SCHEMA_VERSION}")
    for key in _BINDING_STRING_FIELDS:
        if not isinstance(record.get(key), str):
            errors.append(f"workspace_binding {key} must be a string")
    for key, vocabulary in _BINDING_VOCABULARIES:
        if record.get(key) not in vocabulary:
            errors.append(f"workspace_binding {key} is unsupported: {record.get(key)!r}")
    for key, vocabulary in _OPTIONAL_BINDING_VOCABULARIES:
        value = record.get(key)
        if value and value not in vocabulary:
            errors.append(f"workspace_binding {key} is unsupported: {value!r}")
    if record.get("privacy") != BINDING_PRIVACY:
        errors.append("workspace_binding privacy must be metadata_only")
    if record.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("workspace_binding claim_boundary must state the binding boundary")
    for field in _REQUIRED_BINDING_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=True))
    for field in _OPTIONAL_BINDING_REFS:
        errors.extend(reference_errors(record.get(field), field=field, label=_LABEL, required=False))
    errors.extend(_workspace_ref_errors(record.get("workspace_ref")))
    errors.extend(_lifecycle_errors(record))
    if str(record.get("record_id", "")) and str(record.get("record_id", "")) == str(
        record.get("supersedes_binding_ref", "")
    ):
        errors.append("workspace_binding supersedes_binding_ref must not name itself")
    expected_identity = binding_id(
        repository_ref=str(record.get("repository_ref", "")),
        workspace_ref_value=str(record.get("workspace_ref", "")),
    )
    if str(record.get("binding_id", "")) != expected_identity:
        # Without this a hand-edited store could move one workspace's claim onto
        # another workspace's chain, and the release of one would free the other.
        errors.append("workspace_binding binding_id does not identify its own repository and workspace")
    return errors


def _lifecycle_errors(record: Mapping[str, Any]) -> list[str]:
    """The state and its release reason, re-derived.

    Without this a hand-edited store could carry a `held` record that names why
    it ended, or a `released` record that names nothing -- and the active set,
    which is the whole guard, is read straight off `state`.
    """
    errors: list[str] = []
    state = str(record.get("state", ""))
    condition = str(record.get("release_condition", ""))
    reason = str(record.get("release_reason", ""))
    if state == "held" and reason:
        errors.append("workspace_binding release_reason must be empty while the binding is held")
    if state == "released":
        if not reason:
            errors.append("workspace_binding release_reason must name why a released binding ended")
        elif reason in RELEASE_REASONS and condition in _PERMITTED_RELEASE_REASONS:
            if reason not in _PERMITTED_RELEASE_REASONS[condition]:
                errors.append(
                    f"workspace_binding release_reason {reason!r} does not satisfy release_condition "
                    f"{condition!r}"
                )
    return errors


def append_workspace_binding(paths: OmhPaths, record: dict[str, Any]) -> dict[str, Any]:
    """Append one validated binding. Never rewrites, never reorders."""
    errors = validate_workspace_binding(record)
    if errors:
        raise WorkspaceBindingError(errors[0])
    path = paths.runtime_workspace_bindings_path
    # `file_lock` rather than a bare `fcntl.flock`: it holds a real OS lock on
    # both POSIX and Windows, and this store's mutual exclusion has to be a
    # property CI can assert on either platform.
    with file_lock(path, private=True):
        append_store_line(path, record)
    return record


# ---------------------------------------------------------------------------
# The guard: acquire, inspect, release
# ---------------------------------------------------------------------------


def acquire_workspace_binding(
    paths: OmhPaths,
    *,
    repository_ref: str,
    workspace_path: str | Path,
    branch_ref: str,
    base_revision: str,
    handoff_ref: str,
    owner: str,
    safety_profile_revision: str,
    release_condition: str = "observed_terminal_state",
    run_id: str = "",
    now: str = "",
) -> dict[str, Any]:
    """Reserve one canonical workspace for one handoff, or explain who holds it.

    Returns a `workspace_binding_decision/v1` mapping and never raises. A guard
    that raised would be a guard callers wrap in a bare `except`, and #820 AC3
    is that a refusal comes back *with* the way out rather than as a traceback.

    The read of the active set and the append of the new claim happen under one
    lock, so two concurrent acquisitions on one workspace produce one held
    binding and one refusal.
    """
    return acquire_workspace_binding_at(
        paths.runtime_workspace_bindings_path,
        repository_ref=repository_ref,
        workspace_path=workspace_path,
        branch_ref=branch_ref,
        base_revision=base_revision,
        handoff_ref=handoff_ref,
        owner=owner,
        safety_profile_revision=safety_profile_revision,
        release_condition=release_condition,
        run_id=run_id,
        now=now,
    )


def acquire_workspace_binding_at(
    store_path: Path,
    *,
    repository_ref: str,
    workspace_path: str | Path,
    branch_ref: str,
    base_revision: str,
    handoff_ref: str,
    owner: str,
    safety_profile_revision: str,
    release_condition: str = "observed_terminal_state",
    run_id: str = "",
    now: str = "",
) -> dict[str, Any]:
    """`acquire_workspace_binding` against an already-resolved store path.

    Callers that hold a runtime directory rather than an `OmhPaths` use this, so
    every path they touch is derived from the directory they were given and none
    is reconstructed from an environment they are not running in.
    """
    try:
        request = _acquire_request(
            repository_ref=repository_ref,
            workspace_path=workspace_path,
            branch_ref=branch_ref,
            base_revision=base_revision,
            handoff_ref=handoff_ref,
            owner=owner,
        )
    except WorkspaceBindingError as exc:
        return _malformed(store_path, error=str(exc), request={}, action="acquire")
    try:
        with file_lock(store_path, private=True):
            bindings, _ = read_jsonl_objects(store_path)
            verdict = workspace_binding_conflict_in(bindings, request=request, now=now)
            if verdict["reason_code"] or verdict["outcome"] == OUTCOME_ALREADY_HELD:
                # A refusal, or the same handoff restating an unchanged claim.
                # The restatement is returned as it stands rather than appended
                # again: a binding is a reservation, not an event log, and a
                # second identical record would make "when was this taken"
                # ambiguous -- which is the question staleness is measured from.
                return verdict
            head = _head_for(bindings, str(request["binding_id"]))
            record = build_workspace_binding(
                repository_ref=repository_ref,
                workspace_path=workspace_path,
                branch_ref=branch_ref,
                base_revision=base_revision,
                handoff_ref=handoff_ref,
                owner=owner,
                safety_profile_revision=safety_profile_revision,
                release_condition=release_condition,
                state="held",
                run_id=run_id,
                # `now` is the clock for this call: it stamps the new record and
                # it is the instant every prior binding's freshness was judged
                # against above. Two clocks here -- a real stamp and an injected
                # comparison -- is how a binding ends up looking stale the
                # moment it is taken.
                recorded_at=now,
                supersedes_binding_ref=str(head.get("record_id", "")) if head else "",
            )
            append_store_line(store_path, record)
    except WorkspaceBindingError as exc:
        return _malformed(store_path, error=str(exc), request=request, action="acquire")
    except OSError as exc:
        # Covers FileLockTimeout, which is a TimeoutError and therefore an
        # OSError: an unavailable lock is a store that could not be written, and
        # a workspace that could not be reserved is never reported as reserved.
        return _not_written(store_path, error=str(exc), request=request, action="acquire")
    return _decision(
        outcome=OUTCOME_ACQUIRED,
        bound=True,
        reason_code="",
        request=request,
        holder=record,
        now=now,
    )


def inspect_workspace_binding(
    paths: OmhPaths,
    *,
    repository_ref: str,
    workspace_path: str | Path,
    branch_ref: str,
    base_revision: str,
    handoff_ref: str,
    owner: str,
    now: str = "",
) -> dict[str, Any]:
    """What acquiring would answer, without reserving anything.

    The read-only half of the guard: a surface that has to say "this workspace
    is reserved, or here is who owns it" before offering a workspace-bound
    handoff asks this, and nothing is appended. The verdict comes from the same
    function `acquire_workspace_binding` calls, so what a caller is shown and
    what it gets one call later cannot disagree by construction -- with the one
    honest caveat that another handoff may acquire in between, which is why the
    acquisition re-checks under the lock rather than trusting this answer.
    """
    try:
        request = _acquire_request(
            repository_ref=repository_ref,
            workspace_path=workspace_path,
            branch_ref=branch_ref,
            base_revision=base_revision,
            handoff_ref=handoff_ref,
            owner=owner,
        )
    except WorkspaceBindingError as exc:
        return _decision(
            outcome=OUTCOME_REFUSED,
            bound=False,
            reason_code=REFUSAL_MALFORMED,
            request={},
            holder={},
            now=now,
            error=str(exc),
        )
    bindings, _ = read_jsonl_objects(paths.runtime_workspace_bindings_path)
    return workspace_binding_conflict_in(bindings, request=request, now=now)


def release_workspace_binding(
    paths: OmhPaths,
    *,
    repository_ref: str,
    workspace_path: str | Path,
    handoff_ref: str,
    owner: str,
    release_reason: str,
    run_id: str = "",
    now: str = "",
) -> dict[str, Any]:
    """End one binding, or explain why it cannot end yet.

    Two ways in, and they are not interchangeable. `observed_terminal_state` is
    the holder reporting its own work finished, so it is accepted only from the
    handoff and owner on record. `explicit_safe_release` is a deliberate act and
    is accepted from anyone, because it is the recovery a stale binding points
    at -- a release only the absent holder could perform would leave an
    abandoned workspace locked forever.
    """
    return release_workspace_binding_at(
        paths.runtime_workspace_bindings_path,
        repository_ref=repository_ref,
        workspace_path=workspace_path,
        handoff_ref=handoff_ref,
        owner=owner,
        release_reason=release_reason,
        run_id=run_id,
        now=now,
    )


def release_workspace_binding_at(
    store_path: Path,
    *,
    repository_ref: str,
    workspace_path: str | Path,
    handoff_ref: str,
    owner: str,
    release_reason: str,
    run_id: str = "",
    now: str = "",
) -> dict[str, Any]:
    """`release_workspace_binding` against an already-resolved store path."""
    try:
        request = _release_request(
            repository_ref=repository_ref,
            workspace_path=workspace_path,
            handoff_ref=handoff_ref,
            owner=owner,
        )
        if release_reason not in RELEASE_REASONS:
            raise WorkspaceBindingError(f"workspace_binding release_reason is unsupported: {release_reason!r}")
    except WorkspaceBindingError as exc:
        return _malformed(store_path, error=str(exc), request={}, action="release")
    try:
        with file_lock(store_path, private=True):
            bindings, _ = read_jsonl_objects(store_path)
            head = _head_for(bindings, str(request["binding_id"]))
            if not head:
                return _decision(
                    outcome=OUTCOME_REFUSED,
                    bound=False,
                    reason_code=REFUSAL_NOT_HELD,
                    request=request,
                    holder={},
                    now=now,
                )
            if str(head.get("state", "")) != "held":
                # Releasing twice is harmless and is not a refusal: the caller
                # asked for a workspace that is free, and it is free.
                return _decision(
                    outcome=OUTCOME_ALREADY_RELEASED,
                    bound=False,
                    reason_code="",
                    request=request,
                    holder=head,
                    now=now,
                )
            refusal = _release_refusal(head, request=request, release_reason=release_reason)
            if refusal:
                return _decision(
                    outcome=OUTCOME_REFUSED,
                    bound=False,
                    reason_code=refusal,
                    request=request,
                    holder=head,
                    now=now,
                )
            record = build_workspace_binding(
                repository_ref=repository_ref,
                workspace_path=workspace_path,
                # The terms come off the record being released, never off the
                # caller. A release is the end of the claim on file and must not
                # be a second place its branch or revision can be stated.
                branch_ref=str(head.get("branch_ref", "")),
                base_revision=str(head.get("base_revision", "")),
                handoff_ref=str(head.get("handoff_ref", "")),
                owner=str(head.get("owner", "")),
                safety_profile_revision=str(head.get("safety_profile_revision", "")),
                release_condition=str(head.get("release_condition", "")),
                state="released",
                release_reason=release_reason,
                run_id=run_id or str(head.get("run_id", "")),
                recorded_at=now,
                supersedes_binding_ref=str(head.get("record_id", "")),
            )
            append_store_line(store_path, record)
    except WorkspaceBindingError as exc:
        return _malformed(store_path, error=str(exc), request=request, action="release")
    except OSError as exc:
        return _not_written(store_path, error=str(exc), request=request, action="release")
    return _decision(
        outcome=OUTCOME_RELEASED,
        bound=False,
        reason_code="",
        request=request,
        holder=record,
        now=now,
    )


def workspace_binding_conflict_in(
    bindings: Sequence[Mapping[str, Any]],
    *,
    request: Mapping[str, Any],
    now: str = "",
) -> dict[str, Any]:
    """The refusal an acquisition would hit, over bindings already in hand.

    A `workspace_binding_decision/v1` mapping whose `reason_code` is empty when
    nothing stands in the way. Two conflict axes, in this order:

    * the same canonical workspace, in the same repository, held by another
      handoff or owner -- the collision #820 is named after;
    * the same branch, in the same repository, held by another active workspace
      -- two checkouts on one branch, which is the other half of AC1.

    The workspace axis is checked first because it is the more specific answer:
    a caller told "this branch is taken" when its own directory is also taken
    would fix the branch and hit the second refusal immediately.

    Staleness is derived here, at read time, and it refuses rather than
    reclaims. It gets its own reason code on both axes because the recovery
    differs: a fresh conflict is waited out, a stale one is released by a person.
    """
    known = [binding for binding in bindings if isinstance(binding, Mapping)]
    active = active_workspace_bindings(known)
    identity = str(request.get("binding_id", ""))
    for binding in active:
        if str(binding.get("binding_id", "")) != identity:
            continue
        if _is_same_holder(binding, request):
            if _binding_terms(binding) == _binding_terms(request):
                return _decision(
                    outcome=OUTCOME_ALREADY_HELD, bound=True, reason_code="", request=request,
                    holder=binding, now=now,
                )
            return _decision(
                outcome=OUTCOME_REFUSED, bound=False, reason_code=REFUSAL_TERMS_CHANGED, request=request,
                holder=binding, now=now,
            )
        stale = binding_is_stale(binding, now)
        return _decision(
            outcome=OUTCOME_REFUSED,
            bound=False,
            reason_code=REFUSAL_WORKSPACE_STALE if stale else REFUSAL_WORKSPACE_BOUND,
            request=request,
            holder=binding,
            now=now,
        )
    for binding in active:
        if str(binding.get("repository_ref", "")) != str(request.get("repository_ref", "")):
            continue
        if str(binding.get("branch_ref", "")) != str(request.get("branch_ref", "")):
            continue
        stale = binding_is_stale(binding, now)
        return _decision(
            outcome=OUTCOME_REFUSED,
            bound=False,
            reason_code=REFUSAL_BRANCH_STALE if stale else REFUSAL_BRANCH_BOUND,
            request=request,
            holder=binding,
            now=now,
        )
    return _decision(
        outcome=OUTCOME_UNBOUND, bound=False, reason_code="", request=request, holder={}, now=now,
    )


# ---------------------------------------------------------------------------
# Read, derive, validate the store, render
# ---------------------------------------------------------------------------


def read_workspace_bindings_result(paths: OmhPaths) -> tuple[list[dict[str, Any]], list[str]]:
    return read_jsonl_objects(paths.runtime_workspace_bindings_path)


def read_workspace_bindings(
    paths: OmhPaths,
    *,
    run_id: str | None = None,
    binding_id_value: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """The store, in append order, optionally scoped to one run or one binding."""
    bindings, _ = read_workspace_bindings_result(paths)
    if run_id is not None:
        bindings = [binding for binding in bindings if str(binding.get("run_id", "")) == run_id]
    if binding_id_value is not None:
        bindings = [binding for binding in bindings if str(binding.get("binding_id", "")) == binding_id_value]
    if limit is None:
        return bindings
    if limit < 1:
        return []
    return bindings[-limit:]


def latest_binding_in(bindings: Sequence[Mapping[str, Any]], binding_id_value: str) -> dict[str, Any]:
    """The record that speaks for one workspace claim: the one selection rule.

    The store is append-only, so store order is arrival order and the last
    record for a claim is its current state. Every surface that judges a binding
    selects through this function, which is what stops the store validator and
    the conflict predicate from picking different records for one workspace and
    reaching opposite conclusions about it.
    """
    return dict(latest_record_in(bindings, key="binding_id", value=str(binding_id_value)))


def active_workspace_bindings(bindings: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every workspace claim whose current record still holds it.

    Chain heads only. A superseded `held` record is not active -- something
    later replaced it -- and a stale one still is: staleness blocks reuse and
    does not end a binding, for the reason the module docstring gives.
    """
    heads: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        if not isinstance(binding, Mapping):
            continue
        heads[str(binding.get("binding_id", ""))] = dict(binding)
    return [binding for binding in heads.values() if str(binding.get("state", "")) == "held"]


def binding_age_seconds(recorded_at: str, now: str = "") -> int:
    """How old a binding record is, computed now, or `UNKNOWN_AGE_SECONDS`.

    A stamp that does not parse and a stamp in the future both return
    `UNKNOWN_AGE_SECONDS`: neither can be shown to be inside the window, and
    clamping a future stamp to zero would make "edit the timestamp forward" a
    way to hold a workspace indefinitely from disk.
    """
    recorded = _parse_stamp(recorded_at)
    if recorded is None:
        return UNKNOWN_AGE_SECONDS
    reference = _parse_stamp(now) or datetime.now(timezone.utc)
    age = int((reference - recorded).total_seconds())
    if age < 0:
        return UNKNOWN_AGE_SECONDS
    return age


def binding_is_stale(binding: Mapping[str, Any], now: str = "") -> bool:
    """Whether this binding is outside the freshness window, evaluated now."""
    age = binding_age_seconds(str(binding.get("recorded_at", "")), now)
    return age == UNKNOWN_AGE_SECONDS or age > BINDING_STALE_AFTER_SECONDS


def validate_workspace_binding_store(path: Path, *, run_id: str | None = None) -> dict[str, Any]:
    """Validate the binding store, optionally scoped to one run's records.

    Unscoped (`run_id=None`) this is the store's own report: unparseable lines,
    every record's schema, and the integrity of the supersede chain. That chain
    is what makes "which record holds this workspace" answerable, so a fork in
    it is not a cosmetic fault -- it is two records both claiming to be the
    current state of one directory.
    """
    bindings, read_errors = read_jsonl_objects(path)
    binding_count, errors = store_errors(
        path,
        bindings,
        read_errors,
        run_id=run_id,
        validate_record=validate_workspace_binding,
        label=_LABEL,
        id_key="record_id",
        link_key="supersedes_binding_ref",
        noun="binding",
    )
    return {
        "schema_version": WORKSPACE_BINDING_STORE_VALIDATION_SCHEMA_VERSION,
        "path": str(path),
        "run_id": run_id or "",
        "ok": not errors,
        "binding_count": binding_count,
        "mint_failure_count": len(read_workspace_binding_mint_failures(path)),
        "errors": errors,
    }


def workspace_binding_mint_failures_path(store_path: Path) -> Path:
    """Where unrecorded acquisitions are logged, beside the store they failed to reach."""
    return store_path.with_name(WORKSPACE_BINDING_MINT_FAILURE_STORE_NAME)


def read_workspace_binding_mint_failures(store_path: Path) -> list[dict[str, Any]]:
    failures, _ = read_jsonl_objects(workspace_binding_mint_failures_path(store_path))
    return failures


def compact_workspace_binding(binding: Mapping[str, Any], *, now: str = "") -> dict[str, Any]:
    """The user-facing shape, with every field through its class's guard.

    Rendering is the last point at which a hand-edited store reaches a human, so
    nothing is copied through verbatim: identifiers fold to a stable
    non-navigable handle when they are not opaque, and closed vocabularies
    render empty outside their vocabulary. `age_seconds` and `stale` are derived
    here rather than stored, which is the whole point -- nothing on disk says a
    binding is still current.
    """
    return {
        "record_id": redacted_workspace_binding_ref(str(binding.get("record_id", ""))),
        "binding_id": redacted_workspace_binding_ref(str(binding.get("binding_id", ""))),
        "repository_ref": redacted_workspace_binding_ref(str(binding.get("repository_ref", ""))),
        "workspace_ref": _rendered_workspace_ref(binding.get("workspace_ref")),
        "branch_ref": redacted_workspace_binding_ref(str(binding.get("branch_ref", ""))),
        "base_revision": redacted_workspace_binding_ref(str(binding.get("base_revision", ""))),
        "handoff_ref": redacted_workspace_binding_ref(str(binding.get("handoff_ref", ""))),
        "owner": redacted_workspace_binding_ref(str(binding.get("owner", ""))),
        "run_id": redacted_workspace_binding_ref(str(binding.get("run_id", ""))),
        "safety_profile_revision": redacted_workspace_binding_ref(
            str(binding.get("safety_profile_revision", ""))
        ),
        "state": closed_workspace_binding_value(binding.get("state"), BINDING_STATES),
        "release_condition": closed_workspace_binding_value(
            binding.get("release_condition"), RELEASE_CONDITIONS
        ),
        "release_reason": closed_workspace_binding_value(binding.get("release_reason"), RELEASE_REASONS),
        "recorded_at": redacted_workspace_binding_ref(str(binding.get("recorded_at", ""))),
        "supersedes_binding_ref": redacted_workspace_binding_ref(
            str(binding.get("supersedes_binding_ref", ""))
        ),
        "age_seconds": binding_age_seconds(str(binding.get("recorded_at", "")), now),
        "stale": binding_is_stale(binding, now),
    }


def redacted_workspace_binding_ref(value: str) -> str:
    """Fold any handle into a bounded, non-navigable binding reference."""
    return redacted_ref(value, field="workspace_binding_ref")


def closed_workspace_binding_value(value: Any, allowed: tuple[str, ...]) -> str:
    """A closed-vocabulary field renders only what its vocabulary contains."""
    return closed_vocabulary_value(value, allowed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _acquire_request(
    *,
    repository_ref: str,
    workspace_path: str | Path,
    branch_ref: str,
    base_revision: str,
    handoff_ref: str,
    owner: str,
) -> dict[str, Any]:
    """The guarded, canonicalized form of what an acquisition asked for.

    Built before the store is touched, so a malformed request is refused without
    holding the lock, and carrying exactly `_BINDING_TERMS_KEYS` so a request and
    a stored record are fingerprinted over one key set rather than two.
    """
    return {
        **_release_request(
            repository_ref=repository_ref,
            workspace_path=workspace_path,
            handoff_ref=handoff_ref,
            owner=owner,
        ),
        "branch_ref": _opaque(branch_ref, field="workspace_binding branch_ref"),
        "base_revision": _opaque(base_revision, field="workspace_binding base_revision"),
    }


def _release_request(
    *,
    repository_ref: str,
    workspace_path: str | Path,
    handoff_ref: str,
    owner: str,
) -> dict[str, Any]:
    """The identity and the claimed holder, with no terms.

    A release names no branch and no base revision on purpose: it ends the claim
    on record, whatever that claim said. Reading terms from the caller here
    would make a release a second place a binding's branch or revision could be
    stated, and the two statements could disagree.
    """
    workspace = workspace_ref(workspace_path)
    safe_repository = _opaque(repository_ref, field="workspace_binding repository_ref")
    return {
        "binding_id": binding_id(repository_ref=safe_repository, workspace_ref_value=workspace),
        "repository_ref": safe_repository,
        "workspace_ref": workspace,
        "branch_ref": "",
        "base_revision": "",
        "handoff_ref": _opaque(handoff_ref, field="workspace_binding handoff_ref"),
        "owner": _opaque(owner, field="workspace_binding owner"),
    }


def _head_for(bindings: Sequence[Mapping[str, Any]], binding_id_value: str) -> dict[str, Any]:
    return latest_binding_in(bindings, binding_id_value)


def _is_same_holder(binding: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    """Whether one binding is held by the handoff and owner now asking for it.

    Both halves. A second owner reporting the same handoff reference is not the
    same claim -- ownership is exactly what a binding records, and collapsing
    the two would let any caller inherit a reservation by quoting its handoff.
    """
    return str(binding.get("handoff_ref", "")) == str(request.get("handoff_ref", "")) and str(
        binding.get("owner", "")
    ) == str(request.get("owner", ""))


def _release_refusal(
    head: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    release_reason: str,
) -> str:
    """Why this release cannot end this binding, or "" when it can."""
    if release_reason == "observed_terminal_state" and not _is_same_holder(head, request):
        return REFUSAL_NOT_HELD
    condition = str(head.get("release_condition", ""))
    permitted = _PERMITTED_RELEASE_REASONS.get(condition, RELEASE_REASONS)
    if release_reason not in permitted:
        return REFUSAL_CONDITION_UNMET
    return ""


def _binding_terms(record: Mapping[str, Any]) -> str:
    """Which claim a record or a request states, ignoring when it was recorded.

    One key tuple for both sides. `_acquire_request` builds exactly these keys
    for that reason: fingerprinting a request over the keys it happens to carry
    and a record over the keys it happens to carry would never match, and every
    re-acquisition would read as a change of terms.
    """
    return record_fingerprint(record, _BINDING_TERMS_KEYS)


def _decision(
    *,
    outcome: str,
    bound: bool,
    reason_code: str,
    request: Mapping[str, Any],
    holder: Mapping[str, Any],
    now: str,
    error: str = "",
) -> dict[str, Any]:
    """One `workspace_binding_decision/v1` verdict.

    Everything a caller needs to act, and nothing it has to look up: what
    happened, whether it holds the workspace now, why not, who does, how stale
    that claim is, and the recovery -- as a closed id *and* as the sentence for
    it, because #820 AC3 asks for guidance and an id alone is not guidance.
    """
    recovery = REFUSAL_RECOVERY.get(reason_code, "")
    return {
        "schema_version": WORKSPACE_BINDING_DECISION_SCHEMA_VERSION,
        "outcome": outcome,
        "bound": bound,
        "reason_code": reason_code,
        "reason": REFUSAL_REASONS.get(reason_code, ""),
        "recovery_action": recovery,
        "recovery_guidance": RECOVERY_GUIDANCE.get(recovery, ""),
        "error": error,
        "binding_id": redacted_workspace_binding_ref(str(request.get("binding_id", ""))),
        "record_id": redacted_workspace_binding_ref(str(holder.get("record_id", ""))),
        "state": closed_workspace_binding_value(holder.get("state"), BINDING_STATES),
        "holder_owner": redacted_workspace_binding_ref(str(holder.get("owner", ""))),
        "holder_handoff_ref": redacted_workspace_binding_ref(str(holder.get("handoff_ref", ""))),
        "holder_branch_ref": redacted_workspace_binding_ref(str(holder.get("branch_ref", ""))),
        "holder_base_revision": redacted_workspace_binding_ref(str(holder.get("base_revision", ""))),
        "release_condition": closed_workspace_binding_value(
            holder.get("release_condition"), RELEASE_CONDITIONS
        ),
        "requested_workspace_ref": _rendered_workspace_ref(request.get("workspace_ref")),
        "requested_repository_ref": redacted_workspace_binding_ref(str(request.get("repository_ref", ""))),
        "requested_branch_ref": redacted_workspace_binding_ref(str(request.get("branch_ref", ""))),
        "requested_base_revision": redacted_workspace_binding_ref(str(request.get("base_revision", ""))),
        "requested_handoff_ref": redacted_workspace_binding_ref(str(request.get("handoff_ref", ""))),
        "requested_owner": redacted_workspace_binding_ref(str(request.get("owner", ""))),
        "age_seconds": binding_age_seconds(str(holder.get("recorded_at", "")), now) if holder else UNKNOWN_AGE_SECONDS,
        "stale": binding_is_stale(holder, now) if holder else False,
        "stale_after_seconds": BINDING_STALE_AFTER_SECONDS,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _malformed(
    store_path: Path,
    *,
    error: str,
    request: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    _append_mint_failure(store_path, outcome=OUTCOME_REFUSED, error=error, request=request, action=action)
    return _decision(
        outcome=OUTCOME_REFUSED,
        bound=False,
        reason_code=REFUSAL_MALFORMED,
        request=request,
        holder={},
        now="",
        error=error,
    )


def _not_written(
    store_path: Path,
    *,
    error: str,
    request: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    _append_mint_failure(store_path, outcome=OUTCOME_NOT_WRITTEN, error=error, request=request, action=action)
    return _decision(
        outcome=OUTCOME_NOT_WRITTEN,
        bound=False,
        reason_code=REFUSAL_NOT_WRITTEN,
        request=request,
        holder={},
        now="",
        error=error,
    )


def _append_mint_failure(
    store_path: Path,
    *,
    outcome: str,
    error: str,
    request: Mapping[str, Any],
    action: str,
) -> None:
    """Log an acquisition or release that never reached the store, best effort.

    A refusal the caller already holds is the record of last resort; this is so
    a workspace that could not be reserved is visible without the calling flow
    having to invent a channel for it.
    """
    append_sidecar_line(
        workspace_binding_mint_failures_path(store_path),
        {
            "schema_version": WORKSPACE_BINDING_MINT_FAILURE_SCHEMA_VERSION,
            "recorded_at": utc_now(),
            "action": str(action),
            "outcome": str(outcome),
            "binding_id": str(request.get("binding_id", "")),
            "handoff_ref": str(request.get("handoff_ref", "")),
            "owner": str(request.get("owner", "")),
            "error": _bounded_error(error),
        },
    )


def _workspace_ref_errors(value: Any) -> list[str]:
    """A stored workspace reference is a digest handle and never a path."""
    errors = reference_errors(value, field="workspace_ref", label=_LABEL, required=True)
    if errors or not isinstance(value, str):
        return errors
    if not _WORKSPACE_REF.fullmatch(value):
        return [
            f"{_LABEL} workspace_ref must be a canonical-path digest handle; this record carries the "
            "identity of a workspace, never its path"
        ]
    return []


def _rendered_workspace_ref(value: Any) -> str:
    """A workspace reference renders only in the one shape it may be stored in."""
    text = str(value or "")
    return text if _WORKSPACE_REF.fullmatch(text) else ""


def _parse_stamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = strict_fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _opaque(value: str, *, field: str) -> str:
    return opaque_ref(value, field=field, error=WorkspaceBindingError)


def _bounded_error(value: str) -> str:
    """One bounded, secret-free line for the failure log."""
    raw = " ".join(str(value or "").split())
    if not raw:
        return ""
    if is_sensitive_metadata_text(raw):
        return "[redacted]"
    return raw[:200]


def _record_id(recorded_at: str, binding_id_value: str, state: str) -> str:
    return mint_record_id(
        prefix="workspace-binding",
        identity={"binding_id": binding_id_value, "recorded_at": recorded_at, "state": state},
    )
