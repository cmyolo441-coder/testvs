"""`recovery_anchor/v1`: what a baseline was, and how a person could return to it.

After risky work a user asks one question -- "how do I undo this safely?" -- and
today nothing in this tree can answer it. There is no recorded baseline
revision, no bounded record of what was already uncommitted when the work
started, and no written recipe. This module is that record and that answer.

It is only ever a record and an answer. OMH performs no rollback here: nothing
in this module runs a command, spawns a process, opens a socket, or creates,
writes, or deletes a single path. An anchor is metadata; guidance is metadata
plus instruction lines a human runs themselves. That is the whole boundary, and
`CLAIM_BOUNDARY` states it on every record so a reader cannot infer otherwise
from a status line.

Three refusals carry the contract, and each is the reason it exists:

- An anchor with no observed baseline stays `prepared`, and a prepared anchor
  is structurally incapable of describing a recovery: `base_revision` is empty
  and `recovery_recipe` is empty, enforced in both directions by
  `validate_recovery_anchor`. "An anchor was prepared" must never read as "you
  can safely undo this", so the prepared shape has nothing to read that way.
- Guidance for a workspace other than the one the anchor was taken in is
  refused outright. A baseline from another checkout cannot describe this one,
  and offering it would be worse than offering nothing.
- `recoverable` is true for exactly one status. `RECOVERABLE_GUIDANCE_STATUSES`
  holds that one member, and the validator checks the flag against it, so an
  absent, invalid, prepared-only, or mismatched anchor cannot report
  recoverability by any path.

Everything stored is bounded metadata. The workspace is a digest handle rather
than a path, which is both the privacy posture the rest of this package uses
and exactly the comparison the mismatch refusal needs. The dirty state is a
digest plus counts plus a bounded, redacted path sample -- never file contents,
and never an unbounded path list.

Nothing here reads a clock. `created_at` and `baseline_observed_at` are
parameters, and neither reaches an id seed or a digest seed, because a value
that changes between two identical observations turns an equality check into a
race -- the kind the slower CI runner loses first.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
import re
from typing import Any

from ..system.append_only_store import RAW_OR_HIDDEN_KEYS, digest_ref, is_url_shaped, reference_errors
from ..system.metadata_safety import is_sensitive_metadata_text


RECOVERY_ANCHOR_SCHEMA_VERSION = "recovery_anchor/v1"
RECOVERY_GUIDANCE_SCHEMA_VERSION = "recovery_guidance/v1"
RECOVERY_ANCHOR_PRIVACY = "metadata_only"

CLAIM_BOUNDARY = (
    "A recovery anchor is bounded metadata plus written instructions. OMH performs no rollback, runs no "
    "command that changes a workspace, and creates or deletes nothing. A prepared anchor is not a "
    "recoverable baseline, and no anchor is execution, verification, review, CI, merge-readiness, or "
    "merge evidence."
)

# What one anchor is attached to. `handoff` is a prepared handoff that has not
# become a run; `run` is a linked runtime run.
ANCHOR_TARGET_TYPES = ("handoff", "run")

# The two states an anchor can hold. There is no third: either a surface
# observed a baseline revision for the workspace, or none did.
ANCHOR_STATES = ("prepared", "baseline_observed")

# Where an observed baseline came from. Both observing members name a surface
# that already exists in this tree and already requires observed evidence
# before it writes -- the worktree ledger in `omh.coding.worktree_creator` and
# the run journal in `omh.workflows.observation_journal`. `unknown` is the
# absence of an observation, and it is the only value a prepared anchor holds.
BASELINE_SOURCES = ("unknown", "runtime_observation", "worktree_observation")
OBSERVING_BASELINE_SOURCES = ("runtime_observation", "worktree_observation")

# `not_observed` is deliberately distinct from `clean`. Nothing looked is not
# the same as nothing was there, and a recipe that treated the two alike would
# tell someone with uncommitted work that they had none.
DIRTY_STATE_STATES = ("not_observed", "clean", "dirty")

RECOVERY_ANCHOR_KEYS = (
    "anchor_id",
    "anchor_state",
    "base_revision",
    "baseline_observed_at",
    "baseline_source",
    "claim_boundary",
    "created_at",
    "dirty_state",
    "evidence_refs",
    "privacy",
    "recovery_recipe",
    "schema_version",
    "target_id",
    "target_type",
    "workspace_ref",
)

DIRTY_STATE_KEYS = (
    "changed_file_count",
    "digest",
    "paths",
    "paths_truncated",
    "state",
    "untracked_file_count",
)

RECOVERY_GUIDANCE_KEYS = (
    "anchor_id",
    "base_revision",
    "claim_boundary",
    "dirty_state",
    "performs_rollback",
    "reason",
    "recoverable",
    "schema_version",
    "status",
    "steps",
    "workspace_ref",
)

RECOVERY_GUIDANCE_STATUSES = (
    "anchor_absent",
    "anchor_invalid",
    "baseline_not_observed",
    "workspace_mismatch",
    "baseline_available",
)
# The whole of AC3, as a one-member tuple the validator reads. A status added
# without a decision about recoverability lands outside this tuple, which means
# not recoverable -- the safe direction.
RECOVERABLE_GUIDANCE_STATUSES = ("baseline_available",)

# Written for the person who asked the question, not for a log line.
GUIDANCE_REASONS = {
    "anchor_absent": (
        "No recovery anchor was attached to this work, so no baseline was recorded and there is nothing "
        "to return to."
    ),
    "anchor_invalid": (
        "This recovery anchor does not validate, so its baseline cannot be trusted to describe any "
        "workspace."
    ),
    "baseline_not_observed": (
        "This recovery anchor was prepared but no baseline revision was ever observed, so it cannot say "
        "what to return to. A prepared anchor is not a recoverable baseline."
    ),
    "workspace_mismatch": (
        "This recovery anchor was taken in a different workspace, and a baseline from another workspace "
        "cannot describe this one."
    ),
    "baseline_available": (
        "This recovery anchor names an observed baseline for the workspace you asked about. The steps "
        "below are instructions for you to run; OMH runs none of them."
    ),
}

# A baseline is an exact object name. A branch or tag moves, and a baseline that
# moves cannot identify the one state someone wants back.
MAX_BASE_REVISION_CHARS = 64
MIN_BASE_REVISION_CHARS = 7
MAX_DIRTY_PATHS = 20
MAX_PATH_CHARS = 200
MAX_EVIDENCE_REFS = 8
MAX_RECIPE_STEPS = 8
MAX_RECIPE_STEP_CHARS = 200

_LABEL = "recovery_anchor"
# Built from the two bounds rather than repeating them, so the refusal message
# and the pattern cannot disagree about what an object name is.
_OBJECT_NAME = re.compile(rf"^[0-9a-f]{{{MIN_BASE_REVISION_CHARS},{MAX_BASE_REVISION_CHARS}}}$")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
# Rooted in any spelling: a drive letter, a UNC prefix, or one leading separator
# of either kind. The single backslash is the one that bites -- on Windows
# `os.path.join(os.sep, "etc", "shadow")` is `\etc\shadow`, which is root-anchored
# there but was read as project-relative here and stored verbatim.
_ROOTED_PATH = re.compile(r"^(?:[A-Za-z]:|[\\/])")


class RecoveryAnchorError(ValueError):
    """Raised when an anchor cannot be built from what a caller actually holds."""


def workspace_reference(workspace_path: str) -> str:
    """A stable, non-navigable handle for one workspace.

    Normalization is deliberately filesystem-free: `expanduser` plus `normpath`
    plus `normcase`, and no `resolve()`. Resolving would stat the tree, follow
    symlinks, and fold `/tmp` into `/private/tmp` on macOS and a short name into
    a long one on Windows, so the same logical workspace would hash differently
    depending on which machine and which link asked. The handle is machine-local
    either way; what it must be is the same value twice on one machine.
    """
    text = str(workspace_path or "").strip()
    if not text:
        return ""
    normalized = os.path.normcase(os.path.normpath(os.path.expanduser(text)))
    return digest_ref(normalized)


def build_dirty_state_digest(
    *,
    observed: bool,
    changed_paths: Iterable[str] = (),
    untracked_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Bounded dirty state: a digest, two counts, and a redacted path sample.

    The digest covers every path the caller observed, including the ones past
    the sample bound, so two workspaces that differ only beyond the twentieth
    file still produce different digests. Counts come from the lists rather than
    from the caller, so a count cannot disagree with the paths it is counting.
    """
    if not observed:
        return not_observed_dirty_state()
    changed = _unique_sorted(changed_paths)
    untracked = _unique_sorted(untracked_paths)
    combined = _unique_sorted((*changed, *untracked))
    seed = json.dumps({"changed": changed, "untracked": untracked}, sort_keys=True)
    return {
        "state": "dirty" if combined else "clean",
        "digest": f"dirty-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}",
        "changed_file_count": len(changed),
        "untracked_file_count": len(untracked),
        "paths": [_bounded_path(path) for path in combined[:MAX_DIRTY_PATHS]],
        "paths_truncated": len(combined) > MAX_DIRTY_PATHS,
    }


def not_observed_dirty_state() -> dict[str, Any]:
    """The dirty state of a workspace nobody looked at."""
    return {
        "state": "not_observed",
        "digest": "",
        "changed_file_count": 0,
        "untracked_file_count": 0,
        "paths": [],
        "paths_truncated": False,
    }


def recovery_recipe_steps(*, base_revision: str, dirty_state: Mapping[str, Any]) -> list[str]:
    """The written recipe, derived from the anchor's own facts.

    Empty without a baseline: a recipe that cannot name what to return to is not
    a recipe, and shipping a plausible-looking one for a prepared anchor is the
    exact failure this contract exists to prevent.
    """
    if not base_revision:
        return []
    state = str(dirty_state.get("state", "not_observed"))
    if state == "dirty":
        preserve = (
            "Uncommitted work was present when this baseline was recorded, so save it before undoing "
            "anything: git stash push --include-untracked"
        )
    elif state == "clean":
        preserve = "The workspace was clean when this baseline was recorded, so there is nothing to save first."
    else:
        preserve = (
            "No dirty-state observation was recorded, so assume uncommitted work may be present and save "
            "it first: git stash push --include-untracked"
        )
    return [
        "Confirm you are in the workspace this anchor names; guidance is refused for any other workspace.",
        f"See what changed since the baseline: git diff {base_revision}",
        preserve,
        f"Return tracked files to the baseline: git restore --source {base_revision} --staged --worktree .",
        "Untracked files are not part of the baseline, so the stash above is the only copy of them.",
        "OMH runs none of these commands. It performs no rollback and deletes nothing.",
    ]


def build_recovery_anchor(
    *,
    target_type: str,
    target_id: str,
    workspace_path: str,
    created_at: str,
    base_revision: str = "",
    baseline_source: str = "unknown",
    baseline_observed_at: str = "",
    dirty_state: Mapping[str, Any] | None = None,
    evidence_refs: Iterable[str] = (),
) -> dict[str, Any]:
    """Build one anchor, or refuse.

    `created_at` is a parameter rather than a clock read: an anchor is compared
    and digested, and a wall-clock value inside a compared payload is a race.
    Neither timestamp reaches the id seed.
    """
    if target_type not in ANCHOR_TARGET_TYPES:
        raise RecoveryAnchorError(f"{_LABEL} target_type is unsupported: {target_type!r}")
    if baseline_source not in BASELINE_SOURCES:
        raise RecoveryAnchorError(f"{_LABEL} baseline_source is unsupported: {baseline_source!r}")
    safe_target_id = _opaque(target_id, field="target_id")
    safe_created_at = _opaque(created_at, field="created_at")
    workspace_ref = workspace_reference(workspace_path)
    if not workspace_ref:
        raise RecoveryAnchorError(f"{_LABEL} workspace_path is required; an anchor identifies one workspace")
    revision = str(base_revision or "").strip().lower()
    if revision and not _OBJECT_NAME.fullmatch(revision):
        raise RecoveryAnchorError(
            f"{_LABEL} base_revision must be an exact object name of "
            f"{MIN_BASE_REVISION_CHARS} to {MAX_BASE_REVISION_CHARS} hex characters"
        )
    observed = bool(revision) and baseline_source in OBSERVING_BASELINE_SOURCES
    if revision and not observed:
        raise RecoveryAnchorError(
            f"{_LABEL} base_revision requires an observing baseline_source; "
            f"a revision nobody observed is not an observed baseline"
        )
    if not observed:
        # Refusing to carry a half-observed baseline is the point: every field
        # that could read as recoverable is cleared, not merely left unset.
        revision = ""
        baseline_source = "unknown"
        baseline_observed_at = ""
    safe_observed_at = _opaque(baseline_observed_at, field="baseline_observed_at") if baseline_observed_at else ""
    if observed and not safe_observed_at:
        raise RecoveryAnchorError(f"{_LABEL} an observed baseline requires baseline_observed_at")
    state = dict(dirty_state) if isinstance(dirty_state, Mapping) else not_observed_dirty_state()
    record = {
        "schema_version": RECOVERY_ANCHOR_SCHEMA_VERSION,
        "anchor_id": _anchor_id(
            target_type=target_type,
            target_id=safe_target_id,
            workspace_ref=workspace_ref,
            base_revision=revision,
            baseline_source=baseline_source,
            dirty_digest=str(state.get("digest", "")),
        ),
        "anchor_state": "baseline_observed" if observed else "prepared",
        "target_type": target_type,
        "target_id": safe_target_id,
        "workspace_ref": workspace_ref,
        "base_revision": revision,
        "baseline_source": baseline_source,
        "baseline_observed_at": safe_observed_at,
        "dirty_state": state,
        "recovery_recipe": recovery_recipe_steps(base_revision=revision, dirty_state=state),
        "evidence_refs": _bounded_refs(evidence_refs),
        "created_at": safe_created_at,
        "privacy": RECOVERY_ANCHOR_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    errors = validate_recovery_anchor(record)
    if errors:
        raise RecoveryAnchorError(errors[0])
    return record


def build_recovery_guidance(anchor: Mapping[str, Any] | None, *, workspace_path: str) -> dict[str, Any]:
    """Answer "how do I undo this safely?" for one workspace, or refuse and say why.

    Reads no clock and performs no action. Every path out of here that is not
    `baseline_available` returns no steps and `recoverable` false.
    """
    requested_ref = workspace_reference(workspace_path)
    if not isinstance(anchor, Mapping) or not anchor:
        return _guidance("anchor_absent", anchor_id="", workspace_ref=requested_ref)
    # Rendering reads whatever a caller holds, which may have been hand-edited,
    # so the id is folded to a bounded handle before it reaches a reader.
    anchor_id = _rendered_ref(anchor.get("anchor_id", ""))
    if validate_recovery_anchor(dict(anchor)):
        return _guidance("anchor_invalid", anchor_id=anchor_id, workspace_ref=requested_ref)
    if str(anchor.get("workspace_ref", "")) != requested_ref or not requested_ref:
        return _guidance("workspace_mismatch", anchor_id=anchor_id, workspace_ref=requested_ref)
    if str(anchor.get("anchor_state", "")) != "baseline_observed":
        return _guidance("baseline_not_observed", anchor_id=anchor_id, workspace_ref=requested_ref)
    return _guidance(
        "baseline_available",
        anchor_id=anchor_id,
        workspace_ref=requested_ref,
        base_revision=str(anchor.get("base_revision", "")),
        dirty_state=dict(anchor.get("dirty_state", {})),
        steps=[str(step) for step in anchor.get("recovery_recipe", [])],
    )


def validate_recovery_anchor(payload: dict[str, Any]) -> list[str]:
    """Every reason one payload is not a `recovery_anchor/v1` record."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{_LABEL} must be an object"]
    forbidden = sorted(key for key in payload if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    # Both directions: an unsupported key is a field nothing guards, and a
    # missing key is a field a reader will treat as absent rather than as broken.
    extra_keys = sorted(set(payload) - set(RECOVERY_ANCHOR_KEYS) - set(forbidden))
    if extra_keys:
        errors.append(f"{_LABEL} has unsupported keys: {extra_keys}")
    missing = sorted(set(RECOVERY_ANCHOR_KEYS) - set(payload))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if payload.get("schema_version") != RECOVERY_ANCHOR_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {RECOVERY_ANCHOR_SCHEMA_VERSION}")
    for key in ("anchor_id", "anchor_state", "base_revision", "baseline_observed_at", "baseline_source",
                "created_at", "target_id", "target_type", "workspace_ref"):
        if not isinstance(payload.get(key), str):
            errors.append(f"{_LABEL} {key} must be a string")
    if payload.get("target_type") not in ANCHOR_TARGET_TYPES:
        errors.append(f"{_LABEL} target_type is unsupported: {payload.get('target_type')!r}")
    if payload.get("anchor_state") not in ANCHOR_STATES:
        errors.append(f"{_LABEL} anchor_state is unsupported: {payload.get('anchor_state')!r}")
    if payload.get("baseline_source") not in BASELINE_SOURCES:
        errors.append(f"{_LABEL} baseline_source is unsupported: {payload.get('baseline_source')!r}")
    if payload.get("privacy") != RECOVERY_ANCHOR_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {RECOVERY_ANCHOR_PRIVACY}")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(f"{_LABEL} claim_boundary must state the recovery boundary")
    for field in ("anchor_id", "target_id", "created_at", "workspace_ref"):
        errors.extend(reference_errors(payload.get(field), field=field, label=_LABEL, required=True))
    errors.extend(
        reference_errors(payload.get("baseline_observed_at"), field="baseline_observed_at", label=_LABEL, required=False)
    )
    errors.extend(_evidence_ref_errors(payload.get("evidence_refs")))
    errors.extend(_dirty_state_errors(payload.get("dirty_state")))
    errors.extend(_recipe_errors(payload.get("recovery_recipe")))
    errors.extend(_baseline_state_errors(payload))
    return errors


def validate_recovery_guidance(payload: dict[str, Any]) -> list[str]:
    """Every reason one payload is not a `recovery_guidance/v1` answer.

    The load-bearing check is the last one: `recoverable` is compared against
    `RECOVERABLE_GUIDANCE_STATUSES`, so a hand-built or drifted answer that
    claims recoverability from an absent, invalid, prepared-only, or mismatched
    anchor is a validation error rather than a status somebody believes.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["recovery_guidance must be an object"]
    extra_keys = sorted(set(payload) - set(RECOVERY_GUIDANCE_KEYS))
    if extra_keys:
        errors.append(f"recovery_guidance has unsupported keys: {extra_keys}")
    missing = sorted(set(RECOVERY_GUIDANCE_KEYS) - set(payload))
    if missing:
        errors.append(f"recovery_guidance is missing keys: {missing}")
    if payload.get("schema_version") != RECOVERY_GUIDANCE_SCHEMA_VERSION:
        errors.append(f"recovery_guidance schema_version must be {RECOVERY_GUIDANCE_SCHEMA_VERSION}")
    status = payload.get("status")
    if status not in RECOVERY_GUIDANCE_STATUSES:
        errors.append(f"recovery_guidance status is unsupported: {status!r}")
    if payload.get("performs_rollback") is not False:
        errors.append("recovery_guidance performs_rollback must be false; OMH performs no rollback")
    if payload.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append("recovery_guidance claim_boundary must state the recovery boundary")
    if not str(payload.get("reason", "")).strip():
        errors.append("recovery_guidance reason is required")
    errors.extend(_recipe_errors(payload.get("steps"), label="recovery_guidance", field="steps"))
    errors.extend(_dirty_state_errors(payload.get("dirty_state"), label="recovery_guidance"))
    recoverable = payload.get("recoverable")
    if not isinstance(recoverable, bool):
        errors.append("recovery_guidance recoverable must be a boolean")
    elif recoverable:
        if status not in RECOVERABLE_GUIDANCE_STATUSES:
            errors.append(f"recovery_guidance status {status!r} must not report recoverable")
        if not str(payload.get("base_revision", "")).strip():
            errors.append("recovery_guidance recoverable requires a base_revision naming the baseline")
        if not payload.get("steps"):
            errors.append("recovery_guidance recoverable requires the steps it says the reader can run")
    else:
        if payload.get("steps"):
            errors.append("recovery_guidance that is not recoverable must carry no steps")
        if str(payload.get("base_revision", "")).strip():
            errors.append("recovery_guidance that is not recoverable must name no base_revision")
    return errors


def _guidance(
    status: str,
    *,
    anchor_id: str,
    workspace_ref: str,
    base_revision: str = "",
    dirty_state: dict[str, Any] | None = None,
    steps: list[str] | None = None,
) -> dict[str, Any]:
    recoverable = status in RECOVERABLE_GUIDANCE_STATUSES
    return {
        "schema_version": RECOVERY_GUIDANCE_SCHEMA_VERSION,
        "status": status,
        "recoverable": recoverable,
        "reason": GUIDANCE_REASONS[status],
        "anchor_id": anchor_id,
        "workspace_ref": workspace_ref,
        "base_revision": base_revision if recoverable else "",
        # A refusal echoes nothing about the anchor's own workspace. On a
        # mismatch that state describes a checkout the asker is not in.
        "dirty_state": dict(dirty_state or {}) if recoverable else not_observed_dirty_state(),
        "steps": list(steps or []) if recoverable else [],
        "performs_rollback": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _baseline_state_errors(payload: dict[str, Any]) -> list[str]:
    """The prepared/observed split, checked in both directions.

    Either half alone would let the other drift: without the first, a prepared
    anchor could carry a recipe; without the second, an anchor could claim an
    observed baseline while naming nothing to return to.
    """
    errors: list[str] = []
    state = payload.get("anchor_state")
    revision = str(payload.get("base_revision", ""))
    if state == "prepared":
        if revision:
            errors.append(f"{_LABEL} prepared anchor must carry no base_revision")
        if payload.get("baseline_source") != "unknown":
            errors.append(f"{_LABEL} prepared anchor baseline_source must be unknown")
        if payload.get("baseline_observed_at"):
            errors.append(f"{_LABEL} prepared anchor must carry no baseline_observed_at")
        if payload.get("recovery_recipe"):
            errors.append(f"{_LABEL} prepared anchor must carry no recovery_recipe")
    elif state == "baseline_observed":
        if not _OBJECT_NAME.fullmatch(revision):
            errors.append(f"{_LABEL} observed baseline requires base_revision as an exact object name")
        if payload.get("baseline_source") not in OBSERVING_BASELINE_SOURCES:
            errors.append(f"{_LABEL} observed baseline requires an observing baseline_source")
        if not str(payload.get("baseline_observed_at", "")).strip():
            errors.append(f"{_LABEL} observed baseline requires baseline_observed_at")
        if not payload.get("recovery_recipe"):
            errors.append(f"{_LABEL} observed baseline requires a recovery_recipe")
    return errors


def _dirty_state_errors(value: Any, *, label: str = _LABEL) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} dirty_state must be an object"]
    errors: list[str] = []
    extra_keys = sorted(set(value) - set(DIRTY_STATE_KEYS))
    if extra_keys:
        errors.append(f"{label} dirty_state has unsupported keys: {extra_keys}")
    missing = sorted(set(DIRTY_STATE_KEYS) - set(value))
    if missing:
        errors.append(f"{label} dirty_state is missing keys: {missing}")
    if value.get("state") not in DIRTY_STATE_STATES:
        errors.append(f"{label} dirty_state state is unsupported: {value.get('state')!r}")
    for key in ("changed_file_count", "untracked_file_count"):
        count = value.get(key)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            errors.append(f"{label} dirty_state {key} must be a non-negative integer")
    if not isinstance(value.get("paths_truncated"), bool):
        errors.append(f"{label} dirty_state paths_truncated must be a boolean")
    paths = value.get("paths")
    if not isinstance(paths, list):
        errors.append(f"{label} dirty_state paths must be a list")
    elif len(paths) > MAX_DIRTY_PATHS:
        errors.append(f"{label} dirty_state paths must hold at most {MAX_DIRTY_PATHS} entries")
    else:
        for index, path in enumerate(paths):
            if not isinstance(path, str):
                errors.append(f"{label} dirty_state paths[{index}] must be a string")
            elif path != _bounded_path(path):
                errors.append(f"{label} dirty_state paths[{index}] must be a bounded, redacted path")
    if value.get("state") == "not_observed" and (value.get("digest") or paths):
        errors.append(f"{label} dirty_state not_observed must carry no digest and no paths")
    if value.get("state") != "not_observed" and not str(value.get("digest", "")).strip():
        errors.append(f"{label} dirty_state requires a digest once it is observed")
    return errors


def _recipe_errors(value: Any, *, label: str = _LABEL, field: str = "recovery_recipe") -> list[str]:
    if not isinstance(value, list):
        return [f"{label} {field} must be a list"]
    errors: list[str] = []
    if len(value) > MAX_RECIPE_STEPS:
        errors.append(f"{label} {field} must hold at most {MAX_RECIPE_STEPS} steps")
    for index, step in enumerate(value):
        if not isinstance(step, str):
            errors.append(f"{label} {field}[{index}] must be a string")
        elif not step.strip() or len(step) > MAX_RECIPE_STEP_CHARS:
            errors.append(f"{label} {field}[{index}] must be one bounded instruction line")
        elif _CONTROL_CHARS.search(step) or is_sensitive_metadata_text(step) or is_url_shaped(step):
            errors.append(f"{label} {field}[{index}] must be a plain instruction line")
    return errors


def _evidence_ref_errors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return [f"{_LABEL} evidence_refs must be a list"]
    errors: list[str] = []
    if len(value) > MAX_EVIDENCE_REFS:
        errors.append(f"{_LABEL} evidence_refs must hold at most {MAX_EVIDENCE_REFS} entries")
    for index, ref in enumerate(value):
        errors.extend(reference_errors(ref, field=f"evidence_refs[{index}]", label=_LABEL, required=True))
    return errors


def _anchor_id(
    *,
    target_type: str,
    target_id: str,
    workspace_ref: str,
    base_revision: str,
    baseline_source: str,
    dirty_digest: str,
) -> str:
    """Identity from what the anchor is about, never from when it was taken."""
    seed = json.dumps(
        {
            "target_type": target_type,
            "target_id": target_id,
            "workspace_ref": workspace_ref,
            "base_revision": base_revision,
            "baseline_source": baseline_source,
            "dirty_digest": dirty_digest,
        },
        sort_keys=True,
    )
    return f"anchor-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _bounded_path(value: str) -> str:
    """One project-relative path, or a digest handle where it is not one.

    Absolute, home-anchored, escaping, over-long, secret-shaped, URL-shaped, and
    control-carrying values all fold to a handle instead of being stored. The
    function is idempotent, which is what lets the validator re-derive it and
    reject a stored path that did not come through here.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) > MAX_PATH_CHARS:
        return digest_ref(text)
    if _CONTROL_CHARS.search(text) or is_sensitive_metadata_text(text) or is_url_shaped(text):
        return digest_ref(text)
    if text.startswith("~") or _ROOTED_PATH.match(text):
        return digest_ref(text)
    if ".." in re.split(r"[\\/]", text):
        return digest_ref(text)
    return text


def _bounded_refs(refs: Iterable[str]) -> list[str]:
    bounded: list[str] = []
    for ref in refs:
        text = str(ref or "").strip()
        if not text or text in bounded:
            continue
        bounded.append(_opaque(text, field="evidence_refs"))
        if len(bounded) == MAX_EVIDENCE_REFS:
            break
    return bounded


def _rendered_ref(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if reference_errors(text, field="anchor_id", label=_LABEL, required=True):
        return digest_ref(text)
    return text


def _opaque(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    errors = reference_errors(text, field=field, label=_LABEL, required=True)
    if errors:
        raise RecoveryAnchorError(errors[0])
    return text


def _unique_sorted(values: Iterable[str]) -> list[str]:
    cleaned = {str(value).strip() for value in values if str(value).strip()}
    return sorted(cleaned)
