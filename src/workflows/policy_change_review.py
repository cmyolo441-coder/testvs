"""Review one OMH policy or routing change before it becomes the active behavior (#798).

The defect this closes: a change to OMH clarification, routing, handoff, or
verification policy is a behavior change for every intent that policy touches,
and the ones it improves are the ones the author was looking at. The ones it
regresses are the others. Nothing in the tree made a maintainer say, before
rollout, which intents move, which way each of them moves, which coding owners
the change re-scopes, or what evidence would show the change landed without
taking something else down with it.

`policy_change_review/v1` is that statement. One review, for one policy surface,
bound to the baseline it was measured against.

Why a sibling and not an extension
----------------------------------

The nearest neighbour is `improvement_patch_proposal/v1` in
:mod:`workflows.workflow_learning`, and it was the first thing tried. It already
carries a review gate, a regression gate with a replay status, a
`patch_scope.mode` of `proposal_only`, and a target type that includes
`routing`. Extending it was rejected for two reasons that are structural rather
than stylistic:

* **Its subject is different.** An improvement patch proposal is *derived*: it
  exists only downstream of a learning trace, an eval result, and a candidate,
  and its identity keys (`trace_id`, `eval_id`, `candidate_id`) are all
  required. It answers "one observed workflow went wrong; what should change?".
  A policy change review is *authored*: a maintainer proposes a change with no
  trace behind it, and demanding a synthetic trace to record one would make the
  learning family lie about where the proposal came from.
* **Its lifecycle is different.** The patch proposal's status ladder is a
  function of the candidate's decision and a regression replay, and it stops at
  `ready_for_human_patch` -- by design, since `apply_path_available` is `False`
  and the change owner is a later human PR. #798 needs the state *after* that
  handoff: a rollout gated on a matching baseline, observed repository changes,
  and named regression evidence. Bolting a second, differently-gated ladder onto
  a frozen v1 schema read by the export bundle, the learning index, and the
  review queue would change the meaning of a record several surfaces already
  parse.

What is reused instead of reinvented: the hashing scheme is
``quality.skill_governance.policy_decision_digest`` for both the baseline digest
and the review digest, the coding-owner vocabulary is
``coding.executors.EXECUTOR_PROFILES`` rather than a second list of owners, and
the derived-status-plus-content-digest shape is the one
`workflows.skill_pattern_risk_review` established for the same reason.

Not the capability toggle
-------------------------

`omh_capability_policy_change/v1` in `commands.capability_policy` is a different
thing that shares a word. It records that one capability *family* was switched
on or off for one local install: which family, the policy before, the policy
after, and the reverse command. It says nothing about behavior, has no reviewer,
no baseline binding, no intents, and no rollout state, and it applies itself the
moment it is invoked. A review is the opposite: it changes nothing, and it
exists to be argued with first.

AC1 -- every affected intent, with both sides
---------------------------------------------

``intents`` is a list of rows, each naming one intent and both of its behaviors.
A row that states only a before, or only an after, is a validation error naming
the intent and the side it is missing, because one side of a behavior change is
not reviewable: "clarification asks a question here" is a fact about the world,
not a change to it.

An intent whose before and after read the same is deliberately *allowed*. The
user problem is silent regression, and recording "I checked this one and it does
not move" is how a review shows the check happened.

AC2 -- a rejected or superseded review is structurally not the active policy
-----------------------------------------------------------------------------

``review_state`` is **derived**, never passed. :func:`build_policy_change_review`
has no state parameter, no decision parameter, and no rollout parameter, so
there is no argument through which a proposal can assert that it is in force,
and :func:`validate_policy_change_review` re-derives the state and refuses a
payload that disagrees -- a hand-written dict cannot lie either.

Three separate things keep an inactive review inactive:

* ``superseded_by_review_ref`` is the supersession link, and it dominates every
  other input in :func:`derive_review_state`. A review a later review replaced
  reads as ``superseded`` whatever it was decided to be, and
  :func:`record_policy_change_rollout` refuses to touch it at all.
* A rejection is a recorded decision like any other, and ``rejected`` has no
  edge to ``applied``.
* A decision is bound to the content it covers. ``reviewed_digest`` stamps
  ``review_digest``, so material content that moves after the decision leaves
  the review at ``superseded_by_content_change`` rather than carrying the old
  approval onto text nobody read.

:func:`review_reads_as_active` is the one question a surface asks, and
:data:`ACTIVE_REVIEW_STATE` is the only member of :data:`REVIEW_STATES` it is
true for. :data:`REFUSED_REVIEW_STATES` holds the bare words -- ``active``,
``live``, ``in_effect``, ``rolled_out`` -- by name, so a payload asserting one of
them is refused as a rule rather than as an unrecognised value.

AC3 -- applied needs all three, or it is not applied
-----------------------------------------------------

:data:`ROLLOUT_REQUIREMENTS` names them: the observed baseline digest equals the
baseline the review was bound to, at least one repository change was observed,
and the named regression evidence is present and passing. All three, together,
and only from an approved review whose content has not moved. Any one missing
and :func:`derive_review_state` returns ``approved_not_rolled_out``;
:func:`unmet_rollout_requirements` says which, and the validator repeats it in
the error so a refusal is actionable rather than a verdict.

The baseline is the reason the first requirement exists. A review argues about
one specific policy; if the policy in force at rollout time is not the one that
was reviewed, the review does not describe the world and its approval covers
nothing. Both digests come from :func:`policy_change_baseline_digest`.

What OMH did not do
-------------------

Nothing here reads the repository, edits policy source, regenerates a skill,
changes a route, or replays a regression. ``not_observed`` says so per surface.
The rollout block records what a *caller* observed and is only as good as the
surface that supplied it -- which is why the requirement is that the evidence be
named and matched, not that this module went and looked.

Determinism
-----------

No clock is read. ``prepared_at`` is a caller parameter and is excluded from
``review_digest``, because a wall-clock value inside a compared digest turns an
equality check into a race that a slow CI worker loses. ``reviewer_decision``,
``rollout``, ``superseded_by_review_ref``, and ``review_state`` are excluded too:
recording a decision must not move the digest that decision names, and recording
rollout evidence must not invalidate the approval it rests on.

What reads this today
---------------------

Stated because the contract is wider than the wiring. No production surface
mints a review yet; this module is the contract and
``tests/test_policy_change_review.py`` is its only caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..coding.executors import EXECUTOR_PROFILES
from ..quality.skill_governance import policy_decision_digest
from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    is_unsafe_metadata_line,
    opaque_ref,
    reference_errors,
)


POLICY_CHANGE_REVIEW_SCHEMA_VERSION: Final = "policy_change_review/v1"

REVIEW_PRIVACY: Final = "metadata_only"

# The four surfaces #798 scopes this family to. Closed, and short on purpose:
# each one is a place where OMH decides how it will behave toward a user intent,
# and none of them is a place where content is authored.
POLICY_SURFACES: Final = ("clarification", "routing", "handoff", "verification")

# Named so the boundary refuses by name instead of as an unrecognised value.
# Turning an observed user workflow into a reusable skill is the teach-workflow
# capability's job (#838), and a review of a *policy* surface is not the record
# for it.
OUT_OF_SCOPE_SURFACES: Final = ("skill", "skill_creation", "reusable_skill", "teach_workflow")

# Derived from the delegation vocabulary, never restated. A second list of
# coding owners here would be a second thing to keep in agreement with
# `coding.executors`, and the first time they disagreed a review would answer
# for an owner that no longer exists or miss one that does.
POLICY_CHANGE_OWNERS: Final[tuple[str, ...]] = EXECUTOR_PROFILES

# What a policy change does to one coding owner's boundary. Closed so a later
# gate can join on a member instead of parsing a sentence.
# `no_boundary_change` is a real answer rather than an omission, which is what
# keeps "this owner is unaffected" and "nobody considered this owner" apart.
OWNER_BOUNDARY_CHANGES: Final = (
    "handoff_scope_widened",
    "handoff_scope_narrowed",
    "clarification_required_earlier",
    "clarification_no_longer_required",
    "verification_evidence_required",
    "verification_evidence_relaxed",
    "routing_reaches_this_owner",
    "routing_no_longer_reaches_this_owner",
    "no_boundary_change",
)

# A regression case has a result; a repository change is simply a fact that
# happened. That asymmetry is why only one of the two rollout lists carries
# rows. There is no `not_run` member: evidence that was not run is not evidence,
# and leaving it out of the list says so more clearly than naming it.
REGRESSION_OUTCOMES: Final = ("passed", "failed")

# Both are about the proposed change, and neither is about the source of it.
REVIEWER_DECISIONS: Final = ("approve_policy_change", "reject_policy_change")

# The lifecycle of a review. Exactly one member says the change is in force.
REVIEW_STATES: Final = (
    "awaiting_review",
    "rejected",
    "superseded",
    "superseded_by_content_change",
    "approved_not_rolled_out",
    "applied",
)

# The single member of REVIEW_STATES under which this review's after-behavior is
# the behavior now in force.
ACTIVE_REVIEW_STATE: Final = "applied"

# Words a state will not be, held by name so the refusal states a rule. Every
# one of them asserts that routing or guidance is already live, which is a claim
# about the running system that a review is not in a position to make.
REFUSED_REVIEW_STATES: Final = (
    "active",
    "current",
    "deployed",
    "enabled",
    "in_effect",
    "live",
    "merged",
    "passed",
    "released",
    "rolled_out",
    "shipped",
    "verified",
)

# The one sentence each state may be rendered with. Every one says what did or
# did not change in active behavior, so no rendering of any state can read as a
# rollout OMH performed.
REVIEW_STATE_CLAIMS: Final = {
    "awaiting_review": (
        "A policy change is proposed and no reviewer has decided yet. Active routing and guidance are "
        "unchanged."
    ),
    "rejected": (
        "A reviewer rejected the proposed change. It is not the active policy and must not alter routing "
        "or guidance."
    ),
    "superseded": (
        "A later review replaced this one. It is not the active policy, whatever it was decided to be."
    ),
    "superseded_by_content_change": (
        "The reviewed content moved after the decision, so the decision no longer covers it. It is not the "
        "active policy."
    ),
    "approved_not_rolled_out": (
        "A reviewer approved the change and its rollout evidence is incomplete. Active routing and "
        "guidance are unchanged."
    ),
    "applied": (
        "An approved change was rolled out against the matching baseline, with observed repository changes "
        "and named passing regression evidence. OMH neither made nor verified those changes; it recorded "
        "that a caller observed them."
    ),
}

# How a recorded decision stands against the content now in the review.
REVIEWER_DECISION_STATES: Final = ("absent", "current", "stale")

# #798 AC3, as three names a caller can be told it is missing.
ROLLOUT_REQUIREMENTS: Final = (
    "baseline_matches",
    "repository_changes_observed",
    "regression_evidence_named",
)

# What OMH did not do to the policy under review, per surface.
NOT_OBSERVED_SURFACES: Final = (
    "policy_source_edit",
    "routing_behavior_change",
    "clarification_behavior_change",
    "regression_replay",
    "test_execution",
    "repository_write",
)

BASELINE_KEYS: Final = ("baseline_ref", "baseline_digest")
INTENT_KEYS: Final = ("intent_ref", "before", "after")
OWNER_BOUNDARY_KEYS: Final = ("owner", "boundary_change")
REGRESSION_EVIDENCE_KEYS: Final = ("case_ref", "outcome")
ROLLOUT_KEYS: Final = (
    "observed_baseline_digest",
    "observed_repository_changes",
    "regression_evidence",
)
REVIEWER_DECISION_KEYS: Final = ("decided_by", "decision", "reviewed_digest")

POLICY_CHANGE_REVIEW_KEYS: Final = (
    "schema_version",
    "review_ref",
    "policy_surface",
    "baseline",
    "intents",
    "owner_boundaries",
    "not_observed",
    "reviewer_decision",
    "rollout",
    "superseded_by_review_ref",
    "review_state",
    "prepared_at",
    "privacy",
    "review_digest",
    "claim_boundary",
)

# The review's own lifecycle: outside the digest, because recording a decision
# must not move the digest that decision names, and recording the rollout
# evidence an approval asked for must not invalidate the approval.
REVIEW_STATE_KEYS: Final = (
    "reviewer_decision",
    "rollout",
    "superseded_by_review_ref",
    "review_state",
)

# A value that cannot cover itself, two module constants identical in every
# review, and the one field that holds a clock.
DERIVED_REVIEW_KEYS: Final = ("review_digest", "claim_boundary", "privacy", "prepared_at")

# Everything else. Derived rather than listed so a key added to the schema is
# material by default and has to be argued out, never silently left out.
MATERIAL_REVIEW_KEYS: Final = tuple(
    key
    for key in POLICY_CHANGE_REVIEW_KEYS
    if key not in set(REVIEW_STATE_KEYS) | set(DERIVED_REVIEW_KEYS)
)

# Key names under which "the change is already live" arrives. The closed key set
# already refuses every one of them, but it refuses them as unsupported keys,
# and the point of this family is that proposing a change is not making one.
ROLLOUT_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "active",
        "applied_at",
        "ci_passed",
        "deployed",
        "enabled",
        "executed",
        "in_effect",
        "live",
        "merged",
        "passed",
        "released",
        "rolled_out",
        "routing_changed",
        "shipped",
        "succeeded",
        "success",
        "verified",
    }
)

POLICY_CHANGE_REVIEW_CLAIM_BOUNDARY: Final = (
    "This review records a proposed change to one OMH clarification, routing, handoff, or verification "
    "policy surface: the intents it moves, both sides of each of those behaviors, the coding-owner "
    "boundaries it shifts, and a reviewer's decision. OMH does not edit policy source, regenerate skills, "
    "change routing, read the repository, or replay a regression here. Every observation in the rollout "
    "block is supplied by the caller and is only as good as the surface that supplied it. A rejected or "
    "superseded review is never the active policy, and the review is not execution, review, CI, "
    "merge-readiness, or merge evidence."
)

_MAX_BEHAVIOR_CHARS: Final = 240
_MAX_STAMP_CHARS: Final = 64
_MAX_INTENTS: Final = 24
_MAX_OBSERVED_CHANGES: Final = 12
_MAX_REGRESSION_ROWS: Final = 24

_LABEL: Final = "policy change review"
_SHA256_LENGTH: Final = 64


class PolicyChangeReviewError(ValueError):
    """Raised when a review cannot be built, decided, superseded, or rolled out."""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_policy_change_review(
    *,
    review_ref: str,
    policy_surface: str,
    baseline_ref: str,
    baseline_policy: Mapping[str, Any],
    intents: Sequence[Mapping[str, Any]],
    owner_boundaries: Sequence[Mapping[str, Any]],
    prepared_at: str = "",
) -> dict[str, Any]:
    """Mint one proposed policy change review, or refuse.

    There is no `review_state` parameter, no `reviewer_decision` parameter, and
    no `rollout` parameter. A freshly minted review is a proposal and nothing
    else, which is what makes "a proposal cannot read as the active policy" a
    property of the constructor rather than a rule a caller is asked to respect.
    The three later transitions each have their own function, and each of them
    re-derives the state.
    """
    review: dict[str, Any] = {
        "schema_version": POLICY_CHANGE_REVIEW_SCHEMA_VERSION,
        "review_ref": _opaque(review_ref, field=f"{_LABEL} review_ref"),
        "policy_surface": _policy_surface(policy_surface),
        "baseline": {
            "baseline_ref": _opaque(baseline_ref, field=f"{_LABEL} baseline.baseline_ref"),
            "baseline_digest": policy_change_baseline_digest(baseline_policy),
        },
        "intents": _intent_rows(intents),
        "owner_boundaries": _owner_boundary_rows(owner_boundaries),
        "not_observed": {surface: {"status": "not_observed"} for surface in NOT_OBSERVED_SURFACES},
        "prepared_at": _optional_line(prepared_at, field=f"{_LABEL} prepared_at", limit=_MAX_STAMP_CHARS),
        "privacy": REVIEW_PRIVACY,
        "claim_boundary": POLICY_CHANGE_REVIEW_CLAIM_BOUNDARY,
    }
    review["review_digest"] = policy_change_review_digest(review)
    review["reviewer_decision"] = {}
    review["rollout"] = empty_rollout()
    review["superseded_by_review_ref"] = ""
    review["review_state"] = derive_review_state(review)
    _raise_on(validate_policy_change_review(review))
    return review


def policy_change_baseline_digest(baseline_policy: Mapping[str, Any]) -> str:
    """The digest that binds a review to the policy it argues about.

    Reuses `quality.skill_governance.policy_decision_digest`: it is already this
    repository's stable, order-independent content hash over a policy mapping,
    and a second encoding rule here would be a second thing to keep correct.
    The same function produces the digest recorded at rollout time, so the
    baseline comparison is between two values derived the same way.
    """
    if not isinstance(baseline_policy, Mapping) or not baseline_policy:
        raise PolicyChangeReviewError(
            f"{_LABEL} baseline_policy must be a non-empty object describing the policy in force; a change "
            "measured against nothing cannot be reviewed"
        )
    return policy_decision_digest(dict(baseline_policy))


def empty_rollout() -> dict[str, Any]:
    """The rollout block of a review nobody has rolled out.

    Present with all three keys rather than absent, because #798 AC3 is three
    independent requirements and each of them has to be individually missing.
    """
    return {
        "observed_baseline_digest": "",
        "observed_repository_changes": [],
        "regression_evidence": [],
    }


def record_policy_change_decision(
    review: Mapping[str, Any], *, decided_by: str, decision: str
) -> dict[str, Any]:
    """Bind one reviewer's decision to the content that reviewer saw.

    The digest of the material content in front of the reviewer is stamped onto
    the decision, so content that moves afterwards leaves the review superseded
    rather than quietly carrying an approval it outgrew.

    A superseded review is not re-decided: a later review already replaced it,
    and a decision recorded on it now would be a decision about text nothing
    reads.
    """
    _raise_on(validate_policy_change_review(review))
    if str(review.get("superseded_by_review_ref", "") or ""):
        raise PolicyChangeReviewError(
            f"{_LABEL} cannot record a decision on a review superseded by "
            f"{review.get('superseded_by_review_ref')!r}; decide the review that replaced it"
        )
    decided = dict(review)
    decided["reviewer_decision"] = {
        "decided_by": _opaque(decided_by, field=f"{_LABEL} reviewer_decision.decided_by"),
        "decision": _closed(
            decision, allowed=REVIEWER_DECISIONS, field=f"{_LABEL} reviewer_decision.decision"
        ),
        "reviewed_digest": policy_change_review_digest(review),
    }
    decided["review_state"] = derive_review_state(decided)
    _raise_on(validate_policy_change_review(decided))
    return decided


def record_policy_change_rollout(
    review: Mapping[str, Any],
    *,
    observed_baseline_digest: str = "",
    observed_repository_changes: Sequence[str] = (),
    regression_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Attach the evidence a rollout rests on to an approved review.

    Partial evidence is accepted on purpose: evidence arrives in pieces, and a
    review holding two of the three requirements is a useful record of exactly
    what is still outstanding. It simply does not read as `applied` --
    :func:`derive_review_state` decides that, here and in the validator, from
    the evidence rather than from having been called.

    What is refused is attaching evidence at all to a review that no current
    approval covers. That refusal is a convenience, not the safety property:
    even a hand-written payload carrying complete rollout evidence under a
    rejection derives `rejected`.
    """
    _raise_on(validate_policy_change_review(review))
    if str(review.get("superseded_by_review_ref", "") or ""):
        raise PolicyChangeReviewError(
            f"{_LABEL} cannot roll out a review superseded by "
            f"{review.get('superseded_by_review_ref')!r}; roll out the review that replaced it"
        )
    state = derive_review_state(review)
    if state not in {"approved_not_rolled_out", ACTIVE_REVIEW_STATE}:
        raise PolicyChangeReviewError(
            f"{_LABEL} rollout evidence may only be recorded on a review a reviewer approved and whose "
            f"content has not moved since; this review reads {state!r}"
        )
    rolled = dict(review)
    rolled["rollout"] = {
        "observed_baseline_digest": _optional_digest(
            observed_baseline_digest, field=f"{_LABEL} rollout.observed_baseline_digest"
        ),
        "observed_repository_changes": _ref_list(
            observed_repository_changes,
            field=f"{_LABEL} rollout.observed_repository_changes",
            maximum=_MAX_OBSERVED_CHANGES,
        ),
        "regression_evidence": _regression_rows(regression_evidence),
    }
    rolled["review_state"] = derive_review_state(rolled)
    _raise_on(validate_policy_change_review(rolled))
    return rolled


def supersede_policy_change_review(
    review: Mapping[str, Any], *, superseded_by_review_ref: str
) -> dict[str, Any]:
    """Link one review to the review that replaced it.

    The link points forward, from the replaced review to its replacement, so
    whether a review is still the active policy is answerable from that review
    alone and does not require holding the whole sequence.
    """
    _raise_on(validate_policy_change_review(review))
    successor = _opaque(superseded_by_review_ref, field=f"{_LABEL} superseded_by_review_ref")
    if successor == str(review.get("review_ref", "") or ""):
        raise PolicyChangeReviewError(
            f"{_LABEL} superseded_by_review_ref must not name the review itself: {successor}"
        )
    superseded = dict(review)
    superseded["superseded_by_review_ref"] = successor
    superseded["review_state"] = derive_review_state(superseded)
    _raise_on(validate_policy_change_review(superseded))
    return superseded


# ---------------------------------------------------------------------------
# Derived state
# ---------------------------------------------------------------------------


def policy_change_review_digest(review: Mapping[str, Any]) -> str:
    """Content hash of the material part of a review."""
    return policy_decision_digest(material_review_content(review))


def material_review_content(review: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a review the digest covers.

    A missing key projects as `None` rather than raising, so a malformed review
    still has a digest and the validator -- not the hash -- reports the fault.
    """
    return {key: review.get(key) for key in MATERIAL_REVIEW_KEYS}


def reviewer_decision_state(review: Mapping[str, Any]) -> str:
    """Whether a decision exists and still covers the content now in the review."""
    decision = review.get("reviewer_decision")
    if not isinstance(decision, Mapping) or not decision:
        return "absent"
    reviewed = str(decision.get("reviewed_digest", "") or "")
    return "current" if reviewed and reviewed == policy_change_review_digest(review) else "stale"


def rollout_requirements(review: Mapping[str, Any]) -> dict[str, bool]:
    """#798 AC3, evaluated: the matching baseline, observed changes, named evidence.

    Named regression evidence means at least one named case whose recorded
    outcome is `passed` and no named case that failed. A list carrying a failure
    is evidence, and what it is evidence of is that the change is not ready.
    """
    rollout = review.get("rollout")
    rollout = rollout if isinstance(rollout, Mapping) else {}
    baseline = review.get("baseline")
    expected = str(baseline.get("baseline_digest", "") or "") if isinstance(baseline, Mapping) else ""
    observed = str(rollout.get("observed_baseline_digest", "") or "")
    changes = rollout.get("observed_repository_changes")
    evidence = rollout.get("regression_evidence")
    rows = list(evidence) if _is_row_sequence(evidence) else []
    return {
        "baseline_matches": bool(expected) and observed == expected,
        "repository_changes_observed": isinstance(changes, list) and bool(changes),
        "regression_evidence_named": bool(rows)
        and all(str(row.get("outcome", "")) == "passed" for row in rows),
    }


def unmet_rollout_requirements(review: Mapping[str, Any]) -> tuple[str, ...]:
    """Which of #798 AC3's three requirements this review does not yet meet."""
    met = rollout_requirements(review)
    return tuple(name for name in ROLLOUT_REQUIREMENTS if not met.get(name, False))


def derive_review_state(review: Mapping[str, Any]) -> str:
    """The only definition of a review's state.

    Read it as #798 AC2 and AC3 in code. Supersession dominates, because a
    review something else replaced is not the active policy whatever it holds. A
    decision that no longer covers the content is treated as superseded by that
    content, for the same reason. Only an approval that is current reaches the
    rollout gate, and only a complete gate reaches `applied`.
    """
    if str(review.get("superseded_by_review_ref", "") or ""):
        return "superseded"
    state = reviewer_decision_state(review)
    if state == "absent":
        return "awaiting_review"
    if state == "stale":
        return "superseded_by_content_change"
    decision = review.get("reviewer_decision")
    verdict = decision.get("decision") if isinstance(decision, Mapping) else ""
    if verdict != "approve_policy_change":
        return "rejected"
    return ACTIVE_REVIEW_STATE if not unmet_rollout_requirements(review) else "approved_not_rolled_out"


def review_reads_as_active(review: Mapping[str, Any]) -> bool:
    """Whether this review's after-behavior is the behavior now in force.

    The one question a surface should ask before letting a review speak for
    active routing or guidance. Every clause is redundant with `derive_review_state`
    and every clause is here anyway, because #798 AC2 is the property this
    function exists to hold and it should be readable at the call site.
    """
    return (
        str(review.get("review_state", "")) == ACTIVE_REVIEW_STATE
        and derive_review_state(review) == ACTIVE_REVIEW_STATE
        and not str(review.get("superseded_by_review_ref", "") or "")
        and reviewer_decision_state(review) == "current"
    )


def review_state_claim(state: str) -> str:
    """The one sentence a state may be rendered with."""
    text = str(state or "")
    if text not in REVIEW_STATE_CLAIMS:
        raise PolicyChangeReviewError(f"{_LABEL} review_state is unsupported: {text!r}")
    return REVIEW_STATE_CLAIMS[text]


def affected_intents(review: Mapping[str, Any]) -> tuple[str, ...]:
    """The intents this change moves, in the order the review named them."""
    intents = review.get("intents")
    if not _is_row_sequence(intents):
        return ()
    return tuple(str(row.get("intent_ref", "")) for row in intents)


def owner_boundary_impact(review: Mapping[str, Any]) -> dict[str, str]:
    """Each coding owner's boundary change, by owner."""
    rows = review.get("owner_boundaries")
    if not _is_row_sequence(rows):
        return {}
    return {str(row.get("owner", "")): str(row.get("boundary_change", "")) for row in rows}


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate_policy_change_review(review: Any) -> list[str]:
    """Every reason a payload is not a valid review. Both directions on keys."""
    if not isinstance(review, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    claims_rollout = sorted(key for key in review if str(key).lower() in ROLLOUT_CLAIM_KEYS)
    if claims_rollout:
        errors.append(
            f"{_LABEL} must not carry rollout-claim keys: {claims_rollout}; a review proposes a policy "
            "change and never asserts that routing or guidance is already live"
        )
    forbidden = sorted(key for key in review if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    present = {str(key) for key in review}
    missing = sorted(set(POLICY_CHANGE_REVIEW_KEYS) - present)
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    unexpected = sorted(present - set(POLICY_CHANGE_REVIEW_KEYS) - set(claims_rollout) - set(forbidden))
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    if review.get("schema_version") != POLICY_CHANGE_REVIEW_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {POLICY_CHANGE_REVIEW_SCHEMA_VERSION}")
    if review.get("privacy") != REVIEW_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {REVIEW_PRIVACY}")
    if review.get("claim_boundary") != POLICY_CHANGE_REVIEW_CLAIM_BOUNDARY:
        errors.append(
            f"{_LABEL} claim_boundary must state that OMH changes no policy source or routing here and "
            "that a rejected or superseded review is never the active policy"
        )
    if review.get("not_observed") != {
        surface: {"status": "not_observed"} for surface in NOT_OBSERVED_SURFACES
    }:
        errors.append(
            f"{_LABEL} not_observed must mark every one of {list(NOT_OBSERVED_SURFACES)} as not_observed"
        )
    errors.extend(_line_errors(review.get("prepared_at"), field="prepared_at", limit=_MAX_STAMP_CHARS))
    errors.extend(reference_errors(review.get("review_ref"), field="review_ref", label=_LABEL, required=True))
    errors.extend(_surface_errors(review.get("policy_surface")))
    errors.extend(_baseline_errors(review.get("baseline")))
    errors.extend(_intent_errors(review.get("intents")))
    errors.extend(_owner_boundary_errors(review.get("owner_boundaries")))
    errors.extend(_supersession_errors(review))
    errors.extend(_decision_errors(review))
    errors.extend(_rollout_errors(review))
    errors.extend(_state_errors(review))
    errors.extend(_digest_errors(review))
    return errors


def _surface_errors(value: Any) -> list[str]:
    text = str(value or "")
    if text in POLICY_SURFACES:
        return []
    if text.strip().lower() in OUT_OF_SCOPE_SURFACES:
        return [
            f"{_LABEL} policy_surface {text!r} is out of scope for this family; it reviews changes to "
            f"{list(POLICY_SURFACES)} policy, and turning an observed workflow into a reusable skill "
            "belongs to the teach-workflow capability"
        ]
    return [f"{_LABEL} policy_surface is unsupported: {value!r}; allowed: {list(POLICY_SURFACES)}"]


def _baseline_errors(baseline: Any) -> list[str]:
    field = "baseline"
    if not isinstance(baseline, Mapping):
        return [
            f"{_LABEL} {field} must be an object naming the policy this change is measured against; a "
            "change with no baseline cannot be shown to still describe the world at rollout time"
        ]
    errors = _block_key_errors(baseline, keys=BASELINE_KEYS, field=field)
    errors.extend(
        reference_errors(
            baseline.get("baseline_ref"), field=f"{field}.baseline_ref", label=_LABEL, required=True
        )
    )
    if not _is_sha256(str(baseline.get("baseline_digest", "") or "")):
        errors.append(
            f"{_LABEL} {field}.baseline_digest must be a sha256 hex digest of the policy in force when the "
            "change was proposed"
        )
    return errors


def _intent_errors(intents: Any) -> list[str]:
    """#798 AC1: every affected intent, each with a before and an after."""
    field = "intents"
    if not _is_row_sequence(intents):
        return [f"{_LABEL} {field} must be a list of objects, one per affected intent"]
    errors: list[str] = []
    if not intents:
        errors.append(
            f"{_LABEL} {field} must name at least 1 affected intent; a policy change that moves no named "
            "intent is not reviewable"
        )
    if len(intents) > _MAX_INTENTS:
        errors.append(f"{_LABEL} {field} must name at most {_MAX_INTENTS} intents")
    named: list[str] = []
    for index, row in enumerate(intents, start=1):
        reference = str(row.get("intent_ref", "") or "")
        where = repr(reference) if reference else f"row {index}"
        named.append(reference)
        errors.extend(
            reference_errors(
                row.get("intent_ref"), field=f"{field} {where} intent_ref", label=_LABEL, required=True
            )
        )
        unsupported = sorted({str(key) for key in row} - set(INTENT_KEYS))
        if unsupported:
            errors.append(f"{_LABEL} {field} entry {where} has unsupported keys: {unsupported}")
        for side in ("before", "after"):
            value = row.get(side)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"{_LABEL} {field} entry {where} states no {side} behavior; an affected intent needs "
                    "both a before and an after, because one side of a behavior change is not reviewable"
                )
                continue
            errors.extend(
                _line_errors(value, field=f"{field} {where} {side}", limit=_MAX_BEHAVIOR_CHARS, required=True)
            )
    repeated = sorted({reference for reference in named if reference and named.count(reference) > 1})
    if repeated:
        errors.append(f"{_LABEL} {field} names the same intent more than once: {repeated}")
    return errors


def _owner_boundary_errors(rows: Any) -> list[str]:
    """The owner-boundary impact, answered for every coding owner rather than some."""
    field = "owner_boundaries"
    if not _is_row_sequence(rows):
        return [f"{_LABEL} {field} must be a list of objects, one per coding owner"]
    errors: list[str] = []
    named: list[str] = []
    for index, row in enumerate(rows, start=1):
        owner = str(row.get("owner", "") or "")
        where = repr(owner) if owner else f"row {index}"
        named.append(owner)
        errors.extend(_block_key_errors(row, keys=OWNER_BOUNDARY_KEYS, field=f"{field} entry {where}"))
        if owner not in POLICY_CHANGE_OWNERS:
            errors.append(
                f"{_LABEL} {field} names an unknown coding owner: {owner!r}; allowed: "
                f"{list(POLICY_CHANGE_OWNERS)}"
            )
        change = row.get("boundary_change")
        if change not in OWNER_BOUNDARY_CHANGES:
            errors.append(
                f"{_LABEL} {field} boundary_change for {where} is unsupported: {change!r}; allowed: "
                f"{list(OWNER_BOUNDARY_CHANGES)}"
            )
    repeated = sorted({owner for owner in named if owner and named.count(owner) > 1})
    if repeated:
        errors.append(f"{_LABEL} {field} answers for the same coding owner more than once: {repeated}")
    unanswered = [owner for owner in POLICY_CHANGE_OWNERS if owner not in set(named)]
    if unanswered:
        errors.append(
            f"{_LABEL} {field} does not answer for every coding owner: {unanswered}; name "
            "no_boundary_change for an owner the change does not move, so an unaffected owner and an "
            "unconsidered one stay different"
        )
    return errors


def _supersession_errors(review: Mapping[str, Any]) -> list[str]:
    value = review.get("superseded_by_review_ref")
    errors = reference_errors(value, field="superseded_by_review_ref", label=_LABEL, required=False)
    if errors:
        return errors
    if isinstance(value, str) and value and value == str(review.get("review_ref", "") or ""):
        return [f"{_LABEL} superseded_by_review_ref must not name the review itself: {value}"]
    return []


def _decision_errors(review: Mapping[str, Any]) -> list[str]:
    decision = review.get("reviewer_decision")
    if not isinstance(decision, Mapping):
        return [f"{_LABEL} reviewer_decision must be an object, empty when nobody has decided"]
    if not decision:
        return []
    errors = _block_key_errors(decision, keys=REVIEWER_DECISION_KEYS, field="reviewer_decision")
    errors.extend(
        reference_errors(
            decision.get("decided_by"), field="reviewer_decision.decided_by", label=_LABEL, required=True
        )
    )
    if decision.get("decision") not in REVIEWER_DECISIONS:
        errors.append(f"{_LABEL} reviewer_decision.decision is unsupported: {decision.get('decision')!r}")
    if not _is_sha256(str(decision.get("reviewed_digest", "") or "")):
        errors.append(
            f"{_LABEL} reviewer_decision.reviewed_digest must be the sha256 digest of the content that was "
            "reviewed"
        )
    return errors


def _rollout_errors(review: Mapping[str, Any]) -> list[str]:
    field = "rollout"
    rollout = review.get(field)
    if not isinstance(rollout, Mapping):
        return [f"{_LABEL} {field} must be an object carrying the evidence a rollout rests on"]
    errors = _block_key_errors(rollout, keys=ROLLOUT_KEYS, field=field)
    digest = rollout.get("observed_baseline_digest")
    if not isinstance(digest, str):
        errors.append(
            f"{_LABEL} {field}.observed_baseline_digest must be a string, empty when no baseline was observed"
        )
    elif digest and not _is_sha256(digest):
        errors.append(
            f"{_LABEL} {field}.observed_baseline_digest must be a sha256 hex digest of the policy actually "
            "in force"
        )
    errors.extend(
        _ref_list_errors(
            rollout.get("observed_repository_changes"),
            field=f"{field}.observed_repository_changes",
            maximum=_MAX_OBSERVED_CHANGES,
        )
    )
    errors.extend(_regression_evidence_errors(rollout.get("regression_evidence")))
    return errors


def _regression_evidence_errors(evidence: Any) -> list[str]:
    field = "rollout.regression_evidence"
    if not _is_row_sequence(evidence):
        return [f"{_LABEL} {field} must be a list of objects naming one regression case each"]
    errors: list[str] = []
    if len(evidence) > _MAX_REGRESSION_ROWS:
        errors.append(f"{_LABEL} {field} must name at most {_MAX_REGRESSION_ROWS} cases")
    named: list[str] = []
    for index, row in enumerate(evidence, start=1):
        case = str(row.get("case_ref", "") or "")
        where = repr(case) if case else f"row {index}"
        named.append(case)
        errors.extend(_block_key_errors(row, keys=REGRESSION_EVIDENCE_KEYS, field=f"{field} entry {where}"))
        errors.extend(
            reference_errors(
                row.get("case_ref"), field=f"{field} {where} case_ref", label=_LABEL, required=True
            )
        )
        if row.get("outcome") not in REGRESSION_OUTCOMES:
            errors.append(
                f"{_LABEL} {field} outcome for {where} is unsupported: {row.get('outcome')!r}; allowed: "
                f"{list(REGRESSION_OUTCOMES)}"
            )
    repeated = sorted({case for case in named if case and named.count(case) > 1})
    if repeated:
        errors.append(f"{_LABEL} {field} names the same regression case more than once: {repeated}")
    return errors


def _state_errors(review: Mapping[str, Any]) -> list[str]:
    """#798 AC2 and AC3: the state is what the evidence makes it."""
    state = review.get("review_state")
    if str(state).strip().lower() in REFUSED_REVIEW_STATES:
        return [
            f"{_LABEL} review_state may not assert that the change is {state!r}; a review records its own "
            f"lifecycle and never that routing or guidance is live, so one of {list(REVIEW_STATES)} is "
            "required"
        ]
    if state not in REVIEW_STATES:
        return [f"{_LABEL} review_state is unsupported: {state!r}"]
    derived = derive_review_state(review)
    if state == derived:
        return []
    unmet = unmet_rollout_requirements(review)
    detail = f"; unmet rollout requirements: {list(unmet)}" if unmet else ""
    return [
        f"{_LABEL} review_state is {state!r} but the supersession link, the recorded decision, and the "
        f"rollout evidence derive {derived!r}; the state of a review is what its evidence makes it, never "
        f"what a payload asserts{detail}"
    ]


def _digest_errors(review: Mapping[str, Any]) -> list[str]:
    digest = review.get("review_digest")
    if not isinstance(digest, str) or not _is_sha256(digest):
        return [f"{_LABEL} review_digest must be a sha256 hex digest"]
    if digest != policy_change_review_digest(review):
        return [
            f"{_LABEL} review_digest does not match the content it seals; the review was edited after it "
            "was minted"
        ]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_on(errors: list[str]) -> None:
    if errors:
        raise PolicyChangeReviewError("; ".join(errors))


def _policy_surface(value: Any) -> str:
    errors = _surface_errors(value)
    if errors:
        raise PolicyChangeReviewError(errors[0])
    return str(value)


def _intent_rows(intents: Any) -> list[dict[str, Any]]:
    if not _is_row_sequence(intents):
        raise PolicyChangeReviewError(f"{_LABEL} intents must be a list of objects, one per affected intent")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(intents, start=1):
        reference = str(row.get("intent_ref", "") or "")
        where = repr(reference) if reference else f"row {index}"
        unsupported = sorted({str(key) for key in row} - set(INTENT_KEYS))
        if unsupported:
            raise PolicyChangeReviewError(
                f"{_LABEL} intents entry {where} has unsupported keys: {unsupported}"
            )
        for side in ("before", "after"):
            value = row.get(side)
            if not isinstance(value, str) or not value.strip():
                raise PolicyChangeReviewError(
                    f"{_LABEL} intents entry {where} states no {side} behavior; an affected intent needs "
                    "both a before and an after, because one side of a behavior change is not reviewable"
                )
        rows.append(
            {
                "intent_ref": _opaque(row.get("intent_ref"), field=f"{_LABEL} intents {where} intent_ref"),
                "before": _bounded_line(
                    row.get("before"), field=f"{_LABEL} intents {where} before", limit=_MAX_BEHAVIOR_CHARS
                ),
                "after": _bounded_line(
                    row.get("after"), field=f"{_LABEL} intents {where} after", limit=_MAX_BEHAVIOR_CHARS
                ),
            }
        )
    return rows


def _owner_boundary_rows(boundaries: Any) -> list[dict[str, Any]]:
    if not _is_row_sequence(boundaries):
        raise PolicyChangeReviewError(
            f"{_LABEL} owner_boundaries must be a list of objects, one per coding owner"
        )
    rows: list[dict[str, Any]] = []
    for row in boundaries:
        unsupported = sorted({str(key) for key in row} - set(OWNER_BOUNDARY_KEYS))
        if unsupported:
            raise PolicyChangeReviewError(f"{_LABEL} owner_boundaries has unsupported keys: {unsupported}")
        owner = _closed(row.get("owner"), allowed=POLICY_CHANGE_OWNERS, field=f"{_LABEL} owner_boundaries.owner")
        rows.append(
            {
                "owner": owner,
                "boundary_change": _closed(
                    row.get("boundary_change"),
                    allowed=OWNER_BOUNDARY_CHANGES,
                    field=f"{_LABEL} owner_boundaries.boundary_change for {owner!r}",
                ),
            }
        )
    # Owner order is normalized so two reviews answering the same way in a
    # different order produce the same `review_digest`.
    ordered = sorted(rows, key=lambda row: POLICY_CHANGE_OWNERS.index(row["owner"]))
    errors = _owner_boundary_errors(ordered)
    if errors:
        raise PolicyChangeReviewError("; ".join(errors))
    return ordered


def _regression_rows(evidence: Any) -> list[dict[str, Any]]:
    if not _is_row_sequence(evidence):
        raise PolicyChangeReviewError(
            f"{_LABEL} rollout.regression_evidence must be a list of objects naming one regression case each"
        )
    rows: list[dict[str, Any]] = []
    for row in evidence:
        unsupported = sorted({str(key) for key in row} - set(REGRESSION_EVIDENCE_KEYS))
        if unsupported:
            raise PolicyChangeReviewError(
                f"{_LABEL} rollout.regression_evidence has unsupported keys: {unsupported}"
            )
        case = _opaque(row.get("case_ref"), field=f"{_LABEL} rollout.regression_evidence.case_ref")
        rows.append(
            {
                "case_ref": case,
                "outcome": _closed(
                    row.get("outcome"),
                    allowed=REGRESSION_OUTCOMES,
                    field=f"{_LABEL} rollout.regression_evidence.outcome for {case!r}",
                ),
            }
        )
    return rows


def _ref_list(values: Any, *, field: str, maximum: int) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise PolicyChangeReviewError(f"{field} must be a list")
    refs = [_opaque(value, field=field) for value in values]
    if len(set(refs)) != len(refs):
        raise PolicyChangeReviewError(f"{field} must not repeat an entry")
    if len(refs) > maximum:
        raise PolicyChangeReviewError(f"{field} must name at most {maximum} entries")
    return refs


def _ref_list_errors(value: Any, *, field: str, maximum: int) -> list[str]:
    if not isinstance(value, list):
        return [f"{_LABEL} {field} must be a list of opaque references"]
    errors: list[str] = []
    if len(value) > maximum:
        errors.append(f"{_LABEL} {field} must name at most {maximum} entries")
    if len({str(item) for item in value}) != len(value):
        errors.append(f"{_LABEL} {field} must not repeat an entry")
    for item in value:
        errors.extend(reference_errors(item, field=field, label=_LABEL, required=True))
    return errors


def _block_key_errors(block: Any, *, keys: tuple[str, ...], field: str) -> list[str]:
    if not isinstance(block, Mapping):
        return [f"{_LABEL} {field} must be an object"]
    errors: list[str] = []
    present = {str(key) for key in block}
    missing = sorted(set(keys) - present)
    if missing:
        errors.append(f"{_LABEL} {field} is missing keys: {missing}")
    unexpected = sorted(present - set(keys))
    if unexpected:
        errors.append(f"{_LABEL} {field} has unsupported keys: {unexpected}")
    return errors


def _is_row_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(row, Mapping) for row in value)
    )


def _closed(value: Any, *, allowed: tuple[str, ...], field: str) -> str:
    text = str(value or "")
    if text not in allowed:
        raise PolicyChangeReviewError(f"{field} is unsupported: {value!r}; allowed: {list(allowed)}")
    return text


def _opaque(value: Any, *, field: str) -> str:
    return opaque_ref(str(value or ""), field=field, error=PolicyChangeReviewError)


def _optional_digest(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not _is_sha256(text):
        raise PolicyChangeReviewError(f"{field} must be a sha256 hex digest")
    return text


def _bounded_line(value: Any, *, field: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise PolicyChangeReviewError(f"{field} is required")
    if len(text) > limit:
        raise PolicyChangeReviewError(f"{field} must be at most {limit} characters")
    if is_unsafe_metadata_line(text):
        raise PolicyChangeReviewError(
            f"{field} must be one bounded metadata line without secrets, links, or paths"
        )
    return text


def _optional_line(value: Any, *, field: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return _bounded_line(text, field=field, limit=limit) if text else ""


def _line_errors(value: Any, *, field: str, limit: int, required: bool = False) -> list[str]:
    if not isinstance(value, str):
        return [f"{_LABEL} {field} must be a string"]
    if not value.strip():
        return [f"{_LABEL} {field} is required"] if required else []
    if len(value) > limit:
        return [f"{_LABEL} {field} must be at most {limit} characters"]
    if is_unsafe_metadata_line(value):
        return [f"{_LABEL} {field} must not carry secrets, links, paths, or raw text"]
    return []


def _is_sha256(value: str) -> bool:
    if len(value) != _SHA256_LENGTH or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
