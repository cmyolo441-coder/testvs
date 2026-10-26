"""What a third-party skill is worth, what it costs, and what OMH may copy (#793).

The defect this closes: a third-party skill routinely combines a genuinely
useful procedure with broad access, risky instructions, or side effects nobody
described. `plugin_risk_audit/v1` already answers half of that -- it statically
reads one explicitly named local directory and reports which risk categories it
matched -- but a category list is evidence, not advice. It does not say what the
skill is *for*, what authority the procedure needs, how far the scan's word can
be trusted, which part of the pattern OMH may safely reimplement, or which part
the native version must refuse to copy. A user asking "what is useful and risky
about this skill?" gets a list of regex hits.

`skill_pattern_risk_review/v1` is the answer to the question that was asked. One
review, for one skill, citing one audit.

Four blocks, four separate fields
---------------------------------

#793 AC1 requires usefulness, risk, confidence, and native reproduction guidance
to be separated. They are four required keys -- :data:`USEFULNESS_KEYS`,
:data:`RISK_KEYS`, :data:`CONFIDENCE_KEYS`, :data:`NATIVE_REPRODUCTION_KEYS` --
and none of them is computed from another. A review missing one is a validation
error naming it, because the closed key set is checked in both directions.

Separation is the point rather than a layout preference: a skill can be very
useful and very risky at once, the confidence in both of those judgments is a
third thing, and what OMH should build is a fourth. Folding any pair together
produces the summary that made the question necessary.

The scanner is evidence, the reviewer is a decision
---------------------------------------------------

#793 AC2 -- "a clean scan cannot mark the source approved or active" -- is the
boundary the rest rests on, so it is structural rather than documented:

* ``audit_evidence`` is what the scanner found. ``reviewer_decision`` is what a
  person decided. They are different keys, and a review carrying no reviewer
  decision carries ``{}`` there.
* ``review_status`` is **derived**, never passed.
  :func:`build_skill_pattern_risk_review` has no status parameter, so there is
  no argument through which a clean scan can assert approval, and
  :func:`validate_skill_pattern_risk_review` re-derives the status and refuses a
  payload whose status disagrees -- a hand-written dict cannot lie either.
* :data:`REVIEW_STATUSES` contains exactly one approving member and it is
  reachable only from a recorded decision. :func:`review_reads_as_approved` is
  the accessor a surface asks, and it is false for every status a decisionless
  review can hold.
* :data:`REFUSED_REVIEW_STATUSES` holds the bare words -- ``approved``,
  ``active``, ``trusted``, ``clean`` -- by name, so the refusal reads as a rule
  instead of as an unrecognised value. Approval in this schema is always
  *for native reproduction* and never approval of the source: OMH approves what
  it will build, and never the skill it read about.
* A decision is bound to the content it reviewed. ``reviewed_digest`` stamps
  ``review_digest``, and material content that moves afterwards leaves the
  status at ``superseded_by_content_change`` rather than carrying the old
  approval forward.

Risk becomes a constraint or a rejection
----------------------------------------

#793 AC3 is the field that makes the review actionable rather than descriptive.
Every risk category in the cited audit appears exactly once in
``risk.resolutions``, and each one resolves to either a named member of
:data:`NATIVE_CONSTRAINTS` or a named member of :data:`PROHIBITED_BEHAVIORS`.
The name must also appear in the review's own ``native_reproduction`` lists, so
a resolution cannot invoke a constraint the native design never adopted. An
audited category left unresolved is a validation error naming the category, and
a resolution for a category the audit never reported is refused too -- a
reviewer may not manufacture findings the evidence does not carry.

Those two vocabularies are closed on purpose. A free-text constraint is a
sentence somebody wrote; a vocabulary member is something a later gate can join
on. It also keeps the load-bearing fields clear of
:func:`is_unsafe_metadata_line`, which reads security nouns like "secret" and
"credential" as evidence that a stored line carries a secret. The three
free-text fields left in the schema -- ``intended_outcome``, ``procedure_steps``,
``safe_pattern`` -- describe what the skill achieves and what OMH may build, and
none of them needs those nouns.

The risk categories are not restated here.
:data:`AUDITED_RISK_CATEGORIES` is derived from ``plugin_risk_audit``'s own
``RiskCategory``, so widening the scanner widens the review and the two can
never drift into disagreeing about what a category is.

OMH reads about a skill, and nothing more
-----------------------------------------

Nothing in this module imports, installs, registers, enables, or executes the
skill under review, and nothing re-scans it: the audit is passed in as a payload
that ``ops plugin-risk-audit`` already produced. ``not_observed`` says so per
surface, :data:`SKILL_PATTERN_RISK_REVIEW_CLAIM_BOUNDARY` says so in a sentence,
and :data:`EXECUTION_CLAIM_KEYS` refuses a payload shaped to say otherwise by
key name before the closed key set refuses it as unsupported.

Determinism
-----------

No clock is read. ``prepared_at`` is a caller parameter and is excluded from
``review_digest``, because a wall-clock value inside a compared digest turns an
equality check into a race. The hashing scheme is
``quality.skill_governance.policy_decision_digest``, reused rather than
reinvented for the same reason ``native_capability_change_preview`` reuses it: a
second stable-encoding rule in this repository would be a second thing to keep
in agreement with the first.

What reads this today
---------------------

Stated because the contract is wider than the wiring. No production surface
mints a review yet; this module is the contract and
``tests/test_skill_pattern_risk_review.py`` is its only caller. The intended
readers are the native-capability artifacts (#791, #794), which describe what
OMH should build -- this one says which parts of a borrowed pattern they are
allowed to describe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, get_args

from ..quality.skill_governance import policy_decision_digest
from ..system.append_only_store import (
    RAW_OR_HIDDEN_KEYS,
    is_unsafe_metadata_line,
    opaque_ref,
    reference_errors,
)
from .plugin_risk_audit import PLUGIN_RISK_AUDIT_SCHEMA_VERSION, ManifestStatus, RiskCategory


SKILL_PATTERN_RISK_REVIEW_SCHEMA_VERSION: Final = "skill_pattern_risk_review/v1"

# Derived from the scanner, never restated. `plugin_risk_audit.RiskCategory` is
# the single definition of what a risk category is; widening it widens the set
# of categories a review has to resolve, which is the behavior AC3 wants.
AUDITED_RISK_CATEGORIES: Final[tuple[str, ...]] = tuple(get_args(RiskCategory))
AUDIT_MANIFEST_STATUSES: Final[tuple[str, ...]] = tuple(get_args(ManifestStatus))

REVIEW_PRIVACY: Final = "metadata_only"

# What the procedure needs permission to do. `no_additional_authority` is a real
# answer rather than an empty list, so "needs nothing" and "nobody looked" stay
# distinguishable.
REQUIRED_AUTHORITY: Final = (
    "filesystem_read",
    "filesystem_write",
    "process_execution",
    "network_access",
    "package_installation",
    "host_hook_registration",
    "environment_variable_read",
    "user_confirmation",
    "no_additional_authority",
)

# What the procedure needs to touch to do its job.
REQUIRED_DATA: Final = (
    "user_supplied_text",
    "repository_source",
    "local_configuration",
    "environment_variables",
    "credential_material",
    "network_responses",
    "conversation_history",
    "no_additional_data",
)

# What running the skill would do to the host, as the reviewer reads it. This is
# the reviewer's account of behavior and is deliberately not the scanner's
# category list: a regex hit on `fetch(` is evidence, "sends network requests"
# is a judgment about what the skill does with it. `side_effects_unclear` is
# here because the issue's user problem names unclear side effects, and a review
# that cannot say what a skill does must be able to say that.
OBSERVABLE_SIDE_EFFECTS: Final = (
    "writes_local_files",
    "reads_local_files_outside_the_skill",
    "reads_environment_variables",
    "spawns_processes",
    "sends_network_requests",
    "installs_dependencies",
    "registers_host_hooks",
    "mutates_host_configuration",
    "side_effects_unclear",
    "no_side_effects_identified",
)

# The two ways an audited risk category can be answered. There is no third, and
# no "noted" or "acknowledged": AC3 asks for a constraint or a rejection.
RISK_RESOLUTIONS: Final = ("native_constraint", "rejected")

# What the OMH-native version binds itself to. Closed so a later gate can join
# on a member instead of parsing a sentence.
NATIVE_CONSTRAINTS: Final = (
    "no_dynamic_code_execution",
    "no_network_access",
    "no_subprocess_execution",
    "no_runtime_dependencies",
    "no_host_hook_registration",
    "no_credential_material_retained",
    "bounded_local_reads_only",
    "metadata_only_artifacts",
    "explicit_user_invocation_only",
    "deterministic_offline_behavior",
)

# What the OMH-native version must not copy, however useful it looked.
PROHIBITED_BEHAVIORS: Final = (
    "execute_evaluated_source",
    "reach_a_remote_endpoint",
    "spawn_a_host_process",
    "install_or_resolve_dependencies",
    "register_a_host_hook",
    "embed_or_read_credential_material",
    "read_outside_the_named_source",
    "act_without_explicit_invocation",
)

CONFIDENCE_LEVELS: Final = ("low", "medium", "high")

# What the confidence rests on. Each member names material somebody actually
# had, so a stated confidence is attributable rather than felt.
CONFIDENCE_BASES: Final = (
    "cited_static_scan",
    "reviewer_read_supplied_source",
    "supplied_documentation",
    "comparable_native_pattern",
    "prior_review_of_same_source",
)

# What the evidence cannot show. These are the honest limits of a bounded static
# scan of a local directory, enumerated so a review states them instead of
# implying the scan was exhaustive.
EVIDENCE_LIMITS: Final = (
    "static_scan_only_no_execution_observed",
    "classification_can_miss_obfuscated_behavior",
    "classification_can_flag_inert_text",
    "unscanned_file_types_excluded",
    "runtime_behavior_unknown",
    "dependency_behavior_unknown",
    "author_and_provenance_unverified",
    "source_version_not_pinned",
)

# The reviewer's two available decisions, both about what OMH will build.
REVIEWER_DECISIONS: Final = ("approve_native_reproduction", "reject_native_reproduction")

# The lifecycle of the review. Exactly one member approves anything, it names
# what it approves, and it is unreachable without a current reviewer decision.
REVIEW_STATUSES: Final = (
    "awaiting_reviewer_decision",
    "approved_for_native_reproduction",
    "rejected_for_native_reproduction",
    "superseded_by_content_change",
)

# Words a status will not be, held by name so the refusal states a rule. Every
# one of them would describe the *source* rather than OMH's own plan, and none
# of them is something a scan can establish.
REFUSED_REVIEW_STATUSES: Final = (
    "approved",
    "active",
    "allowed",
    "clean",
    "cleared",
    "enabled",
    "installed",
    "passed",
    "safe",
    "trusted",
    "verified",
    "whitelisted",
)

# The one sentence each status may be rendered with. Every one says the source
# was not installed, enabled, or trusted, so no rendering of any status can read
# as adoption of the third-party skill.
REVIEW_STATUS_CLAIMS: Final = {
    "awaiting_reviewer_decision": (
        "A scan was cited and no reviewer has decided yet. A clean scan is not an approval, and the "
        "source is not installed, enabled, or trusted."
    ),
    "approved_for_native_reproduction": (
        "A reviewer approved building the described safe pattern natively under the named constraints. "
        "The source is not installed, enabled, or trusted."
    ),
    "rejected_for_native_reproduction": (
        "A reviewer decided the pattern must not be reproduced natively. The source is not installed, "
        "enabled, or trusted."
    ),
    "superseded_by_content_change": (
        "The reviewed content moved, so the earlier decision no longer covers it. The source is not "
        "installed, enabled, or trusted."
    ),
}

# How the recorded decision stands against the content now in the review.
REVIEWER_DECISION_STATES: Final = ("absent", "current", "stale")

# What OMH did not do to the reviewed skill, per surface, mirroring the audit's
# own `not_observed` block.
NOT_OBSERVED_SURFACES: Final = (
    "skill_import",
    "skill_installation",
    "skill_registration",
    "skill_execution",
    "dependency_installation",
    "network_access",
)

AUDIT_EVIDENCE_KEYS: Final = (
    "schema_version",
    "audit_digest",
    "manifest_status",
    "risk_categories",
    "scanned_file_count",
    "scanned_byte_count",
)
USEFULNESS_KEYS: Final = ("intended_outcome", "procedure_steps", "required_authority", "required_data")
RISK_KEYS: Final = ("side_effects", "resolutions")
RESOLUTION_KEYS: Final = ("category", "resolution", "native_constraint", "prohibited_behavior")
CONFIDENCE_KEYS: Final = ("level", "basis", "evidence_limits")
NATIVE_REPRODUCTION_KEYS: Final = ("safe_pattern", "native_constraints", "prohibited_behaviors")
REVIEWER_DECISION_KEYS: Final = ("decided_by", "decision", "reviewed_digest")

SKILL_PATTERN_RISK_REVIEW_KEYS: Final = (
    "schema_version",
    "skill_ref",
    "audit_evidence",
    "usefulness",
    "risk",
    "confidence",
    "native_reproduction",
    "reviewer_decision",
    "review_status",
    "not_observed",
    "prepared_at",
    "privacy",
    "review_digest",
    "claim_boundary",
)

# The review's own state: outside the digest, because recording a decision must
# not move the digest that the decision names.
REVIEW_STATE_KEYS: Final = ("reviewer_decision", "review_status")

# A value that cannot cover itself, two module constants identical in every
# review, and the one field that holds a clock.
DERIVED_REVIEW_KEYS: Final = ("review_digest", "claim_boundary", "privacy", "prepared_at")

# Everything else. Derived rather than listed so a key added to the schema is
# material by default and has to be argued out, never silently left out.
MATERIAL_REVIEW_KEYS: Final = tuple(
    key
    for key in SKILL_PATTERN_RISK_REVIEW_KEYS
    if key not in set(REVIEW_STATE_KEYS) | set(DERIVED_REVIEW_KEYS)
)

# Key names under which "OMH ran the skill" arrives. The closed key set already
# refuses every one of them, but it refuses them as unsupported keys, and the
# point of this family is that reading about a skill is not running one.
EXECUTION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "activated",
        "active",
        "allowed",
        "approved",
        "enabled",
        "executed",
        "exit_code",
        "imported",
        "installed",
        "invoked",
        "loaded",
        "ran",
        "registered",
        "result",
        "returned",
        "running",
        "safe",
        "started",
        "succeeded",
        "success",
        "trusted",
        "verified",
        "whitelisted",
    }
)

SKILL_PATTERN_RISK_REVIEW_CLAIM_BOUNDARY: Final = (
    "This review reads a cited plugin_risk_audit/v1 payload and a reviewer's judgment about one "
    "third-party skill. OMH never imports, installs, registers, enables, or executes the reviewed skill "
    "and does not re-scan it here. A clean scan is scanner evidence and never an approval: only a "
    "recorded reviewer decision approves anything, it approves building a native pattern rather than the "
    "source, and the review is not execution, verification, review, CI, merge-readiness, or merge "
    "evidence."
)

_MAX_NARRATIVE_CHARS: Final = 400
_MAX_STEP_CHARS: Final = 240
_MAX_STAMP_CHARS: Final = 64
_MAX_PROCEDURE_STEPS: Final = 12
_MAX_VOCABULARY_ITEMS: Final = 12

_LABEL: Final = "skill pattern risk review"
_SHA256_LENGTH: Final = 64


class SkillPatternRiskReviewError(ValueError):
    """Raised when a review cannot be built or a decision cannot be recorded."""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_skill_pattern_risk_review(
    *,
    skill_ref: str,
    audit: Mapping[str, Any],
    intended_outcome: str,
    procedure_steps: Sequence[str],
    required_authority: Sequence[str],
    required_data: Sequence[str],
    side_effects: Sequence[str],
    risk_resolutions: Sequence[Mapping[str, Any]],
    confidence_level: str,
    confidence_basis: Sequence[str],
    evidence_limits: Sequence[str],
    safe_pattern: str,
    native_constraints: Sequence[str],
    prohibited_behaviors: Sequence[str] = (),
    reviewer_decision: Mapping[str, Any] | None = None,
    prepared_at: str = "",
) -> dict[str, Any]:
    """Mint one review, or refuse.

    There is no `review_status` parameter. The status is derived from whether a
    reviewer decision was recorded and whether it still covers the material
    content, which is what makes "a clean scan cannot read as approved" a
    property of the constructor rather than a rule a caller is asked to respect.

    `reviewer_decision` defaults to absent because minting a review is not
    deciding one. A caller that passes one supplies `decided_by` and `decision`;
    the digest it is bound to is stamped here from the content in hand, unless
    the caller is rebuilding a stored review and supplies `reviewed_digest`
    itself.
    """
    review: dict[str, Any] = {
        "schema_version": SKILL_PATTERN_RISK_REVIEW_SCHEMA_VERSION,
        "skill_ref": _opaque(skill_ref, field=f"{_LABEL} skill_ref"),
        "audit_evidence": cite_plugin_risk_audit(audit),
        "usefulness": {
            "intended_outcome": _bounded_line(
                intended_outcome, field=f"{_LABEL} usefulness.intended_outcome", limit=_MAX_NARRATIVE_CHARS
            ),
            "procedure_steps": _line_list(
                procedure_steps, field=f"{_LABEL} usefulness.procedure_steps", limit=_MAX_STEP_CHARS
            ),
            "required_authority": _vocabulary_list(
                required_authority,
                allowed=REQUIRED_AUTHORITY,
                field=f"{_LABEL} usefulness.required_authority",
            ),
            "required_data": _vocabulary_list(
                required_data, allowed=REQUIRED_DATA, field=f"{_LABEL} usefulness.required_data"
            ),
        },
        "risk": {
            "side_effects": _vocabulary_list(
                side_effects, allowed=OBSERVABLE_SIDE_EFFECTS, field=f"{_LABEL} risk.side_effects"
            ),
            "resolutions": _resolution_rows(risk_resolutions),
        },
        "confidence": {
            "level": _closed(confidence_level, allowed=CONFIDENCE_LEVELS, field=f"{_LABEL} confidence.level"),
            "basis": _vocabulary_list(
                confidence_basis, allowed=CONFIDENCE_BASES, field=f"{_LABEL} confidence.basis"
            ),
            "evidence_limits": _vocabulary_list(
                evidence_limits, allowed=EVIDENCE_LIMITS, field=f"{_LABEL} confidence.evidence_limits"
            ),
        },
        "native_reproduction": {
            "safe_pattern": _bounded_line(
                safe_pattern, field=f"{_LABEL} native_reproduction.safe_pattern", limit=_MAX_NARRATIVE_CHARS
            ),
            "native_constraints": _vocabulary_list(
                native_constraints,
                allowed=NATIVE_CONSTRAINTS,
                field=f"{_LABEL} native_reproduction.native_constraints",
            ),
            "prohibited_behaviors": _vocabulary_list(
                prohibited_behaviors,
                allowed=PROHIBITED_BEHAVIORS,
                field=f"{_LABEL} native_reproduction.prohibited_behaviors",
            ),
        },
        "not_observed": {surface: {"status": "not_observed"} for surface in NOT_OBSERVED_SURFACES},
        "prepared_at": _optional_line(prepared_at, field=f"{_LABEL} prepared_at", limit=_MAX_STAMP_CHARS),
        "privacy": REVIEW_PRIVACY,
        "claim_boundary": SKILL_PATTERN_RISK_REVIEW_CLAIM_BOUNDARY,
    }
    review["review_digest"] = skill_pattern_risk_review_digest(review)
    review["reviewer_decision"] = _reviewer_decision(reviewer_decision, current_digest=review["review_digest"])
    review["review_status"] = derive_review_status(review)
    errors = validate_skill_pattern_risk_review(review)
    if errors:
        raise SkillPatternRiskReviewError("; ".join(errors))
    return review


def cite_plugin_risk_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Project the scanner evidence a review cites, or refuse the review.

    A review is a judgment about evidence, so there is no path that produces one
    without evidence: anything that is not a `plugin_risk_audit/v1` payload is
    refused here, before any of the reviewer's own fields are read.

    Only the aggregate is carried over. The audit deliberately exposes no plugin
    source and no root path, and copying its summary keeps that true of the
    review as well.
    """
    if not isinstance(audit, Mapping):
        raise SkillPatternRiskReviewError(
            f"{_LABEL} must cite one {PLUGIN_RISK_AUDIT_SCHEMA_VERSION} payload; a review without scanner "
            "evidence is an opinion"
        )
    version = audit.get("schema_version")
    if version != PLUGIN_RISK_AUDIT_SCHEMA_VERSION:
        raise SkillPatternRiskReviewError(
            f"{_LABEL} must cite a {PLUGIN_RISK_AUDIT_SCHEMA_VERSION} payload, not {version!r}; a review "
            "without scanner evidence is an opinion"
        )
    summary = audit.get("summary")
    source = audit.get("source")
    if not isinstance(summary, Mapping) or not isinstance(source, Mapping):
        raise SkillPatternRiskReviewError(f"{_LABEL} cited audit is missing its summary or its source")
    return {
        "schema_version": PLUGIN_RISK_AUDIT_SCHEMA_VERSION,
        "audit_digest": policy_decision_digest(dict(audit)),
        "manifest_status": _closed(
            source.get("manifest_status"),
            allowed=AUDIT_MANIFEST_STATUSES,
            field=f"{_LABEL} audit_evidence.manifest_status",
        ),
        "risk_categories": _vocabulary_list(
            summary.get("risk_categories") or (),
            allowed=AUDITED_RISK_CATEGORIES,
            field=f"{_LABEL} audit_evidence.risk_categories",
        ),
        "scanned_file_count": _count(
            summary.get("scanned_file_count"), field=f"{_LABEL} audit_evidence.scanned_file_count"
        ),
        "scanned_byte_count": _count(
            summary.get("scanned_byte_count"), field=f"{_LABEL} audit_evidence.scanned_byte_count"
        ),
    }


def record_reviewer_decision(
    review: Mapping[str, Any], *, decided_by: str, decision: str
) -> dict[str, Any]:
    """Bind one reviewer's decision to the content that reviewer saw.

    The only way an approving status is reached. The digest of the material
    content in front of the reviewer is stamped onto the decision, so content
    that moves afterwards leaves the review superseded rather than quietly
    carrying an approval it outgrew.
    """
    errors = validate_skill_pattern_risk_review(review)
    if errors:
        raise SkillPatternRiskReviewError("; ".join(errors))
    decided = dict(review)
    decided["reviewer_decision"] = _reviewer_decision(
        {"decided_by": decided_by, "decision": decision},
        current_digest=skill_pattern_risk_review_digest(review),
    )
    decided["review_status"] = derive_review_status(decided)
    remaining = validate_skill_pattern_risk_review(decided)
    if remaining:
        raise SkillPatternRiskReviewError("; ".join(remaining))
    return decided


# ---------------------------------------------------------------------------
# Derived state
# ---------------------------------------------------------------------------


def skill_pattern_risk_review_digest(review: Mapping[str, Any]) -> str:
    """Content hash of the material part of a review.

    Reuses `quality.skill_governance.policy_decision_digest`: it is already this
    repository's stable, order-independent content hash over a metadata mapping,
    and a second encoding rule here would be a second thing to keep correct.
    """
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
    return "current" if reviewed and reviewed == skill_pattern_risk_review_digest(review) else "stale"


def derive_review_status(review: Mapping[str, Any]) -> str:
    """The only definition of a review's status.

    Read it as the AC2 rule in code: with no decision the status is
    `awaiting_reviewer_decision` and there is no branch that reaches an
    approving value, whatever the scan found.
    """
    state = reviewer_decision_state(review)
    if state == "absent":
        return "awaiting_reviewer_decision"
    if state == "stale":
        return "superseded_by_content_change"
    decision = review.get("reviewer_decision")
    verdict = decision.get("decision") if isinstance(decision, Mapping) else ""
    if verdict == "approve_native_reproduction":
        return "approved_for_native_reproduction"
    return "rejected_for_native_reproduction"


def review_reads_as_approved(review: Mapping[str, Any]) -> bool:
    """Whether anything in this review approves anything.

    The one question a surface should ask. True only for a review whose status
    is the single approving member and whose recorded decision still covers the
    content, so no amount of clean scanner evidence moves it.
    """
    return (
        str(review.get("review_status", "")) == "approved_for_native_reproduction"
        and reviewer_decision_state(review) == "current"
    )


def review_status_claim(status: str) -> str:
    """The one sentence a status may be rendered with."""
    text = str(status or "")
    if text not in REVIEW_STATUS_CLAIMS:
        raise SkillPatternRiskReviewError(f"{_LABEL} review_status is unsupported: {text!r}")
    return REVIEW_STATUS_CLAIMS[text]


def audited_risk_categories(review: Mapping[str, Any]) -> tuple[str, ...]:
    """The risk categories the cited audit reported, in vocabulary order."""
    evidence = review.get("audit_evidence")
    named = set(evidence.get("risk_categories") or ()) if isinstance(evidence, Mapping) else set()
    return tuple(category for category in AUDITED_RISK_CATEGORIES if category in named)


def resolved_risk_categories(review: Mapping[str, Any]) -> tuple[str, ...]:
    """The risk categories the review answered, in vocabulary order."""
    risk = review.get("risk")
    rows = risk.get("resolutions") if isinstance(risk, Mapping) else ()
    if not _is_row_sequence(rows):
        return ()
    named = {str(row.get("category", "")) for row in rows}
    return tuple(category for category in AUDITED_RISK_CATEGORIES if category in named)


def unresolved_risk_categories(review: Mapping[str, Any]) -> tuple[str, ...]:
    """Audited categories the review left without a constraint or a rejection."""
    resolved = set(resolved_risk_categories(review))
    return tuple(category for category in audited_risk_categories(review) if category not in resolved)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate_skill_pattern_risk_review(review: Any) -> list[str]:
    """Every reason a payload is not a valid review. Both directions on keys."""
    if not isinstance(review, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    claims_execution = sorted(key for key in review if str(key).lower() in EXECUTION_CLAIM_KEYS)
    if claims_execution:
        errors.append(
            f"{_LABEL} must not carry execution-claim keys: {claims_execution}; OMH reads about the "
            "reviewed skill and never imports, installs, registers, enables, or runs it"
        )
    forbidden = sorted(key for key in review if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    present = {str(key) for key in review}
    missing = sorted(set(SKILL_PATTERN_RISK_REVIEW_KEYS) - present)
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    unexpected = sorted(
        present - set(SKILL_PATTERN_RISK_REVIEW_KEYS) - set(claims_execution) - set(forbidden)
    )
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    if review.get("schema_version") != SKILL_PATTERN_RISK_REVIEW_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {SKILL_PATTERN_RISK_REVIEW_SCHEMA_VERSION}")
    if review.get("privacy") != REVIEW_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {REVIEW_PRIVACY}")
    if review.get("claim_boundary") != SKILL_PATTERN_RISK_REVIEW_CLAIM_BOUNDARY:
        errors.append(
            f"{_LABEL} claim_boundary must state that OMH never imports, installs, or executes the "
            "reviewed skill and that a clean scan is not an approval"
        )
    if review.get("not_observed") != {
        surface: {"status": "not_observed"} for surface in NOT_OBSERVED_SURFACES
    }:
        errors.append(
            f"{_LABEL} not_observed must mark every one of {list(NOT_OBSERVED_SURFACES)} as not_observed"
        )
    errors.extend(_line_errors(review.get("prepared_at"), field="prepared_at", limit=_MAX_STAMP_CHARS))
    errors.extend(reference_errors(review.get("skill_ref"), field="skill_ref", label=_LABEL, required=True))
    errors.extend(_audit_evidence_errors(review.get("audit_evidence")))
    errors.extend(_usefulness_errors(review.get("usefulness")))
    errors.extend(_risk_errors(review))
    errors.extend(_confidence_errors(review.get("confidence")))
    errors.extend(_native_reproduction_errors(review.get("native_reproduction")))
    errors.extend(_decision_errors(review))
    errors.extend(_digest_errors(review))
    return errors


def _audit_evidence_errors(evidence: Any) -> list[str]:
    """A review with no cited audit is not a review."""
    field = "audit_evidence"
    if not isinstance(evidence, Mapping):
        return [
            f"{_LABEL} {field} must cite one {PLUGIN_RISK_AUDIT_SCHEMA_VERSION} payload; a review without "
            "scanner evidence is an opinion"
        ]
    errors = _block_key_errors(evidence, keys=AUDIT_EVIDENCE_KEYS, field=field)
    if evidence.get("schema_version") != PLUGIN_RISK_AUDIT_SCHEMA_VERSION:
        errors.append(
            f"{_LABEL} {field}.schema_version must be {PLUGIN_RISK_AUDIT_SCHEMA_VERSION}, not "
            f"{evidence.get('schema_version')!r}"
        )
    digest = str(evidence.get("audit_digest", "") or "")
    if not _is_sha256(digest):
        errors.append(f"{_LABEL} {field}.audit_digest must be a sha256 hex digest of the cited audit")
    if evidence.get("manifest_status") not in AUDIT_MANIFEST_STATUSES:
        errors.append(
            f"{_LABEL} {field}.manifest_status is unsupported: {evidence.get('manifest_status')!r}"
        )
    errors.extend(
        _vocabulary_errors(
            evidence.get("risk_categories"),
            allowed=AUDITED_RISK_CATEGORIES,
            field=f"{field}.risk_categories",
            minimum=0,
        )
    )
    for name in ("scanned_file_count", "scanned_byte_count"):
        errors.extend(_count_errors(evidence.get(name), field=f"{field}.{name}"))
    return errors


def _usefulness_errors(usefulness: Any) -> list[str]:
    """#793 AC1's first block: the outcome, the procedure, and what it needs."""
    field = "usefulness"
    if not isinstance(usefulness, Mapping):
        return [f"{_LABEL} {field} must be an object naming the outcome the skill is worth having"]
    errors = _block_key_errors(usefulness, keys=USEFULNESS_KEYS, field=field)
    errors.extend(
        _line_errors(
            usefulness.get("intended_outcome"),
            field=f"{field}.intended_outcome",
            limit=_MAX_NARRATIVE_CHARS,
            required=True,
        )
    )
    errors.extend(
        _line_list_errors(
            usefulness.get("procedure_steps"),
            field=f"{field}.procedure_steps",
            limit=_MAX_STEP_CHARS,
            minimum=1,
            maximum=_MAX_PROCEDURE_STEPS,
        )
    )
    errors.extend(
        _vocabulary_errors(
            usefulness.get("required_authority"),
            allowed=REQUIRED_AUTHORITY,
            field=f"{field}.required_authority",
            minimum=1,
        )
    )
    errors.extend(
        _vocabulary_errors(
            usefulness.get("required_data"), allowed=REQUIRED_DATA, field=f"{field}.required_data", minimum=1
        )
    )
    return errors


def _risk_errors(review: Mapping[str, Any]) -> list[str]:
    """#793 AC1's second block, and AC3 in its enforcing form."""
    field = "risk"
    risk = review.get(field)
    if not isinstance(risk, Mapping):
        return [
            f"{_LABEL} {field} must be an object naming the side effects and how each audited category "
            "resolves"
        ]
    errors = _block_key_errors(risk, keys=RISK_KEYS, field=field)
    effects = risk.get("side_effects")
    errors.extend(
        _vocabulary_errors(
            effects, allowed=OBSERVABLE_SIDE_EFFECTS, field=f"{field}.side_effects", minimum=1
        )
    )
    if isinstance(effects, list) and "no_side_effects_identified" in effects:
        if len(effects) > 1:
            errors.append(
                f"{_LABEL} {field}.side_effects may not name no_side_effects_identified alongside an "
                f"identified effect: {sorted(set(effects) - {'no_side_effects_identified'})}"
            )
        if audited_risk_categories(review):
            errors.append(
                f"{_LABEL} {field}.side_effects claims none while the cited audit reports "
                f"{list(audited_risk_categories(review))}; the reviewer may disagree with the scan, but "
                "not by asserting the scan found nothing"
            )
    errors.extend(_resolution_errors(review))
    return errors


def _resolution_errors(review: Mapping[str, Any]) -> list[str]:
    """Every audited category resolves, and only audited categories resolve."""
    field = "risk.resolutions"
    risk = review.get("risk")
    rows = risk.get("resolutions") if isinstance(risk, Mapping) else None
    if not _is_row_sequence(rows):
        return [f"{_LABEL} {field} must be a list of objects"]
    errors: list[str] = []
    native = review.get("native_reproduction")
    declared_constraints = set(native.get("native_constraints") or ()) if isinstance(native, Mapping) else set()
    declared_prohibitions = (
        set(native.get("prohibited_behaviors") or ()) if isinstance(native, Mapping) else set()
    )
    named: list[str] = []
    for row in rows:
        errors.extend(_block_key_errors(row, keys=RESOLUTION_KEYS, field=field))
        category = str(row.get("category", ""))
        named.append(category)
        if category not in AUDITED_RISK_CATEGORIES:
            errors.append(f"{_LABEL} {field} names an unknown risk category: {category!r}")
        resolution = row.get("resolution")
        constraint = str(row.get("native_constraint", "") or "")
        prohibition = str(row.get("prohibited_behavior", "") or "")
        if resolution == "native_constraint":
            errors.extend(
                _resolution_target_errors(
                    category,
                    value=constraint,
                    other=prohibition,
                    allowed=NATIVE_CONSTRAINTS,
                    declared=declared_constraints,
                    field=f"{field} native_constraint",
                    other_field="prohibited_behavior",
                    declared_field="native_reproduction.native_constraints",
                )
            )
        elif resolution == "rejected":
            errors.extend(
                _resolution_target_errors(
                    category,
                    value=prohibition,
                    other=constraint,
                    allowed=PROHIBITED_BEHAVIORS,
                    declared=declared_prohibitions,
                    field=f"{field} prohibited_behavior",
                    other_field="native_constraint",
                    declared_field="native_reproduction.prohibited_behaviors",
                )
            )
        else:
            errors.append(
                f"{_LABEL} {field} resolution for {category!r} is unsupported: {resolution!r}; a risk "
                f"resolves to one of {list(RISK_RESOLUTIONS)}"
            )
    repeated = sorted({category for category in named if named.count(category) > 1})
    if repeated:
        errors.append(f"{_LABEL} {field} resolves the same risk category more than once: {repeated}")
    audited = audited_risk_categories(review)
    unresolved = [category for category in audited if category not in set(named)]
    if unresolved:
        errors.append(
            f"{_LABEL} leaves audited risk categories unresolved: {unresolved}; every category the cited "
            "audit reports must resolve to a named native constraint or an explicit rejection"
        )
    unaudited = sorted({category for category in named if category in AUDITED_RISK_CATEGORIES} - set(audited))
    if unaudited:
        errors.append(
            f"{_LABEL} {field} resolves risk categories the cited audit never reported: {unaudited}; a "
            "review answers the evidence it cites"
        )
    return errors


def _resolution_target_errors(
    category: str,
    *,
    value: str,
    other: str,
    allowed: tuple[str, ...],
    declared: set[str],
    field: str,
    other_field: str,
    declared_field: str,
) -> list[str]:
    errors: list[str] = []
    if not value:
        errors.append(f"{_LABEL} {field} is required for {category!r}")
    elif value not in allowed:
        errors.append(f"{_LABEL} {field} for {category!r} is unsupported: {value!r}; allowed: {list(allowed)}")
    elif value not in declared:
        errors.append(
            f"{_LABEL} {field} for {category!r} names {value!r}, which {declared_field} does not carry; a "
            "risk cannot resolve to a constraint the native design never adopted"
        )
    if other:
        errors.append(f"{_LABEL} risk.resolutions must leave {other_field} empty for {category!r}")
    return errors


def _confidence_errors(confidence: Any) -> list[str]:
    """#793 AC1's third block: how far the evidence goes."""
    field = "confidence"
    if not isinstance(confidence, Mapping):
        return [f"{_LABEL} {field} must be an object stating how far the cited evidence goes"]
    errors = _block_key_errors(confidence, keys=CONFIDENCE_KEYS, field=field)
    if confidence.get("level") not in CONFIDENCE_LEVELS:
        errors.append(f"{_LABEL} {field}.level is unsupported: {confidence.get('level')!r}")
    errors.extend(
        _vocabulary_errors(confidence.get("basis"), allowed=CONFIDENCE_BASES, field=f"{field}.basis", minimum=1)
    )
    errors.extend(
        _vocabulary_errors(
            confidence.get("evidence_limits"),
            allowed=EVIDENCE_LIMITS,
            field=f"{field}.evidence_limits",
            minimum=1,
        )
    )
    return errors


def _native_reproduction_errors(native: Any) -> list[str]:
    """#793 AC1's fourth block: what OMH may build and what it must not copy."""
    field = "native_reproduction"
    if not isinstance(native, Mapping):
        return [f"{_LABEL} {field} must be an object naming the safe pattern OMH may build"]
    errors = _block_key_errors(native, keys=NATIVE_REPRODUCTION_KEYS, field=field)
    errors.extend(
        _line_errors(
            native.get("safe_pattern"),
            field=f"{field}.safe_pattern",
            limit=_MAX_NARRATIVE_CHARS,
            required=True,
        )
    )
    errors.extend(
        _vocabulary_errors(
            native.get("native_constraints"),
            allowed=NATIVE_CONSTRAINTS,
            field=f"{field}.native_constraints",
            minimum=1,
        )
    )
    errors.extend(
        _vocabulary_errors(
            native.get("prohibited_behaviors"),
            allowed=PROHIBITED_BEHAVIORS,
            field=f"{field}.prohibited_behaviors",
            minimum=0,
        )
    )
    return errors


def _decision_errors(review: Mapping[str, Any]) -> list[str]:
    """#793 AC2: the reviewer's decision, and the status it alone can reach."""
    decision = review.get("reviewer_decision")
    if not isinstance(decision, Mapping):
        return [f"{_LABEL} reviewer_decision must be an object, empty when nobody has decided"]
    errors: list[str] = []
    if decision:
        errors.extend(_block_key_errors(decision, keys=REVIEWER_DECISION_KEYS, field="reviewer_decision"))
        errors.extend(
            reference_errors(
                decision.get("decided_by"), field="reviewer_decision.decided_by", label=_LABEL, required=True
            )
        )
        if decision.get("decision") not in REVIEWER_DECISIONS:
            errors.append(f"{_LABEL} reviewer_decision.decision is unsupported: {decision.get('decision')!r}")
        if not _is_sha256(str(decision.get("reviewed_digest", "") or "")):
            errors.append(
                f"{_LABEL} reviewer_decision.reviewed_digest must be the sha256 digest of the content that "
                "was reviewed"
            )
    status = review.get("review_status")
    if str(status).strip().lower() in REFUSED_REVIEW_STATUSES:
        errors.append(
            f"{_LABEL} review_status may not describe the source as {status!r}; OMH approves a native "
            f"pattern it will build and never the reviewed skill, so one of {list(REVIEW_STATUSES)} is "
            "required"
        )
    elif status not in REVIEW_STATUSES:
        errors.append(f"{_LABEL} review_status is unsupported: {status!r}")
    else:
        derived = derive_review_status(review)
        if status != derived:
            errors.append(
                f"{_LABEL} review_status is {status!r} but the recorded decision derives {derived!r}; the "
                "status of a review is what its reviewer decision makes it, never what a payload asserts"
            )
    return errors


def _digest_errors(review: Mapping[str, Any]) -> list[str]:
    digest = review.get("review_digest")
    if not isinstance(digest, str) or not _is_sha256(digest):
        return [f"{_LABEL} review_digest must be a sha256 hex digest"]
    if digest != skill_pattern_risk_review_digest(review):
        return [
            f"{_LABEL} review_digest does not match the content it seals; the review was edited after it "
            "was minted"
        ]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolution_rows(resolutions: Any) -> list[dict[str, Any]]:
    if not _is_row_sequence(resolutions):
        raise SkillPatternRiskReviewError(f"{_LABEL} risk.resolutions must be a list of objects")
    rows: list[dict[str, Any]] = []
    for row in resolutions:
        unknown = sorted({str(key) for key in row} - set(RESOLUTION_KEYS))
        if unknown:
            raise SkillPatternRiskReviewError(f"{_LABEL} risk.resolutions has unsupported keys: {unknown}")
        category = _closed(
            row.get("category"), allowed=AUDITED_RISK_CATEGORIES, field=f"{_LABEL} risk.resolutions.category"
        )
        resolution = _closed(
            row.get("resolution"), allowed=RISK_RESOLUTIONS, field=f"{_LABEL} risk.resolutions.resolution"
        )
        rows.append(
            {
                "category": category,
                "resolution": resolution,
                "native_constraint": str(row.get("native_constraint", "") or ""),
                "prohibited_behavior": str(row.get("prohibited_behavior", "") or ""),
            }
        )
    return sorted(rows, key=lambda row: AUDITED_RISK_CATEGORIES.index(row["category"]))


def _reviewer_decision(value: Any, *, current_digest: str) -> dict[str, Any]:
    if value is None or value == {}:
        return {}
    if not isinstance(value, Mapping):
        raise SkillPatternRiskReviewError(f"{_LABEL} reviewer_decision must be an object")
    unknown = sorted({str(key) for key in value} - set(REVIEWER_DECISION_KEYS))
    if unknown:
        raise SkillPatternRiskReviewError(f"{_LABEL} reviewer_decision has unsupported keys: {unknown}")
    return {
        "decided_by": _opaque(value.get("decided_by"), field=f"{_LABEL} reviewer_decision.decided_by"),
        "decision": _closed(
            value.get("decision"), allowed=REVIEWER_DECISIONS, field=f"{_LABEL} reviewer_decision.decision"
        ),
        "reviewed_digest": str(value.get("reviewed_digest", "") or current_digest),
    }


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
        raise SkillPatternRiskReviewError(f"{field} is unsupported: {value!r}; allowed: {list(allowed)}")
    return text


def _vocabulary_list(values: Any, *, allowed: tuple[str, ...], field: str) -> list[str]:
    """Closed-vocabulary members in declared order, deduplicated.

    Normalizing the order here is what lets two reviewers who named the same
    members in different orders produce the same `review_digest`. An
    unrecognised member is refused rather than dropped: silently discarding one
    would turn a typo into a review that validates and says something else.
    """
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise SkillPatternRiskReviewError(f"{field} must be a list")
    named = [str(value).strip() for value in values if str(value).strip()]
    unsupported = sorted(set(named) - set(allowed))
    if unsupported:
        raise SkillPatternRiskReviewError(
            f"{field} has unsupported entries: {unsupported}; allowed: {list(allowed)}"
        )
    if len(set(named)) != len(named):
        raise SkillPatternRiskReviewError(f"{field} must not repeat an entry")
    if len(named) > _MAX_VOCABULARY_ITEMS:
        raise SkillPatternRiskReviewError(f"{field} must name at most {_MAX_VOCABULARY_ITEMS} entries")
    return [item for item in allowed if item in set(named)]


def _vocabulary_errors(value: Any, *, allowed: tuple[str, ...], field: str, minimum: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{_LABEL} {field} must be a list of strings"]
    errors: list[str] = []
    if len(value) < minimum:
        errors.append(f"{_LABEL} {field} must name at least {minimum}")
    if len(value) > _MAX_VOCABULARY_ITEMS:
        errors.append(f"{_LABEL} {field} must name at most {_MAX_VOCABULARY_ITEMS} entries")
    if len(set(value)) != len(value):
        errors.append(f"{_LABEL} {field} must not repeat an entry")
    unsupported = sorted(set(value) - set(allowed))
    if unsupported:
        errors.append(f"{_LABEL} {field} has unsupported entries: {unsupported}; allowed: {list(allowed)}")
    return errors


def _line_list(values: Any, *, field: str, limit: int) -> list[str]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise SkillPatternRiskReviewError(f"{field} must be a list")
    lines = [_bounded_line(value, field=field, limit=limit) for value in values]
    if len(set(lines)) != len(lines):
        raise SkillPatternRiskReviewError(f"{field} must not repeat an entry")
    if not lines:
        raise SkillPatternRiskReviewError(f"{field} must name at least 1")
    if len(lines) > _MAX_PROCEDURE_STEPS:
        raise SkillPatternRiskReviewError(f"{field} must name at most {_MAX_PROCEDURE_STEPS} entries")
    return lines


def _line_list_errors(value: Any, *, field: str, limit: int, minimum: int, maximum: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{_LABEL} {field} must be a list of strings"]
    errors: list[str] = []
    if len(value) < minimum:
        errors.append(f"{_LABEL} {field} must name at least {minimum}")
    if len(value) > maximum:
        errors.append(f"{_LABEL} {field} must name at most {maximum} entries")
    if len(set(value)) != len(value):
        errors.append(f"{_LABEL} {field} must not repeat an entry")
    for item in value:
        errors.extend(_line_errors(item, field=field, limit=limit, required=True))
    return errors


def _opaque(value: Any, *, field: str) -> str:
    return opaque_ref(str(value or ""), field=field, error=SkillPatternRiskReviewError)


def _bounded_line(value: Any, *, field: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise SkillPatternRiskReviewError(f"{field} is required")
    if len(text) > limit:
        raise SkillPatternRiskReviewError(f"{field} must be at most {limit} characters")
    if is_unsafe_metadata_line(text):
        raise SkillPatternRiskReviewError(
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


def _count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SkillPatternRiskReviewError(f"{field} must be a non-negative integer")
    return value


def _count_errors(value: Any, *, field: str) -> list[str]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return [f"{_LABEL} {field} must be a non-negative integer"]
    return []


def _is_sha256(value: str) -> bool:
    if len(value) != _SHA256_LENGTH or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
