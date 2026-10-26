"""Ranked OMH-native opportunities for one user goal, from evidence already on disk.

Somebody knows the outcome they want and does not know which public agent
projects have already solved something like it. The raw form of that question --
search the ecosystem, read what comes back -- answers with implementation names,
and an answer made of implementation names is a shopping list. This module
answers the same question with OMH capability statements instead: what could
Hermes do better for this goal, which OMH capability family covers it today, and
which frozen evidence supports saying so.

What is already here, and what this adds
----------------------------------------

Two rankers exist. `capability_roadmap` ranks missing setup and missing runtime
evidence from a probe payload -- an installation question, the same answer for
every user. `popular_plugin_coverage` weighs request families against OMH case
ids -- a breadth heuristic, again the same answer for every user. Neither is
scoped to a goal, neither applies a source policy, and neither knows whether the
material behind its verdict was read last week or last year.

This module is the goal-scoped one. The same catalog and the same observed
material, asked about a different outcome, returns a different ranking, because
the ranking is a function of the goal.

Coverage is read, never invented
--------------------------------

Every opportunity names a `coverage_family_id` from
`capabilities.families.capability_family_projection`, the projection that
already defines OMH's user-facing capability surface. An id outside it is
refused, and the family's label is copied from the live projection rather than
retyped. There is no second inventory of what OMH can do, so there is nothing to
drift out of sync.

Which family a goal belongs to and how much of it that family already delivers
are two different questions, and only the first has a mechanical answer. The
family and its label are OMH's fact, read from the projection; `coverage_state`
is the caller's judgement about their own goal, and is recorded as such.

The product boundary, enforced rather than requested
----------------------------------------------------

An opportunity's own prose may not name a distributable artifact as the answer.
"Install the review plugin" is a validation error here, not a style note: the
whole point of the workflow is that the user asked how Hermes should improve,
and a package name does not answer that. `PACKAGE_ACQUISITION_CUES` refuses on
its own -- an OMH capability statement never needs the word "install" --  while
`DISTRIBUTABLE_ARTIFACT_CUES` refuses only alongside `ARTIFACT_ADOPTION_CUES`,
because this repository's own boundary language ("must not require a plugin at
runtime") says the word without recommending the thing.

Findings are scanned by exactly the opposite rule: not at all. A finding
describes what an observer read in somebody else's project and naming that
project is the honest thing to do. The prohibition is on the answer, not on the
evidence.

Three evidence states, and none of them reads as another
--------------------------------------------------------

`fresh` means an approved source whose cited snapshot was observed inside the
horizon. `stale` means an approved source whose observation is older than the
horizon or whose age cannot be established at all -- an unparseable or missing
observation time is never fresh. `denied` means the source policy does not admit
one of the cited sources, and nothing about its age is even considered.

Only `fresh` supports an opportunity. An opportunity with no fresh evidence is
not dropped, because silently dropping it would hide the denial that caused it;
it moves to `withheld_opportunities` carrying the reason, so widening the policy
or re-observing the source is a visible next step rather than a guess.

Freshness is derived at read time
---------------------------------

No record here stores an expiry. A cited `capability_inspiration_snapshot/v1`
reports when it was observed; this module subtracts that from the `now` it was
handed and compares against a horizon supplied by the same call. Two callers
with different horizons get different verdicts from the same snapshot, which is
correct -- how long evidence stays good is a policy, not a property of the
evidence -- and a snapshot never has to be rewritten because time passed.

`now` is a parameter and this module reads no clock. It lands in `evaluated_at`
for a reader, and it is excluded from `report_id`, so two scouts of the same
material at different moments agree unless a verdict actually moved. A timestamp
inside a compared value would make every comparison a race.

OMH does not search
-------------------

Every byte of observed material arrives from the caller, frozen in snapshots
somebody else already wrote. This module imports nothing that can open a socket
and calls nothing that can. The `claim_boundary` on the report says so, because
the failure this guards against is a reader treating a ranked opportunity as
proof that OMH just surveyed the ecosystem. It did not, and it cannot.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final

from ..capabilities.families import capability_family_projection
from ..routing.executor_cues import contains_boundary_phrase
from ..routing.localization import normalized_phrase, routing_tokens
from .capability_inspiration_snapshot import validate_capability_inspiration_snapshot
from .skill_governance import _stable_encode
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


CAPABILITY_GAP_SCOUT_SCHEMA_VERSION: Final = "capability_gap_scout/v1"

# How much of the goal an existing OMH capability family already delivers, as
# the caller judged it. Ordered weakest-coverage-first because that is the
# ranking order too: the biggest opportunity is where nothing covers the goal.
COVERAGE_STATES: Final[tuple[str, ...]] = ("absent", "partial", "covered")

_COVERAGE_WEIGHTS: Final[dict[str, int]] = {"absent": 3, "partial": 2, "covered": 1}

# What a cited snapshot is worth right now. `denied` is a policy verdict and
# `stale` is a time verdict; they are never collapsed, because the maintainer
# action they ask for is different -- widen the policy, or go and re-read the
# source.
EVIDENCE_STATES: Final[tuple[str, ...]] = ("denied", "fresh", "stale")

# How much fresh evidence stands behind an opportunity, weakest first. This
# counts distinct fresh snapshots and nothing else -- two findings drawn from
# one frozen snapshot are one piece of evidence, not two -- and it is never a
# judgement about whether the idea is good. OMH cannot have one of those.
OPPORTUNITY_CONFIDENCE_LEVELS: Final[tuple[str, ...]] = (
    "unsupported",
    "low",
    "medium",
    "high",
)

# Why an opportunity carries no fresh support. Three values rather than one so
# the reason survives to the reader: an opportunity held back by an unapproved
# source is a different problem from one held back by an old reading.
WITHHELD_REASONS: Final[tuple[str, ...]] = (
    "evidence_denied",
    "evidence_stale",
    "evidence_stale_and_denied",
)

# The single next step the report asks for. Derived, never supplied.
SCOUT_NEXT_ACTIONS: Final[tuple[str, ...]] = (
    "refresh_stale_evidence",
    "review_ranked_opportunities",
    "supply_observed_evidence",
    "widen_source_policy",
)

# Acquiring a distributable artifact. Any one of these refuses an opportunity on
# its own: an OMH capability statement has no legitimate use for them, so a
# false positive costs a rephrase while a false negative ships a shopping list.
PACKAGE_ACQUISITION_CUES: Final[tuple[str, ...]] = (
    "install",
    "installs",
    "installed",
    "installing",
    "installation",
    "installer",
    "reinstall",
    "uninstall",
    "npx",
    "uv add",
    "cargo add",
    "go get",
    "docker pull",
)

# Names for a distributable artifact. Refused only together with an adoption cue
# below, because this repository's own boundary language says these words
# without recommending anything: "must not require the source plugin at runtime"
# is the boundary, not a violation of it. Bare "package" is deliberately absent
# -- `materials-package`, `report-package`, and `deliverable-package` are OMH
# capabilities, and a rule that refuses OMH's own vocabulary is a broken rule.
DISTRIBUTABLE_ARTIFACT_CUES: Final[tuple[str, ...]] = (
    "plugin",
    "plugins",
    "extension",
    "extensions",
    "add-on",
    "add-ons",
    "addon",
    "addons",
    "marketplace",
    "registry entry",
    "mcp server",
    "npm package",
    "pip package",
    "third-party package",
)

# Choosing to take a named artifact on. "require" is deliberately absent: saying
# OMH must not require a plugin is the boundary this repo already states.
ARTIFACT_ADOPTION_CUES: Final[tuple[str, ...]] = (
    "adopt",
    "adopts",
    "adopting",
    "add",
    "adds",
    "adding",
    "enable",
    "enables",
    "enabling",
    "use",
    "uses",
    "using",
    "switch to",
    "integrate",
    "integrates",
    "integrating",
    "bundle",
    "bundles",
    "bundling",
    "wire in",
    "wire up",
    "bring in",
    "pull in",
    "depend on",
    "depends on",
)

# Every claim this report does not make, named so a consumer can check the list
# instead of inferring the boundary from prose.
CAPABILITY_GAP_SCOUT_NOT_OBSERVED: Final[tuple[str, ...]] = (
    "source_search_execution",
    "source_availability",
    "source_content_re_observation",
    "opportunity_acceptance",
    "capability_implementation",
    "coverage_verification",
)

CAPABILITY_GAP_SCOUT_CLAIM_BOUNDARY: Final = (
    "A gap scout ranks OMH-native opportunities against evidence a caller already supplied. OMH ran "
    "no search, fetch, download, or network call to produce it and cannot run one. Freshness is "
    "derived from the observation time the cited snapshot reports, not from re-reading the source. A "
    "ranked opportunity is work OMH could do, never an instruction to install a plugin, extension, "
    "add-on, or registry entry, and it is not maintainer acceptance, implementation, review, CI, or "
    "merge evidence."
)

CAPABILITY_GAP_SCOUT_KEYS: Final[tuple[str, ...]] = (
    "claim_boundary",
    "evaluated_at",
    "freshness_horizon_days",
    "goal",
    "goal_terms",
    "next_action",
    "not_evidence_until_observed",
    "ranked_opportunities",
    "report_id",
    "schema_version",
    "source_policy",
    "summary",
    "withheld_opportunities",
)

CAPABILITY_GAP_SOURCE_POLICY_KEYS: Final[tuple[str, ...]] = (
    "approved_source_count",
    "approved_sources",
    "policy_id",
)

CAPABILITY_GAP_SUMMARY_KEYS: Final[tuple[str, ...]] = (
    "denied_evidence_count",
    "distinct_coverage_family_count",
    "fresh_evidence_count",
    "opportunity_count",
    "stale_evidence_count",
    "withheld_count",
)

CAPABILITY_GAP_OPPORTUNITY_KEYS: Final[tuple[str, ...]] = (
    "confidence",
    "coverage_family_id",
    "coverage_label",
    "coverage_note",
    "coverage_state",
    "denied_evidence_count",
    "evidence",
    "fresh_evidence_count",
    "goal_alignment",
    "opportunity_id",
    "outcome_statement",
    "priority_score",
    "rank",
    "stale_evidence_count",
    "user_value",
    "withheld_reason",
)

CAPABILITY_GAP_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "cited_source_count",
    "denied_sources",
    "evidence_digest",
    "evidence_state",
    "finding",
    "finding_id",
    "observed_age_days",
    "observed_at",
    "snapshot_id",
)

# The shapes a caller hands in, closed on the way in as well as on the way out.
# An unsupported key is refused rather than ignored: a caller who passes
# `priority` expecting it to be honoured is better served by an error than by a
# ranking that silently ignored it.
CAPABILITY_GAP_OPPORTUNITY_INPUT_KEYS: Final[tuple[str, ...]] = (
    "coverage_family_id",
    "coverage_note",
    "coverage_state",
    "evidence",
    "outcome_statement",
    "user_value",
)

CAPABILITY_GAP_EVIDENCE_INPUT_KEYS: Final[tuple[str, ...]] = ("finding", "snapshot")

# How much each signal moves a ranking. Goal alignment dominates by design: this
# report exists to answer one goal, and a large gap nobody asked about must not
# outrank a smaller one that is exactly what the user wants.
GOAL_ALIGNMENT_WEIGHT: Final = 10
COVERAGE_STATE_WEIGHT: Final = 5
FRESH_EVIDENCE_WEIGHT: Final = 2

# Bounded because a ranked answer nobody can read is not an answer, and because
# an unbounded list turns a metadata report into a document store. Over the cap
# is refused rather than truncated: dropping the twenty-first opportunity would
# change the ranking without telling anyone why.
MAX_OPPORTUNITIES: Final = 20
MAX_EVIDENCE_PER_OPPORTUNITY: Final = 10
MAX_APPROVED_SOURCES: Final = 40
MAX_LINE_LENGTH: Final = 200
MAX_URI_LENGTH: Final = 512

DEFAULT_FRESHNESS_HORIZON_DAYS: Final = 90
MIN_FRESHNESS_HORIZON_DAYS: Final = 1
MAX_FRESHNESS_HORIZON_DAYS: Final = 3650

# `observed_age_days` when the cited snapshot's observation time could not be
# read at all. Never `fresh`: an age nobody could establish is not a young one.
AGE_NOT_ESTABLISHED: Final = -1


class CapabilityGapScoutError(ValueError):
    """A scout report that cannot be built without asserting something untrue."""


def package_recommendation_refusals(text: str) -> list[str]:
    """Every reason a line reads as "acquire this" instead of "Hermes could".

    Returned rather than raised so both the builder and the validator can use
    the same judgement, and so the message names the cue that tripped instead of
    leaving an author to guess which word to change.
    """
    folded = normalized_phrase(str(text or ""))
    if not folded:
        return []
    refusals = [
        f"names a package acquisition: {cue!r}"
        for cue in PACKAGE_ACQUISITION_CUES
        if contains_boundary_phrase(folded, (normalized_phrase(cue),))
    ]
    artifacts = [
        cue
        for cue in DISTRIBUTABLE_ARTIFACT_CUES
        if contains_boundary_phrase(folded, (normalized_phrase(cue),))
    ]
    adoptions = [
        cue
        for cue in ARTIFACT_ADOPTION_CUES
        if contains_boundary_phrase(folded, (normalized_phrase(cue),))
    ]
    if artifacts and adoptions:
        refusals.append(
            f"recommends adopting a distributable artifact: {artifacts[0]!r} with {adoptions[0]!r}"
        )
    return refusals


def build_capability_gap_scout(
    *,
    goal: str,
    now: str,
    opportunities: Sequence[Mapping[str, Any]] = (),
    approved_sources: Sequence[str] = (),
    freshness_horizon_days: int = DEFAULT_FRESHNESS_HORIZON_DAYS,
) -> dict[str, Any]:
    """Rank what OMH could do better for one goal against the evidence supplied.

    `now` is a parameter and never a clock read here, so a caller can reproduce
    a report exactly. It is required and must parse: a freshness verdict reached
    against an unreadable clock would be a guess wearing a state name.

    Each entry of `opportunities` carries `CAPABILITY_GAP_OPPORTUNITY_INPUT_KEYS`
    and each of its `evidence` entries carries
    `CAPABILITY_GAP_EVIDENCE_INPUT_KEYS`, whose `snapshot` is a
    `capability_inspiration_snapshot/v1` that must validate. Two entries naming
    the same family and the same outcome are one opportunity; the first
    occurrence's wording stands and the later one contributes its evidence.
    """
    safe_goal = _bounded_line(goal, field="goal")
    if not safe_goal:
        raise CapabilityGapScoutError("capability_gap_scout goal is required")
    goal_terms = sorted(routing_tokens(safe_goal))
    if not goal_terms:
        raise CapabilityGapScoutError(
            "capability_gap_scout goal must carry at least one term the ranking can score"
        )
    evaluated_at = _bounded_line(now, field="now")
    now_moment = _parse_timestamp(evaluated_at)
    if now_moment is None:
        raise CapabilityGapScoutError(
            "capability_gap_scout now must be an ISO 8601 timestamp so freshness can be derived"
        )
    horizon = _validated_horizon(freshness_horizon_days)
    policy = _source_policy(approved_sources)
    approved = set(policy["approved_sources"])

    collapsed: dict[str, dict[str, Any]] = {}
    for entry in _input_sequence(opportunities, field="opportunities"):
        built = _opportunity(
            entry,
            goal_terms=goal_terms,
            approved=approved,
            now_moment=now_moment,
            horizon=horizon,
        )
        existing = collapsed.get(str(built["opportunity_id"]))
        if existing is None:
            collapsed[str(built["opportunity_id"])] = built
            continue
        _merge_evidence(existing, built)
    if len(collapsed) > MAX_OPPORTUNITIES:
        raise CapabilityGapScoutError(
            f"capability_gap_scout opportunities exceeds {MAX_OPPORTUNITIES} distinct entries"
        )

    ranked = sorted(
        (row for row in collapsed.values() if not row["withheld_reason"]), key=_ranking_key
    )
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    withheld = sorted(
        (row for row in collapsed.values() if row["withheld_reason"]), key=_ranking_key
    )
    summary = _summary(ranked, withheld)
    return {
        "schema_version": CAPABILITY_GAP_SCOUT_SCHEMA_VERSION,
        "report_id": _report_id(
            goal=safe_goal,
            horizon=horizon,
            policy_id=str(policy["policy_id"]),
            ranked=ranked,
            withheld=withheld,
        ),
        "goal": safe_goal,
        "goal_terms": goal_terms,
        "evaluated_at": evaluated_at,
        "freshness_horizon_days": horizon,
        "source_policy": policy,
        "ranked_opportunities": ranked,
        "withheld_opportunities": withheld,
        "summary": summary,
        "next_action": _next_action(ranked, summary),
        "claim_boundary": CAPABILITY_GAP_SCOUT_CLAIM_BOUNDARY,
        "not_evidence_until_observed": list(CAPABILITY_GAP_SCOUT_NOT_OBSERVED),
    }


def validate_capability_gap_scout(report: Any) -> list[str]:
    """Every reason a payload is not a `capability_gap_scout/v1`."""
    label = "capability_gap_scout"
    if not isinstance(report, Mapping):
        return [f"{label} must be an object"]
    errors = _key_set_errors(report, CAPABILITY_GAP_SCOUT_KEYS, label)
    if report.get("schema_version") != CAPABILITY_GAP_SCOUT_SCHEMA_VERSION:
        errors.append(f"{label} schema_version must be {CAPABILITY_GAP_SCOUT_SCHEMA_VERSION}")
    for key in ("goal", "report_id", "evaluated_at"):
        if not _non_empty_text(report.get(key)):
            errors.append(f"{label} {key} must be a non-empty string")
    goal_terms = report.get("goal_terms")
    if not isinstance(goal_terms, list) or not all(isinstance(term, str) for term in goal_terms):
        errors.append(f"{label} goal_terms must be a list of strings")
        goal_terms = []
    elif not goal_terms or goal_terms != sorted(set(goal_terms)):
        errors.append(f"{label} goal_terms must be non-empty, sorted, and distinct")
    horizon = report.get("freshness_horizon_days")
    if not _is_int(horizon) or not (
        MIN_FRESHNESS_HORIZON_DAYS <= int(str(horizon)) <= MAX_FRESHNESS_HORIZON_DAYS
    ):
        errors.append(
            f"{label} freshness_horizon_days must be an integer between "
            f"{MIN_FRESHNESS_HORIZON_DAYS} and {MAX_FRESHNESS_HORIZON_DAYS}"
        )
        horizon = MAX_FRESHNESS_HORIZON_DAYS
    if report.get("next_action") not in SCOUT_NEXT_ACTIONS:
        errors.append(f"{label} next_action must be one of {list(SCOUT_NEXT_ACTIONS)}")
    if report.get("claim_boundary") != CAPABILITY_GAP_SCOUT_CLAIM_BOUNDARY:
        errors.append(f"{label} claim_boundary must state that OMH searched nothing itself")
    declared = report.get("not_evidence_until_observed")
    if not isinstance(declared, list) or list(declared) != list(CAPABILITY_GAP_SCOUT_NOT_OBSERVED):
        errors.append(
            f"{label} not_evidence_until_observed must be {list(CAPABILITY_GAP_SCOUT_NOT_OBSERVED)}"
        )
    errors.extend(_source_policy_errors(report.get("source_policy"), label))
    errors.extend(
        _opportunity_list_errors(
            report.get("ranked_opportunities"),
            label=label,
            field="ranked_opportunities",
            goal_terms=[str(term) for term in goal_terms],
            horizon=int(str(horizon)),
            ranked=True,
        )
    )
    errors.extend(
        _opportunity_list_errors(
            report.get("withheld_opportunities"),
            label=label,
            field="withheld_opportunities",
            goal_terms=[str(term) for term in goal_terms],
            horizon=int(str(horizon)),
            ranked=False,
        )
    )
    errors.extend(_cross_list_errors(report, label))
    if not errors:
        errors.extend(_derived_value_errors(report, label))
    return errors


def _opportunity(
    entry: Mapping[str, Any],
    *,
    goal_terms: list[str],
    approved: set[str],
    now_moment: datetime,
    horizon: int,
) -> dict[str, Any]:
    unsupported = sorted(set(entry) - set(CAPABILITY_GAP_OPPORTUNITY_INPUT_KEYS))
    if unsupported:
        raise CapabilityGapScoutError(
            f"capability_gap_scout opportunity has unsupported keys: {unsupported}"
        )
    outcome = _bounded_line(entry.get("outcome_statement", ""), field="outcome_statement")
    if not outcome:
        raise CapabilityGapScoutError("capability_gap_scout opportunity outcome_statement is required")
    user_value = _bounded_line(entry.get("user_value", ""), field="user_value")
    if not user_value:
        raise CapabilityGapScoutError("capability_gap_scout opportunity user_value is required")
    coverage_note = _bounded_line(entry.get("coverage_note", ""), field="coverage_note")
    for field, text in (
        ("outcome_statement", outcome),
        ("user_value", user_value),
        ("coverage_note", coverage_note),
    ):
        refusals = package_recommendation_refusals(text)
        if refusals:
            raise CapabilityGapScoutError(
                f"capability_gap_scout opportunity {field} recommends an installable package "
                f"instead of user value: {refusals[0]}"
            )
    family_id = _bounded_line(entry.get("coverage_family_id", ""), field="coverage_family_id")
    labels = _coverage_labels()
    if family_id not in labels:
        raise CapabilityGapScoutError(
            f"capability_gap_scout opportunity coverage_family_id must be one of {sorted(labels)}"
        )
    coverage_state = str(entry.get("coverage_state", ""))
    if coverage_state not in COVERAGE_STATES:
        raise CapabilityGapScoutError(
            f"capability_gap_scout opportunity coverage_state must be one of {list(COVERAGE_STATES)}"
        )
    evidence = _evidence_list(
        entry.get("evidence", ()), approved=approved, now_moment=now_moment, horizon=horizon
    )
    counts = _evidence_counts(evidence)
    alignment = _goal_alignment(goal_terms, outcome, user_value)
    return {
        "opportunity_id": _opportunity_id(family_id, outcome),
        "outcome_statement": outcome,
        "user_value": user_value,
        "coverage_family_id": family_id,
        "coverage_label": labels[family_id],
        "coverage_state": coverage_state,
        "coverage_note": coverage_note,
        "evidence": evidence,
        "fresh_evidence_count": counts["fresh"],
        "stale_evidence_count": counts["stale"],
        "denied_evidence_count": counts["denied"],
        "confidence": _confidence(evidence),
        "goal_alignment": alignment,
        "priority_score": _priority_score(alignment, coverage_state, counts["fresh"]),
        "rank": 0,
        "withheld_reason": _withheld_reason(counts),
    }


def _merge_evidence(existing: dict[str, Any], duplicate: dict[str, Any]) -> None:
    """Fold a repeated opportunity into the one already held.

    The wording of the first occurrence stands -- two authors describing the
    same gap differently is not two gaps -- but its evidence is the union, since
    dropping a citation because somebody said the same thing twice would lose
    support the report is meant to carry.
    """
    merged = {str(item["finding_id"]): item for item in existing["evidence"]}
    for item in duplicate["evidence"]:
        merged.setdefault(str(item["finding_id"]), item)
    evidence = sorted(merged.values(), key=lambda item: str(item["finding_id"]))
    if len(evidence) > MAX_EVIDENCE_PER_OPPORTUNITY:
        raise CapabilityGapScoutError(
            f"capability_gap_scout opportunity evidence exceeds "
            f"{MAX_EVIDENCE_PER_OPPORTUNITY} entries after collapsing duplicates"
        )
    counts = _evidence_counts(evidence)
    existing["evidence"] = evidence
    existing["fresh_evidence_count"] = counts["fresh"]
    existing["stale_evidence_count"] = counts["stale"]
    existing["denied_evidence_count"] = counts["denied"]
    existing["confidence"] = _confidence(evidence)
    existing["priority_score"] = _priority_score(
        int(existing["goal_alignment"]), str(existing["coverage_state"]), counts["fresh"]
    )
    existing["withheld_reason"] = _withheld_reason(counts)


def _evidence_list(
    entries: Any, *, approved: set[str], now_moment: datetime, horizon: int
) -> list[dict[str, Any]]:
    rows = _input_sequence(entries, field="opportunity evidence")
    if not rows:
        raise CapabilityGapScoutError(
            "capability_gap_scout opportunity requires at least one cited snapshot"
        )
    if len(rows) > MAX_EVIDENCE_PER_OPPORTUNITY:
        raise CapabilityGapScoutError(
            f"capability_gap_scout opportunity evidence exceeds {MAX_EVIDENCE_PER_OPPORTUNITY} entries"
        )
    built: dict[str, dict[str, Any]] = {}
    for entry in rows:
        item = _evidence(entry, approved=approved, now_moment=now_moment, horizon=horizon)
        built.setdefault(str(item["finding_id"]), item)
    return sorted(built.values(), key=lambda item: str(item["finding_id"]))


def _evidence(
    entry: Mapping[str, Any], *, approved: set[str], now_moment: datetime, horizon: int
) -> dict[str, Any]:
    unsupported = sorted(set(entry) - set(CAPABILITY_GAP_EVIDENCE_INPUT_KEYS))
    if unsupported:
        raise CapabilityGapScoutError(
            f"capability_gap_scout evidence has unsupported keys: {unsupported}"
        )
    snapshot = entry.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise CapabilityGapScoutError(
            "capability_gap_scout evidence snapshot must be a capability_inspiration_snapshot/v1 object"
        )
    snapshot_errors = validate_capability_inspiration_snapshot(snapshot)
    if snapshot_errors:
        raise CapabilityGapScoutError(
            f"capability_gap_scout evidence cites an invalid snapshot: {snapshot_errors[0]}"
        )
    finding = _bounded_line(entry.get("finding", ""), field="evidence finding")
    if not finding:
        raise CapabilityGapScoutError("capability_gap_scout evidence finding is required")
    sources = [source for source in snapshot["observed_sources"] if isinstance(source, Mapping)]
    uris = sorted({str(source.get("uri", "")) for source in sources})
    denied = [uri for uri in uris if uri not in approved]
    observed_at = str(snapshot.get("observed_at", ""))
    age = _age_days(observed_at, now_moment)
    snapshot_id = str(snapshot.get("snapshot_id", ""))
    return {
        "finding_id": _finding_id(snapshot_id, finding),
        "snapshot_id": snapshot_id,
        "evidence_digest": str(snapshot.get("evidence_digest", "")),
        "finding": finding,
        "observed_at": observed_at,
        "observed_age_days": age,
        "cited_source_count": len(uris),
        "denied_sources": denied,
        "evidence_state": _evidence_state(denied, age, horizon),
    }


def _evidence_state(denied: list[str], age: int, horizon: int) -> str:
    if denied:
        return "denied"
    if age == AGE_NOT_ESTABLISHED or age > horizon:
        return "stale"
    return "fresh"


def _evidence_counts(evidence: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        state: sum(1 for item in evidence if item.get("evidence_state") == state)
        for state in EVIDENCE_STATES
    }


def _confidence(evidence: Sequence[Mapping[str, Any]]) -> str:
    """How many separate frozen snapshots back this, capped at the vocabulary.

    Distinct snapshots rather than findings: an author who draws three findings
    from one reading has read one thing, and a count that rose with the number
    of sentences would reward paraphrase.
    """
    sources = {
        str(item.get("snapshot_id", ""))
        for item in evidence
        if item.get("evidence_state") == "fresh"
    }
    return OPPORTUNITY_CONFIDENCE_LEVELS[min(len(sources), len(OPPORTUNITY_CONFIDENCE_LEVELS) - 1)]


def _withheld_reason(counts: Mapping[str, int]) -> str:
    if counts["fresh"]:
        return ""
    if counts["denied"] and counts["stale"]:
        return "evidence_stale_and_denied"
    if counts["denied"]:
        return "evidence_denied"
    return "evidence_stale"


def _priority_score(alignment: int, coverage_state: str, fresh_count: int) -> int:
    return (
        alignment * GOAL_ALIGNMENT_WEIGHT
        + _COVERAGE_WEIGHTS.get(coverage_state, 0) * COVERAGE_STATE_WEIGHT
        + fresh_count * FRESH_EVIDENCE_WEIGHT
    )


def _goal_alignment(goal_terms: Sequence[str], outcome: str, user_value: str) -> int:
    """How many distinct goal terms the opportunity says in its own words.

    Only the two fields a user reads as the answer count. Findings are excluded
    on purpose: a ranking that rose with the volume of quoted evidence would
    reward padding the citation list rather than answering the goal.
    """
    spoken = routing_tokens(f"{outcome} {user_value}")
    return len(set(goal_terms) & spoken)


def _ranking_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (-int(row["priority_score"]), str(row["opportunity_id"]))


def _summary(
    ranked: Sequence[Mapping[str, Any]], withheld: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    rows = [*ranked, *withheld]
    return {
        "opportunity_count": len(ranked),
        "withheld_count": len(withheld),
        "distinct_coverage_family_count": len({str(row["coverage_family_id"]) for row in ranked}),
        "fresh_evidence_count": sum(int(row["fresh_evidence_count"]) for row in rows),
        "stale_evidence_count": sum(int(row["stale_evidence_count"]) for row in rows),
        "denied_evidence_count": sum(int(row["denied_evidence_count"]) for row in rows),
    }


def _next_action(ranked: Sequence[Mapping[str, Any]], summary: Mapping[str, int]) -> str:
    """The one step worth taking next, in the order that unblocks the most.

    A denial outranks a stale reading: approving a source costs a policy edit,
    while re-observing one costs somebody going and reading it again.
    """
    if ranked:
        return "review_ranked_opportunities"
    if summary["denied_evidence_count"]:
        return "widen_source_policy"
    if summary["stale_evidence_count"]:
        return "refresh_stale_evidence"
    return "supply_observed_evidence"


def _source_policy(approved_sources: Sequence[str]) -> dict[str, Any]:
    if isinstance(approved_sources, (str, bytes)) or not isinstance(approved_sources, Sequence):
        raise CapabilityGapScoutError(
            "capability_gap_scout approved_sources must be a sequence of source uris"
        )
    if len(approved_sources) > MAX_APPROVED_SOURCES:
        raise CapabilityGapScoutError(
            f"capability_gap_scout approved_sources exceeds {MAX_APPROVED_SOURCES} entries"
        )
    approved: set[str] = set()
    for value in approved_sources:
        uri = str(value or "").strip()
        if not uri:
            continue
        if len(uri) > MAX_URI_LENGTH:
            raise CapabilityGapScoutError(
                f"capability_gap_scout approved source uri exceeds {MAX_URI_LENGTH} characters"
            )
        approved.add(uri)
    ordered = sorted(approved)
    return {
        "policy_id": _policy_id(ordered),
        "approved_sources": ordered,
        "approved_source_count": len(ordered),
    }


def _coverage_labels() -> dict[str, str]:
    """The live capability-family inventory, read rather than restated."""
    projection = capability_family_projection()
    families = projection.get("families")
    if not isinstance(families, list):
        return {}
    return {
        str(family.get("id", "")): str(family.get("label", ""))
        for family in families
        if isinstance(family, Mapping) and str(family.get("id", ""))
    }


def _validated_horizon(freshness_horizon_days: Any) -> int:
    if not _is_int(freshness_horizon_days):
        raise CapabilityGapScoutError("capability_gap_scout freshness_horizon_days must be an integer")
    horizon = int(freshness_horizon_days)
    if not MIN_FRESHNESS_HORIZON_DAYS <= horizon <= MAX_FRESHNESS_HORIZON_DAYS:
        raise CapabilityGapScoutError(
            f"capability_gap_scout freshness_horizon_days must be between "
            f"{MIN_FRESHNESS_HORIZON_DAYS} and {MAX_FRESHNESS_HORIZON_DAYS}"
        )
    return horizon


def _age_days(observed_at: str, now_moment: datetime) -> int:
    """Whole days between an observation and `now`, or `AGE_NOT_ESTABLISHED`.

    An observation dated ahead of `now` is clamped to zero rather than refused.
    Both values are caller-supplied metadata, so a few seconds of clock skew is
    the likely cause and refusing it would turn an ordinary skew into a stale
    verdict.
    """
    observed = _parse_timestamp(observed_at)
    if observed is None:
        return AGE_NOT_ESTABLISHED
    return max(0, (now_moment - observed).days)


def _parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = strict_fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _input_sequence(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CapabilityGapScoutError(
            f"capability_gap_scout {field} must be a sequence of objects"
        )
    rows: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise CapabilityGapScoutError(f"capability_gap_scout {field} entry must be an object")
        rows.append(item)
    return rows


def _bounded_line(value: Any, *, field: str) -> str:
    """One metadata line: whitespace collapsed, length capped, never wrapped.

    Collapsing whitespace also removes a pasted newline, which is what keeps a
    supplied value from arriving as `\\r\\n` on one platform and `\\n` on another
    and producing two different ids for the same text.
    """
    text = " ".join(str(value or "").split())
    if len(text) > MAX_LINE_LENGTH:
        raise CapabilityGapScoutError(
            f"capability_gap_scout {field} exceeds {MAX_LINE_LENGTH} characters"
        )
    return text


def _opportunity_id(family_id: str, outcome: str) -> str:
    seed = _stable_encode({"coverage_family_id": family_id, "outcome": normalized_phrase(outcome)})
    return f"gap-opportunity-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _finding_id(snapshot_id: str, finding: str) -> str:
    seed = _stable_encode({"finding": normalized_phrase(finding), "snapshot_id": snapshot_id})
    return f"gap-finding-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _policy_id(approved: Sequence[str]) -> str:
    seed = _stable_encode({"approved_sources": list(approved)})
    return f"gap-source-policy-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _report_id(
    *,
    goal: str,
    horizon: int,
    policy_id: str,
    ranked: Sequence[Mapping[str, Any]],
    withheld: Sequence[Mapping[str, Any]],
) -> str:
    """Identity over the answer, never over the moment it was produced.

    Every clock reading is excluded: `evaluated_at` at the top, and
    `observed_age_days` inside each finding, which counts up by one every day
    and would otherwise mint a new id each morning while the answer stood still.
    The freshness verdicts themselves stay in, which is the point -- when
    evidence crosses the horizon the answer changed, and the id should say so.
    """
    seed = _stable_encode(
        {
            "freshness_horizon_days": horizon,
            "goal": goal,
            "ranked": [_identity_projection(row) for row in ranked],
            "source_policy_id": policy_id,
            "withheld": [_identity_projection(row) for row in withheld],
        }
    )
    return f"capability-gap-scout-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"


def _identity_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    """One opportunity with the ticking parts of its findings removed."""
    projected = dict(row)
    evidence = row.get("evidence")
    if isinstance(evidence, list):
        projected["evidence"] = [
            {key: value for key, value in item.items() if key != "observed_age_days"}
            for item in evidence
            if isinstance(item, Mapping)
        ]
    return projected


def _source_policy_errors(policy: Any, label: str) -> list[str]:
    if not isinstance(policy, Mapping):
        return [f"{label} source_policy must be an object"]
    errors = _key_set_errors(policy, CAPABILITY_GAP_SOURCE_POLICY_KEYS, f"{label} source_policy")
    if not _non_empty_text(policy.get("policy_id")):
        errors.append(f"{label} source_policy policy_id must be a non-empty string")
    sources = policy.get("approved_sources")
    if not isinstance(sources, list) or not all(isinstance(uri, str) for uri in sources):
        return [*errors, f"{label} source_policy approved_sources must be a list of strings"]
    if sources != sorted(set(sources)):
        errors.append(f"{label} source_policy approved_sources must be sorted and distinct")
    if policy.get("approved_source_count") != len(sources):
        errors.append(f"{label} source_policy approved_source_count does not match approved_sources")
    if policy.get("policy_id") != _policy_id(sources):
        errors.append(f"{label} source_policy policy_id does not match its approved sources")
    return errors


def _opportunity_list_errors(
    rows: Any, *, label: str, field: str, goal_terms: list[str], horizon: int, ranked: bool
) -> list[str]:
    if not isinstance(rows, list):
        return [f"{label} {field} must be a list"]
    if len(rows) > MAX_OPPORTUNITIES:
        return [f"{label} {field} exceeds {MAX_OPPORTUNITIES} entries"]
    errors: list[str] = []
    keys: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"{label} {field}[{index}] must be an object")
            continue
        errors.extend(
            _opportunity_errors(
                row,
                label=f"{label} {field}[{index}]",
                goal_terms=goal_terms,
                horizon=horizon,
                ranked=ranked,
                expected_rank=index + 1 if ranked else 0,
            )
        )
        if _is_int(row.get("priority_score")) and isinstance(row.get("opportunity_id"), str):
            keys.append((-int(str(row["priority_score"])), str(row["opportunity_id"])))
    if keys != sorted(keys):
        errors.append(f"{label} {field} must be sorted by descending priority then opportunity_id")
    return errors


def _opportunity_errors(
    row: Mapping[str, Any],
    *,
    label: str,
    goal_terms: list[str],
    horizon: int,
    ranked: bool,
    expected_rank: int,
) -> list[str]:
    errors = _key_set_errors(row, CAPABILITY_GAP_OPPORTUNITY_KEYS, label)
    outcome = row.get("outcome_statement")
    user_value = row.get("user_value")
    for key in ("opportunity_id", "outcome_statement", "user_value", "coverage_label"):
        if not _non_empty_text(row.get(key)):
            errors.append(f"{label} {key} must be a non-empty string")
    if not isinstance(row.get("coverage_note"), str):
        errors.append(f"{label} coverage_note must be a string")
    for key in ("outcome_statement", "user_value", "coverage_note"):
        refusals = package_recommendation_refusals(str(row.get(key, "")))
        if refusals:
            errors.append(
                f"{label} {key} recommends an installable package instead of user value: {refusals[0]}"
            )
    labels = _coverage_labels()
    family_id = row.get("coverage_family_id")
    if family_id not in labels:
        errors.append(f"{label} coverage_family_id must be one of {sorted(labels)}")
    elif row.get("coverage_label") != labels[str(family_id)]:
        errors.append(f"{label} coverage_label does not match the current capability family")
    if row.get("coverage_state") not in COVERAGE_STATES:
        errors.append(f"{label} coverage_state must be one of {list(COVERAGE_STATES)}")
    if row.get("confidence") not in OPPORTUNITY_CONFIDENCE_LEVELS:
        errors.append(f"{label} confidence must be one of {list(OPPORTUNITY_CONFIDENCE_LEVELS)}")
    withheld_reason = row.get("withheld_reason")
    if ranked and withheld_reason != "":
        errors.append(f"{label} a ranked opportunity must carry no withheld_reason")
    if not ranked and withheld_reason not in WITHHELD_REASONS:
        errors.append(f"{label} withheld_reason must be one of {list(WITHHELD_REASONS)}")
    if row.get("rank") != expected_rank:
        errors.append(f"{label} rank must be {expected_rank}")
    evidence_errors, counts = _evidence_errors(row.get("evidence"), label=label, horizon=horizon)
    errors.extend(evidence_errors)
    if not evidence_errors:
        for state in EVIDENCE_STATES:
            key = f"{state}_evidence_count"
            if row.get(key) != counts[state]:
                errors.append(f"{label} {key} does not match its evidence")
        if _withheld_reason(counts) != (withheld_reason if isinstance(withheld_reason, str) else ""):
            errors.append(f"{label} withheld_reason does not follow from its evidence states")
        evidence = [item for item in row["evidence"] if isinstance(item, Mapping)]
        if row.get("confidence") != _confidence(evidence):
            errors.append(f"{label} confidence does not match its distinct fresh snapshots")
        if ranked and not counts["fresh"]:
            errors.append(f"{label} a ranked opportunity must cite at least one fresh finding")
        if not ranked and counts["fresh"]:
            errors.append(f"{label} a withheld opportunity must cite no fresh finding")
    if isinstance(outcome, str) and isinstance(user_value, str):
        alignment = _goal_alignment(goal_terms, outcome, user_value)
        if row.get("goal_alignment") != alignment:
            errors.append(f"{label} goal_alignment does not match the report goal terms")
        elif not evidence_errors and row.get("priority_score") != _priority_score(
            alignment, str(row.get("coverage_state", "")), counts["fresh"]
        ):
            errors.append(f"{label} priority_score does not follow from its own signals")
    if isinstance(family_id, str) and isinstance(outcome, str):
        if row.get("opportunity_id") != _opportunity_id(family_id, outcome):
            errors.append(f"{label} opportunity_id does not match its family and outcome")
    return errors


def _evidence_errors(
    evidence: Any, *, label: str, horizon: int
) -> tuple[list[str], dict[str, int]]:
    empty = dict.fromkeys(EVIDENCE_STATES, 0)
    if not isinstance(evidence, list) or not evidence:
        return [f"{label} evidence must be a non-empty list"], empty
    if len(evidence) > MAX_EVIDENCE_PER_OPPORTUNITY:
        return [f"{label} evidence exceeds {MAX_EVIDENCE_PER_OPPORTUNITY} entries"], empty
    errors: list[str] = []
    order: list[str] = []
    for index, item in enumerate(evidence):
        item_label = f"{label} evidence[{index}]"
        if not isinstance(item, Mapping):
            errors.append(f"{item_label} must be an object")
            continue
        errors.extend(_single_evidence_errors(item, label=item_label, horizon=horizon))
        order.append(str(item.get("finding_id", "")))
    if order != sorted(order):
        errors.append(f"{label} evidence must be sorted by finding_id")
    if len(set(order)) != len(order):
        errors.append(f"{label} evidence must not repeat a finding_id")
    if errors:
        return errors, empty
    return errors, _evidence_counts(evidence)


def _single_evidence_errors(item: Mapping[str, Any], *, label: str, horizon: int) -> list[str]:
    errors = _key_set_errors(item, CAPABILITY_GAP_EVIDENCE_KEYS, label)
    for key in ("finding_id", "snapshot_id", "evidence_digest", "finding"):
        if not _non_empty_text(item.get(key)):
            errors.append(f"{label} {key} must be a non-empty string")
    if not isinstance(item.get("observed_at"), str):
        errors.append(f"{label} observed_at must be a string")
    state = item.get("evidence_state")
    if state not in EVIDENCE_STATES:
        errors.append(f"{label} evidence_state must be one of {list(EVIDENCE_STATES)}")
    denied = item.get("denied_sources")
    if not isinstance(denied, list) or not all(isinstance(uri, str) for uri in denied):
        errors.append(f"{label} denied_sources must be a list of strings")
        denied = []
    elif denied != sorted(set(denied)):
        errors.append(f"{label} denied_sources must be sorted and distinct")
    count = item.get("cited_source_count")
    if not _is_int(count) or int(str(count)) < 1:
        errors.append(f"{label} cited_source_count must be a positive integer")
    elif len(denied) > int(str(count)):
        errors.append(f"{label} denied_sources cannot outnumber the cited sources")
    age = item.get("observed_age_days")
    if not _is_int(age) or int(str(age)) < AGE_NOT_ESTABLISHED:
        errors.append(f"{label} observed_age_days must be an integer of at least {AGE_NOT_ESTABLISHED}")
        return errors
    if state != _evidence_state([str(uri) for uri in denied], int(str(age)), horizon):
        errors.append(
            f"{label} evidence_state does not follow from its denied sources, age, and the report horizon"
        )
    if isinstance(item.get("snapshot_id"), str) and isinstance(item.get("finding"), str):
        if item.get("finding_id") != _finding_id(str(item["snapshot_id"]), str(item["finding"])):
            errors.append(f"{label} finding_id does not match its snapshot and finding")
    return errors


def _cross_list_errors(report: Mapping[str, Any], label: str) -> list[str]:
    ranked = report.get("ranked_opportunities")
    withheld = report.get("withheld_opportunities")
    if not isinstance(ranked, list) or not isinstance(withheld, list):
        return []
    rows = [row for row in [*ranked, *withheld] if isinstance(row, Mapping)]
    ids = [str(row.get("opportunity_id", "")) for row in rows]
    errors: list[str] = []
    if len(set(ids)) != len(ids):
        errors.append(f"{label} an opportunity_id must not appear twice")
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return [*errors, f"{label} summary must be an object"]
    errors.extend(_key_set_errors(summary, CAPABILITY_GAP_SUMMARY_KEYS, f"{label} summary"))
    if not all(_is_int(summary.get(key)) for key in CAPABILITY_GAP_SUMMARY_KEYS):
        return [*errors, f"{label} summary counts must be integers"]
    countable = all(
        _is_int(row.get(f"{state}_evidence_count")) and isinstance(row.get("coverage_family_id"), str)
        for row in rows
        for state in EVIDENCE_STATES
    )
    if not countable:
        # The per-opportunity pass already reported the malformed rows; totalling
        # them here would raise instead of adding a reason worth reading.
        return errors
    expected = _summary(
        [row for row in ranked if isinstance(row, Mapping)],
        [row for row in withheld if isinstance(row, Mapping)],
    )
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"{label} summary {key} does not match the reported opportunities")
    return errors


def _derived_value_errors(report: Mapping[str, Any], label: str) -> list[str]:
    """Re-derive the report's own identity and next step from its own material."""
    errors: list[str] = []
    ranked = [row for row in report["ranked_opportunities"] if isinstance(row, Mapping)]
    summary = report["summary"]
    if report.get("next_action") != _next_action(ranked, summary):
        errors.append(f"{label} next_action does not follow from its opportunities and evidence")
    report_id = _report_id(
        goal=str(report["goal"]),
        horizon=int(str(report["freshness_horizon_days"])),
        policy_id=str(report["source_policy"]["policy_id"]),
        ranked=ranked,
        withheld=[row for row in report["withheld_opportunities"] if isinstance(row, Mapping)],
    )
    if report.get("report_id") != report_id:
        errors.append(f"{label} report_id does not match its ranked material")
    return errors


def _key_set_errors(payload: Mapping[str, Any], keys: tuple[str, ...], label: str) -> list[str]:
    errors: list[str] = []
    extra = sorted(set(payload) - set(keys))
    if extra:
        errors.append(f"{label} has unsupported keys: {extra}")
    missing = sorted(set(keys) - set(payload))
    if missing:
        errors.append(f"{label} is missing keys: {missing}")
    return errors


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
