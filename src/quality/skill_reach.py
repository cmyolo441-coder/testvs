"""Per-skill routing reach over the two routing-precision corpora.

Audience: agents and operators maintaining the skill catalog. Skill breadth is
the product, so a skill that no chat request can reach is a defect in the
router or in the evidence, never a candidate for pruning. This module projects,
for every installable skill, what the corpora in `routing_precision` actually
prove about it on the deciding surface (`build_chat_interaction_payload`), not
on `omh recommend` ranking:

- `natural_dispatch_cases`: passing intervention cases the router dispatched to
  the skill without an explicit invocation (`route.explicit` is false). A
  message that names a skill turns the routing guards off, so a case that only
  reaches a skill by naming it does not show that an ordinary request can.
- `addressed_dispatch_cases`: passing intervention cases dispatched to the
  skill through an explicit invocation (its name, a sigil, or the assistant it
  addresses).
- `shortlist_first_cases`: passing intervention cases where the router asked
  instead of dispatching and put the skill first (`route.candidate_skill`).
  Shown for context; it is not counted as reach, because the router did not
  pick the skill.
- `negative_control_cases`: passing negative controls that entered the skill's
  territory -- the case forbids it by name, or the router scored it or named it
  as the candidate -- and still did not route there.

Two baselines record the skills the corpora do not yet cover. They only
shrink: an entry must leave when its skill gains coverage, and each entry's
reason is re-measured rather than trusted.
"""

from __future__ import annotations

from typing import Mapping, TypedDict

from ..skills.catalog import builtin_definitions, installable_skill_names
from ..wrapper.contract import build_chat_interaction_payload
from .routing_precision import (
    ROUTING_INTERVENTION_CASES,
    ROUTING_PRECISION_CASES,
    intervention_case_interaction,
    intervention_case_verdict,
    precision_case_interaction,
    precision_case_verdict,
)


SKILL_REACH_SCHEMA_VERSION = "skill_reach/v1"

# The router dispatches the skill, without an explicit invocation, on one of its
# own triggers that is not its name; no intervention sentence proves it yet.
# Verified on every build: the entry fails when no such trigger exists.
REASON_CORPUS_GAP = "corpus_gap"
# The skill is reached when a message addresses it -- a `/omh` sigil, or a turn
# that names Jev -- and by no other trigger. Verified on every build: the entry
# fails unless an addressed intervention case reaches it and no non-name
# trigger does.
REASON_ADDRESSED_BY_DESIGN = "addressed_by_design"
# No negative control uses the skill's vocabulary in another sense.
NEGATIVE_REASONS = (REASON_CORPUS_GAP,)

# Skills with no natural dispatch case in ROUTING_INTERVENTION_CASES, as
# measured when this gate landed. Shrink-only: remove an entry in the same
# commit that adds the intervention case reaching it.
UNREACHED_POSITIVE_BASELINE: Mapping[str, str] = {
    "jev-action-check": REASON_ADDRESSED_BY_DESIGN,
    "jev-ask": REASON_ADDRESSED_BY_DESIGN,
    "jev-done-check": REASON_ADDRESSED_BY_DESIGN,
    "jev-failure-triage": REASON_ADDRESSED_BY_DESIGN,
    "jev-review-gate": REASON_ADDRESSED_BY_DESIGN,
    "jev-route": REASON_ADDRESSED_BY_DESIGN,
    "meta-router": REASON_ADDRESSED_BY_DESIGN,
}

# Skills whose territory no negative control in ROUTING_PRECISION_CASES enters,
# as measured when this gate landed. Shrink-only: remove an entry in the same
# commit that adds the negative control. Every installable skill has one now,
# and the shrink-only ledger in tests/test_skill_reach.py is empty, so no skill
# can be excused here: a new one needs its own negative control.
NO_NEGATIVE_CONTROL_BASELINE: Mapping[str, str] = {}


class SkillReachMeasurement(TypedDict):
    skill: str
    natural_dispatch_cases: int
    addressed_dispatch_cases: int
    shortlist_first_cases: int
    negative_control_cases: int
    natural_trigger_probe: str


class SkillReachRow(SkillReachMeasurement):
    positive_baseline_reason: str
    negative_baseline_reason: str


class SkillReachSummary(TypedDict):
    skill_count: int
    naturally_reached_count: int
    unreached_count: int
    negative_controlled_count: int
    no_negative_control_count: int
    error_count: int


class SkillReachPayload(TypedDict):
    schema_version: str
    source: str
    summary: SkillReachSummary
    skills: list[SkillReachRow]
    errors: list[str]
    claim_boundary: str


def measure_skill_reach(*, source: str = "discord") -> list[SkillReachMeasurement]:
    """Route both corpora once and count, per installable skill, what they prove."""
    skills = installable_skill_names()
    natural = dict.fromkeys(skills, 0)
    addressed = dict.fromkeys(skills, 0)
    shortlisted = dict.fromkeys(skills, 0)
    negative = dict.fromkeys(skills, 0)

    for case in ROUTING_INTERVENTION_CASES:
        interaction = intervention_case_interaction(case, source=source)
        if not intervention_case_verdict(case, interaction, source=source)["passed"]:
            continue
        route = _mapping(interaction.get("route"))
        candidate = str(route.get("candidate_skill") or "")
        if route.get("action") == "clarify" and candidate in shortlisted:
            shortlisted[candidate] += 1
        skill = str(route.get("selected_skill") or "")
        if route.get("action") != "dispatch" or skill not in natural:
            continue
        if route.get("explicit"):
            addressed[skill] += 1
        else:
            natural[skill] += 1

    for case in ROUTING_PRECISION_CASES:
        interaction = precision_case_interaction(case, source=source)
        if not precision_case_verdict(case, interaction, source=source)["passed"]:
            continue
        for skill in _territory_skills(case.forbidden_candidate, _mapping(interaction.get("route"))):
            if skill in negative:
                negative[skill] += 1

    triggers = {definition.name: definition.triggers for definition in builtin_definitions()}
    return [
        {
            "skill": skill,
            "natural_dispatch_cases": natural[skill],
            "addressed_dispatch_cases": addressed[skill],
            "shortlist_first_cases": shortlisted[skill],
            "negative_control_cases": negative[skill],
            "natural_trigger_probe": (
                "" if natural[skill] else natural_trigger_probe(skill, triggers.get(skill, ()), source=source)
            ),
        }
        for skill in skills
    ]


def build_skill_reach_projection(*, source: str = "discord") -> SkillReachPayload:
    measurements = measure_skill_reach(source=source)
    errors = skill_reach_errors(
        measurements,
        positive_baseline=UNREACHED_POSITIVE_BASELINE,
        negative_baseline=NO_NEGATIVE_CONTROL_BASELINE,
    )
    rows: list[SkillReachRow] = [
        {
            **measurement,
            "positive_baseline_reason": UNREACHED_POSITIVE_BASELINE.get(measurement["skill"], ""),
            "negative_baseline_reason": NO_NEGATIVE_CONTROL_BASELINE.get(measurement["skill"], ""),
        }
        for measurement in measurements
    ]
    return {
        "schema_version": SKILL_REACH_SCHEMA_VERSION,
        "source": source,
        "summary": {
            "skill_count": len(rows),
            "naturally_reached_count": sum(1 for row in rows if row["natural_dispatch_cases"]),
            "unreached_count": sum(1 for row in rows if not row["natural_dispatch_cases"]),
            "negative_controlled_count": sum(1 for row in rows if row["negative_control_cases"]),
            "no_negative_control_count": sum(1 for row in rows if not row["negative_control_cases"]),
            "error_count": len(errors),
        },
        "skills": rows,
        "errors": errors,
        "claim_boundary": (
            "Skill reach counts deterministic local routing outcomes of the routing-precision corpora "
            "and of each unreached skill's own trigger phrases. It does not prove live Hermes routing, "
            "that a user phrases a request this way, skill quality, execution, review, CI, or merge evidence."
        ),
    }


def natural_trigger_probe(skill: str, triggers: tuple[str, ...], *, source: str) -> str:
    """Return the first of the skill's own non-name triggers the router dispatches to it.

    The dispatch must not be an explicit invocation, and a trigger that only
    spells the skill's name is skipped: both would show the name reaching the
    skill, which is the case this probe exists to tell apart.
    """
    name = _name_form(skill)
    for trigger in triggers:
        if _name_form(trigger) == name:
            continue
        route = _mapping(build_chat_interaction_payload(trigger, source=source).get("route"))
        if route.get("action") == "dispatch" and route.get("selected_skill") == skill and not route.get("explicit"):
            return trigger
    return ""


def format_skill_reach_summary(payload: Mapping[str, object]) -> str:
    summary = _mapping(payload.get("summary"))
    lines = [
        "OMH skill reach (operators and agents)",
        f"Source: {payload.get('source', 'unknown')}",
        (
            f"Natural positive reach: {summary.get('naturally_reached_count', 0)}/{summary.get('skill_count', 0)} skills; "
            f"unreached: {summary.get('unreached_count', 0)}"
        ),
        (
            f"Negative controls: {summary.get('negative_controlled_count', 0)}/{summary.get('skill_count', 0)} skills; "
            f"uncovered: {summary.get('no_negative_control_count', 0)}"
        ),
    ]
    rows = payload.get("skills")
    unreached = [row for row in rows if isinstance(row, Mapping) and not row.get("natural_dispatch_cases")] if isinstance(rows, list) else []
    if unreached:
        lines.extend(["", "No natural intervention case:"])
        for row in unreached:
            probe = str(row.get("natural_trigger_probe") or "")
            detail = f"own trigger '{probe}' dispatches" if probe else "no non-name trigger dispatches"
            lines.append(
                f"- {row.get('skill')}: {detail}; addressed cases {row.get('addressed_dispatch_cases', 0)}; "
                f"shortlist-first cases {row.get('shortlist_first_cases', 0)}; "
                f"baseline {row.get('positive_baseline_reason') or 'none'}"
            )
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        lines.extend(["", "Errors:"])
        lines.extend(f"- {error}" for error in errors)
    lines.extend(["", f"Boundary: {payload.get('claim_boundary', '')}", "Use --json for the full machine-readable payload."])
    return "\n".join(lines)


def skill_reach_errors(
    measurements: list[SkillReachMeasurement],
    *,
    positive_baseline: Mapping[str, str],
    negative_baseline: Mapping[str, str],
) -> list[str]:
    """Name every skill the corpora leave unproven and every baseline entry that is stale or wrong."""
    installable = {row["skill"] for row in measurements}
    errors: list[str] = []
    for label, baseline in (("UNREACHED_POSITIVE_BASELINE", positive_baseline), ("NO_NEGATIVE_CONTROL_BASELINE", negative_baseline)):
        for skill in sorted(set(baseline) - installable):
            errors.append(f"{skill}: listed in {label} but is not an installable skill; remove the entry")
    for row in measurements:
        skill = row["skill"]
        positive_reason = positive_baseline.get(skill, "")
        if row["natural_dispatch_cases"]:
            if positive_reason:
                errors.append(
                    f"{skill}: reached by {row['natural_dispatch_cases']} natural intervention case(s) "
                    "but still listed in UNREACHED_POSITIVE_BASELINE; remove the entry"
                )
        elif not positive_reason:
            errors.append(
                f"{skill}: no intervention case dispatches to it without naming it; add a "
                "RoutingInterventionCase whose message reaches it in a user's words"
            )
        elif positive_reason == REASON_CORPUS_GAP:
            if not row["natural_trigger_probe"]:
                errors.append(
                    f"{skill}: baseline reason {REASON_CORPUS_GAP} is wrong; none of its own non-name "
                    "triggers dispatches to it, so the router cannot reach it without its name"
                )
        elif positive_reason == REASON_ADDRESSED_BY_DESIGN:
            if not row["addressed_dispatch_cases"] or row["natural_trigger_probe"]:
                errors.append(
                    f"{skill}: baseline reason {REASON_ADDRESSED_BY_DESIGN} is wrong; it needs an addressed "
                    "intervention case and no non-name trigger that dispatches to it"
                )
        else:
            errors.append(f"{skill}: unknown UNREACHED_POSITIVE_BASELINE reason {positive_reason!r}")

        negative_reason = negative_baseline.get(skill, "")
        if row["negative_control_cases"]:
            if negative_reason:
                errors.append(
                    f"{skill}: entered by {row['negative_control_cases']} negative control(s) "
                    "but still listed in NO_NEGATIVE_CONTROL_BASELINE; remove the entry"
                )
        elif not negative_reason:
            errors.append(
                f"{skill}: no negative control uses its vocabulary; add a RoutingPrecisionCase with "
                f"forbidden_candidate='{skill}' that stays unrouted"
            )
        elif negative_reason not in NEGATIVE_REASONS:
            errors.append(f"{skill}: unknown NO_NEGATIVE_CONTROL_BASELINE reason {negative_reason!r}")
    return errors


def _territory_skills(forbidden_candidate: str, route: Mapping[str, object]) -> set[str]:
    skills = {forbidden_candidate, str(route.get("candidate_skill") or "")}
    recommendations = route.get("recommendations")
    if isinstance(recommendations, list):
        for recommendation in recommendations:
            if isinstance(recommendation, Mapping) and _positive_score(recommendation.get("score")):
                skills.add(str(recommendation.get("skill") or ""))
    skills.discard("")
    return skills


def _positive_score(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _name_form(value: str) -> str:
    return " ".join(value.strip().lstrip("$/.").lower().replace("-", " ").split())


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}
