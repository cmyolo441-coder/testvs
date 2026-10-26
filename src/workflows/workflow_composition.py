"""One ordered workflow composed from one compound outcome request (issue #816).

A user asks for a whole outcome once -- "research the payment providers, write
the migration plan, and implement the winner" -- and expects Hermes to present
the ordered workflow that gets there. Today they get one skill: the router
scores the whole sentence and returns its single best match, so the plan
answers the loudest fragment and the rest disappears. Selecting the remaining
capabilities by hand is exactly the work the request was trying to avoid.

A `workflow_composition/v1` record is that ordered workflow. Every step names
the capability it uses, who owns it, what it consumes, what it produces, and
what completing it does and does not prove.

What it is built from
---------------------

Only catalog data. `compound_request_segments` splits the request into the
outcome fragments it stated; each fragment goes through the same
`recommend_skills` scoring every other OMH surface uses; each winning skill maps
to its capability family through the existing family projection; and each step's
inputs, output, and evidence boundary come from that skill's
`SkillDefinition` and its family card. Nothing here carries a per-skill
instruction set that could drift from the catalog it describes.

The one editorial input is `WORKFLOW_COMPOSITION_FAMILY_ORDER`, the phase order
steps are sorted into. It is a claim about which kind of work precedes which,
not about any individual skill, and it is held to the family projection in both
directions by test so a new family cannot be added without placing it.

Ownership
---------

`docs/DIRECTION.md` gives Hermes chat intake, clarification, research, planning,
and narration, and gives implementation to a selected coding owner. A
composition encodes that: a step in the `delegate_coding_and_ship` family is
delegated, every other step is retained by Hermes, and `hermes` is rejected as
the coding owner rather than quietly accepted. That rejection is narrower than
it sounds -- `hermes_coding_team_path/v1` still exists for Hermes-runtime
coding. What a composition may not do is leave implementation with the same
Hermes turn that is narrating the workflow, because then no handoff is ever
prepared and "delegated" becomes a word in a card.

Determinism
-----------

The record is a pure function of the outcome text, the constraints, the
selected coding owner, the available capability set, and the catalog revision.
No clock, no randomness, no model call, no disk read. `catalog_revision()`
enters as a cache key rather than being read inside the memoized builder, so a
catalog change produces a different composition instead of a stale cached one.

Missing capabilities
--------------------

A step whose capability is not available is still ordered into the workflow,
marked `missing`, and listed under `missing_capabilities` with a reason. It is
not dropped, because dropping it would hide an outcome the user asked for, and
nothing here installs it: this module imports no installer and calls none.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from typing import Any, Iterable

from ..capabilities.families import family_for_workflow, family_id_for_workflow
from ..coding.executors import EXECUTOR_PROFILES
from ..install.guidance_projection import catalog_revision
from ..routing.compound_intent import compound_request_segments
from ..routing.recommend import recommend_skills
from ..skills.catalog import installable_skill_names, routable_definitions
from ..skills.catalog_types import SkillDefinition


WORKFLOW_COMPOSITION_SCHEMA_VERSION = "workflow_composition/v1"
WORKFLOW_COMPOSITION_STEP_SCHEMA_VERSION = "workflow_composition_step/v1"
WORKFLOW_COMPOSITION_INTENT_SCHEMA_VERSION = "workflow_composition_intent/v1"
WORKFLOW_COMPOSITION_GAP_SCHEMA_VERSION = "workflow_composition_gap/v1"

WORKFLOW_COMPOSITION_STATES = ("composed", "not_compound", "no_composable_path")
WORKFLOW_COMPOSITION_OWNER_KINDS = ("hermes_retained", "delegated_coding")
WORKFLOW_COMPOSITION_CAPABILITY_STATES = ("available", "missing")
WORKFLOW_COMPOSITION_NEXT_ACTIONS = (
    "present_ordered_workflow",
    "choose_coding_owner",
    "report_missing_capabilities",
    "use_single_capability_plan",
    "ask_which_outcomes_to_compose",
)

# The chat orchestrator. Every retained step is owned by this name, and it is
# the one name a delegated coding step may never carry.
HERMES_RETAINED_OWNER = "hermes"
CODING_OWNER_CHOICE_PENDING = "choose"
WORKFLOW_COMPOSITION_CODING_OWNERS = (
    CODING_OWNER_CHOICE_PENDING,
    *(profile for profile in EXECUTOR_PROFILES if profile != HERMES_RETAINED_OWNER),
)
DELEGATED_CODING_FAMILY = "delegate_coding_and_ship"

# Which kind of work precedes which. Gathering feeds deciding, deciding feeds
# the artifacts and the implementation, operating observes the result, and
# capture happens once there is something worth retaining. Held to the family
# projection in both directions by `tests/test_workflow_composition.py`.
WORKFLOW_COMPOSITION_FAMILY_ORDER = (
    "learn_and_gather",
    "plan_and_decide",
    "create_materials_and_visuals",
    "delegate_coding_and_ship",
    "operate_and_observe",
    "retain_knowledge",
)

WORKFLOW_COMPOSITION_OWNERSHIP_RULE = (
    "Hermes retains chat intake, clarification, research, planning, and narration steps. A step in the "
    "delegate_coding_and_ship family is delegated to the selected coding owner and is never owned by Hermes."
)
WORKFLOW_COMPOSITION_INSTALL_POLICY = (
    "OMH never installs a capability to satisfy a composition. A capability a step needs but that is not "
    "available is reported under missing_capabilities; installing it stays an explicit user action."
)
WORKFLOW_COMPOSITION_DETERMINISM = (
    "Derived from the outcome text, the constraints, the selected coding owner, the available capability "
    "set, and the catalog revision. No clock, no randomness, no model call."
)
WORKFLOW_COMPOSITION_CLAIM_BOUNDARY = (
    "A workflow_composition/v1 record is one prepared ordered workflow. It is not clarification, research, "
    "planning approval, capability installation, coding dispatch, execution, verification, review, CI, "
    "merge-readiness, or merge evidence."
)
WORKFLOW_COMPOSITION_NOT_OBSERVED = (
    "capability_installation",
    "coding_owner_selection",
    "step_start",
    "coding_dispatch",
    "executor_result",
    "verification",
    "review",
)
WORKFLOW_COMPOSITION_GAP_NEXT_ACTION = "report_gap_and_wait_for_an_explicit_install_decision"

WORKFLOW_COMPOSITION_KEYS = (
    "schema_version",
    "composition_id",
    "state",
    "outcome",
    "constraints",
    "catalog_revision",
    "coding_owner",
    "compound_intent",
    "steps",
    "step_count",
    "missing_capabilities",
    "ownership_rule",
    "install_policy",
    "determinism",
    "next_action",
    "claim_boundary",
    "not_evidence_until_observed",
)
WORKFLOW_COMPOSITION_STEP_KEYS = (
    "schema_version",
    "step_id",
    "order",
    "capability",
    "capability_family",
    "capability_status",
    "owner",
    "owner_kind",
    "inputs",
    "output",
    "evidence_boundary",
)
WORKFLOW_COMPOSITION_INTENT_KEYS = (
    "schema_version",
    "recognized",
    "reason",
    "segments",
    "connectors",
    "capability_families",
)
WORKFLOW_COMPOSITION_GAP_KEYS = (
    "schema_version",
    "capability",
    "capability_family",
    "needed_for_step",
    "reason",
    "next_action",
)

# The five fields acceptance criterion 2 requires of every step. Named so the
# validator's error text and the test asserting it read from one list.
WORKFLOW_COMPOSITION_REQUIRED_STEP_FIELDS = (
    "capability",
    "owner",
    "inputs",
    "output",
    "evidence_boundary",
)


@dataclass(frozen=True)
class _ResolvedCapability:
    """One outcome fragment and the catalog capability it scored onto."""

    capability: str
    family: str
    segment: str
    evidence_boundary: str


def build_workflow_composition(
    outcome: str,
    *,
    constraints: Iterable[str] = (),
    coding_owner: str = CODING_OWNER_CHOICE_PENDING,
    available_capabilities: Iterable[str] | None = None,
) -> dict[str, object]:
    """Compose one ordered workflow from one compound outcome request.

    `available_capabilities` defaults to the installable catalog. Passing an
    explicit set is how a caller composes against a host that carries less than
    the full catalog; capabilities outside it are reported, never installed.
    """
    statement = outcome.strip()
    if not statement:
        raise ValueError("workflow composition requires an outcome request")
    if coding_owner == HERMES_RETAINED_OWNER:
        raise ValueError(
            "workflow composition cannot assign coding to hermes: Hermes retains chat, clarification, "
            "research, planning, and narration, and coding delegates to a selected coding owner"
        )
    if coding_owner not in WORKFLOW_COMPOSITION_CODING_OWNERS:
        raise ValueError(f"unsupported workflow composition coding owner: {coding_owner}")
    constraint_key = tuple(text for text in (str(item).strip() for item in constraints) if text)
    available_key = (
        None
        if available_capabilities is None
        else tuple(sorted({text for text in (str(item).strip() for item in available_capabilities) if text}))
    )
    # `catalog_revision()` is resolved here rather than inside the memoized
    # builder so that the revision is part of the cache key: a catalog change
    # produces a new composition instead of returning the one cached before it.
    return _clone(
        _build_workflow_composition_cached(
            statement,
            constraint_key,
            coding_owner,
            available_key,
            catalog_revision(),
        )
    )


@lru_cache(maxsize=512)
def _build_workflow_composition_cached(
    outcome: str,
    constraints: tuple[str, ...],
    coding_owner: str,
    available_key: tuple[str, ...] | None,
    revision: str,
) -> dict[str, object]:
    available = set(installable_skill_names()) if available_key is None else set(available_key)
    segments = compound_request_segments(outcome)
    resolved = _resolved_capabilities(segments.segments)
    state = _state(segments.segments, resolved)
    steps = (
        _steps(resolved, outcome=outcome, constraints=constraints, coding_owner=coding_owner, available=available)
        if state == "composed"
        else []
    )
    gaps = _missing_capabilities(steps)
    return {
        "schema_version": WORKFLOW_COMPOSITION_SCHEMA_VERSION,
        "composition_id": _composition_id(outcome, constraints, coding_owner, available_key, revision),
        "state": state,
        "outcome": outcome,
        "constraints": list(constraints),
        "catalog_revision": revision,
        "coding_owner": coding_owner,
        "compound_intent": {
            "schema_version": WORKFLOW_COMPOSITION_INTENT_SCHEMA_VERSION,
            "recognized": state in {"composed", "no_composable_path"},
            "reason": _intent_reason(state, segments.segments, resolved),
            "segments": list(segments.segments),
            "connectors": list(segments.connectors),
            "capability_families": [item.family for item in resolved],
        },
        "steps": steps,
        "step_count": len(steps),
        "missing_capabilities": gaps,
        "ownership_rule": WORKFLOW_COMPOSITION_OWNERSHIP_RULE,
        "install_policy": WORKFLOW_COMPOSITION_INSTALL_POLICY,
        "determinism": WORKFLOW_COMPOSITION_DETERMINISM,
        "next_action": _next_action(state, steps, gaps),
        "claim_boundary": WORKFLOW_COMPOSITION_CLAIM_BOUNDARY,
        "not_evidence_until_observed": list(WORKFLOW_COMPOSITION_NOT_OBSERVED),
    }


def validate_workflow_composition(payload: Any) -> list[str]:
    """Return every contract violation in one pass; empty means valid."""
    if not isinstance(payload, dict):
        return ["workflow_composition must be an object"]
    errors = _key_set_errors(payload, WORKFLOW_COMPOSITION_KEYS, "workflow_composition")
    if payload.get("schema_version") != WORKFLOW_COMPOSITION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {WORKFLOW_COMPOSITION_SCHEMA_VERSION}")
    if not str(payload.get("composition_id", "")).startswith("composition-"):
        errors.append("composition_id must start with composition-")
    state = payload.get("state")
    if state not in WORKFLOW_COMPOSITION_STATES:
        errors.append(f"state must be one of {list(WORKFLOW_COMPOSITION_STATES)}")
    if not str(payload.get("outcome", "")).strip():
        errors.append("outcome must not be empty")
    if not isinstance(payload.get("constraints"), list):
        errors.append("constraints must be a list")
    if not _is_sha256(str(payload.get("catalog_revision", ""))):
        errors.append("catalog_revision must be a sha256 hex digest")
    coding_owner = payload.get("coding_owner")
    if coding_owner == HERMES_RETAINED_OWNER:
        errors.append("coding_owner must not be hermes: coding is delegated, never retained by the chat orchestrator")
    elif coding_owner not in WORKFLOW_COMPOSITION_CODING_OWNERS:
        errors.append(f"coding_owner must be one of {list(WORKFLOW_COMPOSITION_CODING_OWNERS)}")
    if payload.get("next_action") not in WORKFLOW_COMPOSITION_NEXT_ACTIONS:
        errors.append(f"next_action must be one of {list(WORKFLOW_COMPOSITION_NEXT_ACTIONS)}")
    for key, expected in (
        ("ownership_rule", WORKFLOW_COMPOSITION_OWNERSHIP_RULE),
        ("install_policy", WORKFLOW_COMPOSITION_INSTALL_POLICY),
        ("determinism", WORKFLOW_COMPOSITION_DETERMINISM),
        ("claim_boundary", WORKFLOW_COMPOSITION_CLAIM_BOUNDARY),
    ):
        if payload.get(key) != expected:
            errors.append(f"{key} must state the composition contract verbatim")
    if not set(WORKFLOW_COMPOSITION_NOT_OBSERVED).issubset(set(_string_list(payload.get("not_evidence_until_observed")))):
        errors.append("not_evidence_until_observed must include all workflow-composition boundaries")

    errors.extend(_intent_errors(payload.get("compound_intent")))
    errors.extend(_steps_errors(payload, state))
    errors.extend(_gap_errors(payload))
    return errors


def render_workflow_composition_text(payload: dict[str, Any]) -> str:
    """Plain-text rendering: the ordered workflow, then what is still open."""
    state = str(payload.get("state", ""))
    intent = payload.get("compound_intent")
    reason = str(intent.get("reason", "")) if isinstance(intent, dict) else ""
    lines = [
        f"Workflow composition: {payload.get('composition_id', '')} ({state})",
        f"Outcome: {payload.get('outcome', '')}",
    ]
    constraints = _string_list(payload.get("constraints"))
    if constraints:
        lines.append("Constraints:")
        lines.extend(f"  - {text}" for text in constraints)
    if reason:
        lines.extend(["", reason])
    steps = payload.get("steps")
    if isinstance(steps, list) and steps:
        lines.append("")
        lines.append("Ordered workflow:")
        for step in steps:
            if not isinstance(step, dict):
                continue
            status = str(step.get("capability_status", ""))
            suffix = "  [capability not available]" if status == "missing" else ""
            lines.append(
                f"  {step.get('order', '')}. {step.get('capability', '')} "
                f"— owner: {step.get('owner', '')} ({step.get('owner_kind', '')}){suffix}"
            )
            lines.extend(f"       input:  {text}" for text in _string_list(step.get("inputs")))
            lines.append(f"       output: {step.get('output', '')}")
            lines.append(f"       proves: {step.get('evidence_boundary', '')}")
    gaps = payload.get("missing_capabilities")
    if isinstance(gaps, list) and gaps:
        lines.append("")
        lines.append("Missing capabilities (reported, not installed):")
        for gap in gaps:
            if isinstance(gap, dict):
                lines.append(f"  - {gap.get('capability', '')} ({gap.get('needed_for_step', '')}): {gap.get('reason', '')}")
    lines.append("")
    lines.append(f"Next action: {payload.get('next_action', '')}")
    lines.append(str(payload.get("ownership_rule", "")))
    lines.append(str(payload.get("install_policy", "")))
    lines.append(str(payload.get("claim_boundary", "")))
    return "\n".join(lines)


def _resolved_capabilities(segments: tuple[str, ...]) -> tuple[_ResolvedCapability, ...]:
    """Score each fragment onto one capability, keeping one per family.

    A fragment contributes at any positive score. A score floor was the obvious
    alternative and it does not work: `implement the winner` and
    `add a --json flag` both score 3, so any floor that drops the second drops
    the coding step the first explicitly asked for. Over-inclusion is visible in
    a workflow the user reads before anything starts; omission is not.
    """
    resolved: dict[str, _ResolvedCapability] = {}
    for segment in segments:
        recommendations = recommend_skills(segment, limit=1)
        top = recommendations[0] if recommendations else {}
        capability = str(top.get("skill", ""))
        if not capability or _int_value(top.get("score", 0)) <= 0:
            continue
        family = family_id_for_workflow(capability)
        if not family or family in resolved:
            continue
        resolved[family] = _ResolvedCapability(
            capability=capability,
            family=family,
            segment=segment,
            evidence_boundary=str(top.get("evidence_boundary", "")),
        )
    # Phase order, then family name. Families are unique keys here, so the pair
    # is a total order and the sort does not depend on the order fragments were
    # stated: "implement X after researching Y" and "research Y then implement
    # X" compose to the same workflow.
    return tuple(sorted(resolved.values(), key=lambda item: (_family_rank(item.family), item.family)))


def _family_rank(family: str) -> int:
    if family in WORKFLOW_COMPOSITION_FAMILY_ORDER:
        return WORKFLOW_COMPOSITION_FAMILY_ORDER.index(family)
    return len(WORKFLOW_COMPOSITION_FAMILY_ORDER)


def _state(segments: tuple[str, ...], resolved: tuple[_ResolvedCapability, ...]) -> str:
    if len(segments) < 2:
        return "not_compound"
    if len(resolved) < 2:
        return "no_composable_path"
    return "composed"


def _intent_reason(state: str, segments: tuple[str, ...], resolved: tuple[_ResolvedCapability, ...]) -> str:
    if state == "not_compound":
        if not segments:
            return "The request carries no routable capability signal, so there is nothing to compose."
        return (
            "The request states one outcome. Composition is for a request that asks for more than one; "
            "use the single-capability Hermes plan instead."
        )
    if state == "no_composable_path":
        if not resolved:
            return (
                f"{len(segments)} outcome fragments were recognised, but none of them matched a routable "
                "OMH capability. Name the outcomes to compose."
            )
        return (
            f"{len(segments)} outcome fragments were recognised, but they all resolve to the single capability "
            f"family `{resolved[0].family}`. That is one capability, not a workflow."
        )
    return (
        f"{len(segments)} outcome fragments resolved to {len(resolved)} capability families, ordered by "
        "workflow phase."
    )


def _steps(
    resolved: tuple[_ResolvedCapability, ...],
    *,
    outcome: str,
    constraints: tuple[str, ...],
    coding_owner: str,
    available: set[str],
) -> list[dict[str, object]]:
    steps: list[dict[str, object]] = []
    for index, item in enumerate(resolved):
        order = index + 1
        definition = _definition_for(item.capability)
        delegated = item.family == DELEGATED_CODING_FAMILY
        output = _step_output(definition, item.capability)
        steps.append(
            {
                "schema_version": WORKFLOW_COMPOSITION_STEP_SCHEMA_VERSION,
                "step_id": f"step-{order}",
                "order": order,
                "capability": item.capability,
                "capability_family": item.family,
                "capability_status": "available" if item.capability in available else "missing",
                "owner": coding_owner if delegated else HERMES_RETAINED_OWNER,
                "owner_kind": "delegated_coding" if delegated else "hermes_retained",
                "inputs": _step_inputs(
                    definition,
                    order=order,
                    outcome=outcome,
                    constraints=constraints,
                    segment=item.segment,
                ),
                "output": output,
                "evidence_boundary": _step_evidence_boundary(item, output),
            }
        )
    return steps


def _step_inputs(
    definition: SkillDefinition | None,
    *,
    order: int,
    outcome: str,
    constraints: tuple[str, ...],
    segment: str,
) -> list[str]:
    values = [
        f"requested outcome: {outcome}" if order == 1 else f"output of step-{order - 1}",
        f"outcome fragment this step answers: {segment}",
        *(f"constraint: {text}" for text in constraints),
        *(str(text) for text in (definition.required_inputs if definition else ())),
    ]
    return _dedupe(values)


def _step_output(definition: SkillDefinition | None, capability: str) -> str:
    outputs = [text for text in (str(item).strip() for item in (definition.expected_outputs if definition else ())) if text]
    return "; ".join(outputs) if outputs else f"{capability} result"


def _step_evidence_boundary(item: _ResolvedCapability, output: str) -> str:
    family = family_for_workflow(item.capability)
    not_yet = _string_list(family.get("not_evidence_until_observed"))
    unproven = (
        f"It does not prove {', '.join(not_yet)}."
        if not_yet
        else "It does not prove anything downstream of this step."
    )
    boundary = item.evidence_boundary.strip()
    return " ".join(
        text
        for text in (
            f"Completing this step proves only that `{item.capability}` produced: {output}.",
            unproven,
            boundary,
        )
        if text
    )


def _missing_capabilities(steps: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "schema_version": WORKFLOW_COMPOSITION_GAP_SCHEMA_VERSION,
            "capability": step["capability"],
            "capability_family": step["capability_family"],
            "needed_for_step": step["step_id"],
            "reason": (
                f"`{step['capability']}` is needed for {step['step_id']} but is not among the capabilities "
                "available to this composition."
            ),
            "next_action": WORKFLOW_COMPOSITION_GAP_NEXT_ACTION,
        }
        for step in steps
        if step.get("capability_status") == "missing"
    ]


def _next_action(state: str, steps: list[dict[str, object]], gaps: list[dict[str, object]]) -> str:
    if state == "not_compound":
        return "use_single_capability_plan"
    if state == "no_composable_path":
        return "ask_which_outcomes_to_compose"
    if gaps:
        return "report_missing_capabilities"
    if any(step.get("owner") == CODING_OWNER_CHOICE_PENDING for step in steps):
        return "choose_coding_owner"
    return "present_ordered_workflow"


def _composition_id(
    outcome: str,
    constraints: tuple[str, ...],
    coding_owner: str,
    available_key: tuple[str, ...] | None,
    revision: str,
) -> str:
    seed = json.dumps(
        {
            "outcome": outcome,
            "constraints": list(constraints),
            "coding_owner": coding_owner,
            "available": list(available_key) if available_key is not None else None,
            "catalog_revision": revision,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return "composition-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def _definitions_by_name() -> dict[str, SkillDefinition]:
    return {definition.name: definition for definition in routable_definitions()}


def _definition_for(capability: str) -> SkillDefinition | None:
    return _definitions_by_name().get(capability)


def _intent_errors(intent: object) -> list[str]:
    if not isinstance(intent, dict):
        return ["compound_intent must be an object"]
    errors = _key_set_errors(intent, WORKFLOW_COMPOSITION_INTENT_KEYS, "compound_intent")
    if intent.get("schema_version") != WORKFLOW_COMPOSITION_INTENT_SCHEMA_VERSION:
        errors.append(f"compound_intent schema_version must be {WORKFLOW_COMPOSITION_INTENT_SCHEMA_VERSION}")
    if not isinstance(intent.get("recognized"), bool):
        errors.append("compound_intent recognized must be a boolean")
    if not str(intent.get("reason", "")).strip():
        errors.append("compound_intent reason must say why the request is or is not compound")
    for key in ("segments", "connectors", "capability_families"):
        if not isinstance(intent.get(key), list):
            errors.append(f"compound_intent {key} must be a list")
    return errors


def _steps_errors(payload: dict[str, Any], state: object) -> list[str]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        return ["steps must be a list"]
    errors: list[str] = []
    if payload.get("step_count") != len(steps):
        errors.append("step_count must match steps")
    if state == "composed" and len(steps) < 2:
        errors.append("a composed workflow must carry at least two steps")
    if state in {"not_compound", "no_composable_path"} and steps:
        errors.append(f"state {state} must not carry steps: a single capability is not a workflow")
    for index, step in enumerate(steps):
        for error in _step_errors(step, index):
            errors.append(f"steps[{index}]: {error}")
    return errors


def _step_errors(step: object, index: int) -> list[str]:
    if not isinstance(step, dict):
        return ["step must be an object"]
    errors = _key_set_errors(step, WORKFLOW_COMPOSITION_STEP_KEYS, "step")
    if step.get("schema_version") != WORKFLOW_COMPOSITION_STEP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {WORKFLOW_COMPOSITION_STEP_SCHEMA_VERSION}")
    if step.get("order") != index + 1:
        errors.append("order must be the 1-based position in steps")
    if step.get("step_id") != f"step-{index + 1}":
        errors.append("step_id must be step-<order>")
    for field in WORKFLOW_COMPOSITION_REQUIRED_STEP_FIELDS:
        # A field absent from the step is already reported by the key-set check
        # above; this catches the one present but saying nothing.
        if field in step and _is_blank(step.get(field)):
            errors.append(f"step must name {field}")
    if step.get("capability_status") not in WORKFLOW_COMPOSITION_CAPABILITY_STATES:
        errors.append(f"capability_status must be one of {list(WORKFLOW_COMPOSITION_CAPABILITY_STATES)}")
    if not isinstance(step.get("inputs"), list):
        errors.append("inputs must be a list")
    owner_kind = step.get("owner_kind")
    owner = step.get("owner")
    family = step.get("capability_family")
    if owner_kind not in WORKFLOW_COMPOSITION_OWNER_KINDS:
        errors.append(f"owner_kind must be one of {list(WORKFLOW_COMPOSITION_OWNER_KINDS)}")
    elif owner_kind == "delegated_coding":
        if owner == HERMES_RETAINED_OWNER:
            errors.append("a delegated coding step must not be owned by hermes")
        elif owner not in WORKFLOW_COMPOSITION_CODING_OWNERS:
            errors.append(f"a delegated coding step owner must be one of {list(WORKFLOW_COMPOSITION_CODING_OWNERS)}")
        if family != DELEGATED_CODING_FAMILY:
            errors.append(f"only a {DELEGATED_CODING_FAMILY} step may be delegated")
    else:
        if owner != HERMES_RETAINED_OWNER:
            errors.append(f"a hermes_retained step must be owned by {HERMES_RETAINED_OWNER}")
        if family == DELEGATED_CODING_FAMILY:
            errors.append(f"a {DELEGATED_CODING_FAMILY} step must be delegated, not retained by hermes")
    return errors


def _gap_errors(payload: dict[str, Any]) -> list[str]:
    gaps = payload.get("missing_capabilities")
    if not isinstance(gaps, list):
        return ["missing_capabilities must be a list"]
    errors: list[str] = []
    steps = payload.get("steps")
    missing_steps = (
        {
            str(step.get("step_id", ""))
            for step in steps
            if isinstance(step, dict) and step.get("capability_status") == "missing"
        }
        if isinstance(steps, list)
        else set()
    )
    reported: set[str] = set()
    for index, gap in enumerate(gaps):
        if not isinstance(gap, dict):
            errors.append(f"missing_capabilities[{index}]: gap must be an object")
            continue
        for error in _key_set_errors(gap, WORKFLOW_COMPOSITION_GAP_KEYS, "gap"):
            errors.append(f"missing_capabilities[{index}]: {error}")
        if gap.get("schema_version") != WORKFLOW_COMPOSITION_GAP_SCHEMA_VERSION:
            errors.append(
                f"missing_capabilities[{index}]: schema_version must be {WORKFLOW_COMPOSITION_GAP_SCHEMA_VERSION}"
            )
        if not str(gap.get("reason", "")).strip():
            errors.append(f"missing_capabilities[{index}]: reason must say why the capability is unavailable")
        if gap.get("next_action") != WORKFLOW_COMPOSITION_GAP_NEXT_ACTION:
            errors.append(f"missing_capabilities[{index}]: next_action must leave installation to the user")
        reported.add(str(gap.get("needed_for_step", "")))
    unreported = sorted(missing_steps - reported)
    if unreported:
        errors.append(f"every step with a missing capability must be reported: {', '.join(unreported)}")
    return errors


def _key_set_errors(payload: object, expected: tuple[str, ...], label: str) -> list[str]:
    if not isinstance(payload, dict):
        return [f"{label} must be an object"]
    present = set(payload)
    allowed = set(expected)
    errors: list[str] = []
    missing = sorted(allowed - present)
    if missing:
        errors.append(f"{label} is missing keys: {', '.join(missing)}")
    unexpected = sorted(present - allowed)
    if unexpected:
        errors.append(f"{label} has unexpected keys: {', '.join(unexpected)}")
    return errors


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _is_blank(value: object) -> bool:
    """Whether a required field is present but says nothing."""
    if isinstance(value, list):
        return not [item for item in value if str(item).strip()]
    return not str(value).strip()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _int_value(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _clone(value: Any) -> Any:
    value_type = type(value)
    if value_type is dict:
        return {key: _clone(item) for key, item in value.items()}
    if value_type is list:
        return [_clone(item) for item in value]
    return value
