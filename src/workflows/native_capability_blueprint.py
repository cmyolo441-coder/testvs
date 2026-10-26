"""Native experience blueprints: `native_capability_blueprint/v1` (issue #791).

What was missing
----------------

A useful feature demonstrated in some other agent tool arrives described in that
tool's own vocabulary -- its manifest, its registry entry, its tool definitions,
its hooks, its agents, its settings. None of that says how the behavior should
feel when a person types a sentence at Hermes, which OMH surfaces have to move
for it to exist, or where the claim stops.

`skills/native_capability_surfaces.py` shows the *shape* of what a native
capability touches: a skill definition, a harness, a surface exposure. It shows
it by hardcoding three specific capabilities. There was no contract for
describing a capability that does not exist yet, and so no way to check that a
description of one is complete before somebody starts building it.

What a blueprint is
-------------------

One design artifact for one capability OMH does not have. It names the intent,
the sentences a person would actually type, what Hermes asks back before it
answers, what it produces, which native surfaces have to change, what happens
when an input is missing, what the capability is explicitly not, and the
boundary on all of it.

It is not a claim that the capability exists. `CLAIM_BOUNDARY` says so on every
payload, `IMPLEMENTATION_CLAIM_KEYS` refuses a payload shaped to say otherwise
by key name, and the closed key set refuses the rest.

The surface vocabulary is checkable, not aspirational
-----------------------------------------------------

`NATIVE_CAPABILITY_SURFACES` is not a list of nouns somebody thought a
capability might touch. Every member names a structure that exists in this
repository and that a real gate fails on when a new capability misses it --
`NATIVE_CAPABILITY_SURFACE_ANCHORS` holds the file and the structure for each,
and `docs/ADDING-A-SKILL.md` is where that list comes from.

`REQUIRED_NATIVE_CAPABILITY_SURFACES` is the subset no installable capability
can avoid: the catalog entry, the harness and its evidence ladder, the routing
triggers, the awareness lane and context card, the wrapper action and its label,
the coverage case, the capability family, the regenerated docs, and the
exact-count fixtures. A blueprint that omits one is a validation error naming
the surface and the file it lives in -- which is #791 AC1 in its enforcing form,
and is what makes "identifies every affected surface" something a reader can
check rather than something an author can assert.

The four remaining surfaces are conditional by design. A capability that mints
no local artifact, reads no reviewed memory, prepares no coding handoff, and
needs no bespoke chat card genuinely does not touch them, and requiring them
would make the required list mean less rather than more.

Observed there, required here
-----------------------------

#791 AC3 is the boundary the rest of the contract rests on, and it is modelled
as two fields that cannot be confused for each other:

    observed_source_mechanics  -- what the source host does, as inspiration
    omh_runtime_requirements   -- what OMH will need at runtime

Both are closed vocabularies and they are disjoint. Naming a source-host
mechanic in the requirement list is refused by name, before the generic
unsupported-value check, because "unsupported value" is the wrong sentence for
it: the value is perfectly well-formed and the defect is that a foreign host
became load-bearing. Naming the same mechanic as observed inspiration is fine
and is what the field is for.

There is deliberately no field through which a mechanic can be promoted. AC3
says a source mechanic may not become a runtime requirement "without a separate
approved boundary", and a separate approved boundary is exactly what changing
this schema is: `OMH_RUNTIME_REQUIREMENTS` is a reviewed constant, so widening
it is a diff somebody signs off on rather than a payload somebody writes.

Determinism
-----------

`build_native_capability_blueprint` reads no clock. `prepared_at` is a
parameter and defaults to empty, and it is excluded from `blueprint_digest`
along with the digest itself -- a wall clock inside a compared value turns an
equality check into a race, which this repo has already lost Windows CI time to.

Closed vocabularies are normalized into their declared order before the digest
is taken, so two authors who list the same surfaces in different orders produce
the same digest.

What reads these today
----------------------

Stated because the vocabulary here is wider than the wiring. No production
surface mints a blueprint yet; this module is the contract and
`tests/test_native_capability_blueprint.py` is its only minting caller.

`native_capability_quality_gate` (#795) reads it: the surface vocabulary and the
anchors are the expected set it gates a capability against, and
`blueprint_expected_surfaces` is the accessor it joins on, which refuses an
invalid blueprint rather than returning a partial surface list a gate would
silently pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from ..system.append_only_store import RAW_OR_HIDDEN_KEYS


NATIVE_CAPABILITY_BLUEPRINT_SCHEMA_VERSION: Final[str] = "native_capability_blueprint/v1"

BLUEPRINT_PRIVACY: Final[str] = "metadata_only"

# On every payload. The last clause is the one #791 AC2 rests on: a blueprint is
# a design artifact for work nobody has done.
CLAIM_BOUNDARY: Final[str] = (
    "A native capability blueprint is an OMH-local design artifact describing how one capability should "
    "behave if it is built. It is not dispatch, execution, verification, review, CI, merge-readiness, or "
    "merge evidence, it is not a claim that the capability exists, is installed, or is available, and it "
    "never makes a source host's tools, hooks, agents, settings, manifests, or registries a runtime "
    "requirement of OMH."
)

# Every native surface a new OMH capability can move, in the order a blueprint
# renders them. Derived from `docs/ADDING-A-SKILL.md`: each member names a
# structure that exists in this tree and has a gate behind it.
NATIVE_CAPABILITY_SURFACES: Final[tuple[str, ...]] = (
    "skill_catalog",
    "harness",
    "evidence_ladder",
    "routing_triggers",
    "awareness_lane",
    "workflow_context_card",
    "wrapper_action",
    "next_action_label",
    "chat_card",
    "coverage_case",
    "capability_family",
    "generated_docs",
    "exact_count_fixtures",
    "runtime_artifact",
    "memory_policy",
    "coding_handoff",
)

# Where each surface actually lives. Quoted back in the refusal so a blueprint
# that misses a surface reports the file to edit rather than a bare noun.
NATIVE_CAPABILITY_SURFACE_ANCHORS: Final[dict[str, str]] = {
    "skill_catalog": "the SkillDefinition in src/skills/catalog.py",
    "harness": "the HarnessDefinition in src/skills/catalog.py, checked by `omh harness validate`",
    "evidence_ladder": (
        "HarnessDefinition.evidence_ladder in src/skills/catalog.py, alongside "
        "src/workflows/observation_journal.py"
    ),
    "routing_triggers": "the phrase and token triggers in src/routing/chat.py, each with an overroute guard",
    "awareness_lane": "the lane skills lists in awareness_primer_payload() in src/plugin_bundle/omh/awareness.py",
    "workflow_context_card": "_WORKFLOW_CONTEXT_CARD_BY_WORKFLOW in src/plugin_bundle/omh/awareness.py",
    "wrapper_action": "VISIBLE_ACTIONS and _ACK_PRIMARY_ACTIONS_BY_NEXT_ACTION in src/wrapper/contract.py",
    "next_action_label": "NEXT_ACTION_LABELS in src/routing/action_copy.py",
    "chat_card": "a *_CHAT_CARDS entry or a bespoke renderer in src/wrapper/contract.py",
    "coverage_case": (
        "a ChatCardCoverageCase in src/quality/chat_card_coverage.py or a RoutingInterventionCase in "
        "src/quality/routing_precision.py"
    ),
    "capability_family": (
        "src/capabilities/families.py, projected into src/plugin_bundle/omh/tools/capability_families.json"
    ),
    "generated_docs": "docs/WORKFLOWS.md, docs/ROLES.md, and skills/*/SKILL.md, regenerated from catalog data",
    "exact_count_fixtures": (
        "the exact-count assertions in tests/test_routing_precision.py, tests/test_cli.py, "
        "tests/test_hermes_ux_quality.py, and tests/test_release_smoke.py"
    ),
    "runtime_artifact": "a metadata-only schema-versioned artifact under .omh/, read through src/runtime/artifacts.py",
    "memory_policy": "reviewed OMH-local project memory under .omh/memory/",
    "coding_handoff": "a prepared coding handoff payload from src/coding/coding_delegation.py",
}

# The surfaces no installable capability can avoid. #791 AC1's enforcing set.
REQUIRED_NATIVE_CAPABILITY_SURFACES: Final[tuple[str, ...]] = (
    "skill_catalog",
    "harness",
    "evidence_ladder",
    "routing_triggers",
    "awareness_lane",
    "workflow_context_card",
    "wrapper_action",
    "next_action_label",
    "coverage_case",
    "capability_family",
    "generated_docs",
    "exact_count_fixtures",
)

# Genuinely conditional: a capability that mints no artifact, reads no reviewed
# memory, prepares no coding handoff, and needs no bespoke card does not touch
# these, and requiring them anyway would make the required list mean less.
OPTIONAL_NATIVE_CAPABILITY_SURFACES: Final[tuple[str, ...]] = tuple(
    surface for surface in NATIVE_CAPABILITY_SURFACES if surface not in REQUIRED_NATIVE_CAPABILITY_SURFACES
)

# How a source host packages, exposes, or runs the demonstrated feature. Every
# member is something that belongs to the other system: a manifest it reads, a
# registry it resolves against, an extension API it exposes, a process it runs.
# A blueprint may record any of them as what was observed. None of them may
# appear in `omh_runtime_requirements`.
SOURCE_HOST_MECHANICS: Final[tuple[str, ...]] = (
    "host_plugin_manifest",
    "host_plugin_registry",
    "host_marketplace_listing",
    "host_extension_api",
    "host_tool_definition",
    "host_hook",
    "host_agent_definition",
    "host_setting",
    "host_slash_command",
    "host_background_service",
    "host_remote_endpoint",
)

# What OMH is allowed to need at runtime. Every member is local, deterministic,
# and reachable without the source host: catalog data this repo ships, a
# contract this repo defines, an artifact under .omh/, or something the caller
# hands in. Widening this tuple is the "separate approved boundary" #791 AC3
# names -- a reviewed diff, never a value in a payload.
OMH_RUNTIME_REQUIREMENTS: Final[tuple[str, ...]] = (
    "local_catalog_metadata",
    "local_deterministic_contract",
    "installed_hermes_skill",
    "metadata_only_local_artifact",
    "reviewed_local_memory",
    "prepared_coding_handoff",
    "wrapper_supplied_input",
    "user_supplied_input",
    "omh_cli_command",
)

# What Hermes does before it answers. `answer_directly` is the only policy that
# asks nothing, and it is the only one that may carry no clarifying questions.
CLARIFICATION_POLICIES: Final[tuple[str, ...]] = (
    "answer_directly",
    "clarify_then_answer",
    "clarify_before_handoff",
    "always_clarify_first",
)

# What the capability does when a required input is absent. Four behaviors and
# no "best effort" member: guessing past a missing input is how a prepared
# artifact starts reading like an observed one.
DEGRADATION_BEHAVIORS: Final[tuple[str, ...]] = (
    "ask_for_missing_input",
    "render_partial_with_not_observed",
    "explain_gap_and_stop",
    "route_to_another_workflow",
)

# `docs/DIRECTION.md`, Ownership Boundary, restated as vocabularies so a
# blueprint has to place every part of the capability on one side of the line.
HERMES_RETAINED_WORK: Final[tuple[str, ...]] = (
    "chat_intake",
    "clarification",
    "source_backed_research",
    "planning",
    "skill_and_workflow_narration",
    "user_facing_status_continuity",
)
DELEGATED_WORK: Final[tuple[str, ...]] = (
    "main_implementation_work",
    "code_changes",
    "code_review_fixes",
    "verification_execution",
    "merge_readiness_work",
)

NATIVE_CAPABILITY_BLUEPRINT_KEYS: Final[tuple[str, ...]] = (
    "affected_surfaces",
    "blueprint_digest",
    "capability_id",
    "claim_boundary",
    "clarification_policy",
    "clarifying_questions",
    "degradation_behavior",
    "delegated_work",
    "evidence_steps",
    "example_requests",
    "expected_outputs",
    "hermes_retained_work",
    "intent_summary",
    "non_goals",
    "observed_source_mechanics",
    "omh_runtime_requirements",
    "prepared_at",
    "privacy",
    "schema_version",
)

# Exactly what `blueprint_digest` seals: the design, and nothing else. The three
# module constants are excluded because they are not design decisions, and
# `prepared_at` is excluded because a clock inside a compared value is a race.
BLUEPRINT_DIGEST_KEYS: Final[tuple[str, ...]] = (
    "affected_surfaces",
    "capability_id",
    "clarification_policy",
    "clarifying_questions",
    "degradation_behavior",
    "delegated_work",
    "evidence_steps",
    "example_requests",
    "expected_outputs",
    "hermes_retained_work",
    "intent_summary",
    "non_goals",
    "observed_source_mechanics",
    "omh_runtime_requirements",
)

# Key names under which "this capability exists" arrives. The closed key set
# already refuses every one of them, but it refuses them as unsupported keys,
# and the point of this family is that a design is not a delivery -- so the
# shape is named and refused by name first.
IMPLEMENTATION_CLAIM_KEYS: Final[frozenset[str]] = frozenset(
    {
        "available",
        "built",
        "ci_result",
        "delivered",
        "deployed",
        "enabled",
        "executed",
        "exit_code",
        "implemented",
        "installed",
        "live",
        "merged",
        "observed",
        "observed_at",
        "ran",
        "released",
        "result",
        "reviewed",
        "shipped",
        "status",
        "succeeded",
        "success",
        "verified",
    }
)

_LABEL: Final[str] = "native_capability_blueprint"

# The canonical skill name this blueprint is for: the identifier a catalog entry,
# a tap directory, and a routing key would all use.
_CAPABILITY_ID: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{1,48}$")

# An evidence-ladder step, in the shape the harness definitions already use.
_EVIDENCE_STEP: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{2,48}$")

# An example request is a sentence a person types, not a command they run.
# `docs/DIRECTION.md` rule 1: chat users stay free of command knowledge, so an
# example that teaches one is the wrong example.
_COMMAND_SHAPED: Final[re.Pattern[str]] = re.compile(r"^\s*(?:omh\b|/|-)|--")

# Free-text bounds. A blueprint is authored design prose, and these keep it from
# becoming somewhere a transcript can be parked.
_MAX_TEXT_LENGTH: Final[int] = 200
_MAX_EXAMPLE_REQUESTS: Final[int] = 8
_MIN_EXAMPLE_REQUESTS: Final[int] = 2
_MAX_CLARIFYING_QUESTIONS: Final[int] = 6
_MAX_EXPECTED_OUTPUTS: Final[int] = 8
_MAX_NON_GOALS: Final[int] = 8
_MAX_EVIDENCE_STEPS: Final[int] = 8
_MIN_EVIDENCE_STEPS: Final[int] = 2


class NativeCapabilityBlueprintError(ValueError):
    """Raised when a description of a capability cannot become a blueprint."""


# ---------------------------------------------------------------------------
# Surface vocabulary
# ---------------------------------------------------------------------------


def is_canonical_capability_id(value: Any) -> bool:
    """Whether one value is the canonical skill-name slug a capability is keyed by.

    Exposed because a capability is keyed by the same identifier wherever it is
    described -- a blueprint here, a quality gate in #795 -- and two modules
    deciding separately what that identifier looks like is how they end up
    disagreeing about which capability a payload is about.
    """
    return isinstance(value, str) and bool(_CAPABILITY_ID.match(value))


def blueprint_surface_anchor(surface: str) -> str:
    """The file and structure one surface name refers to, or an empty string."""
    return NATIVE_CAPABILITY_SURFACE_ANCHORS.get(str(surface), "")


def unknown_surfaces(surfaces: Iterable[str]) -> tuple[str, ...]:
    """Every named surface that is not in the vocabulary, sorted."""
    return tuple(sorted({str(surface) for surface in surfaces} - set(NATIVE_CAPABILITY_SURFACES)))


def missing_required_surfaces(surfaces: Iterable[str]) -> tuple[str, ...]:
    """Every required surface the caller did not name, in declared order."""
    named = {str(surface) for surface in surfaces}
    return tuple(surface for surface in REQUIRED_NATIVE_CAPABILITY_SURFACES if surface not in named)


def blueprint_expected_surfaces(blueprint: Mapping[str, Any]) -> tuple[str, ...]:
    """The surfaces one blueprint names, in `NATIVE_CAPABILITY_SURFACES` order.

    The accessor a downstream quality gate joins on. It refuses an invalid
    blueprint outright rather than returning whatever surface list happens to be
    on the payload: a gate handed a partial list would pass a capability for
    surfaces nobody promised, which is the failure the surface vocabulary exists
    to prevent.
    """
    errors = validate_native_capability_blueprint(blueprint)
    if errors:
        raise NativeCapabilityBlueprintError(errors[0])
    named = {str(surface) for surface in blueprint.get("affected_surfaces", [])}
    return tuple(surface for surface in NATIVE_CAPABILITY_SURFACES if surface in named)


# ---------------------------------------------------------------------------
# Build and validate
# ---------------------------------------------------------------------------


def build_native_capability_blueprint(
    *,
    capability_id: str,
    intent_summary: str,
    example_requests: Sequence[str],
    clarification_policy: str,
    hermes_retained_work: Sequence[str],
    expected_outputs: Sequence[str],
    affected_surfaces: Sequence[str],
    omh_runtime_requirements: Sequence[str],
    evidence_steps: Sequence[str],
    degradation_behavior: str,
    non_goals: Sequence[str],
    clarifying_questions: Sequence[str] = (),
    delegated_work: Sequence[str] = (),
    observed_source_mechanics: Sequence[str] = (),
    prepared_at: str = "",
) -> dict[str, Any]:
    """Mint one blueprint, or refuse.

    Closed-vocabulary lists are normalized into their declared order and
    deduplicated, so two authors describing the same capability in different
    orders produce the same `blueprint_digest`. Free-text lists keep the order
    they were written in, because the order examples are read in is a design
    decision.

    Nothing here reads a clock. `prepared_at` is whatever the caller passes and
    is excluded from the digest.
    """
    blueprint: dict[str, Any] = {
        "schema_version": NATIVE_CAPABILITY_BLUEPRINT_SCHEMA_VERSION,
        "capability_id": str(capability_id).strip().casefold(),
        "intent_summary": str(intent_summary).strip(),
        "example_requests": _authored_list(example_requests),
        "clarification_policy": str(clarification_policy).strip(),
        "clarifying_questions": _authored_list(clarifying_questions),
        "hermes_retained_work": _vocabulary_list(hermes_retained_work, HERMES_RETAINED_WORK),
        "delegated_work": _vocabulary_list(delegated_work, DELEGATED_WORK),
        "expected_outputs": _authored_list(expected_outputs),
        "affected_surfaces": _vocabulary_list(affected_surfaces, NATIVE_CAPABILITY_SURFACES),
        "observed_source_mechanics": _vocabulary_list(observed_source_mechanics, SOURCE_HOST_MECHANICS),
        "omh_runtime_requirements": _vocabulary_list(omh_runtime_requirements, OMH_RUNTIME_REQUIREMENTS),
        "evidence_steps": _authored_list(evidence_steps),
        "degradation_behavior": str(degradation_behavior).strip(),
        "non_goals": _authored_list(non_goals),
        "prepared_at": str(prepared_at).strip(),
        "privacy": BLUEPRINT_PRIVACY,
        "claim_boundary": CLAIM_BOUNDARY,
        "blueprint_digest": "",
    }
    blueprint["blueprint_digest"] = blueprint_digest_of(blueprint)
    errors = validate_native_capability_blueprint(blueprint)
    if errors:
        raise NativeCapabilityBlueprintError(errors[0])
    return blueprint


def blueprint_digest_of(blueprint: Mapping[str, Any]) -> str:
    """A sha256 over the design content of a blueprint, covering no clock."""
    identity = {key: blueprint.get(key) for key in BLUEPRINT_DIGEST_KEYS}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def validate_native_capability_blueprint(blueprint: Mapping[str, Any]) -> list[str]:
    """Every reason one payload is not a native capability blueprint."""
    if not isinstance(blueprint, Mapping):
        return [f"{_LABEL} must be an object"]
    errors: list[str] = []
    claims_implementation = sorted(key for key in blueprint if str(key).lower() in IMPLEMENTATION_CLAIM_KEYS)
    if claims_implementation:
        errors.append(
            f"{_LABEL} must not carry implementation-claim keys: {claims_implementation}; a blueprint "
            "describes a capability that does not exist yet and never reports one that does"
        )
    forbidden = sorted(key for key in blueprint if str(key).lower() in RAW_OR_HIDDEN_KEYS)
    if forbidden:
        errors.append(f"{_LABEL} must not carry raw or hidden keys: {forbidden}")
    unexpected = sorted(
        set(blueprint) - set(NATIVE_CAPABILITY_BLUEPRINT_KEYS) - set(claims_implementation) - set(forbidden)
    )
    if unexpected:
        errors.append(f"{_LABEL} has unsupported keys: {unexpected}")
    missing = sorted(set(NATIVE_CAPABILITY_BLUEPRINT_KEYS) - set(blueprint))
    if missing:
        errors.append(f"{_LABEL} is missing keys: {missing}")
    if blueprint.get("schema_version") != NATIVE_CAPABILITY_BLUEPRINT_SCHEMA_VERSION:
        errors.append(f"{_LABEL} schema_version must be {NATIVE_CAPABILITY_BLUEPRINT_SCHEMA_VERSION}")
    if blueprint.get("privacy") != BLUEPRINT_PRIVACY:
        errors.append(f"{_LABEL} privacy must be {BLUEPRINT_PRIVACY}")
    if blueprint.get("claim_boundary") != CLAIM_BOUNDARY:
        errors.append(
            f"{_LABEL} claim_boundary must state the blueprint boundary: a design artifact is not evidence "
            "and is not a claim that the capability exists"
        )
    if not isinstance(blueprint.get("prepared_at"), str):
        errors.append(f"{_LABEL} prepared_at must be a string")
    errors.extend(_identity_errors(blueprint))
    errors.extend(_experience_errors(blueprint))
    errors.extend(_ownership_errors(blueprint))
    errors.extend(_surface_errors(blueprint))
    errors.extend(_boundary_errors(blueprint))
    errors.extend(_digest_errors(blueprint))
    return errors


# ---------------------------------------------------------------------------
# Validation parts
# ---------------------------------------------------------------------------


def _identity_errors(blueprint: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    capability_id = blueprint.get("capability_id")
    if not is_canonical_capability_id(capability_id):
        errors.append(
            f"{_LABEL} capability_id must be the canonical skill name as a lowercase slug: {capability_id!r}"
        )
    errors.extend(_text_errors(blueprint.get("intent_summary"), "intent_summary"))
    return errors


def _experience_errors(blueprint: Mapping[str, Any]) -> list[str]:
    """#791 AC2's first half: the sentences a person would actually type."""
    errors = _text_list_errors(
        blueprint.get("example_requests"),
        "example_requests",
        minimum=_MIN_EXAMPLE_REQUESTS,
        maximum=_MAX_EXAMPLE_REQUESTS,
    )
    if not errors:
        commands = [
            example
            for example in blueprint.get("example_requests", [])
            if _COMMAND_SHAPED.search(str(example))
        ]
        if commands:
            errors.append(
                f"{_LABEL} example_requests must be natural-language requests, not commands: {commands}"
            )
    policy = blueprint.get("clarification_policy")
    if policy not in CLARIFICATION_POLICIES:
        errors.append(f"{_LABEL} clarification_policy is unsupported: {policy!r}")
    questions = blueprint.get("clarifying_questions")
    errors.extend(
        _text_list_errors(questions, "clarifying_questions", minimum=0, maximum=_MAX_CLARIFYING_QUESTIONS)
    )
    if isinstance(questions, list):
        if policy == "answer_directly" and questions:
            errors.append(
                f"{_LABEL} clarifying_questions must be empty when the policy answers directly; a capability "
                "that asks nothing has no question to record"
            )
        if policy in CLARIFICATION_POLICIES and policy != "answer_directly" and not questions:
            errors.append(
                f"{_LABEL} clarifying_questions must name what Hermes asks before it answers under the "
                f"{policy} policy"
            )
    errors.extend(
        _text_list_errors(
            blueprint.get("expected_outputs"), "expected_outputs", minimum=1, maximum=_MAX_EXPECTED_OUTPUTS
        )
    )
    if blueprint.get("degradation_behavior") not in DEGRADATION_BEHAVIORS:
        errors.append(
            f"{_LABEL} degradation_behavior is unsupported: {blueprint.get('degradation_behavior')!r}"
        )
    errors.extend(_text_list_errors(blueprint.get("non_goals"), "non_goals", minimum=1, maximum=_MAX_NON_GOALS))
    return errors


def _ownership_errors(blueprint: Mapping[str, Any]) -> list[str]:
    """`docs/DIRECTION.md` Ownership Boundary, as a check.

    Hermes always does something -- at minimum it takes the request in -- so an
    empty retained list means the blueprint never said where the capability
    lives. Delegated work is legitimately empty: plenty of capabilities hand
    nothing to a coding owner.
    """
    errors = _vocabulary_list_errors(
        blueprint.get("hermes_retained_work"), "hermes_retained_work", HERMES_RETAINED_WORK, minimum=1
    )
    errors.extend(
        _vocabulary_list_errors(blueprint.get("delegated_work"), "delegated_work", DELEGATED_WORK, minimum=0)
    )
    steps = blueprint.get("evidence_steps")
    errors.extend(
        _text_list_errors(steps, "evidence_steps", minimum=_MIN_EVIDENCE_STEPS, maximum=_MAX_EVIDENCE_STEPS)
    )
    if isinstance(steps, list):
        malformed = [step for step in steps if not _EVIDENCE_STEP.match(str(step))]
        if malformed:
            errors.append(
                f"{_LABEL} evidence_steps must be snake_case ladder steps like the harness definitions use: "
                f"{malformed}"
            )
    return errors


def _surface_errors(blueprint: Mapping[str, Any]) -> list[str]:
    """#791 AC1: the blueprint identifies every affected native OMH surface."""
    surfaces = blueprint.get("affected_surfaces")
    if not isinstance(surfaces, list) or not all(isinstance(surface, str) for surface in surfaces):
        return [f"{_LABEL} affected_surfaces must be a list of strings"]
    errors: list[str] = []
    if len(set(surfaces)) != len(surfaces):
        errors.append(f"{_LABEL} affected_surfaces must not repeat a surface")
    unknown = unknown_surfaces(surfaces)
    if unknown:
        errors.append(
            f"{_LABEL} affected_surfaces names surfaces that do not exist in this repository: {list(unknown)}; "
            f"the vocabulary is {list(NATIVE_CAPABILITY_SURFACES)}"
        )
    absent = missing_required_surfaces(surfaces)
    if absent:
        described = [f"{surface} ({blueprint_surface_anchor(surface)})" for surface in absent]
        errors.append(
            f"{_LABEL} affected_surfaces is missing required surfaces: {described}; every installable "
            "capability moves all of them, so a blueprint that omits one is incomplete rather than narrow"
        )
    return errors


def _boundary_errors(blueprint: Mapping[str, Any]) -> list[str]:
    """#791 AC3: observed there never becomes required here."""
    errors = _vocabulary_list_errors(
        blueprint.get("observed_source_mechanics"),
        "observed_source_mechanics",
        SOURCE_HOST_MECHANICS,
        minimum=0,
    )
    requirements = blueprint.get("omh_runtime_requirements")
    if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
        return [*errors, f"{_LABEL} omh_runtime_requirements must be a list of strings"]
    smuggled = sorted(set(requirements) & set(SOURCE_HOST_MECHANICS))
    if smuggled:
        errors.append(
            f"{_LABEL} omh_runtime_requirements must not name a source-host mechanic: {smuggled}; a mechanic "
            "may be recorded in observed_source_mechanics as inspiration, and making one load-bearing at "
            "runtime needs a separate approved boundary that this schema does not carry"
        )
    errors.extend(
        _vocabulary_list_errors(
            [item for item in requirements if item not in set(smuggled)],
            "omh_runtime_requirements",
            OMH_RUNTIME_REQUIREMENTS,
            minimum=1,
        )
    )
    return errors


def _digest_errors(blueprint: Mapping[str, Any]) -> list[str]:
    digest = blueprint.get("blueprint_digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return [f"{_LABEL} blueprint_digest must be a sha256 hex digest"]
    if digest != blueprint_digest_of(blueprint):
        return [
            f"{_LABEL} blueprint_digest does not match the design it seals; the payload was edited after it "
            "was minted"
        ]
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _authored_list(values: Sequence[str]) -> list[str]:
    """Authored prose, in the order it was written, deduplicated."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            cleaned.append(text)
            seen.add(text)
    return cleaned


def _vocabulary_list(values: Sequence[str], vocabulary: tuple[str, ...]) -> list[str]:
    """Closed-vocabulary members in declared order, plus anything unrecognised.

    Unrecognised values are kept rather than dropped: silently discarding one
    would turn a caller's typo into a blueprint that validates and describes
    something nobody asked for.
    """
    named = [str(value).strip() for value in values if str(value).strip()]
    known = [item for item in vocabulary if item in named]
    unknown = sorted({item for item in named if item not in set(vocabulary)})
    return [*known, *unknown]


def _text_errors(value: object, field: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{_LABEL} {field} must be a non-empty string"]
    if len(value) > _MAX_TEXT_LENGTH:
        return [f"{_LABEL} {field} must be at most {_MAX_TEXT_LENGTH} characters"]
    return []


def _text_list_errors(value: object, field: str, *, minimum: int, maximum: int) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{_LABEL} {field} must be a list of strings"]
    errors: list[str] = []
    if len(value) < minimum:
        errors.append(f"{_LABEL} {field} must name at least {minimum}")
    if len(value) > maximum:
        errors.append(f"{_LABEL} {field} must name at most {maximum}")
    if len(set(value)) != len(value):
        errors.append(f"{_LABEL} {field} must not repeat an entry")
    blank = [item for item in value if not item.strip()]
    overlong = [item for item in value if len(item) > _MAX_TEXT_LENGTH]
    if blank:
        errors.append(f"{_LABEL} {field} must not contain an empty entry")
    if overlong:
        errors.append(f"{_LABEL} {field} entries must be at most {_MAX_TEXT_LENGTH} characters")
    return errors


def _vocabulary_list_errors(
    value: object,
    field: str,
    vocabulary: tuple[str, ...],
    *,
    minimum: int,
) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return [f"{_LABEL} {field} must be a list of strings"]
    errors: list[str] = []
    if len(value) < minimum:
        errors.append(f"{_LABEL} {field} must name at least {minimum}")
    if len(set(value)) != len(value):
        errors.append(f"{_LABEL} {field} must not repeat an entry")
    unsupported = sorted({item for item in value if item not in set(vocabulary)})
    if unsupported:
        errors.append(f"{_LABEL} {field} has unsupported entries: {unsupported}; allowed: {list(vocabulary)}")
    return errors
