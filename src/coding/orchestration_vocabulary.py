"""The orchestration vocabulary contract for Hermes-native coding delivery.

This module is a data contract, not a dispatch system — the same shape as
`model_routing.py`: pure data plus one payload function, no I/O, no clock, no
environment reads, no new dependency. It pins the nine terms that separate the
Hermes harness, the Hermes mixture-of-model path, and Maestro so routing copy,
cards, and docs stop blending them. Nothing here selects an executor and
nothing here prepares or dispatches a handoff.

Two facts the entries state and must not overclaim:

- ``claude-code`` is prompt-only on the Maestro RECORD axis. OMH prepares a
  prompt handoff for Claude Code and never runs it from that path, so the
  Maestro entry says "prepares a handoff for" and never claims it as the
  thing that runs the work. Execution is a separate, explicit axis: the
  fanout dispatch bridge (`omh coding fanout dispatch`, and its `omh coding
  run` single-run entry) already spawns local agent CLIs -- including
  claude-code -- as its own operator-invoked, per-invocation opt-in. A
  prepared prompt handoff and a bridge dispatch are two different records;
  neither claim substitutes for the other.
- Mixture and Maestro never share subagent-unit language. Mixture binds a
  model per role to Hermes subagents inside one harness; Maestro hands an
  external coding CLI the work after an explicit user choice.
"""

from __future__ import annotations

from typing import Final


ORCHESTRATION_VOCABULARY_SCHEMA_VERSION: Final[str] = "omh_orchestration_vocabulary/v1"

# The allowed audiences, in the order they widen: normal users see a term in
# ordinary chat copy, operators reach it through explicit commands, and
# maintainers meet it only in source and contract tests.
ORCHESTRATION_VOCABULARY_AUDIENCES: Final[tuple[str, ...]] = (
    "normal_user",
    "operator",
    "maintainer",
)

# The one-line reason `src/routing/action_copy.py` renders for the
# `choose_executor` wrapper action. It lives here because the coding-owner
# question is the entry point of this vocabulary: the user answers it in
# natural language and never needs the word "maestro".
CHOOSE_EXECUTOR_REASON: Final[str] = "asking the user to pick the coding agent"

# The Hermes-harness-default wording, shared verbatim between the
# `hermes_harness` term below and every consumer that must state the default
# in the same words (#953): the `capability_reason` line of a ULW alias route
# quotes it so the reason copy cannot drift from the vocabulary.
HERMES_HARNESS_DEFAULT_WORDING: Final[str] = (
    "absent an explicit coding-owner choice, work runs inside the Hermes "
    "harness and no external coding CLI is selected"
)


_TERMS: Final[tuple[dict[str, object], ...]] = (
    {
        "term": "hermes_harness",
        "meaning": (
            "The default and normal execution path: "
            + HERMES_HARNESS_DEFAULT_WORDING
            + "."
        ),
        "audience": "normal_user",
        "entry_points": (),
        "machinery": (
            "omh.coding.executors.EXECUTOR_PROFILES",
            "omh.coding.executors.WORK_OWNER_MODES",
        ),
    },
    {
        "term": "hermes_mixture",
        "meaning": (
            "Hermes decomposes work into role-separated subagents and binds a "
            "model per role, all inside the Hermes harness; it is a per-role "
            "model binding, never a choice between coding products."
        ),
        "audience": "normal_user",
        "entry_points": (),
        "machinery": ("omh.coding.model_routing.MODEL_ROLES",),
    },
    {
        "term": "hermes_subagent",
        "meaning": (
            "A role-separated unit inside the Hermes harness — the unit the "
            "mixture binds a model to; never an external process."
        ),
        "audience": "normal_user",
        "entry_points": (),
        "machinery": ("omh.coding.model_routing.MODEL_ROLES",),
    },
    {
        "term": "coding_owner",
        "meaning": (
            "The answer to who implements the work: a work-owner mode plus, "
            "when external, the chosen profile; it is resolved by the user's "
            "explicit choice, never silently."
        ),
        "audience": "normal_user",
        "entry_points": ("choose_executor",),
        "machinery": (
            "omh.coding.executors.WORK_OWNER_MODES",
            "omh.routing.owner_preference.read_owner_preference",
        ),
    },
    {
        "term": "maestro",
        "meaning": (
            "The path by which an external coding CLI becomes the coding "
            "owner after an explicit user choice; OMH prepares a handoff for "
            "the chosen CLI on this path and never runs it here -- running it "
            "is a separate, explicit fanout-dispatch-bridge action, not this "
            "prepared record."
        ),
        "audience": "operator",
        "entry_points": ("choose_executor",),
        "machinery": (
            "omh.coding.executors.EXTERNAL_CLI_PROFILES",
            "omh.coding.maestro.facade.build_external_handoff",
        ),
    },
    {
        "term": "external_runtime_handoff",
        "meaning": (
            "A prepared handoff to exactly the omx-runtime, pi-family-runtime, or "
            "omc-runtime profiles; it shares the handoff builder with Maestro "
            "but is not Maestro."
        ),
        "audience": "operator",
        "entry_points": (),
        "machinery": ("omh.coding.maestro.facade.build_external_handoff",),
    },
    {
        "term": "fanout_dispatch",
        "meaning": (
            "The existing explicit local bridge that dispatches frozen work "
            "units to local agent CLIs; opt-in per command, never a default "
            "and never a merge authority."
        ),
        "audience": "operator",
        "entry_points": ("omh coding fanout dispatch",),
        "machinery": ("omh.coding.fanout_dispatch",),
    },
    {
        "term": "prepared",
        "meaning": (
            "The evidence state of an artifact OMH generated locally; "
            "prepared is never execution, review, CI, or merge evidence."
        ),
        "audience": "normal_user",
        "entry_points": (),
        "machinery": (),
    },
    {
        "term": "observed",
        "meaning": (
            "The evidence state of an outcome actually recorded from a real "
            "run; only observed evidence supports execution claims."
        ),
        "audience": "normal_user",
        "entry_points": (),
        "machinery": (),
    },
)


def orchestration_vocabulary_payload() -> dict[str, object]:
    """Return the versioned nine-term orchestration vocabulary contract."""
    return {
        "schema_version": ORCHESTRATION_VOCABULARY_SCHEMA_VERSION,
        "terms": tuple(dict(entry) for entry in _TERMS),
    }
