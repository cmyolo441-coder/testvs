"""One typed Loop operation service behind both the CLI and the native tool.

`omh loop ...` and the `omh_loop` plugin tool are two adapters over the same
vocabulary. Before this module existed, the argparse handlers *were* the
contract: every lifecycle rule reached the workflow functions through one
handler, so a second caller had to re-derive argument shapes, error text, and
output keys, and the two would drift the first time a rule moved.

Three things live here and nowhere else:

* ``LOOP_OPERATION_SPECS`` -- the canonical action manifest. Which actions
  exist, which mutate, which need a loop id, which accept the stale-mutation
  guard, which the native tool exposes, and which JSON keys the CLI prints for
  each one. Every mirror (the tool schema, the CLI adapters, the docs) reads
  this instead of restating it.
* ``run_loop_operation`` -- the raising form, used by the CLI so an existing
  ``OmhError(str(exc))`` handler keeps its exact message and exit behaviour.
* ``loop_operation_envelope`` -- the enveloped form, used by the plugin tool,
  which maps every failure onto a stable code from
  ``LOOP_OPERATION_ERROR_CODES`` instead of leaking an exception string into a
  model-facing payload.

The service records OMH control-plane state only. No action here dispatches an
executor, runs a command, opens a network connection, or observes anything on
the caller's behalf: a successful mutation is an OMH transition and nothing
more, which is what ``prepared_versus_observed`` in the envelope says.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from ..system.local_store import FileLockTimeout
from ..system.paths import OmhPaths
from ..system.record_revision import (
    ConflictingMutationReplay,
    MutationHistoryEvicted,
    StaleRecordMutation,
    record_revision_of,
)
from .goal_loop import (
    LOOP_ACTIONS,
    LOOP_DISPATCH_RECOVERY_OUTCOMES,
    LOOP_EXECUTOR_OPTION_IDS,
    LOOP_STICKY_RULE_DEFAULT_GAP,
    LOOP_STICKY_RULE_DEFAULT_MAX_REPEATS,
    LOOP_STICKY_RULE_REPEAT_MODES,
    LOOP_WORKFLOW_PATTERNS,
    PERMISSION_PROFILES,
    _guarded_cycle_update,
    assess_loopability,
    block_loop_queue_item,
    build_loop_cycle_narration,
    build_loop_goal_driver_handoff,
    build_loop_queue_handoff,
    build_loop_start_card,
    build_loop_status_card,
    create_loop_cycle,
    declare_sticky_rule,
    dispatch_loop_queue_item,
    inspect_loop_queue_item,
    list_loop_cycles,
    list_loop_queue,
    observe_codex_loop_queue_item,
    observe_loop_queue_item,
    read_loop_cycle,
    record_loop_feedback,
    record_loop_goal_driver_observation,
    recover_loop_queue_item_dispatch,
    run_loop_once_result,
    tick_loop_runtime,
    update_loop_permission,
    validate_loop_cycle,
)
from .loop_driver_updates import bind_driver, migrate_driver
from .loop_executor_observations import LoopDriverError

LOOP_OPERATION_REQUEST_SCHEMA: Final[str] = "omh_loop_request/v1"
LOOP_OPERATION_RESULT_SCHEMA: Final[str] = "omh_loop_result/v1"

LOOP_OPERATION_CLAIM_BOUNDARY: Final[str] = (
    "This is an OMH Loop control-plane record. A successful action may record an OMH "
    "transition; it never means an executor was dispatched, code was implemented, a review "
    "ran, CI passed, a change is merge-ready, or anything was merged. Queue items OMH "
    "prepares stay prepared_not_observed until separate observed evidence is recorded."
)

# Stable, model-facing failure vocabulary. The envelope reports one of these
# and a human-readable detail; a caller branches on the code, never the text.
UNKNOWN_ACTION: Final[str] = "unknown_action"
INVALID_REQUEST: Final[str] = "invalid_request"
IDENTITY_REQUIRED: Final[str] = "identity_required"
REVISION_REQUIRED: Final[str] = "revision_required"
LOOP_NOT_FOUND: Final[str] = "loop_not_found"
STALE_REVISION: Final[str] = "stale_revision"
UNPROVABLE_RETRY: Final[str] = "unprovable_retry"
CONFLICTING_REPLAY: Final[str] = "conflicting_replay"
LOOP_RULE_VIOLATION: Final[str] = "loop_rule_violation"
DRIVER_CONTRACT_VIOLATION: Final[str] = "driver_contract_violation"
STORE_UNAVAILABLE: Final[str] = "store_unavailable"
# Raised by the plugin bundle only, which cannot import this module to read the
# constant in the one situation that produces it: the installed OMH package is
# absent, so no Loop state exists to operate on and the tool fails closed. It
# belongs in this vocabulary anyway -- a caller branches on one closed set, not
# on which side of the boundary produced the code.
SERVICE_UNAVAILABLE: Final[str] = "service_unavailable"

LOOP_OPERATION_ERROR_CODES: Final[tuple[str, ...]] = (
    CONFLICTING_REPLAY,
    DRIVER_CONTRACT_VIOLATION,
    IDENTITY_REQUIRED,
    INVALID_REQUEST,
    LOOP_NOT_FOUND,
    LOOP_RULE_VIOLATION,
    REVISION_REQUIRED,
    SERVICE_UNAVAILABLE,
    STALE_REVISION,
    STORE_UNAVAILABLE,
    UNKNOWN_ACTION,
    UNPROVABLE_RETRY,
)

# Field value kinds. Deliberately few: the service validates shape, and a kind
# that needed its own parser would mean the CLI and the tool were sending
# different things under one name.
TEXT: Final[str] = "text"
TEXT_LIST: Final[str] = "text_list"
FLAG: Final[str] = "flag"
COUNT: Final[str] = "count"
OBJECT: Final[str] = "object"

FIELD_KINDS: Final[tuple[str, ...]] = (COUNT, FLAG, OBJECT, TEXT, TEXT_LIST)


@dataclass(frozen=True)
class LoopFieldSpec:
    """One named input of one action, with the shape both adapters must send."""

    name: str
    kind: str
    description: str
    required: bool = False
    enum: tuple[str, ...] = ()
    default: Any = None


@dataclass(frozen=True)
class LoopActionSpec:
    """One Loop operation: what it does, what it needs, and what it returns."""

    action: str
    cli: str
    summary: str
    mutating: bool
    requires_loop_id: bool
    revision_guard: bool
    native_tool: bool
    artifacts: tuple[str, ...]
    fields: tuple[LoopFieldSpec, ...] = ()

    def field_map(self) -> dict[str, LoopFieldSpec]:
        return {spec.name: spec for spec in self.fields}


def _text(name: str, description: str, *, required: bool = False, enum: tuple[str, ...] = (),
          default: Any = "") -> LoopFieldSpec:
    return LoopFieldSpec(name, TEXT, description, required=required, enum=enum, default=default)


def _texts(name: str, description: str, *, required: bool = False,
           enum: tuple[str, ...] = ()) -> LoopFieldSpec:
    return LoopFieldSpec(name, TEXT_LIST, description, required=required, enum=enum, default=())


def _flag(name: str, description: str) -> LoopFieldSpec:
    return LoopFieldSpec(name, FLAG, description, default=False)


def _count(name: str, description: str, *, default: int = 0) -> LoopFieldSpec:
    return LoopFieldSpec(name, COUNT, description, default=default)


_LOOP_CARD_ARTIFACTS: Final[tuple[str, ...]] = ("loop", "status_card")

_ACTION_SPECS: Final[tuple[LoopActionSpec, ...]] = (
    LoopActionSpec(
        "assess",
        "omh loop assess",
        "Classify a goal as loopable before any loop state exists.",
        mutating=False,
        requires_loop_id=False,
        revision_guard=False,
        native_tool=True,
        artifacts=("loopability_assessment",),
        fields=(
            _text("message", "Goal text to classify.", required=True),
            _flag("include_goal", "Echo the raw goal text back in the assessment."),
        ),
    ),
    LoopActionSpec(
        "start_card",
        "omh loop start-card",
        "Shape an ambitious goal into a loop start card without creating loop state.",
        mutating=False,
        requires_loop_id=False,
        revision_guard=False,
        native_tool=False,
        artifacts=("loop_start_card",),
        fields=(
            _text("message", "Goal text to shape into a start card.", required=True),
            _flag("include_goal", "Echo the raw goal text back in the card."),
            _text("source", "Origin label recorded on the card.", default="omh"),
            _text("permission_profile", "Default permission profile offered by the card.",
                  enum=PERMISSION_PROFILES, default="handoff_only"),
            _text("default_executor", "Default executor option offered by the card.",
                  enum=LOOP_EXECUTOR_OPTION_IDS, default="choose"),
        ),
    ),
    LoopActionSpec(
        "start",
        "omh loop start",
        "Create one loop_cycle/v2 record for an accepted, bounded loop goal.",
        mutating=True,
        requires_loop_id=False,
        revision_guard=False,
        native_tool=True,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("goal_summary", "The goal as the user stated it.", required=True),
            _text("goal_reframe", "The bounded first loop goal OMH will actually work on.", required=True),
            _texts("success_criteria", "Observable criteria that would end this loop.", required=True),
            _text("permission_profile", "Authority envelope profile.", enum=PERMISSION_PROFILES,
                  default="handoff_only"),
            _texts("allowed_executors", "Executors this loop may hand off to."),
            _texts("allow_actions", "Loop actions to allow explicitly.", enum=LOOP_ACTIONS),
            _texts("forbid_actions", "Loop actions to forbid explicitly.", enum=LOOP_ACTIONS),
            _text("linked_goal_id", "Existing goal ledger id to bind completion evidence to."),
            _text("source", "Origin label recorded on the cycle.", default="omh"),
            _text("loop_id", "Caller-chosen loop id; omit to derive one from the goal."),
            _flag("allow_unloopable", "Operator escape hatch when the assessment recommends another surface."),
            _text("executor", "Driver executor selection.", default="hermes"),
            _text("work_kind", "Driver work kind.", enum=("coding", "non_coding"), default="non_coding"),
            LoopFieldSpec("capability_snapshot", OBJECT,
                          "Executor capability snapshot that would enable external goal controls."),
            _text("executor_session_ref", "Executor session reference for the selected driver."),
        ),
    ),
    LoopActionSpec(
        "status",
        "omh loop status",
        "Read one loop with its status card, or list every loop when no loop id is given.",
        mutating=False,
        requires_loop_id=False,
        revision_guard=False,
        native_tool=True,
        artifacts=(),
        fields=(_text("loop_id", "Loop to read; omit to list every loop."),),
    ),
    LoopActionSpec(
        "feedback",
        "omh loop feedback",
        "Record one feedback cycle: observed artifacts, an internal gap, or an external wait.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=True,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _texts("observed_artifacts", "References to artifacts already observed this cycle."),
            _text("internal_gap", "The one actionable gap the loop can close locally."),
            _text("external_wait", "What the loop is waiting on outside itself."),
            _flag("context_exhausted", "Checkpoint because the working context ran out."),
            _flag("budget_exhausted", "Checkpoint because the budget ran out."),
        ),
    ),
    LoopActionSpec(
        "permit",
        "omh loop permit",
        "Widen or narrow the authority envelope; forbid always wins over allow.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=True,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _texts("allow_actions", "Loop actions to allow.", enum=LOOP_ACTIONS),
            _texts("forbid_actions", "Loop actions to forbid.", enum=LOOP_ACTIONS),
            _texts("allowed_executors", "Executors to add to the envelope."),
        ),
    ),
    LoopActionSpec(
        "tick",
        "omh loop tick",
        "Prepare the next runtime queue item for this loop.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("trigger", "What caused this tick.",
                  enum=("manual", "scheduled", "wrapper", "automation"), default="manual"),
            _text("cadence", "Cadence label for a scheduled tick."),
            _text("worktree_base", "Worktree base path planned for this tick."),
            _text("worktree_branch", "Worktree branch planned for this tick."),
            _text("subagent_role", "Subagent role planned for this tick."),
            _text("connector", "Connector planned for this tick."),
            _text("connector_action", "Connector action planned for this tick."),
            _text("workflow_pattern", "Workflow pattern for the prepared item.",
                  enum=LOOP_WORKFLOW_PATTERNS, default="single_step"),
            _text("note", "Operator note stored with the prepared item."),
        ),
    ),
    LoopActionSpec(
        "sticky_rule_declare",
        "omh loop sticky-rule declare",
        "Declare a rule the loop re-reads on a cadence instead of forgetting it.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("rule_id", "Stable id for this rule.", required=True),
            _text("text", "The rule itself.", required=True),
            _text("repeat_mode", "How often the rule is re-attached.",
                  enum=LOOP_STICKY_RULE_REPEAT_MODES, default="after_gap"),
            _count("repeat_gap", "Attachments to skip between repeats.",
                   default=LOOP_STICKY_RULE_DEFAULT_GAP),
            _count("max_repeats", "Maximum number of repeats.",
                   default=LOOP_STICKY_RULE_DEFAULT_MAX_REPEATS),
        ),
    ),
    LoopActionSpec(
        "run_once",
        "omh loop run-once",
        "Take one legal advancement: prepare at most one queue item, or report why none was taken.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=True,
        artifacts=("loop", "run_once", "status_card"),
    ),
    LoopActionSpec(
        "goal_driver_handoff",
        "omh loop goal-driver-handoff",
        "Build the prepared handoff text for the selected goal driver. Preparing is not dispatching.",
        mutating=False,
        requires_loop_id=True,
        revision_guard=False,
        native_tool=False,
        artifacts=("goal_driver_handoff",),
        fields=(
            _texts("gate_commands", "Gate commands the driver must run."),
            _count("max_turns", "Turn ceiling for the handoff."),
        ),
    ),
    LoopActionSpec(
        "goal_driver_observe",
        "omh loop goal-driver-observe",
        "Record one bounded observation from the loop's goal driver.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=True,
        artifacts=("goal_driver_observation", "native_goal_status", "loop", "status_card"),
        fields=(
            # Not named `observation`: every OMH plugin tool already spells the
            # host's own invocation metadata `observation`, and one tool where
            # the word means a loop payload instead would be a trap.
            LoopFieldSpec("driver_observation", OBJECT,
                          "The driver observation object, already parsed.", required=True),
        ),
    ),
    LoopActionSpec(
        "driver_bind",
        "omh loop driver-bind",
        "Bind an external executor goal driver from an observed capability snapshot.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            LoopFieldSpec("binding_observation", OBJECT,
                          "The binding observation object, already parsed.", required=True),
        ),
    ),
    LoopActionSpec(
        "migrate_driver",
        "omh loop migrate-driver",
        "Report, or with apply perform, the loop_cycle/v1 to v2 driver migration.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=("loop", "applied", "status_card"),
        fields=(_flag("apply", "Write the migration instead of only reporting it."),),
    ),
    LoopActionSpec(
        "queue_list",
        "omh loop queue list",
        "List this loop's runtime queue items.",
        mutating=False,
        requires_loop_id=True,
        revision_guard=False,
        native_tool=False,
        artifacts=("loop_queue",),
        fields=(_flag("include_observed", "Include items that already carry observed evidence."),),
    ),
    LoopActionSpec(
        "queue_inspect",
        "omh loop queue inspect",
        "Inspect one queue item in full.",
        mutating=False,
        requires_loop_id=True,
        revision_guard=False,
        native_tool=False,
        artifacts=(),
        fields=(_text("queue_id", "Queue item to inspect.", required=True),),
    ),
    LoopActionSpec(
        "queue_handoff",
        "omh loop queue handoff",
        "Build the prepared handoff for one queue item. Preparing is not dispatching.",
        mutating=False,
        requires_loop_id=True,
        revision_guard=False,
        native_tool=False,
        artifacts=("queue_handoff",),
        fields=(_text("queue_id", "Queue item to hand off.", required=True),),
    ),
    LoopActionSpec(
        "queue_dispatch",
        "omh loop queue dispatch",
        "Record that an operator dispatched one queue item to an executor.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("queue_id", "Queue item dispatched.", required=True),
            _text("executor", "Executor the item was dispatched to.",
                  enum=LOOP_EXECUTOR_OPTION_IDS, required=True),
            _text("session_ref", "Executor session reference."),
            _text("thread_ref", "Executor thread reference."),
            _texts("evidence_refs", "References proving the dispatch happened."),
            _text("summary", "Short metadata-only summary."),
        ),
    ),
    LoopActionSpec(
        "queue_recover_dispatch",
        "omh loop queue recover-dispatch",
        "Close out a failed or unknown dispatch attempt and record the replacement.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("queue_id", "Queue item being recovered.", required=True),
            _text("prior_attempt_id", "Dispatch attempt being closed out."),
            _text("prior_outcome", "What happened to the prior attempt.",
                  enum=LOOP_DISPATCH_RECOVERY_OUTCOMES, required=True),
            _texts("outcome_evidence_refs", "References proving the prior outcome.", required=True),
            _text("outcome_summary", "Short metadata-only summary of the prior outcome."),
            _text("executor", "Executor for the replacement dispatch.",
                  enum=LOOP_EXECUTOR_OPTION_IDS, required=True),
            _text("session_ref", "Executor session reference."),
            _text("thread_ref", "Executor thread reference."),
            _texts("evidence_refs", "References proving the replacement dispatch happened."),
            _text("summary", "Short metadata-only summary."),
        ),
    ),
    LoopActionSpec(
        "queue_observe_codex",
        "omh loop queue observe-codex",
        "Observe one queue item from a Codex session log.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=("loop", "narration"),
        fields=(
            _text("queue_id", "Queue item observed.", required=True),
            _text("codex_log_text", "The already-read Codex log text."),
            _text("codex_log_ref", "Reference to the Codex log."),
            _texts("evidence_refs", "References proving the observation.", required=True),
            _text("dispatch_attempt_id", "Dispatch attempt this observation closes."),
            _text("summary", "Short metadata-only summary."),
        ),
    ),
    LoopActionSpec(
        "queue_narrate",
        "omh loop queue narrate",
        "Render the operator-facing narration for this loop or one of its queue items.",
        mutating=False,
        requires_loop_id=True,
        revision_guard=False,
        native_tool=False,
        artifacts=("narration",),
        fields=(_text("queue_id", "Queue item to narrate; omit for the whole loop."),),
    ),
    LoopActionSpec(
        "queue_observe",
        "omh loop queue observe",
        "Record observed evidence against one prepared queue item, moving it out of prepared_not_observed.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=True,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("queue_id", "Queue item observed.", required=True),
            _texts("evidence_refs", "References to the observed evidence.", required=True),
            _texts("worktree_evidence_refs", "Worktree evidence references."),
            _texts("subagent_evidence_refs", "Subagent evidence references."),
            _texts("connector_evidence_refs", "Connector evidence references."),
            _text("dispatch_attempt_id", "Dispatch attempt this observation closes."),
            _text("summary", "Short metadata-only summary."),
        ),
    ),
    LoopActionSpec(
        "queue_block",
        "omh loop queue block",
        "Mark one queue item blocked with the reason that blocks it.",
        mutating=True,
        requires_loop_id=True,
        revision_guard=True,
        native_tool=False,
        artifacts=_LOOP_CARD_ARTIFACTS,
        fields=(
            _text("queue_id", "Queue item to block.", required=True),
            _text("reason", "Why the item is blocked.", required=True),
        ),
    ),
)

LOOP_OPERATION_SPECS: Final[dict[str, LoopActionSpec]] = {spec.action: spec for spec in _ACTION_SPECS}
LOOP_OPERATION_ACTIONS: Final[tuple[str, ...]] = tuple(spec.action for spec in _ACTION_SPECS)
LOOP_TOOL_ACTIONS: Final[tuple[str, ...]] = tuple(
    spec.action for spec in _ACTION_SPECS if spec.native_tool
)

# What the loop's own next_action says to do next, expressed in this
# vocabulary. A loop whose next step has no tool action names the CLI path
# instead of silently suggesting the nearest tool action.
_NEXT_ACTION_ROUTES: Final[dict[str, str]] = {
    "continue_loop": "run_once",
    "observe_runtime_queue": "queue_observe",
    "record_feedback": "feedback",
    "record_external_wait": "feedback",
    "record_checkpoint": "feedback",
    "request_permission": "permit",
    "resolve_runtime_queue_blocker": "status",
    "show_loop_status": "status",
}


@dataclass(frozen=True)
class LoopOperationRequest:
    """One validated call into the service, from either adapter."""

    action: str
    loop_id: str = ""
    fields: Mapping[str, Any] = field(default_factory=dict)
    expected_revision: int | None = None
    mutation_id: str = ""


@dataclass(frozen=True)
class LoopOperationResult:
    """The outcome of one action: the CLI's artifacts plus envelope facts."""

    action: str
    loop_id: str
    artifacts: dict[str, Any]
    record_revision: int
    mutating: bool
    mutation_applied: bool
    warnings: tuple[str, ...] = ()


class LoopRequestError(ValueError):
    """A request that never reached the workflow: bad action, field, or guard."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(message)


def loop_operation_manifest() -> dict[str, Any]:
    """The canonical action manifest, for the tool schema, docs, and tests."""
    return {
        "schema_version": LOOP_OPERATION_REQUEST_SCHEMA,
        "result_schema_version": LOOP_OPERATION_RESULT_SCHEMA,
        "error_codes": list(LOOP_OPERATION_ERROR_CODES),
        "actions": [
            {
                "action": spec.action,
                "cli": spec.cli,
                "summary": spec.summary,
                "mutating": spec.mutating,
                "requires_loop_id": spec.requires_loop_id,
                "revision_guard": spec.revision_guard,
                "native_tool": spec.native_tool,
                "artifacts": list(spec.artifacts),
                "required_fields": [item.name for item in spec.fields if item.required],
                "optional_fields": [item.name for item in spec.fields if not item.required],
            }
            for spec in _ACTION_SPECS
        ],
        "claim_boundary": LOOP_OPERATION_CLAIM_BOUNDARY,
    }


def loop_action_spec(action: str) -> LoopActionSpec:
    spec = LOOP_OPERATION_SPECS.get(str(action))
    if spec is None:
        raise LoopRequestError(UNKNOWN_ACTION, f"unknown loop action: {action!r}")
    return spec


def loop_operation_error_code(exc: BaseException) -> str:
    """Map one failure onto the stable vocabulary. One mapping, both adapters."""
    if isinstance(exc, LoopRequestError):
        return exc.code
    if isinstance(exc, LoopDriverError):
        return DRIVER_CONTRACT_VIOLATION
    if isinstance(exc, StaleRecordMutation):
        return STALE_REVISION
    if isinstance(exc, MutationHistoryEvicted):
        return UNPROVABLE_RETRY
    if isinstance(exc, ConflictingMutationReplay):
        return CONFLICTING_REPLAY
    if isinstance(exc, FileNotFoundError):
        return LOOP_NOT_FOUND
    # FileLockTimeout is a TimeoutError and therefore an OSError; naming it
    # first keeps a contended store from being reported as a missing one.
    if isinstance(exc, (FileLockTimeout, OSError)):
        return STORE_UNAVAILABLE
    if isinstance(exc, TypeError):
        # A workflow function reached with a value of the wrong Python type is
        # a malformed request, not a loop rule the caller broke.
        return INVALID_REQUEST
    # Everything the workflow layer raises to say "this loop does not permit
    # that" is a ValueError, and so is the default: a code invented for one
    # unclassified failure would be a vocabulary nobody can branch on.
    return LOOP_RULE_VIOLATION


# --- request validation ----------------------------------------------------


def _field_value(spec: LoopActionSpec, values: Mapping[str, Any], item: LoopFieldSpec) -> Any:
    if item.name not in values or values[item.name] is None:
        if item.required:
            raise LoopRequestError(
                INVALID_REQUEST, f"{spec.action} requires field {item.name!r}"
            )
        return () if item.kind == TEXT_LIST else item.default
    value = values[item.name]
    if item.kind == TEXT:
        if not isinstance(value, str):
            raise LoopRequestError(INVALID_REQUEST, f"{spec.action}.{item.name} must be a string")
        if item.required and not value.strip():
            raise LoopRequestError(INVALID_REQUEST, f"{spec.action}.{item.name} must not be blank")
        if item.enum and value and value not in item.enum:
            raise LoopRequestError(
                INVALID_REQUEST,
                f"{spec.action}.{item.name} must be one of {', '.join(item.enum)}",
            )
        return value
    if item.kind == TEXT_LIST:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise LoopRequestError(
                INVALID_REQUEST, f"{spec.action}.{item.name} must be an array of strings"
            )
        entries = tuple(value)
        if any(not isinstance(entry, str) for entry in entries):
            raise LoopRequestError(
                INVALID_REQUEST, f"{spec.action}.{item.name} must be an array of strings"
            )
        if item.required and not [entry for entry in entries if entry.strip()]:
            raise LoopRequestError(
                INVALID_REQUEST, f"{spec.action}.{item.name} must not be empty"
            )
        if item.enum:
            unknown = sorted({entry for entry in entries if entry and entry not in item.enum})
            if unknown:
                raise LoopRequestError(
                    INVALID_REQUEST,
                    f"{spec.action}.{item.name} has unknown value(s): {', '.join(unknown)}",
                )
        return entries
    if item.kind == FLAG:
        if not isinstance(value, bool):
            raise LoopRequestError(INVALID_REQUEST, f"{spec.action}.{item.name} must be a boolean")
        return value
    if item.kind == COUNT:
        if isinstance(value, bool) or not isinstance(value, int):
            raise LoopRequestError(INVALID_REQUEST, f"{spec.action}.{item.name} must be an integer")
        return value
    if not isinstance(value, dict):
        raise LoopRequestError(INVALID_REQUEST, f"{spec.action}.{item.name} must be an object")
    return value


def normalized_loop_fields(spec: LoopActionSpec, values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one action's fields, rejecting unknown names before any I/O."""
    known = spec.field_map()
    unknown = sorted(name for name in values if name not in known)
    if unknown:
        raise LoopRequestError(
            INVALID_REQUEST, f"{spec.action} does not accept field(s): {', '.join(unknown)}"
        )
    return {item.name: _field_value(spec, values, item) for item in spec.fields}


# --- execution -------------------------------------------------------------


def _revision_guard(request: LoopOperationRequest, spec: LoopActionSpec) -> dict[str, Any]:
    if not spec.revision_guard:
        return {}
    return {
        "expected_revision": request.expected_revision,
        "mutation_id": request.mutation_id or None,
    }


def _card(paths: OmhPaths, loop_id: str) -> dict[str, Any]:
    return build_loop_status_card(paths, loop_id)


def _with_card(paths: OmhPaths, loop_id: str, cycle: dict[str, Any]) -> dict[str, Any]:
    return {"loop": cycle, "status_card": _card(paths, loop_id)}


def _status_artifacts(paths: OmhPaths, loop_id: str) -> dict[str, Any]:
    if loop_id:
        return {"loop": read_loop_cycle(paths, loop_id), "status_card": _card(paths, loop_id)}
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for loop in list_loop_cycles(paths):
        validation = validate_loop_cycle(loop)
        if not validation["ok"]:
            invalid.append(
                {"loop_id": str(loop.get("loop_id", "unknown")), "errors": validation["errors"]}
            )
            continue
        runtime = loop.get("runtime") if isinstance(loop.get("runtime"), dict) else {}
        valid.append(
            {
                "loop_id": loop["loop_id"],
                "phase": loop["phase"],
                "wait_reason": loop["wait_reason"],
                "permission_profile": loop["authority_envelope"]["permission_profile"],
                "linked_goal_id": loop.get("linked_goal_id", ""),
                "next_action": loop["next_action"],
                "heartbeat_count": runtime.get("heartbeat_count", 0),
                "last_planned_action": runtime.get("last_planned_action", ""),
            }
        )
    return {"loops": valid, "invalid_loops": invalid}


def _run(paths: OmhPaths, request: LoopOperationRequest, spec: LoopActionSpec,
         values: dict[str, Any]) -> dict[str, Any]:
    loop_id = request.loop_id
    guard = _revision_guard(request, spec)
    action = spec.action
    if action == "assess":
        return {
            "loopability_assessment": assess_loopability(
                values["message"], expose_goal=values["include_goal"]
            )
        }
    if action == "start_card":
        return {
            "loop_start_card": build_loop_start_card(
                values["message"],
                include_goal=values["include_goal"],
                source=values["source"],
                default_permission_profile=values["permission_profile"],
                default_executor=values["default_executor"],
            )
        }
    if action == "start":
        cycle = create_loop_cycle(
            paths,
            goal_summary=values["goal_summary"],
            goal_reframe=values["goal_reframe"],
            success_criteria=list(values["success_criteria"]),
            permission_profile=values["permission_profile"],
            allowed_executors=list(values["allowed_executors"]),
            allow_actions=list(values["allow_actions"]),
            forbid_actions=list(values["forbid_actions"]),
            linked_goal_id=values["linked_goal_id"],
            source=values["source"],
            loop_id=values["loop_id"] or None,
            allow_unloopable=values["allow_unloopable"],
            driver_selection={
                "executor": values["executor"],
                "work_kind": values["work_kind"],
                "capability_snapshot": values["capability_snapshot"],
                "session_ref": values["executor_session_ref"] or None,
            },
        )
        return _with_card(paths, str(cycle["loop_id"]), cycle)
    if action == "status":
        return _status_artifacts(paths, values["loop_id"])
    if action == "feedback":
        cycle = record_loop_feedback(
            paths,
            loop_id,
            observed_artifacts=list(values["observed_artifacts"]),
            internal_gap=values["internal_gap"],
            external_wait=values["external_wait"],
            context_exhausted=values["context_exhausted"],
            budget_exhausted=values["budget_exhausted"],
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "permit":
        cycle = update_loop_permission(
            paths,
            loop_id,
            allow_actions=list(values["allow_actions"]),
            forbid_actions=list(values["forbid_actions"]),
            allowed_executors=list(values["allowed_executors"]),
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "tick":
        cycle = tick_loop_runtime(
            paths,
            loop_id,
            trigger=values["trigger"],
            cadence=values["cadence"],
            worktree_base=values["worktree_base"],
            worktree_branch=values["worktree_branch"],
            subagent_role=values["subagent_role"],
            connector=values["connector"],
            connector_action=values["connector_action"],
            workflow_pattern=values["workflow_pattern"],
            note=values["note"],
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "sticky_rule_declare":
        cycle = declare_sticky_rule(
            paths,
            loop_id,
            rule_id=values["rule_id"],
            text=values["text"],
            repeat_mode=values["repeat_mode"],
            repeat_gap=values["repeat_gap"],
            max_repeats=values["max_repeats"],
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "run_once":
        result = run_loop_once_result(paths, loop_id, **guard)
        return {
            "loop": result["loop"],
            "run_once": result["run_once"],
            "status_card": _card(paths, loop_id),
        }
    if action == "goal_driver_handoff":
        return {
            "goal_driver_handoff": build_loop_goal_driver_handoff(
                paths,
                loop_id,
                gate_commands=list(values["gate_commands"]),
                max_turns=values["max_turns"],
            )
        }
    if action == "goal_driver_observe":
        cycle = record_loop_goal_driver_observation(
            paths, loop_id, values["driver_observation"], **guard
        )
        card = _card(paths, loop_id)
        observations = (
            cycle["executor_goal_observations"]
            if card["driver"]["kind"] == "external_executor_goal"
            else cycle["goal_driver_observations"]
        )
        return {
            "goal_driver_observation": observations[-1],
            "native_goal_status": card["native_goal_status"],
            "loop": cycle,
            "status_card": card,
        }
    if action == "driver_bind":
        observation = values["binding_observation"]
        # expected_revision only, matching the surface these two have shipped
        # with: their parsers accept --mutation-id and have never forwarded it.
        # Forwarding it here would change a documented CLI behaviour inside a
        # change whose contract is that it does not.
        cycle = _guarded_cycle_update(
            paths,
            loop_id,
            lambda current: bind_driver(current, observation),
            operation="bind_loop_driver",
            expected_revision=request.expected_revision,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "migrate_driver":
        applied = values["apply"]
        cycle = (
            _guarded_cycle_update(
                paths,
                loop_id,
                migrate_driver,
                operation="migrate_loop_driver",
                expected_revision=request.expected_revision,
            )
            if applied
            else read_loop_cycle(paths, loop_id)
        )
        return {"loop": cycle, "applied": applied, "status_card": _card(paths, loop_id)}
    if action == "queue_list":
        return {
            "loop_queue": list_loop_queue(
                paths, loop_id, include_observed=values["include_observed"]
            )
        }
    if action == "queue_inspect":
        return inspect_loop_queue_item(paths, loop_id, values["queue_id"])
    if action == "queue_handoff":
        return {"queue_handoff": build_loop_queue_handoff(paths, loop_id, values["queue_id"])}
    if action == "queue_dispatch":
        cycle = dispatch_loop_queue_item(
            paths,
            loop_id,
            values["queue_id"],
            executor=values["executor"],
            session_ref=values["session_ref"],
            thread_ref=values["thread_ref"],
            evidence_refs=list(values["evidence_refs"]),
            summary=values["summary"],
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "queue_recover_dispatch":
        cycle = recover_loop_queue_item_dispatch(
            paths,
            loop_id,
            values["queue_id"],
            prior_attempt_id=values["prior_attempt_id"],
            prior_outcome=values["prior_outcome"],
            outcome_evidence_refs=list(values["outcome_evidence_refs"]),
            outcome_summary=values["outcome_summary"],
            executor=values["executor"],
            session_ref=values["session_ref"],
            thread_ref=values["thread_ref"],
            evidence_refs=list(values["evidence_refs"]),
            summary=values["summary"],
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "queue_observe_codex":
        cycle = observe_codex_loop_queue_item(
            paths,
            loop_id,
            values["queue_id"],
            codex_log_text=values["codex_log_text"],
            evidence_refs=list(values["evidence_refs"]),
            codex_log_ref=values["codex_log_ref"],
            summary=values["summary"],
            dispatch_attempt_id=values["dispatch_attempt_id"],
            **guard,
        )
        return {
            "loop": cycle,
            "narration": build_loop_cycle_narration(paths, loop_id, values["queue_id"]),
        }
    if action == "queue_narrate":
        return {"narration": build_loop_cycle_narration(paths, loop_id, values["queue_id"])}
    if action == "queue_observe":
        cycle = observe_loop_queue_item(
            paths,
            loop_id,
            values["queue_id"],
            evidence_refs=list(values["evidence_refs"]),
            worktree_evidence_refs=list(values["worktree_evidence_refs"]),
            subagent_evidence_refs=list(values["subagent_evidence_refs"]),
            connector_evidence_refs=list(values["connector_evidence_refs"]),
            summary=values["summary"],
            dispatch_attempt_id=values["dispatch_attempt_id"],
            **guard,
        )
        return _with_card(paths, loop_id, cycle)
    if action == "queue_block":
        cycle = block_loop_queue_item(
            paths, loop_id, values["queue_id"], reason=values["reason"], **guard
        )
        return _with_card(paths, loop_id, cycle)
    # Unreachable: loop_action_spec() already refused an unknown action, and
    # every action in the manifest is handled above. A new manifest entry with
    # no branch must fail loudly rather than return an empty artifact set.
    raise LoopRequestError(UNKNOWN_ACTION, f"loop action {action!r} has no implementation")


def run_loop_operation(paths: OmhPaths, request: LoopOperationRequest) -> LoopOperationResult:
    """Run one action and return its artifacts, raising the original failure.

    The CLI adapters call this inside the `except` clauses they already had,
    so their error text and exit codes are unchanged.
    """
    spec = loop_action_spec(request.action)
    values = normalized_loop_fields(spec, request.fields)
    if spec.requires_loop_id and not str(request.loop_id).strip():
        raise LoopRequestError(INVALID_REQUEST, f"{spec.action} requires a loop id")
    if not spec.revision_guard and request.expected_revision is not None:
        raise LoopRequestError(
            INVALID_REQUEST, f"{spec.action} does not accept a revision guard"
        )
    before = _revision_before(paths, request, spec)
    artifacts = _run(paths, request, spec, values)
    loop_id = str(request.loop_id or _artifact_loop_id(artifacts))
    after = _artifact_revision(artifacts)
    if after == 0 and loop_id:
        after = _current_revision(paths, loop_id)
    return LoopOperationResult(
        action=spec.action,
        loop_id=loop_id,
        artifacts=artifacts,
        record_revision=after,
        mutating=spec.mutating,
        mutation_applied=bool(spec.mutating and after > before),
    )


def _artifact_loop_id(artifacts: Mapping[str, Any]) -> str:
    cycle = artifacts.get("loop")
    if isinstance(cycle, dict):
        return str(cycle.get("loop_id", ""))
    return ""


def _artifact_revision(artifacts: Mapping[str, Any]) -> int:
    cycle = artifacts.get("loop")
    return record_revision_of(cycle) if isinstance(cycle, dict) else 0


def _current_revision(paths: OmhPaths, loop_id: str) -> int:
    try:
        return record_revision_of(read_loop_cycle(paths, loop_id))
    except (OSError, ValueError):
        return 0


def _revision_before(paths: OmhPaths, request: LoopOperationRequest, spec: LoopActionSpec) -> int:
    if not spec.mutating or not str(request.loop_id).strip():
        return 0
    return _current_revision(paths, str(request.loop_id))


# --- envelope --------------------------------------------------------------


def _next_actions(artifacts: Mapping[str, Any]) -> list[dict[str, str]]:
    card = artifacts.get("status_card")
    if not isinstance(card, dict):
        return []
    loop_next = str(card.get("next_action", ""))
    if not loop_next:
        return []
    routed = _NEXT_ACTION_ROUTES.get(loop_next, "")
    row: dict[str, str] = {"loop_next_action": loop_next}
    if routed:
        row["omh_loop_action"] = routed
        row["cli"] = LOOP_OPERATION_SPECS[routed].cli
    else:
        # A next step outside the tool vocabulary names its CLI path rather
        # than the nearest tool action: routing it to a neighbour would be an
        # invented instruction, not a projection of loop state.
        row["omh_loop_action"] = ""
        row["cli"] = "omh loop status"
    return [row]


def _prepared_versus_observed(spec: LoopActionSpec, mutation_applied: bool) -> dict[str, Any]:
    return {
        "omh_transition_recorded": bool(mutation_applied),
        "executor_dispatched": False,
        "implementation_observed": False,
        "review_observed": False,
        "ci_observed": False,
        "merge_ready": False,
        "merged": False,
        "mutating_action": bool(spec.mutating),
        "claim_boundary": LOOP_OPERATION_CLAIM_BOUNDARY,
    }


LOOP_TOOL_REDACTED_ARTIFACTS: Final[frozenset[str]] = frozenset({"loop"})

# Failures whose exception text is the operating system's, not this repo's.
# `FileNotFoundError` stringifies to the absolute path it could not open, and
# an `OSError` carries whatever the platform put in it, so publishing either
# one hands a model the host's filesystem layout in place of an explanation.
# Every other code in the vocabulary is raised by a workflow function with a
# sentence written here, which is what a caller should read.
_OS_DERIVED_DETAILS: Final[dict[str, str]] = {
    LOOP_NOT_FOUND: (
        "no loop with that id exists in the configured OMH home; list the loops "
        "with action=status and no loop_id"
    ),
    STORE_UNAVAILABLE: (
        "the OMH loop store could not be read or written; the home may be "
        "unavailable, unwritable, or held by another writer"
    ),
}
_REDACTED_HOME: Final[str] = "<omh_home>"


def _safe_detail(code: str, exc: BaseException, omh_home: str) -> str:
    """Say what failed without naming where the store lives.

    The written sentences cover the two codes whose text is the platform's.
    The home prefix is stripped from the rest because a workflow message can
    still quote a path it was handed -- a malformed `cycle.json` is reported as
    `"{path}: {error}"` -- and that is the same leak by another route.
    """
    written = _OS_DERIVED_DETAILS.get(code)
    if written is not None:
        return written
    text = str(exc)
    return text.replace(omh_home, _REDACTED_HOME) if omh_home else text


def loop_operation_envelope(
    paths: OmhPaths,
    request: LoopOperationRequest,
    *,
    redact: frozenset[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Run one action and return a stable success or failure envelope.

    Never raises: a model-facing caller branches on ``status`` and ``error``.
    ``redact`` drops artifacts the caller must not publish. The native tool
    passes ``LOOP_TOOL_REDACTED_ARTIFACTS`` so a chat response carries the
    status card -- the projection the Loop skill already reasons about --
    rather than the whole stored cycle with its transition and observation
    history. Dropping by name rather than selecting by name is deliberate:
    ``status`` returns two different shapes, and an allowlist would silently
    publish nothing for whichever shape it did not anticipate.
    """
    try:
        spec = loop_action_spec(request.action)
    except LoopRequestError as exc:
        return _failure_envelope(request, None, exc)
    try:
        result = run_loop_operation(paths, request)
    except (LoopRequestError, LoopDriverError, OSError, TypeError, ValueError) as exc:
        return _failure_envelope(request, spec, exc, omh_home=str(paths.omh_home))
    artifacts = dict(result.artifacts)
    if redact:
        artifacts = {key: value for key, value in artifacts.items() if key not in redact}
    envelope: dict[str, Any] = {
        "schema_version": LOOP_OPERATION_RESULT_SCHEMA,
        "status": "ok",
        "action": result.action,
        "loop_id": result.loop_id,
        "record_revision": result.record_revision,
        "mutation_applied": result.mutation_applied,
        "warnings": list(result.warnings),
        "next_actions": _next_actions(result.artifacts),
        "prepared_versus_observed": _prepared_versus_observed(spec, result.mutation_applied),
        "claim_boundary": LOOP_OPERATION_CLAIM_BOUNDARY,
    }
    envelope.update(artifacts)
    return envelope


def _failure_envelope(
    request: LoopOperationRequest,
    spec: LoopActionSpec | None,
    exc: BaseException,
    *,
    omh_home: str = "",
) -> dict[str, Any]:
    code = loop_operation_error_code(exc)
    return {
        "schema_version": LOOP_OPERATION_RESULT_SCHEMA,
        "status": "error",
        "action": str(request.action),
        "loop_id": str(request.loop_id),
        "record_revision": 0,
        "mutation_applied": False,
        "error": code,
        "error_detail": _safe_detail(code, exc, omh_home),
        "warnings": [],
        "next_actions": [],
        "prepared_versus_observed": (
            _prepared_versus_observed(spec, False)
            if spec is not None
            else {
                "omh_transition_recorded": False,
                "executor_dispatched": False,
                "implementation_observed": False,
                "review_observed": False,
                "ci_observed": False,
                "merge_ready": False,
                "merged": False,
                "mutating_action": False,
                "claim_boundary": LOOP_OPERATION_CLAIM_BOUNDARY,
            }
        ),
        "supported_actions": list(LOOP_TOOL_ACTIONS),
        "claim_boundary": LOOP_OPERATION_CLAIM_BOUNDARY,
    }


__all__ = [
    "LOOP_OPERATION_ACTIONS",
    "LOOP_OPERATION_CLAIM_BOUNDARY",
    "LOOP_OPERATION_ERROR_CODES",
    "LOOP_OPERATION_REQUEST_SCHEMA",
    "LOOP_OPERATION_RESULT_SCHEMA",
    "LOOP_OPERATION_SPECS",
    "LOOP_TOOL_ACTIONS",
    "LOOP_TOOL_REDACTED_ARTIFACTS",
    "LoopActionSpec",
    "LoopFieldSpec",
    "LoopOperationRequest",
    "LoopOperationResult",
    "LoopRequestError",
    "loop_action_spec",
    "loop_operation_envelope",
    "loop_operation_error_code",
    "loop_operation_manifest",
    "normalized_loop_fields",
    "run_loop_operation",
]
