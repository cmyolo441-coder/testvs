"""Host-side binding for the `omh_loop` tool; the lifecycle lives in OMH core.

Who owns what, and why:

* **OMH core** (`omh.workflows.loop_operations`) owns the Loop operation
  service: the action vocabulary, field validation, the workflow calls, the
  error-code mapping, and the response envelope. The lifecycle it drives is
  thousands of lines of `loop_cycle/v2` rules across a dozen modules. A copy of
  any of that in this bundle would be the CLI/plugin drift the tool exists to
  remove.
* **This module** owns everything core cannot see: which store this call is
  allowed to touch, whether the host gave a session identity, whether the
  caller armed the stale-mutation guard, and what a model-facing payload may
  publish. None of those are answerable from inside a workflow function.

The bundle stays loadable without the `omh` package, so the import is lazy and
guarded, and the failure is closed: with no installed package there is no Loop
state to read, so there is nothing to degrade to. A few core constants are
restated here as literals for the same reason -- `tests/test_loop_tool.py`
pins every one of them against its producer, so a drift is a test failure
rather than a silent disagreement.

A closed failure still says which failure it is: a `package_backend` block
from the shared classifier in `package_backend.py` (#2166).
"""

from __future__ import annotations

from typing import Any

from .package_backend import available_backend, load_backend, package_backend_detail


# Restated from `omh.workflows.loop_operations`; pinned by the parity tests.
LOOP_TOOL_RESULT_SCHEMA = "omh_loop_result/v1"
LOOP_TOOL_CLAIM_BOUNDARY = (
    "This is an OMH Loop control-plane record. A successful action may record an OMH "
    "transition; it never means an executor was dispatched, code was implemented, a review "
    "ran, CI passed, a change is merge-ready, or anything was merged. Queue items OMH "
    "prepares stay prepared_not_observed until separate observed evidence is recorded."
)

SERVICE_UNAVAILABLE = "service_unavailable"
UNKNOWN_ACTION = "unknown_action"
IDENTITY_REQUIRED = "identity_required"
REVISION_REQUIRED = "revision_required"
STORE_UNAVAILABLE = "store_unavailable"
INVALID_REQUEST = "invalid_request"

# The provenance this tool records on a loop it creates. The CLI records
# "omh"; a loop started from a chat turn is not the same origin and says so.
LOOP_TOOL_SOURCE = "omh_loop_tool"

# Every action `omh_loop` exposes, and the service fields each one accepts.
# The service refuses a field an action does not declare, so this map is what
# keeps one flat tool schema from sending `queue_id` into `permit`.
TOOL_ACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "assess": ("message", "include_goal"),
    "start": (
        "goal_summary",
        "goal_reframe",
        "success_criteria",
        "permission_profile",
        "allowed_executors",
        "allow_actions",
        "forbid_actions",
        "linked_goal_id",
        "loop_id",
        "allow_unloopable",
        "executor",
        "work_kind",
        "executor_session_ref",
    ),
    "status": ("loop_id",),
    "feedback": (
        "observed_artifacts",
        "internal_gap",
        "external_wait",
        "context_exhausted",
        "budget_exhausted",
    ),
    "permit": ("allow_actions", "forbid_actions", "allowed_executors"),
    "run_once": (),
    "goal_driver_observe": ("driver_observation",),
    "queue_observe": (
        "queue_id",
        "evidence_refs",
        "worktree_evidence_refs",
        "subagent_evidence_refs",
        "connector_evidence_refs",
        "dispatch_attempt_id",
        "summary",
    ),
}

LOOP_TOOL_ACTIONS: tuple[str, ...] = tuple(sorted(TOOL_ACTION_FIELDS))

# Actions that write. Each one needs the host's session or thread identity.
#
# What that gate is, precisely: a caller that cannot present a host session or
# thread id is not a Hermes tool dispatch, and this tool only exists to serve
# one. It is NOT an attribution record and must not be described as one -- the
# id is checked, reported back in `session_binding`, and never stored. A tool
# mutation and a CLI mutation write byte-identical `cycle.json`, and nothing in
# the OMH home says which session moved a loop.
#
# It is also not an ownership check, deliberately. A loop outlives the
# conversation that started it -- surviving context exhaustion and resuming
# later is the workflow's whole premise -- and the CLI, the operator path for
# every action this tool does not expose, carries no session at all. Binding a
# loop to its first session would strand it when that session ends and split
# the two adapters that this change exists to keep identical. Any session, and
# the CLI, may steer any loop; the stale-revision guard below is what keeps two
# of them from overwriting each other, and it works without knowing who they
# are.
MUTATING_TOOL_ACTIONS = frozenset(
    {"start", "feedback", "permit", "run_once", "goal_driver_observe", "queue_observe"}
)
# Mutations against a loop that already exists. The caller rendered a revision
# before deciding; submitting it is what makes a retry or a racing second
# session provably safe, so the tool refuses the call without it. `start`
# creates the record and has no prior revision to submit.
REVISION_REQUIRED_TOOL_ACTIONS = frozenset(MUTATING_TOOL_ACTIONS - {"start"})
# Actions that name an existing loop. `start` may carry a caller-chosen id as
# a field instead, and `status` lists every loop when none is given.
LOOP_ID_REQUIRED_TOOL_ACTIONS = frozenset(
    {"feedback", "permit", "run_once", "goal_driver_observe", "queue_observe"}
)
# Argument names that would name a store. The schema declares none of them, so
# they can only arrive from a caller trying anyway. Refusing by name is what
# makes the refusal observable on every host: silently dropping an unknown key
# binds correctly and still leaves the caller believing the root was honoured.
ROOT_SELECTING_ARGS = ("hermes_home", "omh_home", "paths", "root", "state_dir")
# Arguments that belong to the call itself rather than to one action, plus the
# host's own invocation metadata, which the handler strips before this module
# sees it. Everything else must be a field of the requested action.
ENVELOPE_ARGS = ("action", "expected_revision", "loop_id", "mutation_id", "observation")

# Native Loop's own path to the same store, named in every refusal.
_CLI_ROUTE = (
    "The omh loop CLI reads and writes the same Loop store when it is given this "
    "profile's OMH home: omh --omh-home <that home> loop status."
)


def _prepared_versus_observed(mutating: bool) -> dict[str, Any]:
    return {
        "omh_transition_recorded": False,
        "executor_dispatched": False,
        "implementation_observed": False,
        "review_observed": False,
        "ci_observed": False,
        "merge_ready": False,
        "merged": False,
        "mutating_action": bool(mutating),
        "claim_boundary": LOOP_TOOL_CLAIM_BOUNDARY,
    }


def error_envelope(action: str, loop_id: str, code: str, detail: str) -> dict[str, Any]:
    """The same failure shape core produces, for failures core never sees."""
    return {
        "schema_version": LOOP_TOOL_RESULT_SCHEMA,
        "status": "error",
        "action": str(action),
        "loop_id": str(loop_id),
        "record_revision": 0,
        "mutation_applied": False,
        "error": str(code),
        "error_detail": str(detail),
        "warnings": [],
        "next_actions": [],
        "prepared_versus_observed": _prepared_versus_observed(
            str(action) in MUTATING_TOOL_ACTIONS
        ),
        "supported_actions": list(LOOP_TOOL_ACTIONS),
        "claim_boundary": LOOP_TOOL_CLAIM_BOUNDARY,
    }


def _expected_revision(args: dict[str, Any]) -> int | None:
    value = args.get("expected_revision")
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _import_loop_service() -> tuple[Any, Any, Any, Any]:
    """The core Loop service; raises whatever this interpreter's import raises.

    Called per use and never cached here: whether `omh` imports is a property
    of the interpreter at that moment, and a module-level answer would outlive
    a package installed, or an import finished, after this bundle loaded.
    """
    from omh.system.paths import resolve_paths
    from omh.workflows.loop_operations import (
        LOOP_TOOL_REDACTED_ARTIFACTS,
        LoopOperationRequest,
        loop_operation_envelope,
    )

    return resolve_paths, LOOP_TOOL_REDACTED_ARTIFACTS, LoopOperationRequest, loop_operation_envelope


def loop_backend_status() -> dict[str, Any]:
    """The `package_backend` block for native Loop on this interpreter, resolved now.

    The import is the check: a top-level `omh` that lacks the Loop service is
    not a backend, and only importing what the tool imports can tell.
    """
    _service, failure = load_backend(_import_loop_service)
    return failure or available_backend()


def loop_backend_detail(backend: dict[str, Any]) -> str:
    """The `package_backend` sentence for native Loop, naming its CLI route."""
    return package_backend_detail(backend, feature="native Loop", cli_route=_CLI_ROUTE)


def run_loop_tool_action(args: dict[str, Any], *, session_ref: str) -> dict[str, Any]:
    """Bind, gate, and run one `omh_loop` action against the configured home."""
    action = str(args.get("action", ""))
    loop_id = str(args.get("loop_id", "") or "")
    if action not in TOOL_ACTION_FIELDS:
        return error_envelope(
            action,
            loop_id,
            UNKNOWN_ACTION,
            "action must be one of " + ", ".join(LOOP_TOOL_ACTIONS),
        )
    named_roots = sorted(name for name in ROOT_SELECTING_ARGS if name in args)
    if named_roots:
        return error_envelope(
            action,
            loop_id,
            INVALID_REQUEST,
            "omh_loop selects no store from its arguments; remove "
            + ", ".join(named_roots)
            + " and it will use the configured OMH home for the active profile",
        )
    if action in MUTATING_TOOL_ACTIONS and not session_ref:
        return error_envelope(
            action,
            loop_id,
            IDENTITY_REQUIRED,
            "this host supplied no session or thread identity, so this call is not "
            "a Hermes tool dispatch and omh_loop will not write; use the omh loop "
            "CLI for an operator or automation path",
        )
    # One flat schema means every field of every action passes host validation
    # for every action, so this is the only place a field sent to the wrong
    # action can be caught. Filtering it away instead would apply the rest of
    # the request and report `ok`, telling a caller its argument took effect
    # when it did not -- the same failure ROOT_SELECTING_ARGS above refuses by
    # name, and the CLI refuses it too, by exiting 2 on an unknown flag.
    foreign = sorted(
        name
        for name in args
        if name not in ENVELOPE_ARGS and name not in TOOL_ACTION_FIELDS[action]
    )
    if foreign:
        accepted = ", ".join(TOOL_ACTION_FIELDS[action]) or "no action-specific fields"
        return error_envelope(
            action,
            loop_id,
            INVALID_REQUEST,
            f"{action} does not accept field(s): " + ", ".join(foreign)
            + f"; it accepts {accepted}",
        )
    if action in LOOP_ID_REQUIRED_TOOL_ACTIONS and not loop_id:
        return error_envelope(action, loop_id, INVALID_REQUEST, f"{action} requires loop_id")
    expected_revision = _expected_revision(args)
    if action in REVISION_REQUIRED_TOOL_ACTIONS and expected_revision is None:
        return error_envelope(
            action,
            loop_id,
            REVISION_REQUIRED,
            "expected_revision is required for this action; read the loop with "
            "action=status and submit the record_revision it reports",
        )
    service, backend = load_backend(_import_loop_service)
    if service is None:
        envelope = error_envelope(
            action,
            loop_id,
            SERVICE_UNAVAILABLE,
            "No Loop state was read or written. " + loop_backend_detail(backend),
        )
        envelope["package_backend"] = backend
        return envelope
    resolve_paths, LOOP_TOOL_REDACTED_ARTIFACTS, LoopOperationRequest, loop_operation_envelope = service
    # Every remaining name is already known to belong to this action; a None
    # value means "not supplied" and takes the service's declared default.
    fields = {
        name: args[name]
        for name in TOOL_ACTION_FIELDS[action]
        if name in args and args[name] is not None
    }
    if action == "start":
        # Provenance is the tool's to record, never the caller's to choose.
        fields["source"] = LOOP_TOOL_SOURCE
    try:
        # No argument reaches this binding. The store is the configured OMH
        # home for the active profile and nothing else, which is why the
        # schema declares no home field for the model to supply.
        paths = resolve_paths()
    except (OSError, ValueError) as error:
        return error_envelope(
            action,
            loop_id,
            STORE_UNAVAILABLE,
            f"OMH runtime home binding is unavailable: {type(error).__name__}",
        )
    envelope = loop_operation_envelope(
        paths,
        LoopOperationRequest(
            action=action,
            loop_id=loop_id,
            fields=fields,
            expected_revision=expected_revision,
            mutation_id=str(args.get("mutation_id", "") or ""),
        ),
        redact=LOOP_TOOL_REDACTED_ARTIFACTS,
    )
    envelope["session_binding"] = {
        "bound": bool(session_ref),
        "session_ref": session_ref,
        "home_source": "configured_omh_home",
        "caller_selected_root": False,
    }
    return envelope


__all__ = [
    "ENVELOPE_ARGS",
    "LOOP_ID_REQUIRED_TOOL_ACTIONS",
    "LOOP_TOOL_ACTIONS",
    "LOOP_TOOL_CLAIM_BOUNDARY",
    "LOOP_TOOL_RESULT_SCHEMA",
    "LOOP_TOOL_SOURCE",
    "MUTATING_TOOL_ACTIONS",
    "ROOT_SELECTING_ARGS",
    "REVISION_REQUIRED_TOOL_ACTIONS",
    "TOOL_ACTION_FIELDS",
    "error_envelope",
    "loop_backend_detail",
    "loop_backend_status",
    "run_loop_tool_action",
]
