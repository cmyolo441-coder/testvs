from __future__ import annotations

import hashlib
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Mapping

from ..plugin_bundle.omh.run_records import (
    CODING_DELEGATION_FILE,
    EVENTS_FILE,
    EXECUTOR_SESSION_FILE,
    RUN_FILE,
)
from ..ingress import CHAT_SOURCES, compact_source_metadata, extract_message_text, extract_source_metadata
from ..routing.chat import CONFIDENCE_LEVELS
from ..executors import CODING_EXECUTOR_TARGETS, executor_label, executor_selection_for_target
from .lifecycle import report_codex_delegation_lifecycle, start_codex_delegation_lifecycle
from .session_handoff_projection import project_valid_session_handoffs
from ..local_store import ensure_dir, ensure_file, read_jsonl_objects, read_runtime_record, utc_now
from ..system.platform_envelope import compact_session_platform
from ..system.record_revision import (
    DuplicateMutationReplay,
    guarded_record_update,
    record_revision_of,
    require_not_terminal,
)
from ..memory import memory_recall_pack_for_handoff, memory_surface_for_source, record_attached_recall_usage
from ..paths import OmhPaths
from ..runtime.records import (
    build_event_record,
    build_wrapper_session_record,
    validate_event_record,
    validate_runtime_observation_record,
    validate_wrapper_session_record,
)
from ..runtime.artifacts import (
    read_runtime_observations_result,
    runtime_observations_for_target,
    runtime_observation_not_applicable,
    summarize_runtime_observation_status,
    validate_runtime_observations_for_wrapper_session,
)
from .contract import (
    CHAT_RESPONSE_SCHEMA_VERSION,
    INTERACTION_MODES,
    _prompt_handoff_show_body,
    build_chat_interaction_payload,
    build_chat_status_interaction,
    headline_with_usage_prefix,
    messenger_rendering_contract,
    render_profile_for_source,
    usage_trace_payload,
)
from .domain_context_attachment import (
    HostProjectBindingFactory,
    build_session_project_binding_factory,
)
from .briefing import build_coding_briefing, chat_response_briefing
from .continuity import build_continuity_briefing
from .hermes_runtime import (
    hermes_coding_team_body,
    hermes_coding_team_claim_boundary,
    hermes_coding_team_extra_action_specs,
)
from .executor_sessions import (
    build_executor_session_status,
    build_executor_session_status_card,
    enhance_chat_response_with_executor_session,
    enhance_status_card_with_executor_session,
    read_executor_session_result,
)
from ..coding_delegation import build_coding_delegation_payload
from ..coding.owner_fit import owner_capability_snapshots
from ..coding.owner_retarget import build_owner_retarget


WRAPPER_SESSION_RESULT_SCHEMA_VERSION = "wrapper_session_result/v1"
# The session event a turn that built a route question appends; the route
# question shadow report reads it as the live-lane route record.
ROUTE_QUESTION_OBSERVED_EVENT = "route_question_observed"
# The statuses that mean "this session already has a prepared handoff", and so
# the only ones an explicit retarget can move: there is nothing to re-project
# before one exists.
PREPARED_HANDOFF_STATUSES = ("handoff_prepared", "prompt_handoff_prepared", "runtime_handoff_prepared")
PLAN_DECISION_TRANSITIONS = {
    "plan_presented": {"accept", "revise", "cancel"},
    "plan_accepted": {"accept", "revise", "cancel"},
    "executor_choice_required": {"cancel"},
    "executor_selected": {"cancel"},
    "prompt_handoff_prepared": set(),
    "runtime_handoff_prepared": set(),
    "revision_requested": {"revise", "cancel"},
    "clarifying": {"cancel"},
    "routed": {"cancel"},
    "cancelled": set(),
    "handoff_prepared": set(),
}


class WrapperSessionError(ValueError):
    pass


def _mutation_digest(*parts: object) -> str:
    """Digest of one operation's own arguments, for replay-conflict detection.

    The core helper compares this against the digest stored for the same
    (operation, mutation_id); a retry that carries different content is then
    refused instead of silently replaying somebody else's result.
    """
    payload = json.dumps(list(parts), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def session_id_for_thread_key(thread_key: str) -> str:
    if not thread_key:
        raise WrapperSessionError("wrapper session requires a thread_key")
    return f"ws-{hashlib.sha256(thread_key.encode('utf-8')).hexdigest()[:24]}"


#: Typed namespace prefix for platform-envelope session ids. A crafted legacy
#: channel_ref/source_event_id can reproduce the exact human thread_key bytes
#: of a platform session; hashing the key inside this canonical JSON envelope
#: keeps the two session families in separate hash namespaces while the
#: stored human thread_key stays identical.
PLATFORM_SESSION_IDENTITY_SCHEMA_VERSION = "omh_platform_session_identity/v1"


def platform_session_id_for_thread_key(thread_key: str) -> str:
    """Session id for a platform-enveloped thread key.

    Same ``ws-<24 hex>`` shape as the legacy derivation, but the hash input
    is the canonical JSON array ``[schema_version, thread_key]`` rather than
    the bare key, so no legacy metadata combination can mint the same id.
    """
    if not thread_key:
        raise WrapperSessionError("wrapper session requires a thread_key")
    basis = json.dumps(
        [PLATFORM_SESSION_IDENTITY_SCHEMA_VERSION, thread_key],
        separators=(",", ":"),
    )
    return f"ws-{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:24]}"


def create_or_resume_wrapper_session(
    paths: OmhPaths,
    event_or_message: dict[str, Any] | str,
    *,
    source: str = "generic",
    mode: str = "auto",
    limit: int = 3,
    min_confidence: str = "high",
    source_metadata: dict[str, str] | None = None,
    executor_target: str = "choose",
    main_agent_model: str = "",
    target_notice: dict[str, object] | None = None,
    record_provenance: dict[str, object] | None = None,
    platform_context: Mapping[str, Any] | None = None,
    design_direction_iteration_context: Mapping[str, object] | None = None,
    hermes_session_id: str = "",
    _host_project_binding_factory: HostProjectBindingFactory | None = None,
) -> dict[str, object]:
    if source not in CHAT_SOURCES:
        raise WrapperSessionError(f"unsupported wrapper session source: {source}")
    if mode not in INTERACTION_MODES:
        raise WrapperSessionError(f"unsupported wrapper session mode: {mode}")
    if min_confidence not in CONFIDENCE_LEVELS:
        raise WrapperSessionError(f"unsupported wrapper session confidence threshold: {min_confidence}")
    # The platform envelope is built and validated inside the payload builder,
    # before this function touches the session store: an unsafe context fails
    # closed ahead of any session directory creation.
    route_question_observations: list[dict[str, object]] = []
    interaction = build_chat_interaction_payload(
        event_or_message,
        source=source,
        mode=mode,
        limit=limit,
        min_confidence=min_confidence,
        include_message=False,
        source_metadata=source_metadata,
        executor_target=executor_target,
        main_agent_model=main_agent_model,
        target_notice=target_notice,
        paths=paths,
        platform_context=platform_context,
        design_direction_iteration_context=design_direction_iteration_context,
        hermes_session_id=hermes_session_id,
        _host_project_binding_factory=build_session_project_binding_factory(
            _host_project_binding_factory,
        ),
        _route_question_sink=route_question_observations,
    )
    thread_key = str(interaction["thread_key"])
    # The presence of a validated platform envelope on the interaction -- not
    # the shape of the key -- selects the platform hash namespace, for both
    # the resume lookup below and the new record built on a miss.
    session_id = (
        platform_session_id_for_thread_key(thread_key)
        if isinstance(interaction.get("platform"), dict)
        else session_id_for_thread_key(thread_key)
    )
    existing = read_wrapper_session(paths, session_id)
    session_dir = _session_dir(paths, session_id)
    if existing:
        append_wrapper_session_event(
            session_dir,
            {
                "event": "session_resumed",
                "message": "wrapper session resumed",
                "data": {"status": existing["status"], "thread_key": thread_key},
            },
        )
        session = existing
        resumed = True
    else:
        session = _session_from_interaction(session_id, interaction, record_provenance=record_provenance)
        # Keep the persisted record, not the pre-write draft: the wrapper echoes
        # session["record_revision"] back on the next mutation, so returning the
        # unbumped draft would make every first mutation look stale.
        session = write_wrapper_session(paths, session)
        append_wrapper_session_event(
            session_dir,
            {
                "event": "session_started",
                "message": "wrapper session started",
                "data": {"status": session["status"], "thread_key": thread_key},
            },
        )
        resumed = False
    # One line in the session's own event log per turn that built a route
    # question -- the log this function already appends to on every call --
    # so the shadow report can join a live turn to the answer recorded on it.
    # Metadata only: the request hash, the deterministic reading, and the
    # question summary (`route_question_observation`).
    for observation in route_question_observations:
        append_wrapper_session_event(
            session_dir,
            {
                "event": ROUTE_QUESTION_OBSERVED_EVENT,
                "message": "route question built",
                "data": observation,
            },
        )
    return {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "resumed": resumed,
        "session": session,
        "interaction": interaction,
        "status": build_wrapper_session_status(paths, session_id),
    }


def execute_design_direction_iteration_session_action(
    paths: OmhPaths,
    session_id: str,
    action: str,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Apply an explicit direction action and advance its session binding together."""
    from .design_direction_iteration_actions import execute_design_direction_iteration_action

    action_result: dict[str, object] | None = None
    action_payload = dict(payload)

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        nonlocal action_result
        route = dict(current["route"])
        context = route.get("design_direction_iteration")
        if action != "prepare_design_direction_iteration":
            if not isinstance(context, dict) or action_payload.get("iteration_id") != context.get("iteration_id"):
                raise WrapperSessionError("design direction action is not bound to this wrapper session")
            if action != "show_design_direction_iteration" and action_payload.get("revision_digest") != context.get("revision_digest"):
                raise WrapperSessionError("stale design direction session binding")
        if action_payload.get("thread_key", current["thread_key"]) != current["thread_key"]:
            raise WrapperSessionError("design direction action names another thread")
        action_payload["thread_key"] = current["thread_key"]
        action_result = execute_design_direction_iteration_action(paths, action, action_payload)
        iteration = action_result.get("iteration")
        if isinstance(iteration, dict):
            snapshots = iteration.get("snapshots")
            if not isinstance(snapshots, list) or not snapshots or not isinstance(snapshots[-1], dict):
                raise WrapperSessionError("design direction result has no current revision")
            route["design_direction_iteration"] = {
                "iteration_id": iteration["iteration_id"],
                "revision_digest": snapshots[-1]["revision_digest"],
            }
        return {**current, "route": route}

    digest = _mutation_digest(action, payload)
    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="design_direction_iteration_action",
        mutation_id=digest,
        mutation_digest=digest,
    )
    if replayed:
        if action == "request_design_direction_memory_review":
            return execute_design_direction_iteration_action(
                paths, action, {**action_payload, "thread_key": session["thread_key"]}
            )
        context = session["route"]["design_direction_iteration"]
        return execute_design_direction_iteration_action(
            paths, "show_design_direction_iteration", {"iteration_id": context["iteration_id"]}
        )
    if action_result is None:
        raise WrapperSessionError("design direction action produced no result")
    return action_result


def open_wrapper_session_decision_gate(
    paths: OmhPaths,
    *,
    session_id: str,
    approver: str,
    safety_profile_revision: str,
    now: str = "",
) -> dict[str, Any]:
    """Open the only remote-decision gate that can advance this plan revision."""
    from ..workflows.decision_gate_receipts import wrapper_gate_binding
    from ..workflows.decision_gates import open_decision_gate

    session = _existing_session(paths, session_id)
    if str(session.get("status", "")) != "plan_presented":
        raise WrapperSessionError("a connector decision gate requires a presented wrapper plan")
    binding = wrapper_gate_binding(session_id, record_revision_of(session))
    return open_decision_gate(
        paths,
        **binding,
        question_code="accept_plan_direction",
        risk_class="reversible",
        choices=("approve", "decline", "defer"),
        approver=approver,
        safety_profile_revision=safety_profile_revision,
        now=now,
    )


def record_plan_decision(
    paths: OmhPaths,
    session_id: str,
    decision: str,
    *,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
) -> dict[str, object]:
    if decision not in {"accept", "revise", "cancel"}:
        raise WrapperSessionError("plan decision must be accept, revise, or cancel")

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        current_status = str(current.get("status", ""))
        allowed = PLAN_DECISION_TRANSITIONS.get(current_status, set())
        if decision not in allowed:
            raise WrapperSessionError(f"cannot {decision.replace('_', ' ')} a wrapper session while status is {current_status}")
        accept_status = _accepted_status_for_session(current)
        updates = {
            "accept": {"status": accept_status, "decision": "plan_accepted"},
            "revise": {"status": "revision_requested", "decision": "plan_revision_requested"},
            "cancel": {"status": "cancelled", "decision": "plan_cancelled"},
        }[decision]
        return build_wrapper_session_record({**current, **updates, "updated_at": utc_now()})

    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="record_plan_decision",
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        mutation_digest=_mutation_digest("record_plan_decision", decision),
    )
    if not replayed:
        append_wrapper_session_event(
            _session_dir(paths, session_id),
            {
                "event": f"plan_{session['decision']}",
                "message": f"wrapper recorded {session['decision']}",
                "data": {"status": session["status"], "decision": session["decision"]},
            },
        )
    return {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session": session,
        "status": build_wrapper_session_status(paths, session_id),
        "replayed": replayed,
    }


def consume_connector_decision_receipt(
    paths: OmhPaths,
    *,
    session_id: str,
    receipt_id: str,
    now: str = "",
) -> dict[str, object]:
    """Advance only the session and plan revision bound when its gate opened."""
    from ..workflows.decision_gate_receipts import wrapper_gate_binding
    from ..workflows.decision_gates import gate_is_expired, latest_gate_in, read_decision_gates, validate_decision_gate

    receipt = next(
        (
            record
            for record in reversed(read_jsonl_objects(paths.runtime_decision_gates_path)[0])
            if record.get("schema_version") == "connector_decision_gate_receipt/v1"
            and record.get("receipt_id") == receipt_id
            and not validate_decision_gate(record)
        ),
        {},
    )
    if not receipt or str(receipt.get("wrapper_session_ref", "")) != session_id:
        raise WrapperSessionError("connector decision receipt does not authorize this wrapper session")
    binding = wrapper_gate_binding(session_id, int(receipt["wrapper_expected_revision"]))
    answer = latest_gate_in(read_decision_gates(paths), str(receipt.get("gate_id", "")))
    if (
        not answer
        or validate_decision_gate(answer)
        or str(answer.get("record_id", "")) != str(receipt.get("gate_answer_record_ref", ""))
        or str(answer.get("resume_digest", "")) != str(receipt.get("expected_resume_digest", ""))
        or str(answer.get("actor", "")) != str(receipt.get("actor", ""))
        or str(answer.get("answered_choice", "")) != str(receipt.get("choice", ""))
        or any(str(answer.get(key, "")) != expected for key, expected in binding.items())
    ):
        raise WrapperSessionError("connector decision receipt is not bound to the current wrapper gate answer")
    decision_by_choice = {"approve": "accept", "decline": "cancel", "defer": "revise"}
    decision = decision_by_choice.get(str(receipt.get("choice", "")), "")
    if not decision:
        raise WrapperSessionError("connector decision receipt choice cannot advance a wrapper session")

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        if gate_is_expired(answer, now):
            raise WrapperSessionError("connector decision receipt expired before wrapper consumption")
        current_status = str(current.get("status", ""))
        allowed = PLAN_DECISION_TRANSITIONS.get(current_status, set())
        if decision not in allowed:
            raise WrapperSessionError(
                f"cannot consume connector decision while wrapper session status is {current_status}"
            )
        accept_status = _accepted_status_for_session(current)
        updates = {
            "accept": {"status": accept_status, "decision": "plan_accepted"},
            "revise": {"status": "revision_requested", "decision": "plan_revision_requested"},
            "cancel": {"status": "cancelled", "decision": "plan_cancelled"},
        }[decision]
        return build_wrapper_session_record({**current, **updates, "updated_at": utc_now()})

    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="consume_connector_decision_receipt",
        expected_revision=int(receipt["wrapper_expected_revision"]),
        mutation_id=f"connector-{receipt_id}",
        mutation_digest=_mutation_digest("consume_connector_decision_receipt", receipt_id),
    )
    if not replayed:
        append_wrapper_session_event(
            _session_dir(paths, session_id),
            {
                "event": "connector_decision_consumed",
                "message": "wrapper session consumed a connector-bound decision receipt",
                "data": {"receipt_id": receipt_id, "decision": session["decision"]},
            },
        )
    return {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session": session,
        "status": build_wrapper_session_status(paths, session_id),
        "replayed": replayed,
    }


def select_wrapper_session_executor(
    paths: OmhPaths,
    session_id: str,
    executor_target: str,
    *,
    dispatch_policy: str | None = None,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
) -> dict[str, object]:
    if executor_target not in CODING_EXECUTOR_TARGETS:
        raise WrapperSessionError(f"unsupported wrapper session executor: {executor_target}")
    selection = executor_selection_for_target(executor_target, action="delegate")
    if selection.choice_required:
        raise WrapperSessionError("select-executor requires a concrete executor profile, not choose")
    policy = dispatch_policy or selection.dispatch_policy

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        require_not_terminal(
            current,
            "status",
            ("cancelled",),
            "select an executor for this wrapper session",
            error_type=WrapperSessionError,
        )
        if current.get("status") not in {"plan_accepted", "executor_choice_required", "executor_selected"}:
            raise WrapperSessionError(f"cannot select executor while status is {current.get('status')}")
        return build_wrapper_session_record(
            {
                **current,
                "status": "executor_selected",
                "work_owner_mode": selection.work_owner_mode,
                "selected_executor_profile": selection.selected_executor_profile,
                "dispatch_policy": policy,
                "updated_at": utc_now(),
            }
        )

    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="select_wrapper_session_executor",
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        mutation_digest=_mutation_digest("select_wrapper_session_executor", executor_target, policy),
    )
    if not replayed:
        append_wrapper_session_event(
            _session_dir(paths, session_id),
            {
                "event": "executor_selected",
                "message": "wrapper session selected executor/runtime profile",
                "data": {
                    "work_owner_mode": selection.work_owner_mode,
                    "selected_executor_profile": selection.selected_executor_profile,
                    "dispatch_policy": policy,
                    "dispatchable": selection.dispatchable,
                },
            },
        )
    return {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session": session,
        "status": build_wrapper_session_status(paths, session_id),
        "replayed": replayed,
    }


def prepare_wrapper_session_handoff(
    paths: OmhPaths,
    session_id: str,
    event_or_message: dict[str, Any] | str,
    *,
    limit: int = 3,
    include_message: bool = False,
    source_metadata: dict[str, str] | None = None,
    executor_target: str | None = None,
    context_pack: dict[str, object] | None = None,
    retarget: bool = False,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    hermes_session_id: str = "",
) -> dict[str, object]:
    session = _existing_session(paths, session_id)
    # expected_revision is threaded into every write this call authorizes
    # instead of being compared once against this unlocked read: the in-lock
    # compare is the guard, and it runs after the replay check, so a client
    # that timed out and retried with the ORIGINAL revision and the SAME
    # mutation_id replays idempotently rather than being refused as stale.
    # `guard_revision` advances when a write here moves the record on.
    guard_revision = expected_revision
    require_not_terminal(
        session,
        "status",
        ("cancelled",),
        "prepare a handoff for this wrapper session",
        error_type=WrapperSessionError,
    )
    # A prepared session re-serves its handoff only for a retry of the SAME
    # message (or an empty show call). A different non-empty message is a
    # follow-up instruction; silently re-serving the old handoff dropped it
    # with exit 0 and no event (issue #754), so it re-prepares instead.
    follow_up_sha = hashlib.sha256(extract_message_text(event_or_message).encode("utf-8")).hexdigest()
    follow_up_message = bool(extract_message_text(event_or_message))
    follow_up = False
    retarget_from = _retarget_source_owner(session, executor_target or "", retarget=retarget)
    if retarget_from:
        # An explicit retarget must not take any of the re-serve short-circuits
        # below. Retargeting normally repeats the ORIGINAL task text, which
        # matches the prepared message, and re-serving would hand back the OLD
        # owner's handoff with exit 0 while silently ignoring the owner the
        # person just asked for.
        #
        # The run-integrity invariants that branch enforces still apply while a
        # run is linked: a retarget detaches this session from its Codex run,
        # so it must not detach from an inconsistent one.
        if session.get("current_run_id"):
            if session["status"] != "handoff_prepared":
                raise WrapperSessionError("wrapper session current_run_id is only valid after handoff preparation")
            if _run_owned_by_other_session(paths, str(session["current_run_id"]), str(session["session_id"])):
                raise WrapperSessionError("wrapper session current_run_id is already linked to another wrapper session")
    elif session.get("current_run_id"):
        if session["status"] != "handoff_prepared":
            raise WrapperSessionError("wrapper session current_run_id is only valid after handoff preparation")
        run_id = str(session["current_run_id"])
        if _run_owned_by_other_session(paths, run_id, str(session["session_id"])):
            raise WrapperSessionError("wrapper session current_run_id is already linked to another wrapper session")
        if not follow_up_message or _prepared_run_message_sha(paths, run_id) == follow_up_sha:
            _ensure_handoff_prepared_event(paths, session, run_id, recovered=True)
            return {
                "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
                "session": session,
                "status": build_wrapper_session_status(paths, session_id),
                "handoff": report_codex_delegation_lifecycle(paths, run_id),
                "replayed": False,
            }
        follow_up = True
    elif session["status"] == "prompt_handoff_prepared":
        if not follow_up_message or _latest_prepared_message_sha(_session_dir(paths, session_id), "prompt_handoff_prepared") == follow_up_sha:
            return {
                "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
                "session": session,
                "status": build_wrapper_session_status(paths, session_id),
                "handoff": {"prompt_handoff": session.get("prompt_handoff", {})},
                "replayed": False,
            }
        follow_up = True
    elif session["status"] == "runtime_handoff_prepared":
        if not follow_up_message or _latest_prepared_message_sha(_session_dir(paths, session_id), "runtime_handoff_prepared") == follow_up_sha:
            return {
                "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
                "session": session,
                "status": build_wrapper_session_status(paths, session_id),
                "handoff": _runtime_session_handoff_envelope(session.get("runtime_handoff", {})),
                "replayed": False,
            }
        follow_up = True
    if session["status"] == "executor_choice_required" and not executor_target:
        raise WrapperSessionError("wrapper session requires executor selection before preparing a handoff")
    if executor_target and follow_up:
        if executor_target != str(session.get("selected_executor_profile") or ""):
            raise WrapperSessionError(
                "a follow-up handoff keeps the selected executor; retarget the accepted plan explicitly to move "
                "it to another executor, or start a new session to replan"
            )
    elif executor_target and not retarget_from:
        selected = select_wrapper_session_executor(
            paths,
            session_id,
            executor_target,
            expected_revision=guard_revision,
            mutation_id=mutation_id,
        )
        session = dict(selected["session"])
        if guard_revision is not None:
            # This call authorized the selection write, so the handoff write
            # that follows must compare against the revision that write
            # produced, not the one the wrapper originally rendered.
            guard_revision = record_revision_of(session)
    if not follow_up and not retarget_from and session["status"] not in {"plan_accepted", "executor_selected"}:
        raise WrapperSessionError("wrapper session plan must be accepted before preparing a handoff")
    if retarget_from:
        # The accepted plan is already bound to this session; only the owner
        # moves. The selection write `select_wrapper_session_executor` performs
        # is skipped because it refuses a session that already has a prepared
        # handoff, and the prepare below rewrites owner, mode, and status in the
        # same guarded transaction that stores the new handoff.
        selected_executor = str(executor_target)
        work_owner_mode = executor_selection_for_target(selected_executor, action="delegate").work_owner_mode
        # Named before the new handoff exists (AC2): building it here means a
        # move that would silently replan the task raises instead of preparing.
        #
        # `follow_up` stays false, so the prepare below does NOT inherit the
        # session's route preference. A follow-up is new text that has to be
        # pulled back onto the accepted lane; a retarget repeats the accepted
        # task itself, and routing it exactly as the prepare it replaces did is
        # what keeps the re-projection faithful to that handoff.
        retarget_record: dict[str, object] | None = _owner_retarget_record(
            paths,
            session,
            extract_message_text(event_or_message),
            limit=limit,
            context_pack=context_pack,
            from_owner=retarget_from,
            to_owner=selected_executor,
        )
    else:
        selected_executor = str(session.get("selected_executor_profile") or "codex")
        work_owner_mode = str(session.get("work_owner_mode") or "")
        retarget_record = None
    if work_owner_mode == "runtime_handoff" and selected_executor:
        return _with_owner_retarget(
            _prepare_runtime_session_handoff(
                paths,
                session,
                event_or_message,
                limit=limit,
                include_message=include_message,
                source_metadata=source_metadata,
                executor_target=selected_executor,
                context_pack=context_pack,
                follow_up=follow_up,
                expected_revision=guard_revision,
                mutation_id=mutation_id,
                hermes_session_id=hermes_session_id,
            ),
            retarget_record,
            paths,
            session_id,
        )
    if selected_executor and selected_executor != "codex":
        return _with_owner_retarget(
            _prepare_prompt_only_session_handoff(
                paths,
                session,
                event_or_message,
                limit=limit,
                include_message=include_message,
                source_metadata=source_metadata,
                executor_target=selected_executor,
                context_pack=context_pack,
                follow_up=follow_up,
                expected_revision=guard_revision,
                mutation_id=mutation_id,
                hermes_session_id=hermes_session_id,
            ),
            retarget_record,
            paths,
            session_id,
        )
    message = extract_message_text(event_or_message)
    metadata = _source_metadata(event_or_message)
    metadata.update({str(key): str(value) for key, value in (source_metadata or {}).items() if str(value)})
    metadata.update({str(key): str(value) for key, value in session.get("source_metadata", {}).items() if str(value)})
    metadata = _compact_coding_source_metadata(metadata)
    message_sha256 = hashlib.sha256(message.encode("utf-8")).hexdigest()
    digest = _mutation_digest("link_prepared_handoff_run", message_sha256, "codex", metadata)
    recovered_run_id = _find_recoverable_prepared_handoff_run(paths, session, message_sha256, metadata)
    if recovered_run_id:
        return _with_owner_retarget(
            _link_prepared_handoff_run(
                paths,
                session,
                recovered_run_id,
                recovered=True,
                expected_revision=guard_revision,
                mutation_id=mutation_id,
                mutation_digest=digest,
            ),
            retarget_record,
            paths,
            session_id,
        )
    follow_up_workflow, follow_up_score = _session_route_preference(session) if follow_up else (None, None)
    lifecycle: dict[str, object] = {}

    def prepare_run() -> str:
        # Runs inside the session lock, after the terminal, replay, and
        # revision guards have passed: a rejected prepare must leave no
        # prepare-started event, no runtime run directory, and no run ledger
        # entry behind (issue #828 AC1).
        append_wrapper_session_event(
            _session_dir(paths, session_id),
            {
                "event": "handoff_prepare_started",
                "message": "wrapper session started preparing coding handoff",
                "data": {"message_sha256": message_sha256, "message_length": len(message)},
            },
        )
        lifecycle.update(
            start_codex_delegation_lifecycle(
                paths,
                message,
                source=str(session["source"]),
                source_metadata=metadata,
                limit=limit,
                include_message=include_message,
                context_pack=context_pack,
                preferred_workflow=follow_up_workflow,
                preferred_workflow_score=follow_up_score,
                force_coding_handoff=follow_up,
                hermes_session_id=hermes_session_id,
            )
        )
        run = lifecycle["run"]
        return str(run["run_id"]) if isinstance(run, dict) else ""

    linked = _link_prepared_handoff_run(
        paths,
        session,
        recovered=False,
        expected_revision=guard_revision,
        mutation_id=mutation_id,
        mutation_digest=digest,
        prepare_run=prepare_run,
    )
    if lifecycle:
        run = lifecycle["run"]
        lifecycle["status"] = report_codex_delegation_lifecycle(paths, str(run["run_id"]) if isinstance(run, dict) else "")
        linked["handoff"] = lifecycle
    return _with_owner_retarget(linked, retarget_record, paths, session_id)


def _retarget_source_owner(session: dict[str, Any], executor_target: str, *, retarget: bool) -> str:
    """The owner an explicit retarget moves away from, or "" when this is not one.

    The guard this relaxes refused EVERY executor change on a follow-up handoff,
    and that intent survives untouched: a follow-up is a delta on an accepted
    plan, silently swapping its owner would hand the next instruction to
    somebody the user never chose, and this returns "" for a follow-up so the
    refusal still fires. What no longer holds is the only escape the old message
    offered --- "start a new session to switch executors". A new session has no
    accepted plan, so that escape charged the user a full replan for a change
    that alters nothing about the work (#812).

    Retargeting is therefore a distinct, named operation rather than a silent
    allow. The caller has to ask for it by name, it only applies to a session
    that already has a prepared handoff, it refuses a move to the owner already
    selected, and it records what it did in `coding_owner_retarget/v1`.
    """
    if not retarget:
        return ""
    if not executor_target:
        raise WrapperSessionError("retargeting requires the coding owner to move the accepted plan to")
    # `select_wrapper_session_executor` is skipped on this path, so the target
    # validation it normally performs has to happen here instead.
    if executor_target == "choose" or executor_target not in CODING_EXECUTOR_TARGETS:
        raise WrapperSessionError(f"unsupported wrapper session executor: {executor_target}")
    status = str(session.get("status", ""))
    if status not in PREPARED_HANDOFF_STATUSES:
        raise WrapperSessionError(
            f"cannot retarget a wrapper session while status is {status}; there is no prepared handoff to re-project"
        )
    current = str(session.get("selected_executor_profile") or "")
    if not current:
        raise WrapperSessionError("cannot retarget a wrapper session that has no selected executor")
    if executor_target == current:
        raise WrapperSessionError(
            f"the accepted plan is already targeted at {executor_target}; retargeting requires a different executor"
        )
    return current


def _owner_retarget_record(
    paths: OmhPaths,
    session: dict[str, Any],
    message: str,
    *,
    limit: int,
    context_pack: dict[str, object] | None,
    from_owner: str,
    to_owner: str,
) -> dict[str, object]:
    """The `coding_owner_retarget/v1` record for one accepted plan changing owner.

    Both sides are re-projections of the same accepted plan, built here from
    identical owner-neutral inputs and differing only in `executor_target`.
    `build_owner_retarget` then compares the two owner-neutral task contracts
    and refuses when any field moved, so a "retarget" that would quietly replan
    the task raises instead of being recorded as one.

    The routing arguments deliberately match the prepare this record describes,
    down to NOT inheriting the session's route preference: a projection routed
    differently from the handoff it accompanies would report a contract that
    handoff does not carry.

    Neither projection reads a memory recall pack or carries source metadata:
    both are owner-specific handoff context that never enters the task
    contract, so reading them here would cost I/O to produce values the
    comparison ignores.
    """
    snapshot_directory = _capability_snapshot_directory(paths)

    def projection(owner: str) -> dict[str, object]:
        return build_coding_delegation_payload(
            message,
            source=str(session["source"]),
            limit=limit,
            executor_target=owner,
            context_pack=context_pack,
            capability_snapshot_directory=snapshot_directory,
        )

    return build_owner_retarget(
        from_payload=projection(from_owner),
        to_payload=projection(to_owner),
        task_source_sha256=hashlib.sha256(message.encode("utf-8")).hexdigest(),
        capability_snapshots=dict(owner_capability_snapshots(snapshot_directory, (from_owner, to_owner))),
    )


def _with_owner_retarget(
    result: dict[str, object],
    retarget_record: dict[str, object] | None,
    paths: OmhPaths,
    session_id: str,
) -> dict[str, object]:
    """Attach and journal the retarget once the new handoff actually exists.

    The record is built before the prepare so a refused move never writes; the
    event is appended after it so the journal never claims a retarget that did
    not finish (issue #828 AC1, applied to this operation).
    """
    if retarget_record is None:
        return result
    append_wrapper_session_event(
        _session_dir(paths, session_id),
        {
            "event": "coding_owner_retargeted",
            "message": "wrapper session retargeted accepted coding work to another owner",
            "data": retarget_record,
        },
    )
    result["coding_owner_retarget"] = retarget_record
    return result


def _capability_snapshot_directory(paths: OmhPaths) -> Path:
    return paths.omh_home / "coding" / "executor-capability-snapshots"


def _session_route_preference(session: dict[str, Any]) -> tuple[str | None, int | None]:
    """The already-routed workflow a follow-up message should inherit.

    A follow-up is a delta on an accepted coding plan ("add tests too"), so
    re-routing its text from scratch either guesses a different workflow or
    fails to produce a handoff at all; the session's stored route is the
    decision the user already accepted.
    """
    route = session.get("route", {}) if isinstance(session.get("route"), dict) else {}
    workflow = str(route.get("selected_skill", "") or "")
    if not workflow:
        return None, None
    try:
        score = int(route.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0
    return workflow, score if score > 0 else None


def _prepare_prompt_only_session_handoff(
    paths: OmhPaths,
    session: dict[str, Any],
    event_or_message: dict[str, Any] | str,
    *,
    limit: int,
    include_message: bool,
    source_metadata: dict[str, str] | None,
    executor_target: str,
    context_pack: dict[str, object] | None,
    follow_up: bool = False,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    hermes_session_id: str = "",
) -> dict[str, object]:
    message = extract_message_text(event_or_message)
    metadata = _source_metadata(event_or_message)
    metadata.update({str(key): str(value) for key, value in (source_metadata or {}).items() if str(value)})
    metadata.update({str(key): str(value) for key, value in session.get("source_metadata", {}).items() if str(value)})
    metadata = _compact_coding_source_metadata(metadata)
    payload = build_coding_delegation_payload(
        message,
        source=str(session["source"]),
        limit=limit,
        include_message=include_message,
        source_metadata=metadata,
        executor_target=executor_target,
        context_pack=context_pack,
        preferred_workflow=(_session_route_preference(session)[0] if follow_up else None),
        preferred_workflow_score=(_session_route_preference(session)[1] if follow_up else None),
        force_coding_handoff=follow_up,
        memory_recall_pack=memory_recall_pack_for_handoff(
            paths,
            message,
            executor_target=executor_target,
            session_id=str(session["session_id"]),
            surface_class=memory_surface_for_source(str(session["source"])),
            hermes_session_id=hermes_session_id,
        ),
        capability_snapshot_directory=_capability_snapshot_directory(paths),
    )
    prompt_handoff = payload.get("prompt_handoff")
    if not isinstance(prompt_handoff, dict):
        raise WrapperSessionError("selected executor produced no prompt handoff")
    session_id = str(session["session_id"])

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        require_not_terminal(
            current,
            "status",
            ("cancelled",),
            "prepare a handoff for this wrapper session",
            error_type=WrapperSessionError,
        )
        # Recall-usage counters are a persisted side effect, so they are
        # recorded only after the guards that can reject have passed.
        record_attached_recall_usage(paths, payload)
        return build_wrapper_session_record(
            {
                **current,
                "status": "prompt_handoff_prepared",
                "work_owner_mode": "prompt_only_handoff",
                "selected_executor_profile": executor_target,
                "dispatch_policy": "prepare_only",
                "prompt_handoff": prompt_handoff,
                # Cleared for the same reason the runtime prepare clears
                # `prompt_handoff`: retargeting (#812) makes a runtime-prepared
                # session reachable from here, and a stale runtime handoff left
                # beside a prompt handoff is a second answer to "what is
                # prepared".
                "runtime_handoff": {},
                "current_run_id": "",
                "updated_at": utc_now(),
            }
        )

    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="prepare_prompt_only_session_handoff",
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        mutation_digest=_mutation_digest(
            "prepare_prompt_only_session_handoff",
            hashlib.sha256(message.encode("utf-8")).hexdigest(),
            executor_target,
            metadata,
        ),
    )
    if not replayed:
        append_wrapper_session_event(
            _session_dir(paths, session_id),
            {
                "event": "prompt_handoff_prepared",
                "message": "wrapper session prepared prompt-only coding handoff",
                "data": {
                    "selected_executor_profile": executor_target,
                    "dispatchable": False,
                    "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
                    "message_length": len(message),
                },
            },
        )
    result: dict[str, object] = {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session": session,
        "status": build_wrapper_session_status(paths, session_id),
        "handoff": {
            "schema_version": "prompt_only_session_handoff/v1",
            "prompt_handoff": prompt_handoff,
            "runtime": {
                "run_created": False,
                "reason": "prompt_only_handoff_is_not_lifecycle_backed",
            },
        },
        "replayed": replayed,
    }
    if include_message and "prompt_handoff_prompt" in payload:
        result["prompt_handoff_prompt"] = payload["prompt_handoff_prompt"]
    return result


def _prepare_runtime_session_handoff(
    paths: OmhPaths,
    session: dict[str, Any],
    event_or_message: dict[str, Any] | str,
    *,
    limit: int,
    include_message: bool,
    source_metadata: dict[str, str] | None,
    executor_target: str,
    context_pack: dict[str, object] | None,
    follow_up: bool = False,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    hermes_session_id: str = "",
) -> dict[str, object]:
    message = extract_message_text(event_or_message)
    metadata = _source_metadata(event_or_message)
    metadata.update({str(key): str(value) for key, value in (source_metadata or {}).items() if str(value)})
    metadata.update({str(key): str(value) for key, value in session.get("source_metadata", {}).items() if str(value)})
    metadata = _compact_coding_source_metadata(metadata)
    payload = build_coding_delegation_payload(
        message,
        source=str(session["source"]),
        limit=limit,
        include_message=include_message,
        source_metadata=metadata,
        executor_target=executor_target,
        context_pack=context_pack,
        preferred_workflow=(_session_route_preference(session)[0] if follow_up else None),
        preferred_workflow_score=(_session_route_preference(session)[1] if follow_up else None),
        force_coding_handoff=follow_up,
        memory_recall_pack=memory_recall_pack_for_handoff(
            paths,
            message,
            executor_target=executor_target,
            session_id=str(session["session_id"]),
            surface_class=memory_surface_for_source(str(session["source"])),
            hermes_session_id=hermes_session_id,
        ),
        capability_snapshot_directory=_capability_snapshot_directory(paths),
    )
    runtime_handoff = payload.get("runtime_handoff")
    if not isinstance(runtime_handoff, dict):
        raise WrapperSessionError("selected runtime produced no runtime handoff")
    session_id = str(session["session_id"])

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        require_not_terminal(
            current,
            "status",
            ("cancelled",),
            "prepare a handoff for this wrapper session",
            error_type=WrapperSessionError,
        )
        # Recall-usage counters are a persisted side effect, so they are
        # recorded only after the guards that can reject have passed.
        record_attached_recall_usage(paths, payload)
        return build_wrapper_session_record(
            {
                **current,
                "status": "runtime_handoff_prepared",
                "work_owner_mode": "runtime_handoff",
                "selected_executor_profile": executor_target,
                "dispatch_policy": "prepare_only",
                "prompt_handoff": {},
                "runtime_handoff": runtime_handoff,
                "current_run_id": "",
                "updated_at": utc_now(),
            }
        )

    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="prepare_runtime_session_handoff",
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        mutation_digest=_mutation_digest(
            "prepare_runtime_session_handoff",
            hashlib.sha256(message.encode("utf-8")).hexdigest(),
            executor_target,
            metadata,
        ),
    )
    if not replayed:
        append_wrapper_session_event(
            _session_dir(paths, session_id),
            {
                "event": "runtime_handoff_prepared",
                "message": "wrapper session prepared runtime coding handoff",
                "data": {
                    "selected_executor_profile": executor_target,
                    "dispatchable": False,
                    "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
                    "message_length": len(message),
                    "team_swarm_guidance": True,
                    "worktree_guidance": True,
                },
            },
        )
    result: dict[str, object] = {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session": session,
        "status": build_wrapper_session_status(paths, session_id),
        "handoff": _runtime_session_handoff_envelope(runtime_handoff),
        "replayed": replayed,
    }
    if include_message and "runtime_handoff_prompt" in payload:
        result["runtime_handoff_prompt"] = payload["runtime_handoff_prompt"]
    return result


def _runtime_session_handoff_envelope(runtime_handoff: object) -> dict[str, object]:
    return {
        "schema_version": "runtime_session_handoff/v1",
        "runtime_handoff": runtime_handoff if isinstance(runtime_handoff, dict) else {},
        "runtime": {
            "run_created": False,
            "reason": "runtime_handoff_is_not_lifecycle_backed",
        },
    }


def build_wrapper_session_status(paths: OmhPaths, session_id: str) -> dict[str, object]:
    session = _existing_session(paths, session_id)
    run_id = str(session.get("current_run_id", ""))
    if run_id:
        runtime_status = report_codex_delegation_lifecycle(paths, run_id)
        interaction = build_chat_status_interaction(
            runtime_status,
            source=str(session["source"]),
            source_metadata={str(key): str(value) for key, value in session.get("source_metadata", {}).items()},
        )
        executor_status = build_executor_session_status(paths, session, linked_status=runtime_status)
        chat_response = enhance_chat_response_with_executor_session(
            interaction["chat_response"],
            executor_status,
        )
        status_card = enhance_status_card_with_executor_session(
            interaction["status_card"],
            executor_status,
        )
        coding_briefing = build_coding_briefing(
            session,
            runtime_status=runtime_status,
            executor_status=executor_status,
        )
        chat_response["coding_briefing"] = chat_response_briefing(coding_briefing)
        chat_response["continuity_briefing"] = build_continuity_briefing(
            {
                "work_summary": coding_briefing.get("work_summary", {}),
                "workspace_isolation": executor_status.get("workspace_isolation", {}),
                "isolation_plan": (
                    executor_status.get("workspace_isolation", {}).get("plan", {})
                    if isinstance(executor_status.get("workspace_isolation"), dict)
                    else {}
                ),
                "runtime_status": runtime_status,
                "runtime_observation": {},
                "executor_status": executor_status,
                "session_status": session["status"],
                "evidence_errors": [
                    str(executor_status[key])
                    for key in ("executor_session_error", "linked_lifecycle_error")
                    if executor_status.get(key)
                ],
            }
        )
        return {
            "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
            "session_id": session_id,
            "thread_key": session["thread_key"],
            "current_run_id": run_id,
            "session_status": session["status"],
            "record_revision": record_revision_of(session),
            "runtime_status": runtime_status,
            "executor_session_status": executor_status,
            "coding_briefing": coding_briefing,
            "status_card": status_card,
            "chat_response": chat_response,
            "claim_boundary": "Execution claims come from the linked runtime run ledger, not the wrapper session.",
        }
    if session["status"] == "runtime_handoff_prepared":
        observations, observation_errors = read_runtime_observations_result(_session_dir(paths, session_id))
        runtime_observation = summarize_runtime_observation_status(
            runtime_observations_for_target(observations, "wrapper_session", session_id)
        )
    else:
        observation_errors = []
        runtime_observation = runtime_observation_not_applicable("session is not a prepared runtime handoff")
    executor_status = build_executor_session_status(paths, session, runtime_status=runtime_observation)
    chat_response = enhance_chat_response_with_executor_session(
        _session_chat_response(session),
        executor_status,
    )
    status_card = build_executor_session_status_card(executor_status)
    hermes_harness = executor_status.get("hermes_coding_harness", {})
    hermes_harness = hermes_harness if isinstance(hermes_harness, dict) else {}
    if hermes_harness:
        chat_response["hermes_coding_harness"] = hermes_harness
        status_card["hermes_coding_harness"] = hermes_harness
    coding_briefing = build_coding_briefing(
        session,
        executor_status=executor_status,
        runtime_observation=runtime_observation,
    )
    chat_response["coding_briefing"] = chat_response_briefing(coding_briefing)
    chat_response["continuity_briefing"] = build_continuity_briefing(
        {
            "work_summary": coding_briefing.get("work_summary", {}),
            "workspace_isolation": executor_status.get("workspace_isolation", {}),
            "isolation_plan": (
                executor_status.get("workspace_isolation", {}).get("plan", {})
                if isinstance(executor_status.get("workspace_isolation"), dict)
                else {}
            ),
            "runtime_status": {},
            "runtime_observation": runtime_observation,
            "executor_status": executor_status,
            "session_status": session["status"],
            "evidence_errors": [
                *observation_errors,
                *(
                    str(executor_status[key])
                    for key in ("executor_session_error", "linked_lifecycle_error")
                    if executor_status.get(key)
                ),
            ],
        }
    )
    prompt_handoff, runtime_handoff = project_valid_session_handoffs(session)
    return {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session_id": session_id,
        "thread_key": session["thread_key"],
        "current_run_id": "",
        "session_status": session["status"],
        "record_revision": record_revision_of(session),
        "work_owner_mode": session.get("work_owner_mode", "external_executor"),
        "selected_executor_profile": session.get("selected_executor_profile"),
        "dispatch_policy": session.get("dispatch_policy", "ask_before_dispatch"),
        "prompt_handoff": prompt_handoff if session["status"] == "prompt_handoff_prepared" else {},
        "runtime_handoff": runtime_handoff if session["status"] == "runtime_handoff_prepared" else {},
        "runtime_observation": runtime_observation,
        "runtime_observation_errors": observation_errors,
        "hermes_coding_harness": hermes_harness,
        "executor_session_status": executor_status,
        "coding_briefing": coding_briefing,
        "status_card": status_card,
        "next_action": _next_action_for_session(session),
        "chat_response": chat_response,
        "claim_boundary": "Wrapper session state is not execution evidence.",
    }


def list_wrapper_sessions(paths: OmhPaths) -> list[dict[str, Any]]:
    if not paths.runtime_wrapper_sessions_dir.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for session_json in sorted(paths.runtime_wrapper_sessions_dir.glob("*/session.json")):
        session = read_runtime_record(session_json)
        if session:
            sessions.append(session)
    return sessions


def show_wrapper_session(paths: OmhPaths, session_id: str) -> dict[str, Any]:
    session_dir = _session_dir(paths, session_id)
    session = read_runtime_record(session_dir / "session.json")
    if not session:
        raise FileNotFoundError(session_id)
    events, event_errors = read_wrapper_session_events_result(session_dir)
    observations, observation_errors = read_runtime_observations_result(session_dir)
    executor_session, executor_session_error = read_executor_session_result(session_dir)
    result = {
        "session": session,
        "events": events,
        "runtime_observations": observations,
        "executor_session": executor_session,
    }
    if event_errors:
        result["event_errors"] = event_errors
    if observation_errors:
        result["runtime_observation_errors"] = observation_errors
    if executor_session_error:
        result["executor_session_error"] = executor_session_error
    return result


def read_wrapper_session(paths: OmhPaths, session_id: str) -> dict[str, Any] | None:
    return read_runtime_record(_session_dir(paths, session_id) / "session.json")


def write_wrapper_session(paths: OmhPaths, session: dict[str, Any]) -> dict[str, Any]:
    record = build_wrapper_session_record(session)
    session_path = _session_dir(paths, str(record["session_id"])) / "session.json"
    result = guarded_record_update(
        session_path,
        mutate=lambda current: build_wrapper_session_record({**record, "record_revision": current.get("record_revision", 0), "applied_mutations": current.get("applied_mutations", {})}),
        operation="write_wrapper_session",
        lock_name="session.json",
        validate=_validated_session_record,
        default={},
    )
    return result.record if isinstance(result, DuplicateMutationReplay) else result


def _validated_session_record(record: dict[str, Any]) -> None:
    errors = validate_wrapper_session_record(record)
    if errors:
        raise WrapperSessionError(errors[0])


def _guarded_session_update(
    paths: OmhPaths,
    session_id: str,
    mutate,
    *,
    operation: str,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    mutation_digest: str | None = None,
) -> tuple[dict[str, Any], bool]:
    """Run one locked read-check-write transaction on session.json.

    Returns (session_record, replayed). A replayed mutation_id returns the
    current record without a write, revision bump, or session event. Replay is
    keyed on (operation, mutation_id), so the same client id reused for a
    different logical operation still applies.
    """
    session_path = _session_dir(paths, session_id) / "session.json"
    result = guarded_record_update(
        session_path,
        mutate=mutate,
        operation=operation,
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        mutation_digest=mutation_digest,
        lock_name="session.json",
        validate=_validated_session_record,
    )
    if isinstance(result, DuplicateMutationReplay):
        return result.record, True
    return result, False


def append_wrapper_session_event(session_dir: Path, event: dict[str, Any]) -> dict[str, Any]:
    item = build_event_record(event)
    errors = validate_event_record(item)
    if errors:
        raise WrapperSessionError(errors[0])
    ensure_dir(session_dir, private=True)
    events_path = session_dir / EVENTS_FILE
    ensure_file(events_path, private=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True) + "\n")
    return item


def read_wrapper_session_events(session_dir: Path) -> list[dict[str, Any]]:
    return read_wrapper_session_events_result(session_dir)[0]


def read_wrapper_session_events_result(session_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    return read_jsonl_objects(session_dir / EVENTS_FILE)


def validate_wrapper_sessions(paths: OmhPaths, session_id: str | None = None) -> dict[str, Any]:
    if session_id:
        session_dirs = [_session_dir(paths, session_id)]
    else:
        session_dirs = (
            sorted(path for path in paths.runtime_wrapper_sessions_dir.glob("*") if path.is_dir())
            if paths.runtime_wrapper_sessions_dir.exists()
            else []
        )
    results = [_validate_wrapper_session_dir(session_dir) for session_dir in session_dirs]
    return {"ok": all(result["ok"] for result in results), "sessions": results}


def _link_prepared_handoff_run(
    paths: OmhPaths,
    session: dict[str, Any],
    run_id: str = "",
    *,
    recovered: bool,
    expected_revision: int | None = None,
    mutation_id: str | None = None,
    mutation_digest: str | None = None,
    prepare_run: Callable[[], str] | None = None,
) -> dict[str, object]:
    """Link a prepared coding run to the session inside one locked transaction.

    ``prepare_run`` is invoked only after the guards that can reject have
    passed, so the run it creates is never an orphan left behind by a refused
    prepare; ``run_id`` is passed directly instead when an existing run is
    being recovered.
    """
    session_id = str(session["session_id"])
    prepared: dict[str, str] = {"run_id": run_id}

    def mutate(current: dict[str, Any]) -> dict[str, Any]:
        require_not_terminal(
            current,
            "status",
            ("cancelled",),
            "link a prepared handoff run to this wrapper session",
            error_type=WrapperSessionError,
        )
        if not prepared["run_id"] and prepare_run is not None:
            prepared["run_id"] = prepare_run()
        if not prepared["run_id"]:
            raise WrapperSessionError("wrapper session handoff preparation produced no runtime run")
        return build_wrapper_session_record(
            {
                **current,
                "status": "handoff_prepared",
                "work_owner_mode": "external_executor",
                "selected_executor_profile": "codex",
                "dispatch_policy": "ask_before_dispatch",
                "prompt_handoff": {},
                "runtime_handoff": {},
                "current_run_id": prepared["run_id"],
                "updated_at": utc_now(),
            }
        )

    session, replayed = _guarded_session_update(
        paths,
        session_id,
        mutate,
        operation="link_prepared_handoff_run",
        expected_revision=expected_revision,
        mutation_id=mutation_id,
        mutation_digest=mutation_digest,
    )
    linked_run_id = prepared["run_id"] or str(session.get("current_run_id", ""))
    if not replayed and linked_run_id:
        _ensure_handoff_prepared_event(paths, session, linked_run_id, recovered=recovered)
    return {
        "schema_version": WRAPPER_SESSION_RESULT_SCHEMA_VERSION,
        "session": session,
        "status": build_wrapper_session_status(paths, session_id),
        "handoff": _existing_lifecycle_payload(paths, linked_run_id) if linked_run_id else {},
        "replayed": replayed,
    }


def _find_recoverable_prepared_handoff_run(
    paths: OmhPaths,
    session: dict[str, Any],
    message_sha256: str,
    expected_metadata: dict[str, str],
) -> str:
    session_dir = _session_dir(paths, str(session["session_id"]))
    if not _has_prepare_started_event(session_dir, message_sha256):
        return ""
    for run_json in sorted(paths.runtime_runs_dir.glob("*/run.json"), reverse=True):
        run = read_runtime_record(run_json)
        if not _is_prepared_coding_run(run):
            continue
        run_id = str(run.get("run_id") or run_json.parent.name)
        if _run_owned_by_other_session(paths, run_id, str(session["session_id"])):
            continue
        coding = read_runtime_record(run_json.parent / CODING_DELEGATION_FILE)
        if not _is_matching_coding_handoff(coding, session, message_sha256, expected_metadata):
            continue
        return run_id
    return ""


def _prepared_run_message_sha(paths: OmhPaths, run_id: str) -> str:
    coding = read_runtime_record(paths.runtime_runs_dir / run_id / CODING_DELEGATION_FILE)
    return str(coding.get("message_sha256", "")) if isinstance(coding, dict) else ""


def _latest_prepared_message_sha(session_dir: Path, event_name: str) -> str:
    for event in reversed(read_wrapper_session_events(session_dir)):
        if event.get("event") == event_name and isinstance(event.get("data"), dict):
            return str(event["data"].get("message_sha256", ""))
    return ""


def _has_prepare_started_event(session_dir: Path, message_sha256: str) -> bool:
    return any(
        event.get("event") == "handoff_prepare_started"
        and isinstance(event.get("data"), dict)
        and event["data"].get("message_sha256") == message_sha256
        for event in read_wrapper_session_events(session_dir)
    )


def _is_prepared_coding_run(run: Any) -> bool:
    return (
        isinstance(run, dict)
        and run.get("artifact_kind") == "prepared_coding_delegation"
        and run.get("phase") == "prepared"
        and run.get("observation_status") == "prepared_not_observed"
    )


def _is_matching_coding_handoff(
    coding: Any,
    session: dict[str, Any],
    message_sha256: str,
    expected_metadata: dict[str, str],
) -> bool:
    if not isinstance(coding, dict):
        return False
    handoff = coding.get("executor_handoff")
    if not isinstance(handoff, dict) or handoff.get("executor_target") != "codex":
        return False
    if coding.get("message_sha256") != message_sha256 or coding.get("status") != "prepared_not_observed":
        return False
    if coding.get("source") != session.get("source"):
        return False
    coding_metadata = coding.get("source_metadata", {})
    if not isinstance(coding_metadata, dict):
        return False
    compact_coding_metadata = _compact_coding_source_metadata(coding_metadata)
    return compact_coding_metadata == expected_metadata


def _ensure_handoff_prepared_event(paths: OmhPaths, session: dict[str, Any], run_id: str, *, recovered: bool) -> None:
    session_dir = _session_dir(paths, str(session["session_id"]))
    if _has_handoff_prepared_event(session_dir, run_id):
        return
    append_wrapper_session_event(
        session_dir,
        {
            "event": "handoff_prepared",
            "message": "wrapper session linked prepared coding handoff",
            "data": {"run_id": run_id, "status": "handoff_prepared", "recovered": recovered},
        },
    )


def _has_handoff_prepared_event(session_dir: Path, run_id: str) -> bool:
    return any(
        event.get("event") == "handoff_prepared" and isinstance(event.get("data"), dict) and event["data"].get("run_id") == run_id
        for event in read_wrapper_session_events(session_dir)
    )


def _run_owned_by_other_session(paths: OmhPaths, run_id: str, session_id: str) -> bool:
    if not paths.runtime_wrapper_sessions_dir.exists():
        return False
    for session_json in sorted(paths.runtime_wrapper_sessions_dir.glob("*/session.json")):
        if session_json.parent.name == session_id:
            continue
        other = read_runtime_record(session_json)
        if other and other.get("current_run_id") == run_id:
            return True
    return False


def _compact_coding_source_metadata(metadata: Any) -> dict[str, str]:
    return compact_source_metadata(metadata)


def _existing_lifecycle_payload(paths: OmhPaths, run_id: str) -> dict[str, object]:
    run_dir = paths.runtime_runs_dir / run_id
    return {
        "schema_version": "coding_lifecycle/v1",
        "run": read_runtime_record(run_dir / RUN_FILE),
        "coding_delegation": read_runtime_record(run_dir / CODING_DELEGATION_FILE),
        "status": report_codex_delegation_lifecycle(paths, run_id),
    }


def _validate_wrapper_session_dir(session_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    session_id = session_dir.name
    session_path = session_dir / "session.json"
    try:
        session = read_runtime_record(session_path)
    except (OSError, JSONDecodeError, ValueError) as exc:
        session = None
        errors.append(f"{session_path}: {exc}")
    if not session:
        errors.append(f"{session_path}: missing session.json")
    else:
        errors.extend(f"{session_path}: {error}" for error in validate_wrapper_session_record(session))
        if session.get("session_id") != session_id:
            errors.append(f"{session_path}: session_id must match directory name")
    events_path = session_dir / EVENTS_FILE
    if events_path.exists():
        events, event_errors = read_jsonl_objects(events_path)
        errors.extend(event_errors)
        for index, event in enumerate(events, start=1):
            errors.extend(f"{events_path}:{index}: {error}" for error in validate_event_record(event))
    else:
        errors.append(f"{events_path}: missing events.jsonl")
    observations_path = session_dir / "runtime_observations.jsonl"
    if observations_path.exists():
        observations, observation_errors = read_runtime_observations_result(session_dir)
        errors.extend(observation_errors)
        for index, observation in enumerate(observations, start=1):
            errors.extend(
                f"{observations_path}:{index}: {error}"
                for error in validate_runtime_observation_record(observation)
            )
        if session:
            errors.extend(validate_runtime_observations_for_wrapper_session(observations_path, session, observations))
    executor_session_path = session_dir / EXECUTOR_SESSION_FILE
    if executor_session_path.exists():
        executor_session, executor_session_error = read_executor_session_result(session_dir)
        if executor_session_error:
            errors.append(executor_session_error)
        if executor_session and executor_session.get("session_id") != session_id:
            errors.append(f"{executor_session_path}: session_id must match directory name")
    return {"session_id": session_id, "ok": not errors, "errors": errors}


def _session_from_interaction(
    session_id: str,
    interaction: dict[str, object],
    *,
    record_provenance: dict[str, object] | None = None,
) -> dict[str, Any]:
    route = interaction.get("route")
    route = route if isinstance(route, dict) else {}
    plan_payload = interaction.get("plan")
    plan = plan_payload.get("plan", {}) if isinstance(plan_payload, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    contract = plan_payload.get("wrapper_contract", {}) if isinstance(plan_payload, dict) else {}
    coding_delegate = contract.get("coding_delegate", {}) if isinstance(contract, dict) else {}
    work_owner_mode = str(interaction.get("work_owner_mode", "external_executor"))
    selected_executor = interaction.get("selected_executor_profile")
    dispatch_policy = str(interaction.get("dispatch_policy", "ask_before_dispatch"))
    platform = interaction.get("platform")
    return build_wrapper_session_record(
        {
            "session_id": session_id,
            "thread_key": interaction["thread_key"],
            "source": interaction["source"],
            "source_metadata": interaction.get("source_metadata", {}),
            "message_sha256": interaction["message_sha256"],
            "message_length": interaction["message_length"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": _status_from_interaction(interaction),
            "decision": "none",
            "route": route,
            "plan": {
                "status": plan.get("status", ""),
                "recommended_workflow": plan.get("recommended_workflow", ""),
                "recommended_harness": plan.get("recommended_harness", ""),
                "coding_delegate_available": str(bool(coding_delegate.get("available", False))).lower(),
            },
            "work_owner_mode": work_owner_mode,
            "selected_executor_profile": selected_executor if isinstance(selected_executor, str) else None,
            "dispatch_policy": dispatch_policy,
            "prompt_handoff": {},
            "runtime_handoff": {},
            "current_run_id": "",
            "record_provenance": record_provenance or {},
            # Compact platform identity only -- refs, capabilities, and limits
            # never persist.
            "platform": compact_session_platform(platform) if isinstance(platform, dict) else None,
        }
    )


def _accepted_status_for_session(session: dict[str, Any]) -> str:
    plan = session.get("plan", {})
    coding_delegate_available = isinstance(plan, dict) and str(plan.get("coding_delegate_available", "")) == "true"
    selected = session.get("selected_executor_profile")
    if coding_delegate_available and not selected:
        return "executor_choice_required"
    return "plan_accepted"


def _status_from_interaction(interaction: dict[str, object]) -> str:
    next_action = str(interaction.get("next_action", ""))
    if next_action == "cancel":
        return "cancelled"
    if next_action == "answer_clarification":
        return "clarifying"
    if next_action == "dispatch_to_workflow":
        return "routed"
    if str(interaction.get("mode", "")) == "route":
        return "routed"
    if next_action == "present_plan":
        return "plan_presented"
    return "plan_presented"


def _source_metadata(event_or_message: dict[str, Any] | str) -> dict[str, str]:
    if not isinstance(event_or_message, dict):
        return {}
    return extract_source_metadata(event_or_message)


def _session_dir(paths: OmhPaths, session_id: str) -> Path:
    return paths.runtime_wrapper_sessions_dir / session_id


def _existing_session(paths: OmhPaths, session_id: str) -> dict[str, Any]:
    session = read_wrapper_session(paths, session_id)
    if not session:
        raise FileNotFoundError(session_id)
    return session


def _next_action_for_session(session: dict[str, Any]) -> str:
    return {
        "plan_presented": "accept_or_revise_plan",
        "clarifying": "answer_clarification",
        "routed": "dispatch_to_workflow",
        "plan_accepted": "prepare_handoff",
        "executor_choice_required": "choose_executor",
        "executor_selected": "prepare_handoff",
        "prompt_handoff_prepared": "show_prompt_handoff",
        "runtime_handoff_prepared": "show_runtime_handoff",
        "revision_requested": "revise_plan",
        "cancelled": "cancelled",
        "handoff_prepared": "show_status",
    }.get(str(session.get("status")), "show_status")


def _session_chat_response(session: dict[str, Any]) -> dict[str, object]:
    status = str(session.get("status", "plan_presented"))
    thread_key = str(session.get("thread_key", ""))
    raw_source_metadata = session.get("source_metadata", {})
    source_metadata = raw_source_metadata if isinstance(raw_source_metadata, dict) else {}

    def response(**kwargs: object) -> dict[str, object]:
        return _chat_response(**kwargs, source_metadata=source_metadata)

    if status == "plan_presented":
        return response(
            kind="plan",
            headline="I drafted a plan for this request.",
            body="Please accept or revise the plan before any coding handoff is prepared.",
            phase="planning",
            next_action="accept_or_revise_plan",
            thread_key=thread_key,
            actions=[
                _action("accept_plan", "Accept plan", "primary"),
                _action("revise_plan", "Revise plan", "secondary"),
                _action("cancel", "Cancel", "secondary"),
            ],
            claim_boundary="A wrapper session plan is not execution evidence.",
        )
    if status == "plan_accepted":
        return response(
            kind="handoff",
            headline="The plan is accepted.",
            body="I can prepare the selected executor/runtime handoff, but no executor or runtime work is observed yet.",
            phase="plan_accepted",
            next_action="prepare_handoff",
            thread_key=thread_key,
            actions=[_action("prepare_handoff", "Prepare handoff", "primary"), _action("cancel", "Cancel", "secondary")],
            claim_boundary="Plan acceptance is not execution evidence.",
        )
    if status == "executor_choice_required":
        return response(
            kind="handoff",
            headline="Choose the coding agent.",
            body="Keep this in Hermes, hand it to Claude Code, hand it to Codex or an oh-my runtime path, or split it across several agents in parallel before this becomes a prepared coding handoff.",
            phase="executor_choice_required",
            next_action="choose_executor",
            thread_key=thread_key,
            actions=[
                _action("choose_executor", "Choose coding agent", "primary"),
                _action("prepare_agent_board_card", "Split across agents", "secondary"),
                _action("cancel", "Cancel", "secondary"),
            ],
            claim_boundary="Choosing a coding agent is not dispatch or implementation evidence.",
        )
    if status == "executor_selected":
        selected = str(session.get("selected_executor_profile") or "Hermes")
        if session.get("work_owner_mode") == "runtime_handoff":
            return response(
                kind="handoff",
                headline="The runtime choice is recorded.",
                body=f"I can prepare the {selected} runtime contract next. No runtime work, team, swarm, or worktree is observed yet.",
                phase="executor_selected",
                next_action="prepare_handoff",
                thread_key=thread_key,
                actions=[_action("prepare_handoff", "Prepare handoff", "primary"), _action("cancel", "Cancel", "secondary")],
                claim_boundary="Runtime selection is not runtime start, dispatch, or implementation evidence.",
            )
        return response(
            kind="handoff",
            headline="The executor choice is recorded.",
            body=f"I can prepare the {selected} handoff next. No executor work is observed yet.",
            phase="executor_selected",
            next_action="prepare_handoff",
            thread_key=thread_key,
            actions=[_action("prepare_handoff", "Prepare handoff", "primary"), _action("cancel", "Cancel", "secondary")],
            claim_boundary="Executor selection is not dispatch evidence.",
        )
    if status == "prompt_handoff_prepared":
        selected = str(session.get("selected_executor_profile") or "executor")
        # Same label seam as the contract path's show-prompt body: the header
        # must read "Prepared prompt for Codex:", never the raw profile id.
        label = executor_label(selected)
        raw_prompt_handoff = session.get("prompt_handoff")
        prompt_handoff = raw_prompt_handoff if isinstance(raw_prompt_handoff, dict) else {}
        return response(
            kind="handoff",
            headline="A prompt handoff is ready.",
            body=_prompt_handoff_show_body(
                label,
                first_line=(
                    f"The {selected} prompt is prepared for copy/pass-through only; "
                    "no runtime run or executor evidence exists yet."
                ),
                prompt_handoff=prompt_handoff,
                delegation_payload={},
            ),
            phase="prompt_handoff_prepared",
            next_action="show_prompt_handoff",
            thread_key=thread_key,
            actions=[
                _action("show_prompt_handoff", "Show prompt", "primary"),
                _action("copy_prompt_handoff", "Copy prompt", "secondary"),
                _action("show_status", "Show status", "secondary"),
            ],
            claim_boundary="Prompt handoff is not dispatch, implementation, review, CI, or merge evidence.",
        )
    if status == "runtime_handoff_prepared":
        selected = str(session.get("selected_executor_profile") or "runtime")
        primary_action = "start_hermes_coding" if selected == "hermes" else "start_runtime"
        primary_label = "Start Hermes coding" if selected == "hermes" else "Start runtime"
        extra_actions = [_action_from_spec(spec) for spec in hermes_coding_team_extra_action_specs()] if selected == "hermes" else []
        body = (
            hermes_coding_team_body()
            if selected == "hermes"
            else f"The {selected} runtime contract is prepared with team/swarm, worker-protocol, and worktree guidance; no runtime evidence exists yet."
        )
        return response(
            kind="handoff",
            headline="A runtime handoff is ready.",
            body=body,
            phase="runtime_handoff_prepared",
            next_action="show_runtime_handoff",
            thread_key=thread_key,
            actions=[
                _action("show_runtime_handoff", "Show runtime", "primary"),
                *extra_actions,
                _action(primary_action, primary_label, "primary", enabled=False),
                _action("prepare_worktree", "Prepare worktree", "secondary", enabled=False),
                _action("start_team", "Start team", "secondary", enabled=False),
                _action("start_swarm", "Start swarm", "secondary", enabled=False),
                _action("show_status", "Show status", "secondary"),
            ],
            claim_boundary=(
                hermes_coding_team_claim_boundary()
                if selected == "hermes"
                else "Runtime handoff is not runtime start, worktree creation, worker dispatch, implementation, review, CI, or merge evidence."
            ),
        )
    if status == "revision_requested":
        return response(
            kind="clarification",
            headline="A plan revision was requested.",
            body="I will keep the request in planning until a revised plan is accepted.",
            phase="revision_requested",
            next_action="revise_plan",
            thread_key=thread_key,
            actions=[_action("revise_plan", "Revise plan", "primary"), _action("cancel", "Cancel", "secondary")],
            claim_boundary="Revision request is not execution evidence.",
        )
    if status == "cancelled":
        return response(
            kind="status",
            headline="This wrapper session is cancelled.",
            body="No handoff or executor work should be inferred from this session.",
            phase="cancelled",
            next_action="cancelled",
            thread_key=thread_key,
            actions=[],
            claim_boundary="Cancelled session has no execution evidence.",
        )
    if status == "clarifying":
        return response(
            kind="clarification",
            headline="I need one clarification before routing this.",
            body="Please answer the clarification before any plan or handoff is prepared.",
            phase="clarifying",
            next_action="answer_clarification",
            thread_key=thread_key,
            actions=[_action("answer:clarify", "Answer clarification", "primary"), _action("cancel", "Cancel", "secondary")],
            claim_boundary="No execution has started.",
        )
    return response(
        kind="status",
        headline="I have a conservative wrapper-session status.",
        body="Session state is available, but execution claims require linked runtime evidence.",
        phase="status",
        next_action="show_status",
        thread_key=thread_key,
        actions=[],
        claim_boundary="Wrapper session state is not execution evidence.",
    )


def _chat_response(
    *,
    kind: str,
    headline: str,
    body: str,
    phase: str,
    next_action: str,
    thread_key: str,
    actions: list[dict[str, object]],
    claim_boundary: str,
    source_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    state = {"phase": phase, "next_action": next_action, "thread_key": thread_key}
    usage_trace = usage_trace_payload(kind=kind, phase=phase, next_action=next_action, state=state)
    headline_with_prefix = headline_with_usage_prefix(headline, str(usage_trace["visible_prefix"]))
    return {
        "schema_version": CHAT_RESPONSE_SCHEMA_VERSION,
        "kind": kind,
        "visibility": "thread",
        "headline": headline_with_prefix,
        "plain_headline": headline,
        "body": body,
        "state": state,
        "usage_trace": usage_trace,
        "messenger_rendering": messenger_rendering_contract(
            visible_prefix=str(usage_trace["visible_prefix"]),
            first_line=headline_with_prefix,
            body=body,
            claim_boundary=claim_boundary,
            render_profile=render_profile_for_source(_source_from_thread_key(thread_key), source_metadata),
            source=_source_from_thread_key(thread_key),
        ),
        "actions": actions,
        "claim_boundary": claim_boundary,
    }


def _source_from_thread_key(thread_key: str) -> str:
    if ":" not in thread_key:
        return "generic"
    return thread_key.split(":", 1)[0] or "generic"


def _action(action_id: str, label: str, style: str, *, enabled: bool = True) -> dict[str, object]:
    return {"id": action_id, "label": label, "style": style, "enabled": enabled, "payload": {}}


def _action_from_spec(spec: dict[str, object]) -> dict[str, object]:
    return _action(
        str(spec.get("id", "")),
        str(spec.get("label", "")),
        str(spec.get("style", "")),
        enabled=bool(spec.get("enabled", True)),
    )
