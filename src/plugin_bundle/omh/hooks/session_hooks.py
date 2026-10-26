from __future__ import annotations

from .. import runtime_paths

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from ..degradation import (
    COMPONENT_DELEGATION_ROUTE_RESTORE,
    degradation_payload,
    runtime_binding_degradation,
)
from ..delegation_ledger import observe_subagent_start, observe_subagent_stop
from ..delegation_route_restore import restore_delegation_baseline
from ..engagement_nudges import record_engagement_observer_failure
from ..host_observation import host_session_id, observe_plugin_hook_call
from ..team_observer import observe_team_dispatch, observe_team_return
from .nudge_budget import (
    DELEGATION_LATCH_FIELD,
    ENGAGEMENT_LOCK,
    latch_engagement,
    note_delegated_session,
)


def subagent_start(**kwargs) -> None:
    """Record that ``child_session_id`` names a delegated lane, not an orchestrator.

    A child is suppressed in its own home/session; the parent gets a durable
    lane_started latch. The host emits this after constructing/attaching the
    child, before it runs: this is child-lifecycle evidence, never completion
    or a model-call receipt. Batch children stay distinct; the parent latch is
    idempotent. Missing lifecycle evidence cannot be replaced by a tool name.

    Observation only: never blocks or starts a spawn. A host that omits the
    callback leaves delegation unknown, not successful.

    Nothing here raises. The caller already wraps the invocation in its own
    quiet block, so a raise would be swallowed and this would simply stop
    recording -- invisibly, which is the failure worth avoiding. So the swallow
    writes a line: otherwise a failure here is observable ONLY as an absence
    (the `delegated_session` decline that never happens), and an absence needs a
    reader who already knew to expect it.
    """
    try:
        observe_plugin_hook_call("subagent_start", kwargs)
        home = str(runtime_paths.plugin_home(kwargs.get("omh_home")))
        runtime_paths.plugin_home(kwargs.get("hermes_home"), hermes=True)
        child = kwargs.get("child_session_id")
        parent = kwargs.get("parent_session_id")
        if not isinstance(child, str) or not child.strip():
            return None
        with ENGAGEMENT_LOCK:
            note_delegated_session(child, omh_home=home)
            if isinstance(parent, str) and parent.strip() and parent != child:
                latch_engagement(parent, DELEGATION_LATCH_FIELD, omh_home=home)
    except Exception as exc:  # noqa: BLE001 - swallowed upstream either way;
        # failing to record one child must not interrupt that child's spawn.
        # Recorded rather than silent: see `record_engagement_observer_failure`.
        record_engagement_observer_failure(type(exc).__name__)
        return None
    # A checked-team helper is bound to the attempt its goal names; any other
    # goal leaves on a substring test (`team_observer`). Never raises.
    observe_team_dispatch(kwargs)
    # The delegation graph's record of who started whom (`delegation_ledger`):
    # ids, turn and OMH's own route, never the goal. Never raises.
    observe_subagent_start(kwargs)
    return None


def subagent_stop(**kwargs) -> None:
    """Mark a checked-team helper as returned, so its part may be checked.

    The host's own end-of-child callback, fired once per child with the same
    `child_session_id` `subagent_start` carried. It is the only return signal
    `omh_team` accepts: the team reads neither a helper's summary nor its
    status, and records nothing about the outcome beyond "it came back".
    Never raises, never blocks.

    The delegation ledger, for the delegation graph, keeps the child's own
    status word and duration; it does not read the summary either.
    """
    observe_team_return(kwargs)
    observe_subagent_stop(kwargs)
    return None


def on_session_start(**kwargs) -> dict[str, object] | None:
    """Put back a delegation route whose writing session is gone.

    The ordinary way back is the end of the task that wrote the route. A
    session killed mid-task never reaches that, so this is the second path,
    and the reason it is a second path rather than the only one: it may act
    only on a route whose recorded writer is NOT a live session, so it cannot
    pull a baseline out from under a session that is still dispatching.

    OMH did not register `on_session_start` before this. It is the host's own
    first-turn lifecycle callback (`hermes_cli.plugins.VALID_HOOKS`), bounded
    and fail-open, and it carries the `session_id` the decision needs -- which
    is why the restore rides it rather than the first `pre_llm_call`, where the
    same work would have to re-derive "is this the first turn" on a hot path
    that runs every turn.

    Nothing here can stop a session starting. A binding failure returns the
    same bounded degradation block the sibling hooks return, a restore that
    fails returns its own, and the host discards the return either way -- the
    point of returning it is that the failure is named rather than absent.
    """
    try:
        omh_home = runtime_paths.plugin_home(kwargs.get("omh_home"))
        hermes_home = runtime_paths.plugin_home(kwargs.get("hermes_home"), hermes=True)
    except (runtime_paths.RuntimeBindingError, OSError, RuntimeError) as exc:
        return runtime_binding_degradation(exc)
    observe_plugin_hook_call("on_session_start", kwargs)
    restore = _restore_route(hermes_home, omh_home, trigger="session_start", orphaned=True)
    payload: dict[str, object] = {"status": "session_start", "route_restore": restore}
    _attach_restore_degradation(payload, restore)
    return payload


def on_session_end(**kwargs) -> dict[str, object] | None:
    """Restore this TASK's delegation route, then checkpoint OMH runtime state.

    Despite the name this is not a session boundary. Hermes fires it from
    `agent/turn_finalizer.py`, whose own comment reads "run_conversation()
    runs once per message", and `docs/SESSION-ACTIVITY-RECEIPTS.md` already
    recorded that it runs at conversation-turn finalization. So a route
    written in a turn comes back at the end of that turn, which is the design
    (see `delegation_route_restore`) and not something to work around. On a
    gateway platform the recorded task id is the session id, so there a route
    survives later turns of the session and a missed restore is retried by
    the next turn end.
    """
    try:
        home = runtime_paths.plugin_home(kwargs.get("omh_home"))
        hermes_home = runtime_paths.plugin_home(kwargs.get("hermes_home"), hermes=True)
    except (runtime_paths.RuntimeBindingError, OSError, RuntimeError) as exc:
        return runtime_binding_degradation(exc)
    observe_plugin_hook_call("on_session_end", kwargs)
    # Scoped to the task that wrote the route, falling back to the session
    # when no task was recorded. This hook fires per turn, so a later turn
    # must not put the baseline back underneath a newer route. The host
    # passes `task_id` here and to the tool; it passes `turn_id` here but NOT
    # to the tool, so the task is the finest scope both ends can name -- and
    # a recorded task decides alone, because a compression split moves the
    # session id mid-turn while the task id stays put (`_writer_matches`).
    # In TUI and CLI a task is one turn; on every gateway platform the task
    # id is the session id, so there the scope is the session.
    writer_session = host_session_id(kwargs)
    writer_task = str(kwargs.get("task_id", "") or "").strip()
    restore = _restore_route(
        hermes_home,
        home,
        trigger="turn_end",
        writer_session=writer_session,
        writer_task=writer_task,
        # This conversation's own route ends with its task too, and only its
        # own: another session's route stays recorded for that session's next
        # dispatch, whatever this restore did to the keys (#2251). Released
        # inside the restore's lock, so the two cannot come apart.
        release_writer=writer_session,
    )
    runtime_dir = home / "runtime"
    if not runtime_dir.exists():
        payload: dict[str, object] = {"status": "no_runtime_state", "route_restore": restore}
        _attach_restore_degradation(payload, restore)
        return payload
    runs_dir = runtime_dir / "runs"
    run_count = len(list(runs_dir.glob("*/run.json"))) if runs_dir.exists() else 0
    state = _read_json(runtime_dir / "state.json")
    if not state and run_count == 0:
        payload = {"status": "no_runtime_state", "route_restore": restore}
        _attach_restore_degradation(payload, restore)
        return payload
    payload = {
        "schema_version": "omh_plugin_session_end/v1",
        "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_state_present": bool(state),
        "latest_run_id": str(state.get("last_run_id", "")) if isinstance(state, dict) else "",
        "run_count": run_count,
        "privacy": "metadata_only",
        "claim_boundary": "This checkpoint proves only that the local OMH plugin hook ran; it is not execution, review, CI, merge, or Hermes reload evidence.",
    }
    path = runtime_dir / "plugin-session-end.json"
    _atomic_write_json(path, payload)
    checkpoint: dict[str, object] = {
        "status": "checkpoint_written",
        "path": str(path),
        "route_restore": restore,
    }
    _attach_restore_degradation(checkpoint, restore)
    return checkpoint


def _restore_route(
    hermes_home: Path,
    omh_home: Path,
    *,
    trigger: str,
    writer_session: str | None = None,
    writer_task: str | None = None,
    orphaned: bool = False,
    release_writer: str | None = None,
) -> dict[str, object]:
    """Call the restore and never let its failure reach the host as a raise.

    The restore module already turns its own expected faults into a `status`,
    so the narrow catch here is for the unexpected one. It is narrow on
    purpose: a bare `except Exception` would also hide a contract break in the
    caller, and this hook is fail-open at the host anyway -- the value added
    by catching is the named status, not the survival.
    """
    try:
        return restore_delegation_baseline(
            hermes_home,
            omh_home=omh_home,
            trigger=trigger,
            require_writer_session=writer_session,
            require_writer_task=writer_task,
            require_writer_not_live=orphaned,
            release_writer=release_writer,
        )
    except (OSError, ValueError, TypeError) as exc:
        return {
            "status": "error",
            "trigger": trigger,
            "error": type(exc).__name__,
            "error_type": type(exc).__name__,
        }


def _attach_restore_degradation(payload: dict[str, object], restore: dict[str, object]) -> None:
    """Name a failed restore in the hook payload instead of leaving an absence.

    Only a real failure degrades. `no_baseline_recorded`, `foreign_edit`,
    `not_last_writer` and `writer_live` are the restore working: each is a
    decision not to touch a value, and reporting them as degradation would put
    a permanent warning in front of every healthy session.

    A restore that put the keys back but could not release the writer's
    conversation route (`writer_routes: unreleased: <type>`) is a failure
    too, reported here rather than left to the six-hour bound that retires
    the entry; likewise a turn end whose own release failed.
    """
    writer_routes = str(restore.get("writer_routes", ""))
    if writer_routes.startswith("unreleased: "):
        payload["omh_degradation"] = degradation_payload(
            [(COMPONENT_DELEGATION_ROUTE_RESTORE, writer_routes.removeprefix("unreleased: "))]
        )
        return
    conversation_route = restore.get("conversation_route")
    if isinstance(conversation_route, dict) and conversation_route.get("status") == "error":
        payload["omh_degradation"] = degradation_payload(
            [(COMPONENT_DELEGATION_ROUTE_RESTORE, str(conversation_route.get("error_type", "error")))]
        )
        return
    if str(restore.get("status", "")) not in ("error", "lock_unavailable"):
        return
    # `error_type` and not `error`: the degradation field is documented as a
    # sanitized exception CLASS NAME, and `safe_error_type` strips a
    # sentence's spaces rather than rejecting it, so passing the message
    # produced labels like `routerestorefailedOSError`.
    error_type = str(restore.get("error_type", "")) or str(restore.get("status", ""))
    payload["omh_degradation"] = degradation_payload(
        [(COMPONENT_DELEGATION_ROUTE_RESTORE, error_type)]
    )


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser().resolve()


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
