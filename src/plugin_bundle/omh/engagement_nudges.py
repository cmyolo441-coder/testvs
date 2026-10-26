"""Behaviour-triggered nudges toward a declared plan and a routed lane.

The reported defect is that OMH almost never engages on an ordinary prompt:
"간단한 작업이든 큰 작업이든 다양하게 테스트해봤지만 거의 todo나 에이전트 호출을
하지 않더라구요." Everything OMH had for this read the user's WORDS -- route
hints, the awareness matcher, the workflow triggers -- so a session that never
said an OMH word got nothing, and the first-turn primer only makes the model
aware, not active.

So the trigger here is the model's own BEHAVIOUR, taken from the host's tool
calls: how many files it has changed, and how much searching it has done
directly. Nothing in the user's message is consulted at any point. A keyword
trigger is what already exists and it is what is failing.

The shape is taken from the upstream `agent-usage-reminder` hook recorded in
the note on code-level borrowings in `docs/SKILL-SOURCES.md`, which solves the
same problem for opencode, and it keeps four of that hook's five properties:

* the trigger is an observed tool outcome, not a phrase;
* the text rides the tool result (`transform_tool_result`), so it costs no
  extra turn and needs no second injection point;
* it is bounded per session (`MAX_ENGAGEMENT_NUDGES`);
* it latches off permanently once the thing happened, rather than decaying.

The fifth, "only nudge the orchestrator", does not transfer as written. Its
opencode form reads the agent name off the session; the Hermes seam carries
`tool_name`, `args`, `result`, `task_id`, `session_id`, `tool_call_id`,
`turn_id`, `api_request_id`, `duration_ms`, `status`, `error_type`,
`error_message` and no agent identity, and a delegated child runs under its own
`session_id`, so per-session keying does not separate it either. The
substitute is `subagent_start`: the host hands over `child_session_id` from the
same code path that creates the child, `nudge_budget` records it, and this
module refuses to nudge a session on that list.

Nothing here raises. Hermes wraps the transform in `except Exception` and logs
at debug (`model_tools._apply_transform_tool_result_hook`), so a handler that
raised would leave no trace at all -- the failure would be invisible rather
than loud. Every decline is counted by reason instead, readable through
`engagement_nudge_declines()`.

The nudges are prepared instruction. Candidate budgets do not prove delivery:
Hermes selects the first transformer string, not necessarily ours. Outcome
observation is independent; see docs/ENGAGEMENT-OBSERVATION.md for its bounds.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Final

from . import runtime_paths
from .hooks.nudge_budget import (
    ENGAGEMENT_LOCK,
    engagement_call,
    DELEGATION_LATCH_FIELD,
    DELEGATION_NUDGES_FIELD,
    DIRECT_READS_FIELD,
    MUTATIONS_FIELD,
    PLAN_LATCH_FIELD,
    PLAN_NUDGES_FIELD,
    bump_engagement_count,
    engagement_count,
    latch_engagement,
    record_distinct_direct_read,
    session_is_delegated,
)
# The one canonicalization of a tool call's arguments in this bundle. The
# repeat guard computes this same digest for the same call at
# `pre_tool_call`; a second hashing path here would be a second answer to
# "is this the same call", and the two would disagree the first time either
# of them changed.
from .tool_bursts import tool_args_digest
from .native_tools import NATIVE_TODO_TOOL, native_tool_presence_clause

ENGAGEMENT_NUDGE_SCHEMA_VERSION: Final = "omh_engagement_nudge/v1"
# Its own JSON key, the way `code_mode_guidance` adds one: a tool result that
# parses as a JSON object must keep parsing after this module has touched it,
# so the text is a new field there and only appended when the result is plain.
ENGAGEMENT_NUDGE_KEY: Final = "omh_engagement"

# Host tool names, copied from Hermes rather than imported -- the bundle may
# only import from inside itself, and Hermes is a different repository, so no
# parity test can exist for these the way it can for the router vocabulary.
# Each set names its source file so a reader can check it by hand.
#
# `agent/tool_result_classification.py`: FILE_MUTATING_TOOL_NAMES. These tools
# CAN change files; their structured outcomes establish whether they did. Not
# widened to `terminal` or `execute_code`: a session that only ran tests has
# done no work a checklist would track, and both of those are mostly that.
FILE_MUTATING_TOOLS: Final[frozenset[str]] = frozenset({"write_file", "patch"})
# `tools/code_execution_tool.py`: SANDBOX_ALLOWED_TOOLS, minus the mutating and
# shell ones. These are the calls a delegated lane would have made off this
# context -- opencode's `TARGET_TOOLS` (grep, glob, webfetch, exa) mapped onto
# the names Hermes actually registers.
DIRECT_READ_TOOLS: Final[frozenset[str]] = frozenset(
    {"read_file", "search_files", "web_search", "web_extract"}
)
# These tool results describe preparation/control/attempts, not child starts.
# Only the host's subagent_start lifecycle event sets the delegation latch.
DELEGATION_TOOLS: Final[frozenset[str]] = frozenset({"delegate_task", "omh_delegate_route"})

# Counter names in the shared per-session map. Defined at `nudge_budget`,
# which owns the map, the eviction policy and the list of which of these
# survive a process restart -- one spelling, in the file that has to name
# them for that last question.
_MUTATIONS: Final = MUTATIONS_FIELD
_DIRECT_READS: Final = DIRECT_READS_FIELD
_PLAN_NUDGES: Final = PLAN_NUDGES_FIELD
_DELEGATION_NUDGES: Final = DELEGATION_NUDGES_FIELD
_PLAN_LATCH: Final = PLAN_LATCH_FIELD
_DELEGATION_LATCH: Final = DELEGATION_LATCH_FIELD

# Three file-mutating calls. One is a typo fix. Two is an edit and its test --
# the two shapes where a checklist is pure overhead, and the two a person holds
# in their head without help. Three is the first count past both, and it is
# also where the HUD stops being able to show what is left without a plan to
# show. The pinned negative case sits at two.
PLAN_NUDGE_FILE_MUTATION_THRESHOLD: Final = 3
# Five DISTINCT direct search/read calls. Below that it is ordinary
# orientation: open a file, grep once, open what the grep found. At five the
# session has run a search pass that one `explore` lane would have run in a
# single call and off this context window, which is the case the nudge exists
# for. opencode fires on the first such call because its target set is
# grep/glob/webfetch only; Hermes' `read_file` is far more ordinary than that,
# so the threshold carries what its narrower set carried. The pinned negative
# case sits at four.
#
# Distinct, because counting calls made a loop indistinguishable from a search
# pass and let it spend the budget meant for one. Measured: session
# `20260919_140745_db409e` issued 203 `search_files` calls, the large majority
# identical; the nudge fired twice, said "delegate this", and was silent for
# the remaining ~180. The session's actual problem was that it was repeating
# one search that returned nothing, and the only OMH mechanism watching
# searches had no way to know (#1701).
DELEGATION_NUDGE_DIRECT_READ_THRESHOLD: Final = 5
# Two per kind per session. opencode allows three; this text rides a tool
# result on a surface that already spends a first-turn primer, and a nudge the
# model has declined twice is not going to work the third time.
MAX_ENGAGEMENT_NUDGES: Final = 2

# "No plan is declared" is read off the session's plan record, which a native
# todo_list write reaches too (`native_todo`), so the nudge names both tools
# and asks for one of them. The tool choice was paid for out of the sentence's
# own length, not the per-session bound in `tests/test_engagement_nudges.py`:
# "the steps you are actually taking" became "your steps", "what is planned
# and what is left" became "what is left", and "has something to reconcile
# ... against" became "can check ... against it".
PLAN_NUDGE_TEXT: Final = (
    "[OMH plan todo] This session has changed files {count} times and no plan is "
    f"declared. Declare one now in ONE tool -- {NATIVE_TODO_TOOL} when "
    f"{native_tool_presence_clause(NATIVE_TODO_TOOL)}, else omh_todo -- with "
    "your steps, exactly one active, so the person sees what is left and the "
    "next turn can check a completion claim against it. Items are "
    "declarations, never execution evidence."
)
# What the two tools actually do, measured in the host rather than inferred
# from their names. This sentence used to end "omh_delegate_route to pick one,
# or delegate_task for a Hermes subagent, and keep working while it runs", and
# that was wrong about both halves:
#
# * `omh_delegate_route` writes `delegation.*` keys that apply to the NEXT
#   dispatch -- its own evidence boundary in `tools/delegate_route_tool.py`
#   says exactly that. Nothing runs, so there is nothing to work alongside.
# * `delegate_task` does run something, and "keep working while it runs" is
#   the one thing the host tells the model not to do about it. Its schema text
#   reads "Background results are delivered only BETWEEN your turns: finish
#   whatever does not depend on them, then give a one-line status and END YOUR
#   TURN. Never wait or poll" (`tools/delegate_tool.py`, `_DESCRIPTION_HEAD`).
#
# Nor can the model choose how it runs. `run_agent._dispatch_delegate_task`
# passes `background=not (depth > 0)`, so a top-level model delegation is
# always backgrounded and the schema-level `background` is unadvertised and
# ignored; the `background=False` default on the Python signature is for
# direct callers. Measured on hermes-agent 0.21.3 -- Hermes is a different
# repository, so no parity test can hold this the way one holds the router
# vocabulary, and the call sites are named here so a reader can check by hand.
DELEGATION_NUDGE_TEXT: Final = (
    "[OMH delegation] This session has run {count} different search/read calls "
    "directly; no child start was observed. Consider a lane for independent work "
    "outside this context: omh_delegate_route picks the model for the next "
    "dispatch, delegate_task spawns the subagent. Never wait or poll on a "
    "dispatched lane -- carry on with what does not depend on it. Routing is "
    "a prepared handoff, never execution, review, CI, or merge evidence."
)

# Why a decline was not a nudge. Counted rather than raised, because the host
# swallows exceptions from this seam: without this, a module that broke would
# be indistinguishable from a session that simply had nothing to say.
_declines: "Counter[str]" = Counter()


def engagement_nudge_declines() -> dict[str, int]:
    """A copy of the decline tally, by reason. Diagnostics, never a gate."""
    return dict(_declines)


def reset_engagement_declines() -> None:
    """Test seam: forget the decline tally. Paired with `reset_nudge_budget`."""
    _declines.clear()


def record_engagement_observer_failure(error_type: str) -> None:
    """Record that the delegated-session observer swallowed a failure.

    `subagent_start` is the only feed for the delegated-session set, and its
    swallow is right: the host already calls it inside its own quiet block, and
    failing to record one child must not interrupt that child's spawn. But a
    swallow with nothing written is observable only as an ABSENCE -- the
    `delegated_session` decline that never happens, which nobody notices
    without already expecting it and counting. Worse, if whatever broke the
    observer also keeps the nudge path away from that session, the two cancel
    and the trace is empty.

    So the REPORT is widened rather than the `except`: one tally entry, in the
    same readout as every other reason this path declined to act, because the
    observer exists only to feed it.
    """
    _declines[f"observer_error:{error_type or 'Unknown'}"] += 1


def annotate_engagement_nudge(
    *,
    tool_name: object,
    result: object,
    args: object = None,
    session_id: str = "",
    omh_home: str = "",
    hermes_home: str = "",
    tool_call_id: str = "",
    turn_id: str = "",
    status: str = "",
) -> str | None:
    """Return *result* carrying a nudge, or ``None`` to pass through untouched.

    ``args`` is the call's arguments, which the host passes this seam already
    coerced to the tool's schema types. They are read for one purpose: to tell
    the same search twice from two different searches. Nothing is stored but
    the digest.

    Fail-open by seam contract and by construction: every path that is not a
    clean, budgeted, unlatched nudge returns ``None`` after recording why.
    """
    try:
        with ENGAGEMENT_LOCK:
            return _annotate(
                tool_name=tool_name, result=result, args=args,
                session_id=session_id, omh_home=omh_home, hermes_home=hermes_home,
                tool_call_id=tool_call_id, turn_id=turn_id, status=status,
            )
    except Exception as exc:  # noqa: BLE001 - see module docstring: the host
        # swallows and debug-logs anything this raises, so a raise here is a
        # silent disappearance. The failure is recorded by type and the tool
        # result passes through unchanged.
        _declines[f"error:{type(exc).__name__}"] += 1
        return None


def _result_object(result: object) -> dict:
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return {}
    return result if isinstance(result, dict) else {}


def _mutation_effect(name: str, data: dict, status: str) -> str:
    """Read the file-tool contract, never success words in free text.

    The host's file_mutation_result_landed checks bytes_written / success.
    Also honor no_change and partial V4A results. A generic error may occur
    after the write, so missing effect evidence is unknown, not no effect.
    """
    if status == "blocked" or data.get("no_change") is True:
        return "none"
    changed_paths = any(isinstance(data.get(field), list) and bool(data[field])
                        for field in ("files_modified", "files_created", "files_deleted"))
    bytes_written = data.get("bytes_written")
    landed = (
        name == "write_file" and type(bytes_written) is int and bytes_written >= 0
    ) or (name == "patch" and data.get("success") is True)
    failed = status not in ("", "ok") or bool(data.get("error")) or data.get("success") is False
    if failed:
        # WriteResult(error=...) includes bytes_written=0 by default: on an
        # error that zero cannot prove even an empty-file write took place.
        known_effect = changed_paths or (type(bytes_written) is int and bytes_written > 0)
        return "partial" if known_effect else "unknown"
    return "landed" if landed else "unknown"


def observe_engagement_outcome(**kwargs: Any) -> dict[str, Any] | None:
    """Keep an observer fault from skipping the other post-tool observers."""
    try:
        return _observe_engagement_outcome(**kwargs)
    except Exception as exc:  # noqa: BLE001 - classified in the policy gate
        record_engagement_observer_failure(type(exc).__name__)
        return None


def _observe_engagement_outcome(
    *, tool_name: object, result: object, args: object = None,
    session_id: str = "", tool_call_id: str = "", turn_id: str = "",
    status: str = "", omh_home: str = "", hermes_home: str = "",
) -> dict[str, Any] | None:
    """Observe a terminal outcome independently of presentation.

    Registry calls post before transform; the outer executor suppresses that
    post and emits its own AFTER transform. The transform may therefore feed
    this same observer, using host call identity to count once in either order.
    No call identity means unknown correlation and no count, not invented work.
    """
    name = str(tool_name or "")
    if name not in FILE_MUTATING_TOOLS | DIRECT_READ_TOOLS | DELEGATION_TOOLS:
        _declines["tool_not_watched"] += 1
        return None
    if not session_id:
        _declines["no_session_id"] += 1
        return None
    if not isinstance(tool_call_id, str) or not tool_call_id or len(tool_call_id) > 512:
        _declines["unknown_call_identity"] += 1
        return None
    home = str(runtime_paths.plugin_home(omh_home or None))
    runtime_paths.plugin_home(hermes_home or None, hermes=True)
    with ENGAGEMENT_LOCK:
        if session_is_delegated(session_id, omh_home=home):
            _declines["delegated_session"] += 1
            return None
        event = engagement_call(session_id, tool_call_id, str(turn_id or ""), name, omh_home=home)
        if event:
            return event
        data = _result_object(result)
        event.update(kind="none", count=0, rendered=False)
        if name in FILE_MUTATING_TOOLS:
            effect = _mutation_effect(name, data, status)
            event["effect"] = effect
            field = {"landed": _MUTATIONS, "partial": "partial_file_mutations", "unknown": "unknown_file_mutations"}.get(effect)
            if field:
                count = bump_engagement_count(session_id, field, omh_home=home)
                if effect == "landed":
                    event.update(kind="mutation", count=count)
        elif name in DIRECT_READ_TOOLS:
            # Attempts still describe search effort, but pre-dispatch refusals do not.
            if status != "blocked":
                bump_engagement_count(session_id, _DIRECT_READS, omh_home=home)
                digest = tool_args_digest(args)
                if digest:
                    count = record_distinct_direct_read(session_id, f"{name}:{digest}", omh_home=home)
                    event.update(kind="read", count=count)
                else:
                    # Carried from the complete-argument-identity change: a call
                    # whose arguments could not be fingerprinted has no identity
                    # to be distinct from, so it is declined by name rather than
                    # counted as a new distinct read.
                    _declines["unknown_read_identity"] += 1
        elif name == "omh_delegate_route":
            action = args.get("action", "set") if isinstance(args, dict) else None
            if (action in ("set", "fallback") and status in ("", "ok")
                    and not data.get("error") and data.get("status") in ("routed", "fell_back", "exhausted_to_inherit")):
                bump_engagement_count(session_id, "route_prepared", omh_home=home)
                event["kind"] = "prepared"
        # delegate_task results (including status/list/stop and failed dispatch)
        # never prove a child existed. subagent_start is the source of that fact.
        return event


def _annotate(
    *, tool_name: object, result: object, args: object, session_id: str,
    omh_home: str, hermes_home: str, tool_call_id: str, turn_id: str, status: str,
) -> str | None:
    event = observe_engagement_outcome(
        tool_name=tool_name, result=result, args=args, session_id=session_id,
        omh_home=omh_home, hermes_home=hermes_home, tool_call_id=tool_call_id,
        turn_id=turn_id, status=status,
    )
    if event is None or event["rendered"]:
        return None
    # This budgets a candidate, NOT delivery: Hermes may select another
    # transformer's string. There is no output-acceptance receipt on this hook.
    event["rendered"] = True
    if event["kind"] not in ("mutation", "read"):
        return None
    home = str(runtime_paths.plugin_home(omh_home or None))
    host_home = str(runtime_paths.plugin_home(hermes_home or None, hermes=True))
    if event["kind"] == "mutation":
        return _plan_nudge(session=session_id, count=event["count"], result=result,
                           omh_home=home, hermes_home=host_home)
    if event["kind"] == "read":
        return _delegation_nudge(session=session_id, count=event["count"], result=result, omh_home=home)
    return None


def _plan_nudge(
    *,
    session: str,
    count: int,
    result: object,
    omh_home: str,
    hermes_home: str,
) -> str | None:
    if count < PLAN_NUDGE_FILE_MUTATION_THRESHOLD:
        _declines["below_work_threshold"] += 1
        return None
    if engagement_count(session, _PLAN_LATCH, omh_home=omh_home):
        _declines["plan_already_declared"] += 1
        return None
    if engagement_count(session, _PLAN_NUDGES, omh_home=omh_home) >= MAX_ENGAGEMENT_NUDGES:
        _declines["plan_budget_spent"] += 1
        return None
    if _plan_declared(session=session, omh_home=omh_home, hermes_home=hermes_home):
        # Latched in the map so the record is read at most a handful of times
        # per session rather than on every mutating call.
        latch_engagement(session, _PLAN_LATCH, omh_home=omh_home)
        _declines["plan_already_declared"] += 1
        return None
    carried = _carry(result, PLAN_NUDGE_TEXT.format(count=count))
    if carried is None:
        _declines["result_not_carryable"] += 1
        return None
    _ = bump_engagement_count(session, _PLAN_NUDGES, omh_home=omh_home)
    return carried


def _delegation_nudge(*, session: str, count: int, result: object, omh_home: str) -> str | None:
    """``count`` is DISTINCT `(tool, argument digest)` pairs, not calls.

    A repeat adds nothing to it, so a loop never reaches the threshold and
    never spends the budget: the whole of #1701 in one substitution.
    """
    if count < DELEGATION_NUDGE_DIRECT_READ_THRESHOLD:
        _declines["below_read_threshold"] += 1
        return None
    if engagement_count(session, _DELEGATION_LATCH, omh_home=omh_home):
        _declines["lane_already_started"] += 1
        return None
    if engagement_count(session, _DELEGATION_NUDGES, omh_home=omh_home) >= MAX_ENGAGEMENT_NUDGES:
        _declines["delegation_budget_spent"] += 1
        return None
    carried = _carry(result, DELEGATION_NUDGE_TEXT.format(count=count))
    if carried is None:
        _declines["result_not_carryable"] += 1
        return None
    _ = bump_engagement_count(session, _DELEGATION_NUDGES, omh_home=omh_home)
    return carried


def _plan_declared(*, session: str, omh_home: str, hermes_home: str) -> bool:
    """Whether this session has a plan record right now, open or just finished.

    Reads the record through the one predicate that owns the question
    (`todo_reconciliation.plan_is_declared`), never a string the model wrote.
    Deliberately NOT `plan_is_established`: that one means "has work left", so
    latching on it would ask for a second plan the moment the first completed.

    The projection retires a finished plan to `absent` after a while, so this
    answers only "is there a plan now". The permanence lives in the caller's
    latch, which is why this is read at most a handful of times per session
    and never again once it answers True.

    An unreadable home answers "no plan", which costs a nudge and never
    suppresses one. That is the right direction for a best-effort read here:
    the alternative is a broken home silently disabling the nudge forever.
    """
    plan_is_declared = _todo_reconciliation.plan_is_declared
    read_omh_todo = _runtime_reader.read_omh_todo

    todo = read_omh_todo(omh_home or None, hermes_home or None, session_ref=session)
    return isinstance(todo, dict) and plan_is_declared(todo)


def _carry(result: object, text: str) -> str | None:
    """Put *text* on *result* without breaking a host result that is JSON.

    A JSON object gains its own key, the way `code_mode_guidance` does, because
    appending to a serialized object makes it unparseable. Anything else is
    plain text and the nudge is appended. A JSON value that is not an object
    (a bare list, a number) is declined rather than guessed at.
    """
    if not isinstance(result, str) or not result:
        return None
    try:
        parsed: Any = json.loads(result)
    except (ValueError, TypeError):
        return f"{result}\n\n{text}"
    if not isinstance(parsed, dict):
        return None
    if ENGAGEMENT_NUDGE_KEY in parsed:
        return None
    parsed[ENGAGEMENT_NUDGE_KEY] = text
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


# Module scope, not call time: Hermes can evict the bundle from sys.modules
# while its callbacks stay live (#1979). Last, so an import cycle finds every name.
from . import runtime_reader as _runtime_reader
from . import todo_reconciliation as _todo_reconciliation
