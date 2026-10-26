"""An edit made while a plan is still unaccepted is a question for a person.

`ralplan` is a planning gate. Its skill has said so in prose since it shipped:
one line routes a full delivery cycle to `ultrawork`, a worked example pairs
"$ralplan implement the refactor now and open the PR" with "stop at the
reviewed plan", and a third line says to start a follow-on engine "only on the
user's explicit go-ahead -- never auto-start an engine from acceptance alone".
A prompt that mixes implementation intent into the ask still produces
implementation. Routing is not what failed: measured on
`build_chat_interaction_payload`, that exact sentence dispatches to `ralplan`
at score 12 with `next_action: present_plan`. The message reaches the right
skill and the skill's own rule is then read past.

So this module adds no prose. It escalates the first file edit of an
unaccepted plan run to the host's human-approval gate, which is the only
directive in `pre_tool_call` a model cannot decline: `block` becomes a tool
result and one measured session read 185 host refusals and repeated the call
anyway, while `approve` is routed to `tools/approval.py::request_tool_approval`
and never returns to the model as text at all. That gate is also, literally,
somewhere to ask the person a question, which is what the owner asked for.

What it reads, and what it refuses to read
------------------------------------------
One record: this session's `omh_todo/v1` plan, through `read_omh_todo`. Two
fields of it decide everything, and neither is anybody's prose.

* `plan_stage == "awaiting_acceptance"` -- a closed-vocabulary field the
  planning writer stamps at declaration time (`todo_store`). Not the plan's
  title, not an item's text, not the model's narration. This repository has
  already paid for the alternative once: the plan's stop criterion used to be
  matched out of item text, and it read "verify the retry is not blocked on
  the session limit" as blocked while missing "waiting on the owner's review"
  -- wrong in both directions on ordinary input (`todo_store`,
  `MAX_TODO_BLOCKED_REASON_CHARS`).
* `own_record is True` -- the projection came from this session's OWN
  record rather than the home-wide fallback. Ownership is asked separately
  and answered first, because the renderer's verdict does not answer it: an
  unattributable plan reads as belonging, deliberately, so that a real plan
  is never hidden from its owner. A gate that stops a turn needs the
  opposite default, and reading the renderer's answer as ownership is the
  defect this gate shipped with (see the call site).
* `status == "established"` -- the projection's own verdict that the record
  is fresh and still has open work. Reused rather than re-derived, because
  staleness and "is there work left" are questions the HUD, the turn-end
  directive and `omh runtime todo show` already answer, and a second copy of
  those rules is how two readers come to disagree about the same plan.

Unknown never accuses
---------------------
The discipline PR #1738 landed for the unarmed-wait directive, applied to a
stronger action. Every path out of the narrow positive case returns None, and
the list is the point rather than an accident of control flow: no record, a
record this session does not own -- including a home-wide one belonging to
nobody -- a stale record, a finished plan, a record this build cannot
classify, an unreadable home, a call with no session id, and a tool that is
not an edit. A wrong escalation stops a person's work
and costs trust; a missed one leaves them exactly where the prose rule already
left them.

`accepted` and absent are both silence, and they are different states on
purpose. `accepted` is the person's recorded go-ahead. Absent is every plan
that is not a planning run at all -- a delivery checklist, a CLI write, a
record written before this field existed -- and reading absence as "not
accepted" would put this gate in front of every edit anyone makes with a todo
list open.

Where there is nobody to ask
----------------------------
Withheld entirely, and this is the half that decides whether the feature is
shippable. `request_tool_approval` documents itself as failing CLOSED: cron
honours `approvals.cron_mode`, and "any OTHER non-interactive non-gateway
context fails CLOSED". So on a surface with no person, escalating does not ask
a question -- it blocks the call, with the host's wording, for a rule nobody
can answer. `hermes chat -q`, the three programmatic platforms, and a
delegated child all read as unattended, and each of them would simply stop.

`escalation_can_reach_a_person` is the same predicate the repeat guard's stage
two already uses, and the fallback is deliberately NOT the repeat guard's. The
repeat guard drops to `block`, because a loop that keeps running is worse than
a loop that is stopped. Here the unguarded case is a session doing ordinary
work, so the fallback is silence: a person running unattended is the person
who least wants a gate they cannot answer, and refusing their edits would make
this feature the reason automation stopped.

Cron is the stated exception, for a reason no code here can fix: a plugin
cannot see `HERMES_CRON_SESSION`, which is a ContextVar and never a process
variable (`hooks/session_attendance`). A cron turn reads as attended. What it
costs is bounded -- at the `approvals.cron_mode` default of `deny` the edit is
refused with the host's wording, and only a cron job that both stamps
`awaiting_acceptance` and then edits files can reach it at all.

Cost
----
One read of one JSON file, plus the session resolution `read_omh_todo` does,
on `write_file` and `patch` only. The tool-name test is first so every other
call pays a set membership, and the attendance predicate is two in-memory maps
ahead of any I/O.
"""

from __future__ import annotations

from typing import Final

from .engagement_nudges import FILE_MUTATING_TOOLS
from .runtime_reader import read_omh_todo
from .todo_store import PLAN_STAGE_AWAITING_ACCEPTANCE

# No schema version is declared here, deliberately. This module's only output
# is a `pre_tool_call` directive, whose shape is the HOST's
# (`_get_pre_tool_call_directive_details` reads `action`, `message` and
# `rule_key` and ignores every other key), so a version constant would name a
# contract OMH does not own and nothing could read.

# The `[a]lways` allowlist grain, set explicitly -- and the reason is not the
# one `request_tool_approval` documents for itself. That function does derive
# a key from a hash of the reason when it is given none, but a plugin
# directive never reaches that branch:
# `hermes_cli/plugins.py::_resolve_block_from_details` calls it as
# `rule_key=details.rule_key or tool_name`. So an omitted key here is not a
# message hash. It is the TOOL NAME, and what a person's `[a]lways` would
# store is `plugin_rule:write_file`.
#
# That is the grain to avoid, for two reasons of different sizes. It is one
# key per tool, so answering `[a]lways` at a `write_file` prompt leaves the
# next `patch` asking the same question. And it is named after the tool
# rather than after the rule, so it is shared with every other OMH rule that
# ever escalates `write_file`: a person settling THIS gate once would be
# permanently allowlisting a tool for rules that do not exist yet.
#
# Constant, and the alternatives were weighed rather than skipped. The person
# is answering one question -- may this session edit before the plan is
# accepted -- so the grain is the gate itself, and both file-mutating tools
# share it because one answer settles both. A key carrying the session would
# bound an `[a]lways` to the run it was asked about, but `[a]lways` writes
# through `approve_permanent` into the profile's own `command_allowlist` in
# config.yaml, so that key would leave one dead entry there per session
# anybody answered in; it would also be indistinguishable from `[s]ession`,
# which the person can already choose. A key carrying the file or a digest of
# the plan re-prompts somebody who has already answered. So: one line in
# their config, visible, removable, and meaning what it says.
PLAN_STAGE_RULE_KEY: Final = "omh_plan_stage_edit"

# Person-facing, and only person-facing. An `approve` directive is resolved by
# the host's gate and never reaches the model as text
# (`hermes_cli/plugins.py::_resolve_block_from_details`), and on a denial the
# model is handed the host's own refusal rather than this. So there is no
# instruction here -- the writer-side rule lives in the `omh_todo` tool
# schema, where the model actually reads it -- and no restatement of what OMH
# is. Two sentences: what the records say, and what each answer does.
PLAN_STAGE_APPROVAL_MESSAGE: Final = (
    "this session declared a plan and has not recorded you accepting it, and this "
    "call edits a file. Approving implements now; denying stops at the reviewed plan."
)


def plan_stage_edit_directive(
    *,
    tool_name: object,
    session_id: object = "",
    omh_home: str = "",
    hermes_home: str = "",
    escalation_allowed: bool = True,
) -> dict[str, str] | None:
    """Escalate a file edit made during an unaccepted plan run, or None to allow.

    The conjunction, in the order it is cheapest to refuse: this call is one
    of the host's two file-mutating tools, the host named a session, a person
    could answer a prompt raised here, and this session's OWN plan record is
    established and stamped `awaiting_acceptance`. Anything else is None.

    The session test is now an ordering choice rather than a correctness
    one -- the ownership test below refuses an unnamed call anyway, since a
    record cannot be owned by no session -- so what it buys is that such a
    call never pays a file read. That is the property its test asserts,
    because the verdict alone cannot tell the two arrangements apart.

    Never raises. Hermes logs a hook exception at WARNING and proceeds, so a
    handler that raised would leave a refusal nobody sees and an edit nobody
    asked about; a read that fails is a record this gate cannot classify,
    which is already spelled None.
    """
    if str(tool_name or "") not in FILE_MUTATING_TOOLS:
        return None
    session = str(session_id or "").strip()
    if not session:
        return None
    if not escalation_allowed:
        return None
    try:
        todo = read_omh_todo(omh_home or None, hermes_home or None, session_ref=session)
    except (OSError, ValueError, TypeError, RuntimeError):
        return None
    if not isinstance(todo, dict):
        return None
    # Ownership FIRST, and as its own test rather than folded into the one
    # below, because the two answer different questions and only one of them
    # was ever being asked here. An earlier version of this gate read
    # `established` alone and claimed it excluded "a record belonging to
    # another session". It does not. The renderer's identity rule reads an
    # unanswerable case as BELONGING, on purpose -- a real plan must never be
    # hidden from its owner on missing evidence
    # (`_todo_belongs_to_another_session`) -- so a stamped HOME-WIDE record
    # projects as `established` for every session id that asks. That record
    # is not hypothetical: the chat writer produces it whenever the host
    # names no session, on the same call that carries the stamp
    # (`tools/todo_tool.py`, `host_session_id`). Measured before the fix, one
    # such record armed this gate for three unrelated session ids and stayed
    # silent only for the empty one a test happened to pin.
    #
    # A gate that stops a person's turn needs the opposite default from a
    # renderer: never escalate on a plan it cannot attribute. `own_record` is
    # that fact -- this projection came from the session's OWN path -- and it
    # is taken from the reader rather than re-derived here, because the
    # reference a caller holds is not always the key a record is written
    # under: a created TUI carries the gateway transport id while records are
    # keyed on the durable session key. `_reading_session` translates that
    # before ownership is decided, so comparing our own raw `session_id`
    # against a stored `session_ref` would refuse a record this session
    # really does own. A stamped record with no owner is unattributable by
    # construction, and refusing it loses nothing.
    if not todo.get("own_record"):
        return None
    # Freshness and open work, which is what `established` actually answers:
    # it excludes `absent` (no record), `stale` (too old) and `all_done` (a
    # finished plan), so those three refusals are one test rather than three
    # re-derivations of rules the HUD owns.
    if str(todo.get("status", "")) != "established":
        return None
    if str(todo.get("plan_stage", "")) != PLAN_STAGE_AWAITING_ACCEPTANCE:
        return None
    return {
        "action": "approve",
        "message": PLAN_STAGE_APPROVAL_MESSAGE,
        "rule_key": PLAN_STAGE_RULE_KEY,
    }
