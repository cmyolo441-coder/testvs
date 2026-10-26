"""Per-turn reconciliation reminder for an open plan todo.

A model that declares a plan (todo init) and then answers "all done" in
chat while the checklist still shows open items leaves the HUD lying to
the user ('작업 다됐다는데 투두는 이렇게 남아있네'). No keyword trigger
can catch every phrasing of a completion claim, but an OPEN plan is a
state, not a phrasing — so while one exists, every turn's context
carries one compact line that binds completion claims to the checklist.
The reminder is awareness (instruction), never state and never
evidence; it stops the moment the plan is all done or cleared.

While the reader's stall finding stands, the same line also carries how
long the checklist has been unchanged. That covers the other half of the
failure: a plan does not only part ways with the session by being
contradicted, it parts ways by being left behind while the session talks
about something else.

The same turn-shaped gap exists one level out: a dispatched unit ends,
the session reports it, and the turn ends without the result being
verified or the plan updated. So the reminder also carries any finished
dispatch the plan has not written down (`dispatch_outcomes`), each with
the verb it owes, plus `DISPATCH_COMPLETION_RULE`. Same boundary: the
lines are instruction and a pointer at a record, never evidence that a
unit did anything.

Those two halves both used to claim the turn, so the dispatch rule has a
second head (`DISPATCH_AFTER_ANSWER_RULE`) that orders it behind the
answer whenever the plan line is the answer-first variant below. One
chain, two heads, and `_open_plan_line` hands the variant back rather
than letting the caller re-derive it.

One more thing is counted rather than repeated. The reconciliation rule
is the only sentence here that fired on every turn for as long as a plan
existed, so it now spends a per-plan turn budget
(`TODO_RECONCILIATION_FULL_TURNS`) and drops to its first clause once
that is out, coming back in full on every write to the plan record. Both
inputs are records -- the plan's `updated_at` and a per-session turn
count in `hooks/nudge_budget` -- and a caller that is not a turn renders
the rule whole.

The per-turn line has one structural limit: it is read at the start of a
turn, so nothing observes a turn that ENDS with open items, and continuation
then waits for the person. `plan_continuation_reading` is the same policy
delivered at the one moment a host offers -- `pre_verify`, whose returned
directive Hermes appends as a synthetic user-role row before re-entering the
SAME turn loop (`agent/turn_stop_gates.py`, `apply_stop_gates`), so the
session is driven onward without anyone being asked for it. Same gate
(`open_plan_position`), same rules, same boundary: it reports what the plan
record says and asserts nothing about what the next turn does. It also hands
back the plan's write stamp, which is how the hook decides whether the run is
still moving and therefore whether more of the host's turn-end budget may be
spent on it (`hooks/nudge_budget.py`).

Both surfaces were right when the session stalled and wrong when the PERSON
steered: "unless something is blocking it, advance the next item" argues with
the person while the session does what they just asked for, and the person is
what is blocking it. A plan-level `deferred_reason` is where that gets
recorded, and `plan_deferral_reason` is the single place either surface asks
whether it holds. The record declares it, the way the item-level block does,
and for the same reason (`recorded_blocked_reason`). Its liveness is decided
in `runtime_reader` against a digest of the items, so resuming the plan lapses
it; a blocked next item still wins over it, being the stronger statement about
why nothing moved.

That field only ever covered the half a model remembers to write down. The
half it does not: someone asks something mid-plan and the drive, sitting in
the context of that very turn, has already said the turn is not discharged by
producing an answer. So the per-turn line reads ONE structural fact about the
turn as well -- whether an inbound message opened it
(`turn_opened_by_message`) -- and carries `TODO_ANSWER_FIRST_RULE` in place of
the drive when one did. That fact is an identity, not an interpretation:
Hermes calls `pre_llm_call` exactly once per turn, from `build_turn_context`
(`agent/turn_context.py`), with the message that started the turn, and a turn
the session drove onward by itself never re-enters that hook at all. What is
still not read is what the message MEANS -- not whether it is a question, not
whether it redirects the work, not its wording in any language. Presence is
the whole test. The turn-end directive reads nothing of the conversation at
all, because its host hands it none: `pre_verify` is called with the plan's
own coordinates and the answer already produced, never with a message.
"""
from __future__ import annotations

from typing import Any

from .dispatch_outcomes import _parse_timestamp, unacknowledged_outcomes
from .todo_evidence import (
    EVIDENCE_CLOSED,
    EVIDENCE_FAILED,
    STORE_READ,
    STORE_UNREADABLE,
    WINDOW_EMPTY,
    WINDOW_HAS_COMMANDS,
    evidence_key,
    item_verdicts,
    valid_evidence,
)
# The per-session turn counter the reconciliation rule's budget spends lives
# with the other bounded per-session maps rather than in a second one here:
# one eviction policy, one key derivation, one reset seam a new test has to
# remember (`reset_nudge_budget`).
from .hooks.nudge_budget import plan_line_turns_on_record
from .hud_text import TODO_UNCHANGED_STATUSES, todo_unchanged_text
from .runtime_reader import DECLARED_TODO_STATUSES, read_omh_todo
# Who opened the turn moved to `turn_authorship` when the route hint became
# its second reader (#1741): a routing surface importing a turn-authorship
# predicate from the plan-checklist module reads wrong, and the alternative
# to moving it is a second copy that can disagree with this one. Re-exported
# here so every import that predates the move keeps working.
from .turn_authorship import (  # noqa: F401 - re-export; see that module's docstring
    PERSON_AUTHORED_DISPLAY_KINDS,
    host_synthesized_turn,
    turn_opened_by_message,
    turn_opened_by_person,
)

try:  # Match a continuation claim on the router's own fold when OMH is installed.
    from omh.routing.visual_qa_cues import contains_cue_phrase as _contains_cue_phrase
except ImportError:  # pragma: no cover - standalone plugin hosts keep the local fold.
    _contains_cue_phrase = None

_MAX_ACTIVE_TEXT_CHARS = 80
# The reminder is one compact block, not a report: three outcome lines plus a
# count of the rest is enough to make the event impossible to miss without
# turning the per-turn context into a dispatch board.
_MAX_OUTCOME_LINES = 3

# The TUI has shown a stopped checklist to the PERSON since the in-flight
# liveness signal landed; the session driving that checklist never saw the
# finding anywhere. A plan that sits with one item active for hours while the
# session answers about other things and ends its turns is the observed
# failure, and it is not a completion claim, so the rule below cannot catch
# it. This sentence states what the reader observed and asks for one sentence
# of accounting. It rides a turn that is already happening and starts
# nothing, which is exactly why it is a context line and not a driver.
#
# It asks for the accounting and stops there. It used to end "and if nothing
# is blocking it, move it", which is `TODO_CONTINUATION_RULE`'s own drive
# repeated about forty characters after that rule says it -- this variant
# renders both, always, in one block. The drive is not weakened by dropping
# the echo: the sentence asking for the next item is still in the same line,
# in its stronger record-backed form.
TODO_UNCHANGED_RULE = (
    "The checklist standing still is an observation, not evidence that the "
    "work failed: say where the plan stands."
)

# The half that was missing. The reconciliation rule below guards a COMPLETION
# CLAIM -- it fires when the model says it is done while items are open. It has
# nothing to say about the far more common way a plan dies: the model answers
# whatever arrived, reports, and ends the turn with the list untouched. A real
# run spent 36 minutes that way, and an earlier one ended with 3 of 5 phases
# pending while every notification got a courteous status reply.
#
# `omh_todo` exists to carry a goal ACROSS turns. A checklist that only ever
# catches a contradiction is a detector, not a plan. So this line states the
# obvious thing nobody was saying: open items mean the work is not finished.
#
# The termination criterion is explicit and it is what keeps this bounded --
# this is a long loop with a stop condition, not an unbounded one. It ends when
# every item is done, or when an item carries a `blocked_reason`. It does not
# end because a turn happened to produce a paragraph.
#
# Two sentences, not four, and the join is where the characters went. The
# premise and the drive were separate sentences saying one thing, so they are
# one sentence with a colon; and "-- not when a turn has produced an answer"
# said, in the stop criterion, what "rather than ending on a status report"
# already says in the drive one clause earlier. The exclusivity that clause
# carried is now carried by "only", which is the stronger form of it: a stop
# criterion that names its two conditions and says they are the only ones
# rules out an answer without having to list it.
TODO_CONTINUATION_RULE = (
    "Open items mean this plan is not finished: unless something is blocking "
    "it, advance the next item in this turn rather than ending on a status "
    "report. It stops only when every item is done or an item carries an "
    "omh_todo blocked_reason."
)

# What the turn-end directive adds to the rule above. The host appends the
# message as a synthetic user-role row, so it has to say what it is: a read of
# the plan record, never a claim that an item ran.
PLAN_CONTINUATION_BOUNDARY = (
    "This directive reports what the plan record says; it is not evidence that "
    "any item ran, passed, or was verified."
)

# The derived state of an item marked done that no recorded fact closes. It is
# never stored -- the record keeps three states -- because it is a verdict
# over the session's records at read time, and a stored verdict would outlive
# the records it was read from. The name is shared with the fanout repair loop
# and the teammate lane so every surface says one word for it.
DONE_UNVERIFIED = "done_unverified"

# Why a done item is still open, each read off a record (`todo_evidence`):
# commands ran inside the item's window and none is bound to it; the call
# bound to it failed; the store holds no result for it (an unknown id, a call
# from before the window, the wrong tool for the kind, an outcome Hermes
# recorded as unknown, a kind nothing resolves yet); or the store could not be
# read. The last is said and never taken as evidence, and it does not drive
# the loop either: a store OMH cannot read is not something the next turn can
# fix by running a command.
EVIDENCE_REASON_NONE = "no_evidence"
EVIDENCE_REASON_FAILED = "evidence_failed"
EVIDENCE_REASON_UNRESOLVED = "evidence_unresolved"
EVIDENCE_REASON_UNREADABLE = "evidence_unreadable"
EVIDENCE_REASONS = (
    EVIDENCE_REASON_NONE,
    EVIDENCE_REASON_FAILED,
    EVIDENCE_REASON_UNRESOLVED,
    EVIDENCE_REASON_UNREADABLE,
)

# What the plan line adds while a done item is not closed by its record. It
# names the two record moves that close the item and nothing else: a passing
# command followed by marking it done again, or an omh_todo blocked_reason,
# the stop criterion's own escape -- a done item carrying one is closed as
# skipped. Rendered only while such an item exists, so a plan without one pays
# nothing.
#
# Both moves name omh_todo, even on a plan whose steps live in Hermes'
# todo_list: re-sending a completed item there is the list standing still
# (`bind_done_items` keeps a done-to-done item exactly as it was), so only
# omh_todo's advance binds the new command to the item.
TODO_EVIDENCE_RULE = (
    "A done mark closes an item only with a passing command recorded for it: "
    "run the check that shows it, then advance it to done again with omh_todo, "
    "or give it an omh_todo blocked_reason saying why no command can show it."
)

# How the plan line says each reason. The line is read by the model and may be
# relayed to a person, so it carries plain words; the reason codes stay in the
# tool's result payload, where a program reads them.
_EVIDENCE_REASON_PHRASES = {
    EVIDENCE_REASON_NONE: "commands ran but none was recorded for it",
    EVIDENCE_REASON_FAILED: "the command recorded for it failed",
    EVIDENCE_REASON_UNRESOLVED: "no recorded result matches it",
    EVIDENCE_REASON_UNREADABLE: "the session record could not be read",
}

# What the plan line says INSTEAD of the continuation rule while the record
# says the person steered elsewhere. It replaces the whole ask rather than
# softening it: a line that both reports a redirection and asks to advance the
# next item is the argument this field exists to end. The stall half goes with
# it -- a deferred plan standing still is the expected state, not a finding.
TODO_DEFERRED_RULE = (
    "The person redirected this session and the plan records it, so this line "
    "is not asking you to advance the next item. Do what they asked for. The "
    "deferral lapses by itself the moment the item list changes, so resuming "
    "the plan needs no clearing step -- and if an item genuinely cannot "
    "proceed, that is an omh_todo blocked_reason on the item, not this."
)

# What the per-turn line says INSTEAD of the continuation rule on a turn an
# inbound message opened. The reported failure, mid-plan: the person asked why
# something might be a model-level problem, the session agreed with nothing
# checked, pivoted straight back to "how shall we proceed?", and when
# challenged offered to fact-check if asked again. "Not when a turn has
# produced an answer" is the sentence that tells a model answering does not
# discharge the turn, and it is delivered into the context of the very turn
# the question arrived on.
#
# It replaces the whole rule rather than softening it, for the reason
# `TODO_DEFERRED_RULE` records: a line that both serves the person and asks
# for the next item is the argument to avoid. The drive is ordered, not
# dropped -- the last sentence still names both ways the turn ends on the
# plan. That is what keeps this inside the stop-criterion contract instead of
# back at the observe-only reminder, and it is why the rule has to be a
# replacement and still carry a resume: `pre_verify` fires only on a turn that
# changed files, so for every non-coding plan this line is the only thing
# driving at all.
#
# The two branches of that last sentence are ordered record-first, and the
# order is the whole of the fix. The gate above is presence-only by design and
# must stay that way, so `그만해, 오늘은 여기까지`, `stop, forget the plan`,
# `thanks!` and `what time is it in Seoul?` all reach this identical sentence.
# Resume-first therefore made "resume the plan" the DEFAULT answer to a person
# who had just said stop, and the only way out of it was a `deferred_reason`
# write they had not asked for. Record-first keeps both options and the stop
# criterion and changes which one the sentence defaults to. Nothing here reads
# the message: two messages meaning opposite things still produce this exact
# string, which `tests/test_person_turn_precedence.py` pins.
# Measured and left alone. Its four sentences are four distinct jobs -- the
# precedence statement, the ask to investigate rather than defer, the guard
# against agreeing with an unchecked claim, and the record-first resume branch
# -- each written against a different half of one observed failure, and the
# only fold available in it ("do not defer it, and do not ask to be asked
# again" into one clause) is worth nine characters and deletes the phrase
# `tests/test_person_turn_precedence.py` pins against the incident wording.
TODO_ANSWER_FIRST_RULE = (
    "A message started this turn, so answering it completely is this turn's "
    "work, ahead of the next plan item. Investigate what it asks with the "
    "tools you have and answer from what you found: do not defer it, and do "
    "not ask to be asked again. Do not agree with a claim you have not "
    "checked -- check it, or say exactly what checking it would need. Then "
    "either record an omh_todo deferred_reason, if they steered the work "
    "elsewhere, or resume the plan."
)

# The instruction half of the reconciliation rule, split out because it is the
# half that survives the turn budget below. Everything after it is the reason
# for the instruction and the evidence boundary on todo writes -- worth saying
# while a plan is new, not worth restating on every turn for the life of the
# plan (a 40-turn session with one plan repeated the whole rule 40 times).
#
# Where the split sits moved once, and nothing was deleted to move it: "either
# finish the remaining items or say which stay open and why" went from the
# always-on half to the budgeted one, so the full rule still says every word
# it said. It belongs on the budgeted side because it is the only clause here
# that DUPLICATES the drive standing beside it -- `TODO_CONTINUATION_RULE`
# already asks for the next item and already names the two conditions that end
# the plan, in record terms rather than in prose. What is left in the
# always-on half is the standing invariant of the record itself (completed
# items marked done, exactly one active), which nothing else states.
#
# It names no tool. The checklist is in whichever ONE tool holds the steps --
# Hermes' todo_list, mirrored into this record (`native_todo`), or omh_todo --
# and naming omh_todo here asked a todo_list session to write the same list a
# second time on every turn.
TODO_RECONCILIATION_RULE_FIRST_CLAUSE = (
    "Before claiming this work is finished, reconcile the checklist: mark "
    "completed items done and keep exactly one item active."
)

# The evidence boundary stays on the budgeted side rather than moving to every
# turn, and the reason is when it can be acted on. It guards a specific
# misreading -- that writing `done` on an item is proof the item ran -- and
# that misreading is only available to a turn that is WRITING todo updates,
# which is the turn right after the record moved. That is exactly the window
# `TODO_RECONCILIATION_FULL_TURNS` keeps open. A quiet turn in between has no
# todo write to mistake for evidence, and the boundary is also carried at the
# two other moments it can be acted on: `omh_todo`'s own result payload
# (`TODO_CLAIM_BOUNDARY`) and the turn-end directive
# (`PLAN_CONTINUATION_BOUNDARY`).
TODO_RECONCILIATION_RULE = (
    f"{TODO_RECONCILIATION_RULE_FIRST_CLAUSE} Either finish the remaining "
    "items or say which stay open and why. A "
    "completion claim in chat while the HUD checklist shows open items is a "
    "visible contradiction. Todo updates are declarations, never execution "
    "evidence."
)

# How many turns of ONE plan record carry the whole reconciliation rule before
# the line drops to its first clause. Every other injection in this bundle
# latches, caps or decays -- code-mode once per session, engagement nudges at
# two and then permanently, the board card once, the route hint per
# fingerprint, the repeat streak after 300 s, dispatch outcomes after 24 h --
# and the plan line was the one that fired on every turn for as long as a plan
# existed, uncapped.
#
# Three, and the count restarts on every write to the plan record, because the
# rule guards a COMPLETION CLAIM and a completion claim is most likely in the
# turns right after the record moves: the model has just marked something done
# and is deciding whether the whole thing is done. So the full rule rides the
# turns that follow a write and the quiet turns in between carry the
# instruction alone. What is counted is a record -- the plan's own
# `updated_at` and a per-session turn count -- never anything read out of the
# conversation.
TODO_RECONCILIATION_FULL_TURNS = 3

# The chain a finished dispatch owes, shared by both heads below so the two
# can never come to owe different verbs.
_DISPATCH_CHAIN = (
    "verify its result, record the outcome on the plan (done or blocked with "
    "reason), then run the recovery or the next item. Do not announce "
    "continuation you have not started."
)

# Written as an obligation for THIS turn because the failure it replaces was
# structurally polite: a status report, then the turn ended, and the unit's
# result sat unverified while the next item never started.
DISPATCH_COMPLETION_RULE = f"A finished dispatch is an event to act on in this turn: {_DISPATCH_CHAIN}"

# What the dispatch block says instead while `TODO_ANSWER_FIRST_RULE` is the
# plan line. Both claim "this turn" and nothing used to subordinate either to
# the other, so a turn someone opened mid-plan carried two competing answers
# to "what is this turn for". The plan line already says the message is the
# turn's work AHEAD of the next plan item; this says the same ordering about
# the dispatch instead of restating the competing claim. Only the head
# differs -- the chain is the same object.
DISPATCH_AFTER_ANSWER_RULE = (
    "A finished dispatch is the event to act on once that answer is given: "
    f"{_DISPATCH_CHAIN}"
)

# Closing phrasings that promise a next step. Matched only to ask whether the
# step was actually armed -- never to suppress the sentence.
#
# Why this match survives while the accusation it used to carry did not. With
# the finding rewritten to state the record, everything left in it is already
# on the same turn: a stall verdict exists only for an established plan, so
# the plan line is rendered beside it, and an unacknowledged outcome is what
# puts the dispatch lines and `DISPATCH_COMPLETION_RULE` there -- and that
# rule already ends "Do not announce continuation you have not started". Key
# this on the record alone and it becomes a third copy of two sentences the
# model is reading anyway, on every turn, uncapped, which is the shape
# `TODO_RECONCILIATION_FULL_TURNS` exists to stop. The phrase is the only
# thing the finding knows that the two lines beside it do not, so it is what
# earns the line its place.
#
# It is not the inference the owner's rule bars. That rule is about STOP
# criteria read from record fields; this is a start criterion, it reads the
# model's OWN closing text rather than anything the person wrote, and a match
# can only ADD an instruction -- no wording, in any language, can stop a plan
# through this path.
CONTINUATION_CLAIM_PHRASES = (
    "계속 진행",
    "이어서 진행",
    "will continue",
    "continuing",
    "proceeding with",
)

# What the finding says now, and what it no longer says. "The previous turn
# announced a continuation but nothing resumed" and "A promise to continue is
# not a continuation" are verdicts on the model, and the trigger above fires
# on ordinary narration: "continuing to read the router tests" is a sentence
# somebody writes while working, and being told it broke a promise is the
# shape that produces an apology and a pivot instead of the next step. So the
# finding reports what was read and what the record says, and asks. The stop
# criterion is unchanged and still explicit: start it, or say it is stopped
# and why.
CONTINUATION_CLAIM_FINDING_TEMPLATE = (
    "The previous turn's closing text names a continuation, and the record "
    "reads: {facts}. Start the next step now -- verify a finished result, "
    "record it on the plan, then dispatch or advance -- or say plainly that "
    "the work is stopped and why."
)
# Each fact is the record it came from, named as a reading rather than as a
# failure. The stall half is the reader's own verdict (`todo.stall`), not an
# elapsed time this function was given, so it may not claim an interval.
CONTINUATION_CLAIM_STALLED_FACT = "the plan checklist is recorded unchanged"
CONTINUATION_CLAIM_OUTSTANDING_FACT = "{count} finished dispatch{plural} unacknowledged on the plan"


def open_todo_reminder(
    *,
    omh_home: str = "",
    hermes_home: str = "",
    session_ref: str = "",
    outcomes: list[dict[str, Any]] | None = None,
    user_message: str = "",
    turn_display_kind: object = "",
    count_turn: bool = False,
) -> str:
    """The per-turn plan line, plus any dispatch outcome nobody wrote down.

    ``session_ref`` is the session whose turn is starting; its own plan is
    the one a completion claim must reconcile against, never another
    session's. ``outcomes`` lets a caller that already read them (the hook
    also needs the count for its honesty check) hand them in rather than
    making this scan the runtime a second time on the same turn.

    ``user_message`` is the message the host says opened this turn, and the
    only thing taken from it is whether there is one (``turn_opened_by_message``
    says why). ``turn_display_kind`` is the host's own typing of the row that
    message arrived on, which is what separates a person from a background
    notice (``host_synthesized_turn``). Both default to absent so a caller
    that does not know stays on the behaviour it has today.

    ``count_turn`` says this call IS one of the session's turns, which is what
    `TODO_RECONCILIATION_FULL_TURNS` counts. Only `pre_llm_call` can say that
    -- Hermes invokes it exactly once per turn -- so it defaults to false and
    every other caller renders the full rule and records nothing, keeping this
    function a pure read for anyone inspecting a plan out of band.
    """
    lines: list[str] = []
    head, answer_first = _open_plan_line(
        omh_home=omh_home,
        hermes_home=hermes_home,
        session_ref=session_ref,
        user_message=user_message,
        turn_display_kind=turn_display_kind,
        count_turn=count_turn,
    )
    if head:
        lines.append(head)
    lines.extend(
        _dispatch_outcome_lines(
            unacknowledged_outcomes(omh_home, hermes_home, session_ref)
            if outcomes is None
            else outcomes,
            after_answer=answer_first,
        )
    )
    return "\n".join(lines)


def answer_first_turn(
    *,
    user_message: str = "",
    turn_display_kind: object = "",
    omh_home: str = "",
    hermes_home: str = "",
    session_ref: str = "",
) -> bool:
    """Whether this turn's plan line is the answer-first variant.

    The one place the question is asked, so the hook's decision to hold the
    continuation-claim finding back cannot drift from the line that made the
    claim. It re-reads the plan rather than taking a record, because the only
    caller already pays for that read inside `open_todo_reminder` and passing
    a record through would put the same projection in two signatures.
    """
    try:
        todo = read_omh_todo(omh_home or None, hermes_home or None, session_ref=session_ref)
    except _READ_FAILURES:
        return False
    unverified = unverified_done_items(todo, hermes_home=hermes_home, session_ref=session_ref)
    return _answer_first_variant(todo, user_message, turn_display_kind, unverified)


def _answer_first_variant(
    todo: dict[str, Any],
    user_message: str,
    turn_display_kind: object = "",
    unverified: list[dict[str, Any]] | None = None,
) -> bool:
    """The branch test `_open_plan_line` applies, factored out so both read it.

    Order matters and mirrors the rendering below: no open work means no
    plan line at all, a live deferral outranks the message, and a blocked next
    item vetoes the deferral and hands the turn back to the message branch.
    """
    if not isinstance(todo, dict) or open_plan_position(todo, unverified) is None:
        return False
    if plan_deferral_reason(todo) and not recorded_blocked_reason(next_open_item(todo, unverified)):
        return False
    return turn_opened_by_person(user_message, turn_display_kind)


def plan_is_established(todo: dict[str, Any]) -> bool:
    """Whether this plan is declared AND still has work in it.

    The single place the string ``"established"`` is compared.
    """
    return todo.get("status") == "established"


def plan_is_declared(todo: dict[str, Any]) -> bool:
    """Whether a plan record exists for this session right now, open or finished.

    A different question from ``plan_is_established``, and the reason the two
    are separate: the engagement nudge asks whether the model declared a plan,
    and a plan that was declared and then completed still means it did. Reading
    that through ``established`` alone would un-latch the nudge the moment a
    plan finished and start asking for another one -- the projection writes
    ``all_done`` for exactly that state.

    It is still only a snapshot. ``read_omh_todo`` retires a finished plan to
    ``absent`` after ALL_DONE_TODO_LINGER_SECONDS, so this cannot answer "did
    this session ever declare one" and the nudge's own latch remembers that.

    A record field, never a text match -- the rule ``recorded_blocked_reason``
    below carries, for the reason written there: a matcher was deleted from
    this module for getting ordinary strings wrong in both directions.
    """
    return str(todo.get("status", "")) in DECLARED_TODO_STATUSES


def open_plan_position(
    todo: dict[str, Any], unverified: list[dict[str, Any]] | None = None
) -> tuple[int, int] | None:
    """``(closed, total)`` while this plan has open work, else ``None``.

    The single place the question "does this plan have open work" is decided.
    The per-turn context line and the turn-end continuation directive are the
    same policy read at two moments -- the line rides a turn that is already
    happening, the directive starts the next one -- so a second copy of this
    condition would let the two disagree about the same plan.

    ``unverified`` is `unverified_done_items`' answer for this plan. Its
    entries that drive (every reason but an unreadable store) count as open,
    including on a plan the projection calls ``all_done`` because every item
    says done -- exactly the plan this exists to keep going. Omitted, every
    done item closes, the answer this gave before evidence existed.
    """
    pending_evidence = len(_driving(unverified))
    if not plan_is_established(todo) and not (
        todo.get("status") == "all_done" and pending_evidence
    ):
        return None
    counts = todo.get("counts") if isinstance(todo.get("counts"), dict) else {}
    done = counts.get("done")
    total = counts.get("total")
    if not isinstance(done, int) or not isinstance(total, int) or total <= 0:
        return None
    closed = max(0, done - pending_evidence)
    if closed >= total:
        return None
    return closed, total


def _driving(unverified: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [entry for entry in unverified or [] if entry.get("reason") != EVIDENCE_REASON_UNREADABLE]


def next_open_item(
    todo: dict[str, Any], unverified: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The item record a continuation would advance: the active one, else the first pending.

    The record, not its text: the stop criterion below reads a field, and
    text is only ever needed for rendering. A plan with nothing active or
    pending whose done marks are not all closed advances the first of those.
    """
    items = todo.get("items") if isinstance(todo.get("items"), list) else []
    found = _first_item(items, "active") or _first_item(items, "pending")
    if found:
        return found
    first = (_driving(unverified) or [{}])[0].get("item")
    if isinstance(first, int) and 1 <= first <= len(items) and isinstance(items[first - 1], dict):
        return items[first - 1]
    return {}


def unverified_done_items(
    todo: dict[str, Any], *, hermes_home: str = "", session_ref: str = ""
) -> list[dict[str, Any]]:
    """The done items their record does not close, each with its 1-based index and reason.

    Asked only of a plan the stop criterion could end -- established, or all
    done -- and only about done items that OMH bound (a ``done_at``) or that
    carry a reference, and that have no ``blocked_reason``: a done item
    carrying one is closed as skipped, the escape the stop criterion already
    names. An item done with none of the binding fields was marked done
    before evidence existed, by the CLI, or by a hand edit, and counts as done
    exactly as it always did -- that is the backward compatibility.

    Each judged item is read from records over its own window, from
    ``window_start`` to ``done_at``:

    * a reference closes when its recorded result, inside the window, is a
      success (`todo_evidence`); otherwise ``evidence_failed`` or
      ``evidence_unresolved``. A reference two items share closes only the
      first;
    * no reference and no evidence-capable call inside the window is a
      conversational item and closes, so an item after a command-backed one
      is never held open for good; calls inside the window, each holding
      another item, make it ``no_evidence``;
    * no store at all closes an unreferenced item and leaves a referenced
      one ``evidence_unresolved``; a store that cannot be read makes it
      ``evidence_unreadable``, reported but not driving (`_driving`).

    Never raises; a failure to read is the ``unreadable`` reading.
    """
    if not isinstance(todo, dict) or todo.get("status") not in {"established", "all_done"}:
        return []
    items = todo.get("items") if isinstance(todo.get("items"), list) else []
    candidates: list[tuple[int, dict[str, Any], dict[str, str] | None]] = []
    seen: set[str] = set()
    duplicates: set[int] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or item.get("state") != "done" or recorded_blocked_reason(item):
            continue
        evidence = valid_evidence(item.get("evidence"))
        if evidence is None and not item.get("done_at"):
            continue
        if evidence is not None:
            key = evidence_key(evidence)
            if key in seen:
                duplicates.add(index)
            seen.add(key)
        candidates.append((index, item, evidence))
    if not candidates:
        return []
    # What every item of the plan holds, open ones included, so each judged
    # item can skip the calls that answer for another.
    holders = [
        (position, evidence_key(checked))
        for position, item in enumerate(items, start=1)
        if isinstance(item, dict) and (checked := valid_evidence(item.get("evidence")))
    ]
    queries = [
        {
            "evidence": evidence,
            "from": _epoch(item.get("window_start")),
            "to": _epoch(item.get("done_at")),
            "held": {key for position, key in holders if position != index},
        }
        for index, item, evidence in candidates
    ]
    dropped = todo.get("dropped_bindings") if isinstance(todo.get("dropped_bindings"), list) else []
    orphans = [
        {"evidence": checked, "at": _epoch(entry.get("dropped_at"))}
        for entry in dropped
        if isinstance(entry, dict) and (checked := valid_evidence(entry))
    ]
    try:
        reading = item_verdicts(hermes_home or None, session_ref, queries, orphans)
    except _READ_FAILURES:
        reading = {"store": STORE_UNREADABLE, "verdicts": []}
    store = reading.get("store")
    verdicts = reading.get("verdicts") if isinstance(reading.get("verdicts"), list) else []
    unverified: list[dict[str, Any]] = []
    for position, (index, item, evidence) in enumerate(candidates):
        if store == STORE_UNREADABLE:
            reason = EVIDENCE_REASON_UNREADABLE
        elif store != STORE_READ:
            if evidence is None:
                continue
            reason = EVIDENCE_REASON_UNRESOLVED
        elif index in duplicates:
            reason = EVIDENCE_REASON_UNRESOLVED
        else:
            verdict = verdicts[position] if position < len(verdicts) else ""
            if verdict in {EVIDENCE_CLOSED, WINDOW_EMPTY}:
                continue
            if verdict == WINDOW_HAS_COMMANDS:
                reason = EVIDENCE_REASON_NONE
            elif verdict == EVIDENCE_FAILED:
                reason = EVIDENCE_REASON_FAILED
            else:
                reason = EVIDENCE_REASON_UNRESOLVED
        unverified.append(
            {"item": index, "state": DONE_UNVERIFIED, "text": item_display_text(item), "reason": reason}
        )
    return unverified


def _first_item(items: list[Any], state: str) -> dict[str, Any] | None:
    return next(
        (item for item in items if isinstance(item, dict) and item.get("state") == state),
        None,
    )


def _epoch(value: object) -> float | None:
    parsed = _parse_timestamp(value) if isinstance(value, str) and value else None
    return parsed.timestamp() if parsed else None


def _unverified_clause(unverified: list[dict[str, Any]]) -> str:
    """`` · item N is marked done, but <plain reason>[ (+K more)]``, or ``""``."""
    if not unverified:
        return ""
    first = unverified[0]
    phrase = _EVIDENCE_REASON_PHRASES.get(first["reason"], "no recorded result matches it")
    clause = f" · item {first['item']} is marked done, but {phrase}"
    if len(unverified) > 1:
        clause += f" (+{len(unverified) - 1} more)"
    return clause


def item_display_text(item: dict[str, Any] | None) -> str:
    """One item's text, truncated for display.

    Truncation belongs here and nowhere else. It used to happen on the way
    OUT of the item lookup, which put a display bound in front of a decision:
    the blocked check saw the first 80 characters of a field capped at 200,
    so a genuinely blocked item whose reason sat past the window read as
    open, and an item truncated mid-phrase could read as blocked.
    """
    if not isinstance(item, dict):
        return ""
    return str(item.get("text", "") or "")[:_MAX_ACTIVE_TEXT_CHARS]


def recorded_blocked_reason(item: dict[str, Any] | None) -> str:
    """The reason an item records for not proceeding, or ``""``.

    ``TODO_CONTINUATION_RULE`` says the plan stops when an item is *recorded*
    blocked with its reason, and a record is not a substring. An earlier form
    of this inferred the state from the item text and was wrong in both
    directions on ordinary input: "verify the retry is not blocked on the
    session limit" read as blocked, while "차단됨: 소유자 승인 대기" and
    "waiting on the owner's review" did not. Every marker that would fix the
    second widens the first, so the plan schema owns the state instead
    (``blocked_reason`` in `todo_store`) and this only reads it.

    ANY non-empty reason counts, including one that says nothing is wrong
    ("none", "n/a", "없음"). That is deliberate, and it is the narrow form of
    the question the matcher got wrong: deciding which wordings are REAL
    blocks needs a list of reasons that do not count, and that list cannot be
    completed in one language let alone four -- the same hole that made text
    inference unfixable. So the field's presence is the declaration and the
    writer owns it. The guard against filling it in speculatively is the tool
    description, which says to omit the field and send it only for an item
    that cannot proceed; a plan un-declares by removing it.
    """
    if not isinstance(item, dict):
        return ""
    reason = item.get("blocked_reason", "")
    # A non-string is corruption, not a declaration, and corruption must not
    # stop a plan. `7` and `{"a": 1}` stringify to something truthy, and
    # reading that as a block is the silent stop this surface exists to end;
    # malformed data fails toward continuing. This does not soften the
    # sentinel rule above: "none" counts because a STRING is a declaration and
    # the writer owns it, while `7` is not a declaration in any language.
    return reason.strip() if isinstance(reason, str) else ""


def plan_deferral_reason(todo: dict[str, Any]) -> str:
    """The reason the person steered this plan elsewhere, while it still holds.

    The projection has already decided liveness: `runtime_reader` fills this
    field only while the recorded digest still matches the items it read, so a
    deferral that has lapsed arrives here as absence and nothing downstream has
    to know the difference. That split is deliberate -- the digest comparison
    lives with the items it compares, and this stays the one question both
    surfaces ask.

    A non-string is read as absent, the same call `recorded_blocked_reason`
    makes and for the same reason: a hand-written record can carry anything,
    `7` stringifies to something truthy, and reading corruption as a
    declaration would stop a plan silently. Malformed data fails toward the
    plan continuing.
    """
    if not isinstance(todo, dict):
        return ""
    reason = todo.get("deferred_reason", "")
    return reason.strip() if isinstance(reason, str) else ""


# Everything the runtime read below can raise, enumerated rather than
# described, because the one thing this function may not do is raise into a
# host that swallows exceptions: Hermes wraps the whole `pre_verify` call in
# `except Exception` and logs at debug, so a handler that raised would end the
# turn silently -- the exact symptom the directive exists to fix.
#
# Four bases cover the chain, and two of them are not obvious from the call:
# `RuntimeBindingError` subclasses `ValueError` (an unbindable home), and the
# reader's state-root guard raises a plain `RuntimeError` for a symlinked home
# or a symlink loop, as does `TodoStoreError`. An earlier version of this list
# left `RuntimeError` out while a comment asserted it was complete.
_READ_FAILURES = (OSError, RuntimeError, ValueError, TypeError)


def plan_continuation_reading(
    *, omh_home: str = "", hermes_home: str = "", session_ref: str = ""
) -> tuple[str, str]:
    """The turn-end message for a session whose plan still has open work, and its stamp.

    ``TODO_CONTINUATION_RULE`` is already the right sentence delivered at the
    wrong moment: ``_open_plan_line`` renders it into the context of a turn
    that is already happening, so a turn that ends with open items ends
    anyway. This is that rule at the one moment a host lets a plugin start the
    next turn instead. The message is empty whenever the plan itself says stop
    -- no plan, a finished plan, or a next item carrying a `blocked_reason` --
    so the directive never argues with the plan's own stop criterion.

    The second value is the plan record's own `updated_at` as the reader
    projected it, `""` when the read failed or the record carries no stamp. It
    rides back with the message rather than being fetched by a second read,
    because the caller spends its remaining turn-end budget on whether the plan
    MOVED between two attempts, and a stamp read separately could describe a
    different plan than the message does. It is returned even when the message
    is empty: a turn continued by something else still needs a baseline for the
    next attempt to measure against.
    """
    # Dropping the dispatch lines with the plan is not a second F3:
    # `unacknowledged_outcomes` opens with this same `read_omh_todo` call and
    # needs the record it returns -- status, `updated_at` as the baseline, and
    # the item text that says which outcomes are already named -- so a plan
    # read that fails leaves it nothing to report either. Measured across a
    # control and eight faults: no fault yields a row here.
    try:
        todo = read_omh_todo(omh_home or None, hermes_home or None, session_ref=session_ref)
    except _READ_FAILURES:
        return "", ""
    if not isinstance(todo, dict):
        return "", ""
    # Every write through `todo_store` restamps this, so it moves on any plan
    # edit rather than only on a completed item -- which is the notion of
    # progress the caller wants. Marking the next item active, re-scoping the
    # list, or recording a reason are all a run advancing, and none of them
    # changes `done/total`.
    stamp = todo.get("updated_at", "")
    stamp = stamp if isinstance(stamp, str) else ""
    unverified = unverified_done_items(todo, hermes_home=hermes_home, session_ref=session_ref)
    position = open_plan_position(todo, unverified)
    item = next_open_item(todo, unverified) if position is not None else {}
    lines: list[str] = []
    # The blocked item stops the PLAN line and nothing else. A finished
    # dispatch nobody wrote down is a separate obligation -- it is frequently
    # the thing that unblocks the item -- so gating both on one item's state
    # would bury the event that ends the wait. A live deferral stops the same
    # half on the same terms: the host appends the directive as a synthetic
    # user-role row and re-enters the turn with it, so issuing one while the
    # person is being served would answer them with the plan they just
    # stepped away from.
    #
    # There is no gate here for the message that opened the turn, and it is
    # not an omission. This fires at the END of a turn that already produced
    # the answer it is handed as `final_response`, which is exactly the moment
    # `TODO_ANSWER_FIRST_RULE`'s own last sentence asks the plan to resume --
    # so the two agree rather than contradict. It also could not read one:
    # Hermes calls `pre_verify` with the session, platform, model, coding
    # posture, attempt, final response and changed paths, and no message.
    if position is not None and not recorded_blocked_reason(item) and not plan_deferral_reason(todo):
        done, total = position
        head = f"[OMH plan todo] {done}/{total} done"
        text = item_display_text(item)
        if text:
            head = f"{head} · next: {text}"
        head += _unverified_clause(unverified)
        rule = (
            f"{TODO_CONTINUATION_RULE} {TODO_EVIDENCE_RULE}"
            if _driving(unverified)
            else TODO_CONTINUATION_RULE
        )
        lines.append(f"{head}. {rule}")
    # Read in its own guard, not folded into the one above: the two lines are
    # independent obligations, so a failed outcome read must not take the plan
    # line with it. `unacknowledged_outcomes` says it never raises and swallows
    # its own read, but it expands the home a second time afterwards, and that
    # expansion is outside its guard.
    try:
        outcomes = unacknowledged_outcomes(omh_home, hermes_home, session_ref)
    except _READ_FAILURES:
        outcomes = []
    lines.extend(_dispatch_outcome_lines(outcomes))
    if not lines:
        return "", stamp
    lines.append(PLAN_CONTINUATION_BOUNDARY)
    return "\n".join(lines), stamp


def _open_plan_line(
    *,
    omh_home: str,
    hermes_home: str,
    session_ref: str,
    user_message: str = "",
    turn_display_kind: object = "",
    count_turn: bool = False,
) -> tuple[str, bool]:
    """The plan line, and whether it is the answer-first variant.

    The flag rides back rather than being re-derived by the caller: the
    dispatch block's head depends on which variant this chose, and a second
    copy of the branch test would let the two disagree about the same turn.
    """
    todo = read_omh_todo(omh_home or None, hermes_home or None, session_ref=session_ref)
    unverified = unverified_done_items(todo, hermes_home=hermes_home, session_ref=session_ref)
    position = open_plan_position(todo, unverified)
    if position is None:
        return "", False
    reconciliation = _reconciliation_rule(
        todo, session_ref=session_ref, count_turn=count_turn
    )
    done, total = position
    items = todo.get("items") if isinstance(todo.get("items"), list) else []
    # The context line names the ACTIVE item and stops there; the directive
    # falls back to the first pending one, because it has to say what to
    # advance and a plan between items has no active entry. Deliberate, and
    # pinned as such.
    active = item_display_text(_first_item(items, "active"))
    head = f"[OMH plan todo] {done}/{total} done"
    if active:
        head = f"{head} · active: {active}"
    head += _unverified_clause(unverified)
    # Rides the drive, so it is carried wherever TODO_CONTINUATION_RULE is and
    # nowhere it is not: a deferred plan or an answer-first turn is not asking
    # for the next item, so it is not asking for this item's command either.
    drive = (
        f"{TODO_CONTINUATION_RULE} {TODO_EVIDENCE_RULE}"
        if _driving(unverified)
        else TODO_CONTINUATION_RULE
    )
    # A blocked next item wins over a deferral, so the line it produces is
    # unchanged here: the block is the stronger statement about why the plan is
    # not moving, and reporting a redirection over it would hide the thing the
    # reader has to act on. Only an unblocked plan reads as deferred.
    deferred = plan_deferral_reason(todo)
    if deferred and not recorded_blocked_reason(next_open_item(todo, unverified)):
        # The reason is rendered whole, not truncated to the item display
        # window: it is the entire content of this variant of the line and is
        # already bounded twice, by the write cap and by the reader's
        # projection of it. Cutting a reason mid-clause can invert what the
        # person asked for, which an item's text cannot do.
        return f"{head} · deferred: {deferred}. {TODO_DEFERRED_RULE} {reconciliation}", False
    # Below the deferral and above the drive. The recorded redirection is the
    # more specific statement and says what was asked for, so it keeps the
    # line when both hold; a message arriving now still beats the drive. The
    # stall half goes with the drive for the reason the deferred branch drops
    # it: "if nothing is blocking it, move it" would re-add, in the same
    # breath, the ask this branch exists to subordinate. The reconciliation
    # rule stays, being about a completion claim rather than about who is
    # owed the turn.
    if turn_opened_by_person(user_message, turn_display_kind):
        return f"{head}. {TODO_ANSWER_FIRST_RULE} {reconciliation}", True
    unchanged = todo_unchanged_text(todo)
    if not unchanged:
        return f"{head}. {drive} {reconciliation}", False
    stall = todo.get("stall") if isinstance(todo.get("stall"), dict) else {}
    # "no tool call in flight" is only true of the quiet finding. The busy one
    # is the more interesting reading and saying the wrong one would make the
    # line refutable on its face.
    observed = (
        "unchanged {age} while calls kept running".format(age=unchanged)
        if stall.get("status") == "unchanged_while_busy"
        else "unchanged {age}, no tool call in flight".format(age=unchanged)
    )
    return (
        f"{head} · {observed}. "
        f"{drive} {reconciliation} {TODO_UNCHANGED_RULE}"
    ), False


def _reconciliation_rule(
    todo: dict[str, Any], *, session_ref: str, count_turn: bool
) -> str:
    """The whole reconciliation rule, or its first clause once the budget is spent.

    Two records decide it and nothing else: the plan's own ``updated_at``,
    which restarts the budget every time the plan is written, and a per-session
    count of turns already rendered against that same stamp.

    Every way of not knowing renders the FULL rule -- a caller that is not a
    turn, a session with no usable id, a record with no stamp, an evicted row.
    That is the opposite of the direction `nudge_budget` fails in for its other
    counters, and deliberately: losing a completion-claim guard is worse than
    repeating it, while losing a nudge costs nothing.
    """
    if not count_turn:
        return TODO_RECONCILIATION_RULE
    stamp = todo.get("updated_at", "")
    already = plan_line_turns_on_record(session_ref, stamp if isinstance(stamp, str) else "")
    if already < TODO_RECONCILIATION_FULL_TURNS:
        return TODO_RECONCILIATION_RULE
    return TODO_RECONCILIATION_RULE_FIRST_CLAUSE


def _dispatch_outcome_lines(
    outcomes: list[dict[str, Any]], *, after_answer: bool = False
) -> list[str]:
    if not outcomes:
        return []
    lines = [
        "dispatch {run_ref}/{unit_id} ended {state}; next: {verb}".format(
            run_ref=outcome.get("run_ref", "unknown"),
            unit_id=outcome.get("unit_id", "unknown"),
            state=_outcome_state(outcome),
            verb=outcome.get("next", "verify_result"),
        )
        for outcome in outcomes[:_MAX_OUTCOME_LINES]
    ]
    remaining = len(outcomes) - len(lines)
    if remaining > 0:
        lines.append(f"(+{remaining} more)")
    lines.append(DISPATCH_AFTER_ANSWER_RULE if after_answer else DISPATCH_COMPLETION_RULE)
    return lines


def _outcome_state(outcome: dict[str, Any]) -> str:
    for key in ("unit_state", "failure_kind", "status"):
        value = str(outcome.get(key, "") or "")
        if value:
            return value
    return "unknown"


def continuation_claim_without_resume(
    text: str, *, todo_stall_status: str, unacknowledged: int
) -> str | None:
    """A finding when a turn promises to continue and nothing was started.

    "계속 진행하겠습니다" with no auto-resume armed is the exact closing the
    incident produced. The claim itself is fine; what makes it a finding is
    the state it was made in -- the plan unchanged, or a finished dispatch
    still unacknowledged. Either one is enough, because either one means the
    promised step did not start. Pure: the caller supplies both states, so
    this never reads a file and never decides on its own that a run is stuck.

    The returned sentence names only the facts that actually hold, so a
    finding raised on an outstanding dispatch alone does not also assert that
    the checklist stopped moving.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    # The reader owns which statuses mean "the checklist stopped moving"; a
    # copy of that pair here would let a third one be added there without this
    # guard ever noticing.
    stalled = str(todo_stall_status or "") in TODO_UNCHANGED_STATUSES
    count = unacknowledged if isinstance(unacknowledged, int) else 0
    outstanding = count > 0
    if not stalled and not outstanding:
        return None
    if not _claims_continuation(text):
        return None
    facts: list[str] = []
    if stalled:
        facts.append(CONTINUATION_CLAIM_STALLED_FACT)
    if outstanding:
        facts.append(
            CONTINUATION_CLAIM_OUTSTANDING_FACT.format(
                count=count, plural="" if count == 1 else "es"
            )
        )
    return CONTINUATION_CLAIM_FINDING_TEMPLATE.format(facts=" and ".join(facts))


def _claims_continuation(text: str) -> bool:
    if _contains_cue_phrase is not None:
        return bool(_contains_cue_phrase(text, CONTINUATION_CLAIM_PHRASES))
    folded = text.casefold()
    return any(phrase.casefold() in folded for phrase in CONTINUATION_CLAIM_PHRASES)
