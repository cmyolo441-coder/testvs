"""Who opened this turn: a person, or Hermes writing a row for itself.

One question, one answer, read by every surface that has a reason to care.
It started life in `todo_reconciliation`, where the only reader was the plan
line's answer-first rule, and moved here when the route hint became the
second: a plugin deciding whether to route has no business importing a
predicate from a module about plan checklists, and the alternative -- a
second copy -- is how two surfaces come to disagree about whether anybody
spoke.

Both halves are records. That text arrived is a fact about the hook's
arguments; who wrote the row it arrived on is a field Hermes stamps. Neither
half reads what the message SAYS, in any language, and nothing here may start
to: deciding "is this a question", "did they redirect me", or "is this a
notice" from wording is the inference #1549 removed from the plan surface for
being wrong in both directions.

`todo_reconciliation` re-exports all four names, so every import that
predates this module keeps working.
"""

from __future__ import annotations


def turn_opened_by_message(user_message: object) -> bool:
    """Whether an inbound message opened the turn this line is riding.

    The single place that question is asked, and it is an identity, not an
    interpretation. Hermes invokes ``pre_llm_call`` exactly once per turn,
    from ``build_turn_context``, handing it ``original_user_message`` -- the
    message that started the turn, which the host keeps free of its own nudge
    injection. A turn the session drove onward by itself never reaches here:
    ``apply_stop_gates`` appends the ``pre_verify`` directive as a synthetic
    user-role row and re-enters the same turn loop, and no second
    ``pre_llm_call`` fires for it. So there is no message here that OMH wrote,
    and presence is a fact about who the turn owes an answer to.

    Presence is therefore the whole test. What the message SAYS is never read
    and must not be: deciding "is this a question" or "did they redirect me"
    from wording is the inference ``recorded_blocked_reason`` records as
    unfixable in one language let alone four. Two messages with opposite
    meanings produce the identical line.

    Absence keeps today's behaviour, deliberately. A host that passes no
    message, and OMH's own tracker-event path in ``pre_llm_call`` -- which
    zeroes it for a host-labelled GitHub event, an event rather than someone
    writing -- leave this false and the continuation drive exactly as it was.
    A plugin that cannot see whether anyone wrote must not begin claiming they
    did.

    Presence alone is not enough to say a PERSON wrote it; that is
    ``turn_opened_by_person`` below, and this stays the presence half.
    """
    return bool(user_message.strip()) if isinstance(user_message, str) else False


# Which `display_kind` values on a user row mean a person typed it. Taken from
# Hermes, which asks the same question in two places and answers it the same
# way: `split_user_originated_turn` (`agent/context_compressor.py`) returns no
# human-authored view for a user row whose `display_kind` is set to anything
# but this, with the comment "other kinds are synthetic"; and
# `list_recent_user_messages` (`hermes_state_search.py`), which feeds /rewind
# and /undo, filters on `display_kind IS NULL OR display_kind = '' OR
# display_kind = 'steer'` because "bookkeeping rows (display_kind set) are
# excluded" while "a /steer row is typed for the renderer but is human input".
#
# Copied rather than imported, the way `engagement_nudges` copies the host's
# tool-name sets and for the same reason: Hermes is a different repository and
# the bundle may not import from it, so no parity test can exist and the
# source is named here instead. Measured against hermes-agent 577990c3a0.
PERSON_AUTHORED_DISPLAY_KINDS = frozenset({"", "steer"})


def host_synthesized_turn(display_kind: object) -> bool:
    """Whether the host wrote this turn's opening row rather than a person.

    A record field, never wording. Hermes opens a turn for its own rows as
    well as for a person's: a background-process completion
    (`display_kind="process_complete"`), an async delegation batch
    (`"async_delegation_complete"`), a model switch (`"model_switch"`), a
    crash-recovery or auto-continue note (`"auto_continue"`), a diagnostic
    (`"internal_notification"`). Each arrives as a `role="user"` row carrying
    real text, so presence cannot tell them apart -- measured in the owner's
    `state.db`, 282 of 1,869 user rows carry such a kind.

    The typing is available in time. `persist_user_display_kind` exists
    precisely so a synthesized turn is typed when its row is WRITTEN rather
    than when the turn ends (`tests/agent/test_synthetic_turn_display_kind.py`
    in the host states the reason: an untyped row mid-turn "paints the raw
    `[System note: ...]` text as if the user had typed it"). The stamp happens
    in `_stage_turn_user_message` at turn start, before `pre_llm_call` runs.

    An unknown or malformed value reads as synthetic, which is the safe
    direction here and the opposite of most guards in the plan module this
    came from: the cost of calling a person's turn synthetic is one turn
    rendering what it rendered before, while the cost of the other error is
    telling the model to answer a person who does not exist.

    That direction is the predicate's own, and the hook does not inherit it.
    `_turn_display_kind` normalizes a non-string field to absence before this
    is called, so a corrupt value arriving through Hermes reads as a person
    and keeps today's behaviour; the branch below is for a direct caller that
    passes the raw field. Both halves are pinned, in
    `tests/test_injected_context_pressure.py`.
    """
    if display_kind is None:
        return False
    if not isinstance(display_kind, str):
        return True
    return display_kind.strip() not in PERSON_AUTHORED_DISPLAY_KINDS


def turn_opened_by_person(user_message: object, display_kind: object = "") -> bool:
    """Whether a PERSON opened this turn, which is who a request-reader serves.

    Both halves are records: that text arrived, and the host's own typing of
    the row it arrived on. Neither reads what the message says.

    ``display_kind`` defaults to absent so a caller that cannot see the row --
    every caller that predates this parameter -- keeps the behaviour it had.
    A plugin that cannot tell who wrote must not start claiming the host did.
    """
    return turn_opened_by_message(user_message) and not host_synthesized_turn(display_kind)
