"""A turn that starts remote work and arms nothing is a stall, not a wait.

A Hermes session cannot wake itself. Exactly two things begin a turn: a
message from a person, and the host's background-process completion notice,
which arrives as a synthetic user-role message. So "waiting for CI" is only a
real wait when the turn armed a background process that exits when CI does. A
turn that says it is waiting and then ends has stopped, and CI passing changes
nothing because nothing is listening (#1721).

Measured on the reporter's own session history: of 48 turn endings that
mentioned CI or a deploy and a wait, one had armed a background waiter and was
woken by it. The longest idle gap was 168 minutes, on a turn whose verification
command was blocked at the human approval gate.

What this module refuses to do, and why
---------------------------------------
It never reads the model's prose. The existing `CONTINUATION_CLAIM_PHRASES`
matcher is the repository's cautionary example here: matching a promise by its
wording accuses a model over ordinary narration, and the 2026-09-19 audit said
so. Both triggers below are structured facts.

* "This turn started remote work" is read from the COMMAND the model passed to
  the `terminal` tool -- structured input the model itself supplied to a tool
  with a schema, which is a different thing from its narration. The command
  set is deliberately tiny and anchored on executable plus subcommand tokens
  after a shell-aware `shlex` split, never a substring, so `git push --help`,
  `echo git push`, and a heredoc body containing the words are all misses.
* "Nothing is armed" is two records, not one. Liveness is `processes.json` in
  the Hermes home, which the host writes on spawn, on move-to-finished and on
  kill (`tools/process_registry.py`) and which omits every exited session, so
  presence in it IS liveness. Whether a live process will NOTIFY comes from
  the background spawn's own tool result, because the host sets the
  notification fields a step after it writes the checkpoint; see
  `SPAWN OBSERVATION` below, which is a reproduced defect in the first draft
  of this module and not a theoretical one. Anything the two together cannot
  settle answers "unknown", never "unarmed" -- this module may decline to
  speak but must never accuse a session that did arm something.
* "This command is blocked at the approval gate" is read from the host's own
  result field: `tools/approval.py::_pending_result` returns
  `status="pending_approval"` with `approval_pending=true`. It is a second
  cause with its own sentence and its own per-turn latch, because a command
  waiting on a person did not run at all and no watcher fixes that. Which
  surfaces it can fire on is bounded and stated: that result is returned only
  when there is no gateway notifier and no CLI panel, which is cron, batch,
  and ask mode without a notifier. On an interactive gateway the call blocks
  in `_await_gateway_decision` and a timeout returns `status="blocked"`
  instead, so this cause does NOT cover the interactive approval stall the
  issue measured at 168 minutes, and neither does the end-of-turn sentence
  below, which keys on the unarmed-wait latch alone.

Why this seam and not `pre_verify`
----------------------------------
`pre_verify` is where the issue proposed acting, and it cannot reach the turn
that matters. The host gates it on `_edited` being non-empty
(`agent/turn_stop_gates.py::_pre_verify_nudge`), and `_turn_file_mutation_paths`
is fed only by `_FILE_MUTATING_TOOLS`, which the host's own test pins as
`frozenset({"write_file", "patch"})`. A turn that runs `git push` and
`gh pr create` and then waits has changed no files, so the hook never fires on
it. That is reported rather than worked around.

`transform_tool_result` fires for every tool through
`model_tools.handle_function_call`, carries `tool_name`, `args`, `result`,
`session_id` and `turn_id`, and lands while the model is still inside the turn
loop -- so both exits the directive offers are actually reachable from it. The
directive is delivered on the call that starts the remote work rather than at
the turn's end, which is the earliest point at which the fact is known and the
last point at which the model can still act on it cheaply.

A turn that receives the directive, arms nothing anyway, and ends is caught
once more at `transform_llm_output`, the one host seam after the tool loop:
`honest_stop_output` appends a sentence telling the person the session has
stopped. It keys on the directive's own per-turn latch and a fresh read of
the process record, never on the final response's wording.

The directive drives to neither exit by force. It names arming a watcher and
telling the person plainly, and it says not to poll in the foreground -- a
foreground poll is what OMH's repeat-call guard escalates to a human gate
(#1719), so recommending one would set up the next stall.

Nothing here raises. Hermes wraps the transform in `except Exception` and logs
at debug (`model_tools._apply_transform_tool_result_hook`), so a handler that
raised would leave no trace at all. Every decline is counted by reason
instead, readable through `remote_wait_declines()`.

What the directive costs, and why it says so little
---------------------------------------------------
It rides a tool result, so every character is paid on a real turn. Two rules
keep it small. Every sentence is either a fact the model cannot look up (a
session cannot wake itself) or an action it can take; nothing restates the
situation back at it. And the evidence boundary lives HERE rather than in the
text: the directive is prepared instruction, and emitting one is not evidence
that a watcher was armed, that the remote work succeeded, or that anyone was
told -- but that sentence is for a reviewer reading this file, not for the
model's context on every push, where it is a claim it cannot act on. An audit
on 2026-09-19 measured OMH text at 36% of all `api_content` in the owner's
primary home and named exactly that kind of line.

The rate is the other half of the cost, and it is stated rather than hidden.
At the moment a push returns a session has almost never armed a watcher --
arming is what you do after pushing -- so "nothing armed" is true on very
nearly every successful push, including the turns that were never going to
wait. Two things bound it: the obligation is phrased conditionally, so a turn
that is not going to wait can read it and move on, and `ARMS_WATCHERS_LATCH`
retires the directive for a session that has been observed arming one. The
alternative to firing this often is reading the model's final response for a
promise, which is the thing this module exists not to do.
"""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Final

from . import runtime_paths
from .hooks.nudge_budget import (
    MAX_TRACKED_SESSIONS,
    engagement_count,
    latch_engagement,
    session_is_delegated,
)

REMOTE_WAIT_NUDGE_SCHEMA_VERSION: Final = "omh_remote_wait/v1"
# Its own JSON key, the way `code_mode_guidance` and `engagement_nudges` add
# one: a tool result that parses as a JSON object must keep parsing after this
# module has touched it, so the text is a new field there and is appended only
# when the result is plain text.
REMOTE_WAIT_NUDGE_KEY: Final = "omh_remote_wait"

# The host tool whose args carry a command line. Copied from Hermes rather than
# imported -- the bundle may only import from inside itself.
TERMINAL_TOOL: Final = "terminal"

# Commands that start work on a remote the session will then have to wait on.
# Anchored on executable plus subcommand tokens, and kept this small on
# purpose: every entry widens what a near miss can look like, and the cost of
# omitting one is a stall this module does not catch, while the cost of a wrong
# one is a directive on a turn that was never going to wait.
# Keyed by executable, with the subcommands under it, rather than written as
# `("git", "push")` tuples. The shape is deliberate and the reason is
# INVARIANT 3 in `tests/test_handoff_safety_contract_enforcement.py`: OMH must
# never push, open, review or merge anything on a forge, and that gate proves
# it structurally, by visiting every list or tuple literal in `src/` whose
# FIRST element is a constant program name -- which is how this repository
# spells an argv. Written as argv-shaped tuples, this table made a module that
# RECOGNISES those commands read as one that RUNS them, and the gate was right
# to say so.
#
# This is not a way around the gate. Its own docstring separates an executed
# argv from a parser, and names `src/coding/work_reporting.py`, which tests
# `"gh pr checks" in command_lower` against an executor's transcript, as
# recognition it deliberately does not visit. This table is the same kind of
# thing: it is compared against a command the MODEL already ran and handed to
# the host's `terminal` tool. Putting the executable in a key rather than in
# argv position makes the data say what it is. And the safety property is
# still proved rather than assumed -- just proved directly, by
# `test_this_module_cannot_execute_anything`, which asserts this module
# imports no subprocess or spawn helper at all. The git half of that gate has
# no exception mechanism for a mutating verb in any case: its failure message
# says to remove the command, not to allowlist it.
REMOTE_WORK_COMMANDS: Final[dict[str, tuple[tuple[str, ...], ...]]] = {
    "git": (("push",),),
    "gh": (("pr", "create"), ("pr", "merge"), ("workflow", "run")),
}


def remote_work_anchors() -> tuple[str, ...]:
    """Every anchor as the ``"git push"``-style phrase the directive names."""
    return tuple(
        f"{executable} {' '.join(subcommand)}"
        for executable, subcommands in REMOTE_WORK_COMMANDS.items()
        for subcommand in subcommands
    )

# A segment carrying one of these is asking about the command, or rehearsing
# it, and starts nothing on a remote.
NON_STARTING_TOKENS: Final[frozenset[str]] = frozenset({"--help", "-h", "--dry-run"})

# Shell tokens that end one command and begin another. `shlex` with
# `punctuation_chars` emits these as their own tokens. Redirections are
# deliberately absent: `<<` introduces a heredoc body, whose lines `shlex`
# flattens into ordinary tokens, and treating it as a separator would make
# `cat <<EOF ... git push ... EOF` read as a push.
SEGMENT_SEPARATORS: Final[frozenset[str]] = frozenset(
    {"&&", "||", "|", "|&", ";", ";;", "&", "(", ")"}
)

# Its presence anywhere in a command disables the newline split; see
# `_command_lines` for why the whole command rather than the heredoc's extent.
HEREDOC_OPERATOR: Final = "<<"

_ENV_ASSIGNMENT: Final = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*=")

# The host's process-record filename in the Hermes home
# (`tools/process_registry.py::_checkpoint_path`).
PROCESS_RECORD_FILENAME: Final = "processes.json"

# Per-turn latch keys. Two causes, two latches: an approval gate and an unarmed
# wait are different facts about the same turn, and suppressing the second
# because the first fired would drop exactly the case it was written for.
UNARMED_WAIT_CAUSE: Final = "remote_wait_unarmed"
APPROVAL_GATE_CAUSE: Final = "remote_wait_approval"

# Per-SESSION latch, in the counter map `engagement_nudges` already uses, so
# this adds no state of its own. Set the first time this session is observed
# arming a notifying background process, and once set the unarmed-wait
# directive never fires for that session again.
#
# The rate is why it exists. At the moment a push returns, a session has
# almost never armed a watcher yet -- arming is what you do AFTER pushing --
# so "nothing armed" is true on very nearly every push, including the many
# turns that were never going to wait. The directive is a reminder of a
# mechanism, and a session that has used the mechanism does not need the
# reminder; latching on that turns a per-push line into a per-session one.
#
# It is observed, not inferred, and from the same place the armed check gets
# its evidence: the host's own background-spawn RESULT, never anything the
# model wrote in prose. See `SPAWN OBSERVATION` below for why the result and
# not the process record.
#
# The residual is stated rather than hidden: a session that arms a watcher for
# one piece of work and later pushes a second without arming gets nothing.
# That is a directive not issued, which is the direction this module fails in.
# The end-of-turn sentence (`honest_stop_output`) does not catch it either: it
# speaks only on a turn whose directive was delivered, and it honours the same
# latch, for the reason given there.
ARMS_WATCHERS_LATCH: Final = "remote_wait_arms_watchers"

# ---------------------------------------------------------------------------
# SPAWN OBSERVATION: why the record alone cannot answer "is anything armed".
#
# The host writes the process checkpoint inside `_track_started`, which runs
# during `_spawn()`, and only THEN sets the notification fields on the session
# object (`tools/terminal_tool_background.py`: `proc_session.notify_on_complete
# = True` and `proc_session.watch_patterns = ...` both come after the spawn
# returned). Nothing writes the checkpoint in between. So the most recently
# armed process is ALWAYS recorded in `processes.json` without its flags,
# until some later write -- another spawn, a completion, a kill -- repairs the
# row. A first draft of this module read those flags straight from the record
# and was reproduced answering "nothing armed" for a session that had just
# armed a watcher: exactly the one failure this module must not have.
#
# So the arming is observed where the host states it plainly, in the
# background spawn's own tool RESULT, which carries the process id as
# `session_id` and `notify_on_complete: true` or a non-empty `watch_patterns`.
# That is better evidence than the record and not merely earlier: the host
# zeroes `notify_on_complete` in the result for a delegated child, whose
# notice will not reach its parent, and `_apply_async_support` zeroes it on a
# surface with no async delivery. A row in the record says none of that.
#
# The record keeps the half it is good at. Presence in it is liveness, since
# the host drops every exited session, so "armed" is: a spawn OMH saw arm
# whose process id is STILL in the record. A live row of this session that
# OMH never saw spawn and that carries no flags is not evidence of absence --
# it may be an arming written before the flags landed -- so it answers
# `None`, which declines, rather than `False`, which would accuse.
MAX_TRACKED_SPAWN_SESSIONS: Final = 64
MAX_TRACKED_SPAWNS_PER_SESSION: Final = 32

# session id -> {host process id: whether the host said it will notify}.
_OBSERVED_SPAWNS: "OrderedDict[str, OrderedDict[str, bool]]" = OrderedDict()

# ---------------------------------------------------------------------------
# PER-TURN LATCH. `transform_tool_result` fires once per tool call, so a
# directive keyed only by session would repeat on every call of a turn.
#
# It lives here rather than in `nudge_budget` beside that module's own maps,
# and the reason is the reset obligation `nudge_budget`'s docstring argues
# about. Everything this pass remembers -- declines, observed spawns, and
# this -- is cleared by one function, `reset_remote_wait_state`, which every
# test of this pass already calls. A second home would split that obligation
# across two names for one feature. It also keeps this branch's diff to
# `nudge_budget` empty, which matters while #1739 is adding a map of its own
# at exactly the anchor a shared map would need.
#
# The turn boundary is a host field, not timing: `turn_id` arrives on every
# tool-result hook, and a row whose recorded turn no longer matches is
# replaced rather than aged out.
_TURN_CAUSES: "OrderedDict[str, tuple[str, set[str]]]" = OrderedDict()


def turn_cause_fired(*, session_id: object, turn_id: object, cause: str) -> bool:
    """Whether *cause* already produced a directive in this session's turn.

    A read, never a write: a caller that cannot go on to deliver the
    directive must not consume the turn's one chance to say it.

    An unkeyed session or turn answers ``True`` -- refused rather than
    shared. Two sessions in one row would latch each other's turns, and the
    direction this module fails in is always toward fewer directives.
    """
    key, turn = _tracked_key(session_id), _tracked_key(turn_id)
    if not key or not turn:
        return True
    recorded = _TURN_CAUSES.get(key)
    return bool(recorded) and recorded[0] == turn and cause in recorded[1]


def record_turn_cause(*, session_id: object, turn_id: object, cause: str) -> None:
    """Record that *cause* produced a directive in this session's turn."""
    key, turn = _tracked_key(session_id), _tracked_key(turn_id)
    if not key or not turn:
        return
    recorded = _TURN_CAUSES.pop(key, None)
    causes = recorded[1] if recorded and recorded[0] == turn else set()
    causes.add(cause)
    _TURN_CAUSES[key] = (turn, causes)
    while len(_TURN_CAUSES) > MAX_TRACKED_SESSIONS:
        # Oldest first, and losing a row costs a session the per-turn
        # suppression, never an extra directive it was spared.
        _ = _TURN_CAUSES.popitem(last=False)


def _tracked_key(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""

# Both texts are written to the rule in the module docstring: every sentence
# is either a fact the model cannot look up or an action it can take. The
# obligation is phrased conditionally because at the moment a push returns the
# model has not yet decided whether to wait, and a directive that assumes it
# will is wrong about most of the turns it lands on.
#
# The unarmed-wait text says "no background process ... will wake it", not
# "nothing will wake it", and the narrowing is load bearing. A background
# process is all this pass can see. An async delegation ALSO wakes a session,
# through the same completion queue but with no `processes.json` row at all
# (`tools/async_delegation.py` puts the event straight on
# `process_registry.completion_queue`), and the owner's history holds 405 such
# wakes. The advice survives that -- a delegation finishing tells the model
# nothing about CI -- but the flat claim would not, so it is not made.
UNARMED_WAIT_TEXT: Final = (
    "[OMH unarmed wait] {command} started remote work, and no background process this "
    "session started will wake it when that finishes. A session resumes only when a "
    "person writes to it or the host delivers a completion notice. If this turn will "
    "wait on that work, either arm the wait now (terminal, background=true, "
    "notify=true, e.g. "
    "`gh pr checks <number> --watch`) or say plainly that the session has stopped and "
    "what to come back for. Do not poll in the foreground."
)
APPROVAL_GATE_TEXT: Final = (
    "[OMH approval gate] This command did not run: it is waiting for a person to "
    "approve it, and the turn cannot approve it or wake itself once it ends. If you "
    "end the turn here, say that as your last words rather than reporting progress."
)

# Why a decline was not a directive. Counted rather than raised, because the
# host swallows exceptions from this seam: without this, a module that broke
# would be indistinguishable from a session that had nothing to say.
_declines: "Counter[str]" = Counter()


# The end-of-turn half (#1721). The directive above rides a tool result while
# the model can still act on it; this sentence is for the turn that received
# the directive, armed nothing anyway, and ended. It is written to the PERSON
# reading the final response, and it states only what the records show: this
# turn started remote work, nothing is armed, and a session resumes only on a
# message. It deliberately does not say the remote work is still running --
# a foreground poll may already have seen it finish -- and it does not say the
# model claimed to be waiting, because nothing here reads what the model said.
#
# The sentence is in the reply's own language (owner decision, 2026-09-28) and
# carries no OMH vocabulary: no `[OMH]` head, no "armed". The language comes
# from the response text alone -- see `reply_language` -- and never from the OS
# locale. The copy is vendored here because this bundle cannot import `omh.*`.
HONEST_STOP_TEXTS: Final[dict[str, str]] = {
    "en": (
        "This session has stopped: the remote work it started will not wake it "
        "when it finishes, so send a message to pick it back up."
    ),
    "ko": (
        "원격 작업이 끝나도 이 세션은 자동으로 다시 시작되지 않습니다. "
        "이어서 하려면 메시지를 보내 주세요."
    ),
    "ja": (
        "リモートの作業が終わっても、このセッションは自動では再開しません。"
        "続けるには、メッセージを送ってください。"
    ),
    "zh": "远程工作完成后，此会话不会自动恢复。如需继续，请发送一条消息。",
}
HONEST_STOP_TEXT: Final = HONEST_STOP_TEXTS["en"]

# When a reply outside code is written in a non-Latin script rather than
# quoting one. A script counts only through a run of at least
# `REPLY_SCRIPT_MIN_RUN` of its characters in a row, so "CI 대기 중" is Korean
# and one stray syllable is nothing. Then its characters have to number at
# least the Latin words beside them -- words, not letters, so "Kubernetes
# deployment rollout status 확인 완료" is not outweighed by the letters of four
# product words. Two kinds of token are dropped whole before either side is
# counted, because neither says what language the reply is written in:
# path- and identifier-shaped tokens (`src/x.py`, `reply_language`,
# `REPLY_SCRIPT_MIN_RUN`, `GitHub`, `설정파일.md`), which a coding reply is full
# of, and quotations ("...", “...”, 「...」, 『...』), which are someone else's
# words: an English reply quoting a Korean sentence stays English, and a
# Korean reply quoting an English error message stays Korean.
REPLY_SCRIPT_MIN_RUN: Final = 2

# A fence closes only on a line of the same character at least as long as the
# opening one, so a ```` block that shows a ``` block is removed whole.
_FENCED_CODE = re.compile(r"(?ms)^[ \t]*((`|~)\2{2,})[^\n]*$.*?(?:^[ \t]*\1\2*[ \t]*$|\Z)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LATIN_WORDS = re.compile(r"[A-Za-z]+")
# A token with `/`, `.` or `_` between two word characters, or a lowercase
# letter directly before an uppercase one, is a path, a file name, an
# identifier or a camel-cased name. A sentence-ending period has no word
# character after it, so "완료." and "done." stay prose.
_NAME_TOKEN = re.compile(r"\S*(?:\w[/._]\w|[a-z][A-Z])\S*")
_HANGUL = "ᄀ-ᇿ㄰-㆏가-힣"
_KANA = "぀-ゟ゠-ヿㇰ-ㇿｦ-ﾟ"
_HAN = "㐀-䶿一-鿿豈-﫿"
_CJK_CHARS = re.compile(f"[{_HANGUL}{_KANA}{_HAN}]")
_QUOTATION = re.compile(r'"[^"\n]*"|“[^”\n]*”|「[^」\n]*」|『[^』\n]*』')
_SCRIPT_RUNS: Final = tuple(
    (language, re.compile(f"[{chars}]{{{REPLY_SCRIPT_MIN_RUN},}}"))
    for language, chars in (("ko", _HANGUL), ("ja", _KANA), ("zh", _HAN))
)


def reply_language(text: object) -> str:
    """The language of *text*: ``ko``, ``ja``, ``zh`` or ``en``.

    Fenced and inline code are removed first, because a Korean string in a
    code block says nothing about the language the reply is written in, and
    so are quotations and path- or identifier-shaped tokens. Then the reply
    is English unless its CJK characters number at least its Latin words. A
    non-Latin reply takes the first script in precedence order -- Hangul,
    then kana, then Han -- that has a run long enough to count, not the
    majority script: a Japanese reply is often mostly kanji, so a majority
    rule would call it Chinese.

    The stated limit: a Japanese reply with no run of two kana (all kanji, or
    kanji with single-kana particles) reads as Chinese. Nothing in the text
    tells the two apart without a dictionary. Anything else, including a
    script with no copy here, is English.
    """
    prose = _INLINE_CODE.sub("", _FENCED_CODE.sub("", str(text or "")))
    prose = _NAME_TOKEN.sub(" ", _QUOTATION.sub(" ", prose))
    if len(_CJK_CHARS.findall(prose)) < len(_LATIN_WORDS.findall(prose)):
        return "en"
    for language, run in _SCRIPT_RUNS:
        if run.search(prose):
            return language
    return "en"


def remote_wait_declines() -> dict[str, int]:
    """A copy of the decline tally, by reason. Diagnostics, never a gate."""
    return dict(_declines)


def reset_remote_wait_state() -> None:
    """Test seam: forget everything this pass remembers.

    One function for both, for the reason `reset_nudge_budget` gives about
    its own maps: this state is process-global, the shard planner reorders
    tests run to run, and an obligation spread over two names is one a new
    test file can half-remember. Every test that touches this pass clears it
    through here.
    """
    _declines.clear()
    _OBSERVED_SPAWNS.clear()
    _TURN_CAUSES.clear()


def observed_spawns(session_id: str) -> dict[str, bool]:
    """What OMH saw this session spawn. Diagnostics and tests, never a gate."""
    return dict(_OBSERVED_SPAWNS.get(session_id) or {})


def annotate_remote_wait(
    *,
    tool_name: object,
    args: object,
    result: object,
    session_id: str = "",
    turn_id: str = "",
    hermes_home: str = "",
) -> str | None:
    """Return *result* carrying the directive, or ``None`` to pass it through.

    Fail-open by seam contract and by construction: every path that is not a
    clean, latched, record-backed directive returns ``None`` after recording
    why.
    """
    try:
        return _annotate(
            tool_name=tool_name,
            args=args,
            result=result,
            session_id=session_id,
            turn_id=turn_id,
            hermes_home=hermes_home,
        )
    except Exception as exc:  # noqa: BLE001 - see module docstring: the host
        # swallows and debug-logs anything this raises, so a raise here is a
        # silent disappearance. The failure is recorded by type and the tool
        # result passes through unchanged.
        _declines[f"error:{type(exc).__name__}"] += 1
        return None


def _annotate(
    *,
    tool_name: object,
    args: object,
    result: object,
    session_id: str,
    turn_id: str,
    hermes_home: str,
) -> str | None:
    if str(tool_name or "") != TERMINAL_TOOL:
        _declines["tool_not_terminal"] += 1
        return None
    session = str(session_id or "").strip()
    turn = str(turn_id or "").strip()
    if not session:
        # An unkeyed session cannot be latched per turn, and an unlatched
        # directive would repeat on every call of the turn.
        _declines["no_session_id"] += 1
        return None
    if not turn:
        # `turn_id` is the only turn boundary this seam can observe. Without it
        # there is no "once per turn", so there is no directive.
        _declines["no_turn_id"] += 1
        return None

    if session_is_delegated(session):
        # A delegated child must not be told to tell "the person" anything,
        # and the watcher it would arm cannot reach its parent -- the host
        # says so itself, in the note it attaches to that very spawn
        # (`_SUBAGENT_NOTIFY_NOTE`). `engagement_nudges` refuses the same
        # sessions for the same reason, through the same predicate.
        _declines["delegated_session"] += 1
        return None

    # Observed before anything else, so a session that armed a watcher on
    # this very call is recorded before it could also be measured.
    spawned = _observe_background_spawn(session, result)
    if spawned is not None:
        if spawned:
            latch_engagement(session, ARMS_WATCHERS_LATCH)
        _declines["background_spawn_observed"] += 1
        return None

    pending: list[tuple[str, str]] = []
    if _approval_pending(result):
        _consider(pending, session=session, turn=turn, cause=APPROVAL_GATE_CAUSE, text=APPROVAL_GATE_TEXT)
    # A command that did not succeed started nothing on a remote, so there is
    # nothing to wait on and nothing to arm. This is what keeps the directive
    # off a push rejected as non-fast-forward, and off a push held at the
    # approval gate -- which never ran, and whose own sentence above is the
    # true thing to say about it.
    started = _started_remote_work(args) if _command_succeeded(result) else ""
    if started:
        _consider_unarmed_wait(
            pending, session=session, turn=turn, started=started, hermes_home=hermes_home
        )
    if not pending:
        return None

    carried = _carry(result, "\n\n".join(text for _, text in pending))
    if carried is None:
        _declines["result_not_carryable"] += 1
        return None
    for cause, _text in pending:
        record_turn_cause(session_id=session, turn_id=turn, cause=cause)
    return carried


def _consider(
    pending: list[tuple[str, str]], *, session: str, turn: str, cause: str, text: str
) -> None:
    """Queue one sentence unless this turn already spent that cause's latch.

    The latch is only READ here and recorded once the directive has actually
    been carried onto the result, so a result the transform could not carry
    does not silently consume the turn's one chance to say this.
    """
    if turn_cause_fired(session_id=session, turn_id=turn, cause=cause):
        _declines[f"latched:{cause}"] += 1
        return
    pending.append((cause, text))


def _consider_unarmed_wait(
    pending: list[tuple[str, str]], *, session: str, turn: str, started: str, hermes_home: str
) -> None:
    if engagement_count(session, ARMS_WATCHERS_LATCH):
        # This session has armed a notifying watcher before, so it has the
        # mechanism and the reminder has nothing left to tell it.
        _declines["session_arms_watchers"] += 1
        return
    armed = armed_waiter_present(session_id=session, hermes_home=hermes_home)
    if armed is None:
        # Unknown is not unarmed. A session whose record cannot be read may
        # well be watching something, and telling it otherwise is the one
        # failure this module must not have.
        _declines["process_record_unreadable"] += 1
        return
    if armed:
        # The record is the same evidence the args are, so it latches too.
        latch_engagement(session, ARMS_WATCHERS_LATCH)
        _declines["waiter_armed"] += 1
        return
    _consider(
        pending,
        session=session,
        turn=turn,
        cause=UNARMED_WAIT_CAUSE,
        text=UNARMED_WAIT_TEXT.format(command=started),
    )


def honest_stop_output(
    *, response_text: object, session_id: object, turn_id: object, hermes_home: str = ""
) -> str | None:
    """Return *response_text* ending in the honest-stop sentence, or ``None``.

    Hermes fires ``transform_llm_output`` once per turn, after the tool loop,
    with the final response and the turn's own ``turn_id`` -- the same value
    every tool hook of that turn received (`agent/turn_context.py` binds it
    once as ``agent._current_turn_id``). It can replace the text; it cannot
    re-enter the loop, so it can deliver the honest stop and never the
    arm-a-watcher half. Those are the two honest exits the directive named,
    and this is the only one still reachable once the turn has ended.

    The decision reads two records and never ``response_text``:

    * this turn's per-turn latch for ``UNARMED_WAIT_CAUSE``, recorded only when
      the in-loop directive was actually carried onto a result, so it means
      "this turn started remote work while nothing was armed";
    * the process record, re-read now, through the same
      ``armed_waiter_present`` the directive used. Arming a watcher after the
      directive is exactly the exit it asked for, and that turn is left alone.

    A session observed arming a notifying watcher at any point is also left
    alone, even when that watcher has already exited: a process that finished
    during the turn queued a completion notice that will still wake it, and
    the record no longer shows it. That is ``ARMS_WATCHERS_LATCH``, the same
    evidence the directive honours.

    Appended rather than rewritten, because the CLI prints only the suffix of
    a transformed response when the new text starts with the streamed one
    (`cli.py::_post_stream_transform_output`); anything else re-prints the
    whole response under a "transformed" banner. Fail-open like every other
    entry here: any failure is counted and the response passes through.
    """
    try:
        return _honest_stop(
            response_text=response_text,
            session_id=session_id,
            turn_id=turn_id,
            hermes_home=hermes_home,
        )
    except Exception as exc:  # noqa: BLE001 - the host swallows and
        # debug-logs anything a transform raises, so a raise here would be a
        # silent disappearance. Recorded by type; the response passes through.
        _declines[f"error:{type(exc).__name__}"] += 1
        return None


def _honest_stop(
    *, response_text: object, session_id: object, turn_id: object, hermes_home: str
) -> str | None:
    if not isinstance(response_text, str) or not response_text.strip():
        _declines["end_of_turn_no_response"] += 1
        return None
    session, turn = _tracked_key(session_id), _tracked_key(turn_id)
    if not session or not turn:
        # `turn_cause_fired` answers True for an unkeyed pair, which is the
        # right answer for suppressing a repeat and the wrong one here.
        _declines["end_of_turn_unkeyed"] += 1
        return None
    if not turn_cause_fired(session_id=session, turn_id=turn, cause=UNARMED_WAIT_CAUSE):
        _declines["end_of_turn_no_unarmed_wait"] += 1
        return None
    if engagement_count(session, ARMS_WATCHERS_LATCH):
        _declines["end_of_turn_session_arms_watchers"] += 1
        return None
    armed = armed_waiter_present(session_id=session, hermes_home=hermes_home)
    if armed is None:
        _declines["end_of_turn_record_unreadable"] += 1
        return None
    if armed:
        _declines["end_of_turn_waiter_armed"] += 1
        return None
    return f"{response_text}\n\n{HONEST_STOP_TEXTS[reply_language(response_text)]}"


def _observe_background_spawn(session: str, result: object) -> bool | None:
    """Record a background spawn from its result; ``None`` if this is not one.

    Returns whether the host said the new process will deliver a notice.

    The result is the host's own statement about what it just started: the
    background path returns the process id as ``session_id`` alongside
    ``notify_on_complete`` / ``watch_patterns``, and the foreground path
    returns no ``session_id`` at all, so the shape identifies itself. It is
    deliberately NOT read from the call's arguments: the advertised parameter
    is `notify`, `notify_on_complete` is an unadvertised legacy alias, and
    reading either would make this depend on which spelling the model chose.
    The result carries the host's answer whichever went in.
    """
    parsed = _parsed_result(result)
    if parsed is None:
        return None
    process_id = str(parsed.get("session_id") or "")
    if not process_id or parsed.get("exit_code") != 0:
        return None
    patterns = parsed.get("watch_patterns")
    notifies = parsed.get("notify_on_complete") is True or (
        isinstance(patterns, list) and bool(patterns)
    )
    _remember_spawn(session, process_id, notifies)
    return notifies


def _remember_spawn(session: str, process_id: str, notifies: bool) -> None:
    spawns = _OBSERVED_SPAWNS.pop(session, None) or OrderedDict()
    _ = spawns.pop(process_id, None)
    spawns[process_id] = notifies
    while len(spawns) > MAX_TRACKED_SPAWNS_PER_SESSION:
        _ = spawns.popitem(last=False)
    _OBSERVED_SPAWNS[session] = spawns
    while len(_OBSERVED_SPAWNS) > MAX_TRACKED_SPAWN_SESSIONS:
        # Oldest first, and eviction fails toward DECLINING rather than
        # accusing: a session whose spawns are forgotten has live rows this
        # module can no longer explain, and an unexplained live row answers
        # `None`.
        _ = _OBSERVED_SPAWNS.popitem(last=False)


def _started_remote_work(args: object) -> str:
    """The anchored remote-work command this call started, or ``""``.

    A call already made with ``background=true`` started something the host is
    tracking, so it is not the unarmed shape this looks for -- whether it will
    WAKE the session is the process record's question, asked separately.
    """
    if not isinstance(args, dict):
        _declines["args_not_mapping"] += 1
        return ""
    if args.get("background") is True:
        _declines["call_was_backgrounded"] += 1
        return ""
    matched = remote_work_command(args.get("command"))
    if not matched:
        _declines["command_not_remote_work"] += 1
    return matched


def remote_work_command(command: object) -> str:
    """The ``"git push"``-style anchor this command line starts, or ``""``.

    Every line and every segment of it is checked, because a multi-line
    terminal block and `cd repo && git push` are both one command that starts
    a push. Each segment is matched on its leading tokens after leading
    environment assignments, so `echo git push` and a heredoc body are misses,
    having a different executable at their head.
    """
    if not isinstance(command, str):
        return ""
    for line in _command_lines(command):
        for segment in _segments(_shell_tokens(line)):
            head = _strip_env_assignments(segment)
            if not head or NON_STARTING_TOKENS.intersection(segment):
                continue
            for subcommand in REMOTE_WORK_COMMANDS.get(head[0], ()):
                if tuple(head[1 : 1 + len(subcommand)]) == subcommand:
                    return f"{head[0]} {' '.join(subcommand)}"
    return ""


def _command_lines(command: str) -> list[str]:
    """What gets tokenized: the whole command, and then each of its lines.

    Both, and the "both" corrects a measured regression. A multi-line block
    is an ordinary shape here -- `git add`, `git commit` and `git push` on
    three lines is one command to the tool -- and `shlex` consumes a newline
    as plain whitespace, so the push sits behind `git add` at the segment's
    head unless the split happens before tokenizing.

    But splitting ALONE loses far more than it finds. A quoted string that
    spans a line boundary leaves every affected line unbalanced,
    `_shell_tokens` answers nothing for each, and the command misses
    completely: `git commit -m "subject\\n\\nbody" && git push` is the shape,
    and it is the shape this repository actually uses. Measured over the
    owner's 10,098 `terminal` calls, the whole-command pass alone anchored
    81 and the per-line pass alone anchored 40. Keeping both anchors 87: the
    original 81 plus the six genuine multi-line gains.

    A heredoc is what makes the per-line pass unsafe, because its BODY is
    lines too: splitting `cat <<EOF / git push / EOF` would read a documented
    command as an executed one. Rather than track heredoc state -- the clever
    string parsing this repository does not allow -- the operator's presence
    anywhere drops the per-line pass. The whole-command pass stays, since it
    never matched a heredoc body to begin with; what is given up is only a
    push on its own line inside such a command, a directive not issued, which
    is the direction this module fails in.
    """
    if HEREDOC_OPERATOR in command:
        return [command]
    return [command, *command.splitlines()]


def _shell_tokens(command: object) -> list[str]:
    """Shell-aware tokens, or ``[]`` for anything that will not tokenize.

    ``punctuation_chars`` is what makes `git push;echo hi` two segments rather
    than one token `push;echo`; plain ``shlex.split`` would miss it.
    """
    if not isinstance(command, str) or not command.strip():
        return []
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        # An unbalanced quote is a command line this module cannot read, and
        # guessing at one is exactly the substring matching it exists to avoid.
        return []


def _segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in SEGMENT_SEPARATORS:
            segments.append([])
            continue
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _strip_env_assignments(segment: list[str]) -> list[str]:
    index = 0
    while index < len(segment) and _ENV_ASSIGNMENT.match(segment[index]):
        index += 1
    return segment[index:]


def armed_waiter_present(*, session_id: str, hermes_home: str = "") -> bool | None:
    """Whether a live background process will wake this session.

    Three answers, and the difference between the last two is the whole
    point. ``True``: something armed is still running. ``False``: nothing is,
    and this module can account for every live process of this session.
    ``None``: it cannot tell, which must never be treated as ``False``.

    Liveness comes from the record. The host writes `processes.json` on spawn,
    on move-to-finished and on kill, and skips every exited session, so an
    entry's presence is its liveness. The file is process-global across
    profiles, so `parent_session_id` -- the host's own name for "session-db id
    of the spawning conversation" -- is what ties an entry to this session; a
    watcher armed by a different session is a different session's.

    Whether a process NOTIFIES comes from the spawn OMH observed, because the
    record's flags are written a step too late; see `SPAWN OBSERVATION`. The
    row's own flags are still honoured when set, since a later checkpoint
    write repairs them and a repaired row is good evidence. What is left over
    -- a live row of this session with no flags that OMH never saw spawn --
    is the case the record cannot settle, and it answers ``None``.
    """
    entries = _process_record(hermes_home)
    if entries is None:
        return None
    observed = _OBSERVED_SPAWNS.get(session_id) or {}
    unexplained = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("parent_session_id") or "") != session_id:
            continue
        process_id = str(entry.get("session_id") or "")
        if observed.get(process_id) is True:
            return True
        if _entry_notifies(entry):
            return True
        if process_id not in observed:
            unexplained = True
    return None if unexplained else False


def _entry_notifies(entry: dict[str, Any]) -> bool:
    """Whether a process-record row's own flags say it will deliver a notice."""
    if entry.get("notify_on_complete") is True:
        return True
    patterns = entry.get("watch_patterns")
    return isinstance(patterns, list) and bool(patterns)


def _process_record(hermes_home: str) -> list[Any] | None:
    """The host's live-process entries, or ``None`` when they cannot be read."""
    try:
        home = runtime_paths.plugin_home(hermes_home or None, hermes=True)
        raw = (Path(home) / PROCESS_RECORD_FILENAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        # A host that has never spawned a background process writes no file.
        # That is a readable answer, and the answer is "nothing is armed".
        return []
    except (OSError, ValueError, RuntimeError):
        # `ValueError` also covers `runtime_paths.RuntimeBindingError`.
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _command_succeeded(result: object) -> bool:
    """Whether the host reported this command finishing cleanly.

    Every `terminal` result carries `exit_code` (`tools/terminal_tool_result.py`
    on the normal path, `_error_json` with -1 on every refusal), so this is one
    host field and not an inference. A result that cannot be parsed cannot
    confirm success, and an unconfirmed command is not one that started remote
    work.

    The one shape worth naming, because an earlier draft of this comment got
    it backwards: `exit_code: None` is NOT a deliberate background launch,
    which reports 0. It is the yield-to-background result
    (`tools/terminal_tool.py`), emitted when a person's message arrives while
    a FOREGROUND command is running, so the `background=true` check above
    never sees it and this is what declines it. That is the right answer --
    a command moved to the background mid-flight has not been observed
    finishing -- and it has one consequence worth recording here. The yield
    path arms `notify_on_complete: true`, and because `_observe_background_spawn`
    also requires `exit_code == 0`, this pass never records that arming. Its
    row therefore reads later as a live process with no flags that OMH never
    saw spawn, which `armed_waiter_present` answers `None` for. It declines
    rather than accusing, which is the direction this module fails in.
    """
    parsed = _parsed_result(result)
    if parsed is None:
        _declines["result_not_parseable"] += 1
        return False
    if parsed.get("exit_code") != 0:
        _declines["command_did_not_succeed"] += 1
        return False
    return True


def _approval_pending(result: object) -> bool:
    """Whether the host reported this call blocked at the human approval gate.

    Both fields are the host's own (`tools/terminal_tool.py::_run_approval_guards`);
    either alone is enough, because a future result that carries only one of
    them still describes a command a person has to approve.
    """
    parsed = _parsed_result(result)
    if parsed is None:
        return False
    return parsed.get("status") == "pending_approval" or parsed.get("approval_pending") is True


def _parsed_result(result: object) -> dict[str, Any] | None:
    if not isinstance(result, str) or not result:
        return None
    try:
        parsed: Any = json.loads(result)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _carry(result: object, text: str) -> str | None:
    """Put *text* on *result* without breaking a host result that is JSON.

    A JSON object gains its own key, the way `engagement_nudges` does, because
    appending to a serialized object makes it unparseable. Anything else is
    plain text and the directive is appended. A JSON value that is not an
    object (a bare list, a number) is declined rather than guessed at.
    """
    if not isinstance(result, str) or not result:
        return None
    try:
        parsed: Any = json.loads(result)
    except (ValueError, TypeError):
        return f"{result}\n\n{text}"
    if not isinstance(parsed, dict):
        return None
    if REMOTE_WAIT_NUDGE_KEY in parsed:
        return None
    parsed[REMOTE_WAIT_NUDGE_KEY] = text
    return json.dumps(parsed)
