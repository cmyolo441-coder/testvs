"""Name the offset when a refused re-read follows a TRUNCATED read.

The measured failure (#1723, owner session ``20260919_140745_db409e``): a model
that needed line 2318 of a 9,172-line file called ``read_file(path)`` with no
``offset``. The read returned lines 1-1666 with ``truncated: true`` and
``next_offset: 1667``. The model kept calling it with the same arguments. Two
more calls returned the same first window in full; after that the host's read
guards answered, twice with a ``status: unchanged`` stub and then 117 times
with ``BLOCKED: ... the content from your earlier read_file result in this
conversation is still current. Proceed with your task using the information you
already have.``

For a truncated read that last sentence is false in the way that matters. Line
2318 was in no result. The refusal asserts possession of content the read never
returned, and neither the stub nor the refusal mentions ``offset`` -- the only
place it appeared was the ``hint`` field of the successful read, which the
model had already ignored three times.

Neither half is OMH's: the model ignored an explicit ``Use offset=1667 to
continue``, and the host wrote the refusal. What OMH has is the one seam that
sees both halves, so this pass carries the missing fact forward:

1. A ``read_file`` result carrying ``truncated: true`` with a usable
   ``next_offset`` and ``total_lines`` records
   ``(session, path, offset) -> window``.
2. A later ``read_file`` refusal or stub whose own call asked for that same
   ``(path, offset)`` gains one bounded note under
   ``TRUNCATED_READ_RECOVERY_KEY`` naming the region that read returned and the
   offset that continues past it.

The window START is half the key rather than payload, and that is the whole
gate. A truncated read is a statement about ONE window: a read from
``offset=5000`` cut at line 6599 says nothing about lines 1-4999, so a record
keyed on the path alone would annotate a refusal for the FIRST window with a
mid-file read's numbers -- a line range no read returned. Keying on the window
also replaces the literal "the refused call's offset is 1" test with the fact
that test was standing in for, and it costs nothing: a model looping on
``offset=5000`` is the same defect and now gets the same note, with its own
numbers.

Three properties this deliberately does not have:

* **No wording match.** Every decision reads a structured field: ``truncated``,
  ``next_offset`` and ``total_lines`` on the way in; ``already_read``,
  ``status == "unchanged"`` and ``guardrail_refusal`` on the way out. The
  refusal's prose is never inspected, so a host that rewrites it keeps working.
* **No guess at the wanted line.** The note states the recorded window and the
  recorded ``next_offset`` and stops. OMH never saw which line the model was
  after, and inventing one would be a claim it cannot support. The note is
  prepared instruction; emitting it is not evidence that any model read it,
  followed it, or reached the region it names.
* **No claim about results it did not see.** One window is recorded per key,
  never the union of a session's reads, so the note says lines past the stop
  were not in THAT result and stops there. "Not returned in any result" would
  be false the moment a later ``offset=1667`` read returned them, and the
  numbers and the sentences around them are held to the same standard.

**The hook runs before the guardrail suffixes are appended.** Real
results persisted in ``~/.hermes/state.db`` carry a trailing
``\\n\\n[Tool loop warning: ...]`` after the JSON object, which would need
``raw_decode`` and a preserved tail to parse -- but that text is appended
strictly AFTER this seam runs, so it can never arrive here.
``model_tools.handle_function_call`` calls ``_apply_transform_tool_result_hook``
on the raw ``_execute_tool`` result as its last act and returns what the hook
returns; only then does ``agent/tool_executor._commit_tool_result`` call
``AIAgent._append_guardrail_observation`` (``run_agent.py``), which is where
``tool_guardrails.append_toolguard_guidance`` adds that suffix, along with the
identical-call stall notice and the result-reference stub. Plain ``json.loads``
is therefore correct here and ``raw_decode`` would be code for a case that
cannot occur.

Nothing here uses a broad ``except``. The house pattern for this seam's
annotating passes is narrow handlers only (``code_mode_guidance``,
``kanban_readback``); the one broad handler on this path
(``engagement_nudges``) earns it by tallying the failure where a reader can
find it, and this module has no such readout, so a broad catch would be the
silent swallow the policy exists to prevent rather than a guarded one.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Final

# The bound the plan record applies to a session reference, reused so a key
# here and every other per-session map in this bundle describe the same string.
from .todo_store import MAX_TODO_SESSION_REF_CHARS

TRUNCATED_READ_RECOVERY_SCHEMA_VERSION: Final = "omh_truncated_read/v1"
# Its own JSON key, beside `omh_guidance`, `omh_engagement` and `omh_readback`:
# a tool result that parses as a JSON object must keep parsing after this pass.
TRUNCATED_READ_RECOVERY_KEY: Final = "omh_truncated_read"
TRUNCATED_READ_TOOL: Final = "read_file"

# Refusal and stub shapes, by KEY. `tools/file_tools.py` writes all three:
# `_dedup_stub_or_block` emits the `status: unchanged` stub and then the
# `already_read` BLOCK, and the consecutive-read guard emits a second
# `already_read` BLOCK at four reads of one region. `guardrail_refusal` is
# `agent/tool_result_classification.GUARDRAIL_REFUSAL_KEY`, which the host sets
# on both blocks -- newer than the measured session, whose rows carry
# `already_read` alone, so all three are recognised independently.
_ALREADY_READ_KEY: Final = "already_read"
_GUARDRAIL_REFUSAL_KEY: Final = "guardrail_refusal"
_STATUS_KEY: Final = "status"
_STATUS_UNCHANGED: Final = "unchanged"

# Sixty-four sessions, the ceiling every other per-session map in this bundle
# uses (`hooks/nudge_budget.MAX_TRACKED_SESSIONS`), and sixteen windows inside
# one. A host outlives every session it runs, so a map keyed by session id only
# grows; eviction is oldest-first and costs an evicted row its note, never an
# extra one.
MAX_TRACKED_READ_SESSIONS: Final = 64
MAX_TRACKED_WINDOWS_PER_SESSION: Final = 16
# A path is a model-supplied string, so the row COUNT alone does not bound the
# bytes. An over-long one is refused rather than truncated: truncating would
# let two paths sharing a long prefix collide, and a collision here does not
# lose a note, it prints the wrong line numbers.
MAX_RECORDED_PATH_CHARS: Final = 1024

_NOTE_PREFIX: Final = "[OMH truncated read]"

_records_lock = threading.Lock()
# session -> (path, start) -> (start, last_line, total_lines, next_offset).
#
# The START is half the key, not just payload. A truncated read is a statement
# about ONE window: `read_file(path, offset=5000)` cut at line 6599 says
# nothing about lines 1-4999, and a record keyed on the path alone would let a
# refusal for the first window be annotated with a mid-file read's numbers.
# Keying on the window the read actually had is also the whole gate: a refusal
# is annotated only when its own call asked for a window some truncated read
# returned, which is the fact a literal "offset is 1" test was standing in for.
# Both levels are insertion-ordered so eviction can take the oldest.
_records: "OrderedDict[str, OrderedDict[tuple[str, int], tuple[int, int, int, int]]]" = (
    OrderedDict()
)


def annotate_truncated_read_recovery(
    *,
    tool_name: object,
    args: object,
    result: object,
    session_id: str = "",
) -> str | None:
    """Return *result* carrying the recovery note, or ``None`` to pass through.

    Fail-open by seam contract: any tool but ``read_file``, a result that is
    not a JSON object, a call with no usable ``path``, and a refusal whose own
    window was never recorded truncated all pass through untouched. Recording a
    truncated read also returns ``None`` -- that call's result is already
    correct.
    """
    if str(tool_name or "") != TRUNCATED_READ_TOOL:
        return None
    if not isinstance(result, str) or not result:
        return None
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or TRUNCATED_READ_RECOVERY_KEY in parsed:
        return None

    call_args = args if isinstance(args, dict) else {}
    path = call_args.get("path")
    if not isinstance(path, str) or not path or len(path) > MAX_RECORDED_PATH_CHARS:
        return None
    session = _session_key(session_id)
    if not session:
        # An unkeyed session shares no row, the way the engagement counters
        # refuse one: measuring one session's reads against another session's
        # refusals is worse than recording nothing.
        return None

    window_key = (path, _requested_offset(call_args))
    if _is_read_refusal(parsed):
        return _note(parsed=parsed, session=session, window_key=window_key)
    _record_if_truncated(parsed=parsed, session=session, window_key=window_key)
    return None


def truncated_read_window(
    session_id: str, path: str, offset: int = 1
) -> tuple[int, int, int, int] | None:
    """The recorded ``(start, last_line, total_lines, next_offset)``, or ``None``.

    ``offset`` is the window start, defaulting to the first window. Diagnostics
    and tests. Never a gate.
    """
    session = _session_key(session_id)
    if not session:
        return None
    with _records_lock:
        return (_records.get(session) or {}).get((path, offset))


def reset_truncated_read_records() -> None:
    """Test seam: forget every recorded window.

    This memory is process-global, and the repository has already paid for that
    shape once (``fanout_dispatch._INTERRUPT_FLAG`` set by one test and
    surfaced as a CI-only failure in whichever test the shard planner ran
    next). Every test that populates it clears it through this function, so the
    obligation is greppable rather than remembered.
    """
    with _records_lock:
        _records.clear()


def _is_read_refusal(parsed: dict[str, Any]) -> bool:
    """Whether this result is one of the host's three read refusal/stub shapes.

    By key, never by wording. Presence of ``already_read`` is enough: the host
    writes it only on the two BLOCK bodies.
    """
    if _ALREADY_READ_KEY in parsed:
        return True
    if parsed.get(_GUARDRAIL_REFUSAL_KEY) is True:
        return True
    return parsed.get(_STATUS_KEY) == _STATUS_UNCHANGED


def _record_if_truncated(
    *, parsed: dict[str, Any], session: str, window_key: tuple[str, int]
) -> None:
    """Remember the window a truncated read returned, if this was one."""
    if parsed.get("truncated") is not True:
        return
    next_offset = _positive_int(parsed.get("next_offset"))
    total_lines = _positive_int(parsed.get("total_lines"))
    if next_offset is None or total_lines is None:
        return
    start = window_key[1]
    last_line = next_offset - 1
    # `next_offset` is the first line NOT returned, so a window that ended
    # before it began, or one running past the end of the file, is incoherent
    # and recorded as nothing rather than printed as a range that never
    # existed. The lower bound is the window's OWN start, not line 1: a read
    # from offset 5000 returning nothing through 4999 is not a short read, it
    # is a result this pass cannot describe.
    if last_line < start or last_line > total_lines:
        return
    with _records_lock:
        windows = _records.pop(session, None)
        if windows is None:
            windows = OrderedDict()
        _ = windows.pop(window_key, None)
        windows[window_key] = (start, last_line, total_lines, next_offset)
        while len(windows) > MAX_TRACKED_WINDOWS_PER_SESSION:
            _ = windows.popitem(last=False)
        _records[session] = windows
        while len(_records) > MAX_TRACKED_READ_SESSIONS:
            _ = _records.popitem(last=False)


def _note(
    *, parsed: dict[str, Any], session: str, window_key: tuple[str, int]
) -> str | None:
    """Add the note when this refusal's own window was recorded truncated.

    The lookup IS the gate. A refused call is annotated only when a truncated
    read of the same path from the same start was recorded, so the note never
    describes a window other than the one the refusal is about.
    """
    with _records_lock:
        window = (_records.get(session) or {}).get(window_key)
    if window is None:
        return None
    start, last_line, total_lines, next_offset = window
    parsed[TRUNCATED_READ_RECOVERY_KEY] = recovery_note_text(
        start=start, last_line=last_line, total_lines=total_lines, next_offset=next_offset
    )
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def recovery_note_text(
    *, start: int, last_line: int, total_lines: int, next_offset: int
) -> str:
    """The bounded note. Every number and every claim came from one record.

    Scoped to THAT read on purpose. OMH keeps one window per key, not the union
    of everything a session has read, so it cannot say a line was absent from
    every result -- a later `offset=1667` read may well have returned it. What
    the record does support is where this read stopped, and that is all the
    note says.
    """
    return (
        f"{_NOTE_PREFIX} An earlier read_file of this path from offset={start} "
        f"was truncated: it returned lines {start}-{last_line} of {total_lines} "
        f"and stopped there, so lines past {last_line} were not in that result. "
        f"This call asked for the same offset. To read another region, call "
        f"read_file with offset set to it and limit to bound it; "
        f"offset={next_offset} continues from where that read stopped."
    )


def _requested_offset(call_args: dict[str, Any]) -> int:
    """The offset the tool actually read from, derived the host's own way.

    ``read_file`` clamps its argument with
    ``file_operations_common.normalize_read_pagination``: ``max(1, int(offset))``
    with the default on a value ``int()`` refuses. So a missing offset, ``0``,
    ``-5``, ``None`` and a junk string all READ FROM LINE 1 and are the case
    this note is for, while ``1667`` and the string ``"1667"`` (which
    ``tools/arg_coercion.coerce_tool_args`` turns into the integer before the
    hook ever sees it) are not. Mirroring the clamp rather than testing for the
    literal ``1`` is what keeps the gate describing the read that happened.
    """
    value = call_args.get("offset")
    try:
        offset = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1
    return max(1, offset)


def _positive_int(value: object) -> int | None:
    """*value* as an ``int`` above zero, or ``None``.

    ``bool`` is excluded although it is an ``int`` subclass: ``truncated`` is
    the only boolean in this shape, and a ``True`` arriving in a line-count
    field is corruption, not a count of one.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _session_key(session_id: object) -> str:
    if not isinstance(session_id, str):
        return ""
    return session_id.strip()[:MAX_TODO_SESSION_REF_CHARS]
