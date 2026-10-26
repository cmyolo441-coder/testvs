"""Code-mode discipline guidance for Hermes ``execute_code`` results.

Hermes' ``execute_code`` tool (programmatic tool calling) already rewards the
code-mode call shape: it refunds the iteration budget for execute_code-only
turns and collapses results to one line under context compression. What the
host does not teach is the discipline that keeps those scripts trustworthy —
a script that swallows its own exception exits 0, and exit 0 then reads as
success in the next model turn.

This module carries that discipline as a bounded, static guidance block that
``transform_tool_result`` appends into the first ``execute_code`` result of a
session, under its own JSON key so the host's result parsing is untouched.
The rules were derived from reading the host implementation
(``tools/code_execution_tool.py``): each ``execute_code`` run is a fresh
child process, so state persists only through files, and the host surfaces
``exit_code`` but cannot see whether the script verified its own effects.

The guidance is prepared instruction only. Injecting it is not evidence that
a script followed it, and an ``exit_code`` of 0 in an annotated result is
still not execution, verification, review, CI, or merge evidence.
"""

from __future__ import annotations

import json
import threading
from typing import Final

CODE_MODE_GUIDANCE_SCHEMA_VERSION: Final = "omh_code_mode_guidance/v1"
CODE_MODE_GUIDANCE_KEY: Final = "omh_guidance"
CODE_MODE_GUIDANCE_TOOL: Final = "execute_code"

# One rule per failure mode this repo can name, each grounded in an observed
# host fact rather than a style preference. Keep the block bounded: it rides
# inside a tool result, so every character here is context spent on every
# session's first execute_code call.
CODE_MODE_DISCIPLINE_RULES: Final[tuple[str, ...]] = (
    "Never swallow failures: no bare except or except-pass; let errors "
    "propagate so they surface as a non-zero exit_code.",
    "exit_code 0 is not verified success. After a script mutates files or "
    "state, re-read the mutated target and confirm the change before "
    "reporting it done.",
    "Route every derived number (counts, sums, set differences, date math) "
    "through code and print it; never report a figure the script did not "
    "print.",
    "Destructive operations run as a dry run first: print the affected set, "
    "then mutate only after it is confirmed.",
    "Scripts do not share variables — each execute_code run is a fresh "
    "process. Persist intermediate state to files and read it back instead "
    "of re-fetching the same data.",
)

_GUIDANCE_HEADER: Final = "[OMH Code-Mode Discipline]"

# Session-scoped delivery dedup. In-process state is sufficient and correct
# here: the injection rides the Hermes process that runs the hook, and
# re-delivering once after a host restart is acceptable (even useful), so a
# cross-process ledger would add I/O without adding a guarantee.
_delivered_lock = threading.Lock()
_delivered_sessions: set[str] = set()


def code_mode_guidance_text() -> str:
    """The bounded guidance block, static and free of any result content."""
    lines = [_GUIDANCE_HEADER]
    lines.extend(f"- {rule}" for rule in CODE_MODE_DISCIPLINE_RULES)
    return "\n".join(lines)


def annotate_execute_code_result(
    *,
    tool_name: object,
    result: object,
    session_id: str = "",
) -> str | None:
    """Return *result* with the guidance key added, or ``None`` to pass through.

    Fail-open by seam contract: anything that is not a clean ``execute_code``
    JSON-object result — or a session that already received the guidance —
    passes through untouched.
    """
    if str(tool_name or "") != CODE_MODE_GUIDANCE_TOOL:
        return None
    if not isinstance(result, str) or not result:
        return None
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or CODE_MODE_GUIDANCE_KEY in parsed:
        return None
    if not _claim_delivery(str(session_id or "")):
        return None
    parsed[CODE_MODE_GUIDANCE_KEY] = code_mode_guidance_text()
    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _claim_delivery(session_id: str) -> bool:
    """Atomically claim one delivery per session (process-wide when unkeyed)."""
    with _delivered_lock:
        if session_id in _delivered_sessions:
            return False
        _delivered_sessions.add(session_id)
    return True


def _reset_delivery_state() -> None:
    """Test seam: forget which sessions were served."""
    with _delivered_lock:
        _delivered_sessions.clear()
