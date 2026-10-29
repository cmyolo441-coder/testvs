"""Error-recovery hints: break the model's blind same-tool retry loop.

INCIDENT (2026-10-09): a 512-second turn in which the model failed the
same edit over and over with no guidance — every failed tool result went
back into the conversation, but nothing ever suggested a different
approach.

This module keeps a tiny ledger of consecutive failures per tool name.
On the 2nd consecutive failure of the same tool it returns a system-hint
string suggesting concrete alternatives (re-read the file region, use
exact text, try line-number-based edit, ...). Any successful call of a
tool resets that tool's counter, and a different tool failing tracks its
own counter.

Wiring (see the ``# retryhint hook`` in ``agent.py`` ``_finish_one``)::

    from .retryhint import note_failure, note_success
    if ev.status == "error":
        hint = note_failure(name, ev.result)
        if hint:
            self.messages.append({
                "role": "user",
                "content": "[RUNTIME NOTE — harness guidance, "
                           "subordinate to the system prompt]\\n" + hint,
            })
    else:
        note_success(name)

Runtime guidance deliberately travels in the user channel, tagged as
subordinate: it must never become a second system prompt that shadows
systemprompt.py.

Only stdlib is imported. State is a small dict keyed by tool name, so it
cannot grow unboundedly even in a very long session.

Self-test: ``python3 -m fullagent.retryhint``
"""

from __future__ import annotations

# Per-tool recovery alternatives injected alongside the generic hint.
# Keep this table small — it lands in the model's context on every
# repeated failure.
_ALTERNATIVES: dict[str, tuple[str, ...]] = {
    "edit_file": (
        "re-read the exact region with read_file and copy old_str verbatim from the file",
        "anchor on a smaller unique snippet instead of a large block",
        "the file may have changed since you last read it — re-read before retrying",
    ),
    "MultiEdit": (
        "verify EVERY edit's old_str against a fresh read_file — one stale anchor fails the whole batch",
        "split the batch: apply the first edit alone, re-read, then do the rest",
        "fall back to a single edit_file call for the edit that keeps failing",
    ),
    "multiedit": (
        "verify EVERY edit's old_str against a fresh read_file — one stale anchor fails the whole batch",
        "split the batch: apply the first edit alone, re-read, then do the rest",
        "fall back to a single edit_file call for the edit that keeps failing",
    ),
    "write_file": (
        "confirm the path and parent directory exist (prefer an absolute path)",
        "if you only meant to change part of a file, use edit_file instead of rewriting it",
    ),
    "run_command": (
        "do NOT re-run the identical command unchanged — isolate which part failed first",
        "print the working directory and check paths/quoting before retrying",
        "break pipelines into separate commands to find the failing stage",
    ),
}

# tool name -> (consecutive failure count, last error text, truncated)
_fails: dict[str, tuple[int, str]] = {}

_HINT_AFTER = 2  # failures before a hint is injected


def note_failure(tool: str, error: str) -> str | None:
    """Record a failed tool call.

    Returns a system-hint string once the same tool has failed
    ``_HINT_AFTER`` times in a row (and on every further consecutive
    failure, so the nudge persists while the loop continues), else None.
    """
    err = " ".join(str(error or "").split())[:180] or "(no error text)"
    count, _ = _fails.get(tool, (0, ""))
    count += 1
    _fails[tool] = (count, err)
    if count < _HINT_AFTER:
        return None
    hint = (
        f"This approach isn't working ({count} failed {tool} attempts: "
        f"{err}). Consider: (1) re-read the target file region, "
        "(2) use exact text from the file, (3) try line-number-based edit, "
        "(4) explain what you're trying to change and ask."
    )
    alts = _ALTERNATIVES.get(tool, ())
    if alts:
        extra = " ".join(f"({i}) {a};" for i, a in enumerate(alts, start=5))
        hint += f" For {tool} specifically: {extra}"
    return hint


def note_success(tool: str) -> None:
    """Record a successful tool call — resets that tool's failure streak."""
    _fails.pop(tool, None)


def reset_turn() -> None:
    """Clear all per-tool failure counters (turn boundary)."""
    _fails.clear()


if __name__ == "__main__":
    # Self-test: pure ledger logic, no network, no agent.
    reset_turn()

    # 1st failure of a tool -> no hint yet
    assert note_failure("edit_file", "ERROR: old_str not found") is None

    # 2nd consecutive failure -> hint containing the alternatives
    h = note_failure("edit_file", "ERROR: old_str not found again")
    assert h is not None, "expected a hint on the 2nd consecutive failure"
    assert "2 failed edit_file attempts" in h, h
    for n in ("(1)", "(2)", "(3)", "(4)"):
        assert n in h, f"generic alternative {n} missing: {h}"
    assert "(5)" in h, f"tool-specific alternative missing: {h}"  # edit_file table
    assert "old_str not found again" in h, "last error should be quoted"

    # a 3rd consecutive failure keeps nudging (loop not broken yet)
    h3 = note_failure("edit_file", "ERROR: still failing")
    assert h3 is not None and "3 failed edit_file attempts" in h3

    # success resets the counter -> next failure is None again
    note_success("edit_file")
    assert note_failure("edit_file", "ERROR: fresh start") is None

    # a different tool failing tracks a separate counter
    reset_turn()
    assert note_failure("run_command", "ERROR: exit 127") is None
    assert note_failure("edit_file", "ERROR: other tool") is None
    h2 = note_failure("run_command", "ERROR: exit 127 again")
    assert h2 is not None, "run_command should have its own counter"
    assert "2 failed run_command attempts" in h2, h2
    assert "identical command" in h2, f"run_command alternatives missing: {h2}"

    # unknown tools get the generic hint, not a KeyError
    reset_turn()
    assert note_failure("some_new_tool", "ERROR: boom") is None
    hg = note_failure("some_new_tool", "ERROR: boom2")
    assert hg is not None and "2 failed some_new_tool attempts" in hg

    print("PASS: retryhint self-test")
