"""Text the OMH HUD shows that its readers format the same way: elapsed times, counts, the stalled-checklist age.

`runtime_reader` (the panel payload, the text HUD line, the per-turn
reminder), the delegation view and the OMH TUI use these. They live apart from
`runtime_reader` so a reader that only formats text (the TUI's spinner and
agent rows) does not import the whole HUD reader. Nothing here imports more
than `typing`.
"""

from __future__ import annotations

from typing import Any

# The two statuses that ARE the finding; every other value is a way of being
# silent. Named once so a caller cannot cover one and miss the other.
TODO_UNCHANGED_STATUSES: frozenset[str] = frozenset({"unchanged", "unchanged_while_busy"})


def elapsed_text(seconds: float) -> str:
    """``12s`` / ``4m 3s`` / ``2h 01m``.

    Byte-identical to the widget's own ``elapsedText``, so the same elapsed
    figure reads the same whether it arrives through the TUI panel, the text
    HUD line, or the per-turn reminder.
    """
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m {total % 60}s"
    return f"{total // 3600}h {total // 60 % 60:02d}m"


def compact_count(value: float) -> str:
    """`950`, `1.2k`, `3M`, `70B`, `1.2T`: a count in at most four characters and a unit."""
    for size, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= size:
            return f"{value / size:.1f}".removesuffix(".0") + suffix
    return str(int(value))


def todo_unchanged_text(todo: dict[str, Any]) -> str:
    """How long this checklist has been unchanged, when that is the finding.

    Empty for every other stall status, so a caller renders the hint by
    appending this and nothing else; the projection's own status stays the
    single place the rule lives.
    """
    stall = todo.get("stall")
    # Both findings carry the age. `unchanged_while_busy` is the same fact with
    # calls running, and dropping it here would have silently reinstated the
    # liveness suppression the status split exists to remove.
    if not isinstance(stall, dict) or stall.get("status") not in TODO_UNCHANGED_STATUSES:
        return ""
    seconds = todo.get("updated_age_seconds")
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
        return ""
    return elapsed_text(seconds)
