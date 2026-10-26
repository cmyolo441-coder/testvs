"""Full-width diff bands via the Hermes ``transform_tool_result`` seam.

The Hermes TUI paints diff add/delete lines with a text-run background, so
the colored band ends wherever each line's text ends — a ragged highlighter
look the owner rejected. The fill geometry is renderer-owned and OMH never
patches Hermes, but Hermes ships an explicit canonicalization seam for
exactly this: ``transform_tool_result`` receives the final tool-result
string and may replace it.

OMH's transform pads every painted diff line (``+``/``-``) with trailing
spaces to the diff block's widest line, measured in TERMINAL CELLS (East
Asian wide characters occupy two), so the text-run background renders as one
uniform rectangle. Content is unchanged beyond trailing whitespace — the
diff still applies, the model reads the same change — and any parse doubt
returns ``None`` so the result passes through untouched (the seam is
fail-open by contract).

The padded string is also what the model reads, so the padding is bounded in
characters by ``MAX_BAND_PADDING_RATIO``: past it the band drops to a narrower
line's width and the wider lines stay unpadded, the same right edge
``MAX_BAND_CELLS`` already leaves on an outlier.
"""

from __future__ import annotations

import json
from typing import Any

# Padding stops at this cell width: lines wider than the cap wrap in the TUI
# anyway, and padding everything to a pathological outlier would bloat every
# other line. A capped block still reads as a uniform band for normal diffs.
MAX_BAND_CELLS = 160

# Trailing-space padding may not exceed the diff's own length in CHARACTERS,
# so the transform can never more than double what it is handed, in the unit
# the result is billed in. The cap above is a ceiling, not a bound: under it
# the band is still the widest line, so one long line sets it for every short
# one. The #1729 case -- 118 short lines beside one 152-cell line -- grew from
# 3,310 to 18,682 chars that way. Over the budget the band steps down to the
# widest painted line whose padding fits, and the lines above it keep the
# unpadded right edge MAX_BAND_CELLS already leaves on an outlier.
#
# Characters, although the band itself is a cell width. The padding is spaces,
# one character and one cell each, so the spend is the same number in either
# unit; only the diff it is measured against differs, and a double-width line
# is two cells per one character. A cell-denominated budget therefore bounds
# nothing the result is billed in: measured, a CJK-heavy diff reached 2.27x in
# characters under a cell budget of 1.0.
#
# 1.0 rather than a tighter fraction because the step-down lands on a real line
# width either way, so a tighter budget reshapes more ordinary diffs without
# recovering more on the pathological one.
MAX_BAND_PADDING_RATIO = 1.0

_PAINTED_PREFIXES = ("+", "-")


def _cell_width(text: str) -> int:
    """Terminal cell width with the same wide ranges the HUD widget uses."""
    width = 0
    for char in text:
        code = ord(char)
        wide = code >= 0x1100 and (
            code <= 0x115F
            or code in (0x2329, 0x232A)
            or 0x2E80 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3
            or 0xF900 <= code <= 0xFAFF
            or 0xFE10 <= code <= 0xFE6F
            or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6
        )
        width += 2 if wide else 1
    return width


def _looks_like_diff(text: str) -> bool:
    lines = text.splitlines()
    has_hunk = any(line.startswith("@@") for line in lines)
    has_marker = any(line.startswith(("--- ", "+++ ")) for line in lines)
    painted = sum(1 for line in lines if line.startswith(_PAINTED_PREFIXES))
    return (has_hunk or has_marker) and painted >= 2


def _padding_cells(widths: list[int], band: int) -> int:
    """Cells of trailing space a band this wide would add to those lines.

    A space is one cell and one character, so this is also the number of
    characters added, which is what the budget is a fraction of.
    """
    return sum(band - width for width in widths if width < band)


def _affordable_band(widths: list[int], band: int, budget: int) -> int:
    """Widest painted line at or under ``band`` whose padding fits ``budget``.

    Only widths a line actually has are considered: a band between two of them
    covers no line the lower one misses, so the difference is padding nobody
    sees. Walking the sorted widths, everything before ``index`` is at or under
    ``width``, so ``index * width - below`` is that band's padding.
    """
    best = 0
    below = 0
    for index, width in enumerate(sorted(widths)):
        if width > band:
            break
        if index * width - below <= budget:
            best = width
        below += width
    return best


def pad_diff_lines(text: str) -> str:
    """Pad painted diff lines to the widest band the budget allows, in cells."""
    lines = text.splitlines()
    band = min(
        MAX_BAND_CELLS,
        max((_cell_width(line) for line in lines), default=0),
    )
    painted_widths = [
        _cell_width(line) for line in lines if line.startswith(_PAINTED_PREFIXES)
    ]
    budget = int(len(text) * MAX_BAND_PADDING_RATIO)
    if _padding_cells(painted_widths, band) > budget:
        band = _affordable_band(painted_widths, band, budget)
    padded = []
    for line in lines:
        if line.startswith(_PAINTED_PREFIXES):
            gap = band - _cell_width(line)
            padded.append(line + " " * gap if gap > 0 else line)
        else:
            padded.append(line)
    trailer = "\n" if text.endswith("\n") else ""
    return "\n".join(padded) + trailer


def transform_tool_result(**kwargs: Any) -> str | None:
    """Return a padded replacement result, or ``None`` to leave it alone."""
    result = kwargs.get("result")
    if not isinstance(result, str) or "@@" not in result and "--- " not in result:
        return None
    try:
        parsed = json.loads(result)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        diff = parsed.get("diff")
        if not isinstance(diff, str) or not _looks_like_diff(diff):
            return None
        padded = pad_diff_lines(diff)
        if padded == diff:
            return None
        parsed["diff"] = padded
        try:
            return json.dumps(parsed, ensure_ascii=False)
        except (TypeError, ValueError):
            return None
    if _looks_like_diff(result):
        padded = pad_diff_lines(result)
        return padded if padded != result else None
    return None
