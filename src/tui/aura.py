"""The aura a `dag-ulw` in the composer draws (owner request 2026-10-23, after Claude Code's `ultrathink`).

While the draft holds `dag-ulw`, `dag ulw` or `/dag-ulw` (any case, a whole
token, though a Korean particle may follow it: `dag-ulwx`, `undag ulw` and
a path's `src/dag-ulw/` are none), the
token's letters run violet, magenta, pink and lavender (`Theme.aura`) with a
near-white band sweeping through them (`Theme.aura_glint`), and the
composer's two rules are drawn heavy (`AURA_RULE`) in the same gradient with
a band of their own travelling along them. Nothing else moves: the rows, the
caret and the cells are the draft's own.

It moves only at the truecolor depth with motion on, and only while the
token is there, so the app ticks for it then and never otherwise
(`App._aura_live`). At 256 or 16 colours it stands still; at the `NO_COLOR`
depth, and on a background where its band would not show
(`theme.aura_moves`), the token and the heavy rules are bold and still.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from .style import Line, Span, Style

AURA_RULE = "━"  # the composer's rules while the aura is on: heavy, the cells of `frame.RULE`
GLINT_CELLS = 2  # the band through the token
RULE_GLINT_CELLS = 16  # the band along a rule
RULE_GLINT_SPEED = 4  # cells the rule's band travels a phase

# A whole `dag-ulw`, `dag ulw` or `/dag-ulw` token: nothing a word or a path goes on with on either side. It ends on an
# ASCII class, so a Korean particle (`dag-ulw로`, `/dag-ulw를`) leaves it whole, as the router reads it.
_KEYWORD = re.compile(r"(?<![\w./-])/?dag[- ]ulw(?![A-Za-z0-9_/-])", re.IGNORECASE)


def aura_ranges(text: str) -> list[tuple[int, int]]:
    """(start, end) str indexes of each `dag-ulw` token in `text`."""
    return [(found.start(), found.end()) for found in _KEYWORD.finditer(text)]


def _step(at: int, span: int, steps: int) -> int:
    """The gradient step of cell `at` of `span`: the first cell the first step, the last the last."""
    return at * (steps - 1) // max(span - 1, 1)


def aura_style(base: Sequence[Style], glint: Sequence[Style], offset: int, span: int, phase: int) -> Style:
    """The style of the token's cell `offset` of `span` at `phase`: its step of the gradient, from the band's while the
    band, `GLINT_CELLS` wide and a cell a phase, passes over it (it enters from the left and leaves at the right)."""
    band = phase % (span + 2 * GLINT_CELLS) - GLINT_CELLS
    step = _step(offset, span, len(base))
    return glint[step] if band <= offset < band + GLINT_CELLS else base[step]


def aura_rule(width: int, base: Sequence[Style], glint: Sequence[Style], phase: int) -> Line:
    """A composer rule `width` cells of `AURA_RULE` in the gradient, a band `RULE_GLINT_CELLS` wide travelling along it
    `RULE_GLINT_SPEED` cells a phase."""
    if width <= 0 or not base:
        return ()
    band = (phase * RULE_GLINT_SPEED) % (width + RULE_GLINT_CELLS) - RULE_GLINT_CELLS
    spans: list[Span] = []
    for at in range(width):
        step = _step(at, width, len(base))
        style = glint[step] if band <= at < band + RULE_GLINT_CELLS else base[step]
        if spans and spans[-1].style == style:
            spans[-1] = Span(spans[-1].text + AURA_RULE, style)
        else:
            spans.append(Span(AURA_RULE, style))
    return tuple(spans)
