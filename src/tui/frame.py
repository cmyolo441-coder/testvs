"""One repaint: from what the app holds to the rows `LiveRegion.render` draws.

The app loop calls `build_frame` once per repaint with the terminal size, the
parts it holds (sticky prompt, streaming tail, slot, composer, info line)
and the input mode, then hands the result to
`LiveRegion.render(frame.lines, frame.cursor, cols, flush=lines)`.

The composer is drawn as Claude Code draws it: a full-width rule `─…─` above
it, its input rows full width behind the `❯ ` prompt mark (no side edges, no
corners), and a full-width rule under it, both rules in the rule colour
whatever the input mode. The rules take the rows the layout gives the
composer frame (`layout.allocate`), so a 6-row terminal draws the top rule
alone, and below `layout.MIN_ROWS` rows or `layout.MIN_COLS` columns the input
rows have no rule at all. A `/name` in the draft that resolves to a command or
skill (`known_slash`, `slash.known_names`) is drawn bold in its colour
(`Theme.slash`). A `dag-ulw` token in the draft is drawn in the aura instead
(`aura`), and the rules are drawn heavy in its gradient while it is there. While the composer holds a `!` line for Hermes' shell the
prompt mark is drawn bold in the shell family's hue instead of the prompt colour; the rules
stay as they are. The info line under the bottom rule is a line the app
builds, its text `INFO_INSET` cells in; the frame only places it. The agent
view the app builds (`agents`) is drawn under the info line, the region's last
rows.

The composer follows the content until the screen is full, then stays on the
bottom rows. With `min_height` 0 a frame is exactly as tall as what it holds
(`layout.allocate`), and `LiveRegion` draws it right under the output, so the
composer sits under the last row printed, the panel, prompt slot, completion
menu, tail and sticky prompt above it, and the rows under it (the bottom
rule, by the terminal size alone, `layout.under_composer`, the info line and
the agent view) under it. When something above the composer grows,
the composer moves down with it until the region's last row is the screen's
last row, and from then on the screen scrolls instead; when something above
it shrinks, the composer moves up by as many rows. Once the region's last row
is the screen's last row, the app passes the region's height as
`min_height`, so what shrinks there becomes pad rows under the live block,
above the working row and its tip, and the composer does not move up: when the
agent view under it shrinks, it moves down by as many rows.

An empty composer shows `placeholder` after the prompt glyph in the hint
colour while the app asks for it (no slot open). It is drawn, never typed:
the caret stays right after the glyph and the composer's text stays empty.

While the person picks from a list (`composer` is None) the frame draws no
composer at all: no rules, no input rows, and no caret (`Frame.cursor` is None,
and `LiveRegion` hides the cursor) unless `slot_caret` puts one in the slot (a
picker's filter row or the history search's query row, where typing goes). The slot stands in the
composer's place, right above the info line, with the rows `layout.choice_cap`
gives it. The rows under the composer are the same either way.

The attachment row (the images queued for the next message) is the composer's
first row under the top rule while there is one: it counts against the composer's cap.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .aura import AURA_RULE, aura_ranges, aura_rule, aura_style
from .composer import PROMPT, Composer
from .layout import BLANK, Budget, allocate, caret_row, compose, composer_cap
from .style import TRUECOLOR, Line, Span, Style
from .slash import slash_spans, slash_token_ranges
from .theme import DEFAULT_THEME, KIND_FAMILIES, Theme, aura_moves
from .width import cell_width, line_width, truncate, truncate_line

RULE = "─"  # the composer's rules, above and under it, full width
INFO_INSET = 2  # cells before the info line's text, under the rule
STICKY_PREFIX = "↳ you: "
STICKY_CUT = "…"  # where the reminder's prompt is cut
BROWSING_HINT = "↑↓ browsing · esc to type"  # in the composer's place while the agent list or the dag tree has the keys

MODE_NORMAL = "normal"
MODE_ACCEPT = "accept"
MODE_PLAN = "plan"
MODE_ACCEPT_PLAN = "accept+plan"  # approvals off and the plan flag on, both at once
MODE_SHELL = "shell"  # the composer holds a `!` line: it goes to Hermes' shell.exec, not to the model


@dataclass(frozen=True)
class Frame:
    lines: tuple[Line, ...]
    cursor: tuple[int, int] | None  # (row, col) of the composer caret within the region; None with no composer
    height: int
    budget: Budget


def build_frame(
    *,
    cols: int,
    rows: int,
    composer: Composer | None,
    info: Line,
    sticky: Line | None = None,
    tail: Sequence[Line] = (),
    slot: Sequence[Line] = (),
    theme: Theme = DEFAULT_THEME,
    prompt: str = PROMPT,
    attachments: Line | None = None,
    mode: str = MODE_NORMAL,
    placeholder: str = "",
    slot_caret: tuple[int, int] | None = None,
    text_question: bool = False,
    min_height: int = 0,
    below: Sequence[Line] = (),
    agents: Sequence[Line] = (),
    composer_blank: bool = False,
    tail_floor: int = 0,
    known_slash: frozenset[str] = frozenset(),
    aura: int = 0,
    aura_depth: str = TRUECOLOR,
) -> Frame:
    """The region for one repaint; `min_height` rows at least, pad rows under the live block making up the rest.

    `attachments` is drawn as the composer's first row when the composer may
    have two rows; a terminal too short for that shows the input alone.
    `mode` MODE_SHELL draws the prompt mark in the shell family's hue; `info` is the
    region's last row. `composer` None draws no
    composer (nor the attachment row): the slot takes its place, and the caret
    is `slot_caret`, a (row, col) within `slot`, or none where that is None or
    its row is cut.
    `text_question` says the slot holds a question answered with text, whose
    label keeps a row on a tiny terminal (`layout.allocate`). `below` is drawn
    under the bottom edge, above the info line: the completion menu where the
    app puts it under the composer. `agents` is the agent view, drawn under
    the info line as the region's last rows, sized by its producer to
    `layout.agent_view_cap`. `composer_blank` keeps the composer's rules and
    draws its rows as the dim `BROWSING_HINT` and blank rows of the same
    count, with no caret: the agent list or the dag tree under them has the
    keys, and nothing moves as it opens or closes.
    `tail_floor` is the tail's last rows the agent view leaves it
    (`layout.allocate`): the working row while a turn runs. `known_slash`
    names the commands and skills a `/name` in the draft is drawn for (none
    in a draft holding a paste chip). `aura` is the phase a `dag-ulw` token's
    gradient and its band stand at (`aura`), painted at `aura_depth`: the
    token, over its `/name` colour, and both rules, drawn heavy, while the
    draft holds one.
    """
    cap = composer_cap(rows)
    composer_lines: list[Line] = []
    glow: list[tuple[int, int]] = []
    base: tuple[Style, ...] = ()
    glint: tuple[Style, ...] = ()
    caret_line = caret_col = 0
    if composer is not None:
        chips = [truncate_line(attachments, cols)] if attachments is not None and cap >= 2 else []
        mark = theme.kinds[KIND_FAMILIES.index("shell")] if mode == MODE_SHELL else theme.prompt
        marks = slash_token_ranges(composer.text, known_slash) if known_slash and not composer.pastes else []
        glow = aura_ranges(composer.text) if theme.aura and not composer.pastes and not composer.masked else []
        base, glint = (tuple(replace(style, depth=aura_depth) for style in styles) if aura_depth != TRUECOLOR
                       else styles for styles in (theme.aura, theme.aura_glint))
        input_lines, (caret_line, caret_col) = composer.render(
            cols, cap - len(chips), prompt, mark, theme.muted, marks, theme.slash,
            glow, lambda offset, length: aura_style(base, glint, offset, length, aura),
        )
        if placeholder and not composer.text and input_lines:
            input_lines[0] = truncate_line((*input_lines[0], Span(placeholder, theme.hint)), cols)
        composer_lines = chips + input_lines
        caret_line += len(chips)
    budget = allocate(
        rows,
        cols,
        sticky=sticky is not None,
        tail_rows=len(tail),
        slot_rows=len(slot),
        composer_rows=len(composer_lines),
        min_height=min_height,
        composer_hidden=composer is None,
        text_question=text_question,
        below_rows=len(below),
        agents_rows=len(agents),
        tail_floor=tail_floor,
    )
    rule = rule_line(cols, theme)
    if glow and not composer_blank:
        rule = aura_rule(cols, base, glint, aura) if aura_moves(theme) else (
            Span(AURA_RULE * cols, replace(theme.rule, bold=True)),)
    lines = compose(
        budget,
        sticky=sticky,
        tail=tail,
        slot=slot,
        rule=rule,
        composer=_browsing_rows(len(composer_lines), cols, theme) if composer_blank else composer_lines,
        bottom_rule=rule,
        info=truncate_line(info, cols),
        below=below,
        agents=agents,
    )
    if composer is not None and not composer_blank:
        cursor: tuple[int, int] | None = (caret_row(budget, caret_line), caret_col)
    elif slot_caret is not None and 0 <= slot_caret[0] < budget.slot:
        cursor = (budget.pad + budget.sticky + budget.tail + slot_caret[0], slot_caret[1])
    else:
        cursor = None
    return Frame(
        lines=tuple(lines),
        cursor=cursor,
        height=budget.total,
        budget=budget,
    )


def _browsing_rows(count: int, cols: int, theme: Theme) -> list[Line]:
    """The composer's rows while a list under it has the keys: the hint, dim, then blank rows; as many as it had."""
    if count <= 0:
        return []
    return [truncate_line((Span(" " * INFO_INSET), Span(BROWSING_HINT, theme.hint)), cols), *[BLANK] * (count - 1)]


def rule_line(width: int, theme: Theme = DEFAULT_THEME) -> Line:
    """One of the composer's rules: `RULE` across `width`, in the rule colour."""
    return (Span(RULE * max(width, 0), theme.rule),)


def sticky_line(prompt_text: str, width: int, theme: Theme = DEFAULT_THEME, known: frozenset[str] = frozenset()) -> Line:
    """`↳ you: <first line of the prompt>` cut to `width`, shown while a turn runs.

    It has no band, as the prompt's echo in the transcript has none: `↳ you:`
    in the running blue (a turn runs), which no transcript row starts with, the prompt in the
    muted text colour (a `/name` in it that resolves, `known`, bold in its
    colour, as on the prompt's own row), and a dim `…` where it is cut. So it
    reads apart from the transcript rows right above it by its words and
    colours.
    """
    if width <= 0:
        return ()
    first = prompt_text.strip().split("\n", 1)[0]
    lead = Span(STICKY_PREFIX, theme.running)  # a turn runs
    line = (lead, *slash_spans(first, known, theme.muted, theme.slash))
    if line_width(line) <= width:
        return line
    room = width - cell_width(STICKY_PREFIX) - cell_width(STICKY_CUT)
    if room <= 0:
        return (*truncate_line(line, width - len(STICKY_CUT), ellipsis=""), Span(STICKY_CUT, theme.dim))
    # The words are cut before they are coloured: a token the cut goes through is no token (`/dag-ulw` cut to `/dag`).
    cut = truncate(first, room, ellipsis="")
    whole = [(start, end) for start, end in slash_token_ranges(first, known) if end <= len(cut)]
    spans = slash_spans(cut, known, theme.muted, theme.slash, whole if first.startswith(cut) else [])
    return (lead, *spans, Span(STICKY_CUT, theme.dim))
