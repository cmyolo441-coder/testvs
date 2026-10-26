"""Render the markdown subset model replies use, one complete source line at a time.

A reply streams in token by token, and a line that has left the live region is
in the terminal's scrollback for good. So the renderer never looks
ahead: `render_line` turns ONE complete source line into rows, and the only
thing a line can learn from the lines before it is `MarkdownState` (an open
fence and its language, the multi-line string or comment a highlighted fence
line ended inside, the content column of the last list item). Rendering
a reply line by line, threading the state, gives exactly the rows rendering it
whole gives; that equality is what lets the transcript flush a reply while it
is still streaming.

The subset: ATX headings, bullet and numbered lists (wrapped rows hang under
the item text), blockquotes, fenced code (verbatim, hard-wrapped by cells, no
inline markup inside; the opening fence shows only its language as a dim label,
or nothing, and the closing fence shows nothing; a language `highlight.py` knows
is syntax-coloured, any other stays one code colour), ```diff fences whose added
and removed lines are full-row bands, inline `code`, **bold**, *italic* / _italic_, links
`[t](u)` -> `t (u)`, and horizontal rules. Tables pass through as plain text.
Constructs that need a later line to decide an earlier one (setext headings,
indented code) are left out on purpose.

Every row is plain `Line`s for `screen.py` to paint. `clean_text` runs first:
whole escape sequences (CSI, OSC, the other control strings, two-byte escapes)
are dropped, then tabs are expanded and every remaining control character is
dropped, so rows are measured on what will show and carry none of the
`[38;2;…m` debris that dropping only the ESC byte would leave
(`screen.paint_line` strips any sequence again before a row is written).
Malformed markdown renders as literal text; nothing here raises on any input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import unicodedata

from .escapes import strip_escapes
from .highlight import highlight_line, highlights
from .style import Line, Span, Style
from .theme import Theme
from .width import TAB_CELLS, cell_width, cluster_width, clusters, is_unpainted, line_width, wrap_line

BULLET = "•"
RULE = "─"
QUOTE_BAR = "│ "
DIFF_LANGUAGES = frozenset({"diff", "patch"})

_FENCE_CHARS = ("`", "~")
_MIN_FENCE_RUN = 3
_MAX_BLOCK_INDENT = 3  # CommonMark: four leading spaces stop a line being a heading, rule or quote
_MAX_HEADING_LEVEL = 6
_BULLET_MARKERS = frozenset("-*+")
_ORDERED_DELIMITERS = frozenset(".)")
_MAX_ORDINAL_DIGITS = 9
_DIGITS = "0123456789"
_RULE_CHARS = frozenset("-*_")
_MIN_RULE_RUN = 3
_ESCAPABLE = frozenset("\\`*_[]()#+-.!|>~")



@dataclass(frozen=True)
class MarkdownState:
    """What one source line hands the next: the open fence, and the list it is in."""

    fence: str = ""  # the opening run (``` or ~~~~...) while inside a fenced block
    fence_lang: str = ""
    fence_indent: int = 0
    list_indent: int = 0  # content column of the most recent list item; 0 outside a list
    code_carry: str = ""  # the string or comment opener a highlighted fence line ended inside

    @property
    def in_fence(self) -> bool:
        return bool(self.fence)


def clean_text(text: str) -> str:
    """One source line made safe to measure: escapes dropped whole, tabs expanded, controls and bidi controls dropped.

    The transcript runs model and tool text through it so widths and wraps are
    computed on what will show. The guarantee that no escape sequence reaches
    the terminal is not this function's: `screen.paint_line` strips every row,
    whatever produced it.
    """
    return "".join(ch for ch in strip_escapes(text).expandtabs(TAB_CELLS) if not is_unpainted(ch))


def wrap_prefixed(spans: Sequence[Span], first: Line, rest: Line, width: int) -> list[Line]:
    """`spans` wrapped to `width`, the first row behind `first` and every other row behind `rest`.

    A prefix that leaves no room for text is dropped rather than overflowing the row.
    """
    width = max(1, width)
    prefix_cells = max(line_width(first), line_width(rest))
    if prefix_cells >= width:
        first, rest, prefix_cells = (), (), 0
    rows = wrap_line(tuple(spans), width - prefix_cells)
    return [(*(first if index == 0 else rest), *row) for index, row in enumerate(rows)]


def render_line(source: str, state: MarkdownState, width: int, theme: Theme) -> tuple[list[Line], MarkdownState]:
    """The rows for one complete source line, and the state the next line starts from."""
    text = clean_text(source)
    width = max(1, width)
    if state.in_fence:
        if _closes_fence(text, state.fence):
            return [], replace(state, fence="", fence_lang="", fence_indent=0, code_carry="")
        return _code_rows(text, state, width, theme)
    opened = _fence_open(text)
    if opened is not None:
        fence, lang = opened
        indent = _leading_spaces(text)
        list_indent = state.list_indent if indent else 0
        label = _verbatim_rows(" " * indent + lang, width, theme.muted) if lang else []
        return label, MarkdownState(fence, lang, indent, list_indent)
    if not text.strip():
        return [()], state
    return _block_rows(text, state, width, theme)


def render_lines(
    sources: Sequence[str], state: MarkdownState, width: int, theme: Theme
) -> tuple[list[Line], MarkdownState]:
    """`render_line` over several complete source lines, threading the state."""
    rows: list[Line] = []
    for source in sources:
        lines, state = render_line(source, state, width, theme)
        rows.extend(lines)
    return rows, state


def render_markdown(text: str, width: int, theme: Theme) -> list[Line]:
    """A whole document at `width`, from a fresh state."""
    return render_lines(text.split("\n"), MarkdownState(), width, theme)[0]


# ── block level ───────────────────────────────────────────────────────────


def _leading_spaces(text: str) -> int:
    return len(text) - len(text.lstrip(" "))


def _run_of(text: str, ch: str) -> str:
    """The run of `ch` that `text` starts with."""
    return text[: len(text) - len(text.lstrip(ch))]


def _fence_open(text: str) -> tuple[str, str] | None:
    stripped = text.lstrip(" ")
    for ch in _FENCE_CHARS:
        run = _run_of(stripped, ch)
        if len(run) < _MIN_FENCE_RUN:
            continue
        info = stripped[len(run) :].strip()
        if ch == "`" and "`" in info:
            return None  # ```a``` on one line is inline code, not a fence
        return run, info.split()[0].lower() if info else ""
    return None


def _closes_fence(text: str, fence: str) -> bool:
    stripped = text.lstrip(" ")
    run = _run_of(stripped, fence[0])
    return len(run) >= len(fence) and not stripped[len(run) :].strip()


def _heading_text(body: str) -> str | None:
    hashes = _run_of(body, "#")
    if not 1 <= len(hashes) <= _MAX_HEADING_LEVEL:
        return None
    if len(body) > len(hashes) and body[len(hashes)] != " ":
        return None
    return body[len(hashes) :].strip()


def _is_rule(body: str) -> bool:
    marks = body.replace(" ", "")
    return len(marks) >= _MIN_RULE_RUN and marks[0] in _RULE_CHARS and marks == marks[0] * len(marks)


def _list_item(body: str) -> tuple[str, str] | None:
    """(marker as painted, item text) when `body` opens a list item."""
    if body[:1] in _BULLET_MARKERS and body[1:2] in ("", " "):
        return f"{BULLET} ", body[1:].lstrip(" ")
    digits = body[: len(body) - len(body.lstrip(_DIGITS))]
    if not 1 <= len(digits) <= _MAX_ORDINAL_DIGITS:
        return None
    rest = body[len(digits) :]
    if rest[:1] in _ORDERED_DELIMITERS and rest[1:2] in ("", " "):
        return f"{digits}{rest[0]} ", rest[1:].lstrip(" ")
    return None


def _block_rows(text: str, state: MarkdownState, width: int, theme: Theme) -> tuple[list[Line], MarkdownState]:
    indent = _leading_spaces(text)
    body = text[indent:]
    if indent <= _MAX_BLOCK_INDENT:
        heading = _heading_text(body)
        if heading is not None:
            return wrap_prefixed(_inline_spans(heading, theme.heading, theme), (), (), width), replace(state, list_indent=0)
        if _is_rule(body):
            return [(Span(RULE * width, theme.dim),)], replace(state, list_indent=0)
        if body.startswith(">"):
            return _quote_rows(body, width, theme), state
    pad: Line = (Span(" " * indent),) if indent else ()
    if body.startswith("|"):
        return wrap_prefixed((Span(body, theme.text),), pad, pad, width), state
    item = _list_item(body)
    if item is not None:
        marker, content = item
        column = indent + cell_width(marker)
        first = (*pad, Span(marker, theme.muted))
        rows = wrap_prefixed(_inline_spans(content, theme.text, theme), first, (Span(" " * column),), width)
        return rows, replace(state, list_indent=column)
    if not indent:
        state = replace(state, list_indent=0)
    elif state.list_indent:
        pad = (Span(" " * state.list_indent),)  # a continuation line hangs under its item's text
    return wrap_prefixed(_inline_spans(body, theme.text, theme), pad, pad, width), state


def _quote_rows(body: str, width: int, theme: Theme) -> list[Line]:
    depth = 0
    rest = body
    while rest.startswith(">"):
        depth += 1
        rest = rest[1:]
        if rest.startswith(" "):
            rest = rest[1:]
    bar: Line = (Span(QUOTE_BAR * depth, theme.quote),)
    return wrap_prefixed(_inline_spans(rest, theme.quote, theme), bar, bar, width)


# ── fenced code ───────────────────────────────────────────────────────────


def hard_wrap(spans: Sequence[Span], width: int) -> list[Line]:
    """Styled spans cut into rows of at most `width` cells, keeping every space and never splitting a cluster."""
    rows: list[Line] = []
    row: list[Span] = []
    used = 0
    for span in spans:
        piece = ""
        for cluster in clusters(span.text):
            cluster_cells = cluster_width(cluster)
            if (row or piece) and used + cluster_cells > width:
                if piece:
                    row.append(Span(piece, span.style))
                    piece = ""
                rows.append(tuple(row))
                row, used = [], 0
            piece += cluster
            used += cluster_cells
        if piece:
            row.append(Span(piece, span.style))
    rows.append(tuple(row))
    return rows


def band_rows(text: str, width: int, band: Style) -> list[Line]:
    """`text` hard-wrapped, each row padded to `width` in the band style (the TUI paints the whole row)."""
    rows: list[Line] = []
    for row in hard_wrap((Span(text, band),), width):
        filled = "".join(span.text for span in row)
        rows.append((Span(filled + " " * max(0, width - line_width(row)), band),))
    return rows


def _verbatim_rows(text: str, width: int, style: Style) -> list[Line]:
    return hard_wrap((Span(text, style),), width)


def _code_rows(text: str, state: MarkdownState, width: int, theme: Theme) -> tuple[list[Line], MarkdownState]:
    if state.fence_lang in DIFF_LANGUAGES:
        return _diff_fence_rows(text, state, width, theme), state
    if not highlights(state.fence_lang):
        return _verbatim_rows(text, width, theme.code), state
    spans, carry = highlight_line(unicodedata.normalize("NFC", text), state.fence_lang, state.code_carry, theme)
    return hard_wrap(spans, width), replace(state, code_carry=carry)


def _diff_fence_rows(text: str, state: MarkdownState, width: int, theme: Theme) -> list[Line]:
    body = text[min(state.fence_indent, _leading_spaces(text)) :]
    if body.startswith(("+++", "---")):
        return _verbatim_rows(text, width, theme.label)
    if body.startswith("@@"):
        return _verbatim_rows(text, width, theme.muted)
    if body.startswith("+"):
        return band_rows(text, width, theme.diff_add)
    if body.startswith("-"):
        return band_rows(text, width, theme.diff_remove)
    return _verbatim_rows(text, width, theme.code)


# ── inline ────────────────────────────────────────────────────────────────


def _code_span(text: str, start: int) -> tuple[str, int] | None:
    """(code text, index after the closing run) for a backtick run at `start`, or None if unclosed."""
    ticks = _run_of(text[start:], "`")
    search = start + len(ticks)
    while True:
        close = text.find(ticks, search)
        if close < 0:
            return None
        after = close + len(ticks)
        if text[after : after + 1] != "`":
            break
        search = after + len(_run_of(text[after:], "`"))
    code = text[start + len(ticks) : close]
    if len(code) >= 2 and code[0] == " " and code[-1] == " " and code.strip():
        code = code[1:-1]
    return code, after


def _link(text: str, start: int) -> tuple[str, str, int] | None:
    """(label, url, index after the link) for `[label](url)` at `start`."""
    close = text.find("]", start + 1)
    if close < 0 or text[close + 1 : close + 2] != "(":
        return None
    end = text.find(")", close + 2)
    if end < 0:
        return None
    label = text[start + 1 : close]
    url = text[close + 2 : end].strip()
    if not label or not url:
        return None
    return label, url, end + 1


def _opens_emphasis(text: str, start: int, marker: str) -> bool:
    after = start + len(marker)
    if text[after : after + 1] in ("", " "):
        return False
    return not (marker[0] == "_" and start > 0 and text[start - 1].isalnum())


def _closes_emphasis(text: str, at: int, marker: str) -> bool:
    if text[at - 1] == " ":
        return False
    if len(marker) == 1 and (text[at - 1] == marker or text[at + 1 : at + 2] == marker):
        return False  # a lone * never closes on half of a ** run
    return not (marker[0] == "_" and text[at + len(marker) : at + len(marker) + 1].isalnum())


def _emphasis(text: str, start: int) -> tuple[str, bool, int] | None:
    """(inner text, strong, index after the closing marker) for emphasis opening at `start`.

    The FIRST valid closer wins, and whether a position closes depends only on
    the text around it. So emphasis never nests inside its own kind (the outer
    would have taken the inner closer), and recursion is at most one level per
    marker kind (`**`, `__`, `*`, `_`) however many markers a line holds.
    """
    marker = text[start : start + 2] if text[start : start + 2] in ("**", "__") else text[start]
    if not _opens_emphasis(text, start, marker):
        return None
    search = start + len(marker) + 1
    while True:
        close = text.find(marker, search)
        if close < 0:
            return None
        if _closes_emphasis(text, close, marker):
            return text[start + len(marker) : close], len(marker) == 2, close + len(marker)
        search = close + 1


def _inline_spans(text: str, style: Style, theme: Theme) -> list[Span]:
    """Inline markup in one line of block text, as spans over the block's `style`."""
    spans: list[Span] = []
    plain: list[str] = []

    def take_plain() -> None:
        if plain:
            spans.append(Span("".join(plain), style))
            plain.clear()

    index = 0
    while index < len(text):
        ch = text[index]
        if ch == "\\" and text[index + 1 : index + 2] in _ESCAPABLE:
            plain.append(text[index + 1])
            index += 2
            continue
        if ch == "`":
            code = _code_span(text, index)
            if code is None:
                ticks = _run_of(text[index:], "`")
                plain.append(ticks)
                index += len(ticks)
                continue
            take_plain()
            if code[0]:
                spans.append(Span(code[0], theme.code))
            index = code[1]
            continue
        if ch == "[":
            link = _link(text, index)
            if link is not None:
                take_plain()
                label, url, index = link
                spans.append(Span(label, theme.link))
                if url != label:
                    spans.append(Span(f" ({url})", theme.muted))
                continue
        if ch in "*_":
            emphasis = _emphasis(text, index)
            if emphasis is not None:
                take_plain()
                inner, strong, index = emphasis
                inner_style = replace(style, bold=True) if strong else replace(style, italic=True)
                spans.extend(_inline_spans(inner, inner_style, theme))
                continue
        plain.append(ch)
        index += 1
    take_plain()
    return spans
