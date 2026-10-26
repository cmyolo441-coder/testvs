"""The in-memory pager: `/history`, `/logs` and long slash output, on the alternate screen.

The pager never touches the live region or the terminal's scrollback. It
switches to the alternate screen, paints a page of already-rendered lines and
a footer, and switches back when the person leaves, which restores the main
screen (live region included) exactly as it was. Nothing it shows is written
anywhere else. After its first frame a pager paints only the rows that changed
since its last one, until the size changes or `invalidate` says something else
cleared the screen, each frame one synchronized update as the live region's.

`Pager` shows lines rendered beforehand. `HistoryPager` (`/history`) renders
the transcript's blocks only as a page reaches them, so a long session opens
at the cost of one page.

Keys: j / Down one line down, k / Up one line up, Space / PageDown a page
down, b / PageUp a page up, g / Home the top, G / End the bottom, q / Esc
leave.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

from .keys import CHAR, DOWN, END, ESCAPE, HOME, PAGE_DOWN, PAGE_UP, UP, Key, Paste
from .screen import HIDE_CURSOR, SYNC_BEGIN, SYNC_END, paint_line
from .style import CSI, Line, Span
from .theme import Theme

ALT_SCREEN_ON = f"{CSI}?1049h"
ALT_SCREEN_OFF = f"{CSI}?1049l"
FOOTER_KEYS = "j/k line · space/b page · g/G ends · q leaves"
# The most blocks a `HistoryPager` keeps rendered; one scrolled back to after it left is rendered again.
HISTORY_BLOCKS_KEPT = 256


def _move_to(row: int) -> str:
    """Column 1 of 0-based screen row `row`."""
    return f"{CSI}{row + 1};1H"


@dataclass
class Pager:
    FOOTER_ROWS: ClassVar[int] = 1  # the rows `footer` paints under the page

    title: str
    lines: Sequence[Line]
    top: int = 0
    # What the last frame left on each screen row (the line and its bytes), and the size it was drawn at.
    _painted: dict[int, tuple[Line, str]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _size: tuple[int, int] | None = field(default=None, init=False, repr=False, compare=False)

    def invalidate(self) -> None:
        """Forget what is on the screen: the next frame paints every row."""
        self._painted = {}
        self._size = None

    def body_rows(self, rows: int) -> int:
        return max(rows - self.FOOTER_ROWS, 1)

    def _last_top(self, rows: int) -> int:
        return max(len(self.lines) - self.body_rows(rows), 0)

    def handle_key(self, event: Key | Paste, rows: int) -> bool:
        """Move by one key; False when the person left the pager."""
        if not isinstance(event, Key):
            return True
        page = self.body_rows(rows)
        char = event.char if event.name == CHAR and not event.ctrl and not event.alt else ""
        if event.name == ESCAPE or char == "q":
            return False
        if event.name == DOWN or char == "j":
            self._scroll(1, rows)
        elif event.name == UP or char == "k":
            self._scroll(-1, rows)
        elif event.name == PAGE_DOWN or char == " ":
            self._scroll(page, rows)
        elif event.name == PAGE_UP or char == "b":
            self._scroll(-page, rows)
        elif event.name == HOME or char == "g":
            self._scroll_to_end(False, rows)
        elif event.name == END or char == "G":
            self._scroll_to_end(True, rows)
        return True

    def _scroll(self, step: int, rows: int) -> None:
        self.top = min(max(self.top + step, 0), self._last_top(rows))

    def _scroll_to_end(self, bottom: bool, rows: int) -> None:
        self.top = self._last_top(rows) if bottom else 0

    def _page(self, cols: int, rows: int, theme: Theme) -> tuple[list[Line], str]:
        """The body rows of the current page, and the footer's position text."""
        body = self.body_rows(rows)
        self.top = min(self.top, self._last_top(rows))
        last = min(self.top + body, len(self.lines))
        position = f"{self.top + 1 if self.lines else 0}-{last}/{len(self.lines)}"
        return list(self.lines[self.top : self.top + body]), position

    def frame(self, cols: int, rows: int, theme: Theme) -> str:
        """The alternate screen's rows for the current page that differ from the last frame, footer on the last row.

        "" when no row changed.
        """
        if self._size != (cols, rows):
            self.invalidate()
            self._size = (cols, rows)
        shown, position = self._page(cols, rows, theme)
        parts: list[str] = []
        body = self.body_rows(rows)
        for row in range(body):
            self._paint(row, shown[row] if row < len(shown) else (), cols, parts)
        footer = self.footer(position, theme)
        for at, line in enumerate(footer):
            if rows - len(footer) + at >= 0:  # a screen shorter than the footer keeps its last rows, over the page
                self._paint(rows - len(footer) + at, line, cols, parts)
        return SYNC_BEGIN + HIDE_CURSOR + "".join(parts) + SYNC_END if parts else ""

    def footer(self, position: str, theme: Theme) -> list[Line]:
        """The `FOOTER_ROWS` rows under the page: the title, where the page is, and the keys."""
        return [(Span(f" {self.title} · {position} · {FOOTER_KEYS} ", theme.status),)]

    def _paint(self, row: int, line: Line, cols: int, parts: list[str]) -> None:
        """Add screen row `row`'s bytes to `parts`, unless the last frame left this very line there."""
        held = self._painted.get(row)
        if held is not None and held[0] == line:
            return
        painted = paint_line(line, cols)
        self._painted[row] = (line, painted)
        parts.append(_move_to(row) + painted)


def _no_rows(index: int, width: int, theme: Theme) -> list[Line]:
    return []


@dataclass
class HistoryPager(Pager):
    """`/history`: the transcript's blocks, each rendered only when a page reaches it.

    `render(index, width, theme)` gives block `index`'s rows; a blank row goes
    above each block that has rows but the first such block. The page's top
    is a block and a row inside it (`item`, `offset`), so opening, moving and
    jumping to either end render only the blocks the page and the last page
    reach, at the frame's width. A move asked for before the first frame is
    made at it; after that each move is made at once, as `Pager` makes it.
    The footer counts messages (blocks), not rows. At most
    `HISTORY_BLOCKS_KEPT` blocks' rows are kept, for one width and theme, the
    last block's always once drawn (with a cap of one, it and one other).
    """

    lines: Sequence[Line] = ()
    count: int = 0
    render: Callable[[int, int, Theme], list[Line]] = _no_rows
    item: int = 0
    offset: int = 0
    _step: int = field(default=0, init=False, repr=False, compare=False)
    _to_end: bool | None = field(default=None, init=False, repr=False, compare=False)
    _kept: dict[int, list[Line]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _drawn: tuple[int, Theme] | None = field(default=None, init=False, repr=False, compare=False)
    # The width and theme the kept rows, the first block with rows and the last page's top are for.
    _kept_for: tuple[int, Theme] | None = field(default=None, init=False, repr=False, compare=False)
    _first: int | None = field(default=None, init=False, repr=False, compare=False)
    # The last page's top, found for (body rows, count) at `_kept_for`; every frame bounds by it.
    _last_page: tuple[tuple[int, int], tuple[int, int]] | None = field(default=None, init=False, repr=False, compare=False)

    def _scroll(self, step: int, rows: int) -> None:
        self._step += step
        self._settle_if_drawn(rows)

    def _scroll_to_end(self, bottom: bool, rows: int) -> None:
        self._step, self._to_end = 0, bottom
        self._settle_if_drawn(rows)

    def _settle_if_drawn(self, rows: int) -> None:
        if self._drawn is not None:
            self._settle(self._drawn[0], self.body_rows(rows), self._drawn[1])

    def _rows(self, index: int, cols: int, theme: Theme) -> list[Line]:
        """Block `index`'s rows at `cols`, the blank row above it included."""
        rows = self._rendered(index, cols, theme)
        return [(), *rows] if rows and index > self._first_with_rows(cols, theme) else rows

    def _rendered(self, index: int, cols: int, theme: Theme) -> list[Line]:
        """Block `index`'s rows at `cols`, rendered once while kept; the block used longest ago leaves first.

        The last block never leaves: it is the one block that can still
        change (a reply streaming while `/history` is open), and holding it
        keeps the page, and the last page's top found from it, as first drawn.
        When it is the only block kept and that fills the cache (a cap of
        one), nothing leaves, so one block is held beside it.
        """
        rows = self._kept.pop(index, None)
        if rows is None:
            rows = self.render(index, cols, theme)
            if len(self._kept) >= HISTORY_BLOCKS_KEPT:
                oldest = next((kept for kept in self._kept if kept != self.count - 1), None)
                if oldest is not None:
                    del self._kept[oldest]
        self._kept[index] = rows
        return rows

    def shows_nothing(self, cols: int, theme: Theme) -> bool:
        """Whether no block has a row at `cols`; the rows rendered to tell are kept for the first frame."""
        self._keep_for(cols, theme)
        return self._first_with_rows(cols, theme) >= self.count

    def _keep_for(self, cols: int, theme: Theme) -> None:
        """Let go of every kept row and position found at another width or theme."""
        if self._kept_for is None or self._kept_for[0] != cols or self._kept_for[1] is not theme:
            self._kept, self._first, self._last_page = {}, None, None
            self._kept_for = (cols, theme)

    def _first_with_rows(self, cols: int, theme: Theme) -> int:
        """The first block with any rows (`count` when none has any), found once per width and theme."""
        if self._first is None:
            index = 0
            while index < self.count and not self._rendered(index, cols, theme):
                index += 1
            self._first = index
        return self._first

    def _last_page_top(self, body: int, cols: int, theme: Theme) -> tuple[int, int]:
        """Where the page starts when its last row is the last block's last row (the top, for less than a page).

        Found once per body height and block count at the width and theme
        drawn, so a frame that moves nothing walks no block for it.
        """
        key = (body, self.count)
        if self._last_page is not None and self._last_page[0] == key:
            return self._last_page[1]
        index, total = self.count, 0
        while index > 0 and total < body:
            index -= 1
            total += len(self._rows(index, cols, theme))
        top = index, max(total - body, 0)
        self._last_page = (key, top)
        return top

    def _settle(self, cols: int, body: int, theme: Theme) -> None:
        """Make the moves asked for, then keep the top a real row no lower than the last page's."""
        if self._to_end is not None:
            self.item, self.offset = self._last_page_top(body, cols, theme) if self._to_end else (0, 0)
        self.offset += self._step
        self._step, self._to_end = 0, None
        while self.offset < 0 and self.item > 0:
            self.item -= 1
            self.offset += len(self._rows(self.item, cols, theme))
        self.offset = max(self.offset, 0)
        while self.item < self.count - 1 and self.offset >= len(self._rows(self.item, cols, theme)):
            self.offset -= len(self._rows(self.item, cols, theme))
            self.item += 1
        self.item, self.offset = min((self.item, self.offset), self._last_page_top(body, cols, theme))

    def _page(self, cols: int, rows: int, theme: Theme) -> tuple[list[Line], str]:
        self._keep_for(cols, theme)
        self._drawn = (cols, theme)
        body = self.body_rows(rows)
        self._settle(cols, body, theme)
        shown: list[Line] = []
        index, skip = self.item, self.offset
        while index < self.count and len(shown) < body:
            shown.extend(self._rows(index, cols, theme)[skip:])
            index, skip = index + 1, 0
        first = self.item + 1 if self.count else 0
        return shown[:body], f"{first}-{index}/{self.count} messages"
