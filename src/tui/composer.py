"""The composer: the multi-line input a prompt is typed into.

`Composer` is a state machine with no IO: `handle` takes one decoded key or
paste and returns what the app should do about it, `render` draws the text at
a width and reports where the caret is in cells, so the app parks the
hardware cursor there and an IME pre-edit (Korean, Japanese) shows in place.

Text is kept NFC and the caret is a str index on a cluster boundary
(`width.clusters`): a Hangul syllable, an emoji with a skin tone, a ZWJ family
or a flag is one step for the arrows and one deletion for Backspace/Delete.
A decomposed (NFD) paste is composed when it is inserted. Control characters
other than newline and tab are dropped on insert; a tab is kept and drawn as
`TAB_CELLS` spaces.

Rows wrap between words, as the transcript's echo of the sent prompt does: a
row that overflows ends before the last word that starts on it after a space
(or tab), so the spaces end the row and the next row starts with that word; a
word longer than a row breaks at the cell, cluster by cluster. Masked input
wraps by the cell alone. The spaces stay in the text and on the row they end,
so every caret position maps to exactly one (row, col). A continuation row
starts with a space only after a row that holds one word, or only spaces, and
has no room left for that space (a tab takes four cells): a space cannot hang
past the edge, where the caret before it would sit. A row filled to the last
cell is followed by an empty row, so the caret after it never sits past the
right edge. Continuation rows are indented by the prompt's width. When the
input needs more than `max_rows` rows, the visible window scrolls just enough
to keep the caret row inside it.

Keys: printable text inserts, with Alt too unless the chord came as a kitty or
modifyOtherKeys report (the legacy bytes `ESC x` are also Esc then x arriving in
one read, and the x is not lost with the Esc), except Alt+B and Alt+F, legacy
`ESC b` and `ESC f` or reported, which move a word back and forward as
readline's do (Terminal.app sends Option+Left/Right that way); Enter submits,
except right after a backslash, which it turns into a newline; Alt+Enter,
Shift+Enter and Ctrl+J insert a newline (Ctrl+Enter is Enter here; the app
decides what its submit means); Backspace/Delete remove one cluster;
Left/Right move one cluster, with Ctrl or Alt one word; Up/Down move one
visual row and browse history at the first and last row; Home/End and
Ctrl+A/Ctrl+E go to the start/end of the logical line; Ctrl+U/Ctrl+K kill to
the start/end of the line (at the edge, the newline itself); Ctrl+W,
Alt+Backspace and Ctrl+Backspace delete the word before the caret; a paste
inserts as is, newlines kept; Tab completes; Esc cancels and keeps the text;
Ctrl+C clears the text and cancels, or reports an interrupt when there is no
text; Ctrl+D exits on empty input and deletes forward otherwise; Ctrl+_ (what
Ctrl+-, Ctrl+/ and Ctrl+7 send on most terminals, or Ctrl+- reported as
itself) undoes the last edit and Ctrl+Y redoes it.

Undo steps back through the edits since the last submit, at most
`UNDO_CAP`: each key or paste that changed the text is one step, a run of
characters typed one after another at the caret is one step, and so is an
edit the app makes (`replace`, `restore`). A step puts back the text, its
paste chips and the caret. A new edit drops what redo held; a submit drops
both; masked input keeps none.

A paste of more than `PASTE_CHIP_LINES` lines or `PASTE_CHIP_CHARS`
characters (Claude Code's thresholds) goes in as one chip,
`[Pasted text #1 +60 lines]`, its text held in `pastes` (memory only, until
the composer empties). The chip is atomic: the caret steps over it and
never lands inside, and a deletion that reaches it (Backspace, Delete,
Ctrl+W, Ctrl+U, Ctrl+K) takes it whole. A submit sends the text with every
chip expanded where it stands and reports the compact text beside it
(`ComposerAction.shown`), which is what the transcript shows; history keeps
the expanded text. Masked input takes a paste as typed.

History is held oldest first, at most `HISTORY_CAP` entries; a submit
appends unless the text is blank or repeats the newest entry, or
`records_history` is off (the app turns it off while the composer answers a
question Hermes asked). Each entry it records is handed to `on_record` too,
which the app points at Hermes' history file (`history.py`). Masked input (secrets, sudo) draws one bullet per
cluster, refuses newlines, never browses history, never enters it, and is
shown as bullets by `repr` too.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import unicodedata

from .keys import (
    BACKSPACE,
    CHAR,
    DELETE,
    DOWN,
    END,
    ENTER,
    ESCAPE,
    HOME,
    LEFT,
    RIGHT,
    TAB,
    UP,
    Key,
    Paste,
)
from .style import DEFAULT_STYLE, Line, Span, Style
from .width import TAB_CELLS, cell_width, cluster_width, clusters, is_control

PROMPT = "❯ "
MASK = "•"
NEWLINE = "\n"
TAB_CHAR = "\t"
# What a row may end on: a space, as in the transcript's wrap (`width.wrap`), or a tab, drawn as spaces. A no-break
# space is not one.
WORD_BREAKS = frozenset({" ", TAB_CHAR})
# The most entries history holds, the oldest leaving first: as many as Hermes' own TUI keeps (ui-tui `history.ts`).
HISTORY_CAP = 1000
# A paste longer than this many lines, or this many characters, becomes one chip.
PASTE_CHIP_LINES = 3
PASTE_CHIP_CHARS = 800
PASTE_CHIP = "[Pasted text #{number} +{lines} lines]"
PASTE_CHIP_ONE_LINE = "[Pasted text #{number} · {chars:,} chars]"
UNDO_CAP = 100
UNDO_CHARS = frozenset("_-")  # with Ctrl
REDO_CHAR = "y"  # with Ctrl

NONE = "none"
SUBMIT = "submit"
CANCEL = "cancel"
INTERRUPT = "interrupt"
EXIT = "exit"
COMPLETE = "complete"


@dataclass(frozen=True)
class ComposerAction:
    kind: str
    text: str = ""  # the submitted text, for SUBMIT only, every paste chip expanded
    shown: str = ""  # the submitted text as the composer showed it, when chips made it differ from `text`
    chips: tuple[tuple[str, str], ...] = ()  # each chip in `shown`, its label and its text (`Composer.restore`)


NOTHING = ComposerAction(NONE)

# Readline's Alt+B and Alt+F, legacy or reported; the legacy bytes are also what stock macOS Terminal.app sends for
# Option+Left and Option+Right.
ALT_WORD_MOVES = {"b": LEFT, "f": RIGHT}


@dataclass(frozen=True)
class _Row:
    start: int  # cluster index of the row's first cluster
    end: int  # cluster index after the row's last cluster (a newline is never inside a row)
    hard: bool  # the row ends its logical line, so the position at `end` belongs to it


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


@dataclass(frozen=True)
class _Shape:
    """One text value measured: its clusters, their boundaries, and what each draws as and how wide."""

    text: str
    masked: bool
    units: list[str]
    bounds: list[int]  # str index of every cluster boundary, from 0 to len(text)
    shown: list[str]
    widths: list[int | None]  # `None` for a newline


def _shape(text: str, masked: bool) -> _Shape:
    units = clusters(text)
    bounds = [0]
    shown: list[str] = []
    widths: list[int | None] = []
    for unit in units:
        bounds.append(bounds[-1] + len(unit))
        if unit == NEWLINE:
            shown.append("")
            widths.append(None)
        elif masked:
            shown.append(MASK)
            widths.append(1)
        elif unit == TAB_CHAR:
            shown.append(" " * TAB_CELLS)
            widths.append(TAB_CELLS)
        else:
            shown.append(unit)
            widths.append(cluster_width(unit))
    return _Shape(text, masked, units, bounds, shown, widths)


def _cluster_index(bounds: Sequence[int], caret: int) -> int:
    """How many clusters lie before `caret`; a caret inside a cluster counts as its start."""
    return max(bisect_right(bounds, caret) - 1, 0)


def _clean(text: str, masked: bool) -> str:
    kept: list[str] = []
    for ch in text.replace("\r\n", NEWLINE).replace("\r", NEWLINE):
        if ch == NEWLINE:
            if not masked:
                kept.append(ch)
        elif ch == TAB_CHAR or not is_control(ch):
            kept.append(ch)
    return "".join(kept)


def _layout(widths: Sequence[int | None], avail: int, breaks: Sequence[bool]) -> list[_Row]:
    """Visual rows for clusters of these widths (`None` is a newline) at `avail` cells.

    `breaks[i]` says cluster i is a space or a tab a row may end on. A row
    that overflows ends before the last word on it that follows one, so the
    spaces end that row and the next starts with the word. With no such word
    it ends at the cell: a word longer than a row breaks there, and a space
    with no room left after a row of one word, or of only spaces, starts the
    next row. After a word moves down, the cluster that overflowed is
    measured again on the word's new row.
    """
    rows: list[_Row] = []
    start = used = 0
    for index, width in enumerate(widths):
        if width is None:
            rows.extend(_close_line(start, index, used, avail))
            start, used = index + 1, 0
            continue
        while used and used + width > avail:
            words = range(index, start, -1) if breaks else ()
            cut = next((at for at in words if breaks[at - 1] and not breaks[at]), index)
            rows.append(_Row(start, cut, hard=False))
            start, used = cut, sum(cells or 0 for cells in widths[cut:index])
        used += width
    rows.extend(_close_line(start, len(widths), used, avail))
    return rows


def _close_line(start: int, end: int, used: int, avail: int) -> list[_Row]:
    if used >= avail:  # full to the edge: the position after it gets a row of its own
        return [_Row(start, end, hard=False), _Row(end, end, hard=True)]
    return [_Row(start, end, hard=True)]


def _locate(rows: Sequence[_Row], widths: Sequence[int | None], index: int) -> tuple[int, int]:
    """(visual row, cells from the row start) of the position before cluster `index`."""
    for number, row in enumerate(rows):
        if row.start <= index < row.end or (index == row.end and row.hard):
            return number, sum(width or 0 for width in widths[row.start : index])
    return len(rows) - 1, 0


def _position_at(row: _Row, widths: Sequence[int | None], col: int) -> int:
    """The rightmost position in `row` whose column is at most `col`."""
    last = row.end if row.hard else row.end - 1
    best = row.start
    used = 0
    for index in range(row.start, last):
        used += widths[index] or 0
        if used > col:
            break
        best = index + 1
    return best


def _word_start(units: Sequence[str], index: int) -> int:
    while index > 0 and units[index - 1].isspace():
        index -= 1
    while index > 0 and not units[index - 1].isspace():
        index -= 1
    return index


def _word_end(units: Sequence[str], index: int) -> int:
    while index < len(units) and units[index].isspace():
        index += 1
    while index < len(units) and not units[index].isspace():
        index += 1
    return index


@dataclass
class Composer:
    text: str = ""
    caret: int = 0
    history: list[str] = field(default_factory=list)
    history_index: int | None = None
    masked: bool = False
    records_history: bool = True
    on_record: Callable[[str], None] | None = field(default=None, repr=False, compare=False)
    pastes: dict[str, str] = field(default_factory=dict, repr=False, compare=False)  # chip label -> its text
    _paste_count: int = field(default=0, init=False, repr=False, compare=False)
    _draft: str = field(default="", init=False, repr=False, compare=False)
    _avail: int | None = field(default=None, init=False, repr=False, compare=False)
    _scroll: int = field(default=0, init=False, repr=False, compare=False)
    # The text measured, and its rows at one width: a caret move or a redraw reuses them until the text changes.
    _shaped: _Shape | None = field(default=None, init=False, repr=False, compare=False)
    _laid: tuple[_Shape, int, list[_Row]] | None = field(default=None, init=False, repr=False, compare=False)
    # Undo and redo: (text, caret, paste chips) before each step, newest last.
    _undo: list[tuple[str, int, dict[str, str]]] = field(default_factory=list, init=False, repr=False, compare=False)
    _redo: list[tuple[str, int, dict[str, str]]] = field(default_factory=list, init=False, repr=False, compare=False)
    _typing_at: int | None = field(default=None, init=False, repr=False, compare=False)  # where the typed run's caret is

    def __post_init__(self) -> None:
        self._set(self.text, min(max(self.caret, 0), len(self.text)))

    def __repr__(self) -> str:
        text = MASK * len(clusters(self.text)) if self.masked else self.text
        return f"Composer(text={text!r}, caret={self.caret}, masked={self.masked})"

    # -- editing primitives ---------------------------------------------------

    def _measured(self) -> _Shape:
        """The text's `_Shape`, worked out again only when the text or the mask changed."""
        shape = self._shaped
        if shape is None or shape.masked != self.masked or shape.text != self.text:
            shape = self._shaped = _shape(self.text, self.masked)
        return shape

    def _rows(self, avail: int) -> tuple[_Shape, list[_Row]]:
        """The measured text and its visual rows at `avail` cells, laid out again only when either changed."""
        shape = self._measured()
        laid = self._laid
        if laid is None or laid[0] is not shape or laid[1] != avail:
            breaks = [] if shape.masked else [unit in WORD_BREAKS for unit in shape.units]
            laid = self._laid = (shape, avail, _layout(shape.widths, avail, breaks))
        return shape, laid[2]

    def _set(self, text: str, caret: int) -> None:
        """Store `text` as NFC with the caret moved to the matching cluster boundary."""
        full = _nfc(text)
        self.text = full
        bounds = self._measured().bounds
        self.caret = bounds[_cluster_index(bounds, min(len(_nfc(text[:caret])), len(full)))]

    def _replace(self, start: int, end: int, insert: str = "") -> None:
        if self.pastes and start < end:  # a deletion that reaches a chip takes it whole
            for chip_start, chip_end in self._chip_spans():
                if chip_start < end and chip_end > start:
                    start, end = min(start, chip_start), max(end, chip_end)
        self._set(self.text[:start] + insert + self.text[end:], start + len(insert))
        if self.pastes:
            self.pastes = {label: text for label, text in self.pastes.items() if label in self.text}
            self._caret_off_chip(forward=True)

    def _chip_spans(self) -> list[tuple[int, int]]:
        """Every chip's (start, end) in the text, in order."""
        spans: list[tuple[int, int]] = []
        for label in self.pastes:
            index = self.text.find(label)
            while index >= 0:
                spans.append((index, index + len(label)))
                index = self.text.find(label, index + len(label))
        return sorted(spans)

    def _caret_off_chip(self, forward: bool) -> None:
        """A caret that fell inside a chip goes to its end (`forward`) or its start."""
        for start, end in self._chip_spans():
            if start < self.caret < end:
                self.caret = end if forward else start
                return

    def expanded(self, text: str | None = None) -> str:
        """`text` (the composer's own by default) with every chip replaced by the text it holds."""
        text = self.text if text is None else text
        if not self.pastes:
            return text
        parts: list[str] = []
        done = 0
        for start, end in self._chip_spans() if text == self.text else []:
            parts.append(text[done:start])
            parts.append(self.pastes[text[start:end]])
            done = end
        parts.append(text[done:])
        return "".join(parts)

    def _insert_paste(self, text: str) -> None:
        """A paste: a long one (and not masked) as a chip, anything else as typed.

        A chip's number is one no label in the text has, chips `restore` put back included.
        """
        cleaned = _clean(text, self.masked)
        lines = len(cleaned.rstrip(NEWLINE).split(NEWLINE))
        if self.masked or not (lines > PASTE_CHIP_LINES or len(cleaned) > PASTE_CHIP_CHARS):
            self._insert(text)
            return
        label = ""
        while not label or label in self.text:
            self._paste_count += 1
            label = (
                PASTE_CHIP.format(number=self._paste_count, lines=lines)
                if lines > 1
                else PASTE_CHIP_ONE_LINE.format(number=self._paste_count, chars=len(cleaned))
            )
        self.pastes[label] = cleaned
        self._replace(self.caret, self.caret, label)

    def _insert(self, text: str) -> None:
        cleaned = _clean(text, self.masked)
        if cleaned:
            self._replace(self.caret, self.caret, cleaned)

    def replace(self, start: int, end: int, text: str) -> None:
        """Put `text` (cleaned as a paste is) in place of `self.text[start:end]`; the caret lands after it. One undo step."""
        before = self._state()
        self._replace(start, end, _clean(text, self.masked))
        self._step(before, typed=False)

    def restore(self, shown: str, chips: Sequence[tuple[str, str]]) -> None:
        """Put back, at the caret, a text a submit showed as `shown`: its `chips` (label, text) are chips again."""
        before = self._state()
        self.pastes.update(chips)
        self._replace(self.caret, self.caret, _clean(shown, self.masked))
        self._step(before, typed=False)

    def _state(self) -> tuple[str, int, dict[str, str]]:
        return self.text, self.caret, dict(self.pastes)

    def _step(self, before: tuple[str, int, dict[str, str]], typed: bool) -> None:
        """Keep `before` as an undo step when the text changed since, unless it continues a typed run."""
        if before[0] == self.text:
            if before[1] != self.caret:
                self._typing_at = None  # a caret move ends the run
            return
        if not self.masked and not (typed and self._typing_at == before[1]):
            self._undo.append(before)
            del self._undo[:-UNDO_CAP]
        self._redo.clear()
        self._typing_at = self.caret if typed else None

    def _take_step(self, source: list[tuple[str, int, dict[str, str]]], target: list[tuple[str, int, dict[str, str]]]) -> None:
        """Undo (from `_undo`) or redo (from `_redo`) one step; the state left goes on the other list."""
        if not source:
            return
        target.append(self._state())
        text, caret, pastes = source.pop()
        self.pastes = dict(pastes)
        self._set(text, caret)
        self._typing_at = None
        self.history_index = None  # the text is no history entry any more
        self._draft = ""

    def _units(self) -> tuple[list[str], list[int], int]:
        """Clusters, their boundaries, and the caret's cluster index."""
        shape = self._measured()
        return shape.units, shape.bounds, _cluster_index(shape.bounds, self.caret)

    def _load(self, text: str) -> None:
        self._set(text, len(text))
        self._scroll = 0

    def _clear(self) -> None:
        self._load("")
        self.history_index = None
        self._draft = ""
        self.pastes = {}
        self._paste_count = 0

    # -- key handling ---------------------------------------------------------

    def handle(self, event: Key | Paste) -> ComposerAction:
        if isinstance(event, Key) and event.name == CHAR and event.ctrl and not event.alt:
            if event.char in UNDO_CHARS:
                self._take_step(self._undo, self._redo)
                return NOTHING
            if event.char == REDO_CHAR:
                self._take_step(self._redo, self._undo)
                return NOTHING
        before = self._state()
        action = self._handle(event)
        if action.kind == SUBMIT:
            self._undo.clear()
            self._redo.clear()
            self._typing_at = None
        else:
            typed = isinstance(event, Key) and event.name == CHAR and not event.ctrl and not event.alt
            self._step(before, typed)
        return action

    def _handle(self, event: Key | Paste) -> ComposerAction:
        if isinstance(event, Paste):
            self._insert_paste(event.text)
            return NOTHING
        if event.name == ENTER:
            adds_line = event.alt or event.shift
            if not adds_line and not self.masked and self.text[self.caret - 1 : self.caret] == "\\":
                self._replace(self.caret - 1, self.caret, NEWLINE)
                return NOTHING
            if adds_line:
                self._insert(NEWLINE)
                return NOTHING
            return self._submit()
        if event.name == CHAR:
            if event.ctrl:
                return self._control(event.char)
            if event.alt and event.char in ALT_WORD_MOVES:
                self._move_or_delete(Key(ALT_WORD_MOVES[event.char], alt=True))
            elif not (event.alt and event.protocol):
                self._insert(event.char)
            return NOTHING
        if event.name == TAB:
            return NOTHING if event.shift else ComposerAction(COMPLETE)
        if event.name == ESCAPE:
            return ComposerAction(CANCEL)
        if event.name in (UP, DOWN):
            self._vertical(-1 if event.name == UP else 1)
            return NOTHING
        self._move_or_delete(event)
        return NOTHING

    def _move_or_delete(self, key: Key) -> None:
        units, bounds, index = self._units()
        by_word = key.ctrl or key.alt
        if key.name == BACKSPACE and index > 0:
            self._replace(bounds[_word_start(units, index) if by_word else index - 1], self.caret)
        elif key.name == DELETE and index < len(units):
            self._replace(self.caret, bounds[index + 1])
        elif key.name == LEFT:
            self.caret = bounds[_word_start(units, index) if by_word else max(index - 1, 0)]
            self._caret_off_chip(forward=False)
        elif key.name == RIGHT:
            self.caret = bounds[_word_end(units, index) if by_word else min(index + 1, len(units))]
            self._caret_off_chip(forward=True)
        elif key.name == HOME:
            self.caret = self._line_start()
        elif key.name == END:
            self.caret = self._line_end()

    def _control(self, char: str) -> ComposerAction:
        if char == "c":
            if not self.text:
                return ComposerAction(INTERRUPT)
            self._clear()
            return ComposerAction(CANCEL)
        if char == "d":
            if not self.text:
                return ComposerAction(EXIT)
            self._move_or_delete(Key(DELETE))
        elif char == "j":
            self._insert(NEWLINE)
        elif char == "a":
            self.caret = self._line_start()
        elif char == "e":
            self.caret = self._line_end()
        elif char == "u":
            start = self._line_start()
            self._replace(start - 1 if start == self.caret and start > 0 else start, self.caret)
        elif char == "k":
            end = self._line_end()
            self._replace(self.caret, end + 1 if end == self.caret and end < len(self.text) else end)
        elif char == "w":
            units, bounds, index = self._units()
            self._replace(bounds[_word_start(units, index)], self.caret)
        return NOTHING

    def _line_start(self) -> int:
        return self.text.rfind(NEWLINE, 0, self.caret) + 1

    def _line_end(self) -> int:
        end = self.text.find(NEWLINE, self.caret)
        return len(self.text) if end < 0 else end

    def record(self, text: str, sent: bool = True) -> None:
        """Keep `text`, trimmed as Hermes' history file keeps it, as the newest history entry, unless it is blank,
        repeats the newest, or nothing may be recorded.

        Only a `sent` line goes on to `on_record`; a draft the person cleared
        stays in this session's history alone.
        """
        entry = text.strip()
        if self.records_history and not self.masked and entry and (not self.history or self.history[-1] != entry):
            self.history.append(entry)
            del self.history[:-HISTORY_CAP]
            if sent and self.on_record is not None:
                self.on_record(entry)

    def _submit(self) -> ComposerAction:
        shown = self.text
        text = self.expanded()
        chips = tuple((label, pasted) for label, pasted in self.pastes.items() if label in shown)
        self.record(text)
        self._clear()
        return ComposerAction(SUBMIT, text, shown if shown != text else "", chips)

    # -- rows and history -----------------------------------------------------

    def _vertical(self, step: int) -> None:
        if self._avail is not None:
            shape, rows = self._rows(self._avail)
        else:  # never rendered: logical lines only
            shape = self._measured()
            shape, rows = self._rows(sum(width or 0 for width in shape.widths) + 1)
        widths, bounds = shape.widths, shape.bounds
        row, col = _locate(rows, widths, _cluster_index(bounds, self.caret))
        target = row + step
        if 0 <= target < len(rows):
            self.caret = bounds[_position_at(rows[target], widths, col)]
            self._caret_off_chip(forward=True)
        elif step < 0:
            self._history_back()
        else:
            self._history_forward()

    def _history_back(self) -> None:
        if self.masked or not self.history or self.history_index == 0:
            return
        if self.history_index is None:
            self._draft = self.text
            self.history_index = len(self.history) - 1
        else:
            self.history_index -= 1
        self._load(self.history[self.history_index])

    def _history_forward(self) -> None:
        if self.masked or self.history_index is None:
            return
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self._load(self.history[self.history_index])
            return
        self.history_index = None
        self._load(self._draft)
        self._draft = ""

    # -- drawing --------------------------------------------------------------

    def render(
        self,
        width: int,
        max_rows: int,
        prompt: str = PROMPT,
        prompt_style: Style = DEFAULT_STYLE,
        chip_style: Style = DEFAULT_STYLE,
        marks: Sequence[tuple[int, int]] = (),
        mark_style: Style = DEFAULT_STYLE,
        glow: Sequence[tuple[int, int]] = (),
        glow_style: Callable[[int, int], Style] | None = None,
    ) -> tuple[list[Line], tuple[int, int]]:
        """The visible rows at `width` cells, and the caret as (row, col) within them; paste chips in `chip_style`,
        the `marks` (str index ranges of the text, such as a `/name` that resolves) in `mark_style`, and the `glow`
        ranges (`aura`) a style a character, `glow_style(offset, length)` with `offset` from the range's start in a range
        `length` long, over a mark."""
        prompt_cells = cell_width(prompt)
        avail = max(width - prompt_cells, 1)
        self._avail = avail
        shape, rows = self._rows(avail)
        shown, widths = shape.shown, shape.widths
        row, col = _locate(rows, widths, _cluster_index(shape.bounds, self.caret))
        visible = max(max_rows, 1)
        scroll = min(max(self._scroll, row - visible + 1), row)
        scroll = max(min(scroll, len(rows) - visible), 0)
        self._scroll = scroll
        indent = Span(" " * prompt_cells)
        chips = self._chip_spans() if self.pastes and not self.masked else []
        lines: list[Line] = []
        for number in range(scroll, min(len(rows), scroll + visible)):
            lead = Span(prompt, prompt_style) if number == 0 else indent
            start, end = rows[number].start, rows[number].end
            if chips or marks or glow:
                body_spans = _chip_styled(shown, shape.bounds, start, end, chips, chip_style, marks, mark_style,
                                          glow, glow_style)
                lines.append((lead, *body_spans))
                continue
            body = "".join(shown[start:end])
            lines.append((lead, Span(body)) if body else (lead,))
        return lines, (row - scroll, prompt_cells + col)


def _chip_styled(
    shown: Sequence[str], bounds: Sequence[int], start: int, end: int, chips: Sequence[tuple[int, int]], style: Style,
    marks: Sequence[tuple[int, int]] = (), mark_style: Style = DEFAULT_STYLE,
    glow: Sequence[tuple[int, int]] = (), glow_style: Callable[[int, int], Style] | None = None,
) -> list[Span]:
    """Clusters `start` to `end` as spans, the ones inside a chip (by str index) in `style`, inside a glow range in
    `glow_style` of their offset in it, inside a mark in `mark_style`."""
    spans: list[Span] = []
    for index in range(start, end):
        inside = any(chip_start <= bounds[index] < chip_end for chip_start, chip_end in chips)
        lit = None if inside or glow_style is None else next(
            (glow_style(bounds[index] - lit_start, lit_end - lit_start) for lit_start, lit_end in glow
             if lit_start <= bounds[index] < lit_end),
            None)
        marked = not inside and any(mark_start <= bounds[index] < mark_end for mark_start, mark_end in marks)
        wanted = style if inside else lit if lit is not None else mark_style if marked else DEFAULT_STYLE
        if spans and spans[-1].style == wanted:
            spans[-1] = Span(spans[-1].text + shown[index], wanted)
        else:
            spans.append(Span(shown[index], wanted))
    return spans
