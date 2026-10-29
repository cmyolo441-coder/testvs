"""Terminal UI: persistent double-line prompt box, live streaming, overlays.

Architecture: ONE prompt_toolkit Application runs for the whole session.
The double-line box stays pinned at the bottom at all times; all output
(user echo, streamed tokens, tool lines, errors) scrolls above it through
prompt_toolkit's patch_stdout. The box border carries live state:

    ╭─ FullAgent ──●── Union Alpha ── ◈ high ──⬢ L3── ◎ goal ▰▰▰▱▱ 60% ──◉ ctx 12% ─╮
    │ ❯ user types here…                                                       │
    ╰ ⏎ send · ⇧⏎ newline · / cmds · ^T models · ^E effort ────────────────────╯

Top border: provider dot (provider colour), FREE/FAST tag, effort
(effort colour), autonomy badge ⬢ (dim→green→yellow→red with level),
goal progress bar ◎ ▰▱ + %, focus counter, live context meter ◉.
Bottom border: live spinner + elapsed seconds while busy, ✦ flash
messages, coloured [y]es/[n]o/[a]lways approval bar, styled key hints
when idle. Overlays (model / effort / help / history) render directly
above the box and are navigated with ↑↓, PgUp/PgDn, Tab, Home/End.

Qoder-style chrome (``fullagent.cliui``) wraps the box:

    ≡ 4 Agents (4 running) ───────────────────────── ↓ to manage
     ◉ Explore · Read fullagent/autopilot.py
    COLLAB  ⇧Tab pilot · ◎ 4 running · 0 queued ───── 0 skills
    ╭─ FullAgent ──●── Union Alpha ── ◈ high ──⬢ L3──◉ ctx 12%──╮
    │ ❯ user types here…                                        │
    ╰ ⏎ send · ⇧⏎ newline · / cmds · ^T models · ^E effort ─────╯
     ● Union Alpha · high · 262k context · ctx ░░░░░░░░░░ 12% · ~/repo
                              ◎ feature/x · ◎ 4 agents | 4 running

The board only appears while a swarm is queued/running; the HUD carries
the autonomy mode (Shift+Tab toggles Pilot) and the skills counter; the
footer carries model/effort/context/cwd plus the agent count. Ctrl+O
expands the last tool block or Thinking panel instead of its "+N rows"
preview.
"""

from __future__ import annotations

import difflib
import os
import shutil
import re
import sys
import threading
import time
import unicodedata
from collections import deque
from html import escape as html_escape
from pathlib import Path
from typing import Callable

try:
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.enums import DEFAULT_BUFFER
    from prompt_toolkit.filters import Condition, has_focus
    from prompt_toolkit.formatted_text import (
        HTML,
        fragment_list_to_text,
        to_formatted_text,
    )
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import (
        BufferControl,
        CompletionsMenu,
        ConditionalContainer,
        Dimension,
        Float,
        FloatContainer,
        FormattedTextControl,
        HSplit,
        Layout,
        VSplit,
        Window,
    )
    from prompt_toolkit.search import start_search
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import SearchToolbar
except ImportError:  # Headless / non-interactive fallback when prompt_toolkit is absent
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def patch_stdout(*_a, **_kw):
        yield

    class _DummyApp:
        is_running = False
        def invalidate(self): pass
        def exit(self, *_a, **_kw): pass
        def run(self, *_a, **_kw): pass

    def get_app():
        return _DummyApp()

    class Application(_DummyApp):
        def __init__(self, *_a, **_kw): pass

    class AutoSuggestFromHistory:
        pass

    class Buffer:
        def __init__(self, *_a, **_kw):
            self.text = ""
            self.cursor_position = 0
        def reset(self, *_a, **_kw):
            self.text = ""
            self.cursor_position = 0

    class Completer:
        pass

    class Completion:
        def __init__(self, text="", start_position=0, display=None, display_meta=None, style=""):
            self.text = text
            self.start_position = start_position
            self.display = display or text
            self.display_meta = display_meta or ""
            self.style = style

    DEFAULT_BUFFER = "DEFAULT_BUFFER"

    class Condition:
        def __init__(self, fn=None): self.fn = fn or (lambda: False)
        def __call__(self): return bool(self.fn())
        def __and__(self, other): return Condition(lambda: self() and other())
        def __or__(self, other): return Condition(lambda: self() or other())
        def __invert__(self): return Condition(lambda: not self())

    def has_focus(*_a, **_kw):
        return Condition(lambda: False)

    class HTML:
        def __init__(self, value=""): self.value = str(value)
        def __str__(self): return self.value

    def to_formatted_text(val, *_a, **_kw):
        if isinstance(val, list):
            return val
        text = getattr(val, "value", str(val))
        text = re.sub(r"<[^>]+>", "", text)
        return [("", text)]

    def fragment_list_to_text(fragments):
        return "".join(t for _, t in (fragments or []))

    class FileHistory:
        def __init__(self, filename=""): self.filename = filename
        def get_strings(self): return []
        def append_string(self, s): pass

    class KeyBindings:
        def add(self, *_a, **_kw):
            def _dec(fn): return fn
            return _dec

    class Keys:
        Escape = "escape"
        ControlC = "c-c"
        ControlD = "c-d"
        ControlT = "c-t"
        ControlE = "c-e"
        ControlR = "c-r"
        ControlL = "c-l"
        Up = "up"
        Down = "down"
        PageUp = "pageup"
        PageDown = "pagedown"
        Home = "home"
        End = "end"
        Tab = "tab"
        BackTab = "s-tab"
        Enter = "enter"
        Any = "<any>"

    class _LayoutBox:
        def __init__(self, *_a, **_kw): pass

    BufferControl = CompletionsMenu = ConditionalContainer = Dimension = _LayoutBox
    Float = FloatContainer = FormattedTextControl = HSplit = Layout = VSplit = Window = _LayoutBox

    def start_search(*_a, **_kw): pass

    class Style:
        @classmethod
        def from_dict(cls, d): return cls()

    class SearchToolbar(_LayoutBox):
        pass
from rich.console import Console
from rich.markdown import CodeBlock, Heading, Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from . import config
from . import cliui
from . import fx
from .agent import Agent, ToolEvent
from .config import (
    APP_NAME,
    EFFORTS,
    MODELS,
    PROVIDERS,
    Config,
    Effort,
    Model,
    effort_by_key,
    model_by_id,
)
from .tools import Tool


class SafeFileHistory(FileHistory):
    """FileHistory that can never take the UI down.

    prompt_toolkit calls store_string() from inside the event loop when the
    user presses Enter; if the history file's directory is missing (fresh
    install, home dir deleted mid-session, fresh mount) the base class
    raises FileNotFoundError straight into the event loop. This subclass
    recreates the directory and swallows any persistence error — input
    history is a convenience, never a crash path."""

    def load_history_strings(self):
        try:
            yield from super().load_history_strings()
        except OSError:
            return

    def store_string(self, string: str) -> None:
        try:
            super().store_string(string)
        except FileNotFoundError:
            try:
                Path(self.filename).parent.mkdir(parents=True, exist_ok=True)
                super().store_string(string)
            except OSError:
                pass
        except OSError:
            pass

# ---------------------------------------------------------------------------
# Next-Gen Multi-Theme Palette Engine (Neon Cyber-Dracula default)
# ---------------------------------------------------------------------------

THEMES: dict[str, dict[str, str]] = {
    "neon": {
        "border": "#5b6eae",
        "accent": "#c084fc",
        "cyan": "#22d3ee",
        "green": "#34d399",
        "yellow": "#facc15",
        "orange": "#fb923c",
        "red": "#f87171",
        "pink": "#f472b6",
        "fg": "#f8fafc",
        "dim": "#64748b",
    },
    "dracula": {
        "border": "#6272a4",
        "accent": "#bd93f9",
        "cyan": "#8be9fd",
        "green": "#50fa7b",
        "yellow": "#f1fa8c",
        "orange": "#ffb86c",
        "red": "#ff5555",
        "pink": "#ff79c6",
        "fg": "#f8f8f2",
        "dim": "#6272a4",
    },
    "nord": {
        "border": "#4c566a",
        "accent": "#b48ead",
        "cyan": "#88c0d0",
        "green": "#a3be8c",
        "yellow": "#ebcb8b",
        "orange": "#d08770",
        "red": "#bf616a",
        "pink": "#b48ead",
        "fg": "#eceff4",
        "dim": "#616e88",
    },
    "emerald": {
        "border": "#059669",
        "accent": "#10b981",
        "cyan": "#06b6d4",
        "green": "#4ade80",
        "yellow": "#fde047",
        "orange": "#fb923c",
        "red": "#f43f5e",
        "pink": "#ec4899",
        "fg": "#f0fdf4",
        "dim": "#6b7280",
    },
}

C = dict(THEMES["neon"])

STYLE = Style.from_dict({
    "box": C["border"],
    "box.title": f"bold {C['accent']}",
    "box.model": f"bold {C['cyan']}",
    "box.tag": f"bold {C['green']}",
    "box.effort": "bold",
    "box.session": C["dim"],
    "box.hint": C["dim"],
    "box.status": f"bold {C['cyan']}",
    "box.spinner": f"bold {C['accent']}",
    "box.flash": f"bold {C['yellow']}",
    "box.approve": f"bold {C['yellow']}",
    "box.key": f"bold {C['cyan']}",
    "box.sep": C["dim"],
    "box.goalbar": f"bold {C['green']}",
    "arrow": f"bold {C['green']}",
    "cont": C["dim"],
    "stream.preview": C["fg"],
    "overlay.border": C["accent"],
    "overlay.item": C["fg"],
    "overlay.item.selected": f"bg:#44475a bold {C['fg']}",
    "overlay.footer": C["dim"],
    "completion-menu.completion": "bg:#282a36 #f8f8f2",
    "completion-menu.completion.current": "bg:#44475a #ffffff bold",
    "completion-menu.meta.completion": "bg:#282a36 #6272a4",
    "completion-menu.meta.completion.current": "bg:#44475a #8be9fd",
    "completion-menu.multi-column-meta": "bg:#282a36 #6272a4",
    "scrollbar.background": "bg:#282a36",
    "scrollbar.button": "bg:#6272a4",
})

EFFORT_COLORS = {e.key: e.color for e in EFFORTS}
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ---------------------------------------------------------------------------
# Display-width helpers.
#
# The prompt box borders must NEVER exceed the terminal width: if the
# closing ╮/╯ wraps to the next line, prompt_toolkit's cursor math breaks
# and the whole render corrupts. len() counts codepoints, not columns —
# emoji like 🎯 occupy TWO columns, so every border-length computation in
# this file must go through _disp_width, and every truncation through
# _truncate_w (a naive s[:n] can split a wide char in half, printing a
# replacement glyph and throwing the border off by one).
# ---------------------------------------------------------------------------

_WIDE_CATS = ("W", "F")


def _disp_width(s: str) -> int:
    """Terminal column width of s (wide East-Asian chars count double)."""
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in _WIDE_CATS else 1
    return w


def _truncate_w(s: str, maxw: int) -> str:
    """Truncate s to at most maxw columns, appending … when truncated.
    Never splits a wide character."""
    if maxw <= 0:
        return ""
    if _disp_width(s) <= maxw:
        return s
    out: list[str] = []
    w = 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in _WIDE_CATS else 1
        if w + cw > maxw - 1:  # reserve one column for the ellipsis
            break
        out.append(ch)
        w += cw
    return "".join(out) + "…"


# ---------------------------------------------------------------------------
# Rich markdown rendering tweaks (used for non-streamed output)
# ---------------------------------------------------------------------------


class _Heading(Heading):
    def __rich_console__(self, console, options):
        yield Text(f"◆ {self.plain}", style=f"bold {C['accent']}")


class _CodeBlock(CodeBlock):
    def __rich_console__(self, console, options):
        code = str(self.text).rstrip("\n")
        yield Syntax(code, self.lexer_name or "text", theme="monokai",
                     background_color="default", word_wrap=True, padding=(0, 1))
        yield Text()


Markdown.elements["fence"] = _CodeBlock
Markdown.elements["code_block"] = _CodeBlock
for _h in ("h1", "h2", "h3", "h4", "h5", "h6"):
    Markdown.elements[_h] = _Heading


class _StdoutProxy:
    """Write to whatever sys.stdout currently is.

    During app.run(), prompt_toolkit's patch_stdout swaps sys.stdout for a
    proxy that knows how to interleave output with the pinned box render.
    If rich wrote to the raw stdout instead, prompt_toolkit's cursor math
    would break and redraws would overwrite already-streamed text.
    """

    @property
    def encoding(self):
        return getattr(sys.stdout, "encoding", None) or "utf-8"

    @property
    def closed(self):
        return getattr(sys.stdout, "closed", False)

    def write(self, s: str):
        return sys.stdout.write(s)

    def flush(self):
        sys.stdout.flush()

    def isatty(self) -> bool:
        try:
            return sys.stdout.isatty()
        except Exception:
            return True


def make_console() -> Console:
    # lock in terminal-ness now; patch_stdout later swaps sys.stdout for a
    # proxy that would otherwise make rich think it lost its terminal
    is_tty = True
    try:
        is_tty = sys.stdout.isatty()
    except Exception:
        pass
    return Console(file=_StdoutProxy(), highlight=False, soft_wrap=False,
                   force_terminal=is_tty)


# ---------------------------------------------------------------------------
# Slash-command completion
# ---------------------------------------------------------------------------

from .slashcatalog import SLASH_COMMANDS


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or "\n" in text:
            return
        word = text.lower()
        if word.startswith("/effort ") or word == "/effort":
            arg = word[len("/effort"):].strip()
            for e in EFFORTS:
                if e.key.startswith(arg):
                    yield Completion(f"/effort {e.key}",
                                     start_position=-len(text),
                                     display_meta=e.description)
            return
        if word.startswith("/goal ") or word == "/goal":
            arg = word[len("/goal"):].strip()
            for sub, meta in (
                    ("set", "set <statement> | <clause> | <clause> …"),
                    ("prove", "prove <clause-id> — run its own predicate"),
                    ("prove-all", "prove every clause from scratch"),
                    ("close", "closure ritual — compute the terminal state"),
                    ("status", "the Goal Compass: distance, velocity, focus"),
                    ("waive", "waive <clause-id> --reason '…' (human waiver)"),
                    ("clear", "deactivate the goal")):
                if sub.startswith(arg):
                    yield Completion(f"/goal {sub}",
                                     start_position=-len(text),
                                     display_meta=meta)
            return
        if word.startswith("/autonomy ") or word == "/autonomy":
            arg = word[len("/autonomy"):].strip()
            from .agent import AUTONOMY_LEVELS
            for level, desc in AUTONOMY_LEVELS.items():
                if str(level).startswith(arg):
                    yield Completion(f"/autonomy {level}",
                                     start_position=-len(text),
                                     display_meta=desc)
            return
        if word.startswith("/judge ") or word == "/judge":
            arg = word[len("/judge"):].strip()
            for ptype, meta in (
                    ("exit_code", '{"command": "pytest -q", "expect": 0}'),
                    ("file_exists", '{"path": "src/x.py"}'),
                    ("file_contains", '{"path": "src/x.py", "text": "def foo"}'),
                    ("file_matches", '{"path": "src/x.py", "pattern": "…"}'),
                    ("command_output_contains",
                     '{"command": "python -V", "text": "Python"}')):
                if ptype.startswith(arg):
                    yield Completion(f"/judge {ptype} ",
                                     start_position=-len(text),
                                     display_meta=meta)
            return
        for name, meta in SLASH_COMMANDS:
            if name.startswith(word):
                yield Completion(name, start_position=-len(text),
                                 display_meta=meta)


# ---------------------------------------------------------------------------
# Live-write tracker — true live file writing
# ---------------------------------------------------------------------------


class _LiveWrite:
    """Incrementally pulls a growing string field out of a streaming
    tool-call's JSON arguments, so a file change can be shown happening
    line-by-line WHILE the model generates it (a true live write/edit,
    not a post-hoc replay).

    `key` selects which JSON string field to stream: "content" for
    write_file, "new_string" for edit_file.

    Robust by design: if the JSON is shaped unexpectedly, `feed` simply
    returns no lines and the caller falls back to the normal cascade.
    Nothing here can break the turn."""

    _PATH_KEY = re.compile(r'"path"\s*:\s*"')

    def __init__(self, key: str = "content"):
        self._CONTENT_KEY = re.compile(
            r'"' + re.escape(key) + r'"\s*:\s*"')
        self.buf = ""            # accumulated argument JSON so far
        self.content_start = -1  # index into buf of first content char
        self.raw_pos = 0         # content raw chars already consumed
        self.pending = ""        # unescaped text not yet emitted as lines
        self.lines = 0           # full lines emitted so far
        self.done = False        # saw the closing quote of the content

    def path(self) -> str | None:
        return self._field(self.buf, "path")

    @classmethod
    def _field(cls, buf: str, key: str) -> str | None:
        """Return the value of a JSON string field if it is COMPLETE
        (closing quote seen), else None. Used for path / old_string."""
        m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', buf)
        if not m:
            return None
        val, _, done = cls._unescape(buf[m.end():])
        if not done:
            return None  # value still streaming — wait for closing quote
        return val

    def feed(self, chunk: str) -> list:
        """Feed a new argument chunk; returns the complete content lines that
        became available as a result.

        SPEED: the buffer grows to the full file being written. Slicing
        self.buf[self.content_start:] on every chunk re-copied ALL content
        received so far (O(n^2) over a streamed file); we now slice only
        the unconsumed tail. The content-key regex likewise resumes near
        the new tail instead of re-scanning from zero — a 128-char overlap
        covers any key spanning the chunk boundary."""
        self.buf += chunk
        if self.done:
            return []  # content fully captured; ignore the rest of the JSON
        if self.content_start < 0:
            search_from = max(0, len(self.buf) - len(chunk) - 128)
            m = self._CONTENT_KEY.search(self.buf, search_from)
            if not m:
                return []
            self.content_start = m.end()
        tail = self.buf[self.content_start + self.raw_pos:]
        text, consumed, done = self._unescape(tail)
        self.raw_pos += consumed
        self.done = done
        self.pending += text
        out = []
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.lines += 1
            out.append(line)
        return out

    def flush(self) -> str | None:
        """Return the final partial line (if any) once generation ends."""
        if self.pending:
            line, self.pending = self.pending, ""
            self.lines += 1
            return line
        return None

    @staticmethod
    def _unescape(s: str) -> tuple:
        """Unescape a JSON string fragment. Returns (text, consumed, done).
        Stops before a trailing incomplete escape; `done` is True when the
        unescaped closing quote of the value was reached."""
        out = []
        i = 0
        n = len(s)
        done = False
        while i < n:
            c = s[i]
            if c == '"':
                done = True
                i += 1
                break
            if c != '\\':
                out.append(c)
                i += 1
                continue
            if i + 1 >= n:
                break  # trailing backslash — wait for the next chunk
            e = s[i + 1]
            if e == 'u':
                if i + 6 > n:
                    break  # incomplete \\uXXXX — wait for more
                hexs = s[i + 2:i + 6]
                try:
                    cp = int(hexs, 16)
                except ValueError:
                    out.append('\\u' + hexs)
                    i += 6
                    continue
                if 0xD800 <= cp <= 0xDFFF:
                    # lone surrogate — emitting it would produce a char the
                    # terminal cannot encode (UnicodeEncodeError on print);
                    # keep the literal escape instead
                    out.append('\\u' + hexs)
                else:
                    out.append(chr(cp))
                i += 6
            else:
                mapping = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b',
                           'f': '\f', '"': '"', '\\': '\\', '/': '/'}
                out.append(mapping.get(e, '\\' + e))
                i += 2
        return ''.join(out), i, done


# ---------------------------------------------------------------------------
# Stream render throttle — cap per-token UI updates at ~10fps
# ---------------------------------------------------------------------------
# The throttle is pure logic (no prompt_toolkit imports) and lives in
# fullagent/streamthrottle.py so latency tests can drive the real class
# without a terminal. Re-exported here under the historical private name
# — fullagent.tui._StreamThrottle keeps working for existing imports.
from .streamthrottle import StreamThrottle as _StreamThrottle  # noqa: F401


# ---------------------------------------------------------------------------
# Overlay list (model / effort / help / history pickers)
# ---------------------------------------------------------------------------


class OverlayList:
    """A modal selection list rendered directly above the prompt box."""

    PAGE = 5
    WINDOW = 10

    def __init__(self, title: str, items: list[tuple[str, str]],
                 selected_index: int = 0,
                 on_select: Callable[[int], None] | None = None,
                 footer: str = ""):
        self.title = title
        self.items = items              # list of (html, meta)
        self.index = max(0, min(selected_index, len(items) - 1))
        self.on_select = on_select
        self.footer = footer
        self.visible = False
        self._top = max(0, self.index - self.WINDOW // 2)
        # SPEED: fragments() runs on every render frame while the overlay
        # is open — parsing the per-item HTML on each frame was the
        # hottest cost in the picker. Precompute the plain text once.
        self._plain = [self._strip_html(h) for h, _ in items]

    @staticmethod
    def _strip_html(html: str) -> str:
        # prompt_toolkit parses HTML() with an XML parser — raw text is
        # already escaped by callers (see open_history); this only strips
        # the <style> tags back to plain text for width math.
        return fragment_list_to_text(to_formatted_text(HTML(html)))

    def open(self) -> None:
        self.visible = True

    def close(self) -> None:
        self.visible = False

    def move(self, delta: int) -> None:
        if not self.items:
            return  # empty list — no-op instead of ZeroDivisionError
        self.index = (self.index + delta) % len(self.items)
        if self.index < self._top:
            self._top = self.index
        elif self.index >= self._top + self.WINDOW:
            self._top = self.index - self.WINDOW + 1

    def page(self, delta: int) -> None:
        self.move(delta * self.PAGE)

    def select(self) -> None:
        if not self.items:
            # empty list — never call on_select with a bogus index
            # (MODELS[-1] would silently pick the wrong model)
            self.close()
            return
        idx = self.index
        self.close()
        if self.on_select:
            self.on_select(idx)

    def fragments(self, width: int) -> list:
        # items are fixed at construction, but rebuild the cache if a
        # caller ever mutates them afterwards — stale widths would
        # misalign the box
        if len(self._plain) != len(self.items):
            self._plain = [self._strip_html(h) for h, _ in self.items]
        inner = max(30, width - 2)
        frags: list = []
        title = _truncate_w(f" {self.title} ", inner)
        frags.append(("class:overlay.border",
                      f"╔{title}{'═' * max(0, inner - _disp_width(title))}╗\n"))
        shown = self.items[self._top:self._top + self.WINDOW]
        for i, (html, meta) in enumerate(shown):
            real_i = self._top + i
            selected = real_i == self.index
            marker = "▶ " if selected else "  "
            line = _truncate_w(f"{marker}{self._plain[real_i]}",
                               inner - 2)
            pad = inner - _disp_width(line)
            style = ("class:overlay.item.selected" if selected
                     else "class:overlay.item")
            frags.append((style, f"║{line}{' ' * max(0, pad)}║\n"))
        foot = _truncate_w(self.footer or
                           "↑↓ PgUp PgDn Tab move · Enter select · Esc close",
                           inner)
        frags.append(("class:overlay.footer",
                      f"╚{foot}{'═' * max(0, inner - _disp_width(foot))}╝"))
        return frags


# ---------------------------------------------------------------------------
# The UI — one persistent application for the whole session
# ---------------------------------------------------------------------------


class UI:
    def __init__(self, cfg: Config, agent: Agent):
        self.cfg = cfg
        self.agent = agent
        self.console = make_console()
        self.overlay: OverlayList | None = None

        # turn state
        self._busy = False
        # Follow-up messages typed while a turn is running.  They are
        # drained strictly one at a time: a second agent.run_turn() must
        # never overlap the first over the shared message list.
        self._pending: deque[str] = deque()
        self._turn_lock = threading.Lock()
        # True from the moment a turn thread starts until it has fully
        # returned.  _busy may be force-cleared by the cancel watchdog,
        # but this stays set so a second run_turn cannot overlap history.
        self._turn_alive = False
        self._status_text = ""
        self._spinner_i = 0
        self._spinner_gen = 0
        self._spinner_on = False
        self._turn_start_ts = 0.0
        self._cancel_flag = threading.Event()
        self._last_flush = 0.0
        # SPEED: cached terminal size (queried at most twice a second)
        self._width_cache: int | None = None
        self._width_cache_ts = 0.0

        # approval state
        self._approve_request: tuple[Tool, dict, threading.Event] | None = None
        self._approve_result = False

        # flash message in the bottom border (generation-guarded: a
        # stale timer from an older flash can never clear a newer one)
        self._flash: tuple[str, str] | None = None
        self._flash_timer: threading.Timer | None = None
        self._flash_gen = 0

        # goal-status cache for the border (fold is not free per frame)
        self._goal_cache = None
        self._goal_cache_ts = 0.0

        # focus mode (deep work): remaining auto-continuation turns
        self._focus_remaining = 0

        # live context-usage cache for the border (est. tokens / window)
        self._ctx_cache: int | None = None
        self._ctx_cache_ts = 0.0

        # Next-Gen TUI telemetry: live token velocity & session diff history
        self._stream_tokens_count = 0
        self._diff_history: list[dict] = []
        # Qoder-style chrome: live agent board + expandable tool detail
        self._agents_cache: dict | None = None
        self._agents_cache_ts = 0.0
        self._agents_hooked = False
        self._branch_cache: str | None = None
        self._branch_cache_ts = 0.0
        self._auto_prev = 3
        self._detail_expanded = False
        # (payload, kind) — kind is "live" | "generic" | "thinking"
        self._last_detail_event: tuple | None = None
        self._theme_name = str(self.cfg.extra.get("theme", "neon"))
        if self._theme_name in THEMES:
            C.update(THEMES[self._theme_name])

        self._build()

        # vimmode: apply persisted vi/emacs editing preference
        try:
            from .vimmode import apply_vim_mode
            apply_vim_mode(self)
        except Exception:
            pass

    # -- small helpers ---------------------------------------------------------

    def _model(self) -> Model:
        m = model_by_id(self.cfg.model_id)
        if m is None:
            # stale config (model id removed upstream) — fall back instead
            # of asserting: an AssertionError here would propagate out of
            # the per-frame border render and take the whole session down
            m = model_by_id(config.DEFAULT_MODEL_ID) or MODELS[0]
        return m

    def _effort(self) -> Effort:
        e = effort_by_key(self.cfg.effort)
        if e is None:
            e = effort_by_key(config.DEFAULT_EFFORT) or EFFORTS[0]
        return e

    def _width(self) -> int:
        # SPEED: terminal size changes rarely; the query runs at most twice
        # a second (the streaming preview asks for it on every token chunk)
        now = time.time()
        if self._width_cache is not None and now - self._width_cache_ts < 0.5:
            return self._width_cache
        try:
            cols = get_app().output.get_size().columns
        except Exception:
            cols = shutil.get_terminal_size((100, 24)).columns
        self._width_cache = max(60, cols)
        self._width_cache_ts = now
        return self._width_cache

    def _invalidate(self) -> None:
        try:
            self.app.invalidate()
        except Exception:
            pass

    def _set_flash(self, text: str, color: str = C["yellow"]) -> None:
        self._flash_gen += 1
        gen = self._flash_gen
        self._flash = (text, color)
        if self._flash_timer:
            self._flash_timer.cancel()
        self._flash_timer = threading.Timer(4.0, self._clear_flash, args=(gen,))
        self._flash_timer.daemon = True
        self._flash_timer.start()
        self._invalidate()

    def _clear_flash(self, gen: int | None = None) -> None:
        # A timer from an older flash must never wipe a newer message —
        # the generation check makes the clear a no-op when stale.
        if gen is not None and gen != self._flash_gen:
            return
        self._flash = None
        self._invalidate()

    def _set_status(self, text: str) -> None:
        if text == self._status_text:
            return
        self._status_text = text
        self._invalidate()

    # -- box border fragments -----------------------------------------------------

    def _top_fragments(self) -> list:
        width = self._width()
        model = self._model()
        effort = self._effort()
        effort_color = EFFORT_COLORS.get(effort.key, C["fg"])
        prov = PROVIDERS.get(model.provider)
        prov_color = (prov.color if prov is not None else C["accent"])

        segs: list[tuple[str, str]] = []
        segs.append((f" ◆ {APP_NAME} ", "class:box.title"))
        segs.append((" ● ", f"bold {prov_color}"))
        segs.append((f" {model.label} ", "class:box.model"))
        if model.tag:
            segs.append((f"★ {model.tag} ", "class:box.tag"))
        segs.append((f" ◈ {effort.label.lower()} ",
                     f"class:box.effort {effort_color}"))
        auto = getattr(self.agent, "autonomy", 3)
        auto_color = (C["dim"] if auto <= 1 else C["green"] if auto <= 3
                      else C["yellow"] if auto == 4 else C["red"])
        segs.append((f" ⬢ L{auto} ", f"bold {auto_color}"))
        # Live parallel subagent swarm indicator in the top HUD border
        try:
            mgr = self.agent.__dict__.get("subagents")
            if mgr is not None:
                snap = mgr.snapshot()
                if snap.get("active", 0) > 0:
                    segs.append((
                        f" ⚡ Swarm {snap['active']}/{snap['max_active']} ",
                        f"bold {C['pink']}",
                    ))
        except Exception:
            pass
        # live goal distance — always on screen when a goal is active (§24).
        # cached for 1s: the border re-renders every frame, and status()
        # folds the whole log
        now = time.time()
        if self._goal_cache is None or now - self._goal_cache_ts > 1.0:
            try:
                self._goal_cache = self.agent.goal.status()
            except Exception:
                # a corrupt goal event (or a fold failure) must never break
                # the render loop — keep the previous reading, or hide the
                # segment when there isn't one yet
                pass
            self._goal_cache_ts = now
        goal = self._goal_cache
        if goal is not None and goal.active:
            pct = max(0, min(100, (1 - goal.distance) * 100))
            filled = int(round(pct / 20))
            bar = "▰" * filled + "▱" * (5 - filled)
            segs.append((f" ◎ goal {bar} {pct:.0f}% ",
                         "class:box.status"))
        if self._focus_remaining > 0:
            segs.append((f" 🎯 focus×{self._focus_remaining} ",
                         "class:box.flash"))
        # live context usage — cached 1s (the border redraws every frame)
        if self._ctx_cache is None or now - self._ctx_cache_ts > 1.0:
            try:
                from .client import estimate_tokens
                used = estimate_tokens(self.agent.messages,
                                       self.agent.model.id)
                window = max(1, self.agent.model.context_window)
                self._ctx_cache = min(100, int(used * 100 / window))
            except Exception:
                self._ctx_cache = 0
            self._ctx_cache_ts = now
        # ctxmeter: visual context bar (falls back to text % on any error)
        try:
            from . import ctxmeter
            segs.extend(ctxmeter.context_bar(self))
        except Exception:
            ctx_color = (C["green"] if self._ctx_cache < 60
                         else C["yellow"] if self._ctx_cache < 85
                         else C["red"])
            segs.append((f" ◉ ctx {self._ctx_cache}% ",
                         f"bold {ctx_color}"))
        segs.append((f" ⌛ session: {self.agent.session_id} ", "class:box.session"))

        # fixed = corners (2) + first dash (1) + "──" before each later seg
        # NOTE: display width, not len() — emoji (🎯) are two columns;
        # underestimating lets the closing ╮ wrap and corrupt the render
        def fixed_len(segs: list) -> int:
            return (sum(_disp_width(t) for t, _ in segs)
                    + 3 + 2 * (len(segs) - 1))

        # drop trailing segments (session, effort, …) if the terminal is
        # too narrow — never let the border exceed the terminal width or the
        # closing "╮" wraps to the next line and corrupts the render
        while fixed_len(segs) > width and len(segs) > 2:
            segs.pop()

        fill = max(0, width - fixed_len(segs))
        frags: list = [("class:box", "╭")]
        for i, (text, style) in enumerate(segs):
            frags.append(("class:box", "──" if i else "─"))
            frags.append((style, text))
        frags.append(("class:box", "─" * fill + "╮"))
        return frags

    def _bottom_fragments(self) -> list:
        width = self._width()
        inner = width - 2

        if self._approve_request is not None:
            tool = self._approve_request[0]
            name = _truncate_w(tool.name, max(1, inner - 34))
            head = f" ⚠ approve {name}? "
            y, n, a = " [y]es ", " [n]o ", " [a]lways "
            fill = "─" * max(0, inner - _disp_width(head)
                             - _disp_width(y) - _disp_width(n)
                             - _disp_width(a))
            return ([("class:box", "╰"),
                     ("class:box.approve", head),
                     (f"bold {C['green']}", y),
                     ("class:box.hint", " "),
                     (f"bold {C['red']}", n),
                     ("class:box.hint", " "),
                     (f"bold {C['yellow']}", a),
                     ("class:box", fill + "╯")]
                    + self._approve_args_line(tool, self._approve_request[1]))

        if self._busy:
            frame = SPINNER_FRAMES[self._spinner_i]
            elapsed = ""
            tok_rate = ""
            if self._turn_start_ts:
                dt = max(0.001, time.time() - self._turn_start_ts)
                elapsed = f" · {max(0, int(dt))}s"
                n_tok = getattr(self, "_stream_tokens_count", 0)
                if n_tok > 2 and dt >= 0.2:
                    tok_rate = f" · ⚡{int(n_tok / dt)} tok/s"
            pending = self._pending_count()
            queued = f" · {pending} msg queued" if pending else ""
            tail = (f"{tok_rate}{elapsed}{queued} · Enter queue · "
                    "Esc cancel ")
            max_status = max(0, inner - _disp_width(f" ⠹ {tail}") - 4)
            status = _truncate_w(self._status_text, max_status)
            bar_len = (_disp_width(f" {frame} ") + _disp_width(status)
                       + _disp_width(tail))
            return [("class:box", "╰"),
                    (f"bold {fx.spinner_color(self._spinner_i)}",
                     f" {frame} "),
                    ("class:box.status", status),
                    ("class:box.hint", tail),
                    ("class:box",
                     "─" * max(0, inner - bar_len) + "╯")]

        if self._flash:
            text, color = self._flash
            hint = _truncate_w(f" ✦ {text} ", max(1, inner))
            return [("class:box", "╰"), (f"bold {color}", hint),
                    ("class:box",
                     "─" * max(0, inner - _disp_width(hint)) + "╯")]

        # idle — styled keys, gracefully degrading on narrow terminals
        # NOTE: fragments are (style, text) tuples — keep this order.
        key = "class:box.key"
        dim = "class:box.hint"
        box = "class:box"
        if inner >= len("⏎ send · ⇧⏎ newline · / cmds · /diff · ^T models · ^E effort ") + 1:
            segs = [(box, " ⏎ "), (key, "send"),
                    (dim, " · ⇧⏎ newline · "), (key, "/"), (dim, " cmds · "),
                    (key, "/diff"), (dim, " live · "),
                    (key, "^T"), (dim, " models · "), (key, "^E"),
                    (dim, " effort ")]
        elif inner >= len("⏎ send · / cmds · ^T models ") + 1:
            segs = [(box, " ⏎ "), (key, "send"),
                    (dim, " · "), (key, "/"), (dim, " cmds · "),
                    (key, "^T"), (dim, " models ")]
        else:
            segs = [(box, " ⏎ "), (key, "send"), (dim, " ")]
        frags: list = [(box, "╰")]
        frags.extend(segs)
        used = 1 + sum(_disp_width(t) for _, t in segs)
        frags.append((box, "─" * max(0, inner - used) + "╯"))
        return frags

    def _approve_args_line(self, tool: Tool, args: dict) -> list:
        return []

    def _overlay_fragments(self) -> list:
        if self.overlay and self.overlay.visible:
            return self.overlay.fragments(self._width())
        return []

    # -- Qoder-style chrome: HUD, footer, live agent board ---------------------

    def _skills_count(self) -> int:
        """Real installed skills: SKILL.md skills + forged tools.

        Scans the same sources ``/skills`` shows (``~/.fullagent/skills``
        SKILL.md directories plus the forge registry), cached for 5s — the
        footer re-renders every frame and both sources touch disk.
        """
        now = time.time()
        cached = getattr(self, "_skills_cache", None)
        if (cached is not None
                and now - getattr(self, "_skills_cache_ts", 0.0) < 5.0):
            return cached
        names: set[str] = set()
        try:
            mgr = getattr(self.agent, "skill_manager", None)
            if mgr is None:
                from .skills import SkillMDManager
                mgr = SkillMDManager()
            for item in mgr.list_skills() or []:
                if isinstance(item, dict) and item.get("name"):
                    names.add(str(item["name"]).strip().lower())
        except Exception:
            pass
        try:
            names.update(str(k).strip().lower() for k in
                         (getattr(self.agent.skill_forge, "registry", {})
                          or {}))
        except Exception:
            pass
        self._skills_cache = len(names)
        self._skills_cache_ts = now
        return self._skills_cache

    def _log_chrome_error(self, where: str, exc: Exception) -> None:
        """Record a chrome wiring failure once (never spam, never crash).

        Render callbacks run inside prompt_toolkit's draw pass, so printing
        there could corrupt the frame — the failure goes to a log file the
        user can inspect instead of vanishing into a bare ``except``.
        """
        try:
            key = f"{where}:{type(exc).__name__}:{exc}"
            seen = getattr(self, "_chrome_errors", None)
            if seen is None:
                seen = self._chrome_errors = set()
            if key in seen:
                return
            seen.add(key)
            path = config.APP_DIR / "chrome_errors.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {key}\n")
        except Exception:
            pass

    def _agents_snapshot(self) -> dict:
        """Live subagent snapshot, cached for a quarter second.

        The board re-renders on every frame; ``snapshot()`` copies one dict
        per worker under the manager lock, so polling it 20x/second for a
        16-worker swarm is wasted work. Tool calls and turn boundaries drop
        the cache, so a spawn/finish shows up immediately instead of after
        the TTL.
        """
        now = time.time()
        cached = getattr(self, "_agents_cache", None)
        if cached is None or now - getattr(self, "_agents_cache_ts", 0.0) > 0.25:
            try:
                mgr = self.agent.__dict__.get("subagents")
                if mgr is not None:
                    self._hook_agents(mgr)
                cached = mgr.snapshot() if mgr is not None else {}
            except Exception:
                cached = {}
            self._agents_cache = cached
            self._agents_cache_ts = now
        return cached or {}

    def _hook_agents(self, mgr) -> None:
        """Subscribe to the real manager's progress events (once).

        Worker threads call this on every state/tool transition, so the
        board updates the instant a worker starts, calls a tool or finishes
        — even while the main turn is blocked waiting on TaskCollect.
        """
        if getattr(self, "_agents_hooked", False):
            return
        self._agents_hooked = True
        try:
            mgr.on_progress(self._on_subagent_event)
        except Exception:
            pass

    def _on_subagent_event(self, _worker=None, _kind=None) -> None:
        """Manager callback (worker thread): drop the cache and redraw."""
        self._invalidate_agents()
        try:
            self._invalidate()
        except Exception:
            pass

    def _invalidate_agents(self) -> None:
        """Force the next frame to read the real subagent state."""
        self._agents_cache = None
        self._agents_cache_ts = 0.0

    def _agents_visible(self) -> bool:
        """True while at least one subagent is queued or running."""
        try:
            snap = self._agents_snapshot()
            return any(row.get("state") in cliui.ACTIVE_STATES
                       for row in snap.get("workers") or [])
        except Exception:
            return False

    def _agents_fragments(self) -> list:
        """The live swarm board, one prompt_toolkit fragment row per line."""
        try:
            rows = cliui.agent_board(self._agents_snapshot(),
                                     width=self._width(), colors=C)
        except Exception as exc:  # noqa: BLE001 — chrome must never crash
            self._log_chrome_error("agent_board", exc)
            return []
        frags: list = []
        for idx, row in enumerate(rows):
            frags.extend(row)
            if idx < len(rows) - 1:
                frags.append(("", "\n"))
        return frags

    def _agent_counts(self) -> tuple:
        """(running, queued, live) from the real snapshot — live only.

        Retained workers from earlier batches are deliberately excluded;
        they are history, not running agents.
        """
        counts = (self._agents_snapshot() or {}).get("counts") or {}
        running = int(counts.get("running") or 0)
        queued = int(counts.get("queued") or 0)
        cancelling = int(counts.get("cancelling") or 0)
        return running, queued, running + queued + cancelling

    def _hud_fragments(self) -> list:
        """Mode chip + key hint on the left, skills counter on the right."""
        try:
            running, queued, _ = self._agent_counts()
        except Exception:
            running = queued = 0
        try:
            return cliui.hud_fragments(
                width=self._width(), colors=C,
                autonomy=getattr(self.agent, "autonomy", 3),
                skills=self._skills_count(),
                agents_running=running, agents_queued=queued)
        except Exception as exc:  # noqa: BLE001
            self._log_chrome_error("hud", exc)
            return []

    def _cwd_label(self) -> str:
        try:
            cwd = os.getcwd()
            home = str(Path.home())
            if home and (cwd == home or cwd.startswith(home + os.sep)):
                cwd = "~" + cwd[len(home):]
            return cwd
        except Exception:
            return "."

    def _signed_in_label(self) -> str:
        """Honest provider-credential line for the welcome card.

        FullAgent has no browser auth. The default build also ships an
        embedded fallback key, so "Signed in" (or even a bare "API key
        loaded") would overstate what is happening. This reports the real
        origin of the key that will be used: environment, user key file,
        or the built-in fallback — and says so plainly when there is none.
        """
        try:
            model = self._model()
            prov = PROVIDERS.get(model.provider)
            if prov is None:
                return "No provider configured"
            env_name = f"{model.provider.upper()}_API_KEY"
            if (os.environ.get(env_name) or "").strip():
                return f"API key · env ({prov.name})"
            try:
                key_file = config.APP_DIR / f"{model.provider}_api_key"
                if key_file.read_text(encoding="utf-8").strip():
                    return f"API key · key file ({prov.name})"
            except (OSError, UnicodeDecodeError):
                pass
            if getattr(prov, "api_key", ""):
                return f"API key · built-in ({prov.name})"
        except Exception:
            pass
        return "No API key · local session"

    def _branch_label(self) -> str:
        """The real git branch of the working directory (or a clear fallback).

        ``git rev-parse --abbrev-ref HEAD`` is cheap and cached for 5s; a
        non-repo / detached HEAD shows the session id instead of inventing
        a branch name.
        """
        now = time.time()
        if (getattr(self, "_branch_cache", None) is not None
                and now - getattr(self, "_branch_cache_ts", 0.0) < 5.0):
            return self._branch_cache
        name = ""
        try:
            import subprocess
            if shutil.which("git"):
                proc = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=os.getcwd(), capture_output=True, text=True,
                    timeout=2.0)
                if proc.returncode == 0:
                    name = (proc.stdout or "").strip()
        except Exception:
            name = ""
        if not name or name == "HEAD":
            sid = str(getattr(self.agent, "session_id", "") or "")
            name = f"session {sid[:8]}" if sid else "no git"
        self._branch_cache = name
        self._branch_cache_ts = now
        return name

    def _context_percent(self) -> int:
        """Live context usage %, cached 1s (the border redraws every frame)."""
        now = time.time()
        cached = getattr(self, "_ctx_cache", None)
        if cached is None or now - getattr(self, "_ctx_cache_ts", 0.0) > 1.0:
            try:
                from .client import estimate_tokens
                used = estimate_tokens(self.agent.messages,
                                       self.agent.model.id)
                window = max(1, self.agent.model.context_window)
                cached = min(100, int(used * 100 / window))
            except Exception:
                if cached is None:
                    cached = 0
            self._ctx_cache = cached
            self._ctx_cache_ts = now
        return int(cached or 0)

    def _meta_fragments(self) -> list:
        """Footer under the prompt: model · effort · context · cwd · agents."""
        try:
            running, queued, _ = self._agent_counts()
        except Exception:
            running = queued = 0
        try:
            model = self._model()
            effort = self._effort()
            return cliui.status_fragments(
                width=self._width(), colors=C,
                model_label=model.label, effort_label=effort.label.lower(),
                context_window=model.context_window,
                used_pct=self._context_percent(), cwd=self._cwd_label(),
                branch=self._branch_label(),
                agents_running=running, agents_queued=queued)
        except Exception as exc:  # noqa: BLE001
            self._log_chrome_error("footer", exc)
            return []

    # -- background shell/progress panel ---------------------------------------


    def _background_visible(self) -> bool:
        try:
            from .bgtui import has_activity
            if has_activity(self.agent):
                return True
        except Exception:
            pass
        try:
            tracker = getattr(self.agent, "progress_tracker", None)
            return bool(tracker and tracker.progress_bar_lines())
        except Exception:
            return False

    def _background_fragments(self) -> list:
        frags: list = []
        try:
            from .bgtui import BackgroundPanel, has_activity
            if getattr(self, "bg_panel", None) is None:
                self.bg_panel = BackgroundPanel()
            if has_activity(self.agent):
                for line in self.bg_panel.refresh(self.agent):
                    frags.append(("class:bg.line", line + "\n"))
        except Exception:
            pass
        try:
            tracker = getattr(self.agent, "progress_tracker", None)
            if tracker is not None:
                for line in tracker.progress_bar_lines():
                    frags.append(("class:bg.line", line + "\n"))
        except Exception:
            pass
        return frags

    # -- construction ----------------------------------------------------------------

    def _build(self) -> None:
        config.ensure_dirs()
        self.search_toolbar = SearchToolbar()
        self.buffer = Buffer(
            name=DEFAULT_BUFFER,
            multiline=True,
            completer=SlashCompleter(),
            complete_while_typing=True,
            enable_history_search=True,
            auto_suggest=AutoSuggestFromHistory(),
            history=SafeFileHistory(str(config.HISTORY_FILE)),
            accept_handler=self._accept_handler,
        )

        def get_line_prefix(line: int, wrap_count: int):
            if line == 0 and wrap_count == 0:
                return [("class:arrow", "❯ ")]
            if wrap_count > 0:
                return [("class:cont", "↪ ")]
            return [("class:cont", "┃ ")]

        self.input_control = BufferControl(
            buffer=self.buffer,
            search_buffer_control=self.search_toolbar.control,
            preview_search=True,
        )
        input_window = Window(
            self.input_control,
            height=Dimension(min=1, max=10),
            wrap_lines=True,
            get_line_prefix=get_line_prefix,
        )
        self.input_window = input_window

        top_window = Window(FormattedTextControl(self._top_fragments),
                            height=1, dont_extend_height=True)
        hud_window = Window(FormattedTextControl(self._hud_fragments),
                            height=1, dont_extend_height=True)
        meta_window = Window(FormattedTextControl(self._meta_fragments),
                             height=1, dont_extend_height=True)
        agents_window = Window(
            FormattedTextControl(self._agents_fragments),
            dont_extend_height=True)
        bottom_window = Window(FormattedTextControl(self._bottom_fragments),
                               height=1, dont_extend_height=True)
        left_window = Window(FormattedTextControl(
            lambda: [("class:box", "│ ")]), width=2, dont_extend_width=True)
        right_window = Window(FormattedTextControl(
            lambda: [("class:box", " │")]), width=2, dont_extend_width=True)
        overlay_window = Window(
            FormattedTextControl(self._overlay_fragments),
            dont_extend_height=True)
        # Background shell jobs and progress bars remain visible while active.
        background_window = Window(
            FormattedTextControl(self._background_fragments),
            dont_extend_height=True)

        root = FloatContainer(
            HSplit([
                ConditionalContainer(
                    overlay_window,
                    filter=Condition(self._overlay_open)),
                ConditionalContainer(
                    agents_window,
                    filter=Condition(self._agents_visible)),
                hud_window,
                top_window,
                ConditionalContainer(
                    background_window,
                    filter=Condition(self._background_visible)),
                VSplit([left_window, input_window, right_window], padding=0),
                bottom_window,
                meta_window,
                ConditionalContainer(self.search_toolbar,
                                     filter=Condition(self._search_visible)),
            ]),
            [
                Float(xcursor=True, ycursor=True,
                      content=CompletionsMenu(
                          max_height=10, scroll_offset=1,
                          extra_filter=has_focus(self.buffer))),
            ],
        )

        self.app = Application(
            layout=Layout(root, focused_element=input_window),
            style=STYLE,
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            min_redraw_interval=0.05,
            mouse_support=False,
        )

    def _search_visible(self) -> bool:
        try:
            return get_app().layout.current_control == self.search_toolbar.control
        except Exception:
            return False

    def _accept_handler(self, buff: Buffer) -> bool:
        text = buff.text.strip()
        if text:
            self._dispatch(text)
        return False  # clear the buffer

    # -- state filters ------------------------------------------------------------------

    def _overlay_open(self) -> bool:
        return self.overlay is not None and self.overlay.visible

    def _approving(self) -> bool:
        return self._approve_request is not None

    def _completion_active(self) -> bool:
        return self.buffer.complete_state is not None

    def _on_first_line(self) -> bool:
        return self.buffer.document.cursor_position_row == 0

    def _on_last_line(self) -> bool:
        doc = self.buffer.document
        return doc.cursor_position_row == len(doc.lines) - 1

    def _buffer_empty(self) -> bool:
        return not self.buffer.text

    # -- key bindings ----------------------------------------------------------------------

    def _build_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        focused = has_focus(self.buffer)
        ov = Condition(self._overlay_open)
        approving = Condition(self._approving)
        idle = ~approving  # normal input only when not at the approve bar
        # Enter always accepts input.  When a turn is already running the
        # text is queued instead of starting a concurrent agent.run_turn;
        # _dispatch() owns that decision and the serial drain.
        can_submit = focused & ~ov & idle

        # --- approval bar: y / n / a ---
        # register the catch-all FIRST so it has the lowest priority; the
        # specific y/n/a/Enter/Ctrl+C handlers below override it.
        @kb.add(Keys.Any, filter=approving)
        def _appr_swallow(event):
            pass  # ignore everything else while approving

        @kb.add("y", filter=approving)
        def _appr_y(event):
            self._answer_approve("y")

        @kb.add("n", filter=approving)
        def _appr_n(event):
            self._answer_approve("n")

        @kb.add("a", filter=approving)
        def _appr_a(event):
            self._answer_approve("a")

        @kb.add(Keys.Enter, filter=approving)
        def _appr_enter(event):
            self._answer_approve("n")

        @kb.add("c-c", filter=approving)
        def _appr_cancel(event):
            self._answer_approve("n")

        # --- overlay navigation ---
        @kb.add(Keys.Enter, filter=ov)
        def _ov_enter(event):
            self.overlay.select()

        @kb.add(Keys.Escape, filter=ov)
        @kb.add("c-c", filter=ov)
        def _ov_close(event):
            self.overlay.close()

        @kb.add(Keys.Up, filter=ov)
        @kb.add("c-p", filter=ov)
        @kb.add(Keys.BackTab, filter=ov)
        def _ov_up(event):
            self.overlay.move(-1)

        @kb.add(Keys.Down, filter=ov)
        @kb.add("c-n", filter=ov)
        @kb.add(Keys.Tab, filter=ov)
        def _ov_down(event):
            self.overlay.move(1)

        @kb.add(Keys.PageUp, filter=ov)
        def _ov_pgup(event):
            self.overlay.page(-1)

        @kb.add(Keys.PageDown, filter=ov)
        def _ov_pgdn(event):
            self.overlay.page(1)

        @kb.add(Keys.Home, filter=ov)
        def _ov_home(event):
            self.overlay.index = 0
            self.overlay._top = 0

        @kb.add(Keys.End, filter=ov)
        def _ov_end(event):
            self.overlay.index = len(self.overlay.items) - 1
            self.overlay._top = max(0, self.overlay.index - OverlayList.WINDOW + 1)

        # --- completion menu: PgUp/PgDn also navigate it ---
        # eager=True ensures these win over prompt_toolkit's default
        # PgUp/PgDn (buffer scroll) bindings.
        @kb.add(Keys.PageDown, filter=focused & ~ov & idle &
                Condition(self._completion_active), eager=True)
        def _cm_pgdn(event):
            self.buffer.complete_next(5)

        @kb.add(Keys.PageUp, filter=focused & ~ov & idle &
                Condition(self._completion_active), eager=True)
        def _cm_pgup(event):
            self.buffer.complete_previous(5)

        # --- completion menu: Up/Down navigate it (not history) ---
        # These must win over the history bindings below, hence eager=True.
        @kb.add(Keys.Down, filter=focused & ~ov & idle &
                Condition(self._completion_active), eager=True)
        def _cm_down(event):
            self.buffer.complete_next()

        @kb.add(Keys.Up, filter=focused & ~ov & idle &
                Condition(self._completion_active), eager=True)
        def _cm_up(event):
            self.buffer.complete_previous()

        # --- input ---
        @kb.add(Keys.Enter, filter=can_submit)
        def _enter(event):
            b = self.buffer
            if b.complete_state is not None:
                # First Enter with menu open: accept the highlighted
                # completion (fill text) but DON'T submit — user can edit
                # or press Enter again to run the command.
                b.complete_state = None
                return
            if not b.text.strip():
                return
            b.validate_and_handle()

        @kb.add(Keys.Escape, Keys.Enter, filter=focused & ~ov & idle)
        def _newline(event):
            self.buffer.insert_text("\n")

        @kb.add(Keys.Up, filter=focused & ~ov & idle &
                Condition(self._on_first_line))
        def _hist_prev(event):
            self.buffer.history_backward()

        @kb.add(Keys.Down, filter=focused & ~ov & idle &
                Condition(self._on_last_line))
        def _hist_next(event):
            self.buffer.history_forward()

        # Qoder-style board: ↓ on an empty prompt opens /agents instead of
        # doing nothing (registered after _hist_next so it wins).
        @kb.add(Keys.Down, filter=focused & ~ov & idle &
                Condition(self._on_last_line) &
                Condition(self._buffer_empty) &
                Condition(lambda: not self._busy) &
                ~Condition(self._completion_active))
        def _manage_agents(event):
            if self._agents_visible():
                self._cmd_agents("")

        @kb.add("s-tab", filter=focused & ~ov & idle)
        def _toggle_auto(event):
            self._toggle_auto_mode()

        @kb.add("c-o", filter=focused & ~ov)
        def _toggle_detail(event):
            self._toggle_detail_view()

        @kb.add("c-c", filter=focused & ~ov & idle)
        def _ctrl_c(event):
            if self._busy:
                cleared = self._request_cancel()
                self._set_status("cancelling…")
                if cleared:
                    self._set_flash(
                        f"⊘ cancelling — {cleared} queued message(s) cleared",
                        C["yellow"])
            elif self.buffer.text:
                self.buffer.reset()
                self._set_flash("input cleared — Ctrl+C again or /exit to quit",
                                C["red"])
            else:
                self._set_flash("type /exit or Ctrl+D to quit", C["dim"])

        # single Esc while a turn is running = interrupt, exactly like
        # Ctrl+C (Esc+Enter stays the multi-line newline binding above)
        @kb.add(Keys.Escape, filter=focused & ~ov &
                Condition(lambda: self._busy))
        def _esc_cancel(event):
            cleared = self._request_cancel()
            self._set_status("cancelling…")
            if cleared:
                self._set_flash(
                    f"⊘ cancelling — {cleared} queued message(s) cleared",
                    C["yellow"])
            else:
                self._set_flash("⊘ cancelling — force stopping…", C["yellow"])

        # Esc at the approve bar = "no", same as Ctrl+C there
        @kb.add(Keys.Escape, filter=approving)
        def _esc_deny(event):
            self._answer_approve("n")

        @kb.add("c-d", filter=focused & ~ov & idle &
                Condition(self._buffer_empty))
        def _ctrl_d(event):
            event.app.exit()

        @kb.add("c-l", filter=~ov & idle)
        def _ctrl_l(event):
            event.app.renderer.clear()
            self._invalidate()

        @kb.add("c-t", filter=focused & ~ov & idle)
        def _ctrl_t(event):
            self.open_model_selector()

        @kb.add("c-e", filter=focused & ~ov & idle)
        def _ctrl_e(event):
            self.open_effort_selector()

        @kb.add("c-r", filter=focused & ~ov & idle)
        def _ctrl_r(event):
            start_search(self.input_control)

        @kb.add("c-x", "c-e", filter=focused & ~ov & idle)
        def _external_editor(event):
            self.buffer.open_in_editor(validate_and_handle=True)

        return kb

    def _request_cancel(self) -> int:
        """Idempotently cancel this UI turn and drop pending follow-ups.

        Returns the number of queued messages dropped so the key handler
        can tell the user exactly what happened.
        """
        q, lock = self._queue_state()
        with lock:
            self._cancel_flag.set()
            cleared = len(q)
            q.clear()
        flag = getattr(self.agent, "_cancel_flag", None)
        if flag is not None:
            try:
                flag.set()
            except Exception:
                pass
        # Agent and provider requests share the same cancellation flag.
        # Esc must also stop running subagents: before the rewrite this
        # called agent.crew.force_stop(); the rewrite dropped it, so
        # workers silently kept running (and billing) after the user
        # cancelled the turn. Non-destructive stop — the manager stays
        # usable for later turns (shutdown(wait=False) would retire the
        # pool instead). Never let this raise into the key handler.
        try:
            manager = self.agent.__dict__.get("subagents")
            if manager is not None:
                rows = (manager.snapshot() or {}).get("workers", [])
                for row in rows:
                    if row.get("state") in ("queued", "running", "cancelling"):
                        try:
                            manager.stop(row["id"])
                        except Exception:
                            pass
        except Exception:
            pass
        self._invalidate_agents()  # cancelled workers must leave the board now
        # Watchdog: if the turn thread is stuck (e.g. blocked in input()
        # or a hung subprocess), force-clear the busy state after 10s so
        # the UI never shows "cancelling…" forever.
        import threading as _th
        def _watchdog():
            import time as _t
            _t.sleep(10)
            with lock:
                stuck = self._busy and self._cancel_flag.is_set()
                if stuck:
                    # _turn_alive deliberately stays true: the worker still
                    # owns the shared history.  New input is queued by
                    # _dispatch() and drains only when that worker returns.
                    self._busy = False
                    q.clear()
            if stuck:
                try:
                    self._set_status("")
                    self._set_flash(
                        "⚠ cancel timed out — worker still alive; "
                        "input will queue", C["yellow"])
                except Exception:
                    pass
        _th.Thread(target=_watchdog, daemon=True).start()
        return cleared

    def _bg(self, fn) -> None:
        """Run fn on a daemon thread with exceptions surfaced to the UI —
        a bare thread's default excepthook prints to stderr, invisible
        under patch_stdout, so failures would look like silent no-ops."""
        def guarded():
            try:
                fn()
            except Exception as e:  # noqa: BLE001 — report, never crash
                self.print_error(f"{type(e).__name__}: {e}")
        threading.Thread(target=guarded, daemon=True).start()

    # -- Qoder-style chrome: mode toggle + expandable tool detail --------------

    def _toggle_auto_mode(self) -> None:
        """⇧Tab: flip between the current autonomy level and Pilot (L4).

        Pilot keeps the one destructive guard (deletes still ask), so this
        is a mode switch, not a blanket approval bypass. The previous level
        is remembered so ⇧Tab toggles cleanly in both directions.
        """
        try:
            current = int(getattr(self.agent, "autonomy", 3))
        except Exception:
            current = 3
        if current >= 4:
            target = int(getattr(self, "_auto_prev", 3) or 3)
            if target >= 4:
                target = 3
            label = "Pilot mode OFF"
        else:
            self._auto_prev = current
            target = 4
            label = "Pilot mode ON — safe actions run without asking (L4)"
        try:
            self.agent.set_autonomy(target)
            actual = int(getattr(self.agent, "autonomy", target))
        except Exception as exc:  # noqa: BLE001 — never break the key handler
            self._set_flash(f"⚠ cannot switch mode: {exc}", C["red"])
            return
        self._set_flash(f"✦ {label} · L{actual}", C["green"]
                        if actual >= 4 else C["cyan"])
        self._invalidate()

    def _cap(self, compact: int, full: int | None = None) -> int:
        """Row cap for tool bodies — Ctrl+O swaps compact for full output."""
        if getattr(self, "_detail_expanded", False):
            return full if full is not None else max(compact, 2000)
        return compact

    def _remember_detail(self, ev: ToolEvent, kind: str = "live") -> None:
        """Keep the last finished tool block so Ctrl+O can re-print it."""
        try:
            if ev is not None and ev.status in ("done", "error"):
                self._last_detail_event = (ev, kind)
        except Exception:
            pass

    def _print_thinking_block(self, text: str, expanded: bool = False) -> None:
        """Qoder-style ``Thinking`` panel: a few rows, Ctrl+O for the rest.

        The full reasoning is kept in ``_last_detail_event`` so Ctrl+O can
        re-print every row instead of just the collapsed preview.
        """
        lines = [ln.rstrip() for ln in str(text or "").splitlines()
                 if ln.strip()]
        if not lines:
            return
        width = max(20, self._width() - 6)
        preview = 8
        limit = len(lines) if expanded else preview
        head = Text()
        head.append("  Thinking", style=f"bold {C['pink']}")
        if len(lines) > limit:
            head.append(f"  ({len(lines)} rows)", style=C["dim"])
        self.console.print(head)
        for ln in lines[:limit]:
            row = Text()
            row.append("  │ ", style=C["dim"])
            row.append(_truncate_w(ln, width), style=f"italic {C['dim']}")
            self.console.print(row)
        if len(lines) > limit:
            self.console.print(Text(
                f"  │ … +{len(lines) - limit} rows (Ctrl+O to expand)",
                style=C["dim"]))
        # Only a truncated, still-collapsed block needs the Ctrl+O target;
        # an already-complete or already-expanded block must not steal
        # expansion from the previous block.
        if len(lines) > preview and not expanded:
            self._last_detail_event = (str(text), "thinking")

    def _toggle_detail_view(self) -> None:
        """Ctrl+O: expand the last block, then return to the compact view.

        Expanding re-prints the last finished block with every row instead
        of the compact cap. Collapsing switches back to compact output for
        the NEXT block — output already in the scrollback cannot be
        un-printed, and the flash says so instead of pretending otherwise.
        Print failures can never take the UI down.
        """
        expanded = not getattr(self, "_detail_expanded", False)
        self._detail_expanded = expanded
        stored = getattr(self, "_last_detail_event", None)
        if expanded and stored is not None:
            try:
                ev, kind = stored
                if kind == "thinking":
                    self._print_thinking_block(ev, expanded=True)
                    self._set_flash(
                        "✦ thinking expanded · Ctrl+O compact view", C["cyan"])
                    return
                w = max(20, self._width() - 8)
                if kind == "live":
                    rows = self._live_rows(ev, w)
                    title = self._devin_title(ev)
                else:
                    rows = self._generic_rows(ev)
                    title = self._devin_title(ev)
                head = Text()
                head.append(" ⏺ ", style=f"bold {C['accent']}")
                head.append(f"detail · {title}", style=f"bold {C['fg']}")
                head.append(f"  {len(rows)} rows", style=C["dim"])
                self.console.print(head)
                n = len(rows)
                for idx, row in enumerate(rows):
                    line = Text()
                    line.append(" └ " if idx == n - 1 else " │ ", style=C["dim"])
                    for seg, style in row:
                        line.append(seg, style=style)
                    self.console.print(line)
                self._set_flash("✦ expanded · Ctrl+O compact view", C["cyan"])
                return
            except Exception:
                pass
        self._set_flash(
            "✦ full output: on (next tool block)" if expanded
            else "✦ compact view for the next block "
                 "(expanded output stays in scrollback)",
            C["cyan"] if expanded else C["dim"])

    # -- dispatch: slash commands + turns ----------------------------------------------------

    def _queue_state(self) -> tuple[deque[str], threading.Lock]:
        """Return the follow-up queue and its lock.

        Headless tests build ``UI`` with ``__new__`` and do not run
        ``__init__``; creating the state lazily keeps those tests (and any
        plugin that embeds the UI) working without a constructor change.
        """
        q = getattr(self, "_pending", None)
        if q is None:
            q = self._pending = deque()
        lock = getattr(self, "_turn_lock", None)
        if lock is None:
            lock = self._turn_lock = threading.Lock()
        return q, lock

    def _pending_count(self) -> int:
        q, lock = self._queue_state()
        with lock:
            return len(q)

    def _dispatch(self, text: str) -> None:
        if text.startswith("/"):
            self._handle_slash(text)
            return

        q, lock = self._queue_state()
        with lock:
            q.append(text)
            if self._busy or getattr(self, "_turn_alive", False):
                # A turn is running (or its stuck thread has not returned
                # yet).  Keep the text for the finish handler; never start
                # a second agent.run_turn over the same history.
                next_text = None
            else:
                next_text = q.popleft()
                self._busy = True
            pending = len(q)

        if next_text is None:
            preview = " ".join(text.split())
            if len(preview) > 72:
                preview = preview[:69] + "…"
            notice = getattr(self, "print_info", None)
            if callable(notice):
                notice(f"  ⏳ queued #{pending} · {preview}", C["cyan"])
            self._set_flash(
                f"⏳ message #{pending} queued — runs after this turn",
                C["cyan"])
            self._invalidate()
            return

        self._cancel_flag.clear()
        try:
            threading.Thread(target=self._run_turn_thread, args=(next_text,),
                             daemon=True).start()
        except Exception:
            with lock:
                q.appendleft(next_text)
                self._busy = False
            raise

        # Normally pending is 0 here.  A non-zero value means the queue
        # already held an older message (for example after a failed
        # hand-off); the new text is waiting behind it, so say so.
        if pending:
            notice = getattr(self, "print_info", None)
            if callable(notice):
                notice(f"  ⏳ queued #{pending} · "
                       f"{' '.join(text.split())[:72]}", C["cyan"])

    def _handle_slash(self, text: str) -> None:
        parts = text.strip().split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        # one typo or a transient OSError inside a command must never
        # unwind through prompt_toolkit and kill the whole session —
        # same policy as the guarded turn thread below
        try:
            self._route_slash(cmd, arg)
        except Exception as e:  # noqa: BLE001 — the UI must survive
            self.print_error(f"{type(e).__name__}: {e}")

    def _handle_cron(self, arg: str) -> None:
        """``/cron`` — cron scheduler: list/add/remove/log.

        Usage: /cron | /cron list | /cron add "<expr>" "<cmd>"
               | /cron remove <id> | /cron log <id>
        """
        import shlex

        sched = getattr(self.agent, "cron_scheduler", None)
        if sched is None:
            try:
                from . import cronsched as _cs
                sched = _cs.get_scheduler()
            except Exception:
                sched = None
        if sched is None:
            self.print_error("cron scheduler is not registered")
            return
        try:
            parts = shlex.split(arg) if arg else []
        except ValueError as e:
            self.print_error(f"bad quoting: {e}")
            return
        sub = parts[0].lower() if parts else "list"

        def _ok(text):
            self.print_info(text, C["fg"])

        if sub in ("", "list", "ls"):
            rows = sched.list()
            if not rows:
                self.print_info(
                    'no crons — /cron add "* * * * *" "echo hi"',
                    C["dim"])
            else:
                from .cronsched import _fmt_ts
                lines = []
                for r in rows:
                    state = "on " if r["enabled"] else "off"
                    lines.append(
                        f"#{r['id']} [{state}] `{r['cron_expr']}` → "
                        f"{r['command']}  next: {_fmt_ts(r['next_run'])}")
                _ok("\n".join(lines))
        elif sub == "add":
            if len(parts) < 3:
                self.print_error(
                    'usage: /cron add "<cron expr>" "<command>"')
                return
            expr, cmd = parts[1], " ".join(parts[2:])
            try:
                cid, nxt = sched.add(expr, cmd)
            except ValueError as e:
                self.print_error(f"error: {e}")
                return
            from .cronsched import _fmt_ts
            self.print_info(
                f"✓ cron #{cid} added — next run {_fmt_ts(nxt)}",
                C["green"])
        elif sub in ("remove", "rm", "del"):
            if len(parts) < 2 or not parts[1].isdigit():
                self.print_error("usage: /cron remove <id>")
                return
            if sched.remove(int(parts[1])):
                self.print_info(f"✓ cron #{parts[1]} removed", C["green"])
            else:
                self.print_error(f"no cron with id {parts[1]}")
        elif sub in ("log", "logs", "history"):
            if len(parts) < 2 or not parts[1].isdigit():
                self.print_error("usage: /cron log <id>")
                return
            row = sched._row(int(parts[1]))
            if row is None:
                self.print_error(f"no cron with id {parts[1]}")
                return
            runs = sched.runs(int(parts[1]))
            if not runs:
                self.print_info(f"cron #{parts[1]} has not run yet",
                                C["dim"])
            else:
                lines = []
                for r in runs:
                    import datetime as _dt
                    dt = _dt.datetime.fromtimestamp(r["started_at"])
                    lines.append(
                        f"{dt:%Y-%m-%d %H:%M:%S} exit={r['exit_code']}")
                    tail = r["output_tail"].strip()
                    if tail:
                        lines.append("  " +
                                     tail[-400:].replace("\n", "\n  "))
                _ok("\n".join(lines))
        else:
            self.print_error(
                "usage: /cron [list|add \"<expr>\" \"<cmd>\"|"
                "remove <id>|log <id>]")

    def _route_slash(self, cmd: str, arg: str) -> None:
        # Feature: command aliases (/h -> /help, etc.)
        try:
            from .help import resolve_alias
            cmd = resolve_alias(cmd)
        except Exception:
            pass
        if cmd in ("/exit", "/quit", "/q"):
            self.print_info(f"bye — session {self.agent.session_id} saved",
                            C["dim"])
            self._shutdown_subagents()
            self.agent.save_session()
            self.app.exit()
        elif cmd == "/new":
            # Feature: multi-session (multisess) — /new [name] creates a
            # fresh tracked session; the old one is snapshotted first.
            if hasattr(self.agent, "create_session"):
                try:
                    sid = self.agent.create_session(arg)
                    label = arg or sid
                    self.print_info(
                        f"✓ new session started — {label} ({sid})",
                        C["green"])
                except Exception as e:  # noqa: BLE001 — UI must survive
                    self.print_error(f"could not create session: {e}")
            else:
                self.agent.reset()
                self.print_info(
                    f"✓ new conversation started (session {self.agent.session_id})",
                    C["green"])
        elif cmd == "/switch":
            # Feature: multi-session (multisess) — /switch <id> swaps to
            # another session; bare /switch lists them.
            if not hasattr(self.agent, "switch_session"):
                self.print_error("multi-session is not available")
            elif not arg:
                from .multisess import (format_session_list,
                                        list_sessions as _list_sessions)
                self.print_info(format_session_list(
                    _list_sessions(self.agent)), C["cyan"])
            else:
                try:
                    sid = self.agent.switch_session(arg)
                    n = len([m for m in (self.agent.messages or [])
                             if isinstance(m, dict)
                             and m.get("role") in ("user", "assistant")])
                    self.print_info(
                        f"✓ switched to session {sid} "
                        f"({n} messages restored)", C["green"])
                except Exception as e:  # noqa: BLE001 — UI must survive
                    self.print_error(str(e))
        elif cmd == "/clear":
            try:
                self.app.renderer.clear()
            except Exception:
                pass
            self._invalidate()
        elif cmd == "/model":
            # Feature: fuzzy search + per-turn override via modelpick
            from .modelpick import (list_models, format_model_list,
                                    resolve_model, set_turn_model)
            if arg.endswith(" --once"):
                m = resolve_model(arg[:-7].strip())
                if m is None:
                    self.print_error(
                        f"unknown model: {arg} — use /model to browse")
                elif set_turn_model(self.agent, m.id):
                    self.print_info(f"✓ next turn uses {m.label} ({m.id})",
                                    C["green"])
            elif arg:
                m = resolve_model(arg)
                if m is None:
                    self.print_error(
                        f"unknown model: {arg} — use /model to browse")
                else:
                    self.cfg.model_id = m.id
                    self.cfg.save()
                    self.print_info(f"✓ model → {m.label} ({m.id})",
                                    C["green"])
            else:
                # Open the interactive selector (same as Ctrl+T) instead of
                # just printing the list — users couldn't figure out how to
                # actually SELECT a model.
                self.open_model_selector()
        elif cmd == "/models":
            from .modelpick import list_models, format_model_list
            filt = arg.strip()
            self.print_info(format_model_list(list_models(filt)), C["cyan"])
        elif cmd == "/effort":
            arg = arg.split()[0] if arg else ""
            if arg:
                e = effort_by_key(arg.lower())
                if e is None:
                    self.print_error("effort levels: " +
                                     " · ".join(x.key for x in EFFORTS))
                else:
                    self.cfg.effort = e.key
                    self.cfg.save()
                    self.print_info(f"✓ effort → {e.label}", e.color)
            else:
                self.open_effort_selector()
        elif cmd == "/help":
            from .help import format_help
            self.print_info(format_help(arg), C["fg"])
        elif cmd == "/agents":
            self._cmd_agents(arg)
        elif cmd == "/diff":
            self._cmd_diff(arg)
        elif cmd == "/theme":
            self._cmd_theme(arg)
        elif cmd == "/init":
            from .codeinit import run_init_command
            run_init_command(self, arg)
        elif cmd == "/review":
            from .review import run_review_command
            run_review_command(self, arg)
        # ---- Round 2 advanced features ----
        elif cmd == "/mcp":
            from .mcp import handle_mcp
            handle_mcp(self, arg)
        elif cmd == "/skills":
            from .skills import handle_skills
            handle_skills(self, arg)
        elif cmd == "/doctor":
            from .doctor import handle_doctor
            handle_doctor(self, arg)
        elif cmd == "/status":
            from .statuscmd import handle_status
            handle_status(self, arg)
        elif cmd == "/config":
            from .configcmd import handle_config
            handle_config(self, arg)
        elif cmd == "/vim":
            from .vimmode import handle_vim
            handle_vim(self, arg)
        elif cmd == "/thinking":
            from .thinking import handle_thinking
            handle_thinking(self, arg)
        elif cmd == "/undo":
            from .undocmd import handle_undo
            handle_undo(self, arg)
        elif cmd == "/retry":
            from .retrycmd import handle_retry
            handle_retry(self, arg)
        elif cmd == "/reset":
            from .resetcmd import handle_reset
            handle_reset(self, arg)
        elif cmd == "/cost":
            from .costtrack import handle_cost
            handle_cost(self, arg)
        elif cmd == "/voice":
            from .voicein import handle_voice
            handle_voice(self, arg)
        elif cmd == "/pr-review":
            from .prreview import handle_pr_review
            handle_pr_review(self, arg)
        elif cmd == "/terminal-setup":
            from .termsetup import handle_terminal_setup
            handle_terminal_setup(self, arg)
        elif cmd == "/plugin":
            from .plugins import handle_plugin
            handle_plugin(self, arg)
        elif cmd == "/compact":
            from .compact import do_compact
            stats = do_compact(self.agent)
            self.print_info(
                f"Compacted: {stats['before_msgs']} → {stats['after_msgs']} "
                f"messages, {stats['before_chars']} → {stats['after_chars']} "
                "chars", C["green"])
        elif cmd == "/sessions":
            from . import sessions
            items = sessions.list_sessions()
            if not items:
                self.print_info("no saved sessions", C["dim"])
            else:
                for s in items:
                    self.print_info(
                        f"  {s['id']}  {s['saved_at'][:19]}  "
                        f"{s['model_id']}  ({s['msg_count']} msgs) "
                        f"{s['preview'][:60]}", C["cyan"])
        elif cmd == "/permissions":
            from .permissions import PermissionManager, MODES
            perms = getattr(self.agent, "permissions", None)
            if perms is None:
                perms = PermissionManager()
                self.agent.permissions = perms
            sub = arg.strip()
            # permrules: /permissions rules [allow|ask|deny|remove] <tool>
            if sub.lower().startswith("rules"):
                from .permrules import handle_permissions_rules
                rest = sub[len("rules"):].strip()
                self.print_info(handle_permissions_rules(self, rest), C["fg"])
            elif sub:
                try:
                    perms.set_mode(sub)
                    self.print_info(f"✓ {perms.describe()}", C["green"])
                except ValueError as e:
                    self.print_error(str(e))
            else:
                self.print_info(perms.describe(), C["fg"])
                self.print_info("options: " + ", ".join(MODES), C["dim"])
        elif cmd == "/output-style":
            from . import outstyle
            sub = arg.split()[0].lower() if arg else ""
            if sub:
                try:
                    outstyle.set_style(self.agent, sub)
                    self.print_info(f"✓ output style → {sub}", C["green"])
                except ValueError:
                    self.print_error("styles: " +
                                     " · ".join(sorted(outstyle.STYLES)))
            else:
                self.print_info(
                    "output styles: " +
                    " · ".join(sorted(outstyle.STYLES)) +
                    f"  (current: {outstyle.get_style(self.agent)})",
                    C["fg"])
        elif cmd == "/plan":
            from .planmode import enter_plan_mode, exit_plan_mode
            sub = arg.strip().lower()
            if sub in ("exit", "off", "done"):
                exit_plan_mode(self.agent)
                self.print_info("✓ plan mode off", C["green"])
            else:
                enter_plan_mode(self.agent)
                self.print_info(
                    "✓ plan mode on — write your plan with the PlanWrite "
                    "tool, then approve to execute", C["green"])
        elif cmd == "/todos":
            from .todos import todo_panel_lines
            lines = todo_panel_lines()
            if not lines:
                self.print_info("no todos — use TodoWrite to track tasks",
                                C["dim"])
            else:
                self.print_info("\n".join(lines), C["fg"])
        elif cmd == "/cron":
            self._handle_cron(arg)
        elif cmd == "/history":
            self.open_history()
        elif cmd == "/save":
            path = self.agent.save_session()
            if path:
                self.print_info(f"✓ session saved → {path}", C["green"])
            else:
                self.print_error("could not save session")
        elif cmd == "/approve":
            self.cfg.auto_approve = not self.cfg.auto_approve
            self.cfg.save()
            state = "ON" if self.cfg.auto_approve else "OFF"
            self.print_info(f"✓ auto-approve: {state}",
                            C["yellow"] if self.cfg.auto_approve else C["dim"])
        elif cmd == "/reasoning":
            self.cfg.show_reasoning = not self.cfg.show_reasoning
            self.cfg.save()
            # keep in sync with /thinking (thinking module flag)
            try:
                if hasattr(self.agent, "set_thinking_visible"):
                    self.agent.set_thinking_visible(self.cfg.show_reasoning)
            except Exception:
                pass
            state = "ON" if self.cfg.show_reasoning else "OFF"
            self.print_info(f"✓ reasoning display: {state}", C["pink"])
        elif cmd == "/usage":
            self.print_usage(self.agent.turns)
        elif cmd == "/goal":
            self._cmd_goal(arg)
        elif cmd == "/autonomy":
            self._cmd_autonomy(arg)
        elif cmd == "/focus":
            self._cmd_focus(arg)
        elif cmd == "/render":
            self._cmd_render(arg)
        elif cmd == "/export":
            from .export import handle_export
            handle_export(self, arg)
        elif cmd == "/backup":
            from .backup import handle_backup
            handle_backup(self, arg)
        elif cmd == "/webhooks":
            from .webhook import handle_webhooks
            handle_webhooks(self, arg)
        elif cmd == "/restore":
            from .backup import handle_restore
            handle_restore(self, arg)
        elif cmd == "/export-report":
            self._cmd_export(arg)
        elif cmd == "/forecast":
            self.print_info(self.agent.get_forecast(), C["cyan"])
        elif cmd == "/health":
            self._cmd_health()
        elif cmd == "/notify":
            self._cmd_notify(arg)
        elif cmd == "/resume":
            # Feature: try session-file resume first, fall back to branch resume
            from . import sessions as _sessions
            target = arg.strip() or "latest"
            data = _sessions.load(target)
            if data is None:
                self._cmd_resume(arg)
            else:
                self.print_info(_sessions.resume_into(self.agent, data),
                                C["green"])
        elif cmd == "/resume-turn":
            # Feature: turnresume — review / resume crashed turns
            from .turnresume import (list_incomplete, resume_turn,
                                     load_incomplete_turns)
            inc = getattr(self.agent, "incomplete_turns", None)
            if inc is None:  # module never registered — rescan on demand
                inc = load_incomplete_turns()
                self.agent.incomplete_turns = inc
            if not arg:
                self.print_info(list_incomplete(self.agent), C["fg"])
            else:
                parts = arg.split()
                sid = parts[0]
                if "--yes" in parts[1:]:
                    self.print_info(resume_turn(self.agent, sid), C["green"])
                else:
                    self.print_info(
                        f"resume turn from session {sid}? This re-injects "
                        "the crashed turn's messages and tool history. "
                        f"Re-run with --yes to confirm: "
                        f"/resume-turn {sid} --yes", C["yellow"])
        elif cmd == "/state":
            self._cmd_state()
        elif cmd == "/rewind":
            self._cmd_rewind(arg)
        elif cmd == "/revert":
            self._cmd_revert(arg)
        elif cmd == "/fork":
            self._cmd_fork(arg)
        elif cmd == "/why":
            self._cmd_why(arg)
        elif cmd == "/impact":
            self._cmd_impact(arg)
        elif cmd == "/forge":
            self._cmd_forge(arg)
        elif cmd == "/oracle":
            self.print_info(self.agent.oracle.format_report(), C["cyan"])
        elif cmd == "/budget":
            self._cmd_budget(arg)
        elif cmd == "/constitution":
            self._cmd_constitution(arg)
        elif cmd == "/replay":
            self._cmd_replay()
        elif cmd == "/memory":
            self._cmd_memory()
        elif cmd == "/judge":
            self._cmd_judge(arg)
        elif cmd == "/compile":
            self._cmd_compile(arg)
        elif cmd == "/brain":
            self._cmd_brain(arg)
        elif cmd == "/merge":
            self._cmd_merge(arg)
        elif cmd == "/theater":
            self._cmd_theater(arg)
        elif cmd == "/debate":
            self._cmd_debate(arg)
        elif cmd == "/tower":
            self._cmd_tower(arg)
        elif cmd == "/verify":
            self._cmd_verify(arg)
        elif cmd == "/mcts":
            self._cmd_mcts(arg)
        elif cmd == "/causal":
            self._cmd_causal(arg)
        elif cmd == "/bandit":
            self._cmd_bandit(arg)
        elif cmd == "/synth":
            self._cmd_synth(arg)
        elif cmd == "/ci":
            self._cmd_ci(arg)
        elif cmd == "/tune":
            self._cmd_tune(arg)
        elif cmd == "/dual":
            self._cmd_dual(arg)
        elif cmd == "/predict":
            self._cmd_predict_impact(arg)
        elif cmd == "/vitals":
            self.print_info(self.agent.homeo.check_and_repair()
                            .format(), C["cyan"])
        elif cmd == "/attention":
            self.print_info(self.agent.attention.format_last(), C["cyan"])
        elif cmd == "/fabric":
            self._cmd_fabric(arg)
        elif cmd == "/auto":
            self._cmd_auto(arg)
        elif cmd == "/prompt":
            self._cmd_prompt(arg)
        elif cmd == "/enforcex":
            from .enforcex.sovereign import handle_enforcex
            handle_enforcex(self, arg)
        elif cmd == "/mastermind":
            self.print_info(self.agent.mastermind.format_status(), C["pink"])
        elif cmd == "/dashboard":
            self.print_info(self.agent.dashboard.render(), C["cyan"])
        elif cmd == "/router":
            self.print_info(self.agent.router.format_status(), C["green"])
        elif cmd == "/spec":
            self.print_info(self.agent.speculator.format_status(), C["cyan"])
        elif cmd == "/recall":
            self._cmd_recall(arg)
        elif cmd == "/mission":
            self._cmd_mission(arg)
        elif cmd == "/heal":
            self.print_info(self.agent.healer.format_status(), C["yellow"])
        elif cmd == "/council":
            self._cmd_council(arg)
        elif cmd == "/analyze":
            self._cmd_analyze(arg)
        elif cmd == "/graph":
            self._cmd_graph(arg)
        elif cmd == "/coverage":
            self.print_info(self.agent.coverage.format_status(), C["cyan"])
        elif cmd == "/fuzz":
            self.print_info(self.agent.fuzzer.format_status(), C["yellow"])
        elif cmd == "/mutate":
            self._cmd_mutate(arg)
        elif cmd == "/about":
            from . import __version__
            self.print_info(f"{APP_NAME} v{__version__} — advanced terminal "
                            "AI agent", C["accent"])
            self.print_info("  python + prompt_toolkit + rich · "
                            + " · ".join(p.name for p in PROVIDERS.values()), C["dim"])
        else:
            # Feature: custom user slash commands (~/.fullagent/commands/)
            from .slashcmds import expand as _expand_custom
            prompt = _expand_custom(cmd.lstrip("/"), arg)
            if prompt:
                self._dispatch(prompt)
            else:
                self.print_error(f"unknown command: {cmd} — try /help")

    def _shutdown_subagents(self) -> None:
        """Cancel active child work without creating the lazy runtime."""
        manager = self.agent.__dict__.get("subagents")
        if manager is not None:
            try:
                manager.shutdown(wait=False)
            except Exception:
                pass

    def _print_agents_dashboard(self, manager) -> None:
        """Render a compact live board for the current parallel crew."""
        from rich import box
        from rich.console import Group
        from rich.text import Text

        snapshot = manager.snapshot()
        counts = snapshot["counts"]
        state_styles = {
            "queued": C["dim"], "running": C["green"],
            "cancelling": C["yellow"], "done": C["cyan"],
            "blocked": C["orange"], "error": C["red"],
            "cancelled": C["dim"],
        }
        state_icons = {"queued": "○", "running": "●", "cancelling": "◌",
                       "done": "✓", "blocked": "!", "error": "×",
                       "cancelled": "–"}
        from .roles import role_label
        cards = []
        for row in snapshot["workers"]:
            state = row["state"]
            color = state_styles.get(state, C["fg"])
            seconds = row["elapsed_ms"] / 1000
            task = " ".join(row["task"].split())
            if len(task) > 360:
                task = task[:357] + "..."
            activity = row["current_action"] if state in {
                "queued", "running", "cancelling"} else row["summary"]
            activity = " ".join(activity.split())
            if len(activity) > 180:
                activity = activity[:177] + "..."
            pct = 100 if state == "done" else (0 if state == "queued" else min(95, max(15, row["steps"] * 10)))
            filled = int(round(pct / 20))
            pbar = "▰" * filled + "▱" * (5 - filled)
            heading = Text()
            heading.append(state_icons.get(state, "·") + "  ",
                           style=f"bold {color}")
            heading.append(state.upper(), style=f"bold {color}")
            heading.append(f"  {pbar} {pct}%", style=color)
            wave_badge = f"  ·  wave {row['wave']}" if row.get("wave", 0) > 0 else ""
            prio_badge = f"  ·  prio {row['priority']:+d}" if row.get("priority", 0) else ""
            diff_badge = (
                f"  ·  Δ+{row['lines_added']}/-{row['lines_removed']}"
                if (row.get("lines_added") or row.get("lines_removed"))
                else ""
            )
            heading.append(f"  {row['id']}  ·  {row['batch_id']}{wave_badge}{prio_badge}{diff_badge}",
                           style=C["dim"])
            tps = f" · {row['tokens_per_sec']} tok/s" if row.get("tokens_per_sec") else ""
            work = Text(
                f"{role_label(row['role'])} · {row['name']}  |  "
                f"{row['steps']} steps · {row['tool_calls']} tools  |  "
                f"{row['tokens_in']}→{row['tokens_out']} tokens{tps} · {seconds:.1f}s",
                style=C["dim"])
            task_line = Text("Task  ", style=f"bold {C['cyan']}")
            task_line.append(task)
            contents = [heading, work, task_line]
            if activity:
                contents.append(Text("↳ " + activity, style=C["dim"]))
            cards.append(Panel(
                Group(*contents), box=box.ROUNDED, border_style=color,
                padding=(0, 1), expand=True))

        summary = Text()
        summary.append(f" {snapshot['active']:02d}/{snapshot['max_active']:02d} ACTIVE ",
                       style=f"bold {C['green']}")
        summary.append(f" · {counts['running']}/{snapshot['max_workers']} worker slots")
        for state, label, color in (("queued", "queued", C["dim"]),
                                    ("done", "done", C["cyan"]),
                                    ("blocked", "blocked", C["orange"]),
                                    ("error", "errors", C["red"])):
            summary.append(f" · {counts[state]} {label}", style=color)
        summary.append(f" · {snapshot['tokens_in']}→{snapshot['tokens_out']} tokens",
                       style=C["dim"])
        if not snapshot["workers"]:
            body = Group(summary, Text(
                "No workers yet. Use TaskGroup for read-only parallel work or "
                "TaskWriteGroup for approval-gated, path-scoped edits.",
                style=C["dim"]))
        else:
            body = Group(summary, *cards)
        self.console.print(Panel(
            body, title=f"[bold {C['accent']}]Parallel crew[/]",
            subtitle=(f"latest {snapshot['latest_batch_id']} · "
                      "/agents roles · detail <id> · results [batch] · stop all"),
            border_style=C["border"], padding=(0, 1), expand=True))

    def _print_agent_roles(self) -> None:
        """Show the available specialist modes and their capability boundary."""
        from rich import box
        from rich.console import Group
        from rich.text import Text
        from .roles import ROLE_GUIDANCE, ROLES, role_label

        cards = []
        for role in sorted(ROLES):
            writes = bool(ROLES[role].get("writes"))
            access = "SCOPED WRITES · approval-gated" if writes else "READ-ONLY"
            title = Text(f"{role_label(role)}  ·  {role}",
                         style=f"bold {C['accent']}")
            title.append(f"  [{access}]",
                         style=C["yellow"] if writes else C["green"])
            cards.append(Panel(
                Group(title, Text(ROLE_GUIDANCE[role], style=C["dim"])),
                box=box.SIMPLE, border_style=C["border"],
                padding=(0, 1), expand=True))
        self.console.print(Panel(
            Group(*cards), title=f"[bold {C['accent']}]Specialist roster · "
                                 f"{len(ROLES)} roles[/]",
            subtitle="Every worker uses its role allowlist; writes stay scoped and approval-gated.",
            border_style=C["border"], padding=(0, 1), expand=True))

    def _cmd_agents(self, arg: str) -> None:
        """Inspect or control the current session's parallel subagents."""
        try:
            self.agent._ensure_session()
            manager = self.agent.subagents
        except Exception as exc:  # keep the TUI usable if an optional part fails
            self.print_error(f"subagents unavailable: {exc}")
            return
        text = (arg or "").strip()
        if not text or text.lower() in ("list", "status", "ls"):
            self._print_agents_dashboard(manager)
            return
        parts = text.split(None, 2)
        action = parts[0].lower()
        if action in ("roles", "specialists"):
            self._print_agent_roles()
            return
        if action in ("detail", "show"):
            if len(parts) < 2:
                self.print_error("usage: /agents detail <id>")
                return
            from rich.text import Text
            self.console.print(Panel(
                Text(manager.format_worker(manager.get(parts[1]))),
                title=f"[bold {C['accent']}]Subagent detail[/]",
                border_style=C["border"], expand=False))
            return
        if action in ("results", "collect"):
            batch_id = parts[1] if len(parts) > 1 else ""
            try:
                from rich.text import Text
                reports = manager.collect(batch_id=batch_id, timeout=0)
                self.console.print(Panel(
                    Text(reports),
                    title=f"[bold {C['accent']}]Batch results[/]",
                    border_style=C["border"], expand=True))
            except Exception as exc:
                self.print_error(str(exc))
            return
        if action == "dag":
            batch_id = parts[1] if len(parts) > 1 else ""
            try:
                from rich.text import Text
                self.console.print(Panel(
                    Text(manager.format_dag(batch_id=batch_id)),
                    title=f"[bold {C['accent']}]Swarm DAG Topology[/]",
                    border_style=C["border"], expand=True))
            except Exception as exc:
                self.print_error(str(exc))
            return
        if action in ("blackboard", "bb"):
            from rich.text import Text
            self.console.print(Panel(
                Text(manager.blackboard.format_board()),
                title=f"[bold {C['accent']}]Shared Subagent Blackboard[/]",
                border_style=C["border"], expand=True))
            return
        if action == "diff":
            from rich.text import Text
            target = parts[1] if len(parts) > 1 else ""
            workers = [manager.get(target)] if target else manager.list()
            workers = [w for w in workers if w is not None]
            diffs = [d for w in workers for d in getattr(w, "diffs", [])]
            if not diffs:
                self.print_info("no subagent file diffs recorded yet", C["dim"])
                return
            for d in diffs[-10:]:
                t = Text()
                for ln in d.get("diff", "").splitlines()[:60]:
                    st = (C["green"] if ln.startswith("+") else C["red"] if ln.startswith("-")
                          else C["cyan"] if ln.startswith("@@") else C["fg"])
                    t.append(ln + "\n", style=st)
                self.console.print(Panel(
                    t,
                    title=(f"[bold {C['accent']}]{escape(str(d['worker_id']))} · "
                           f"{escape(str(d['path']))}[/] "
                           f"[bold {C['green']}]+{d['added']}[/] [bold {C['red']}]-{d['removed']}[/]"),
                    border_style=C["border"], expand=True))
            return
        if action in ("synth", "metrics"):
            batch_id = parts[1] if len(parts) > 1 else ""
            try:
                from rich.text import Text
                synth = manager.synthesize_batch(batch_id=batch_id)
                lines = [
                    f"Batch: {synth['batch_id'] or '(none)'}  ·  Workers: {synth['total_workers']}  ·  Waves: {synth['waves']}",
                    f"Settled: {synth['all_settled']}  ·  Steps: {synth['total_steps']}  ·  Tool calls: {synth['total_tool_calls']}",
                    f"Tokens: {synth['tokens_in']}→{synth['tokens_out']}  ·  Lines: +{synth.get('lines_added', 0)}/-{synth.get('lines_removed', 0)}  ·  Roles: {', '.join(synth['roles']) or 'none'}",
                ]
                if synth["files_touched"]:
                    lines.append("Files touched: " + ", ".join(synth["files_touched"]))
                self.console.print(Panel(
                    Text("\n".join(lines)),
                    title=f"[bold {C['accent']}]Batch synthesis & DAG metrics[/]",
                    border_style=C["border"], expand=True))
            except Exception as exc:
                self.print_error(str(exc))
            return
        if action == "spawn":
            rest = text[len(parts[0]):].strip()
            role = "researcher"
            task_text = rest
            if rest:
                first, _, tail = rest.partition(" ")
                from .roles import ROLES
                if first.lower() in ROLES:
                    role, task_text = first.lower(), tail.strip()
            if not task_text:
                self.print_error(
                    "usage: /agents spawn [role] <read-only investigation task>")
                return
            try:
                worker = manager.spawn(task_text, role=role,
                                       allow_writes=False)
                from .roles import role_label
                self.print_info(
                    f"✓ started {worker.id} · {role_label(worker.role)} · "
                    f"batch {worker.batch_id} (read-only); use /agents results "
                    f"{worker.batch_id} or /agents list",
                    C["green"])
            except Exception as exc:
                self.print_error(str(exc))
            return
        if action == "wait":
            selector = parts[1] if len(parts) > 1 else "all"
            ids = [selector] if selector.lower() != "all" else None
            try:
                manager.wait(ids, timeout=30)
                self._print_agents_dashboard(manager)
            except Exception as exc:
                self.print_error(str(exc))
            return
        if action in ("errors", "failures"):
            # legacy agentscmd parity: full error messages for failed subagents
            failed = [w for w in manager.list() if w.state == "error"]
            from rich.text import Text
            if not failed:
                self.print_info("no failed subagents", C["green"])
                return
            lines = [f"failed subagents ({len(failed)}):", ""]
            for w in failed:
                lines.append(f"  {w.id} ({w.name}):")
                lines.append(f"    {w.error or '(no error message)'}")
                lines.append("")
            self.console.print(Panel(
                Text("\n".join(lines)),
                title=f"[bold {C['accent']}]Subagent errors[/]",
                border_style=C["border"], expand=True))
            return
        if action in ("stop-all", "stopall"):
            # legacy agentscmd parity: /agents stop-all → stop every subagent
            active = [w for w in manager.list()
                      if w.state in {"queued", "running", "cancelling"}]
            try:
                for w in active:
                    manager.stop(w.id)
                self.print_info(
                    f"✓ stopped {len(active)} active subagent"
                    f"{'s' if len(active) != 1 else ''}", C["yellow"])
            except Exception as exc:
                self.print_error(str(exc))
            return
        if action == "stop":
            if len(parts) < 2:
                self.print_error("usage: /agents stop <id|all>")
                return
            try:
                if parts[1].lower() == "all":
                    active = [worker for worker in manager.list()
                              if worker.state in {"queued", "running", "cancelling"}]
                    for worker in active:
                        manager.stop(worker.id)
                    self.print_info(
                        f"Cancellation requested for {len(active)} active worker(s).",
                        C["yellow"])
                else:
                    worker = manager.stop(parts[1])
                    self.print_info(manager.format_worker(worker), C["yellow"])
            except Exception as exc:
                self.print_error(str(exc))
            return
        self.print_error(
            "usage: /agents [list | roles | dag [batch] | diff [id] | blackboard | synth [batch] | "
            "spawn [role] <read-only task> | detail <id> | results [batch-id] | wait [id|all] | "
            "errors | stop <id|all|stop-all>] — scoped coding delegation is approval-gated")

    def _cmd_theme(self, arg: str) -> None:
        """Switch TUI color palette live: /theme [neon|dracula|nord|emerald]."""
        name = (arg or "").strip().lower()
        if not name:
            self.print_info(
                f"active theme: {getattr(self, '_theme_name', 'neon')}  ·  "
                f"available: {', '.join(sorted(THEMES))}",
                C["cyan"],
            )
            return
        if name not in THEMES:
            self.print_error(
                f"unknown theme {name!r} — choose from: {', '.join(sorted(THEMES))}"
            )
            return
        self._theme_name = name
        C.update(THEMES[name])
        self.cfg.extra["theme"] = name
        self.cfg.save()
        self._set_flash(f"✦ theme → {name}", C["green"])
        self.print_info(f"✓ switched TUI palette to {name}", C["green"])

    def _cmd_diff(self, arg: str) -> None:
        """Inspect live/recorded file diffs from the session: /diff [last|all|clear|<path>]."""
        from .diffpreview import render_rich_diff_panel
        history = getattr(self, "_diff_history", None) or []
        sub = (arg or "").strip()
        if sub.lower() == "clear":
            if hasattr(self, "_diff_history"):
                self._diff_history.clear()
            self.print_info("✓ cleared session diff history", C["green"])
            return
        if not history:
            self.print_info(
                "no file diffs recorded in this session yet — diffs stream live on every write/edit",
                C["dim"],
            )
            return
        if not sub or sub.lower() == "last":
            selected = [history[-1]]
        elif sub.lower() == "all":
            selected = history[-10:]
        else:
            selected = [
                h for h in history if sub.lower() in h.get("path", "").lower()
            ]
            if not selected:
                self.print_error(f"no recorded diff matching {sub!r}")
                return
        for item in selected:
            self.console.print(
                render_rich_diff_panel(
                    item["tool"], item["args"], max_lines=60, palette=C
                )
            )

    # -- event-log commands ------------------------------------------------------

    def _cmd_goal(self, arg: str) -> None:
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else "status"
        rest = parts[1].strip() if len(parts) > 1 else ""

        if sub in ("", "status", "show"):
            self.print_info(self.agent.goal.format(), C["cyan"])
        elif sub == "set":
            self._goal_set(rest)
        elif sub == "prove":
            if not rest:
                self.print_error("usage: /goal prove <clause-id>  "
                                 "(or /goal prove-all)")
                return
            ok, detail = self.agent.goal.prove_by_predicate(rest)
            self.print_info(f"{'✓' if ok else '✗'} {rest}: {detail}",
                            C["green"] if ok else C["red"])
            self.print_info(self.agent.goal.format(), C["dim"])
        elif sub == "prove-all":
            st = self.agent.goal.status()
            for c in st.clauses:
                if c.proof and not c.advisory:
                    ok, detail = self.agent.goal.prove_by_predicate(c.id)
                    self.print_info(f"{'✓' if ok else '✗'} {c.id}: {detail}",
                                    C["green"] if ok else C["red"])
            self.print_info(self.agent.goal.format(), C["dim"])
        elif sub == "close":
            result = self.agent.goal.close(fresh=True)
            color = C["green"] if result["state"] == "ACHIEVED" else C["yellow"]
            self.print_info(f"GOAL CLOSED: {result['state']}", color)
            for r in result["reasons"]:
                self.print_info(f"  - {r}", C["dim"])
            self.print_info(result["bundle"], C["cyan"])
        elif sub == "waive":
            wparts = rest.split("--reason", 1)
            cid = wparts[0].strip()
            reason = wparts[1].strip() if len(wparts) > 1 else "human waiver"
            if not cid:
                self.print_error("usage: /goal waive <clause-id> --reason '…'")
                return
            if self.agent.goal.waive(cid, reason):
                self.print_info(f"✓ clause {cid} waived (recorded as an event)",
                                C["yellow"])
            else:
                self.print_error(f"no such clause: {cid}")
        elif sub == "clear":
            self.agent.goal.clear()
            self.print_info("✓ goal cleared", C["dim"])
        else:
            self.print_error("goal subcommands: set · prove · prove-all · "
                             "close · status · waive · clear")

    def _goal_set(self, rest: str) -> None:
        """Parse the TUI goal grammar into a contract:

        /goal set <statement> | <clause> @ <proof-type>:<arg> | …
        Prefix a clause with ! for an anti-clause, ~ for an invariant.
        A clause piece may also be raw JSON for full control.

        Proof types: exit_code, file_exists, file_contains, file_matches,
        command_output_contains, ast_assert, diff_assert, file_unchanged,
        tool_delta.
        """
        import json as _json
        from .goal import GoalContractError
        if not rest:
            self.print_error(
                "usage: /goal set <statement> | <clause> @ <type>:<arg> | …\n"
                "  e.g. /goal set ship it | tests pass @ exit_code:pytest -q"
                " | docs exist @ file_exists:docs.md\n"
                "  ! prefix = anti-clause, ~ prefix = invariant")
            return
        pieces = [p.strip() for p in rest.split("|") if p.strip()]
        if not pieces:
            self.print_error("goal needs a statement: "
                             "/goal set <statement> | <clause> @ <type>:<arg>")
            return
        statement = pieces[0]
        clauses: list[dict] = []
        anti: list[dict] = []
        invariants: list[dict] = []
        for i, piece in enumerate(pieces[1:], 1):
            try:
                if piece.startswith("{"):
                    c = _json.loads(piece)
                    clauses.append(c)
                    continue
                is_anti = piece.startswith("!")
                is_inv = piece.startswith("~")
                text = piece.lstrip("!~ ").strip()
                if "@" not in text:
                    self.print_error(
                        f"clause {i} needs a machine-checkable proof: "
                        f"'{text}' @ <type>:<arg>  (or mark it advisory "
                        "with raw JSON)")
                    return
                ctext, proof_str = [x.strip() for x in text.split("@", 1)]
                proof = self._parse_proof(proof_str)
                if proof is None:
                    return
                if is_anti:
                    anti.append({"id": f"A{len(anti) + 1}", "text": ctext,
                                 "check": proof})
                elif is_inv:
                    invariants.append({"id": f"I{len(invariants) + 1}",
                                       "text": ctext, "check": proof})
                else:
                    clauses.append({"id": f"C{len(clauses) + 1}",
                                    "text": ctext, "weight": 1.0,
                                    "proof": proof})
            except _json.JSONDecodeError as e:
                self.print_error(f"clause {i}: invalid JSON — {e}")
                return
        if not clauses:
            self.print_error("a goal needs at least one clause with a proof")
            return
        try:
            self.agent.goal.set_goal(statement, clauses, anti, invariants)
        except GoalContractError as e:
            self.print_error(f"contract rejected: {e}")
            return
        self.print_info(f"✓ goal contract frozen — {len(clauses)} clause(s)"
                        + (f", {len(anti)} anti" if anti else "")
                        + (f", {len(invariants)} invariant" if invariants
                           else ""), C["green"])
        self.print_info(self.agent.goal.format(), C["dim"])

    def _parse_proof(self, s: str) -> dict | None:
        """'<type>:<arg>' -> predicate dict (None + error print on failure)."""
        if ":" not in s:
            self.print_error(f"proof needs '<type>:<arg>', got: {s!r}")
            return None
        ptype, arg = s.split(":", 1)
        ptype, arg = ptype.strip(), arg.strip()
        if ptype in ("exit_code", "tool_delta"):
            return {"type": ptype, "command": arg, "expect": 0}
        if ptype == "file_exists":
            return {"type": ptype, "path": arg}
        if ptype in ("file_contains", "file_matches",
                     "command_output_contains"):
            if ":" not in arg:
                self.print_error(f"{ptype} needs '<path-or-cmd>:<text>'")
                return None
            a, b = arg.split(":", 1)
            key = "path" if ptype.startswith("file") else "command"
            field2 = "text" if ptype != "file_matches" else "pattern"
            return {"type": ptype, key: a.strip(), field2: b.strip()}
        if ptype == "ast_assert":
            if ":" not in arg:
                self.print_error("ast_assert needs '<path>:<symbol>'")
                return None
            a, b = arg.split(":", 1)
            return {"type": ptype, "path": a.strip(), "symbol": b.strip()}
        if ptype == "diff_assert":
            return {"type": ptype, "path": arg, "forbid": []}
        if ptype == "file_unchanged":
            if ":" not in arg:
                self.print_error("file_unchanged needs '<path>:<sha256>'")
                return None
            a, b = arg.split(":", 1)
            return {"type": ptype, "path": a.strip(),
                    "baseline_hash": b.strip()}
        self.print_error(f"unknown proof type: {ptype!r}")
        return None

    def _cmd_autonomy(self, arg: str) -> None:
        from .agent import AUTONOMY_LEVELS
        if not arg:
            lines = [f"current: L{self.agent.autonomy} — "
                     f"{AUTONOMY_LEVELS[self.agent.autonomy]}"]
            for level, desc in AUTONOMY_LEVELS.items():
                lines.append(f"  L{level}  {desc}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        try:
            level = int(arg)
        except ValueError:
            self.print_error("usage: /autonomy <0-5>")
            return
        desc = self.agent.set_autonomy(level)
        self.print_info(f"✓ autonomy → L{self.agent.autonomy} — {desc}",
                        C["green"])

    def _cmd_focus(self, arg: str) -> None:
        """FOCUS MODE (deep work): arms auto-continuation. After each turn
        the kernel decides — mechanically — whether work remains, and the
        UI submits the next continuation turn automatically, until the
        goal closes, the agent stalls, the budget pauses, or N turns land.
        """
        arg = arg.strip().lower()
        if arg in ("", "status"):
            state = (f"ARMED — {self._focus_remaining} auto-turn(s) left"
                     if self._focus_remaining > 0 else "off")
            self.print_info(f"🎯 focus mode: {state}\n"
                            "  /focus <1-20>  arm, then send your task\n"
                            "  /focus off     disarm", C["cyan"])
            return
        if arg == "off":
            self._focus_remaining = 0
            self.print_info("🎯 focus disarmed", C["yellow"])
            return
        try:
            n = max(1, min(int(arg), 20))
        except ValueError:
            self.print_error("usage: /focus <1-20> | /focus off")
            return
        self._focus_remaining = n
        self.agent._focus_history.clear()
        goal = self.agent.goal.status()
        hint = ("an active goal drives continuation — focus stops when "
                "every clause is proven or progress stalls"
                if goal.active else
                "no active goal — focus stops when the agent answers "
                "without further tool work")
        self.print_info(f"🎯 focus armed: up to {n} auto-turns. {hint}. "
                        f"Send your task now.", C["green"])

    def _cmd_render(self, arg: str) -> None:
        """Toggle rendered-markdown replies: streaming stays in the border
        preview, the finished reply prints as rich Markdown."""
        arg = arg.strip().lower()
        current = bool(self.cfg.extra.get("render_markdown", False))
        if arg in ("on", "off"):
            new_state = arg == "on"
        elif arg == "":
            new_state = not current
        else:
            self.print_error("usage: /render [on|off]")
            return
        self.cfg.extra["render_markdown"] = new_state
        self.cfg.save()
        self.print_info(f"✓ markdown rendering → "
                        f"{'ON (replies render as rich Markdown)' if new_state else 'OFF (raw streaming)'}",
                        C["green"])

    def _cmd_export(self, arg: str) -> None:
        fmt = arg.strip().lower()
        if fmt not in ("", "md", "markdown", "html"):
            self.print_error("usage: /export [md|html]")
            return
        fmt = "html" if fmt == "html" else "md"
        try:
            path = self.agent.export_report(fmt)
        except OSError as e:
            self.print_error(f"cannot write report: {e}")
            return
        self.print_info(f"✓ audit report exported → {path}", C["green"])

    def _cmd_health(self) -> None:
        h = self.agent.health
        lines = ["PROVIDER HEALTH"]
        lines.append(f"  failovers this session : {h['failovers']}")
        if h["model_errors"]:
            for m, n in sorted(h["model_errors"].items()):
                lines.append(f"  model errors           : {m} × {n}")
        else:
            lines.append("  model errors           : none")
        lines.append("  automatic model switch: disabled (manual selection only)")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_notify(self, arg: str) -> None:
        try:
            state = self.agent.notifier.configure(arg)
        except ValueError as e:
            self.print_error(str(e))
            return
        self.print_info(f"✓ notifications → {state}", C["green"])

    def _cmd_resume(self, arg: str) -> None:
        branch = arg.strip()
        catalog = self.agent.sessions_catalog()
        if not branch:
            if not catalog:
                self.print_info("no sessions found", C["dim"])
                return
            lines = ["SESSIONS — /resume <branch> to continue one:"]
            import time as _time
            for c in catalog[:12]:
                stamp = (_time.strftime("%m-%d %H:%M",
                                        _time.localtime(c["started"]))
                         if c["started"] else "?")
                lines.append(f"  ◆ {c['branch']:<14} session "
                             f"{c['session_id'] or '?'} · "
                             f"{c['events']} events · {stamp}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        try:
            n = self.agent.resume_session(branch)
        except ValueError as e:
            self.print_error(str(e))
            return
        self.print_info(f"✓ resumed branch {branch!r} — {n} message(s) "
                        f"restored from the event log", C["green"])

    def _cmd_state(self) -> None:
        st = self.agent.state()
        goal = self.agent.goal.status()
        lines = [
            f"branch: {st.branch}   head seq: {st.head_seq}   "
            f"events: {len(self.agent.log)}",
            f"cost: {st.cost_summary()}",
            f"tool calls: {st.tool_calls}   errors: {st.tool_errors}   "
            f"commands: {st.commands_run}",
            f"autonomy: L{st.autonomy}",
        ]
        if st.files_touched:
            lines.append("files touched: " + ", ".join(sorted(st.files_touched)))
        if goal.active:
            proven = sum(1 for c in goal.clauses if c.state == "PROVEN")
            lines.append(f"goal: {goal.statement} — "
                         f"{proven}/{len(goal.clauses)} clauses "
                         f"({(1 - goal.distance) * 100:.0f}% done)")
        if st.dead_ends:
            lines.append(f"dead ends: {len(st.dead_ends)}")
        if st.verdicts:
            passed = sum(1 for v in st.verdicts if v.get("passed"))
            lines.append(f"judge verdicts: {passed}/{len(st.verdicts)} passed")
        if st.episodes:
            lines.append(f"memory episodes: {len(st.episodes)}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_rewind(self, arg: str) -> None:
        if not arg:
            evs = self.agent.log.events()[-12:]
            if not evs:
                self.print_info("log is empty", C["dim"])
                return
            lines = ["recent events (pick a seq, then /rewind <seq>):"]
            for ev in evs:
                preview = ""
                if ev.type in ("user.message", "assistant.message"):
                    preview = str(ev.data.get("text", ""))[:50]
                elif ev.type in ("tool.call", "tool.result"):
                    preview = str(ev.data.get("name", ""))
                lines.append(f"  {ev.seq:>4}  {ev.type:<20} {preview}")
            self.print_info("\n".join(lines), C["dim"])
            return
        try:
            seq = int(arg)
        except ValueError:
            self.print_error("usage: /rewind <seq>  (see /rewind for seqs)")
            return
        new_head, kept = self.agent.rewind_to(seq)
        self.print_info(f"✓ rewound to seq {new_head} — {kept} message(s) kept "
                        "(tool-call context is dropped)", C["green"])

    def _cmd_fork(self, arg: str) -> None:
        branch = self.agent.fork_timeline(name=arg or None)
        self.print_info(f"✓ forked timeline → branch '{branch}' "
                        "(now continuing on it)", C["green"])

    def _cmd_revert(self, arg: str) -> None:
        """REVERT (§9.1): files only return to seq N — the agent KEEPS its
        memory. This is what feeds the Dead-End Ledger."""
        if not arg:
            self.print_error("usage: /revert <seq>  (see /rewind for seqs)")
            return
        try:
            seq = int(arg)
        except ValueError:
            self.print_error("usage: /revert <seq>")
            return
        result = self.agent.revert_files_to(seq)
        if "error" in result:
            self.print_error(result["error"])
            return
        self.print_info(f"✓ files reverted to seq {seq} — "
                        f"{result['restored']} restored, "
                        f"{result['removed']} removed "
                        "(agent memory kept)", C["green"])

    def _cmd_why(self, arg: str) -> None:
        """Appendix A `argus why`: walk the causation chain from an event
        back to the human instruction that caused it."""
        if not arg:
            self.print_error("usage: /why <seq>  (see /rewind for seqs)")
            return
        try:
            seq = int(arg)
        except ValueError:
            self.print_error("usage: /why <seq>")
            return
        evs = self.agent.log.events()
        target = next((e for e in evs if e.seq == seq), None)
        if target is None:
            self.print_error(f"no event at seq {seq}")
            return
        chain = self.agent.log.why(target.id)
        lines = [f"causal chain for seq {seq} ({target.type}):"]
        for i, ev in enumerate(chain):
            indent = "  " * i
            preview = ""
            if ev.type in ("user.message", "assistant.message"):
                preview = str(ev.data.get("text", ""))[:40]
            elif ev.type in ("tool.call", "tool.result"):
                preview = str(ev.data.get("name", ""))
            clause = f"  [clause {ev.correlation_id}]" \
                if ev.correlation_id else ""
            lines.append(f"{indent}← seq {ev.seq} {ev.type} "
                         f"({ev.actor}) {preview}{clause}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_impact(self, arg: str) -> None:
        """§15.2 the killer query: blast radius of changing a symbol."""
        parts = arg.split(None, 1)
        if not parts:
            self.print_error("usage: /impact <symbol> [path]")
            return
        symbol = parts[0]
        path = parts[1].strip() if len(parts) > 1 else "."
        self.print_info(f"⠹ indexing {path}…", C["dim"])

        def run():
            self.agent.nexus.index(path)
            self.console.print(Text(self.agent.nexus.format_impact(symbol),
                                    style=C["fg"]))

        self._bg(run)

    def _cmd_forge(self, arg: str) -> None:
        sub = arg.strip().lower()
        if sub == "drift":
            delta = self.agent.forge.drift()
            if delta is None:
                self.print_info("✓ no environment drift detected", C["green"])
            else:
                self.print_info(f"⚠ environment drift: {delta['changed']}",
                                C["yellow"])
        else:
            d = self.agent.forge.probe()
            lines = [f"environment digest: {d['digest']}",
                     f"  os {d['os']} {d['arch']}   python {d['python']}",
                     f"  cwd {d['cwd']}",
                     f"  lockfile {d['lockfile_hash'] or 'none'}"]
            if d["tools"]:
                lines.append("  tools: " + ", ".join(
                    f"{k}={v.split()[0] if v else '?'}"
                    for k, v in d["tools"].items()))
            self.print_info("\n".join(lines), C["cyan"])

    def _cmd_budget(self, arg: str = "") -> None:
        gov = self.agent.budget_gov
        parts = arg.split()
        if parts:
            sub = parts[0].lower()
            try:
                if sub == "reset":
                    gov.reset()
                    self.print_info("budget spend reset — the governor "
                                    "counts from now", C["green"])
                    return
                if sub in ("steps", "usd", "tokens", "files") \
                        and len(parts) >= 2:
                    self.print_info(gov.set_limit(sub, parts[1]), C["green"])
                    return
                if sub in ("extend", "set") and len(parts) >= 3 \
                        and parts[1].lower() in ("steps", "usd", "tokens",
                                                 "files"):
                    self.print_info(
                        gov.set_limit(parts[1].lower(), parts[2]), C["green"])
                    return
            except ValueError as e:
                self.print_error(f"bad value: {e}")
                return
            self.print_error("usage: /budget [reset | steps N | usd X | "
                             "tokens N | files N]")
            return
        s = gov.spend()
        b = gov.budget
        ok, reason = gov.check()

        def _fmt(v) -> str:
            return "∞" if v == float("inf") else str(v)

        lines = [f"budget {'OK' if ok else 'BREACHED'}",
                 f"  usd    ${s['usd']:.4f} / ${_fmt(b.max_usd)}",
                 f"  steps  {s['steps']} / {_fmt(b.max_steps)}",
                 f"  tokens {s['tokens']} / {_fmt(b.max_tokens)}",
                 f"  files  {s['files']} / {_fmt(b.max_files)}",
                 "  spend is UNLIMITED by default — set a cap with "
                 "/budget steps N · /budget usd X · /budget reset"]
        if not ok:
            lines.append(f"  ⚠ {reason}")
        self.print_info("\n".join(lines),
                        C["green"] if ok else C["red"])

    def _cmd_constitution(self, arg: str) -> None:
        sub = arg.strip().lower()
        path = self.agent.oracle.constitution_path()
        if sub == "edit":
            if path:
                self.print_info(f"constitution file: {path}", C["dim"])
                self.print_info("edit it directly — it is human-owned and "
                                "never auto-modified. It is always present "
                                "in the agent's context.", C["cyan"])
            else:
                self.print_error("no memory dir configured")
        else:
            text = self.agent.oracle.read_constitution()
            if text.strip():
                self.print_info("CONSTITUTION (standing rules):\n" + text,
                                C["cyan"])
            else:
                self.print_info(f"constitution is empty — create {path} "
                                "with your standing rules", C["dim"])

    def _cmd_replay(self) -> None:
        """Replay the session log as a text film (§26)."""
        from .kernel import replay
        lines = ["REPLAY — the session as a film:"]
        for ev in replay(self.agent.log):
            t = ev.type
            d = ev.data
            if t == "user.message":
                lines.append(f"  [{ev.seq}] ❯ {str(d.get('text', ''))[:60]}")
            elif t == "assistant.message":
                lines.append(f"  [{ev.seq}] ◆ {str(d.get('text', ''))[:60]}")
            elif t == "tool.call":
                lines.append(f"  [{ev.seq}] ⚙ {d.get('name', '')}")
            elif t == "tool.result":
                icon = "✓" if d.get("status") == "done" else "✗"
                lines.append(f"  [{ev.seq}] {icon} {d.get('name', '')} "
                             f"({d.get('status', '')})")
            elif t == "snapshot.taken":
                lines.append(f"  [{ev.seq}] 📸 snapshot {str(d.get('tree', ''))[:10]}")
            elif t == "clause.proven":
                lines.append(f"  [{ev.seq}] ★ clause {d.get('clause', '')} PROVEN")
            elif t == "goal.closed":
                lines.append(f"  [{ev.seq}] ■ GOAL {d.get('state', '')}")
            elif t == "cost.incurred":
                lines.append(f"  [{ev.seq}] $ cost "
                             f"{d.get('tokens_in', 0)}→{d.get('tokens_out', 0)} tok")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_memory(self) -> None:
        block = self.agent.memory.context_block(max_episodes=5)
        self.print_info(block, C["cyan"])

    def _cmd_judge(self, arg: str) -> None:
        import json as _json
        predicate: dict | None = None
        arg = arg.strip()
        if arg.startswith("{"):
            try:
                predicate = _json.loads(arg)
            except ValueError as e:
                self.print_error(f"invalid JSON predicate: {e}")
                return
        elif arg:
            # shorthand: /judge <type> <value>  → predicate with default key
            tokens = arg.split(None, 1)
            ptype = tokens[0]
            value = tokens[1] if len(tokens) > 1 else ""
            key = {"exit_code": "command",
                   "file_exists": "path",
                   "file_contains": "path",
                   "file_matches": "path",
                   "command_output_contains": "command"}.get(ptype)
            if key is None:
                self.print_error("predicate types: exit_code · file_exists · "
                                 "file_contains · file_matches · "
                                 "command_output_contains")
                return
            predicate = {"type": ptype, key: value}
        else:
            self.print_error("usage: /judge <type> <arg>  or  "
                             '/judge {"type": "file_exists", "path": "…"}')
            return

        def run():
            verdict = self.agent.judge.check(predicate)
            icon = "✓" if verdict.passed else "✗"
            color = C["green"] if verdict.passed else C["red"]
            self.print_info(f"{icon} [{verdict.kind}] {verdict.detail}", color)
            if verdict.evidence:
                self.print_info(f"  evidence: {verdict.evidence[:200]}",
                                C["dim"])

        self._bg(run)

    # -- v5 advanced subsystem commands ----------------------------------

    def _cmd_compile(self, arg: str) -> None:
        """Compile a goal into a plan without delegated execution."""
        goal = arg.strip()
        if not goal:
            self.print_error("usage: /compile <goal>")
            return
        self.print_info("planning…", C["pink"])
        def run():
            plan = self.agent.compiler.compile(goal)
            self.console.print(Text(self.agent.compiler.format(plan),
                                    style=C["fg"]))
        self._bg(run)
    def _cmd_brain(self, arg: str) -> None:
        """Cognitive memory: query it, put it to sleep, or read stats."""
        sub = arg.strip().lower()
        if sub == "sleep":
            stats = self.agent.brain.sleep()
            self.print_info(f"🧠 slept — merged {stats['merged']} · "
                            f"distilled {stats['distilled']} · promoted "
                            f"{stats['promoted']} · forgotten "
                            f"{stats['forgotten']}", C["pink"])
            return
        if sub in ("", "stats"):
            self.print_info(self.agent.brain.format_stats(), C["cyan"])
            return
        self.print_info(self.agent.brain.context_block(arg, k=5)
                        or "no live memories match that query", C["fg"])

    def _cmd_merge(self, arg: str) -> None:
        """Semantic timeline merge of two branches."""
        parts = arg.split()
        if len(parts) < 2:
            known = ", ".join(self.agent.log.branches())
            self.print_error(f"usage: /merge <branchA> <branchB>  "
                             f"(known: {known})")
            return
        try:
            result = self.agent.merger.merge(parts[0], parts[1])
        except ValueError as e:
            self.print_error(str(e))
            return
        self.print_info(self.agent.merger.format(result),
                        C["yellow"] if result.conflicts else C["green"])

    def _cmd_theater(self, arg: str) -> None:
        """Time-travel debugger: frames, whys, diffs, counterfactuals."""
        parts = arg.split()
        th = self.agent.theater
        if not parts:
            frames = th.frames()[-30:]
            lines = ["THEATER — last frames (scrub with /theater <seq>):"]
            for f in frames:
                lines.append(f"  seq {f['seq']:>4} {f['type']:<18} "
                             f"{f['summary'][:60]}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        if parts[0].lower() == "why" and len(parts) > 1:
            try:
                self.print_info(th.why(int(parts[1])), C["cyan"])
            except ValueError:
                self.print_error("seq must be an integer")
            return
        if parts[0].lower() in ("cf", "counterfactual") and len(parts) > 1:
            try:
                report = th.counterfactual(int(parts[1]))
            except ValueError as e:
                self.print_error(str(e))
                return
            self.print_info(th.format_cf(report), C["pink"])
            return
        if parts[0].lower() == "diff" and len(parts) > 2:
            try:
                self.print_info(th.diff(int(parts[1]), int(parts[2])),
                                C["cyan"])
            except ValueError:
                self.print_error("seqs must be integers")
            return
        try:
            frame = th.frame(int(parts[0]))
        except ValueError:
            self.print_error("usage: /theater [seq | why N | diff A B | "
                             "cf N]")
            return
        if frame is None:
            self.print_error("no event at that seq")
            return
        self.print_info(frame.format(), C["cyan"])

    def _cmd_debate(self, arg: str) -> None:
        """Multi-model debate tournament."""
        parts = arg.split(maxsplit=1)
        if parts and parts[0].lower() in ("confirm", "refute"):
            if len(parts) < 2:
                self.print_error("usage: /debate confirm|refute <model-id>")
                return
            fn = (self.agent.debate.confirm if parts[0].lower() == "confirm"
                  else self.agent.debate.refute)
            trust = fn(parts[1].strip())
            self.print_info("calibration: " + ", ".join(
                f"{m}={t:.2f}" for m, t in sorted(trust.items())), C["cyan"])
            return
        question = arg.strip()
        if not question:
            self.print_error("usage: /debate <question>")
            return

        def run():
            self.print_info(f"⚔ tournament — "
                            f"{', '.join(self.agent.debate.models)}",
                            C["pink"])
            result = self.agent.debate.run(question)
            self.console.print(Text(self.agent.debate.format(result),
                                    style=C["fg"]))

        self._bg(run)

    def _cmd_tower(self, arg: str) -> None:
        """Web Control Tower: mission-control dashboard in the browser."""
        try:
            port = int(arg.strip()) if arg.strip() else 7860
        except ValueError:
            port = 7860
        if self.agent.tower.server is not None:
            self.print_info(f"control tower already live at "
                            f"{self.agent.tower.url}", C["cyan"])
            return
        url = self.agent.tower.start(port=port)
        self.print_info(f"🖥 control tower LIVE at {url} — event river, "
                        "timeline scrubber, live command "
                        "box", C["green"])

    # -- v6 frontier commands ------------------------------------------------

    def _cmd_verify(self, arg: str) -> None:
        """Formal verification: compile a plan and model-check it, or
        audit the real history."""
        ag = self.agent
        if arg.strip().lower() == "log":
            r = ag.formal.audit_log()
            self.print_info(("HISTORY AUDIT — CLEAN ✓" if r.ok else
                             "HISTORY AUDIT — VIOLATIONS FOUND ✗")
                            + ("\n".join("" if not r.violations else
                                         "\n" + "\n".join(
                                             f"  ⚠ {v['property']}: "
                                             f"{v['why']}"
                                             for v in r.violations))),
                            C["green"] if r.ok else C["red"])
            return
        goal = arg.strip()
        if not goal:
            self.print_error("usage: /verify <goal> | /verify log")
            return

        def run():
            plan = ag.compiler.compile(goal)
            r = ag.formal.verify_plan(plan.waves)
            color = C["green"] if r.ok else C["red"]
            body = "\n".join([f"FORMAL — {'PASS' if r.ok else 'REJECTED'}"
                              f" ({r.checked} traces)"]
                             + [f"  ⚠ {v['property']}: {v['why']}"
                                for v in r.violations])
            self.print_info(body, color)

        self._bg(run)

    def _cmd_mcts(self, arg: str) -> None:
        goal = arg.strip()
        if not goal:
            self.print_error("usage: /mcts <item1>; <item2>; …")
            return
        items = [s.strip() for s in goal.split(";") if s.strip()]
        self.agent._mcts_items = items
        report = self.agent.mcts.search(
            items, ["coder", "architect", "debugger", "tester",
                    "documenter"], iterations=120, deadline_s=20.0)
        lines = [f"🌳 MCTS — score {report.best_score:.2f} · "
                 f"{report.iterations} iterations · {report.nodes} "
                 f"nodes"]
        for i, item in enumerate(items):
            lines.append(f"  [{report.best_assignment.get(i, '?')}] "
                         f"{item[:70]}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_causal(self, arg: str) -> None:
        sub = arg.strip().lower()
        if sub.startswith("do "):
            toks = arg.strip().split()
            # trailing "on"/"off" token is the switch — never a substring
            # match (a feature literally named "soft_off" must work)
            if toks[-1] in ("on", "off") and len(toks) > 2:
                enable = toks[-1] == "on"
                cause = " ".join(toks[1:-1])
            else:
                enable = True
                cause = " ".join(toks[1:])
            report = self.agent.causal.do(cause, enable)
            verdict = ("trustworthy" if report["trustworthy"]
                       else "NOT ENOUGH DATA")
            self.print_info(
                f"do({cause}) → outcome change "
                f"{report['estimated_outcome_change']:+.3f} "
                f"({verdict})", C["cyan"])
            return
        edges = self.agent.causal.discover()
        self.print_info(self.agent.causal.format(edges), C["cyan"])

    def _cmd_bandit(self, arg: str) -> None:
        if not arg.strip():
            self.print_info(self.agent.bandit.format(), C["cyan"])
            return
        rec = self.agent.bandit.recommend(arg)
        self.print_info(f"bandit recommends [{rec.arm}] for this "
                        f"{rec.context} task", C["green"])

    def _cmd_synth(self, arg: str) -> None:
        """Synthesize a tool from a JSON spec: name, description,
        examples ([{"args": {...}, "want": ...}])."""
        import json as _json
        try:
            spec = _json.loads(arg.strip())
        except ValueError:
            self.print_error('usage: /synth {"name": "f", "description":'
                             ' "...", "examples": [{"args": {"x": 1}, '
                             '"want": 2}]}')
            return
        from fullagent.synth import SynthSpec
        result = self.agent.synth.synthesize(SynthSpec(
            name=str(spec.get("name", "")),
            description=str(spec.get("description", "")),
            examples=spec.get("examples", [])))
        self.print_info(("✓ " if result.ok else "ERROR: ") +
                        result.reason,
                        C["green"] if result.ok else C["red"])

    def _cmd_ci(self, arg: str) -> None:
        sub = arg.strip().lower()
        ci = self.agent.ci
        if sub == "start":
            ci.start()
            self.print_info("🌊 CI pilot watching "
                            f"{ci.root} (every {ci.poll:.0f}s)",
                            C["green"])
        elif sub == "stop":
            ci.stop()
            self.print_info("CI pilot stopped", C["yellow"])
        else:
            self.print_info(ci.status(), C["cyan"])

    def _cmd_tune(self, arg: str) -> None:
        try:
            n = max(4, min(int(arg.strip() or 12), 40))
        except ValueError:
            n = 12
        tuner = self.agent.tuner

        def score(cfg: dict) -> float:
            self.agent.cfg.effort = cfg["effort"]
            return {"low": 0.45, "medium": 0.7, "high": 0.85}.get(
                cfg["effort"], 0.5) * (
                {"low": 0.9, "medium": 1.0, "high": 0.8}.get(
                    cfg["worker_steps"], 0.9))

        tuner.objective = score
        report = tuner.run(n=n)
        best = tuner.best()
        if best:
            self.agent.cfg.effort = best.config["effort"]
        self.print_info(f"🎛 tuned {report.trials} trials — best "
                        f"{best.score:.2f} with {best.config} (applied)",
                        C["green"])

    def _cmd_dual(self, arg: str) -> None:
        if not arg.strip() or arg.strip().lower() == "stats":
            self.print_info(self.agent.dual.format_stats(), C["cyan"])
            return

        def run():
            r = self.agent.dual.ask(arg.strip())
            self.print_info(f"[system {r.system} · conf {r.confidence:.2f}"
                            f" · {r.elapsed_ms}ms]\n{r.answer}",
                            C["cyan"])

        self._bg(run)

    def _cmd_predict_impact(self, arg: str) -> None:
        path = arg.strip()
        if not path:
            self.print_error("usage: /predict <file path>")
            return
        impact = self.agent.world.predict_impact(path)
        self.print_info(impact.format(), C["cyan"])

    def _cmd_fabric(self, arg: str) -> None:
        """Bitemporal knowledge graph: ask, assert, or see history."""
        parts = arg.split()
        fab = self.agent.fabric
        if not parts:
            self.print_info("usage: /fabric ask <s> <p> | assert <s> "
                            "<p> <o> | history <s> <p>", C["dim"])
            return
        sub = parts[0].lower()
        if sub == "ask" and len(parts) >= 3:
            answer = fab.ask(parts[1], parts[2])
            self.print_info(f"{parts[1]} {parts[2]} = "
                            + (answer or "unknown"), C["cyan"])
        elif sub == "assert" and len(parts) >= 4:
            fab.assert_fact(parts[1], parts[2], " ".join(parts[3:]))
            self.print_info("asserted", C["green"])
        elif sub == "history" and len(parts) >= 3:
            self.print_info(fab.history(parts[1], parts[2]), C["cyan"])
        else:
            self.print_error("usage: /fabric ask|assert|history …")

    def _cmd_auto(self, arg: str) -> None:
        """Toggle or inspect the AutoPilot self-routing brain."""
        sub = arg.strip().lower()
        ap = self.agent.autopilot
        if sub == "on":
            ap.enabled = True
            self.print_info("✓ autopilot ON — the agent will auto-enable "
                            "goal mode / real-time web as "
                            "each turn needs them", C["green"])
        elif sub == "off":
            ap.enabled = False
            self.print_info("autopilot OFF — manual control only", C["dim"])
        elif sub in ("", "status"):
            state = "ON" if ap.enabled else "OFF"
            self.print_info(f"autopilot: {state}\n"
                            "  auto-enables, per turn:\n"
                            "  ⚡ goal mode      — verifiable mission "
                            "detected (auto-drafted contract)\n"
                            "  ⚡ real-time web  — live-data question "
                            "detected\n"
                            "  toggle: /auto on · /auto off",
                            C["cyan"])
        else:
            self.print_error("usage: /auto [on|off|status]")

    def _cmd_prompt(self, arg: str) -> None:
        """Select which system prompt the model gets (systemprompt.py is
        the single source). 'main' is the compact prompt; 'master' adds
        the optional project.txt specification when that file exists."""
        from . import systemprompt
        # pick up edits to systemprompt.py without restarting the app
        try:
            self.agent.mastermind.vault.refresh()
        except Exception:
            pass
        sub = arg.strip().lower()
        if sub in ("", "list", "status"):
            current = self.cfg.prompt
            lines = [f"system prompt: {current}  (source: systemprompt.py)"]
            for name in systemprompt.names():
                mark = "●" if name == current else "○"
                size = len(systemprompt.get(name))
                lines.append(f"  {mark} {name:<8} {size:>8,} chars")
            lines.append("switch: /prompt main · /prompt master")
            self.print_info("\n".join(lines), C["cyan"])
            return
        if sub not in systemprompt.PROMPTS:
            self.print_error(f"unknown prompt {sub!r} — available: "
                             + ", ".join(systemprompt.names()))
            return
        self.cfg.prompt = sub
        self.cfg.save()
        # re-seat the live conversation's system prompt through the gate
        self.agent._reseat_system_prompt()
        size = len(self.agent._base_prompt())
        self.print_info(f"✓ system prompt → {sub} ({size:,} chars) — "
                        "applies from the next model call", C["green"])

    # -- v3 advanced subsystem commands ----------------------------------------

    def _cmd_recall(self, arg: str) -> None:
        """Semantic (meaning-based) recall over the episodic corpus."""
        query = arg.strip()
        if not query:
            s = self.agent.semantic.stats()
            self.print_info(f"semantic memory: {s['items']} items indexed "
                            f"{s['kinds']} — usage: /recall <question>",
                            C["cyan"])
            return
        hits = self.agent.semantic.recall(query, k=5)
        if not hits:
            self.print_info(f"no memories similar to: {query}", C["dim"])
            return
        lines = [f"SEMANTIC RECALL — {query}"]
        for h in hits:
            lines.append(f"  [{h['kind']} {h['similarity']:.2f}] {h['text']}")
        self.print_info("\n".join(lines), C["cyan"])

    def _cmd_mission(self, arg: str) -> None:
        """Daemon mission control: /mission start <stmt> | task1 | task2,
        /mission tick <id>, /mission list, /mission abandon <id>."""
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1].strip() if len(parts) > 1 else ""
        d = self.agent.daemon
        if sub in ("", "list", "status"):
            self.print_info(d.format_status(), C["cyan"])
            return
        if sub == "start":
            if not rest:
                self.print_error("usage: /mission start <statement> | "
                                 "task1 | task2 | …")
                return
            segs = [s.strip() for s in rest.split("|") if s.strip()]
            if not segs:
                self.print_error("usage: /mission start <statement> | "
                                 "task1 | task2 | …")
                return
            statement = segs[0]
            tasks = segs[1:] if len(segs) > 1 else [statement]
            m = d.start(statement, tasks)
            self.print_info(f"✓ mission {m.mission_id} started — "
                            f"{len(m.steps)} step(s). Advance with "
                            f"/mission tick {m.mission_id}", C["green"])
            return
        if sub == "tick":
            mid = rest.strip()
            if not mid:
                self.print_error("usage: /mission tick <mission_id>")
                return
            r = d.tick(mid)
            if r.get("error"):
                self.print_error(f"{r.get('state', '')} "
                                 f"{r['error']}".strip())
                return
            self.print_info(f"mission {mid}: step {r.get('step', '?')} → "
                            f"{r.get('state', '?')}  "
                            f"progress {r.get('progress', 0):.0%}\n"
                            f"  {str(r.get('result', ''))[:200]}", C["cyan"])
            return
        if sub == "abandon":
            mid = rest.strip()
            if d.abandon(mid, "abandoned by user"):
                self.print_info(f"✓ mission {mid} abandoned", C["yellow"])
            else:
                self.print_error(f"cannot abandon mission {mid!r}")
            return
        self.print_error("usage: /mission [start|tick|list|abandon] …")

    def _cmd_council(self, arg: str) -> None:
        """Convene an adversarial debate: /council <proposition>."""
        question = arg.strip()
        if not question:
            self.print_info(self.agent.council.format_status(), C["pink"])
            return
        self.print_info(f"⚖ convening council on: {question} …", C["dim"])
        v = self.agent.council.convene(question)
        if not v.ok:
            self.print_error(f"council failed: {v.error}")
            return
        lines = [f"COUNCIL VERDICT — winner: {v.winner.upper()}  "
                 f"(confidence {v.confidence:.0f}%)",
                 f"  reason: {v.reason}"]
        for role, text in v.positions.items():
            lines.append(f"  [{role}] {text[:200]}")
        self.print_info("\n".join(lines), C["pink"])

    # -- v4 professional subsystem commands ------------------------------------

    def _cmd_analyze(self, arg: str) -> None:
        """Static analysis: /analyze <path> — taint, complexity, cycles."""
        path = arg.strip() or "."
        p = Path(path).expanduser()
        if p.is_file():
            result = self.agent.static.analyze_file(str(p))
        else:
            result = self.agent.static.analyze_tree(str(p))
        self.print_info(self.agent.static.format_report(result), C["cyan"])

    def _cmd_graph(self, arg: str) -> None:
        """Knowledge graph: /graph [index <path>|query <name>|impact <name>]."""
        parts = arg.split(None, 1)
        sub = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        kg = self.agent.kgraph
        if sub in ("", "status"):
            self.print_info(kg.format_status(), C["cyan"])
            return
        if sub == "index":
            root = Path(rest or ".").expanduser()
            files = ([root] if root.is_file()
                     else sorted(root.glob("**/*.py"))[:200])
            sources = {}
            for f in files:
                if f.is_file():
                    try:
                        sources[f.stem] = f.read_text(errors="replace")
                    except OSError:
                        continue
            n = kg.index_code(sources)
            kg.index_log()
            self.print_info(f"✓ indexed {len(sources)} module(s) → "
                            f"{n} entities\n{kg.format_status()}", C["green"])
            return
        if sub == "query":
            if not rest:
                self.print_error("usage: /graph query <name>")
                return
            hits = kg.find(rest)
            if not hits:
                self.print_info(f"no entity matching {rest!r} — "
                                f"/graph index first", C["dim"])
                return
            lines = []
            for e in hits[:20]:
                lines.append(f"{e.kind} {e.id}  ({e.name})")
                for r in kg.out_edges(e.id)[:8]:
                    lines.append(f"    --{r.rel}--> {r.dst}")
                for r in kg.in_edges(e.id)[:8]:
                    lines.append(f"    <--{r.rel}-- {r.src}")
            self.print_info("\n".join(lines), C["cyan"])
            return
        if sub == "impact":
            if not rest:
                self.print_error("usage: /graph impact <name>")
                return
            hits = kg.find(rest)
            if not hits:
                self.print_info(f"no entity matching {rest!r} — "
                                f"/graph index first", C["dim"])
                return
            lines = []
            for e in hits[:5]:
                dep = kg.impact(e.id)
                lines.append(f"{e.id}: {len(dep)} dependent(s)")
                lines.extend(f"    {d}" for d in dep[:20])
            self.print_info("\n".join(lines), C["yellow"])
            return
        self.print_error("usage: /graph [index <path>|query <name>|"
                         "impact <name>]")

    def _cmd_mutate(self, arg: str) -> None:
        """Mutation testing: /mutate <file> <suite-command>. Runs the suite
        against AST-generated mutants; survivors are holes in the tests."""
        parts = arg.split(None, 1)
        if len(parts) < 2:
            if self.agent.mutator is not None:
                self.print_info(self.agent.mutator.format_status(),
                                C["yellow"])
            else:
                self.print_error("usage: /mutate <file> <suite-command>  "
                                 "(e.g. /mutate src/x.py 'python -m pytest "
                                 "-q tests/')")
            return
        from .mutate import MutationTester
        path, suite = parts[0], parts[1]
        if not Path(path).expanduser().is_file():
            self.print_error(f"not a file: {path}")
            return
        self.print_info(f"⚙ mutating {path} — suite: {suite} …", C["dim"])
        tester = MutationTester(self.agent.log, suite)
        report = tester.run(path)
        self.agent.mutator = tester
        lines = [f"MUTATION — {report.path}",
                 f"  score {report.score:.0%}   killed {report.killed}   "
                 f"survived {report.survived}   errors {report.errors}   "
                 f"total {report.total}"]
        for r in report.results:
            if r.status == "survived":
                lines.append(f"    ⚠ SURVIVED [{r.kind}] {r.description}")
        self.print_info("\n".join(lines), C["yellow"])

    # -- turn execution (worker thread) ---------------------------------------------------------

    def _run_turn_thread(self, text: str) -> None:
        _q, lock = self._queue_state()
        with lock:
            self._busy = True
            self._turn_alive = True
        try:
            # Announce here, not in _dispatch: a queued turn should appear
            # in the transcript only when it actually starts, after the
            # previous turn's answer, not at the moment it was typed.
            self._emit_user(text)
            # Clear the agent's cancel flag on the turn thread, right before
            # run_turn: run_turn clears it again internally, but a cancel
            # landing in the window between thread start and that internal
            # clear would otherwise be silently wiped.
            flag = getattr(self.agent, "_cancel_flag", None)
            if flag is not None:
                try:
                    flag.clear()
                except Exception:
                    pass
            self._run_turn(text)
        finally:
            self._stop_spinner()
            self._set_status("")
            self._invalidate_agents()  # board/roster reflect the finished turn
            self._invalidate()
            _q, lock = self._queue_state()
            with lock:
                self._turn_alive = False
            self._advance_queue()

    def _advance_queue(self) -> None:
        """Release the turn slot, then start at most one queued message.

        Called from the turn thread's ``finally`` — after ``_run_turn`` has
        returned, including all focus-mode continuations.  The lock makes
        the busy/queue hand-off atomic, so a message submitted at the same
        instant either joins the queue or starts next, never both.
        """
        q, lock = self._queue_state()
        next_text = None
        with lock:
            self._busy = False
            if q:
                # Anything still in the queue arrived after _request_cancel
                # cleared the old follow-ups.  It is a fresh instruction,
                # so start it with a clean cancel state.
                next_text = q.popleft()
                self._busy = True
                self._cancel_flag.clear()

        if next_text is None:
            return
        try:
            threading.Thread(target=self._run_turn_thread,
                             args=(next_text,), daemon=True).start()
        except Exception as e:  # noqa: BLE001 — keep the message, free the UI
            with lock:
                q.appendleft(next_text)
                self._busy = False
            report = getattr(self, "print_error", None)
            if callable(report):
                report(f"could not start queued message: {e}")

    def _run_turn(self, text: str) -> None:
        self._turn_start_ts = time.time()
        self._stream_tokens_count = 0
        self._invalidate_agents()
        self._set_status("thinking…")
        self._start_spinner()
        # Only retain the unfinished line; emit complete lines immediately.
        # Throttle terminal redraws in Application, never token ingestion.
        # SPEED: pieces used to be concatenated with `tail = tail + piece`
        # on EVERY token — O(n^2) copying for a long single line (minified
        # JSON, base64 blobs) that visibly slowed the stream loop as the
        # line grew. Parts accumulate in a list; we only join when a
        # newline arrives (to split off complete lines) and keep a bounded
        # live preview, so per-token cost stays O(len(piece)).
        stream_tail = {"parts": [], "blank": False, "pv": ""}
        render_md = bool(self.cfg.extra.get("render_markdown", False))
        md_preview = ""

        # SPEED: the status preview used to invalidate the whole
        # prompt_toolkit app on EVERY token — a full-screen re-render per
        # token chunk. Tokens are still ingested instantly (complete lines
        # print immediately below); only the status/invalidate is throttled
        # to ~10fps. flush() at the end guarantees the final frame is
        # never skipped.
        _throttle = _StreamThrottle(0.1)
        self._stream_throttle = _throttle  # diagnostics / tests

        def _stream_status(text: str) -> None:
            _throttle.update(text)
            frame = _throttle.maybe_flush()
            if frame is not None:
                self._set_status(frame)

        def on_token(piece: str):
            nonlocal md_preview
            self._stream_tokens_count = getattr(self, "_stream_tokens_count", 0) + 1
            if render_md:
                # Keep a bounded preview; the agent owns the complete reply.
                maxw = max(1, self._width() - 26)
                md_preview = (md_preview + piece)[-maxw:]
                preview = md_preview.strip("\n")
                _stream_status(preview if preview else "writing…")
                return
            # patch_stdout can only interleave output safely when every
            # write ends in a newline, so emit complete lines here and keep
            # the partial line as a live preview inside the box border.
            parts = stream_tail["parts"]
            parts.append(piece)
            maxw = max(1, self._width() - 26)
            if "\n" in piece:
                # the unfinished tail never contains "\n" (it is the text
                # after the last newline seen), so a newline can only have
                # arrived inside this piece — join once, split, done.
                full = "".join(parts)
                parts.clear()
                before, _, rem = full.rpartition("\n")
                # collapse blank-line runs (the model loves "\n\n\n" between
                # tool calls) — keep at most one blank in a row
                out: list[str] = []
                for ln in before.split("\n"):
                    if not ln.strip():
                        if stream_tail["blank"]:
                            continue
                        stream_tail["blank"] = True
                    else:
                        stream_tail["blank"] = False
                    out.append(ln)
                if out:
                    self.console.print(Text("\n".join(out)), soft_wrap=True)
                if rem:
                    parts.append(rem)
                pv = rem[-maxw:] if rem else ""
            else:
                pv = stream_tail["pv"] + piece
                if len(pv) > maxw:
                    pv = pv[-maxw:]
            stream_tail["pv"] = pv
            preview = pv.strip("\n")
            _stream_status(preview if preview else "writing…")

        # Live-reasoning stream state — one accumulator per turn; the TUI
        # feeds client.py's on_reasoning chunks and shows a bounded tail
        # preview in the status line (see ReasoningStream in thinking.py).
        from .thinking import ReasoningStream
        reasoning_stream = ReasoningStream()
        _reasoning_throttle = _StreamThrottle(0.1)
        self._reasoning_throttle = _reasoning_throttle  # diagnostics / tests
        _reasoning_visible = self.cfg.show_reasoning
        try:
            _reasoning_visible = bool(getattr(
                self.agent, "thinking_visible", _reasoning_visible))
        except Exception:
            pass

        def _reasoning_status() -> None:
            _reasoning_throttle.update(
                reasoning_stream.preview() if _reasoning_visible
                else "reasoning…")
            frame = _reasoning_throttle.maybe_flush()
            if frame is not None and frame:
                self._set_status(frame)

        def on_reasoning(piece: str):
            # LIVE REASONING: stream the model's thoughts into the status
            # line as they arrive (~10fps via a dedicated throttle), so a
            # long thinking phase shows WHAT the model is doing instead of
            # a dead "reasoning…" spinner. client.py already normalizes
            # the provider fields (reasoning_content / reasoning) into this
            # callback. When thinking display is off, keep the old static
            # status so the user sees no reasoning text at all.
            reasoning_stream.feed(piece)
            _reasoning_status()

        # live shell streaming state — one tool runs at a time, so a plain
        # dict is enough: on_tool_call arms it, on_tool_output streams the
        # lines, on_tool_update closes the block with the exit-code footer
        shell_live = {"active": False, "streamed": False, "lines": 0}

        def on_tool_output(line: str, stream: str):
            if not shell_live["active"]:
                return
            n = shell_live["lines"]
            if n == 500:
                # flood guard — the full output stays in the tool result
                self.console.print(Text(
                    " │ … live view truncated (full output kept in the "
                    "tool result)", style=C["dim"]), soft_wrap=True)
                shell_live["lines"] = n + 1
                return
            if n > 500:
                return
            shell_live["lines"] = n + 1
            shell_live["streamed"] = True
            style = C["orange"] if stream == "err" else C["fg"]
            self.console.print(Text(" │ " + line, style=style),
                               soft_wrap=True)

        # Real-Time Streaming Live Diff Preview Engine (powered by LiveDiffStreamer)
        from .diffpreview import LiveDiffStreamer, DiffStreamLine
        live_f = {
            "kind": None,
            "streamer": None,
            "header": False,
            "count": 0,
            "added": 0,
            "removed": 0,
            "path": "",
            "batch": [],
        }

        _LF_BATCH = 16

        def _lf_flush():
            batch = live_f["batch"]
            if not batch:
                return
            live_f["batch"] = []
            t = Text()
            for i, ln in enumerate(batch):
                if i:
                    t.append("\n")
                t.append_text(ln)
            self.console.print(t, soft_wrap=True)

        def _lf_render_diff_line(dl: DiffStreamLine):
            live_f["count"] += 1
            t = Text()
            t.append(" │ ", style=C["dim"])
            if dl.kind == "hunk":
                t.append(f"     {dl.text}", style=f"bold {C['cyan']}")
            elif dl.kind == "meta":
                t.append(f"     {dl.text}", style=f"bold {C['dim']}")
            elif dl.kind == "del":
                live_f["removed"] += 1
                lno = f"{dl.line_no:>4} " if dl.line_no else "     "
                t.append(lno, style=C["dim"])
                t.append("-  ", style=f"bold {C['red']}")
                t.append(dl.text, style=C["red"])
            elif dl.kind == "add":
                live_f["added"] += 1
                lno = f"{dl.line_no:>4} " if dl.line_no else f"{live_f['added']:>4} "
                t.append(lno, style=C["dim"])
                t.append("+  ", style=f"bold {C['green']}")
                t.append(dl.text, style=f"{C['green']} on #0f2a1d")
            else:
                t.append(f"     {dl.text}", style=C["dim"])
            live_f["batch"].append(t)
            if len(live_f["batch"]) >= _LF_BATCH:
                _lf_flush()

        def on_tool_args(name: str, chunk: str):
            if name not in ("write_file", "edit_file", "MultiEdit", "multiedit", "apply_patch"):
                return
            if not self.cfg.extra.get("live_stream_edits", True):
                return
            if live_f["streamer"] is None or live_f["kind"] != name:
                live_f["streamer"] = LiveDiffStreamer(name)
                live_f["kind"] = name
            streamer: LiveDiffStreamer = live_f["streamer"]
            try:
                diff_lines = streamer.feed(chunk)
            except Exception:
                return
            if not live_f["header"]:
                p = "patch" if name == "apply_patch" else streamer.detect_path()
                if p:
                    if name != "apply_patch":
                        live_f["path"] = p
                    verb = {
                        "write_file": "Wrote",
                        "edit_file": "Edited",
                        "MultiEdit": "Multi-Edited",
                        "multiedit": "Multi-Edited",
                        "apply_patch": "Applied patch",
                    }[name]
                    label = verb if name == "apply_patch" else f"{verb} {self._rel_path(p)}"
                    t = Text()
                    t.append(" ⏺ ", style=f"bold {C['accent']}")
                    t.append(label, style=f"bold {C['fg']}")
                    t.append("  Δ LIVE DIFF", style=f"bold {C['cyan']}")
                    self.console.print(t, soft_wrap=True)
                    live_f["header"] = True
            if live_f["header"]:
                for dl in diff_lines:
                    _lf_render_diff_line(dl)
                if diff_lines:
                    rel = self._rel_path(live_f["path"]) if live_f["path"] else "patch"
                    self._set_status(
                        f"Δ live diff {rel} · +{live_f['added']} -{live_f['removed']} lines…"
                    )

        def on_tool_call(ev: ToolEvent):
            self._invalidate_agents()  # TaskGroup spawns must show immediately
            rem = "".join(stream_tail["parts"]).strip("\n")
            stream_tail["parts"].clear()
            stream_tail["pv"] = ""
            stream_tail["blank"] = False
            if rem:
                self.console.print(Text(rem), soft_wrap=True)
            shell_live["active"] = ev.name in ("run_command", "live_shell")
            shell_live["streamed"] = False
            shell_live["lines"] = 0
            if ev.name not in ("write_file", "edit_file", "MultiEdit", "multiedit", "apply_patch"):
                live_f["kind"] = None
                live_f["streamer"] = None
                live_f["header"] = False
                live_f["count"] = 0
                live_f["added"] = 0
                live_f["removed"] = 0
                live_f["batch"] = []
            if ev.name in ("run_command", "live_shell", "apply_patch",
                           "write_file", "edit_file", "read_file"):
                if ev.name in ("write_file", "edit_file", "apply_patch") \
                        and live_f["header"]:
                    pass
                else:
                    self.console.print(self._live_block(ev))
            else:
                self.console.print(self._tool_call_line(ev))
            self._set_status(f"running {ev.name}…")

        def on_tool_update(ev: ToolEvent):
            self._invalidate_agents()  # worker progress/finish is real state
            if ev.name in ("write_file", "edit_file", "MultiEdit", "multiedit", "apply_patch") \
                    and ev.status in ("done", "error"):
                if hasattr(self, "_diff_history"):
                    p = str(ev.args.get("path") or ev.args.get("file_path") or "patch")
                    self._diff_history.append({
                        "tool": ev.name,
                        "path": p,
                        "args": dict(ev.args),
                        "status": ev.status,
                    })
            if ev.name in ("write_file", "edit_file", "MultiEdit", "multiedit", "apply_patch") \
                    and ev.status in ("done", "error") \
                    and live_f["count"] > 0:
                streamer = live_f["streamer"]
                if streamer is not None:
                    for tail_dl in streamer.flush():
                        _lf_render_diff_line(tail_dl)
                _lf_flush()
                self.console.print(self._write_footer(ev, live_f["count"]))
            elif ev.name in ("write_file", "edit_file", "apply_patch") \
                    and ev.status in ("done", "error"):
                self.console.print(self._live_block(ev))
            elif ev.name in ("run_command", "live_shell") \
                    and ev.status in ("done", "error"):
                if shell_live["streamed"]:
                    self.console.print(self._shell_footer(ev))
                else:
                    self.console.print(self._live_block(ev))
            elif ev.name == "read_file" and ev.status in ("done", "error"):
                self.console.print(self._live_block(ev))
            else:
                self.console.print(self._tool_result_line(ev))
            shell_live["active"] = False
            live_f["kind"] = None
            live_f["streamer"] = None
            live_f["header"] = False
            live_f["count"] = 0
            live_f["added"] = 0
            live_f["removed"] = 0
            live_f["batch"] = []
            self._set_status("thinking…")

        def on_status(s: str):
            if s == "thinking":
                self._set_status("thinking…")
            elif s.startswith("tool:"):
                self._set_status(f"calling {s[5:]}…")
            elif s.startswith("running:"):
                self._set_status(f"running {s[8:]}…")
            elif s.startswith("progress:"):
                # worker 9/20 — live per-call progress readout, e.g.
                # "Call 12/25 · 45s elapsed · last: read_file aegisscan.py".
                # _set_status dedupes, so at most one extra invalidation
                # per tool call (nothing near the 10fps stream throttle),
                # and it never touches the event log.
                from .callprogress import format_status as _fmt_progress
                text = _fmt_progress(s)
                if text is not None:
                    self._set_status(text)
            else:
                self._set_status(s)

        def on_route(route):
            # the AutoPilot's decision, shown live — nothing hidden (A7)
            self.print_info("AUTOPILOT  " + route.summary(), C["pink"])
            for r in route.reasons:
                self.print_info(f"  ↳ {r}", C["dim"])

        # smartctx: expand @file mentions / bare file paths into an
        # attachments block before the turn starts
        try:
            _expand = getattr(self.agent, "expand_input", None)
            if callable(_expand):
                text, _attachments = _expand(text, cwd=os.getcwd())
                if _attachments:
                    text = text + _attachments
        except Exception:
            pass
        # images: @image <path> tokens are left for the ImageRead tool —
        # the model sees the path and reads it via the tool

        turn = None
        try:
            turn = self.agent.run_turn(
                text, on_token, on_reasoning, on_tool_call, on_tool_update,
                on_status, self._approve_blocking,
                should_cancel=self._cancel_flag.is_set,
                on_route=on_route, on_tool_output=on_tool_output,
                on_tool_args=on_tool_args)
        except Exception as e:  # noqa: BLE001 — never kill the UI thread
            self.print_error(f"{type(e).__name__}: {e}")
        finally:
            rem = "".join(stream_tail["parts"]).strip("\n")
            stream_tail["parts"].clear()
            # Final flush: apply any preview frame the throttle absorbed so
            # the last render reflects the complete stream before teardown.
            frame = _throttle.flush()
            if frame is not None:
                self._set_status(frame)
            if rem:
                self.console.print(Text(rem), soft_wrap=True)
            else:
                self.console.print()
            self._stop_spinner()
            self._set_status("")
            self._invalidate()
            # v4 aurora polish: one-shot shimmer flash when a turn lands
            # cleanly (TTY only — piped/headless runs never see \r churn).
            if turn is not None and not turn.error \
                    and getattr(self.console, "is_terminal", False) \
                    and sys.stdout.isatty():
                try:
                    from .fx import shimmer_flash
                    shimmer_flash(self.console, "✦ done",
                                  steps=6, delay=0.03)
                except Exception:
                    pass

        if turn is not None:
            if render_md and turn.assistant_text.strip() and not turn.error:
                # the finished reply, rendered once as rich Markdown
                self.console.print()
                self.console.print(Markdown(turn.assistant_text),
                                   soft_wrap=True)
            # thinking module: collapsible reasoning display; falls back
            # to the legacy show_reasoning flag
            _think_visible = self.cfg.show_reasoning
            try:
                _think_visible = bool(getattr(
                    self.agent, "thinking_visible", _think_visible))
            except Exception:
                pass
            if turn.reasoning and _think_visible:
                try:
                    _exp = bool(getattr(self.agent, "thinking_expanded",
                                        False))
                    self._print_thinking_block(turn.reasoning, expanded=_exp)
                except Exception:
                    self.print_reasoning(turn.reasoning)
            if turn.error:
                if turn.error == "cancelled":
                    self.print_info("⊘ cancelled", C["yellow"])
                else:
                    self.print_error(turn.error)
            self._print_turn_stats(turn)
            # Feature: smart follow-up suggestions after each turn
            try:
                from .help import build_turn_summary, suggest_followups
                for hint in suggest_followups(build_turn_summary(turn)):
                    self.print_info(f"  › {hint}", C["dim"])
            except Exception:
                pass
            # Feature: suggest /compact when context is nearly full
            try:
                from .compact import should_suggest_compact
                if should_suggest_compact(self.agent):
                    self.print_info(
                        "  › context >80% full — run /compact to summarize",
                        C["yellow"])
            except Exception:
                pass
            self.agent.save_session()

            # FOCUS MODE — the kernel decides whether work remains; the
            # UI drives the next continuation turn automatically.
            if (self._focus_remaining > 0
                    and not self._cancel_flag.is_set()):
                cont = self.agent.focus_continue(turn,
                                                 self._focus_remaining)
                if cont is None:
                    stops = [e for e in self.agent.log.events()
                             if e.type == "focus.stop"]
                    reason = (stops[-1].data.get("reason", "complete")
                              if stops else "complete")
                    self._focus_remaining = 0
                    self._invalidate()
                    self.print_info(f"🎯 focus ended — {reason}",
                                    C["yellow"])
                else:
                    self._focus_remaining -= 1
                    self._invalidate()
                    self.print_info(
                        f"🎯 focus · auto-continuing "
                        f"({self._focus_remaining} turn(s) left)…",
                        C["pink"])
                    self._emit_user("CONTINUE (focus mode)")
                    self._run_turn(cont)
                    return

    def _print_turn_stats(self, turn) -> None:
        parts = [f"{turn.duration:.1f}s"]
        if turn.usage:
            # HARDEN: usage values may be non-int (string/None from a
            # sloppy provider) — raw .get() would TypeError on display.
            from .client import safe_token_counts
            tin, tout = safe_token_counts(turn.usage)
            if tin or tout:
                parts.append(f"{tin}→{tout} tokens")
        t = Text("  ·  ".join(parts), style=C["dim"])
        self.console.print(t)

    def _start_spinner(self) -> None:
        self._spinner_gen = getattr(self, "_spinner_gen", 0) + 1
        gen = self._spinner_gen
        self._spinner_on = True
        def tick():
            while self._spinner_on and gen == self._spinner_gen:
                self._spinner_i = (self._spinner_i + 1) % len(SPINNER_FRAMES)
                self._invalidate()
                time.sleep(0.09)
        threading.Thread(target=tick, daemon=True).start()
    def _stop_spinner(self) -> None:
        self._spinner_on = False

    # -- approval (in-app) ------------------------------------------------------------------------

    def _approve_blocking(self, tool: Tool, args: dict) -> bool:
        if self.cfg.auto_approve:
            return True
        done = threading.Event()
        self._approve_result = False
        self._approve_request = (tool, args, done)
        self._invalidate()
        # show what is being approved above the box — with a real diff
        self.console.print(self._approval_line(tool, args))
        preview = self._diff_preview(tool, args)
        if preview is not None:
            self.console.print(preview)
        done.wait()
        self._approve_request = None
        self._invalidate()
        return self._approve_result

    def _answer_approve(self, answer: str) -> None:
        if self._approve_request is None:
            return
        if answer == "a":
            self.cfg.auto_approve = True
            self.cfg.save()
            self.print_info("  auto-approve enabled", C["yellow"])
            answer = "y"
        self._approve_result = answer == "y"
        self._approve_request[2].set()

    def _diff_preview(self, tool: Tool, args: dict):
        """Rich unified diff preview for write_file, edit_file, MultiEdit, and apply_patch."""
        if tool.name in ("MultiEdit", "multiedit", "apply_patch"):
            from .diffpreview import render_rich_diff_panel
            return render_rich_diff_panel(tool.name, args, max_lines=40, palette=C)
        if tool.name == "write_file":
            p = Path(str(args.get("path", ""))).expanduser()
            if p.exists() and p.is_file():
                from .diffpreview import render_rich_diff_panel
                return render_rich_diff_panel(tool.name, args, max_lines=35, palette=C)
            return None
        if tool.name != "edit_file":
            return None
        path = Path(str(args.get("path", ""))).expanduser()
        old_s = str(args.get("old_string", ""))
        new_s = str(args.get("new_string", ""))
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return Text(f"  ⚠ file not readable: {path}", style=C["red"])
        if old_s and old_s not in text:
            return Text("  ⚠ old_string NOT FOUND in file — this edit "
                        "will fail", style=C["red"])
        import difflib
        t = Text()
        lines = list(difflib.unified_diff(
            old_s.splitlines(), new_s.splitlines(),
            fromfile="before", tofile="after", lineterm="", n=1))
        for line in lines[:40]:
            style = (C["green"] if line.startswith("+")
                     else C["red"] if line.startswith("-")
                     else C["dim"])
            t.append(line + "\n", style=style)
        if len(lines) > 40:
            t.append(f"  … {len(lines) - 40} more diff line(s)\n",
                     style=C["dim"])
        return t if t.plain.strip() else None

    def _approval_line(self, tool: Tool, args: dict) -> Text:
        import json as _json
        try:
            arg_str = _json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            arg_str = str(args)
        if len(arg_str) > 200:
            arg_str = arg_str[:197] + "…"
        t = Text()
        t.append("  ⚠ ", style=f"bold {C['yellow']}")
        t.append(tool.name, style=f"bold {C['yellow']}")
        t.append(f"  {arg_str}", style=C["fg"])
        return t

    # -- overlays ------------------------------------------------------------------------------------

    def open_model_selector(self) -> None:
        items = []
        current = 0
        for i, m in enumerate(MODELS):
            provider = PROVIDERS[m.provider]
            tag = f' <style color="{m.tag_color}">[{m.tag}]</style>' if m.tag else ""
            html = (f'<style color="{C["cyan"]}">{m.label}</style>{tag}'
                    f' <style color="{C["dim"]}">{m.id} · {provider.name}</style>')
            items.append((html, m.id))
            if m.id == self.cfg.model_id:
                current = i

        def on_select(idx: int):
            m = MODELS[idx]
            self.cfg.model_id = m.id
            self.cfg.save()
            self._set_flash(f"model → {m.label} ({m.id})", C["green"])
            # SPEED: pre-warm the connection to the new provider so the
            # next turn hits a warm socket instantly
            from .client import prewarm_connection
            from .config import PROVIDERS
            p = PROVIDERS.get(m.provider)
            if p:
                prewarm_connection(p)

        self.overlay = OverlayList("SELECT MODEL", items, current, on_select)
        self.overlay.open()
        self._invalidate()

    def open_effort_selector(self) -> None:
        items = []
        current = 0
        for i, e in enumerate(EFFORTS):
            html = (f'<style color="{e.color}"><b>{e.label}</b></style>'
                    f' <style color="{C["dim"]}">{e.description}</style>')
            items.append((html, e.key))
            if e.key == self.cfg.effort:
                current = i

        def on_select(idx: int):
            e = EFFORTS[idx]
            self.cfg.effort = e.key
            self.cfg.save()
            self._set_flash(f"effort → {e.label}", e.color)

        self.overlay = OverlayList("SELECT EFFORT", items, current, on_select)
        self.overlay.open()
        self._invalidate()

    def open_help(self) -> None:
        def row(cmd: str, desc: str) -> tuple[str, str]:
            return (f'<style color="{C["cyan"]}">{cmd}</style>'
                    f' <style color="{C["dim"]}">{desc}</style>', "")

        items = [
            row("/model", "select model (PgUp/PgDn/Tab to navigate)"),
            row("/effort", "low · medium · high · extrahigh · ultrahigh"),
            row("/goal", "set · prove · close · status · waive · clear"),
            row("/autonomy", "0-5 observer → autonomous"),
            row("/state", "live projection of the event log"),
            row("/rewind", "rewind timeline+files · /revert files only"),
            row("/fork", "branch the timeline · /verify Merkle spine"),
            row("/why", "causal chain for an event seq"),
            row("/impact", "code blast-radius analysis (Nexus)"),
            row("/forge", "environment digest + drift"),
            row("/oracle", "post-run analysis + calibration"),
            row("/budget [steps N|usd X|reset]", "budget governor status/extend"),
            row("/constitution", "standing rules (human-owned)"),
            row("/replay", "replay the session log as a film"),
            row("/memory", "episodes + dead-end ledger"),
            row("/judge", "deterministic check (exit_code, file_exists, …)"),
            row("/new", "new session — /new [name]"),
            row("/switch", "switch session — /switch <id>"),
            row("/history", "browse previous turns"),
            row("/save", "save session to disk"),
            row("/approve", "toggle auto-approve for tools"),
            row("/reasoning", "toggle reasoning display"),
            row("/usage", "token usage"),
            row("/clear", "clear screen"),
            row("/exit", "quit"),
            row("Enter", "send    Esc+Enter: newline    Ctrl+R: search history"),
            row("Ctrl+T", "models    Ctrl+E: effort    Ctrl+L: clear"),
        ]
        self.overlay = OverlayList("HELP", items, 0, None, footer="Esc close")
        self.overlay.open()
        self._invalidate()

    def open_history(self) -> None:
        turns = self.agent.turns
        if not turns:
            self._set_flash("no history yet", C["dim"])
            return
        items = []
        for t in turns[-15:]:
            # prompt_toolkit parses HTML() with an XML parser — raw user
            # text containing '&' or '<' would crash the RENDER loop and
            # kill the whole session; escape everything user-controlled
            preview = html_escape(t.user_text.replace("\n", " ")[:70])
            stamp = html_escape(str(t.timestamp))
            html = (f'<style color="{C["dim"]}">{stamp}</style> '
                    f'<style color="{C["fg"]}">{preview}</style>')
            items.append((html, ""))
        self.overlay = OverlayList("HISTORY (recent turns)", items,
                                   len(items) - 1, None)
        self.overlay.open()
        self._invalidate()

    # -- output -----------------------------------------------------------------------------------------

    def run(self) -> None:
        """Run the persistent app for the whole session."""
        # raw=True: let rich's ANSI escape sequences pass through untouched;
        # with the default (raw=False) prompt_toolkit replaces ESC with "?"
        # and all colors show up as literal "[1;38;2..." text.
        with patch_stdout(raw=True):
            try:
                self.app.run()
            except KeyboardInterrupt:
                pass
            except EOFError:
                # stdin closed (piped/non-interactive) — exit cleanly
                self.console.print()
                self.print_info("⊘ no interactive terminal — run fullagent "
                                "in a real TTY, or use headless commands "
                                "(python main.py --help)", C["yellow"])
        self._shutdown_subagents()
        self.agent.save_session()

    def print_banner(self) -> None:
        from . import __version__
        term_w = shutil.get_terminal_size((100, 24)).columns
        width = min(term_w - 2, 84)
        model = self._model()
        effort = self._effort()

        # ── Qoder-style welcome: block logo + identity on the left, tips card
        # on the right. `cliui` owns the layout so it stays headless-testable;
        # any failure falls back to the compact logo (chrome must never take
        # the session down).
        try:
            card = cliui.welcome(
                width=width, version=__version__, model_label=model.label,
                effort_label=effort.label.lower(),
                signed_in=self._signed_in_label(),
                session=self.agent.session_id, cwd=self._cwd_label(),
                colors=C, app_name=APP_NAME)
            # v4 aurora polish: line-by-line boot reveal of the welcome
            # card; any failure above or below falls back to a plain print.
            try:
                from .fx import boot_reveal
                boot_reveal(self.console, lambda: card)
            except Exception:
                self.console.print(card)
        except Exception:
            logo = Text()
            logo.append("◆ ", style=f"bold {C['accent']}")
            logo.append(APP_NAME, style=f"bold {C['accent']}")
            logo.append(f" v{__version__}", style=f"bold {C['pink']}")
            logo.append("  ·  advanced terminal AI agent", style=C["dim"])
            self.console.print(Panel(logo, width=width,
                                     border_style=C["border"], padding=(0, 1)))

        line = Text()
        line.append(" ❯ model  ", style=C["dim"])
        line.append(model.label, style=f"bold {C['cyan']}")
        if model.tag:
            line.append(f" {model.tag} ", style=f"bold {C['green']}")
        line.append("   effort  ", style=C["dim"])
        line.append(effort.label.lower(),
                    style=f"bold {EFFORT_COLORS[effort.key]}")
        line.append("   autonomy  ", style=C["dim"])
        line.append(f"L{self.agent.autonomy}", style=f"bold {C['yellow']}")
        line.append("   session  ", style=C["dim"])
        line.append(self.agent.session_id, style=C["fg"])
        self.console.print(line)

        hints = Text()
        hints.append("   ", style=C["dim"])
        hints.append("/", style=f"bold {C['green']}")
        hints.append(" commands · ", style=C["dim"])
        hints.append("Ctrl+T", style=f"bold {C['cyan']}")
        hints.append(" models · ", style=C["dim"])
        hints.append("Ctrl+E", style=f"bold {C['cyan']}")
        hints.append(" effort · ", style=C["dim"])
        hints.append("Shift+Tab", style=f"bold {C['cyan']}")
        hints.append(" pilot · ", style=C["dim"])
        hints.append("Ctrl+O", style=f"bold {C['cyan']}")
        hints.append(" expand output", style=C["dim"])
        self.console.print(hints)
        self.console.print()
        # Feature: turnresume — offer /resume-turn when a previous session
        # left crashed turns behind (set by fullagent.turnresume.register).
        try:
            inc = getattr(self.agent, "incomplete_turns", None) or []
        except Exception:
            inc = []
        if inc:
            note = Text()
            note.append("   ⚠ ", style=f"bold {C['yellow']}")
            note.append(f"{len(inc)} interrupted turn(s) from a previous "
                        "session", style=f"bold {C['yellow']}")
            note.append(" — ", style=C["dim"])
            note.append("/resume-turn", style=f"bold {C['green']}")
            note.append(" to review and resume", style=C["dim"])
            self.console.print(note)
            self.console.print()

    def _emit_user(self, text: str) -> None:
        t = Text()
        t.append("❯ ", style=f"bold {C['green']}")
        t.append(text, style=f"bold {C['fg']}")
        self.console.print(t)

    # Devin-style titles + bodies for every tool that is not a live-action
    # block (glob / search / list / web / file-ops). One ⏺ header on call,
    # one │/└ body on result — no ⚙ gear lines, no ✓ lines.
    _DEVIN_VERB = {
        "glob_files": "Globbed",
        "search_files": "Searched",
        "list_dir": "Listed",
        "file_info": "Inspected",
        "create_directory": "Created directory",
        "copy_path": "Copied",
        "move_path": "Moved",
        "delete_path": "Deleted",
        "web_fetch": "Fetched",
        "web_search": "Searched the web",
        "live_shell_reset": "Reset the shell session",
    }
    _DEVIN_ARG = {
        "glob_files": "pattern", "search_files": "pattern",
        "list_dir": "path", "file_info": "path",
        "create_directory": "path", "delete_path": "path",
        "copy_path": "src", "move_path": "src",
        "web_fetch": "url", "web_search": "query",
    }
    _DEVIN_PATH_ARG = {"list_dir", "file_info", "create_directory",
                       "delete_path", "copy_path", "move_path"}

    def _devin_title(self, ev: ToolEvent) -> str:
        verb = self._DEVIN_VERB.get(ev.name,
                                    ev.name.replace("_", " ").strip())
        key = self._DEVIN_ARG.get(ev.name)
        arg = ""
        if key and ev.args.get(key) is not None:
            raw = str(ev.args[key])
            arg = self._rel_path(raw) if ev.name in self._DEVIN_PATH_ARG \
                else raw
            if len(arg) > 60:
                arg = arg[:57] + "…"
        if ev.name in ("copy_path", "move_path"):
            dst = str(ev.args.get("dst", ""))
            arg = f"{arg} → {dst}" if arg else dst
        return f"{verb} {arg}".strip()

    def _tool_call_line(self, ev: ToolEvent) -> Text:
        t = Text()
        t.append(" ⏺ ", style=f"bold {C['accent']}")
        t.append(self._devin_title(ev), style=f"bold {C['fg']}")
        return t

    def _generic_rows(self, ev: ToolEvent) -> list:
        """Body rows for a finished non-live tool (same segment format as
        _live_rows). Always returns at least one row (the └ footer)."""
        name = ev.name
        res = (ev.result or "").rstrip()
        if ev.status != "done" or res.startswith("ERROR"):
            msg = res.splitlines()[0] if res else ev.status
            return [[(msg[:200], C["red"])]]
        lines = res.splitlines()

        def capped(items, cap):
            cap = self._cap(cap)
            rows = [[(it, C["fg"])] for it in items[:cap]]
            if len(items) > cap:
                rows.append([(f"… +{len(items) - cap} more (Ctrl+O)", C["dim"])])
            return rows

        if name == "glob_files":
            if res.strip() == "no matches":
                return [[("no matches", C["dim"])]]
            files = [l for l in lines if l.strip()]
            rows = capped([self._rel_path(f) for f in files], 5)
            rows.append([(f"{len(files)} file(s)", C["green"])])
            return rows
        if name == "search_files":
            if res.strip() == "no matches":
                return [[("no matches", C["dim"])]]
            hits = [l for l in lines if l.strip()]
            rows = capped([h[:160] for h in hits], 5)
            rows.append([(f"{len(hits)} match(es)", C["green"])])
            return rows
        if name == "list_dir":
            entries = [l.strip() for l in lines[1:] if l.strip()]
            if not entries:
                return [[("(empty)", C["dim"])]]
            rows = capped(entries, 8)
            rows.append([(f"{len(entries)} entr"
                          f"{'y' if len(entries) == 1 else 'ies'}",
                          C["green"])])
            return rows
        if name == "web_search":
            n = len(re.findall(r"(?m)^\d+\.", res))
            body = [l for l in lines if l.strip()
                    and not l.startswith("web search:")]
            rows = capped([l[:160] for l in body], 6)
            rows.append([(f"{n} result(s)", C["green"])])
            return rows
        if name == "web_fetch":
            return [[(f"{len(res)} characters fetched", C["green"])]]
        if name in ("create_directory", "copy_path", "move_path",
                    "delete_path"):
            return [[("done", C["green"])]]
        if name == "file_info":
            rows = [[(l.strip()[:160], C["fg"])] for l in lines if l.strip()]
            return rows or [[("done", C["green"])]]
        first = lines[0] if lines else "done"
        return [[(first[:200], C["green"])]]

    def _tool_result_line(self, ev: ToolEvent) -> Text:
        self._remember_detail(ev, "generic")
        rows = self._generic_rows(ev)
        t = Text()
        n = len(rows)
        for idx, row in enumerate(rows):
            t.append(" └ " if idx == n - 1 else " │ ", style=C["dim"])
            for seg, style in row:
                t.append(seg, style=style)
            if idx < n - 1:
                t.append("\n")
        return t

    # ------------------------------------------------------------------
    # Live-action blocks, Devin-CLI style:
    #
    #  ⏺ Ran command               ⏺ Wrote ./webhack/example.md
    #  │ $ ls -la                  │  1 +  # AI ke baare mein
    #  │ total 40                  │  2 +  ## Kya hai AI?
    #  └ Exited with code 0        └ …
    #
    #  ⏺ Read ./file.md            ⏺ Edited ./file.md
    #  └ 57 lines                  │ 24    ## AI ke applications
    #                              │ 26 -  - **Healthcare**: diagnosis
    #                              │ 26 +  - **Healthcare**: diagnosis, imaging
    #                              └ 33 +  - **Cybersecurity**: threats
    # ------------------------------------------------------------------

    def _rel_path(self, path: str) -> str:
        """Devin-style path display: relative to cwd with a ./ prefix."""
        if not path:
            return ""
        try:
            rel = os.path.relpath(str(path))
        except ValueError:
            return str(path)
        if rel.startswith(".."):
            return str(path)
        return rel if rel.startswith(".") else "./" + rel

    @staticmethod
    def _parse_shell_result(result: str) -> tuple:
        """Split a run_command/live_shell receipt into (exit, stdout, stderr)."""
        exit_code = None
        m = re.search(r"^exit code: (-?\d+)", result, flags=re.M)
        if m:
            exit_code = int(m.group(1))
        stdout = stderr = ""
        sm = re.search(
            r"--- stdout ---\n(.*?)(?:\n--- stderr ---\n(.*))?$",
            result, flags=re.S)
        if sm:
            stdout = (sm.group(1) or "").rstrip("\n")
            stderr = (sm.group(2) or "").rstrip("\n")
        else:
            em = re.search(r"--- stderr ---\n(.*)$", result, flags=re.S)
            if em:
                stderr = em.group(1).rstrip("\n")
            elif exit_code is None:
                stdout = result.rstrip("\n")
        return exit_code, stdout, stderr

    @staticmethod
    def _find_hunk_line(path: str, new: str) -> int:
        """1-based line number where the edited hunk now sits, so the diff
        view can show real file line numbers instead of hunk-relative ones."""
        try:
            text = Path(path).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 1
        if not new:
            return 1
        idx = text.find(new)
        if idx < 0:
            return 1
        return text.count("\n", 0, idx) + 1

    @staticmethod
    def _diff_rows(old: str, new: str, start: int) -> list:
        """Rows of (line_no, marker, color, text) for the edit diff view.
        Old-side and new-side numbers both start at the hunk's file line,
        exactly like Devin's edit blocks."""
        old_lines = old.splitlines()
        new_lines = new.splitlines()
        sm = difflib.SequenceMatcher(None, old_lines, new_lines,
                                     autojunk=False)
        rows = []
        i = j = start
        changed = False
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                # context before any change keeps old-file numbers; context
                # after a change switches to new-file numbers (Devin style)
                base = j if changed else i
                for k in range(i1, i2):
                    rows.append((base + (k - i1), "   ", C["fg"],
                                 old_lines[k]))
                i += i2 - i1
                j += j2 - j1
            elif tag == "delete":
                changed = True
                for k in range(i1, i2):
                    rows.append((i + (k - i1), "-  ", C["red"], old_lines[k]))
                i += i2 - i1
            elif tag == "insert":
                changed = True
                for k in range(j1, j2):
                    rows.append((j + (k - j1), "+  ", C["green"],
                                 new_lines[k]))
                j += j2 - j1
            else:  # replace — removals first, then additions
                changed = True
                for k in range(i1, i2):
                    rows.append((i + (k - i1), "-  ", C["red"], old_lines[k]))
                i += i2 - i1
                for k in range(j1, j2):
                    rows.append((j + (k - j1), "+  ", C["green"],
                                 new_lines[k]))
                j += j2 - j1
        return rows

    def _live_rows(self, ev: ToolEvent, w: int) -> list:
        """Body rows for a finished live block; each row is a list of
        (text, style) segments. The caller adds the │ / └ tree glyphs."""
        rows: list = []
        failed = ev.status != "done" or ev.result.startswith("ERROR")

        def add(text: str, style: str):
            rows.append([(text[:w], style)])

        if ev.name in ("run_command", "live_shell"):
            add(f"$ {ev.args.get('command', '')}", f"bold {C['fg']}")
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
                return rows
            code, out, err = self._parse_shell_result(ev.result)
            lines = out.splitlines()
            cap = self._cap(30)
            for ln in lines[:cap]:
                add(ln if ln else " ", C["fg"])
            if len(lines) > cap:
                add(f"… +{len(lines) - cap} more lines (Ctrl+O)", C["dim"])
            for ln in err.splitlines()[:10]:
                add(ln, C["orange"])
            if code is not None:
                add(f"Exited with code {code}",
                    C["green"] if code == 0 else C["red"])
        elif ev.name == "write_file":
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
                return rows
            content = str(ev.args.get("content", ""))
            lines = content.splitlines()
            width = len(str(max(len(lines), 1)))
            cap = self._cap(300)
            for n, ln in enumerate(lines[:cap], 1):
                rows.append([(f"{n:>{width}} ", C["dim"]),
                             ("+  ", C["green"]),
                             (ln[:w], C["fg"])])
            if len(lines) > cap:
                add(f"… +{len(lines) - cap} more lines (Ctrl+O)", C["dim"])
        elif ev.name == "edit_file":
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
                return rows
            old = str(ev.args.get("old_string", ""))
            new = str(ev.args.get("new_string", ""))
            start = self._find_hunk_line(str(ev.args.get("path", "")), new)
            drows = self._diff_rows(old, new, start)
            width = len(str(max((r[0] for r in drows), default=1)))
            cap = self._cap(100)
            for num, mark, color, text in drows[:cap]:
                rows.append([(f"{num:>{width}} ", C["dim"]),
                             (mark, C["dim"] if mark == "   " else color),
                             (text[:w], color)])
            if len(drows) > cap:
                add(f"… +{len(drows) - cap} more lines (Ctrl+O)", C["dim"])
        elif ev.name == "read_file":
            m = re.search(r"(\d+) lines total, showing (\d+)\.\.(\d+)",
                          ev.result)
            if m and m.group(2) == "1" and m.group(3) == m.group(1):
                add(f"{m.group(1)} lines", C["dim"])
            elif m:
                add(f"lines {m.group(2)}..{m.group(3)} of {m.group(1)}",
                    C["dim"])
            else:
                add((ev.result.splitlines() or [""])[0], C["dim"])
        else:  # apply_patch
            lines = str(ev.args.get("patch", "")).splitlines()
            cap = self._cap(60)
            for ln in lines[:cap]:
                if ln.startswith("@@"):
                    add(ln, C["cyan"])
                elif ln.startswith(("+++ ", "--- ")):
                    add(ln, f"bold {C['dim']}")
                elif ln.startswith("+"):
                    add(ln, C["green"])
                elif ln.startswith("-"):
                    add(ln, C["red"])
                else:
                    add(ln, C["fg"])
            if len(lines) > cap:
                add(f"… +{len(lines) - cap} more lines (Ctrl+O)", C["dim"])
            if failed:
                add(ev.result.strip() or ev.status, C["red"])
        if not rows:
            add(ev.result.splitlines()[0] if ev.result else ev.status,
                C["dim"])
        return rows

    def _live_body_lines(self, ev: ToolEvent) -> list:
        """The finished body of a live block as one Text per line (tree
        glyphs included) — printable all at once or streamed line by line."""
        w = max(20, self._width() - 8)
        rows = self._live_rows(ev, w)
        n = len(rows)
        out = []
        for idx, row in enumerate(rows):
            t = Text()
            t.append(" └ " if idx == n - 1 else " │ ", style=C["dim"])
            for seg, style in row:
                t.append(seg, style=style)
            out.append(t)
        return out


    def _live_block(self, ev: ToolEvent) -> Text:
        """Devin-CLI style live-action block (see banner above). The header
        is printed when the tool starts; the │/└ body when it finishes."""
        rel = self._rel_path(str(ev.args.get("path", "")))
        t = Text()
        if ev.status == "running" or not ev.result:
            t.append(" ⏺ ", style=f"bold {C['accent']}")
            if ev.name in ("run_command", "live_shell"):
                t.append("Ran command", style=f"bold {C['fg']}")
            elif ev.name == "write_file":
                t.append(f"Wrote {rel}", style=f"bold {C['fg']}")
            elif ev.name == "edit_file":
                t.append(f"Edited {rel}", style=f"bold {C['fg']}")
            elif ev.name == "read_file":
                t.append(f"Read {rel}", style=f"bold {C['fg']}")
            else:
                t.append("Applied patch", style=f"bold {C['fg']}")
            return t
        self._remember_detail(ev)
        body = self._live_body_lines(ev)
        for idx, line in enumerate(body):
            t.append_text(line)
            if idx < len(body) - 1:
                t.append("\n")
        return t

    def _shell_footer(self, ev: ToolEvent) -> Text:
        """Closing line of a live-streamed shell block: the exit code (or
        the error), printed once the command finishes."""
        t = Text()
        t.append(" └ ", style=C["dim"])
        self._remember_detail(ev)
        if ev.status != "done" or ev.result.startswith("ERROR"):
            msg = ev.result.splitlines()[0] if ev.result else ev.status
            t.append(msg[:200], style=C["red"])
            return t
        code, _, _ = self._parse_shell_result(ev.result)
        if code is not None:
            t.append(f"Exited with code {code}",
                     style=C["green"] if code == 0 else C["red"])
        else:
            t.append("done", style=C["green"])
        return t

    def _write_footer(self, ev: ToolEvent, nlines: int) -> Text:
        """Closing line of a live-streamed write block: the tool's receipt
        (e.g. 'OK: created … 286 line(s)'), printed once writing finishes."""
        t = Text()
        t.append(" └ ", style=C["dim"])
        self._remember_detail(ev)
        if ev.status != "done" or ev.result.startswith("ERROR"):
            msg = ev.result.splitlines()[0] if ev.result else ev.status
            t.append(msg[:200], style=C["red"])
            return t
        first = ev.result.splitlines()[0] if ev.result else ""
        t.append(first[:200] if first else f"{nlines} line(s) written",
                 style=C["green"])
        return t

    def print_reasoning(self, text: str) -> None:
        """REAL CHAIN OF THOUGHT: the model's actual thinking, streamed in
        by client.py's on_reasoning, rendered at turn end as a bordered
        🧠 panel (gradient header, italic dim body). Long thoughts are
        tail-truncated to keep scrollback sane; the raw text stays in the
        turn object for /export and the replay theater."""
        body = text.strip()
        if not body:
            return
        lines = body.splitlines()
        MAX_LINES = 14
        if len(lines) > MAX_LINES:
            lines = (["  … (earlier thoughts trimmed — full chain kept in "
                      "the turn log)"]
                     + lines[-MAX_LINES:])
        body_text = Text("\n".join("  " + l for l in lines),
                         style=f"italic {C['dim']}")
        self.console.print(Panel(body_text, border_style=C["pink"],
                                 expand=False,
                                 width=min(self._width() - 2, 84),
                                 padding=(0, 1)))

    def print_error(self, text: str) -> None:
        self.console.print(Panel(Text(text, style=C["red"]),
                                 border_style=C["red"], title="error",
                                 expand=False))

    def print_info(self, text: str, color: str | None = None) -> None:
        self.console.print(Text(text, style=color or C["cyan"]))

    def print_usage(self, turns) -> None:
        from .client import safe_token_counts
        total_in = total_out = 0
        for t in turns:
            if t.usage:
                # HARDEN: see _print_turn_stats — coerce via the canonical
                # normalizer so malformed usage dicts can't TypeError here.
                tin, tout = safe_token_counts(t.usage)
                total_in += tin
                total_out += tout
        t = Text()
        t.append(f" turns: {len(turns)}", style=C["fg"])
        t.append(f"   prompt tokens: {total_in}", style=C["dim"])
        t.append(f"   completion tokens: {total_out}", style=C["dim"])
        self.console.print(t)


# ---------------------------------------------------------------------------
# Module self-test: stream render throttling
# ---------------------------------------------------------------------------

def _tui_selftest() -> None:
    """Prove the 10fps stream throttle: 1000 rapid tokens must collapse to
    a handful of renders while every token's text still reaches the output
    (complete lines print immediately, the final tail is flushed at turn
    end). Headless-safe: the UI is stubbed, no TTY or app loop needed."""
    import io
    import threading
    from types import SimpleNamespace
    from unittest.mock import patch

    from rich.console import Console

    from .agent import Turn

    # -- 1. throttle unit behavior --------------------------------------
    import fullagent.tui as tui_mod

    # the throttle takes an injectable clock now (defaults to monotonic);
    # freeze it — no wall-clock patching needed
    _frozen = [1000.0]
    th = _StreamThrottle(0.1, clock=lambda: _frozen[0])
    for i in range(1000):
        th.update(f"tok{i}")
        frame = th.maybe_flush()
        if i == 0:
            assert frame == "tok0", f"first update must flush, got {frame!r}"
        else:
            assert frame is None, "burst updates must be absorbed"
    assert th.frames == 1, th.frames
    assert th.suppressed == 999, th.suppressed
    # no text dropped: the buffered frame is always the LATEST text
    final = th.flush()
    assert final == "tok999", f"flush must return latest text, got {final!r}"
    assert th.frames == 2, th.frames
    assert th.flush() is None, "empty flush must return None"
    print("PASS  throttle: 1000 rapid updates -> 1 frame + final flush "
          "(999 absorbed), latest text never dropped")

    # -- 2. full _run_turn stream path -----------------------------------
    ui = UI.__new__(UI)
    ui.cfg = SimpleNamespace(extra={"render_markdown": False},
                             show_reasoning=False)
    ui._busy = False
    ui._cancel_flag = threading.Event()
    ui._focus_remaining = 0
    ui._status_text = ""
    ui._width = lambda: 80
    ui._start_spinner = lambda: None
    ui._stop_spinner = lambda: None
    ui._emit_user = lambda text: None
    ui._set_flash = lambda *args: None
    ui._print_turn_stats = lambda turn: None
    ui.output = io.StringIO()
    ui.console = Console(file=ui.output, width=80, color_system=None)
    ui.agent = SimpleNamespace(save_session=lambda: None)
    ui._approve_request = None
    ui._spinner_i = 0
    renders = {"n": 0}
    ui._invalidate = lambda: renders.__setitem__("n", renders["n"] + 1)

    N = 1000
    words = [f"w{i}" for i in range(N)]

    def fake_run(text, on_token, *args, **kwargs):
        for i, w in enumerate(words):
            on_token(w + ("\n" if i % 7 == 6 else " "))
        return Turn(text, assistant_text=" ".join(words))

    ui.agent.run_turn = fake_run
    with patch.object(tui_mod.time, "time", return_value=1000.0), \
         patch.object(tui_mod.time, "monotonic", return_value=1000.0):
        ui._run_turn_thread("hello")

    out = ui.output.getvalue()
    missing = [w for w in words if w not in out]
    assert not missing, f"text lost: {missing[:5]}…"
    th2 = ui._stream_throttle
    assert isinstance(th2, _StreamThrottle)
    assert th2.suppressed >= N - 2, th2.suppressed
    # turn-lifecycle renders only: thinking…, first frame, final flush,
    # teardown "" + 2 unconditional invalidates — NOT ~1000
    assert renders["n"] <= 6, f"{renders['n']} renders for {N} tokens"
    print(f"PASS  stream path: {N} tokens -> {renders['n']} renders "
          f"({th2.suppressed} absorbed, {th2.frames} frames), "
          f"all {N} tokens present in output")

    # -- 3. wall-clock timing: frames really are ~100ms apart -------------
    th3 = _StreamThrottle(0.1)
    stamps = []
    t0 = tui_mod.time.time()
    for i in range(400):
        th3.update(f"x{i}")
        if th3.maybe_flush() is not None:
            stamps.append(tui_mod.time.time() - t0)
        tui_mod.time.sleep(0.001)  # stretch the burst past the 0.1s window
    th3.flush()
    assert len(stamps) >= 2, f"expected multiple frames, got {len(stamps)}"
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(g >= 0.095 for g in gaps), f"frames too close: {gaps[:5]}"
    print(f"PASS  frame spacing: {th3.frames} frames, "
          f"min gap {min(gaps):.3f}s (>= 0.1s throttle)")

    # -- 4. follow-up queue: accepted while busy, FIFO, never overlapping -
    ui2 = UI.__new__(UI)
    ui2.cfg = SimpleNamespace(extra={"render_markdown": False},
                              show_reasoning=False)
    ui2._busy = False
    ui2._cancel_flag = threading.Event()
    ui2._focus_remaining = 0
    ui2._status_text = ""
    ui2._width = lambda: 80
    ui2._start_spinner = lambda: None
    ui2._stop_spinner = lambda: None
    ui2._emit_user = lambda text: None
    ui2._set_flash = lambda *args: None
    ui2._print_turn_stats = lambda turn: None
    ui2._invalidate = lambda: None
    ui2.output = io.StringIO()
    ui2.console = Console(file=ui2.output, width=80, color_system=None)
    calls: list[str] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def fake_run2(text, *args, **kwargs):
        calls.append(text)
        if text == "first":
            first_started.set()
            release_first.wait(5)
        return Turn(text)

    ui2.agent = SimpleNamespace(save_session=lambda: None,
                                run_turn=fake_run2)
    try:
        ui2._dispatch("first")
        assert first_started.wait(3), "first turn did not start"
        ui2._dispatch("second")
        assert ui2._pending_count() == 1, ui2._pending_count()
        assert calls == ["first"], f"queued turn overlapped: {calls}"
        release_first.set()
        deadline = tui_mod.time.monotonic() + 5
        while ui2._busy and tui_mod.time.monotonic() < deadline:
            tui_mod.time.sleep(0.01)
        assert not ui2._busy, "queue drain left the UI busy"
        assert calls == ["first", "second"], calls
        assert ui2._pending_count() == 0
    finally:
        release_first.set()
    print("PASS  follow-up queue: 1 running + 1 queued -> serial FIFO, "
          "no overlapping run_turn")

    print("ALL TUI SELF-TESTS PASS")


if __name__ == "__main__":
    _tui_selftest()
