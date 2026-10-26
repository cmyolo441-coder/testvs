"""Terminal styling: truecolor palette, markdown streaming, banner.

The palette is the one thing kept from oh-my-hermes' skin engine (turquoise
``#00CED1`` on deep teal). Everything else here is a small ANSI writer with no
dependencies, so rendering stays allocation-light on every token.
"""

from __future__ import annotations

import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, TextIO

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
ITALIC = "\x1b[3m"
UNDERLINE = "\x1b[4m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_INLINE_RE = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`|\[[^\]\n]+\]\([^)\n]+\))")


@dataclass(frozen=True)
class Palette:
    name: str
    accent: str
    title: str
    text: str
    dim: str
    ok: str
    warn: str
    err: str
    tool: str
    code: str
    rule: str
    user: str
    thinking: str


PALETTES: Dict[str, Palette] = {
    "hermes": Palette(
        name="hermes",
        accent="#00CED1",
        title="#7FDBFF",
        text="#E4FAFF",
        dim="#3E7C8A",
        ok="#6ED7B0",
        warn="#FFA726",
        err="#EF5350",
        tool="#00CED1",
        code="#9FE8FF",
        rule="#1E8A99",
        user="#9FE8FF",
        thinking="#4A7580",
    ),
    "amber": Palette(
        name="amber",
        accent="#FFB000",
        title="#FFD27F",
        text="#FFF3DC",
        dim="#8A6B3E",
        ok="#8FD694",
        warn="#FFC857",
        err="#FF7A6B",
        tool="#FFB000",
        code="#FFE0A3",
        rule="#8A5A1E",
        user="#FFE0A3",
        thinking="#8A6B3E",
    ),
    "mono": Palette(
        name="mono",
        accent="#D8DEE9",
        title="#FFFFFF",
        text="#E5E9F0",
        dim="#6C7687",
        ok="#C9D1D9",
        warn="#E5E9F0",
        err="#FFFFFF",
        tool="#D8DEE9",
        code="#F0F4FA",
        rule="#4C566A",
        user="#FFFFFF",
        thinking="#6C7687",
    ),
}


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def visible_width(text: str) -> int:
    width = 0
    for char in strip_ansi(text):
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
    return width


def truncate(text: str, width: int, suffix: str = "…") -> str:
    plain = strip_ansi(text)
    if visible_width(plain) <= width:
        return text
    out, used = [], 0
    for char in plain:
        step = 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        if used + step > width - 1:
            break
        out.append(char)
        used += step
    return "".join(out) + suffix


def _color_enabled(stream: TextIO) -> bool:
    override = os.environ.get("OMH_COLOR", "").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    if override in {"1", "true", "yes", "on"}:
        return True
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM", "") in {"dumb", ""}:
        return False
    try:
        return bool(stream.isatty())
    except Exception:  # pragma: no cover - exotic streams
        return False


class Theme:
    """ANSI helper; every method returns plain text when colour is off."""

    def __init__(
        self,
        name: str = "hermes",
        *,
        enabled: Optional[bool] = None,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.palette = PALETTES.get(name.lower(), PALETTES["hermes"])
        stream = stream or sys.stdout
        self.enabled = _color_enabled(stream) if enabled is None else enabled

    # -- primitives -----------------------------------------------------
    @staticmethod
    def _truecolor(hex_color: str, background: bool = False) -> str:
        value = hex_color.lstrip("#")
        if len(value) != 6:
            return ""
        r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
        return f"\x1b[{48 if background else 38};2;{r};{g};{b}m"

    def paint(
        self,
        text: str,
        *,
        fg: Optional[str] = None,
        bold: bool = False,
        dim: bool = False,
        italic: bool = False,
        underline: bool = False,
    ) -> str:
        if not self.enabled or not text:
            return text
        prefix = ""
        if fg:
            prefix += self._truecolor(fg)
        if bold:
            prefix += BOLD
        if dim:
            prefix += DIM
        if italic:
            prefix += ITALIC
        if underline:
            prefix += UNDERLINE
        return f"{prefix}{text}{RESET}" if prefix else text

    # -- semantic helpers ----------------------------------------------
    def accent(self, text: str) -> str:
        return self.paint(text, fg=self.palette.accent)

    def title(self, text: str) -> str:
        return self.paint(text, fg=self.palette.title, bold=True)

    def text(self, text: str) -> str:
        return self.paint(text, fg=self.palette.text)

    def muted(self, text: str) -> str:
        return self.paint(text, fg=self.palette.dim)

    def ok(self, text: str) -> str:
        return self.paint(text, fg=self.palette.ok)

    def warn(self, text: str) -> str:
        return self.paint(text, fg=self.palette.warn)

    def err(self, text: str) -> str:
        return self.paint(text, fg=self.palette.err)

    def tool(self, text: str) -> str:
        return self.paint(text, fg=self.palette.tool, bold=True)

    def code(self, text: str) -> str:
        return self.paint(text, fg=self.palette.code)

    def user_label(self, text: str) -> str:
        return self.paint(text, fg=self.palette.user, bold=True)

    def bold(self, text: str) -> str:
        return self.paint(text, bold=True)

    def rule(self, width: Optional[int] = None) -> str:
        width = width or terminal_width()
        return self.paint("─" * max(4, width), fg=self.palette.rule)


def terminal_width() -> int:
    try:
        return max(40, os.get_terminal_size().columns)
    except OSError:
        return 80


@dataclass(frozen=True)
class Glyphs:
    bullet: str
    tool: str
    ok: str
    fail: str
    prompt: str
    bar: str
    branch: str


UNICODE_GLYPHS = Glyphs("•", "◆", "✔", "✖", "❯", "│", "└")
ASCII_GLYPHS = Glyphs("*", ">", "+", "x", ">", "|", "`")


def glyphs(stream: Optional[TextIO] = None) -> Glyphs:
    stream = stream or sys.stdout
    encoding = (getattr(stream, "encoding", "") or "utf-8").lower()
    return ASCII_GLYPHS if encoding.startswith(("ascii", "cp1252", "latin")) else UNICODE_GLYPHS


class MarkdownStream:
    """Incrementally turns streamed markdown into ANSI text.

    Complete lines are rendered as they arrive, so the first token paints
    immediately instead of waiting for the response to finish.
    """

    def __init__(self, theme: Theme, width: Optional[int] = None) -> None:
        self.theme = theme
        self.width = width or terminal_width()
        self._buf = ""
        self._in_code = False
        self._fence = ""

    @property
    def in_code(self) -> bool:
        return self._in_code

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buf += text
        out: List[str] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            out.append(self._render_line(line))
            out.append("\n")
        return "".join(out)

    def flush(self) -> str:
        out = ""
        if self._buf:
            out += self._render_line(self._buf)
            self._buf = ""
        if self._in_code:
            self._in_code = False
            out += self._fence_line(close=True)
        return out

    # -- internals ------------------------------------------------------
    def _fence_line(self, close: bool = False) -> str:
        theme = self.theme
        if close:
            return theme.muted("  └" + "─" * 3) + "\n"
        label = self._fence or "code"
        return theme.muted("  ┌─ ") + theme.accent(label) + "\n"

    def _render_line(self, line: str) -> str:
        theme = self.theme
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        if stripped.startswith("```"):
            if self._in_code:
                self._in_code = False
                self._fence = ""
                return self._fence_line(close=True).rstrip("\n")
            self._in_code = True
            self._fence = stripped[3:].strip()
            return self._fence_line().rstrip("\n")

        if self._in_code:
            return theme.muted("  │ ") + theme.code(line)

        if not stripped:
            return ""

        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            return theme.accent("▌ ") + theme.title(heading.group(2).strip())

        if re.match(r"^([-*_])\1{2,}$", stripped):
            return theme.muted("─" * min(self.width - 4, 48))

        bullet = re.match(r"^([-*+])\s+(.*)$", stripped)
        if bullet:
            return indent + theme.accent("• ") + self._inline(bullet.group(2))

        numbered = re.match(r"^(\d+[.)])\s+(.*)$", stripped)
        if numbered:
            return indent + theme.accent(numbered.group(1) + " ") + self._inline(numbered.group(2))

        quote = re.match(r"^>\s?(.*)$", stripped)
        if quote:
            return indent + theme.muted("│ ") + theme.paint(
                strip_ansi(self._inline(quote.group(1))), italic=True
            )

        return indent + self._inline(stripped)

    def _inline(self, text: str) -> str:
        theme = self.theme
        out: List[str] = []
        for part in _INLINE_RE.split(text):
            if not part:
                continue
            if part.startswith("**") and part.endswith("**") and len(part) > 4:
                out.append(theme.bold(part[2:-2]))
            elif part.startswith("`") and part.endswith("`") and len(part) > 2:
                out.append(theme.code(part[1:-1]))
            elif part.startswith("[") and "](" in part and part.endswith(")"):
                label, url = part[1:-1].split("](", 1)
                out.append(theme.paint(label, underline=True))
                out.append(" " + theme.muted(url))
            else:
                out.append(theme.text(part))
        return "".join(out)


def render_markdown(text: str, theme: Theme, width: Optional[int] = None) -> str:
    stream = MarkdownStream(theme, width)
    return stream.feed(text) + stream.flush()


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_FRAMES_ASCII = ["|", "/", "-", "\\"]
THINKING_VERBS = ["thinking", "routing", "reading", "planning", "checking", "writing"]

LOGO = [
    " ██████╗ ███╗   ███╗██╗  ██╗",
    "██╔═══██╗████╗ ████║██║  ██║",
    "██║   ██║██╔████╔██║███████║",
    "╚██████╔╝██║ ╚═╝ ██║██╔══██║",
    " ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝",
]

LOGO_ASCII = [
    "  ###  #   # #   #",
    " #   # ## ## #   #",
    " #   # # # # #####",
    " #   # #   # #   #",
    "  ###  #   # #   #",
]


def logo_lines() -> List[str]:
    encoding = (getattr(sys.stdout, "encoding", "") or "utf-8").lower()
    return LOGO_ASCII if encoding.startswith(("ascii", "cp1252", "latin")) else LOGO


def colorize_diff(text: str, theme: Theme) -> str:
    out: List[str] = []
    for line in text.splitlines():
        if line.startswith(("+++", "---")):
            out.append(theme.muted(line))
        elif line.startswith("+"):
            out.append(theme.ok(line))
        elif line.startswith("-"):
            out.append(theme.err(line))
        elif line.startswith("@@"):
            out.append(theme.accent(line))
        else:
            out.append(theme.muted(line))
    return "\n".join(out)


def shorten_path(path: str, limit: int = 48) -> str:
    if len(path) <= limit:
        return path
    return "…" + path[-(limit - 1) :]


def table(theme: Theme, headers: List[str], rows: List[List[str]], *, max_width: int = 100) -> str:
    """Small aligned table; cells are truncated before ANSI is applied."""
    widths = [visible_width(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], visible_width(cell))
    budget = max(20, max_width - 3 * max(0, len(widths) - 1))
    while sum(widths) > budget:
        widest = widths.index(max(widths))
        if widths[widest] <= 8:
            break
        widths[widest] -= 2

    def cell(text: str, width: int) -> str:
        text = truncate(text, width)
        return text + " " * (width - visible_width(text))

    head = "   ".join(theme.bold(cell(h, widths[i])) for i, h in enumerate(headers))
    rule = theme.muted("   ".join("─" * widths[i] for i in range(len(widths))))
    body = [
        "   ".join(cell(str(value), widths[i]) for i, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([head, rule, *body])


def banner(
    theme: Theme,
    *,
    version: str,
    model_label: str,
    workspace: str,
    strategy: str,
    profile_count: int,
    provider_count: int,
) -> str:
    g = glyphs()
    logo = [
        theme.paint(line, fg=theme.palette.accent if i % 2 == 0 else theme.palette.title, bold=True)
        for i, line in enumerate(logo_lines())
    ]
    meta = [
        theme.accent(f"{g.tool} ") + theme.title("OMH") + theme.muted(f"  v{version} · terminal agent harness"),
        "",
        theme.muted("  model    ") + theme.text(model_label),
        theme.muted("  pool     ")
        + theme.text(
            f"{profile_count} model{'s' if profile_count != 1 else ''} · "
            f"{provider_count} provider{'s' if provider_count != 1 else ''} · {strategy}"
        ),
        theme.muted("  cwd      ") + theme.text(shorten_path(workspace, 64)),
        "",
        theme.muted("  /help commands   ") + theme.accent("!") + theme.muted(" shell   ") + theme.accent("esc") + theme.muted(" / ") + theme.accent("ctrl-c") + theme.muted(" cancel"),
    ]
    return "\n".join(logo + [""] + meta)


def status_line(
    theme: Theme,
    *,
    model: str,
    seconds: float,
    tokens: int,
    cost: float,
    step: int,
) -> str:
    g = glyphs()
    parts = [
        theme.muted(f"{g.tool} ") + theme.accent(model),
        theme.muted(f"{seconds:.1f}s"),
        theme.muted(f"{tokens:,} tok"),
    ]
    if cost > 0:
        parts.append(theme.muted(f"${cost:.4f}"))
    if step > 1:
        parts.append(theme.muted(f"{step} steps"))
    return theme.muted("  ") + theme.muted(" · ").join(parts) + "\n"


def tool_line(theme: Theme, name: str, summary: str, *, running: bool = False) -> str:
    g = glyphs()
    icon = theme.accent(g.tool) if running else theme.ok(g.ok)
    return theme.muted("  ") + icon + " " + theme.tool(name) + (
        theme.muted("  " + summary) if summary else ""
    )


def error_block(theme: Theme, message: str, hint: str = "") -> str:
    lines = [theme.muted("  ") + theme.err("✖ ") + theme.err(message)]
    if hint:
        lines.append(theme.muted("    " + hint))
    return "\n".join(lines)


