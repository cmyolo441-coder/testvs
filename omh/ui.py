"""Terminal UI: spinner, streamed markdown, tool cards, prompt, approvals."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from .theme import (
    MarkdownStream,
    SPINNER_FRAMES,
    SPINNER_FRAMES_ASCII,
    Theme,
    colorize_diff,
    error_block,
    glyphs,
    status_line,
    terminal_width,
    tool_line,
    truncate,
)
from .tools.base import ToolSpec


class Console:
    def __init__(
        self,
        theme: Theme,
        *,
        stream: Optional[TextIO] = None,
        error_stream: Optional[TextIO] = None,
    ) -> None:
        self.theme = theme
        self.stream = stream or sys.stdout
        self.error_stream = error_stream or sys.stderr

    @property
    def interactive(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except Exception:
            return False

    def write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def line(self, text: str = "") -> None:
        self.write(text + "\n")

    def error(self, text: str) -> None:
        self.error_stream.write(text + "\n")
        self.error_stream.flush()


class Spinner:
    """A single-line spinner on stderr; never touches the transcript."""

    def __init__(
        self,
        theme: Theme,
        *,
        stream: Optional[TextIO] = None,
        interval: float = 0.08,
        enabled: Optional[bool] = None,
    ) -> None:
        self.theme = theme
        self.stream = stream or sys.stderr
        self.interval = interval
        try:
            tty = bool(self.stream.isatty())
        except Exception:
            tty = False
        self.enabled = tty if enabled is None else (enabled and tty)
        encoding = (getattr(self.stream, "encoding", "") or "utf-8").lower()
        self._frames = (
            SPINNER_FRAMES_ASCII
            if encoding.startswith(("ascii", "cp1252", "latin"))
            else SPINNER_FRAMES
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active = False
        self.label = "thinking"
        self._started = 0.0

    def start(self, label: str = "thinking") -> None:
        if not self.enabled or self._thread is not None:
            return
        self.label = label
        self._stop.clear()
        self._started = time.monotonic()
        self._active = True
        self._thread = threading.Thread(target=self._spin, daemon=True, name="omh-spinner")
        self._thread.start()

    def set_label(self, label: str) -> None:
        self.label = label

    def _spin(self) -> None:
        index = 0
        while not self._stop.wait(self.interval):
            frame = self._frames[index % len(self._frames)]
            elapsed = time.monotonic() - self._started
            text = f"  {frame} {self.label}  {elapsed:.1f}s"
            self.stream.write("\r\x1b[2K" + self.theme.muted(text))
            self.stream.flush()
            index += 1

    def stop(self) -> None:
        if not self.enabled or not self._active:
            return
        self._active = False
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=0.3)
        self.stream.write("\r\x1b[2K")
        self.stream.flush()

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()


class TurnRenderer:
    """Consumes agent events and paints one assistant turn."""

    def __init__(
        self,
        console: Console,
        spinner: Spinner,
        *,
        show_usage: bool = True,
    ) -> None:
        self.console = console
        self.theme = console.theme
        self.spinner = spinner
        self.show_usage = show_usage
        self._markdown: Optional[MarkdownStream] = None
        self._printed = False
        self._started = 0.0
        self._usage: Dict[str, Any] = {}
        self._model = ""
        self._tool_count = 0

    def begin(self, label: str = "thinking") -> None:
        self._markdown = MarkdownStream(self.theme)
        self._printed = False
        self._started = time.monotonic()
        self._usage = {}
        self._model = ""
        self._tool_count = 0
        self.spinner.start(label)

    def set_label(self, label: str) -> None:
        self.spinner.set_label(label)

    def text(self, chunk: str, model: str = "") -> None:
        self.spinner.stop()
        if model:
            self._model = model
        if not self._printed:
            self.console.line()
            suffix = (
                self.theme.muted("  " + self._model) if self._model else ""
            )
            self.console.line(self.theme.accent("◆ ") + self.theme.title("omh") + suffix)
            self._printed = True
        if self._markdown is not None:
            rendered = self._markdown.feed(chunk)
            if rendered:
                self.console.write(rendered)

    def info(self, message: str) -> None:
        self.spinner.stop()
        self.console.line(self.theme.muted("  · " + message))

    def tool_start(self, name: str, summary: str, model: str = "") -> None:
        self.spinner.stop()
        if not self._printed:
            if model:
                self._model = model
            self.console.line()
            suffix = self.theme.muted("  " + self._model) if self._model else ""
            self.console.line(self.theme.accent("◆ ") + self.theme.title("omh") + suffix)
            self._printed = True
        self._tool_count += 1
        self.console.line(
            tool_line(
                self.theme,
                name,
                truncate(summary, terminal_width() - 30),
                running=True,
            )
        )
        self.spinner.start(f"running {name}")

    def tool_result(self, name: str, result: Any) -> None:
        self.spinner.stop()
        g = glyphs()
        icon = self.theme.ok(g.ok) if result.ok else self.theme.err(g.fail)
        status = "ok" if result.ok else "failed"
        self.console.line(
            self.theme.muted("  ") + icon + " " + self.theme.muted(f"{name}  {status}")
        )
        for line in self._preview(name, result):
            self.console.line(line)
        self.spinner.start("thinking")

    def _preview(self, name: str, result: Any) -> List[str]:
        text = (result.output or "").strip()
        if not text:
            return []
        lines = text.splitlines()
        out: List[str] = []
        if not result.ok:
            for line in lines[:8]:
                out.append(self.theme.muted("    ") + self.theme.err(truncate(line, terminal_width() - 6)))
            if len(lines) > 8:
                out.append(self.theme.muted(f"    … {len(lines) - 8} more lines"))
            return out
        if name == "edit_file":
            diff_lines = lines[1:] if lines and lines[0].startswith("edited") else lines
            shown = [line for line in diff_lines if line.startswith(("@@", "+", "-"))][:14]
            if shown:
                out.append("    " + colorize_diff("\n    ".join(shown), self.theme))
            return out
        if name in {"run_shell", "write_file"}:
            for line in lines[:6]:
                out.append(self.theme.muted("    " + truncate(line, terminal_width() - 6)))
            if len(lines) > 6:
                out.append(self.theme.muted(f"    … {len(lines) - 6} more lines"))
            return out
        out.append(self.theme.muted("    " + truncate(lines[0], terminal_width() - 6)))
        return out

    def usage(self, data: Dict[str, Any]) -> None:
        self._usage = data
        if data.get("model"):
            self._model = str(data["model"])

    def error(self, error: BaseException) -> None:
        self.spinner.stop()
        self._flush_markdown()
        message = getattr(error, "message", str(error))
        hint = getattr(error, "hint", "")
        self.console.line()
        self.console.line(error_block(self.theme, message, hint))

    def done(self, data: Dict[str, Any]) -> None:
        self.spinner.stop()
        self._flush_markdown()
        if self.show_usage and self._usage:
            elapsed = time.monotonic() - self._started
            self.console.write(
                status_line(
                    self.theme,
                    model=self._usage.get("model") or self._model or "model",
                    seconds=elapsed,
                    tokens=int(self._usage.get("tokens") or 0),
                    cost=float(self._usage.get("cost") or 0.0),
                    step=int(self._usage.get("step") or 0),
                )
            )
        reason = data.get("reason")
        if reason == "cancelled":
            self.console.line(self.theme.warn("  cancelled"))
        elif reason == "max_steps":
            self.console.line(
                self.theme.warn(
                    "  " + str(data.get("message") or "step limit reached")
                )
            )

    def _flush_markdown(self) -> None:
        if self._markdown is not None:
            tail = self._markdown.flush()
            if tail:
                self.console.write(tail)
            self._markdown = None
        if self._printed:
            self.console.line()


class Prompt:
    """Readline-backed input with history, tab completion and safe EOF."""

    def __init__(self, theme: Theme, history_file: str, *, stream: Optional[TextIO] = None) -> None:
        self.theme = theme
        self.history_file = Path(history_file).expanduser()
        self.stream = stream
        self.words: List[str] = []
        self._readline: Any = None
        try:
            import readline  # noqa: F401

            self._readline = readline
            try:
                readline.read_history_file(str(self.history_file))
            except (OSError, AttributeError):
                pass
            try:
                readline.set_history_length(2000)
                readline.parse_and_bind("tab: complete")
                readline.set_completer(self._complete)
                readline.set_completer_delims(" \t\n")
            except AttributeError:  # pragma: no cover - libedit
                pass
        except ImportError:  # pragma: no cover - windows without pyreadline
            self._readline = None

    def _complete(self, text: str, state: int) -> Optional[str]:
        options = [word for word in self.words if word.startswith(text)]
        return options[state] if state < len(options) else None

    def set_words(self, words: List[str]) -> None:
        self.words = sorted(set(words))

    def ask(self, marker: str = "❯ ") -> Optional[str]:
        prompt = self.theme.paint(marker, fg=self.theme.palette.user, bold=True)
        try:
            line = input(prompt)
        except EOFError:
            return None
        if line.strip():
            self._remember(line)
        return line

    def _remember(self, line: str) -> None:
        if self._readline is None:
            return
        try:
            if self._readline.get_current_history_length():
                last = self._readline.get_history_item(
                    self._readline.get_current_history_length()
                )
                if last == line:
                    return
            self._readline.add_history(line)
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            self._readline.write_history_file(str(self.history_file))
        except (OSError, AttributeError):
            pass


def ask_approval(
    console: Console,
    spec: ToolSpec,
    arguments: Dict[str, Any],
    summary: str,
) -> bool:
    theme = console.theme
    console.line()
    console.line(
        theme.paint("  approval ", fg=theme.palette.warn, bold=True)
        + theme.tool(spec.name)
        + (theme.muted("  " + summary) if summary else "")
    )
    if spec.name == "write_file":
        for line in str(arguments.get("content", "")).splitlines()[:8]:
            console.line(theme.muted("    │ " + truncate(line, terminal_width() - 10)))
        extra = len(str(arguments.get("content", "")).splitlines()) - 8
        if extra > 0:
            console.line(theme.muted(f"    │ … {extra} more lines"))
    elif spec.name == "edit_file":
        for line in str(arguments.get("old_string", "")).splitlines()[:4]:
            console.line(theme.err("    - " + truncate(line, terminal_width() - 10)))
        for line in str(arguments.get("new_string", "")).splitlines()[:4]:
            console.line(theme.ok("    + " + truncate(line, terminal_width() - 10)))
    try:
        answer = input(theme.muted("  run? [y/N] ")).strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


