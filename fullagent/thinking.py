"""Thinking display — show the model's reasoning in the TUI.

Reads reasoning text from every message shape ``client.py`` produces,
formats it collapsed (preview) or expanded (full), and exposes the
``/thinking`` TUI command. No imports of ``.agent``/``.tui``/``.client``
at module level, so there are no import cycles.

Shapes handled by :func:`extract_thinking` (mirrors ``client.py``):
    - ``message["reasoning"]`` / ``message["reasoning_content"]`` — plain
      strings, set by ``client.assistant_message()`` and read by
      ``client._result_from_json()`` / ``agent.py`` history handling.
    - ``message["reasoning_details"]`` — list of parts, e.g. OpenRouter
      style ``[{"type": "reasoning.text", "text": "..."}]``.
    - content blocks — ``message["content"]`` as a list of blocks with
      ``type == "reasoning"`` (``"reasoning"``/``"text"``/``"content"``
      keys) or Anthropic style ``type == "thinking"`` (``"thinking"`` key).

Public API:
    - :func:`extract_thinking` -- pull reasoning text out of a message.
    - :func:`format_thinking` -- collapsed/expanded render-ready string.
    - :class:`ReasoningStream` -- incremental reasoning accumulator for
      live TUI streaming (feed per-chunk, bounded tail ``preview()``).
    - :func:`handle_thinking` -- the ``/thinking`` TUI command.
    - :func:`register` -- wire ``agent.thinking_visible`` + helpers.

``python3 -m fullagent.thinking`` runs the built-in self-test.
"""

from __future__ import annotations

import json
import os
from typing import Any

# ~/.fullagent/config.json — read/written directly, merging other keys, so
# this module never depends on config.py (and never touches _EMBEDDED_KEYS).
_APP_DIR = os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent")
_CONFIG_KEY = "thinking_visible"
_EXPANDED_KEY = "thinking_expanded"


def _config_path() -> str:
    return os.path.join(_APP_DIR, "config.json")


def _read_config() -> dict:
    try:
        with open(_config_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_config(data: dict) -> None:
    """Persist, merging with whatever else lives in config.json.

    Best-effort: a failed write must never break a TUI command."""
    try:
        os.makedirs(_APP_DIR, exist_ok=True)
        tmp = _config_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, _config_path())
    except OSError:
        pass


def _persist_visible(visible: bool) -> None:
    data = _read_config()
    data[_CONFIG_KEY] = bool(visible)
    _write_config(data)


def _persist_expanded(expanded: bool) -> None:
    data = _read_config()
    data[_EXPANDED_KEY] = bool(expanded)
    _write_config(data)


def _load_visible() -> bool:
    return bool(_read_config().get(_CONFIG_KEY, True))


def _load_expanded() -> bool:
    return bool(_read_config().get(_EXPANDED_KEY, False))


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def _coerce_text(value: Any) -> str:
    """Best-effort text extraction from a reasoning part; never raises."""
    try:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, dict):
            for key in ("text", "reasoning", "thinking", "content"):
                got = _coerce_text(value.get(key))
                if got:
                    return got
            return ""
        if isinstance(value, (list, tuple)):
            return "".join(_coerce_text(v) for v in value)
        return ""
    except Exception:
        return ""


def extract_thinking(message: dict) -> str:
    """Pull reasoning text from a message dict. Never raises.

    Checks, in order: ``reasoning``/``reasoning_content`` strings,
    ``reasoning_details`` part lists, and ``content`` blocks of type
    ``reasoning``/``thinking``. Returns ``""`` when nothing is found.
    """
    try:
        if not isinstance(message, dict):
            return ""
        # 1. plain-string keys (client.assistant_message / _result_from_json)
        for key in ("reasoning_content", "reasoning"):
            text = message.get(key)
            if isinstance(text, str) and text.strip():
                return text
        # 2. reasoning_details lists (OpenRouter-style part dicts)
        details = message.get("reasoning_details")
        if isinstance(details, (list, tuple)):
            text = _coerce_text(details).strip()
            if text:
                return text
        # 3. content blocks (OpenAI / Anthropic style)
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = str(block.get("type") or "")
                if btype in ("reasoning", "thinking", "reasoning_content"):
                    got = _coerce_text(
                        block.get("reasoning", block.get("thinking",
                                                         block.get("text"))))
                    # fall back to the generic dict walk for odd shapes
                    if not got:
                        got = _coerce_text(block)
                    if got:
                        parts.append(got)
            if parts:
                return "".join(parts).strip()
        return ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _dim_indent(text: str) -> str:
    """Two-space indent + dim ANSI, for the TUI's raw console output."""
    lines = text.splitlines() or [""]
    return "\n".join(f"{_DIM}  {line}{_RESET}" for line in lines)


def format_thinking(text: str, collapsed: bool = True) -> str:
    """Render-ready reasoning string.

    Collapsed: first 2 lines + "… (N more lines, /thinking to expand)".
    Expanded: the full text. Both are dim + indented. "" in, "" out.
    """
    text = (text or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    if collapsed and len(lines) > 2:
        head = "\n".join(lines[:2])
        more = len(lines) - 2
        return _dim_indent(
            head + f"\n… ({more} more lines, /thinking to expand)")
    return _dim_indent("\n".join(lines))


# ---------------------------------------------------------------------------
# live reasoning streaming — incremental accumulator + bounded preview
# ---------------------------------------------------------------------------


class ReasoningStream:
    """Incremental reasoning accumulator for live TUI streaming.

    ``client.py`` fires the ``on_reasoning`` callback per reasoning chunk
    (provider fields already normalized there: ``delta.reasoning_content``
    or ``delta.reasoning`` — provider differences never reach this class).
    The TUI feeds each chunk here and asks for a bounded ``preview()`` at
    ~10fps (the caller throttles with ``tui._StreamThrottle``), so the user
    watches the model's thought process evolve instead of a static
    "reasoning…" spinner.

    Chunks are kept in a list — O(1) append, joined once per read (the
    same O(n) discipline as ``client._chat_stream_once``). The live
    preview renders the single-line TAIL of the text: the latest thought
    is what the user watches, and the cost stays O(tail) per frame no
    matter how long the stream grows.
    """

    def __init__(self, tail_chars: int = 90):
        self._parts: list[str] = []
        self._chars = 0             # total reasoning chars fed
        self.chunks = 0             # total reasoning chunks fed
        self._tail_chars = max(1, tail_chars)

    def feed(self, piece: Any) -> None:
        """Accumulate one reasoning chunk. Never raises; ignores empties."""
        try:
            if not piece:
                return
            text = piece if isinstance(piece, str) else str(piece)
            if not text:
                return
            self._parts.append(text)
            self._chars += len(text)
            self.chunks += 1
        except Exception:
            pass

    @property
    def text(self) -> str:
        """The full accumulated reasoning text."""
        return "".join(self._parts)

    @property
    def chars(self) -> int:
        """Total reasoning characters fed so far."""
        return self._chars

    def preview(self, tail_chars: int | None = None) -> str:
        """Bounded single-line preview of the reasoning so far.

        Takes the tail of the accumulated text (the latest thought),
        collapses newlines/whitespace runs to single spaces, and prefixes
        "💭 " ("💭 …" when the tail was cut from a longer stream).
        Returns "" when nothing (non-blank) has arrived yet. Never raises.
        """
        try:
            limit = (self._tail_chars if tail_chars is None
                     else max(1, tail_chars))
            if self._chars == 0:
                return ""
            raw = "".join(self._parts)
            if not raw.strip():
                return ""
            truncated = len(raw) > limit
            tail = raw[-limit:] if truncated else raw
            single = " ".join(tail.split())
            if not single:
                return ""
            return ("💭 …" if truncated else "💭 ") + single
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# /thinking command
# ---------------------------------------------------------------------------

USAGE = (
    "usage:\n"
    "  /thinking            toggle reasoning display\n"
    "  /thinking on|off     show or hide reasoning\n"
    "  /thinking expand     show the full reasoning text\n"
    "  /thinking collapse   show only the 2-line preview"
)


def _agent_from_ui(ui: Any) -> Any:
    agent = getattr(ui, "agent", None)
    if agent is None:
        raise RuntimeError("/thinking needs the TUI host agent")
    return agent


def _tell(ui: Any, text: str) -> None:
    """Print through the TUI; fall back to stdout for duck-typed UIs."""
    printer = getattr(ui, "print_info", None)
    if callable(printer):
        try:
            printer(text)
            return
        except Exception:
            pass
    print(text)


def handle_thinking(ui: Any, arg: str) -> None:
    """``/thinking`` — toggle/set/expand the reasoning display.

    Prints confirmation via ``ui.print_info`` (falls back to stdout).
    Persists ``thinking_visible`` (and ``thinking_expanded``) into
    ``~/.fullagent/config.json``, merging any other keys.
    """
    agent = _agent_from_ui(ui)
    word = (arg or "").strip().lower()

    if word in ("", "toggle"):
        new = not bool(getattr(agent, "thinking_visible", True))
        agent.set_thinking_visible(new)
        _tell(ui, f"✓ thinking display: {'ON' if new else 'OFF'}")
    elif word == "on":
        agent.set_thinking_visible(True)
        _tell(ui, "✓ thinking display: ON")
    elif word == "off":
        agent.set_thinking_visible(False)
        _tell(ui, "✓ thinking display: OFF")
    elif word == "expand":
        agent.set_thinking_expanded(True)
        agent.set_thinking_visible(True)
        _tell(ui, "✓ thinking display: expanded")
    elif word == "collapse":
        agent.set_thinking_expanded(False)
        _tell(ui, "✓ thinking display: collapsed (2-line preview)")
    else:
        _tell(ui, f"unknown option: {word}\n{USAGE}")


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Wire thinking display into an agent (duck-typed, no imports).

    Attaches ``agent.thinking_visible`` (persisted, default True),
    ``agent.thinking_expanded`` (persisted, default False) and helpers:
    ``agent.set_thinking_visible(bool)``, ``agent.set_thinking_expanded
    (bool)``, plus the module's ``extract_thinking``/``format_thinking``
    for the TUI render path.
    """
    agent.thinking_visible = _load_visible()
    agent.thinking_expanded = _load_expanded()

    def set_thinking_visible(value: bool) -> bool:
        agent.thinking_visible = bool(value)
        _persist_visible(agent.thinking_visible)
        return agent.thinking_visible

    def set_thinking_expanded(value: bool) -> bool:
        agent.thinking_expanded = bool(value)
        _persist_expanded(agent.thinking_expanded)
        return agent.thinking_expanded

    agent.set_thinking_visible = set_thinking_visible
    agent.set_thinking_expanded = set_thinking_expanded
    agent.extract_thinking = extract_thinking
    agent.format_thinking = format_thinking


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.thinking`  →  PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    import tempfile

    # Isolate persistence: redirect the config dir for the self-test.
    global _APP_DIR
    tmp = tempfile.mkdtemp(prefix="thinking_selftest_")
    _APP_DIR = tmp

    # -- extract_thinking: every shape client.py produces -----------------
    m1 = {"role": "assistant", "content": "hi",
          "reasoning_content": "deep thought 1"}
    assert extract_thinking(m1) == "deep thought 1"

    m2 = {"role": "assistant", "content": None, "reasoning": "deep thought 2",
          "tool_calls": []}
    assert extract_thinking(m2) == "deep thought 2"

    m3 = {"reasoning_details": [
        {"type": "reasoning.text", "text": "part one. "},
        {"type": "reasoning.text", "text": "part two."},
    ]}
    assert extract_thinking(m3) == "part one. part two.", extract_thinking(m3)

    m4 = {"role": "assistant", "content": [
        {"type": "text", "text": "hello"},
        {"type": "reasoning", "reasoning": "block thought"},
    ]}
    assert extract_thinking(m4) == "block thought"

    m5 = {"content": [{"type": "thinking", "thinking": "anthropic style"}]}
    assert extract_thinking(m5) == "anthropic style"

    # precedence: reasoning_content wins over reasoning
    m6 = {"reasoning": "r", "reasoning_content": "rc"}
    assert extract_thinking(m6) == "rc"

    # nothing → ""
    assert extract_thinking({"role": "assistant", "content": "hi"}) == ""
    assert extract_thinking({}) == ""
    assert extract_thinking({"reasoning": "   "}) == ""
    # never raises on garbage
    assert extract_thinking(None) == ""
    assert extract_thinking("nope") == ""
    assert extract_thinking({"reasoning_details": [{"weird": object()}]}) == ""

    # -- format_thinking --------------------------------------------------
    five = "\n".join(f"line {i}" for i in range(1, 6))
    collapsed = format_thinking(five, collapsed=True)
    assert "line 1" in collapsed and "line 2" in collapsed
    assert "line 3" not in collapsed
    assert "(3 more lines, /thinking to expand)" in collapsed
    assert collapsed.startswith("\x1b[2m  "), repr(collapsed[:12])  # dim+indent

    short = "only\ntwo"
    assert "more lines" not in format_thinking(short, collapsed=True)
    assert "only" in format_thinking(short, collapsed=True)

    expanded = format_thinking(five, collapsed=False)
    for i in range(1, 6):
        assert f"line {i}" in expanded
    assert "(3 more lines" not in expanded

    assert format_thinking("") == ""
    assert format_thinking("   ") == ""

    # -- ReasoningStream: live incremental accumulation -------------------
    # Simulate a stream of chunks exactly like client._chat_stream_once
    # delivers them to the TUI's on_reasoning callback: the accumulator
    # must grow incrementally, and the preview must always reflect the
    # latest text.
    rs = ReasoningStream(tail_chars=40)
    assert rs.text == "" and rs.chars == 0 and rs.chunks == 0
    assert rs.preview() == "", "no preview before any chunk"

    streamed: list[str] = []
    chunks = ["The user wants ", "a quick sort.\n", "Plan: partition ",
              "around a pivot, then recurse on both halves."]
    seen_tails: list[str] = []
    for i, c in enumerate(chunks):
        rs.feed(c)
        streamed.append(c)
        # full text grows incrementally — nothing dropped, nothing reordered
        assert rs.text == "".join(streamed), f"chunk {i}"
        assert rs.chunks == i + 1
        assert rs.chars == sum(len(s) for s in streamed)
        pv = rs.preview()
        assert pv, f"preview must be non-empty after chunk {i}"
        assert pv.startswith("💭 "), repr(pv)
        assert "\n" not in pv, "preview must be single-line"
        seen_tails.append(pv)

    # the preview follows the stream: the last preview ends with the
    # tail of the latest chunk
    assert seen_tails[-1].endswith("recurse on both halves."), seen_tails[-1]
    # incremental growth: later previews show later text
    assert "pivot" in seen_tails[-1] or "recurse" in seen_tails[-1]

    # long stream → bounded tail with ellipsis marker
    rs2 = ReasoningStream(tail_chars=20)
    for _ in range(50):
        rs2.feed("lorem ipsum dolor sit amet ")
    p2 = rs2.preview()
    assert p2.startswith("💭 …"), repr(p2[:10])
    assert len(p2) <= len("💭 …") + 20, len(p2)
    assert "\n" not in p2

    # explicit tail_chars override
    p3 = rs2.preview(tail_chars=5)
    assert p3.startswith("💭 …") and len(p3) <= len("💭 …") + 5, repr(p3)

    # garbage never breaks the stream
    rs3 = ReasoningStream()
    rs3.feed("")
    rs3.feed(None)
    rs3.feed("   \n  ")
    assert rs3.preview() == "", "blank-only stream has no preview"
    rs3.feed("real thought")
    assert "real thought" in rs3.preview()
    rs3.feed(123)                       # non-str coerced, not crashing
    assert "123" in rs3.text

    # whitespace collapse: newlines/tabs become single spaces
    rs4 = ReasoningStream(tail_chars=200)
    rs4.feed("line one\n\nline\t\ttwo")
    assert rs4.preview() == "💭 line one line two", repr(rs4.preview())

    # -- register + persistence round-trip --------------------------------
    class FakeAgent:
        pass

    a = FakeAgent()
    register(a)
    assert a.thinking_visible is True      # default
    assert a.thinking_expanded is False    # default
    assert callable(a.set_thinking_visible)
    assert a.extract_thinking is extract_thinking
    assert a.format_thinking is format_thinking

    a.set_thinking_visible(False)
    assert a.thinking_visible is False
    # fresh register reads the persisted value
    b = FakeAgent()
    register(b)
    assert b.thinking_visible is False
    # other keys in config.json are preserved
    with open(os.path.join(tmp, "config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    assert raw["thinking_visible"] is False
    raw["some_other_key"] = "kept"
    with open(os.path.join(tmp, "config.json"), "w",
              encoding="utf-8") as fh:
        json.dump(raw, fh)
    a.set_thinking_visible(True)
    with open(os.path.join(tmp, "config.json"), encoding="utf-8") as fh:
        merged = json.load(fh)
    assert merged["thinking_visible"] is True
    assert merged["some_other_key"] == "kept"

    # corrupted config.json → defaults, never raises
    with open(os.path.join(tmp, "config.json"), "w",
              encoding="utf-8") as fh:
        fh.write("{not json")
    c = FakeAgent()
    register(c)
    assert c.thinking_visible is True

    # -- handle_thinking with a fake UI -----------------------------------
    class FakeUI:
        def __init__(self, agent):
            self.agent = agent
            self.shown: list[str] = []

        def print_info(self, text, color=None):
            self.shown.append(text)

    d = FakeAgent()
    register(d)
    ui = FakeUI(d)

    handle_thinking(ui, "")            # toggle True → False
    assert d.thinking_visible is False
    assert "OFF" in ui.shown[-1]

    handle_thinking(ui, "")            # toggle False → True
    assert d.thinking_visible is True
    assert "ON" in ui.shown[-1]

    handle_thinking(ui, "off")
    assert d.thinking_visible is False
    handle_thinking(ui, "on")
    assert d.thinking_visible is True

    handle_thinking(ui, "expand")
    assert d.thinking_expanded is True and d.thinking_visible is True
    handle_thinking(ui, "collapse")
    assert d.thinking_expanded is False

    before = d.thinking_visible
    handle_thinking(ui, "bogus")
    assert d.thinking_visible == before
    assert "usage:" in ui.shown[-1]

    # persisted across the fake-UI toggles
    e = FakeAgent()
    register(e)
    assert e.thinking_visible == d.thinking_visible
    assert e.thinking_expanded == d.thinking_expanded

    print("PASS")


if __name__ == "__main__":
    _selftest()
