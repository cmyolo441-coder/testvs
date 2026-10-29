"""fx — pure visual-effects layer for the TUI (v4 "aurora" polish).

Everything here is additive chrome: a function that fails or a terminal
that can't render it must never take a session down. Callers wrap uses
in try/except and fall back to plain output.

Effects:
- gradient_text      per-character truecolor gradient (rainbow / custom)
- spinner_color      color-cycling spinner so the busy glyph shimmers
- boot_reveal        line-by-line reveal of the welcome card (typewriter)
- shimmer_flash      one-shot "✦" flash with a rising gradient

Self-test: python -m fullagent.fx
"""

from __future__ import annotations

import time
from typing import Callable

# Rainbow used for gradients (truecolor, degrades via rich's color system
# to the nearest 256/16 color on limited terminals).
RAINBOW = ["#ff5f6d", "#ff7b54", "#ffb26b", "#ffd56b", "#7bed9f",
           "#4ecdc4", "#45aaf2", "#a55eea"]

# Spinner glyph colors, cycled by frame index — the busy spinner breathes
# through the palette instead of sitting on one flat color.
SPINNER_COLORS = ["#45aaf2", "#4ecdc4", "#7bed9f", "#ffd56b",
                  "#ff7b54", "#a55eea"]


def _hex(c: str) -> str:
    """Truecolor hex passthrough; rich accepts these directly."""
    return c


def gradient_text(text: str, palette: list[str] | None = None) -> object:
    """Return a rich Text with a per-character horizontal gradient.

    Uses a lazy import so headless modules that import fx never require
    rich (the TUI always has it).
    """
    from rich.text import Text

    pal = palette or RAINBOW
    t = Text()
    n = max(1, len(text))
    for i, ch in enumerate(text):
        t.append(ch, style=_hex(pal[int(i * (len(pal) - 1) / max(1, n - 1))]))
    return t


def spinner_color(i: int) -> str:
    """Color for spinner frame index i (cycles through the palette)."""
    if not SPINNER_COLORS:
        return "cyan"
    return SPINNER_COLORS[i % len(SPINNER_COLORS)]


def boot_reveal(console, render: Callable[[], object],
                delay: float = 0.028) -> None:
    """Reveal a rendered block line-by-line (Devin/Qoder-style boot).

    `render()` returns a rich renderable (already built once, so layout
    never changes mid-reveal); we capture its lines via the console
    export and print them back one at a time. Falls back to a single
    plain print on any failure.
    """
    try:
        from rich.text import Text

        with console.capture() as cap:
            console.print(render())
        lines = cap.get().splitlines()
        for line in lines:
            # Colors are enabled for the real TUI, so capture returns ANSI
            # sequences. Printing that raw string would make Rich escape the
            # codes (and lose the gradient); parse them back into a Text so
            # the reveal preserves every truecolor span.
            console.print(Text.from_ansi(line))
            if delay > 0:
                time.sleep(delay)
    except Exception:
        try:
            console.print(render())
        except Exception:
            pass


def shimmer_flash(console, text: str, steps: int = 8,
                  delay: float = 0.035) -> None:
    """Print `text` with a one-shot brightness shimmer (rising edge)."""
    from rich.text import Text

    try:
        for k in range(max(1, steps)):
            t = Text()
            color = _hex(SPINNER_COLORS[k % len(SPINNER_COLORS)])
            t.append(text, style=f"bold {color}")
            console.print(t, end="\r", soft_wrap=False)
            time.sleep(delay)
        t = Text()
        t.append(text, style=f"bold {_hex(RAINBOW[-1])}")
        console.print(t)
    except Exception:
        console.print(text)


# ── self-test ────────────────────────────────────────────────────────────
def _self_test() -> int:
    from rich.console import Console
    import io

    console = Console(file=io.StringIO(), force_terminal=False, width=60)

    t = gradient_text("FULLAGENT")
    assert len(str(t)) == len("FULLAGENT"), "gradient length mismatch"

    assert spinner_color(0) == SPINNER_COLORS[0]
    assert spinner_color(len(SPINNER_COLORS)) == SPINNER_COLORS[0], \
        "spinner colors must cycle"

    # boot_reveal prints every line and never raises
    boot_reveal(console, lambda: "hello", delay=0)
    out = console.file.getvalue()
    assert "hello" in out, "boot_reveal lost content"

    # broken renderer must not raise
    def _boom():
        raise RuntimeError("boom")
    boot_reveal(console, _boom, delay=0)

    # bad spinner index wraps
    assert spinner_color(999) in SPINNER_COLORS

    print("FX SELF-TEST PASS (gradient, spinner cycle, reveal, fallbacks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
