"""Visual context-usage meter for the TUI border.

Renders a block-style usage bar (``████████░░░░░░``) as
prompt_toolkit ``(text, style)`` fragments, the same fragment shape the
border code in :mod:`fullagent.tui` uses. Colour thresholds mirror the
ones the border applies today:

    - ``< 60%``  green
    - ``< 85%``  yellow
    - ``>= 85%`` red

Hex colours match ``tui.C`` so the meter looks identical to the
hand-rolled ``ctx`` segment it replaces.

Public API:
    - :func:`render_meter` -- bar for a raw percentage, honours a width.
    - :func:`context_bar` -- pulls the live estimate from the UI/agent
      exactly like ``tui._top_fragments`` does today and returns
      border-ready fragments with an ``NN%`` label.
    - :func:`register` -- no-op wire-up for the feature-modules tuple.
"""

from __future__ import annotations

import math
import time
from typing import Any, List, Tuple

Fragment = Tuple[str, str]

# mirrors tui.C for the three meter colours
GREEN = "#50fa7b"
YELLOW = "#f1fa8c"
RED = "#ff5555"

FILLED = "\u2588"  # █
EMPTY = "\u2591"   # ░

THRESHOLD_GREEN = 60.0
THRESHOLD_YELLOW = 85.0

DEFAULT_WIDTH = 20
MIN_WIDTH = 4
MAX_WIDTH = 80

# same 1s cache the border uses (the border re-renders every frame;
# estimate_tokens walks the whole message list)
_CACHE_TTL = 1.0


def _clamp_pct(used_pct: Any) -> float:
    """Sanitise anything into a 0..100 float (NaN/None/negatives safe)."""
    try:
        pct = float(used_pct)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(pct) or math.isinf(pct):
        return 0.0
    return max(0.0, min(100.0, pct))


def _clamp_width(width: Any) -> int:
    try:
        w = int(width)
    except (TypeError, ValueError):
        return DEFAULT_WIDTH
    return max(MIN_WIDTH, min(MAX_WIDTH, w))


def meter_color(used_pct: float) -> str:
    """Hex colour for *used_pct* per the border thresholds."""
    pct = _clamp_pct(used_pct)
    if pct < THRESHOLD_GREEN:
        return GREEN
    if pct < THRESHOLD_YELLOW:
        return YELLOW
    return RED


def render_meter(used_pct: float, width: int = DEFAULT_WIDTH) -> List[Fragment]:
    """Render a context-usage block bar as prompt_toolkit fragments.

    Args:
        used_pct: context used, in percent (clamped to 0..100).
        width: bar width in characters; clamped to ``MIN_WIDTH..MAX_WIDTH``.

    Returns:
        A single ``(bar_text, style)`` fragment, e.g.
        ``[("██████░░░░", "bold #50fa7b")]``. Bar text is exactly
        *width* display cells wide.
    """
    pct = _clamp_pct(used_pct)
    w = _clamp_width(width)
    filled = int(round(pct / 100.0 * w))
    filled = max(0, min(w, filled))
    bar = FILLED * filled + EMPTY * (w - filled)
    style = f"bold {meter_color(pct)}"
    return [(bar, style)]


def _fresh_pct(agent: Any) -> float:
    """Mirror of how tui._top_fragments computes context usage today."""
    from .client import estimate_tokens
    messages = getattr(agent, "messages", [])
    model = getattr(agent, "model", None)
    model_id = getattr(model, "id", "") if model is not None else ""
    window = getattr(model, "context_window", 0) if model is not None else 0
    used = estimate_tokens(messages, model_id)
    if not window or window <= 0:
        return 0.0
    return min(100.0, used * 100.0 / window)


def cached_pct(ui: Any) -> float:
    """Return the 0..100 context percentage, cached 1s like the border.

    Reuses ``ui._ctx_cache``/``ui._ctx_cache_ts`` when present so the
    coordinator can share one reading with the old ``◉ ctx`` segment.
    Never raises — on failure returns the last good reading, else 0.
    """
    now = time.time()
    cache = getattr(ui, "_ctx_cache", None)
    cache_ts = getattr(ui, "_ctx_cache_ts", 0.0)
    if cache is None or now - cache_ts > _CACHE_TTL:
        try:
            agent = getattr(ui, "agent", None)
            new = _fresh_pct(agent) if agent is not None else 0.0
        except Exception:
            new = cache if isinstance(cache, (int, float)) else 0.0
        try:
            ui._ctx_cache = new
            ui._ctx_cache_ts = now
        except Exception:
            pass
        return float(new)
    return float(cache)


def context_bar(ui: Any, width: Any = None) -> List[Fragment]:
    """Border-ready fragments for the context meter with an ``NN%`` label.

    Pulls the live estimate from ``ui`` exactly like
    ``tui._top_fragments`` does (same ``estimate_tokens`` call, same
    1s cache). ``width`` is the bar width in characters; when ``None`` a
    compact 10-wide bar is used so the border segment stays short.
    """
    pct = _clamp_pct(cached_pct(ui))
    pct_int = int(round(pct))
    bar_width = DEFAULT_WIDTH // 2 if width is None else _clamp_width(width)
    bar_frags = render_meter(pct, width=bar_width)
    color = meter_color(pct)
    label = f" ◉ ctx {pct_int}% "
    return [(label, f"bold {color}"), *bar_frags, (" ", "class:box")]


def register(agent: Any) -> None:
    """Wire into an agent (duck-typed).

    The meter is pure presentation — nothing to register on the agent
    itself. This exists so the module slots into the feature-modules
    tuple in ``Agent._register_feature_modules`` like every other module.
    """
    try:
        agent.ctx_meter_enabled = True
    except Exception:
        pass


if __name__ == "__main__":
    # -- self-test -----------------------------------------------------
    fails: List[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            fails.append(name)

    # thresholds: block counts + colours
    for pct, filled_expect, color_expect in (
        (0, 0, GREEN),
        (25, 5, GREEN),
        (59, 12, GREEN),
        (60, 12, YELLOW),
        (84, 17, YELLOW),
        (85, 17, RED),
        (100, 20, RED),
    ):
        frags = render_meter(pct, width=20)
        check(f"render_meter({pct}) one fragment", len(frags) == 1)
        bar, style = frags[0]
        check(f"render_meter({pct}) width 20", len(bar) == 20)
        check(f"render_meter({pct}) filled {filled_expect}",
              bar.count(FILLED) == filled_expect
              and bar.count(EMPTY) == 20 - filled_expect)
        check(f"render_meter({pct}) color",
              style == f"bold {color_expect}")

    # custom width honoured
    bar10, _ = render_meter(50, width=10)[0]
    check("width=10 honoured", len(bar10) == 10
          and bar10.count(FILLED) == 5)
    # width clamping
    check("width=1 clamped to MIN", len(render_meter(50, width=1)[0][0]) == MIN_WIDTH)
    check("width=999 clamped to MAX",
          len(render_meter(50, width=999)[0][0]) == MAX_WIDTH)

    # weird inputs never crash
    for weird in (None, float("nan"), float("inf"), float("-inf"),
                  -5, 150, "40", object()):
        try:
            f = render_meter(weird)
            bar, style = f[0]
            check(f"weird {weird!r} -> valid fragment",
                  len(bar) == DEFAULT_WIDTH
                  and bar.count(FILLED) + bar.count(EMPTY) == DEFAULT_WIDTH)
        except Exception as e:  # noqa: BLE001
            check(f"weird {weird!r} no crash ({e})", False)

    # context_bar against a fake UI mirroring tui.py's shape
    class FakeModel:
        id = "test-model"
        context_window = 200000

    class FakeUI:
        def __init__(self):
            self.agent = type("A", (), {})()
            self.agent.messages = [{"role": "user",
                                    "content": "x" * 4000}]
            self.agent.model = FakeModel()
            self._ctx_cache = None
            self._ctx_cache_ts = 0.0

    ui = FakeUI()
    frags = context_bar(ui)
    check("context_bar returns fragments",
          frags and all(isinstance(t, str) and isinstance(s, str)
                        for t, s in frags))
    check("context_bar label has %",
          any("%" in t for t, _ in frags))
    # second call uses the 1s cache (no second estimate crash path)
    frags2 = context_bar(ui)
    check("context_bar cached call stable", frags == frags2)

    # context_bar with no agent must not raise
    try:
        empty = context_bar(object())
        check("context_bar no-agent no crash",
              any("0%" in t for t, _ in empty))
    except Exception as e:  # noqa: BLE001
        check(f"context_bar no-agent no crash ({e})", False)

    # register
    class FakeAgent:
        pass

    fa = FakeAgent()
    register(fa)
    check("register sets flag", getattr(fa, "ctx_meter_enabled", False))

    print()
    if fails:
        print(f"{len(fails)} FAILURES")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASS")
