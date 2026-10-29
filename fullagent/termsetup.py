"""`/terminal-setup` — terminal capability diagnostics.

Read-only check: reports truecolor/256-color support, Unicode locale,
terminal size and emoji rendering ability. Never changes anything;
prints concrete fix hints (export ... / locale guidance) for every
failed check.

``python3 -m fullagent.termsetup`` runs the built-in self-test.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Tuple

TRUECOLOR_VALUES = ("truecolor", "24bit")


def detect_truecolor(env: Dict[str, str] | None = None) -> bool:
    """True when COLORTERM advertises truecolor/24bit."""
    env = os.environ if env is None else env
    return (env.get("COLORTERM", "") or "").strip().lower() in TRUECOLOR_VALUES


def detect_256color(env: Dict[str, str] | None = None) -> bool:
    """True when TERM names a 256-color terminal (or truecolor is present)."""
    env = os.environ if env is None else env
    if detect_truecolor(env):
        return True
    return "256color" in (env.get("TERM", "") or "").lower()


def detect_unicode(env: Dict[str, str] | None = None) -> bool:
    """True when LC_ALL/LC_CTYPE/LANG advertises a UTF-8 locale."""
    env = os.environ if env is None else env
    for key in ("LC_ALL", "LC_CTYPE", "LANG"):
        val = (env.get(key, "") or "").lower().replace("-", "")
        if "utf8" in val or "utf-8" in val:
            return True
    return False


def terminal_size() -> Tuple[int, int]:
    """Current terminal size as (cols, rows); never raises."""
    try:
        size = shutil.get_terminal_size(fallback=(80, 24))
        return max(1, int(size.columns)), max(1, int(size.rows))
    except Exception:
        return (80, 24)


def emoji_width_ok() -> bool:
    """Best-effort emoji width check; never raises.

    True when the process can measure wide characters: ``wcwidth`` (if
    installed) reports width 2 for an emoji, else we fall back to the
    Python-level proxy of ``len("\U0001f600".encode("utf-16-le"))`` and the
    UTF-8 locale detection. Any failure returns False, never raises.
    """
    try:
        try:
            import wcwidth  # type: ignore

            width = wcwidth.wcswidth("\U0001f600")
            return width == 2
        except ImportError:
            pass
        # Heuristic fallback: python encodes the emoji to 4 bytes and the
        # locale supports UTF-8 -> the terminal very likely renders it.
        return len("\U0001f600".encode("utf-8")) == 4 and detect_unicode()
    except Exception:
        return False


def diagnose() -> List[Dict[str, Any]]:
    """Return one result dict per check: name / ok / detail / fix."""
    results: List[Dict[str, Any]] = []

    colorterm = os.environ.get("COLORTERM", "") or "(unset)"
    results.append({
        "name": "truecolor",
        "ok": detect_truecolor(),
        "detail": f"COLORTERM={colorterm}",
        "fix": ("export COLORTERM=truecolor in your shell profile, or "
                "switch to a truecolor terminal (e.g. WezTerm, kitty, "
                "Alacritty, modern tmux)"),
    })

    term = os.environ.get("TERM", "") or "(unset)"
    results.append({
        "name": "256 colors",
        "ok": detect_256color(),
        "detail": f"TERM={term}",
        "fix": "export TERM=xterm-256color (inside tmux/screen use tmux-256color)",
    })

    locale = (os.environ.get("LC_ALL")
              or os.environ.get("LC_CTYPE")
              or os.environ.get("LANG") or "(unset)")
    results.append({
        "name": "unicode locale",
        "ok": detect_unicode(),
        "detail": f"locale={locale}",
        "fix": ("use a UTF-8 locale, e.g. 'export LC_ALL=en_US.UTF-8'; "
                "on Debian/Ubuntu run 'dpkg-reconfigure locales'"),
    })

    cols, rows = terminal_size()
    results.append({
        "name": "terminal size",
        "ok": cols >= 80 and rows >= 24,
        "detail": f"{cols}x{rows}",
        "fix": "enlarge the terminal to at least 80x24 columns/rows",
    })

    results.append({
        "name": "emoji width",
        "ok": emoji_width_ok(),
        "detail": ("wcwidth reports emoji width 2"
                   if emoji_width_ok() else "cannot confirm 2-cell emoji width"),
        "fix": ("pip install wcwidth for accurate detection; use a font "
                "with emoji glyphs (Nerd Fonts / Noto Color Emoji)"),
    })
    return results


def handle_terminal_setup(ui, arg: str) -> None:
    """TUI handler: print the diagnostics report. Diagnose only, changes nothing."""
    _ = arg  # accepted for signature parity; no subcommands yet
    ui.print_info("terminal diagnostics (read-only — nothing was changed):")
    for r in diagnose():
        mark = "\u2713" if r["ok"] else "\u2717"  # ✓ / ✗
        ui.print_info(f"  {mark} {r['name']}: {r['detail']}")
        if not r["ok"]:
            ui.print_info(f"    fix: {r['fix']}")
    if not all(r["ok"] for r in diagnose()):
        ui.print_error("some checks failed — apply the 'fix:' hints above "
                       "and re-run /terminal-setup")


def register(agent: Any) -> None:
    """Wire into an agent (duck-typed). The check is TUI-side; nothing to add."""
    agent.termsetup_ready = True


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.termsetup`  →  PASS
# ---------------------------------------------------------------------------

class _FakeUI:
    def __init__(self):
        self.infos = []
        self.errors = []

    def print_info(self, text, color=None):
        self.infos.append(text)

    def print_error(self, text):
        self.errors.append(text)


def _selftest() -> None:
    # --- detector matrix with injected envs ---
    assert detect_truecolor({"COLORTERM": "truecolor"}) is True
    assert detect_truecolor({"COLORTERM": "24bit"}) is True
    assert detect_truecolor({"COLORTERM": "TRUECOLOR"}) is True
    assert detect_truecolor({"COLORTERM": "256color"}) is False
    assert detect_truecolor({}) is False

    assert detect_256color({"TERM": "xterm-256color"}) is True
    assert detect_256color({"COLORTERM": "truecolor", "TERM": "xterm"}) is True
    assert detect_256color({"TERM": "xterm"}) is False
    assert detect_256color({}) is False

    assert detect_unicode({"LANG": "en_US.UTF-8"}) is True
    assert detect_unicode({"LC_ALL": "C.utf8"}) is True
    assert detect_unicode({"LANG": "C"}) is False
    assert detect_unicode({}) is False

    # default env (None) delegates to os.environ and never raises
    assert isinstance(detect_truecolor(), bool)
    assert isinstance(detect_256color(), bool)
    assert isinstance(detect_unicode(), bool)

    # --- terminal_size ---
    cols, rows = terminal_size()
    assert isinstance(cols, int) and isinstance(rows, int)
    assert cols >= 1 and rows >= 1

    # --- emoji_width_ok never raises ---
    assert isinstance(emoji_width_ok(), bool)

    # --- diagnose() shape ---
    report = diagnose()
    assert isinstance(report, list) and len(report) == 5, report
    names = [r["name"] for r in report]
    assert names == ["truecolor", "256 colors", "unicode locale",
                     "terminal size", "emoji width"], names
    for r in report:
        assert set(r.keys()) == {"name", "ok", "detail", "fix"}, r
        assert isinstance(r["ok"], bool)
        assert isinstance(r["detail"], str) and r["detail"]
        assert isinstance(r["fix"], str) and r["fix"]

    # --- handler prints ✓/✗ and fix hints, changes nothing ---
    before_env = dict(os.environ)
    ui = _FakeUI()
    handle_terminal_setup(ui, "")
    assert dict(os.environ) == before_env, "handler must not change env"
    joined = "\n".join(ui.infos)
    assert "terminal diagnostics" in joined, ui.infos
    assert "✓" in joined or "✗" in joined, ui.infos
    failed = [r for r in report if not r["ok"]]
    if failed:
        assert "fix:" in joined, ui.infos
        assert ui.errors, "failing checks should add an error line"
    else:
        assert not ui.errors

    # --- register is additive ---
    class FakeAgent:
        pass

    a = FakeAgent()
    register(a)
    assert a.termsetup_ready is True

    print("PASS")


if __name__ == "__main__":
    _selftest()
