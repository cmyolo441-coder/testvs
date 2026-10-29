"""Regression coverage for visual polish and stale-edit edge cases.

These tests are pure/headless: Rich writes to a StringIO and filestat uses a
temporary file. No network, no provider calls, no interactive terminal.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import importlib.util
import io
import re

from rich.console import Console

from fullagent import cliui, filestat, fx, prewarm


def test_boot_reveal_preserves_truecolor_gradient():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor",
                      width=60)
    fx.boot_reveal(console, lambda: fx.gradient_text("ABC"), delay=0)
    out = buf.getvalue()
    assert re.sub(r"\x1b\[[0-9;]*m", "", out).strip() == "ABC"
    assert "\x1b[38;2;" in out, "ANSI gradient was escaped instead of rendered"


def test_welcome_logo_uses_multicolour_gradient():
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, color_system="truecolor",
                      width=100)
    colors = {
        "border": "#5b6eae", "accent": "#c084fc", "cyan": "#22d3ee",
        "green": "#34d399", "yellow": "#facc15", "orange": "#fb923c",
        "red": "#f87171", "pink": "#f472b6", "fg": "#f8fafc",
        "dim": "#64748b",
    }
    console.print(cliui.welcome(
        width=100, version="9.9.9", model_label="Test Model",
        effort_label="max", signed_in="API key", session="abcd1234",
        cwd="~/repo", colors=colors))
    out = buf.getvalue()
    assert "Tips for getting started" in out
    assert out.count("\x1b[38;2;") >= 8


def test_stale_check_keeps_content_digest_defence(tmp_path):
    path = tmp_path / "same-stat.txt"
    path.write_text("xy\n", encoding="utf-8")

    class Agent:
        pass

    agent = Agent()
    filestat.register(agent)
    filestat.note_read(agent, str(path))
    current = filestat._stat(str(path))
    assert current is not None

    # Simulate a coarse filesystem where only the digest changed: mtime and
    # size are identical, so a two-field baseline would miss the rewrite.
    store = filestat._store_for(agent)
    store[filestat._key(str(path))] = (current[0], current[1], "0" * 32)
    warning = filestat.check_stale(agent, str(path))
    assert "changed since you read it" in warning


def test_prewarm_stat_and_filestat_share_the_same_fingerprint(tmp_path):
    path = tmp_path / "watched.py"
    path.write_text("v1", encoding="utf-8")
    first = prewarm._stat(str(path))
    assert first == filestat._stat(str(path))
    assert first is not None and len(first) == 3

    path.write_text("v2", encoding="utf-8")
    second = prewarm._stat(str(path))
    assert second != first


def test_enforcex_package_has_module_entrypoint():
    spec = importlib.util.find_spec("fullagent.enforcex.__main__")
    assert spec is not None and spec.origin
