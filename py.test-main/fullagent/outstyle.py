"""Output styles for agent responses (Claude Code `/output-style`).

Three named styles, each a short system-prompt instruction snippet that the
agent appends to its system prompt so the LLM answers in that register:

- ``concise``      — terse, short sentences, no preamble.
- ``detailed``     — thorough: edge cases, commands, file paths. (default)
- ``explanatory``  — teach as you go: explain what and why before each action.

Usage (wired in by the parent agent — see report):
    from fullagent import outstyle
    outstyle.register(agent)              # sets agent.output_style, adds tool
    outstyle.set_style(agent, "concise")   # raises ValueError on unknown
    outstyle.style_instruction(agent)     # -> snippet for the system prompt

The choice persists in ``~/.fullagent/output_style`` (single word).

This module never imports ``.agent`` — it is duck-typed against whatever
agent object ``register`` receives.
"""
from __future__ import annotations

import os
from pathlib import Path

STYLES = {
    "concise": "Be terse. Short sentences. No preamble, no filler.",
    "detailed": "Be thorough: cover edge cases, show commands and file paths.",
    "explanatory": "Teach as you go: explain what and why before each action.",
}

DEFAULT_STYLE = "detailed"


def _store_path() -> Path:
    """Where the style choice is persisted (single word)."""
    override = os.environ.get("FULLAGENT_OUTSTYLE_FILE")
    if override:
        return Path(override)
    base = Path(os.environ["FULLAGENT_HOME"]).expanduser() if os.environ.get("FULLAGENT_HOME") else (Path.home() / ".fullagent")
    return base / "output_style"


def load_style() -> str:
    """Read the persisted style; fall back to DEFAULT_STYLE."""
    try:
        name = _store_path().read_text(encoding="utf-8").strip().lower()
    except OSError:
        return DEFAULT_STYLE
    return name if name in STYLES else DEFAULT_STYLE


def _persist(name: str) -> None:
    path = _store_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n", encoding="utf-8")
    except OSError:
        pass  # persistence is best-effort; the in-memory value still wins


def set_style(agent, name: str) -> str:
    """Set the agent's output style; raises ValueError on unknown names."""
    key = (name or "").strip().lower()
    if key not in STYLES:
        raise ValueError(
            f"unknown output style {name!r}; choose from: "
            + ", ".join(sorted(STYLES)))
    agent.output_style = key
    _persist(key)
    return key


def get_style(agent) -> str:
    """Return the agent's current style; default is DEFAULT_STYLE."""
    name = getattr(agent, "output_style", None)
    return name if name in STYLES else DEFAULT_STYLE


def style_instruction(agent) -> str:
    """System-prompt snippet for the agent's current output style."""
    return STYLES[get_style(agent)]


def register(agent) -> None:
    """Restore the persisted style on ``agent`` and add the ``OutputStyle``
    tool so the LLM can switch styles itself during a turn."""
    agent.output_style = load_style()

    from .tools import Tool

    def _switch(style: str = "") -> str:
        try:
            return f"output style → {set_style(agent, style)}"
        except ValueError as exc:
            return str(exc)

    agent.tools["OutputStyle"] = Tool(
        "OutputStyle",
        "Switch the response output style. "
        "Args: style (concise | detailed | explanatory).",
        {"type": "object",
         "properties": {"style": {"type": "string",
                                  "enum": sorted(STYLES)}},
         "required": ["style"]},
        _switch,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile
    from types import SimpleNamespace

    with tempfile.TemporaryDirectory() as td:
        os.environ["FULLAGENT_OUTSTYLE_FILE"] = str(Path(td) / "output_style")

        agent = SimpleNamespace(tools={})
        register(agent)
        assert get_style(agent) == "detailed", "default should be detailed"
        assert isinstance(style_instruction(agent), str)

        for name in STYLES:
            set_style(agent, name)
            assert get_style(agent) == name, name
            assert style_instruction(agent) == STYLES[name], name

        # persisted round-trip: fresh agent picks up the saved word
        fresh = SimpleNamespace(tools={})
        register(fresh)
        assert get_style(fresh) == "explanatory", "persisted style should reload"
        assert _store_path().read_text().strip() == "explanatory"

        # invalid style raises
        for bad in ("", "balanced", "VERBOSE", None):
            try:
                set_style(agent, bad)  # type: ignore[arg-type]
            except ValueError:
                pass
            else:
                raise AssertionError(f"set_style({bad!r}) should raise")

        # unknown tool arg returns the error string instead of raising
        result = agent.tools["OutputStyle"].handler(style="bogus")
        assert "unknown output style" in result

        # get_style tolerates garbage attributes
        agent.output_style = "junk"
        assert get_style(agent) == "detailed"

        del os.environ["FULLAGENT_OUTSTYLE_FILE"]
    print("PASS")


if __name__ == "__main__":
    _self_test()
