"""Vi keybindings toggle for the TUI input line (``/vim``).

Claude Code CLI-style: toggle between emacs (default) and vi editing
mode for the prompt input, persisting the choice across sessions.

Public API for the TUI and the agent:
    - :func:`register` -- attach the persisted preference to an agent
      (duck-typed) as ``agent.vim_mode``. Called once at agent startup.
    - :func:`handle_vim` -- TUI slash-command handler: ``/vim`` toggles,
      ``/vim on`` / ``/vim off`` set explicitly. Prints via the UI.
    - :func:`apply_vim_mode` -- helper the coordinator calls right after
      creating the TUI so the prompt starts in the right editing mode;
      ``handle_vim`` calls it too so the switch is immediate.
    - :func:`load_vim_mode` / :func:`set_vim_mode` -- persistence helpers.

The preference persists as the ``vim_mode`` key in
``~/.fullagent/config.json`` (``$FULLAGENT_HOME`` is honoured). The write
merges with the existing file — other keys are never clobbered.

The prompt_toolkit hook point is the ``Application`` object the TUI owns
(``ui.app`` in ``fullagent/tui.py``): its ``editing_mode`` attribute is
set to ``EditingMode.VI`` / ``EditingMode.EMACS``. A ``ui.vim_mode``
flag is stamped as well so the TUI builder can re-apply the mode
whenever it (re)creates the prompt session.

This module never imports ``.agent`` / ``.tui`` / ``prompt_toolkit`` at
module level, so there are no import cycles and the self-test needs no
terminal. The ``prompt_toolkit.enums.EditingMode`` import happens lazily
inside :func:`_set_app_editing_mode`.

``python3 -m fullagent.vimmode`` runs the built-in self-test.
"""

from __future__ import annotations

import json
import os
from typing import Any

CONFIG_KEY = "vim_mode"

USAGE = (
    "usage:\n"
    "  /vim            toggle vi keybindings on/off\n"
    "  /vim on         enable vi keybindings\n"
    "  /vim off        disable vi keybindings (emacs, default)"
)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def _config_path() -> str:
    """Path to the merged config file.

    Mirrors ``fullagent.config._pick_app_dir``'s first two candidates:
    ``$FULLAGENT_HOME/config.json``, else ``~/.fullagent/config.json``.
    """
    home = os.environ.get("FULLAGENT_HOME")
    if home:
        return os.path.join(home, "config.json")
    return os.path.join(os.path.expanduser("~"), ".fullagent", "config.json")


def load_vim_mode() -> bool:
    """Read the persisted preference; default ``False`` (emacs)."""
    cfg = _user_config()
    if cfg is not None:
        try:
            return bool(cfg.get(CONFIG_KEY))
        except Exception:
            pass  # fall through to the direct reader
    return _load_vim_mode_direct()


def _user_config():
    """A ``configcmd.UserConfig`` pointed at the vim store, or ``None``.

    Lazy import: ``configcmd`` is a sibling feature module and may not be
    importable in every environment; the direct JSON reader/writer below
    is the fallback.
    """
    try:
        from . import configcmd
        return configcmd.UserConfig(_config_path())
    except Exception:
        return None


def _load_vim_mode_direct() -> bool:
    """Fallback reader: direct JSON read of ``vim_mode``."""
    try:
        with open(_config_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return bool(data.get(CONFIG_KEY, False))
    except (OSError, ValueError):
        pass
    return False


def _persist_vim_mode(enabled: bool) -> None:
    """Write the preference, merging with the existing config file.

    Prefers ``configcmd.UserConfig`` (schema-validated, atomic) when
    available; falls back to a direct atomic JSON write. Either way other
    keys are never clobbered, and a crash mid-write can't leave
    truncated JSON behind.
    """
    cfg = _user_config()
    if cfg is not None:
        try:
            cfg.set(CONFIG_KEY, "true" if enabled else "false")
            return
        except Exception:
            pass  # fall through to the direct writer
    _persist_vim_mode_direct(enabled)


def _persist_vim_mode_direct(enabled: bool) -> None:
    """Fallback writer: direct atomic JSON write, merging existing keys."""
    path = _config_path()
    data: dict = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        data = {}
    data[CONFIG_KEY] = bool(enabled)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass  # persistence is best-effort; the in-memory value still wins


# ---------------------------------------------------------------------------
# core: flip the prompt_toolkit editing mode on the live UI
# ---------------------------------------------------------------------------

def is_vim_mode(agent: Any) -> bool:
    """Return the agent's current vim-mode preference (default False)."""
    return bool(getattr(agent, "vim_mode", False))


def _set_app_editing_mode(ui: Any, enabled: bool) -> bool:
    """Apply the mode to the prompt_toolkit ``Application`` on ``ui``.

    Sets ``ui.app.editing_mode`` to ``EditingMode.VI`` / ``EMACS`` and
    stamps ``ui.vim_mode`` so the TUI builder can re-apply the mode
    whenever it (re)creates the prompt session. Returns ``True`` when an
    app object was found and updated, ``False`` otherwise (the flag is
    still stamped either way).
    """
    enabled = bool(enabled)
    ui.vim_mode = enabled
    app = getattr(ui, "app", None)
    if app is None:
        return False
    try:
        from prompt_toolkit.enums import EditingMode
    except ImportError:
        return False
    try:
        app.editing_mode = EditingMode.VI if enabled else EditingMode.EMACS
        return True
    except Exception:
        return False


def apply_vim_mode(ui: Any) -> bool:
    """Apply the agent's persisted vim preference to a (new) UI.

    The coordinator calls this right after constructing the TUI so the
    prompt starts in the right editing mode; :func:`handle_vim` calls it
    too so a toggle takes effect immediately. Safe on duck-typed fakes.
    Returns the effective mode.
    """
    agent = getattr(ui, "agent", None)
    if agent is not None:
        enabled = is_vim_mode(agent)
    else:
        enabled = bool(getattr(ui, "vim_mode", False))
    _set_app_editing_mode(ui, enabled)
    return enabled


def set_vim_mode(ui: Any, enabled: bool) -> bool:
    """Set the preference on the agent, persist it, and apply it to the UI.

    Returns the effective mode.
    """
    enabled = bool(enabled)
    agent = getattr(ui, "agent", None)
    if agent is not None:
        agent.vim_mode = enabled
    _persist_vim_mode(enabled)
    _set_app_editing_mode(ui, enabled)
    return enabled


# ---------------------------------------------------------------------------
# slash-command handler
# ---------------------------------------------------------------------------

def handle_vim(ui: Any, arg: str) -> None:
    """``/vim`` — toggle vi keybindings for the TUI input line.

    ``/vim`` toggles, ``/vim on`` / ``/vim off`` set explicitly. The
    choice persists in ``~/.fullagent/config.json`` and the prompt's
    editing mode switches immediately. Prints feedback via the UI.
    """
    info = getattr(ui, "print_info", None) or print
    err = getattr(ui, "print_error", None) or print

    agent = getattr(ui, "agent", None)
    if agent is None:
        err("/vim needs the TUI host agent")
        return

    text = (arg or "").strip().lower()
    if text in ("", "toggle"):
        enabled = not is_vim_mode(agent)
    elif text in ("on", "1", "true", "yes", "enable", "enabled"):
        enabled = True
    elif text in ("off", "0", "false", "no", "disable", "disabled"):
        enabled = False
    else:
        err(USAGE)
        return

    set_vim_mode(ui, enabled)
    mode = "vi" if enabled else "emacs"
    info(f"✓ vi keybindings {'on' if enabled else 'off'} — "
         f"input editing mode is now {mode}")


# ---------------------------------------------------------------------------
# agent wiring
# ---------------------------------------------------------------------------

def register(agent: Any) -> None:
    """Attach the persisted vim preference to ``agent``.

    Sets ``agent.vim_mode`` (bool) from ``~/.fullagent/config.json`` and
    does nothing else — the coordinator applies it to the UI with
    :func:`apply_vim_mode` after TUI creation.
    """
    agent.vim_mode = load_vim_mode()


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.vimmode`  →  PASS
# ---------------------------------------------------------------------------

class _FakeEditingMode:
    VI = "VI"
    EMACS = "EMACS"


def _install_fake_editing_mode() -> None:
    """Provide prompt_toolkit.enums.EditingMode for the test env only."""
    import sys
    import types
    pt = sys.modules.get("prompt_toolkit")
    if pt is None:
        pt = types.ModuleType("prompt_toolkit")
        sys.modules["prompt_toolkit"] = pt
    enums = sys.modules.get("prompt_toolkit.enums")
    if enums is None:
        enums = types.ModuleType("prompt_toolkit.enums")
        enums.EditingMode = _FakeEditingMode
        sys.modules["prompt_toolkit.enums"] = enums
    pt.enums = enums


class _FakeApp:
    def __init__(self):
        self.editing_mode = _FakeEditingMode.EMACS


class _FakeAgent:
    def __init__(self):
        self.vim_mode = False


class _FakeUI:
    def __init__(self, with_app: bool = True):
        self.agent = _FakeAgent()
        self.app = _FakeApp() if with_app else None
        self.messages: list = []

    def print_info(self, msg, color=None):
        self.messages.append(("info", msg))

    def print_error(self, msg):
        self.messages.append(("error", msg))


def _selftest() -> None:
    import tempfile

    _install_fake_editing_mode()

    # isolate the config file for the self-test
    tmp = tempfile.mkdtemp(prefix="vimmode_selftest_")
    os.environ["FULLAGENT_HOME"] = tmp
    cfg_file = os.path.join(tmp, "config.json")

    # --- default is emacs/off ---
    assert load_vim_mode() is False

    # --- register reads persisted value ---
    a = _FakeAgent()
    register(a)
    assert a.vim_mode is False

    # --- /vim toggles on, flips editing_mode immediately ---
    ui = _FakeUI()
    register(ui.agent)
    handle_vim(ui, "")
    assert ui.agent.vim_mode is True
    assert ui.vim_mode is True
    assert ui.app.editing_mode == _FakeEditingMode.VI
    assert any("on" in m[1] for m in ui.messages if m[0] == "info"), ui.messages

    # --- persistence round-trip + merge (other keys survive) ---
    with open(cfg_file, "w", encoding="utf-8") as fh:
        json.dump({"theme": "dracula", "model_id": "abc"}, fh)
    handle_vim(ui, "off")
    assert ui.agent.vim_mode is False
    assert ui.app.editing_mode == _FakeEditingMode.EMACS
    with open(cfg_file, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["vim_mode"] is False, data
    assert data["theme"] == "dracula" and data["model_id"] == "abc", data

    # a fresh agent picks up the persisted value
    b = _FakeAgent()
    register(b)
    assert b.vim_mode is False

    with open(cfg_file, encoding="utf-8") as fh:
        data = json.load(fh)
    data["vim_mode"] = True
    with open(cfg_file, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    c = _FakeAgent()
    register(c)
    assert c.vim_mode is True

    # --- explicit on/off ---
    handle_vim(ui, "on")
    assert ui.agent.vim_mode is True
    assert ui.app.editing_mode == _FakeEditingMode.VI
    handle_vim(ui, "off")
    assert ui.agent.vim_mode is False
    assert ui.app.editing_mode == _FakeEditingMode.EMACS
    handle_vim(ui, "toggle")
    assert ui.agent.vim_mode is True

    # --- unknown arg: usage error, mode unchanged ---
    before = ui.agent.vim_mode
    handle_vim(ui, "banana")
    assert ui.agent.vim_mode is before
    assert any("usage:" in m[1] for m in ui.messages if m[0] == "error")

    # --- corrupt config file degrades to default, toggle re-creates ---
    with open(cfg_file, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    assert load_vim_mode() is False
    handle_vim(ui, "on")
    assert json.load(open(cfg_file, encoding="utf-8"))["vim_mode"] is True

    # --- apply_vim_mode: UI without an app still gets the flag ---
    ui2 = _FakeUI(with_app=False)
    ui2.agent.vim_mode = True
    assert apply_vim_mode(ui2) is True
    assert ui2.vim_mode is True

    # --- apply_vim_mode falls back to the UI flag when agent is absent ---
    ui3 = _FakeUI(with_app=True)
    ui3.agent = None
    ui3.vim_mode = True
    assert apply_vim_mode(ui3) is True
    assert ui3.app.editing_mode == _FakeEditingMode.VI

    # --- handle_vim survives a missing agent ---
    ui4 = _FakeUI()
    ui4.agent = None
    handle_vim(ui4, "on")  # must not raise
    assert any(m[0] == "error" for m in ui4.messages)

    # --- fallback: direct JSON writer when configcmd is unavailable ---
    import sys as _sys
    with open(cfg_file, "w", encoding="utf-8") as fh:
        json.dump({"theme": "dracula", "vim_mode": False}, fh)
    saved = _sys.modules.pop("fullagent.configcmd", None)
    _sys.modules["fullagent.configcmd"] = None  # force ImportError
    try:
        assert load_vim_mode() is False
        ui5 = _FakeUI()
        register(ui5.agent)
        handle_vim(ui5, "on")
        assert ui5.agent.vim_mode is True
        assert load_vim_mode() is True
        data = json.load(open(cfg_file, encoding="utf-8"))
        assert data["vim_mode"] is True, data
        assert data["theme"] == "dracula", data  # merge preserved
    finally:
        if saved is not None:
            _sys.modules["fullagent.configcmd"] = saved
        else:
            _sys.modules.pop("fullagent.configcmd", None)

    print("PASS")


if __name__ == "__main__":
    _selftest()
