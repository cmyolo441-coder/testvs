"""TUI `/config` command — interactive config editor.

Reads/writes user settings at ``~/.fullagent/config.json`` through a
:class:`UserConfig` with a known-keys schema. No imports of
``.agent``/``.tui``/``.config`` at module level, so there are no
import cycles (this module deliberately does NOT reuse
``fullagent.config`` — API keys must never be settable from here).

Public API for the TUI and the agent:
    - :func:`register` -- wire the feature into an agent (duck-typed);
      attaches ``agent.user_config`` (a :class:`UserConfig`).
    - :func:`handle_config` -- TUI handler: ``/config`` subcommands.

Usage from the TUI::

    /config                 show all settings
    /config list            show all settings
    /config get <key>       show one setting
    /config set <key> <val> set one setting (type-coerced + validated)
    /config reset <key>     reset one setting to its default

API keys are NEVER settable here: any key containing "key"/"token"/
"secret" (case-insensitive) is refused with an explanatory message.

``python3 -m fullagent.configcmd`` runs the built-in self-test.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

# Path constant — the self-test monkeypatches this to a temp file.
CONFIG_PATH = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"), "config.json")

USAGE = (
    "usage:\n"
    "  /config                 show all settings\n"
    "  /config list            show all settings\n"
    "  /config get <key>       show one setting\n"
    "  /config set <key> <val> set one setting\n"
    "  /config reset <key>     reset one setting to its default\n"
    "\n"
    "API keys cannot be set here — use environment variables instead."
)

# ---------------------------------------------------------------------------
# known-keys schema: key -> {"type": str|bool|choice, ... , "default": ...}
# ---------------------------------------------------------------------------

OUTPUT_STYLES = ("concise", "detailed", "explanatory")

SCHEMA = {
    "default_model": {
        "type": "str",
        "default": "",
        "desc": "default model id (empty = provider default)",
    },
    "auto_approve": {
        "type": "bool",
        "default": False,
        "desc": "auto-approve tool calls without prompting",
    },
    "vim_mode": {
        "type": "bool",
        "default": False,
        "desc": "vim keybindings in the TUI editor",
    },
    "output_style": {
        "type": "choice",
        "choices": OUTPUT_STYLES,
        "default": "concise",
        "desc": "response style: concise / detailed / explanatory",
    },
    "thinking_visible": {
        "type": "bool",
        "default": True,
        "desc": "show thinking blocks in the TUI",
    },
    "theme": {
        "type": "str",
        "default": "default",
        "desc": "colour theme name",
    },
}

_SENSITIVE_HINTS = ("key", "token", "secret")


def _is_sensitive(key: str) -> bool:
    kl = (key or "").lower()
    return any(h in kl for h in _SENSITIVE_HINTS)


def _coerce_bool(raw: str):
    low = (raw or "").strip().lower()
    if low in ("true", "1", "yes", "y", "on"):
        return True
    if low in ("false", "0", "no", "n", "off"):
        return False
    return None


def _coerce_value(spec: dict, raw: str):
    """Coerce ``raw`` to the schema type; returns (ok, value, error)."""
    stype = spec["type"]
    if stype == "bool":
        v = _coerce_bool(raw)
        if v is None:
            return False, None, (
                f"expected a boolean (true/false, yes/no, on/off), got {raw!r}"
            )
        return True, v, ""
    if stype == "choice":
        choices = spec["choices"]
        if raw not in choices:
            return False, None, (
                f"expected one of {', '.join(choices)}, got {raw!r}"
            )
        return True, raw, ""
    # "str"
    return True, raw, ""


class UserConfig:
    """User settings with schema validation and atomic persistence."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or CONFIG_PATH
        self._lock = threading.Lock()
        self._data: dict = {}
        self.load()

    # -- persistence ----------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            raw = {}
        with self._lock:
            self._data = dict(raw) if isinstance(raw, dict) else {}

    def _persist_locked(self) -> None:
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=d or ".", prefix=".config-", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp, self.path)  # atomic on POSIX
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # -- accessors ------------------------------------------------------
    def valid_keys(self) -> list:
        return sorted(SCHEMA)

    def get(self, key: str):
        """Return the effective value (stored or default)."""
        key = (key or "").strip()
        if key not in SCHEMA:
            raise KeyError(f"unknown setting: {key!r}")
        with self._lock:
            if key in self._data:
                return self._data[key]
        return SCHEMA[key]["default"]

    def list_all(self) -> dict:
        """Return all settings with effective values."""
        return {k: self.get(k) for k in self.valid_keys()}

    def set(self, key: str, raw_value: str):
        """Validate, type-coerce, store and persist ``key``.

        Returns the coerced value. Raises ``KeyError`` for unknown keys,
        ``ValueError`` for sensitive keys or invalid values.
        """
        key = (key or "").strip()
        if _is_sensitive(key):
            raise ValueError(
                f"refusing to store {key!r}: API keys, tokens and secrets "
                "must never live in ~/.fullagent/config.json — use "
                "environment variables instead."
            )
        if key not in SCHEMA:
            raise KeyError(
                f"unknown setting: {key!r}\nvalid keys: "
                + ", ".join(self.valid_keys())
            )
        ok, value, err = _coerce_value(SCHEMA[key], raw_value)
        if not ok:
            raise ValueError(f"invalid value for {key!r}: {err}")
        with self._lock:
            self._data[key] = value
            self._persist_locked()
        return value

    def reset(self, key: str):
        """Reset ``key`` to its default (removes any stored override)."""
        key = (key or "").strip()
        if key not in SCHEMA:
            raise KeyError(
                f"unknown setting: {key!r}\nvalid keys: "
                + ", ".join(self.valid_keys())
            )
        with self._lock:
            self._data.pop(key, None)
            self._persist_locked()
        return SCHEMA[key]["default"]


def register(agent) -> None:
    """Attach ``agent.user_config`` (a :class:`UserConfig`)."""
    agent.user_config = UserConfig()


def _user_config_from_ui(ui) -> UserConfig:
    agent = getattr(ui, "agent", None)
    if agent is None:
        raise RuntimeError("/config needs the TUI host agent")
    cfg = getattr(agent, "user_config", None)
    if cfg is None:
        raise RuntimeError(
            "/config not initialised — configcmd.register(agent) was not run"
        )
    return cfg


def handle_config(ui, arg: str) -> None:
    """Dispatch a `/config` argument; prints via ``ui``."""
    text = (arg or "").strip()
    try:
        cfg = _user_config_from_ui(ui)
    except RuntimeError as exc:
        ui.print_error(str(exc))
        return
    parts = text.split(None, 2) if text else []
    sub = parts[0].lower() if parts else "list"

    if sub in ("list", "ls", ""):
        lines = ["settings:"]
        for key in cfg.valid_keys():
            spec = SCHEMA[key]
            val = cfg.get(key)
            desc = spec.get("desc", "")
            lines.append(f"  {key} = {val!r}   ({desc})")
        ui.print_info("\n".join(lines))
        return

    if sub == "get":
        if len(parts) < 2:
            ui.print_error("usage: /config get <key>")
            return
        key = parts[1]
        try:
            ui.print_info(f"{key} = {cfg.get(key)!r}")
        except KeyError as exc:
            ui.print_error(str(exc) + f"\nvalid keys: {', '.join(cfg.valid_keys())}")
        return

    if sub == "set":
        if len(parts) < 3:
            ui.print_error("usage: /config set <key> <value>")
            return
        key, raw_value = parts[1], parts[2]
        try:
            value = cfg.set(key, raw_value)
        except (KeyError, ValueError) as exc:
            ui.print_error(str(exc))
            return
        ui.print_info(f"✓ {key} = {value!r}")
        return

    if sub == "reset":
        if len(parts) < 2:
            ui.print_error("usage: /config reset <key>")
            return
        key = parts[1]
        try:
            value = cfg.reset(key)
        except KeyError as exc:
            ui.print_error(str(exc) + f"\nvalid keys: {', '.join(cfg.valid_keys())}")
            return
        ui.print_info(f"✓ {key} reset to {value!r}")
        return

    ui.print_error(f"unknown subcommand: {sub}\n{USAGE}")


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.configcmd`  →  PASS
# ---------------------------------------------------------------------------

class _FakeUI:
    def __init__(self, agent):
        self.agent = agent
        self.infos = []
        self.errors = []

    def print_info(self, msg):
        self.infos.append(msg)

    def print_error(self, msg):
        self.errors.append(msg)


class _FakeAgent:
    pass


def _selftest() -> None:
    import tempfile as _tf

    # temp config path — monkeypatch the path constant
    tmpdir = _tf.mkdtemp(prefix="configcmd-selftest-")
    tmppath = os.path.join(tmpdir, "config.json")
    global CONFIG_PATH
    old_path = CONFIG_PATH
    CONFIG_PATH = tmppath
    try:
        agent = _FakeAgent()
        register(agent)
        cfg = agent.user_config
        assert isinstance(cfg, UserConfig), type(cfg)
        assert cfg.path == tmppath, cfg.path

        # defaults
        assert cfg.get("default_model") == "", repr(cfg.get("default_model"))
        assert cfg.get("auto_approve") is False
        assert cfg.get("vim_mode") is False
        assert cfg.get("output_style") == "concise"
        assert cfg.get("thinking_visible") is True
        assert cfg.get("theme") == "default"

        # set with type coercion
        assert cfg.set("auto_approve", "true") is True
        assert cfg.get("auto_approve") is True
        assert cfg.set("vim_mode", "ON") is True
        assert cfg.set("thinking_visible", "no") is False
        assert cfg.set("output_style", "detailed") == "detailed"
        assert cfg.set("theme", "solarized") == "solarized"
        assert cfg.set("default_model", "deepseek-v4.1-flash-free") == "deepseek-v4.1-flash-free"

        # validation: bad bool
        try:
            cfg.set("auto_approve", "maybe")
            raise AssertionError("expected ValueError for bad bool")
        except ValueError as e:
            assert "boolean" in str(e), e

        # validation: bad choice
        try:
            cfg.set("output_style", "verbose")
            raise AssertionError("expected ValueError for bad choice")
        except ValueError as e:
            assert "concise" in str(e), e

        # unknown keys rejected with helpful message
        try:
            cfg.set("frobnicate", "x")
            raise AssertionError("expected KeyError for unknown key")
        except KeyError as e:
            assert "frobnicate" in str(e) and "valid keys" in str(e), e
        try:
            cfg.get("frobnicate")
            raise AssertionError("expected KeyError on get")
        except KeyError:
            pass

        # API-key refusal (case-insensitive)
        for nasty in ("api_key", "APIKEY", "github_token", "Token", "jwt_secret",
                      "client_secret", "secret_sauce"):
            try:
                cfg.set(nasty, "xxx")
                raise AssertionError(f"expected ValueError for {nasty}")
            except ValueError as e:
                assert "never live in" in str(e) or "environment variables" in str(e), e

        # round-trip persistence: fresh UserConfig reads the file back
        cfg2 = UserConfig()
        assert cfg2.get("auto_approve") is True
        assert cfg2.get("output_style") == "detailed"
        assert cfg2.get("theme") == "solarized"
        # file is atomic-write json, not the default config module's format
        with open(tmppath, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["auto_approve"] is True, on_disk

        # list_all
        all_ = cfg.list_all()
        assert set(all_) == set(SCHEMA), set(all_)
        assert all_["output_style"] == "detailed"

        # reset
        assert cfg.reset("auto_approve") is False
        assert cfg.get("auto_approve") is False
        cfg3 = UserConfig()
        assert cfg3.get("auto_approve") is False  # persisted removal

        # handler: /config list
        ui = _FakeUI(agent)
        handle_config(ui, "")
        assert ui.infos and "settings:" in ui.infos[0], ui.infos
        assert "output_style" in ui.infos[0], ui.infos
        assert not ui.errors, ui.errors

        ui = _FakeUI(agent)
        handle_config(ui, "list")
        assert "settings:" in ui.infos[0]

        # handler: get
        ui = _FakeUI(agent)
        handle_config(ui, "get output_style")
        assert "detailed" in ui.infos[0], ui.infos
        ui = _FakeUI(agent)
        handle_config(ui, "get bogus")
        assert ui.errors and "unknown setting" in ui.errors[0], ui.errors

        # handler: set
        ui = _FakeUI(agent)
        handle_config(ui, "set vim_mode true")
        assert "✓ vim_mode = True" in ui.infos[0], ui.infos
        assert cfg.get("vim_mode") is True

        # handler: set unknown key → helpful error
        ui = _FakeUI(agent)
        handle_config(ui, "set nope 1")
        assert ui.errors and "valid keys" in ui.errors[0], ui.errors

        # handler: set api key → refusal
        ui = _FakeUI(agent)
        handle_config(ui, "set openai_api_key sk-123")
        assert ui.errors and "environment variables" in ui.errors[0], ui.errors

        # handler: set invalid value → validation error
        ui = _FakeUI(agent)
        handle_config(ui, "set output_style verbose")
        assert ui.errors and "concise" in ui.errors[0], ui.errors

        # handler: reset
        ui = _FakeUI(agent)
        handle_config(ui, "reset vim_mode")
        assert "✓ vim_mode reset to False" in ui.infos[0], ui.infos
        assert cfg.get("vim_mode") is False

        # handler: unknown subcommand → usage
        ui = _FakeUI(agent)
        handle_config(ui, "frobnicate")
        assert ui.errors and "usage:" in ui.errors[0], ui.errors

        # handler: missing agent.user_config → clear error, no crash
        ui = _FakeUI(_FakeAgent())
        handle_config(ui, "list")
        assert ui.errors and "register" in ui.errors[0], ui.errors
    finally:
        CONFIG_PATH = old_path

    print("PASS")


if __name__ == "__main__":
    _selftest()
