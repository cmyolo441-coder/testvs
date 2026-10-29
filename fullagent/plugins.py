"""User plugins: manifest-driven subprocess tools in ``~/.fullagent/plugins/``.

A plugin is a directory ``~/.fullagent/plugins/<name>/`` containing a
``manifest.json``::

    {"name": "...", "version": "...", "description": "...",
     "tools": [{"name": "...", "description": "...",
                "script": "tool.py", "timeout": 30}]}

Public API for the TUI and the agent:
    - :func:`register` -- discover plugins and register every *enabled*
      plugin tool in ``agent.tools`` as ``plugin_<plugin>_<tool>``.
    - :func:`handle_plugin` -- ``/plugin`` TUI command handler.

A plugin tool runs its script as a subprocess (no shell): argv is
``[sys.executable, script_path]``, the tool's JSON args go to stdin and
stdout is the result string. Stderr on failure becomes ``"ERROR: ..."``.
Timeouts are mandatory and enforced. Script paths are validated to stay
inside the plugin directory (``..`` and absolute escapes are rejected).

The enabled set persists in ``~/.fullagent/plugins.json`` as
``{"enabled": ["name", ...]}``; when the file is missing every
discovered plugin defaults to enabled.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Set

from .tools import Tool, RISK_CONFIRM

BASE_DIR = os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent")
PLUGINS_DIR = os.path.join(BASE_DIR, "plugins")
ENABLED_FILE = os.path.join(BASE_DIR, "plugins.json")

MANIFEST_NAME = "manifest.json"

# Plugin / tool names must be safe to embed in a tool registry key.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Sanity ceiling for tool timeouts (seconds); timeouts are mandatory.
MAX_TIMEOUT = 600

REQUIRED_MANIFEST_FIELDS = ("name", "version", "description", "tools")
REQUIRED_TOOL_FIELDS = ("name", "description", "script", "timeout")

USAGE = (
    "usage:\n"
    "  /plugin                    list plugins\n"
    "  /plugin list               list plugins\n"
    "  /plugin enable <name>      enable a plugin (restart to apply)\n"
    "  /plugin disable <name>     disable a plugin (restart to apply)"
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _safe_name(value: Any) -> Optional[str]:
    """Return a registry-safe name, or None when invalid."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if not _NAME_RE.match(v):
        return None
    return v


def _script_path(plugin_dir: str, script: Any) -> Optional[str]:
    """Resolve a tool's script, confined to ``plugin_dir`` (or None).

    Rejects absolute paths and anything that escapes the plugin dir via
    ``..`` (checked on the real path, so symlinks cannot escape either).
    """
    if not isinstance(script, str) or not script.strip():
        return None
    if os.path.isabs(script):
        return None
    base = os.path.realpath(plugin_dir)
    joined = os.path.normpath(os.path.join(base, script.strip()))
    real = os.path.realpath(joined)
    if real != base and not real.startswith(base + os.sep):
        return None
    return real


def _validate_tool(plugin_dir: str, raw: Any) -> tuple:
    """Validate one tool entry; returns (tool_dict, error)."""
    if not isinstance(raw, dict):
        return None, "tool entry must be a JSON object"
    for field in REQUIRED_TOOL_FIELDS:
        if field not in raw:
            return None, f"missing required field {field!r}"
    name = _safe_name(raw.get("name"))
    if name is None:
        return None, f"invalid tool name {raw.get('name')!r}"
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        return None, f"tool {name!r}: description must be a non-empty string"
    script_path = _script_path(plugin_dir, raw.get("script"))
    if script_path is None:
        return None, (
            f"tool {name!r}: script must be a relative path inside the "
            f"plugin directory (rejected {raw.get('script')!r})")
    timeout = raw.get("timeout")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        return None, f"tool {name!r}: timeout is mandatory (seconds)"
    timeout = float(timeout)
    if not (0 < timeout <= MAX_TIMEOUT):
        return None, (
            f"tool {name!r}: timeout must be between 0 and {MAX_TIMEOUT}s")
    return {
        "name": name,
        "description": description.strip(),
        "script_path": script_path,
        "timeout": timeout,
    }, ""


def _validate_manifest(plugin_dir: str, raw: Any) -> tuple:
    """Validate a manifest; returns (plugin_dict, errors[list])."""
    errors: List[str] = []
    if not isinstance(raw, dict):
        return None, ["manifest must be a JSON object"]
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in raw:
            errors.append(f"missing required field {field!r}")
    name = _safe_name(raw.get("name"))
    if name is None:
        errors.append(f"invalid plugin name {raw.get('name')!r}")
    version = raw.get("version")
    if not isinstance(version, str) or not version.strip():
        errors.append("version must be a non-empty string")
    description = raw.get("description")
    if not isinstance(description, str) or not description.strip():
        errors.append("description must be a non-empty string")
    raw_tools = raw.get("tools")
    if raw_tools is not None and not isinstance(raw_tools, list):
        errors.append("tools must be a list")
    if errors:
        return None, errors
    plugin = {
        "name": name,
        "version": version.strip(),
        "description": description.strip(),
        "dir": plugin_dir,
        "tools": [],
        "rejected": [],
    }
    for raw_tool in raw_tools or []:
        tool, err = _validate_tool(plugin_dir, raw_tool)
        if tool is None:
            plugin["rejected"].append(err)
        else:
            plugin["tools"].append(tool)
    return plugin, []


# ---------------------------------------------------------------------------
# Discovery + enabled set
# ---------------------------------------------------------------------------

class PluginManager:
    """Discovers plugins and persists the enabled set."""

    def __init__(self, plugins_dir: Optional[str] = None,
                 enabled_file: Optional[str] = None) -> None:
        self.plugins_dir = plugins_dir or PLUGINS_DIR
        self.enabled_file = enabled_file or ENABLED_FILE
        self._lock = threading.Lock()
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.invalid: Dict[str, str] = {}
        self._enabled: Set[str] = set()
        self.reload()

    # -- discovery ------------------------------------------------------
    def reload(self) -> None:
        plugins: Dict[str, Dict[str, Any]] = {}
        invalid: Dict[str, str] = {}
        try:
            entries = sorted(os.listdir(self.plugins_dir))
        except OSError:
            entries = []
        for entry in entries:
            plugin_dir = os.path.join(self.plugins_dir, entry)
            if not os.path.isdir(plugin_dir):
                continue
            if _safe_name(entry) is None:
                invalid[entry] = "directory name is not a valid plugin name"
                continue
            manifest_path = os.path.join(plugin_dir, MANIFEST_NAME)
            try:
                with open(manifest_path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except FileNotFoundError:
                invalid[entry] = "missing manifest.json"
                continue
            except (OSError, ValueError) as exc:
                invalid[entry] = f"unreadable manifest: {exc}"
                continue
            plugin, errors = _validate_manifest(plugin_dir, raw)
            if plugin is None:
                invalid[entry] = "; ".join(errors)
                continue
            plugins[plugin["name"]] = plugin
        with self._lock:
            self.plugins = plugins
            self.invalid = invalid
            self._enabled = self._load_enabled(set(plugins))

    def _load_enabled(self, discovered: Set[str]) -> Set[str]:
        try:
            with open(self.enabled_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return set(discovered)  # default: everything enabled
        enabled = data.get("enabled") if isinstance(data, dict) else None
        if not isinstance(enabled, list):
            return set(discovered)
        return {n for n in enabled
                if isinstance(n, str) and n in discovered}

    def enabled(self) -> Set[str]:
        with self._lock:
            return set(self._enabled)

    # -- enable / disable -------------------------------------------------
    def _save_enabled(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.enabled_file), exist_ok=True)
            with open(self.enabled_file, "w", encoding="utf-8") as fh:
                json.dump({"enabled": sorted(self._enabled)}, fh, indent=2)
        except OSError:
            pass  # persistence is best-effort; never break the TUI

    def set_enabled(self, name: str, enabled: bool) -> str:
        """Enable/disable a plugin; returns an error string or ''."""
        name = (name or "").strip()
        with self._lock:
            if name not in self.plugins and name not in self.invalid:
                return f"unknown plugin {name!r}"
            if enabled:
                self._enabled.add(name)
            else:
                self._enabled.discard(name)
            self._save_enabled()
        return ""


# The manager for module-level helpers; set by register() or on demand.
_manager: Optional[PluginManager] = None
_manager_lock = threading.Lock()


def _get_manager() -> PluginManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = PluginManager()
        return _manager


# ---------------------------------------------------------------------------
# Tool handler: run the script as a subprocess (no shell)
# ---------------------------------------------------------------------------

def _make_handler(plugin: Dict[str, Any], tool: Dict[str, Any]):
    script_path = tool["script_path"]
    timeout = tool["timeout"]
    label = f"plugin_{plugin['name']}_{tool['name']}"

    def _run(**kwargs: Any) -> str:
        if not os.path.isfile(script_path):
            return f"ERROR: plugin script missing for '{label}'"
        try:
            payload = json.dumps(kwargs)
        except (TypeError, ValueError) as exc:
            return f"ERROR: plugin '{label}' args not JSON-serializable: {exc}"
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                input=payload,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                cwd=plugin["dir"],
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: plugin '{label}' timed out after {timeout:g}s"
        except OSError as exc:
            return f"ERROR: plugin '{label}' failed to start: {exc}"
        if proc.returncode != 0:
            detail = proc.stderr.strip() or f"exit code {proc.returncode}"
            return f"ERROR: plugin '{label}' failed: {detail}"
        return proc.stdout

    return _run


def _tool_description(plugin: Dict[str, Any], tool: Dict[str, Any]) -> str:
    return (
        f"[plugin {plugin['name']} v{plugin['version']}] "
        f"{tool['description']}"
    )


def register(agent: Any) -> None:
    """Discover plugins and register enabled plugin tools on an agent.

    Each enabled tool lands in ``agent.tools`` as
    ``plugin_<plugin>_<tool>``. The manager is stored on
    ``agent.plugins``; module-level helpers (:func:`handle_plugin`)
    then serve this host. Safe to call twice (existing names win).
    """
    global _manager
    with _manager_lock:
        _manager = PluginManager()
    agent.plugins = _manager
    enabled = _manager.enabled()
    for pname, plugin in _manager.plugins.items():
        if pname not in enabled:
            continue
        for tool in plugin["tools"]:
            tool_name = f"plugin_{pname}_{tool['name']}"
            if tool_name in agent.tools:
                continue
            agent.tools[tool_name] = Tool(
                name=tool_name,
                description=_tool_description(plugin, tool),
                parameters={"type": "object"},  # open arg schema
                handler=_make_handler(plugin, tool),
                risk=RISK_CONFIRM,  # plugin scripts are user-supplied code
            )


# ---------------------------------------------------------------------------
# /plugin TUI command
# ---------------------------------------------------------------------------

def _list_text(mgr: PluginManager) -> str:
    if not mgr.plugins and not mgr.invalid:
        return "no plugins — add one under ~/.fullagent/plugins/<name>/"
    enabled = mgr.enabled()
    lines = [f"plugins ({len(mgr.plugins)}):"]
    for name in sorted(mgr.plugins):
        plugin = mgr.plugins[name]
        state = "enabled" if name in enabled else "disabled"
        n_tools = len(plugin["tools"])
        lines.append(
            f"  {name:<20} v{plugin['version']:<10} {state:<8} "
            f"{n_tools} tool{'s' if n_tools != 1 else ''}")
        if plugin["rejected"]:
            lines.append(
                f"    {'':<20} rejected: "
                + "; ".join(plugin["rejected"])[:120])
    for name in sorted(mgr.invalid):
        lines.append(f"  {name:<20} invalid: {mgr.invalid[name][:120]}")
    return "\n".join(lines)


def handle_plugin(ui, arg: str) -> None:
    """Dispatch a ``/plugin`` argument; prints via ``ui.print_info``.

    ``/plugin list`` shows name, version, enabled state and tool count;
    ``/plugin enable|disable <name>`` persists the enabled set and notes
    that a restart is needed for the change to take effect.
    """
    mgr = getattr(getattr(ui, "agent", None), "plugins", None)
    if not isinstance(mgr, PluginManager):
        mgr = _get_manager()
    text = (arg or "").strip()
    if not text or text.lower() in ("list", "ls"):
        ui.print_info(_list_text(mgr))
        return
    parts = text.split(None, 1)
    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if sub == "enable":
        err = mgr.set_enabled(rest, True)
        if err:
            ui.print_info(err + "\n" + _list_text(mgr))
        else:
            ui.print_info(
                f"✓ plugin '{rest.strip()}' enabled "
                f"(restart to apply)\n{_list_text(mgr)}")
        return
    if sub == "disable":
        err = mgr.set_enabled(rest, False)
        if err:
            ui.print_info(err + "\n" + _list_text(mgr))
        else:
            ui.print_info(
                f"✓ plugin '{rest.strip()}' disabled "
                f"(restart to apply)\n{_list_text(mgr)}")
        return
    ui.print_info(f"unknown subcommand: {sub}\n{USAGE}")


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.plugins`  →  PASS
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    import tempfile

    # Isolate plugin dirs for the self-test.
    _tmp = tempfile.mkdtemp(prefix="plugins_selftest_")
    PLUGINS_DIR = os.path.join(_tmp, "plugins")
    ENABLED_FILE = os.path.join(_tmp, "plugins.json")
    os.makedirs(os.path.join(PLUGINS_DIR, "demo"))
    os.makedirs(os.path.join(PLUGINS_DIR, "evil"))

    _echo_src = (
        "import json, sys\n"
        "args = json.load(sys.stdin)\n"
        "print(json.dumps({'echo': args, 'ok': True}))\n"
    )
    _slow_src = "import time\ntime.sleep(30)\n"
    _fail_src = "import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n"

    with open(os.path.join(PLUGINS_DIR, "demo", "echo_tool.py"), "w") as fh:
        fh.write(_echo_src)
    with open(os.path.join(PLUGINS_DIR, "demo", "slow_tool.py"), "w") as fh:
        fh.write(_slow_src)
    with open(os.path.join(PLUGINS_DIR, "demo", "fail_tool.py"), "w") as fh:
        fh.write(_fail_src)
    _demo_manifest = {
        "name": "demo", "version": "0.1.0", "description": "demo plugin",
        "tools": [
            {"name": "echo", "description": "echo args back",
             "script": "echo_tool.py", "timeout": 10},
            {"name": "slow", "description": "sleeps past its timeout",
             "script": "slow_tool.py", "timeout": 1},
            {"name": "fail", "description": "exits non-zero",
             "script": "fail_tool.py", "timeout": 10},
            {"name": "missing", "description": "no script field",
             "timeout": 5},
        ],
    }
    with open(os.path.join(PLUGINS_DIR, "demo", "manifest.json"), "w") as fh:
        json.dump(_demo_manifest, fh)

    # Path-traversal plugin: script escapes the plugin dir.
    _evil_manifest = {
        "name": "evil", "version": "1", "description": "evil plugin",
        "tools": [
            {"name": "bad", "description": "tries to escape",
             "script": "../evil.py", "timeout": 5},
            {"name": "abs", "description": "absolute escape",
             "script": "/etc/passwd", "timeout": 5},
        ],
    }
    with open(os.path.join(PLUGINS_DIR, "evil", "manifest.json"), "w") as fh:
        json.dump(_evil_manifest, fh)

    class _FakeAgent:
        def __init__(self):
            self.tools = {}

    class _FakeUI:
        def __init__(self):
            self.agent = _FakeAgent()
            self.lines: list = []

        def print_info(self, text, color=None):
            self.lines.append(str(text))

    a = _FakeAgent()
    register(a)

    # Discovery + registration
    assert "plugin_demo_echo" in a.tools, a.tools.keys()
    assert "plugin_demo_slow" in a.tools
    assert "plugin_demo_fail" in a.tools
    assert "plugin_demo_missing" not in a.tools, "tool without script"
    assert "plugin_evil_bad" not in a.tools, "path traversal not blocked"
    assert "plugin_evil_abs" not in a.tools, "absolute escape not blocked"
    assert hasattr(a, "plugins")
    mgr = a.plugins
    assert len(mgr.plugins["demo"]["tools"]) == 3
    assert len(mgr.plugins["evil"]["rejected"]) == 2
    assert "demo" in mgr.enabled()  # default: all discovered enabled

    # Subprocess round-trip: args -> stdin, stdout -> result
    out = a.tools["plugin_demo_echo"].handler(answer=42, text="hi")
    parsed = json.loads(out)
    assert parsed == {"echo": {"answer": 42, "text": "hi"}, "ok": True}, out

    # Timeout enforced
    out = a.tools["plugin_demo_slow"].handler()
    assert out.startswith("ERROR:") and "timed out" in out, out

    # Non-zero exit -> ERROR with stderr detail
    out = a.tools["plugin_demo_fail"].handler()
    assert out == "ERROR: plugin 'plugin_demo_fail' failed: boom", out

    # /plugin list via the TUI handler
    ui = _FakeUI()
    handle_plugin(ui, "")
    blob = "\n".join(ui.lines)
    assert "demo" in blob and "0.1.0" in blob and "enabled" in blob, blob
    assert "3 tools" in blob, blob
    assert "evil" in blob, blob

    # enable/disable persistence
    ui.lines.clear()
    handle_plugin(ui, "disable demo")
    blob = "\n".join(ui.lines)
    assert "disabled" in blob and "restart to apply" in blob, blob
    assert "demo" not in json.load(open(ENABLED_FILE))["enabled"]
    assert "demo" not in PluginManager(PLUGINS_DIR,
                                       ENABLED_FILE).enabled()
    handle_plugin(ui, "disable demo")  # idempotent
    handle_plugin(ui, "enable demo")
    blob = "\n".join(ui.lines)
    assert "enabled" in blob and "restart to apply" in blob, blob
    assert json.load(open(ENABLED_FILE)) == {"enabled": ["demo", "evil"]}

    # unknown plugin / unknown subcommand
    ui.lines.clear()
    handle_plugin(ui, "enable nope")
    assert "unknown plugin" in "\n".join(ui.lines)
    handle_plugin(ui, "bogus")
    assert "usage:" in "\n".join(ui.lines)

    # Manifest schema validation: missing fields -> invalid plugin
    os.makedirs(os.path.join(PLUGINS_DIR, "broken"))
    with open(os.path.join(PLUGINS_DIR, "broken", "manifest.json"),
              "w") as fh:
        json.dump({"name": "broken"}, fh)  # missing version/description/tools
    mgr2 = PluginManager(PLUGINS_DIR, ENABLED_FILE)
    assert "broken" in mgr2.invalid
    assert "missing required field" in mgr2.invalid["broken"]

    shutil.rmtree(_tmp, ignore_errors=True)
    print("plugins self-test PASSED")
