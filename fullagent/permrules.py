"""Per-tool permission rules: allow / ask / deny lists.

Claude Code-style ``permissions.json`` rules that compose with the round-1
permission *modes* from :mod:`fullagent.permissions`
(default / plan / acceptEdits / bypassPermissions).

Config file ``~/.fullagent/permissions.json``::

    {"allow": ["read_file", "glob_files"],
     "ask":   ["run_command"],
     "deny":  ["delete_path"]}

Missing file (or unreadable file) = defaults: a small READ_ONLY set runs
freely, every other tool falls back to the *mode* verdict from
``fullagent.permissions`` (i.e. treated as ``"ask"`` here — the existing
approval flow decides).

:class:`PermissionRulesManager`::

    - check(tool_name) -> "allow" | "ask" | "deny"
    - should_ask(tool_name) -> bool   # for the approval flow
    - is_denied(tool_name) -> bool    # hard block
    - reload()                        # re-read the JSON file

Precedence: deny > ask > allow > default.
"deny" is a hard block: it wins over allow/ask rules AND over the round-1
modes (even bypassPermissions). "allow"/"ask" refine the mode verdict for
listed tools only; unlisted tools are left to the mode.

Tool-name matching is case-sensitive and also matches the exact string the
user typed; names are stripped of surrounding whitespace. Entries that are
not non-empty strings are ignored on load.

TUI integration (wired by the coordinator — do NOT import tui here):
``handle_permissions_rules(ui, arg)`` renders the current rules and applies
``allow|ask|deny <tool>`` / ``remove <tool>`` updates, persisting to the
JSON file. ``/permissions rules ...`` dispatches to it.

Usage::

    from fullagent import permrules

    permrules.register(agent)      # sets agent.perm_rules
    agent.perm_rules.check("read_file")        # "allow" (default) or rule
    agent.perm_rules.should_ask("run_command") # True/False for approval flow

Self-test: ``python3 -m fullagent.permrules`` → PASS.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional

ALLOW = "allow"
ASK = "ask"
DENY = "deny"
VERDICTS = (ALLOW, ASK, DENY)

# Read-only tools that run freely when no rule mentions them.
# Kept deliberately small: anything mutating falls back to "ask"
# (i.e. the round-1 mode verdict / approval flow decides).
READ_ONLY = frozenset({
    "read_file",
    "glob_files",
    "grep",
    "search_files",
    "web_fetch",
    "WebFetch",
    "web_search",
    "TodoRead",
    "list_dir",
    "Read",
    "Glob",
    "Grep",
    "WebSearch",
})

PERM_FILE = os.path.join(os.environ.get("FULLAGENT_HOME") or os.path.join(os.path.expanduser("~"), ".fullagent"),
                         "permissions.json")


class PermissionRulesManager:
    """Per-tool allow/ask/deny rules backed by permissions.json."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path or PERM_FILE
        self._lock = threading.Lock()
        self._rules: Dict[str, set] = {ALLOW: set(), ASK: set(),
                                      DENY: set()}
        self.reload()

    # -- config path ------------------------------------------------------
    @property
    def path(self) -> str:
        return self._path

    # -- persistence ------------------------------------------------------
    def reload(self) -> None:
        """Re-read the JSON file. Missing/unreadable file → empty rules."""
        with self._lock:
            self._rules = {ALLOW: set(), ASK: set(), DENY: set()}
            try:
                with open(self._path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                return
            if not isinstance(data, dict):
                return
            for verdict in VERDICTS:
                entries = data.get(verdict, [])
                if isinstance(entries, list):
                    self._rules[verdict] = {
                        str(e).strip() for e in entries
                        if isinstance(e, str) and str(e).strip()
                    }

    def save(self) -> None:
        """Persist current rules to the JSON file (best-effort)."""
        with self._lock:
            self._write_locked(
                {v: sorted(self._rules[v]) for v in VERDICTS})

    def _write_locked(self, data: dict) -> None:
        # Atomic replace: a crash mid-write used to truncate the file,
        # which reload() treats as "no rules" — silently dropping DENY
        # entries and reopening tools the user had blocked.
        try:
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = (f"{self._path}.{os.getpid()}."
                   f"{threading.get_ident()}.tmp")
            try:
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                os.replace(tmp, self._path)
            except OSError:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except OSError:
            pass  # persistence is best-effort; never break the flow

    # -- mutation ---------------------------------------------------------
    def set_rule(self, tool_name: str, verdict: str) -> None:
        """Put *tool_name* on one list, removing it from the others.

        Raises ValueError on unknown verdict.
        """
        if verdict not in VERDICTS:
            raise ValueError(
                "unknown verdict %r; expected one of: %s"
                % (verdict, ", ".join(VERDICTS)))
        name = str(tool_name).strip()
        if not name:
            raise ValueError("tool name must be non-empty")
        with self._lock:
            for v in VERDICTS:
                self._rules[v].discard(name)
            self._rules[verdict].add(name)
            self._persist_locked()

    def remove_rule(self, tool_name: str) -> bool:
        """Drop *tool_name* from all lists. Returns True if one existed."""
        name = str(tool_name).strip()
        with self._lock:
            found = any(name in self._rules[v] for v in VERDICTS)
            for v in VERDICTS:
                self._rules[v].discard(name)
            if found:
                self._persist_locked()
            return found

    def _persist_locked(self) -> None:
        self._write_locked({v: sorted(self._rules[v]) for v in VERDICTS})

    # -- verdicts ---------------------------------------------------------
    def check(self, tool_name: str) -> str:
        """Return "allow" | "ask" | "deny" for *tool_name*.

        Precedence: deny > ask > allow > default. Default is "allow" for
        the READ_ONLY set, "ask" for everything else (the caller falls
        back to the round-1 mode verdict / existing approval flow).
        """
        name = str(tool_name).strip()
        with self._lock:
            if name in self._rules[DENY]:
                return DENY
            if name in self._rules[ASK]:
                return ASK
            if name in self._rules[ALLOW]:
                return ALLOW
        return ALLOW if name in READ_ONLY else ASK

    def should_ask(self, tool_name: str) -> bool:
        """True when the approval flow should prompt for *tool_name*.

        "allow" → False (run freely), "ask" → True (prompt),
        "deny" → False here because denied tools are hard-blocked
        (see :meth:`is_denied`) rather than prompted for.
        """
        return self.check(tool_name) == ASK

    def is_denied(self, tool_name: str) -> bool:
        """True when *tool_name* is hard-blocked by a deny rule."""
        return self.check(tool_name) == DENY

    # -- display ----------------------------------------------------------
    def rules(self) -> Dict[str, List[str]]:
        """Current rules as sorted lists (for display)."""
        with self._lock:
            return {v: sorted(self._rules[v]) for v in VERDICTS}

    def describe(self) -> str:
        """Multi-line summary of the current rules (for the TUI)."""
        lines = [f"rules file: {self._path}"]
        rules = self.rules()
        for verdict in VERDICTS:
            items = rules[verdict]
            lines.append(f"  {verdict}: "
                         + (", ".join(items) if items else "(none)"))
        lines.append("defaults: read-only tools "
                     + ", ".join(sorted(READ_ONLY)[:6]) + "… allowed; "
                     + "everything else follows the permission mode "
                     + "(/permissions)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "PermissionRulesManager(%s)" % (self._path,)


# ---------------------------------------------------------------------------
# Agent wiring (duck-typed; never imports agent.py)
# ---------------------------------------------------------------------------

def register(agent: Any) -> PermissionRulesManager:
    """Attach a PermissionRulesManager to the agent.

    Call once during agent setup. Stores the manager on
    ``agent.perm_rules``; ``agent.perm_rules.should_ask(tool_name)`` is the
    hook the approval flow uses.
    """
    agent.perm_rules = PermissionRulesManager()
    return agent.perm_rules


# ---------------------------------------------------------------------------
# TUI handler — the coordinator wires `/permissions rules ...` to this.
# Returns text; the TUI prints it (same pattern as the TUI help handler).
# ---------------------------------------------------------------------------

def handle_permissions_rules(ui: Any, arg: str) -> str:
    """Handle `/permissions rules [allow|ask|deny|remove <tool>]`.

    *ui* is the TUI (duck-typed; only ``ui.agent`` is used). No args →
    show current rules. ``allow|ask|deny <tool>`` → set + persist the
    rule. ``remove <tool>`` → drop the tool from all lists.
    """
    agent = getattr(ui, "agent", None)
    mgr = getattr(agent, "perm_rules", None) if agent is not None else None
    if mgr is None:
        mgr = PermissionRulesManager()
        if agent is not None:
            agent.perm_rules = mgr

    parts = str(arg or "").split()
    if not parts:
        return mgr.describe()
    if len(parts) < 2:
        return ("usage: /permissions rules "
                "[allow|ask|deny|remove] <tool>")

    action, tool = parts[0].lower(), parts[1]
    if action in (ALLOW, ASK, DENY):
        try:
            mgr.set_rule(tool, action)
        except ValueError as e:
            return f"error: {e}"
        return f"✓ {tool}: {action} (saved to {mgr.path})"
    if action in ("remove", "rm", "clear", "unset"):
        if mgr.remove_rule(tool):
            return f"✓ {tool}: rule removed (saved to {mgr.path})"
        return f"{tool}: no rule to remove"
    return ("usage: /permissions rules "
            "[allow|ask|deny|remove] <tool>")


# ---------------------------------------------------------------------------
# Self-test: `python3 -m fullagent.permrules` → PASS
# ---------------------------------------------------------------------------
def _self_test() -> None:
    import tempfile

    tmp = tempfile.mkdtemp(prefix="permrules_selftest_")
    cfg = os.path.join(tmp, "permissions.json")

    class FakeAgent:
        pass

    # 1. Missing file → defaults: read-only allowed, others ask.
    mgr = PermissionRulesManager(path=cfg)
    assert mgr.check("read_file") == "allow"
    assert mgr.check("glob_files") == "allow"
    assert mgr.check("web_fetch") == "allow"
    assert mgr.check("run_command") == "ask"   # falls back to mode flow
    assert mgr.check("write_file") == "ask"
    assert mgr.check("delete_path") == "ask"
    assert mgr.should_ask("run_command") is True
    assert mgr.should_ask("read_file") is False
    assert mgr.is_denied("delete_path") is False

    # 2. Config file with all three lists; precedence deny > ask > allow.
    with open(cfg, "w", encoding="utf-8") as fh:
        json.dump({"allow": ["read_file", "run_command"],
                   "ask": ["write_file"],
                   "deny": ["run_command", "delete_path"]},
                  fh)
    mgr.reload()
    assert mgr.check("run_command") == "deny", "deny must beat allow"
    assert mgr.check("write_file") == "ask"
    assert mgr.check("delete_path") == "deny"
    assert mgr.check("read_file") == "allow"
    assert mgr.is_denied("run_command") is True
    assert mgr.should_ask("run_command") is False  # denied, not asked
    assert mgr.should_ask("write_file") is True

    # deny beats allow for a read-only tool too
    mgr.set_rule("glob_files", DENY)
    assert mgr.check("glob_files") == "deny"

    # ask beats allow
    mgr.set_rule("read_file", ASK)
    assert mgr.check("read_file") == "ask"
    # then deny beats ask
    mgr.set_rule("read_file", DENY)
    assert mgr.check("read_file") == "deny"

    # 3. Persistence round-trip: set_rule persists, fresh manager sees it.
    mgr.set_rule("my_tool", ALLOW)
    fresh = PermissionRulesManager(path=cfg)
    assert fresh.check("my_tool") == "allow"
    assert "my_tool" in fresh.rules()["allow"]
    assert "delete_path" in fresh.rules()["deny"]

    # remove_rule drops it → back to default
    assert fresh.remove_rule("my_tool") is True
    assert fresh.check("my_tool") == "ask"
    assert fresh.remove_rule("my_tool") is False

    # 4. Bad config (invalid JSON / wrong shape) → defaults, no crash.
    with open(cfg, "w", encoding="utf-8") as fh:
        fh.write("{not json")
    mgr.reload()
    assert mgr.check("run_command") == "ask"
    with open(cfg, "w", encoding="utf-8") as fh:
        json.dump(["not", "a", "dict"], fh)
    mgr.reload()
    assert mgr.check("run_command") == "ask"

    # 5. register() wiring.
    agent = FakeAgent()
    mgr2 = register(agent)
    assert isinstance(agent.perm_rules, PermissionRulesManager)
    assert mgr2 is agent.perm_rules
    assert callable(agent.perm_rules.should_ask)
    assert agent.perm_rules.should_ask("write_file") is True
    assert agent.perm_rules.should_ask("read_file") is False

    # 6. Handler (duck-typed UI).
    class FakeUI:
        def __init__(self, a):
            self.agent = a

    ui = FakeUI(FakeAgent())
    out = handle_permissions_rules(ui, "")
    assert "rules file:" in out and "allow:" in out and "deny:" in out, out
    out = handle_permissions_rules(ui, "deny delete_path")
    assert "delete_path" in out and "deny" in out, out
    assert ui.agent.perm_rules.check("delete_path") == "deny"
    out = handle_permissions_rules(ui, "allow run_command")
    assert ui.agent.perm_rules.check("run_command") == "allow"
    out = handle_permissions_rules(ui, "ask run_command")
    assert ui.agent.perm_rules.check("run_command") == "ask"
    out = handle_permissions_rules(ui, "remove run_command")
    assert ui.agent.perm_rules.check("run_command") == "ask"  # default
    out = handle_permissions_rules(ui, "bogus")
    assert "usage" in out
    out = handle_permissions_rules(ui, "bogus foo")
    assert "usage" in out
    # handler persists: new manager for same path sees the rule
    path = ui.agent.perm_rules.path
    assert PermissionRulesManager(path=path).check("delete_path") == "deny"
    # handler works even with a UI lacking agent.perm_rules (lazy init)
    out = handle_permissions_rules(FakeUI(FakeAgent()), "")
    assert "rules file:" in out

    # 7. set_rule validation.
    try:
        mgr2.set_rule("x", "maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("set_rule with bad verdict did not raise")
    try:
        mgr2.set_rule("   ", "allow")
    except ValueError:
        pass
    else:
        raise AssertionError("set_rule with empty name did not raise")

    print("PASS")


if __name__ == "__main__":
    _self_test()
