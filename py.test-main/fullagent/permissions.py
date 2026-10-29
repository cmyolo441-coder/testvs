"""Permission modes: default / plan / acceptEdits / bypassPermissions.

Controls whether a tool call may proceed automatically, and whether the
agent should ask the user for approval first.

Semantics of ``allows(tool_name, risk) -> bool``:

* ``default`` — read-only ("safe") tools run freely; mutating tools
  (``risk != "safe"``) need approval (returns ``False``).
* ``plan`` — all non-safe tools are blocked (returns ``False``) until the
  plan is approved. The *caller* (planmode.py / TUI) decides the UX:
  blocked-for-approval vs. blocked-until-plan-approved.
* ``acceptEdits`` — file-edit tools are auto-allowed; shell commands still
  need approval; safe tools always allowed.
* ``bypassPermissions`` — everything allowed, no prompts.

``allows`` returns a plain boolean only; the caller decides whether
``False`` means "ask for approval" or "block outright".

Usage::

    from fullagent.permissions import PermissionManager, MODES
    from fullagent import permissions

    perms = PermissionManager()          # starts in "default"
    perms.set_mode("plan")
    perms.allows("write_file", risk="confirm")  # False -> blocked/ask

    permissions.register(agent)          # sets agent.permissions
"""

from __future__ import annotations

from typing import Any

# The four Claude-Code-style permission modes, in order of strictness.
MODES = ("default", "plan", "acceptEdits", "bypassPermissions")

# File-editing tools: auto-allowed under "acceptEdits".
EDIT_TOOLS = {
    "write_file",
    "edit_file",
    "apply_patch",
    "NotebookEdit",
    "NotebookInsert",
    "NotebookDelete",
    "MultiEdit",
}

# Shell-ish tools: never auto-allowed, even under "acceptEdits".
SHELL_TOOLS = {"run_command", "live_shell", "Bash", "bash"}

SAFE = "safe"  # matches fullagent.tools.RISK_SAFE


class PermissionManager:
    """Decides whether a tool call may proceed under the current mode."""

    def __init__(self, mode: str = "default") -> None:
        self._mode = "default"
        self.set_mode(mode)

    # -- mode -----------------------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, name: str) -> None:
        """Switch permission mode. Raises ValueError on unknown mode."""
        if name not in MODES:
            raise ValueError(
                "unknown permission mode %r; expected one of: %s"
                % (name, ", ".join(MODES))
            )
        self._mode = name

    # -- verdict ----------------------------------------------------------------
    def allows(self, tool_name: str, risk: str = SAFE) -> bool:
        """Return True if the tool may run without approval in the current mode.

        ``risk`` mirrors fullagent.tools.RISK_SAFE / RISK_CONFIRM:
        "safe" (read-only) vs. anything else (mutating / needs attention).
        A ``False`` result means the caller should either ask the user for
        approval or block the call, depending on the caller's UX.
        """
        mode = self._mode
        if mode == "bypassPermissions":
            return True
        if risk == SAFE:
            return True
        if mode == "default":
            return False  # mutating tools need approval
        if mode == "plan":
            return False  # all non-safe tools blocked
        if mode == "acceptEdits":
            if tool_name in EDIT_TOOLS:
                return True
            return False  # shell + other mutating tools need approval
        return False  # defensive: unknown mode never allows

    # -- display -----------------------------------------------------------------
    def describe(self) -> str:
        """One-line summary of the current permission mode (for the TUI)."""
        summaries = {
            "default": "default: safe tools run freely, mutating tools need approval",
            "plan": "plan: all non-safe tools blocked until the plan is approved",
            "acceptEdits": "acceptEdits: file edits auto-allowed, shell commands need approval",
            "bypassPermissions": "bypassPermissions: all tools allowed, no approval prompts",
        }
        return summaries.get(self._mode, "unknown mode")

    def __repr__(self) -> str:
        return "PermissionManager(mode=%r)" % (self._mode,)


def register(agent: Any) -> PermissionManager:
    """Attach a PermissionManager to the agent.

    Call once during agent setup in agent.py, right after the agent is
    constructed. Returns the new manager.
    """
    agent.permissions = PermissionManager()
    return agent.permissions


# ---------------------------------------------------------------------------
# Self-test: `python3 -m fullagent.permissions` → PASS
# ---------------------------------------------------------------------------
def _self_test() -> None:
    pm = PermissionManager()
    assert pm.mode == "default"

    # default: safe allowed, mutating needs approval
    assert pm.allows("read_file", "safe") is True
    assert pm.allows("write_file", "confirm") is False
    assert pm.allows("run_command", "confirm") is False

    pm.set_mode("plan")
    assert pm.mode == "plan"
    assert pm.allows("read_file", "safe") is True
    assert pm.allows("write_file", "confirm") is False
    assert pm.allows("run_command", "confirm") is False

    pm.set_mode("acceptEdits")
    assert pm.mode == "acceptEdits"
    assert pm.allows("read_file", "safe") is True
    for tool in EDIT_TOOLS:
        assert pm.allows(tool, "confirm") is True, tool
    assert pm.allows("run_command", "confirm") is False
    assert pm.allows("live_shell", "confirm") is False
    assert pm.allows("delete_path", "confirm") is False

    pm.set_mode("bypassPermissions")
    assert pm.mode == "bypassPermissions"
    assert pm.allows("read_file", "safe") is True
    assert pm.allows("write_file", "confirm") is True
    assert pm.allows("run_command", "confirm") is True
    assert pm.allows("anything_at_all", "confirm") is True

    # invalid mode raises
    try:
        pm.set_mode("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("set_mode('nope') did not raise ValueError")
    assert pm.mode == "bypassPermissions"  # failed set must not change mode

    try:
        PermissionManager("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("constructor with bad mode did not raise")

    # register() wiring
    class FakeAgent:
        pass

    agent = FakeAgent()
    mgr = register(agent)
    assert isinstance(agent.permissions, PermissionManager)
    assert mgr is agent.permissions
    assert agent.permissions.mode == "default"

    # describe() is a one-line str for every mode
    for mode in MODES:
        pm.set_mode(mode)
        d = pm.describe()
        assert isinstance(d, str) and len(d.splitlines()) == 1, mode

    print("PASS")


if __name__ == "__main__":
    _self_test()
