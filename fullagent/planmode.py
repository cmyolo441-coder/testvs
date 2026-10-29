"""Plan mode: the agent writes a plan and waits for approval before acting.

When plan mode is active, every mutating tool (risk != "safe") is blocked
until the plan has been approved. Safe (read-only) tools and PlanWrite
itself always run.

Integrates with the permission-modes contract (fullagent.permissions):
uses try/except ImportError with a local fallback so this module works
standalone, even before fullagent.permissions exists.
"""

from __future__ import annotations

from typing import Any, Optional

from .tools import RISK_SAFE, Tool

# ---------------------------------------------------------------------------
# Permission-modes contract (built by another worker; fallback if missing)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on sibling feature's presence
    from .permissions import PermissionManager, MODES  # type: ignore
    _HAS_PERMISSIONS = True
except ImportError:
    _HAS_PERMISSIONS = False
    MODES = ("normal", "plan", "auto_accept")
    PermissionManager = None  # type: ignore

_PLAN_MODE_NAME = "plan"

# Module-level pointer to the agent currently driving plan mode, so
# should_block_mutating_tool(tool_name, risk) works without an agent arg.
_current_agent: Any = None


def _set_current_agent(agent: Any) -> None:
    global _current_agent
    _current_agent = agent


def is_plan_mode(agent: Optional[Any] = None) -> bool:
    """True when plan mode is active on the given (or current) agent."""
    ag = agent if agent is not None else _current_agent
    return bool(ag is not None and getattr(ag, "_plan_mode", False))


def enter_plan_mode(agent: Any) -> None:
    """Put the agent into plan mode: reset approval state, empty the plan."""
    agent._plan_mode = True
    agent._plan_approved = False
    agent.plan_text = None
    _set_current_agent(agent)
    # Integrate with the permission-modes contract: switch the manager
    # to the "plan" mode so the rest of the system knows what's going on.
    mgr = getattr(agent, "permissions", None) or getattr(agent, "_permissions", None)
    if mgr is not None and hasattr(mgr, "set_mode"):
        try:
            mgr.set_mode(_PLAN_MODE_NAME)
        except Exception:
            pass


def exit_plan_mode(agent: Any) -> None:
    """Leave plan mode and return to the normal permission mode."""
    agent._plan_mode = False
    agent._plan_approved = False
    mgr = getattr(agent, "permissions", None) or getattr(agent, "_permissions", None)
    if mgr is not None and hasattr(mgr, "set_mode"):
        for _name in ("default", "normal"):
            try:
                mgr.set_mode(_name)
                break
            except Exception:
                continue
    global _current_agent
    if _current_agent is agent:
        _current_agent = None


def get_plan(agent: Any) -> Optional[str]:
    """Return the current plan text, or None if no plan has been written."""
    return getattr(agent, "plan_text", None)


def approve_plan(agent: Optional[Any] = None) -> None:
    """Approve the plan: clears the gate so mutating tools may run again."""
    ag = agent if agent is not None else _current_agent
    if ag is not None:
        ag._plan_approved = True


def plan_requires_approval(agent: Optional[Any] = None) -> bool:
    """True when a plan has been written in plan mode but not yet approved."""
    ag = agent if agent is not None else _current_agent
    if ag is None:
        return False
    return (bool(getattr(ag, "_plan_mode", False))
            and getattr(ag, "plan_text", None) is not None
            and not bool(getattr(ag, "_plan_approved", False)))


def should_block_mutating_tool(tool_name: str, risk: str,
                               agent: Optional[Any] = None
                               ) -> tuple[bool, str]:
    """Plan-mode gate: (blocked, message).

    In plan mode every tool whose risk is not "safe" is blocked until
    approve_plan() is called. PlanWrite is never blocked (the agent must
    be able to revise its plan); safe tools always pass.
    """
    ag = agent if agent is not None else _current_agent
    if ag is None or not getattr(ag, "_plan_mode", False):
        return False, ""
    if tool_name == "PlanWrite" or risk == RISK_SAFE:
        return False, ""
    if getattr(ag, "_plan_approved", False):
        return False, ""
    return (True,
            "Plan mode: mutating tools are blocked until the plan is "
            f"approved. Tool '{tool_name}' was blocked. Write the plan with "
            "PlanWrite and ask the user to approve it first.")


# ---------------------------------------------------------------------------
# Tool: PlanWrite
# ---------------------------------------------------------------------------
def _make_plan_write_tool(agent: Any) -> Tool:
    def plan_write(plan: str) -> str:
        agent.plan_text = plan
        n_lines = len(plan.splitlines())
        return (f"Plan recorded ({n_lines} lines). "
                "Mutating tools remain blocked until the user approves "
                "this plan.")

    return Tool(
        "PlanWrite",
        "Write your step-by-step plan in markdown before acting. "
        "Args: plan (markdown string). Plan mode blocks mutating tools "
        "until the user approves the plan.",
        {"type": "object",
         "properties": {"plan": {"type": "string",
                                 "description": "Step-by-step plan in markdown"}},
         "required": ["plan"]},
        plan_write,
        risk=RISK_SAFE,
    )


def register(agent: Any) -> None:
    """Expose PlanWrite on the agent and initialize plan-mode state.

    Call once during agent setup, e.g. right after the tool registry is
    built in agent.py.
    """
    if not hasattr(agent, "_plan_mode"):
        agent._plan_mode = False
    if not hasattr(agent, "_plan_approved"):
        agent._plan_approved = False
    if not hasattr(agent, "plan_text"):
        agent.plan_text = None
    tools = getattr(agent, "tools", None)
    if isinstance(tools, dict):
        tools["PlanWrite"] = _make_plan_write_tool(agent)
    _set_current_agent(agent)


# ---------------------------------------------------------------------------
# Self-test: python3 -m fullagent.planmode
# ---------------------------------------------------------------------------
def _selftest() -> None:
    from types import SimpleNamespace

    from .tools import RISK_CONFIRM

    agent = SimpleNamespace(tools={
        "Read": Tool("Read", "read a file",
                     {"type": "object", "properties": {}}, lambda: "ok",
                     risk=RISK_SAFE),
        "Write": Tool("Write", "write a file",
                      {"type": "object", "properties": {}}, lambda: "ok",
                      risk=RISK_CONFIRM),
    })

    checks: list[tuple[str, bool]] = []

    def check(name: str, cond: bool) -> None:
        checks.append((name, cond))
        print(("PASS" if cond else "FAIL"), "-", name)

    register(agent)
    check("register adds PlanWrite tool", "PlanWrite" in agent.tools)
    check("plan mode off initially", not is_plan_mode(agent))
    check("no gate before plan mode",
          should_block_mutating_tool("Write", RISK_CONFIRM) == (False, ""))
    check("no plan text initially", get_plan(agent) is None)

    enter_plan_mode(agent)
    check("plan mode active", is_plan_mode(agent))

    blocked, msg = should_block_mutating_tool("Write", RISK_CONFIRM)
    check("mutating tool blocked in plan mode", blocked and "approve" in msg)
    check("safe tool not blocked in plan mode",
          should_block_mutating_tool("Read", RISK_SAFE) == (False, ""))
    check("PlanWrite itself not blocked in plan mode",
          should_block_mutating_tool("PlanWrite", RISK_SAFE) == (False, ""))
    check("nothing to approve before a plan is written",
          not plan_requires_approval())

    # LLM writes its plan via the PlanWrite tool.
    result = agent.tools["PlanWrite"].handler(
        plan="# Fix login bug\n\n1. Reproduce the bug\n2. Patch handler\n3. Run tests")
    check("PlanWrite returns confirmation", "Plan recorded" in result)
    check("plan stored on agent.plan_text",
          get_plan(agent) == "# Fix login bug\n\n1. Reproduce the bug\n2. Patch handler\n3. Run tests")
    check("approval required after plan written", plan_requires_approval())

    blocked, _ = should_block_mutating_tool("Write", RISK_CONFIRM)
    check("still blocked after plan written", blocked)

    approve_plan(agent)
    check("approval flag cleared", not plan_requires_approval())
    check("mutating tool allowed after approval",
          should_block_mutating_tool("Write", RISK_CONFIRM) == (False, ""))

    exit_plan_mode(agent)
    check("plan mode exited", not is_plan_mode(agent))
    check("fallback permissions contract carries 'plan' mode",
          "plan" in MODES)

    failed = [n for n, ok in checks if not ok]
    if failed:
        print(f"\nSELF-TEST FAILED: {len(failed)}/{len(checks)} checks failed")
        raise SystemExit(1)
    print(f"\nSELF-TEST PASS: {len(checks)}/{len(checks)} checks passed")


if __name__ == "__main__":
    _selftest()
