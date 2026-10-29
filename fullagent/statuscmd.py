"""TUI ``/status`` command — session status dashboard.

Claude Code-style: one panel showing version, model/provider, session id,
uptime, context-window usage, todo progress, background tasks and the
permission mode. All integration points are optional and guarded with
``getattr`` so a missing module (todos, bgsh, permissions) can never
crash the dashboard.

Public API:
    - :func:`register` -- attach ``agent.status_info()`` (returns a dict);
      also records the session start time on the agent.
    - :func:`handle_status` -- TUI entry point ``(ui, arg) -> None``; prints
      a multi-line status panel via ``ui.print_info`` / ``ui.print_error``.

``python3 -m fullagent.statuscmd`` runs the built-in self-test.
"""

from __future__ import annotations

import time
from typing import Any, Dict

USAGE = "usage:\n  /status   show session status dashboard"


# -- helpers ---------------------------------------------------------------

def _estimate(agent) -> int:
    """Context usage percent via the existing estimator (0-100), guarded."""
    try:
        from .client import estimate_tokens
        model = getattr(agent, "model", None)
        model_id = getattr(model, "id", None)
        messages = getattr(agent, "messages", None)
        if model is None or model_id is None or messages is None:
            return 0
        used = estimate_tokens(messages, model_id)
        window = max(1, getattr(model, "context_window", 0) or 0)
        return max(0, min(100, int(used * 100 / window)))
    except Exception:
        return 0


def _todo_progress(agent):
    """Return (done, total) from the todos module state, guarded."""
    try:
        from . import todos as _todos_mod  # noqa: F401  (ensures module importable)
    except Exception:
        return 0, 0
    del _todos_mod
    try:
        tm = getattr(agent, "todo_manager", None)
        if tm is None:
            tm = getattr(agent, "todos", None)
        items = tm.todos if tm is not None and hasattr(tm, "todos") else None
        if items is None:
            get_fn = getattr(tm, "get_todos", None)
            items = get_fn() if callable(get_fn) else None
        if not items:
            return 0, 0
        total = len(items)
        done = sum(1 for t in items
                   if isinstance(t, dict) and t.get("status") == "completed")
        return done, total
    except Exception:
        return 0, 0


def _background_tasks(agent) -> int:
    """Count active bgsh shell jobs, guarded against missing integration."""
    try:
        from . import bgsh
        tasks = bgsh.list_background_tasks() or []
        return sum(1 for task in tasks
                   if isinstance(task, dict)
                   and task.get("status") not in ("done", "failed", "killed"))
    except Exception:
        return 0
def _permission_mode(agent) -> str:
    """Current permission mode string, guarded."""
    try:
        pm = getattr(agent, "permissions", None)
        if pm is None:
            return "unknown (permissions module not loaded)"
        mode = pm.mode() if callable(getattr(pm, "mode", None)) else getattr(pm, "_mode", None)
        return str(mode) if mode else "default"
    except Exception:
        return "unknown"


def _fmt_uptime(seconds: float) -> str:
    secs = max(0, int(seconds))
    h, secs = divmod(secs, 3600)
    m, s = divmod(secs, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _context_bar(pct: int, width: int = 20) -> str:
    filled = int(round(pct / 100 * width))
    return "▰" * filled + "▱" * (width - filled)


# -- public API ------------------------------------------------------------

def status_info(agent) -> Dict[str, Any]:
    """Collect the session status dict for ``agent``. Never raises."""
    try:
        from . import __version__ as _v
        version = _v
    except Exception:
        version = "?"
    model = getattr(agent, "model", None)
    model_id = getattr(model, "id", "?") or "?"
    provider = getattr(model, "provider", None)
    if not provider:
        try:
            prov_fn = getattr(agent, "provider", None)
            provider = (prov_fn().name if callable(prov_fn) else None) or "?"
        except Exception:
            provider = "?"
    start = getattr(agent, "_status_start_time", None)
    uptime = max(0.0, time.time() - start) if isinstance(start, (int, float)) else 0.0
    todo_done, todo_total = _todo_progress(agent)
    return {
        "version": version,
        "model": model_id,
        "provider": str(provider),
        "session_id": getattr(agent, "session_id", "?") or "?",
        "uptime_seconds": uptime,
        "context_pct": _estimate(agent),
        "todos_done": todo_done,
        "todos_total": todo_total,
        "background_tasks": _background_tasks(agent),
        "permission_mode": _permission_mode(agent),
    }


def handle_status(ui, arg: str) -> None:
    """TUI ``/status`` handler: print the status dashboard panel."""
    arg = (arg or "").strip()
    if arg in ("-h", "--help", "help"):
        ui.print_info(USAGE)
        return
    try:
        agent = getattr(ui, "agent", None)
        info = status_info(agent) if agent is not None else {}
        if not info:
            ui.print_error("status: no agent attached to the TUI")
            return
        pct = info["context_pct"]
        lines = [
            "fullagent status",
            f"  version:    {info['version']}",
            f"  model:      {info['model']} ({info['provider']})",
            f"  session:    {info['session_id']}",
            f"  uptime:     {_fmt_uptime(info['uptime_seconds'])}",
            f"  context:    {_context_bar(pct)} {pct}%",
            f"  todos:      {info['todos_done']}/{info['todos_total']} done",
            f"  background: {info['background_tasks']} task(s)",
            f"  perms:      {info['permission_mode']}",
        ]
        ui.print_info("\n".join(lines))
    except Exception as e:  # noqa: BLE001 — status must never crash the TUI
        ui.print_error(f"status failed: {e}")


def register(agent) -> None:
    """Wire ``agent.status_info()``; records the session start time once."""
    if not hasattr(agent, "_status_start_time"):
        try:
            agent._status_start_time = time.time()
        except Exception:
            pass
    try:
        agent.status_info = lambda: status_info(agent)  # noqa: E731
    except Exception:
        pass


# -- self-test --------------------------------------------------------------

def _selftest() -> int:
    from types import SimpleNamespace

    failures = []

    def check(name, cond):
        print(("PASS" if cond else "FAIL") + " " + name)
        if not cond:
            failures.append(name)

    # 1. bare-minimum agent: everything optional missing
    bare = SimpleNamespace()
    register(bare)
    info = bare.status_info()
    check("bare.status_info returns dict", isinstance(info, dict))
    check("bare dict has all keys",
          all(k in info for k in ("version", "model", "provider",
                                  "session_id", "uptime_seconds", "context_pct",
                                  "todos_done", "todos_total",
                                  "background_tasks", "permission_mode")))
    check("bare context pct 0", info["context_pct"] == 0)
    check("bare todos 0/0", info["todos_done"] == 0 and info["todos_total"] == 0)
    check("bare background 0", info["background_tasks"] == 0)
    check("bare perms unknown string",
          isinstance(info["permission_mode"], str) and info["permission_mode"])
    check("bare uptime small", 0 <= info["uptime_seconds"] < 5)

    # 2. full agent: all optional modules present
    class FakeModel:
        id = "deepseek-v4.1-flash-free"
        provider = "kios"
        context_window = 128000

    class FakeTodoManager:
        todos = [
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "completed"},
            {"content": "c", "status": "pending"},
        ]

    class FakePerms:
        def mode(self):
            return "plan"

    full = SimpleNamespace(model=FakeModel(),
                           session_id="abc12345",
                           messages=[{"role": "user", "content": "hi"}],
                           todo_manager=FakeTodoManager(),
                           permissions=FakePerms())
    register(full)
    info2 = full.status_info()
    check("full model id", info2["model"] == "deepseek-v4.1-flash-free")
    check("full provider", info2["provider"] == "kios")
    check("full session", info2["session_id"] == "abc12345")
    check("full todos 2/3", info2["todos_done"] == 2 and info2["todos_total"] == 3)
    check("full background defaults empty", info2["background_tasks"] == 0)
    check("full perms plan", info2["permission_mode"] == "plan")
    check("full context sane", 0 <= info2["context_pct"] <= 100)

    # 3. handler prints the panel (with and without agent)
    class FakeUI:
        def __init__(self, agent=None):
            self.agent = agent
            self.out = []
            self.errs = []

        def print_info(self, s):
            self.out.append(s)

        def print_error(self, s):
            self.errs.append(s)

    ui = FakeUI(full)
    handle_status(ui, "")
    panel = "\n".join(ui.out)
    check("handle prints version line", "version:" in panel)
    check("handle prints model line", "deepseek-v4.1-flash-free (kios)" in panel)
    check("handle prints context line", "context:" in panel)
    check("handle prints todos line", "todos:" in panel)
    check("handle prints perms line", "perms:      plan" in panel)

    ui2 = FakeUI(None)
    handle_status(ui2, "")
    check("handle no-agent -> error", bool(ui2.errs))

    ui3 = FakeUI(full)
    handle_status(ui3, "--help")
    check("handle --help usage", any("usage:" in s for s in ui3.out))

    # 4. importable through the package path used by the feature-modules tuple
    import importlib
    mod = importlib.import_module("fullagent.statuscmd")
    check("module importable as fullagent.statuscmd", mod is not None)
    check("register callable", callable(mod.register))
    check("handle_status callable", callable(mod.handle_status))

    print(("PASS" if not failures else "FAIL") +
          f" self-test ({len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
