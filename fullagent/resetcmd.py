"""TUI ``/reset`` command — the escape hatch for a tangled conversation.

Incident (2026-10-09): a 512-second runaway turn left the conversation
full of failed tool attempts and their error messages. Every later
attempt kept going in circles because all that failure was still in the
history, and the user had no way to recover the conversation.

``/retry`` repairs the *last* turn. ``/reset`` is the bigger hammer for
when the whole history is too tangled to salvage — after caps, cancels,
and repeated errors, every subsequent call keeps failing because the
context itself is poisoned. ``/reset`` drops the conversation and its
per-turn caches entirely, but keeps the durable state:

Kept:
  * the session — ``session_id``, the event ``log``, config, model
  * the goal contract — ``agent.goal`` (GoalContract) untouched
  * the todos — ``agent.todo_manager`` untouched
  * the system prompt — the sealed mastermind prompt is reseated
  * one auto-summary message of what was accomplished so far (built
    from :mod:`fullagent.compact`'s heuristic summarizer, compressed
    to a single paragraph), so context isn't fully lost

Cleared:
  * ``agent.messages`` — rebuilt as [system, summary]
  * ``agent.turns`` — failed Turn records can't poison speculation
  * tool dedup cache (:func:`fullagent.tooldedup.reset`)
  * file-stat baselines (:func:`fullagent.filestat.clear`)
  * stall/focus history (``agent._focus_history``) — the per-turn
    stall counter and loopdetect detector are already turn-local, but
    the session-long focus distance history is a stale wedge signal

Usage from the TUI::

    /reset    drop history + per-turn caches, keep session/goal/todos

``python3 -m fullagent.resetcmd`` runs the built-in self-test.
"""

from __future__ import annotations

from typing import Any


#: Max bullet lines condensed into the one-paragraph auto-summary.
MAX_SUMMARY_BULLETS = 5
#: Hard cap on the summary paragraph length (chars).
MAX_SUMMARY_CHARS = 600


def _role(msg: Any) -> str:
    try:
        return msg.get("role", "") or ""
    except AttributeError:
        return ""


def build_reset_summary(messages: list) -> str:
    """Compress ``messages`` into one paragraph of what was accomplished.

    Reuses :mod:`fullagent.compact`'s heuristic decision-line extraction
    (user requests + "Ran <tool>" lines + decision-ish prose) over *all*
    non-system messages — unlike :func:`~fullagent.compact.do_compact`,
    which keeps a verbatim tail, ``/reset`` drops everything, so the
    summary covers the whole conversation. Never raises; returns an
    empty string when there is nothing worth summarizing.
    """
    try:
        from . import compact
    except Exception:  # noqa: BLE001 — a missing module must not kill /reset
        return ""
    try:
        non_system = [m for m in (messages or [])
                      if isinstance(m, dict) and _role(m) != "system"]
        lines = compact._extract_decision_lines(non_system)
    except Exception:  # noqa: BLE001 — degrade to an empty summary
        return ""
    bullets = [ln for ln in lines[:MAX_SUMMARY_BULLETS] if ln]
    paragraph = "; ".join(bullets).strip()
    if len(paragraph) > MAX_SUMMARY_CHARS:
        cut = paragraph[:MAX_SUMMARY_CHARS].rsplit(";", 1)[0].strip()
        paragraph = (cut or paragraph[:MAX_SUMMARY_CHARS].strip()) + "…"
    return paragraph


def reset_conversation(agent: Any) -> dict:
    """Clear history + per-turn caches, keep session/goal/todos + summary.

    Returns ``{"dropped": N, "summary": str, "messages": M}`` where ``N``
    is the number of messages dropped and ``M`` the size of the rebuilt
    message list (system + summary, if any). Never raises — a failed
    partial reset is worse than no reset.
    """
    from . import tooldedup

    stats: dict = {"dropped": 0, "summary": "", "messages": 0}
    try:
        messages = list(getattr(agent, "messages", None) or [])
    except Exception:  # noqa: BLE001
        messages = []
    stats["dropped"] = len(messages)

    try:
        stats["summary"] = build_reset_summary(messages)
    except Exception:  # noqa: BLE001
        stats["summary"] = ""

    # Keep: the system prompt message (reseated below), the session,
    # the goal contract, and todos are simply not touched.
    system = next((m for m in messages if _role(m) == "system"), None)

    new_messages: list = []
    if isinstance(system, dict):
        new_messages.append(system)
    try:
        from . import compact

        if stats["summary"]:
            new_messages.append({
                "role": "user",
                "content": compact.SUMMARY_MARKER + "\n" + stats["summary"],
            })
    except Exception:  # noqa: BLE001
        if stats["summary"]:
            new_messages.append({
                "role": "user",
                "content": "SUMMARY: " + stats["summary"],
            })
    try:
        agent.messages = new_messages
    except Exception:  # noqa: BLE001
        pass

    # Refresh the sealed system prompt through the mastermind gate
    # (best effort — duck-typed agents may not have it).
    try:
        reseat = getattr(agent, "_reseat_system_prompt", None)
        if callable(reseat):
            reseat()
    except Exception:  # noqa: BLE001
        pass

    # Clear: failed turn records (speculation never references them).
    try:
        agent.turns = []
    except Exception:  # noqa: BLE001
        pass
    try:
        agent._turn_user_text = ""
    except Exception:  # noqa: BLE001
        pass

    # Clear: per-turn caches.
    try:
        tooldedup.reset(agent)
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import filestat

        filestat.clear(agent)
    except Exception:  # noqa: BLE001
        pass
    # stall/focus history: session-long wedge signal — start clean
    try:
        fh = getattr(agent, "_focus_history", None)
        if fh is not None:
            fh.clear()
    except Exception:  # noqa: BLE001
        pass

    try:
        stats["messages"] = len(getattr(agent, "messages", None) or [])
    except Exception:  # noqa: BLE001
        pass
    try:
        log = getattr(agent, "log", None)
        if log is not None:
            log.append("turn.reset",
                       {"dropped_messages": stats["dropped"],
                        "summary_chars": len(stats["summary"])},
                       actor="system")
    except Exception:  # noqa: BLE001
        pass
    return stats


def handle_reset(ui: Any, arg: str) -> None:
    """TUI ``/reset`` handler: clear history, keep the durable state.

    Never raises — one reset hiccup must not kill the TUI session.
    """
    agent = getattr(ui, "agent", None)
    if agent is None:
        ui.print_error("/reset needs the TUI host agent")
        return
    if getattr(ui, "_busy", False):
        # Never overlap two agent loops — same rule as _dispatch.
        ui.print_error("/reset: a turn is already running — "
                       "Ctrl+C to cancel it first")
        return
    try:
        stats = reset_conversation(agent)
        ui.print_info(
            f"History cleared ({stats['dropped']} messages dropped). "
            "Goal and todos kept."
        )
    except Exception as e:  # noqa: BLE001 — the UI must survive
        ui.print_error(f"/reset failed: {e}")


def register(agent: Any) -> None:
    """Expose ``/reset`` to the feature-module loop (duck-typed)."""
    agent.reset_supported = True


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.resetcmd` → PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("resetcmd self-test")

    class FakeLog:
        def __init__(self):
            self.events = []

        def append(self, etype, data, **kw):
            self.events.append((etype, data))

    class FakeGoal:
        def __init__(self):
            self.statement = "build the widget"

    class FakeTodos:
        def __init__(self):
            self.items = [{"text": "write code", "status": "done"},
                          {"text": "ship it", "status": "pending"}]

    class FakeAgent:
        pass

    def tangled_agent():
        agent = FakeAgent()
        agent.session_id = "abc123"
        agent.messages = [
            {"role": "system", "content": "sealed system prompt"},
            {"role": "user", "content": "build the widget"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1",
                             "function": {"name": "run_command",
                                          "arguments": '{"cmd": "make widget"}'}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "Error: build failed"},
            {"role": "assistant", "content": "I built the widget successfully"},
            {"role": "user", "content": "now add tests"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c2",
                             "function": {"name": "write_file",
                                          "arguments": '{"path": "test_widget.py"}'}}]},
            {"role": "tool", "tool_call_id": "c2",
             "content": "Error: permission denied"},
        ]
        agent.turns = [object(), object()]
        agent._turn_user_text = "stale"
        agent.goal = FakeGoal()
        agent.todo_manager = FakeTodos()
        agent.log = FakeLog()
        agent._focus_history = [0.1, 0.2, 0.1]
        # tool dedup cache (per-turn cache)
        from . import tooldedup

        tooldedup.store(agent, "run_command", {"cmd": "make widget"},
                        "Error: build failed")
        # file-stat baselines (per-turn cache)
        from . import filestat

        filestat.register(agent)
        filestat.note_read(agent, __file__)
        return agent

    # 1. tangled history → cleared, one-paragraph summary kept
    agent = tangled_agent()
    stats = reset_conversation(agent)
    assert stats["dropped"] == 8, stats
    roles = [m.get("role") for m in agent.messages]
    assert roles[0] == "system", roles
    assert agent.messages[0]["content"] == "sealed system prompt"
    summaries = [m for m in agent.messages
                 if "Previous conversation summary:" in str(m.get("content"))]
    assert len(summaries) == 1, agent.messages
    para = summaries[0]["content"].split("Previous conversation summary:",
                                         1)[1].strip()
    assert para and "\n" not in para, para  # single paragraph
    assert "build the widget" in para, para  # user intent survived
    assert "Ran run_command" in para or "ran" in para.lower(), para
    assert stats["messages"] == len(agent.messages)
    print("  [ok] history cleared (8 dropped), one-paragraph auto-summary kept")

    # 2. kept: session, goal contract, todos untouched
    assert agent.session_id == "abc123"
    assert isinstance(agent.goal, FakeGoal)
    assert agent.goal.statement == "build the widget"
    assert isinstance(agent.todo_manager, FakeTodos)
    assert len(agent.todo_manager.items) == 2
    assert any(e[0] == "turn.reset" for e in agent.log.events)
    assert agent.log.events[-1][1]["dropped_messages"] == 8
    print("  [ok] session, goal contract, todos kept; turn.reset logged")

    # 3. cleared: turns, per-turn caches, stall/focus history
    assert agent.turns == []
    assert agent._turn_user_text == ""
    from . import tooldedup, filestat

    assert tooldedup._cache(agent) == {}, tooldedup._cache(agent)
    assert filestat._store_for(agent) == {}, filestat._store_for(agent)
    assert agent._focus_history == []
    print("  [ok] turns + dedup cache + file-stat baselines + focus history "
          "cleared")

    # 4. empty history → no crash, still prints and keeps system message
    agent2 = FakeAgent()
    agent2.messages = [{"role": "system", "content": "sys"}]
    agent2.goal = FakeGoal()
    agent2.todo_manager = FakeTodos()
    agent2.log = FakeLog()
    s2 = reset_conversation(agent2)
    assert s2["dropped"] == 1
    assert s2["summary"] == ""
    assert [m["role"] for m in agent2.messages] == ["system"]
    print("  [ok] empty/trivial history → clean no-op reset")

    # 5. handle_reset: happy path prints the exact confirmation
    class FakeUI:
        def __init__(self, agent):
            self.agent = agent
            self._busy = False
            self.infos = []
            self.errors = []

        def print_info(self, text, *a):
            self.infos.append(text)

        def print_error(self, text, *a):
            self.errors.append(text)

    ui = FakeUI(tangled_agent())
    handle_reset(ui, "")
    assert not ui.errors, ui.errors
    assert ui.infos == ["History cleared (8 messages dropped). "
                        "Goal and todos kept."], ui.infos
    print("  [ok] handle_reset prints the exact confirmation")

    # 6. busy → refused, state untouched
    agent3 = tangled_agent()
    ui3 = FakeUI(agent3)
    ui3._busy = True
    handle_reset(ui3, "")
    assert ui3.errors and "already running" in ui3.errors[0]
    assert len(agent3.messages) == 8
    assert agent3.turns != []
    print("  [ok] busy turn → refused, state untouched")

    # 7. no agent → clean error, no raise
    ui4 = FakeUI(None)
    ui4.agent = None
    handle_reset(ui4, "")
    assert ui4.errors and "host agent" in ui4.errors[0]
    print("  [ok] no agent → clean error")

    # 8. junk input to build_reset_summary never raises
    assert build_reset_summary(None) == ""
    assert build_reset_summary(["junk", None, 42]) == ""
    print("  [ok] build_reset_summary tolerates junk")

    # 9. register() exposes the module to the feature loop
    register(FakeAgent())
    print("  [ok] register() exposes reset_supported")

    print("PASS")


if __name__ == "__main__":
    _selftest()
