"""TUI ``/retry`` command — conversation repair after a derailed turn.

Incident (2026-10-09): a 512-second runaway turn left the conversation
full of failed tool attempts and their error messages. Every later
attempt kept going in circles because all that failure was still in the
history, and the user had no way to recover the conversation.

``/retry`` fixes that:

1. :func:`repair_conversation` (pure, TUI-free) drops everything the
   derailed turn emitted after the original user prompt — failed
   ``tool_calls``, their ``tool`` result/error messages, and partial
   assistant text — and keeps the original user prompt/goal intact.
2. :func:`handle_retry` (TUI wiring, same handler shape as
   :func:`fullagent.undocmd.handle_undo`) also drops the failed
   ``Turn`` record, so speculation and recent-tool context never
   reference the derailed attempts, then restarts the turn cleanly on
   the turn worker thread — the same path a normal submit takes, so
   ``run_turn`` appends the user message itself.

This complements :mod:`fullagent.turnresume` (resumes turns that
crashed mid-flight in a *dead* process) and :mod:`fullagent.undocmd`
(reverts *file* mutations): ``/retry`` is for the turn that finished
badly and poisoned the context, where the user wants a clean re-run
of the same prompt.

Usage from the TUI::

    /retry    drop the failed turn's attempts, keep my prompt, run again

``python3 -m fullagent.retrycmd`` runs the built-in self-test.
"""

from __future__ import annotations

import threading
from typing import Any


def repair_conversation(messages: list) -> list:
    """Drop the failed turn's tool attempts + errors, keep the prompt.

    Everything after the last ``user`` message — assistant text,
    ``tool_calls`` blocks, ``tool`` result/error messages — belongs to
    the derailed turn and is removed. Earlier history (including prior
    turns) is untouched. The original user prompt message stays, so the
    result is ready for a fresh attempt.

    Pure function: never mutates the input, never touches the agent.
    Non-dict entries are tolerated. If there is no user message at
    all, returns a copy unchanged.
    """
    src = list(messages or [])
    last_user = -1
    for i in range(len(src) - 1, -1, -1):
        try:
            if src[i].get("role") == "user":
                last_user = i
                break
        except AttributeError:
            continue
    if last_user < 0:
        return src
    return src[: last_user + 1]


def _original_prompt(agent: Any) -> str | None:
    """The user prompt of the failed turn — the retry goal."""
    for m in reversed(getattr(agent, "messages", None) or []):
        try:
            if m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        except AttributeError:
            continue
    return None


def handle_retry(ui: Any, arg: str) -> None:
    """TUI ``/retry`` handler: repair the conversation, restart the turn.

    Never raises — one repair hiccup must not kill the TUI session
    (the slash dispatcher has its own guard too; belt and suspenders).
    """
    agent = getattr(ui, "agent", None)
    if agent is None:
        ui.print_error("/retry needs the TUI host agent")
        return
    if getattr(ui, "_busy", False):
        # Never overlap two agent loops — same rule as _dispatch.
        ui.print_error("/retry: a turn is already running — "
                       "Ctrl+C to cancel it first")
        return
    prompt = _original_prompt(agent)
    if prompt is None:
        ui.print_error("/retry: no user prompt in this conversation — "
                       "nothing to retry")
        return
    try:
        messages = list(getattr(agent, "messages", None) or [])
        repaired = repair_conversation(messages)
        dropped = len(messages) - len(repaired)
        # The failed Turn record holds the same derailed tool attempts —
        # drop it so speculation / recent-tools never reference them.
        # Identity check on user_text: only pop when it is this turn.
        turns = getattr(agent, "turns", None)
        if turns:
            last = turns[-1]
            if getattr(last, "user_text", None) == prompt:
                turns.pop()
        # run_turn appends the user message itself (agent.py) — hand it
        # a clean slate whose history ends before the prompt. The
        # trailing user message was the failed turn's prompt; the fresh
        # turn re-appends it.
        agent.messages = repaired[:-1]
        try:
            log = getattr(agent, "log", None)
            if log is not None:
                log.append("turn.retry",
                           {"dropped_messages": dropped,
                            "prompt_chars": len(prompt)},
                           actor="system")
        except Exception:
            pass
        ui.print_info(
            f"↻ retrying with a cleaned context — dropped {dropped} "
            f"failed message(s), restarting from your prompt")
        # Same claim-and-start path as a normal submit (_dispatch): busy
        # on the UI thread, clear a stale cancel, run on the worker.
        # The prompt is NOT re-emitted — it is already on screen from
        # the first attempt.
        ui._busy = True
        try:
            ui._cancel_flag.clear()
        except Exception:
            pass
        threading.Thread(target=ui._run_turn_thread, args=(prompt,),
                         daemon=True).start()
    except Exception as e:  # noqa: BLE001 — the UI must survive
        try:
            ui._busy = False
        except Exception:
            pass
        ui.print_error(f"/retry failed: {e}")


def register(agent: Any) -> None:
    """Expose ``/retry`` to the feature-module loop (duck-typed).

    Retry needs no agent-side state — everything is read on demand
    from ``agent.messages`` / ``agent.turns``. This exists so the
    ``Agent._register_feature_modules`` loop picks the module up like
    every other feature module.
    """
    agent.retry_supported = True


# ---------------------------------------------------------------------------
# self-test: `python3 -m fullagent.retrycmd` → PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("retrycmd self-test")

    PROMPT = "install numpy and verify the import"

    def failed_turn():
        """The exact incident shape: prompt, failed attempts, errors,
        partial assistant text."""
        return [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "name": "run_command",
                             "args": {"cmd": "pip install numpy"}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": "Error: command timed out after 120s"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c2", "name": "read_file",
                             "args": {"path": "/nope.py"}}]},
            {"role": "tool", "tool_call_id": "c2",
             "content": "Error: file not found: /nope.py"},
            {"role": "assistant",
             "content": "It seems the commands are failing, let me try…"},
        ]

    # 1. incident shape → failed attempts + errors gone, prompt intact
    msgs = failed_turn()
    fixed = repair_conversation(msgs)
    assert fixed == [{"role": "user", "content": PROMPT}], fixed
    assert all(m.get("role") != "tool" for m in fixed)
    assert not any("Error" in str(m.get("content", "")) for m in fixed)
    assert fixed[0]["content"] == PROMPT  # user prompt intact
    print("  [ok] failed turn repaired — attempts/errors dropped, "
          "prompt intact, ready for a fresh attempt")

    # 2. earlier history is preserved, only the last turn is dropped
    hist = ([{"role": "system", "content": "sys"},
             {"role": "user", "content": "first task"},
             {"role": "assistant", "content": "done"}]
            + failed_turn())
    fixed = repair_conversation(hist)
    assert [m.get("role") for m in fixed] == [
        "system", "user", "assistant", "user"], fixed
    assert fixed[-1]["content"] == PROMPT
    print("  [ok] earlier history preserved; only the failed turn dropped")

    # 3. no user message → copy returned unchanged (never the same obj)
    weird = [{"role": "assistant", "content": "hi"}]
    out = repair_conversation(weird)
    assert out == weird and out is not weird
    assert repair_conversation([]) == []
    assert repair_conversation(None) == []
    print("  [ok] no user message → unchanged copy")

    # 4. input is never mutated; non-dict entries tolerated
    msgs = failed_turn()
    snapshot = [dict(m) for m in msgs]
    repair_conversation(msgs + ["junk", None, 42])
    assert [dict(m) for m in msgs] == snapshot
    print("  [ok] input not mutated; junk entries tolerated")

    # 5. handle_retry: happy path repairs, drops the failed Turn, and
    #    restarts the turn on the worker thread with the same prompt
    class FakeTurn:
        def __init__(self, user_text):
            self.user_text = user_text
            self.tools = ["failed-attempt"]

    class FakeLog:
        def __init__(self):
            self.events = []

        def append(self, etype, data, **kw):
            self.events.append((etype, data))

    class FakeUI:
        def __init__(self, agent):
            self.agent = agent
            self._busy = False
            self._cancel_flag = threading.Event()
            self.infos = []
            self.errors = []
            self.restarted = threading.Event()
            self.restart_prompt = None

        def print_info(self, text, *a):
            self.infos.append(text)

        def print_error(self, text, *a):
            self.errors.append(text)

        def _run_turn_thread(self, text):
            self.restart_prompt = text
            self.restarted.set()

    class FakeAgent:
        pass

    agent = FakeAgent()
    agent.messages = ([{"role": "system", "content": "sys"}]
                      + failed_turn())
    agent.turns = [FakeTurn(PROMPT)]
    agent.log = FakeLog()
    ui = FakeUI(agent)

    handle_retry(ui, "")
    assert ui.restarted.wait(5), "turn was not restarted on the worker"
    assert ui.restart_prompt == PROMPT, ui.restart_prompt
    # messages: repaired history minus the trailing user message —
    # run_turn re-appends it itself, so no duplicate.
    assert agent.messages == [{"role": "system", "content": "sys"}], \
        agent.messages
    # failed Turn record dropped (speculation never sees the attempts)
    assert agent.turns == []
    assert any(e[0] == "turn.retry" for e in agent.log.events)
    assert ui.infos and "dropped 5 failed message(s)" in ui.infos[-1], \
        ui.infos
    assert not ui.errors
    print("  [ok] handle_retry: context repaired, failed Turn dropped, "
          "turn restarted with the original prompt")

    # 6. busy → refusal, no restart, no mutation
    agent2 = FakeAgent()
    agent2.messages = failed_turn()
    agent2.turns = [FakeTurn(PROMPT)]
    ui2 = FakeUI(agent2)
    ui2._busy = True
    handle_retry(ui2, "")
    assert not ui2.restarted.is_set()
    assert ui2.errors and "already running" in ui2.errors[0]
    assert len(agent2.messages) == 6 and len(agent2.turns) == 1
    print("  [ok] busy turn → refused, state untouched")

    # 7. no agent / no prompt → clean errors, no raise
    ui3 = FakeUI(None)
    ui3.agent = None
    handle_retry(ui3, "")
    assert ui3.errors and "host agent" in ui3.errors[0]
    agent4 = FakeAgent()
    agent4.messages = [{"role": "assistant", "content": "hi"}]
    ui4 = FakeUI(agent4)
    handle_retry(ui4, "")
    assert ui4.errors and "no user prompt" in ui4.errors[0]
    assert not ui4.restarted.is_set()
    print("  [ok] no agent / no prompt → clean errors")

    # 8. a Turn record that is NOT this turn is left alone
    agent5 = FakeAgent()
    agent5.messages = ([{"role": "system", "content": "sys"}]
                       + failed_turn())
    agent5.turns = [FakeTurn("some older task")]
    ui5 = FakeUI(agent5)
    handle_retry(ui5, "")
    assert ui5.restarted.wait(5)
    assert len(agent5.turns) == 1  # foreign Turn kept
    assert agent5.turns[0].user_text == "some older task"
    print("  [ok] unrelated Turn record preserved")

    # 9. register() exposes the module to the feature loop
    register(agent)
    assert agent.retry_supported is True
    print("  [ok] register() exposes retry_supported")

    print("PASS")


if __name__ == "__main__":
    _selftest()
