"""Turn-loop guardrail proof tests (worker 3/20 deep bug hunt).

Drives the REAL Agent.run_turn end-to-end with an injected fake provider
at the chat_stream seam — no mocks of the code under test: tool parsing,
dispatch_block, _execute_tool, _finish_one, the adaptive cap, the
retry-loop detector, the stall guard, _seal_pending_tool_calls and
focus/reset state all run for real.

Proves:
  1. A guardrail firing mid-block STOPS the walk: the remaining tool calls
     in the same block never execute (before the fix, dispatch_block's
     "return True to stop early" contract was ignored and every call in
     the block ran anyway — including mutating tools).
  2. The unexecuted tail is marked blocked AND pairing-sealed, so the
     session stays usable afterwards (no orphan tool_calls ->
     invalid_request_error on the next model call).
  3. The no-progress stall guard fires on the 3rd identical error in a
     row (before the fix it fired on the 4th while its message said 3).
  4. reset() clears _focus_history (before the fix, stale stall/done
     ticks leaked into the next focus session).

Run:  python3 -m pytest tests/test_turn_loop_guards.py -q
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import json
import os
import sys
import tempfile
import time

# Isolate app state BEFORE fullagent.config computes APP_DIR at import.
_tmp_home = tempfile.mkdtemp(prefix="turnloop_home_")
os.environ["FULLAGENT_HOME"] = _tmp_home

from pathlib import Path as _Path
_REPO = str(_Path(__file__).resolve().parent.parent)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

try:
    import pytest
except ImportError:
    import importlib.util as _ilu
    _compat_spec = _ilu.spec_from_file_location(
        "pytest", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
    )
    pytest = _ilu.module_from_spec(_compat_spec)
    _sys.modules["pytest"] = pytest
    _sys.modules["tests._pytest_compat"] = pytest
    _compat_spec.loader.exec_module(pytest)


import fullagent.agent as agent_mod
from fullagent.client import StreamResult
from fullagent.config import Config
from fullagent.tools import Tool


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def _tc(tool_id: str, name: str, args: dict) -> dict:
    return {"id": tool_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class ScriptedProvider:
    """Fake chat_stream: yields canned StreamResults in order, then a
    plain-text reply forever. Counts calls."""

    def __init__(self, first=()):
        self._queue = list(first)
        self.calls = 0

    def __call__(self, provider, model, effort, messages, schemas,
                 **kwargs):
        self.calls += 1
        if self._queue:
            return self._queue.pop(0)
        return StreamResult(content="done", reasoning="",
                            tool_calls=[], usage=None, model="test")


def _callbacks():
    seen = {"token": [], "tool_call": [], "tool_update": [],
            "status": []}
    return seen, {
        "on_token": seen["token"].append,
        "on_reasoning": lambda s: None,
        "on_tool_call": seen["tool_call"].append,
        "on_tool_update": seen["tool_update"].append,
        "on_status": seen["status"].append,
        "approve": lambda tool, args: True,
        "should_cancel": lambda: False,
    }


@pytest.fixture()
def agent(monkeypatch, tmp_path):
    """A real Agent with isolated app state and a scriptable provider."""
    import fullagent.config as config_mod
    monkeypatch.setattr(config_mod, "APP_DIR", tmp_path / "app")
    monkeypatch.setattr(config_mod, "EVENT_LOG_FILE",
                        tmp_path / "app" / "eventlog.jsonl")
    monkeypatch.setattr(config_mod, "SESSIONS_DIR",
                        tmp_path / "app" / "sessions")
    monkeypatch.setattr(config_mod, "CONFIG_FILE",
                        tmp_path / "app" / "config.json")
    config_mod.ensure_dirs()
    ag = agent_mod.Agent(Config())
    scripted = ScriptedProvider()
    monkeypatch.setattr(agent_mod, "chat_stream", scripted)
    ag._scripted = scripted
    return ag


def _run(ag, text="test turn"):
    seen, cbs = _callbacks()
    turn = ag.run_turn(text, cbs["on_token"], cbs["on_reasoning"],
                       cbs["on_tool_call"], cbs["on_tool_update"],
                       cbs["on_status"], cbs["approve"],
                       should_cancel=cbs["should_cancel"])
    return turn, seen


def _tool_call_events(ag, name):
    return [e for e in ag.log.events()
            if e.type == "tool.call" and e.data.get("name") == name]


def _assert_paired(messages):
    """Every assistant tool_call id must have a tool response.

    This is the invariant providers enforce: one orphan and the next
    request dies with a pairing / invalid_request_error."""
    assistant_ids = [tc.get("id")
                     for m in messages
                     if m.get("role") == "assistant" and m.get("tool_calls")
                     for tc in m["tool_calls"]]
    tool_ids = {m.get("tool_call_id") for m in messages
                if m.get("role") == "tool" and m.get("tool_call_id")}
    orphans = [cid for cid in assistant_ids if cid not in tool_ids]
    assert not orphans, f"orphan tool_calls would break the next " \
                        f"provider call: {orphans}"


# --------------------------------------------------------------------------
# 1 + 2. guardrail stops the block; tail is blocked + sealed; session lives
# --------------------------------------------------------------------------

def test_loopdetect_stops_block_and_seals_tail(agent):
    """5 identical failing write_file calls in ONE block.

    The retry-loop detector fires on the 3rd identical failure. The 4th
    and 5th calls must never execute (pre-fix: they ran), must be marked
    blocked, and the conversation must stay paired so the session keeps
    working afterwards."""
    # write_file with a null byte in content: deterministic, byte-identical
    # ERROR on every call (the transmission-corruption guard), and
    # write_file is never parallelized, so the sequential walk is tested.
    block = [_tc(f"call_{i}", "write_file",
                 {"path": "/tmp/loopdetect_proof.txt",
                  "content": "x\x00y"})
             for i in range(5)]
    agent._scripted._queue.append(
        StreamResult(content="", reasoning="", tool_calls=block,
                     usage=None, model="test"))

    turn, _ = _run(agent, "write the file five times")

    # the guardrail fired and named itself
    assert "Loop detected" in (turn.error or ""), turn.error
    # exactly 3 executions — calls 4 and 5 never ran
    assert len(_tool_call_events(agent, "write_file")) == 3, \
        "guardrail fired but the rest of the block still executed"
    # the unexecuted tail is marked blocked, not left "running"
    skipped = [t for t in turn.tools if t.status == "blocked"]
    assert len(skipped) == 2, \
        f"expected 2 blocked tail calls, got {len(skipped)}"
    assert all(t.result.startswith("SKIPPED") for t in skipped)
    # pairing invariant holds: 3 real responses + 2 sealed
    _assert_paired(agent.messages)

    # the session is still usable — the next turn runs cleanly
    turn2, _ = _run(agent, "are you still alive")
    assert turn2.error == "", turn2.error
    assert turn2.assistant_text == "done"


# --------------------------------------------------------------------------
# 3. stall guard fires on the 3rd identical error (not the 4th)
# --------------------------------------------------------------------------

def test_stall_guard_fires_on_third_identical_error(agent):
    """Same tool, same byte-identical error, DIFFERENT args.

    The retry-loop detector keys on (tool, args, error) so it stays
    silent here; the stall guard keys on (tool, result-hash) and must
    fire exactly on the 3rd identical error — matching its message."""
    def _boom(**kwargs):
        return "ERROR: synthetic failure"

    agent.tools["boom_test"] = Tool(
        "boom_test", "fixture: always fails with a byte-identical error",
        {"type": "object",
         "properties": {"n": {"type": "integer"}}}, _boom)
    block = [_tc(f"boom_{i}", "boom_test", {"n": i}) for i in range(4)]
    agent._scripted._queue.append(
        StreamResult(content="", reasoning="", tool_calls=block,
                     usage=None, model="test"))

    turn, _ = _run(agent, "boom four times")

    assert "No progress detected" in (turn.error or ""), turn.error
    assert "3 times in a row" in turn.error
    # the 4th call never executed — the guard fired on the 3rd
    assert len(_tool_call_events(agent, "boom_test")) == 3, \
        "stall guard fired late (4th identical error still executed)"
    assert len([t for t in turn.tools if t.status == "blocked"]) == 1
    _assert_paired(agent.messages)


# --------------------------------------------------------------------------
# 4. reset() clears focus stall/done state
# --------------------------------------------------------------------------

def test_reset_clears_focus_history(agent):
    """Stale _focus_history must not leak into the next session.

    focus_continue's done-detection stops when the tail is [0.0, 0.0];
    without a clear on reset, a previous session's ticks could stop a
    brand-new focus session on its second tick."""
    agent._focus_history.extend([1.0, 0.0, 0.0])
    agent.reset()
    assert agent._focus_history == [], \
        f"reset() leaked focus state: {agent._focus_history}"


def test_focus_done_detection_needs_fresh_ticks(agent):
    """Sanity: two consecutive no-work focus ticks stop focus, using
    only the current session's history."""
    from types import SimpleNamespace

    agent._focus_history.clear()
    idle = SimpleNamespace(error="", tools=[])
    assert agent.focus_continue(idle, 5) is not None  # 1st tick: continue
    assert agent.focus_continue(idle, 4) is None      # 2nd tick: done


# --------------------------------------------------------------------------
# 5. TaskCollect/TaskOutput waits on live subagents are not tool work
# --------------------------------------------------------------------------

def test_task_collect_wait_does_not_trip_tool_time_budget(agent, monkeypatch):
    """The real end-to-end regression: two 0.2s TaskCollect waits while
    background subagents are still active must not consume the 0.05s
    tool-time budget. Before the fix, the first wait tripped the cap,
    the second collection was blocked, and the parent turn died before
    its workers could report back."""
    import fullagent.config as config_mod

    agent._ensure_session()
    assert agent.tools["TaskCollect"].background_wait is True
    assert agent.tools["TaskOutput"].background_wait is True

    monkeypatch.setattr(config_mod, "TURN_TIME_BUDGET_S", 0.05)
    monkeypatch.setattr(config_mod, "TURN_TIME_BUDGET_SLOW_S", 0.05)

    def slow_collect(*args, **kwargs):
        time.sleep(0.2)
        return ("COLLECTED 1 SUBAGENT REPORT(S) · "
                "1 still active after the wait")

    monkeypatch.setattr(agent.subagents, "collect", slow_collect)
    # Simulate the reported scenario: the collect waits time out while
    # background workers are still active.
    monkeypatch.setattr(agent.subagents, "active_count", lambda: 1)

    block = [_tc(f"collect_{i}", "TaskCollect", {"wait_seconds": 1})
             for i in range(2)]
    agent._scripted._queue.append(
        StreamResult(content="", reasoning="", tool_calls=block,
                     usage=None, model="test"))

    turn, _ = _run(agent, "wait for the running subagents")

    assert turn.error == "", turn.error
    assert [t.status for t in turn.tools if t.name == "TaskCollect"] == \
        ["done", "done"]
    assert agent._scripted.calls == 2  # one tool block + one final reply
    _assert_paired(agent.messages)


if __name__ == "__main__":
    import importlib.util as _ilu
    if "tests._pytest_compat" not in _sys.modules:
        _cs = _ilu.spec_from_file_location(
            "tests._pytest_compat", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
        )
        _cm = _ilu.module_from_spec(_cs)
        _sys.modules["tests._pytest_compat"] = _cm
        _cs.loader.exec_module(_cm)
    raise SystemExit(_sys.modules["tests._pytest_compat"].run_module_tests(_sys.modules[__name__]))
