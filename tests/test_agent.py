"""Agent loop: streaming, tool calls, parallel execution, trimming."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from omh.agent import Agent, estimate_cost
from omh.config import ModelProfile, ProviderConfig
from omh.providers.base import ChatMessage, Chunk, ToolCall, Usage
from omh.router import ModelRouter
from omh.session import SessionStore
from omh.tools import default_registry

try:  # discovery from the repo root imports this as tests.test_agent
    from .support import ScriptedProvider, make_config, responder
except ImportError:  # pragma: no cover - direct module loading
    from support import ScriptedProvider, make_config, responder


def make_agent(
    tmp: str,
    script,
    *,
    approval: str = "auto",
    approver=None,
    max_steps: int = 6,
    tools_enabled: bool = True,
    history_limit: int = 40,
    max_parallel_tools: int = 4,
    provider_available: bool = True,
):
    config = make_config(
        tmp,
        models=[ModelProfile("model-a", "a", label="a", input_cost=1.0, output_cost=2.0)],
        providers={"a": ProviderConfig(name="a", api_key="k")},
        approval_mode=approval,
        max_steps=max_steps,
        history_limit=history_limit,
        tools_enabled=tools_enabled,
    )
    config.router.max_parallel_tools = max_parallel_tools
    provider = ScriptedProvider("a", script)
    router = ModelRouter(config, {"a": provider} if provider_available else {})
    store = SessionStore(Path(tmp) / "sessions")
    session = store.new()
    registry = default_registry(str(tmp), approval_mode=approval)
    agent = Agent(config, router, registry, session, approver=approver)
    return agent, provider, session, registry


def tool_call(name: str, identifier: str, **arguments) -> ToolCall:
    return ToolCall(identifier, name, json.dumps(arguments))


class TextFlowTest(unittest.TestCase):
    def test_text_response_and_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, session, _ = make_agent(
                tmp, [responder("Hello", " world", model="model-a")]
            )
            events = list(agent.run("hi"))
            kinds = [event.kind for event in events]
            self.assertEqual(kinds[0], "step")
            self.assertEqual(kinds[-1], "done")
            self.assertEqual("".join(e.text for e in events if e.kind == "text"), "Hello world")
            self.assertEqual(events[-1].get("reason"), "complete")
            self.assertEqual([m.role for m in session.messages], ["user", "assistant"])
            self.assertEqual(session.messages[0].content, "hi")
            self.assertEqual(provider.calls, 1)
            self.assertIsNotNone(provider.seen[0]["tools"])
            self.assertIn("workspace", provider.seen[0]["messages"][0].content)

    def test_tools_are_offered_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, _, _ = make_agent(tmp, [responder("ok")])
            list(agent.run("hi"))
            names = [tool["function"]["name"] for tool in provider.seen[0]["tools"]]
            self.assertIn("read_file", names)
            self.assertIn("run_shell", names)

    def test_tools_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, _, _ = make_agent(
                tmp, [responder("ok")], tools_enabled=False
            )
            list(agent.run("hi"))
            self.assertIsNone(provider.seen[0]["tools"])

    def test_cost_and_usage_are_accounted(self) -> None:
        def with_usage(**_) -> None:
            yield Chunk(text="ok", model="model-a")
            yield Chunk(finish="stop", usage=Usage(1000, 500), model="model-a")

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, _ = make_agent(tmp, [with_usage])
            events = list(agent.run("hi"))
            usage_event = next(e for e in events if e.kind == "usage")
            self.assertEqual(usage_event.get("tokens"), 1500)
            self.assertAlmostEqual(usage_event.get("cost"), 0.002)
            self.assertEqual(session.usage.total, 1500)
            self.assertAlmostEqual(session.cost, 0.002)
            self.assertEqual(session.model, "model-a")

    def test_empty_response_is_not_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, _ = make_agent(tmp, [[Chunk(finish="stop")]])
            events = list(agent.run("hi"))
            self.assertEqual(events[-1].get("reason"), "complete")
            self.assertEqual([m.role for m in session.messages], ["user"])

    def test_estimate_cost_without_profile(self) -> None:
        self.assertEqual(estimate_cost(None, Usage(1, 1)), 0.0)
        self.assertEqual(estimate_cost(ModelProfile("m", "a"), None), 0.0)


class ToolFlowTest(unittest.TestCase):
    def test_tool_call_then_answer(self) -> None:
        def write_step(**_) -> None:
            yield Chunk(
                tool_calls=[tool_call("write_file", "c1", path="out.txt", content="hi\n")],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, session, _ = make_agent(
                tmp, [write_step, responder("done", model="model-a")]
            )
            events = list(agent.run("write a file"))
            self.assertTrue((Path(tmp) / "out.txt").exists())
            starts = [e for e in events if e.kind == "tool_start"]
            results = [e for e in events if e.kind == "tool_result"]
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].get("result").ok)
            self.assertEqual(events[-1].get("reason"), "complete")
            self.assertEqual(provider.calls, 2)
            roles = [message.role for message in session.messages]
            self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
            tool_message = session.messages[2]
            self.assertTrue(tool_message.content.startswith("ok"))
            self.assertEqual(tool_message.tool_call_id, "c1")

    def test_denied_tool_is_reported_to_the_model(self) -> None:
        def write_step(**_) -> None:
            yield Chunk(
                tool_calls=[tool_call("write_file", "c1", path="no.txt", content="x")],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, session, _ = make_agent(
                tmp,
                [write_step, responder("understood", model="model-a")],
                approval="ask",
                approver=lambda spec, args, summary: False,
            )
            events = list(agent.run("write"))
            result = next(e for e in events if e.kind == "tool_result").get("result")
            self.assertFalse(result.ok)
            self.assertFalse((Path(tmp) / "no.txt").exists())
            second_call_messages = provider.seen[1]["messages"]
            tool_messages = [m for m in second_call_messages if m.role == "tool"]
            self.assertEqual(len(tool_messages), 1)
            self.assertIn("denied", tool_messages[0].content)

    def test_approved_tool_runs(self) -> None:
        def write_step(**_) -> None:
            yield Chunk(
                tool_calls=[tool_call("write_file", "c1", path="yes.txt", content="ok")],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, _, _ = make_agent(
                tmp,
                [write_step, responder("done", model="model-a")],
                approval="ask",
                approver=lambda spec, args, summary: True,
            )
            list(agent.run("write"))
            self.assertTrue((Path(tmp) / "yes.txt").exists())

    def test_tools_run_in_parallel(self) -> None:
        command = "python3 -c 'import time; time.sleep(0.5)'"

        def two_tools(**_) -> None:
            yield Chunk(
                tool_calls=[
                    tool_call("run_shell", "c1", command=command),
                    tool_call("run_shell", "c2", command=command),
                ],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, _ = make_agent(
                tmp,
                [two_tools, responder("done", model="model-a")],
                max_parallel_tools=2,
            )
            started = time.monotonic()
            events = list(agent.run("run twice"))
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 0.9)
            results = [e for e in events if e.kind == "tool_result"]
            self.assertEqual(len(results), 2)
            self.assertTrue(all(e.get("result").ok for e in results))
            tool_ids = [m.tool_call_id for m in session.messages if m.role == "tool"]
            self.assertEqual(tool_ids, ["c1", "c2"])

    def test_max_steps_stops_the_loop(self) -> None:
        def list_step(**_) -> None:
            yield Chunk(
                tool_calls=[tool_call("list_dir", "c1", path=".")],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, _, _ = make_agent(tmp, [list_step], max_steps=2)
            events = list(agent.run("loop"))
            self.assertEqual(events[-1].get("reason"), "max_steps")
            self.assertEqual(provider.calls, 2)
            self.assertEqual(len([e for e in events if e.kind == "tool_result"]), 2)


class RobustnessTest(unittest.TestCase):
    def test_cancel_keeps_partial_text(self) -> None:
        def slow(**_) -> None:
            yield Chunk(text="one", model="model-a")
            yield Chunk(text="two", model="model-a")
            yield Chunk(finish="stop", model="model-a")

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, _ = make_agent(tmp, [slow])
            seen = []
            for event in agent.run("hi"):
                seen.append(event)
                if event.kind == "text":
                    agent.cancel()
            self.assertEqual("".join(e.text for e in seen if e.kind == "text"), "one")
            self.assertEqual(seen[-1].get("reason"), "cancelled")
            self.assertEqual(session.messages[-1].content, "one")

    def test_stream_error_after_partial_output(self) -> None:
        from omh.errors import ProviderError

        def partial(**_) -> None:
            yield Chunk(text="half", model="model-a")
            raise ProviderError("cut")

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, _ = make_agent(tmp, [partial])
            events = list(agent.run("hi"))
            self.assertIn("half", [e.text for e in events if e.kind == "text"])
            errors = [e for e in events if e.kind == "error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(events[-1].get("reason"), "error")
            self.assertEqual(session.messages[-1].content, "half")

    def test_no_provider_reports_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, _, _, _ = make_agent(tmp, [responder("never")], provider_available=False)
            events = list(agent.run("hi"))
            error = next(e for e in events if e.kind == "error").get("error")
            self.assertIn("no model is available", error.message)
            self.assertTrue(error.hint)

    def test_cancel_during_tools_records_cancelled_results(self) -> None:
        def list_step(**_) -> None:
            yield Chunk(
                tool_calls=[tool_call("list_dir", "c1", path=".")],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, registry = make_agent(tmp, [list_step])

            def boom(*args, **kwargs):
                raise KeyboardInterrupt

            registry.execute = boom  # type: ignore[assignment]
            events = list(agent.run("go"))
            self.assertEqual(events[-1].get("reason"), "cancelled")
            tool_messages = [m for m in session.messages if m.role == "tool"]
            self.assertEqual(len(tool_messages), 1)
            self.assertIn("cancelled", tool_messages[0].content)

    def test_zero_length_tool_id_is_generated(self) -> None:
        def list_step(**_) -> None:
            yield Chunk(
                tool_calls=[ToolCall("", "list_dir", "{}")],
                finish="tool_calls",
                model="model-a",
            )

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, session, _ = make_agent(
                tmp, [list_step, responder("done", model="model-a")]
            )
            list(agent.run("go"))
            tool_messages = [m for m in session.messages if m.role == "tool"]
            self.assertTrue(tool_messages[0].tool_call_id.startswith("call_1_"))


class HistoryTest(unittest.TestCase):
    def test_history_is_trimmed_at_a_user_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, session, _ = make_agent(
                tmp, [responder("ok", model="model-a")], history_limit=6
            )
            for index in range(12):
                session.messages.append(ChatMessage.user(f"question {index}"))
                session.messages.append(ChatMessage.assistant(f"answer {index}"))
            list(agent.run("final"))
            sent = provider.seen[0]["messages"]
            self.assertEqual(sent[0].role, "system")
            tail = sent[1:]
            self.assertLessEqual(len(tail), 6)
            self.assertEqual(tail[0].role, "user")
            self.assertEqual(tail[-1].content, "final")

    def test_incomplete_tool_history_is_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, session, _ = make_agent(
                tmp, [responder("ok", model="model-a")]
            )
            session.messages.append(ChatMessage.user("old"))
            session.messages.append(
                ChatMessage.assistant("thinking", [ToolCall("gone", "read_file", "{}")])
            )
            session.messages.append(ChatMessage.user("next"))
            list(agent.run("now"))
            sent = provider.seen[0]["messages"]
            self.assertFalse(any(m.tool_calls for m in sent))
            self.assertFalse(any(m.role == "tool" for m in sent))
            self.assertEqual([m.content for m in sent[2:]], ["thinking", "next", "now"])

    def test_complete_tool_history_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent, provider, session, _ = make_agent(
                tmp, [responder("ok", model="model-a")]
            )
            session.messages.append(ChatMessage.user("old"))
            session.messages.append(
                ChatMessage.assistant("", [ToolCall("kept", "read_file", "{}")])
            )
            session.messages.append(ChatMessage.tool("kept", "ok", "read_file"))
            session.messages.append(ChatMessage.user("next"))
            list(agent.run("now"))
            sent = provider.seen[0]["messages"]
            tools = [m for m in sent if m.role == "tool"]
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].tool_call_id, "kept")
            self.assertTrue(any(m.tool_calls for m in sent))

    def test_system_prompt_includes_skill_and_workspace(self) -> None:
        from omh.skills import Skill

        with tempfile.TemporaryDirectory() as tmp:
            agent, _, _, _ = make_agent(tmp, [responder("ok")])
            agent.skill = Skill("review", "review code", "BODY OF SKILL", "/x")
            prompt = agent.system_prompt()
            self.assertIn("Active skill: review", prompt)
            self.assertIn("BODY OF SKILL", prompt)
            self.assertIn(tmp, prompt)
            self.assertIn("approval: auto", prompt)


if __name__ == "__main__":
    unittest.main()


