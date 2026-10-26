"""Provider dialects, SSE decoding and error mapping."""

from __future__ import annotations

import io
import json
import unittest
from typing import Any, Dict, List
from unittest import mock
from urllib.error import HTTPError

from omh.config import ProviderConfig
from omh.errors import AuthError, ProviderError, RateLimitError
from omh.providers.anthropic import AnthropicProvider, convert_messages, convert_tools
from omh.providers.base import ChatMessage, ToolCall, Usage
from omh.providers.http import map_http_error, post_sse, sse_json
from omh.providers.openai import OpenAIProvider, _Accumulator


def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(
        ProviderConfig(
            name="test",
            base_url="https://example.test/v1",
            api_key="secret",
        )
    )


class BaseTypesTest(unittest.TestCase):
    def test_tool_call_roundtrip(self) -> None:
        call = ToolCall("id1", "read_file", '{"path": "x"}')
        self.assertEqual(call.parsed(), {"path": "x"})
        self.assertEqual(ToolCall.from_dict(call.to_dict()).name, "read_file")

    def test_bad_tool_arguments_are_reported(self) -> None:
        with self.assertRaises(ValueError):
            ToolCall("id", "x", "{bad").parsed()
        with self.assertRaises(ValueError):
            ToolCall("id", "x", "[]").parsed()
        self.assertEqual(ToolCall("id", "x", "").parsed(), {})

    def test_message_serialisation(self) -> None:
        assistant = ChatMessage.assistant("hi", [ToolCall("1", "f", "{}")])
        raw = assistant.to_dict()
        self.assertEqual(raw["role"], "assistant")
        self.assertEqual(raw["tool_calls"][0]["function"]["name"], "f")
        tool = ChatMessage.tool("1", "ok", "f").to_dict()
        self.assertEqual(tool["tool_call_id"], "1")
        self.assertEqual(ChatMessage.from_dict(raw).tool_calls[0].id, "1")

    def test_usage_add(self) -> None:
        usage = Usage(2, 3)
        usage.add(Usage(1, 1))
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.total), (3, 4, 7))
        usage.add(None)
        self.assertEqual(usage.total, 7)


class OpenAITest(unittest.TestCase):
    def test_text_stream_and_usage(self) -> None:
        frames = [
            (None, json.dumps({"model": "m1", "choices": [{"delta": {"content": "Hel"}}]})),
            (None, json.dumps({"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]})),
            (None, json.dumps({"usage": {"prompt_tokens": 3, "completion_tokens": 2}})),
        ]
        with mock.patch("omh.providers.openai.post_sse", return_value=iter(frames)) as sent:
            chunks = list(
                openai_provider().stream(
                    model="m1", messages=[ChatMessage.user("hi")], tools=None
                )
            )
        self.assertEqual("".join(chunk.text for chunk in chunks), "Hello")
        self.assertEqual(chunks[-1].finish, "stop")
        self.assertEqual(chunks[-1].usage.total, 5)
        self.assertEqual(chunks[0].model, "m1")
        payload = sent.call_args.args[1]
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(sent.call_args.args[2]["Authorization"], "Bearer secret")

    def test_streamed_tool_calls_accumulate(self) -> None:
        frames = [
            (None, json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_a", "function": {"name": "read_file", "arguments": '{"pa'}}
            ]}}]})),
            (None, json.dumps({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'th": "x"}'}}
            ]}, "finish_reason": "tool_calls"}]})),
        ]
        with mock.patch("omh.providers.openai.post_sse", return_value=iter(frames)):
            chunks = list(
                openai_provider().stream(
                    model="m",
                    messages=[ChatMessage.user("go")],
                    tools=[{"type": "function", "function": {"name": "read_file"}}],
                )
            )
        final = chunks[-1]
        self.assertEqual(final.finish, "tool_calls")
        self.assertEqual(len(final.tool_calls or []), 1)
        call = final.tool_calls[0]
        self.assertEqual(call.name, "read_file")
        self.assertEqual(call.parsed(), {"path": "x"})

    def test_stream_options_400_falls_back(self) -> None:
        frames = [(None, json.dumps({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}))]
        error = ProviderError("HTTP 400: stream_options unsupported", status=400)
        with mock.patch(
            "omh.providers.openai.post_sse", side_effect=[error, iter(frames)]
        ) as sent:
            chunks = list(
                openai_provider().stream(model="m", messages=[ChatMessage.user("x")])
            )
        self.assertEqual("".join(chunk.text for chunk in chunks), "ok")
        self.assertNotIn("stream_options", sent.call_args.args[1])

    def test_provider_error_frame_is_raised(self) -> None:
        frames = [(None, json.dumps({"error": {"message": "bad model"}}))]
        with mock.patch("omh.providers.openai.post_sse", return_value=iter(frames)):
            with self.assertRaises(ProviderError):
                list(openai_provider().stream(model="m", messages=[]))

    def test_accumulator_ignores_unfinished_calls(self) -> None:
        acc = _Accumulator()
        acc.add({"index": 0, "function": {"arguments": "{}"}})
        self.assertEqual(acc.calls(), [])


class AnthropicTest(unittest.TestCase):
    def test_message_conversion_and_merging(self) -> None:
        messages = [
            ChatMessage.system("sys"),
            ChatMessage.user("hi"),
            ChatMessage.assistant("yo", [ToolCall("t1", "grep", '{"pattern": "x"}')]),
            ChatMessage.tool("t1", "result", "grep"),
            ChatMessage.user("again"),
        ]
        system, turns = convert_messages(messages)
        self.assertEqual(system, "sys")
        self.assertEqual(turns[0]["role"], "user")
        self.assertEqual(turns[1]["role"], "assistant")
        self.assertEqual(turns[1]["content"][0]["text"], "yo")
        self.assertEqual(turns[1]["content"][1]["type"], "tool_use")
        self.assertEqual(turns[1]["content"][1]["input"], {"pattern": "x"})
        self.assertEqual(len(turns), 3)
        self.assertEqual(turns[2]["role"], "user")
        self.assertEqual(len(turns[2]["content"]), 2)
        self.assertEqual(turns[2]["content"][0]["type"], "tool_result")

    def test_tool_conversion(self) -> None:
        tools = convert_tools(
            [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
        )
        self.assertEqual(tools[0]["name"], "f")
        self.assertIn("input_schema", tools[0])

    def test_streaming_text_and_tool_use(self) -> None:
        frames = [
            ("message_start", json.dumps({"type": "message_start", "message": {"model": "claude", "usage": {"input_tokens": 4, "output_tokens": 1}}})),
            ("content_block_start", json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})),
            ("content_block_delta", json.dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}})),
            ("content_block_start", json.dumps({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tool1", "name": "grep", "input": {}}})),
            ("content_block_delta", json.dumps({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"pattern": "x"}'}})),
            ("message_delta", json.dumps({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}})),
            ("message_stop", json.dumps({"type": "message_stop"})),
        ]
        provider = AnthropicProvider(
            ProviderConfig(
                name="anthropic",
                kind="anthropic",
                base_url="https://api.anthropic.com",
                api_key="k",
            )
        )
        with mock.patch("omh.providers.anthropic.post_sse", return_value=iter(frames)) as sent:
            chunks = list(provider.stream(model="claude", messages=[ChatMessage.user("hi")]))
        self.assertEqual("".join(chunk.text for chunk in chunks), "Hi")
        final = chunks[-1]
        self.assertEqual(final.finish, "tool_calls")
        self.assertEqual(final.tool_calls[0].parsed(), {"pattern": "x"})
        self.assertEqual(final.usage.prompt_tokens, 4)
        self.assertEqual(final.usage.completion_tokens, 5)
        payload = sent.call_args.args[1]
        self.assertTrue(payload["stream"])
        self.assertEqual(sent.call_args.args[2]["x-api-key"], "k")
        self.assertEqual(sent.call_args.args[2]["anthropic-version"], "2023-06-01")

    def test_error_event_is_raised(self) -> None:
        frames = [("error", json.dumps({"type": "error", "error": {"message": "nope"}}))]
        provider = AnthropicProvider(
            ProviderConfig(
                name="a",
                kind="anthropic",
                base_url="https://api.anthropic.com",
                api_key="k",
            )
        )
        with mock.patch("omh.providers.anthropic.post_sse", return_value=iter(frames)):
            with self.assertRaises(ProviderError):
                list(provider.stream(model="m", messages=[]))


class FakeResponse:
    def __init__(self, lines: List[bytes]) -> None:
        self.lines = lines
        self.headers: Dict[str, str] = {}

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def __iter__(self):
        return iter(self.lines)


class HttpTest(unittest.TestCase):
    def test_post_sse_parses_events(self) -> None:
        response = FakeResponse(
            [
                b": keep-alive\n",
                b"\n",
                b"event: message_start\n",
                b'data: {"type": "message_start"}\n',
                b"\n",
                b"data: [DONE]\n",
            ]
        )
        with mock.patch("omh.providers.http._OPENER.open", return_value=response) as opened:
            events = list(post_sse("https://example.test/v1", {"a": 1}, {"X": "y"}))
        self.assertEqual(
            events, [("message_start", '{"type": "message_start"}'), (None, "[DONE]")]
        )
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1")
        self.assertEqual(request.get_header("X"), "y")

    def test_sse_json_stops_on_done_and_rejects_garbage(self) -> None:
        decoded = list(sse_json(iter([(None, '{"a": 1}'), (None, "[DONE]")]), "p"))
        self.assertEqual(decoded[0][1], {"a": 1})
        self.assertEqual(len(decoded), 1)
        with self.assertRaises(ProviderError):
            list(sse_json(iter([(None, "not json")]), "p"))

    def test_map_http_error(self) -> None:
        def http_error(
            code: int, body: bytes, headers: Dict[str, str] | None = None
        ) -> HTTPError:
            return HTTPError("u", code, "msg", headers or {}, io.BytesIO(body))

        rate = map_http_error(
            http_error(429, b'{"error": {"message": "slow down"}}', {"Retry-After": "2"}),
            "p",
        )
        self.assertIsInstance(rate, RateLimitError)
        self.assertEqual(rate.retry_after, 2.0)
        self.assertIn("slow down", rate.message)

        auth = map_http_error(http_error(401, b"{}"), "p")
        self.assertIsInstance(auth, AuthError)
        self.assertFalse(auth.retryable)

        bad = map_http_error(http_error(400, b"{}"), "p")
        self.assertFalse(bad.retryable)

    def test_post_sse_raises_mapped_error(self) -> None:
        error = HTTPError(
            "u", 503, "down", {}, io.BytesIO(b'{"error": {"message": "maintenance"}}')
        )
        with mock.patch("omh.providers.http._OPENER.open", side_effect=error):
            with self.assertRaises(ProviderError) as caught:
                list(post_sse("https://example.test/v1", {}, {}))
        self.assertEqual(caught.exception.status, 503)


if __name__ == "__main__":
    unittest.main()


