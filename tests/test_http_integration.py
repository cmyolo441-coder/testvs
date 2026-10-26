"""Real HTTP round trips against local SSE servers (no network access needed)."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from omh.config import ProviderConfig
from omh.providers.anthropic import AnthropicProvider
from omh.providers.base import ChatMessage
from omh.providers.openai import OpenAIProvider


class Server:
    def __init__(self, handler) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class OpenAISSEHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        OpenAISSEHandler.requests.append(
            {"headers": dict(self.headers), "body": body}
        )
        has_tool = any(message.get("role") == "tool" for message in body["messages"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if has_tool:
            frames = [
                {"choices": [{"delta": {"content": "done"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            ]
        else:
            frames = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_http",
                                        "function": {
                                            "name": "list_dir",
                                            "arguments": '{"path": "."}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            ]
        for frame in frames:
            self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args) -> None:
        return


class AnthropicSSEHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        AnthropicSSEHandler.requests.append(
            {"headers": dict(self.headers), "body": body}
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "model": "claude-test",
                        "usage": {"input_tokens": 6, "output_tokens": 1},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi from anthropic"},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 4},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        for event, payload in events:
            self.wfile.write(
                f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()
            )
            self.wfile.flush()

    def log_message(self, *args) -> None:
        return


class HttpIntegrationTest(unittest.TestCase):
    def test_openai_real_stream_with_tool_call(self) -> None:
        OpenAISSEHandler.requests = []
        server = Server(OpenAISSEHandler)
        try:
            provider = OpenAIProvider(
                ProviderConfig(
                    name="local", base_url=server.base_url, api_key="http-key"
                )
            )
            chunks = list(
                provider.stream(
                    model="test-model",
                    messages=[ChatMessage.user("list")],
                    tools=[{"type": "function", "function": {"name": "list_dir"}}],
                )
            )
            self.assertEqual(chunks[-1].finish, "tool_calls")
            call = chunks[-1].tool_calls[0]
            self.assertEqual(call.name, "list_dir")
            self.assertEqual(call.parsed(), {"path": "."})

            followup = list(
                provider.stream(
                    model="test-model",
                    messages=[
                        ChatMessage.user("list"),
                        ChatMessage.assistant("", [call]),
                        ChatMessage.tool("call_http", "ok", "list_dir"),
                    ],
                )
            )
            self.assertEqual("".join(chunk.text for chunk in followup), "done")
            self.assertEqual(followup[-1].usage.total, 11)

            sent = OpenAISSEHandler.requests[0]
            self.assertEqual(sent["headers"].get("Authorization"), "Bearer http-key")
            self.assertTrue(sent["body"]["stream"])
            self.assertEqual(sent["body"]["model"], "test-model")
        finally:
            server.close()

    def test_anthropic_real_stream(self) -> None:
        AnthropicSSEHandler.requests = []
        server = Server(AnthropicSSEHandler)
        try:
            provider = AnthropicProvider(
                ProviderConfig(
                    name="anthropic",
                    kind="anthropic",
                    base_url=server.base_url.replace("/v1", ""),
                    api_key="ant-key",
                )
            )
            chunks = list(
                provider.stream(
                    model="claude-test",
                    messages=[
                        ChatMessage.system("be brief"),
                        ChatMessage.user("hi"),
                    ],
                )
            )
            self.assertEqual("".join(chunk.text for chunk in chunks), "hi from anthropic")
            self.assertEqual(chunks[-1].finish, "stop")
            self.assertEqual(chunks[-1].usage.total, 10)
            sent = AnthropicSSEHandler.requests[0]
            headers = {key.lower(): value for key, value in sent["headers"].items()}
            self.assertEqual(headers.get("x-api-key"), "ant-key")
            self.assertEqual(headers.get("anthropic-version"), "2023-06-01")
            self.assertEqual(sent["body"]["system"], "be brief")
        finally:
            server.close()


if __name__ == "__main__":
    unittest.main()

