"""NVIDIA NIM streaming payload test — the reported curl, end to end.

The reported working request against build.nvidia.com (NVIDIA NIM) is:

    https://integrate.api.nvidia.com/v1/chat/completions
    Accept: text/event-stream                # because stream=true
    {"model": "mistralai/mistral-large-4-1t-a52b",
     "messages": [{"role": "user", "content": "hii"}],
     "max_tokens": 4096, "stream": true, "temperature": 0.5,
     "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0,
     "tools": [describe_harry_potter_character], "tool_choice": "auto",
     "reasoning_effort": "high"}

This file proves the app's own send path reproduces that request and
consumes the SSE body it answers with:

  A. config ships the `nvidia` provider + the mistral-large model
     (window smaller than the 200k effort budget — the regression below);
  B. build_payload() puts stream=True, the tool schema, tool_choice
     "auto" and reasoning_effort "high" on the wire (stream=False for
     blocking mode);
  C. _base_headers() negotiates Accept: text/event-stream when streaming
     and Accept: application/json otherwise — the curl's own branch;
  D. a real NIM-shaped SSE body (reasoning_content deltas + tool_call
     argument fragments split across chunks) is parsed by the REAL
     _chat_stream_once pipeline into one tool call with valid JSON args.

Only `_http` is faked, exactly like tests/test_sse_malformed.py — payload
building and SSE parsing are the real code.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import dataclasses
import json
import os
import sys
import unittest

# allow `python tests/test_nvidia_stream_payload.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from unittest.mock import patch

from fullagent import client, config
from fullagent.client import (_base_headers, _chat_stream_once, build_payload)
from fullagent.config import Provider, effort_by_key, model_by_id

NVIDIA_MODEL_ID = "mistralai/mistral-large-4-1t-a52b"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# The exact tool schema sent by the reported curl.
HP_TOOL = {
    "type": "function",
    "function": {
        "name": "describe_harry_potter_character",
        "description": "Returns information and images of Harry Potter "
                       "characters.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [
                        "Harry James Potter", "Hermione Jean Granger",
                        "Ron Weasley", "Fred Weasley", "George Weasley",
                        "Bill Weasley", "Percy Weasley", "Charlie Weasley",
                        "Ginny Weasley", "Molly Weasley", "Arthur Weasley",
                        "Neville Longbottom", "Luna Lovegood",
                        "Draco Malfoy",
                        "Albus Percival Wulfric Brian Dumbledore",
                        "Minerva McGonagall", "Remus Lupin",
                        "Rubeus Hagrid", "Sirius Black", "Severus Snape",
                        "Bellatrix Lestrange", "Lord Voldemort",
                        "Cedric Diggory", "Nymphadora Tonks",
                        "James Potter",
                    ],
                    "description": "Name of the Harry Potter character",
                },
            },
            "required": ["name"],
        },
    },
}

# `extrahigh` maps to reasoning_effort="high" — the value in the curl.
CURL_EFFORT = "extrahigh"


def _model():
    model = model_by_id(NVIDIA_MODEL_ID)
    assert model is not None, f"{NVIDIA_MODEL_ID} must ship in config.MODELS"
    return model


def _provider(api_key: str = "nvapi-test-key"):
    return Provider(key="nvidia", name="NVIDIA NIM",
                    base_url=NVIDIA_BASE_URL, api_key=api_key,
                    color="#76b900")


def _chunk(delta, finish_reason=None, usage=None):
    """One `chat.completion.chunk` event, NIM-shaped."""
    event = {
        "id": "chatcmpl-nim-1",
        "object": "chat.completion.chunk",
        "created": 1793434426,
        "model": NVIDIA_MODEL_ID,
        "choices": [{"index": 0, "delta": delta,
                     "finish_reason": finish_reason}],
    }
    if usage is not None:
        event["usage"] = usage
    return event


# A realistic NIM stream for "hii": hidden reasoning first, then a tool
# call whose JSON arguments arrive in two fragments, then the finish
# chunk and the usage-only chunk.
NIM_STREAM = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"reasoning_content": "The user said hi. "}),
    _chunk({"reasoning_content": "No character was named, so I'll pick one."}),
    _chunk({"tool_calls": [{
        "index": 0, "id": "call_HP01", "type": "function",
        "function": {"name": "describe_harry_potter_character",
                     "arguments": ""}}]}),
    _chunk({"tool_calls": [{"index": 0,
                            "function": {"arguments": '{"name": "Harry'}}]}),
    _chunk({"tool_calls": [{"index": 0,
                            "function": {"arguments": ' James Potter"}'}}]}),
    _chunk({}, finish_reason="tool_calls"),
    _chunk({}, usage={"prompt_tokens": 91, "completion_tokens": 37,
                      "total_tokens": 128}),
]


class FakeStreamResponse:
    """Stand-in for requests.Response: iter_lines() yields raw SSE bytes."""

    def __init__(self, events):
        self.status_code = 200
        self.headers = {"Content-Type": "text/event-stream; charset=utf-8"}
        self._events = events
        self.closed = False
        self.text = ""

    def iter_lines(self):
        for event in self._events:
            yield ("data: " + json.dumps(event)).encode("utf-8")
            yield b""  # SSE event terminator (blank line)
        yield b"data: [DONE]"
        yield b""

    def json(self):
        raise ValueError("streamed response has no JSON body")

    def close(self):
        self.closed = True


class RecordingSession:
    """Fake requests.Session that records the outbound request."""

    def __init__(self, response):
        self._response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self._response


class NvidiaConfigTests(unittest.TestCase):
    def test_provider_and_model_are_configured(self):
        provider = config.PROVIDERS["nvidia"]
        self.assertEqual(provider.name, "NVIDIA NIM")
        self.assertEqual(provider.base_url, NVIDIA_BASE_URL)
        model = _model()
        self.assertEqual(model.provider, "nvidia")
        self.assertTrue(model.supports_tools)
        self.assertTrue(model.supports_reasoning)
        self.assertEqual(model.context_window, 1_000_000)
        self.assertGreaterEqual(model.context_window, config.MAX_TOKENS)

    def test_missing_key_is_a_clear_error_not_a_bad_request(self):
        with self.assertRaises(client.APIError) as ctx:
            client._check_api_key(_provider(api_key=""), _model())
        message = str(ctx.exception)
        self.assertIn("no API key configured", message)
        self.assertIn("NVIDIA_API_KEY", message)


class NvidiaPayloadTests(unittest.TestCase):
    def test_streaming_payload_matches_the_reported_curl(self):
        payload = build_payload(_model(), effort_by_key(CURL_EFFORT),
                                [{"role": "user", "content": "hii"}],
                                [HP_TOOL], stream=True)
        # Same body keys as the curl's payload.json — nothing more, nothing
        # less (top_p/frequency_penalty/presence_penalty are the curl's
        # defaults and are not sent by the app at all).
        self.assertEqual(sorted(payload),
                         ["max_tokens", "messages", "model",
                          "reasoning_effort", "stream", "temperature",
                          "tool_choice", "tools"])
        self.assertEqual(payload["model"], NVIDIA_MODEL_ID)
        self.assertIs(payload["stream"], True)
        self.assertEqual(payload["messages"],
                         [{"role": "user", "content": "hii"}])
        self.assertEqual(payload["tools"], [HP_TOOL])
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertGreater(payload["max_tokens"], 0)
        self.assertLessEqual(payload["max_tokens"],
                             client._MAX_TOKENS_CAP["nvidia"])
        # The body must serialize exactly like `curl -d @payload.json`.
        self.assertIn('"stream": true', json.dumps(payload))

    def test_blocking_payload_turns_stream_off(self):
        payload = build_payload(_model(), effort_by_key(CURL_EFFORT),
                                [{"role": "user", "content": "hii"}],
                                [HP_TOOL], stream=False)
        self.assertIs(payload["stream"], False)

    def test_accept_header_follows_the_stream_flag(self):
        provider = _provider()
        streaming = _base_headers(provider, stream=True)
        self.assertEqual(streaming["Accept"], "text/event-stream")
        self.assertEqual(streaming["Content-Type"], "application/json")
        self.assertEqual(streaming["Authorization"],
                         "Bearer nvapi-test-key")
        blocking = _base_headers(provider, stream=False)
        self.assertEqual(blocking["Accept"], "application/json")


class SmallWindowRegressionTests(unittest.TestCase):
    """A window smaller than the effort's token budget used to be fatal.

    `_ensure_window_fit` compared input + the RAW requested budget
    (config.MAX_TOKENS = 200k) against the window, so every 128k model
    rejected even a three-word history locally with "conversation is too
    large" — the NVIDIA Mistral route could never be called at all.
    """

    def test_small_history_fits_a_small_window(self):
        model = _model()
        effort = effort_by_key(config.DEFAULT_EFFORT)
        tokens = client._ensure_window_fit(
            model, [{"role": "user", "content": "hii"}], [HP_TOOL], effort)
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, model.context_window)

    def test_max_tokens_is_clamped_inside_the_window(self):
        model = _model()
        payload = build_payload(model, effort_by_key(config.DEFAULT_EFFORT),
                                [{"role": "user", "content": "hii"}],
                                [HP_TOOL], stream=True)
        self.assertLess(payload["max_tokens"], model.context_window)

    def test_oversized_input_still_raises(self):
        tiny = dataclasses.replace(_model(), context_window=2_000)
        messages = [{"role": "user", "content": "x" * 40_000}]
        with self.assertRaises(client.APIError) as ctx:
            build_payload(tiny, effort_by_key("low"), messages, None,
                          stream=True)
        self.assertIn("context window", str(ctx.exception))


class NvidiaStreamTests(unittest.TestCase):
    def _run(self):
        model = _model()
        payload = build_payload(model, effort_by_key(CURL_EFFORT),
                                [{"role": "user", "content": "hii"}],
                                [HP_TOOL], stream=True)
        provider = _provider()
        headers = _base_headers(provider, stream=True)
        url = provider.base_url.rstrip("/") + "/chat/completions"
        response = FakeStreamResponse(NIM_STREAM)
        session = RecordingSession(response)
        seen = {"tokens": [], "reasoning": [], "tool_names": [],
                "arg_fragments": []}

        def _on_tool_args(name, fragment):
            seen["arg_fragments"].append((name, fragment))

        with patch.object(client, "_http", return_value=session):
            result = _chat_stream_once(
                url, headers, payload,
                on_token=seen["tokens"].append,
                on_reasoning=seen["reasoning"].append,
                on_tool_start=seen["tool_names"].append,
                on_tool_args=_on_tool_args,
                should_cancel=None, timeout=30.0)
        return result, session, response, seen

    def test_wire_request_is_the_streaming_curl(self):
        _result, session, response, _seen = self._run()
        self.assertEqual(len(session.calls), 1)
        url, kwargs = session.calls[0]
        self.assertEqual(url, NVIDIA_BASE_URL + "/chat/completions")
        self.assertIs(kwargs["stream"], True)
        self.assertEqual(kwargs["headers"]["Accept"], "text/event-stream")
        self.assertEqual(kwargs["headers"]["Content-Type"],
                         "application/json")
        self.assertEqual(kwargs["headers"]["Authorization"],
                         "Bearer nvapi-test-key")
        body = kwargs["json"]
        self.assertIs(body["stream"], True)
        self.assertEqual(body["model"], NVIDIA_MODEL_ID)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertEqual(
            body["tools"][0]["function"]["name"],
            "describe_harry_potter_character")
        self.assertTrue(response.closed)

    def test_sse_stream_parses_reasoning_and_tool_call(self):
        result, _session, _response, seen = self._run()
        self.assertEqual(result.model, NVIDIA_MODEL_ID)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual("".join(seen["reasoning"]),
                         "The user said hi. No character was named, "
                         "so I'll pick one.")
        self.assertEqual(result.reasoning, "".join(seen["reasoning"]))
        self.assertEqual(result.content, "")
        self.assertEqual(seen["tokens"], [])
        self.assertEqual(seen["tool_names"],
                         ["describe_harry_potter_character"])
        self.assertEqual(seen["arg_fragments"],
                         [("describe_harry_potter_character",
                           '{"name": "Harry'),
                          ("describe_harry_potter_character",
                           ' James Potter"}')])
        self.assertEqual(len(result.tool_calls), 1)
        call = result.tool_calls[0]
        self.assertEqual(call["id"], "call_HP01")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"],
                         "describe_harry_potter_character")
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"name": "Harry James Potter"})
        self.assertEqual(result.usage,
                         {"prompt_tokens": 91, "completion_tokens": 37,
                          "total_tokens": 128})


    def test_public_chat_stream_sends_the_curls_stream_request(self):
        """chat_stream() — the path the agent actually calls — must put
        the same body/headers on the wire as the reported curl."""
        model = _model()
        provider = _provider()
        session = RecordingSession(FakeStreamResponse(NIM_STREAM))
        with patch.object(client, "_http", return_value=session):
            result = client.chat_stream(
                provider, model, effort_by_key(CURL_EFFORT),
                [{"role": "user", "content": "hii"}], [HP_TOOL],
                timeout=30.0)
        url, kwargs = session.calls[0]
        self.assertEqual(url, NVIDIA_BASE_URL + "/chat/completions")
        self.assertIs(kwargs["stream"], True)
        self.assertEqual(kwargs["headers"]["Accept"], "text/event-stream")
        self.assertIs(kwargs["json"]["stream"], True)
        self.assertEqual(kwargs["json"]["tool_choice"], "auto")
        self.assertEqual(kwargs["json"]["reasoning_effort"], "high")
        self.assertEqual(result.tool_calls[0]["function"]["name"],
                         "describe_harry_potter_character")
        self.assertEqual(
            json.loads(result.tool_calls[0]["function"]["arguments"]),
            {"name": "Harry James Potter"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

