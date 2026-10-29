"""SSE malformed-shape hardening tests.

Malformed provider data previously caused
`AttributeError: 'str' object has no attribute ...` when a provider sends
unexpected event shapes. `_iter_sse_events` yields whatever `json.loads`
returns — strings, ints, lists — and every `.get()` downstream was a
potential crash. These tests feed a REAL malformed SSE byte stream
through the REAL pipeline (`_iter_sse_events` ->
`_chat_stream_once`) and prove:
  1. no AttributeError / TypeError is raised on ANY malformed shape,
  2. every valid event interleaved in the mess still parses correctly,
  3. genuine `error` events still raise APIError (not swallowed).
Only `_http` is mocked (fake response), exactly like
tests/test_stream_latency.py — the event parser is the real thing.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import json
import os
import sys
import unittest

# allow `python tests/test_sse_malformed.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from unittest.mock import patch

from fullagent import client
from fullagent.client import (APIError, _chat_stream_once, _result_from_json)


class FakeResponse:
    """Stand-in for requests.Response: iter_lines() yields raw SSE bytes."""

    def __init__(self, payloads):
        self.status_code = 200
        self.headers = {"Content-Type": "text/event-stream"}
        self._payloads = payloads
        self.closed = False

    def iter_lines(self):
        for p in self._payloads:
            yield ("data: " + json.dumps(p)).encode("utf-8")
            yield b""  # SSE event terminator (blank line)
        yield b"data: [DONE]"
        yield b""

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def post(self, *args, **kwargs):
        return self._resp


# Every malformed shape the parser must survive. Each entry is the JSON
# payload of one SSE data line.
MALFORMED_PAYLOADS = [
    "event as plain string",          # event: str
    42,                               # event: int
    [1, 2, 3],                        # event: list
    None,                             # event: null (skipped silently)
    {"choices": "choices-as-string"},  # choices: str (was: AttributeError)
    {"choices": 42},                  # choices: int
    {"choices": {"not": "a list"}},    # choices: dict
    {"choices": ["choice-as-string"]},  # choice: str (was: AttributeError)
    {"choices": [None]},              # choice: null
    {"choices": [{"delta": "delta-as-string"}]},  # delta: str (was crash)
    {"choices": [{"delta": None}]},    # delta: null
    {"choices": [{"delta": {"content": {"d": "ict"}}}]},  # content: dict (was TypeError at join)
    {"choices": [{"delta": {"content": 123}}]},          # content: int (was TypeError at join)
    {"choices": [{"delta": {"content": ["a", "list"]}}]},  # content: list
    {"choices": [{"delta": {"reasoning_content": ["r"]}}]},  # reasoning: list (was TypeError)
    {"choices": [{"delta": {"tool_calls": "tc-as-string"}}]},  # tool_calls: str (was crash)
    {"choices": [{"delta": {"tool_calls": 42}}]},        # tool_calls: int
    {"choices": [{"delta": {"tool_calls": ["entry-string"]}}]},  # tc entry: str (was crash)
    {"choices": [{"delta": {"tool_calls": [None]}}]},    # tc entry: null
    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": "fn-string"}]}}]},  # function: str (was crash)
    {"choices": [{"delta": {"tool_calls": [{"index": "notanint", "id": "x", "function": {"name": 123, "arguments": {"a": 1}}}]}}]},
    # index: str + name/arguments non-str — must coerce, never crash
    {"choices": [{"delta": {"tool_calls": [{"index": None, "function": {"name": None}}]}}]},
    {"usage": "usage-as-string"},      # usage: str (already guarded)
    {"usage": {"prompt_tokens": 5}},    # usage: valid dict
]

# Valid events interleaved between the malformed ones — all of these
# MUST survive and appear in the final StreamResult.
VALID_PAYLOADS = [
    {"choices": [{"delta": {"content": "Hello "}}]},
    {"choices": [{"delta": {"content": "world"}}]},
    {"choices": [{"delta": {"tool_calls": [
        {"index": 1, "id": "call_1",
         "function": {"name": "write_file", "arguments": "{\"p\":1}"}}]}}]},
    {"choices": [{"delta": {"content": "!"}, "finish_reason": "stop"}]},
]


def _run(payloads):
    resp = FakeResponse(payloads)
    seen = []
    tools_started = []
    with patch.object(client, "_http", return_value=FakeSession(resp)):
        result = _chat_stream_once(
            "http://fake/v1/chat/completions", {}, {},
            on_token=seen.append,
            on_reasoning=None,
            on_tool_start=tools_started.append,
            on_tool_args=None,
            should_cancel=None, timeout=30.0)
    return result, seen, tools_started


class MalformedStreamTests(unittest.TestCase):
    def test_malformed_events_never_crash(self):
        """Every malformed shape above passes through the real parser
        without AttributeError/TypeError. (Before the fix, the str-shaped
        ones raised AttributeError: 'str' object has no attribute 'get'.)"""
        payloads = []
        for m in MALFORMED_PAYLOADS:
            payloads.append(m)
        for v in VALID_PAYLOADS:
            payloads.append(v)
        result, seen, tools_started = _run(payloads)
        # If we get here, no exception was raised — the assertion below
        # documents the survivor count for the report.
        self.assertTrue(True, f"parsed {len(payloads)} events without crash")

    def test_valid_events_survive_malformed_stream(self):
        """Valid content/tool-call events interleaved with garbage must
        still parse — malformed events are SKIPPED or canonicalized,
        never fatal. (Non-str content pieces are canonicalized to text
        by _canonical_text, e.g. Anthropic-style blocks are joined —
        the invariant is no crash + no lost valid data.)"""
        payloads = []
        it = iter(MALFORMED_PAYLOADS)
        for v in VALID_PAYLOADS:
            payloads.append(next(it))
            payloads.append(v)
        payloads.extend(it)
        result, seen, tools_started = _run(payloads)

        # Valid content pieces arrive in order, with no crash in between.
        self.assertEqual(seen[:2], ["Hello ", "world"])
        self.assertIn("!", seen)
        content = result.content
        self.assertLess(content.index("Hello "), content.index("world"))
        self.assertLess(content.index("world"), content.index("!"))
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage, {"prompt_tokens": 5})

        # The valid tool call survives intact, byte-for-byte.
        self.assertIn({"id": "call_1", "type": "function",
                       "function": {"name": "write_file",
                                    "arguments": "{\"p\":1}"}},
                      result.tool_calls)
        self.assertIn("write_file", tools_started)
        # Every emitted tool call is shape-safe: dict function with
        # str name/arguments — no consumer can AttributeError on it.
        for tc in result.tool_calls:
            self.assertIsInstance(tc, dict)
            fn = tc["function"]
            self.assertIsInstance(fn, dict)
            self.assertIsInstance(fn["name"], str)
            self.assertIsInstance(fn["arguments"], str)

    def test_error_events_still_raise_apierror(self):
        """Hardening must not swallow real error events: dict AND
        string error shapes both raise APIError."""
        for err in ({"message": "boom", "code": 400}, "boom-string"):
            with self.assertRaises(APIError):
                _run([{"error": err}])

    def test_result_from_json_malformed_shapes(self):
        """The blocking/JSON-fallback parser gets the same treatment —
        no AttributeError on any of these."""
        for bad in [
            {"choices": "choices-as-string"},
            {"choices": [{"message": "message-as-string"}]},
            {"choices": [{"message": None}]},
            {"choices": [{"message": {"content": {"d": "ict"}}}]},
            {"choices": [{"message": {"content": "x", "tool_calls":
                                      [{"function": "fn-string"}]}}]},
            {"choices": ["choice-as-string"]},
            {"model": "m"},
        ]:
            r = _result_from_json(bad, "")
            self.assertIsInstance(r.content, str)
            self.assertIsInstance(r.tool_calls, list)
        # and a fully valid body still parses end to end
        r = _result_from_json(
            {"model": "m", "choices": [{"message": {"content": "hi"},
                                        "finish_reason": "stop"}]}, "")
        self.assertEqual((r.content, r.finish_reason, r.model),
                         ("hi", "stop", "m"))


def _selftest() -> int:
    """Standalone proof: `python tests/test_sse_malformed.py`."""
    print("== SSE malformed-shape hardening self-test ==")
    suite = unittest.TestLoader().loadTestsFromTestCase(MalformedStreamTests)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(_selftest())
