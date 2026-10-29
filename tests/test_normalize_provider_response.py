"""Provider response-shape normalization tests for fullagent.client.

Proves that normalize_provider_response() converts every malformed
provider shape into a canonical StreamResult-safe dict BEFORE parsing,
so no shape can raise ``AttributeError: 'str' object has no attribute
...`` inside a turn.

Shapes covered (observed against OpenAI-compatible gateways):
  * usage as a string
  * tool_calls as a bare string, or a list containing strings
  * missing choices
  * choices as a string, a single dict, or a list containing strings
  * choice.message as a string
  * choice.delta as a string (streaming events)
  * content as Anthropic-style blocks [{"type":"text","text":"hi"}]
  * tool_call.function as a JSON string, or a plain string
  * tool_call.function.arguments as a dict
  * tool_call.index as a string
  * finish_reason as a non-string, model as a non-string
  * non-dict top-level payload -> APIError (never AttributeError)

Plus two end-to-end tests over a raw-socket HTTP server:
  * an SSE stream of malformed events -> chat_stream() returns a clean
    StreamResult with no AttributeError;
  * a non-SSE JSON body when stream=true was requested (the
    stream-mode JSON fallback) -> chat_stream() returns a clean result.

Run:  python3 -m pytest tests/test_normalize_provider_response.py -x -q
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import copy
import json
import socket
import threading
import unittest

from fullagent.client import (APIError, StreamResult, _result_from_json,
                              chat_stream, normalize_provider_response)
from fullagent.config import Effort, Model, Provider


# ---------------------------------------------------------------------------
# unit: normalize_provider_response canonical output
# ---------------------------------------------------------------------------

class NormalizeUnitTests(unittest.TestCase):
    def test_usage_as_string(self):
        data = {"model": "m", "usage": "prompt_tokens=12,total=34",
                "choices": []}
        norm = normalize_provider_response(data, "kilo")
        self.assertIsNone(norm["usage"])
        self.assertEqual(norm["model"], "m")
        self.assertEqual(norm["choices"], [])

    def test_missing_choices(self):
        norm = normalize_provider_response({"model": "m"}, "kilo")
        self.assertEqual(norm["choices"], [])
        # _result_from_json must not crash and must fall back to model_id
        res = _result_from_json({"model": "m"}, "fallback-id")
        self.assertIsInstance(res, StreamResult)
        self.assertEqual(res.model, "m")
        self.assertEqual(res.content, "")
        self.assertEqual(res.tool_calls, [])

    def test_choices_as_string(self):
        data = {"model": "m", "choices": "some gateway error text"}
        res = _result_from_json(data, "mid")  # must not AttributeError
        self.assertEqual(res.content, "")
        self.assertEqual(res.tool_calls, [])

    def test_choices_as_single_dict(self):
        data = {"choices": {"message": {"content": "hi"}}}
        norm = normalize_provider_response(data, "kilo")
        self.assertEqual(len(norm["choices"]), 1)
        self.assertEqual(norm["choices"][0]["message"]["content"], "hi")

    def test_choice_list_with_string_entry(self):
        data = {"choices": ["garbage", {"message": {"content": "ok"}}]}
        res = _result_from_json(data, "mid")  # choices[0] is a str
        self.assertIsInstance(res, StreamResult)
        self.assertEqual(res.content, "")  # placeholder choice, no crash

    def test_message_as_string(self):
        data = {"choices": [{"message": "just a string",
                             "finish_reason": "stop"}]}
        res = _result_from_json(data, "mid")
        self.assertEqual(res.content, "")
        self.assertEqual(res.finish_reason, "stop")

    def test_tool_calls_as_bare_string(self):
        data = {"choices": [{"message": {"content": "hi",
                                         "tool_calls": "read_file"}}]}
        res = _result_from_json(data, "mid")
        self.assertEqual(res.content, "hi")
        self.assertEqual(res.tool_calls, [])

    def test_tool_calls_list_of_strings(self):
        data = {"choices": [{"message": {"tool_calls": ["a", "b", 42]}}]}
        res = _result_from_json(data, "mid")
        self.assertEqual(res.tool_calls, [])

    def test_function_as_json_string(self):
        fn = json.dumps({"name": "read", "arguments": {"path": "/x"}})
        data = {"choices": [{"message": {"tool_calls": [
            {"id": "c1", "function": fn}]}}]}
        res = _result_from_json(data, "mid")
        self.assertEqual(len(res.tool_calls), 1)
        tc = res.tool_calls[0]
        self.assertEqual(tc["function"]["name"], "read")
        # arguments must be a JSON string parseable downstream
        self.assertEqual(json.loads(tc["function"]["arguments"]),
                         {"path": "/x"})

    def test_function_as_plain_string(self):
        data = {"choices": [{"message": {"tool_calls": [
            {"id": "c1", "function": "read"}]}}]}
        res = _result_from_json(data, "mid")
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0]["function"]["name"], "read")

    def test_arguments_as_dict(self):
        data = {"choices": [{"message": {"tool_calls": [
            {"id": "c1",
             "function": {"name": "w", "arguments": {"a": 1}}}]}}]}
        res = _result_from_json(data, "mid")
        args = res.tool_calls[0]["function"]["arguments"]
        self.assertIsInstance(args, str)
        self.assertEqual(json.loads(args), {"a": 1})

    def test_content_as_blocks(self):
        data = {"choices": [{"message": {"content": [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"}]}}]}
        res = _result_from_json(data, "mid")
        self.assertEqual(res.content, "hello world")

    def test_reasoning_as_dict(self):
        data = {"choices": [{"message": {"reasoning_content": {"x": 1}}}]}
        res = _result_from_json(data, "mid")
        self.assertIsInstance(res.reasoning, str)

    def test_finish_reason_non_string(self):
        data = {"choices": [{"message": {"content": "x"},
                             "finish_reason": {"weird": True}}]}
        res = _result_from_json(data, "mid")
        self.assertTrue(res.finish_reason is None
                        or isinstance(res.finish_reason, str))

    def test_model_non_string(self):
        norm = normalize_provider_response({"model": 123}, "kilo")
        self.assertIsInstance(norm["model"], str)

    def test_non_dict_top_level_raises_apierror(self):
        for bad in (["a", "b"], "oops", 42, None):
            with self.assertRaises(APIError):
                normalize_provider_response(bad, "kilo")
            with self.assertRaises(APIError):
                _result_from_json(bad, "mid")

    def test_delta_as_string_stream_mode(self):
        event = {"choices": [{"delta": "just a string"}]}
        norm = normalize_provider_response(event, "kilo", for_stream=True)
        delta = norm["choices"][0]["delta"]
        # the exact access pattern of the _chat_stream_once event loop
        # must work on the normalized event
        self.assertEqual(delta.get("content"), "")
        self.assertEqual(delta.get("tool_calls"), [])
        self.assertEqual(delta.get("reasoning"), "")

    def test_stream_tool_call_index_as_string(self):
        event = {"choices": [{"delta": {"tool_calls": [
            {"index": "0", "id": "c1",
             "function": {"name": "r", "arguments": "{}"}}]}}]}
        norm = normalize_provider_response(event, "kilo", for_stream=True)
        tc = norm["choices"][0]["delta"]["tool_calls"][0]
        self.assertIsInstance(tc["index"], int)
        self.assertEqual(tc["index"], 0)
        self.assertEqual(tc["function"]["name"], "r")

    def test_stream_tool_calls_as_string(self):
        event = {"choices": [{"delta": {"tool_calls": "read,write"}}]}
        norm = normalize_provider_response(event, "kilo",
                                           for_stream=True)
        tcs = norm["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(tcs, [])
        # iterating the normalized list like the event loop does is safe
        for tc in tcs:
            tc.get("index", 0)

    def test_input_not_mutated(self):
        data = {"model": "m", "usage": "junk",
                "choices": [{"message": {"content": [{"type": "text",
                                                     "text": "hi"}],
                                        "tool_calls": ["x"]}}]}
        snapshot = copy.deepcopy(data)
        normalize_provider_response(data, "kilo")
        self.assertEqual(data, snapshot)

    def test_canonical_shape_type_invariants(self):
        """Every field of the canonical dict has its promised type."""
        data = {"model": None, "usage": [1, 2],
                "choices": [{"delta": {"content": 7,
                                       "tool_calls": [{"index": None}]},
                             "finish_reason": 5}]}
        norm = normalize_provider_response(data, "kilo", for_stream=True)
        self.assertIsInstance(norm["model"], str)
        self.assertTrue(norm["usage"] is None
                        or isinstance(norm["usage"], dict))
        self.assertIsInstance(norm["choices"], list)
        for ch in norm["choices"]:
            self.assertIsInstance(ch, dict)
            self.assertTrue(ch["finish_reason"] is None
                            or isinstance(ch["finish_reason"], str))
            for key in ("message", "delta"):
                m = ch[key]
                self.assertIsInstance(m["content"], str)
                self.assertIsInstance(m["reasoning"], str)
                self.assertIsInstance(m["tool_calls"], list)
                for tc in m["tool_calls"]:
                    self.assertIsInstance(tc, dict)
                    fn = tc["function"]
                    self.assertIsInstance(fn, dict)
                    self.assertIsInstance(fn["name"], str)
                    self.assertIsInstance(fn["arguments"], str)


# ---------------------------------------------------------------------------
# end-to-end: malformed wire shapes through chat_stream
# ---------------------------------------------------------------------------

def _serve(handler):
    """Raw-socket HTTP server on 127.0.0.1:0; return (socket, port)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port = srv.getsockname()[1]

    def _read_headers(conn):
        conn.settimeout(10)
        data = b""
        try:
            while b"\r\n\r\n" not in data:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            header, _, body = data.partition(b"\r\n\r\n")
            content_length = 0
            for line in header.split(b"\r\n")[1:]:
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
                    break
            while len(body) < content_length:
                chunk = conn.recv(min(65536, content_length - len(body)))
                if not chunk:
                    break
                body += chunk
        except OSError:
            pass

    def _loop():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return
            _read_headers(conn)
            t = threading.Thread(target=handler, args=(conn,), daemon=True)
            t.start()

    threading.Thread(target=_loop, daemon=True).start()
    return srv, port


def _mk(base_url):
    prov = Provider(key="t", name="t", base_url=base_url, api_key="dummy",
                    color="red")
    mod = Model(id="t", provider="t", label="t")
    eff = Effort(key="t", label="T", color="red", max_tokens=None,
                 temperature=0.0, reasoning_effort=None, description="t")
    return prov, mod, eff


MSGS = [{"role": "user", "content": "hi"}]


def _malformed_sse(conn):
    """SSE stream where every event carries a malformed provider shape."""
    try:
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: text/event-stream\r\n"
                     b"Connection: close\r\n\r\n"
                     # usage as string, choices as string
                     b'data: {"model": 123, "usage": "prompt_tokens=12",'
                     b'"choices": "broken"}\n\n'
                     # delta as a string, finish_reason as int
                     b'data: {"choices": [{"delta": "just a string",'
                     b'"finish_reason": 7}]}\n\n'
                     # content blocks + tool_calls as bare string
                     b'data: {"choices": [{"delta": {"content": '
                     b'[{"type": "text", "text": "hello"}],'
                     b'"tool_calls": "read_file"}}]}\n\n'
                     # index as string, arguments as dict
                     b'data: {"choices": [{"delta": {"tool_calls": '
                     b'[{"index": "0", "id": "c1", "function": '
                     b'{"name": "read", "arguments": {"path": "/tmp/x"}}}]},'
                     b'"finish_reason": "stop"}]}\n\n'
                     b"data: [DONE]\n\n")
    except OSError:
        pass
    finally:
        conn.close()


def _malformed_json_body(conn):
    """Non-SSE JSON body even though stream=true was requested."""
    body = json.dumps({
        "model": "m", "usage": "junk",
        "choices": [{"message": {
            "content": [{"type": "text", "text": "hi"}],
            "tool_calls": ["nope", 42]},
            "finish_reason": None}],
    }).encode()
    try:
        conn.sendall(b"HTTP/1.1 200 OK\r\n"
                     b"Content-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                     b"Connection: close\r\n\r\n" + body)
    except OSError:
        pass
    finally:
        conn.close()


class MalformedWireTests(unittest.TestCase):
    def test_malformed_sse_stream(self):
        """Every malformed SSE event shape normalizes; the turn completes
        with a clean StreamResult and no AttributeError."""
        srv, port = _serve(_malformed_sse)
        try:
            prov, mod, eff = _mk(f"http://127.0.0.1:{port}")
            seen_tokens = []
            res = chat_stream(prov, mod, eff, MSGS, None,
                              on_token=seen_tokens.append, timeout=15)
        finally:
            srv.close()
        self.assertIsInstance(res, StreamResult)
        self.assertEqual(res.content, "hello")
        self.assertEqual(seen_tokens, ["hello"])
        # usage-as-string must not leak through as a str
        self.assertTrue(res.usage is None or isinstance(res.usage, dict))
        # one real tool call survived; the bare-string tool_calls did not
        self.assertEqual(len(res.tool_calls), 1)
        tc = res.tool_calls[0]
        self.assertEqual(tc["function"]["name"], "read")
        self.assertEqual(json.loads(tc["function"]["arguments"]),
                         {"path": "/tmp/x"})
        self.assertEqual(res.finish_reason, "stop")

    def test_malformed_json_fallback(self):
        """stream=true requested but provider returned a plain JSON body
        with malformed shapes -> clean StreamResult, no AttributeError."""
        srv, port = _serve(_malformed_json_body)
        try:
            prov, mod, eff = _mk(f"http://127.0.0.1:{port}")
            res = chat_stream(prov, mod, eff, MSGS, None, timeout=15)
        finally:
            srv.close()
        self.assertIsInstance(res, StreamResult)
        self.assertEqual(res.content, "hi")
        self.assertEqual(res.tool_calls, [])
        self.assertTrue(res.usage is None or isinstance(res.usage, dict))


if __name__ == "__main__":
    unittest.main()
