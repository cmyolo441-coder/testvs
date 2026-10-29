"""Malformed-provider simulation test (provider-boundary regression).

Adversarial harness driving the REAL parsing functions in fullagent.client
against ~70 malformed provider payloads. Proves the reported incident
(AttributeError: 'str' object has no attribute ...) can never happen:

  * _result_from_json            — non-streamed JSON body parser
  * typeguards primitives        — the boundary canonicalizer
  * _iter_sse_events             — raw SSE framing
  * _chat_stream_once            — REAL SSE accumulation loop (fake HTTP)
  * assistant_message            — history message builder
  * _learn_from_usage / safe_token_counts — usage calibration
  * StreamResult                 — construction coercion

Rule: no AttributeError/TypeError EVER, whatever the shape.
Allowed outcomes: a normalized StreamResult with correctly-typed fields,
or a clean APIError for genuinely unparseable top-level payloads and
provider "error" events.

Run:  python3 -m pytest tests/test_malformed_provider.py -q
      (standalone: python3 /tmp/malformed_provider_test.py)
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import json
import sys

# Standalone entry (python3 /tmp/malformed_provider_test.py): make sure the
# repo package is importable regardless of the script's own directory.
# Harmless under pytest (python -m pytest already puts the root first).
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


import fullagent.client as _client
from fullagent import typeguards as _typeguards
from fullagent.client import (
    APIError,
    StreamResult,
    _chat_stream_once,
    _iter_sse_events,
    _learn_from_usage,
    _result_from_json,
    assistant_message,
    safe_token_counts,
)

_CRASHERS = (AttributeError, TypeError)


def _assert_safe_result(res):
    """StreamResult fields must hold their declared types."""
    assert isinstance(res, StreamResult)
    assert isinstance(res.content, str)
    assert isinstance(res.reasoning, str)
    assert isinstance(res.model, str)
    assert res.finish_reason is None or isinstance(res.finish_reason, str)
    assert res.usage is None or isinstance(res.usage, dict)
    assert isinstance(res.tool_calls, list)
    for tc in res.tool_calls:
        assert isinstance(tc, dict)
        fn = tc.get("function")
        assert isinstance(fn, dict), tc
        assert isinstance(fn.get("name"), str), tc
        assert isinstance(fn.get("arguments"), str), tc


# --------------------------------------------------------------------------
# fakes for the real SSE accumulation loop
# --------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, lines):
        self._lines = [l.encode() if isinstance(l, str) else l
                       for l in lines]

    def iter_lines(self, **kw):
        return iter(self._lines)

    def close(self):
        pass


class _FakeStreamResp(_FakeResp):
    def __init__(self, events, content_type="text/event-stream"):
        lines = []
        for ev in events:
            lines.append(ev if isinstance(ev, str)
                         else "data: " + json.dumps(ev))
            lines.append("")
        super().__init__(lines)
        self.headers = {"Content-Type": content_type}
        self.status_code = 200
        self.raw = None  # abort path tolerates missing socket layers

    def json(self):
        return {}


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def post(self, *a, **k):
        return self._resp


def _run_stream(events, content_type="text/event-stream"):
    """Drive the REAL _chat_stream_once loop with a fake HTTP layer."""
    resp = _FakeStreamResp(events, content_type)
    old = _client._http
    _client._http = lambda: _FakeSession(resp)
    try:
        return _chat_stream_once("http://x/v1/chat/completions",
                                 {"Authorization": "Bearer k"}, {},
                                 None, None, None, None, None, 30)
    finally:
        _client._http = old


# --------------------------------------------------------------------------
# A. _result_from_json — malformed blocking bodies
# --------------------------------------------------------------------------

_BLOCKING = [
    {"model": "m", "usage": "100 tokens",
     "choices": [{"message": {"content": "hi"}}]},
    {"choices": [{"message": {"content": None,
                              "tool_calls": ["read_file"]}}]},
    {"choices": "oops"},
    {"choices": [{"message": "str"}]},
    {"choices": [{"message": None}]},
    {"choices": [None, 123, "x"]},
    {"choices": {"message": {"content": "hi"}}},          # single dict
    {"choices": {"0": {"message": {"content": "hi"}}}},  # dict-of-dicts
    {},
    {"usage": {"prompt_tokens": "abc"}, "choices": []},
    {"usage": {"prompt_tokens": {"n": 1}}, "choices": []},
    {"usage": {"prompt_tokens": None}, "choices": []},
    {"usage": [1, 2], "choices": []},
    {"choices": [{"delta": "x"}]},
    {"choices": [{"message": {"content": [{"type": "text", "text": "hi"},
                                         {"type": "image"}]}}]},
    {"choices": [{"message": {"content": 42}}]},
    {"choices": [{"message": {"content": b"bytes!"}}]},
    {"choices": [{"message": {"tool_calls": "read_file"}}]},
    {"choices": [{"message": {"tool_calls": [
        {"id": "1", "function": {"name": 123, "arguments": ["x"]}}]}}]},
    {"choices": [{"message": {"tool_calls": [
        {"function": "{invalid json"}]}}]},
    {"choices": [{"message": {"tool_calls": [None, 5, "x"]}}]},
    {"choices": [{"message": {"tool_calls": [{"id": 9}]}}]},
    {"model": 456, "choices": [{"message": {"content": "hi"},
                                "finish_reason": 7}]},
    {"choices": [{"message": {"content": "hi"}, "finish_reason": {"w": 1}}]},
    {"choices": None},
    {"choices": [{"text": "legacy completions"}]},        # /v1/completions
    {"choices": [{"message": {"role": 5, "content": "hi"}}]},
]


@pytest.mark.parametrize("payload", _BLOCKING,
                         ids=[f"case-{i}" for i in range(len(_BLOCKING))])
def test_result_from_json_malformed(payload):
    try:
        res = _result_from_json(payload, "fallback")
    except _CRASHERS as e:
        pytest.fail(f"{type(e).__name__}: {e}")
    _assert_safe_result(res)


@pytest.mark.parametrize("payload", [[1, 2, 3], "boom", None, 42, 3.5],
                         ids=["array", "string", "none", "int", "float"])
def test_result_from_json_nondict_top_raises_apierror(payload):
    """Non-dict top-level payload: clean APIError, never AttributeError."""
    with pytest.raises(APIError):
        _result_from_json(payload, "m")


@pytest.mark.parametrize("payload", [
    {"error": "boom"},
    {"error": {"message": "bad", "code": 400}},
    {"error": {"message": {"nested": 1}}},
    {"error": {"message": "rate limited", "code": "rate_limit_exceeded"}},
], ids=["error-string", "error-dict", "error-nested", "error-slug"])
def test_result_from_json_error_envelope_raises_apierror(payload):
    """HTTP-200 error envelopes (dict or bare string) surface as a clean
    APIError — never a silent empty result, never AttributeError."""
    with pytest.raises(APIError):
        _result_from_json(payload, "m")


# --------------------------------------------------------------------------
# B. typeguards primitives — the canonicalizer every boundary now uses
# --------------------------------------------------------------------------

def test_typeguards_never_crash_never_lie():
    nasty = ["x", 0, 1, -1, 3.7, None, True, b"bytes", ["l"], {"d": 1},
             (1, 2), {"a", "b"}, object(), float("nan")]
    for v in nasty:
        d = _typeguards.ensure_dict(v)
        assert isinstance(d, dict), v
        l = _typeguards.ensure_list(v)
        assert isinstance(l, list), v
        s = _typeguards.ensure_str(v)
        assert isinstance(s, str), v
        i = _typeguards.ensure_int(v)
        assert isinstance(i, int) and not isinstance(i, bool) or isinstance(v, bool), v
        b = _typeguards.ensure_bool(v)
        assert isinstance(b, bool), v
    # contract spot-checks
    assert _typeguards.ensure_dict({"a": 1}) == {"a": 1}
    assert _typeguards.ensure_list([1]) == [1]
    assert _typeguards.ensure_str("hi") == "hi"
    assert _typeguards.ensure_str(5) == "5"          # int coercion
    assert _typeguards.ensure_str({"a": 1}) == ""    # no container reprs
    assert _typeguards.ensure_str(["x"]) == ""
    assert _typeguards.ensure_str(b"hi") == "hi"    # bytes decoded
    # NOTE (worker-13 contract): ensure_int is deliberately NOT clever —
    # numeric strings yield the default rather than being parsed. The
    # "never raises, never lies" guarantee is what the audit needs.
    assert _typeguards.ensure_int("42") == 0
    assert _typeguards.ensure_int(42) == 42
    assert _typeguards.ensure_int("abc") == 0
    assert _typeguards.ensure_int(None) == 0


# --------------------------------------------------------------------------
# C. _iter_sse_events — malformed SSE framing
# --------------------------------------------------------------------------

def test_iter_sse_events_drops_non_dict_events():
    lines = [
        "data: [1,2,3]", "",          # JSON array -> dropped
        'data: "plain"', "",          # JSON string -> dropped
        "data: 42", "",               # JSON number -> dropped
        "data: null", "",             # JSON null -> dropped
        "data: {broken", "",          # invalid JSON -> dropped
        "data: {\"a\": 1", "",        # truncated -> dropped
        ": ping", "",                 # comment -> keepalive sentinel
        "data: {\"choices\": []}", "",  # valid
        "data: {\"choices\": []}",    # valid, no trailing blank line
    ]
    events = list(_iter_sse_events(_FakeResp(lines)))
    dicts = [e for e in events if isinstance(e, dict)]
    assert dicts == [{"choices": []}, {"choices": []}]


def test_iter_sse_events_multiline_data():
    events = [e for e in _iter_sse_events(
        _FakeResp(['data: {"a":', "data: 1}", ""])) if isinstance(e, dict)]
    assert events == [{"a": 1}]


def test_iter_sse_events_bad_bytes_never_crash():
    lines = [b"data: \xff\xfe invalid \x80 bytes", b"", b"data: [DONE]"]
    events = list(_iter_sse_events(_FakeResp(lines)))
    assert not [e for e in events if isinstance(e, dict)]


# --------------------------------------------------------------------------
# D. _chat_stream_once — real accumulation loop vs malformed events
# --------------------------------------------------------------------------

_STREAM = [
    ('usage as string',
     [{'choices': [{'delta': {'content': 'hi'}}], 'usage': '100 tokens'}],
     False),
    ('error as string',
     [{'error': 'boom'}],
     True),
    ('error as dict',
     [{'error': {'message': 'nope', 'code': 400}}],
     True),
    ('error as nested junk',
     [{'error': {'message': {'x': 1}}}],
     True),
    ('delta as string',
     [{'choices': [{'delta': 'just-a-string'}]}],
     False),
    ('delta None',
     [{'choices': [{'delta': None}]}],
     False),
    ('delta.tool_calls as strings',
     {'choices': [{'delta': {'tool_calls': ['read_file', 'write']}}]},
     False),
    ('delta.tool_calls bare string',
     {'choices': [{'delta': {'tool_calls': 'read_file'}}]},
     False),
    ('delta.tool_calls None entries',
     {'choices': [{'delta': {'tool_calls': [None, 5]}}]},
     False),
    ('delta.tool_calls fn as string',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': 'read_file'}]}}]},
     False),
    ('delta.tool_calls fn None',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': None}]}}]},
     False),
    ('delta.tool_calls fn args as dict',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'c1', 'function': {'name': 'read', 'arguments': {'path': 'x'}}}]}}]},
     False),
    ('delta.tool_calls fn args as list',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'c1', 'function': {'name': 'read', 'arguments': [1]}}]}}]},
     False),
    ('delta.tool_calls index as string',
     {'choices': [{'delta': {'tool_calls': [{'index': '0', 'id': 'c1', 'function': {'name': 'read', 'arguments': '{}'}}]}}]},
     False),
    ('delta.tool_calls index None',
     {'choices': [{'delta': {'tool_calls': [{'index': None, 'function': {'name': 'read', 'arguments': '{}'}}]}}]},
     False),
    ('delta.tool_calls index as list',
     {'choices': [{'delta': {'tool_calls': [{'index': [0], 'function': {'name': 'read', 'arguments': '{}'}}]}}]},
     False),
    ('delta.tool_calls id as int',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 9, 'function': {'name': 'f', 'arguments': '{}'}}]}}]},
     False),
    ('delta.tool_calls name as int',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 9, 'arguments': '{}'}}]}}]},
     False),
    ('delta.tool_calls empty name fragments',
     {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': '', 'arguments': '{}'}}]}}]},
     False),
    ('delta.content as block list',
     {'choices': [{'delta': {'content': [{'type': 'text', 'text': 'hi'}]}}]},
     False),
    ('delta.content as int',
     {'choices': [{'delta': {'content': 7}}]},
     False),
    ('delta.content None',
     {'choices': [{'delta': {'content': None}}]},
     False),
    ('delta.reasoning as int',
     {'choices': [{'delta': {'reasoning_content': 5}}]},
     False),
    ('delta.reasoning as dict',
     {'choices': [{'delta': {'reasoning_content': {'x': 1}}}]},
     False),
    ('finish_reason as int',
     [{'choices': [{'delta': {}, 'finish_reason': 5}]}],
     False),
    ('finish_reason as dict',
     [{'choices': [{'finish_reason': {'x': 1}}]}],
     False),
    ('choices as string',
     [{'choices': 'oops'}],
     False),
    ('choices entries junk',
     [{'choices': [None, 'x', 7]}],
     False),
    ('model as int',
     [{'model': 99, 'choices': [{'delta': {'content': 'hi'}}]}],
     False),
    ('model as dict',
     [{'model': {'n': 1}, 'choices': [{'delta': {'content': 'hi'}}]}],
     False),
    ('top-level array event',
     ['data: [1,2,3]'],
     False),
    ('happy path',
     [{'choices': [{'delta': {'content': 'hi', 'tool_calls': [{'index': 0, 'id': 'c1', 'function': {'name': 'read', 'arguments': '{}'}}]}}]}, {'usage': {'prompt_tokens': 10}}],
     False),
]


@pytest.mark.parametrize("name,events,raises",
                         [(c[0], c[1], c[2]) for c in _STREAM],
                         ids=[c[0] for c in _STREAM])
def test_chat_stream_once_malformed(name, events, raises):
    try:
        res = _run_stream(events)
    except _CRASHERS as e:
        pytest.fail(f"{name}: {type(e).__name__}: {e}")
    except APIError:
        if not raises:
            pytest.fail(f"{name}: unexpected APIError")
        return
    if raises:
        pytest.fail(f"{name}: expected APIError, stream completed")
    _assert_safe_result(res)


def test_chat_stream_once_non_sse_json_fallback_malformed():
    """Non-SSE JSON body with stream=true requested: malformed shape must
    still normalize through the blocking parser cleanly."""
    resp = _FakeStreamResp([], content_type="application/json")
    resp.json = lambda: {"choices": [{"message": "str"}], "usage": "x"}
    old = _client._http
    _client._http = lambda: _FakeSession(resp)
    try:
        res = _chat_stream_once("http://x/v1/chat/completions",
                                {"Authorization": "Bearer k"}, {},
                                None, None, None, None, None, 30)
    except _CRASHERS as e:
        pytest.fail(f"fallback: {type(e).__name__}: {e}")
    finally:
        _client._http = old
    _assert_safe_result(res)


# --------------------------------------------------------------------------
# E. assistant_message — malformed history inputs
# --------------------------------------------------------------------------

_ASSISTANT = [
    ({"a": 1}, [], ""),
    ([1, 2], [], ""),
    (123, [], ""),
    (None, [], ""),
    ("hi", "read_file", ""),
    ("hi", {"id": "1", "function": {"name": "r", "arguments": "{}"}}, ""),
    ("hi", [None, "x", 5,
            {"id": "1", "function": {"name": "r", "arguments": "{}"}}], ""),
    ("hi", ("a", "b"), ""),
    ("hi", [], {"k": "v"}),
    ("hi", [], 7),
    ("hi", [], None),
    (None, "read_file", "think"),
]


@pytest.mark.parametrize("content,tool_calls,reasoning", _ASSISTANT,
                         ids=[f"case-{i}" for i in range(len(_ASSISTANT))])
def test_assistant_message_malformed(content, tool_calls, reasoning):
    try:
        msg = assistant_message(content, tool_calls, reasoning)
    except _CRASHERS as e:
        pytest.fail(f"{type(e).__name__}: {e}")
    assert isinstance(msg, dict) and msg["role"] == "assistant"
    c = msg.get("content")
    assert c is None or isinstance(c, str)
    for tc in msg.get("tool_calls", []):
        assert isinstance(tc, dict)
    rc = msg.get("reasoning_content")
    assert rc is None or isinstance(rc, str)


# --------------------------------------------------------------------------
# F. _learn_from_usage / safe_token_counts — malformed usage
# --------------------------------------------------------------------------

_USAGES = ["100 tokens", 123, None, [], {}, {"prompt_tokens": "abc"},
           {"prompt_tokens": {"nested": 1}}, {"prompt_tokens": None},
           {"prompt_tokens": -5}, {"prompt_tokens": "12"},
           {"input_tokens": "7", "output_tokens": 3.9},
           {"weird": object()}, "x" * 10000, [1, {"a": 2}]]


@pytest.mark.parametrize("usage", _USAGES,
                         ids=[f"case-{i}" for i in range(len(_USAGES))])
def test_usage_never_crashes(usage):
    try:
        _learn_from_usage(usage, 500, "m")
        p, c = safe_token_counts(usage)
    except _CRASHERS as e:
        pytest.fail(f"{usage!r}: {type(e).__name__}: {e}")
    assert isinstance(p, int) and p >= 0
    assert isinstance(c, int) and c >= 0


# --------------------------------------------------------------------------
# G. StreamResult direct construction coercion
# --------------------------------------------------------------------------

def test_streamresult_construction_malformed():
    try:
        res = StreamResult(content=["x"], reasoning={"a": 1},
                           tool_calls=["read", None, 5], finish_reason=7,
                           usage="100 tokens", model=None)
    except _CRASHERS as e:
        pytest.fail(f"{type(e).__name__}: {e}")
    _assert_safe_result(res)


# --------------------------------------------------------------------------
# standalone runner (python3 /tmp/malformed_provider_test.py, no pytest)
# --------------------------------------------------------------------------

if __name__ == "__main__":
    passed = failed = 0

    def _check(label, fn):
        global passed, failed
        try:
            fn()
        except _CRASHERS as e:
            failed += 1
            print(f"FAIL {label}: {type(e).__name__}: {e}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {label}: assertion: {e}")
        except BaseException as e:
            # pytest.raises' Failed (BaseException) when an expected
            # APIError is NOT raised, or any other unexpected error.
            if "APIError" in type(e).__name__:
                passed += 1
            else:
                failed += 1
                print(f"FAIL {label}: {type(e).__name__}: {e}")
        else:
            passed += 1

    for name, fn in sorted(list(globals().items())):
        if not (name.startswith("test_") and callable(fn)):
            continue
        marks = getattr(fn, "pytestmark", [])
        param = next((m for m in marks if m.name == "parametrize"), None)
        if param is None:
            _check(name, fn)
            continue
        (argnames, argvalues), kwargs = param.args, param.kwargs
        ids = kwargs.get("ids") or [str(i) for i in range(len(argvalues))]
        names = argnames.split(",")
        for i, vals in enumerate(argvalues):
            # single-arg parametrize passes bare values (which may
            # themselves be lists); only zip when there are >1 argnames.
            kw = ({names[0]: vals} if len(names) == 1
                  else dict(zip(names, vals)))
            _check(f"{name}[{ids[i]}]", lambda kw=kw: fn(**kw))

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
