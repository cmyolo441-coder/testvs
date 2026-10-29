"""Empty-content + tool_calls test battery (sprint worker 16/20).

Some strict OpenAI-compatible gateways
reject assistant turns that carry tool_calls with empty/null content:
  - content:null  -> 400 "expected a string, got null"
  - content:""    -> 400 "text content blocks must be non-empty"

The canonical fix: assistant turns with tool_calls ALWAYS carry a
non-empty string content — a single-space placeholder " " when the
model produced no text. This file proves:

  A. assistant_message() applies the invariant at build time
     (None / "" / missing-key shapes, passthrough of real text,
      reasoning_content still attached, malformed calls handled).
  B. _sanitize_messages() repairs stale/foreign history at the wire
     (None / "" / missing key / non-string content), never raises,
     and leaves valid messages untouched.
  C. build_payload() — the real send path — emits a JSON-serializable
     payload where every assistant turn with tool_calls has non-empty
     string content.

Run:  python3 -m pytest tests/test_empty_content_toolcalls.py -q
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import copy
import json
import sys

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


from fullagent.client import (
    _CONTENT_PLACEHOLDER,
    _sanitize_messages,
    assistant_message,
    build_payload,
    validate_history,
)
import fullagent.client as _client
from fullagent.config import Effort


# -- helpers ----------------------------------------------------------------

def _tc(tid="call_1", name="read", args="{}"):
    return {"id": tid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _tool(tid="call_1", content="result"):
    return {"role": "tool", "tool_call_id": tid, "content": content}


def _hist(assistant_msg):
    """Valid tool-turn history around one assistant message."""
    return [
        {"role": "user", "content": "look at this"},
        assistant_msg,
        _tool("call_1"),
    ]


def _model():
    return _client.Model(id="kilo-auto/free", provider="kilo",
                         label="t")


def _effort():
    return Effort("high", "HIGH", "#50fa7b", None, 0.6, None, "d")


# ==========================================================================
# A. assistant_message() — canonical fix at build time
# ==========================================================================

class TestAssistantMessage:
    def test_none_content_with_tool_calls_gets_placeholder(self):
        m = assistant_message(None, [_tc()], "")
        assert m["content"] == " " == _CONTENT_PLACEHOLDER
        assert isinstance(m["content"], str)
        assert m["tool_calls"] == [_tc()]

    def test_empty_string_content_with_tool_calls_gets_placeholder(self):
        m = assistant_message("", [_tc()], "")
        assert m["content"] == " "

    def test_real_text_preserved_verbatim(self):
        m = assistant_message("I'll read the file now.", [_tc()], "")
        assert m["content"] == "I'll read the file now."

    def test_placeholder_passthrough(self):
        # Already the placeholder — not mangled into something else.
        m = assistant_message(" ", [_tc()], "")
        assert m["content"] == " "

    def test_whitespace_content_kept(self):
        m = assistant_message("  \n ", [_tc()], "")
        assert m["content"] == "  \n "

    def test_none_content_without_tool_calls_unchanged(self):
        # Out of scope: a bare empty turn is a different failure class
        # (the sanitizer drops content-less, call-less assistant turns).
        m = assistant_message(None, [], "")
        assert m["content"] is None
        assert "tool_calls" not in m

    def test_empty_string_without_tool_calls_unchanged(self):
        m = assistant_message("", [], "")
        assert m["content"] == ""
        assert "tool_calls" not in m

    def test_reasoning_content_still_attached(self):
        m = assistant_message(None, [_tc()], "thinking hard")
        assert m["content"] == " "
        assert m["reasoning_content"] == "thinking hard"

    def test_empty_reasoning_content_fallback(self):
        m = assistant_message(None, [_tc()], "")
        assert m["content"] == " "
        assert m["reasoning_content"] == ""

    def test_dict_content_coerced_not_placeholder(self):
        m = assistant_message({"a": 1}, [_tc()], "")
        assert isinstance(m["content"], str) and m["content"] != " "
        assert json.loads(m["content"]) == {"a": 1}

    def test_numeric_content_coerced_not_placeholder(self):
        m = assistant_message(5, [_tc()], "")
        assert m["content"] == "5"

    def test_all_malformed_calls_dropped_no_placeholder(self):
        # Garbage tool_calls normalize away -> no calls -> no placeholder.
        m = assistant_message(None, ["read", None, 5], "")
        assert "tool_calls" not in m
        assert m["content"] is None

    def test_never_raises_on_garbage(self):
        for content, calls, reasoning in [
            (None, None, None),
            (object(), [{"x": object()}], object()),
            (b"\xff\xfe", "read_file", b"\x80"),
        ]:
            m = assistant_message(content, calls, reasoning)  # must not raise
            assert m["role"] == "assistant"
            if m.get("tool_calls"):
                c = m["content"]
                assert isinstance(c, str) and c != ""

    def test_wire_invariant_property(self):
        # The invariant: tool_calls present => non-empty str content.
        cases = [None, "", " ", "text", {"k": "v"}, [1], 0, True]
        for content in cases:
            m = assistant_message(content, [_tc()], "")
            if m.get("tool_calls"):
                assert isinstance(m["content"], str), content
                assert m["content"] != "", content


# ==========================================================================
# B. _sanitize_messages() — repair pass for stale/foreign history
# ==========================================================================

class TestSanitizeMessages:
    def test_none_content_repaired(self):
        h = _hist({"role": "assistant", "content": None,
                   "tool_calls": [_tc()]})
        assert _sanitize_messages(h) is True
        assert h[1]["content"] == " "

    def test_empty_string_content_repaired(self):
        h = _hist({"role": "assistant", "content": "",
                   "tool_calls": [_tc()]})
        assert _sanitize_messages(h) is True
        assert h[1]["content"] == " "

    def test_missing_content_key_repaired(self):
        h = _hist({"role": "assistant", "tool_calls": [_tc()]})
        assert "content" not in h[1]
        assert _sanitize_messages(h) is True
        assert h[1]["content"] == " "

    def test_non_string_content_coerced(self):
        h = _hist({"role": "assistant",
                   "content": [{"type": "text", "text": "hi"}],
                   "tool_calls": [_tc()]})
        assert _sanitize_messages(h) is True
        assert isinstance(h[1]["content"], str) and h[1]["content"] != ""

    def test_valid_message_untouched(self):
        h = _hist({"role": "assistant", "content": "doing it",
                   "tool_calls": [_tc()],
                   "reasoning_content": "r"})
        assert _sanitize_messages(h) is False
        assert h[1]["content"] == "doing it"

    def test_placeholder_content_not_re_repaired(self):
        h = _hist({"role": "assistant", "content": " ",
                   "tool_calls": [_tc()],
                   "reasoning_content": "r"})
        assert _sanitize_messages(h) is False
        assert h[1]["content"] == " "

    def test_non_assistant_messages_untouched(self):
        h = [{"role": "user", "content": None},
             {"role": "system", "content": ""}]
        before = copy.deepcopy(h)
        _sanitize_messages(h)
        assert h == before

    def test_call_less_empty_assistant_left_alone(self):
        # Out of scope for this worker: a content-less, call-less
        # assistant turn is NOT repaired here (it never carried
        # tool_calls). Locked in as-is — the "content or tool_calls
        # must be set" class belongs to a different worker.
        h = [{"role": "user", "content": "x"},
             {"role": "assistant", "content": None}]
        before = copy.deepcopy(h)
        _sanitize_messages(h)
        assert h == before

    def test_no_tool_calls_anywhere_no_fix(self):
        h = [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "hi"}]
        assert _sanitize_messages(h) is False

    def test_never_raises(self):
        bad = [None, "x", 5, [{"role": "assistant"}],
               [{"role": "assistant", "content": None,
                 "tool_calls": [None, "x", {}]}],
               [{"role": "assistant", "tool_calls": [{"id": "c",
                 "function": {"name": "f", "arguments": object()}}]}]]
        for h in bad:
            if isinstance(h, list):
                _sanitize_messages(h)  # must not raise


# ==========================================================================
# C. build_payload() — the real wire path
# ==========================================================================

class TestBuildPayload:
    def test_wire_payload_never_empty_content_with_calls(self):
        h = _hist({"role": "assistant", "content": None,
                   "tool_calls": [_tc()]})
        p = build_payload(_model(), _effort(), h, None)
        for m in p["messages"]:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                assert isinstance(m["content"], str)
                assert m["content"] != ""
        json.dumps(p)  # payload must be wire-serializable

    def test_stale_session_shape_repaired_on_wire(self):
        # Simulates a session/cassette predating the invariant:
        # content key missing entirely, no reasoning fields.
        h = [{"role": "user", "content": "go"},
             {"role": "assistant", "tool_calls": [_tc("call_9")]},
             {"role": "tool", "tool_call_id": "call_9",
              "content": "done"}]
        p = build_payload(_model(), _effort(), h, None)
        asm = next(m for m in p["messages"]
                   if m.get("role") == "assistant"
                   and m.get("tool_calls"))
        assert asm["content"] == " "
        assert "reasoning_content" in asm

    def test_clean_history_passes_through(self):
        h = [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "here you go"}]
        p = build_payload(_model(), _effort(), h, None)
        assert p["messages"][1]["content"] == "here you go"


# ==========================================================================
# D. validate_history diagnostics — no false positives on the placeholder
# ==========================================================================

def _diag_codes(history):
    return [d.code for d in validate_history(history)]


class TestDiagnostics:
    def test_none_content_still_warns(self):
        h = _hist({"role": "assistant", "content": None,
                   "tool_calls": [_tc()]})
        assert "empty_content_with_tool_calls" in _diag_codes(h)

    def test_empty_string_still_warns(self):
        h = _hist({"role": "assistant", "content": "",
                   "tool_calls": [_tc()]})
        assert "empty_content_with_tool_calls" in _diag_codes(h)

    def test_placeholder_does_not_warn(self):
        # The canonical " " placeholder is wire-safe; warning on it
        # would flag every tool turn after the fix.
        h = _hist({"role": "assistant", "content": " ",
                   "tool_calls": [_tc()]})
        assert "empty_content_with_tool_calls" not in _diag_codes(h)

    def test_builtin_message_shape_does_not_warn(self):
        # End to end: what assistant_message() builds for a text-less
        # tool turn must not trip the diagnostic.
        m = assistant_message(None, [_tc()], "")
        assert "empty_content_with_tool_calls" not in _diag_codes(_hist(m))


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
