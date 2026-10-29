"""StreamResult field-type validation tests (provider-boundary validation).

Malformed provider data previously caused
`AttributeError: 'str' object has no attribute ...`.

StreamResult (fullagent/client.py) is the provider result container with
fields: content, reasoning, tool_calls, finish_reason, usage, model.
Providers and cassette replays can hand it wrong-typed values
(usage as a string, tool_calls as ["string"], content as None), and the
dataclass used to accept them silently — the crash then happened
*downstream* (tui.py `turn.usage.get(...)`, agent.py `tc["function"]`).

These tests prove:
  1. constructing StreamResult with wrong types coerces every field to
     its declared type (usage dict-or-None, tool_calls list-of-dicts,
     content/reasoning/model str, finish_reason str-or-None),
  2. the exact downstream read patterns from the crash report do not
     raise on the coerced result,
  3. _result_from_json coerces malformed provider payloads the same way.

Run:  python3 -m pytest tests/test_streamresult_validation.py -q
  or:  python3 tests/test_streamresult_validation.py
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import os
import sys
import unittest

# allow `python tests/test_streamresult_validation.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from fullagent.client import StreamResult, _result_from_json


def _wrong_typed() -> StreamResult:
    """Every field given a wrong type, like a sloppy provider would."""
    return StreamResult(
        content=None,                       # type: ignore[arg-type]
        reasoning=123,                      # type: ignore[arg-type]
        tool_calls=["string", 42, None],    # type: ignore[arg-type]
        finish_reason=5,                    # type: ignore[arg-type]
        usage="not a dict",                 # type: ignore[arg-type]
        model={"m": 1},                     # type: ignore[arg-type]
    )


class TestCoercion(unittest.TestCase):
    def test_wrong_types_coerced_at_construction(self):
        r = _wrong_typed()
        self.assertIsInstance(r.content, str)
        self.assertEqual(r.content, "")
        self.assertIsInstance(r.reasoning, str)
        self.assertEqual(r.reasoning, "")
        self.assertIsInstance(r.model, str)
        self.assertEqual(r.model, "")
        # usage: dict-or-None only — a non-empty string is truthy and
        # would sail past `if usage` guards into usage.get()
        self.assertIsNone(r.usage)
        # tool_calls: always a list of dicts; junk entries dropped
        self.assertIsInstance(r.tool_calls, list)
        self.assertEqual(r.tool_calls, [])
        self.assertTrue(all(isinstance(tc, dict)
                            for tc in r.tool_calls))
        # finish_reason: str-or-None; non-str junk coerces to ""
        self.assertIsInstance(r.finish_reason, str)

    def test_mixed_tool_calls_keep_valid_drop_junk(self):
        good = {"id": "c1", "type": "function",
                "function": {"name": "read", "arguments": "{}"}}
        r = StreamResult(tool_calls=["junk", good, None])  # type: ignore[arg-type]
        self.assertEqual(r.tool_calls, [good])

    def test_non_list_tool_calls_becomes_empty(self):
        r = StreamResult(tool_calls="nope")  # type: ignore[arg-type]
        self.assertEqual(r.tool_calls, [])
        r = StreamResult(tool_calls={"a": 1})  # type: ignore[arg-type]
        self.assertEqual(r.tool_calls, [])

    def test_finish_reason_none_stays_none(self):
        self.assertIsNone(StreamResult(finish_reason=None).finish_reason)
        self.assertEqual(StreamResult(finish_reason="stop").finish_reason,
                         "stop")

    def test_valid_values_untouched(self):
        tc = {"id": "c1", "type": "function",
              "function": {"name": "read", "arguments": "{}"}}
        r = StreamResult(content="hi", reasoning="r", tool_calls=[tc],
                         finish_reason="stop",
                         usage={"prompt_tokens": 3}, model="m1")
        self.assertEqual(r.content, "hi")
        self.assertEqual(r.reasoning, "r")
        self.assertEqual(r.tool_calls, [tc])
        self.assertEqual(r.finish_reason, "stop")
        self.assertEqual(r.usage, {"prompt_tokens": 3})
        self.assertEqual(r.model, "m1")

    def test_defaults_are_typed(self):
        r = StreamResult()
        self.assertEqual(r.content, "")
        self.assertEqual(r.reasoning, "")
        self.assertEqual(r.tool_calls, [])
        self.assertIsNone(r.finish_reason)
        self.assertIsNone(r.usage)
        self.assertEqual(r.model, "")

    def test_normalize_idempotent_and_returns_self(self):
        r = StreamResult(content="x", usage={"a": 1})
        self.assertIs(r.normalize(), r)
        r.normalize()
        self.assertEqual(r.content, "x")
        self.assertEqual(r.usage, {"a": 1})

    def test_post_construction_assignment_repaired_by_normalize(self):
        # Direct attribute assignment bypasses __post_init__; normalize()
        # (called before both parse paths return) must repair it.
        r = StreamResult()
        r.usage = "string-usage"    # type: ignore[assignment]
        r.tool_calls = ["nope"]     # type: ignore[assignment]
        r.content = None            # type: ignore[assignment]
        r.model = 7                 # type: ignore[assignment]
        r.normalize()
        self.assertIsNone(r.usage)
        self.assertEqual(r.tool_calls, [])
        self.assertEqual(r.content, "")
        self.assertEqual(r.model, "")


class TestDownstreamNoCrash(unittest.TestCase):
    """Replay the exact downstream read patterns from the crash report
    against a wrong-typed StreamResult — none may raise."""

    def test_usage_get_pattern(self):
        # tui.py: turn.usage.get("prompt_tokens", 0); agent._emit_cost
        r = _wrong_typed()
        usage = r.usage  # agent.py does: turn.usage = result.usage
        tin = (usage.get("prompt_tokens", 0) or 0) if usage else 0
        tout = (usage.get("completion_tokens", 0) or 0) if usage else 0
        self.assertEqual((tin, tout), (0, 0))

    def test_tool_call_iteration_pattern(self):
        # agent.py: for tc in result.tool_calls: fn = tc["function"]
        r = _wrong_typed()
        names = [tc["function"].get("name", "") for tc in r.tool_calls]
        self.assertEqual(names, [])

    def test_content_concat_pattern(self):
        # agent.py: turn.assistant_text += result.content
        #          turn.reasoning += result.reasoning
        r = _wrong_typed()
        text = ""
        text += r.content
        text += r.reasoning
        self.assertEqual(text, "")

    def test_cassette_style_construction(self):
        # agent.py cassette replay passes stored values straight in
        stored = {"content": None, "reasoning": None,
                  "tool_calls": "broken", "usage": [1, 2]}
        r = StreamResult(content=stored.get("content", ""),
                         reasoning=stored.get("reasoning", ""),
                         tool_calls=stored.get("tool_calls", []),
                         usage=stored.get("usage"),
                         model="m")
        self.assertEqual(r.content, "")
        self.assertEqual(r.reasoning, "")
        self.assertEqual(r.tool_calls, [])
        self.assertIsNone(r.usage)
        # ... and the downstream reads must not crash
        _ = r.usage.get("x", 0) if r.usage else 0
        _ = [tc["function"] for tc in r.tool_calls]


class TestResultFromJson(unittest.TestCase):
    def test_malformed_payload_coerced(self):
        data = {
            "model": 123,
            "usage": "lots",
            "choices": [{
                "message": {
                    "content": ["part", {"type": "text", "text": "hi"}],
                    "reasoning_content": 42,
                    "tool_calls": ["bad",
                                   {"id": "c1", "type": "function",
                                    "function": {"name": "r",
                                                 "arguments": "{}"}}],
                },
                "finish_reason": 7,
            }],
        }
        r = _result_from_json(data, "fallback")
        self.assertIsInstance(r.content, str)
        self.assertIsInstance(r.reasoning, str)
        self.assertIsInstance(r.model, str)
        self.assertIsNone(r.usage)
        self.assertIsInstance(r.finish_reason, str)
        self.assertTrue(all(isinstance(tc, dict) for tc in r.tool_calls))
        # downstream reads must not crash
        _ = r.usage.get("prompt_tokens", 0) if r.usage else 0
        _ = [tc["function"]["name"] for tc in r.tool_calls]

    def test_choices_as_string(self):
        r = _result_from_json({"choices": "nope"}, "m")
        self.assertEqual(r.content, "")
        self.assertIsNone(r.finish_reason)

    def test_message_as_string(self):
        r = _result_from_json({"choices": [{"message": "just a string"}]},
                              "m")
        self.assertEqual(r.content, "")
        self.assertIsInstance(r.content, str)

    def test_no_choices(self):
        r = _result_from_json({"model": "m"}, "fallback")
        self.assertEqual(r.model, "m")
        self.assertEqual(r.content, "")

    def test_wellformed_untouched(self):
        data = {"model": "m1", "usage": {"prompt_tokens": 5},
                "choices": [{"message": {"content": "hello"},
                             "finish_reason": "stop"}]}
        r = _result_from_json(data, "fallback")
        self.assertEqual(r.content, "hello")
        self.assertEqual(r.model, "m1")
        self.assertEqual(r.usage, {"prompt_tokens": 5})
        self.assertEqual(r.finish_reason, "stop")


if __name__ == "__main__":
    unittest.main(verbosity=2)
