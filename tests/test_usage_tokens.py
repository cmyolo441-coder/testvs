"""Usage-normalisation tests for fullagent.client.usage_tokens.

Proves the single safe extractor returns valid (non-negative int,
non-negative int) tuples for EVERY usage shape providers have been seen
to return -- and never raises. Covers the reported crash:

    AttributeError: 'str' object has no attribute 'get'

from ``result.usage.get("prompt_tokens")`` when a provider returns usage
as a string (client.py / agent.py / tui.py call sites).

Also exercises the thin wrappers that now delegate to usage_tokens:
client._learn_from_usage, agent.Agent._emit_cost, jsonout._usage.

Run:  python3 -m pytest tests/test_usage_tokens.py -q
   or: python3 -m unittest tests.test_usage_tokens -v
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import unittest

from fullagent import client
from fullagent.client import usage_tokens
from fullagent import jsonout


# 12 usage shapes: every one must yield (int >= 0, int >= 0), never raise.
SHAPES = [
    ("none", None, (0, 0)),
    ("empty_dict", {}, (0, 0)),
    ("normal_dict", {"prompt_tokens": 10, "completion_tokens": 20}, (10, 20)),
    ("string", "some provider garbage", (0, 0)),
    ("empty_string", "", (0, 0)),
    ("int", 123, (0, 0)),
    ("float", 4.5, (0, 0)),
    ("list", [1, 2], (0, 0)),
    ("string_values", {"prompt_tokens": "10", "completion_tokens": "20"},
     (10, 20)),
    ("none_values", {"prompt_tokens": None, "completion_tokens": None},
     (0, 0)),
    ("wrong_keys", {"wrong": 1}, (0, 0)),
    ("junk_values", {"prompt_tokens": "abc", "completion_tokens": [1]},
     (0, 0)),
    ("aliases", {"input_tokens": 7, "output_tokens": 8}, (7, 8)),
    ("alias_plus_primary",
     {"prompt_tokens": 5, "input_tokens": 99, "completion_tokens": 6,
      "output_tokens": 99}, (5, 6)),
    ("negative", {"prompt_tokens": -3, "completion_tokens": -1}, (0, 0)),
    ("float_values", {"prompt_tokens": 10.9, "completion_tokens": 2.1},
     (10, 2)),
    ("bool_values", {"prompt_tokens": True, "completion_tokens": False},
     (1, 0)),
]


class UsageTokensTest(unittest.TestCase):
    def test_all_shapes_never_raise_and_return_valid_ints(self):
        for name, shape, expected in SHAPES:
            with self.subTest(shape=name):
                # Must never raise -- this is the reported AttributeError.
                result = usage_tokens(shape)
                self.assertIsInstance(result, tuple, name)
                self.assertEqual(len(result), 2, name)
                tin, tout = result
                self.assertIsInstance(tin, int, name)
                self.assertIsInstance(tout, int, name)
                self.assertNotIsInstance(tin, bool, name)
                self.assertNotIsInstance(tout, bool, name)
                self.assertGreaterEqual(tin, 0, name)
                self.assertGreaterEqual(tout, 0, name)
                self.assertEqual(result, expected, name)

    def test_old_crash_pattern_now_safe(self):
        # The exact reported crash: usage arrives as a string and the old
        # code called result.usage.get("prompt_tokens").
        usage = "truncated-usage-string"
        with self.assertRaises(AttributeError):
            usage.get("prompt_tokens")  # old pattern -- proves the bug existed
        tin, tout = usage_tokens(usage)  # new pattern -- safe
        self.assertEqual((tin, tout), (0, 0))

    def test_learn_from_usage_never_raises(self):
        for name, shape, _ in SHAPES:
            with self.subTest(shape=name):
                # Must never raise for any shape (previously AttributeError
                # on strings/ints, which was NOT caught by its try/except).
                client._learn_from_usage(shape, sent_chars=100,
                                         model_id="test-model")

    def test_jsonout_usage_never_raises(self):
        for name, shape, _ in SHAPES:
            with self.subTest(shape=name):
                turn = type("T", (), {"usage": shape})()
                out = jsonout._usage(turn)
                self.assertEqual(set(out), {"input_tokens", "output_tokens"})
                self.assertIsInstance(out["input_tokens"], int)
                self.assertIsInstance(out["output_tokens"], int)
                self.assertGreaterEqual(out["input_tokens"], 0)
                self.assertGreaterEqual(out["output_tokens"], 0)
        # normal dict still flows through
        turn = type("T", (), {"usage": {"prompt_tokens": 3,
                                       "completion_tokens": 4}})()
        self.assertEqual(jsonout._usage(turn),
                         {"input_tokens": 3, "output_tokens": 4})

    def test_agent_emit_cost_never_raises(self):
        from fullagent.agent import Agent

        class _Status:
            focus = "focus-1"

        class _Goal:
            def status(self):
                return _Status()

        class _Log:
            def __init__(self):
                self.events = []

            def append(self, name, data, correlation_id=None):
                self.events.append((name, data, correlation_id))

        class _Model:
            id = "test-model"

        for name, shape, expected in SHAPES:
            with self.subTest(shape=name):
                dummy = type("D", (), {})()
                dummy.goal = _Goal()
                dummy.log = _Log()
                dummy.model = _Model()
                dummy.cost_tracker = None
                # Must never raise for any shape.
                Agent._emit_cost(dummy, shape)
                tin, tout = expected
                if tin or tout:
                    self.assertEqual(len(dummy.log.events), 1, name)
                    _n, data, _c = dummy.log.events[0]
                    self.assertEqual(data["tokens_in"], tin, name)
                    self.assertEqual(data["tokens_out"], tout, name)
                else:
                    self.assertEqual(dummy.log.events, [], name)

    def test_token_accumulation_pattern(self):
        # Mirrors cumulative usage accounting: total += usage_tokens(result.usage)
        tokens_in = tokens_out = 0
        for shape in ("bad string", 42, None, {},
                      {"prompt_tokens": "15", "completion_tokens": None}):
            tin, tout = usage_tokens(shape)
            tokens_in += tin
            tokens_out += tout
        self.assertEqual((tokens_in, tokens_out), (15, 0))


if __name__ == "__main__":
    unittest.main()
