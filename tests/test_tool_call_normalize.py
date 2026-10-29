"""Worker 7/20 — tool_calls normalization: prove ONE canonical shape
everywhere. Feeds garbage (strings, None, ints, partial dicts, dicts
with string `function`) through normalize_tool_call and asserts the
output is always a valid dict or skipped (None)."""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fullagent.client import normalize_tool_call, _result_from_json


def _valid(tc):
    """Assert the canonical shape — mirrors what downstream .get() calls need."""
    assert isinstance(tc, dict), f"not a dict: {tc!r}"
    assert set(tc) == {"id", "type", "function"}, f"bad keys: {tc!r}"
    assert isinstance(tc["id"], str) and tc["id"], f"bad id: {tc!r}"
    assert tc["type"] == "function", f"bad type: {tc!r}"
    fn = tc["function"]
    assert isinstance(fn, dict), f"function not a dict: {tc!r}"
    assert set(fn) == {"name", "arguments"}, f"bad fn keys: {tc!r}"
    assert isinstance(fn["name"], str) and fn["name"], f"bad name: {tc!r}"
    assert isinstance(fn["arguments"], str), f"arguments not str: {tc!r}"
    return tc


CASES = [
    # (input, expectation: "valid" -> valid dict; "skip" -> None; plus optional checks)
    (None, "skip"),
    ("", "skip"),
    ("call_1", "skip"),                      # bare string is not a tool call
    (42, "skip"),
    (3.14, "skip"),
    (["x"], "skip"),
    ({"id": "a"}, "skip"),                   # no function at all
    ({"function": None}, "skip"),
    ({"function": 42}, "skip"),
    ({"function": {}}, "skip"),              # no name
    ({"function": {"name": ""}}, "skip"),    # empty name
    ({"function": {"name": None}}, "skip"),
    ({"function": {"name": 7}}, "skip"),     # non-str name
    # --- valid recoveries ---
    ({"id": "c1", "function": {"name": "read", "arguments": '{"p":1}'}},
     "valid"),
    ({"function": {"name": "read"}}, "valid"),          # missing id -> fallback id
    ({"id": None, "function": {"name": "read", "arguments": None}},
     "valid"),                                          # None id/args
    ({"id": 123, "function": {"name": "read", "arguments": {"p": 1}}},
     "valid"),                                          # int id, dict args -> JSON str
    ({"function": '{"name": "run", "arguments": "{\\"c\\":\\"ls\\"}"}'},
     "valid"),                                          # JSON-string function payload
    ({"function": "run"}, "valid"),                     # plain string -> name
    ({"function": '{"broken json'}, "valid"),           # bad JSON string -> name
    ({"id": "x", "type": "function", "extra": 1,
      "function": {"name": "w", "arguments": "{}", "extra2": 2}},
     "valid"),                                          # extra keys stripped
]


def test_normalizer_garbage():
    for i, (inp, expect) in enumerate(CASES):
        out = normalize_tool_call(inp)
        if expect == "skip":
            assert out is None, f"case {i}: {inp!r} should be skipped, got {out!r}"
        else:
            _valid(out)
    print(f"PASS: {len(CASES)} garbage inputs -> always valid dict or skipped")


def test_normalizer_id_uniqueness():
    outs = [normalize_tool_call({"function": {"name": "x"}}) for _ in range(50)]
    ids = [o["id"] for o in outs]
    assert len(set(ids)) == 50, "fallback ids must be unique"
    print("PASS: 50 missing-id calls -> 50 unique fallback ids")


def test_normalizer_dict_args_become_json():
    out = normalize_tool_call({"function": {"name": "w",
                                             "arguments": {"path": "a.txt"}}})
    _valid(out)
    assert json.loads(out["function"]["arguments"]) == {"path": "a.txt"}
    print("PASS: dict arguments json-encoded (not Python repr)")


def test_normalizer_json_string_function():
    out = normalize_tool_call({"id": "s1", "function":
        '{"name": "run", "arguments": "{\\"cmd\\": \\"ls\\"}"}'})
    _valid(out)
    assert out["function"]["name"] == "run"
    assert json.loads(out["function"]["arguments"]) == {"cmd": "ls"}
    print("PASS: JSON-string function payload parsed")


def test_result_from_json_garbage_tool_calls():
    """End-to-end: provider JSON body with garbage tool_calls must not crash
    and must only produce valid dicts on result.tool_calls."""
    body = {"model": "m", "choices": [{
        "finish_reason": "tool_calls",
        "message": {"content": "",
                    "tool_calls": [
                        "a string", None, 7,
                        {"id": "ok1", "function": {"name": "read",
                                                  "arguments": '{"p":1}'}},
                        {"function": '{"name": "run", "arguments": "{}"}'},
                        {"function": {"name": "", "arguments": "{}"}},
                        {"function": {"name": "w", "arguments": {"p": "x"}}},
                    ]}}]}
    result = _result_from_json(body, "m")
    assert len(result.tool_calls) == 3, result.tool_calls
    for tc in result.tool_calls:
        _valid(tc)
    # dict arguments must be real JSON, not str() repr
    w = [t for t in result.tool_calls if t["function"]["name"] == "w"][0]
    assert json.loads(w["function"]["arguments"]) == {"p": "x"}
    print("PASS: _result_from_json drops 4 garbage entries, keeps 3 valid")


def test_downstream_get_never_crashes():
    """Simulate the tool-loop downstream pattern on normalizer output."""
    for inp, _ in CASES:
        out = normalize_tool_call(inp)
        if out is None:
            continue
        fn = out["function"]          # was tc["function"] KeyError before
        name = fn.get("name", "")     # was AttributeError on str function
        args = fn.get("arguments")
        cid = out.get("id")           # was AttributeError on str tc
        assert name and isinstance(args, str) and cid
    print("PASS: downstream .get()/[] pattern never crashes on output")


if __name__ == "__main__":
    test_normalizer_garbage()
    test_normalizer_id_uniqueness()
    test_normalizer_dict_args_become_json()
    test_normalizer_json_string_function()
    test_result_from_json_garbage_tool_calls()
    test_downstream_get_never_crashes()
    print("\nALL NORMALIZER TESTS PASSED")
