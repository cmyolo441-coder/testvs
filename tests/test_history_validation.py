"""History validation diagnostics test battery (sprint worker 12/20).

Drives the REAL validate_history() in fullagent.client against ~20
malformed histories. Proves each diagnostic code fires exactly where it
should — so the next `invalid_request_error` from the provider comes
with a precise, index-stamped explanation instead of a guessing game.

Also covers:
  * clean histories produce ZERO diagnostics (no false positives)
  * integration: build_payload() logs the diagnostics before every send
    (warning level; offending message dumped in debug mode)
  * validate_history never raises, whatever the input shape

Run:  python3 -m pytest tests/test_history_validation.py -q
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import logging
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
    HISTORY_MAX_MESSAGE_CHARS,
    HistoryDiagnostic,
    validate_history,
)
import fullagent.client as _client


# -- helpers ----------------------------------------------------------------

def _codes(diags):
    return [d.code for d in diags]


def _tc(tid, name="read", args="{}"):
    return {"id": tid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _assistant_with_call(tid, content="let me look"):
    return {"role": "assistant", "content": content,
            "tool_calls": [_tc(tid)]}


def _tool(tid, content="result"):
    return {"role": "tool", "tool_call_id": tid, "content": content}


CLEAN = [
    {"role": "system", "content": "be helpful"},
    {"role": "user", "content": "list files"},
    _assistant_with_call("call_1"),
    _tool("call_1", "a.txt\nb.txt"),
    {"role": "assistant", "content": "Here are your files."},
]


def _fresh_clean():
    """Deep-ish copy so tests never mutate the shared CLEAN."""
    import copy
    return copy.deepcopy(CLEAN)


# -- (a) non-dict messages -----------------------------------------------------

def test_a1_string_message():
    h = _fresh_clean()
    h.insert(1, "i am not a dict")
    diags = validate_history(h)
    assert "non_dict_message" in _codes(diags)
    assert any(d.index == 1 for d in diags if d.code == "non_dict_message")


def test_a2_none_and_int_messages():
    h = [None, 42, {"role": "user", "content": "hi"}]
    diags = validate_history(h)
    assert _codes(diags).count("non_dict_message") == 2
    assert all(d.severity == "error" for d in diags
               if d.code == "non_dict_message")


# -- (b) unknown roles ---------------------------------------------------------

def test_b1_unknown_role():
    h = [{"role": "admin", "content": "x"}]
    assert "unknown_role" in _codes(validate_history(h))


def test_b2_missing_role():
    h = [{"content": "no role here"}]
    assert "unknown_role" in _codes(validate_history(h))


def test_b3_function_role_flagged():
    # legacy OpenAI "function" role — our client never emits it
    h = [{"role": "function", "name": "f", "content": "x"}]
    assert "unknown_role" in _codes(validate_history(h))


# -- (c) orphan tool_calls -----------------------------------------------------

def test_c1_orphan_tool_call():
    h = [
        {"role": "user", "content": "do it"},
        _assistant_with_call("call_orphan"),
    ]
    diags = validate_history(h)
    assert "orphan_tool_call" in _codes(diags)
    d = next(d for d in diags if d.code == "orphan_tool_call")
    assert d.index == 1 and d.severity == "error"
    assert "call_orphan" in d.detail


def test_c2_answered_call_is_not_orphan():
    assert "orphan_tool_call" not in _codes(validate_history(_fresh_clean()))


# -- (d) tool responses without a preceding assistant tool_call -----------------

def test_d1_orphan_tool_response():
    h = [
        {"role": "user", "content": "hi"},
        _tool("call_ghost"),
    ]
    diags = validate_history(h)
    assert "orphan_tool_response" in _codes(diags)


def test_d2_tool_response_missing_id():
    h = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "result without id"},
    ]
    diags = validate_history(h)
    assert "tool_response_missing_id" in _codes(diags)


def test_d3_tool_response_out_of_order():
    # the tool answer comes BEFORE the assistant message that issued it
    h = [
        {"role": "user", "content": "hi"},
        _tool("call_1"),
        _assistant_with_call("call_1"),
    ]
    diags = validate_history(h)
    codes = _codes(diags)
    assert "tool_response_out_of_order" in codes
    assert "orphan_tool_response" not in codes  # id WAS issued — precision


def test_d4_duplicate_tool_response():
    h = _fresh_clean()
    h.insert(4, _tool("call_1", "second answer"))
    assert "duplicate_tool_response_id" in _codes(validate_history(h))


# -- (e) empty assistant content WITH tool_calls --------------------------------

def test_e1_null_content_with_tool_calls():
    m = _assistant_with_call("call_1")
    m["content"] = None
    h = [{"role": "user", "content": "x"}, m, _tool("call_1")]
    diags = validate_history(h)
    assert "empty_content_with_tool_calls" in _codes(diags)
    d = next(d for d in diags if d.code == "empty_content_with_tool_calls")
    assert d.severity == "warning"  # some providers reject, not all


def test_e2_empty_string_content_with_tool_calls():
    m = _assistant_with_call("call_1", content="")
    h = [{"role": "user", "content": "x"}, m, _tool("call_1")]
    assert "empty_content_with_tool_calls" in _codes(validate_history(h))


def test_e3_content_present_no_diagnostic():
    assert "empty_content_with_tool_calls" not in _codes(
        validate_history(_fresh_clean()))


# -- (f) duplicate tool_call ids --------------------------------------------------

def test_f1_duplicate_tool_call_id():
    h = [
        {"role": "user", "content": "x"},
        _assistant_with_call("call_dup"),
        _assistant_with_call("call_dup"),
        _tool("call_dup"),
    ]
    diags = validate_history(h)
    assert "duplicate_tool_call_id" in _codes(diags)
    d = next(d for d in diags if d.code == "duplicate_tool_call_id")
    assert d.index == 2 and d.severity == "error"


def test_f2_missing_tool_call_id():
    h = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "doing",
         "tool_calls": [{"type": "function",
                         "function": {"name": "f", "arguments": "{}"}}]},
    ]
    assert "tool_call_missing_id" in _codes(validate_history(h))


# -- (g) oversize messages ------------------------------------------------------

def test_g1_oversize_message():
    h = [{"role": "user",
          "content": "x" * (HISTORY_MAX_MESSAGE_CHARS + 100)}]
    diags = validate_history(h)
    assert "oversize_message" in _codes(diags)
    d = next(d for d in diags if d.code == "oversize_message")
    assert d.severity == "warning"


def test_g2_normal_size_no_diagnostic():
    assert "oversize_message" not in _codes(validate_history(_fresh_clean()))


# -- clean histories -------------------------------------------------------------

def test_clean_history_zero_diagnostics():
    assert validate_history(_fresh_clean()) == []


def test_empty_history_zero_diagnostics():
    assert validate_history([]) == []


def test_history_not_a_list():
    diags = validate_history("not a list")  # type: ignore[arg-type]
    assert _codes(diags) == ["history_not_a_list"]


# -- integration: build_payload logs diagnostics ---------------------------------

def _test_model():
    return _client.Model(id="m", provider="k", label="m")


def _test_effort():
    return _client.Effort("high", "HIGH", "#50fa7b", 4096, 0.6, None,
                          "test effort")


def test_build_payload_logs_diagnostics(caplog, monkeypatch):
    model, effort = _test_model(), _test_effort()
    # FullAgent's configured logger owns its stderr handler and disables
    # propagation. Re-enable it for this capture so caplog sees the record.
    monkeypatch.setattr(logging.getLogger("fullagent"), "propagate", True)
    h = [
        {"role": "user", "content": "hi"},
        _assistant_with_call("call_orphan"),  # (c) orphan
    ]
    with caplog.at_level(logging.WARNING, logger="fullagent.client"):
        _client.build_payload(model, effort, h, None, stream=False)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("orphan_tool_call" in m for m in msgs), msgs


def test_build_payload_debug_dumps_offender(caplog, monkeypatch):
    model, effort = _test_model(), _test_effort()
    monkeypatch.setattr(logging.getLogger("fullagent"), "propagate", True)
    h = [{"role": "bogus", "content": "MARKER_XYZ"}]
    with caplog.at_level(logging.DEBUG, logger="fullagent.client"):
        _client.build_payload(model, effort, h, None, stream=False)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("unknown_role" in m for m in msgs), msgs
    assert any("MARKER_XYZ" in m for m in msgs), msgs  # offender dumped


def test_clean_history_logs_nothing(caplog, monkeypatch):
    model, effort = _test_model(), _test_effort()
    monkeypatch.setattr(logging.getLogger("fullagent"), "propagate", True)
    with caplog.at_level(logging.DEBUG, logger="fullagent.client"):
        _client.build_payload(model, effort, _fresh_clean(), None,
                              stream=False)
    assert not [r for r in caplog.records
                if "history validation" in r.getMessage()]


# -- validate_history never raises -----------------------------------------------

@pytest.mark.parametrize("bad", [
    [None], [42], [["nested"]], [{"role": "assistant", "tool_calls": None}],
    [{"role": "assistant", "tool_calls": "notalist"}],
    [{"role": "assistant", "tool_calls": [{"id": 123}]}],
    [{"role": "tool", "tool_call_id": 0}],
    [{"role": "assistant", "content": {"weird": ["shape"]}}],
    [{"role": "user", "content": b"bytes"}],
])
def test_never_raises(bad):
    diags = validate_history(bad)
    assert isinstance(diags, list)
    assert all(isinstance(d, HistoryDiagnostic) for d in diags)


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
