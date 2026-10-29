"""Rigorous test battery for _sanitize_messages tool-call pairing.

Covers: orphan calls, orphan responses, out-of-order messages,
duplicate ids, empty ids, degenerate messages, reasoning_content.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import copy

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


from fullagent.client import _sanitize_messages


def _call(cid, name="read", args="{}"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args}}


def _assistant(calls=None, content=None):
    m = {"role": "assistant"}
    if content is not None:
        m["content"] = content
    if calls is not None:
        m["tool_calls"] = calls
    return m


def _tool(tid, content="ok"):
    m = {"role": "tool", "content": content}
    if tid is not None:
        m["tool_call_id"] = tid
    return m


def _user(content="hi"):
    return {"role": "user", "content": content}


def _pair_ids(messages):
    """Assert pairing invariants; return (call_ids, response_ids)."""
    call_ids = []
    for m in messages:
        for c in m.get("tool_calls", []) or []:
            call_ids.append(str(c["id"]))
    resp_ids = [str(m["tool_call_id"]) for m in messages
                if m.get("role") == "tool"]
    # non-empty, unique call ids
    assert all(call_ids), f"empty call id in {call_ids}"
    assert len(set(call_ids)) == len(call_ids), f"dup call ids {call_ids}"
    # every call has exactly one response, every response has a call
    assert sorted(call_ids) == sorted(resp_ids), \
        f"calls {call_ids} vs responses {resp_ids}"
    # order: each response comes after its issuing message
    first_issue = {}
    for i, m in enumerate(messages):
        for c in m.get("tool_calls", []) or []:
            first_issue.setdefault(str(c["id"]), i)
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            tid = str(m["tool_call_id"])
            assert first_issue[tid] < i, f"response {tid} before its call"
    return call_ids, resp_ids


# 1. valid history is untouched -------------------------------------------
def test_valid_pair_untouched():
    msgs = [_user(), _assistant([_call("a")], "thinking"),
            _tool("a"), _assistant(content="done")]
    msgs[1]["reasoning_content"] = "r"  # present -> nothing to heal
    before = copy.deepcopy(msgs)
    assert _sanitize_messages(msgs) is False
    assert msgs == before
    _pair_ids(msgs)


# 2. orphan call, NO tool messages at all (the `or not tool_response_ids`
#    bug: old code kept everything when the response set was empty) -------
def test_orphan_call_no_tool_messages_dropped():
    msgs = [_user("do x"), _assistant([_call("a")], "on it")]
    assert _sanitize_messages(msgs) is True
    assert msgs[1].get("tool_calls") is None
    assert msgs[1]["content"] == "on it"  # content preserved
    _pair_ids(msgs)


# 3. orphan call alongside a valid pair ------------------------------------
def test_orphan_call_dropped_valid_pair_kept():
    msgs = [_user(),
            _assistant([_call("a"), _call("orphan")], "x"),
            _tool("a"),
            _user("more")]
    assert _sanitize_messages(msgs) is True
    assert [c["id"] for c in msgs[1]["tool_calls"]] == ["a"]
    _pair_ids(msgs)


# 4. orphan tool response (no matching call anywhere) ----------------------
def test_orphan_response_dropped():
    msgs = [_user(), _tool("ghost"), _assistant(content="hi")]
    assert _sanitize_messages(msgs) is True
    assert len(msgs) == 2
    assert all(m.get("role") != "tool" for m in msgs)


# 5. response whose call was stripped becomes orphan and is dropped --------
def test_response_dropped_after_call_stripped():
    # call "a" has an id but its message also carries a malformed call
    # that normalizes away; "a" itself is fine — instead: assistant
    # message with calls but the RESPONSE references a different id.
    msgs = [_user(),
            _assistant([_call("a")], "x"),
            _tool("a"), _tool("stray")]
    assert _sanitize_messages(msgs) is True
    assert [m.get("tool_call_id") for m in msgs
            if m.get("role") == "tool"] == ["a"]
    _pair_ids(msgs)


# 6. out-of-order: response BEFORE its issuing call ------------------------
def test_out_of_order_response_dropped():
    msgs = [_user(),
            _tool("a"),                       # response before the call
            _assistant([_call("a")], "x")]
    assert _sanitize_messages(msgs) is True
    # response dropped (it preceded its call); the call is now an orphan
    # and must be dropped too
    assert all(m.get("role") != "tool" for m in msgs)
    asst = [m for m in msgs if m.get("role") == "assistant"][0]
    assert asst.get("tool_calls") is None
    _pair_ids(msgs)


# 7. duplicate call ids across two assistant messages ----------------------
def test_duplicate_call_ids_second_dropped():
    msgs = [_user(),
            _assistant([_call("a")], "one"),
            _tool("a"),
            _user("again"),
            _assistant([_call("a")], "two"),   # duplicate id
            _tool("a")]
    assert _sanitize_messages(msgs) is True
    calls = [c["id"] for m in msgs for c in m.get("tool_calls", []) or []]
    assert calls == ["a"]
    # exactly one response survives (first issuer wins)
    resps = [m for m in msgs if m.get("role") == "tool"]
    assert len(resps) == 1
    _pair_ids(msgs)


# 8. duplicate responses for the same call id ------------------------------
def test_duplicate_responses_second_dropped():
    msgs = [_user(),
            _assistant([_call("a")], "x"),
            _tool("a", "first"),
            _tool("a", "second")]
    assert _sanitize_messages(msgs) is True
    resps = [m for m in msgs if m.get("role") == "tool"]
    assert len(resps) == 1 and resps[0]["content"] == "first"
    _pair_ids(msgs)


# 9. empty-id call ----------------------------------------------------------
def test_empty_id_call_dropped():
    msgs = [_user(),
            _assistant([_call(""), _call("a")], "x"),
            _tool("a")]
    assert _sanitize_messages(msgs) is True
    assert [c["id"] for c in msgs[1]["tool_calls"]] == ["a"]
    _pair_ids(msgs)


# 10. tool message with missing tool_call_id --------------------------------
def test_idless_tool_message_dropped():
    msgs = [_user(),
            _assistant([_call("a")], "x"),
            _tool("a"),
            _tool(None, "mystery")]
    assert _sanitize_messages(msgs) is True
    assert all("tool_call_id" in m for m in msgs
               if m.get("role") == "tool")
    _pair_ids(msgs)


# 11. degenerate empty assistant message dropped after stripping ------------
def test_empty_assistant_message_dropped():
    msgs = [_user("do x"),
            _assistant([_call("orphan")]),   # no content, orphan call
            _user("hello?")]
    assert _sanitize_messages(msgs) is True
    assert len(msgs) == 2
    assert [m["role"] for m in msgs] == ["user", "user"]
    _pair_ids(msgs)


# 12. reasoning_content added to surviving tool turn ------------------------
def test_reasoning_content_added():
    msgs = [_user(),
            _assistant([_call("a")]),        # no reasoning, no content
            _tool("a")]
    # call "a" is paired so the assistant message survives (it has calls)
    assert _sanitize_messages(msgs) is True
    assert msgs[1]["reasoning_content"] == ""
    _pair_ids(msgs)


# 13. multi-turn valid history stays valid -----------------------------------
def test_multi_turn_valid_untouched():
    msgs = [_user("a"),
            _assistant([_call("a1"), _call("a2")], "r"),
            _tool("a1"), _tool("a2"),
            _user("b"),
            _assistant([_call("b1")], "r2"),
            _tool("b1"),
            _assistant(content="done")]
    msgs[1]["reasoning_content"] = "r"
    msgs[5]["reasoning_content"] = "r2"
    before = copy.deepcopy(msgs)
    assert _sanitize_messages(msgs) is False
    assert msgs == before
    _pair_ids(msgs)


# 14. mutation is in-place (callers rely on the same list object) ------------
def test_mutates_in_place():
    msgs = [_user(), _assistant([_call("x")], "y")]
    ident = id(msgs)
    _sanitize_messages(msgs)
    assert id(msgs) == ident


# 15. malformed (string) tool_calls normalized, orphan dropped ---------------
def test_string_tool_call_handled():
    msgs = [_user(),
            _assistant(["not-a-dict"], "x"),
            _assistant(content="fine")]
    assert _sanitize_messages(msgs) is True
    assert msgs[1].get("tool_calls") is None
    _pair_ids(msgs)


# 16. response for a call issued by a LATER assistant message ----------------
def test_response_for_later_call_dropped():
    msgs = [_user(),
            _assistant([_call("late")], "second"),
            _tool("late"),                    # valid for "late"...
            _assistant(content="first")]
    # reorder so the response precedes nothing valid: here it IS after
    # its issuer, so it stays. Flip: response before issuer.
    msgs2 = [_user(),
             _tool("late"),                    # before issuer
             _assistant([_call("late")], "x")]
    assert _sanitize_messages(msgs2) is True
    assert all(m.get("role") != "tool" for m in msgs2)
    asst = [m for m in msgs2 if m.get("role") == "assistant"][0]
    assert asst.get("tool_calls") is None
    _pair_ids(msgs2)
    # and the in-order variant only gains reasoning_content
    msgs[1]["reasoning_content"] = "r"
    before = copy.deepcopy(msgs)
    assert _sanitize_messages(msgs) is False
    assert msgs == before


# 17. empty history / no tool content at all ----------------------------------
def test_plain_history_untouched():
    msgs = [_user("hi"), _assistant(content="hello")]
    before = copy.deepcopy(msgs)
    assert _sanitize_messages(msgs) is False
    assert msgs == before


# 18. non-dict entries are skipped, not crashed on ---------------------------
def test_non_dict_entries_skipped():
    msgs = [_user(), "junk", None,
            _assistant([_call("a")], "x"), _tool("a")]
    msgs[3]["reasoning_content"] = "r"
    assert _sanitize_messages(msgs) is False  # pairs valid; junk skipped
    _pair_ids([m for m in msgs if isinstance(m, dict)])


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
