"""Role-sequence validation + repair test battery (worker 15/20).

Drives the REAL _repair_role_sequence in fullagent.client against 14
broken histories plus one build_payload end-to-end check. Every case
asserts the 5 rules on the OUTPUT:

  1. first message is system or user
  2. no two consecutive assistant messages
  3. every tool message immediately follows the assistant message
     containing its tool_call_id
  4. no leading tool messages
  5. a user message exists

Run:  python3 -m pytest tests/test_role_sequence_repair.py -q
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


from fullagent.client import _repair_role_sequence, build_payload
from fullagent.config import Effort, Model


def asst(content=None, calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


def call(cid):
    return {"id": cid, "type": "function",
            "function": {"name": "bash", "arguments": "{}"}}


def tool(cid, content="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def check_rules(msgs):
    """Assert all 5 role-sequence rules hold."""
    assert msgs, "empty history"
    assert msgs[0]["role"] in ("system", "user"), \
        f"rule 1: first role is {msgs[0]['role']!r}"
    assert any(m["role"] == "user" for m in msgs), "rule 5: no user message"
    assert msgs[0]["role"] != "tool", "rule 4: leading tool message"
    for a, b in zip(msgs, msgs[1:]):
        assert not (a["role"] == "assistant" and b["role"] == "assistant"), \
            "rule 2: consecutive assistant messages"
    # rule 3: tool messages must form a contiguous block immediately
    # following the assistant holding their tool_call_id
    asst_ids = {}
    for m in msgs:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                asst_ids[str(tc["id"])] = m
    owner_pos = {id(m): i for i, m in enumerate(msgs)
                 if m["role"] == "assistant"}
    for i, m in enumerate(msgs):
        if m["role"] != "tool":
            continue
        tid = str(m.get("tool_call_id"))
        assert tid in asst_ids, f"rule 3: orphan tool {tid!r}"
        owner = asst_ids[tid]
        j = owner_pos[id(owner)]
        block = msgs[j + 1:i + 1]
        assert block and all(
            b["role"] == "tool"
            and str(b.get("tool_call_id")) in
            {str(tc["id"]) for tc in owner.get("tool_calls") or []}
            for b in block), \
            f"rule 3: tool {tid!r} not in its assistant's tool block"


def run(broken):
    msgs = [dict(m) if isinstance(m, dict) else m for m in broken]
    repairs = _repair_role_sequence(msgs)
    check_rules(msgs)
    return msgs, repairs


def test_consecutive_assistants_merged():
    msgs, repairs = run([
        {"role": "user", "content": "hi"},
        asst("first part"),
        asst("second part"),
    ])
    assert len([m for m in msgs if m["role"] == "assistant"]) == 1
    assert msgs[1]["content"] == "first part\nsecond part"
    assert repairs, "expected a repair to be logged"


def test_three_consecutive_assistants_merged():
    msgs, _ = run([
        {"role": "user", "content": "hi"},
        asst("a"), asst("b"), asst("c"),
    ])
    assert msgs[1]["content"] == "a\nb\nc"


def test_leading_tool_messages_dropped():
    msgs, repairs = run([
        tool("x1"), tool("x2"),
        {"role": "user", "content": "hi"},
    ])
    assert all(m["role"] != "tool" for m in msgs)
    assert any("leading tool" in r for r in repairs)


def test_stranded_tool_reordered_after_assistant():
    msgs, repairs = run([
        {"role": "user", "content": "run it"},
        asst("calling", calls=[call("c1")]),
        {"role": "user", "content": "oops, surgery inserted me"},
        tool("c1", "done"),
    ])
    idx = [m["role"] for m in msgs]
    a = idx.index("assistant")
    t = [i for i, m in enumerate(msgs)
         if m["role"] == "tool" and m["tool_call_id"] == "c1"][0]
    assert t == a + 1
    assert any("reordered" in r for r in repairs)


def test_tool_before_its_assistant_reordered():
    msgs, _ = run([
        {"role": "user", "content": "hi"},
        tool("c9", "early"),
        asst("calling", calls=[call("c9")]),
    ])
    idx = [m["role"] for m in msgs]
    a = idx.index("assistant")
    assert msgs[a + 1]["role"] == "tool"
    assert msgs[a + 1]["tool_call_id"] == "c9"


def test_orphan_tool_dropped():
    msgs, repairs = run([
        {"role": "user", "content": "hi"},
        asst("plain"),
        tool("ghost", "no such call"),
    ])
    assert all(m.get("tool_call_id") != "ghost" for m in msgs)
    assert any("orphan" in r for r in repairs)


def test_tool_without_call_id_dropped():
    msgs, _ = run([
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "no id"},
    ])
    assert all(m["role"] != "tool" for m in msgs)


def test_leading_assistant_gets_placeholder_user():
    msgs, repairs = run([asst("hello there")])
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert any("placeholder" in r for r in repairs)


def test_no_user_message_gets_placeholder():
    msgs, repairs = run([
        {"role": "system", "content": "be nice"},
        asst("hello"),
    ])
    assert any(m["role"] == "user" for m in msgs)
    assert any("placeholder" in r for r in repairs)


def test_empty_history_gets_placeholder_user():
    msgs, repairs = run([])
    assert msgs == [{"role": "user", "content": "(continue)"}]
    assert repairs


def test_invalid_roles_and_non_dicts_dropped():
    msgs, repairs = run([
        {"role": "user", "content": "hi"},
        {"role": "developer", "content": "bogus"},
        "not a dict",
        None,
        asst("ok"),
    ])
    assert all(isinstance(m, dict) and
               m["role"] in ("system", "user", "assistant", "tool")
               for m in msgs)
    assert len(msgs) == 2
    assert any("invalid role" in r for r in repairs)


def test_multiple_tools_reordered_in_original_order():
    msgs, _ = run([
        {"role": "user", "content": "go"},
        tool("c2", "second"),
        asst("calling", calls=[call("c1"), call("c2")]),
        tool("c1", "first"),
    ])
    idx = [m["role"] for m in msgs]
    a = idx.index("assistant")
    assert msgs[a + 1]["tool_call_id"] == "c2"   # original relative order kept
    assert msgs[a + 2]["tool_call_id"] == "c1"


def test_merged_assistants_keep_tool_calls():
    msgs, _ = run([
        {"role": "user", "content": "go"},
        asst("part one"),
        asst("part two", calls=[call("c5")]),
        tool("c5", "result"),
    ])
    assistants = [m for m in msgs if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert [tc["id"] for tc in assistants[0]["tool_calls"]] == ["c5"]


def test_valid_sequence_untouched():
    original = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "run ls"},
        asst("calling", calls=[call("c1")]),
        tool("c1", "out"),
        asst("done"),
    ]
    msgs, repairs = run(original)
    assert repairs == []
    assert [m["role"] for m in msgs] == \
        ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[3]["content"] == "out"


def test_build_payload_end_to_end():
    broken = [
        tool("z0"),
        {"role": "user", "content": "do it"},
        asst("a1"),
        asst("a2", calls=[call("z1")]),
        {"role": "user", "content": "surgery"},
        tool("z1", "res"),
    ]
    model = Model(id="t", provider="kilo", label="t",
                  supports_tools=False, supports_reasoning=False)
    effort = Effort(key="low", label="low", color="grey", max_tokens=64,
                    temperature=0.2, reasoning_effort=None,
                    description="test")
    payload = build_payload(model, effort, broken, tools=None, stream=False)
    check_rules(payload["messages"])
    assert payload["messages"][0]["role"] in ("system", "user")


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
