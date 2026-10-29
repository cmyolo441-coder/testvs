"""PERF (worker 2/20): hard per-request prompt token budget.

``prune_messages()`` in agent.py caps *message count* (window=40) but not
*token size*. Measured on a real 25-call turn: call 1 sent ~13.6k tokens,
call 19 sent ~179.6k tokens (13x growth, +25k in a single step). The
culprit is the "newest N tool results stay verbatim" rule: two 80KB tool
dumps ride along at ~25k tokens each on EVERY call until they rotate out.
Payload size oscillates wildly instead of staying bounded, and the
400-retry path in ``_complete`` sent ``self.messages`` completely unpruned
(~193k tokens at 25 calls — truly unbounded).

This module enforces a hard token ceiling on the exact payload
``Agent._complete()`` sends, so per-request cost is bounded no matter how
long the turn runs. ``self.messages`` stays canonical (checkpoints, event
log, TUI keep everything); only the wire view shrinks.

Shrink order (cheapest fidelity loss first):
  1. per-result char caps on tool outputs (newest keep a larger cap),
  2. drop oldest complete turn-groups from the front — injected system
     hints first, then oldest user/assistant+tool segments.
Never dropped: messages[0] (system prompt), the first user message (task
framing), and the final group (current turn context). assistant+tool_call
pairs are always dropped as a unit so the provider never sees an orphan
``tool`` message.
"""
from __future__ import annotations

import json

DEFAULT_BUDGET_TOKENS = 100_000   # hard per-request ceiling
MIN_BUDGET_TOKENS = 8_000         # below this the system prompt alone wins
_NEWEST_VERBATIM_CAP = 8_000      # chars per tool result, newest 2
_OLDER_RESULT_CAP = 2_000         # chars per tool result, older ones
_TRUNC_NOTE = "\n…[truncated to budget — re-read the file or re-run the command if needed]"


def _estimate_tokens(msgs: list, model_id: str = "") -> int:
    from .client import estimate_tokens_fast
    return estimate_tokens_fast(json.dumps(msgs, default=str), model_id)


def prompt_wire_tokens(model_msgs: list, schemas: list | None,
                       model_id: str = "") -> int:
    """Estimated tokens of the full request body (messages + tool schemas)."""
    return _estimate_tokens({"messages": model_msgs, "tools": schemas or []},
                            model_id)


def schemas_tokens(schemas: list | None, model_id: str = "") -> int:
    """Token cost of the tool-schema block alone (reserved from the budget)."""
    if not schemas:
        return 0
    return _estimate_tokens(schemas, model_id)


def _truncate_tool_results(msgs: list) -> int:
    """Cap every tool result to a char budget; newest two keep more.

    Returns the number of results truncated. Mutates the (copied) list.
    """
    tool_idx = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
    n = 0
    for rank, i in enumerate(reversed(tool_idx)):
        cap = _NEWEST_VERBATIM_CAP if rank < 2 else _OLDER_RESULT_CAP
        content = msgs[i].get("content", "")
        if isinstance(content, str) and len(content) > cap:
            msgs[i]["content"] = content[:cap] + _TRUNC_NOTE
            n += 1
    return n


def _droppable_groups(msgs: list) -> list[list[int]]:
    """Oldest-first index groups safe to drop.

    Excludes index 0 (system prompt), the first user message (task
    framing), and the final group (current turn context). assistant
    messages travel with their tool results so pairs never split.
    """
    n = len(msgs)
    groups: list[list[int]] = []
    i = 1
    first_user_seen = (n > 0 and msgs[0].get("role") == "user")
    while i < n:
        role = msgs[i].get("role")
        if role == "user" and not first_user_seen:
            first_user_seen = True   # task framing: never dropped
            i += 1
            continue
        if role == "system":
            groups.append([i])        # injected hints drop first
            i += 1
        elif role == "user":
            j = i + 1
            while j < n and msgs[j].get("role") in ("assistant", "tool",
                                                    "system"):
                j += 1
            groups.append(list(range(i, j)))
            i = j
        elif role == "assistant":
            j = i + 1
            while j < n and msgs[j].get("role") == "tool":
                j += 1
            groups.append(list(range(i, j)))
            i = j
        else:                          # orphan tool / unknown: drop singly
            groups.append([i])
            i += 1
    if groups:
        groups.pop()                   # keep the current turn's context
    return groups


def bound_prompt(model_msgs: list,
                 budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                 model_id: str = "",
                 reserved_tokens: int = 0) -> list:
    """Return a copy of ``model_msgs`` guaranteed (best-effort) under budget.

    ``reserved_tokens`` is the token cost of everything else on the wire
    (tool schemas) — the message view is shrunk so that
    messages + reserved <= budget. Never mutates the input. If even the
    system prompt + first user + final group exceed the budget, returns
    that minimal core unchanged.
    """
    try:
        budget = max(MIN_BUDGET_TOKENS, int(budget_tokens or 0))
    except (TypeError, ValueError):
        budget = DEFAULT_BUDGET_TOKENS
    try:
        reserved = max(0, int(reserved_tokens or 0))
    except (TypeError, ValueError):
        reserved = 0
    # shallow-copy dicts so truncation never mutates agent state
    msgs = [dict(m) if isinstance(m, dict) else m for m in model_msgs]
    over = lambda: _estimate_tokens(msgs, model_id) + reserved > budget
    if not over():
        return msgs
    _truncate_tool_results(msgs)
    if not over():
        return msgs
    shift = 0
    for group in _droppable_groups(msgs):
        if not over():
            break
        for i in sorted((g - shift for g in group), reverse=True):
            del msgs[i]
        shift += len(group)
    return msgs


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> int:
    """Simulate a 25-call turn with hostile tool outputs; prove the per-
    request payload is bounded under the budget on every call."""
    import random
    random.seed(7)
    budget = 100_000
    system = {"role": "system", "content": "S" * 2500}
    schemas = [{"type": "function",
                "function": {"name": f"tool_{i:03d}",
                             "description": "D" * 300,
                             "parameters": {"type": "object"}}}
               for i in range(118)]                      # ~41k chars, real size
    small = "OK\n" + "x" * 220
    big = "test ... ok\n" * 1200                          # ~38KB pytest dump
    huge = "y" * 80_000                                   # 80KB file read
    outs = [small] * 14 + [big] * 7 + [huge] * 4
    random.shuffle(outs)

    messages = [system, {"role": "user", "content": "do the thing"}]
    from .promptbudget import schemas_tokens as _st
    reserved = _st(schemas)
    worst, smallest = 0, 10 ** 18
    for call in range(1, 26):
        bounded = bound_prompt(messages, budget, reserved_tokens=reserved)
        toks = prompt_wire_tokens(bounded, schemas)
        worst = max(worst, toks)
        smallest = min(smallest, toks)
        assert toks <= budget, f"call {call}: {toks:,} > budget {budget:,}"
        assert bounded[0] is not system  # copied, not aliased
        assert bounded[0]["content"] == system["content"]
        assert any(m.get("role") == "user"
                   and m.get("content") == "do the thing" for m in bounded), \
            "first user message (task framing) must survive"
        # no orphan tool messages: every tool msg needs a matching
        # assistant tool_call earlier in the payload
        seen_calls: set[str] = set()
        for m in bounded:
            for tc in (m.get("tool_calls") or []):
                seen_calls.add(tc.get("id"))
            if m.get("role") == "tool":
                assert m.get("tool_call_id") in seen_calls, \
                    f"orphan tool message at call {call}"
        assert len(bounded) >= 2
        messages.append({"role": "assistant", "content": f"step {call}",
                         "tool_calls": [{"id": f"c{call}", "type": "function",
                                         "function": {"name": "run_command",
                                                      "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{call}",
                         "content": outs[call - 1]})
        if call % 6 == 0:
            messages.append({"role": "system", "content": "HINT: retry differently"})
    # input never mutated: the 80KB dump must still be intact in `messages`
    assert any(len(str(m.get("content", ""))) == 80_000 for m in messages), \
        "bound_prompt must not mutate the agent's canonical history"
    print(f"promptbudget self-test PASS — 25 calls, worst {worst:,} tok, "
          f"smallest {smallest:,} tok, budget {budget:,} "
          f"(unbounded old path reached ~193k and grew linearly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
