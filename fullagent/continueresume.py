"""Continue-from-summary — "continue" starts a FRESH turn, not a replay.

Root cause (worker 11/20): when a turn hit the 25-tool-call cap and the
user said "continue", ``run_turn`` kept the ENTIRE capped-turn history in
``agent.messages`` (~25 assistant tool_call messages + ~25 tool result
messages, routinely 50k+ chars of stale context). The provider answered
the bloated follow-up request with ``invalid_request_error``.

The fix: when the user says "continue" after a capped turn, the history
is rewritten to::

    [system prompt, original user request, assistant summary of prior work]

and the normal flow then appends the user's "continue" message. The model
resumes from the summary instead of re-reading 25 calls of history.

The summary is deterministic and extractive — no model call needed:

* user goal (the original request, verbatim, capped)
* tools used: one line per tool call — name, key args, status,
  short result preview
* files touched
* the assistant's last partial reply (if any)
* pending items: why the turn stopped + which tools errored

Nothing is lost for audit purposes: the event log still holds the full
history; only the *model-visible* context is reseeded.

Public API:

* :func:`is_continue_request` — is this user text a "continue"?
* :func:`summarize_capped_turn` — deterministic extractive summary text.
* :func:`seed_continue_history` — duck-typed agent hook; call at the top
  of ``run_turn``. Returns True when the history was reseeded.

``python3 -m fullagent.continueresume`` runs the built-in self-test,
including the <10% token-size proof.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# continue detection
# ---------------------------------------------------------------------------

_CONTINUE_RE = re.compile(
    r"^\s*(please\s+)?(continue|resume|keep\s+going|carry\s+on|go\s+on|"
    r"proceed|carry\s+on\s+please|yes\s+continue|do\s+continue)"
    r"\s*[.!…]*\s*$",
    re.IGNORECASE,
)


def is_continue_request(text: str) -> bool:
    """True when the user text is a bare "continue"-style request."""
    return bool(text) and bool(_CONTINUE_RE.match(str(text)))


# a previous turn counts as "capped" when its error is one of the stop
# messages that invite the user to continue
_CAP_MARKERS = (
    "Turn stopped after",          # 25 tool calls
    "stopped after",               # "... tool iterations"
)


def _is_capped_stop(error: str) -> bool:
    if not error:
        return False
    err = error.strip()
    if err.startswith("Turn stopped after"):
        return True
    if "stopped after" in err and ("tool calls" in err
                                   or "tool iterations" in err):
        return True
    return False


# ---------------------------------------------------------------------------
# deterministic extractive summary
# ---------------------------------------------------------------------------

_SUM_GOAL_CAP = 300
_SUM_TOOL_CAP = 5         # max tool lines in the summary (newest-first window)
_SUM_RESULT_PREVIEW = 60
_SUM_TEXT_CAP = 1600      # hard ceiling for the whole summary body

_ARG_KEYS = ("path", "file", "filename", "target", "command", "cmd",
             "pattern", "query", "url", "prompt", "content", "text")


def _cap(s: str, n: int) -> str:
    s = str(s or "")
    s = " ".join(s.split())  # collapse whitespace/newlines to one line
    return s if len(s) <= n else s[:n] + "…"


def _args_preview(args: Any) -> str:
    """Most informative 1-2 args for a tool call, human readable."""
    if not isinstance(args, dict) or not args:
        return ""
    bits = []
    for key in _ARG_KEYS:
        val = args.get(key)
        if val:
            bits.append(f"{key}={_cap(val, 60)}")
            if len(bits) == 2:
                break
    if not bits:  # fall back to the first two keys
        for key in list(args)[:2]:
            bits.append(f"{key}={_cap(args[key], 40)}")
    return " ".join(bits)


def summarize_capped_turn(turn: Any) -> str:
    """Build the deterministic summary text for a capped turn.

    Duck-typed on Turn: user_text, assistant_text, tools (ToolEvents
    with name/args/result/status), error.
    """
    tools = list(getattr(turn, "tools", []) or [])
    lines: list[str] = []
    files: list[str] = []

    goal = _cap(getattr(turn, "user_text", "") or "", _SUM_GOAL_CAP)
    lines.append(
        "[SUMMARY OF PRIOR TURN — that turn hit the tool-call cap and "
        "stopped. Continue EXACTLY where it left off; do not restart "
        "from scratch.]")
    lines.append("")
    lines.append(f"User goal: {goal}")
    lines.append("")
    lines.append(f"Work done ({len(tools)} tool calls):")

    # files from ALL calls (not just the summary window)
    for ev in tools:
        args = getattr(ev, "args", {}) or {}
        if isinstance(args, dict):
            for key in ("path", "file", "filename", "target"):
                val = args.get(key)
                if val and str(val) not in files:
                    files.append(str(val))

    errors: list[str] = []
    # newest calls matter most for resuming — keep the tail of the list
    window = tools[-_SUM_TOOL_CAP:]
    skipped = len(tools) - len(window)
    if skipped:
        lines.append(f"  … {skipped} earlier calls omitted (see event log)")
    for i, ev in enumerate(window, skipped + 1):
        name = getattr(ev, "name", "") or "?"
        args = getattr(ev, "args", {}) or {}
        status = getattr(ev, "status", "") or "?"
        result = getattr(ev, "result", "") or ""
        ap = _args_preview(args)
        lines.append(
            f"  {i}. {name} {ap} — {status} — "
            f"{_cap(result, _SUM_RESULT_PREVIEW)}")
        if status == "error":
            errors.append(f"{name}({_cap(_args_preview(args), 40)})")

    if files:
        lines.append("")
        lines.append(f"Files touched: {', '.join(files[:10])}")

    tail = _cap(getattr(turn, "assistant_text", "") or "", 200)
    if tail:
        lines.append("")
        lines.append(f"Assistant's last partial reply: {tail}")

    lines.append("")
    err = (getattr(turn, "error", "") or "").strip()
    lines.append(f"Why it stopped: {err or 'tool-call cap reached'}")
    if errors:
        lines.append(f"Calls that errored ({len(errors)}): "
                     f"{', '.join(errors[:6])}")
    lines.append("")
    lines.append("PENDING: whatever in the user goal above is not yet "
                 "done. Pick up from the last completed step — do not "
                 "repeat successful work.")

    body = "\n".join(lines)
    return body if len(body) <= _SUM_TEXT_CAP else body[:_SUM_TEXT_CAP] + "…"


# ---------------------------------------------------------------------------
# agent hook
# ---------------------------------------------------------------------------


def seed_continue_history(agent: Any, user_text: str) -> bool:
    """Reseed the model-visible history when the user says "continue".

    Duck-typed on the Agent: ``turns`` (list of Turn), ``messages``
    (list of message dicts), ``log`` (optional, event log).

    Returns True when the history was replaced with
    ``[system, original user request, assistant summary]``. The caller
    then appends the user's "continue" message as usual, so the final
    shape is [system, user(original), assistant(summary), user("continue")].

    Returns False (history untouched) when:
      * the user text is not a continue request, or
      * there is no previous turn, or
      * the previous turn did not stop on the tool-call cap.
    """
    if not is_continue_request(user_text):
        return False
    turns = getattr(agent, "turns", None) or []
    if not turns:
        return False
    prev = turns[-1]
    if not _is_capped_stop(getattr(prev, "error", "") or ""):
        return False
    messages = getattr(agent, "messages", None)
    if not isinstance(messages, list) or not messages:
        return False

    summary = summarize_capped_turn(prev)

    # keep only the leading system message(s); everything else — the full
    # capped-turn history — is replaced by the summary
    kept: list[dict] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system" and not kept:
            kept.append(m)
        elif kept and isinstance(m, dict) and m.get("role") == "system":
            kept.append(m)
        else:
            break

    original_request = (getattr(prev, "user_text", "") or "").strip()
    new_messages: list[dict] = list(kept)
    new_messages.append({"role": "user", "content": original_request})
    new_messages.append({
        "role": "assistant",
        "content": summary,
    })
    agent.messages = new_messages

    # SOVEREIGN ENFORCEMENT: if index 0 wasn't a system message, `kept`
    # was empty and the history now has NO prompt at all — re-seat it.
    try:
        if not agent.messages or agent.messages[0].get("role") != "system":
            agent._reseat_system_prompt()
    except Exception:
        pass

    log = getattr(agent, "log", None)
    if log is not None and hasattr(log, "append"):
        try:
            log.append("turn.continue_reseeded", {
                "dropped_messages": len(messages) - len(kept),
                "kept_messages": len(kept),
                "summary_chars": len(summary),
                "prev_tool_calls": len(getattr(prev, "tools", []) or []),
            }, actor="kernel")
        except Exception:  # noqa: BLE001 — logging never breaks a turn
            pass
    return True


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------


def _check(cond: bool, label: str) -> None:
    if not cond:
        raise AssertionError(f"FAILED: {label}")
    print(f"  ok: {label}")


def _make_fake_agent(n_calls: int = 25):
    """Simulate an agent whose last turn hit the tool-call cap."""
    from types import SimpleNamespace

    system = {"role": "system", "content": "SYS " + "x" * 4000}
    messages = [system,
                {"role": "user",
                 "content": "Refactor the payment module to use Stripe."}]
    tools = []
    for i in range(n_calls):
        messages.append({
            "role": "assistant", "tool_calls": [{
                "id": f"call_{i}",
                "function": {"name": "run_command",
                             "arguments": '{"cmd": "pytest -x"}'}}]})
        messages.append({
            "role": "tool", "tool_call_id": f"call_{i}",
            "content": f"test output {i} " + "y" * 2000})
        tools.append(SimpleNamespace(
            name="run_command" if i % 2 else "write_file",
            args={"cmd": "pytest -x"} if i % 2 else {"path": f"pay_{i}.py"},
            result=("FAILED test_auth" if i % 5 == 0 else "ok") + " z" * 500,
            status="error" if i % 5 == 0 else "done"))
    prev = SimpleNamespace(
        user_text="Refactor the payment module to use Stripe.",
        assistant_text="Working through the refactor…",
        tools=tools,
        error="Turn stopped after 25 tool calls — ask me to continue.")
    events = []

    class _Log:
        def append(self, name, data=None, **kw):
            events.append(name)

    return SimpleNamespace(messages=messages, turns=[prev], log=_Log()), \
        events


def _estimate(obj: Any) -> int:
    try:
        from .client import estimate_tokens
        return estimate_tokens(obj, "")
    except Exception:  # noqa: BLE001
        import json
        return len(json.dumps(obj, default=str)) // 4


def _selftest() -> None:
    print("continueresume self-test:")

    # --- detection ---
    for yes in ("continue", "Continue", "  CONTINUE  ", "continue!",
                "please continue", "yes continue", "resume", "keep going",
                "carry on", "go on", "proceed"):
        _check(is_continue_request(yes), f"detects {yes!r}")
    for no in ("continue the refactor with tests", "don't continue",
               "hello", "", "what about continue?"):
        _check(not is_continue_request(no),
               f"rejects {no!r}")

    # --- reseed replaces the full 25-call history ---
    agent, events = _make_fake_agent(25)
    before = _estimate(agent.messages)
    assert before > 0
    _check(seed_continue_history(agent, "continue"),
           "reseeds after capped turn")
    msgs = agent.messages
    _check(len(msgs) == 3, f"3 messages after reseed (got {len(msgs)})")
    _check(msgs[0]["role"] == "system", "system prompt kept")
    _check(msgs[1]["role"] == "user"
           and "Stripe" in msgs[1]["content"], "original request kept")
    _check(msgs[2]["role"] == "assistant"
           and "SUMMARY OF PRIOR TURN" in msgs[2]["content"],
           "summary is the assistant message")
    summary = msgs[2]["content"]
    _check("User goal:" in summary, "summary carries the user goal")
    _check("25 tool calls" in summary, "summary counts the tool calls")
    _check("Files touched:" in summary, "summary lists files touched")
    _check("PENDING" in summary, "summary states pending items")
    _check("pay_0.py" in summary, "summary names real files from args")
    _check(len(summary) <= 1600, f"summary capped ({len(summary)} chars)")
    _check("turn.continue_reseeded" in events, "log event recorded")

    after = _estimate(msgs)
    ratio = after / before
    print(f"  size: before={before} tokens, after={after} tokens "
          f"({ratio:.1%})")
    _check(ratio < 0.10,
           f"continued turn sends <10% of original history ({ratio:.1%})")

    # --- no reseed when it shouldn't fire ---
    agent2, _ = _make_fake_agent(25)
    n0 = len(agent2.messages)
    _check(not seed_continue_history(agent2, "what's next?"),
           "no reseed on non-continue text")
    _check(len(agent2.messages) == n0, "history untouched")
    agent3, _ = _make_fake_agent(25)
    agent3.turns[0].error = ""  # previous turn finished normally
    _check(not seed_continue_history(agent3, "continue"),
           "no reseed when previous turn was not capped")
    agent4, _ = _make_fake_agent(25)
    agent4.turns = []  # brand-new session
    _check(not seed_continue_history(agent4, "continue"),
           "no reseed with no previous turn")

    print("PASS: continueresume self-test — all checks passed")


if __name__ == "__main__":
    _selftest()
