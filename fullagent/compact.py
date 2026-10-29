"""Conversation compaction (/compact) — Claude Code style.

Offline, heuristic summarization of the in-memory message list so a long
session can be squashed back down when the context window fills up.

Nothing here imports ``.agent`` — everything is duck-typed against
``agent.messages`` (a list of OpenAI-style ``{"role": ..., "content": ...}``
dicts) and ``agent.model.context_window``. Every public function is
total: on any unexpected shape it degrades to keeping the first and last
messages rather than raising.
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# tunables
# ---------------------------------------------------------------------------

TAIL_KEEP = 10               # last N messages are always kept verbatim
MAX_SUMMARY_LINES = 40       # hard cap on heuristic summary length
SUGGEST_THRESHOLD = 0.80     # suggest /compact once estimated usage passes this
DEFAULT_CONTEXT_WINDOW = 200_000
SUMMARY_MARKER = "Previous conversation summary:"

# decision-ish language in assistant text worth preserving
_DECISION_RES = [
    re.compile(r"\b(i'?ll|i have decided|we decided|decided to)\b", re.I),
    re.compile(r"\b(will now|going to|about to|planning to)\b", re.I),
    re.compile(r"(?m)^\s*(plan|decision|decisions|note|todo|next steps?)\s*:", re.I),
]

# file paths with a code/config extension mentioned in prose
_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(\[])(~?[\w\-.]+(?:/[\w\-.]+)+\.\w+|[\w\-.~]+\."
    r"(?:py|js|ts|tsx|jsx|go|rs|java|c|h|cpp|md|txt|rst|yaml|yml|json|toml|"
    r"cfg|ini|sh|bash|html|css|sql|proto))",
    re.I,
)

# tool names whose arguments carry a file path being written
_FILE_WRITE_TOOLS = {"write", "edit", "multiedit", "apply_patch", "create_file"}


# ---------------------------------------------------------------------------
# small duck-typed helpers
# ---------------------------------------------------------------------------

def _messages(agent) -> list:
    try:
        msgs = getattr(agent, "messages", None)
        return list(msgs) if msgs else []
    except Exception:
        return []


def _role(msg) -> str:
    try:
        return str(msg.get("role", "")) if isinstance(msg, dict) else ""
    except Exception:
        return ""


def _content(msg) -> str:
    try:
        if not isinstance(msg, dict):
            return ""
        c = msg.get("content")
        if c is None:
            return ""
        return c if isinstance(c, str) else str(c)
    except Exception:
        return ""


def _is_summary_msg(msg) -> bool:
    return _role(msg) == "user" and _content(msg).lstrip().startswith(SUMMARY_MARKER)


def _total_chars(messages) -> int:
    total = 0
    for m in messages:
        try:
            total += len(_content(m))
        except Exception:
            pass
    return total


def _tool_calls(msg) -> list:
    try:
        tcs = msg.get("tool_calls") if isinstance(msg, dict) else None
        return list(tcs) if tcs else []
    except Exception:
        return []


def _tool_name_and_path(tc) -> tuple[str, str | None]:
    """Return (tool name, file path arg if it looks like a write)."""
    name, path = "", None
    try:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        name = str(fn.get("name", "") or "")
        raw_args = fn.get("arguments", "")
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else {}
        if not isinstance(args, dict):
            args = {}
        for key in ("path", "file", "filepath", "file_path", "filename"):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                path = v.strip()
                break
    except Exception:
        pass
    return name, path


def _truncate(s: str, n: int = 160) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# heuristic summary
# ---------------------------------------------------------------------------

def _extract_decision_lines(middle: list) -> list[str]:
    """Pull "key decision" lines out of assistant messages in the middle chunk."""
    lines: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        line = _truncate(line)
        if line and line not in seen:
            seen.add(line)
            lines.append(line)

    for msg in middle:
        role = _role(msg)
        if role == "assistant":
            text = _content(msg)
            # 1) decision-ish prose lines
            for raw in text.splitlines():
                line = raw.strip().lstrip("-*•> ").strip()
                if len(line) < 8:
                    continue
                if any(rx.search(line) for rx in _DECISION_RES):
                    add(line)
            # 2) file paths mentioned in prose
            for m in _PATH_RE.finditer(text):
                add("Referenced file: " + m.group(1))
            # 3) tool calls made -> "Ran <tool>" (+ path for writes)
            for tc in _tool_calls(msg):
                name, path = _tool_name_and_path(tc)
                if not name:
                    continue
                if name in _FILE_WRITE_TOOLS and path:
                    add("Ran %s on %s" % (name, path))
                else:
                    add("Ran %s" % name)
        elif role == "user" and not _is_summary_msg(msg):
            # one short line per user request so intent survives compaction
            first = _content(msg).strip().splitlines()
            if first and first[0].strip():
                add("User asked: " + _truncate(first[0].strip(), 140))

    if len(lines) > MAX_SUMMARY_LINES:
        dropped = len(lines) - MAX_SUMMARY_LINES
        lines = lines[:MAX_SUMMARY_LINES]
        lines.append("…(%d more lines omitted)" % dropped)
    return lines


def _split_conversation(messages: list) -> tuple[list, list, list]:
    """Split into (prefix, middle, tail).

    prefix  = [system?] (messages[0] if role == "system")
    tail    = last TAIL_KEEP messages, verbatim
    middle  = everything in between, to be summarized
    """
    msgs = list(messages)
    prefix: list = []
    rest = msgs
    if rest and _role(rest[0]) == "system":
        prefix = [rest[0]]
        rest = rest[1:]
    if len(rest) <= TAIL_KEEP:
        return prefix, [], rest
    return prefix, rest[:-TAIL_KEEP], rest[-TAIL_KEEP:]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def compact_conversation(agent) -> dict:
    """Build a compaction report without mutating the agent.

    Returns a dict with before/after counts, the summary text, how many
    messages were summarized vs kept, and the ``new_messages`` list that
    :func:`do_compact` would install.
    """
    try:
        messages = _messages(agent)
        # drop any stale summary message from a previous compaction so we
        # never stack summaries on top of each other
        old_summary = next((m for m in messages if _is_summary_msg(m)), None)
        messages = [m for m in messages if not _is_summary_msg(m)]
        prefix, middle, tail = _split_conversation(messages)

        before_msgs = len(messages) + (1 if old_summary else 0)
        before_chars = _total_chars(messages) + (
            len(_content(old_summary)) if old_summary else 0)

        if not middle:
            # nothing worth summarizing — keep prior summary if there was
            # one, else worst case keep first+last
            kept = prefix + ([old_summary] if old_summary else []) + tail
            if not kept:
                kept = messages[:1] + messages[-1:]
            return {
                "before_msgs": before_msgs,
                "after_msgs": len(kept),
                "before_chars": before_chars,
                "after_chars": _total_chars(kept),
                "summary": "",
                "summary_lines": 0,
                "summarized_msgs": 0,
                "kept_msgs": len(kept),
                "new_messages": kept,
                "did_compact": False,
            }

        decision_lines = _extract_decision_lines(middle)
        if decision_lines:
            body = "\n".join("- " + ln for ln in decision_lines)
        else:
            body = "- (no key decisions extracted; %d messages summarized)" % len(middle)
        summary_text = SUMMARY_MARKER + "\n" + body
        summary_msg = {"role": "user", "content": summary_text}

        new_messages = prefix + [summary_msg] + tail
        return {
            "before_msgs": before_msgs,
            "after_msgs": len(new_messages),
            "before_chars": before_chars,
            "after_chars": _total_chars(new_messages),
            "summary": summary_text,
            "summary_lines": len(decision_lines),
            "summarized_msgs": len(middle),
            "kept_msgs": len(prefix) + len(tail),
            "new_messages": new_messages,
            "did_compact": True,
        }
    except Exception:
        # absolute worst case: keep first + last, never crash
        messages = _messages(agent)
        kept = []
        if messages:
            kept = [messages[0]] + ([messages[-1]] if len(messages) > 1 else [])
        chars = _total_chars(messages)
        return {
            "before_msgs": len(messages),
            "after_msgs": len(kept),
            "before_chars": chars,
            "after_chars": _total_chars(kept),
            "summary": "",
            "summary_lines": 0,
            "summarized_msgs": 0,
            "kept_msgs": len(kept),
            "new_messages": kept,
            "did_compact": False,
        }


def should_suggest_compact(agent) -> bool:
    """True when estimated token usage exceeds 80% of the model context window.

    Tokens are estimated as chars/4 (heuristic, works offline).
    """
    try:
        messages = _messages(agent)
        if not messages:
            return False
        est_tokens = _total_chars(messages) / 4.0
        model = getattr(agent, "model", None)
        window = getattr(model, "context_window", None) if model is not None else None
        try:
            window = int(window)
        except (TypeError, ValueError):
            window = DEFAULT_CONTEXT_WINDOW
        if window <= 0:
            window = DEFAULT_CONTEXT_WINDOW
        return (est_tokens / window) > SUGGEST_THRESHOLD
    except Exception:
        return False


def do_compact(agent) -> dict:
    """Rewrite ``agent.messages`` to the compacted form.

    New shape: [system?, summary_message, *last_10]. Returns stats
    ``{before_msgs, after_msgs, before_chars, after_chars}``.
    """
    report = compact_conversation(agent)
    stats = {
        "before_msgs": report["before_msgs"],
        "after_msgs": report["after_msgs"],
        "before_chars": report["before_chars"],
        "after_chars": report["after_chars"],
    }
    try:
        agent.messages = report["new_messages"]
    except Exception:
        pass
    # SOVEREIGN ENFORCEMENT: the rewrite can drop or displace messages[0];
    # re-seat the sealed prompt immediately so canonical history never
    # lives without its constitution (fixes a /compact prompt-drop hole).
    try:
        agent._reseat_system_prompt()
    except Exception:
        pass
    return stats


# ---------------------------------------------------------------------------
# self-test: python3 -m fullagent.compact
# ---------------------------------------------------------------------------

def _selftest() -> None:
    class FakeModel:
        context_window = 200_000

    class FakeAgent:
        def __init__(self):
            self.model = FakeModel()
            self.messages = []

    def umsg(text):
        return {"role": "user", "content": text}

    def amsg(text, tools=None):
        m = {"role": "assistant", "content": text}
        if tools:
            m["tool_calls"] = tools
        return m

    def tc(name, path=None):
        args = {"path": path} if path else {}
        return {"id": "1", "function": {"name": name,
                                       "arguments": json.dumps(args)}}

    # --- build a 50-message conversation with decisions sprinkled in ---
    agent = FakeAgent()
    agent.messages.append({"role": "system", "content": "You are a coding assistant."})
    agent.messages.append(umsg("Build me a web scraper in Python."))
    for i in range(48):
        if i % 6 == 0:
            agent.messages.append(umsg("request #%d: add feature %d" % (i, i)))
        elif i % 6 == 1:
            agent.messages.append(amsg(
                "I'll implement the retry logic with exponential backoff now."))
        elif i % 6 == 2:
            agent.messages.append(amsg("Done.", tools=[tc("write", "scraper.py")]))
        elif i % 6 == 3:
            agent.messages.append({"role": "tool", "tool_call_id": "1",
                                   "content": "wrote 120 lines"})
        elif i % 6 == 4:
            agent.messages.append(amsg("Plan: first parse the HTML, then extract links."))
        else:
            agent.messages.append(amsg("Small filler reply number %d." % i))
    assert len(agent.messages) == 50, len(agent.messages)

    # --- compact_conversation report ---
    report = compact_conversation(agent)
    assert report["before_msgs"] == 50
    assert report["did_compact"] is True
    new = report["new_messages"]
    assert _role(new[0]) == "system", "system prompt must survive"
    assert _is_summary_msg(new[1]), "summary must be second"
    assert len(new) == 2 + TAIL_KEEP, len(new)
    assert new[2:] == agent.messages[-TAIL_KEEP:], "tail must be verbatim"
    s = report["summary"]
    assert s.startswith(SUMMARY_MARKER)
    assert "Ran write on scraper.py" in s, s
    assert "retry logic" in s, s
    assert report["summary_lines"] <= MAX_SUMMARY_LINES
    assert report["after_chars"] < report["before_chars"]

    # --- do_compact rewrites in place ---
    stats = do_compact(agent)
    assert stats["before_msgs"] == 50
    assert stats["after_msgs"] == 12
    assert stats["before_chars"] > stats["after_chars"]
    assert len(agent.messages) == 12
    assert _role(agent.messages[0]) == "system"
    assert _is_summary_msg(agent.messages[1])
    # second compaction must not stack summaries
    stats2 = do_compact(agent)
    assert stats2["after_msgs"] == 12, stats2
    assert sum(1 for m in agent.messages if _is_summary_msg(m)) == 1

    # --- tiny conversation: no-op, never crash ---
    tiny = FakeAgent()
    tiny.messages = [{"role": "user", "content": "hi"}]
    r = compact_conversation(tiny)
    assert r["did_compact"] is False
    assert r["after_msgs"] == 1
    empty = FakeAgent()
    assert do_compact(empty)["after_msgs"] == 0

    # --- should_suggest_compact threshold ---
    big = FakeAgent()
    # >80% of 200k tokens => >160k tokens => >640k chars
    big.messages = [umsg("x" * 700_000)]
    assert should_suggest_compact(big) is True
    small = FakeAgent()
    small.messages = [umsg("hello")]
    assert should_suggest_compact(small) is False
    # missing model -> default 200k window, still fine
    nomodel = FakeAgent()
    del nomodel.model
    nomodel.messages = [umsg("hello")]
    assert should_suggest_compact(nomodel) is False
    # exactly at boundary: 80% -> not >, just under threshold must be False
    edge = FakeAgent()
    edge.messages = [umsg("x" * 640_000)]  # exactly 160k est tokens = 80%
    assert should_suggest_compact(edge) is False

    print("PASS")


if __name__ == "__main__":
    _selftest()
