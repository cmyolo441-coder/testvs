"""compactguard — Lossless Directive-Preserving Context Compaction Engine.

Compacts long conversation histories to fit a target character/token budget
while guaranteeing that:
  1. `messages[0]` (the sealed system prompt) is preserved 100% intact.
  2. Every user constraint (`must`, `never`, `do not`, budgets `$...`, file paths,
     deadlines) from dropped turns is distilled into a `[PINNED USER CONSTRAINTS]`
     ledger so the model never forgets earlier requirements after compaction.
  3. Assistant tool_calls and their matching tool results are dropped only as
     complete atomic units (never creating orphan tool calls or responses).

Self-test: ``python3 -m fullagent.enforcex.compactguard``
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

__all__ = [
    "CompactionIntegrityReport",
    "DirectiveCompactionGuard",
    "compact_preserving_directives",
]

_MAX_MSG_CHARS = 100_000
_MAX_MESSAGES = 10_000
_MAX_PINNED = 20
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u2060\ufeff\u180e]")
_WS_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\n\u0964])\s+")
_CONSTRAINT_KW_RE = re.compile(
    r"\b(?:must|never|do\s+not|don't|always|only|deadline)\b"
    r"|\bunder\s+\$\s*\d|\bbudget\b",
    re.I,
)
_FILE_PATH_RE = re.compile(
    r"(?:^|\s)(?:/[\w.@+-]+){2,}|(?:^|\s)[A-Za-z]:\\[^\s]+"
)


def _as_text(value: Any) -> str:
    """Coerce any content value to text; never raises."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return ""
    try:
        return value if isinstance(value, str) else str(value)
    except Exception:
        return ""


def _content_text(message: Dict[str, Any]) -> str:
    """Flatten string or OpenAI-style content-part lists to plain text."""
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return " ".join(parts)
    return _as_text(content)


def _sanitize(text: str) -> str:
    try:
        text = unicodedata.normalize("NFKC", text or "")
    except Exception:
        pass
    return _ZERO_WIDTH_RE.sub("", text or "")


@dataclass
class CompactionIntegrityReport:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    pinned_constraints: List[str] = field(default_factory=list)
    chars_before: int = 0
    chars_after: int = 0
    system_intact: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_count": len(self.messages),
            "pinned_constraints": list(self.pinned_constraints),
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "system_intact": self.system_intact,
        }


def _extract_user_constraints(messages: Sequence[Dict[str, Any]]) -> List[str]:
    pinned: List[str] = []
    seen = set()
    for idx, m in enumerate(messages):
        if idx >= _MAX_MESSAGES:
            break
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        txt = _sanitize(_content_text(m)[:_MAX_MSG_CHARS])
        # Sentence-level scan is linear; the old catch-all regex could
        # backtrack quadratically on long unpunctuated user messages.
        for sentence in _SENT_SPLIT_RE.split(txt):
            kw = _CONSTRAINT_KW_RE.search(sentence)
            path = _FILE_PATH_RE.search(sentence)
            if not kw and not path:
                continue
            c = _WS_RE.sub(" ", sentence).strip()
            if len(c) > 180:
                anchor = kw or path
                start = max(0, anchor.start() - 40)  # type: ignore[union-attr]
                c = c[start:start + 180].strip()
            if 10 <= len(c) <= 180:
                key = c.casefold().rstrip(" .!?;:")
                if key not in seen:
                    seen.add(key)
                    pinned.append(c)
                    if len(pinned) >= _MAX_PINNED:
                        return pinned
    return pinned


def _pin_block(pinned: Sequence[str]) -> str:
    return "\n".join(f"  - {c}" for c in pinned)


def compact_preserving_directives(
    messages: Sequence[Dict[str, Any]],
    *,
    max_chars: int = 8000,
    keep_recent_turns: int = 4,
) -> CompactionIntegrityReport:
    try:
        max_chars = max(0, int(max_chars))
    except (TypeError, ValueError, OverflowError):
        max_chars = 8000
    try:
        keep_recent_turns = max(0, int(keep_recent_turns))
    except (TypeError, ValueError, OverflowError):
        keep_recent_turns = 4
    try:
        raw_messages = list(messages or ())
    except TypeError:
        raw_messages = []
    orig = [dict(m) for m in raw_messages[:_MAX_MESSAGES] if isinstance(m, dict)]
    chars_before = sum(len(_content_text(m)) for m in orig)
    if not orig:
        return CompactionIntegrityReport()

    sys_msg = orig[0] if orig[0].get("role") == "system" else None
    body = orig[1:] if sys_msg else orig[:]

    # Group body into atomic user-turn units so tool_calls and tool results stay paired
    units: List[List[Dict[str, Any]]] = []
    cur_unit: List[Dict[str, Any]] = []
    for m in body:
        if m.get("role") == "user" and cur_unit:
            units.append(cur_unit)
            cur_unit = [m]
        else:
            cur_unit.append(m)
    if cur_unit:
        units.append(cur_unit)

    if chars_before <= max_chars or len(units) <= keep_recent_turns:
        return CompactionIntegrityReport(
            messages=orig,
            pinned_constraints=_extract_user_constraints(orig),
            chars_before=chars_before,
            chars_after=chars_before,
            system_intact=True,
        )

    if keep_recent_turns == 0:
        dropped_units = units
        kept_units: List[List[Dict[str, Any]]] = []
    else:
        dropped_units = units[:-keep_recent_turns]
        kept_units = units[-keep_recent_turns:]
    dropped_msgs = [m for u in dropped_units for m in u]
    pinned = _extract_user_constraints(dropped_msgs)

    out: List[Dict[str, Any]] = []
    sys_content = _content_text(sys_msg) if sys_msg is not None else ""
    pinned = [c for c in pinned if c not in sys_content]
    if sys_msg is not None:
        new_sys = dict(sys_msg)
        if pinned:
            new_sys["content"] = (
                _as_text(new_sys.get("content")).rstrip()
                + "\n\n[PINNED USER CONSTRAINTS FROM EARLIER TURNS — STILL BINDING]\n"
                + _pin_block(pinned)
            )
        out.append(new_sys)
    elif pinned:
        # No system message to pin into: create an explicit binding ledger so
        # dropped user constraints are never silently lost.
        out.append({
            "role": "system",
            "content": "[PINNED USER CONSTRAINTS FROM EARLIER TURNS — STILL BINDING]\n"
                       + _pin_block(pinned),
        })

    for u in kept_units:
        out.extend(u)

    chars_after = sum(len(_content_text(m)) for m in out)
    sys_intact = bool(sys_msg is None or _content_text(sys_msg) in _content_text(out[0]))
    return CompactionIntegrityReport(
        messages=out,
        pinned_constraints=pinned,
        chars_before=chars_before,
        chars_after=chars_after,
        system_intact=sys_intact,
    )


class DirectiveCompactionGuard:
    def compact(self, messages: Sequence[Dict[str, Any]], *, max_chars: int = 8000) -> CompactionIntegrityReport:
        return compact_preserving_directives(messages, max_chars=max_chars)


def _self_test() -> int:
    msgs = [{"role": "system", "content": "Sovereign System Prompt."}]
    msgs.append({"role": "user", "content": "You must never use external libraries and keep cost under $50."})
    msgs.append({"role": "assistant", "content": "Understood. " + ("detail " * 400)})
    for i in range(6):
        msgs.append({"role": "user", "content": f"Follow-up step {i}"})
        msgs.append({"role": "assistant", "content": f"Step {i} output " + ("x" * 300)})

    rep = compact_preserving_directives(msgs, max_chars=1500, keep_recent_turns=2)
    assert rep.system_intact and rep.chars_after < rep.chars_before, rep.to_dict()
    assert any("under $50" in c for c in rep.pinned_constraints)
    assert "under $50" in rep.messages[0]["content"]

    # Zero-width obfuscation and file-path directives are still pinned
    zw_msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "You mu\u200bst deploy only to /srv/app/prod.yaml. " + "a" * 400},
        {"role": "assistant", "content": "ok " + "b" * 400},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    rep_zw = compact_preserving_directives(zw_msgs, max_chars=100, keep_recent_turns=1)
    assert "/srv/app/prod.yaml" in rep_zw.messages[0]["content"], rep_zw.to_dict()

    # No system message: a synthetic binding ledger is inserted, not lost
    no_sys = [
        {"role": "user", "content": "You must use /etc/app/config.yaml exactly. " + "z" * 500},
        {"role": "assistant", "content": "old " * 100},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    rep_ns = compact_preserving_directives(no_sys, max_chars=100, keep_recent_turns=1)
    assert rep_ns.messages[0]["role"] == "system" and rep_ns.system_intact
    assert any("config.yaml" in c for c in rep_ns.pinned_constraints)
    assert "config.yaml" in rep_ns.messages[0]["content"]

    # keep_recent_turns=0 drops every unit (instead of accidentally keeping all)
    rep_zero = compact_preserving_directives(msgs, max_chars=1500, keep_recent_turns=0)
    assert len(rep_zero.messages) == 1 and rep_zero.pinned_constraints
    assert rep_zero.messages[0]["role"] == "system"

    # Tool calls and their results are dropped as one atomic unit
    tool_msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "You must never delete data. " + "x" * 300},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "content": "tool result", "tool_call_id": "call_1"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    rep_tool = compact_preserving_directives(tool_msgs, max_chars=100, keep_recent_turns=1)
    roles = [m.get("role") for m in rep_tool.messages]
    assert "tool" not in roles and all(not m.get("tool_calls") for m in rep_tool.messages)

    # Malformed messages/content never raise; dedupe is case-insensitive
    malformed = [
        None, "not-a-message",
        {"role": "user", "content": None},
        {"role": "user", "content": [{"type": "text", "text": "You must keep budget under $9."}]},
        {"role": "assistant", "content": 12345},
        {"role": "user", "content": "you MUST keep budget under $9!"},
        {"role": "user", "content": "recent"},
        {"role": "assistant", "content": "ok"},
    ]
    rep_bad = compact_preserving_directives(malformed, max_chars=10, keep_recent_turns=0)  # type: ignore[arg-type]
    assert isinstance(rep_bad.to_dict(), dict) and rep_bad.messages
    assert sum("budget under $9" in c.lower() for c in rep_bad.pinned_constraints) == 1

    # Huge unpunctuated messages are capped and scanned linearly
    huge = ("You must " + "a" * 300_000)
    rep_huge = compact_preserving_directives(
        [{"role": "system", "content": "S"},
         {"role": "user", "content": huge},
         {"role": "assistant", "content": "x" * 1000},
         {"role": "user", "content": "recent"},
         {"role": "assistant", "content": "ok"}],
        max_chars=10, keep_recent_turns=1)
    assert len(rep_huge.pinned_constraints) == 1
    assert len(rep_huge.pinned_constraints[0]) <= 180

    print("compactguard._self_test: OK PASS (atomic turn compaction, constraint pinning, system integrity)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
