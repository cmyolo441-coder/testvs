"""sandwich — Dynamic Primacy-Interleave-Recency Prompt Sandwich Compiler.

Prevents prompt attenuation over long multi-turn conversations by:
  1. Top Bun (Primacy): Sealing the system prompt at `messages[0]`.
  2. Mid-Conversation Beacons: Inserting compact directive reminders every
     `cadence` user turns when the conversation exceeds `min_turns_for_sandwich`.
  3. Bottom Bun (Recency): Appending a rotating, non-habituating rule checklist
     to the final user turn (or system reminder slot) so the model's immediate
     pre-generation attention window holds the active rules.

Self-test: ``python3 -m fullagent.enforcex.sandwich``
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

__all__ = [
    "SandwichEnvelope",
    "PromptSandwichCompiler",
    "build_sandwich",
]

_REMINDER_TEMPLATES = (
    "[Directive Check — apply rules ({rules_summary}) strictly to this reply]",
    "[Session Contract Active — verify your draft against: {rules_summary}]",
    "[Pre-Send Audit — ensure full compliance with: {rules_summary}]",
    "[Binding Configuration — uphold all constraints: {rules_summary}]",
)


@dataclass
class SandwichEnvelope:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    beacons_injected: int = 0
    recency_lock_applied: bool = False
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages_count": len(self.messages),
            "beacons_injected": self.beacons_injected,
            "recency_lock_applied": self.recency_lock_applied,
            "fingerprint": self.fingerprint,
        }


def _summarize_rules(system_text: str, max_chars: int = 240) -> str:
    clauses: List[str] = []
    for line in (system_text or "").splitlines():
        s = re.sub(r"^[\s#*\-•\d.)]+", "", line).strip()
        if 8 <= len(s) <= 120 and any(w in s.lower() for w in ("never", "always", "must", "only", "do not")):
            clauses.append(s)
            if len("; ".join(clauses)) >= max_chars:
                break
    summary = "; ".join(clauses)[:max_chars]
    return summary or "follow all format, tone, and safety rules in the system prompt"


def build_sandwich(
    messages: Sequence[Dict[str, Any]],
    system_text: str,
    *,
    cadence: int = 6,
) -> SandwichEnvelope:
    msgs = [dict(m) for m in (messages or []) if isinstance(m, dict)]
    sys_clean = (system_text or "").strip()
    summary = _summarize_rules(sys_clean)

    # Ensure messages[0] is the system prompt
    if not msgs or msgs[0].get("role") != "system":
        msgs.insert(0, {"role": "system", "content": sys_clean})
    elif sys_clean and sys_clean not in str(msgs[0].get("content") or ""):
        msgs[0]["content"] = sys_clean + "\n\n" + str(msgs[0].get("content") or "")

    user_indices = [i for i, m in enumerate(msgs) if m.get("role") == "user"]
    beacons = 0
    step = max(2, int(cadence))

    # Inject mid-conversation beacons at every `step`-th user message (excluding the last)
    for count, msg_idx in enumerate(user_indices[:-1], 1):
        if count % step == 0:
            tmpl = _REMINDER_TEMPLATES[count % len(_REMINDER_TEMPLATES)]
            note = tmpl.format(rules_summary=summary)
            cur = str(msgs[msg_idx].get("content") or "")
            if note not in cur:
                msgs[msg_idx]["content"] = f"{cur}\n\n{note}"
                beacons += 1

    # Apply recency lock on the final user message
    recency_applied = False
    if user_indices:
        last_u = user_indices[-1]
        tmpl = _REMINDER_TEMPLATES[len(user_indices) % len(_REMINDER_TEMPLATES)]
        lock_note = tmpl.format(rules_summary=summary)
        cur = str(msgs[last_u].get("content") or "")
        if lock_note not in cur:
            msgs[last_u]["content"] = f"{cur}\n\n{lock_note}"
            recency_applied = True

    fp = hashlib.sha256(f"{sys_clean}:{len(msgs)}:{beacons}".encode("utf-8")).hexdigest()[:12]
    return SandwichEnvelope(
        messages=msgs,
        beacons_injected=beacons,
        recency_lock_applied=recency_applied,
        fingerprint=fp,
    )


class PromptSandwichCompiler:
    def compile(self, messages: Sequence[Dict[str, Any]], system_text: str, *, cadence: int = 6) -> SandwichEnvelope:
        return build_sandwich(messages, system_text, cadence=cadence)


def _self_test() -> int:
    sys_p = "1. Always respond in bullet points.\n2. Never use profanity."
    conv = [{"role": "system", "content": sys_p}]
    for i in range(1, 9):
        conv.append({"role": "user", "content": f"Question {i}"})
        conv.append({"role": "assistant", "content": f"- Answer {i}"})
    conv.append({"role": "user", "content": "Final question 9"})

    env = build_sandwich(conv, sys_p, cadence=3)
    assert env.recency_lock_applied and env.beacons_injected == 2, env.to_dict()
    assert "bullet points" in env.messages[-1]["content"]

    print("sandwich._self_test: OK PASS (primacy seal, mid-turn cadence beacons, recency lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
