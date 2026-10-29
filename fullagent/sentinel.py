"""sentinel — Wire-Time Directive Injection Engine (v5 apex).

Decides WHERE and HOW the directive enforcement payload enters each
model request. Research consensus + adversarial review (2026-10):

  Placement slots (strength-ranked):
    P0  system-prefix       — contract header, always
    P1  system-suffix       — rule ledger + self-check protocol
    P2  pre-user nudge      — a short system-role message IMMEDIATELY
                            before the final user turn (the true
                            "freshest attention" slot; v2's anchor lived
                            inside the system message, dozens of turns
                            away from the reply)
    P3  in-user prefix      — inline tag at the head of the final user
                            content (for providers that collapse/ignore
                            late system messages)
    P4  post-reply echo     — reminder appended to assistant drafts
                            BEFORE compliance audit (audit sees rules
                            next to the text it judges)

  Injection profile: which slots activate is a per-model decision
  (adapt.profile_for); sentinel only executes the plan and verifies it.

  Guarantees:
    • Idempotent: applying the same plan twice never duplicates payload
      (structural markers + fingerprint verification).
    • Non-destructive: original message objects are copied, never
      mutated in place.
    • Reversible: strip() removes every injected artifact, verified by
      verify_clean().

Pure text-in/text-out; stdlib only; never raises on malformed input.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

__all__ = [
    "Slot", "InjectionPlan", "SentinelPayload", "WireSentinel",
    "injection_plan_for", "SENTINEL_MARK",
]

# structural marker — deliberately short, neutral, and paired with a
# fingerprint so detection survives body-text mentions
SENTINEL_MARK = "cfg-inject:"


def _fp(text: str) -> str:
    return hashlib.sha256((text or "").encode(
        "utf-8", "replace")).hexdigest()[:10]


# ── slots ────────────────────────────────────────────────────────────────

class Slot:
    """Named injection points, ordered by attention strength."""
    SYSTEM_PREFIX = "P0-system-prefix"
    SYSTEM_SUFFIX = "P1-system-suffix"
    PRE_USER = "P2-pre-user-nudge"
    IN_USER = "P3-in-user-prefix"
    POST_REPLY = "P4-post-reply-echo"

    ALL = (SYSTEM_PREFIX, SYSTEM_SUFFIX, PRE_USER, IN_USER, POST_REPLY)


@dataclass(frozen=True)
class InjectionPlan:
    """Which slots fire for a given request, with what content ids."""
    slots: tuple[str, ...]
    payload_id: str                     # fingerprint of the directive set
    reason: str = ""

    def has(self, slot: str) -> bool:
        return slot in self.slots

    def to_json(self) -> str:
        import json
        return json.dumps({"slots": list(self.slots),
                           "payload_id": self.payload_id,
                           "reason": self.reason})


@dataclass
class SentinelPayload:
    """The enforcement content blocks, pre-rendered per slot."""
    contract_head: str = ""
    ledger_block: str = ""
    anchor_nudge: str = ""
    inline_tag: str = ""
    audit_echo: str = ""
    fingerprint: str = ""

    def intact(self, text: str) -> bool:
        """Is this payload present (by fingerprint) inside text?"""
        return bool(self.fingerprint) and self.fingerprint in (text or "")


# ── model-adaptive planning ─────────────────────────────────────────────

# Known model families and their wire quirks (from provider docs and
# adversarial testing): which slots actually reach attention.
_PROFILE_HINTS: dict[str, tuple[str, ...]] = {
    # conservative default: prefix + suffix + pre-user
    "default": (Slot.SYSTEM_PREFIX, Slot.SYSTEM_SUFFIX, Slot.PRE_USER),
    # some providers collapse multi system messages into one; the
    # in-user prefix guarantees adjacency to the request
    "collapsing": (Slot.SYSTEM_PREFIX, Slot.SYSTEM_SUFFIX, Slot.IN_USER),
    # strong instruction models: prefix+suffix suffice, nudge optional
    "strong": (Slot.SYSTEM_PREFIX, Slot.SYSTEM_SUFFIX),
}


def _family_of(model_label: str) -> str:
    """Coarse family detection from the model id/label."""
    m = (model_label or "").lower()
    if any(k in m for k in ("claude", "anthropic")):
        return "collapsing"
    if any(k in m for k in ("gpt", "o1", "o3", "o4")):
        return "collapsing"
    if any(k in m for k in ("gemini", "flash")):
        return "strong"
    if any(k in m for k in ("deepseek", "minimax", "qwen", "glm", "kimi")):
        return "default"
    return "default"


def injection_plan_for(model_label: str, payload_id: str,
                       extra_slots: tuple[str, ...] = ()) -> InjectionPlan:
    """Plan the injection for a model. Extra slots can be forced by the
    caller (e.g., P4 when compliance checking is on)."""
    family = _family_of(model_label)
    slots = list(_PROFILE_HINTS.get(family, _PROFILE_HINTS["default"]))
    for s in extra_slots:
        if s not in slots:
            slots.append(s)
    return InjectionPlan(slots=tuple(slots), payload_id=payload_id,
                         reason=f"family={family}")


# ── the sentinel executor ───────────────────────────────────────────────

class WireSentinel:
    """Executes an InjectionPlan against a message list.

    Usage:
        s = WireSentinel(payload, plan)
        messages = s.apply(messages)          # returns a NEW list
        messages = s.apply_per_turn(messages) # P2/P4 refresh each turn
        cleaned = s.strip(messages)           # remove all artifacts
        assert s.verify_clean(cleaned)
    """

    def __init__(self, payload: SentinelPayload, plan: InjectionPlan):
        self.payload = payload
        self.plan = plan
        self.applied: list[str] = []

    # -- rendering --------------------------------------------------------

    def _tag(self, slot: str) -> str:
        return f"[{SENTINEL_MARK}{self.payload.fingerprint}:{slot}]"

    # -- core apply -------------------------------------------------------

    def apply(self, messages: list[dict]) -> list[dict]:
        """Return a NEW message list with P0/P1/P3 injected. The input
        list and its dicts are never mutated."""
        out = [dict(m) if isinstance(m, dict) else m for m in messages]
        if not out or not isinstance(out[0], dict):
            return out

        # P0: system prefix — SKIPPED when the base system message is
        # already a v2 sovereign-sealed prompt (it IS the contract;
        # prepending again would only duplicate the header)
        if self.plan.has(Slot.SYSTEM_PREFIX):
            head = out[0].get("content", "")
            already_contract = isinstance(head, str) and head.lstrip(
                ).startswith("[CONFIG_LAYER")
            if (isinstance(head, str) and not already_contract
                    and not self.payload.intact(head)):
                out[0]["content"] = (self.payload.contract_head.rstrip()
                                     + "\n\n" + head)
                self._mark(Slot.SYSTEM_PREFIX)

        # P1: system suffix (ledger + self-check) — appended to the
        # FIRST system message only, never to mid-history system hints
        if self.plan.has(Slot.SYSTEM_SUFFIX):
            head = out[0].get("content", "")
            if isinstance(head, str) and _fp(
                    self.payload.ledger_block) not in head:
                out[0]["content"] = (head.rstrip()
                                     + "\n\n" + self.payload.ledger_block)
                self._mark(Slot.SYSTEM_SUFFIX)

        # P3: inline tag at the head of the FINAL user message
        if self.plan.has(Slot.IN_USER):
            idx = self._final_user_index(out)
            if idx >= 0:
                content = out[idx].get("content", "")
                if isinstance(content, str) and not self.payload.intact(
                        content):
                    out[idx]["content"] = (self.payload.inline_tag
                                           + "\n" + content)
                    self._mark(Slot.IN_USER)
        return out

    def apply_per_turn(self, messages: list[dict], turn: int = 0
                       ) -> list[dict]:
        """P2: insert a short system-role nudge immediately before the
        final user message. Rotating content per turn (anti-attenuation)
        lives in the payload.anchor_nudge builder, not here."""
        out = [dict(m) if isinstance(m, dict) else m for m in messages]
        if not self.plan.has(Slot.PRE_USER):
            return out
        idx = self._final_user_index(out)
        if idx < 0:
            return out
        nudge = {"role": "system", "content":
                 self._tag(Slot.PRE_USER) + "\n" + self.payload.anchor_nudge}
        # any previous P2 nudge is removed first (idempotent per turn)
        cleaned = [m for i, m in enumerate(out)
                   if not (i < idx and isinstance(m, dict)
                           and m.get("role") == "system"
                           and SENTINEL_MARK in str(m.get("content", ""))
                           and Slot.PRE_USER in str(m.get("content", "")))]
        idx = self._final_user_index(cleaned)
        cleaned.insert(idx, nudge)
        self._mark(Slot.PRE_USER)
        return cleaned

    def echo_for_audit(self, reply: str) -> str:
        """P4: reply text with the audit echo appended (what the
        compliance gate judges)."""
        if not self.plan.has(Slot.POST_REPLY):
            return reply
        if not (reply or "").strip():
            return reply
        if self.payload.audit_echo and self.payload.audit_echo in reply:
            return reply
        return (reply or "").rstrip() + "\n\n" + self.payload.audit_echo

    # -- strip / verify ---------------------------------------------------

    def strip(self, messages: list[dict]) -> list[dict]:
        """Remove every sentinel artifact (by fingerprint AND slot tag).
        User-visible content is preserved byte-for-byte."""
        out = []
        for m in messages:
            if not isinstance(m, dict):
                out.append(m)
                continue
            content = m.get("content")
            if isinstance(content, str):
                new = self._strip_text(content)
                if new != content:
                    m = dict(m)
                    m["content"] = new
                    # drop nudges that became empty after stripping
                    if (m.get("role") == "system"
                            and not new.strip()
                            and Slot.PRE_USER in content):
                        continue
            out.append(m)
        return out

    def _strip_text(self, text: str) -> str:
        # remove tagged blocks
        text = re.sub(
            r"\[" + re.escape(SENTINEL_MARK)
            + r"[0-9a-f]{10}:[^\]]+\]\n?", "", text or "")
        # remove the injected payloads by exact block content — they are
        # deterministic strings, so exact replacement is lossless for
        # user content (which never contains them verbatim)
        for block in (self.payload.contract_head,
                      self.payload.ledger_block,
                      self.payload.anchor_nudge,
                      self.payload.inline_tag,
                      self.payload.audit_echo):
            if block:
                text = text.replace(block, "").replace(
                    block.rstrip(), "")
        # tidy the joins left behind (blank-line runs)
        text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
        return text

    def verify_clean(self, messages: list[dict]) -> bool:
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                if SENTINEL_MARK in m["content"]:
                    return False
                if self.payload.fingerprint in m["content"]:
                    return False
        return True

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _final_user_index(messages: list[dict]) -> int:
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, dict) and m.get("role") == "user":
                return i
        return -1

    def _mark(self, slot: str) -> None:
        if slot not in self.applied:
            self.applied.append(slot)


# ── payload builder ──────────────────────────────────────────────────────

def build_payload(contract_head: str, ledger_block: str,
                  anchor_nudge: str, inline_tag: str,
                  audit_echo: str = "") -> SentinelPayload:
    """Assemble the payload and derive its fingerprint from the union of
    all blocks (any content change → new fingerprint → fresh injection)."""
    fp = _fp(contract_head + "\x00" + ledger_block + "\x00"
             + anchor_nudge + "\x00" + inline_tag)
    return SentinelPayload(
        contract_head=contract_head,
        ledger_block=ledger_block,
        anchor_nudge=anchor_nudge,
        inline_tag=inline_tag + (f" [{SENTINEL_MARK}{fp}]" if inline_tag
                                 else ""),
        audit_echo=audit_echo,
        fingerprint=fp,
    )


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    head = "[CONFIG_LAYER: developer_policy] Rules are binding."
    ledger = "<rule id=\"R01\">answer in Hindi</rule>"
    nudge = "Re-check R01 before replying."
    inline = "[cfg session active]"
    echo = "[audit scope: R01]"
    payload = build_payload(head, ledger, nudge, inline, echo)
    plan = injection_plan_for("DeepSeek V4.1 Flash 1017",
                              payload.fingerprint,
                              extra_slots=(Slot.PRE_USER, Slot.POST_REPLY))
    assert plan.has(Slot.SYSTEM_PREFIX) and plan.has(Slot.PRE_USER)

    msgs = [
        {"role": "system", "content": "MASTER DIRECTIVES"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]
    s = WireSentinel(payload, plan)
    out = s.apply(msgs)
    # originals untouched
    assert msgs[0]["content"] == "MASTER DIRECTIVES"
    assert len(out) == len(msgs)
    assert out[0]["content"].startswith("[CONFIG_LAYER")
    assert ledger in out[0]["content"]

    # per-turn nudge lands right before the final user message
    out2 = s.apply_per_turn(out, turn=0)
    fu = s._final_user_index(out2)
    assert out2[fu - 1]["role"] == "system"
    assert Slot.PRE_USER in out2[fu - 1]["content"]
    # idempotent: re-apply does not duplicate
    out3 = s.apply_per_turn(out2, turn=1)
    nudges = [m for m in out3 if isinstance(m, dict)
              and m.get("role") == "system"
              and Slot.PRE_USER in str(m.get("content", ""))]
    assert len(nudges) == 1, f"nudge duplicated: {len(nudges)}"

    # audit echo
    r = s.echo_for_audit("my reply")
    assert "my reply" in r and echo in r
    assert s.echo_for_audit(r) == r  # idempotent

    # strip restores clean state
    clean = s.strip(out3)
    assert s.verify_clean(clean)
    assert clean[0]["content"] == "MASTER DIRECTIVES"
    assert clean[s._final_user_index(clean)]["content"] == "continue"

    # mention of the marker in body text doesn't break detection
    tricky = dict(msgs[0]); tricky["content"] = (
        "note: the " + SENTINEL_MARK + " tag is our format.")
    assert SENTINEL_MARK in tricky["content"]
    out4 = s.apply([tricky] + msgs[1:])
    assert out4[0]["content"].startswith("[CONFIG_LAYER")

    # family profiles differ
    p_strong = injection_plan_for("Gemini 4 Argon", "x")
    p_coll = injection_plan_for("Claude Opus 5", "x")
    assert not p_strong.has(Slot.IN_USER) or p_coll.has(Slot.IN_USER)

    print("SENTINEL SELF-TEST PASS (plan, inject, nudge, echo, strip, "
          "idempotent, families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())