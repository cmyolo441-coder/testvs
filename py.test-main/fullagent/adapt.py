"""adapt — Provider & Model Enforcement Profiles (v5 apex).

Not every model sees the prompt the same way. Providers differ in:
  - system-message handling (merge, collapse, position-weight)
  - attention decay over long system prompts
  - tool-call wiring (which changes how tool output reaches attention)
  - reasoning fields (thinking models weight self-checks differently)
  - safety filters (what framing gets up- vs down-weighted)

adapt codifies this knowledge as PROFILES: per-model-family enforcement
parameters chosen from measurements and adversarial review. Consumers:

  - sentinel.injection_plan_for -> which slots fire
  - apex.build                  -> ledger sizing multiplier
  - comply.run_pipeline_plan    -> audit strictness
  - agent._compliance_gate      -> retry budget

Profile fields (all have safe defaults; unknown models -> "default"):
  slot_set          which sentinel slots fire
  ledger_ratio      enforcement overhead as a fraction of doc tokens
  anchor_every      inject the rotating anchor every N turns (1 = always)
  audit_mode        "full"|"spot"|"adaptive" default strictness
  max_audit_retries compliance-gate retry budget
  fence_tool_output shield tool output before it reaches the model
  trust_long_system provider keeps >30k system prompts fully attended
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Profile", "profile_for", "available_families", "PROFILE_COUNT"]


# ── profile dataclass ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Profile:
    family: str
    slot_set: tuple[str, ...]
    ledger_ratio: float = 0.06
    anchor_every: int = 1
    audit_mode: str = "adaptive"        # "full" | "spot" | "adaptive"
    max_audit_retries: int = 1
    fence_tool_output: bool = True
    trust_long_system: bool = False
    notes: str = ""

    def summary(self) -> str:
        slots = ",".join(s.split("-", 1)[0] for s in self.slot_set)
        return (f"{self.family}: slots={slots} "
                f"ledger={self.ledger_ratio:.0%} "
                f"anchor/={self.anchor_every} audit={self.audit_mode} "
                f"retries={self.max_audit_retries} "
                f"fence={self.fence_tool_output} "
                f"longsys={self.trust_long_system}")


# slot name constants (mirror sentinel.Slot without importing it --
# adapt must stay importable from anywhere, including headless tools)
_S_PREFIX = "P0-system-prefix"
_S_SUFFIX = "P1-system-suffix"
_S_PREUSER = "P2-pre-user-nudge"
_S_INUSER = "P3-in-user-prefix"
_S_ECHO = "P4-post-reply-echo"

# ── family profiles ──────────────────────────────────────────────────────

_PROFILES: dict[str, Profile] = {
    # DeepSeek / MiniMax / Qwen / GLM / Kimi -- open-weight chat models:
    # multi-system handling varies; pre-user nudge guarantees adjacency
    "openweight": Profile(
        family="openweight",
        slot_set=(_S_PREFIX, _S_SUFFIX, _S_PREUSER),
        ledger_ratio=0.07,
        anchor_every=1,
        audit_mode="adaptive",
        max_audit_retries=1,
        trust_long_system=False,
        notes="attention decays over >20k system prompts; keep the "
              "anchor in the rotation"),
    # Claude / Anthropic -- strong instruction-following, merges system
    # blocks; system-level config framing is respected, in-user tags
    # unnecessary
    "anthropic": Profile(
        family="anthropic",
        slot_set=(_S_PREFIX, _S_SUFFIX),
        ledger_ratio=0.06,
        anchor_every=2,
        audit_mode="adaptive",
        max_audit_retries=1,
        trust_long_system=True,
        notes="long system prompts well attended; rotate anchors "
              "less often to avoid boilerplate fatigue"),
    # OpenAI GPT / o-series -- merges multiple system messages; the
    # in-user prefix is the reliable adjacency slot
    "openai": Profile(
        family="openai",
        slot_set=(_S_PREFIX, _S_SUFFIX, _S_INUSER),
        ledger_ratio=0.06,
        anchor_every=1,
        audit_mode="adaptive",
        max_audit_retries=1,
        trust_long_system=True,
        notes="developer-message vocabulary up-weighted; keep the "
              "in-user tag for tool-heavy turns"),
    # Gemini -- strong on long context; prefix + suffix + pre-user,
    # lean overhead
    "gemini": Profile(
        family="gemini",
        slot_set=(_S_PREFIX, _S_SUFFIX, _S_PREUSER),
        ledger_ratio=0.05,
        anchor_every=2,
        audit_mode="adaptive",
        max_audit_retries=1,
        trust_long_system=True,
        notes="1M-context family; enforcement overhead kept lean"),
    # small / free-tier models -- weakest attention; everything on,
    # plus a bigger ledger share and one extra audit retry
    "small": Profile(
        family="small",
        slot_set=(_S_PREFIX, _S_SUFFIX, _S_PREUSER, _S_INUSER),
        ledger_ratio=0.09,
        anchor_every=1,
        audit_mode="full",
        max_audit_retries=2,
        trust_long_system=False,
        notes="max protection for weak attention; the ledger is the "
              "primary carrier, not the long document"),
    # unknown / new providers -- conservative default
    "default": Profile(
        family="default",
        slot_set=(_S_PREFIX, _S_SUFFIX, _S_PREUSER),
        ledger_ratio=0.06,
        anchor_every=1,
        audit_mode="adaptive",
        max_audit_retries=1,
        notes="conservative default for unrecognized model ids"),
}

# ── family detection ─────────────────────────────────────────────────────

_FAMILY_MATCHERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("anthropic", ("claude", "anthropic", "fable", "opus", "sonnet",
                   "haiku")),
    ("openai", ("gpt-", "gpt_", "o1", "o3", "o4", "chatgpt", "openai",
                "astra")),
    ("gemini", ("gemini",)),
    ("openweight", ("deepseek", "minimax", "qwen", "glm", "kimi", "mimo",
                    "hy3", "hy4", "grok", "llama", "mistral", "jev")),
)

# size hints: small/free-tier markers bump the protection level
_SMALL_MARKERS = re.compile(
    r"\b(free|mini|tiny|small|lite|nano|8b|3b|1b|7b)\b", re.I)


def _size_class(model_label: str) -> str:
    return "small" if _SMALL_MARKERS.search(model_label or "") \
        else "standard"


def profile_for(model_label: str) -> Profile:
    """Resolve the enforcement profile for a model id/label. Never
    raises; unknown models get the conservative default."""
    label = (model_label or "").lower()
    for family, keys in _FAMILY_MATCHERS:
        if any(k in label for k in keys):
            prof = _PROFILES[family]
            if _size_class(model_label) == "small":
                # small variants of big families get the small profile's
                # strictness while keeping the family's slot knowledge
                small = _PROFILES["small"]
                return Profile(
                    family=prof.family + "-small",
                    slot_set=small.slot_set,
                    ledger_ratio=max(prof.ledger_ratio,
                                     small.ledger_ratio),
                    anchor_every=1,
                    audit_mode=small.audit_mode,
                    max_audit_retries=small.max_audit_retries,
                    fence_tool_output=True,
                    trust_long_system=False,
                    notes=prof.notes + " | small-tier strictness")
            return prof
    return _PROFILES["default"]


def available_families() -> list[str]:
    return sorted({p.family.split("-")[0] for p in _PROFILES.values()})


PROFILE_COUNT = len(_PROFILES)


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    # family resolution
    p = profile_for("DeepSeek V4.1 Flash 1017")
    assert p.family == "openweight", p.family
    assert p.fence_tool_output

    pa = profile_for("claude-opus-5")
    assert pa.family == "anthropic" and pa.trust_long_system
    assert _S_INUSER not in pa.slot_set

    pg = profile_for("Gemini 4 Argon")
    assert pg.family == "gemini"

    po = profile_for("gpt-5.6-sol")
    assert po.family == "openai" and _S_INUSER in po.slot_set

    # small-tier strictness
    pf = profile_for("deepseek-v4.1-flash-free")
    assert pf.family == "openweight-small", pf.family
    assert pf.max_audit_retries >= 2
    assert pf.ledger_ratio >= p.ledger_ratio
    assert not pf.trust_long_system

    # unknown model -> default, never raises
    pd = profile_for("brand-new-model-9000")
    assert pd.family == "default"
    assert profile_for("") == pd
    assert profile_for(None) == pd  # type: ignore[arg-type]

    # all profiles well-formed
    for fam, prof in _PROFILES.items():
        assert prof.slot_set, fam
        assert 0.03 <= prof.ledger_ratio <= 0.15, fam
        assert prof.audit_mode in ("full", "spot", "adaptive"), fam
        assert 0 <= prof.max_audit_retries <= 3, fam

    assert len(available_families()) >= 5
    assert "slots=" in p.summary()

    print("ADAPT SELF-TEST PASS (families, small-tier, defaults, "
          f"{PROFILE_COUNT} profiles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())