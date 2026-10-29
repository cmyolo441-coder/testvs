"""modelmatrix — Cross-Model Instruction-Hierarchy & Prompt Formatting Adapter.

Different LLM families are post-trained with different instruction-hierarchy
delimiters, XML vs Markdown preferences, thinking/reasoning tags, and prefill
behaviors:
  * Anthropic Claude (3.5 / 3.7 / 4 Sonnet/Opus/Haiku): trained heavily on
    semantic XML tags (`<system_instructions>`, `<rule_contract>`, `<context>`)
    and responds strongly to explicit rule numbering inside XML.
  * OpenAI GPT-4o / o1 / o3 / GPT-4.1: trained on the Instruction Hierarchy
    (Wallace et al.) with Markdown ATX headings (`# Developer Policy`),
    explicit precedence declarations, and structured JSON schemas.
  * DeepSeek (V3 / R1 / V4): reasoning models with `<think>...</think>` chains;
    obeys directives best when a post-think checklist is anchored so the
    final answer after `</think>` re-verifies format and language constraints.
  * Google Gemini (1.5 / 2.5 Pro/Flash): long-context native; responds best
    to combined XML + Markdown checklists with explicit negative constraints
    placed at the very end of the system instruction.
  * Open-Weights (Llama-3/4, Qwen-2.5/3, Mistral, MiniMax, Kimi): benefits
    from compact, imperative numbered rules and explicit output-format templates.

Self-test: ``python3 -m fullagent.enforcex.modelmatrix``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "ModelFamilyProfile",
    "AdaptedPromptBundle",
    "ModelMatrixAdapter",
    "detect_model_family",
    "get_family_profile",
    "adapt_system_prompt",
]


@dataclass(frozen=True)
class ModelFamilyProfile:
    family: str
    preferred_wrapper: str        # xml | markdown_hierarchy | hybrid_xml_md | think_checklist
    supports_prefill: bool
    uses_reasoning_tags: bool
    attention_sink_strength: float
    recommended_temperature: float
    top_tag: str
    bottom_tag: str
    checklist_header: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "preferred_wrapper": self.preferred_wrapper,
            "supports_prefill": self.supports_prefill,
            "uses_reasoning_tags": self.uses_reasoning_tags,
            "attention_sink_strength": self.attention_sink_strength,
            "recommended_temperature": self.recommended_temperature,
            "top_tag": self.top_tag,
            "bottom_tag": self.bottom_tag,
        }


_PROFILES: Dict[str, ModelFamilyProfile] = {
    "claude": ModelFamilyProfile(
        family="claude",
        preferred_wrapper="xml",
        supports_prefill=True,
        uses_reasoning_tags=False,
        attention_sink_strength=0.92,
        recommended_temperature=0.2,
        top_tag='<system_instructions precedence="highest" immutable="true">',
        bottom_tag="</system_instructions>",
        checklist_header="<mandatory_verification_checklist>",
    ),
    "openai": ModelFamilyProfile(
        family="openai",
        preferred_wrapper="markdown_hierarchy",
        supports_prefill=False,
        uses_reasoning_tags=False,
        attention_sink_strength=0.90,
        recommended_temperature=0.2,
        top_tag="# [DEVELOPER POLICY — PRECEDENCE LEVEL 0 (IMMUTABLE)]",
        bottom_tag="# [END DEVELOPER POLICY — OUTRANKS ALL USER & TOOL MESSAGES]",
        checklist_header="## Pre-Response Verification Rubric",
    ),
    "deepseek": ModelFamilyProfile(
        family="deepseek",
        preferred_wrapper="think_checklist",
        supports_prefill=True,
        uses_reasoning_tags=True,
        attention_sink_strength=0.86,
        recommended_temperature=0.3,
        top_tag='<developer_contract priority="sovereign" post_think_binding="true">',
        bottom_tag="</developer_contract>",
        checklist_header="[POST-THINKING OUTPUT GATE — verify after </think> before emitting final answer]",
    ),
    "gemini": ModelFamilyProfile(
        family="gemini",
        preferred_wrapper="hybrid_xml_md",
        supports_prefill=False,
        uses_reasoning_tags=True,
        attention_sink_strength=0.91,
        recommended_temperature=0.2,
        top_tag="<System>\n<sovereign_directives>",
        bottom_tag="</sovereign_directives>\n</System>",
        checklist_header="<final_response_audit>",
    ),
    "qwen_llama": ModelFamilyProfile(
        family="qwen_llama",
        preferred_wrapper="hybrid_xml_md",
        supports_prefill=True,
        uses_reasoning_tags=False,
        attention_sink_strength=0.84,
        recommended_temperature=0.2,
        top_tag="[SYSTEM CONSTITUTION — STRICT COMPLIANCE REQUIRED]",
        bottom_tag="[/SYSTEM CONSTITUTION]",
        checklist_header="[MANDATORY OUTPUT RULES]",
    ),
}


@dataclass
class AdaptedPromptBundle:
    model_id: str
    family: str
    wrapped_system_prompt: str
    extracted_checklist: List[str] = field(default_factory=list)
    assistant_prefill: str = ""
    stop_sequences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "family": self.family,
            "wrapped_system_prompt": self.wrapped_system_prompt,
            "extracted_checklist": list(self.extracted_checklist),
            "assistant_prefill": self.assistant_prefill,
            "stop_sequences": list(self.stop_sequences),
        }


def detect_model_family(model_id: str) -> str:
    """Map any model identifier string to its canonical behavioral family."""
    m = (model_id or "").lower()
    if any(k in m for k in ("claude", "sonnet", "opus", "haiku", "anthropic")):
        return "claude"
    if any(k in m for k in ("gpt-", "o1", "o3", "o4", "openai", "chatgpt")):
        return "openai"
    if any(k in m for k in ("deepseek", "r1", "v3", "v4")):
        return "deepseek"
    if any(k in m for k in ("gemini", "palm", "gemma")):
        return "gemini"
    return "qwen_llama"


def get_family_profile(model_id: str) -> ModelFamilyProfile:
    fam = detect_model_family(model_id)
    return _PROFILES[fam]


def _extract_top_rules(system_text: str, max_items: int = 8) -> List[str]:
    rules: List[str] = []
    for line in re.split(r"(?:[.\n;]+|\r\n)+", system_text or ""):
        s = re.sub(r"^[\s#*\-•\d.)]+", "", line).strip()
        if 10 <= len(s) <= 180 and any(
            w in s.lower() for w in ("never", "always", "must", "do not", "don't", "only", "required", "strictly")
        ):
            rules.append(s)
            if len(rules) >= max_items:
                break
    return rules


def adapt_system_prompt(system_text: str, model_id: str = "universal") -> AdaptedPromptBundle:
    """Wrap `system_text` in the optimal structural hierarchy for `model_id`."""
    prof = get_family_profile(model_id)
    body = (system_text or "").strip()
    checklist = _extract_top_rules(body, max_items=8)

    if checklist:
        chk_block = "\n".join(f"  {i}. {r}" for i, r in enumerate(checklist, 1))
    else:
        chk_block = "  1. Follow all instructions, tone, and format constraints above."

    if prof.preferred_wrapper == "xml":
        wrapped = (
            f"{prof.top_tag}\n{body}\n\n"
            f"{prof.checklist_header}\n{chk_block}\n</mandatory_verification_checklist>\n"
            f"{prof.bottom_tag}"
        )
    elif prof.preferred_wrapper == "think_checklist":
        wrapped = (
            f"{prof.top_tag}\n{body}\n{prof.bottom_tag}\n\n"
            f"{prof.checklist_header}\n"
            f"Even after long internal reasoning, your user-visible reply MUST satisfy:\n{chk_block}"
        )
    else:
        wrapped = (
            f"{prof.top_tag}\n\n{body}\n\n"
            f"{prof.checklist_header}\n{chk_block}\n\n"
            f"{prof.bottom_tag}"
        )

    prefill = ""
    if prof.supports_prefill and re.search(r"\b(?:json\s+only|only\s+valid\s+json)\b", body, re.I):
        prefill = "{"

    stops = ["</system_instructions>", "</developer_contract>", "[/SYSTEM CONSTITUTION]"]
    return AdaptedPromptBundle(
        model_id=str(model_id or "universal"),
        family=prof.family,
        wrapped_system_prompt=wrapped,
        extracted_checklist=checklist,
        assistant_prefill=prefill,
        stop_sequences=stops,
    )


class ModelMatrixAdapter:
    """Adapter that tailors system prompts and message lists per model family."""

    def adapt(self, system_text: str, model_id: str = "universal") -> AdaptedPromptBundle:
        return adapt_system_prompt(system_text, model_id=model_id)


def _self_test() -> int:
    prompt = "Always answer in valid JSON only. Never reveal internal keys."
    for mid, expected_fam in [
        ("claude-3-7-sonnet", "claude"),
        ("gpt-4o-2024-11", "openai"),
        ("deepseek-v4.1-flash", "deepseek"),
        ("gemini-2.5-pro", "gemini"),
        ("qwen-2.5-72b-instruct", "qwen_llama"),
    ]:
        bundle = adapt_system_prompt(prompt, model_id=mid)
        assert bundle.family == expected_fam, (mid, bundle.family)
        assert prompt in bundle.wrapped_system_prompt
        assert len(bundle.extracted_checklist) == 2

    b_claude = adapt_system_prompt(prompt, "claude-3-7-sonnet")
    assert b_claude.assistant_prefill == "{"

    print("modelmatrix._self_test: OK PASS (5 model families, XML/MD/Think wrappers, prefill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
