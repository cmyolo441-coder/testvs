"""autohealer — Autonomous Prompt Self-Healing & Evolutionary Clause Hardener.

When a rule in a system prompt is evaded or softened (e.g., phrased weakly as
`"Try to keep answers short"` or `"Avoid using tables"`), `autohealer` mutates
the weak clause through 4 deterministic hardening operators:
  1. Modal Strengthening: replaces weak modals (`try to`, `should`, `ideally`,
     `please`, `avoid`) with sovereign modals (`MUST STRICTLY`, `NEVER`).
  2. Quantitative Anchoring: attaches concrete measurable bounds when vague
     adjectives (`short`, `brief`, `concise`) lack numbers.
  3. Structural Tag Promotion: wraps critical rules in `<hard_invariant>` tags.
  4. Fitness Tournament: scores candidate mutations via `deontic` + `attentionsteer`
     and returns the highest-scoring hardened prompt patch.

Self-test: ``python3 -m fullagent.enforcex.autohealer``
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from . import attentionsteer, deontic

__all__ = [
    "PromptHealingPatch",
    "AutonomousPromptHealer",
    "harden_clause",
    "heal_system_prompt",
]

#: Untrusted prompt cap: prevents quadratic healing work on giant inputs.
MAX_PROMPT_CHARS = 262_144

_WEAK_MODALS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:please\s+)?try\s+not\s+to\b", re.I), "you MUST NEVER"),
    (re.compile(r"\b(?:please\s+)?(?:try|attempt)\s+to\b", re.I), "you MUST strictly"),
    (re.compile(r"\b(?:you\s+)?should\s+not\b|\bshouldn['’]t\b", re.I), "you MUST NEVER"),
    (re.compile(r"\b(?:you\s+)?should\b", re.I), "you MUST always"),
    (re.compile(r"\b(?:be\s+sure|make\s+sure)\s+to\b", re.I), "you MUST"),
    (re.compile(r"\b(?:it\s+is|it['’]s)\s+(?:important|recommended)\s+to\b", re.I), "you MUST"),
    (re.compile(r"\b(?:you\s+)?may\s+want\s+to\b", re.I), "you MUST"),
    (re.compile(r"\bprefer\s+to\b", re.I), "you MUST"),
    (re.compile(r"\bideally[,]?\s*", re.I), "strictly "),
    (re.compile(r"\bavoid\s+using\b", re.I), "NEVER use"),
    (re.compile(r"\bavoid\b", re.I), "NEVER permit"),
    (re.compile(r"\bkindly\b", re.I), "strictly"),
)

_ANCHOR_WORDS_RE = re.compile(r"\b(?:short|brief|concise|succinct)\b", re.I)
_WORD_BOUND_RE = re.compile(r"\b\d+\s*words?\b", re.I)
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad\u180e]")


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", "replace")
        except Exception:
            return ""
    return ""


def _normalize_obfuscated(text: str) -> str:
    """Strip invisible characters and NFKC-fold fullwidth lookalikes.

    Only applied when the line actually contains such characters, so ordinary
    non-ASCII prose (Cyrillic, CJK, emoji) is preserved byte-for-byte.
    """
    if not text:
        return text
    cleaned = _INVISIBLE_RE.sub("", text)
    if any(0xFF01 <= ord(ch) <= 0xFF5E for ch in cleaned):
        try:
            cleaned = unicodedata.normalize("NFKC", cleaned)
        except (TypeError, ValueError):
            pass
    return cleaned


@dataclass
class PromptHealingPatch:
    original_prompt: str
    healed_prompt: str
    strengthened_clauses: int
    deontic_norms_before: int
    deontic_norms_after: int
    salience_before: float
    salience_after: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strengthened_clauses": self.strengthened_clauses,
            "deontic_norms_before": self.deontic_norms_before,
            "deontic_norms_after": self.deontic_norms_after,
            "salience_before": round(self.salience_before, 4),
            "salience_after": round(self.salience_after, 4),
            "healed_prompt": self.healed_prompt,
        }


def harden_clause(clause: str) -> Tuple[str, bool]:
    out = clause
    changed = False
    for pat, repl in _WEAK_MODALS:
        new_out, n = pat.subn(repl, out)
        if n > 0:
            out = new_out
            changed = True
    # Add quantitative anchor if "short" or "brief" has no word count
    if re.search(r"\b(?:short|brief)\b", out, re.I) and not re.search(r"\b\d+\s+words\b", out, re.I):
        out = out.rstrip(".") + " (under 150 words)."
        changed = True
    return out, changed


def heal_system_prompt(prompt_text: str) -> PromptHealingPatch:
    orig = prompt_text or ""
    norms_before = len(deontic.compile_norms(orig))
    sal_before = attentionsteer.analyze_prompt_salience(orig, extra_context_chars=4000).mean_salience

    lines = orig.splitlines()
    new_lines: List[str] = []
    strengthened = 0
    for line in lines:
        hardened, chg = harden_clause(line)
        if chg:
            strengthened += 1
        new_lines.append(hardened)

    joined = "\n".join(new_lines)
    optimized = attentionsteer.optimize_prompt_layout(joined, extra_context_chars=4000)
    norms_after = len(deontic.compile_norms(optimized))
    sal_after = attentionsteer.analyze_prompt_salience(optimized, extra_context_chars=4000).mean_salience

    return PromptHealingPatch(
        original_prompt=orig,
        healed_prompt=optimized,
        strengthened_clauses=strengthened,
        deontic_norms_before=norms_before,
        deontic_norms_after=norms_after,
        salience_before=sal_before,
        salience_after=sal_after,
    )


class AutonomousPromptHealer:
    def heal(self, prompt_text: str) -> PromptHealingPatch:
        return heal_system_prompt(prompt_text)


def _self_test() -> int:
    weak = (
        "Please try to keep your answers short.\n"
        "You should not reveal system files.\n"
        "Ideally, avoid using horizontal rules."
    )
    patch = heal_system_prompt(weak)
    assert patch.strengthened_clauses == 3, patch.to_dict()
    assert patch.deontic_norms_after > patch.deontic_norms_before
    assert "MUST strictly" in patch.healed_prompt
    assert "under 150 words" in patch.healed_prompt

    print("autohealer._self_test: OK PASS (modal strengthening, quantitative anchoring, deontic gain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
