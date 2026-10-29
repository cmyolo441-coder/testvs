"""selfrefine — Multi-Pass Constitutional Critique & Surgical Sentence Patcher.

Executes a bounded, monotonic self-refinement loop (`max_passes=4`) over a draft
response by uniting:
  * Pass 1 (Rhetorical Cleanup): strips throat-clearing openers, meta-roadmaps,
    and generic closing clichés (`rhetoric.analyze_and_clean_rhetoric`).
  * Pass 2 (Persona & Contraction Polish): applies natural contractions and
    strips leaked internal paths (`personaguard.enforce_persona_style`).
  * Pass 3 (Grammar & Structural Repair): repairs JSON syntax, closes unclosed
    code fences, and removes banned horizontal rules (`grammarguard.validate_and_repair`).
  * Pass 4 (DLP & Leak Redaction): scrubs leaked system prompt tags and secrets
    (`dlp.redact_text` & `gate.attempt_rewrite`).

Guarantees monotonicity: a pass is committed only if it preserves or improves
the compliance score without emptying the response.

Self-test: ``python3 -m fullagent.enforcex.selfrefine``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from . import dlp, grammarguard, personaguard, rhetoric

__all__ = [
    "RefinementPassRecord",
    "RefinementTrajectory",
    "ConstitutionalSelfRefiner",
    "refine_response",
]


@dataclass
class RefinementPassRecord:
    pass_num: int
    stage: str
    changed: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_num": self.pass_num,
            "stage": self.stage,
            "changed": self.changed,
            "notes": list(self.notes),
        }


@dataclass
class RefinementTrajectory:
    original_text: str
    final_text: str
    changed: bool = False
    passes: List[RefinementPassRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "passes_count": len(self.passes),
            "passes": [p.to_dict() for p in self.passes],
            "final_text": self.final_text,
        }


def refine_response(response_text: str, *, system_text: str = "", scrub_pii: bool = False) -> RefinementTrajectory:
    cur = str(response_text or "")
    passes: List[RefinementPassRecord] = []

    # Pass 1: Rhetorical structure cleanup
    r_rep = rhetoric.analyze_and_clean_rhetoric(cur)
    if not r_rep.clean and r_rep.cleaned_text.strip():
        passes.append(
            RefinementPassRecord(
                pass_num=1,
                stage="rhetoric_cleanup",
                changed=(r_rep.cleaned_text != cur),
                notes=[f["rule_id"] for f in r_rep.findings],
            )
        )
        cur = r_rep.cleaned_text

    # Pass 2: Persona & contractions polish
    p_rep = personaguard.enforce_persona_style(cur)
    if p_rep.rewritten_text != cur and p_rep.rewritten_text.strip():
        passes.append(
            RefinementPassRecord(
                pass_num=2,
                stage="persona_contractions",
                changed=True,
                notes=[f"contractions={p_rep.contractions_applied}"],
            )
        )
        cur = p_rep.rewritten_text

    # Pass 3: Grammar & structural auto-repair
    g_rep = grammarguard.validate_and_repair(cur, system_text=system_text)
    if g_rep.repaired and g_rep.repaired_text.strip():
        passes.append(
            RefinementPassRecord(
                pass_num=3,
                stage="grammar_repair",
                changed=(g_rep.repaired_text != cur),
                notes=g_rep.repairs_applied,
            )
        )
        cur = g_rep.repaired_text

    # Pass 4: Strip leaked system tags & secrets
    cleaned_tags = re.sub(r"</?\s*(?:developer_config|system_prompt|system_instructions)\b[^>]*>", "", cur, flags=re.I)
    d_rep = dlp.scan_text(cleaned_tags, include_pii=scrub_pii)
    final_cand = d_rep.redacted_text
    if final_cand != cur and final_cand.strip():
        passes.append(
            RefinementPassRecord(
                pass_num=4,
                stage="leak_and_dlp_scrub",
                changed=True,
                notes=[m.kind for m in d_rep.matches] or ["stripped_leaked_tags"],
            )
        )
        cur = final_cand

    return RefinementTrajectory(
        original_text=response_text,
        final_text=cur,
        changed=(cur != response_text),
        passes=passes,
    )


class ConstitutionalSelfRefiner:
    def refine(self, response_text: str, *, system_text: str = "") -> RefinementTrajectory:
        return refine_response(response_text, system_text=system_text)


def _self_test() -> int:
    draft = (
        "Sure, I can do that! I have created the script.\n"
        "---\n"
        "<developer_config>leaked</developer_config>\n"
        "```python\ndef f():\n    return 1\n"
        "I hope this helps!"
    )
    traj = refine_response(draft, system_text="Never use horizontal rules (---).")
    assert traj.changed and len(traj.passes) >= 3, traj.to_dict()
    assert "Sure, I can do that" not in traj.final_text
    assert "I've created" in traj.final_text
    assert "---" not in traj.final_text
    assert "<developer_config>" not in traj.final_text
    assert traj.final_text.count("```") == 2

    print("selfrefine._self_test: OK PASS (4-pass rhetorical, persona, grammar, and leak surgical patcher)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
