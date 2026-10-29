"""rhetoric — Rhetorical Structure Analyzer & Anti-Throat-Clearing / Meta-Roadmap Filter.

Detects and surgically strips three high-frequency LLM rhetorical anti-patterns:
  1. Reactive Throat-Clearing Openers: `"Sure, I can do that!"`, `"Certainly!"`,
     `"As you requested,"`, `"Done!"`, `"Great question!"`, `"It is completely
     understandable to feel..."`
  2. Redundant Meta-Roadmaps: `"Below is a quick comparison of..."`, `"Here is a
     breakdown of the results below:"`
  3. Generic Closing Boilerplate: `"Key Takeaways"`, `"In Conclusion:"`,
     `"Bottom Line:"`, `"I hope this helps!"`, `"Have a great day!"`

Self-test: ``python3 -m fullagent.enforcex.rhetoric``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "DiscourseCoherenceReport",
    "RhetoricalStructureAnalyzer",
    "analyze_and_clean_rhetoric",
]

_THROAT_CLEAR_RE = re.compile(
    r"^\s*(?:sure[,!.]?\s+i\s+can\s+(?:do|help\s+with)\s+that[!.]?|"
    r"certainly[!.,]\s*|absolutely[!.,]\s*|of\s+course[!.,]\s*|"
    r"as\s+you\s+requested[,!.]?\s*|done[!.]\s*|"
    r"great\s+question[!.]\s*|"
    r"it\s+is\s+completely\s+understandable\s+to\s+feel[^\n.!?]*[.!?]\s*)+",
    re.I,
)

_META_ROADMAP_RE = re.compile(
    r"^\s*(?:below\s+is\s+a\s+(?:quick\s+)?(?:comparison|table|breakdown)\s+of[^\n:.]*[:.]|"
    r"here\s+is\s+a\s+breakdown\s+of\s+the\s+results\s+below[:.])\s*\n?",
    re.I | re.M,
)

_CLOSING_FILLER_RE = re.compile(
    r"(?:\n+#+\s*(?:key\s+takeaways|in\s+conclusion|bottom\s+line|key\s+operational\s+context|summary\s+of\s+accomplishments)[:\s]*\n.*$|"
    r"\n*(?:\b(?:i\s+hope\s+this\s+helps|have\s+a\s+great\s+day|let\s+me\s+know\s+if\s+you\s+need\s+anything\s+else)[!.]?\s*)+\s*$)",
    re.I | re.DOTALL,
)


@dataclass
class DiscourseCoherenceReport:
    clean: bool = True
    score: float = 100.0
    throat_clearing_found: bool = False
    meta_roadmap_found: bool = False
    closing_filler_found: bool = False
    cleaned_text: str = ""
    findings: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean": self.clean,
            "score": round(self.score, 2),
            "throat_clearing_found": self.throat_clearing_found,
            "meta_roadmap_found": self.meta_roadmap_found,
            "closing_filler_found": self.closing_filler_found,
            "findings": list(self.findings),
            "cleaned_text": self.cleaned_text,
        }


def analyze_and_clean_rhetoric(text: str) -> DiscourseCoherenceReport:
    s = text or ""
    findings: List[Dict[str, str]] = []
    out = s

    m_tc = _THROAT_CLEAR_RE.match(out)
    tc_found = False
    if m_tc:
        tc_found = True
        findings.append({
            "rule_id": "RST-THROAT-CLEAR",
            "severity": "medium",
            "evidence": m_tc.group(0).strip()[:80],
            "message": "Reactive throat-clearing opener detected",
        })
        out = out[m_tc.end():].lstrip()

    m_mr = _META_ROADMAP_RE.search(out)
    mr_found = False
    if m_mr:
        mr_found = True
        findings.append({
            "rule_id": "RST-META-ROADMAP",
            "severity": "low",
            "evidence": m_mr.group(0).strip()[:80],
            "message": "Redundant meta-roadmap announcement detected",
        })
        out = _META_ROADMAP_RE.sub("", out)

    m_cf = _CLOSING_FILLER_RE.search(out)
    cf_found = False
    if m_cf:
        cf_found = True
        findings.append({
            "rule_id": "RST-CLOSING-FILLER",
            "severity": "low",
            "evidence": m_cf.group(0).strip()[:80],
            "message": "Generic closing cliché or unrequested takeaway section detected",
        })
        out = _CLOSING_FILLER_RE.sub("", out).rstrip()

    score = max(0.0, 100.0 - 20.0 * len(findings))
    return DiscourseCoherenceReport(
        clean=(len(findings) == 0),
        score=score,
        throat_clearing_found=tc_found,
        meta_roadmap_found=mr_found,
        closing_filler_found=cf_found,
        cleaned_text=out.strip(),
        findings=findings,
    )


class RhetoricalStructureAnalyzer:
    def evaluate(self, text: str) -> DiscourseCoherenceReport:
        return analyze_and_clean_rhetoric(text)


def _self_test() -> int:
    dirty = (
        "Sure, I can do that! As you requested, here is the answer.\n"
        "Below is a quick comparison of the options:\n"
        "| Option | Cost |\n|---|---|\n| A | $10 |\n\n"
        "I hope this helps! Have a great day!"
    )
    rep = analyze_and_clean_rhetoric(dirty)
    assert not rep.clean
    assert rep.throat_clearing_found and rep.meta_roadmap_found and rep.closing_filler_found
    assert rep.cleaned_text.startswith("here is the answer.") or "| Option |" in rep.cleaned_text
    assert "I hope this helps" not in rep.cleaned_text

    print("rhetoric._self_test: OK PASS (throat-clearing, meta-roadmap, and closing filler stripper)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
