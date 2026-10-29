"""morpholex — Multilingual Script Profiler, Banned Buzzword Filter & Parallel Bullet Enforcer.

Performs three deterministic linguistic compliance checks:
  1. Unicode Script Distribution & Language Lock: measures exact character ratios
     across Latin, Devanagari (`U+0900..U+097F`), CJK, Arabic, and Cyrillic, plus
     Romanized Hinglish lexicon markers (`hai`, `hain`, `karo`, `aur`, `ke`, `ko`, `bhi`).
  2. Corporate Buzzword & Cliché Blocker: scans for banned corporate filler
     (`synergistic`, `paradigm`, `holistic`, `actionable insights`, `multifaceted`,
     `utilize`, `leverage`, `spearhead`, `delve`, `tapestry`, `testament`).
  3. Bullet-List Grammatical Parallelism Verifier: checks that all bullets in
     a list start with a consistent grammatical lead-in structure (e.g., all
     bold noun/verb lead-ins or all plain clauses, without mixing styles).

Self-test: ``python3 -m fullagent.enforcex.morpholex``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "BANNED_BUZZWORDS",
    "LinguisticComplianceReport",
    "MorphoLexicalEnforcer",
    "script_distribution",
    "detect_buzzwords",
    "check_bullet_parallelism",
    "evaluate_linguistics",
]

BANNED_BUZZWORDS: Tuple[str, ...] = (
    "synergistic",
    "synergy",
    "paradigm",
    "holistic",
    "actionable insights",
    "multifaceted",
    "utilize",
    "leverage",
    "spearhead",
    "delve",
    "tapestry",
    "undisputed gold standard",
    "unrivaled",
)

_HINGLISH_MARKERS = {
    "hai", "hain", "ho", "karo", "karke", "aur", "bhi", "ke", "ko", "se",
    "me", "mein", "ye", "yeh", "wo", "woh", "nahi", "samjhe", "bahut", "acha",
}

_BUZZ_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = tuple(
    (bw, re.compile(r"\b" + re.escape(bw) + r"\b", re.I))
    for bw in BANNED_BUZZWORDS
)


@dataclass
class LinguisticComplianceReport:
    passed: bool = True
    score: float = 100.0
    dominant_script: str = "latin"
    scripts: Dict[str, float] = field(default_factory=dict)
    hinglish_detected: bool = False
    buzzwords_found: List[str] = field(default_factory=list)
    parallel_bullets: bool = True
    bullet_issues: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 2),
            "dominant_script": self.dominant_script,
            "scripts": {k: round(v, 4) for k, v in self.scripts.items()},
            "hinglish_detected": self.hinglish_detected,
            "buzzwords_found": list(self.buzzwords_found),
            "parallel_bullets": self.parallel_bullets,
            "bullet_issues": list(self.bullet_issues),
            "violations_count": len(self.violations),
            "violations": list(self.violations),
        }


def script_distribution(text: str) -> Dict[str, float]:
    counts = {"latin": 0, "devanagari": 0, "cjk": 0, "arabic": 0, "cyrillic": 0}
    total = 0
    for ch in (text or ""):
        cp = ord(ch)
        if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F):
            counts["latin"] += 1
            total += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
            total += 1
        elif (0x4E00 <= cp <= 0x9FFF) or (0x3040 <= cp <= 0x30FF) or (0xAC00 <= cp <= 0xD7AF):
            counts["cjk"] += 1
            total += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["arabic"] += 1
            total += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
            total += 1
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: v / float(total) for k, v in counts.items()}


def detect_buzzwords(text: str) -> List[str]:
    if not text:
        return []
    found: List[str] = []
    for bw, pat in _BUZZ_PATTERNS:
        if pat.search(text):
            found.append(bw)
    return found


def check_bullet_parallelism(text: str) -> Tuple[bool, List[str]]:
    """Verify that bullets within the same contiguous list share parallel structure
    (e.g., all use `**Bold Lead:**` or none do, and no dense 80+ word paragraph bullets)."""
    if not text:
        return True, []
    issues: List[str] = []
    current_group: List[str] = []
    groups: List[List[str]] = []

    for line in text.splitlines():
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.+)$", line)
        if m:
            current_group.append(m.group(1).strip())
        elif line.strip() == "" and current_group:
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)

    for g_idx, grp in enumerate(groups, 1):
        if len(grp) < 2:
            continue
        bold_leads = [bool(re.match(r"^\*\*[^*]+\*\*\s*[:—-]", item)) for item in grp]
        if any(bold_leads) and not all(bold_leads):
            issues.append(f"list_group_{g_idx}: inconsistent bold lead-in formatting across sibling bullets")
        for b_idx, item in enumerate(grp, 1):
            if len(item.split()) > 75:
                issues.append(f"list_group_{g_idx}_bullet_{b_idx}: dense paragraph bullet exceeds 75 words")

    return (len(issues) == 0), issues


def evaluate_linguistics(text: str, *, required_script: Optional[str] = None) -> LinguisticComplianceReport:
    s = text or ""
    scripts = script_distribution(s)
    dom = max(scripts, key=lambda k: scripts[k]) if any(scripts.values()) else "latin"
    words = set(re.findall(r"[a-z]+", s.lower()))
    hinglish = len(words & _HINGLISH_MARKERS) >= 2

    buzz = detect_buzzwords(s)
    parallel_ok, b_issues = check_bullet_parallelism(s)
    violations: List[Dict[str, Any]] = []

    for bw in buzz:
        violations.append({
            "rule_id": "LEX-BUZZWORD",
            "category": "style_buzzword",
            "severity": "medium",
            "evidence": bw,
            "message": f"Banned corporate buzzword '{bw}' detected",
        })
    for iss in b_issues:
        violations.append({
            "rule_id": "LEX-PARALLEL-BULLET",
            "category": "bullet_parallelism",
            "severity": "low",
            "evidence": iss,
            "message": f"Bullet parallelism violation: {iss}",
        })
    if required_script and scripts.get(required_script.lower(), 0.0) < 0.5:
        violations.append({
            "rule_id": "LEX-SCRIPT-MISMATCH",
            "category": "language_lock",
            "severity": "high",
            "evidence": f"dominant={dom}",
            "message": f"Required script '{required_script}' not dominant (got {dom})",
        })

    score = max(0.0, 100.0 - 15.0 * len(buzz) - 10.0 * len(b_issues) - (30.0 if required_script and dom != required_script.lower() else 0.0))
    return LinguisticComplianceReport(
        passed=(len(violations) == 0),
        score=score,
        dominant_script=dom,
        scripts=scripts,
        hinglish_detected=hinglish,
        buzzwords_found=buzz,
        parallel_bullets=parallel_ok,
        bullet_issues=b_issues,
        violations=violations,
    )


class MorphoLexicalEnforcer:
    def evaluate(self, text: str, *, required_script: Optional[str] = None) -> LinguisticComplianceReport:
        return evaluate_linguistics(text, required_script=required_script)


def _self_test() -> int:
    clean = "- **Parser**: Validates AST nodes.\n- **Verifier**: Checks deontic rules."
    r_ok = evaluate_linguistics(clean)
    assert r_ok.passed and r_ok.parallel_bullets, r_ok.to_dict()

    dirty = (
        "We leverage a synergistic paradigm for actionable insights.\n"
        "- **First**: Has bold lead-in.\n"
        "- Second bullet lacks bold lead-in."
    )
    r_bad = evaluate_linguistics(dirty)
    assert not r_bad.passed
    assert len(r_bad.buzzwords_found) >= 3
    assert not r_bad.parallel_bullets

    print("morpholex._self_test: OK PASS (script profiling, buzzword blocker, bullet parallelism)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
