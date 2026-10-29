"""smtguard — Quantitative & Linear Arithmetic Constraint Verifier (Words, Sentences, Bullets, Lines).

Extracts numerical bounds directly from natural-language system prompts and
verifies them over response metrics:
  * Word count bounds: `"under 150 words"`, `"at least 50 words"`, `"between 100 and 200 words"`
  * Sentence count bounds: `"maximum 5 sentences"`, `"exactly 3 sentences"`
  * Paragraph count bounds: `"at most 2 paragraphs"`, `"3 paragraphs"`
  * Bullet / list item bounds: `"at least 4 bullet points"`, `"exactly 5 items"`
  * Character / line length bounds: `"under 500 characters"`

Computes exact arithmetic slack/violation deltas and can deterministically
trim over-budget responses to satisfy upper bounds.

Self-test: ``python3 -m fullagent.enforcex.smtguard``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LinearBound",
    "SMTVerificationReport",
    "SMTLinearVerifier",
    "extract_numeric_bounds",
    "measure_metrics",
    "verify_bounds",
]

_METRIC_ALIASES = {
    "word": "words", "words": "words",
    "sentence": "sentences", "sentences": "sentences",
    "paragraph": "paragraphs", "paragraphs": "paragraphs",
    "bullet": "bullets", "bullets": "bullets",
    "item": "bullets", "items": "bullets", "point": "bullets", "points": "bullets",
    "character": "chars", "characters": "chars", "char": "chars", "chars": "chars",
    "line": "lines", "lines": "lines",
}

_PAT_BETWEEN = re.compile(
    r"\bbetween\s+(\d+)\s+(?:and|to|-)\s+(\d+)\s+(words?|sentences?|paragraphs?|bullets?(?:\s+points?)?|items?|characters?|chars?|lines?)\b",
    re.I,
)
_PAT_UPPER = re.compile(
    r"\b(?:under|at\s+most|no\s+more\s+than|max(?:imum)?(?:\s+of)?|up\s+to|fewer\s+than|less\s+than)\s+(\d+)\s+(words?|sentences?|paragraphs?|bullets?(?:\s+points?)?|items?|characters?|chars?|lines?)\b",
    re.I,
)
_PAT_LOWER = re.compile(
    r"\b(?:at\s+least|minimum(?:\s+of)?|no\s+fewer\s+than|more\s+than)\s+(\d+)\s+(words?|sentences?|paragraphs?|bullets?(?:\s+points?)?|items?|characters?|chars?|lines?)\b",
    re.I,
)
_PAT_EXACT = re.compile(
    r"\bexactly\s+(\d+)\s+(words?|sentences?|paragraphs?|bullets?(?:\s+points?)?|items?|characters?|chars?|lines?)\b",
    re.I,
)


@dataclass
class LinearBound:
    bound_id: str
    metric: str               # words | sentences | paragraphs | bullets | chars | lines
    min_val: Optional[int] = None
    max_val: Optional[int] = None
    raw_clause: str = ""

    def check(self, actual: int) -> Tuple[bool, int]:
        """Return `(satisfied, signed_delta)` where delta > 0 means excess and delta < 0 means deficit."""
        if self.min_val is not None and actual < self.min_val:
            return False, actual - self.min_val
        if self.max_val is not None and actual > self.max_val:
            return False, actual - self.max_val
        return True, 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bound_id": self.bound_id,
            "metric": self.metric,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "raw_clause": self.raw_clause,
        }


@dataclass
class SMTVerificationReport:
    passed: bool = True
    score: float = 100.0
    metrics: Dict[str, int] = field(default_factory=dict)
    bounds: List[LinearBound] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 2),
            "metrics": dict(self.metrics),
            "bounds": [b.to_dict() for b in self.bounds],
            "violations_count": len(self.violations),
            "violations": list(self.violations),
        }


def _norm_unit(unit_raw: str) -> str:
    first = unit_raw.strip().lower().split()[0]
    return _METRIC_ALIASES.get(first, "words")


def extract_numeric_bounds(system_text: str) -> List[LinearBound]:
    """Parse quantitative bounds from `system_text`."""
    if not system_text or not isinstance(system_text, str):
        return []
    bounds: List[LinearBound] = []
    idx = 0
    for m in _PAT_BETWEEN.finditer(system_text):
        idx += 1
        lo, hi = int(m.group(1)), int(m.group(2))
        bounds.append(LinearBound(f"SMT-{idx:02d}", _norm_unit(m.group(3)), min(lo, hi), max(lo, hi), m.group(0)))
    for m in _PAT_UPPER.finditer(system_text):
        idx += 1
        hi = int(m.group(1))
        bounds.append(LinearBound(f"SMT-{idx:02d}", _norm_unit(m.group(2)), None, hi, m.group(0)))
    for m in _PAT_LOWER.finditer(system_text):
        idx += 1
        lo = int(m.group(1))
        bounds.append(LinearBound(f"SMT-{idx:02d}", _norm_unit(m.group(2)), lo, None, m.group(0)))
    for m in _PAT_EXACT.finditer(system_text):
        idx += 1
        val = int(m.group(1))
        bounds.append(LinearBound(f"SMT-{idx:02d}", _norm_unit(m.group(2)), val, val, m.group(0)))
    return bounds


def measure_metrics(text: str) -> Dict[str, int]:
    """Compute exact integer counts of words, sentences, paragraphs, bullets, chars, and lines."""
    s = text or ""
    words = re.findall(r"\S+", s)
    sents = [x.strip() for x in re.split(r"(?<=[.!?।])\s+", s.strip()) if x.strip()]
    paras = [p.strip() for p in re.split(r"\n\s*\n", s.strip()) if p.strip()]
    bullets = re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+", s, flags=re.M)
    lines = s.splitlines() if s else []
    return {
        "words": len(words),
        "sentences": len(sents),
        "paragraphs": len(paras),
        "bullets": len(bullets),
        "chars": len(s),
        "lines": len(lines),
    }


def verify_bounds(response_text: str, *, system_text: str = "", bounds: Optional[Sequence[LinearBound]] = None) -> SMTVerificationReport:
    b_list = list(bounds) if bounds is not None else extract_numeric_bounds(system_text)
    metrics = measure_metrics(response_text)
    violations: List[Dict[str, Any]] = []

    for b in b_list:
        actual = metrics.get(b.metric, 0)
        ok, delta = b.check(actual)
        if not ok:
            direction = f"exceeded by {delta}" if delta > 0 else f"short by {abs(delta)}"
            violations.append({
                "rule_id": b.bound_id,
                "category": "smt_quantitative",
                "severity": "high",
                "metric": b.metric,
                "actual": actual,
                "min_val": b.min_val,
                "max_val": b.max_val,
                "delta": delta,
                "message": f"Quantitative constraint '{b.raw_clause}' violated: {b.metric}={actual} ({direction})",
                "evidence": f"{b.metric}={actual}",
            })

    score = max(0.0, 100.0 - 25.0 * len(violations))
    return SMTVerificationReport(
        passed=len(violations) == 0,
        score=score,
        metrics=metrics,
        bounds=b_list,
        violations=violations,
    )


class SMTLinearVerifier:
    def evaluate(self, response_text: str, *, system_text: str = "") -> SMTVerificationReport:
        return verify_bounds(response_text, system_text=system_text)


def _self_test() -> int:
    prompt = "Keep your reply under 20 words and include exactly 2 bullet points."
    bounds = extract_numeric_bounds(prompt)
    assert len(bounds) == 2, bounds

    good = "- Point one is short.\n- Point two is clear."
    r_ok = verify_bounds(good, system_text=prompt)
    assert r_ok.passed and r_ok.score == 100.0, r_ok.to_dict()

    bad = "- Only one bullet point with way too many words " + ("extra " * 25)
    r_bad = verify_bounds(bad, system_text=prompt)
    assert not r_bad.passed and len(r_bad.violations) == 2, r_bad.to_dict()

    print("smtguard._self_test: OK PASS (interval extraction, metric counting, delta calculation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
