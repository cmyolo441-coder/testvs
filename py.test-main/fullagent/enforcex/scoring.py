"""scoring — Weighted violation scoring, grading and drift tracking.

Pure functions + a small stateful drift tracker that turn a list of
violations (as produced by detect.run_all / responseanaly /
injection.scan, or by the gate's local fallbacks) into:

  * a per-category weighted penalty,
  * a total score in [0, 100] (100 = perfectly compliant),
  * a letter grade (A+ .. F),
  * a verdict (pass / warn / fail),
  * a drift signal (is compliance improving or degrading over time?).

Design notes
------------
* Violations are dicts: {"category": str, "severity": str,
  "rule_id": str, "evidence": str, "confidence": float, ...}.
  Unknown fields are tolerated; missing fields get defaults.
* Category weights encode how much each category *matters*; severity
  multipliers encode how much a violation *hurts*; confidence scales
  the penalty down for low-confidence detections.
* Diminishing penalties: the Nth violation in one category contributes
  less than the first, so a wall of small nits does not nuke a reply
  that is otherwise fine.
* Everything is persistable via plain dicts (Score.to_dict /
  DriftTracker.to_dict + from_dict). Thread-safe via a lock in the
  tracker. Stdlib only; never raises on malformed input.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "CATEGORY_WEIGHTS",
    "SEVERITY_MULTIPLIERS",
    "GRADE_THRESHOLDS",
    "Score",
    "normalize_violation",
    "score_violations",
    "DriftTracker",
]


# ── weight tables ────────────────────────────────────────────────────────

CATEGORY_WEIGHTS: Dict[str, float] = {
    "identity": 3.0,
    "security": 3.0,
    "injection": 3.0,
    "leak": 2.8,
    "content_policy": 2.5,
    "safety": 2.5,
    "format": 1.5,
    "process": 1.4,
    "factual": 1.3,
    "language": 1.2,
    "scope": 1.2,
    "tone": 1.0,
    "style": 1.0,
    "other": 1.0,
    # previously missing sub-signal weights: "quality" (repetition /
    # structural quality from responseanaly) used to collapse into
    # "other" = 1.0 and was clearly under-weighted; "hint" covers
    # covert self-disclosure of instructions, "refusal" covers
    # blanket refusal of a benign job.
    "quality": 1.2,
    "hint": 1.8,
    "refusal": 1.5,
}

# Category aliases: spellings that must score identically to the
# canonical key (lookup only; the stored value is never rewritten).
_CATEGORY_ALIASES: Dict[str, str] = {
    "content-policy": "content_policy",
    "content policy": "content_policy",
    "contentpolicy": "content_policy",
    "prompt_leak": "leak",
    "prompt-leak": "leak",
    "promptleak": "leak",
    "jailbreak": "injection",
    "prompt-injection": "injection",
    "prompt_injection": "injection",
    "leakage": "leak",
    "infosec": "security",
}

SEVERITY_MULTIPLIERS: Dict[str, float] = {
    "critical": 3.0,
    "high": 2.0,
    "medium": 1.2,
    "low": 0.6,
    "info": 0.2,
}

# Severity aliases so hostile reports phrased differently still bind.
_SEVERITY_ALIASES: Dict[str, str] = {
    "fatal": "critical",
    "urgent": "critical",
    "blocker": "critical",
    "severe": "high",
    "major": "high",
    "important": "high",
    "moderate": "medium",
    "normal": "medium",
    "minor": "low",
    "trivial": "low",
    "nit": "low",
    "informational": "info",
    "notice": "info",
    "hint": "info",
}

GRADE_THRESHOLDS: Sequence[tuple[float, str]] = (
    (97.0, "A+"),
    (92.0, "A"),
    (85.0, "B"),
    (75.0, "C"),
    (60.0, "D"),
    (40.0, "E"),
    (0.0, "F"),
)

_DEFAULT_SEVERITY = "medium"
_DEFAULT_CONFIDENCE = 1.0
# Penalty budget per category before diminishing returns kick in fully.
_CATEGORY_SATURATION = 12.0
# Diminishing-returns exponent (>1 = concave).
_DIMINISH_EXPONENT = 1.35
# Bound free-text fields copied into normalized violations (DoS guard).
_MAX_FIELD = 8192
# Hard cap on violations scored per call; the count is still exact.
_MAX_VIOLATIONS = 50_000
# Word tokenizer used for tolerant string / kind classification.
_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_SEVERITY_ORDER = ("critical", "high", "medium", "low", "info")
_CATEGORY_ORDER = ("identity", "security", "injection", "leak",
                   "content_policy", "safety", "format", "process",
                   "factual", "language", "scope", "tone", "style")


def _finite_float(value: Any, default: float) -> float:
    """Coerce to a finite float; return ``default`` on anything else."""
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_field(value: Any, cap: int = _MAX_FIELD) -> str:
    """Stringify a field defensively and bound its length."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        return ""
    return text[:cap]


def _category_weight(category: str) -> float:
    key = (category or "other").strip().lower()
    key = _CATEGORY_ALIASES.get(key, key)
    return CATEGORY_WEIGHTS.get(key, CATEGORY_WEIGHTS["other"])


def _severity_multiplier(severity: str) -> float:
    key = (severity or _DEFAULT_SEVERITY).strip().lower()
    key = _SEVERITY_ALIASES.get(key, key)
    return SEVERITY_MULTIPLIERS.get(key, SEVERITY_MULTIPLIERS[_DEFAULT_SEVERITY])


def _clamp_score(value: Any) -> float:
    """Bounded 0..100 score; NaN/inf collapse to 0.0 (fail closed)."""
    return max(0.0, min(100.0, _finite_float(value, 0.0)))


# ── dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Score:
    """Final compliance score for one response."""
    total: float                     # 0..100, higher = more compliant
    grade: str                       # "A+" .. "F"
    category_scores: Dict[str, float]  # category -> 0..100 sub-score
    violations_count: int
    drift: float = 0.0               # filled by DriftTracker.update()
    verdict: str = "pass"            # pass | warn | fail

    def to_dict(self) -> Dict[str, Any]:
        cats: Dict[str, float] = {}
        for k, v in (self.category_scores or {}).items():
            cats[str(k)] = round(_clamp_score(v), 2)
        return {
            "total": round(_clamp_score(self.total), 2),
            "grade": str(self.grade),
            "category_scores": cats,
            "violations_count": max(0, int(self.violations_count or 0)),
            "drift": round(_finite_float(self.drift, 0.0), 4),
            "verdict": str(self.verdict),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Score":
        try:
            data = d if isinstance(d, dict) else {}
            cats: Dict[str, float] = {}
            raw_cats = data.get("category_scores")
            if isinstance(raw_cats, dict):
                for k, v in list(raw_cats.items())[:512]:
                    cats[str(k)] = _clamp_score(v)
            try:
                count = max(0, int(data.get("violations_count", 0) or 0))
            except (TypeError, ValueError, OverflowError):
                count = 0
            grade = str(data.get("grade", "F") or "F")
            verdict = str(data.get("verdict", "fail") or "fail")
            return cls(
                total=_clamp_score(data.get("total", 0.0)),
                grade=grade,
                category_scores=cats,
                violations_count=count,
                drift=_finite_float(data.get("drift", 0.0), 0.0),
                verdict=verdict,
            )
        except Exception:
            return cls(total=0.0, grade="F", category_scores={},
                       violations_count=0, verdict="fail")


# ── normalization ────────────────────────────────────────────────────────

def normalize_violation(raw: Any) -> Dict[str, Any]:
    """Coerce anything violation-shaped into a normalized dict.

    Accepts dicts, (category, severity) tuples, bare strings, or objects
    exposing violation-like attributes.  Fields are bounded in size and
    all numeric values are finite.  Never raises.
    """
    out: Dict[str, Any] = {
        "category": "other",
        "severity": _DEFAULT_SEVERITY,
        "rule_id": "",
        "evidence": "",
        "confidence": _DEFAULT_CONFIDENCE,
        "message": "",
    }
    try:
        if isinstance(raw, dict):
            for key in ("category", "severity", "rule_id", "evidence",
                        "message", "kind", "type"):
                if key in raw and raw[key] is not None:
                    out[key] = _safe_field(raw[key])
            conf = raw.get("confidence")
            if conf is not None:
                parsed = _finite_float(conf, None)  # type: ignore[arg-type]
                if parsed is not None:
                    out["confidence"] = min(1.0, max(0.1, parsed))
            if out.get("kind") and not out["message"]:
                out["message"] = out["kind"]
        elif isinstance(raw, (tuple, list)) and len(raw) >= 1:
            out["category"] = _safe_field(raw[0])
            if len(raw) >= 2:
                out["severity"] = _safe_field(raw[1])
            if len(raw) >= 3 and raw[2] is not None:
                out["rule_id"] = _safe_field(raw[2])
        elif isinstance(raw, str):
            low = _safe_field(raw).lower()
            tokens = _WORD_RE.findall(low)
            for sev in _SEVERITY_ORDER:
                if sev in tokens:
                    out["severity"] = sev
                    break
            for cat in _CATEGORY_ORDER:
                if cat in tokens:
                    out["category"] = cat
                    break
            out["message"] = raw[:_MAX_FIELD]
        elif raw is not None:
            # Tolerant object form: category=/severity=/rule_id=/evidence=
            for key in ("category", "severity", "rule_id", "evidence",
                        "message"):
                value = getattr(raw, key, None)
                if value is not None:
                    out[key] = _safe_field(value)
            conf = getattr(raw, "confidence", None)
            if conf is not None:
                parsed = _finite_float(conf, None)  # type: ignore[arg-type]
                if parsed is not None:
                    out["confidence"] = min(1.0, max(0.1, parsed))
    except Exception:
        pass
    # canonical category/severity casing
    try:
        category = (out.get("category") or "other").strip().lower()
    except Exception:
        category = "other"
    out["category"] = category or "other"
    try:
        sev = (out.get("severity") or _DEFAULT_SEVERITY).strip().lower()
    except Exception:
        sev = _DEFAULT_SEVERITY
    sev = _SEVERITY_ALIASES.get(sev, sev)
    out["severity"] = sev if sev in SEVERITY_MULTIPLIERS else _DEFAULT_SEVERITY
    # evidence-based reporting: when no explicit evidence was supplied,
    # fall back to the message/kind so downstream explanations always
    # have a concrete anchor.
    if not out.get("evidence"):
        out["evidence"] = out.get("message", "") or out.get("kind", "")
    for key in ("category", "severity", "rule_id", "evidence", "message"):
        out[key] = _safe_field(out.get(key, ""))
    out["confidence"] = _finite_float(out.get("confidence"),
                                      _DEFAULT_CONFIDENCE)
    out["confidence"] = min(1.0, max(0.1, out["confidence"]))
    kind = out.get("kind")
    if kind is not None:
        out["kind"] = _safe_field(kind)
    return out


# ── core scoring ─────────────────────────────────────────────────────────

def _grade_for(total: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if total >= threshold:
            return grade
    return "F"


def _verdict_for(total: float, violations: List[Dict[str, Any]],
                 rules: Optional[Any]) -> str:
    has_critical = any(v["severity"] == "critical" for v in violations)
    has_high = any(v["severity"] == "high" for v in violations)
    if has_critical or total < 40.0:
        return "fail"
    if total < 75.0 or has_high:
        return "warn"
    return "pass"


def score_violations(violations: Iterable[Any], rules: Optional[Any] = None) -> Score:
    """Compute a weighted Score from a list of raw violations.

    ``rules`` is optional context (a rule list/dict); it is tolerated
    but not required — scoring only needs the violations themselves.

    The mapping is monotonic: adding a violation never raises the score,
    and every returned value is finite and bounded to [0, 100].
    """
    normalized: List[Dict[str, Any]] = []
    has_critical = False
    has_high = False
    total_count = 0
    try:
        iterator = iter(violations or [])
    except Exception:
        iterator = iter(())
    for raw in iterator:
        if raw is None:
            continue
        total_count += 1
        if total_count > _MAX_VIOLATIONS:
            # Keep counting without storing unbounded input; the penalty
            # is already saturated at this point for any sane weighting.
            continue
        v = normalize_violation(raw)
        if v["severity"] == "critical":
            has_critical = True
        elif v["severity"] == "high":
            has_high = True
        normalized.append(v)

    # group raw penalties per category
    per_category: Dict[str, List[float]] = {}
    for v in normalized:
        weight = _category_weight(v["category"])
        mult = _severity_multiplier(v["severity"])
        conf = min(1.0, max(0.0, _finite_float(v.get("confidence"), 1.0)))
        raw_penalty = 10.0 * weight * mult * conf
        if not math.isfinite(raw_penalty):
            continue
        per_category.setdefault(v["category"], []).append(raw_penalty)

    category_penalties: Dict[str, float] = {}
    for cat, penalties in per_category.items():
        penalties.sort(reverse=True)
        # diminishing penalties: sum(p_i / i) for i = 1..n, capped softly
        total = 0.0
        for i, p in enumerate(penalties, start=1):
            total += p / (i ** (1.0 / _DIMINISH_EXPONENT))
        # saturate: category can never eat more than _CATEGORY_SATURATION
        # * weight points off the total
        cap = _CATEGORY_SATURATION * _category_weight(cat)
        if total > cap:
            total = cap + (total - cap) * 0.25  # soft overshoot allowed
        category_penalties[cat] = total if math.isfinite(total) else cap

    gross_penalty = sum(category_penalties.values())
    if not math.isfinite(gross_penalty):
        gross_penalty = 1e9
    # convert gross penalty (unbounded-ish) into a 0..100 score via a
    # smooth curve: score = 100 * exp(-penalty / 60)
    total_score = 100.0 * math.exp(-gross_penalty / 60.0)
    total_score = _clamp_score(total_score)
    if has_critical:
        # fail closed: any confirmed critical violation caps the total
        # regardless of confidence scaling (monotone: a cap only lowers)
        total_score = min(total_score, 60.0)
    elif has_high:
        total_score = min(total_score, 74.0)
    elif total_score < 75.0 and (gross_penalty > 1e9 or not math.isfinite(gross_penalty)):
        total_score = 0.0

    category_scores: Dict[str, float] = {}
    for cat in sorted(set(per_category) | {v["category"] for v in normalized}):
        pen = category_penalties.get(cat, 0.0)
        category_scores[cat] = _clamp_score(100.0 * math.exp(-pen / 30.0))

    grade = _grade_for(total_score)
    verdict_violations = normalized
    if has_critical or has_high:
        # ensure severity seen beyond the storage cap still drives verdict
        verdict_violations = normalized + [{
            "severity": "critical" if has_critical else "high"}]
    verdict = _verdict_for(total_score, verdict_violations, rules)
    return Score(
        total=total_score,
        grade=grade,
        category_scores=category_scores,
        violations_count=total_count,
        drift=0.0,
        verdict=verdict,
    )


# ── drift tracking ───────────────────────────────────────────────────────

class DriftTracker:
    """Exponential moving average of scores with trend direction.

    Usage::

        t = DriftTracker()
        t.update(score_a); t.update(score_b)
        print(t.summary())
    """

    def __init__(self, alpha: float = 0.3, window: int = 100) -> None:
        try:
            alpha_f = float(alpha)
        except (TypeError, ValueError, OverflowError):
            alpha_f = 0.3
        self._alpha = alpha_f if 0.0 < alpha_f <= 1.0 else 0.3
        try:
            window_i = int(window)
        except (TypeError, ValueError, OverflowError):
            window_i = 100
        self._window = max(10, min(10_000, window_i))
        self._ema: Optional[float] = None
        self._count = 0
        self._scores: List[float] = []
        self._last: Optional[Score] = None
        self._lock = threading.Lock()

    def update(self, score: Any) -> Score:
        """Feed a Score (or numeric total). Returns the updated Score
        with its ``drift`` field set to (ema_after - ema_before)."""
        raw_value = getattr(score, "total", score)
        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError):
            value = float("nan")
        if not math.isfinite(value):
            # malformed sample: leave the tracker untouched, return a
            # graceful zero drift instead of raising or poisoning the EMA
            with self._lock:
                if isinstance(score, Score):
                    score.drift = 0.0
                    return score
                return self._last or Score(total=0.0, grade="F",
                                           category_scores={},
                                           violations_count=0,
                                           verdict="fail")
        value = max(0.0, min(100.0, value))
        with self._lock:
            before = self._ema
            if self._ema is None:
                self._ema = value
                drift = 0.0
            else:
                self._ema = self._alpha * value + (1.0 - self._alpha) * self._ema
                drift = (self._ema or 0.0) - (before or 0.0)
            self._count += 1
            self._scores.append(value)
            if len(self._scores) > self._window:
                self._scores = self._scores[-self._window:]
            if isinstance(score, Score):
                score.drift = drift
                self._last = score
                return score
            last = self._last or Score(total=value, grade=_grade_for(value),
                                       category_scores={}, violations_count=0)
            last.drift = drift
            self._last = last
            return last

    # -- introspection ----------------------------------------------------

    @property
    def ema(self) -> Optional[float]:
        with self._lock:
            return self._ema

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def trend(self) -> str:
        """One of: improving | degrading | stable | unknown."""
        with self._lock:
            if len(self._scores) < 3:
                return "unknown"
            half = max(1, len(self._scores) // 2)
            older = sum(self._scores[:-half]) / max(1, len(self._scores) - half)
            newer = sum(self._scores[-half:]) / half
            delta = newer - older
            if delta > 2.0:
                return "improving"
            if delta < -2.0:
                return "degrading"
            return "stable"

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            scores = list(self._scores)
        if not scores:
            return {"count": 0, "ema": None, "trend": "unknown",
                    "min": None, "max": None, "mean": None}
        half = max(1, len(scores) // 2)
        return {
            "count": self._count,
            "ema": round(self._ema, 2) if self._ema is not None else None,
            "trend": self.trend(),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
            "mean": round(sum(scores) / len(scores), 2),
            "recent_mean": round(sum(scores[-half:]) / half, 2),
        }

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "alpha": self._alpha,
                "window": self._window,
                "ema": self._ema,
                "count": self._count,
                "scores": list(self._scores),
            }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DriftTracker":
        data = d if isinstance(d, dict) else {}
        try:
            tracker = cls(alpha=data.get("alpha", 0.3),
                          window=data.get("window", 100))
        except Exception:
            return cls()
        try:
            ema = data.get("ema")
            if isinstance(ema, (int, float)):
                ema_f = float(ema)
                tracker._ema = ema_f if math.isfinite(ema_f) else None
            tracker._count = max(0, int(data.get("count", 0) or 0))
            raw_scores = data.get("scores", [])
            scores: List[float] = []
            for item in list(raw_scores)[:tracker._window] if isinstance(
                    raw_scores, (list, tuple)) else []:
                value = _finite_float(item, float("nan"))
                if math.isfinite(value):
                    scores.append(max(0.0, min(100.0, value)))
            tracker._scores = scores[-tracker._window:]
        except Exception:
            return cls()
        return tracker


# ── explanations / breakdown ─────────────────────────────────────────────

@dataclass
class PenaltyLine:
    """One row of a human-readable penalty breakdown."""
    category: str
    severity: str
    rule_id: str
    penalty: float
    capped: bool
    detail: str


def penalty_breakdown(violations: Iterable[Any]) -> List[PenaltyLine]:
    """Explain exactly where the score went, line by line.

    Useful for reports and retry prompts: each row shows the raw
    weighted penalty for one violation and whether the per-category
    cap kicked in. Violations are listed biggest-penalty-first.
    """
    lines: List[PenaltyLine] = []
    per_category_totals: Dict[str, float] = {}
    ordered = [normalize_violation(v) for v in (violations or []) if v is not None]
    # Deterministic ordering: biggest penalty first, then stable textual
    # tie-breakers so equal-weight violations never reorder between runs.
    decorated = []
    for v in ordered:
        weight = _category_weight(v["category"])
        mult = _severity_multiplier(v["severity"])
        conf = min(1.0, max(0.0, _finite_float(v.get("confidence"), 1.0)))
        raw = 10.0 * weight * mult * conf
        decorated.append((raw, v))
    decorated.sort(key=lambda item: (
        -round(item[0], 9),
        str(item[1].get("category", "")),
        str(item[1].get("severity", "")),
        str(item[1].get("rule_id", "")),
        str(item[1].get("evidence", ""))[:64],
        str(item[1].get("message", ""))[:64],
    ))
    for raw, v in decorated:
        weight = _category_weight(v["category"])
        mult = _severity_multiplier(v["severity"])
        conf = min(1.0, max(0.0, _finite_float(v.get("confidence"), 1.0)))
        cat = v["category"]
        per_category_totals[cat] = per_category_totals.get(cat, 0.0) + raw
        cap = _CATEGORY_SATURATION * weight
        capped = per_category_totals[cat] > cap
        detail = (
            f"weight={weight:.1f} x sev={mult:.1f} x conf={conf:.2f}"
            f" -> {raw:.2f} pts"
            + (" (category cap applied)" if capped else "")
        )
        lines.append(PenaltyLine(
            category=cat,
            severity=v["severity"],
            rule_id=str(v.get("rule_id", "")),
            penalty=round(raw, 2),
            capped=capped,
            detail=detail,
        ))
    return lines


def explain(score: Score, violations: Iterable[Any] = ()) -> str:
    """Short plain-English explanation of a Score."""
    try:
        total = _clamp_score(getattr(score, "total", 0.0))
        grade = str(getattr(score, "grade", "F"))
        verdict = str(getattr(score, "verdict", "fail"))
        count = max(0, int(getattr(score, "violations_count", 0) or 0))
        drift = _finite_float(getattr(score, "drift", 0.0), 0.0)
    except Exception:
        total, grade, verdict, count, drift = 0.0, "F", "fail", 0, 0.0
    parts: List[str] = [f"score {total:.1f}/100 (grade {grade}, "
                        f"verdict {verdict})"]
    if count:
        parts.append(f"{count} violation(s)")
    breakdown = penalty_breakdown(violations)
    if breakdown:
        worst = breakdown[0]
        parts.append(f"worst: {worst.category}/{worst.severity} "
                     f"({worst.penalty:.1f} pts)")
    if drift:
        direction = "up" if drift > 0 else "down"
        parts.append(f"drift {direction} {abs(drift):.1f}")
    return "; ".join(parts)


def merge_history(scores: Iterable[Any]) -> Dict[str, Any]:
    """Aggregate a sequence of Score objects (or dicts) into summary
    statistics suitable for persistence alongside a DriftTracker."""
    totals: List[float] = []
    grades: Dict[str, int] = {}
    verdicts: Dict[str, int] = {}
    try:
        iterator = iter(scores or [])
    except Exception:
        iterator = iter(())
    for item in iterator:
        try:
            if isinstance(item, dict):
                total = item.get("total")
                grade = item.get("grade", "?")
                verdict = item.get("verdict", "?")
            else:
                total = getattr(item, "total", None)
                grade = getattr(item, "grade", "?")
                verdict = getattr(item, "verdict", "?")
        except Exception:
            continue
        value = _finite_float(total, float("nan"))
        if math.isfinite(value):
            totals.append(max(0.0, min(100.0, value)))
        gkey = _safe_field(grade, 32)
        vkey = _safe_field(verdict, 32)
        grades[gkey] = grades.get(gkey, 0) + 1
        verdicts[vkey] = verdicts.get(vkey, 0) + 1
    if not totals:
        return {"count": 0}
    return {
        "count": len(totals),
        "min": round(min(totals), 2),
        "max": round(max(totals), 2),
        "mean": round(sum(totals) / len(totals), 2),
        "grades": grades,
        "verdicts": verdicts,
    }


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> None:  # pragma: no cover
    # clean response
    s0 = score_violations([])
    assert s0.total > 99.0 and s0.grade in ("A+", "A") and s0.verdict == "pass", s0

    # one critical security violation should fail hard
    s1 = score_violations([
        {"category": "security", "severity": "critical",
         "rule_id": "SEC-1", "evidence": "ignored instructions"},
    ])
    assert s1.total < 80.0 and s1.verdict == "fail", s1

    # low-severity nits should only warn at worst
    s2 = score_violations([
        {"category": "tone", "severity": "low"},
        {"category": "format", "severity": "low"},
    ])
    assert s2.total > 70.0, s2

    # diminishing: 5 tiny violations < 1 critical in impact ordering
    nits = score_violations([
        {"category": "tone", "severity": "low"} for _ in range(5)
    ])
    hard = score_violations([{"category": "security", "severity": "critical"}])
    assert hard.total < nits.total

    # normalization of weird input
    v = normalize_violation(("security", "critical"))
    assert v["category"] == "security" and v["severity"] == "critical"
    v = normalize_violation(None)
    assert v["category"] == "other"

    # dict round-trip
    d = s1.to_dict()
    again = Score.from_dict(d)
    assert abs(again.total - s1.total) < 0.01 and again.grade == s1.grade

    # drift tracker
    t = DriftTracker()
    for total in (90.0, 85.0, 80.0, 70.0, 60.0):
        t.update(Score(total=total, grade="C", category_scores={},
                       violations_count=1, verdict="warn"))
    summ = t.summary()
    assert summ["count"] == 5 and summ["trend"] == "degrading", summ
    restored = DriftTracker.from_dict(t.to_dict())
    assert restored.count == 5 and restored.ema == t.ema
    t2 = DriftTracker()
    for total in (60.0, 70.0, 80.0, 90.0):
        t2.update(total)
    assert t2.summary()["trend"] == "improving"

    # breakdown + explain + merge
    lines = penalty_breakdown([
        {"category": "security", "severity": "critical", "rule_id": "S1"},
        {"category": "tone", "severity": "low"},
    ])
    assert lines[0].penalty >= lines[-1].penalty and lines[0].capped is True
    assert lines[-1].severity == "low" and lines[-1].capped is False
    text = explain(s1, [{"category": "security", "severity": "critical"}])
    assert "score" in text and "worst" in text
    hist = merge_history([s1.to_dict(), s2.to_dict()])
    assert hist["count"] == 2 and "mean" in hist
    assert merge_history([]) == {"count": 0}

    # ── adversarial additions ─────────────────────────────────────────
    # monotonicity: appending violations can never raise the score
    base_v = [
        {"category": "tone", "severity": "low"},
        {"category": "security", "severity": "high", "confidence": 0.7},
        {"category": "format", "severity": "medium", "confidence": 0.4},
    ]
    prev_score = score_violations([]).total
    for i in range(1, len(base_v) + 1):
        cur = score_violations(base_v[:i]).total
        assert cur <= prev_score + 1e-9, (i, cur, prev_score)
        prev_score = cur
    dup = score_violations(base_v * 3)
    assert dup.total <= score_violations(base_v).total + 1e-9

    # NaN / inf / non-numeric confidence never poison the score
    for bad_conf in (float("nan"), float("inf"), "not-a-number", None, -5, 1e300):
        s_bad = score_violations([{"category": "security", "severity": "high",
                                   "confidence": bad_conf}])
        assert math.isfinite(s_bad.total) and 0.0 <= s_bad.total <= 100.0, s_bad
        assert all(math.isfinite(v) for v in s_bad.category_scores.values())
    assert score_violations([{"confidence": float("nan")}]).verdict in (
        "pass", "warn", "fail")

    # normalization fixes: kind becomes message/evidence, substring
    # "low" in "follow" no longer mislabels severity
    kv = normalize_violation({"kind": "leak_tag", "category": "leak"})
    assert kv["message"] == "leak_tag" and kv["evidence"] == "leak_tag", kv
    assert normalize_violation("please follow the style guide")["severity"] == "medium"
    assert normalize_violation("critical security issue")["severity"] == "critical"
    assert normalize_violation("critical security issue")["category"] == "security"
    assert normalize_violation((1, 2, 3))["rule_id"] == "3"
    assert normalize_violation(object())["category"] == "other"

    # deterministic breakdown for equal-weight violations
    ties = [{"category": "tone", "severity": "low", "rule_id": "B"},
            {"category": "tone", "severity": "low", "rule_id": "A"}]
    order_fwd = [ln.rule_id for ln in penalty_breakdown(ties)]
    order_rev = [ln.rule_id for ln in penalty_breakdown(list(reversed(ties)))]
    assert order_fwd == order_rev, (order_fwd, order_rev)

    # malformed persistence / tracker input never raise
    restored_bad = Score.from_dict({"total": "nan", "drift": "inf",
                                    "violations_count": "x",
                                    "category_scores": {"tone": "bad"}})
    assert restored_bad.total == 0.0 and restored_bad.drift == 0.0
    t_bad = DriftTracker(alpha="bad", window="bad")  # type: ignore[arg-type]
    t_bad.update("not-a-number")
    t_bad.update(Score(total=float("nan"), grade="F", category_scores={},
                       violations_count=0))
    assert t_bad.count == 0 and t_bad.summary()["count"] == 0
    restored_t = DriftTracker.from_dict({"alpha": "x", "scores": "bad"})
    assert restored_t.count == 0

    # huge inputs are bounded, never blow up
    huge = normalize_violation("critical " + "x" * 1_000_000)
    assert len(huge["message"]) <= _MAX_FIELD
    assert math.isfinite(score_violations([huge]).total)
    assert merge_history([{"total": float("nan"), "grade": "A"},
                          {"total": 80.5, "grade": "B"}])["count"] == 1

    print("scoring._self_test: OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
