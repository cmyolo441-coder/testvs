"""benchsuite — Verifiable Instruction-Following Benchmark (IFEval) & Elo Rating Engine.

Implements 12 deterministic, programmatic IFEval instruction verifiers
(Zhou et al., "Instruction-Following Evaluation for Large Language Models"):
  1. `no_commas`: response contains zero commas (`,`)
  2. `json_format`: entire response is valid JSON
  3. `lowercase_only`: all alphabetic characters are lowercase
  4. `uppercase_only`: all alphabetic characters are uppercase
  5. `title_wrapped`: response contains a title wrapped in `<<Title>>`
  6. `quotation_wrapped`: entire response is wrapped in double quotes
  7. `postscript`: response ends with a `P.S.` or `P.P.S.` postscript
  8. `min_words`: word count >= N
  9. `max_words`: word count <= N
  10. `exact_bullets`: bullet count == N
  11. `include_keywords`: all specified keywords appear
  12. `exclude_keywords`: none of the specified forbidden words appear

Also computes exact Elo rating updates between competing models/configurations.

Self-test: ``python3 -m fullagent.enforcex.benchsuite``
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

__all__ = [
    "BenchmarkScorecard",
    "ComplianceBenchmarkSuite",
    "verify_ifeval_constraint",
    "update_elo",
]

#: ASCII commas plus common fullwidth / Arabic comma confusables.  A "no
#: commas" constraint that ignores these is trivially evaded.
_COMMA_CHARS = (",", "\uff0c", "\u060c", "\u3001")
_TITLE_RE = re.compile(r"<<[^<>\n]{1,200}>>")
_POSTSCRIPT_RE = re.compile(r"\bP\.(?:P\.)?S\.")
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_MAX_TEXT = 1_000_000
_MAX_CONSTRAINTS = 1024


def _text(value: Any) -> str:
    """Coerce to str for constraint checks (non-str => empty, never raises)."""
    if isinstance(value, str):
        return value[:_MAX_TEXT]
    return ""


def _as_int(value: Any, default: int) -> int:
    """Best-effort int conversion; invalid/NaN inputs fall back to `default`."""
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _keywords(value: Any):
    """Normalize a keyword parameter into a case-folded tuple of str.

    A bare string is treated as one keyword (not an iterable of letters),
    ``None`` means "no keywords", and malformed iterables are ignored.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.casefold(),)
    try:
        items = list(value)
    except TypeError:
        return ()
    out = []
    for item in items:
        if isinstance(item, (str, bytes)):
            try:
                s = item.decode("utf-8", "replace") if isinstance(item, bytes) else item
            except Exception:
                continue
            out.append(unicodedata.normalize("NFKC", s).casefold())
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out.append(str(item))
    return tuple(out)


@dataclass
class BenchmarkScorecard:
    total_constraints: int = 0
    passed_constraints: int = 0
    strict_accuracy: float = 100.0
    results: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_constraints": self.total_constraints,
            "passed_constraints": self.passed_constraints,
            "strict_accuracy": round(self.strict_accuracy, 2),
            "results": dict(self.results),
        }


def verify_ifeval_constraint(name: str, response_text: str, param: Any = None) -> bool:
    """Check one IFEval constraint.  Malformed inputs fail closed (False)."""
    if not isinstance(name, str):
        return False
    if not isinstance(response_text, str):
        return False
    s = response_text[:_MAX_TEXT]
    stripped = s.strip()
    if name == "no_commas":
        return not any(ch in s for ch in _COMMA_CHARS)
    if name == "json_format":
        if not stripped:
            return False
        try:
            json.loads(stripped)
            return True
        except (ValueError, RecursionError):
            return False
    if name == "lowercase_only":
        alpha = [c for c in s if c.isalpha()]
        return bool(alpha) and all(c.islower() for c in alpha)
    if name == "uppercase_only":
        alpha = [c for c in s if c.isalpha()]
        return bool(alpha) and all(c.isupper() for c in alpha)
    if name == "title_wrapped":
        return bool(_TITLE_RE.search(s))
    if name == "quotation_wrapped":
        return len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"')
    if name == "postscript":
        return bool(_POSTSCRIPT_RE.search(s))
    if name == "min_words":
        n = _as_int(param, 1)
        return len(s.split()) >= max(0, n)
    if name == "max_words":
        n = _as_int(param, 100)
        return len(s.split()) <= max(0, n)
    if name == "exact_bullets":
        n = _as_int(param, 1)
        bullets = _BULLET_RE.findall(s)
        return len(bullets) == max(0, n)
    if name == "include_keywords":
        low = unicodedata.normalize("NFKC", s).casefold()
        return all(w in low for w in _keywords(param))
    if name == "exclude_keywords":
        low = unicodedata.normalize("NFKC", s).casefold()
        return all(w not in low for w in _keywords(param))
    return False


def update_elo(rating_a: float, rating_b: float, score_a: float, *, k_factor: float = 32.0) -> Tuple[float, float]:
    """Compute updated Elo ratings `(new_a, new_b)` given `score_a in {0.0, 0.5, 1.0}`.

    Non-finite ratings, scores or `k_factor` leave both ratings unchanged
    (never propagates NaN into a rating table).  `score_a` is clamped to
    [0, 1] and a negative k-factor is treated as 0.
    """
    try:
        ra = float(rating_a)
        rb = float(rating_b)
        sa = float(score_a)
        k = float(k_factor)
    except (TypeError, ValueError, OverflowError):
        return rating_a, rating_b
    if not (math.isfinite(ra) and math.isfinite(rb) and math.isfinite(sa) and math.isfinite(k)):
        return rating_a, rating_b
    sa = max(0.0, min(1.0, sa))
    k = max(0.0, k)
    exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
    exp_b = 1.0 - exp_a
    new_a = ra + k * (sa - exp_a)
    new_b = rb + k * ((1.0 - sa) - exp_b)
    return new_a, new_b


class ComplianceBenchmarkSuite:
    def evaluate(self, response_text: str, constraints: Dict[str, Any]) -> BenchmarkScorecard:
        results: Dict[str, bool] = {}
        if not isinstance(constraints, dict):
            return BenchmarkScorecard()
        for idx, (cname, param) in enumerate(constraints.items()):
            if idx >= _MAX_CONSTRAINTS:
                break
            try:
                results[str(cname)] = verify_ifeval_constraint(cname, response_text, param)
            except Exception:  # defensive: a malformed constraint must not abort the suite
                results[str(cname)] = False
        tot = len(results)
        passed = sum(1 for ok in results.values() if ok)
        acc = (passed / float(tot) * 100.0) if tot > 0 else 100.0
        return BenchmarkScorecard(
            total_constraints=tot,
            passed_constraints=passed,
            strict_accuracy=acc,
            results=results,
        )


def _self_test() -> int:
    suite = ComplianceBenchmarkSuite()
    sample = (
        "<<Architecture Overview>>\n"
        "- first module verified without commas\n"
        "- second module verified cleanly\n"
        "P.S. all checks green"
    )
    card = suite.evaluate(
        sample,
        {
            "no_commas": True,
            "title_wrapped": True,
            "exact_bullets": 2,
            "postscript": True,
            "include_keywords": ["architecture", "module"],
            "exclude_keywords": ["synergy"],
            "max_words": 40,
        },
    )
    assert card.strict_accuracy == 100.0, card.to_dict()

    # A bare string keyword is one keyword, not an iterable of characters.
    assert verify_ifeval_constraint("include_keywords", "The ARCHITECTURE is set.", "architecture")
    assert not verify_ifeval_constraint("include_keywords", "nothing here", "architecture")

    # Fullwidth commas still violate "no commas" (evasion resistance).
    assert not verify_ifeval_constraint("no_commas", "alpha\uff0cbeta")
    assert verify_ifeval_constraint("no_commas", "alpha beta")

    # Casefold + NFKC keyword matching (German sharp s, fullwidth latin).
    assert verify_ifeval_constraint("include_keywords", "STRASSE", "Stra\u00dfe")
    assert verify_ifeval_constraint("include_keywords", "\uff21\uff22\uff23", "abc")
    assert verify_ifeval_constraint("exclude_keywords", "clean text", ["README", "TODO"])

    # Malformed params fail soft and never raise.
    for bad in (None, "abc", [], {}, float("nan")):
        assert verify_ifeval_constraint("min_words", "one two", bad) is not None
        assert verify_ifeval_constraint("exact_bullets", "- a", bad) is not None
        assert verify_ifeval_constraint("include_keywords", "abc", bad) is not None
    assert verify_ifeval_constraint("min_words", "one two", None) is True
    assert verify_ifeval_constraint("exact_bullets", "- a\n- b", None) is False
    assert verify_ifeval_constraint("unknown_constraint", "abc", 1) is False
    assert verify_ifeval_constraint(None, "abc", 1) is False  # type: ignore[arg-type]

    # Non-str responses fail closed; empty JSON is not valid JSON.
    for bad_resp in (None, 42, b"bytes", [], {}):
        assert verify_ifeval_constraint("json_format", bad_resp) is False
    assert verify_ifeval_constraint("json_format", "") is False

    # Non-dict constraint sets yield a neutral, empty scorecard.
    assert suite.evaluate("text", None).total_constraints == 0  # type: ignore[arg-type]
    # Deterministic evaluation.
    assert suite.evaluate(sample, {"title_wrapped": True}) == suite.evaluate(sample, {"title_wrapped": True})

    ra, rb = update_elo(1500.0, 1500.0, 1.0)
    assert ra == 1516.0 and rb == 1484.0
    # Out-of-range scores clamp, invalid inputs leave ratings untouched.
    assert update_elo(1500.0, 1500.0, 9.0) == update_elo(1500.0, 1500.0, 1.0)
    for bad in (float("nan"), None, "not-a-number"):
        assert update_elo(1500.0, 1600.0, bad) == (1500.0, 1600.0)
    # Zero-sum check: equal ratings, k=32.
    na, nb = update_elo(1500.0, 1500.0, 0.5)
    assert abs((na - 1500.0) + (nb - 1500.0)) < 1e-9

    print("benchsuite._self_test: OK PASS (12 IFEval verifiable constraints + Elo rating updates)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
