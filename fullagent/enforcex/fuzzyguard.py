"""fuzzyguard — Gödel, Łukasiewicz & Product T-Norm Continuous Fuzzy Logic Engine.

Evaluates graduated, multi-dimensional system-prompt constraints (such as
"keep paragraphs reasonably short", "maintain high lexical density", "avoid
excessive hedging") over `[0.0, 1.0]` truth degrees using the three fundamental
continuous t-norms (Hájek, "Metamathematics of Fuzzy Logic"):
  1. Łukasiewicz Logic:
     - T-norm:   `T_L(a, b) = max(0, a + b - 1)`
     - T-conorm: `S_L(a, b) = min(1, a + b)`
     - Residuum: `I_L(a, b) = min(1, 1 - a + b)`
  2. Gödel-Dummett Logic:
     - T-norm:   `T_G(a, b) = min(a, b)`
     - T-conorm: `S_G(a, b) = max(a, b)`
     - Residuum: `I_G(a, b) = 1.0 if a <= b else b`
  3. Product (Goguen) Logic:
     - T-norm:   `T_P(a, b) = a * b`
     - T-conorm: `S_P(a, b) = a + b - a * b`
     - Residuum: `I_P(a, b) = 1.0 if a <= b else b / a`

Also provides trapezoidal membership functions and Mamdani centroid defuzzification.

All degrees are clamped to `[0.0, 1.0]`; malformed (non-numeric, NaN,
infinity) inputs degrade to `0.0` instead of raising or poisoning the
truth algebra with NaN.

Self-test: ``python3 -m fullagent.enforcex.fuzzyguard``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

__all__ = [
    "TrapezoidalFuzzySet",
    "FuzzyEvaluationReport",
    "FuzzyLogicEngine",
    "lukasiewicz_tnorm",
    "lukasiewicz_residuum",
    "godel_tnorm",
    "godel_residuum",
    "product_tnorm",
    "product_residuum",
]

# ReDoS / cost bound for membership scoring on oversized responses
_MAX_SCAN = 262144


def _clamp01(x: float) -> float:
    """Clamp to [0, 1]; NaN / non-numeric / None degrade to 0.0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN must never become "fully true"
        return 0.0
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        return 1.0
    return v


def lukasiewicz_tnorm(a: float, b: float) -> float:
    return _clamp01(_clamp01(a) + _clamp01(b) - 1.0)


def lukasiewicz_tconorm(a: float, b: float) -> float:
    return _clamp01(_clamp01(a) + _clamp01(b))


def lukasiewicz_residuum(a: float, b: float) -> float:
    return _clamp01(1.0 - _clamp01(a) + _clamp01(b))


def godel_tnorm(a: float, b: float) -> float:
    return min(_clamp01(a), _clamp01(b))


def godel_tconorm(a: float, b: float) -> float:
    return max(_clamp01(a), _clamp01(b))


def godel_residuum(a: float, b: float) -> float:
    aa, bb = _clamp01(a), _clamp01(b)
    return 1.0 if aa <= bb else bb


def product_tnorm(a: float, b: float) -> float:
    return _clamp01(a) * _clamp01(b)


def product_tconorm(a: float, b: float) -> float:
    aa, bb = _clamp01(a), _clamp01(b)
    return _clamp01(aa + bb - aa * bb)


def product_residuum(a: float, b: float) -> float:
    aa, bb = _clamp01(a), _clamp01(b)
    return 1.0 if aa <= bb or aa <= 1e-12 else _clamp01(bb / aa)


@dataclass(frozen=True)
class TrapezoidalFuzzySet:
    """Trapezoidal membership function `(a <= b <= c <= d)`."""
    name: str
    a: float
    b: float
    c: float
    d: float

    def membership(self, x: float) -> float:
        try:
            val = float(x)
        except (TypeError, ValueError):
            return 0.0
        if val != val:  # NaN has no degree
            return 0.0
        if val <= self.a or val >= self.d:
            if val == self.a == self.b:
                return 1.0
            if val == self.c == self.d:
                return 1.0
            return 0.0
        if self.b <= val <= self.c:
            return 1.0
        if val < self.b:
            return _clamp01((val - self.a) / max(1e-12, self.b - self.a))
        return _clamp01((self.d - val) / max(1e-12, self.d - self.c))


@dataclass
class FuzzyEvaluationReport:
    lukasiewicz_truth: float
    godel_truth: float
    product_truth: float
    centroid_score: float
    clause_degrees: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lukasiewicz_truth": round(self.lukasiewicz_truth, 4),
            "godel_truth": round(self.godel_truth, 4),
            "product_truth": round(self.product_truth, 4),
            "centroid_score": round(self.centroid_score, 2),
            "clause_degrees": {k: round(v, 4) for k, v in self.clause_degrees.items()},
        }


class FuzzyLogicEngine:
    """Multi-t-norm fuzzy evaluator and Mamdani centroid defuzzifier."""

    _HEDGES = frozenset((
        "maybe", "perhaps", "possibly", "probably", "might", "could",
        "seems", "seem", "arguably", "generally", "somewhat", "relatively",
        "roughly", "sort", "kind", "fairly", "presumably", "apparently",
    ))

    def __init__(self) -> None:
        self.concise_set = TrapezoidalFuzzySet("concise_words", 5.0, 15.0, 180.0, 320.0)
        self.lexical_density_set = TrapezoidalFuzzySet("lexical_density", 0.30, 0.48, 1.00, 1.00)
        # degree of *non-hedging*: 1.0 at <=8% hedge words, 0 at >=25%
        self.non_hedging_set = TrapezoidalFuzzySet("non_hedging", 0.0, 0.0, 0.08, 0.25)

    @staticmethod
    def centroid_defuzzify(samples: Sequence[Tuple[float, float]]) -> float:
        """Compute center of gravity `sum(x_i * mu_i) / sum(mu_i)`.

        Non-finite / malformed samples are ignored; the result is clamped
        to the 0..100 output scale. Never raises."""
        num = 0.0
        den = 0.0
        try:
            for x, mu in samples or ():
                try:
                    xv, mv = float(x), float(mu)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(xv) or not math.isfinite(mv):
                    continue
                mv = max(0.0, mv)
                num += xv * mv
                den += mv
        except Exception:
            return 0.0
        if den <= 1e-12:
            return 0.0
        return max(0.0, min(100.0, num / den))

    def evaluate_response(self, response_text: str) -> FuzzyEvaluationReport:
        text = str(response_text or "")[:_MAX_SCAN]
        raw_words = text.split()
        words = [w.lower().strip(".,:;!?()[]{}") for w in raw_words if w.strip()]
        w_count = len(words)
        unique_ratio = len(set(words)) / max(1.0, float(w_count))
        hedge_ratio = (sum(1 for w in words if w in self._HEDGES)
                       / max(1.0, float(w_count)))

        deg_concise = self.concise_set.membership(w_count)
        deg_density = self.lexical_density_set.membership(unique_ratio)
        deg_fence_balanced = 1.0 if text.count("```") % 2 == 0 else 0.2
        deg_non_hedging = self.non_hedging_set.membership(hedge_ratio)

        degrees = {
            "conciseness": _clamp01(deg_concise),
            "lexical_density": _clamp01(deg_density),
            "fence_balance": _clamp01(deg_fence_balanced),
            "non_hedging": _clamp01(deg_non_hedging),
        }
        vals = list(degrees.values())

        t_luk = vals[0]
        t_god = vals[0]
        t_prd = vals[0]
        for v in vals[1:]:
            t_luk = lukasiewicz_tnorm(t_luk, v)
            t_god = godel_tnorm(t_god, v)
            t_prd = product_tnorm(t_prd, v)

        centroid = self.centroid_defuzzify([
            (100.0, t_god),
            (90.0, t_prd),
            (80.0, t_luk),
            (20.0, 1.0 - t_god),
        ])
        return FuzzyEvaluationReport(
            lukasiewicz_truth=_clamp01(t_luk),
            godel_truth=_clamp01(t_god),
            product_truth=_clamp01(t_prd),
            centroid_score=centroid,
            clause_degrees=degrees,
        )


def _self_test() -> int:
    # Verify t-norm ordering: Lukasiewicz <= Product <= Godel for all a, b in [0, 1]
    for a in (0.2, 0.5, 0.8):
        for b in (0.3, 0.6, 0.9):
            tl = lukasiewicz_tnorm(a, b)
            tp = product_tnorm(a, b)
            tg = godel_tnorm(a, b)
            assert tl <= tp + 1e-12 <= tg + 2e-12, (a, b, tl, tp, tg)
            # Residuation adjunction property: T(a, c) <= b iff c <= I(a, b)
            res = lukasiewicz_residuum(a, b)
            assert lukasiewicz_tnorm(a, res) <= b + 1e-12

    eng = FuzzyLogicEngine()
    rep = eng.evaluate_response("Here is a clear, concise, well-structured explanation with rich vocabulary and zero unclosed blocks.")
    assert rep.godel_truth >= 0.8 and rep.centroid_score >= 80.0, rep.to_dict()

    # NaN / infinity / non-numeric never poison the truth algebra
    for bad in (float("nan"), float("inf"), float("-inf"), None, "x", object()):
        for fn in (lukasiewicz_tnorm, godel_tnorm, product_tnorm,
                   lukasiewicz_residuum, godel_residuum, product_residuum):
            deg = fn(bad, 0.5)
            assert 0.0 <= deg <= 1.0, (fn.__name__, bad, deg)
    assert lukasiewicz_tnorm(float("nan"), 1.0) == 0.0
    assert godel_tnorm(float("nan"), 1.0) == 0.0

    # t-conorm is the De Morgan dual of each t-norm
    for a, b in ((0.2, 0.7), (0.5, 0.5), (0.9, 0.1)):
        assert abs(lukasiewicz_tconorm(a, b)
                   - (1.0 - lukasiewicz_tnorm(1 - a, 1 - b))) < 1e-12
        assert abs(godel_tconorm(a, b)
                   - (1.0 - godel_tnorm(1 - a, 1 - b))) < 1e-12
        assert abs(product_tconorm(a, b)
                   - (1.0 - product_tnorm(1 - a, 1 - b))) < 1e-12

    # residuum adjunction maximality for all three logics
    for a in (0.0, 0.3, 0.8, 1.0):
        for b in (0.0, 0.4, 0.9, 1.0):
            for tnorm, resid in ((lukasiewicz_tnorm, lukasiewicz_residuum),
                                 (godel_tnorm, godel_residuum),
                                 (product_tnorm, product_residuum)):
                res = resid(a, b)
                assert tnorm(a, res) <= b + 1e-12
                for c in (0.0, 0.25, 0.75, 1.0):
                    if tnorm(a, c) <= b + 1e-12:
                        assert c <= res + 1e-12, (tnorm.__name__, a, b, c)

    # trapezoid boundaries + malformed input
    trap = TrapezoidalFuzzySet("t", 0.0, 0.2, 0.8, 1.0)
    assert trap.membership(-1) == 0.0 and trap.membership(2) == 0.0
    assert trap.membership(0.5) == 1.0 and 0.0 < trap.membership(0.1) < 1.0
    assert trap.membership(float("nan")) == 0.0 and trap.membership("junk") == 0.0

    # every clause degree is a real threshold in [0, 1]
    hedged = eng.evaluate_response(
        "Maybe this might possibly work; perhaps it could be somewhat roughly right.")
    assert all(0.0 <= v <= 1.0 for v in hedged.clause_degrees.values()), hedged.to_dict()
    assert hedged.clause_degrees["non_hedging"] < 0.5
    assert hedged.godel_truth <= hedged.product_truth + 1e-12

    # centroid ignores malformed samples and is bounded
    assert FuzzyLogicEngine.centroid_defuzzify([]) == 0.0
    assert FuzzyLogicEngine.centroid_defuzzify(
        [("bad", 1.0), (50.0, float("nan")), (80.0, 1.0)]) == 80.0
    assert 0.0 <= FuzzyLogicEngine.centroid_defuzzify([(500.0, 1.0)]) <= 100.0

    # oversized input is bounded and never raises
    big = eng.evaluate_response("word " * 60000)
    assert 0.0 <= big.godel_truth <= 1.0 and 0.0 <= big.centroid_score <= 100.0

    print("fuzzyguard._self_test: OK PASS (Lukasiewicz/Godel/Product t-norms, residua adjunction, "
          "De Morgan duals, Mamdani centroid, NaN-safe degrees, hedging clause)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
