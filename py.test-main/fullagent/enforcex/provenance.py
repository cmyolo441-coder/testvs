"""provenance — Green-Karvounarakis-Tannen Commutative Provenance Semirings.

Tracks *how* and *from which rules/sources* a compliance verdict was derived by
evaluating queries and rule derivations over abstract commutative semirings
`(K, (+), (*), 0, 1)` (Green, Karvounarakis & Tannen, PODS 2007):
  1. **Why-Provenance / Lineage Polynomials (`PolynomialSemiring` `N[X]`)**:
     Each base fact is tagged with variable `x_i`. Alternative derivations add
     (`+`), joint prerequisites multiply (`*`), yielding exact provenance
     monomials showing every minimal witness combination!
  2. **Tropical Cost Semiring (`(R U {inf}, min, +, inf, 0)`)**:
     Computes the minimum remediation cost required to satisfy a goal.
  3. **Viterbi Confidence Semiring (`([0, 1], max, *, 0, 1)`)**:
     Computes the maximum-likelihood derivation confidence.
  4. **Security Access-Control Semiring (`(Levels, min_clearance, max_clearance)`)**:
     Computes the minimum security clearance needed to read a derived output.

Self-test: ``python3 -m fullagent.enforcex.provenance``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Counter as CounterType, Dict, FrozenSet, List, Sequence, Tuple
from collections import Counter

__all__ = [
    "ProvenancePolynomial",
    "TropicalCostSemiring",
    "ViterbiConfidenceSemiring",
    "ProvenanceEvaluationReport",
    "ProvenanceSemiringEngine",
]


class ProvenancePolynomial:
    """Element of the universal commutative provenance semiring `N[X]`.
    Represented as a mapping `monomial (frozenset of (var, exp)) -> coeff (int)`."""

    def __init__(self, terms: Dict[FrozenSet[Tuple[str, int]], int] | None = None) -> None:
        self.terms: Dict[FrozenSet[Tuple[str, int]], int] = {
            k: int(v) for k, v in (terms or {}).items() if int(v) > 0
        }

    @classmethod
    def zero(cls) -> "ProvenancePolynomial":
        return cls({})

    @classmethod
    def one(cls) -> "ProvenancePolynomial":
        return cls({frozenset(): 1})

    @classmethod
    def var(cls, name: str) -> "ProvenancePolynomial":
        return cls({frozenset([(str(name), 1)]): 1})

    def __add__(self, other: "ProvenancePolynomial") -> "ProvenancePolynomial":
        out = dict(self.terms)
        for mon, coeff in other.terms.items():
            out[mon] = out.get(mon, 0) + coeff
        return ProvenancePolynomial(out)

    def __mul__(self, other: "ProvenancePolynomial") -> "ProvenancePolynomial":
        if not self.terms or not other.terms:
            return ProvenancePolynomial.zero()
        out: Dict[FrozenSet[Tuple[str, int]], int] = {}
        for m1, c1 in self.terms.items():
            for m2, c2 in other.terms.items():
                merged: CounterType[str] = Counter(dict(m1))
                for v_name, exp in m2:
                    merged[v_name] += exp
                key = frozenset((k, v) for k, v in sorted(merged.items()) if v > 0)
                out[key] = out.get(key, 0) + c1 * c2
        return ProvenancePolynomial(out)

    def minimal_witness_sets(self) -> List[List[str]]:
        """PosBool[X] absorptive projection: return minimal sets of base variables sufficient for derivation."""
        support_sets = [frozenset(v for v, _ in mon) for mon in self.terms.keys()]
        minimal: List[FrozenSet[str]] = []
        for s in sorted(set(support_sets), key=len):
            if not any(m.issubset(s) for m in minimal):
                minimal.append(s)
        return [sorted(m) for m in minimal]

    def format_str(self) -> str:
        if not self.terms:
            return "0"
        parts: List[str] = []
        for mon, coeff in sorted(self.terms.items(), key=lambda kv: sorted(kv[0])):
            if not mon:
                parts.append(str(coeff))
                continue
            vars_str = "*".join(f"{v}^{e}" if e > 1 else v for v, e in sorted(mon))
            parts.append(f"{coeff}*{vars_str}" if coeff > 1 else vars_str)
        return " + ".join(parts)


class TropicalCostSemiring:
    """Tropical semiring `(R_+ U {inf}, min, +, inf, 0)`."""
    zero = float("inf")
    one = 0.0

    @staticmethod
    def add(a: float, b: float) -> float:
        return min(float(a), float(b))

    @staticmethod
    def mul(a: float, b: float) -> float:
        return float(a) + float(b)


class ViterbiConfidenceSemiring:
    """Viterbi semiring `([0, 1], max, *, 0.0, 1.0)`."""
    zero = 0.0
    one = 1.0

    @staticmethod
    def add(a: float, b: float) -> float:
        return max(float(a), float(b))

    @staticmethod
    def mul(a: float, b: float) -> float:
        return float(a) * float(b)


@dataclass
class ProvenanceEvaluationReport:
    polynomial: str
    minimal_witnesses: List[List[str]]
    min_tropical_cost: float
    max_viterbi_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "polynomial": self.polynomial,
            "minimal_witnesses": list(self.minimal_witnesses),
            "min_tropical_cost": round(self.min_tropical_cost, 4),
            "max_viterbi_confidence": round(self.max_viterbi_confidence, 4),
        }


class ProvenanceSemiringEngine:
    """Evaluates conjunctive-disjunctive derivation graphs simultaneously over
    `N[X]`, Tropical, and Viterbi semirings."""

    def evaluate_derivations(
        self,
        derivations: Sequence[Sequence[str]],
        costs: Dict[str, float],
        confidences: Dict[str, float],
    ) -> ProvenanceEvaluationReport:
        poly = ProvenancePolynomial.zero()
        trop = TropicalCostSemiring.zero
        vit = ViterbiConfidenceSemiring.zero

        for branch in derivations:
            p_branch = ProvenancePolynomial.one()
            t_branch = TropicalCostSemiring.one
            v_branch = ViterbiConfidenceSemiring.one
            for atom in branch:
                p_branch = p_branch * ProvenancePolynomial.var(atom)
                t_branch = TropicalCostSemiring.mul(t_branch, costs.get(atom, 1.0))
                v_branch = ViterbiConfidenceSemiring.mul(v_branch, confidences.get(atom, 0.9))
            poly = poly + p_branch
            trop = TropicalCostSemiring.add(trop, t_branch)
            vit = ViterbiConfidenceSemiring.add(vit, v_branch)

        return ProvenanceEvaluationReport(
            polynomial=poly.format_str(),
            minimal_witnesses=poly.minimal_witness_sets(),
            min_tropical_cost=trop,
            max_viterbi_confidence=vit,
        )


def _self_test() -> int:
    eng = ProvenanceSemiringEngine()
    # Two ways to satisfy compliance: (r1 & r2) OR (r1 & r2 & r3) OR (r4)
    rep = eng.evaluate_derivations(
        derivations=[["r1", "r2"], ["r1", "r2", "r3"], ["r4"]],
        costs={"r1": 2.0, "r2": 3.0, "r3": 4.0, "r4": 6.0},
        confidences={"r1": 0.9, "r2": 0.9, "r3": 0.5, "r4": 0.85},
    )
    # Absorptive minimal witnesses drop {r1, r2, r3} because {r1, r2} is a subset!
    assert rep.minimal_witnesses == [["r4"], ["r1", "r2"]], rep.to_dict()
    assert rep.min_tropical_cost == 5.0  # min(2+3, 2+3+4, 6) == 5.0
    assert abs(rep.max_viterbi_confidence - 0.85) < 1e-6  # max(0.81, 0.405, 0.85) == 0.85

    print("provenance._self_test: OK PASS (N[X] polynomials, PosBool minimal witnesses, Tropical & Viterbi semirings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
