"""intuitionistic — Heyting Algebra & Constructive Intuitionistic Logic Verifier.

In classical Boolean logic, `p v ~p` (Law of Excluded Middle) and `~~p -> p`
(Double Negation Elimination) hold without an explicit proof witness. In
sovereign compliance auditing, claiming a requirement holds merely because its
negation was not proved is unsound.

Implements a finite **Heyting Algebra** `(H, <=, /\\, \\/, ->, 0, 1)` where:
  * Every proposition is backed by an explicit set of constructive evidence
    spans (`WitnessTerm`, Brouwer-Heyting-Kolmogorov BHK interpretation).
  * Heyting implication `a -> b` is the greatest element `c` such that
    `c /\\ a <= b` (relative pseudo-complement).
  * Intuitionistic negation `~a` is `a -> 0`, so `a \\/ ~a` is strictly `< 1`
    whenever `a` has only partial/unverified constructive evidence!

Self-test: ``python3 -m fullagent.enforcex.intuitionistic``
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Sequence, Set, Tuple

__all__ = [
    "ConstructiveWitness",
    "HeytingTruthValue",
    "IntuitionisticVerdict",
    "HeytingAlgebraEngine",
]


@dataclass(frozen=True)
class ConstructiveWitness:
    rule_id: str
    evidence_span: str
    start_char: int
    end_char: int
    verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "evidence_span": self.evidence_span[:100],
            "start_char": self.start_char,
            "end_char": self.end_char,
            "verified": self.verified,
        }


@dataclass(frozen=True)
class HeytingTruthValue:
    """Element of the open-set topology Heyting algebra `O(X)` over verification worlds `X`."""
    universe: FrozenSet[str]
    open_set: FrozenSet[str]
    witnesses: Tuple[ConstructiveWitness, ...] = ()

    @property
    def is_top(self) -> bool:
        return self.open_set == self.universe

    @property
    def is_bottom(self) -> bool:
        return len(self.open_set) == 0

    @property
    def measure(self) -> float:
        if not self.universe:
            return 1.0
        return len(self.open_set) / float(len(self.universe))


class HeytingAlgebraEngine:
    """BHK Constructive Witness & Topological Heyting Algebra Evaluator."""

    def __init__(self, worlds: Sequence[str] = ("syntax", "semantics", "grounding", "safety")) -> None:
        self.universe: FrozenSet[str] = frozenset(str(w) for w in worlds)

    def top(self) -> HeytingTruthValue:
        return HeytingTruthValue(self.universe, self.universe)

    def bottom(self) -> HeytingTruthValue:
        return HeytingTruthValue(self.universe, frozenset())

    def meet(self, a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Conjunction `a /\\ b`: intersection of open sets and union of witnesses."""
        return HeytingTruthValue(
            self.universe,
            a.open_set & b.open_set,
            a.witnesses + b.witnesses,
        )

    def join(self, a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Disjunction `a \\/ b`: union of open sets."""
        return HeytingTruthValue(
            self.universe,
            a.open_set | b.open_set,
            a.witnesses + b.witnesses,
        )

    def implies(self, a: HeytingTruthValue, b: HeytingTruthValue) -> HeytingTruthValue:
        """Heyting relative pseudo-complement `a -> b = Int((X \\ a) U b)`.
        In our ordered Kripke chain `w_0 <= w_1 <= ... <= w_k`, an open set is
        upward-closed; for general open sets with interior operator, worlds
        where `a` holds must also satisfy `b`, and worlds where `a` fails only
        enter the interior if their upward cone satisfies `a -> b`."""
        worlds_list = sorted(self.universe)
        valid: Set[str] = set()
        for idx, w in enumerate(worlds_list):
            # Upward cone from w
            cone = set(worlds_list[idx:])
            if (cone & a.open_set).issubset(b.open_set):
                valid.add(w)
        return HeytingTruthValue(self.universe, frozenset(valid), b.witnesses)

    def negate(self, a: HeytingTruthValue) -> HeytingTruthValue:
        """Intuitionistic negation `~a = (a -> 0)`."""
        return self.implies(a, self.bottom())

    def double_negate(self, a: HeytingTruthValue) -> HeytingTruthValue:
        """Compute `~~a`. In Heyting algebra, `a <= ~~a`, but `~~a == a` need not hold!"""
        return self.negate(self.negate(a))

    def construct_witness_for_pattern(self, rule_id: str, pattern: str, text: str) -> HeytingTruthValue:
        """BHK constructive proof search: find explicit substring witness matching `pattern`."""
        m = re.search(pattern, text or "", re.I | re.M)
        if m:
            w = ConstructiveWitness(
                rule_id=rule_id,
                evidence_span=m.group(0),
                start_char=m.start(),
                end_char=m.end(),
                verified=True,
            )
            return HeytingTruthValue(self.universe, self.universe, (w,))
        return self.bottom()


@dataclass
class IntuitionisticVerdict:
    constructively_proved: bool
    truth_measure: float
    excluded_middle_gap: float
    witnesses: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constructively_proved": self.constructively_proved,
            "truth_measure": round(self.truth_measure, 4),
            "excluded_middle_gap": round(self.excluded_middle_gap, 4),
            "witnesses": list(self.witnesses),
        }


def _self_test() -> int:
    ha = HeytingAlgebraEngine(("w1", "w2", "w3", "w4"))
    # Partial truth value (upward cone {w3, w4})
    partial = HeytingTruthValue(ha.universe, frozenset({"w3", "w4"}))
    neg_p = ha.negate(partial)
    lem = ha.join(partial, neg_p)
    # In Intuitionistic Heyting algebra, partial \/ ~partial is NOT Top!
    assert not lem.is_top, lem

    # Constructive BHK witness search
    v1 = ha.construct_witness_for_pattern("R-JSON", r"\{.*\}", '{"ok": true}')
    v2 = ha.construct_witness_for_pattern("R-KEY", r'"ok"\s*:', '{"ok": true}')
    conj = ha.meet(v1, v2)
    assert conj.is_top and len(conj.witnesses) == 2

    print("intuitionistic._self_test: OK PASS (Heyting pseudo-complement, non-trivial LEM gap, BHK witnesses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
