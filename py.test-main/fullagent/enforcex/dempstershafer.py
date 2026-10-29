"""dempstershafer — Dempster-Shafer Theory of Evidence, Belief/Plausibility & Pignistic Transform.

Implements Glenn Shafer's **Mathematical Theory of Evidence** (1976) and
Smets' **Transferable Belief Model** over a frame of discernment
`Theta = {safe, warning, violation}`:
  * **Basic Probability Assignment (BPA / Mass Function `m: 2^Theta -> [0, 1]`)**:
    `m(empty) = 0` and `sum_{A subseteq Theta} m(A) = 1`. Mass assigned to
    `Theta` itself represents explicit epistemic ignorance!
  * **Belief Function (`Bel(A) = sum_{B subseteq A} m(B)`)**: lower bound on probability.
  * **Plausibility Function (`Pl(A) = sum_{B & A != empty} m(B) = 1 - Bel(Theta \\ A)`)**:
    upper bound on probability.
  * **Dempster's Rule of Orthogonal Combination (`m_1 (+) m_2`)**:
    `m_{12}(C) = (1 / (1 - K)) * sum_{A & B = C} m_1(A) * m_2(B)`, where
    `K = sum_{A & B = empty} m_1(A) * m_2(B)` is the inter-sensor conflict mass.
  * **Smets' Pignistic Probability (`BetP(omega) = sum_{omega in A} m(A) / |A|`)**.

Self-test: ``python3 -m fullagent.enforcex.dempstershafer``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional, Sequence, Tuple

__all__ = [
    "MassAssignment",
    "DempsterShaferReport",
    "DempsterShaferEngine",
]

_EPS = 1e-12


def _finite(value: Any) -> Optional[float]:
    """Best-effort finite float; ``None`` for junk / NaN / infinities."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _clean_frame(frame: Any) -> FrozenSet[str]:
    if frame is None:
        return frozenset()
    if isinstance(frame, str):
        return frozenset({frame})
    try:
        return frozenset(str(x) for x in frame)
    except TypeError:
        return frozenset()


class MassAssignment:
    """Basic Probability Assignment `m: 2^Theta -> [0, 1]`."""

    def __init__(self, frame: Iterable[str], masses: Optional[Dict[FrozenSet[str], float]] = None) -> None:
        self.frame: FrozenSet[str] = _clean_frame(frame)
        raw: Dict[FrozenSet[str], float] = {}
        items = masses.items() if isinstance(masses, dict) else ()
        for focal, val in items:
            try:
                fset = frozenset(str(x) for x in focal) & self.frame
            except TypeError:
                continue
            fval = _finite(val)
            if fset and fval is not None and fval > 0.0:
                raw[fset] = raw.get(fset, 0.0) + fval
        tot = sum(raw.values())
        if not self.frame or tot <= _EPS or not math.isfinite(tot):
            # Vacuous BPA: all mass on ignorance (Theta), or empty frame.
            self.masses: Dict[FrozenSet[str], float] = (
                {self.frame: 1.0} if self.frame else {}
            )
        else:
            self.masses = {k: v / tot for k, v in raw.items()}

    def belief(self, hypothesis: Iterable[str]) -> float:
        try:
            h = frozenset(str(x) for x in hypothesis) & self.frame
        except TypeError:
            return 0.0
        return max(0.0, min(1.0, sum(
            v for focal, v in self.masses.items() if focal.issubset(h))))

    def plausibility(self, hypothesis: Iterable[str]) -> float:
        try:
            h = frozenset(str(x) for x in hypothesis) & self.frame
        except TypeError:
            return 0.0
        return max(0.0, min(1.0, sum(
            v for focal, v in self.masses.items() if len(focal & h) > 0)))

    def pignistic_probabilities(self) -> Dict[str, float]:
        """Smets' pignistic transformation `BetP(x) = sum_{x in A} m(A) / |A|`."""
        bet: Dict[str, float] = {x: 0.0 for x in self.frame}
        for focal, mass in self.masses.items():
            if not focal:
                continue
            share = mass / float(len(focal))
            for elem in focal:
                bet[elem] += share
        return bet

    def discount(self, alpha: float) -> "MassAssignment":
        """Shafer's discounting: shrink evidence by ``alpha`` in [0, 1].

        ``m'(A) = alpha * m(A)`` for proper subsets and the remaining mass is
        transferred to the frame (ignorance).
        """
        a = _finite(alpha)
        a = 0.0 if a is None else max(0.0, min(1.0, a))
        out: Dict[FrozenSet[str], float] = {}
        for focal, mass in self.masses.items():
            if focal == self.frame:
                out[focal] = out.get(focal, 0.0) + mass
            else:
                out[focal] = out.get(focal, 0.0) + a * mass
        if self.frame:
            out[self.frame] = out.get(self.frame, 0.0) + (1.0 - a)
        return MassAssignment(self.frame, out)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly mass table (focal sets sorted, masses rounded)."""
        return {
            "frame": sorted(self.frame),
            "masses": [
                {"focal": sorted(focal), "mass": round(v, 6)}
                for focal, v in sorted(self.masses.items(),
                                       key=lambda kv: (len(kv[0]), sorted(kv[0])))
            ],
        }



@dataclass
class DempsterShaferReport:
    conflict_k: float
    belief_intervals: Dict[str, Tuple[float, float]]
    pignistic: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_k": round(self.conflict_k, 4),
            "belief_intervals": {k: (round(v[0], 4), round(v[1], 4)) for k, v in self.belief_intervals.items()},
            "pignistic": {k: round(v, 4) for k, v in self.pignistic.items()},
        }


class DempsterShaferEngine:
    """Combines multi-sensor evidence via Dempster's Rule of Combination."""

    def __init__(self, frame: Sequence[str] = ("safe", "warning", "violation")) -> None:
        self.frame: FrozenSet[str] = _clean_frame(frame)

    def combine(self, m1: MassAssignment, m2: MassAssignment) -> Tuple[MassAssignment, float]:
        """Compute `m1 (+) m2` and conflict mass `K in [0, 1]`.

        Total conflict (`K == 1`) has no Dempster combination; Yager's rule is
        used as a fallback (all mass to Theta) and `K = 1` is reported.
        """
        if not isinstance(m1, MassAssignment) or not isinstance(m2, MassAssignment):
            return MassAssignment(self.frame), 1.0
        combined: Dict[FrozenSet[str], float] = {}
        conflict_k = 0.0
        for a, v1 in m1.masses.items():
            for b, v2 in m2.masses.items():
                inter = a & b
                prod = v1 * v2
                if not inter:
                    conflict_k += prod
                else:
                    combined[inter] = combined.get(inter, 0.0) + prod

        total = sum(combined.values())
        if conflict_k >= 1.0 - _EPS or total <= _EPS:
            # Total conflict fallback: Yager's rule assigns conflict mass to
            # universal ignorance Theta (nothing can be concluded).
            return MassAssignment(self.frame), 1.0

        norm = 1.0 / total
        normalized = {k: v * norm for k, v in combined.items() if v > 0.0}
        return MassAssignment(self.frame, normalized), min(1.0, max(0.0, conflict_k))

    def combine_all(self, assignments: Sequence[MassAssignment]) -> Tuple[MassAssignment, float]:
        """Sequentially combine a sequence; returns final BPA and max step conflict."""
        base: Optional[MassAssignment] = None
        worst_k = 0.0
        for nxt in assignments or ():
            if not isinstance(nxt, MassAssignment):
                continue
            if base is None:
                base = nxt
                continue
            base, k = self.combine(base, nxt)
            worst_k = max(worst_k, k)
        if base is None:
            return MassAssignment(self.frame), 0.0
        return base, worst_k

    def evaluate_sensors(self, assignments: Sequence[MassAssignment]) -> DempsterShaferReport:
        base, total_k = self.combine_all(assignments)
        intervals: Dict[str, Tuple[float, float]] = {}
        for x in sorted(self.frame):
            intervals[x] = (base.belief([x]), base.plausibility([x]))
        return DempsterShaferReport(
            conflict_k=total_k,
            belief_intervals=intervals,
            pignistic=base.pignistic_probabilities(),
        )


def _self_test() -> int:
    eng = DempsterShaferEngine(("safe", "warning", "violation"))
    # Sensor 1 assigns 0.70 to {safe} and 0.30 to universal ignorance Theta
    s1 = MassAssignment(eng.frame, {frozenset(["safe"]): 0.70, eng.frame: 0.30})
    # Sensor 2 assigns 0.80 to {safe} and 0.20 to Theta
    s2 = MassAssignment(eng.frame, {frozenset(["safe"]): 0.80, eng.frame: 0.20})

    rep = eng.evaluate_sensors([s1, s2])
    # Combined mass on {safe} = 1 - (0.30 * 0.20) = 0.94, ignorance = 0.06, conflict K = 0.0!
    bel_safe, pl_safe = rep.belief_intervals["safe"]
    assert abs(bel_safe - 0.94) < 1e-6 and abs(pl_safe - 1.0) < 1e-6
    assert abs(sum(rep.pignistic.values()) - 1.0) < 1e-6

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. conflict mass is the exact cross-product of disjoint focal sets
    c1 = MassAssignment(eng.frame, {frozenset(["safe"]): 0.9, frozenset(["violation"]): 0.1})
    c2 = MassAssignment(eng.frame, {frozenset(["warning"]): 0.8, frozenset(["violation"]): 0.2})
    comb, k = eng.combine(c1, c2)
    assert abs(k - 0.98) < 1e-9, k
    assert abs(comb.belief(["violation"]) - 1.0) < 1e-9, comb.to_dict()

    # 2. total conflict falls back to vacuous ignorance without dividing by zero
    total, k_total = eng.combine(
        MassAssignment(eng.frame, {frozenset(["safe"]): 1.0}),
        MassAssignment(eng.frame, {frozenset(["violation"]): 1.0}),
    )
    assert k_total == 1.0 and abs(total.belief(eng.frame) - 1.0) < 1e-9
    assert abs(sum(total.pignistic_probabilities().values()) - 1.0) < 1e-9

    # 3. Bel(A) == 1 - Pl(Theta \ A) duality holds for every singleton
    for x in eng.frame:
        rest = eng.frame - {x}
        assert abs(comb.belief([x]) - (1.0 - comb.plausibility(rest))) < 1e-9

    # 4. junk / NaN / negative / infinite masses never raise and stay normalized
    dirty = MassAssignment(eng.frame, {
        frozenset(["safe"]): 2.0,
        frozenset(["warning"]): float("nan"),
        frozenset(["violation"]): float("inf"),
        frozenset(["safe", "warning"]): -0.5,
        frozenset(["safe", "violation"]): "not-a-number",  # type: ignore[dict-item]
    })
    assert abs(sum(dirty.masses.values()) - 1.0) < 1e-9
    assert dirty.belief(None) == 0.0  # type: ignore[arg-type]

    # 5. discounting moves residual mass to ignorance (Shafer)
    certain = MassAssignment(eng.frame, {frozenset(["safe"]): 1.0})
    discounted = certain.discount(0.6)
    assert abs(discounted.belief(["safe"]) - 0.6) < 1e-9
    assert abs(discounted.plausibility(["safe"]) - 1.0) < 1e-9
    assert abs(sum(discounted.masses.values()) - 1.0) < 1e-9

    # 6. empty / malformed sensor lists give the vacuous report, not a crash
    empty_rep = eng.evaluate_sensors([])
    assert empty_rep.conflict_k == 0.0 and empty_rep.pignistic
    malformed = eng.evaluate_sensors([None, "junk", certain])  # type: ignore[list-item]
    assert malformed.belief_intervals["safe"] == (1.0, 1.0)
    assert DempsterShaferEngine(()).evaluate_sensors([]).belief_intervals == {}


    print("dempstershafer._self_test: OK PASS (BPA mass functions, Bel/Pl intervals, Dempster combination, BetP)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
