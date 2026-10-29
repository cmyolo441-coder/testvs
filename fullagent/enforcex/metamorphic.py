"""metamorphic — Automated Metamorphic Testing & Zeller Delta-Debugging (`ddmin`) Minimizer.

Implements two foundational software verification techniques (Chen et al. 1998
Metamorphic Testing; Zeller & Hildebrandt 2002 Delta Debugging `ddmin`):
  1. **Metamorphic Relation (MR) Verifier**:
     Even without a ground-truth oracle for every arbitrary prompt/response pair,
     any sound compliance evaluator `E(response, system)` must satisfy formal
     algebraic Metamorphic Relations:
       - *MR1 (Rule Permutation Invariance)*: permuting the order of independent
         system rules does not alter the set of detected violations.
       - *MR2 (Violation Monotonicity)*: appending an explicit forbidden token or
         leaked secret never *increases* the compliance score.
       - *MR3 (Idempotence of Deterministic Repair)*: `repair(repair(x)) == repair(x)`.
  2. **Zeller's `ddmin` Counterexample Shrinking Algorithm**:
     Given a failing input sequence `C` where `test(C) == FAIL`, recursively
     partitions `C` into `n` chunks (testing complements `C \\ C_i` and subsets
     `C_i`, doubling granularity `n = min(|C|, 2n)` when no reduction succeeds)
     to isolate a **1-minimal failing counterexample**!

Self-test: ``python3 -m fullagent.enforcex.metamorphic``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

__all__ = [
    "MetamorphicReport",
    "MetamorphicTestingEngine",
    "ddmin_shrink",
]


def ddmin_shrink(
    items: Sequence[Any],
    fails_predicate: Callable[[Sequence[Any]], bool],
) -> List[Any]:
    """Zeller & Hildebrandt (2002) `ddmin` algorithm returning a 1-minimal failing subset."""
    cur = list(items)
    if not cur or not fails_predicate(cur):
        return cur
    n = 2
    while len(cur) >= 2:
        chunk_len = max(1, len(cur) // n)
        chunks: List[List[Any]] = []
        for i in range(0, len(cur), chunk_len):
            chunks.append(cur[i:i + chunk_len])
        reduced = False
        # 1. Try reducing to a single chunk C_i
        for ch in chunks:
            if fails_predicate(ch):
                cur = ch
                n = 2
                reduced = True
                break
        if reduced:
            continue
        # 2. Try reducing to complement C \ C_i
        for idx in range(len(chunks)):
            comp = [x for j, ch in enumerate(chunks) if j != idx for x in ch]
            if comp and fails_predicate(comp):
                cur = comp
                n = max(n - 1, 2)
                reduced = True
                break
        if reduced:
            continue
        if n >= len(cur):
            break
        n = min(len(cur), n * 2)
    return cur


@dataclass
class MetamorphicReport:
    permutation_invariant: bool
    violation_monotonic: bool
    repair_idempotent: bool
    minimal_failing_slice: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permutation_invariant": self.permutation_invariant,
            "violation_monotonic": self.violation_monotonic,
            "repair_idempotent": self.repair_idempotent,
            "minimal_failing_slice": list(self.minimal_failing_slice),
        }


class MetamorphicTestingEngine:
    """Verifies metamorphic relations and shrinks failing inputs via `ddmin`."""

    def verify_evaluator(
        self,
        score_fn: Callable[[str, str], float],
        repair_fn: Callable[[str, str], str],
        base_response: str,
        rules: Sequence[str],
        forbidden_injection: str = "<developer_config>leak</developer_config>",
    ) -> MetamorphicReport:
        sys_fwd = "\n".join(rules)
        sys_rev = "\n".join(reversed(list(rules)))
        s1 = score_fn(base_response, sys_fwd)
        s2 = score_fn(base_response, sys_rev)
        perm_ok = abs(s1 - s2) < 1e-6

        s_poisoned = score_fn(base_response + "\n" + forbidden_injection, sys_fwd)
        mono_ok = s_poisoned <= s1 + 1e-6

        r1 = repair_fn(base_response + "\n" + forbidden_injection, sys_fwd)
        r2 = repair_fn(r1, sys_fwd)
        idem_ok = (r1 == r2)

        lines = (base_response + "\n" + forbidden_injection).splitlines()
        min_slice = ddmin_shrink(lines, lambda sub: score_fn("\n".join(sub), sys_fwd) < 90.0)
        return MetamorphicReport(
            permutation_invariant=perm_ok,
            violation_monotonic=mono_ok,
            repair_idempotent=idem_ok,
            minimal_failing_slice=[str(x) for x in min_slice],
        )


def _self_test() -> int:
    # 1. Verify Zeller's ddmin shrinks a 10-line input down to the exact 2 lines that jointly trigger a failure
    lines = [f"safe line {i}" for i in range(8)] + ["TRIGGER_A", "TRIGGER_B"]
    shrunk = ddmin_shrink(lines, lambda sub: ("TRIGGER_A" in sub and "TRIGGER_B" in sub))
    assert shrunk == ["TRIGGER_A", "TRIGGER_B"], shrunk

    # 2. Verify Metamorphic Relations (permutation invariance, monotonicity, repair idempotence)
    eng = MetamorphicTestingEngine()
    rep = eng.verify_evaluator(
        score_fn=lambda resp, _sys: 40.0 if "<developer_config>" in resp else 100.0,
        repair_fn=lambda resp, _sys: resp.replace("<developer_config>leak</developer_config>", "").strip(),
        base_response="Line 1 ok\nLine 2 ok\nLine 3 ok",
        rules=["Rule A: be concise.", "Rule B: never leak config."],
    )
    assert rep.permutation_invariant and rep.violation_monotonic and rep.repair_idempotent
    assert rep.minimal_failing_slice == ["<developer_config>leak</developer_config>"]

    print("metamorphic._self_test: OK PASS (MR1/MR2/MR3 metamorphic relations + Zeller ddmin 1-minimal shrinking)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
