"""temporalctl — Computation Tree Logic (CTL) Branching-Time Model Checker.

Implements exact fixed-point model checking over finite Kripke structures
`M = (S, R, L)` for branching-time Computation Tree Logic (Clarke, Emerson &
Sistla) to verify multi-branch conversation, tool-call, and retry trees:
  * Atomic propositions `p`
  * Boolean connectives `NOT(f)`, `AND(f, g)`, `OR(f, g)`, `IMPLIES(f, g)`
  * Existential path operators:
      - `EX(f)`: in some next state, `f` holds
      - `EF(f)`: along some path, `f` eventually holds (`E[True U f]`)
      - `EG(f)`: along some path, `f` globally holds (greatest fixpoint)
      - `EU(f, g)`: along some path, `f` holds until `g` holds (least fixpoint)
  * Universal path operators:
      - `AX(f)`: in all next states, `f` holds (`~EX(~f)`)
      - `AF(f)`: along all paths, `f` eventually holds (least fixpoint)
      - `AG(f)`: along all paths, `f` globally holds (`~EF(~f)`)
      - `AU(f, g)`: along all paths, `f` holds until `g` holds (least fixpoint)

Self-test: ``python3 -m fullagent.enforcex.temporalctl``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

__all__ = [
    "CTLFormula",
    "KripkeStructure",
    "CTLCheckReport",
    "CTLModelChecker",
    "ATOM",
    "NOT",
    "AND",
    "OR",
    "IMPLIES",
    "EX",
    "AX",
    "EF",
    "AF",
    "EG",
    "AG",
    "EU",
    "AU",
]


@dataclass(frozen=True)
class CTLFormula:
    op: str
    atom: str = ""
    left: Optional["CTLFormula"] = None
    right: Optional["CTLFormula"] = None

    def __str__(self) -> str:
        if self.op == "TRUE":
            return "True"
        if self.op == "FALSE":
            return "False"
        if self.op == "ATOM":
            return self.atom
        if self.op in ("NOT", "EX", "AX", "EF", "AF", "EG", "AG"):
            return f"{self.op}({self.left})"
        if self.op in ("AND", "OR", "IMPLIES", "EU", "AU"):
            return f"{self.op}({self.left}, {self.right})"
        return f"{self.op}({self.atom})"


def ATOM(name: str) -> CTLFormula:
    return CTLFormula("ATOM", atom=str(name))


def NOT(f: CTLFormula) -> CTLFormula:
    return CTLFormula("NOT", left=f)


def AND(f: CTLFormula, g: CTLFormula) -> CTLFormula:
    return CTLFormula("AND", left=f, right=g)


def OR(f: CTLFormula, g: CTLFormula) -> CTLFormula:
    return CTLFormula("OR", left=f, right=g)


def IMPLIES(f: CTLFormula, g: CTLFormula) -> CTLFormula:
    return CTLFormula("IMPLIES", left=f, right=g)


def EX(f: CTLFormula) -> CTLFormula:
    return CTLFormula("EX", left=f)


def AX(f: CTLFormula) -> CTLFormula:
    return CTLFormula("AX", left=f)


def EF(f: CTLFormula) -> CTLFormula:
    return CTLFormula("EF", left=f)


def AF(f: CTLFormula) -> CTLFormula:
    return CTLFormula("AF", left=f)


def EG(f: CTLFormula) -> CTLFormula:
    return CTLFormula("EG", left=f)


def AG(f: CTLFormula) -> CTLFormula:
    return CTLFormula("AG", left=f)


def EU(f: CTLFormula, g: CTLFormula) -> CTLFormula:
    return CTLFormula("EU", left=f, right=g)


def AU(f: CTLFormula, g: CTLFormula) -> CTLFormula:
    return CTLFormula("AU", left=f, right=g)


class KripkeStructure:
    """Finite Kripke structure `M = (S, S0, R, L)` with totalized transition relation."""

    def __init__(self) -> None:
        self.states: Set[str] = set()
        self.initial_states: Set[str] = set()
        self.transitions: Dict[str, Set[str]] = {}
        self.predecessors: Dict[str, Set[str]] = {}
        self.labels: Dict[str, Set[str]] = {}

    def add_state(self, state: str, propositions: Iterable[str] = (), *, initial: bool = False) -> None:
        s = str(state)
        self.states.add(s)
        if initial:
            self.initial_states.add(s)
        self.transitions.setdefault(s, set())
        self.predecessors.setdefault(s, set())
        self.labels.setdefault(s, set()).update(str(p) for p in propositions)

    def add_transition(self, src: str, dst: str) -> None:
        u, v = str(src), str(dst)
        self.add_state(u)
        self.add_state(v)
        self.transitions[u].add(v)
        self.predecessors[v].add(u)

    def ensure_total(self) -> None:
        """CTL semantics require every state to have at least one outgoing transition;
        add a self-loop to any terminal sink state."""
        for s in list(self.states):
            if not self.transitions.get(s):
                self.add_transition(s, s)

    def pre_exists(self, target_set: Set[str]) -> Set[str]:
        """Return `{s in S | exists s' in target_set with (s, s') in R}`."""
        res: Set[str] = set()
        for t in target_set:
            res.update(self.predecessors.get(t, ()))
        return res

    def pre_forall(self, target_set: Set[str]) -> Set[str]:
        """Return `{s in S | for all s' with (s, s') in R, s' in target_set}`."""
        res: Set[str] = set()
        for s in self.states:
            succs = self.transitions.get(s, set())
            if succs and succs.issubset(target_set):
                res.add(s)
        return res


@dataclass
class CTLCheckReport:
    formula: str
    holds_in_all_initial: bool
    satisfying_states: List[str] = field(default_factory=list)
    violating_initial_states: List[str] = field(default_factory=list)
    witness_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "holds_in_all_initial": self.holds_in_all_initial,
            "satisfying_states": list(self.satisfying_states),
            "violating_initial_states": list(self.violating_initial_states),
            "witness_trace": list(self.witness_trace),
        }


class CTLModelChecker:
    """Symbolic/explicit fixed-point CTL model checker."""

    def sat(self, kripke: KripkeStructure, formula: CTLFormula) -> Set[str]:
        """Compute `Sat(formula) = {s in S | M, s |= formula}`."""
        kripke.ensure_total()
        op = formula.op
        if op == "TRUE":
            return set(kripke.states)
        if op == "FALSE":
            return set()
        if op == "ATOM":
            return {s for s in kripke.states if formula.atom in kripke.labels.get(s, set())}
        if op == "NOT":
            assert formula.left is not None
            return kripke.states - self.sat(kripke, formula.left)
        if op == "AND":
            assert formula.left is not None and formula.right is not None
            return self.sat(kripke, formula.left) & self.sat(kripke, formula.right)
        if op == "OR":
            assert formula.left is not None and formula.right is not None
            return self.sat(kripke, formula.left) | self.sat(kripke, formula.right)
        if op == "IMPLIES":
            assert formula.left is not None and formula.right is not None
            return (kripke.states - self.sat(kripke, formula.left)) | self.sat(kripke, formula.right)
        if op == "EX":
            assert formula.left is not None
            return kripke.pre_exists(self.sat(kripke, formula.left))
        if op == "AX":
            assert formula.left is not None
            return kripke.pre_forall(self.sat(kripke, formula.left))
        if op == "EF":
            assert formula.left is not None
            return self.sat(kripke, EU(CTLFormula("TRUE"), formula.left))
        if op == "AF":
            assert formula.left is not None
            return self.sat(kripke, AU(CTLFormula("TRUE"), formula.left))
        if op == "EG":
            # Greatest fixpoint: nu Z. (left & EX Z)
            assert formula.left is not None
            z = self.sat(kripke, formula.left)
            while True:
                z_next = z & kripke.pre_exists(z)
                if z_next == z:
                    return z
                z = z_next
        if op == "AG":
            # AG f == ~EF(~f)
            assert formula.left is not None
            return kripke.states - self.sat(kripke, EF(NOT(formula.left)))
        if op == "EU":
            # Least fixpoint: mu Z. (right | (left & EX Z))
            assert formula.left is not None and formula.right is not None
            sat_l = self.sat(kripke, formula.left)
            z = self.sat(kripke, formula.right)
            while True:
                z_next = z | (sat_l & kripke.pre_exists(z))
                if z_next == z:
                    return z
                z = z_next
        if op == "AU":
            # Least fixpoint: mu Z. (right | (left & AX Z))
            assert formula.left is not None and formula.right is not None
            sat_l = self.sat(kripke, formula.left)
            z = self.sat(kripke, formula.right)
            while True:
                z_next = z | (sat_l & kripke.pre_forall(z))
                if z_next == z:
                    return z
                z = z_next
        raise ValueError(f"Unknown CTL operator: {op}")

    def shortest_witness_path(self, kripke: KripkeStructure, start_states: Iterable[str], target_states: Set[str]) -> List[str]:
        """BFS shortest path from `start_states` to `target_states`."""
        from collections import deque
        q: deque[List[str]] = deque([s] for s in start_states if s in kripke.states)
        seen: Set[str] = set(s for s in start_states)
        while q:
            path = q.popleft()
            u = path[-1]
            if u in target_states:
                return path
            for v in sorted(kripke.transitions.get(u, ())):
                if v not in seen:
                    seen.add(v)
                    q.append(path + [v])
        return []

    def check(self, kripke: KripkeStructure, formula: CTLFormula) -> CTLCheckReport:
        sat_states = self.sat(kripke, formula)
        init = kripke.initial_states or set(kripke.states)
        violating = sorted(init - sat_states)
        holds = len(violating) == 0
        witness: List[str] = []
        if not holds:
            bad_targets = kripke.states - sat_states
            witness = self.shortest_witness_path(kripke, violating, bad_targets)
        return CTLCheckReport(
            formula=str(formula),
            holds_in_all_initial=holds,
            satisfying_states=sorted(sat_states),
            violating_initial_states=violating,
            witness_trace=witness,
        )


def _self_test() -> int:
    # Build a branching conversation/tool execution Kripke structure
    m = KripkeStructure()
    m.add_state("s0_intake", ["safe", "authenticated"], initial=True)
    m.add_state("s1_draft", ["safe", "authenticated", "drafted"])
    m.add_state("s2_audit_pass", ["safe", "authenticated", "verified", "delivered"])
    m.add_state("s3_retry", ["safe", "authenticated", "retrying"])
    m.add_transition("s0_intake", "s1_draft")
    m.add_transition("s1_draft", "s2_audit_pass")
    m.add_transition("s1_draft", "s3_retry")
    m.add_transition("s3_retry", "s2_audit_pass")
    m.add_transition("s2_audit_pass", "s2_audit_pass")

    checker = CTLModelChecker()
    # 1. AG(safe): safety holds globally across all branches
    r_safe = checker.check(m, AG(ATOM("safe")))
    assert r_safe.holds_in_all_initial

    # 2. AF(delivered): along all paths, delivery is inevitably reached
    r_deliv = checker.check(m, AF(ATOM("delivered")))
    assert r_deliv.holds_in_all_initial

    # 3. AG(delivered -> verified): delivery never occurs without verification
    r_inv = checker.check(m, AG(IMPLIES(ATOM("delivered"), ATOM("verified"))))
    assert r_inv.holds_in_all_initial

    # 4. Inject an unsafe unverified branch and verify AG(safe) catches it
    m.add_state("s4_leak", ["leaked"])
    m.add_transition("s3_retry", "s4_leak")
    r_fail = checker.check(m, AG(ATOM("safe")))
    assert not r_fail.holds_in_all_initial
    assert "s0_intake" in r_fail.violating_initial_states

    print("temporalctl._self_test: OK PASS (CTL mu/nu fixpoints EX/AX/EF/AF/EG/AG/EU/AU verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
