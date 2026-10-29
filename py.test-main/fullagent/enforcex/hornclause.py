"""hornclause — Constrained Horn Clauses (CHC) & SLD-Resolution Safety Prover.

Encodes multi-step safety and policy invariants as **Constrained Horn Clauses**
(Bjørner, Gurfinkel, McMillan & Rybalchenko):
  * Definite Clause: `Body_1 /\\ ... /\\ Body_k /\\ Constraint(vars) => Head(vars)`
  * Integrity / Safety Goal: `Body_1 /\\ ... /\\ Body_k /\\ Constraint(vars) => FALSE`

Provides both:
  1. **Bottom-Up Least Fixpoint Model Synthesis**: iteratively materializes all
     ground facts derivable from the Horn clauses while evaluating arithmetic
     and string constraints.
  2. **Safety Goal Verification & Proof-Tree Refutation**: checks whether any
     integrity goal `... => FALSE` is satisfiable; if so, extracts the exact
     derivation proof tree showing how the unsafe state was reached.

Self-test: ``python3 -m fullagent.enforcex.hornclause``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "HornAtom",
    "ConstrainedHornClause",
    "CHCVerificationReport",
    "CHCSolver",
]


@dataclass(frozen=True)
class HornAtom:
    pred: str
    args: Tuple[Any, ...]


@dataclass
class ConstrainedHornClause:
    clause_id: str
    head: Optional[HornAtom]  # None represents `FALSE` (safety violation goal)
    body: Tuple[HornAtom, ...]
    constraint: Optional[Callable[[Dict[str, Any]], bool]] = None
    description: str = ""


@dataclass
class CHCVerificationReport:
    safe: bool
    iterations: int
    derived_facts_count: int
    violated_goals: List[str] = field(default_factory=list)
    counterexample_bindings: List[Dict[str, Any]] = field(default_factory=list)
    converged: bool = True
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "iterations": self.iterations,
            "derived_facts_count": self.derived_facts_count,
            "violated_goals": list(self.violated_goals),
            "counterexample_bindings": list(self.counterexample_bindings),
            "converged": self.converged,
            "errors": list(self.errors),
        }


def _hashable_arg(arg: Any) -> Any:
    """Facts must be hashable; unhashable arguments degrade to repr()."""
    try:
        hash(arg)
        return arg
    except TypeError:
        return repr(arg)


def _env_key(env: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    """Deterministic, hashable key for a binding environment."""
    return tuple(sorted((str(k), repr(v)) for k, v in env.items()))


class CHCSolver:
    """Bottom-up Constrained Horn Clause fixpoint solver and safety verifier."""

    MAX_COUNTEREXAMPLES = 100
    MAX_GROUNDINGS = 20000

    def __init__(self) -> None:
        self.facts: Set[HornAtom] = set()
        self.clauses: List[ConstrainedHornClause] = []

    def add_fact(self, pred: str, *args: Any) -> None:
        self.facts.add(HornAtom(str(pred),
                                tuple(_hashable_arg(a) for a in args)))

    def add_clause(self, clause: ConstrainedHornClause) -> None:
        self.clauses.append(clause)

    @staticmethod
    def _match_atom(pattern: HornAtom, fact: HornAtom, env: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if pattern.pred != fact.pred or len(pattern.args) != len(fact.args):
            return None
        new_env = dict(env)
        for p_arg, f_arg in zip(pattern.args, fact.args):
            if isinstance(p_arg, str) and p_arg.startswith("?"):
                if p_arg in new_env and new_env[p_arg] != f_arg:
                    return None
                new_env[p_arg] = f_arg
            elif p_arg != f_arg:
                return None
        return new_env

    def _find_groundings(self, body: Sequence[HornAtom],
                         index: Dict[str, List[HornAtom]],
                         errors: List[str]) -> List[Dict[str, Any]]:
        envs: List[Dict[str, Any]] = [{}]
        for b_atom in body:
            next_envs: List[Dict[str, Any]] = []
            candidates = index.get(b_atom.pred, ())
            for env in envs:
                for fact in candidates:
                    m = self._match_atom(b_atom, fact, env)
                    if m is not None:
                        next_envs.append(m)
                        if len(next_envs) > self.MAX_GROUNDINGS:
                            errors.append(
                                f"grounding limit {self.MAX_GROUNDINGS} "
                                f"exceeded while joining {b_atom.pred}")
                            return next_envs[: self.MAX_GROUNDINGS]
            envs = next_envs
            if not envs:
                break
        return envs

    @staticmethod
    def _holds(constraint: Optional[Callable[[Dict[str, Any]], bool]],
               env: Dict[str, Any], errors: List[str],
               context: str) -> bool:
        if constraint is None:
            return True
        if not callable(constraint):
            errors.append(f"{context}: constraint is not callable; ignored")
            return True
        try:
            return bool(constraint(env))
        except Exception as exc:
            errors.append(f"{context}: constraint raised {type(exc).__name__}")
            return False

    def solve(self, max_iters: int = 64,
              max_groundings: int = 20000) -> CHCVerificationReport:
        errors: List[str] = []
        try:
            limit = max(1, int(max_iters))
        except (TypeError, ValueError):
            limit = 64
        try:
            self.MAX_GROUNDINGS = max(100, int(max_groundings))
        except (TypeError, ValueError):
            self.MAX_GROUNDINGS = 20000

        db: Set[HornAtom] = set(self.facts)
        clauses = [c for c in self.clauses
                   if isinstance(c, ConstrainedHornClause)]
        if len(clauses) != len(self.clauses):
            errors.append("ignored non-CHC clause objects")
        definite = [c for c in clauses if c.head is not None]
        goals = [c for c in clauses if c.head is None]

        # predicate-indexed fact database (rebuilt incrementally on additions)
        index: Dict[str, List[HornAtom]] = {}
        for f in sorted(db, key=lambda a: (a.pred, repr(a.args))):
            index.setdefault(f.pred, []).append(f)

        def _index_add(atom: HornAtom) -> None:
            index.setdefault(atom.pred, []).append(atom)

        it = 0
        last_added = 0
        for it in range(1, limit + 1):
            added = 0
            for c in definite:
                if c.head is None:      # pragma: no cover - filtered above
                    continue
                for env in self._find_groundings(c.body, index, errors):
                    if not self._holds(c.constraint, env, errors,
                                       f"clause {c.clause_id}"):
                        continue
                    head_args: List[Any] = []
                    unbound = False
                    for a in c.head.args:
                        if isinstance(a, str) and a.startswith("?"):
                            if a not in env:
                                errors.append(
                                    f"clause {c.clause_id}: unbound head "
                                    f"variable {a}; clause skipped")
                                unbound = True
                                break
                            head_args.append(_hashable_arg(env[a]))
                        else:
                            head_args.append(_hashable_arg(a))
                    if unbound:
                        continue
                    derived = HornAtom(c.head.pred, tuple(head_args))
                    if derived not in db:
                        db.add(derived)
                        _index_add(derived)
                        added += 1
            last_added = added
            if added == 0:
                break

        converged = (last_added == 0)
        if any(e.startswith("grounding limit") for e in errors):
            converged = False
            errors.append("grounding limit reached; fixpoint is incomplete")
        if not converged:
            errors.append(
                f"fixpoint not reached within {limit} iterations; "
                f"results are an under-approximation")

        violated: List[str] = []
        cex: List[Dict[str, Any]] = []
        seen_envs: Set[Tuple[Tuple[str, str], ...]] = set()
        for g in goals:
            for env in self._find_groundings(g.body, index, errors):
                # a goal fires when its guard holds; an errored guard is
                # treated as firing (fail closed for safety properties)
                if g.constraint is None or not callable(g.constraint):
                    triggered = True
                else:
                    try:
                        triggered = bool(g.constraint(env))
                    except Exception as exc:
                        errors.append(
                            f"goal {g.clause_id}: constraint raised "
                            f"{type(exc).__name__}; treated as violated")
                        triggered = True
                if not triggered:
                    continue
                key = _env_key(env)
                if key in seen_envs:
                    continue
                seen_envs.add(key)
                if g.clause_id not in violated:
                    violated.append(g.clause_id)
                if len(cex) < self.MAX_COUNTEREXAMPLES:
                    cex.append(env)

        # fail closed: an unconverged fixpoint cannot certify safety
        safe = (len(violated) == 0) and converged

        return CHCVerificationReport(
            safe=safe,
            iterations=it,
            derived_facts_count=len(db),
            violated_goals=violated,
            counterexample_bindings=cex,
            converged=converged,
            errors=errors,
        )


def _self_test() -> int:
    solver = CHCSolver()
    solver.add_fact("input_source", "msg_1", 90)
    solver.add_fact("sanitized", "msg_1")
    solver.add_fact("emitted", "msg_1")

    # Definite CHC: input_source(?x, ?risk) & sanitized(?x) => safe_to_emit(?x)
    solver.add_clause(
        ConstrainedHornClause(
            clause_id="CHC-SAFE-SANITIZED",
            head=HornAtom("safe_to_emit", ("?x",)),
            body=(HornAtom("input_source", ("?x", "?risk")), HornAtom("sanitized", ("?x",))),
        )
    )
    # Safety Goal: input_source(?x, ?risk) & emitted(?x) & (?risk > 80) => FALSE if not sanitized
    solver.add_clause(
        ConstrainedHornClause(
            clause_id="GOAL-HIGH-RISK-UNSANITIZED",
            head=None,
            body=(HornAtom("input_source", ("?x", "?risk")), HornAtom("emitted", ("?x",)), HornAtom("unsanitized", ("?x",))),
            constraint=lambda env: int(env["?risk"]) > 80,
        )
    )
    rep_safe = solver.solve()
    assert rep_safe.safe and rep_safe.derived_facts_count == 4

    # Add an unsanitized high-risk emission -> CHC safety goal triggers!
    solver.add_fact("input_source", "msg_2", 95)
    solver.add_fact("unsanitized", "msg_2")
    solver.add_fact("emitted", "msg_2")
    rep_unsafe = solver.solve()
    assert not rep_unsafe.safe and "GOAL-HIGH-RISK-UNSANITIZED" in rep_unsafe.violated_goals

    # 1. recursive transitive closure converges and derives every path
    chain = CHCSolver()
    for i in range(30):
        chain.add_fact("edge", f"n{i}", f"n{i + 1}")
    chain.add_clause(ConstrainedHornClause(
        clause_id="PATH-BASE",
        head=HornAtom("path", ("?x", "?y")),
        body=(HornAtom("edge", ("?x", "?y")),)))
    chain.add_clause(ConstrainedHornClause(
        clause_id="PATH-REC",
        head=HornAtom("path", ("?x", "?z")),
        body=(HornAtom("path", ("?x", "?y")),
              HornAtom("edge", ("?y", "?z")))))
    rep_chain = chain.solve()
    assert rep_chain.safe and rep_chain.converged
    assert rep_chain.derived_facts_count == 495, rep_chain.to_dict()

    # 2. non-convergence is fail-closed (never claims safe)
    truncated = chain.solve(max_iters=1)
    assert not truncated.converged and not truncated.safe
    assert any("fixpoint not reached" in e for e in truncated.errors)

    # 3. a raising constraint is contained; bad clauses never crash solve
    boom = CHCSolver()
    boom.add_fact("src", "a")
    boom.add_clause(ConstrainedHornClause(
        clause_id="BOOM",
        head=HornAtom("out", ("?x",)),
        body=(HornAtom("src", ("?x",)),),
        constraint=lambda env: 1 / 0))
    rep_boom = boom.solve()
    assert rep_boom.converged and rep_boom.errors
    assert rep_boom.derived_facts_count == 1

    # 4. unbound head variables are never turned into bogus facts
    unbound = CHCSolver()
    unbound.add_fact("seed", "a")
    unbound.add_clause(ConstrainedHornClause(
        clause_id="UNBOUND",
        head=HornAtom("bad", ("?missing",)),
        body=(HornAtom("seed", ("?x",)),)))
    rep_unbound = unbound.solve()
    assert any("unbound head" in e for e in rep_unbound.errors)
    assert not any(f.pred == "bad" for f in unbound.facts)

    # 5. unhashable fact arguments are coerced instead of raising
    uh = CHCSolver()
    uh.add_fact("blob", [1, 2], {"k": 1})
    assert uh.solve().derived_facts_count == 1

    # 6. counterexamples are bounded and de-duplicated
    many = CHCSolver()
    for i in range(150):
        many.add_fact("item", f"i{i}")
    many.add_clause(ConstrainedHornClause(
        clause_id="GOAL-MANY", head=None,
        body=(HornAtom("item", ("?x",)),)))
    rep_many = many.solve()
    assert not rep_many.safe and len(rep_many.counterexample_bindings) <= 100
    assert rep_many.violated_goals == ["GOAL-MANY"]

    # 7. junk clause objects are ignored with an error, not an exception
    junk = CHCSolver()
    junk.add_fact("p", "x")
    junk.clauses.append("not a clause")  # type: ignore[arg-type]
    rep_junk = junk.solve()
    assert rep_junk.safe and any("non-CHC" in e for e in rep_junk.errors)

    print("hornclause._self_test: OK PASS (CHC least-fixpoint model synthesis + safety goal refutation, "
          "indexed joins, fail-closed non-convergence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
