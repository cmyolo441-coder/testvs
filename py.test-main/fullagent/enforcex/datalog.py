"""datalog — Stratified Datalog Deductive Rule Engine (Bottom-Up Fixpoint Evaluator).

Evaluates declarative Horn clauses with stratified negation over relational
security and compliance facts:
  * Facts (EDB): `("calls", "turn1", "read_file", ".env")`
  * Rules (IDB): `Head :- PosAtom1, PosAtom2, ~NegAtom1`
  * Variables are strings starting with uppercase (`"?X"` or `"X"`), constants
    start with lowercase or literals.

Computes the exact least fixpoint model and supports relational queries.
Negation is evaluated under *predicate stratification*: a negated predicate is
always fully computed in a strictly lower stratum before it is tested (rules
with recursion through negation are rejected and reported in ``errors`` rather
than producing an unsound model). Evaluation uses a semi-naive delta fixpoint
with per-predicate indexes.

Self-test: ``python3 -m fullagent.enforcex.datalog``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Atom",
    "DatalogRule",
    "StratifiedDatalogEngine",
]

Fact = Tuple[str, ...]

_MAX_FACTS = 1_000_000


@dataclass(frozen=True)
class Atom:
    pred: str
    terms: Tuple[str, ...]
    negated: bool = False


@dataclass(frozen=True)
class DatalogRule:
    head: Atom
    body: Tuple[Atom, ...]
    stratum: int = 0


def _is_var(term: str) -> bool:
    return bool(term) and (term.startswith("?") or term[0].isupper())


def _vars_of(atom: Atom) -> Set[str]:
    return {t for t in atom.terms if _is_var(t)}


def _match_atom(atom: Atom, fact: Fact, subst: Dict[str, str]) -> Optional[Dict[str, str]]:
    if not fact or atom.pred != fact[0] or len(atom.terms) != len(fact) - 1:
        return None
    new_s = dict(subst)
    for t, val in zip(atom.terms, fact[1:]):
        if _is_var(t):
            if t in new_s and new_s[t] != val:
                return None
            new_s[t] = val
        elif t != val:
            return None
    return new_s


def _instantiate(atom: Atom, subst: Dict[str, str]) -> Fact:
    return (atom.pred,) + tuple(subst.get(t, t) for t in atom.terms)


def _rule_is_safe(rule: DatalogRule) -> bool:
    """A rule is safe when every head/negated variable is bound positively."""
    positive: Set[str] = set()
    for atom in rule.body:
        if not atom.negated:
            positive |= _vars_of(atom)
    if not _vars_of(rule.head) <= positive:
        return False
    for atom in rule.body:
        if atom.negated and not _vars_of(atom) <= positive:
            return False
    return True


class StratifiedDatalogEngine:
    """Bottom-up fixpoint evaluator for stratified Datalog programs.

    Attributes:
        facts: materialized fact set (EDB seeds plus every derived fact).
        rules: registered rules in insertion order.
        errors: human-readable reasons rules were skipped (unsafe /
            unstratifiable); cleared on each ``materialize()`` call.
    """

    def __init__(self) -> None:
        self.facts: Set[Fact] = set()
        self.rules: List[DatalogRule] = []
        self.errors: List[str] = []

    # ── program construction ────────────────────────────────────────────

    def add_fact(self, pred: str, *args: str) -> None:
        self.facts.add((str(pred),) + tuple(str(a) for a in args))

    def add_rule(self, head: Atom, body: Sequence[Atom], *, stratum: int = 0) -> None:
        self.rules.append(DatalogRule(head=head, body=tuple(body), stratum=int(stratum)))

    # ── stratification ──────────────────────────────────────────────────

    @staticmethod
    def _sccs(nodes: Sequence[str], edges: Dict[str, Set[str]]) -> List[List[str]]:
        """Iterative Tarjan SCC so deep rule graphs never hit the recursion limit."""
        index: Dict[str, int] = {}
        low: Dict[str, int] = {}
        on_stack: Set[str] = set()
        stack: List[str] = []
        out: List[List[str]] = []
        counter = 0
        for root in nodes:
            if root in index:
                continue
            work: List[Tuple[str, int, List[str]]] = [(root, 0, sorted(edges.get(root, ())))]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, pos, succ = work[-1]
                if pos < len(succ):
                    work[-1] = (node, pos + 1, succ)
                    nxt = succ[pos]
                    if nxt not in index:
                        index[nxt] = low[nxt] = counter
                        counter += 1
                        stack.append(nxt)
                        on_stack.add(nxt)
                        work.append((nxt, 0, sorted(edges.get(nxt, ()))))
                    elif nxt in on_stack:
                        low[node] = min(low[node], index[nxt])
                else:
                    work.pop()
                    if work:
                        parent = work[-1][0]
                        low[parent] = min(low[parent], low[node])
                    if low[node] == index[node]:
                        comp: List[str] = []
                        while True:
                            w = stack.pop()
                            on_stack.discard(w)
                            comp.append(w)
                            if w == node:
                                break
                        out.append(comp)
        return out

    def _stratify(self, rules: List[DatalogRule]) -> Tuple[Dict[str, int], List[DatalogRule]]:
        """Assign a stratum to every IDB predicate; drop unstratifiable rules.

        Returns ``(predicate_strata, usable_rules)``. A rule is dropped when its
        head (or another predicate in the same SCC) depends negatively on a
        predicate of that SCC, or when it is unsafe.
        """
        usable = [r for r in rules if _rule_is_safe(r)]
        for r in rules:
            if r not in usable:
                self.errors.append(
                    f"unsafe rule skipped (unbound head/negated variable): {r.head.pred}"
                )

        while usable:
            idb = {r.head.pred for r in usable}
            edges: Dict[str, Set[str]] = {p: set() for p in idb}
            neg_edges: Set[Tuple[str, str]] = set()
            for r in usable:
                for atom in r.body:
                    if atom.pred in idb:
                        edges[r.head.pred].add(atom.pred)
                        if atom.negated:
                            neg_edges.add((r.head.pred, atom.pred))
            comps = self._sccs(sorted(idb), edges)
            comp_of: Dict[str, int] = {}
            for ci, comp in enumerate(comps):
                for p in comp:
                    comp_of[p] = ci
            bad = {ci for ci, comp in enumerate(comps)
                   if any((a, b) in neg_edges and a in comp and b in comp
                          for a in comp for b in comp)}
            if not bad:
                break
            kept: List[DatalogRule] = []
            for r in usable:
                if comp_of.get(r.head.pred) in bad:
                    self.errors.append(
                        f"unstratifiable rule skipped (recursion through negation): {r.head.pred}"
                    )
                else:
                    kept.append(r)
            usable = kept

        idb = {r.head.pred for r in usable}
        declared: Dict[str, int] = {}
        for r in usable:
            declared[r.head.pred] = max(declared.get(r.head.pred, 0), int(r.stratum))
        deps: Dict[str, Set[Tuple[str, bool]]] = {p: set() for p in idb}
        for r in usable:
            for atom in r.body:
                if atom.pred in idb:
                    deps[r.head.pred].add((atom.pred, bool(atom.negated)))
        strata: Dict[str, int] = {}

        def compute(pred: str, visiting: Set[str]) -> int:
            if pred in strata:
                return strata[pred]
            if pred in visiting:  # positive cycle: all members share a stratum
                return 0
            visiting.add(pred)
            best = declared.get(pred, 0)
            for dep, negated in deps.get(pred, ()):
                child = compute(dep, visiting) + (1 if negated else 0)
                if child > best:
                    best = child
            visiting.discard(pred)
            strata[pred] = best
            return best

        for p in sorted(idb):
            compute(p, set())
        return strata, usable

    # ── evaluation ──────────────────────────────────────────────────────

    @staticmethod
    def _build_index(db: Set[Fact]) -> Dict[str, List[Tuple[str, ...]]]:
        index: Dict[str, List[Tuple[str, ...]]] = {}
        for fact in db:
            index.setdefault(fact[0], []).append(fact[1:])
        return index

    @staticmethod
    def _join(rule: DatalogRule, db_index: Dict[str, List[Tuple[str, ...]]],
              delta_index: Optional[Dict[str, List[Tuple[str, ...]]]],
              delta_pos: int) -> List[Dict[str, str]]:
        """Join the positive body atoms of ``rule``; one atom may come from delta."""
        positives = [(i, a) for i, a in enumerate(rule.body) if not a.negated]

        def size(pair: Tuple[int, Atom]) -> int:
            pos, atom = pair
            if delta_index is not None and pos == delta_pos:
                return len(delta_index.get(atom.pred, ()))
            return len(db_index.get(atom.pred, ()))

        # Delta atom first, then smallest relation first (classic join ordering).
        positives.sort(key=lambda pair: (0 if pair[0] == delta_pos else 1, size(pair)))
        results: List[Dict[str, str]] = [{}]
        for pos, atom in positives:
            src = delta_index if (delta_index is not None and pos == delta_pos) else db_index
            candidates = src.get(atom.pred, ())
            arity = len(atom.terms)
            nxt: List[Dict[str, str]] = []
            for subst in results:
                for args in candidates:
                    if len(args) != arity:
                        continue
                    new_s = _match_atom(atom, (atom.pred,) + args, subst)
                    if new_s is not None:
                        nxt.append(new_s)
            results = nxt
            if not results:
                break
        return results

    def _derive_round(self, rules: Sequence[DatalogRule],
                      db_index: Dict[str, List[Tuple[str, ...]]],
                      delta_index: Optional[Dict[str, List[Tuple[str, ...]]]]) -> Set[Fact]:
        out: Set[Fact] = set()
        for rule in rules:
            n_pos = sum(1 for a in rule.body if not a.negated)
            if n_pos == 0:
                substitutions: List[Dict[str, str]] = [{}]
            elif delta_index is None:
                substitutions = self._join(rule, db_index, None, -1)
            else:
                substitutions = []
                for pos, atom in enumerate(rule.body):
                    if atom.negated or not delta_index.get(atom.pred):
                        continue
                    substitutions.extend(self._join(rule, db_index, delta_index, pos))
            for subst in substitutions:
                if any(
                    any(_match_atom(na, (na.pred,) + args, subst) is not None
                        for args in db_index.get(na.pred, ()))
                    for na in rule.body if na.negated
                ):
                    continue
                out.add(_instantiate(rule.head, subst))
        return out

    def materialize(self, *, max_facts: int = _MAX_FACTS) -> Set[Fact]:
        """Evaluate all strata to fixpoint and return the full materialized fact set."""
        self.errors = []
        strata, rules = self._stratify(list(self.rules))
        if not rules:
            return set(self.facts)
        db: Set[Fact] = set(self.facts)
        cap = max(1, int(max_facts))

        by_stratum: Dict[int, List[DatalogRule]] = {}
        for r in rules:
            by_stratum.setdefault(strata.get(r.head.pred, 0), []).append(r)

        for st in sorted(by_stratum):
            st_rules = by_stratum[st]
            db_index = self._build_index(db)
            # First round: every fact is "new" relative to this stratum.
            delta_index: Optional[Dict[str, List[Tuple[str, ...]]]] = db_index
            while True:
                derived = self._derive_round(st_rules, db_index, delta_index)
                new_facts = derived - db
                if not new_facts:
                    break
                db |= new_facts
                db_index = self._build_index(db)
                if len(db) > cap:
                    self.errors.append(f"fact cap reached ({cap}); fixpoint truncated")
                    break
                delta_index = self._build_index(new_facts)

        self.facts = db
        return db

    def query(self, pred: str, *pattern: str) -> List[Tuple[str, ...]]:
        self.materialize()
        q_atom = Atom(str(pred), tuple(str(p) for p in pattern))
        out: List[Tuple[str, ...]] = []
        for f in sorted(self.facts):
            if _match_atom(q_atom, f, {}) is not None:
                out.append(f[1:])
        return out


def _self_test() -> int:
    eng = StratifiedDatalogEngine()
    # Taint propagation + negation rule:
    # tainted_session(S) :- read_secret(S, F)
    # exfil_violation(S, U) :- tainted_session(S), net_post(S, U), ~sanitized(S)
    eng.add_fact("read_secret", "sess_1", ".env")
    eng.add_fact("net_post", "sess_1", "https://evil.example")
    eng.add_fact("read_secret", "sess_2", ".env")
    eng.add_fact("sanitized", "sess_2")
    eng.add_fact("net_post", "sess_2", "https://api.internal")

    eng.add_rule(
        Atom("tainted_session", ("?S",)),
        [Atom("read_secret", ("?S", "?F"))],
        stratum=0,
    )
    eng.add_rule(
        Atom("exfil_violation", ("?S", "?U")),
        [
            Atom("tainted_session", ("?S",)),
            Atom("net_post", ("?S", "?U")),
            Atom("sanitized", ("?S",), negated=True),
        ],
        stratum=1,
    )

    hits = eng.query("exfil_violation", "?S", "?U")
    assert hits == [("sess_1", "https://evil.example")], hits

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. empty program: no facts, no crash
    empty = StratifiedDatalogEngine()
    assert empty.materialize() == set() and empty.query("anything") == []

    # 2. negation is auto-stratified even when every rule declares stratum 0
    auto = StratifiedDatalogEngine()
    auto.add_fact("edge", "a", "b")
    auto.add_fact("edge", "b", "c")
    auto.add_fact("blocked", "b")
    auto.add_rule(Atom("reach", ("?X", "?Y")), [Atom("edge", ("?X", "?Y"))])
    auto.add_rule(
        Atom("reach", ("?X", "?Z")),
        [Atom("reach", ("?X", "?Y")), Atom("edge", ("?Y", "?Z"))],
    )
    auto.add_rule(
        Atom("unsafe", ("?X",)),
        [Atom("reach", ("?X", "?Y")), Atom("blocked", ("?Y",), negated=True)],
    )
    assert auto.query("unsafe", "?X") == [("a",), ("b",)], auto.query("unsafe", "?X")
    assert ("reach", "a", "c") in auto.materialize()

    # 3. repeated materialize is idempotent
    before = set(auto.materialize())
    assert auto.materialize() == before

    # 4. unsafe rule (unbound head variable) is skipped, never fabricates "?X"
    unsafe = StratifiedDatalogEngine()
    unsafe.add_fact("seed", "s")
    unsafe.add_rule(Atom("bad", ("?X",)), [Atom("seed", ("?Y",))])
    assert unsafe.materialize() == {("seed", "s")}
    assert unsafe.errors and "unsafe" in unsafe.errors[0]

    # 5. recursion through negation is rejected instead of looping / lying
    loop = StratifiedDatalogEngine()
    loop.add_rule(Atom("p", ("?X",)), [Atom("q", ("?X",))])
    loop.add_rule(Atom("q", ("?X",)), [Atom("p", ("?X",)), Atom("p", ("?X",), negated=True)])
    assert loop.materialize() == set()
    assert any("unstratifiable" in e for e in loop.errors), loop.errors

    # 6. large transitive closure terminates with the exact count (semi-naive)
    chain = StratifiedDatalogEngine()
    for i in range(100):
        chain.add_fact("next", f"n{i}", f"n{i + 1}")
    chain.add_rule(Atom("path", ("?X", "?Y")), [Atom("next", ("?X", "?Y"))])
    chain.add_rule(
        Atom("path", ("?X", "?Z")),
        [Atom("path", ("?X", "?Y")), Atom("next", ("?Y", "?Z"))],
    )
    assert len(chain.query("path", "?X", "?Y")) == 100 * 101 // 2
    # 7. constants in query patterns filter exactly
    assert chain.query("next", "n0", "?Y") == [("n0", "n1")]
    assert chain.query("next", "n0", "n2") == []

    print("datalog._self_test: OK PASS (stratified semi-naive fixpoint, unification, negation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
