"""egraph — Equality Saturation (E-Graph), Union-Find Congruence Closure & Cost Extractor.

Implements a compact **E-Graph** (Nelson-Oppen congruence closure + Willsey et
al. `egg` equality saturation) to canonicalize equivalent expressions, rules,
and code fragments:
  * **E-Node `f(c_1, ..., c_k)`**: an operator symbol applied to child **E-Class** IDs.
  * **E-Class**: an equivalence class of E-Nodes representing the same semantic value.
  * **Path-Compressed Union-Find**: maintains canonical E-Class representatives.
  * **Upward Congruence Closure (`rebuild()`)**: whenever `find(c1) == find(c2)`,
    parent E-Nodes `f(c1)` and `f(c2)` are automatically merged into the same E-Class!
  * **Minimum-Cost AST Extraction**: bottom-up dynamic programming extracts the
    simplest canonical term from any E-Class.

Self-test: ``python3 -m fullagent.enforcex.egraph``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

__all__ = [
    "ENode",
    "TermAST",
    "EGraph",
]

_MAX_DEPTH = 256


@dataclass(frozen=True)
class ENode:
    op: str
    children: Tuple[int, ...] = ()

    def canonicalize(self, find_fn: Callable[[int], int]) -> "ENode":
        return ENode(self.op, tuple(find_fn(c) for c in self.children))

    def sort_key(self) -> Tuple[str, Tuple[int, ...]]:
        return (self.op, self.children)


@dataclass(frozen=True)
class TermAST:
    op: str
    args: Tuple["TermAST", ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.op
        return f"({self.op} " + " ".join(str(a) for a in self.args) + ")"


class EGraph:
    """Hashconsed E-Graph with congruence closure and minimum-cost extraction.

    Union-find uses union-by-rank plus path halving, so deep rewrite chains stay
    near-constant time; ``rebuild()`` restores the upward-congruence invariant.
    """

    def __init__(self) -> None:
        self.parent: List[int] = []
        self.rank: List[int] = []
        self.eclasses: Dict[int, Set[ENode]] = {}
        self.memo: Dict[ENode, int] = {}
        self.worklist: List[int] = []

    # ── union-find ──────────────────────────────────────────────────────

    def find(self, eclass_id: int) -> int:
        """Canonical representative; unknown ids are returned unchanged."""
        try:
            root = int(eclass_id)
        except (TypeError, ValueError):
            return -1
        if root < 0 or root >= len(self.parent):
            return root
        while self.parent[root] != root:
            self.parent[root] = self.parent[self.parent[root]]
            root = self.parent[root]
        return root

    def _new_class(self) -> int:
        new_id = len(self.parent)
        self.parent.append(new_id)
        self.rank.append(0)
        self.eclasses[new_id] = set()
        return new_id

    def add_enode(self, node: ENode) -> int:
        if not isinstance(node, ENode):
            return -1
        canon = node.canonicalize(self.find)
        existing = self.memo.get(canon)
        if existing is not None:
            return self.find(existing)
        new_id = self._new_class()
        self.eclasses[new_id] = {canon}
        self.memo[canon] = new_id
        return new_id

    def add_term(self, term: TermAST) -> int:
        if not isinstance(term, TermAST):
            return -1
        child_ids = tuple(self.add_term(a) for a in term.args)
        return self.add_enode(ENode(term.op, child_ids))

    def union(self, id1: int, id2: int) -> int:
        """Merge two E-Classes (union by rank). Returns the representative."""
        r1, r2 = self.find(id1), self.find(id2)
        if r1 == r2:
            return r1
        if r1 < 0 or r2 < 0 or r1 >= len(self.parent) or r2 >= len(self.parent):
            return r1 if r1 >= 0 else r2
        if self.rank[r1] < self.rank[r2]:
            r1, r2 = r2, r1
        self.parent[r2] = r1
        if self.rank[r1] == self.rank[r2]:
            self.rank[r1] += 1
        self.eclasses.setdefault(r1, set()).update(self.eclasses.pop(r2, set()))
        self.worklist.append(r1)
        return r1

    # ── congruence closure ──────────────────────────────────────────────

    def rebuild(self) -> None:
        """Restore upward congruence closure invariant across all E-Classes until fixpoint."""
        for _ in range(10_000):
            merges = self._canonicalize_and_merge()
            self.worklist.clear()
            if not merges:
                break
        self._compact()

    def _canonicalize_and_merge(self) -> int:
        """Merge congruent parents; returns the number of merges performed."""
        merges = 0
        signature: Dict[ENode, int] = {}
        for cid in sorted(self.eclasses.keys()):
            root = self.find(cid)
            if root != cid:
                continue
            for node in sorted(self.eclasses.get(root, ()), key=ENode.sort_key):
                canon = node.canonicalize(self.find)
                previous = signature.get(canon)
                if previous is not None:
                    previous = self.find(previous)
                    if previous != self.find(root):
                        root = self.union(root, previous)
                        merges += 1
                signature[canon] = self.find(root)
        if merges:
            # Node sets were merged by union(); canonicalize them from scratch.
            memo: Dict[ENode, int] = {}
            rebuilt: Dict[int, Set[ENode]] = {}
            for cid in sorted(self.eclasses.keys()):
                root = self.find(cid)
                if root != cid:
                    continue
                members: Set[ENode] = set()
                for node in self.eclasses.get(root, ()):
                    canon = node.canonicalize(self.find)
                    members.add(canon)
                    memo[canon] = root
                rebuilt[root] = members
            self.eclasses = rebuilt
            self.memo = memo
            self._compact()
        return merges

    def _compact(self) -> None:
        """Drop non-representative keys and rebuild the hashcons memo."""
        compact: Dict[int, Set[ENode]] = {}
        memo: Dict[ENode, int] = {}
        for cid in sorted(self.eclasses.keys()):
            root = self.find(cid)
            members = compact.setdefault(root, set())
            for node in self.eclasses.get(cid, ()):
                canon = node.canonicalize(self.find)
                members.add(canon)
                memo[canon] = root
        self.eclasses = compact
        self.memo = memo

    def equivalent(self, id1: int, id2: int) -> bool:
        self.rebuild()
        return self.find(id1) == self.find(id2)

    # ── extraction ──────────────────────────────────────────────────────

    def extract_best(self, eclass_id: int,
                     op_costs: Optional[Dict[str, float]] = None) -> Tuple[float, TermAST]:
        """Extract minimum-cost TermAST from `eclass_id` via Bellman-Ford DP.

        Cyclic e-classes without a finite-cost base, unknown ids and malformed
        cost tables degrade to ``(inf, TermAST("?"))`` instead of raising.
        """
        self.rebuild()
        costs = op_costs if isinstance(op_costs, dict) else {}
        best_cost: Dict[int, float] = {}
        best_node: Dict[int, ENode] = {}

        changed = True
        while changed:
            changed = False
            for cid in sorted(self.eclasses.keys()):
                root = self.find(cid)
                for node in sorted(self.eclasses.get(root, ()), key=ENode.sort_key):
                    if not all(self.find(ch) in best_cost for ch in node.children):
                        continue
                    try:
                        op_cost = float(costs.get(node.op, 1.0))
                    except (TypeError, ValueError):
                        op_cost = 1.0
                    if op_cost != op_cost:  # NaN guard
                        op_cost = 1.0
                    c_val = op_cost + sum(best_cost[self.find(ch)] for ch in node.children)
                    if c_val < best_cost.get(root, float("inf")) - 1e-12:
                        best_cost[root] = c_val
                        best_node[root] = node
                        changed = True

        target = self.find(eclass_id)
        if target not in best_node:
            return float("inf"), TermAST("?")

        def build_ast(cid: int, seen: Set[int], depth: int) -> TermAST:
            r = self.find(cid)
            node = best_node.get(r)
            if node is None or r in seen or depth >= _MAX_DEPTH:
                return TermAST(node.op if node is not None else "?")
            return TermAST(node.op, tuple(
                build_ast(ch, seen | {r}, depth + 1) for ch in node.children))

        return best_cost[target], build_ast(target, set(), 0)


def _self_test() -> int:
    eg = EGraph()
    # Build terms: (mul x 2), (shl x 1), and (div (mul x 2) 2)
    x = eg.add_term(TermAST("x"))
    two = eg.add_term(TermAST("2"))
    one = eg.add_term(TermAST("1"))
    mul_x_2 = eg.add_enode(ENode("mul", (x, two)))
    shl_x_1 = eg.add_enode(ENode("shl", (x, one)))
    div_mul = eg.add_enode(ENode("div", (mul_x_2, two)))
    div_shl = eg.add_enode(ENode("div", (shl_x_1, two)))

    # Assert rewrite equivalence: (mul x 2) == (shl x 1)
    eg.union(mul_x_2, shl_x_1)
    # Upward congruence closure automatically proves (div (mul x 2) 2) == (div (shl x 1) 2)!
    assert eg.equivalent(div_mul, div_shl)

    # Assert algebraic simplification: (div (mul x 2) 2) == x
    eg.union(div_mul, x)
    cost, simplest = eg.extract_best(div_shl)
    assert cost == 1.0 and str(simplest) == "x", (cost, str(simplest))

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. congruence propagates through three nested levels on both sides
    g = EGraph()
    a = g.add_term(TermAST("a"))
    b = g.add_term(TermAST("b"))
    g.union(a, b)
    ga = g.add_enode(ENode("g", (a,)))
    gb = g.add_enode(ENode("g", (b,)))
    assert g.equivalent(ga, gb)
    fga = g.add_enode(ENode("f", (ga,)))
    fgb = g.add_enode(ENode("f", (gb,)))
    assert g.equivalent(fga, fgb)

    # 2. structural hashcons: adding the same term twice yields one class
    h = EGraph()
    t1 = h.add_term(TermAST("add", (TermAST("v"), TermAST("1"))))
    t2 = h.add_term(TermAST("add", (TermAST("v"), TermAST("1"))))
    assert t1 == t2 and len(h.eclasses) == 3

    # 3. union is idempotent and transitivity is preserved after rebuild
    before = len(h.eclasses)
    assert h.union(t1, t1) == t1
    h.add_term(TermAST("mul", (TermAST("v"), TermAST("2"))))
    assert len(h.eclasses) == before + 2
    h.rebuild()
    assert h.find(t1) == h.find(t2)

    # 4. parents become congruent only after their children are unioned
    k = EGraph()
    u = k.add_term(TermAST("u"))
    v = k.add_term(TermAST("v"))
    p = k.add_enode(ENode("p", (u,)))
    q = k.add_enode(ENode("p", (v,)))
    assert not k.equivalent(p, q)
    k.union(u, v)
    assert k.equivalent(p, q)

    # 5. custom operator costs steer extraction to the cheaper alternative
    m = EGraph()
    expensive = m.add_term(TermAST("expensive"))
    cheap = m.add_term(TermAST("cheap"))
    root = m.add_enode(ENode("root", (expensive, cheap)))
    m.union(cheap, root)
    cost_c, ast_c = m.extract_best(root, {"cheap": 1.0, "root": 50.0})
    assert cost_c == 1.0 and str(ast_c) == "cheap", (cost_c, str(ast_c))

    # 6. unknown / malformed requests fail soft, never raise
    inf_cost, fallback = k.extract_best(987654)
    assert inf_cost == float("inf") and str(fallback) == "?"
    assert k.extract_best(p, "not-a-cost-map")[0] >= 0.0  # type: ignore[arg-type]

    print("egraph._self_test: OK PASS (Union-Find hashconsing, upward congruence closure, min-cost AST extraction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
