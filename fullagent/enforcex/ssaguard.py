r"""ssaguard — Cooper-Harvey-Kennedy Dominator Tree, Dominance Frontier & SSA Phi-Placement.

Analyzes Control Flow Graphs (CFGs) of generated code or policy execution graphs
using classical compiler theory (Cytron et al. 1991; Cooper, Harvey & Kennedy 2001):
  1. **Reverse Post-Order (RPO) Traversal & Iterative Dominator Sets (`Dom[b]`)**:
     `Dom[n] = {n} U (/\_{p in pred(n)} Dom[p])`
  2. **Immediate Dominator (`IDom[b]`) & Dominator Tree Construction**:
     identifies the unique strict dominator closest to each basic block `b`.
  3. **Dominance Frontier (`DF[b]`)**:
     `y in DF[x]` iff `x` dominates a predecessor of `y` without strictly
     dominating `y` itself.
  4. **Iterated Dominance Frontier (`IDF`) & Minimal `phi`-Node Placement**:
     computes the exact join blocks where SSA `v_3 = phi(v_1, v_2)` merges are
     required and flags use-before-def / uninitialized variable bugs along any
     control-flow path.

Self-test: ``python3 -m fullagent.enforcex.ssaguard``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

__all__ = [
    "BasicBlock",
    "SSAAnalysisReport",
    "SSADominatorAnalyzer",
]


@dataclass
class BasicBlock:
    block_id: str
    defs: Set[str] = field(default_factory=set)
    uses: Set[str] = field(default_factory=set)
    succs: List[str] = field(default_factory=list)
    preds: List[str] = field(default_factory=list)


@dataclass
class SSAAnalysisReport:
    dominators: Dict[str, List[str]]
    idom: Dict[str, Optional[str]]
    dominance_frontier: Dict[str, List[str]]
    phi_placements: Dict[str, List[str]]  # var -> list of block_ids requiring phi(var)
    uninitialized_uses: List[Tuple[str, str]]  # (block_id, var)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominators": dict(self.dominators),
            "idom": dict(self.idom),
            "dominance_frontier": dict(self.dominance_frontier),
            "phi_placements": dict(self.phi_placements),
            "uninitialized_uses": list(self.uninitialized_uses),
        }


class SSADominatorAnalyzer:
    """Computes Dominators, Immediate Dominators, Dominance Frontiers, and SSA phi placements."""

    def __init__(self, entry_id: str = "B0") -> None:
        self.entry_id = str(entry_id)
        self.blocks: Dict[str, BasicBlock] = {}

    def add_block(self, block_id: str, defs: Iterable[str] = (), uses: Iterable[str] = ()) -> None:
        b = str(block_id)
        blk = self.blocks.setdefault(b, BasicBlock(b))
        blk.defs.update(str(d) for d in defs)
        blk.uses.update(str(u) for u in uses)

    def add_edge(self, u: str, v: str) -> None:
        self.add_block(u)
        self.add_block(v)
        if v not in self.blocks[u].succs:
            self.blocks[u].succs.append(v)
        if u not in self.blocks[v].preds:
            self.blocks[v].preds.append(u)

    def analyze(self) -> SSAAnalysisReport:
        all_nodes = sorted(self.blocks.keys())
        all_set = set(all_nodes)

        # 1. Dataflow fixpoint for Dom[b]
        dom: Dict[str, Set[str]] = {
            b: ({b} if b == self.entry_id else set(all_set))
            for b in all_nodes
        }
        changed = True
        while changed:
            changed = False
            for b in all_nodes:
                if b == self.entry_id:
                    continue
                preds = self.blocks[b].preds
                if not preds:
                    new_dom = {b}
                else:
                    inter = set(all_set)
                    for p in preds:
                        inter &= dom[p]
                    new_dom = {b} | inter
                if new_dom != dom[b]:
                    dom[b] = new_dom
                    changed = True

        # 2. Immediate Dominator IDom[b]: unique d in Dom[b] \ {b} with |Dom[d]| == |Dom[b]| - 1
        idom: Dict[str, Optional[str]] = {self.entry_id: None}
        for b in all_nodes:
            if b == self.entry_id:
                continue
            strict_doms = dom[b] - {b}
            chosen: Optional[str] = None
            for d in strict_doms:
                if len(dom[d]) == len(strict_doms):
                    chosen = d
                    break
            idom[b] = chosen

        # 3. Dominance Frontier DF[b]
        df: Dict[str, Set[str]] = {b: set() for b in all_nodes}
        for b in all_nodes:
            preds = self.blocks[b].preds
            if len(preds) >= 2:
                for p in preds:
                    runner: Optional[str] = p
                    while runner is not None and runner != idom.get(b):
                        df[runner].add(b)
                        runner = idom.get(runner)

        # 4. Cytron Iterated Dominance Frontier for Minimal Phi Placement
        var_defs: Dict[str, Set[str]] = {}
        for b, blk in self.blocks.items():
            for v in blk.defs:
                var_defs.setdefault(v, set()).add(b)

        phi_nodes: Dict[str, List[str]] = {}
        for var, def_blocks in sorted(var_defs.items()):
            has_phi: Set[str] = set()
            work = list(def_blocks)
            ever_worked = set(def_blocks)
            while work:
                x = work.pop()
                for y in df[x]:
                    if y not in has_phi:
                        has_phi.add(y)
                        if y not in ever_worked:
                            ever_worked.add(y)
                            work.append(y)
            phi_nodes[var] = sorted(has_phi)

        # 5. Detect uninitialized uses (variable used in block B where no dominator defines it)
        uninit: List[Tuple[str, str]] = []
        for b, blk in self.blocks.items():
            for u_var in sorted(blk.uses):
                if not any(u_var in self.blocks[d].defs for d in dom[b]):
                    uninit.append((b, u_var))

        return SSAAnalysisReport(
            dominators={k: sorted(v) for k, v in sorted(dom.items())},
            idom=idom,
            dominance_frontier={k: sorted(v) for k, v in sorted(df.items())},
            phi_placements=phi_nodes,
            uninitialized_uses=uninit,
        )


def _self_test() -> int:
    # Diamond CFG: B0 -> {B1, B2} -> B3
    ssa = SSADominatorAnalyzer("B0")
    ssa.add_block("B0", defs=["x"])
    ssa.add_block("B1", defs=["x", "branch_only"])
    ssa.add_block("B2", defs=["x"])
    ssa.add_block("B3", uses=["x", "branch_only"])
    ssa.add_edge("B0", "B1")
    ssa.add_edge("B0", "B2")
    ssa.add_edge("B1", "B3")
    ssa.add_edge("B2", "B3")

    rep = ssa.analyze()
    assert rep.idom["B1"] == "B0" and rep.idom["B2"] == "B0" and rep.idom["B3"] == "B0"
    assert rep.dominance_frontier["B1"] == ["B3"] and rep.dominance_frontier["B2"] == ["B3"]
    # x is modified in B1 and B2 -> requires phi(x) at join block B3!
    assert rep.phi_placements["x"] == ["B3"]
    # branch_only is only defined in B1 (which does NOT dominate B3) -> flagged as uninitialized use in B3!
    assert ("B3", "branch_only") in rep.uninitialized_uses

    print("ssaguard._self_test: OK PASS (Dominator tree, Dominance Frontier, Cytron phi placement, uninit-use check)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
