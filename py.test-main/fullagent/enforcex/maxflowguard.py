"""maxflowguard — Dinic's Level-Graph Blocking-Flow & Min-Cut Exfiltration Bottleneck Analyzer.

Models multi-stage agent/subagent/tool information pipelines as a capacitated
directed network `G = (V, E, c)` and computes:
  1. **Dinic's $O(V^2 E)$ Maximum Flow**:
     - Constructs layered BFS level graph `level[v] = level[u] + 1` on the
       residual network `G_f`.
     - Pushes blocking flows via DFS with current-arc pointer optimization
       (`ptr[u]`) so dead-end arcs are never revisited in the same phase.
  2. **Max-Flow Min-Cut Theorem (`(S, T)` partition & cut edges)**:
     - Identifies the exact minimum-capacity edge cut separating untrusted/secret
       sources `s` from external egress sinks `t` so the minimum number of
       chokepoint gates can throttle or zero out exfiltration bandwidth.

Self-test: ``python3 -m fullagent.enforcex.maxflowguard``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = [
    "FlowEdge",
    "MinCutReport",
    "DinicMaxFlowNetwork",
]


@dataclass
class FlowEdge:
    u: int
    v: int
    cap: float
    flow: float = 0.0
    rev_idx: int = 0


@dataclass
class MinCutReport:
    max_flow: float
    source_side_nodes: List[int] = field(default_factory=list)
    sink_side_nodes: List[int] = field(default_factory=list)
    min_cut_edges: List[Tuple[int, int, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_flow": round(self.max_flow, 4),
            "source_side_nodes": list(self.source_side_nodes),
            "sink_side_nodes": list(self.sink_side_nodes),
            "min_cut_edges": [(u, v, round(c, 4)) for u, v, c in self.min_cut_edges],
        }


class DinicMaxFlowNetwork:
    """Dinic's blocking-flow algorithm with current-arc optimization and min-cut extraction."""

    def __init__(self, num_nodes: int) -> None:
        self.n = max(1, int(num_nodes))
        self.adj: List[List[FlowEdge]] = [[] for _ in range(self.n)]
        self.level: List[int] = [-1 for _ in range(self.n)]
        self.ptr: List[int] = [0 for _ in range(self.n)]

    def add_edge(self, u: int, v: int, capacity: float) -> None:
        fwd = FlowEdge(u=u, v=v, cap=float(capacity), flow=0.0, rev_idx=len(self.adj[v]))
        bwd = FlowEdge(u=v, v=u, cap=0.0, flow=0.0, rev_idx=len(self.adj[u]))
        self.adj[u].append(fwd)
        self.adj[v].append(bwd)

    def _bfs_level_graph(self, s: int, t: int) -> bool:
        self.level = [-1 for _ in range(self.n)]
        self.level[s] = 0
        q: deque[int] = deque([s])
        while q:
            u = q.popleft()
            for e in self.adj[u]:
                if e.cap - e.flow > 1e-12 and self.level[e.v] == -1:
                    self.level[e.v] = self.level[u] + 1
                    q.append(e.v)
        return self.level[t] != -1

    def _dfs_blocking_flow(self, u: int, t: int, pushed: float) -> float:
        if pushed <= 1e-12 or u == t:
            return pushed
        while self.ptr[u] < len(self.adj[u]):
            cid = self.ptr[u]
            e = self.adj[u][cid]
            rem = e.cap - e.flow
            if self.level[u] + 1 == self.level[e.v] and rem > 1e-12:
                tr = self._dfs_blocking_flow(e.v, t, min(pushed, rem))
                if tr > 1e-12:
                    e.flow += tr
                    self.adj[e.v][e.rev_idx].flow -= tr
                    return tr
            self.ptr[u] += 1
        return 0.0

    def compute_min_cut(self, source: int, sink: int) -> MinCutReport:
        total_flow = 0.0
        while self._bfs_level_graph(source, sink):
            self.ptr = [0 for _ in range(self.n)]
            while True:
                pushed = self._dfs_blocking_flow(source, sink, float("inf"))
                if pushed <= 1e-12:
                    break
                total_flow += pushed

        # Reachable vertices in the final residual graph form S; complement forms T
        s_side = [i for i in range(self.n) if self.level[i] != -1]
        s_set = set(s_side)
        t_side = [i for i in range(self.n) if i not in s_set]
        cut_edges: List[Tuple[int, int, float]] = []
        for u in s_side:
            for e in self.adj[u]:
                if e.cap > 0 and e.v not in s_set:
                    cut_edges.append((u, e.v, e.cap))

        return MinCutReport(
            max_flow=total_flow,
            source_side_nodes=s_side,
            sink_side_nodes=t_side,
            min_cut_edges=cut_edges,
        )


def _self_test() -> int:
    # 6-node pipeline: 0 (secret vault) -> {1, 2} (subagents) -> {3, 4} (formatters) -> 5 (external sink)
    net = DinicMaxFlowNetwork(6)
    net.add_edge(0, 1, 16.0)
    net.add_edge(0, 2, 13.0)
    net.add_edge(1, 2, 10.0)
    net.add_edge(1, 3, 12.0)
    net.add_edge(2, 1, 4.0)
    net.add_edge(2, 4, 14.0)
    net.add_edge(3, 2, 9.0)
    net.add_edge(3, 5, 20.0)
    net.add_edge(4, 3, 7.0)
    net.add_edge(4, 5, 4.0)

    rep = net.compute_min_cut(0, 5)
    # Classic CLRS max-flow network: max_flow == 23.0, and sum of min-cut edge capacities == 12 + 7 + 4 == 23.0!
    cut_cap_sum = sum(c for _, _, c in rep.min_cut_edges)
    assert abs(rep.max_flow - 23.0) < 1e-6 and abs(cut_cap_sum - 23.0) < 1e-6, rep.to_dict()

    print("maxflowguard._self_test: OK PASS (Dinic level-graph blocking flow + Max-Flow Min-Cut duality=23.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
