"""spectralgraph — Normalized Graph Laplacian, Fiedler Vector (lambda_2) & Cheeger Cut.

Analyzes the structural connectivity of discourse/rule/entity graphs `G = (V, E, w)`
via Spectral Graph Theory (Chung 1997):
  * **Symmetric Normalized Laplacian**: `L_sym = I - D^{-1/2} A D^{-1/2}`, whose
    eigenvalues satisfy `0 = lambda_1 <= lambda_2 <= ... <= lambda_n <= 2`.
  * **Ground-State Deflation & Fiedler Pair (`lambda_2, v_2`)**:
    projects out the trivial stationary eigenvector `u_1_i = sqrt(deg(i))` and
    runs inverse/shifted power iteration on `(2I - L_sym)` to extract the
    **Fiedler Value** `lambda_2` (Algebraic Connectivity) and **Fiedler Vector**.
  * **Sweep-Cut Cheeger Partition (`phi(S)`)**:
    sweeps prefix cuts ordered by the Fiedler vector coordinates `v_2[i] / sqrt(deg(i))`
    to find the minimum-conductance bipartition `phi(G) = cut(S, V\\S) / min(vol(S), vol(V\\S))`,
    satisfying Cheeger's Inequality `lambda_2 / 2 <= phi(G) <= sqrt(2 * lambda_2)`.

Self-test: ``python3 -m fullagent.enforcex.spectralgraph``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

__all__ = [
    "SpectralCutReport",
    "SpectralGraphAnalyzer",
]


@dataclass
class SpectralCutReport:
    fiedler_lambda2: float
    cheeger_conductance: float
    cheeger_lower_bound: float
    cheeger_upper_bound: float
    cluster_a: List[int] = field(default_factory=list)
    cluster_b: List[int] = field(default_factory=list)
    fiedler_vector: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fiedler_lambda2": round(self.fiedler_lambda2, 4),
            "cheeger_conductance": round(self.cheeger_conductance, 4),
            "cheeger_lower_bound": round(self.cheeger_lower_bound, 4),
            "cheeger_upper_bound": round(self.cheeger_upper_bound, 4),
            "cluster_a": list(self.cluster_a),
            "cluster_b": list(self.cluster_b),
            "fiedler_vector": [round(x, 4) for x in self.fiedler_vector],
        }


class SpectralGraphAnalyzer:
    """Computes Normalized Laplacian Fiedler eigenvalue `lambda_2` and Cheeger sweep cut."""

    def analyze(self, adj: Sequence[Sequence[float]], iterations: int = 80) -> SpectralCutReport:
        n = len(adj)
        if n < 2:
            return SpectralCutReport(0.0, 0.0, 0.0, 0.0, list(range(n)), [], [1.0] * n)

        deg = [sum(max(0.0, float(adj[i][j])) for j in range(n)) for i in range(n)]
        inv_sqrt_d = [1.0 / math.sqrt(d) if d > 1e-12 else 0.0 for d in deg]

        # Normalized Laplacian L_sym[i][j] = delta_{ij} - A[i][j] / sqrt(d_i d_j)
        l_sym = [[0.0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            l_sym[i][i] = 1.0 if deg[i] > 1e-12 else 0.0
            for j in range(n):
                w = max(0.0, float(adj[i][j]))
                if w > 0.0:
                    l_sym[i][j] -= w * inv_sqrt_d[i] * inv_sqrt_d[j]

        # Trivial eigenvector u0 of L_sym with eigenvalue 0: u0[i] = sqrt(d_i) / sqrt(vol(G))
        u0 = [math.sqrt(max(0.0, d)) for d in deg]
        u0_norm = math.sqrt(sum(x * x for x in u0)) or 1.0
        u0 = [x / u0_norm for x in u0]

        # Power iteration on M = 2*I - L_sym orthogonalized against u0 -> largest eigenvector of M is lambda_2 of L_sym!
        v = [math.sin(i + 1.0) for i in range(n)]
        for _ in range(max(20, iterations)):
            # Project out u0
            dot0 = sum(v[i] * u0[i] for i in range(n))
            v = [v[i] - dot0 * u0[i] for i in range(n)]
            # Multiply by (2I - L_sym)
            w_vec = [0.0 for _ in range(n)]
            for i in range(n):
                lv = sum(l_sym[i][j] * v[j] for j in range(n))
                w_vec[i] = 2.0 * v[i] - lv
            dot0_w = sum(w_vec[i] * u0[i] for i in range(n))
            w_vec = [w_vec[i] - dot0_w * u0[i] for i in range(n)]
            norm = math.sqrt(sum(x * x for x in w_vec)) or 1.0
            v = [x / norm for x in w_vec]

        # Rayleigh quotient v^T L_sym v
        lv_final = [sum(l_sym[i][j] * v[j] for j in range(n)) for i in range(n)]
        lambda2 = max(0.0, sum(v[i] * lv_final[i] for i in range(n)))

        # Fiedler sweep cut along embedding coordinates f[i] = v[i] / sqrt(d_i)
        embedding = [(v[i] * inv_sqrt_d[i], i) for i in range(n)]
        embedding.sort(key=lambda pair: pair[0])
        order = [idx for _, idx in embedding]

        total_vol = sum(deg)
        best_phi = float("inf")
        best_k = 1
        s_set = set()
        vol_s = 0.0
        cut_val = 0.0

        for k in range(1, n):
            u = order[k - 1]
            s_set.add(u)
            vol_s += deg[u]
            for j in range(n):
                w = max(0.0, float(adj[u][j]))
                if w > 0:
                    if j in s_set:
                        cut_val -= w
                    else:
                        cut_val += w
            denom = min(vol_s, total_vol - vol_s)
            if denom > 1e-12:
                phi = cut_val / denom
                if phi < best_phi:
                    best_phi = phi
                    best_k = k

        cluster_a = sorted(order[:best_k])
        cluster_b = sorted(order[best_k:])
        if best_phi == float("inf"):
            best_phi = 0.0

        return SpectralCutReport(
            fiedler_lambda2=lambda2,
            cheeger_conductance=best_phi,
            cheeger_lower_bound=0.5 * lambda2,
            cheeger_upper_bound=math.sqrt(2.0 * lambda2),
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            fiedler_vector=v,
        )


def _self_test() -> int:
    # Two tightly connected triangles {0,1,2} and {3,4,5} joined by a single weak bridge (2, 3) of weight 0.1
    adj = [
        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    ]
    rep = SpectralGraphAnalyzer().analyze(adj)
    # Fiedler sweep cut cleanly recovers the two triangles {0,1,2} and {3,4,5}!
    assert {tuple(rep.cluster_a), tuple(rep.cluster_b)} == {(0, 1, 2), (3, 4, 5)}, rep.to_dict()
    # Cheeger inequality holds: lambda_2 / 2 <= phi(G) <= sqrt(2 * lambda_2)
    assert rep.cheeger_lower_bound - 1e-4 <= rep.cheeger_conductance <= rep.cheeger_upper_bound + 1e-4

    print("spectralgraph._self_test: OK PASS (Normalized Laplacian Fiedler vector, Cheeger cut & inequality)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
