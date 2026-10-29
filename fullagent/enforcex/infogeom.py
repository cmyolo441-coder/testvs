"""infogeom — Fisher-Rao Geodesic Distance, Bhattacharyya Metric & Sinkhorn Optimal Transport.

Measures geometric shift between probability distributions on the statistical
simplex `Delta^{n-1}` (Amari, "Information Geometry and Its Applications";
Cuturi 2013, "Sinkhorn Distances"):
  1. **Square-Root Hellinger Embedding (`sqrt(p_i)` on the positive orthant of
     the unit sphere `S^{n-1}_+`)**:
     The Fisher information metric `g_{ij}(p) = delta_{ij} / p_i` pulls back to
     4 times the standard spherical metric on `S^{n-1}_+`.
  2. **Exact Fisher-Rao Geodesic Distance**:
     `d_FR(p, q) = 2 * arccos( sum_i sqrt(p_i * q_i) ) in [0, pi]`
  3. **Bhattacharyya Coefficient (`BC(p, q) = sum_i sqrt(p_i * q_i)`) & Distance
     (`D_B = -ln BC(p, q)`)**.
  4. **Geodesic Interpolation (`gamma(t)` for `t in [0, 1]`)** along the Fisher-Rao
     Riemannian manifold via spherical linear interpolation (`slerp`) of `sqrt(p)`.
  5. **Entropic Sinkhorn-Knopp Optimal Transport (`W_epsilon(p, q)`)**:
     iteratively scales `u, v` on Gibbs kernel `K = exp(-C / epsilon)` so
     `diag(u) K diag(v)` has marginals `p` and `q`.

Self-test: ``python3 -m fullagent.enforcex.infogeom``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "InformationGeometryReport",
    "InformationGeometryEngine",
    "normalize_simplex",
    "bhattacharyya_coefficient",
    "fisher_rao_distance",
    "fisher_rao_geodesic",
    "sinkhorn_optimal_transport",
]


def normalize_simplex(weights: Sequence[float], eps: float = 1e-12) -> List[float]:
    cleaned = [max(eps, float(w)) for w in (weights or (1.0,))]
    tot = sum(cleaned)
    return [x / tot for x in cleaned]


def bhattacharyya_coefficient(p: Sequence[float], q: Sequence[float]) -> float:
    pn = normalize_simplex(p)
    qn = normalize_simplex(q)
    k = min(len(pn), len(qn))
    bc = sum(math.sqrt(pn[i] * qn[i]) for i in range(k))
    return max(0.0, min(1.0, bc))


def fisher_rao_distance(p: Sequence[float], q: Sequence[float]) -> float:
    """Exact Riemannian geodesic distance under the Fisher Information Metric on `Delta^{n-1}`."""
    bc = bhattacharyya_coefficient(p, q)
    return 2.0 * math.acos(max(-1.0, min(1.0, bc)))


def fisher_rao_geodesic(p: Sequence[float], q: Sequence[float], t: float = 0.5) -> List[float]:
    """Compute point `gamma(t)` at fraction `t in [0, 1]` along the Fisher-Rao geodesic."""
    pn = normalize_simplex(p)
    qn = normalize_simplex(q)
    k = min(len(pn), len(qn))
    tau = max(0.0, min(1.0, float(t)))
    u = [math.sqrt(pn[i]) for i in range(k)]
    v = [math.sqrt(qn[i]) for i in range(k)]
    cos_omega = max(-1.0, min(1.0, sum(u[i] * v[i] for i in range(k))))
    omega = math.acos(cos_omega)
    if omega < 1e-8:
        interp = [(1.0 - tau) * pn[i] + tau * qn[i] for i in range(k)]
        return normalize_simplex(interp)
    sin_omega = math.sin(omega)
    a = math.sin((1.0 - tau) * omega) / sin_omega
    b = math.sin(tau * omega) / sin_omega
    gamma = [((a * u[i] + b * v[i]) ** 2) for i in range(k)]
    return normalize_simplex(gamma)


def sinkhorn_optimal_transport(
    p: Sequence[float],
    q: Sequence[float],
    cost_matrix: Sequence[Sequence[float]],
    *,
    epsilon: float = 0.1,
    iterations: int = 60,
) -> Tuple[float, List[List[float]]]:
    """Entropic-regularized Optimal Transport via Sinkhorn-Knopp matrix scaling."""
    pn = normalize_simplex(p)
    qn = normalize_simplex(q)
    n = len(pn)
    m = len(qn)
    eps = max(1e-4, float(epsilon))
    k_mat = [
        [math.exp(-float(cost_matrix[i][j]) / eps) for j in range(m)]
        for i in range(n)
    ]
    u = [1.0 / n for _ in range(n)]
    v = [1.0 / m for _ in range(m)]

    for _ in range(max(10, iterations)):
        # u = p ./ (K v)
        for i in range(n):
            kv = sum(k_mat[i][j] * v[j] for j in range(m))
            u[i] = pn[i] / max(1e-18, kv)
        # v = q ./ (K^T u)
        for j in range(m):
            ktu = sum(k_mat[i][j] * u[i] for i in range(n))
            v[j] = qn[j] / max(1e-18, ktu)

    plan = [[u[i] * k_mat[i][j] * v[j] for j in range(m)] for i in range(n)]
    ot_cost = sum(plan[i][j] * float(cost_matrix[i][j]) for i in range(n) for j in range(m))
    return ot_cost, plan


@dataclass
class InformationGeometryReport:
    fisher_rao_rad: float
    bhattacharyya_coeff: float
    bhattacharyya_dist: float
    sinkhorn_cost: float
    midpoint_distribution: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fisher_rao_rad": round(self.fisher_rao_rad, 4),
            "bhattacharyya_coeff": round(self.bhattacharyya_coeff, 4),
            "bhattacharyya_dist": round(self.bhattacharyya_dist, 4),
            "sinkhorn_cost": round(self.sinkhorn_cost, 4),
            "midpoint_distribution": [round(x, 4) for x in self.midpoint_distribution],
        }


class InformationGeometryEngine:
    def compare_distributions(self, p: Sequence[float], q: Sequence[float]) -> InformationGeometryReport:
        pn = normalize_simplex(p)
        qn = normalize_simplex(q)
        n = min(len(pn), len(qn))
        cost = [[abs(i - j) / max(1.0, float(n - 1)) for j in range(n)] for i in range(n)]
        bc = bhattacharyya_coefficient(pn[:n], qn[:n])
        fr = fisher_rao_distance(pn[:n], qn[:n])
        bd = -math.log(max(1e-12, bc))
        ot_c, _ = sinkhorn_optimal_transport(pn[:n], qn[:n], cost)
        mid = fisher_rao_geodesic(pn[:n], qn[:n], 0.5)
        return InformationGeometryReport(
            fisher_rao_rad=fr,
            bhattacharyya_coeff=bc,
            bhattacharyya_dist=bd,
            sinkhorn_cost=ot_c,
            midpoint_distribution=mid,
        )


def _self_test() -> int:
    eng = InformationGeometryEngine()
    p = [0.80, 0.15, 0.05]
    q = [0.05, 0.15, 0.80]
    rep = eng.compare_distributions(p, q)
    # Verify geodesic midpoint bisects Fisher-Rao distance: d(p, mid) == d(mid, q) == 0.5 * d(p, q)
    d_p_mid = fisher_rao_distance(p, rep.midpoint_distribution)
    d_mid_q = fisher_rao_distance(rep.midpoint_distribution, q)
    assert abs(d_p_mid - d_mid_q) < 1e-6 and abs((d_p_mid + d_mid_q) - rep.fisher_rao_rad) < 1e-6
    assert rep.sinkhorn_cost > 0.5

    print("infogeom._self_test: OK PASS (Fisher-Rao geodesic midpoint bisection + Sinkhorn-Knopp optimal transport)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
