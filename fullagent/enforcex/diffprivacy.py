"""diffprivacy — (epsilon, delta)-Differential Privacy, Renyi DP Accountant & Mechanisms.

Provides mathematically rigorous Differential Privacy primitives (Dwork & Roth
2014; Mironov 2017 Rényi Differential Privacy) for sanitizing aggregate
telemetry, audit metrics, and prompt statistics:
  1. **Laplace Mechanism (`Lap(Delta_1 / epsilon)`)**: pure `(epsilon, 0)`-DP
     via inverse-CDF sampling with `os.urandom` cryptographic randomness.
  2. **Analytic Gaussian Mechanism (`N(0, sigma^2)`)**: `(epsilon, delta)`-DP
     with `sigma = Delta_2 * sqrt(2 * ln(1.25 / delta)) / epsilon`.
  3. **Exponential Mechanism (`exp(epsilon * u(x, r) / (2 * Delta_u))`)**:
     selects discrete candidates privately with probability proportional to utility.
  4. **Rényi Differential Privacy (RDP) Composition & Conversion to `(epsilon, delta)`-DP**:
     tracks cumulative RDP `epsilon_RDP(alpha)` across queries at orders
     `alpha in {2, 4, 8, 16, 32, 64}` and converts via Mironov Proposition 3:
     `epsilon(delta) = min_{alpha > 1} (epsilon_RDP(alpha) + ln(1 / delta) / (alpha - 1))`.

Self-test: ``python3 -m fullagent.enforcex.diffprivacy``
"""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PrivacyBudgetState",
    "DifferentialPrivacyAccountant",
    "laplace_noise",
    "gaussian_noise",
    "exponential_mechanism_probs",
]


def _crypto_uniform_open01() -> float:
    """Cryptographically secure uniform float in `(0, 1)` from `os.urandom`."""
    raw = struct.unpack(">Q", os.urandom(8))[0]
    return (raw + 1.0) / float((1 << 64) + 2.0)


def _finite(value: Any, fallback: float) -> float:
    """Finite float with fallback; never raises."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(f) or math.isinf(f):
        return fallback
    return f


def laplace_noise(scale_b: float) -> float:
    """Sample from `Laplace(0, b)` using exact inverse transform `F^{-1}(u)`."""
    b = max(1e-12, _finite(scale_b, 1e-12))
    u = _crypto_uniform_open01() - 0.5
    sign = 1.0 if u >= 0.0 else -1.0
    tail = max(1e-15, 1.0 - 2.0 * abs(u))
    return -b * sign * math.log(tail)


def gaussian_noise(sigma: float) -> float:
    """Sample from `N(0, sigma^2)` via Box-Muller transform."""
    s = max(1e-12, _finite(sigma, 1e-12))
    u1 = _crypto_uniform_open01()
    u2 = _crypto_uniform_open01()
    z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return s * z0


def exponential_mechanism_probs(
    scores: Sequence[float],
    *,
    epsilon: float = 1.0,
    sensitivity: float = 1.0,
) -> List[float]:
    """Compute exact selection probabilities for the McSherry-Talwar Exponential Mechanism.

    Non-finite scores are treated as zero utility; exponentiation is shifted by
    the maximum so ``exp`` never overflows.
    """
    if not scores:
        return []
    try:
        score_list = [float(u) for u in scores]
    except (TypeError, ValueError):
        return []
    clean = [s if math.isfinite(s) else 0.0 for s in score_list]
    eps = max(1e-6, _finite(epsilon, 1e-6))
    sens = max(1e-6, _finite(sensitivity, 1e-6))
    scaled = [(eps * u) / (2.0 * sens) for u in clean]
    m = max(scaled) if scaled else 0.0
    weights = [math.exp(x - m) for x in scaled]
    tot = sum(weights)
    if not math.isfinite(tot) or tot <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [w / tot for w in weights]



@dataclass
class PrivacyBudgetState:
    queries_count: int
    basic_epsilon: float
    rdp_converted_epsilon: float
    delta: float
    within_budget: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queries_count": self.queries_count,
            "basic_epsilon": round(self.basic_epsilon, 4),
            "rdp_converted_epsilon": round(self.rdp_converted_epsilon, 4),
            "delta": self.delta,
            "within_budget": self.within_budget,
        }


class DifferentialPrivacyAccountant:
    """Tracks cumulative privacy loss using Rényi Differential Privacy (RDP) orders."""

    ORDERS: Tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 8.0, 16.0, 32.0, 64.0)

    def __init__(self, max_epsilon: float = 5.0, target_delta: float = 1e-5) -> None:
        self.max_epsilon = max(0.0, _finite(max_epsilon, 5.0))
        self.target_delta = max(1e-12, _finite(target_delta, 1e-5))
        self.rdp_totals: Dict[float, float] = {a: 0.0 for a in self.ORDERS}
        self.basic_epsilon = 0.0
        self.laplace_epsilon = 0.0
        self.queries_count = 0

    def record_gaussian_query(self, sensitivity_l2: float, sigma: float) -> float:
        """Gaussian mechanism with noise `sigma` satisfies `(alpha, alpha * Delta_2^2 / (2 * sigma^2))`-RDP.

        Invalid (NaN/inf/zero-noise) parameters are ignored rather than
        poisoning the accountant.
        """
        sens = _finite(sensitivity_l2, -1.0)
        s = _finite(sigma, -1.0)
        if sens < 0.0 or s <= 0.0:
            return self.effective_epsilon()
        self.queries_count += 1
        d2 = sens * sens
        s2 = s * s
        for alpha in self.ORDERS:
            self.rdp_totals[alpha] += (alpha * d2) / (2.0 * s2)
        eps_single = sens * math.sqrt(2.0 * math.log(1.25 / self.target_delta)) / s
        self.basic_epsilon += eps_single
        return self.effective_epsilon()

    def record_laplace_query(self, epsilon_per_query: float = 1.0) -> float:
        """Compose a pure `(epsilon, 0)`-DP Laplace query by basic composition."""
        eps = _finite(epsilon_per_query, -1.0)
        if eps < 0.0:
            return self.effective_epsilon()
        self.queries_count += 1
        self.laplace_epsilon += eps
        self.basic_epsilon += eps
        return self.effective_epsilon()

    def effective_epsilon(self, delta: Optional[float] = None) -> float:
        """Convert accumulated RDP to `(epsilon, delta)`-DP (Mironov Prop. 3).

        Basic composition is used only for a single Gaussian query (where its
        per-query delta is the target delta) or for pure-DP Laplace queries;
        for multiple Gaussian queries the RDP conversion is the sound bound.
        """
        d = max(1e-15, _finite(delta, self.target_delta))
        if self.queries_count == 0:
            return 0.0
        best = float("inf")
        for alpha in self.ORDERS:
            eps_a = self.rdp_totals[alpha] + math.log(1.0 / d) / (alpha - 1.0)
            if eps_a < best:
                best = eps_a
        if self.laplace_epsilon > 0.0:
            # Basic composition of pure DP with the RDP-converted Gaussian part.
            return max(0.0, best + self.laplace_epsilon)
        gaussian_queries = self.queries_count
        if gaussian_queries == 1:
            best = min(best, self.basic_epsilon)
        return max(0.0, best)

    def state(self) -> PrivacyBudgetState:
        eff = self.effective_epsilon()
        return PrivacyBudgetState(
            queries_count=self.queries_count,
            basic_epsilon=self.basic_epsilon,
            rdp_converted_epsilon=eff,
            delta=self.target_delta,
            within_budget=(eff <= self.max_epsilon),
        )


def _self_test() -> int:
    acc = DifferentialPrivacyAccountant(max_epsilon=10.0, target_delta=1e-5)
    for _ in range(10):
        acc.record_gaussian_query(sensitivity_l2=1.0, sigma=5.0)
    st = acc.state()
    # RDP sub-linear composition is strictly tighter than naive basic composition!
    assert st.rdp_converted_epsilon < st.basic_epsilon, st.to_dict()
    assert st.within_budget

    probs = exponential_mechanism_probs([10.0, 20.0, 50.0], epsilon=2.0, sensitivity=5.0)
    assert abs(sum(probs) - 1.0) < 1e-6 and probs[2] > probs[1] > probs[0]

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. mechanisms stay finite for junk scales (no NaN / inf propagation)
    for junk in (float("nan"), float("inf"), float("-inf"), "junk", None):
        assert math.isfinite(laplace_noise(junk))  # type: ignore[arg-type]
        assert math.isfinite(gaussian_noise(junk))  # type: ignore[arg-type]
    assert abs(laplace_noise(0.0)) < 1e-9
    assert abs(gaussian_noise(0.0)) < 1e-9

    # 2. exponential mechanism cannot overflow or return NaN probabilities
    extreme = exponential_mechanism_probs(
        [1e308, -1e308, float("nan"), float("inf")], epsilon=5.0, sensitivity=1.0)
    assert abs(sum(extreme) - 1.0) < 1e-9
    assert all(math.isfinite(p) and p >= 0.0 for p in extreme)
    assert exponential_mechanism_probs([]) == []
    assert abs(sum(exponential_mechanism_probs([1, 1])) - 1.0) < 1e-12

    # 3. invalid Gaussian parameters are ignored, not accounted as privacy loss
    guard = DifferentialPrivacyAccountant(max_epsilon=1.0)
    assert guard.record_gaussian_query(1.0, 0.0) == 0.0
    assert guard.record_gaussian_query(float("nan"), 5.0) == 0.0
    assert guard.record_gaussian_query(1.0, float("inf")) == 0.0
    assert guard.queries_count == 0 and guard.state().within_budget

    # 4. more noise always buys less epsilon; composition grows monotonically
    loud = DifferentialPrivacyAccountant(max_epsilon=100.0)
    loud.record_gaussian_query(1.0, 10.0)
    quiet = DifferentialPrivacyAccountant(max_epsilon=100.0)
    quiet.record_gaussian_query(1.0, 1.0)
    assert quiet.effective_epsilon() > loud.effective_epsilon()
    before_eps = loud.effective_epsilon()
    loud.record_gaussian_query(1.0, 10.0)
    assert loud.effective_epsilon() > before_eps

    # 5. a single analytic-Gaussian query may use the tighter basic bound
    single = DifferentialPrivacyAccountant(target_delta=1e-5)
    single.record_gaussian_query(1.0, 5.0)
    st_single = single.state()
    assert abs(st_single.rdp_converted_epsilon - round(single.basic_epsilon, 4)) < 1e-4

    # 6. pure-DP Laplace queries compose additively and are counted
    lap = DifferentialPrivacyAccountant(max_epsilon=5.0)
    lap.record_laplace_query(0.4)
    lap.record_laplace_query(0.4)
    assert lap.queries_count == 2 and abs(lap.basic_epsilon - 0.8) < 1e-9
    assert lap.effective_epsilon() >= 0.8

    # 7. a tighter delta can only increase the reported epsilon
    d_acc = DifferentialPrivacyAccountant(target_delta=1e-5)
    for _ in range(3):
        d_acc.record_gaussian_query(1.0, 4.0)
    assert d_acc.effective_epsilon(1e-9) > d_acc.effective_epsilon(1e-3)

    print("diffprivacy._self_test: OK PASS (Laplace/Gaussian/Exponential mechanisms + Renyi DP composition)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
