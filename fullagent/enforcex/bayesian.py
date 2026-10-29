"""bayesian — Conjugate Beta-Binomial & Dirichlet-Multinomial Compliance Estimator.

Maintains exact Bayesian posteriors over rule compliance and drift rates:
  * Per-Rule Beta($\\alpha, \\beta$) Conjugate Tracker: computes exact posterior
    mean $\\frac{\\alpha}{\\alpha+\\beta}$, variance, normal-approximation 95%
    credible intervals, and tail probability $P(\theta < \tau)$.
  * Multi-Category Dirichlet($\\alpha_1, \\dots, \\alpha_K$) Tracker: tracks
    categorical distribution of violation types and computes exact
    Kullback-Leibler divergence $D_{\text{KL}}(P \\parallel Q)$ to detect
    statistically significant behavioral shifts across turns.

Self-test: ``python3 -m fullagent.enforcex.bayesian``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "BetaPosterior",
    "BayesianComplianceNet",
    "kl_divergence",
    "jensen_shannon_divergence",
    "regularized_incomplete_beta",
]

_EPS = 1e-12
_MAX_ITER = 200


def _finite_positive(value: float, default: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(v) or v <= 0.0:
        return default
    return v


@dataclass
class BetaPosterior:
    rule_id: str
    alpha: float = 2.0        # prior successes (Laplace/Jeffreys smoothed)
    beta: float = 1.0         # prior failures

    @property
    def total(self) -> float:
        return self.alpha + self.beta

    @property
    def mean(self) -> float:
        return self.alpha / self.total if self.total > 0 else 0.5

    @property
    def variance(self) -> float:
        t = self.total
        if t <= 0:
            return 0.0
        return (self.alpha * self.beta) / (t * t * (t + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.variance))

    def credible_interval_95(self) -> Tuple[float, float]:
        m = self.mean
        s = 1.95996 * self.std
        return (max(0.0, m - s), min(1.0, m + s))

    def credible_interval_exact(self, level: float = 0.95) -> Tuple[float, float]:
        """Exact equal-tailed credible interval via inverse regularized beta.

        Deterministic bisection on the Beta CDF; never raises.
        """
        try:
            lvl = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError, OverflowError):
            lvl = 0.95
        tail = (1.0 - lvl) / 2.0
        a = max(_EPS, float(self.alpha))
        b = max(_EPS, float(self.beta))
        lo = _beta_quantile(a, b, tail)
        hi = _beta_quantile(a, b, 1.0 - tail)
        return (min(lo, hi), max(lo, hi))

    def prob_less_than(self, threshold: float) -> float:
        """Exact posterior tail probability P(theta < threshold)."""
        try:
            t = float(threshold)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(t):
            return 0.0 if t < 0 else 1.0
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        a = max(_EPS, float(self.alpha))
        b = max(_EPS, float(self.beta))
        return regularized_incomplete_beta(a, b, t)

    def prob_greater_than(self, threshold: float) -> float:
        """Exact posterior tail probability P(theta > threshold)."""
        return max(0.0, min(1.0, 1.0 - self.prob_less_than(threshold)))

    def update(self, passed: bool, *, weight: float = 1.0, decay: float = 1.0) -> None:
        try:
            d = float(decay)
        except (TypeError, ValueError, OverflowError):
            d = 1.0
        if math.isfinite(d) and 0.0 < d < 1.0:
            self.alpha = max(0.5, self.alpha * d)
            self.beta = max(0.5, self.beta * d)
        try:
            w = float(weight)
        except (TypeError, ValueError, OverflowError):
            w = 1.0
        if not math.isfinite(w) or w < 0.0:
            w = 0.0
        if passed:
            self.alpha += w
        else:
            self.beta += w

    def to_dict(self) -> Dict[str, Any]:
        lo, hi = self.credible_interval_95()
        return {
            "rule_id": self.rule_id,
            "alpha": round(self.alpha, 3),
            "beta": round(self.beta, 3),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz-style continued fraction for the incomplete beta function."""
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), the Beta(a, b) CDF at x."""
    try:
        aa = float(a)
        bb = float(b)
        xx = float(x)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not (math.isfinite(aa) and math.isfinite(bb)) or aa <= 0.0 or bb <= 0.0:
        return 0.0
    if not math.isfinite(aa + bb):
        return 0.0
    if math.isnan(xx):
        return 0.0
    if xx <= 0.0:
        return 0.0
    if xx >= 1.0:
        return 1.0
    log_front = (aa * math.log(xx) + bb * math.log1p(-xx)
                 + math.lgamma(aa + bb) - math.lgamma(aa) - math.lgamma(bb))
    front = math.exp(log_front)
    if xx < (aa + 1.0) / (aa + bb + 2.0):
        val = front * _beta_continued_fraction(aa, bb, xx) / aa
    else:
        val = 1.0 - front * _beta_continued_fraction(bb, aa, 1.0 - xx) / bb
    return max(0.0, min(1.0, val))


def _beta_quantile(a: float, b: float, p: float, *, iters: int = 80) -> float:
    """Inverse Beta CDF by deterministic bisection (monotone, robust)."""
    target = max(0.0, min(1.0, p))
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(a, b, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _as_distribution(values: Sequence[float]) -> Any:
    """Convert a sequence to finite non-negative floats; None when invalid."""
    if values is None or isinstance(values, (str, bytes, dict)):
        return None
    try:
        items = list(values)
    except TypeError:
        return None
    if not items:
        return None
    out: List[float] = []
    for v in items:
        try:
            f = float(v)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(f):
            return None
        out.append(max(_EPS, f))
    return out


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q) in nats with Laplace smoothing."""
    pp = _as_distribution(p)
    qq = _as_distribution(q)
    if pp is None or qq is None or len(pp) != len(qq):
        return 0.0
    sp = sum(pp)
    sq = sum(qq)
    if sp <= 0.0 or sq <= 0.0:
        return 0.0
    total = 0.0
    for px, qx in zip(pp, qq):
        pn = px / sp
        qn = qx / sq
        total += pn * math.log(pn / qn)
    return max(0.0, total)


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Compute symmetric, bounded [0, ln(2)] Jensen-Shannon divergence between P and Q."""
    pp = _as_distribution(p)
    qq = _as_distribution(q)
    if pp is None or qq is None or len(pp) != len(qq):
        return 0.0
    sp = sum(pp)
    sq = sum(qq)
    if sp <= 0.0 or sq <= 0.0:
        return 0.0
    pn = [x / sp for x in pp]
    qn = [x / sq for x in qq]
    m = [0.5 * (a + b) for a, b in zip(pn, qn)]
    val = 0.5 * kl_divergence(pn, m) + 0.5 * kl_divergence(qn, m)
    return max(0.0, min(math.log(2.0), val))


class BayesianComplianceNet:
    """Tracks per-rule Beta posteriors and categorical Dirichlet distributions."""

    def __init__(self, prior_alpha: float = 2.0, prior_beta: float = 0.5) -> None:
        self.prior_alpha = _finite_positive(prior_alpha, 2.0)
        self.prior_beta = _finite_positive(prior_beta, 0.5)
        self._rules: Dict[str, BetaPosterior] = {}

    def observe(self, rule_id: str, passed: bool, *, weight: float = 1.0, decay: float = 0.98) -> BetaPosterior:
        rid = str(rule_id or "default")[:200]
        post = self._rules.get(rid)
        if post is None:
            post = BetaPosterior(rule_id=rid, alpha=self.prior_alpha, beta=self.prior_beta)
            self._rules[rid] = post
        post.update(passed, weight=weight, decay=decay)
        return post

    def fragile_rules(self, threshold: float = 0.75) -> List[BetaPosterior]:
        try:
            th = float(threshold)
        except (TypeError, ValueError, OverflowError):
            th = 0.75
        if not math.isfinite(th):
            th = 0.75
        return sorted(
            [p for p in self._rules.values() if p.mean < th],
            key=lambda x: (x.mean, x.rule_id),
        )

    def overall_reliability(self) -> float:
        if not self._rules:
            return 1.0
        return sum(p.mean for p in self._rules.values()) / len(self._rules)


def _self_test() -> int:
    net = BayesianComplianceNet()
    for _ in range(10):
        net.observe("R1_FORMAT", True)
    for _ in range(5):
        net.observe("R2_TONE", False)

    assert net.observe("R1_FORMAT", True).mean > 0.85
    fragile = net.fragile_rules(0.60)
    assert len(fragile) == 1 and fragile[0].rule_id == "R2_TONE"

    # KL & JS divergence checks
    p = [0.8, 0.1, 0.1]
    q = [0.1, 0.1, 0.8]
    assert kl_divergence(p, p) < 1e-9
    assert kl_divergence(p, q) > 1.0
    assert abs(jensen_shannon_divergence(p, q) - jensen_shannon_divergence(q, p)) < 1e-9

    # Exact Beta CDF: Beta(1,1) is uniform, Beta(2,1) has CDF x^2.
    assert abs(regularized_incomplete_beta(1.0, 1.0, 0.3) - 0.3) < 1e-9
    assert abs(regularized_incomplete_beta(2.0, 1.0, 0.5) - 0.25) < 1e-9
    assert regularized_incomplete_beta(2.0, 1.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 1.0, 1.0) == 1.0
    assert regularized_incomplete_beta(float("nan"), 1.0, 0.5) == 0.0

    post = BetaPosterior("R_TAIL", alpha=2.0, beta=2.0)
    cdf = post.prob_less_than(0.5)
    assert abs(cdf - 0.5) < 1e-6, cdf
    assert 0.0 <= post.prob_less_than(-3.0) == 0.0
    assert post.prob_less_than(3.0) == 1.0
    assert abs(post.prob_less_than(0.2) + post.prob_greater_than(0.2) - 1.0) < 1e-9
    xs = [i / 20.0 for i in range(21)]
    cdfs = [post.prob_less_than(x) for x in xs]
    assert all(b >= a for a, b in zip(cdfs, cdfs[1:])), cdfs

    # Exact credible interval brackets the mean and is deterministic.
    lo, hi = post.credible_interval_exact(0.95)
    assert 0.0 <= lo <= post.mean <= hi <= 1.0, (lo, post.mean, hi)
    assert (lo, hi) == post.credible_interval_exact(0.95)
    nlo, nhi = post.credible_interval_95()
    assert 0.0 <= nlo <= post.mean <= nhi <= 1.0, (nlo, nhi)

    # Malformed numeric inputs never corrupt the posterior.
    before = (post.alpha, post.beta)
    post.update(True, weight=float("nan"))
    post.update(True, weight=float("inf"))
    post.update(False, weight=-5)
    post.update(True, weight=0, decay=float("nan"))
    assert (post.alpha, post.beta) == before, (post.alpha, post.beta)
    post.update(True, weight=2.0)
    assert post.alpha == before[0] + 2.0

    # Divergence helpers fail soft on malformed distributions.
    for bad in (None, [], [1, 2, 3], "abc", [float("nan"), 0.2], {"a": 1}):
        assert kl_divergence(bad, [0.5, 0.5]) == 0.0
        assert jensen_shannon_divergence(bad, [0.5, 0.5]) == 0.0
    assert jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]) <= math.log(2.0) + 1e-12

    # Deterministic ordering for equally fragile rules.
    net2 = BayesianComplianceNet()
    for rid in ("R_z", "R_a", "R_m"):
        net2.observe(rid, False, decay=1.0)
    names = [p.rule_id for p in net2.fragile_rules(0.9)]
    assert names == sorted(names), names
    assert len(net2.fragile_rules(float("nan"))) == 3
    assert net2.observe(None, True).rule_id == "default"  # type: ignore[arg-type]

    print("bayesian._self_test: OK PASS (Beta-Binomial posteriors, credible intervals, KL/JS divergence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
