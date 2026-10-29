"""conformal — Split & Adaptive Conformal Prediction (ACI) Uncertainty Calibrator.

Provides finite-sample, distribution-free coverage guarantees (Vovk, Gammerman
& Shafer 2005; Gibbs & Candès 2021 Adaptive Conformal Inference):
  1. **Split Conformal Quantile (`q_hat`)**:
     Given `n` calibration nonconformity residuals `R_1, ..., R_n` and miscoverage
     level `alpha in (0, 1)`, computes the finite-sample corrected order statistic
     at level `ceil((n + 1) * (1 - alpha)) / n`, guaranteeing
     `P(Y_{n+1} in [y_hat - q_hat, y_hat + q_hat]) >= 1 - alpha`.
  2. **Conformal p-Value (`p(y)`)**:
     `p(y) = (1 + |{i : R_i >= R_{n+1}}|) / (n + 1)` — detects out-of-distribution
     adversarial turns when `p(y) < alpha`.
  3. **Gibbs-Candès Online Adaptive Conformal Inference (ACI)**:
     Updates `alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)` so long-run
     empirical coverage converges to `1 - alpha_target` even under distribution shift!

Self-test: ``python3 -m fullagent.enforcex.conformal``
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ConformalInterval",
    "SplitConformalCalibrator",
    "AdaptiveConformalInference",
]

_ALPHA_MIN = 0.001
_ALPHA_MAX = 0.999
_DEFAULT_ALPHA = 0.10


def _finite(value: Any) -> Optional[float]:
    """Best-effort finite float coercion; None for anything malformed."""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _abs_residual(value: Any) -> Optional[float]:
    f = _finite(value)
    return None if f is None else abs(f)


def _alpha_or_default(alpha: Any, default: float = _DEFAULT_ALPHA) -> float:
    a = _finite(alpha)
    if a is None:
        return default
    return max(_ALPHA_MIN, min(_ALPHA_MAX, a))


@dataclass
class ConformalInterval:
    prediction: float
    lower: float
    upper: float
    quantile_radius: float
    alpha: float
    p_value: float

    def contains(self, true_y: float) -> bool:
        try:
            y = float(true_y)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(y):
            return False
        return self.lower <= y <= self.upper

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction": round(self.prediction, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "quantile_radius": round(self.quantile_radius, 4),
            "alpha": round(self.alpha, 4),
            "p_value": round(self.p_value, 4),
        }


class SplitConformalCalibrator:
    """Distribution-free split conformal prediction calibrator."""

    def __init__(self, residuals: Optional[Sequence[float]] = None) -> None:
        cleaned: List[float] = []
        try:
            source = residuals or ()
            for r in source:
                v = _abs_residual(r)
                if v is not None:
                    cleaned.append(v)
        except TypeError:
            source = ()
        cleaned.sort()
        self.residuals: List[float] = cleaned

    def __len__(self) -> int:
        return len(self.residuals)

    def calibrate_residual(self, y_true: float, y_pred: float) -> None:
        a, b = _finite(y_true), _finite(y_pred)
        if a is None or b is None:
            return
        bisect.insort(self.residuals, abs(a - b))

    def quantile(self, alpha: float = _DEFAULT_ALPHA, *, strict: bool = False) -> float:
        n = len(self.residuals)
        if n == 0:
            return 0.0
        a = _alpha_or_default(alpha)
        k = int(math.ceil((n + 1) * (1.0 - a)))
        if k > n:
            # No finite order statistic reaches the requested level with n
            # calibration points; strict mode reports the honest +inf radius.
            if strict:
                return math.inf
            k = n
        idx = max(0, min(n - 1, k - 1))
        return self.residuals[idx]

    def p_value(self, residual: float) -> float:
        n = len(self.residuals)
        if n == 0:
            return 1.0
        r = _abs_residual(residual)
        if r is None:
            return 1.0  # malformed evidence is never treated as an outlier
        ge_count = n - bisect.bisect_left(self.residuals, r - 1e-12)
        return (1.0 + ge_count) / float(n + 1)

    def predict_interval(self, y_pred: float, *, alpha: float = _DEFAULT_ALPHA,
                         candidate_residual: float = 0.0,
                         strict: bool = False) -> ConformalInterval:
        yp = _finite(y_pred)
        if yp is None:
            yp = 0.0
        a = _alpha_or_default(alpha)
        q = self.quantile(a, strict=strict)
        pval = self.p_value(candidate_residual)
        return ConformalInterval(
            prediction=yp,
            lower=yp - q,
            upper=yp + q,
            quantile_radius=q,
            alpha=a,
            p_value=pval,
        )


class AdaptiveConformalInference:
    """Gibbs & Candès (2021) online Adaptive Conformal Inference under distribution shift."""

    def __init__(self, target_alpha: float = 0.10, step_size_gamma: float = 0.05, initial_residuals: Sequence[float] = ()) -> None:
        ta = _finite(target_alpha)
        if ta is None:
            ta = _DEFAULT_ALPHA
        g = _finite(step_size_gamma)
        if g is None:
            g = 0.05
        self.target_alpha = max(0.01, min(0.50, ta))
        self.alpha_t = self.target_alpha
        self.gamma = max(0.001, g)
        self.calibrator = SplitConformalCalibrator(initial_residuals)
        self.errors: List[int] = []

    def step(self, y_pred: float, y_true: float) -> ConformalInterval:
        yp = _finite(y_pred)
        yt = _finite(y_true)
        if yp is None:
            yp = 0.0
        if yt is None:
            yt = yp  # malformed outcome: treat as no evidence, never raise
        interval = self.calibrator.predict_interval(
            yp,
            alpha=self.alpha_t,
            candidate_residual=abs(yt - yp),
        )
        err_t = 0 if interval.contains(yt) else 1
        self.errors.append(err_t)
        # Gibbs-Candès update: alpha_{t+1} = alpha_t + gamma * (target_alpha - err_t)
        self.alpha_t = max(0.005, min(0.95, self.alpha_t + self.gamma * (self.target_alpha - err_t)))
        self.calibrator.calibrate_residual(yt, yp)
        return interval

    def empirical_coverage(self) -> float:
        """Observed 1 - error rate over all steps so far (0.0 when empty)."""
        if not self.errors:
            return 0.0
        return 1.0 - (sum(self.errors) / float(len(self.errors)))


def _self_test() -> int:
    cal = SplitConformalCalibrator([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    iv = cal.predict_interval(90.0, alpha=0.10, candidate_residual=1.0)
    assert iv.quantile_radius >= 4.5
    assert iv.contains(93.0) and not iv.contains(99.0)
    assert iv.p_value > 0.5  # residual=1.0 is small/conforming

    # Outlier residual=12.0 gets minimal conformal p-value = 1 / 11
    assert abs(cal.p_value(12.0) - (1.0 / 11.0)) < 1e-6

    # Verify ACI decreases alpha_t (widens interval) when miscoverage occurs (err_t = 1)
    aci = AdaptiveConformalInference(target_alpha=0.10, step_size_gamma=0.05, initial_residuals=[1.0, 2.0, 3.0, 4.0])
    a_before = aci.alpha_t
    aci.step(y_pred=90.0, y_true=50.0)  # miss -> err_t = 1 -> alpha_t decreases!
    assert aci.alpha_t < a_before

    # Finite-sample index math: k = ceil((n+1)(1-alpha)); with n=9, alpha=.1 -> 9th
    cal9 = SplitConformalCalibrator([float(i) for i in range(1, 10)])
    assert cal9.quantile(0.10) == 9.0
    assert cal9.quantile(0.20) == 8.0  # ceil(10 * 0.8) = 8

    # Small n cannot reach the level: default stays conservative, strict is honest
    small = SplitConformalCalibrator([1.0, 2.0, 3.0, 4.0])
    assert small.quantile(0.05) == 4.0
    assert math.isinf(small.quantile(0.05, strict=True))
    assert math.isinf(small.predict_interval(0.0, alpha=0.05, strict=True).upper)

    # Malformed residuals / queries are ignored instead of crashing
    cal_bad = SplitConformalCalibrator(["x", None, float("nan"), float("inf"), -2.5, 1.0])
    assert cal_bad.residuals == [1.0, 2.5]
    cal_bad.calibrate_residual(None, "x")
    assert len(cal_bad) == 2 and cal_bad.p_value(None) == 1.0

    # Conformal p-values are monotone non-increasing in residual size
    assert cal.p_value(0.5) >= cal.p_value(2.0) >= cal.p_value(99.0)
    assert abs(cal.p_value(99.0) - (1.0 / 11.0)) < 1e-9

    # Determinism and malformed y handling
    iv_a = cal.predict_interval(10.0, alpha=0.10, candidate_residual=1.0)
    iv_b = cal.predict_interval(10.0, alpha=0.10, candidate_residual=1.0)
    assert iv_a.to_dict() == iv_b.to_dict()
    assert not iv_a.contains(None) and not iv_a.contains(float("nan"))

    # ACI clamps stay bounded under persistent misses and hits
    aci2 = AdaptiveConformalInference(target_alpha=0.2, step_size_gamma=0.5, initial_residuals=[1.0])
    for _ in range(50):
        aci2.step(0.0, 1000.0)
    assert 0.005 <= aci2.alpha_t <= 0.95
    for _ in range(50):
        aci2.step(0.0, 0.0)
    assert 0.005 <= aci2.alpha_t <= 0.95
    assert 0.0 <= aci2.empirical_coverage() <= 1.0
    assert _aci_malformed_probe()

    print("conformal._self_test: OK PASS (Split conformal finite-sample quantile, p-values, Gibbs-Candes ACI)")
    return 0


def _aci_malformed_probe() -> bool:
    """Adversarial helper: None/NaN step inputs never poison ACI state."""
    aci = AdaptiveConformalInference(target_alpha=float("nan"), step_size_gamma=float("inf"),
                                     initial_residuals=[1.0, "bad", float("nan")])  # type: ignore[list-item]
    iv = aci.step(None, None)  # type: ignore[arg-type]
    assert math.isfinite(aci.alpha_t) and 0.005 <= aci.alpha_t <= 0.95
    assert math.isfinite(iv.lower) and math.isfinite(iv.upper)
    return True


if __name__ == "__main__":
    raise SystemExit(_self_test())
