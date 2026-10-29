"""cusum — Page's Two-Sided Tabular CUSUM & Shiryaev-Roberts Change-Point Detector.

Provides sequential quickest-detection algorithms (Page 1954, Shiryaev 1963,
Roberts 1966, Lorden 1971) to detect multi-turn compliance degradation or
adversarial drift with minimax optimal detection delay:
  1. **Page's Two-Sided Tabular CUSUM**:
     - Upper statistic: `S_t^+ = max(0, S_{t-1}^+ + (x_t - mu_0 - k))`
     - Lower statistic: `S_t^- = max(0, S_{t-1}^- + (mu_0 - k - x_t))`
     - Triggers alarm when `max(S_t^+, S_t^-) > h`, where `k = delta / 2` is the
       reference allowance and `h` controls the Average Run Length (`ARL_0`).
  2. **Shiryaev-Roberts-Pollak (SRP) Statistic**:
     - `R_t = (1 + R_{t-1}) * LR_t`, where `LR_t = exp(delta * (x_t - mu_0 - delta/2) / sigma^2)`
       is the Gaussian likelihood ratio between pre-change `N(mu_0, sigma^2)`
       and post-change `N(mu_1, sigma^2)`.

Self-test: ``python3 -m fullagent.enforcex.cusum``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

__all__ = [
    "ChangePointStep",
    "SequentialDriftDetector",
]


def _finite(value: Any, default: float) -> float:
    """Best-effort finite float coercion (never raises)."""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if math.isfinite(f) else default


@dataclass
class ChangePointStep:
    step: int
    value: float
    cusum_pos: float
    cusum_neg: float
    shiryaev_roberts: float
    alarm: bool
    direction: str  # none | upward_shift | downward_shift

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "value": round(self.value, 4),
            "cusum_pos": round(self.cusum_pos, 4),
            "cusum_neg": round(self.cusum_neg, 4),
            "shiryaev_roberts": round(self.shiryaev_roberts, 4),
            "alarm": self.alarm,
            "direction": self.direction,
        }


class SequentialDriftDetector:
    """Two-sided Page CUSUM and Shiryaev-Roberts sequential change-point detector."""

    def __init__(
        self,
        baseline_mean: float = 95.0,
        sigma: float = 4.0,
        shift_magnitude: float = 8.0,
        cusum_threshold_h: float = 16.0,
        sr_threshold: float = 120.0,
        *,
        reset_after_alarm: bool = False,
    ) -> None:
        self.mu0 = _finite(baseline_mean, 0.0)
        self.sigma = max(1e-3, _finite(sigma, 1.0))
        self.delta = abs(_finite(shift_magnitude, 0.0))
        self.k = 0.5 * self.delta
        self.h = max(0.0, _finite(cusum_threshold_h, 16.0))
        self.sr_threshold = max(1e-9, _finite(sr_threshold, 120.0))
        self.reset_after_alarm = bool(reset_after_alarm)

        self.step_count = 0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.sr_stat = 0.0

    def reset(self) -> None:
        self.step_count = 0
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.sr_stat = 0.0

    def update(self, x: float) -> ChangePointStep:
        self.step_count += 1
        val = _finite(x, self.mu0)  # malformed/non-finite sample = no evidence
        self.s_pos = max(0.0, self.s_pos + (val - self.mu0 - self.k))
        self.s_neg = max(0.0, self.s_neg + (self.mu0 - self.k - val))

        # Log-likelihood ratio for downward compliance drop (mu_1 = mu_0 - delta)
        if self.delta > 0.0:
            z_drop = (self.mu0 - val)
            log_lr = (self.delta * (z_drop - 0.5 * self.delta)) / (self.sigma ** 2)
            lr = math.exp(max(-20.0, min(20.0, log_lr)))
            self.sr_stat = min(1e9, (1.0 + self.sr_stat) * lr)
        else:
            # zero drift hypothesis: no evidence, so no Shiryaev-Roberts drift
            self.sr_stat = 0.0

        alarm = False
        direction = "none"
        if self.s_neg > self.h or self.sr_stat > self.sr_threshold:
            alarm = True
            direction = "downward_shift"
        elif self.s_pos > self.h:
            alarm = True
            direction = "upward_shift"

        step = ChangePointStep(
            step=self.step_count,
            value=val,
            cusum_pos=self.s_pos,
            cusum_neg=self.s_neg,
            shiryaev_roberts=self.sr_stat,
            alarm=alarm,
            direction=direction,
        )
        if alarm and self.reset_after_alarm:
            self.s_pos = 0.0
            self.s_neg = 0.0
            self.sr_stat = 0.0
        return step


def _self_test() -> int:
    det = SequentialDriftDetector(baseline_mean=95.0, sigma=3.0, shift_magnitude=10.0, cusum_threshold_h=15.0)
    # 10 nominal observations around 95.0 -> zero alarms
    for v in (95.0, 96.0, 94.0, 95.5, 94.5, 95.0, 96.2, 94.8, 95.1, 95.0):
        st = det.update(v)
        assert not st.alarm

    # Sudden compliance degradation to 80.0 -> CUSUM & Shiryaev-Roberts quickly trigger downward_shift alarm!
    triggered = False
    for _ in range(4):
        st = det.update(80.0)
        if st.alarm:
            triggered = True
            assert st.direction == "downward_shift"
            break
    assert triggered

    # Upward drift must be reported as an upward shift (symmetric detector)
    up = SequentialDriftDetector(baseline_mean=95.0, sigma=3.0, shift_magnitude=10.0, cusum_threshold_h=15.0)
    up_alarm = None
    for _ in range(5):
        st = up.update(110.0)
        if st.alarm:
            up_alarm = st
            break
    assert up_alarm is not None and up_alarm.direction == "upward_shift"

    # Reset is a full reset: step numbering and statistics restart at zero
    det.reset()
    assert det.step_count == 0 and det.s_pos == 0.0 and det.s_neg == 0.0 and det.sr_stat == 0.0
    assert det.update(95.0).step == 1

    # Zero drift magnitude must not manufacture false SR alarms on nominal data
    flat = SequentialDriftDetector(baseline_mean=95.0, sigma=3.0, shift_magnitude=0.0,
                                   cusum_threshold_h=15.0, sr_threshold=5.0)
    assert not any(flat.update(95.0).alarm for _ in range(200))

    # Malformed / non-finite observations are neutral, never crash the detector
    for bad in (None, "abc", float("nan"), float("inf"), object()):
        st_bad = det.update(bad)  # type: ignore[arg-type]
        assert not st_bad.alarm and st_bad.value == det.mu0

    # reset_after_alarm clears statistics for the next episode but keeps the clock
    auto = SequentialDriftDetector(baseline_mean=95.0, sigma=3.0, shift_magnitude=10.0,
                                   cusum_threshold_h=15.0, reset_after_alarm=True)
    first = None
    for _ in range(4):
        st = auto.update(80.0)
        if st.alarm:
            first = st
            break
    assert first is not None and auto.s_pos == 0.0 and auto.s_neg == 0.0 and auto.sr_stat == 0.0
    assert auto.update(80.0).step > first.step

    # SR statistic is monotone while a persistent shift keeps arriving
    sr_vals = []
    probe = SequentialDriftDetector(baseline_mean=95.0, sigma=10.0, shift_magnitude=10.0,
                                    cusum_threshold_h=1e9, sr_threshold=1e9)
    for _ in range(4):
        sr_vals.append(probe.update(80.0).shiryaev_roberts)
    assert all(b > a for a, b in zip(sr_vals, sr_vals[1:]))

    print("cusum._self_test: OK PASS (Page two-sided CUSUM + Shiryaev-Roberts quickest change-point detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
