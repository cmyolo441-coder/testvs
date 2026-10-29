r"""counterfactual — Structural Causal Model (SCM) & Pearl `do()` Intervention Analyzer.

Implements Pearl's 3-step Counterfactual Algorithm over prompt compliance:
  1. Abduction: Given an observed turn `(clause_salience, context_noise,
     adversarial_pressure, observed_score)`, infer the exogenous background
     disturbance $U = u$.
  2. Action (`do(X = x')`): Intervene on the structural graph by setting
     `clause_salience = 1.0` (e.g., via recency pinning) or `adversarial_pressure = 0.0`
     (via input sanitization).
  3. Prediction: Compute the counterfactual compliance score $Y_{X \leftarrow x'}(u)$
     and the Average Causal Effect (ACE) to rank which intervention yields the
     largest compliance gain.

Self-test: ``python3 -m fullagent.enforcex.counterfactual``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

__all__ = [
    "TurnObservation",
    "CounterfactualReport",
    "CausalComplianceEngine",
    "evaluate_counterfactuals",
]


def _finite(value: Any, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return f if math.isfinite(f) else default


def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    return max(lo, min(hi, _finite(value, default)))


@dataclass
class TurnObservation:
    clause_salience: float = 0.5      # [0, 1]
    context_noise: float = 0.3        # [0, 1]
    adversarial_pressure: float = 0.2 # [0, 1]
    observed_score: float = 70.0      # [0, 100]


@dataclass
class CounterfactualReport:
    exogenous_u: float
    observed_score: float
    cf_pin_anchor_score: float
    cf_sanitize_input_score: float
    cf_compact_context_score: float
    ace_pin_anchor: float
    ace_sanitize_input: float
    ace_compact_context: float
    best_intervention: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exogenous_u": round(self.exogenous_u, 4),
            "observed_score": round(self.observed_score, 2),
            "cf_pin_anchor_score": round(self.cf_pin_anchor_score, 2),
            "cf_sanitize_input_score": round(self.cf_sanitize_input_score, 2),
            "cf_compact_context_score": round(self.cf_compact_context_score, 2),
            "ace_pin_anchor": round(self.ace_pin_anchor, 2),
            "ace_sanitize_input": round(self.ace_sanitize_input, 2),
            "ace_compact_context": round(self.ace_compact_context, 2),
            "best_intervention": self.best_intervention,
        }


def _structural_y(salience: float, noise: float, pressure: float, u: float) -> float:
    """Structural equation for compliance score Y in [0, 100]."""
    # Base compliance driven by salience, degraded by noise, pressure, and interaction term
    eff_salience = salience * (1.0 - 0.35 * noise)
    raw = 50.0 + 48.0 * eff_salience - 38.0 * pressure - 18.0 * noise + u
    return max(0.0, min(100.0, raw))


_DEFAULTS = {"clause_salience": 0.5, "context_noise": 0.3,
             "adversarial_pressure": 0.2, "observed_score": 70.0}


def _field(obs: Any, name: str) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, _DEFAULTS[name])
    return getattr(obs, name, _DEFAULTS[name])


def _coerce_obs(obs: Any) -> TurnObservation:
    """Coerce dicts / malformed observations into a bounded TurnObservation."""
    return TurnObservation(
        clause_salience=_clamp(_field(obs, "clause_salience"), 0.0, 1.0, 0.5),
        context_noise=_clamp(_field(obs, "context_noise"), 0.0, 1.0, 0.3),
        adversarial_pressure=_clamp(_field(obs, "adversarial_pressure"), 0.0, 1.0, 0.2),
        observed_score=_clamp(_field(obs, "observed_score"), 0.0, 100.0, 70.0),
    )


def evaluate_counterfactuals(obs: TurnObservation) -> CounterfactualReport:
    obs = _coerce_obs(obs)
    # Step 1: Abduction — solve for exogenous noise U from observed outcome
    predicted_no_u = _structural_y(obs.clause_salience, obs.context_noise, obs.adversarial_pressure, 0.0)
    u = obs.observed_score - predicted_no_u

    # Step 2 & 3: Action do(X = x') & Prediction under fixed exogenous U = u
    cf_anchor = _structural_y(1.0, obs.context_noise, obs.adversarial_pressure, u)
    cf_sanitize = _structural_y(obs.clause_salience, obs.context_noise, 0.0, u)
    cf_compact = _structural_y(obs.clause_salience, 0.05, obs.adversarial_pressure, u)

    ace_anchor = max(0.0, cf_anchor - obs.observed_score)
    ace_sanitize = max(0.0, cf_sanitize - obs.observed_score)
    ace_compact = max(0.0, cf_compact - obs.observed_score)

    # Rank by the *raw* individual causal effect, so that when every
    # intervention hurts equally (reported ACE clamps at 0) the ordering is
    # still defined by the least harmful effect rather than list position.
    candidates = [
        ("pin_recency_anchor", ace_anchor, cf_anchor - obs.observed_score),
        ("sanitize_adversarial_input", ace_sanitize, cf_sanitize - obs.observed_score),
        ("compact_context_noise", ace_compact, cf_compact - obs.observed_score),
    ]
    best = max(candidates, key=lambda x: x[2])[0]

    return CounterfactualReport(
        exogenous_u=u,
        observed_score=obs.observed_score,
        cf_pin_anchor_score=cf_anchor,
        cf_sanitize_input_score=cf_sanitize,
        cf_compact_context_score=cf_compact,
        ace_pin_anchor=ace_anchor,
        ace_sanitize_input=ace_sanitize,
        ace_compact_context=ace_compact,
        best_intervention=best,
    )


class CausalComplianceEngine:
    def analyze(self, obs: TurnObservation) -> CounterfactualReport:
        return evaluate_counterfactuals(obs)


def _self_test() -> int:
    # Turn with buried prompt (low salience=0.25) -> pinning anchor has largest ACE
    obs1 = TurnObservation(clause_salience=0.25, context_noise=0.2, adversarial_pressure=0.1, observed_score=55.0)
    r1 = evaluate_counterfactuals(obs1)
    assert r1.best_intervention == "pin_recency_anchor", r1.to_dict()
    assert r1.cf_pin_anchor_score > obs1.observed_score + 20.0

    # Turn with heavy adversarial injection (pressure=0.85) -> sanitizing input has largest ACE
    obs2 = TurnObservation(clause_salience=0.90, context_noise=0.1, adversarial_pressure=0.85, observed_score=52.0)
    r2 = evaluate_counterfactuals(obs2)
    assert r2.best_intervention == "sanitize_adversarial_input", r2.to_dict()

    # Malformed / out-of-range observations are clamped, never NaN
    obs_bad = TurnObservation(clause_salience=float("nan"), context_noise=99.0,
                              adversarial_pressure=-3.0, observed_score=float("inf"))
    r_bad = evaluate_counterfactuals(obs_bad)
    for value in r_bad.to_dict().values():
        assert not (isinstance(value, float) and math.isnan(value)), r_bad.to_dict()
    assert r_bad.observed_score == 70.0 and r_bad.exogenous_u == 70.0 - _structural_y(0.5, 1.0, 0.0, 0.0)

    # Dict observations are accepted and equivalent to the dataclass form
    obs_dict = {"clause_salience": 0.25, "context_noise": 0.2,
                "adversarial_pressure": 0.1, "observed_score": 55.0}
    r_dict = evaluate_counterfactuals(obs_dict)  # type: ignore[arg-type]
    assert r_dict.to_dict() == r1.to_dict()

    # Structural equation is monotone in the expected directions
    assert _structural_y(0.9, 0.2, 0.2, 0.0) > _structural_y(0.2, 0.2, 0.2, 0.0)
    assert _structural_y(0.5, 0.2, 0.8, 0.0) < _structural_y(0.5, 0.2, 0.1, 0.0)
    assert _structural_y(0.5, 0.9, 0.2, 0.0) < _structural_y(0.5, 0.1, 0.2, 0.0)

    # Ablation edges: already-pinned salience or already-clean pressure yields no ACE
    r_pinned = evaluate_counterfactuals(TurnObservation(1.0, 0.1, 0.5, 60.0))
    assert r_pinned.ace_pin_anchor == 0.0
    r_clean = evaluate_counterfactuals(TurnObservation(0.6, 0.1, 0.0, 60.0))
    assert r_clean.ace_sanitize_input == 0.0

    # Deterministic repeated evaluation and engine/none robustness
    assert evaluate_counterfactuals(obs1).to_dict() == r1.to_dict()
    assert CausalComplianceEngine().analyze(obs1).to_dict() == r1.to_dict()
    r_none = CausalComplianceEngine().analyze(None)  # type: ignore[arg-type]
    assert math.isfinite(r_none.exogenous_u) and 0.0 <= r_none.observed_score <= 100.0

    print("counterfactual._self_test: OK PASS (Pearl abduction-action-prediction + do-calculus ACE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
