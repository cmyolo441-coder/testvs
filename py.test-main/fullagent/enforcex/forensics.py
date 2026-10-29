r"""forensics — Exact Shapley-Value Blame Attribution & Incident Forensics Engine.

When a model response violates the system prompt, attributes exact cooperative
game-theoretic blame across the 4 root-cause factors $N = \{1, 2, 3, 4\}$:
  1. `attention_burying`: critical directive was buried in the mid-context trough
  2. `adversarial_user_input`: user prompt contained override/injection pressure
  3. `conflict_ambiguity`: system prompt contained conflicting or vague norms
  4. `tool_output_noise`: tool outputs injected distracting or hostile text

Computes exact Shapley values $\phi_i(v)$ over all $2^4 = 16$ coalitions so the
blame shares sum strictly to the total compliance loss (`efficiency axiom`).

Self-test: ``python3 -m fullagent.enforcex.forensics``
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet

__all__ = [
    "FACTORS",
    "ForensicInvestigationReport",
    "IncidentForensicsEngine",
    "compute_shapley_blame",
]

FACTORS = (
    "attention_burying",
    "adversarial_user_input",
    "conflict_ambiguity",
    "tool_output_noise",
)


@dataclass
class ForensicInvestigationReport:
    total_loss: float
    shapley_blame: Dict[str, float]
    blame_shares_pct: Dict[str, float]
    primary_culprit: str
    remediation_playbook: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_loss": round(self.total_loss, 3),
            "shapley_blame": {k: round(v, 4) for k, v in self.shapley_blame.items()},
            "blame_shares_pct": {k: round(v, 2) for k, v in self.blame_shares_pct.items()},
            "primary_culprit": self.primary_culprit,
            "remediation_playbook": self.remediation_playbook,
        }


_PLAYBOOKS = {
    "attention_burying": "Pin buried rules into a Recency Attention Anchor via attentionsteer.optimize_prompt_layout().",
    "adversarial_user_input": "Sanitize user input and wrap in hierarchyguard Ring-2 fences before model dispatch.",
    "conflict_ambiguity": "Resolve deontic norm antinomies via deontic.resolve_conflicts() and autohealer.",
    "tool_output_noise": "Pass tool outputs through toolguard.sanitize_tool_output() and compactguard.",
}


def _clamp01(value: Any) -> float:
    """Clamp an intensity to [0, 1]; NaN / non-numeric degrade to 0.0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def compute_shapley_blame(
    *,
    attention_burying: float = 0.0,
    adversarial_user_input: float = 0.0,
    conflict_ambiguity: float = 0.0,
    tool_output_noise: float = 0.0,
    **extra: Any,
) -> ForensicInvestigationReport:
    """Compute exact Shapley values over the 4 root-cause factors in [0, 1].

    Unknown keyword arguments and malformed intensities are ignored so the
    forensics entry point never raises on bad telemetry."""
    intensities = {
        "attention_burying": _clamp01(attention_burying),
        "adversarial_user_input": _clamp01(adversarial_user_input),
        "conflict_ambiguity": _clamp01(conflict_ambiguity),
        "tool_output_noise": _clamp01(tool_output_noise),
    }
    base_weights = {
        "attention_burying": 28.0,
        "adversarial_user_input": 38.0,
        "conflict_ambiguity": 22.0,
        "tool_output_noise": 18.0,
    }

    def characteristic_v(coalition: FrozenSet[str]) -> float:
        if not coalition:
            return 0.0
        linear = sum(base_weights[f] * intensities[f] for f in coalition)
        # Superadditive synergy when buried attention + adversarial input co-occur
        synergy = 0.0
        if "attention_burying" in coalition and "adversarial_user_input" in coalition:
            synergy += 14.0 * intensities["attention_burying"] * intensities["adversarial_user_input"]
        if "adversarial_user_input" in coalition and "tool_output_noise" in coalition:
            synergy += 10.0 * intensities["adversarial_user_input"] * intensities["tool_output_noise"]
        return min(100.0, linear + synergy)

    n = len(FACTORS)
    n_fact = math.factorial(n)
    shapley: Dict[str, float] = {f: 0.0 for f in FACTORS}

    # exact 2^4 coalition table (cached per call; v() is deterministic)
    coalition_value: Dict[FrozenSet[str], float] = {}
    for size in range(n + 1):
        for subset in itertools.combinations(FACTORS, size):
            fs = frozenset(subset)
            coalition_value[fs] = characteristic_v(fs)

    # Shapley weight depends only on |S|: |S|!(n-|S|-1)!/n!
    weight_by_size = [
        (math.factorial(k) * math.factorial(n - k - 1)) / float(n_fact)
        for k in range(n)
    ]
    for f in FACTORS:
        others = [x for x in FACTORS if x != f]
        for r in range(len(others) + 1):
            weight = weight_by_size[r]
            for subset in itertools.combinations(others, r):
                s_set = frozenset(subset)
                marginal = (coalition_value[s_set | {f}]
                            - coalition_value[s_set])
                shapley[f] += weight * marginal

    total_loss = coalition_value[frozenset(FACTORS)]
    if total_loss > 0:
        shares = {k: (v / total_loss) * 100.0 for k, v in shapley.items()}
    else:
        shares = {k: 0.0 for k in FACTORS}

    culprit = max(FACTORS, key=lambda f: shapley[f])
    return ForensicInvestigationReport(
        total_loss=total_loss,
        shapley_blame=shapley,
        blame_shares_pct=shares,
        primary_culprit=culprit,
        remediation_playbook=_PLAYBOOKS[culprit],
    )


class IncidentForensicsEngine:
    def investigate(self, **kwargs: float) -> ForensicInvestigationReport:
        return compute_shapley_blame(**kwargs)


def _self_test() -> int:
    rep = compute_shapley_blame(
        attention_burying=0.8,
        adversarial_user_input=0.9,
        conflict_ambiguity=0.1,
        tool_output_noise=0.0,
    )
    # Shapley efficiency axiom: sum of Shapley values equals v(N) - v(empty) == total_loss
    assert abs(sum(rep.shapley_blame.values()) - rep.total_loss) < 1e-6, rep.to_dict()
    assert abs(sum(rep.blame_shares_pct.values()) - 100.0) < 1e-5
    assert rep.primary_culprit == "adversarial_user_input"

    # 1. zero intensities: no blame, no division by zero, no raise
    zero = compute_shapley_blame()
    assert zero.total_loss == 0.0
    assert all(v == 0.0 for v in zero.shapley_blame.values())
    assert all(v == 0.0 for v in zero.blame_shares_pct.values())
    assert zero.primary_culprit in FACTORS

    # 2. malformed intensities are clamped, never raise
    junk = compute_shapley_blame(
        attention_burying=None, adversarial_user_input="bad",
        conflict_ambiguity=float("nan"), tool_output_noise=0.0,
        surprising_extra="ignored")
    assert junk.total_loss == 0.0, junk.to_dict()
    # +inf saturates to the maximum intensity (fail-closed blame)
    saturated = compute_shapley_blame(tool_output_noise=float("inf"))
    assert saturated.total_loss == 18.0, saturated.to_dict()
    high = compute_shapley_blame(tool_output_noise=5.0)
    assert high.total_loss == 18.0 and high.shapley_blame["tool_output_noise"] == 18.0

    # 3. efficiency axiom holds across mixed intensities (bounded 0..100)
    for vals in ((1, 1, 1, 1), (0.5, 0.25, 0.75, 0.1), (0, 1, 0, 1)):
        r = compute_shapley_blame(
            attention_burying=vals[0], adversarial_user_input=vals[1],
            conflict_ambiguity=vals[2], tool_output_noise=vals[3])
        assert 0.0 <= r.total_loss <= 100.0
        assert abs(sum(r.shapley_blame.values()) - r.total_loss) < 1e-9
        assert abs(sum(r.blame_shares_pct.values())
                   - (100.0 if r.total_loss else 0.0)) < 1e-9

    # 4. blame is monotone in a factor's intensity
    low = compute_shapley_blame(adversarial_user_input=0.2)
    higher = compute_shapley_blame(adversarial_user_input=0.6)
    assert (higher.shapley_blame["adversarial_user_input"]
            > low.shapley_blame["adversarial_user_input"])

    # 5. every blamed factor receives a non-empty remediation playbook
    assert all(compute_shapley_blame(**{f: 1.0}).remediation_playbook
               for f in FACTORS)

    print("forensics._self_test: OK PASS (exact 2^4 coalition Shapley values, efficiency axiom verified, "
          "clamped malformed intensities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
