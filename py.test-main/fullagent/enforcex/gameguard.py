"""gameguard — Game-Theoretic Stackelberg Security Game & Regret-Matching Defense Engine.

Models prompt enforcement as a two-player Stackelberg / Zero-Sum game between:
  * Defender (Leader / System Prompt Operator): chooses a probability distribution
    over defense strategies (`STRICT_GATE`, `SANDWICH_RE_ANCHOR`, `REWRITE_SCRUB`,
    `DEONTIC_AUDIT`, `CANARY_TRIPWIRE`).
  * Attacker (Follower / Adversarial Input): chooses an attack vector (`DIRECT_OVERRIDE`,
    `ROLEPLAY_JAILBREAK`, `OBFUSCATED_CIPHER`, `MULTI_TURN_EROSION`, `INDIRECT_TOOL`).

Computes the Defender's optimal mixed strategy via Multiplicative Weights /
Regret Matching over `T` rounds and selects the highest-utility defense posture.

Self-test: ``python3 -m fullagent.enforcex.gameguard``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

__all__ = [
    "DEFENDER_ACTIONS",
    "ATTACKER_ACTIONS",
    "EquilibriumStrategyReport",
    "StackelbergDefenseGame",
    "solve_equilibrium",
]

DEFENDER_ACTIONS: Tuple[str, ...] = (
    "STRICT_GATE",
    "SANDWICH_RE_ANCHOR",
    "REWRITE_SCRUB",
    "DEONTIC_AUDIT",
    "CANARY_TRIPWIRE",
)

ATTACKER_ACTIONS: Tuple[str, ...] = (
    "DIRECT_OVERRIDE",
    "ROLEPLAY_JAILBREAK",
    "OBFUSCATED_CIPHER",
    "MULTI_TURN_EROSION",
    "INDIRECT_TOOL",
)

# Defender payoff matrix U[d][a] in [0, 1] (higher = better defense against attack `a`)
_PAYOFF_MATRIX: Tuple[Tuple[float, ...], ...] = (
    # DIRECT, ROLEPLAY, CIPHER, MULTITURN, INDIRECT
    (0.95,    0.70,     0.65,   0.60,      0.75),  # STRICT_GATE
    (0.80,    0.92,     0.70,   0.95,      0.82),  # SANDWICH_RE_ANCHOR
    (0.85,    0.78,     0.94,   0.72,      0.88),  # REWRITE_SCRUB
    (0.88,    0.90,     0.75,   0.85,      0.80),  # DEONTIC_AUDIT
    (0.92,    0.82,     0.88,   0.78,      0.96),  # CANARY_TRIPWIRE
)


@dataclass
class EquilibriumStrategyReport:
    defender_distribution: Dict[str, float]
    attacker_distribution: Dict[str, float]
    game_value: float
    recommended_posture: str
    iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defender_distribution": {k: round(v, 4) for k, v in self.defender_distribution.items()},
            "attacker_distribution": {k: round(v, 4) for k, v in self.attacker_distribution.items()},
            "game_value": round(self.game_value, 4),
            "recommended_posture": self.recommended_posture,
            "iterations": self.iterations,
        }


def solve_equilibrium(
    observed_attacks: Sequence[str] = (),
    *,
    iterations: int = 200,
    learning_rate: float = 0.25,
) -> EquilibriumStrategyReport:
    """Compute minimax equilibrium via Multiplicative Weights Update (MWU)."""
    n_def = len(DEFENDER_ACTIONS)
    n_atk = len(ATTACKER_ACTIONS)

    # Prior weights boosted by any observed attack types in the session
    atk_idx_map = {name: i for i, name in enumerate(ATTACKER_ACTIONS)}
    w_def = [1.0] * n_def
    w_atk = [1.0] * n_atk
    for obs in (observed_attacks or ()):
        idx = atk_idx_map.get(str(obs).upper())
        if idx is not None:
            w_atk[idx] += 1.5

    cum_def = [0.0] * n_def
    cum_atk = [0.0] * n_atk

    iters = max(10, int(iterations))
    eta = max(0.01, float(learning_rate))

    for _ in range(iters):
        s_def = sum(w_def)
        s_atk = sum(w_atk)
        p_def = [w / s_def for w in w_def]
        p_atk = [w / s_atk for w in w_atk]

        for i in range(n_def):
            cum_def[i] += p_def[i]
        for j in range(n_atk):
            cum_atk[j] += p_atk[j]

        # Expected payoff for each defender action against current p_atk
        for i in range(n_def):
            payoff_i = sum(_PAYOFF_MATRIX[i][j] * p_atk[j] for j in range(n_atk))
            w_def[i] *= math.exp(eta * payoff_i)

        # Attacker wants to MINIMIZE defender payoff (zero-sum)
        for j in range(n_atk):
            def_payoff_j = sum(p_def[i] * _PAYOFF_MATRIX[i][j] for i in range(n_def))
            atk_gain_j = 1.0 - def_payoff_j
            w_atk[j] *= math.exp(eta * atk_gain_j)

    avg_def = [c / iters for c in cum_def]
    avg_atk = [c / iters for c in cum_atk]

    # Expected game value at equilibrium
    val = sum(
        avg_def[i] * _PAYOFF_MATRIX[i][j] * avg_atk[j]
        for i in range(n_def)
        for j in range(n_atk)
    )
    best_idx = max(range(n_def), key=lambda i: avg_def[i])
    return EquilibriumStrategyReport(
        defender_distribution={DEFENDER_ACTIONS[i]: avg_def[i] for i in range(n_def)},
        attacker_distribution={ATTACKER_ACTIONS[j]: avg_atk[j] for j in range(n_atk)},
        game_value=val,
        recommended_posture=DEFENDER_ACTIONS[best_idx],
        iterations=iters,
    )


class StackelbergDefenseGame:
    def solve(self, observed_attacks: Sequence[str] = ()) -> EquilibriumStrategyReport:
        return solve_equilibrium(observed_attacks)


def _self_test() -> int:
    rep = solve_equilibrium(["MULTI_TURN_EROSION", "ROLEPLAY_JAILBREAK"], iterations=150)
    assert abs(sum(rep.defender_distribution.values()) - 1.0) < 1e-6
    assert abs(sum(rep.attacker_distribution.values()) - 1.0) < 1e-6
    assert 0.70 <= rep.game_value <= 1.0, rep.to_dict()
    assert rep.recommended_posture in DEFENDER_ACTIONS

    print("gameguard._self_test: OK PASS (MWU Nash/Stackelberg equilibrium, value=" + f"{rep.game_value:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
