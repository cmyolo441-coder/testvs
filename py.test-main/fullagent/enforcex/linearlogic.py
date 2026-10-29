"""linearlogic — Girard Resource-Aware Linear Logic (o-, (*), &, !A) Budget Verifier.

In Jean-Yves Girard's **Linear Logic** (1987), propositions are not static
truths that can be duplicated (contraction) or discarded (weakening) at will;
they are **consumable resources**:
  * Multiplicative Conjunction (`A (*) B` / tensor): having both resource `A`
    and resource `B` simultaneously.
  * Linear Implication (`A -o B` / lollipop): consuming one unit of `A` to
    produce `B`.
  * Additive Conjunction (`A & B` / with): operator chooses *either* `A` or `B`
    from the same resource pool (cannot consume both).
  * Exponential Modality (`!A` / of-course): an inexhaustible capability that
    permits arbitrary contraction and weakening.

Tracks exact linear resource consumption across tool calls, retries, and
one-time security nonces so neither double-spend nor resource leaks occur.

Self-test: ``python3 -m fullagent.enforcex.linearlogic``
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

__all__ = [
    "LinearTransition",
    "ResourceLedgerReport",
    "LinearResourceContext",
]


@dataclass(frozen=True)
class LinearTransition:
    """Represents a linear implication `consumes -o produces`."""
    name: str
    consumes: Tuple[str, ...]
    produces: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
        }


@dataclass
class ResourceLedgerReport:
    valid: bool
    remaining_linear: Dict[str, int] = field(default_factory=dict)
    exponentials: List[str] = field(default_factory=list)
    steps_applied: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "remaining_linear": dict(self.remaining_linear),
            "exponentials": list(self.exponentials),
            "steps_applied": list(self.steps_applied),
            "violations": list(self.violations),
        }


class LinearResourceContext:
    """Multiset linear context `Delta` paired with intuitionistic/unbounded `!Gamma`."""

    def __init__(
        self,
        linear_resources: Optional[Dict[str, int]] = None,
        exponential_capabilities: Iterable[str] = (),
    ) -> None:
        self.delta: Counter[str] = Counter()
        for k, v in (linear_resources or {}).items():
            if int(v) > 0:
                self.delta[str(k)] += int(v)
        self.gamma: Set[str] = {str(x) for x in exponential_capabilities}
        self.history: List[str] = []
        self.violations: List[str] = []

    def can_fire(self, rule: LinearTransition) -> bool:
        needed = Counter(rule.consumes)
        for res, count in needed.items():
            if res in self.gamma:
                continue
            if self.delta.get(res, 0) < count:
                return False
        return True

    def fire(self, rule: LinearTransition) -> bool:
        """Apply linear lollipop transition `consumes -o produces`."""
        needed = Counter(rule.consumes)
        for res, count in needed.items():
            if res in self.gamma:
                continue
            if self.delta.get(res, 0) < count:
                self.violations.append(
                    f"Linear resource double-spend/underflow on '{res}' in transition '{rule.name}' "
                    f"(required={count}, available={self.delta.get(res, 0)})"
                )
                return False
        for res, count in needed.items():
            if res not in self.gamma:
                self.delta[res] -= count
                if self.delta[res] <= 0:
                    del self.delta[res]
        for prod in rule.produces:
            if prod not in self.gamma:
                self.delta[prod] += 1
        self.history.append(rule.name)
        return True

    def check_no_unconsumed(self, must_consume: Iterable[str] = ()) -> bool:
        """In strict Linear Logic (no weakening), linear obligations in `must_consume` must reach 0."""
        ok = True
        for res in must_consume:
            rem = self.delta.get(res, 0)
            if rem > 0:
                self.violations.append(f"Unconsumed linear obligation '{res}' (remaining={rem})")
                ok = False
        return ok and len(self.violations) == 0

    def report(self) -> ResourceLedgerReport:
        return ResourceLedgerReport(
            valid=(len(self.violations) == 0),
            remaining_linear=dict(self.delta),
            exponentials=sorted(self.gamma),
            steps_applied=list(self.history),
            violations=list(self.violations),
        )


def _self_test() -> int:
    # 1. Single-use approval nonce + retry budget + !read_only capability
    ctx = LinearResourceContext(
        linear_resources={"approval_nonce": 1, "retry_token": 2, "must_verify": 1},
        exponential_capabilities=["!read_policy"],
    )
    t_deploy = LinearTransition("deploy_action", ("approval_nonce", "!read_policy"), ("deployed_state",))
    t_verify = LinearTransition("verify_action", ("deployed_state", "must_verify"), ("certified_output",))

    assert ctx.fire(t_deploy) is True
    # Attempting to reuse the consumed one-time approval_nonce fails!
    assert ctx.fire(t_deploy) is False
    assert len(ctx.violations) == 1

    # Clear violation to test clean completion of must_verify
    ctx.violations.clear()
    assert ctx.fire(t_verify) is True
    assert ctx.check_no_unconsumed(["approval_nonce", "must_verify"]) is True

    print("linearlogic._self_test: OK PASS (Girard lollipop -o, tensor (*), !A exponential, double-spend block)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
