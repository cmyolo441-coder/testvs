"""refineguard — Liquid Refinement Types ({v: T | p(v)}) & Dependent Schema Checker.

Extends standard structural type checking (JSON Schema / Python types) with
**Liquid Refinement Types** (Rondon, Kawaguchi & Jhala) and **Dependent Record
Refinements** where the refinement predicate of field `k_2` can reference the
runtime value of field `k_1`:
  * Base Refinement `{v: int | 1 <= v <= 100}`
  * Dependent Record `Sigma(start: int, {end: int | end >= start})`
  * Cardinality Dependency `Sigma(count: int, {items: list | len(items) == count})`
  * Sum Conservation `Sigma(total: float, {parts: list | sum(parts) == total})`
  * Subtyping Check `T1 <: T2` via finite witness domain implication.

Self-test: ``python3 -m fullagent.enforcex.refineguard``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

__all__ = [
    "RefinedFieldSpec",
    "RefinementViolation",
    "RefinementCheckReport",
    "LiquidRefinementChecker",
]


@dataclass
class RefinedFieldSpec:
    field_name: str
    expected_type: type
    predicate: Callable[[Any, Dict[str, Any]], bool]
    description: str


@dataclass
class RefinementViolation:
    field_name: str
    kind: str  # missing_field | type_error | refinement_failed
    message: str
    actual_value: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_name": self.field_name,
            "kind": self.kind,
            "message": self.message,
            "actual_value": repr(self.actual_value)[:100],
        }


@dataclass
class RefinementCheckReport:
    valid: bool
    checked_fields: int
    violations: List[RefinementViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "checked_fields": self.checked_fields,
            "violations": [v.to_dict() for v in self.violations],
        }


class LiquidRefinementChecker:
    """Checks dependent record refinements `{v: T | phi(v, env)}` and finite subtyping."""

    def __init__(self) -> None:
        self.specs: List[RefinedFieldSpec] = []

    def add_field(
        self,
        field_name: str,
        expected_type: type,
        predicate: Callable[[Any, Dict[str, Any]], bool],
        description: str,
    ) -> None:
        self.specs.append(
            RefinedFieldSpec(
                field_name=str(field_name),
                expected_type=expected_type,
                predicate=predicate,
                description=str(description),
            )
        )

    def check_record(self, record: Dict[str, Any]) -> RefinementCheckReport:
        env = dict(record or {})
        violations: List[RefinementViolation] = []
        for spec in self.specs:
            if spec.field_name not in env:
                violations.append(
                    RefinementViolation(
                        field_name=spec.field_name,
                        kind="missing_field",
                        message=f"Required refined field '{spec.field_name}' is missing",
                    )
                )
                continue
            val = env[spec.field_name]
            if not isinstance(val, spec.expected_type) or (spec.expected_type is int and isinstance(val, bool)):
                violations.append(
                    RefinementViolation(
                        field_name=spec.field_name,
                        kind="type_error",
                        message=f"Expected base type {spec.expected_type.__name__} for '{spec.field_name}'",
                        actual_value=val,
                    )
                )
                continue
            try:
                ok = bool(spec.predicate(val, env))
            except Exception as exc:
                ok = False
                violations.append(
                    RefinementViolation(
                        field_name=spec.field_name,
                        kind="refinement_failed",
                        message=f"Refinement evaluation error on '{spec.field_name}': {exc}",
                        actual_value=val,
                    )
                )
                continue
            if not ok:
                violations.append(
                    RefinementViolation(
                        field_name=spec.field_name,
                        kind="refinement_failed",
                        message=f"Liquid refinement violated on '{spec.field_name}': {spec.description}",
                        actual_value=val,
                    )
                )
        return RefinementCheckReport(
            valid=(len(violations) == 0),
            checked_fields=len(self.specs),
            violations=violations,
        )

    @staticmethod
    def is_subtype(
        pred_sub: Callable[[Any], bool],
        pred_super: Callable[[Any], bool],
        domain_samples: Iterable[Any],
    ) -> Tuple[bool, Optional[Any]]:
        """Check refinement subtyping `{v | pred_sub(v)} <: {v | pred_super(v)}`.
        Returns `(True, None)` if valid across `domain_samples`, else `(False, counterexample)`."""
        for x in domain_samples:
            if pred_sub(x) and not pred_super(x):
                return False, x
        return True, None


def _self_test() -> int:
    checker = LiquidRefinementChecker()
    checker.add_field("min_score", int, lambda v, _: 0 <= v <= 100, "{v: int | 0 <= v <= 100}")
    checker.add_field("max_score", int, lambda v, env: v >= env.get("min_score", 0) and v <= 100, "{v: int | min_score <= v <= 100}")
    checker.add_field("count", int, lambda v, _: v >= 1, "{v: int | v >= 1}")
    checker.add_field("items", list, lambda v, env: len(v) == env.get("count", -1), "{v: list | len(v) == count}")

    rep_ok = checker.check_record({"min_score": 70, "max_score": 95, "count": 2, "items": ["a", "b"]})
    assert rep_ok.valid

    rep_bad = checker.check_record({"min_score": 80, "max_score": 60, "count": 3, "items": ["only_one"]})
    assert not rep_bad.valid and len(rep_bad.violations) == 2

    # Subtyping check: {v: int | 10 <= v <= 20} <: {v: int | 0 <= v <= 100}
    sub_ok, witness = LiquidRefinementChecker.is_subtype(
        lambda x: 10 <= x <= 20,
        lambda x: 0 <= x <= 100,
        range(-50, 150),
    )
    assert sub_ok and witness is None

    sub_bad, counterex = LiquidRefinementChecker.is_subtype(
        lambda x: 0 <= x <= 120,
        lambda x: 0 <= x <= 100,
        range(0, 150),
    )
    assert not sub_bad and counterex == 101

    print("refineguard._self_test: OK PASS (Dependent Sigma records, liquid predicates, subtyping counterexamples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
