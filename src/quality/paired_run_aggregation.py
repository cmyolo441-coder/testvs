"""Per-arm aggregation and Pareto comparison for paired-run decisions."""

from __future__ import annotations

from typing import Sequence, TypeVar

from .paired_run_model import (
    Aggregate,
    AggregationScope,
    ArmAggregate,
    ArmRole,
    BehaviorVerdict,
    InfrastructureStatus,
    ParetoOutcome,
    RecordedResult,
    RunResultInput,
)
from .paired_run_values import is_digest

_AggregateValue = TypeVar("_AggregateValue")


def aggregate_results(
    results: Sequence[RunResultInput | RecordedResult],
    scope: AggregationScope = AggregationScope(),
) -> Aggregate:
    by_task: dict[str, dict[ArmRole, BehaviorVerdict]] = {}
    for item in results:
        if item.infrastructure_status is InfrastructureStatus.OBSERVED:
            by_task.setdefault(item.task_id, {})[item.arm] = item.behavior_verdict
    variant_wins = False
    baseline_wins = False
    comparable_count = 0
    for arms in by_task.values():
        if set(arms) != {ArmRole.BASELINE, ArmRole.VARIANT}:
            continue
        comparable_count += 1
        baseline = arms[ArmRole.BASELINE]
        variant = arms[ArmRole.VARIANT]
        variant_wins = variant_wins or (
            variant is BehaviorVerdict.PASS and baseline is BehaviorVerdict.FAIL
        )
        baseline_wins = baseline_wins or (
            baseline is BehaviorVerdict.PASS and variant is BehaviorVerdict.FAIL
        )
    if comparable_count == 0:
        outcome = ParetoOutcome.INCONCLUSIVE
    elif variant_wins and baseline_wins:
        outcome = ParetoOutcome.TRADEOFF
    elif variant_wins:
        outcome = ParetoOutcome.VARIANT_DOMINATES
    elif baseline_wins:
        outcome = ParetoOutcome.BASELINE_DOMINATES
    else:
        outcome = ParetoOutcome.NO_OBSERVED_DIFFERENCE
    return Aggregate(
        _arm_aggregate(results, ArmRole.BASELINE),
        _arm_aggregate(results, ArmRole.VARIANT),
        comparable_count,
        scope.task_set_digest,
        scope.baseline_exposure_digest,
        scope.variant_exposure_digest,
        outcome.value,
    )


def _arm_aggregate(
    results: Sequence[RunResultInput | RecordedResult],
    role: ArmRole,
) -> ArmAggregate:
    rows = tuple(item for item in results if item.arm is role)
    return ArmAggregate(
        sum(item.behavior_verdict is BehaviorVerdict.PASS for item in rows),
        sum(item.behavior_verdict is BehaviorVerdict.FAIL for item in rows),
        sum(item.infrastructure_status is InfrastructureStatus.INFRA_ERROR for item in rows),
        sum(item.infrastructure_status is InfrastructureStatus.NOT_OBSERVED for item in rows),
    )


def aggregate_shape_errors(value: _AggregateValue) -> tuple[str, ...]:
    expected = {
        "baseline", "variant", "comparable_task_count", "task_set_digest",
        "baseline_exposure_digest", "variant_exposure_digest",
    }
    arm_expected = {"observed_pass", "observed_fail", "infra_error", "not_observed"}
    if not isinstance(value, dict) or set(value) != expected:
        return ("aggregate keys are not closed",)
    errors: list[str] = []
    for label in ("baseline", "variant"):
        arm = value.get(label)
        if not isinstance(arm, dict) or set(arm) != arm_expected:
            errors.append(f"aggregate.{label} keys are not closed")
            continue
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in arm.values()
        ):
            errors.append(f"aggregate.{label} counts must be non-negative integers")
    comparable = value.get("comparable_task_count")
    if isinstance(comparable, bool) or not isinstance(comparable, int) or comparable < 0:
        errors.append("aggregate.comparable_task_count must be a non-negative integer")
    for field in ("task_set_digest", "baseline_exposure_digest", "variant_exposure_digest"):
        digest = value.get(field)
        if not isinstance(digest, str) or not is_digest(digest):
            errors.append(f"aggregate.{field} must be a lowercase sha256 digest")
    return tuple(errors)
