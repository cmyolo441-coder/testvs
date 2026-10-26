"""Boundary validation for pure paired-run dispatch planning."""

from __future__ import annotations

from ..quality.paired_run_model import ArmRole, InfrastructureStatus, PairedRunDecision
from .paired_run_dispatch_model import (
    ApprovalState,
    ArmDispatchTarget,
    CostTimeBound,
    DispatchBudgets,
    NamedConcurrencyBudget,
    PairedRunDispatchConfig,
    PairedRunDispatchPlanError,
    SharedResourceMode,
)


def validate_dispatch_inputs(
    decision: PairedRunDecision,
    config: PairedRunDispatchConfig,
) -> dict[ArmRole, ArmDispatchTarget]:
    """Refuse every unfair or divergent plan before cells are constructed."""
    targets = _validated_targets(decision, config)
    _validate_unobserved_matrix(decision)
    _validate_budget_shape(config.budgets, targets)
    expected_count = len(decision.tasks) * len(ArmRole)
    if decision.max_total_runs < expected_count:
        raise PairedRunDispatchPlanError(
            "max_total_runs cannot fund the exact task-by-arm evaluation matrix"
        )
    _validate_total_bounds(decision, config.budgets, targets)
    _validate_shared_resources(decision, config, targets)
    return targets


def named_limits(
    limits: tuple[NamedConcurrencyBudget, ...],
    label: str,
) -> dict[str, int]:
    result: dict[str, int] = {}
    for limit in limits:
        if not limit.name or limit.name in result:
            raise PairedRunDispatchPlanError(
                f"{label} concurrency budgets must have unique names"
            )
        _require_positive_int(limit.maximum, f"{label} concurrency maximum")
        result[limit.name] = limit.maximum
    return result


def _validated_targets(
    decision: PairedRunDecision,
    config: PairedRunDispatchConfig,
) -> dict[ArmRole, ArmDispatchTarget]:
    if not isinstance(config.approval, ApprovalState):
        raise PairedRunDispatchPlanError("approval must be explicit")
    if not isinstance(config.dry_run, bool):
        raise PairedRunDispatchPlanError("dry_run must be a boolean")
    if not isinstance(config.shared_resource_mode, SharedResourceMode):
        raise PairedRunDispatchPlanError("shared_resource_mode is unsupported")
    targets: dict[ArmRole, ArmDispatchTarget] = {}
    for target in config.targets:
        if not isinstance(target.arm, ArmRole) or target.arm in targets:
            raise PairedRunDispatchPlanError("targets must contain each arm exactly once")
        if not target.provider or not target.executor or not target.model:
            raise PairedRunDispatchPlanError(
                "target provider, executor, and model must be non-empty"
            )
        if target.shared_resource_key == "":
            raise PairedRunDispatchPlanError("shared_resource_key cannot be empty")
        targets[target.arm] = target
    if set(targets) != set(ArmRole):
        raise PairedRunDispatchPlanError(
            "targets must contain baseline and variant exactly once"
        )
    frozen_arms = {
        ArmRole.BASELINE: decision.baseline,
        ArmRole.VARIANT: decision.variant,
    }
    for role, target in targets.items():
        frozen = frozen_arms[role]
        if target.executor != frozen.executor:
            raise PairedRunDispatchPlanError("executor substitution is forbidden")
        if target.model != frozen.model:
            raise PairedRunDispatchPlanError("model substitution is forbidden")
    return targets


def _validate_unobserved_matrix(decision: PairedRunDecision) -> None:
    if any(
        result.infrastructure_status is not InfrastructureStatus.NOT_OBSERVED
        for result in decision.results
    ):
        raise PairedRunDispatchPlanError(
            "paired_run_decision already contains observed attempts and cannot be replanned"
        )


def _validate_budget_shape(
    budgets: DispatchBudgets,
    targets: dict[ArmRole, ArmDispatchTarget],
) -> None:
    _require_positive_int(budgets.global_concurrency, "global_concurrency")
    executor_limits = named_limits(budgets.executor_concurrency, "executor")
    provider_limits = named_limits(budgets.provider_concurrency, "provider")
    for target in targets.values():
        if target.executor not in executor_limits:
            raise PairedRunDispatchPlanError("missing executor concurrency budget")
        if target.provider not in provider_limits:
            raise PairedRunDispatchPlanError("missing provider concurrency budget")
        _validate_bound(target.local_estimate, "local estimate")
        _validate_bound(target.provider_estimate, "provider estimate")
    _validate_bound(budgets.local_bound, "local bound")
    _validate_bound(budgets.provider_bound, "provider bound")


def _validate_bound(bound: CostTimeBound, label: str) -> None:
    _require_positive_int(bound.cost_units, f"{label} cost_units")
    _require_positive_int(bound.seconds, f"{label} seconds")


def _require_positive_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PairedRunDispatchPlanError(f"{label} must be a positive integer")


def _validate_total_bounds(
    decision: PairedRunDecision,
    budgets: DispatchBudgets,
    targets: dict[ArmRole, ArmDispatchTarget],
) -> None:
    multiplier = len(decision.tasks)
    local_cost = sum(target.local_estimate.cost_units * multiplier for target in targets.values())
    local_seconds = sum(target.local_estimate.seconds * multiplier for target in targets.values())
    provider_cost = sum(target.provider_estimate.cost_units * multiplier for target in targets.values())
    provider_seconds = sum(target.provider_estimate.seconds * multiplier for target in targets.values())
    if local_cost > budgets.local_bound.cost_units:
        raise PairedRunDispatchPlanError("local cost bound refuses the full paired matrix")
    if local_seconds > budgets.local_bound.seconds:
        raise PairedRunDispatchPlanError("local time bound refuses the full paired matrix")
    if provider_cost > budgets.provider_bound.cost_units:
        raise PairedRunDispatchPlanError("provider cost bound refuses the full paired matrix")
    if provider_seconds > budgets.provider_bound.seconds:
        raise PairedRunDispatchPlanError("provider time bound refuses the full paired matrix")


def _validate_shared_resources(
    decision: PairedRunDecision,
    config: PairedRunDispatchConfig,
    targets: dict[ArmRole, ArmDispatchTarget],
) -> None:
    keys = [
        target.shared_resource_key
        for _ in decision.tasks
        for target in targets.values()
        if target.shared_resource_key is not None
    ]
    if config.shared_resource_mode is SharedResourceMode.REFUSE and len(set(keys)) != len(keys):
        raise PairedRunDispatchPlanError("shared resource reuse is refused before planning")
