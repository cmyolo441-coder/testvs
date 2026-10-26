"""Shared paired-run receipt binding comparison without model imports."""

from __future__ import annotations

from dataclasses import dataclass

from ..coding.hermes_child_evaluation import HermesChildEvaluationBinding


@dataclass(frozen=True, slots=True)
class ExpectedEvaluationBinding:
    task_id: str
    acceptance_criteria_ref: str
    input_digest: str
    arm: str
    executor: str
    model: str
    exposure_digest: str
    execution_revision: str
    timeout_seconds: int


def evaluation_binding_errors(
    binding: HermesChildEvaluationBinding | None,
    expected: ExpectedEvaluationBinding,
) -> tuple[str, ...]:
    """Return every dimension by which a signed binding misses one row."""
    if binding is None:
        return ("attempted result receipt requires a complete evaluation binding",)
    mismatches: list[str] = []
    for field in (
        "task_id",
        "acceptance_criteria_ref",
        "input_digest",
        "arm",
        "executor",
        "model",
        "exposure_digest",
        "execution_revision",
        "timeout_seconds",
    ):
        if getattr(binding, field) != getattr(expected, field):
            mismatches.append(f"receipt evaluation binding {field} does not match result context")
    return tuple(mismatches)


def receipt_reference_uniqueness_errors(refs: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(refs)) != len(refs):
        return ("receipt references must be unique across attempted result rows",)
    return ()
