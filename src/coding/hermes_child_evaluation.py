"""Typed closed evaluation context sealed into Hermes child receipts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Final, TypeAlias

from ..system.metadata_safety import require_opaque_metadata_ref

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_BINDING_FIELDS: Final = frozenset({
    "task_id",
    "acceptance_criteria_ref",
    "input_digest",
    "arm",
    "executor",
    "model",
    "exposure_digest",
    "execution_revision",
    "timeout_seconds",
})
_DIGEST: Final = re.compile(r"^[0-9a-f]{64}$")
_ARMS: Final = frozenset({"baseline", "variant"})
MAX_EVALUATION_TIMEOUT_SECONDS: Final = 3_600


EvaluationBindingRecord: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class HermesChildEvaluationContext:
    task_id: str
    acceptance_criteria_ref: str
    input_digest: str
    arm: str
    executor: str
    exposure_digest: str
    execution_revision: str


@dataclass(frozen=True, slots=True)
class HermesChildEvaluationBinding:
    task_id: str
    acceptance_criteria_ref: str
    input_digest: str
    arm: str
    executor: str
    model: str
    exposure_digest: str
    execution_revision: str
    timeout_seconds: int | float

    def to_record(self) -> EvaluationBindingRecord:
        return {
            "task_id": self.task_id,
            "acceptance_criteria_ref": self.acceptance_criteria_ref,
            "input_digest": self.input_digest,
            "arm": self.arm,
            "executor": self.executor,
            "model": self.model,
            "exposure_digest": self.exposure_digest,
            "execution_revision": self.execution_revision,
            "timeout_seconds": self.timeout_seconds,
        }


def seal_evaluation_binding(
    context: HermesChildEvaluationContext,
    model: str,
    timeout_seconds: float,
) -> HermesChildEvaluationBinding:
    binding = HermesChildEvaluationBinding(
        context.task_id,
        context.acceptance_criteria_ref,
        context.input_digest,
        context.arm,
        context.executor,
        model,
        context.exposure_digest,
        context.execution_revision,
        timeout_seconds,
    )
    if errors := evaluation_binding_errors(binding.to_record()):
        raise ValueError("invalid Hermes child evaluation context: " + "; ".join(errors))
    return binding


def parse_evaluation_binding(
    value: JsonValue,
) -> HermesChildEvaluationBinding | None:
    if evaluation_binding_errors(value) or not isinstance(value, dict):
        return None
    task_id = value.get("task_id")
    criteria = value.get("acceptance_criteria_ref")
    input_digest = value.get("input_digest")
    arm = value.get("arm")
    executor = value.get("executor")
    model = value.get("model")
    exposure_digest = value.get("exposure_digest")
    revision = value.get("execution_revision")
    timeout = value.get("timeout_seconds")
    if not (
        isinstance(task_id, str)
        and isinstance(criteria, str)
        and isinstance(input_digest, str)
        and isinstance(arm, str)
        and isinstance(executor, str)
        and isinstance(model, str)
        and isinstance(exposure_digest, str)
        and isinstance(revision, str)
        and not isinstance(timeout, bool)
        and isinstance(timeout, (int, float))
    ):
        return None
    return HermesChildEvaluationBinding(
        task_id, criteria, input_digest, arm, executor, model,
        exposure_digest, revision, timeout,
    )


def evaluation_binding_errors(value: JsonValue) -> list[str]:
    if not isinstance(value, dict):
        return ["evaluation_binding must be an object"]
    errors: list[str] = []
    if frozenset(value) != _BINDING_FIELDS:
        errors.append("evaluation_binding must use the closed schema")
    for field in (
        "task_id",
        "acceptance_criteria_ref",
        "executor",
        "model",
        "execution_revision",
    ):
        try:
            _ = require_opaque_metadata_ref(
                value.get(field),
                field=f"evaluation_binding.{field}",
            )
        except ValueError:
            errors.append(f"evaluation_binding.{field} must be metadata-only")
    for field in ("input_digest", "exposure_digest"):
        item = value.get(field)
        if not isinstance(item, str) or _DIGEST.fullmatch(item) is None:
            errors.append(f"evaluation_binding.{field} must be a lowercase sha256 digest")
    if value.get("arm") not in _ARMS:
        errors.append("evaluation_binding.arm is invalid")
    timeout = value.get("timeout_seconds")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or not 0 < timeout <= MAX_EVALUATION_TIMEOUT_SECONDS
    ):
        errors.append("evaluation_binding.timeout_seconds must be a positive bounded number")
    return errors
