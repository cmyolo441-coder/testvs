"""Closed metric observations; boundary inputs follow the stdlib artifact protocol."""
from collections.abc import Mapping
from enum import StrEnum
from typing import Final, TypedDict, assert_never

from .lifecycle_growth_configuration_values import closed, reference, references

METRIC_BOUNDARY: Final = (
    "Local metric reconciliation of caller-supplied observations only; not provider "
    "execution, approval, or causal evidence. No queries or provider errors are retained."
)


class MetricInputError(ValueError):
    """A stable redacted category, never arbitrary provider exception text."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class MetricRole(StrEnum):
    PRIMARY = "primary"
    GUARDRAIL = "guardrail"


class MetricState(StrEnum):
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    DEGRADED = "degraded"
    PASSED = "passed"
    FAILED = "failed"
    MISSING = "missing"
    ERRORED = "errored"
    INDETERMINATE = "indeterminate"


class MetricErrorCategory(StrEnum):
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_INPUT = "invalid_input"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    PROVIDER_FAILURE = "provider_failure"
    UNKNOWN = "unknown"


class MetricResult(TypedDict):
    metric_ref: str
    role: MetricRole
    state: MetricState
    evidence_refs: list[str]
    error_category: MetricErrorCategory | None


class MetricReview(TypedDict):
    complete: bool
    rollback: bool
    reasons: list[str]
    results: list[MetricResult]
    missing_metric_refs: list[str]


def parse_result(value: object) -> MetricResult:
    source = closed(value, set(MetricResult.__annotations__))
    try:
        role = MetricRole(reference(source["role"]))
        state = MetricState(reference(source["state"]))
        category = None if source["error_category"] is None else MetricErrorCategory(reference(source["error_category"]))
    except ValueError as exc:
        raise MetricInputError("metric_result_invalid") from exc
    refs = references(source["evidence_refs"])
    match state:
        case MetricState.IMPROVED | MetricState.UNCHANGED | MetricState.DEGRADED:
            if role != MetricRole.PRIMARY:
                raise MetricInputError("metric_role_mismatch")
        case MetricState.PASSED | MetricState.FAILED:
            if role != MetricRole.GUARDRAIL:
                raise MetricInputError("metric_role_mismatch")
        case MetricState.MISSING:
            if refs:
                raise MetricInputError("metric_result_contradiction")
        case MetricState.ERRORED:
            if category is None:
                raise MetricInputError("metric_error_category_required")
        case MetricState.INDETERMINATE:
            pass
        case unreachable:
            assert_never(unreachable)
    if state not in (MetricState.MISSING, MetricState.ERRORED, MetricState.INDETERMINATE) and not refs:
        raise MetricInputError("metric_result_evidence_required")
    if state != MetricState.ERRORED and category is not None:
        raise MetricInputError("metric_result_contradiction")
    return {"metric_ref": reference(source["metric_ref"]), "role": role, "state": state,
            "evidence_refs": refs, "error_category": category}


def parse_results(value: object, limit: int = 9) -> list[MetricResult]:
    if not isinstance(value, list) or len(value) > limit:
        raise MetricInputError("metric_results_bound")
    results = [parse_result(item) for item in value]
    if len({item["metric_ref"] for item in results}) != len(results):
        raise MetricInputError("metric_result_duplicate")
    return results


def expected_metrics(plan: Mapping[str, object]) -> dict[str, MetricRole]:
    """Derive exact refs, never a caller-selected expected count or leaf subset."""
    primary = reference(plan.get("primary_metric_ref"))
    guardrails = references(plan.get("guardrail_metric_refs"))
    if primary in guardrails or len(set(guardrails)) != len(guardrails):
        raise MetricInputError("metric_result_duplicate")
    return {primary: MetricRole.PRIMARY, **dict.fromkeys(guardrails, MetricRole.GUARDRAIL)}


def reconcile_results(results: list[MetricResult], expected: Mapping[str, MetricRole]) -> MetricReview:
    reasons: list[str] = []
    rollback = False
    for item in results:
        ref = item["metric_ref"]
        if ref not in expected:
            raise MetricInputError("metric_result_unexpected")
        if item["role"] != expected[ref]:
            raise MetricInputError("metric_role_mismatch")
        match item["state"]:
            case MetricState.MISSING:
                reasons.append("metric_result_missing")
            case MetricState.ERRORED:
                reasons.append("metric_result_errored")
            case MetricState.INDETERMINATE:
                reasons.append("metric_result_indeterminate")
            case MetricState.FAILED:
                rollback = True
            case MetricState.IMPROVED | MetricState.UNCHANGED | MetricState.DEGRADED | MetricState.PASSED:
                pass
            case unreachable:
                assert_never(unreachable)
    missing = sorted(set(expected) - {item["metric_ref"] for item in results})
    if missing:
        reasons.append("metric_result_missing")
    return {"complete": not reasons, "rollback": rollback, "reasons": list(dict.fromkeys(reasons)),
            "results": results, "missing_metric_refs": missing}
