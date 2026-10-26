from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .lifecycle_growth_analysis import (
    IN_FLIGHT_ANALYSIS_STATES,
    analysis_run_state,
    analysis_status_errors,
    analysis_status_record,
)
from .lifecycle_growth_safety import STEP_TRACE_STATES, step_outcome_records, step_outcomes_errors
from .lifecycle_growth_values import (
    CLAIM_BOUNDARY,
    PREPARED_STATUS,
    count_errors,
    metadata_ref,
    metadata_refs,
    positive_count,
    refs_errors,
    require_nonnegative_count,
    require_state,
    state_errors,
)


_COUNT_FIELDS = (
    "eligible_count",
    "attempted_count",
    "delivered_count",
    "displayed_count",
    "acted_count",
    "outcome_count",
)


def build_growth_measurement_readout(*, lifecycle_growth_id: str, eligible_count: int, attempted_count: int, delivered_count: int, displayed_count: int, acted_count: int, outcome_count: int, runtime_days_observed: int, denominator_state: str, data_freshness_state: str, instrumentation_state: str, sample_ratio_state: str, cross_exposure_state: str, overlap_state: str, primary_metric_state: str, guardrail_state: str, causal_claim_status: str, rollback_state: str, provider_evidence_refs: Sequence[str], actual_exposure_evidence_refs: Sequence[str], data_evidence_refs: Sequence[str], runtime_evidence_refs: Sequence[str], causal_evidence_refs: Sequence[str], step_outcomes: Sequence[Mapping[str, str]], step_trace_state: str, analysis_status: Mapping[str, Any]) -> dict[str, object]:
    counts = {
        "eligible_count": require_nonnegative_count(eligible_count, field="eligible_count"),
        "attempted_count": require_nonnegative_count(attempted_count, field="attempted_count"),
        "delivered_count": require_nonnegative_count(delivered_count, field="delivered_count"),
        "displayed_count": require_nonnegative_count(displayed_count, field="displayed_count"),
        "acted_count": require_nonnegative_count(acted_count, field="acted_count"),
        "outcome_count": require_nonnegative_count(outcome_count, field="outcome_count"),
    }
    runtime_days = require_nonnegative_count(runtime_days_observed, field="runtime_days_observed")
    record: dict[str, object] = {
        "schema_version": "growth_measurement_readout/v1",
        "status": PREPARED_STATUS,
        "lifecycle_growth_id": metadata_ref(lifecycle_growth_id, field="lifecycle_growth_id"),
        **counts,
        "runtime_days_observed": runtime_days,
        "denominator_state": require_state(denominator_state, field="denominator_state", allowed=("known", "unknown")),
        "data_freshness_state": require_state(data_freshness_state, field="data_freshness_state", allowed=("fresh", "stale", "unknown")),
        "instrumentation_state": require_state(instrumentation_state, field="instrumentation_state", allowed=("healthy", "broken", "unknown")),
        "sample_ratio_state": require_state(sample_ratio_state, field="sample_ratio_state", allowed=("match", "mismatch", "unknown")),
        "cross_exposure_state": require_state(cross_exposure_state, field="cross_exposure_state", allowed=("absent", "detected", "unknown")),
        "overlap_state": require_state(overlap_state, field="overlap_state", allowed=("absent", "detected", "unknown")),
        "primary_metric_state": require_state(primary_metric_state, field="primary_metric_state", allowed=("improved", "not_improved", "inconclusive")),
        "guardrail_state": require_state(guardrail_state, field="guardrail_state", allowed=("passed", "failed", "inconclusive")),
        "causal_claim_status": require_state(causal_claim_status, field="causal_claim_status", allowed=("established", "not_established", "unknown")),
        "rollback_state": require_state(rollback_state, field="rollback_state", allowed=("not_triggered", "triggered")),
        "provider_evidence_refs": metadata_refs(provider_evidence_refs, field="provider_evidence_refs", required=counts["delivered_count"] > 0),
        "actual_exposure_evidence_refs": metadata_refs(actual_exposure_evidence_refs, field="actual_exposure_evidence_refs", required=counts["displayed_count"] > 0),
        "data_evidence_refs": metadata_refs(data_evidence_refs, field="data_evidence_refs", required=denominator_state == "known"),
        "runtime_evidence_refs": metadata_refs(runtime_evidence_refs, field="runtime_evidence_refs", required=runtime_days > 0),
        "causal_evidence_refs": metadata_refs(causal_evidence_refs, field="causal_evidence_refs", required=causal_claim_status == "established"),
        "step_outcomes": step_outcome_records(step_outcomes),
        "step_trace_state": require_state(step_trace_state, field="step_trace_state", allowed=STEP_TRACE_STATES),
        "analysis_status": analysis_status_record(analysis_status),
        "disposition": "",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    record["disposition"] = derive_readout_disposition(record)
    return record


def derive_readout_disposition(record: Mapping[str, Any]) -> str:
    """The four-word disposition, gated first on safety and then on the analysis run.

    A rollback signal is an observed guardrail breach and outranks everything:
    holding it because an analysis is queued would keep a harmful treatment
    live. Below that, an in-flight run means the numbers are not final, so the
    readout is `review` no matter how good they look, and any run state other
    than `completed` leaves the numbers unbacked. Elapsed time never enters
    here: a delayed run is still a queued or running one.
    """
    if record.get("rollback_state") == "triggered" or record.get("guardrail_state") == "failed":
        return "rollback"
    run_state = analysis_run_state(record)
    if run_state in IN_FLIGHT_ANALYSIS_STATES:
        return "review"
    if run_state != "completed":
        return "insufficient_data"
    if not positive_count(record.get("eligible_count")) or not positive_count(record.get("displayed_count")) or not positive_count(record.get("outcome_count")):
        return "insufficient_data"
    if record.get("denominator_state") != "known" or record.get("data_freshness_state") != "fresh":
        return "insufficient_data"
    if record.get("instrumentation_state") != "healthy" or record.get("sample_ratio_state") != "match" or record.get("cross_exposure_state") != "absent" or record.get("overlap_state") != "absent":
        return "review"
    if record.get("primary_metric_state") != "improved" or record.get("guardrail_state") != "passed":
        return "insufficient_data"
    if record.get("causal_claim_status") != "established" or not _has_refs(record.get("causal_evidence_refs")):
        return "insufficient_data"
    if not _has_refs(record.get("actual_exposure_evidence_refs")) or not _has_refs(record.get("runtime_evidence_refs")):
        return "insufficient_data"
    return "ship"


def validate_readout(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in _COUNT_FIELDS:
        errors.extend(count_errors(record.get(field), field=field))
    errors.extend(count_errors(record.get("runtime_days_observed"), field="runtime_days_observed"))
    counts: list[int] = []
    for field in _COUNT_FIELDS:
        value = record.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            counts.append(value)
    if len(counts) == len(_COUNT_FIELDS) and any(left < right for left, right in zip(counts, counts[1:])):
        errors.append("readout funnel counts must not increase")
    _extend_state_errors(errors, record)
    errors.extend(refs_errors(record.get("provider_evidence_refs"), field="provider_evidence_refs", required=positive_count(record.get("delivered_count"))))
    errors.extend(refs_errors(record.get("actual_exposure_evidence_refs"), field="actual_exposure_evidence_refs", required=positive_count(record.get("displayed_count"))))
    errors.extend(refs_errors(record.get("data_evidence_refs"), field="data_evidence_refs", required=record.get("denominator_state") == "known"))
    errors.extend(refs_errors(record.get("runtime_evidence_refs"), field="runtime_evidence_refs", required=positive_count(record.get("runtime_days_observed"))))
    errors.extend(refs_errors(record.get("causal_evidence_refs"), field="causal_evidence_refs", required=record.get("causal_claim_status") == "established"))
    errors.extend(step_outcomes_errors(record.get("step_outcomes")))
    errors.extend(state_errors(record.get("step_trace_state"), field="step_trace_state", allowed=STEP_TRACE_STATES))
    errors.extend(analysis_status_errors(record.get("analysis_status")))
    if record.get("disposition") != derive_readout_disposition(record):
        errors.append("readout disposition must match derived disposition")
    return errors


def _extend_state_errors(errors: list[str], record: Mapping[str, Any]) -> None:
    for field, allowed in (
        ("denominator_state", ("known", "unknown")),
        ("data_freshness_state", ("fresh", "stale", "unknown")),
        ("instrumentation_state", ("healthy", "broken", "unknown")),
        ("sample_ratio_state", ("match", "mismatch", "unknown")),
        ("cross_exposure_state", ("absent", "detected", "unknown")),
        ("overlap_state", ("absent", "detected", "unknown")),
        ("primary_metric_state", ("improved", "not_improved", "inconclusive")),
        ("guardrail_state", ("passed", "failed", "inconclusive")),
        ("causal_claim_status", ("established", "not_established", "unknown")),
        ("rollback_state", ("not_triggered", "triggered")),
    ):
        errors.extend(state_errors(record.get(field), field=field, allowed=allowed))


def _has_refs(value: Any) -> bool:
    return isinstance(value, list) and bool(value)
