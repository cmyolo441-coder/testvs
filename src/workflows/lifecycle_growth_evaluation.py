"""Lifecycle promotion interpretation; independent safety signals win over drift."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .lifecycle_growth_analysis import analysis_run_state, analysis_status_field
from .lifecycle_growth_configuration import review_configuration
from .lifecycle_growth_launch import lifecycle_growth_evaluation_context
from .lifecycle_growth_exposure import review_exposure_evidence
from .lifecycle_growth_readout import derive_readout_disposition
from .lifecycle_growth_metrics import review_metric_coverage
from .lifecycle_growth_validation import expected_errors, experiment_hold_reasons

LIFECYCLE_GROWTH_READOUT_SCHEMA_VERSION: Final = "lifecycle_growth_readout/v1"


def evaluate_lifecycle_growth(
    experiment: Mapping[str, object], readout: Mapping[str, object], *, evaluation_context: object = None,
    exposure_evidence: object = None, audience_review: object = None, configuration_binding: object = None,
    metric_coverage: object = None, metric_plan_binding: object = None,
) -> dict[str, object]:
    """Existing public API keeps keyword companions instead of rewriting v1 records."""
    errors = expected_errors(experiment, "growth_experiment_plan/v1", "experiment")
    errors.extend(expected_errors(readout, "growth_measurement_readout/v1", "readout"))
    if not errors and experiment.get("lifecycle_growth_id") != readout.get("lifecycle_growth_id"):
        errors.append("experiment and readout lifecycle_growth_id differ")
    structural_errors = bool(errors)
    result = _readout_lifecycle_growth(readout, experiment, exposure_evidence)
    errors.extend(experiment_hold_reasons(experiment))
    observed_days = readout.get("runtime_days_observed")
    minimum_days = experiment.get("minimum_runtime_days")
    if isinstance(observed_days, int) and isinstance(minimum_days, int) and observed_days < minimum_days:
        errors.append("minimum runtime has not elapsed")
    if errors:
        if result["disposition"] != "rollback":
            result["disposition"] = "insufficient_data"
        result["interpretation_state"] = "HOLD"
    context = lifecycle_growth_evaluation_context(evaluation_context, displayed_count=max(0, _count(readout, "displayed_count")))
    reasons = _errors(result["evidence_reason_codes"]) + _errors(context.get("evidence_reason_codes", []))
    if context.get("evidence_reason_codes") and result["disposition"] != "rollback":
        result.update(disposition="insufficient_data", interpretation_state="HOLD")
    supplied = {"experiment": experiment, "readout": readout, "configuration_binding": configuration_binding}
    if exposure_evidence is not None:
        supplied["exposure_evidence"] = exposure_evidence
    if audience_review is not None:
        supplied["audience_review"] = audience_review
    configuration_reasons = review_configuration(supplied)
    reasons.extend(configuration_reasons)
    if configuration_reasons:
        result["interpretation_state"] = "HOLD"
        if result["disposition"] != "rollback":
            # Configuration precedes analysis/exposure. Structural errors still win.
            result["disposition"] = "review" if not errors and any(code in configuration_reasons for code in (
                "configuration_drift", "configuration_binding_mismatch", "configuration_seal_conflict",
            )) else "insufficient_data"
    metrics = review_metric_coverage({**supplied, "metric_coverage": metric_coverage, "metric_plan_binding": metric_plan_binding})
    reasons.extend(metrics["reasons"])
    if metrics["rollback"] and not structural_errors:
        result.update(disposition="rollback", interpretation_state="HOLD")
    if metrics["reasons"]:
        result["interpretation_state"] = "HOLD"
        if result["disposition"] != "rollback" and not configuration_reasons:
            result["disposition"] = "review" if metrics["complete"] and not errors else "insufficient_data"
    return {
        "schema_version": LIFECYCLE_GROWTH_READOUT_SCHEMA_VERSION,
        "interpretation_state": result["interpretation_state"], "disposition": result["disposition"],
        "assignment_unit": experiment.get("assignment_unit", ""), "exposure_unit": experiment.get("exposure_unit", ""),
        **{key: result[key] for key in ("actual_exposure_count", "delivery_count", "populations", "channels",
            "runtime_days_observed", "analysis_run_state", "analysis_observed_at", "analysis_delay_state")},
        "artifact_errors": _errors(result.get("artifact_errors")) + errors,
        "claim_boundary": "Assignment is not exposure; evaluation is derived from bounded caller-supplied metadata only.",
        "evidence_reason_codes": list(dict.fromkeys(reasons)), "configuration_integrity": not configuration_reasons,
        "blocked": bool(errors or reasons or result["interpretation_state"] == "HOLD"),
        "metric_completeness": metrics["complete"], "metric_results": metrics["results"],
        "missing_metric_refs": metrics["missing_metric_refs"], "metric_composites": metrics["composites"],
    }


def readout_lifecycle_growth(
    readout: Mapping[str, object], *, experiment: Mapping[str, object] | None = None,
    exposure_evidence: object = None, audience_review: object = None, configuration_binding: object = None,
    metric_coverage: object = None, metric_plan_binding: object = None,
) -> dict[str, object]:
    """Read historic artifacts without integrity, or evaluate all supplied companions."""
    if experiment is not None:
        return evaluate_lifecycle_growth(experiment, readout, exposure_evidence=exposure_evidence,
            audience_review=audience_review, configuration_binding=configuration_binding,
            metric_coverage=metric_coverage, metric_plan_binding=metric_plan_binding)
    result = _readout_lifecycle_growth(readout, None, exposure_evidence)
    result.update(configuration_integrity=False, metric_completeness=False, metric_results=[], missing_metric_refs=[], metric_composites=[])
    result["evidence_reason_codes"] = [*_errors(result["evidence_reason_codes"]), "configuration_identity_legacy", "metric_coverage_legacy"]
    return result


def _readout_lifecycle_growth(
    readout: Mapping[str, object], experiment: Mapping[str, object] | None, exposure_evidence: object,
) -> dict[str, object]:
    errors = expected_errors(readout, "growth_measurement_readout/v1", "readout")
    disposition = derive_readout_disposition(readout)
    exposure = review_exposure_evidence(exposure_evidence, experiment, readout)
    reasons = list(exposure.reasons)
    if experiment is None:
        reasons.append("experiment_missing")
    if reasons and disposition == "ship":
        disposition = "review" if "channel_partial_delivery" in reasons else "insufficient_data"
    return {
        "schema_version": LIFECYCLE_GROWTH_READOUT_SCHEMA_VERSION,
        "interpretation_state": "READY" if not errors and disposition == "ship" else "HOLD",
        "disposition": disposition if not errors else "insufficient_data",
        "evidence_reason_codes": reasons, "blocked": bool(errors or reasons or disposition != "ship"),
        "populations": {"eligible": _count(readout, "eligible_count"), "assigned": exposure.assigned_count,
                        "attempted": _count(readout, "attempted_count"), "reached": _count(readout, "displayed_count"),
                        "converted": _count(readout, "outcome_count")},
        "channels": list(exposure.channels), "actual_exposure_count": _count(readout, "displayed_count"),
        "delivery_count": _count(readout, "delivered_count"), "action_count": _count(readout, "acted_count"),
        "outcome_count": _count(readout, "outcome_count"), "runtime_days_observed": _count(readout, "runtime_days_observed"),
        "analysis_run_state": analysis_run_state(readout), "analysis_observed_at": analysis_status_field(readout, "observed_at"),
        "analysis_delay_state": analysis_status_field(readout, "delay_state"), "artifact_errors": errors,
        "claim_boundary": "This derived readout is not provider, delivery, display, outcome, or causal evidence. Analysis-run state and its observation time are reported as the provider observed them, and elapsed time never restates them.",
    }


def _errors(value: object) -> list[str]:
    return value if isinstance(value, list) and all(isinstance(error, str) for error in value) else ["readout errors are invalid"]


def _count(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
