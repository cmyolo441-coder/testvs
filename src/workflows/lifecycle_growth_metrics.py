"""Versioned metric coverage construction and exact-plan decision reconciliation."""
from collections.abc import Mapping
from typing import Final, TypedDict

from .lifecycle_growth_artifacts import validate_experiment
from .lifecycle_growth_readout import validate_readout
from .lifecycle_growth_configuration import parse_binding
from .lifecycle_growth_configuration_values import artifact_digest, closed, digest, record, reference
from .lifecycle_growth_metric_members import (
    MEMBER_SCHEMA, PLAN_BINDING_SCHEMA, MetricComposite, parse_composites,
    parse_member_set, parse_plan_binding, review_composites,
)
from .lifecycle_growth_metric_values import (
    METRIC_BOUNDARY, MetricInputError, MetricResult, MetricReview, MetricRole,
    MetricState, expected_metrics, parse_results, reconcile_results,
)

METRIC_SCHEMA: Final = "lifecycle_growth_metric_coverage/v1"


class MetricCoverage(TypedDict):
    schema_version: str
    lifecycle_growth_id: str
    configuration_digest: str
    experiment_digest: str
    analysis_run_ref: str
    readout_digest: str
    metric_plan_binding_digest: str | None
    results: list[MetricResult]
    composites: list[MetricComposite]
    claim_boundary: str


class CoverageReview(MetricReview):
    composites: list[MetricComposite]


def parse_metric_coverage(value: object) -> MetricCoverage:
    source = record(value)
    # Composite metadata is optional for atomic metrics, not an extensions slot.
    source = closed({"composites": [], "metric_plan_binding_digest": None, **source}, set(MetricCoverage.__annotations__))
    if source["schema_version"] != METRIC_SCHEMA or source["claim_boundary"] != METRIC_BOUNDARY:
        raise MetricInputError("metric_coverage_constants_invalid")
    return {"schema_version": METRIC_SCHEMA, "lifecycle_growth_id": reference(source["lifecycle_growth_id"]),
            "configuration_digest": digest(source["configuration_digest"]),
            "experiment_digest": digest(source["experiment_digest"]),
            "analysis_run_ref": reference(source["analysis_run_ref"]), "readout_digest": digest(source["readout_digest"]),
            "metric_plan_binding_digest": None if source["metric_plan_binding_digest"] is None else digest(source["metric_plan_binding_digest"]),
            "results": parse_results(source["results"]), "composites": parse_composites(source["composites"]),
            "claim_boundary": METRIC_BOUNDARY}


def build_metric_coverage(payload: Mapping[str, object]) -> MetricCoverage:
    """Adapters supply outcomes; this builder only binds their metadata to artifacts."""
    source = closed({"composites": [], "metric_plan_binding": None, **payload},
                    {"experiment", "readout", "configuration_binding", "results", "composites", "metric_plan_binding"})
    plan, readout = record(source["experiment"]), record(source["readout"])
    if validate_experiment(plan) or validate_readout(readout):
        raise MetricInputError("metric_artifacts_invalid")
    binding = parse_binding(source["configuration_binding"])
    seal = binding["seal"]
    if seal is None:
        raise MetricInputError("metric_configuration_mismatch")
    coverage = parse_metric_coverage({"schema_version": METRIC_SCHEMA,
        "lifecycle_growth_id": plan.get("lifecycle_growth_id"),
        "configuration_digest": seal["observation"]["configuration_digest"],
        "experiment_digest": artifact_digest(plan), "readout_digest": artifact_digest(readout),
        "analysis_run_ref": record(readout.get("analysis_status")).get("run_ref"),
        "results": source["results"], "composites": source["composites"],
        "metric_plan_binding_digest": None if source["metric_plan_binding"] is None else artifact_digest(parse_plan_binding(source["metric_plan_binding"])),
        "claim_boundary": METRIC_BOUNDARY})
    review_metric_coverage({**source, "metric_coverage": coverage})
    return coverage


def review_metric_coverage(payload: Mapping[str, object]) -> CoverageReview:
    """Missing observations hold; only bound, expected, evidenced harm is rollback."""
    plan, readout = record(payload["experiment"]), record(payload["readout"])
    expected = expected_metrics(plan)
    raw = payload.get("metric_coverage")
    if raw is None:
        return {"complete": False, "rollback": False, "reasons": ["metric_coverage_legacy"],
                "results": [], "missing_metric_refs": sorted(expected), "composites": []}
    coverage = parse_metric_coverage(raw)
    result = reconcile_results(coverage["results"], expected)
    plan_binding = None if payload.get("metric_plan_binding") is None else parse_plan_binding(payload["metric_plan_binding"])
    composites = review_composites(plan_binding, coverage["composites"], expected)
    reasons = [*result["reasons"], *composites["reasons"]]
    config = None if payload.get("configuration_binding") is None else parse_binding(payload["configuration_binding"])
    seal = config["seal"] if config else None
    actual_plan = artifact_digest(plan)
    artifacts_match = (
        coverage["experiment_digest"] == actual_plan
        and coverage["readout_digest"] == artifact_digest(readout)
        and coverage["analysis_run_ref"] == record(readout.get("analysis_status")).get("run_ref")
        and coverage["lifecycle_growth_id"] == plan.get("lifecycle_growth_id") == readout.get("lifecycle_growth_id")
    )
    seal_matches = seal is not None and coverage["configuration_digest"] == seal["observation"]["configuration_digest"]
    binding_matches = artifacts_match and seal_matches
    if not binding_matches:
        reasons.append("metric_configuration_mismatch")
    membership_matches = coverage["metric_plan_binding_digest"] == (artifact_digest(plan_binding) if plan_binding else None)
    if plan_binding is not None:
        membership_matches &= (plan_binding["experiment_digest"] == actual_plan
                               and plan_binding["configuration_digest"] == coverage["configuration_digest"])
    if not membership_matches:
        reasons.append("metric_composite_members_mismatch")
    if result["complete"]:
        primary = next(item for item in result["results"] if item["role"] == MetricRole.PRIMARY)
        guards = [item for item in result["results"] if item["role"] == MetricRole.GUARDRAIL]
        aggregate_guard = "failed" if any(item["state"] == MetricState.FAILED for item in guards) else "passed"
        if primary["state"] != readout.get("primary_metric_state") or aggregate_guard != readout.get("guardrail_state"):
            reasons.append("metric_aggregate_mismatch")
        if primary["state"] != MetricState.IMPROVED:
            reasons.append("metric_decision_ineligible")
    return {"complete": result["complete"] and composites["complete"] and binding_matches and membership_matches,
            "rollback": artifacts_match and (seal is None or seal_matches) and (result["rollback"] or (membership_matches and composites["rollback"])),
            "reasons": list(dict.fromkeys(reasons)), "results": result["results"],
            "missing_metric_refs": result["missing_metric_refs"], "composites": composites["composites"]}


def validate_metric_artifact(value: object) -> list[str]:
    source = record(value)
    parsers = {METRIC_SCHEMA: parse_metric_coverage, MEMBER_SCHEMA: parse_member_set, PLAN_BINDING_SCHEMA: parse_plan_binding}
    try:
        schema = reference(source.get("schema_version"))
        parser = parsers.get(schema)
        if parser is None:
            raise MetricInputError("metric_schema_invalid")
        parser(source)
    except ValueError as exc:
        return [str(exc)]
    return []
