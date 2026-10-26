"""Reviewed composite membership identity, separate from observed metric coverage."""
from collections.abc import Mapping
from typing import Final, TypedDict

from .lifecycle_growth_configuration_values import (
    artifact_digest, closed, digest, reference, references, timestamp,
)
from .lifecycle_growth_metric_values import (
    MetricInputError, MetricResult, MetricRole, MetricState, parse_results, reconcile_results,
)

MEMBER_SCHEMA: Final = "lifecycle_metric_member_set/v1"
PLAN_BINDING_SCHEMA: Final = "lifecycle_growth_metric_plan_binding/v1"


class MetricMemberSet(TypedDict):
    schema_version: str
    metric_ref: str
    member_refs: list[str]
    member_set_digest: str
    observed_at: str
    evidence_refs: list[str]


class MetricPlanBinding(TypedDict):
    schema_version: str
    experiment_digest: str
    configuration_digest: str
    composite_memberships: list[MetricMemberSet]


class MetricComposite(TypedDict):
    member_set: MetricMemberSet
    results: list[MetricResult]


class CompositeReview(TypedDict):
    complete: bool
    reasons: list[str]
    rollback: bool
    composites: list[MetricComposite]


def parse_member_set(value: object) -> MetricMemberSet:
    source = closed(value, set(MetricMemberSet.__annotations__))
    members = references(source["member_refs"])
    parent = reference(source["metric_ref"])
    if not members or members != sorted(set(members)) or parent in members:
        raise MetricInputError("metric_composite_members_mismatch")
    projection = {"schema_version": MEMBER_SCHEMA, "metric_ref": parent, "member_refs": members}
    if source["schema_version"] != MEMBER_SCHEMA or digest(source["member_set_digest"]) != artifact_digest(projection):
        raise MetricInputError("metric_composite_members_mismatch")
    refs = references(source["evidence_refs"])
    if not refs:
        raise MetricInputError("metric_member_evidence_required")
    return {"schema_version": MEMBER_SCHEMA, "metric_ref": parent, "member_refs": members,
            "member_set_digest": digest(source["member_set_digest"]),
            "observed_at": timestamp(source["observed_at"]), "evidence_refs": refs}


def parse_plan_binding(value: object) -> MetricPlanBinding:
    source = closed(value, set(MetricPlanBinding.__annotations__))
    raw = source["composite_memberships"]
    if source["schema_version"] != PLAN_BINDING_SCHEMA or not isinstance(raw, list) or len(raw) > 9:
        raise MetricInputError("metric_plan_binding_invalid")
    members = [parse_member_set(item) for item in raw]
    if len({item["metric_ref"] for item in members}) != len(members):
        raise MetricInputError("metric_result_duplicate")
    return {"schema_version": PLAN_BINDING_SCHEMA, "experiment_digest": digest(source["experiment_digest"]),
            "configuration_digest": digest(source["configuration_digest"]), "composite_memberships": members}


def parse_composites(value: object) -> list[MetricComposite]:
    if not isinstance(value, list) or len(value) > 9:
        raise MetricInputError("metric_composites_bound")
    composites: list[MetricComposite] = []
    for item in value:
        source = closed(item, set(MetricComposite.__annotations__))
        composites.append({"member_set": parse_member_set(source["member_set"]),
                           "results": parse_results(source["results"], 8)})
    if len({item["member_set"]["metric_ref"] for item in composites}) != len(composites):
        raise MetricInputError("metric_result_duplicate")
    return composites


def review_composites(binding: MetricPlanBinding | None, composites: list[MetricComposite],
                      expected: Mapping[str, MetricRole]) -> CompositeReview:
    """Observed membership cannot define its own expected set or replace its parent."""
    reviewed = {item["metric_ref"]: item for item in binding["composite_memberships"]} if binding else {}
    observed = {item["member_set"]["metric_ref"]: item for item in composites}
    if (set(reviewed) | set(observed)) - set(expected):
        raise MetricInputError("metric_result_unexpected")
    reasons: list[str] = []
    rollback = False
    eligible = True
    if set(reviewed) != set(observed):
        reasons.append("metric_composite_members_mismatch")
    for parent, item in observed.items():
        frozen = reviewed.get(parent)
        if frozen is None or item["member_set"] != frozen:
            reasons.append("metric_composite_members_mismatch")
            continue
        result = reconcile_results(item["results"], dict.fromkeys(frozen["member_refs"], expected[parent]))
        reasons.extend(result["reasons"])
        rollback |= result["rollback"]
        eligible &= all(member["state"] not in (MetricState.UNCHANGED, MetricState.DEGRADED) for member in item["results"])
    complete = not reasons
    if not eligible:
        reasons.append("metric_decision_ineligible")
    return {"complete": complete, "reasons": list(dict.fromkeys(reasons)), "rollback": rollback, "composites": composites}
