"""Prepared launch configuration, separate from closed lifecycle v1 artifacts.

Provider-neutral concept adoption only: ordered audiences, explicitly approved
promotion carry, separate graduation, and precise missing-evidence reasons.
No upstream implementation or enterprise schema is used. Rollout observation
and rollback prerequisites are OMH's own cleanup policy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import TypeGuard, TypedDict

from .lifecycle_growth_safety import route_lifecycle_workflow_mutation
from .lifecycle_growth_values import (
    CLAIM_BOUNDARY,
    PREPARED_STATUS,
    metadata_ref,
    metadata_refs,
    require_nonnegative_count,
    require_state,
)


class _AudienceRule(TypedDict):
    rule_ref: str
    evaluation_domain_ref: str | None
    bucketing_domain_ref: str | None
    bucketing_subject: str
    condition_refs: list[str]
    rollout_share: int | float
    result_kind: str
    variant_ref: str | None
    reachable: bool | None


_RULE_KEYS = frozenset(_AudienceRule.__annotations__) - {"reachable"}


def _ref(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return metadata_ref(value, field=field)


def _refs(value: object, field: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence of references")
    values: Sequence[object] = value
    refs: list[str] = []
    for ref in values:
        if not isinstance(ref, str):
            raise ValueError(f"{field} must contain strings")
        refs.append(ref)
    return metadata_refs(refs, field=field, required=False)


def _is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def _mapping(value: object, field: str) -> Mapping[object, object]:
    if not _is_mapping(value):
        raise ValueError(f"{field} must be an object")
    return value


def _state(value: object, field: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is invalid")
    return require_state(value, field=field, allowed=allowed)


def _share(value: object, field: str) -> int | float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not 0 <= value <= 100 or not math.isfinite(value)):
        raise ValueError(f"{field} must be a finite share from 0 to 100")
    return value


def _record(schema: str, lifecycle_growth_id: str, **fields: object) -> dict[str, object]:
    return {"schema_version": schema, "status": PREPARED_STATUS,
            "lifecycle_growth_id": _ref(lifecycle_growth_id, "lifecycle_growth_id"),
            **fields, "claim_boundary": CLAIM_BOUNDARY}


def _audience_rule(value: object) -> _AudienceRule:
    value = _mapping(value, "rule")
    if set(value) != set(_RULE_KEYS):
        raise ValueError("rules entries must carry exactly the audience rule input keys")
    kind = _state(value["result_kind"], "result_kind", ("on", "variant", "split"))
    variant = value["variant_ref"]
    if kind != "variant" and variant is not None:
        raise ValueError("variant_ref must be null unless result_kind is variant")
    return {
        "rule_ref": _ref(value["rule_ref"], "rule_ref"),
        "evaluation_domain_ref": None if value["evaluation_domain_ref"] is None else _ref(value["evaluation_domain_ref"], "evaluation_domain_ref"),
        "bucketing_domain_ref": None if value["bucketing_domain_ref"] is None else _ref(value["bucketing_domain_ref"], "bucketing_domain_ref"),
        "bucketing_subject": _state(value["bucketing_subject"], "bucketing_subject", ("person", "group", "device")),
        "condition_refs": _refs(value["condition_refs"], "condition_refs"),
        "rollout_share": _share(value["rollout_share"], "rollout_share"),
        "result_kind": kind,
        "variant_ref": _ref(variant, "variant_ref") if kind == "variant" else None,
        "reachable": None,
    }


def build_launch_audience_review(
    *, lifecycle_growth_id: str, evaluation_semantics: str,
    rules: object, holdout_exclusion_share: int | float,
) -> dict[str, object]:
    """Analyze ordered configuration, not membership or actual exposure.

    Null domain references mean unknown. Reachable means not provably shadowed,
    not that a subject satisfies conditions or receives treatment. A catch-all
    requires the same evaluation domain, bucketing domain AND subject. There
    is no special group-index or device exception.
    """
    semantics = _state(evaluation_semantics, "evaluation_semantics", ("first_match", "unknown"))
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)) or len(rules) > 32:
        raise ValueError("rules must contain at most 32 audience rule records")
    values: Sequence[object] = rules
    records = [_audience_rule(value) for value in values]
    if len({record["rule_ref"] for record in records}) != len(records):
        raise ValueError("rule_ref must be unique within an audience")
    shadowed: set[tuple[str, str, str]] = set()
    unreachable: list[str] = []
    unknown: list[str] = []
    for record in records:
        evaluation = record["evaluation_domain_ref"]
        bucketing = record["bucketing_domain_ref"]
        if semantics == "unknown" or evaluation is None or bucketing is None:
            unknown.append(record["rule_ref"])
            continue
        domain = (evaluation, bucketing, record["bucketing_subject"])
        record["reachable"] = domain not in shadowed
        if not record["reachable"]:
            unreachable.append(record["rule_ref"])
        elif not record["condition_refs"] and record["rollout_share"] == 100:
            shadowed.add(domain)
    return _record(
        "launch_audience_review/v1", lifecycle_growth_id,
        evaluation_semantics=semantics, rules=records,
        holdout_exclusion_share=_share(holdout_exclusion_share, "holdout_exclusion_share"),
        unreachable_rule_refs=unreachable, unknown_rule_refs=unknown,
        verdict="HOLD" if unreachable or unknown or semantics == "unknown" else "READY",
    )


def build_launch_promotion_preflight(
    *, lifecycle_growth_id: str, source_environment_ref: str, target_environment_ref: str,
    dependency_refs_satisfied: Sequence[str], dependency_refs_to_create: Sequence[str],
    schedule_refs: Sequence[str], approvals: Mapping[str, bool],
    safety: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Prepare disabled promotion; carry consent is not promotion approval.

    The optional existing safety policy supplies the promotion gate. An absent
    policy never implies approval. Carry lists describe a proposal, not creation
    or scheduling, even when the gate is READY.
    """
    supplied = _mapping(approvals, "approvals")
    if (set(supplied) != {"carry_dependencies", "carry_schedules"}
            or any(type(value) is not bool for value in supplied.values())):
        raise ValueError("approvals must contain exactly two boolean carry approvals")
    carry = {"carry_dependencies": supplied["carry_dependencies"] is True,
             "carry_schedules": supplied["carry_schedules"] is True}
    policy: Mapping[object, object] = _mapping(safety, "safety") if safety is not None else {}
    satisfied = _refs(dependency_refs_satisfied, "dependency_refs_satisfied")
    to_create = _refs(dependency_refs_to_create, "dependency_refs_to_create")
    schedules = _refs(schedule_refs, "schedule_refs")
    if set(satisfied) & set(to_create):
        raise ValueError("satisfied and to-create dependencies must be distinct")
    gate = route_lifecycle_workflow_mutation({field: policy.get(field) for field in (
        "workflow_content_state", "mutation_route", "promotion_decision_state", "promotion_result_state",
    )})
    warnings = ["dependencies_unresolved"] if to_create and not carry["carry_dependencies"] else []
    return _record(
        "launch_promotion_preflight/v1", lifecycle_growth_id,
        source_environment_ref=_ref(source_environment_ref, "source_environment_ref"),
        target_environment_ref=_ref(target_environment_ref, "target_environment_ref"),
        target_enabled_state="disabled", dependency_refs_satisfied=satisfied,
        dependency_refs_to_create=to_create, schedule_refs=schedules, approvals=carry,
        carried_dependency_refs=to_create.copy() if carry["carry_dependencies"] else [],
        carried_schedule_refs=schedules.copy() if carry["carry_schedules"] else [],
        promotion_gate=gate, warnings=warnings,
        verdict="HOLD" if warnings or gate["verdict"] != "READY" else "READY",
    )


def build_launch_graduation_check(
    *, lifecycle_growth_id: str, rollout_observed_state: str,
    evidence_refs: Sequence[str], rollback_conditions_state: str,
) -> dict[str, object]:
    """Propose a separate cleanup only from attributed rollout/rollback assertions."""
    rollout = _state(rollout_observed_state, "rollout_observed_state", ("complete", "partial", "unknown"))
    rollback = _state(rollback_conditions_state, "rollback_conditions_state", ("satisfied", "unsatisfied", "unknown"))
    refs = _refs(evidence_refs, "evidence_refs")
    reasons: list[str] = []
    if rollout != "complete":
        reasons.append("rollout_" + rollout)
    if not refs:
        reasons.append("rollout_evidence_absent")
    if rollback != "satisfied":
        reasons.append("rollback_conditions_" + rollback)
    return _record(
        "launch_graduation_check/v1", lifecycle_growth_id,
        rollout_observed_state=rollout, evidence_refs=refs,
        rollback_conditions_state=rollback, reason_codes=reasons,
        cleanup_action="not_proposed" if reasons else "proposed",
        verdict="HOLD" if reasons else "READY",
    )


def prepare_lifecycle_launch(operation: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Adapt closed public operation inputs into the pure launch contracts.

    Unknown or missing keys are refused without echoing untrusted key names.
    These operations return proposals only and never write or invoke a provider.
    """
    required = {
        "audience": {"lifecycle_growth_id", "evaluation_semantics", "rules", "holdout_exclusion_share"},
        "promote": {"lifecycle_growth_id", "source_environment_ref", "target_environment_ref",
                    "dependency_refs_satisfied", "dependency_refs_to_create", "schedule_refs", "approvals"},
        "graduate": {"lifecycle_growth_id", "rollout_observed_state", "evidence_refs", "rollback_conditions_state"},
    }
    if operation not in required:
        raise ValueError("lifecycle launch operation is unsupported")
    optional: set[str] = {"safety"} if operation == "promote" else set()
    if not required[operation] <= payload.keys() or payload.keys() - required[operation] - optional:
        raise ValueError("lifecycle launch input keys are invalid")
    lifecycle_id = _ref(payload["lifecycle_growth_id"], "lifecycle_growth_id")
    if operation == "audience":
        return build_launch_audience_review(
            lifecycle_growth_id=lifecycle_id,
            evaluation_semantics=_state(payload["evaluation_semantics"], "evaluation_semantics", ("first_match", "unknown")),
            rules=payload["rules"],
            holdout_exclusion_share=_share(payload["holdout_exclusion_share"], "holdout_exclusion_share"),
        )
    if operation == "graduate":
        return build_launch_graduation_check(
            lifecycle_growth_id=lifecycle_id,
            rollout_observed_state=_state(payload["rollout_observed_state"], "rollout_observed_state", ("complete", "partial", "unknown")),
            evidence_refs=_refs(payload["evidence_refs"], "evidence_refs"),
            rollback_conditions_state=_state(payload["rollback_conditions_state"], "rollback_conditions_state", ("satisfied", "unsatisfied", "unknown")),
        )
    supplied = _mapping(payload["approvals"], "approvals")
    if set(supplied) != {"carry_dependencies", "carry_schedules"}:
        raise ValueError("approvals must contain exactly two boolean carry approvals")
    approvals: dict[str, bool] = {}
    for field in ("carry_dependencies", "carry_schedules"):
        value = supplied[field]
        if not isinstance(value, bool):
            raise ValueError("approvals must contain exactly two boolean carry approvals")
        approvals[field] = value
    safety = payload.get("safety")
    policy: Mapping[object, object] = _mapping(safety, "safety") if safety is not None else {}
    return build_launch_promotion_preflight(
        lifecycle_growth_id=lifecycle_id,
        source_environment_ref=_ref(payload["source_environment_ref"], "source_environment_ref"),
        target_environment_ref=_ref(payload["target_environment_ref"], "target_environment_ref"),
        dependency_refs_satisfied=_refs(payload["dependency_refs_satisfied"], "dependency_refs_satisfied"),
        dependency_refs_to_create=_refs(payload["dependency_refs_to_create"], "dependency_refs_to_create"),
        schedule_refs=_refs(payload["schedule_refs"], "schedule_refs"), approvals=approvals,
        safety={field: policy.get(field) for field in (
            "workflow_content_state", "mutation_route", "promotion_decision_state", "promotion_result_state",
        )},
    )


def lifecycle_growth_evaluation_context(
    evaluation_context: object, *, displayed_count: int,
) -> dict[str, object]:
    """Validate optional context and return a conservative evaluation overlay.

    Absent context returns no fields, preserving old caller output exactly.
    Resolved/observed context never overrides the existing data-health, runtime,
    approval, or rollback decisions. Malformed context raises ValueError.
    Integration applies this overlay after the existing evaluator's checks.
    """
    if evaluation_context is None:
        return {}
    evaluation_context = _mapping(evaluation_context, "evaluation_context")
    if set(evaluation_context) != {"experiment_reference_state", "baseline_exposure_state"}:
        raise ValueError("evaluation_context must carry exactly reference and baseline states")
    reference = _state(evaluation_context["experiment_reference_state"], "experiment_reference_state", ("resolved", "deleted", "unknown"))
    baseline = _state(evaluation_context["baseline_exposure_state"], "baseline_exposure_state", ("observed", "absent", "unknown"))
    count = require_nonnegative_count(displayed_count, field="displayed_count")
    reasons: list[str] = []
    if reference != "resolved":
        reasons.append("experiment_reference_" + reference)
    if baseline != "observed":
        reasons.append("baseline_exposure_" + baseline)
    if count == 0:
        reasons.append("exposure_absent")
    result: dict[str, object] = {"evidence_reason_codes": reasons, "blocked": reference == "deleted"}
    if reasons:
        result.update(disposition="insufficient_data", interpretation_state="HOLD")
    return result
