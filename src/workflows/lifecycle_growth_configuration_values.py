"""Versioned canonical assignment projection and bounded configuration parsing."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
import hashlib
import json
import re
from typing import Final, TypeGuard

from .lifecycle_growth_artifacts import validate_experiment
from .lifecycle_growth_launch import build_launch_audience_review
from .lifecycle_growth_values import artifact_shape_errors, metadata_ref
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

CANONICAL_SCHEMA: Final = "lifecycle_assignment_configuration/v1"
CONFIGURATION_BOUNDARY: Final = (
    "Local reconciliation of caller-supplied metadata only; not provider observation, "
    "approval, execution, or causal evidence. No provider reads or writes occur."
)
PLAN_FIELDS: Final = (
    "treatment_ref", "control_ref", "assignment_unit", "exposure_unit",
    "sticky_assignment_policy", "assignment_event_ref", "actual_exposure_event_ref",
    "holdout_state", "holdout_rationale_ref",
)
RULE_FIELDS: Final = (
    "rule_ref", "evaluation_domain_ref", "bucketing_domain_ref", "bucketing_subject",
    "condition_refs", "rollout_share", "result_kind", "variant_ref",
)


class ConfigurationInputError(ValueError):
    """A bounded category, never an echo of raw adapter metadata."""

    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


def is_record(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and all(isinstance(key, str) for key in value)


def record(value: object) -> Mapping[str, object]:
    if not is_record(value):
        raise ConfigurationInputError("configuration_object_invalid")
    return value


def closed(value: object, keys: set[str]) -> Mapping[str, object]:
    result = record(value)
    if set(result) != keys:
        raise ConfigurationInputError("configuration_keys_invalid")
    return result


def reference(value: object) -> str:
    if not isinstance(value, str):
        raise ConfigurationInputError("configuration_reference_invalid")
    try:
        return metadata_ref(value, field="configuration_reference")
    except ValueError as exc:
        raise ConfigurationInputError("configuration_reference_invalid") from exc


def optional_reference(value: object) -> str | None:
    return None if value is None else reference(value)


def references(value: object, limit: int = 8) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise ConfigurationInputError("configuration_references_invalid")
    return [reference(item) for item in value]


def digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ConfigurationInputError("configuration_digest_invalid")
    return value


def timestamp(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value) is None:
        raise ConfigurationInputError("configuration_timestamp_invalid")
    try:
        strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigurationInputError("configuration_timestamp_invalid") from exc
    return value


def artifact_digest(value: Mapping[str, object]) -> str:
    """Hash actual artifact JSON, separately from assignment normalization."""
    try:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ConfigurationInputError("configuration_json_invalid") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _decimal_share(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationInputError("configuration_share_invalid")
    number = Decimal(str(value))
    if not number.is_finite() or not 0 <= number <= 100:
        raise ConfigurationInputError("configuration_share_invalid")
    return "0" if number == 0 else format(number.normalize(), "f")


def canonical_configuration(artifacts: Mapping[str, object]) -> dict[str, object]:
    """Validate both launch artifacts; retain array order and normalize shares."""
    experiment = record(artifacts.get("experiment"))
    if artifact_shape_errors(experiment)[1] or validate_experiment(experiment):
        raise ConfigurationInputError("configuration_experiment_invalid")
    audience = record(artifacts.get("audience_review"))
    audience_keys = {"schema_version", "status", "lifecycle_growth_id", "evaluation_semantics", "rules",
                     "holdout_exclusion_share", "unreachable_rule_refs", "unknown_rule_refs", "verdict", "claim_boundary"}
    closed(audience, audience_keys)
    rules = audience["rules"]
    if not isinstance(rules, list) or len(rules) > 32:
        raise ConfigurationInputError("configuration_rules_invalid")
    inputs = []
    for item in rules:
        rule = closed(item, {*RULE_FIELDS, "reachable"})
        inputs.append({key: rule[key] for key in RULE_FIELDS})
    semantics = audience["evaluation_semantics"]
    share = audience["holdout_exclusion_share"]
    if not isinstance(semantics, str) or not isinstance(share, (int, float)):
        raise ConfigurationInputError("configuration_audience_invalid")
    rebuilt = build_launch_audience_review(
        lifecycle_growth_id=reference(audience["lifecycle_growth_id"]),
        evaluation_semantics=semantics, rules=inputs, holdout_exclusion_share=share,
    )
    normalized_rules = rebuilt["rules"]
    if not isinstance(normalized_rules, list):
        raise ConfigurationInputError("configuration_rules_invalid")
    comparison = {**audience, "lifecycle_growth_id": reference(audience["lifecycle_growth_id"]),
                  "rules": normalized_rules,
                  "unreachable_rule_refs": references(audience["unreachable_rule_refs"], 32),
                  "unknown_rule_refs": references(audience["unknown_rule_refs"], 32)}
    if (rebuilt != comparison or comparison["lifecycle_growth_id"] != reference(experiment["lifecycle_growth_id"])
            or any(record(raw)["reachable"] is not record(normalized)["reachable"]
                   for raw, normalized in zip(rules, normalized_rules))):
        raise ConfigurationInputError("configuration_audience_invalid")
    inputs = [{key: record(rule)[key] for key in RULE_FIELDS} for rule in normalized_rules]
    return {
        "schema_version": CANONICAL_SCHEMA,
        "experiment": {key: reference(experiment[key]) if experiment[key] != "" else "" for key in PLAN_FIELDS},
        "audience_review": {"evaluation_semantics": semantics,
            "rules": [{**rule, "rollout_share": _decimal_share(rule["rollout_share"])} for rule in inputs],
            "holdout_exclusion_share": _decimal_share(share)},
    }


def configuration_digest(artifacts: Mapping[str, object]) -> str:
    return artifact_digest(canonical_configuration(artifacts))


def configuration_is_known(artifacts: Mapping[str, object], revision: str | None) -> bool:
    """Unresolved semantics and mutable condition references cannot prove identity."""
    audience = record(artifacts["audience_review"])
    rules = audience["rules"]
    if not isinstance(rules, list):
        raise ConfigurationInputError("configuration_rules_invalid")
    if audience["evaluation_semantics"] == "unknown" or record(artifacts["experiment"])["holdout_state"] == "unknown":
        return False
    for item in rules:
        rule = record(item)
        if rule["evaluation_domain_ref"] is None or rule["bucketing_domain_ref"] is None:
            return False
        if revision is None and any(re.search(r"(?:_v\d+|sha256_[0-9a-f]{64})$", ref) is None for ref in references(rule["condition_refs"])):
            return False
    return True
