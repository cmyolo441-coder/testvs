from __future__ import annotations

from collections.abc import Callable
from typing import Final

from .sales_pipeline_models import JsonObject, JsonValue

_SCOPE_KEYS: Final = frozenset(
    (
        "schema_version", "status", "source_ref", "as_of", "reviewed_at", "cohort_id", "horizon_end",
        "included_motions", "included_owner_ids", "reporting_currency", "amount_semantics", "stage_definitions",
        "forecast_definitions", "scenario_definitions", "conversion_bases", "freshness", "known_data_quality_gaps", "opportunity_count",
        "decision_owner_id", "raw_export_retained", "claim_boundary",
    )
)
_HEALTH_KEYS: Final = frozenset(
    (
        "schema_version", "status", "source_ref", "as_of", "opportunity_count", "movement", "aging", "stalls",
        "slips", "exit_criteria_gaps", "next_step_quality", "concentration", "duplicate_ids", "missing_owner_ids",
        "deal_exceptions", "claim_boundary",
    )
)
_FORECAST_KEYS: Final = frozenset(
    (
        "schema_version", "status", "source_ref", "as_of", "reporting_currency", "amount_semantics",
        "seller_states", "scenarios", "buyer_commitments", "weighted_seller_scenario", "calibration", "confidence",
        "evidence_limits", "authoritative_finance_forecast", "claim_boundary",
    )
)
_OUTCOME_KEYS: Final = frozenset(
    (
        "schema_version", "status", "source_ref", "cohort_id", "horizon_end", "outcome_counts", "reason_counts",
        "contradictions", "evidence_gaps", "research_follow_ups", "causal_claims_established", "claim_boundary",
    )
)
_RENEWAL_KEYS: Final = frozenset(
    (
        "schema_version", "status", "source_ref", "horizon_end", "signals", "open_risks", "evidence_gaps",
        "contradictions", "expansion_hypotheses", "claim_boundary",
    )
)
_HANDOFF_KEYS: Final = frozenset(
    (
        "schema_version", "status", "source_ref", "decision_owner_id", "selected_account_follow_ups",
        "proposed_crm_corrections", "connector", "specialist_routes", "claim_boundary",
    )
)
_SCHEMA_KEYS: Final = {
    "sales_pipeline_scope/v1": _SCOPE_KEYS,
    "sales_pipeline_health/v1": _HEALTH_KEYS,
    "sales_forecast_assessment/v1": _FORECAST_KEYS,
    "sales_outcome_learning_annex/v1": _OUTCOME_KEYS,
    "sales_renewal_risk_annex/v1": _RENEWAL_KEYS,
    "sales_pipeline_handoff/v1": _HANDOFF_KEYS,
}


def _bounded_errors(value: JsonValue, path: str = "artifact", depth: int = 0) -> list[str]:
    if depth > 8:
        return [f"{path} exceeds maximum nesting"]
    if isinstance(value, str):
        if len(value) > 500 or "\n" in value or "://" in value or "@" in value:
            return [f"{path} contains unsafe or unbounded metadata"]
        return []
    if isinstance(value, list):
        if len(value) > 1000:
            return [f"{path} exceeds maximum items"]
        return [error for index, item in enumerate(value) for error in _bounded_errors(item, f"{path}[{index}]", depth + 1)]
    if isinstance(value, dict):
        if len(value) > 100:
            return [f"{path} exceeds maximum fields"]
        return [error for key, item in value.items() for error in _bounded_errors(item, f"{path}.{key}", depth + 1)]
    return []


def _shape_errors(record: JsonObject, schema: str, keys: frozenset[str]) -> list[str]:
    label = schema.removesuffix("/v1")
    errors: list[str] = []
    if set(record) != set(keys):
        errors.append(f"{label} keys are invalid")
    if record.get("schema_version") != schema:
        errors.append(f"{label} schema_version is invalid")
    if record.get("status") not in ("READY", "HOLD", "prepared_not_observed"):
        errors.append(f"{label} status is invalid")
    errors.extend(_bounded_errors(record))
    return errors


def _list_fields(record: JsonObject, label: str, names: tuple[str, ...]) -> list[str]:
    return [f"{label} {name} must be a list" for name in names if not isinstance(record.get(name), list)]


def _row_shape_errors(record: JsonObject, label: str, field: str, keys: frozenset[str]) -> list[str]:
    rows = record.get(field)
    if not isinstance(rows, list):
        return []
    if any(not isinstance(row, dict) or set(row) != set(keys) for row in rows):
        return [f"{label} {field} rows are invalid"]
    return []


def validate_sales_pipeline_scope(record: JsonObject) -> list[str]:
    errors = _shape_errors(record, "sales_pipeline_scope/v1", _SCOPE_KEYS)
    errors.extend(_list_fields(record, "sales_pipeline_scope", (
        "included_motions", "included_owner_ids", "stage_definitions", "forecast_definitions", "scenario_definitions",
        "conversion_bases", "known_data_quality_gaps",
    )))
    errors.extend(_row_shape_errors(record, "sales_pipeline_scope", "stage_definitions", frozenset(
        ("stage_id", "meaning", "required_exit_criteria", "stall_after_days")
    )))
    errors.extend(_row_shape_errors(record, "sales_pipeline_scope", "forecast_definitions", frozenset(
        ("category_id", "meaning", "included_scenario_ids")
    )))
    errors.extend(_row_shape_errors(record, "sales_pipeline_scope", "scenario_definitions", frozenset(
        ("scenario_id", "meaning")
    )))
    errors.extend(_row_shape_errors(record, "sales_pipeline_scope", "conversion_bases", frozenset(
        ("source_currency", "target_currency", "numerator", "denominator", "evidence_ref", "observed_at")
    )))
    freshness = record.get("freshness")
    if not isinstance(freshness, dict) or set(freshness) != {"limit_days", "age_seconds", "status"}:
        errors.append("sales_pipeline_scope freshness is invalid")
    if record.get("status") not in ("READY", "HOLD"):
        errors.append("sales_pipeline_scope status must be READY or HOLD")
    if record.get("raw_export_retained") is not False:
        errors.append("sales_pipeline_scope must not retain raw export")
    gaps = record.get("known_data_quality_gaps")
    if record.get("status") == "READY" and gaps != []:
        errors.append("sales_pipeline_scope READY status cannot contain data quality gaps")
    if record.get("status") == "HOLD" and not gaps:
        errors.append("sales_pipeline_scope HOLD status requires data quality gaps")
    return errors


def validate_sales_pipeline_health(record: JsonObject) -> list[str]:
    errors = _shape_errors(record, "sales_pipeline_health/v1", _HEALTH_KEYS)
    if record.get("status") != "prepared_not_observed":
        errors.append("sales_pipeline_health status must be prepared_not_observed")
    errors.extend(_list_fields(record, "sales_pipeline_health", (
        "movement", "aging", "stalls", "slips", "exit_criteria_gaps", "next_step_quality", "duplicate_ids",
        "missing_owner_ids", "deal_exceptions",
    )))
    if record.get("duplicate_ids") != [] or record.get("missing_owner_ids") != []:
        errors.append("sales_pipeline_health calculations require duplicate-free owned opportunities")
    concentration = record.get("concentration")
    if not isinstance(concentration, dict) or set(concentration) != {
        "total_amount_minor", "account_shares", "owner_shares", "threshold_basis_points", "exceptions"
    }:
        errors.append("sales_pipeline_health concentration is invalid")
    return errors


def validate_sales_forecast_assessment(record: JsonObject) -> list[str]:
    errors = _shape_errors(record, "sales_forecast_assessment/v1", _FORECAST_KEYS)
    if record.get("status") != "prepared_not_observed":
        errors.append("sales_forecast_assessment status must be prepared_not_observed")
    errors.extend(_list_fields(record, "sales_forecast_assessment", (
        "seller_states", "scenarios", "buyer_commitments", "evidence_limits",
    )))
    errors.extend(_row_shape_errors(record, "sales_forecast_assessment", "seller_states", frozenset(
        ("opportunity_id", "category_id", "probability_basis_points", "state_source")
    )))
    errors.extend(_row_shape_errors(record, "sales_forecast_assessment", "scenarios", frozenset(
        ("scenario_id", "amount_minor", "basis", "buyer_commitment_inferred")
    )))
    errors.extend(_row_shape_errors(record, "sales_forecast_assessment", "buyer_commitments", frozenset(
        ("opportunity_id", "state", "evidence_refs", "state_source")
    )))
    if record.get("authoritative_finance_forecast") is not False:
        errors.append("sales_forecast_assessment cannot be authoritative finance forecast")
    weighted = record.get("weighted_seller_scenario")
    if not isinstance(weighted, dict) or set(weighted) != {"amount_minor", "excluded_opportunity_ids", "basis"}:
        errors.append("sales_forecast_assessment weighted seller scenario is invalid")
    calibration = record.get("calibration")
    calibration_keys = {
        "availability", "basis", "matched_opportunity_ids", "prior_forecast_amount_minor",
        "observed_won_amount_minor", "absolute_error_minor", "excluded_reason",
    }
    if (
        not isinstance(calibration, dict)
        or set(calibration) != calibration_keys
        or calibration.get("availability") not in ("available", "unavailable")
    ):
        errors.append("sales_forecast_assessment calibration is invalid")
    return errors


def validate_sales_outcome_learning_annex(record: JsonObject) -> list[str]:
    errors = _shape_errors(record, "sales_outcome_learning_annex/v1", _OUTCOME_KEYS)
    if record.get("status") != "prepared_not_observed":
        errors.append("sales_outcome_learning_annex status must be prepared_not_observed")
    errors.extend(_list_fields(record, "sales_outcome_learning_annex", (
        "outcome_counts", "reason_counts", "contradictions", "evidence_gaps", "research_follow_ups",
    )))
    if record.get("causal_claims_established") is not False:
        errors.append("sales_outcome_learning_annex cannot establish causal claims")
    return errors


def validate_sales_renewal_risk_annex(record: JsonObject) -> list[str]:
    errors = _shape_errors(record, "sales_renewal_risk_annex/v1", _RENEWAL_KEYS)
    if record.get("status") != "prepared_not_observed":
        errors.append("sales_renewal_risk_annex status must be prepared_not_observed")
    errors.extend(_list_fields(record, "sales_renewal_risk_annex", (
        "signals", "open_risks", "evidence_gaps", "contradictions", "expansion_hypotheses",
    )))
    return errors


def validate_sales_pipeline_handoff(record: JsonObject) -> list[str]:
    errors = _shape_errors(record, "sales_pipeline_handoff/v1", _HANDOFF_KEYS)
    if record.get("status") != "prepared_not_observed":
        errors.append("sales_pipeline_handoff status must be prepared_not_observed")
    errors.extend(_list_fields(record, "sales_pipeline_handoff", (
        "selected_account_follow_ups", "proposed_crm_corrections", "specialist_routes",
    )))
    connector = record.get("connector")
    if connector != {"availability": "unavailable", "mutation_observed": False, "delivery_observed": False}:
        errors.append("sales_pipeline_handoff connector boundary is invalid")
    errors.extend(_row_shape_errors(record, "sales_pipeline_handoff", "selected_account_follow_ups", frozenset(
        ("opportunity_id", "owner_id", "due_on", "action_id", "exit_criterion_id", "evidence_refs", "state")
    )))
    correction_keys = frozenset(
        ("object_id", "object_type", "field_id", "proposed_value", "evidence_refs", "owner_id", "approval_state")
    )
    if _row_shape_errors(record, "sales_pipeline_handoff", "proposed_crm_corrections", correction_keys):
        errors.append("sales_pipeline_handoff proposed CRM correction is incomplete")
    return errors


_VALIDATORS: Final[dict[str, Callable[[JsonObject], list[str]]]] = {
    "sales_pipeline_scope/v1": validate_sales_pipeline_scope,
    "sales_pipeline_health/v1": validate_sales_pipeline_health,
    "sales_forecast_assessment/v1": validate_sales_forecast_assessment,
    "sales_outcome_learning_annex/v1": validate_sales_outcome_learning_annex,
    "sales_renewal_risk_annex/v1": validate_sales_renewal_risk_annex,
    "sales_pipeline_handoff/v1": validate_sales_pipeline_handoff,
}


def validate_sales_pipeline_artifact(record: JsonValue) -> list[str]:
    if not isinstance(record, dict):
        return ["sales pipeline artifact must be an object"]
    schema = record.get("schema_version")
    if not isinstance(schema, str) or schema not in _SCHEMA_KEYS:
        return ["sales pipeline artifact schema_version is unsupported"]
    return _VALIDATORS[schema](record)
