"""Shared lifecycle artifact validation, preserving the six historic schemas."""
from collections.abc import Mapping

from .lifecycle_growth_artifacts import (
    validate_audience, validate_brief, validate_experiment, validate_handoff, validate_safety,
)
from .lifecycle_growth_configuration import BINDING_SCHEMA, validate_configuration_artifact
from .lifecycle_growth_configuration_identity import IDENTITY_SCHEMA
from .lifecycle_growth_exposure import is_exposure_evidence, validate_exposure_evidence
from .lifecycle_growth_readout import validate_readout
from .lifecycle_growth_metrics import METRIC_SCHEMA, MEMBER_SCHEMA, PLAN_BINDING_SCHEMA, validate_metric_artifact
from .lifecycle_growth_values import artifact_shape_errors


def validate_lifecycle_growth_artifact(record: object) -> list[str]:
    """Return structural errors without changing readable historic artifacts."""
    if isinstance(record, Mapping) and record.get("schema_version") in (IDENTITY_SCHEMA, BINDING_SCHEMA):
        return validate_configuration_artifact(record)
    if isinstance(record, Mapping) and record.get("schema_version") in (METRIC_SCHEMA, MEMBER_SCHEMA, PLAN_BINDING_SCHEMA):
        return validate_metric_artifact(record)
    if is_exposure_evidence(record):
        return validate_exposure_evidence(record)
    schema, errors = artifact_shape_errors(record)
    if not schema or not isinstance(record, Mapping):
        return errors
    validators = {
        "lifecycle_growth_brief/v1": validate_brief,
        "audience_trigger_policy/v1": validate_audience,
        "lifecycle_safety_policy/v1": validate_safety,
        "growth_experiment_plan/v1": validate_experiment,
        "growth_measurement_readout/v1": validate_readout,
        "growth_handoff_disposition/v1": validate_handoff,
    }
    validator = validators.get(schema)
    if validator is None:
        return errors + ["lifecycle growth artifact schema_version is unsupported"]
    return errors + validator(record)


def expected_errors(record: Mapping[str, object], schema: str, label: str) -> list[str]:
    if record.get("schema_version") != schema:
        return [f"{label} artifact has the wrong schema"]
    errors = validate_lifecycle_growth_artifact(record)
    return [f"{label} artifact is invalid"] if errors else []


def experiment_hold_reasons(experiment: Mapping[str, object]) -> list[str]:
    return [reason for field, value, reason in (
        ("approval_state", "approved", "human approval is absent"),
        ("holdout_state", "preserved", "holdout is not preserved"),
        ("data_health_state", "healthy", "experiment data health is not healthy"),
    ) if experiment.get(field) != value]
