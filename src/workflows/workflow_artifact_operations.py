"""Closed metadata-only operations for the implemented workflow artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any, Final

from ..runtime.decision_prototypes import persist_decision_prototype
from ..system.paths import OmhPaths
from .decision_prototypes import (
    compact_decision_prototype_receipt,
    observe_decision_prototype,
    prepare_decision_prototype,
    validate_decision_prototype,
)
from .decision_receipt_handoffs import build_channel_aware_product_brief_handoff, build_decision_receipt_handoff
from .lifecycle_growth_contracts import (
    evaluate_lifecycle_growth,
    prepare_lifecycle_growth,
    readout_lifecycle_growth,
    validate_lifecycle_growth_artifact,
)
from .lifecycle_growth_launch import prepare_lifecycle_launch
from .lifecycle_growth_configuration import build_configuration_binding
from .lifecycle_growth_metrics import build_metric_coverage
from .lifecycle_growth_configuration_values import ConfigurationInputError
from .product_discovery_validation import (
    append_product_discovery_artifact,
    discovery_audience_gate,
    discovery_audience_gate_with_feedback,
    evaluate_channel_feedback,
    evaluate_product_discovery,
    prepare_product_discovery,
    validate_product_discovery_artifact,
)
from .sales_pipeline_artifacts import validate_sales_pipeline_artifact
from .sales_pipeline_handoff import prepare_sales_pipeline_handoff
from .sales_pipeline_review import evaluate_sales_pipeline_review, prepare_sales_pipeline_review
from .workflow_artifact_operations_build import (
    build_lifecycle_growth_artifacts,
    build_channel_feedback_ledger_input,
    build_product_discovery_package,
)
from .workflow_artifact_operations_sales import (
    SalesWorkflowArtifactInputError,
    sales_handoff_input,
    sales_review_input,
)


class WorkflowArtifactOperationError(ValueError):
    """Raised when a closed workflow artifact operation cannot accept its input."""


Operation = Callable[[OmhPaths, Mapping[str, Any]], dict[str, Any]]
WORKFLOW_ARTIFACT_OPERATIONS: Final[dict[str, tuple[str, ...]]] = {
    "decision-prototype": ("prepare", "validate", "observe", "receipt", "handoff", "persist"),
    "lifecycle-growth": ("build", "prepare", "validate", "evaluate", "readout", "audience", "promote", "graduate", "configuration", "metrics"),
    "product-discovery-validation": ("build", "prepare", "validate", "audience-gate", "evaluate", "handoff", "append", "channel-feedback"),
    "sales-pipeline-review": ("prepare", "validate", "evaluate", "handoff"),
}


def run_workflow_artifact_operation(
    paths: OmhPaths, workflow: str, operation: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one supported metadata-only operation without retaining request bodies."""
    handler = _DISPATCH.get((workflow, operation))
    if handler is None:
        raise WorkflowArtifactOperationError("workflow artifact operation is unsupported")
    try:
        result = handler(paths, payload)
    except (SalesWorkflowArtifactInputError, ValueError) as exc:
        raise WorkflowArtifactOperationError(str(exc)) from exc
    return {
        "schema_version": "workflow_artifact_operation_result/v1",
        "workflow": workflow,
        "operation": operation,
        "result": result,
        "claim_boundary": "This operation handles bounded metadata only. It does not schedule, execute, send, mutate a CRM, or promote production work.",
    }


def _prototype_prepare(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return prepare_decision_prototype(payload)


def _prototype_validate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"valid": not validate_decision_prototype(payload), "errors": validate_decision_prototype(payload)}


def _prototype_observe(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return observe_decision_prototype(_required_mapping(payload, "artifact"), _required_mapping(payload, "observation"))


def _prototype_receipt(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return compact_decision_prototype_receipt(payload)


def _prototype_handoff(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_decision_receipt_handoff(payload, target_workflow="ralplan")


def _prototype_persist(paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return persist_decision_prototype(paths, payload)


def _lifecycle_build(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_lifecycle_growth_artifacts(payload)


def _lifecycle_prepare(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) - {"brief", "audience", "safety", "experiment", "handoff", "readout", "exposure_evidence",
                       "evaluation_context", "audience_review", "configuration_binding", "metric_coverage", "metric_plan_binding"}:
        raise ConfigurationInputError("configuration_operation_keys_invalid")
    return prepare_lifecycle_growth(payload)


def _lifecycle_configuration(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(build_configuration_binding(payload))


def _lifecycle_metrics(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return dict(build_metric_coverage(payload))


def _lifecycle_validate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_lifecycle_growth_artifact(payload)
    return {"valid": not errors, "errors": errors}


def _lifecycle_evaluate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) - {"experiment", "readout", "evaluation_context", "exposure_evidence", "audience_review", "configuration_binding", "metric_coverage", "metric_plan_binding"}:
        raise ConfigurationInputError("configuration_operation_keys_invalid")
    return evaluate_lifecycle_growth(
        _required_mapping(payload, "experiment"), _required_mapping(payload, "readout"),
        evaluation_context=payload.get("evaluation_context"),
        exposure_evidence=payload.get("exposure_evidence"),
        audience_review=payload.get("audience_review"),
        configuration_binding=payload.get("configuration_binding"),
        metric_coverage=payload.get("metric_coverage"), metric_plan_binding=payload.get("metric_plan_binding"),
    )


def _lifecycle_readout(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    if "readout" in payload:
        return _lifecycle_evaluate(_paths, payload)
    return readout_lifecycle_growth(payload)


def _lifecycle_audience(_paths: OmhPaths, payload: Mapping[str, object]) -> dict[str, object]:
    return prepare_lifecycle_launch("audience", payload)


def _lifecycle_promote(_paths: OmhPaths, payload: Mapping[str, object]) -> dict[str, object]:
    return prepare_lifecycle_launch("promote", payload)


def _lifecycle_graduate(_paths: OmhPaths, payload: Mapping[str, object]) -> dict[str, object]:
    return prepare_lifecycle_launch("graduate", payload)


def _discovery_build(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return build_product_discovery_package(payload)


def _discovery_prepare(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return prepare_product_discovery(**_discovery_package(payload))


def _discovery_validate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_product_discovery_artifact(payload)
    return {"valid": not errors, "errors": errors}


def _discovery_audience_gate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return discovery_audience_gate(payload)


def _discovery_channel_feedback(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"frame", "gtm", "portfolio", "feedback_ledger", "now"}:
        raise WorkflowArtifactOperationError("channel feedback operation keys are invalid")
    frame = _required_mapping(payload, "frame")
    feedback = _required_mapping(payload, "feedback_ledger")
    ledger = build_channel_feedback_ledger_input(feedback)
    disposition = evaluate_channel_feedback(frame=frame, gtm=_required_mapping(payload, "gtm"),
        portfolio=_required_mapping(payload, "portfolio"), feedback_ledger=feedback, now=_required_string(payload, "now"))
    return {"ledger": ledger, "disposition": disposition,
            "audience_gate": discovery_audience_gate_with_feedback(frame, disposition)}


def _discovery_evaluate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return evaluate_product_discovery(_required_mapping(payload, "package"), now=_required_string(payload, "now"))


def _discovery_handoff(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    if "receipt" in payload or "disposition" in payload:
        if set(payload) != {"receipt", "disposition"}:
            raise WorkflowArtifactOperationError("channel feedback handoff keys are invalid")
        return build_channel_aware_product_brief_handoff(
            _required_mapping(payload, "receipt"), _required_mapping(payload, "disposition")
        )
    return build_decision_receipt_handoff(payload, target_workflow="product-brief")


def _discovery_append(paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return append_product_discovery_artifact(paths, payload)


def _sales_prepare(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return asdict(prepare_sales_pipeline_review(sales_review_input(payload)))


def _sales_validate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    errors = validate_sales_pipeline_artifact(payload)
    return {"valid": not errors, "errors": errors}


def _sales_evaluate(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    return asdict(evaluate_sales_pipeline_review(sales_review_input(payload)))


def _sales_handoff(_paths: OmhPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    review = sales_review_input(_required_mapping(payload, "review"))
    actions = _required_mapping(payload, "actions")
    return prepare_sales_pipeline_handoff(sales_handoff_input(actions, review))


def _discovery_package(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    required = ("frame", "ledger", "plan", "portfolio", "gtm")
    if set(payload) != set(required):
        raise WorkflowArtifactOperationError("product discovery package keys are invalid")
    return {key: _required_mapping(payload, key) for key in required}


def _required_mapping(payload: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = payload.get(field)
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise WorkflowArtifactOperationError(f"{field} must be an object")
    return value


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise WorkflowArtifactOperationError(f"{field} must be a string")
    return value


_DISPATCH: Final[dict[tuple[str, str], Operation]] = {
    ("decision-prototype", "prepare"): _prototype_prepare,
    ("decision-prototype", "validate"): _prototype_validate,
    ("decision-prototype", "observe"): _prototype_observe,
    ("decision-prototype", "receipt"): _prototype_receipt,
    ("decision-prototype", "handoff"): _prototype_handoff,
    ("decision-prototype", "persist"): _prototype_persist,
    ("lifecycle-growth", "build"): _lifecycle_build,
    ("lifecycle-growth", "prepare"): _lifecycle_prepare,
    ("lifecycle-growth", "validate"): _lifecycle_validate,
    ("lifecycle-growth", "evaluate"): _lifecycle_evaluate,
    ("lifecycle-growth", "readout"): _lifecycle_readout,
    ("lifecycle-growth", "audience"): _lifecycle_audience,
    ("lifecycle-growth", "promote"): _lifecycle_promote,
    ("lifecycle-growth", "graduate"): _lifecycle_graduate,
    ("lifecycle-growth", "configuration"): _lifecycle_configuration,
    ("lifecycle-growth", "metrics"): _lifecycle_metrics,
    ("product-discovery-validation", "build"): _discovery_build,
    ("product-discovery-validation", "prepare"): _discovery_prepare,
    ("product-discovery-validation", "validate"): _discovery_validate,
    ("product-discovery-validation", "audience-gate"): _discovery_audience_gate,
    ("product-discovery-validation", "evaluate"): _discovery_evaluate,
    ("product-discovery-validation", "handoff"): _discovery_handoff,
    ("product-discovery-validation", "append"): _discovery_append,
    ("product-discovery-validation", "channel-feedback"): _discovery_channel_feedback,
    ("sales-pipeline-review", "prepare"): _sales_prepare,
    ("sales-pipeline-review", "validate"): _sales_validate,
    ("sales-pipeline-review", "evaluate"): _sales_evaluate,
    ("sales-pipeline-review", "handoff"): _sales_handoff,
}
