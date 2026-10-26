from __future__ import annotations

from .show_shape_models import (
    HandoffShapeInput,
    PlanShapeInput,
    ReviewShapeInput,
    ShapeLens,
    StatusShapeInput,
    WorkArtifactShapeInput,
)


PLAN_SCHEMAS = frozenset(("work_plan/v1", "plan/v1"))
STATUS_SCHEMAS = frozenset(("work_status/v1", "status/v1"))
HANDOFF_SCHEMAS = frozenset(
    ("coding_prompt_handoff/v1", "coding_runtime_handoff/v1", "handoff/v1")
)
REVIEW_SCHEMAS = frozenset(("review_change/v1", "review/v1"))
BRIEFING_SCHEMAS = frozenset(("coding_briefing/v1",))
MAX_FIELD_CHARS = 80
SENSITIVE_MARKERS = ("api_key", "authorization", "credential", "password", "secret", "token", "prompt")


def source_error(source: WorkArtifactShapeInput, lens: ShapeLens) -> str:
    if source.source_schema not in schemas_for(source):
        return "unsupported_source_schema"
    if not source.source_artifact_id or not source.evidence_state:
        return "missing_source_metadata"
    if not source.nodes:
        return "insufficient_shape_data"
    if not safe_references(
        (source.source_artifact_id, source.source_schema, source.evidence_state)
    ):
        return "unsafe_source_content"
    if lens not in lenses_for(source):
        return "lens_not_supported_for_source_schema"
    node_ids: set[str] = set()
    for node in source.nodes:
        if not node.node_id or node.node_id in node_ids:
            return "invalid_node_id"
        node_ids.add(node.node_id)
        if not node.source_refs:
            return "missing_node_source_refs"
        if not safe_reference(node.node_id) or not safe_text(node.label):
            return "unsafe_source_content"
        if not safe_references(node.source_refs):
            return "unsafe_source_content"
        if node.state and not safe_text(node.state):
            return "unsafe_source_content"
        if node.owner and not safe_text(node.owner):
            return "unsafe_source_content"
        if lens == "state" and not safe_text(node.state):
            return "insufficient_state_data"
        if lens == "ownership" and not safe_text(node.owner):
            return "insufficient_ownership_data"
        if lens == "change" and node.change == "unknown":
            return "insufficient_change_data"
    for edge in source.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            return "unknown_edge_endpoint"
        if not edge.source_refs:
            return "missing_edge_source_refs"
        if not safe_references(
            (edge.source_id, edge.target_id, *edge.source_refs)
        ) or (edge.label and not safe_text(edge.label)):
            return "unsafe_source_content"
    return ""


def schemas_for(source: WorkArtifactShapeInput) -> frozenset[str]:
    if isinstance(source, PlanShapeInput):
        return PLAN_SCHEMAS
    if isinstance(source, StatusShapeInput):
        return STATUS_SCHEMAS
    if isinstance(source, HandoffShapeInput):
        return HANDOFF_SCHEMAS
    if isinstance(source, ReviewShapeInput):
        return REVIEW_SCHEMAS
    return all_supported_schemas()


def lenses_for(source: WorkArtifactShapeInput) -> frozenset[str]:
    if isinstance(source, PlanShapeInput):
        return frozenset(("flow", "structure", "state", "change"))
    if isinstance(source, StatusShapeInput):
        return frozenset(("state", "structure"))
    if isinstance(source, HandoffShapeInput):
        return frozenset(("flow", "ownership", "structure", "state"))
    if isinstance(source, ReviewShapeInput):
        return frozenset(("change", "structure"))
    return lenses_for_schema(source.source_schema)


def lenses_for_schema(schema: str) -> frozenset[str]:
    if schema in PLAN_SCHEMAS:
        return frozenset(("flow", "structure", "state", "change"))
    if schema in STATUS_SCHEMAS:
        return frozenset(("state", "structure"))
    if schema in HANDOFF_SCHEMAS:
        return frozenset(("flow", "ownership", "structure", "state"))
    if schema in REVIEW_SCHEMAS:
        return frozenset(("change", "structure"))
    if schema in BRIEFING_SCHEMAS:
        return frozenset(("flow", "structure", "change", "state", "ownership"))
    return frozenset()


def all_supported_schemas() -> frozenset[str]:
    return (
        PLAN_SCHEMAS
        | STATUS_SCHEMAS
        | HANDOFF_SCHEMAS
        | REVIEW_SCHEMAS
        | BRIEFING_SCHEMAS
    )


def safe_values(values: tuple[str, ...]) -> bool:
    return all(safe_text(value) for value in values)


def safe_references(values: tuple[str, ...]) -> bool:
    return all(safe_reference(value) for value in values)


def safe_reference(value: str) -> bool:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > MAX_FIELD_CHARS
        or "\n" in normalized
        or "\r" in normalized
    ):
        return False
    folded = normalized.casefold()
    folded = folded.replace("prompt_handoff", "").replace("handoff_prompt", "")
    return not any(marker in folded for marker in SENSITIVE_MARKERS)


def safe_text(value: str) -> bool:
    normalized = value.strip()
    if not normalized or len(normalized) > MAX_FIELD_CHARS or "\n" in normalized or "\r" in normalized:
        return False
    folded = normalized.casefold()
    return not any(marker in folded for marker in SENSITIVE_MARKERS)
