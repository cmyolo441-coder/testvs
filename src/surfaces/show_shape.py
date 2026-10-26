from __future__ import annotations

from .show_shape_models import (
    Availability,
    ChangeState,
    GenericShapeInput,
    HandoffShapeInput,
    PlanShapeInput,
    ReviewShapeInput,
    ShapeEdge,
    ShapeFormat,
    ShapeLens,
    ShapeNode,
    ShapeOmission,
    ShowShapeCapabilities,
    StatusShapeInput,
    UNCHANGED_SOURCE_CLAIM_BOUNDARY,
    WORK_ARTIFACT_SHOW_SHAPE_SCHEMA_VERSION,
    WorkArtifactShapeInput,
    WorkArtifactShowShape,
    MAX_BODY_CHARS,
)
from .show_shape_projector import bounded_bullets, bounded_graph, bounded_omissions, coerce_source
from .show_shape_render import legend, render_body
from .show_shape_validation import safe_text, source_error


def build_work_artifact_show_shape(
    source: object,
    *,
    lens: str = "flow",
    format: str = "ascii",
    capabilities: ShowShapeCapabilities = ShowShapeCapabilities(),
) -> WorkArtifactShowShape:
    """Render a bounded, source-cited structural projection without inference."""
    selected_lens = _shape_lens(lens)
    selected_format = _shape_format(format)
    artifact, source_reason = coerce_source(source)
    if selected_lens is None:
        return _unavailable("unsupported_lens", artifact, selected_format=selected_format)
    if selected_format is None:
        return _unavailable("unsupported_format", artifact, selected_lens=selected_lens)
    if artifact is None:
        return _unavailable(source_reason, None, selected_lens=selected_lens, selected_format=selected_format)
    if selected_format == "mermaid" and not capabilities.mermaid_observed:
        return _unavailable("mermaid_capability_not_observed", artifact, selected_lens=selected_lens, selected_format=selected_format)
    validation_reason = source_error(artifact, selected_lens)
    if validation_reason:
        return _unavailable(validation_reason, artifact, selected_lens=selected_lens, selected_format=selected_format)
    if selected_format == "diff" and selected_lens != "change":
        return _unavailable("format_requires_change_lens", artifact, selected_lens=selected_lens, selected_format=selected_format)
    nodes, edges, omissions = bounded_graph(artifact)
    bullets, bullet_omissions = bounded_bullets(artifact.bullets)
    omissions = bounded_omissions((*artifact.omissions, *omissions, *bullet_omissions))
    body = render_body(artifact, selected_lens, selected_format, nodes, edges)
    if len(body) > MAX_BODY_CHARS:
        return _unavailable("render_budget_exhausted", artifact, selected_lens=selected_lens, selected_format=selected_format, omissions=omissions)
    return WorkArtifactShowShape(
        availability="available",
        reason="",
        source_artifact_id=artifact.source_artifact_id,
        source_schema=artifact.source_schema,
        lens=selected_lens,
        format=selected_format,
        evidence_state=artifact.evidence_state,
        body=body,
        bullets=bullets,
        legend=legend(selected_lens, selected_format),
        nodes=nodes,
        edges=edges,
        omissions=omissions,
    )


def _unavailable(
    reason: str,
    source: WorkArtifactShapeInput | None,
    *,
    selected_lens: ShapeLens | None = None,
    selected_format: ShapeFormat | None = None,
    omissions: tuple[ShapeOmission, ...] = (),
) -> WorkArtifactShowShape:
    safe_source = source if source is not None and safe_text(source.source_artifact_id) and safe_text(source.source_schema) else None
    return WorkArtifactShowShape(
        availability="unavailable",
        reason=reason,
        source_artifact_id=safe_source.source_artifact_id if safe_source else "",
        source_schema=safe_source.source_schema if safe_source else "",
        lens=selected_lens or "flow",
        format=selected_format or "ascii",
        evidence_state=safe_source.evidence_state if safe_source and safe_text(safe_source.evidence_state) else "",
        body="",
        bullets=(),
        legend=("Shape unavailable; no structure, causality, or evidence is inferred.",),
        nodes=(),
        edges=(),
        omissions=omissions,
    )


def _shape_lens(value: str) -> ShapeLens | None:
    if value == "flow":
        return "flow"
    if value == "structure":
        return "structure"
    if value == "change":
        return "change"
    if value == "state":
        return "state"
    if value == "ownership":
        return "ownership"
    return None


def _shape_format(value: str) -> ShapeFormat | None:
    if value == "ascii":
        return "ascii"
    if value == "tree":
        return "tree"
    if value == "diff":
        return "diff"
    if value == "mermaid":
        return "mermaid"
    return None


__all__ = [
    "Availability", "ChangeState", "GenericShapeInput", "HandoffShapeInput", "PlanShapeInput", "ReviewShapeInput",
    "ShapeEdge", "ShapeFormat", "ShapeLens", "ShapeNode", "ShapeOmission", "ShowShapeCapabilities", "StatusShapeInput",
    "UNCHANGED_SOURCE_CLAIM_BOUNDARY", "WORK_ARTIFACT_SHOW_SHAPE_SCHEMA_VERSION", "WorkArtifactShapeInput",
    "WorkArtifactShowShape", "build_work_artifact_show_shape",
]
