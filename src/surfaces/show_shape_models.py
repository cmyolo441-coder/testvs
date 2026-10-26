from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


WORK_ARTIFACT_SHOW_SHAPE_SCHEMA_VERSION = "work_artifact_show_shape/v1"
UNCHANGED_SOURCE_CLAIM_BOUNDARY = (
    "An unchanged marker reports only that the supplied source had no observed change marker; "
    "it does not prove source equality, execution, review, CI, or merge."
)
MAX_BODY_CHARS = 4_000

ShapeLens = Literal["flow", "structure", "change", "state", "ownership"]
ShapeFormat = Literal["ascii", "tree", "diff", "mermaid"]
Availability = Literal["available", "unavailable"]
ChangeState = Literal["added", "removed", "modified", "unchanged", "unknown"]


@dataclass(frozen=True, slots=True)
class ShapeNode:
    node_id: str
    label: str
    source_refs: tuple[str, ...]
    state: str = ""
    owner: str = ""
    change: ChangeState = "unknown"


@dataclass(frozen=True, slots=True)
class ShapeEdge:
    source_id: str
    target_id: str
    source_refs: tuple[str, ...]
    label: str = ""


@dataclass(frozen=True, slots=True)
class ShapeOmission:
    item_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class WorkArtifactShapeInput:
    source_artifact_id: str
    source_schema: str
    evidence_state: str
    nodes: tuple[ShapeNode, ...]
    edges: tuple[ShapeEdge, ...] = ()
    bullets: tuple[str, ...] = ()
    omissions: tuple[ShapeOmission, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanShapeInput(WorkArtifactShapeInput):
    pass


@dataclass(frozen=True, slots=True)
class StatusShapeInput(WorkArtifactShapeInput):
    pass


@dataclass(frozen=True, slots=True)
class HandoffShapeInput(WorkArtifactShapeInput):
    pass


@dataclass(frozen=True, slots=True)
class ReviewShapeInput(WorkArtifactShapeInput):
    pass


@dataclass(frozen=True, slots=True)
class GenericShapeInput(WorkArtifactShapeInput):
    pass


@dataclass(frozen=True, slots=True)
class ShowShapeCapabilities:
    mermaid_observed: bool = False


@dataclass(frozen=True, slots=True)
class WorkArtifactShowShape:
    availability: Availability
    reason: str
    source_artifact_id: str
    source_schema: str
    lens: ShapeLens
    format: ShapeFormat
    evidence_state: str
    body: str
    bullets: tuple[str, ...]
    legend: tuple[str, ...]
    nodes: tuple[ShapeNode, ...]
    edges: tuple[ShapeEdge, ...]
    omissions: tuple[ShapeOmission, ...]
    claim_boundary: str = UNCHANGED_SOURCE_CLAIM_BOUNDARY

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": WORK_ARTIFACT_SHOW_SHAPE_SCHEMA_VERSION,
            "availability": self.availability,
            "reason": self.reason,
            "source_artifact_id": self.source_artifact_id,
            "source_schema": self.source_schema,
            "lens": self.lens,
            "format": self.format,
            "evidence_state": self.evidence_state,
            "body": self.body,
            "bullets": list(self.bullets),
            "legend": list(self.legend),
            "nodes": [_node_dict(node) for node in self.nodes],
            "edges": [_edge_dict(edge) for edge in self.edges],
            "omissions": [_omission_dict(omission) for omission in self.omissions],
            "claim_boundary": self.claim_boundary,
        }


def _node_dict(node: ShapeNode) -> dict[str, object]:
    return {
        "node_id": node.node_id,
        "label": node.label,
        "source_refs": list(node.source_refs),
        "state": node.state,
        "owner": node.owner,
        "change": node.change,
    }


def _edge_dict(edge: ShapeEdge) -> dict[str, object]:
    return {
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "label": edge.label,
        "source_refs": list(edge.source_refs),
    }


def _omission_dict(omission: ShapeOmission) -> dict[str, str]:
    return {"item_id": omission.item_id, "reason": omission.reason}
