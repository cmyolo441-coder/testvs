from __future__ import annotations

from typing import Mapping

from .show_shape_models import GenericShapeInput, ShapeEdge, ShapeNode, ShapeOmission, WorkArtifactShapeInput
from .show_shape_validation import all_supported_schemas, safe_text


MAX_NODES = 6
MAX_EDGES = 6
MAX_REFS_PER_ITEM = 2
MAX_OMISSIONS = 16


def coerce_source(source: object) -> tuple[WorkArtifactShapeInput | None, str]:
    if isinstance(source, WorkArtifactShapeInput):
        return source, ""
    if not isinstance(source, Mapping):
        return None, "unsupported_source_input"
    schema = value(source, "source_schema") or value(source, "schema_version")
    artifact_id = value(source, "source_artifact_id") or value(source, "artifact_id")
    evidence_state = value(source, "evidence_state")
    if schema not in all_supported_schemas():
        return None, "unsupported_source_schema"
    nodes = nodes_from_mapping(source.get("nodes"))
    edges = edges_from_mapping(source.get("edges"))
    if nodes is None or edges is None:
        return None, "insufficient_generic_shape_data"
    return (
        GenericShapeInput(
            source_artifact_id=artifact_id,
            source_schema=schema,
            evidence_state=evidence_state,
            nodes=nodes,
            edges=edges,
            bullets=strings(source.get("bullets")),
            omissions=omissions_from_mapping(source.get("omissions")),
        ),
        "",
    )


def bounded_graph(source: WorkArtifactShapeInput) -> tuple[tuple[ShapeNode, ...], tuple[ShapeEdge, ...], tuple[ShapeOmission, ...]]:
    omissions: list[ShapeOmission] = []
    kept_nodes = source.nodes[:MAX_NODES]
    if len(source.nodes) > len(kept_nodes):
        omissions.append(ShapeOmission("nodes", "node_limit"))
    nodes = tuple(bounded_node(node, omissions) for node in kept_nodes)
    kept_ids = {node.node_id for node in nodes}
    eligible_edges = tuple(edge for edge in source.edges if edge.source_id in kept_ids and edge.target_id in kept_ids)
    if len(eligible_edges) < len(source.edges):
        omissions.append(ShapeOmission("edges", "node_limit"))
    kept_edges = eligible_edges[:MAX_EDGES]
    if len(eligible_edges) > len(kept_edges):
        omissions.append(ShapeOmission("edges", "edge_limit"))
    return nodes, tuple(bounded_edge(edge, omissions) for edge in kept_edges), tuple(omissions)


def bounded_bullets(bullets: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[ShapeOmission, ...]]:
    safe = tuple(bullet for bullet in bullets if safe_text(bullet))
    omissions: list[ShapeOmission] = []
    if len(safe) < len(bullets):
        omissions.append(ShapeOmission("bullets", "unsafe_source_content"))
    selected = safe[:3]
    if len(safe) > len(selected):
        omissions.append(ShapeOmission("bullets", "bullet_limit"))
    return selected, tuple(omissions)


def bounded_omissions(omissions: tuple[ShapeOmission, ...]) -> tuple[ShapeOmission, ...]:
    safe = tuple(omission for omission in omissions if safe_text(omission.item_id) and safe_text(omission.reason))
    if len(safe) <= MAX_OMISSIONS:
        return safe
    return (*safe[: MAX_OMISSIONS - 1], ShapeOmission("omissions", "omission_limit"))


def bounded_node(node: ShapeNode, omissions: list[ShapeOmission]) -> ShapeNode:
    refs = node.source_refs[:MAX_REFS_PER_ITEM]
    if len(refs) < len(node.source_refs):
        omissions.append(ShapeOmission(f"node:{node.node_id}", "source_ref_limit"))
    return ShapeNode(node.node_id, node.label, refs, node.state, node.owner, node.change)


def bounded_edge(edge: ShapeEdge, omissions: list[ShapeOmission]) -> ShapeEdge:
    refs = edge.source_refs[:MAX_REFS_PER_ITEM]
    if len(refs) < len(edge.source_refs):
        omissions.append(ShapeOmission(f"edge:{edge.source_id}->{edge.target_id}", "source_ref_limit"))
    return ShapeEdge(edge.source_id, edge.target_id, refs, edge.label)


def value(source: Mapping[str, object], key: str) -> str:
    item = source.get(key)
    return item if isinstance(item, str) else ""


def strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def nodes_from_mapping(raw: object) -> tuple[ShapeNode, ...] | None:
    if not isinstance(raw, (list, tuple)):
        return None
    nodes: list[ShapeNode] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        change = value(item, "change")
        if change not in {"added", "removed", "modified", "unchanged", "unknown"}:
            return None
        nodes.append(
            ShapeNode(
                value(item, "node_id"),
                value(item, "label"),
                strings(item.get("source_refs")),
                value(item, "state"),
                value(item, "owner"),
                change,
            )
        )
    return tuple(nodes)


def edges_from_mapping(raw: object) -> tuple[ShapeEdge, ...] | None:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        return None
    edges: list[ShapeEdge] = []
    for item in raw:
        if not isinstance(item, Mapping):
            return None
        edges.append(ShapeEdge(value(item, "source_id"), value(item, "target_id"), strings(item.get("source_refs")), value(item, "label")))
    return tuple(edges)


def omissions_from_mapping(raw: object) -> tuple[ShapeOmission, ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(
        ShapeOmission(value(item, "item_id"), value(item, "reason"))
        for item in raw
        if isinstance(item, Mapping)
    )
