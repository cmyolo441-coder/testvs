from __future__ import annotations

from .show_shape_models import ShapeEdge, ShapeFormat, ShapeLens, ShapeNode, WorkArtifactShapeInput


def render_body(
    source: WorkArtifactShapeInput,
    lens: ShapeLens,
    format: ShapeFormat,
    nodes: tuple[ShapeNode, ...],
    edges: tuple[ShapeEdge, ...],
) -> str:
    if format == "mermaid":
        return render_mermaid(source, lens, nodes, edges)
    if format == "diff":
        return render_diff(source, nodes, edges)
    return render_text(source, lens, format, nodes, edges)


def render_text(
    source: WorkArtifactShapeInput,
    lens: ShapeLens,
    format: ShapeFormat,
    nodes: tuple[ShapeNode, ...],
    edges: tuple[ShapeEdge, ...],
) -> str:
    lines = [
        f"source: {source.source_artifact_id} [{source.source_schema}]",
        f"lens: {lens}; evidence: {source.evidence_state}",
    ]
    node_prefix = "-" if format == "tree" else "[node]"
    for node in nodes:
        lines.append(f"{node_prefix} {node.node_id}: {node.label}{node_detail(node, lens)} [refs: {', '.join(node.source_refs)}]")
    for edge in edges:
        label = f" ({edge.label})" if edge.label else ""
        lines.append(f"link: {edge.source_id} -> {edge.target_id}{label} [refs: {', '.join(edge.source_refs)}]")
    return "\n".join(lines)


def render_diff(source: WorkArtifactShapeInput, nodes: tuple[ShapeNode, ...], edges: tuple[ShapeEdge, ...]) -> str:
    lines = [
        f"--- {source.source_artifact_id} [{source.source_schema}]",
        f"+++ shape change; evidence: {source.evidence_state}",
    ]
    prefixes = {"added": "+", "removed": "-", "modified": "~", "unchanged": " "}
    for node in nodes:
        lines.append(f"{prefixes[node.change]} {node.label} [node: {node.node_id}; refs: {', '.join(node.source_refs)}]")
    for edge in edges:
        label = f" ({edge.label})" if edge.label else ""
        lines.append(f"  link: {edge.source_id} -> {edge.target_id}{label} [refs: {', '.join(edge.source_refs)}]")
    return "\n".join(lines)


def render_mermaid(
    source: WorkArtifactShapeInput,
    lens: ShapeLens,
    nodes: tuple[ShapeNode, ...],
    edges: tuple[ShapeEdge, ...],
) -> str:
    lines = [f"%% source: {source.source_artifact_id} [{source.source_schema}]", f"%% lens: {lens}; evidence: {source.evidence_state}", "flowchart TD"]
    identifiers = {node.node_id: f"n{index}" for index, node in enumerate(nodes, start=1)}
    for node in nodes:
        lines.append(f"  {identifiers[node.node_id]}[\"{mermaid_label(node.label, lens, node)}\"] %% refs: {', '.join(node.source_refs)}")
    for edge in edges:
        label = f"|{edge.label.replace('|', '/')}|" if edge.label else ""
        lines.append(f"  {identifiers[edge.source_id]} -->{label} {identifiers[edge.target_id]} %% refs: {', '.join(edge.source_refs)}")
    return "\n".join(lines)


def mermaid_label(label: str, lens: ShapeLens, node: ShapeNode) -> str:
    return f"{label.replace(chr(34), chr(39))}{node_detail(node, lens)}"


def node_detail(node: ShapeNode, lens: ShapeLens) -> str:
    if lens == "state":
        return f" [state: {node.state}]"
    if lens == "ownership":
        return f" [owner: {node.owner}]"
    if lens == "change":
        return f" [change: {node.change}]"
    return ""


def legend(lens: ShapeLens, format: ShapeFormat) -> tuple[str, ...]:
    format_note = "Mermaid is rendered only when the surface capability is observed." if format == "mermaid" else "Text output is safe for plain ASCII-capable surfaces."
    return (
        "Every rendered node and link carries supplied source refs; links describe source structure, not causality.",
        f"Lens: {lens}; evidence state is propagated without upgrade.",
        format_note,
    )
