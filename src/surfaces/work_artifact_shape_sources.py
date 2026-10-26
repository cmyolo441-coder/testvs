"""Typed projection of persisted wrapper payloads into per-artifact shape sources.

This module owns the parsing half of the shape surface: it reads the same
already-persisted payloads the copy manifest reads (the coding briefing and
the session's prepared handoffs) and narrows them into mapping-typed shape
sources carrying exact source schemas, evidence states, and field-level refs.
It performs no I/O, mutates nothing, and never upgrades a prepared artifact
into observed evidence; rendering and availability decisions stay with the
committed ``show_shape`` facade.
"""

from __future__ import annotations

from collections.abc import Mapping

from .work_artifact_briefing_shapes import briefing_shape_source
from .work_artifact_copy import _ARTIFACT_IDS


def work_artifact_shape_sources(
    briefing: Mapping[str, object],
    *,
    prompt_handoff: Mapping[str, object] | None = None,
    runtime_handoff: Mapping[str, object] | None = None,
    session_id: str = "",
) -> dict[str, Mapping[str, object] | None]:
    """Project each copy artifact's shape source from the same persisted payloads.

    Reads only the payloads the copy manifest already reads: the coding
    briefing and the session's prepared handoffs. The handoff artifact prefers
    the runtime handoff when both exist; a prompt handoff beside it is still
    projected with its exact schema and is traced through its own registered
    shape projector.
    """

    prefix_session = session_id or str(briefing.get("session_id", ""))
    sources: dict[str, Mapping[str, object] | None] = {}
    for artifact_id in _ARTIFACT_IDS:
        if artifact_id == "handoff_prompt":
            if runtime_handoff:
                sources[artifact_id] = _handoff_shape_source(
                    prefix_session, runtime_handoff
                )
            elif prompt_handoff:
                sources[artifact_id] = _prompt_handoff_shape_source(
                    prefix_session, prompt_handoff
                )
            else:
                sources[artifact_id] = None
        else:
            sources[artifact_id] = (
                briefing_shape_source(artifact_id, prefix_session, briefing)
                if briefing
                else None
            )
    return sources


def _handoff_shape_source(
    session_id: str, handoff: Mapping[str, object]
) -> dict[str, object]:
    """Project the prepared handoff's own fields into a facade shape source.

    Every node and edge is emitted only when its backing field is recorded, and
    carries a source ref naming that exact field, so nothing structural is
    inferred from an absent contract.
    """

    status = str(handoff.get("status", ""))
    profile = _mapping(handoff.get("runtime_profile"))
    owner = str(
        handoff.get("selected_executor_profile") or profile.get("profile") or ""
    )
    prefix = f"{session_id}#runtime_handoff"
    nodes: list[dict[str, object]] = []
    if status:
        nodes.append(
            _shape_node(
                "handoff",
                "Prepared runtime handoff",
                (f"{prefix}.status",),
                state=status,
                owner=owner,
            )
        )
    if profile:
        nodes.append(
            _shape_node(
                "executor",
                str(profile.get("label") or "Selected runtime"),
                (f"{prefix}.runtime_profile",),
                state=status,
                owner=owner,
            )
        )
    observation = _mapping(handoff.get("observation_contract"))
    if observation:
        nodes.append(
            _shape_node(
                "observation",
                "Runtime observation",
                (f"{prefix}.observation_contract",),
                state="not_observed",
                owner="omh",
            )
        )
    edges: list[dict[str, object]] = []
    if status and profile and handoff.get("dispatch_contract"):
        edges.append(
            _shape_edge(
                "handoff", "executor", (f"{prefix}.dispatch_contract",), "dispatches"
            )
        )
    if profile and observation:
        edges.append(
            _shape_edge(
                "executor",
                "observation",
                (f"{prefix}.observation_contract",),
                "reports",
            )
        )
    return {
        "source_artifact_id": prefix,
        "source_schema": str(handoff.get("schema_version", "")),
        "evidence_state": status,
        "nodes": nodes,
        "edges": edges,
        # The runtime's own ownership list, bounded by the facade's bullet budget.
        "bullets": _string_list(
            _mapping(handoff.get("runtime_brief")).get("runtime_owns")
        ),
    }


def _prompt_handoff_shape_source(
    session_id: str, handoff: Mapping[str, object]
) -> dict[str, object]:
    owner = str(handoff.get("selected_executor_profile", ""))
    prefix = f"{session_id}#prompt_handoff"
    nodes = [
        _shape_node(
            "handoff",
            "Prepared handoff",
            (f"{prefix}.schema_version",),
            state="prepared_not_observed",
            owner=owner,
        ),
        _shape_node(
            "executor",
            "Selected executor",
            (f"{prefix}.selected_executor_profile",),
            state="prepared_not_observed",
            owner=owner,
        ),
    ]
    return {
        "source_artifact_id": prefix,
        "source_schema": str(handoff.get("schema_version", "")),
        "evidence_state": "prepared_not_observed",
        "nodes": nodes,
        "edges": [
            _shape_edge(
                "handoff",
                "executor",
                (f"{prefix}.selected_executor_profile",),
                "targets",
            )
        ],
        "bullets": [],
    }


def _shape_node(
    node_id: str, label: str, refs: tuple[str, ...], *, state: str = "", owner: str = ""
) -> dict[str, object]:
    return {
        "node_id": node_id,
        "label": label,
        "source_refs": list(refs),
        "state": state,
        "owner": owner,
        "change": "unknown",
    }


def _shape_edge(
    source_id: str, target_id: str, refs: tuple[str, ...], label: str
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "target_id": target_id,
        "source_refs": list(refs),
        "label": label,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, dict) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]
