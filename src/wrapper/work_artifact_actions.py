from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..surfaces.show_shape_models import ShowShapeCapabilities
from ..surfaces.work_artifact_copy import (
    WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION,
    build_work_artifact_copy_manifest,
    select_work_artifact,
)
from ..surfaces.work_artifact_shapes import (
    SHAPE_CLAIM_BOUNDARY,
    show_work_artifact_shape,
    work_artifact_shape_availability,
    work_artifact_shape_sources,
)

LIST_ACTION = "list_work_artifacts"
SELECT_ACTION = "select_work_artifact"
SHOW_SHAPE_ACTION = "show_work_artifact_shape"


def build_work_artifact_copy_action(
    status_payload: dict[str, Any],
    *,
    artifact_id: str = "",
) -> dict[str, Any]:
    """List the current work item's copyable artifacts, or hand back exactly one.

    Reads an already-built ``wrapper_session_result/v1`` payload; it performs no
    session I/O, records nothing, and leaves ``next_action`` at ``show_status``
    so copying a block never advances the session toward dispatch or evidence.
    """

    briefing = _object(status_payload.get("coding_briefing"))
    manifest = build_work_artifact_copy_manifest(
        briefing,
        prompt_handoff=_object(status_payload.get("prompt_handoff")),
    )
    base = {
        "schema_version": WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION,
        "session_id": str(status_payload.get("session_id", "")),
        "next_action": "show_status",
        "claim_boundary": str(manifest["claim_boundary"]),
    }
    if artifact_id:
        return {
            **base,
            "action": SELECT_ACTION,
            "artifact": select_work_artifact(manifest, artifact_id),
        }
    shapes = work_artifact_shape_availability(_shape_sources(status_payload, briefing))
    return {
        **base,
        "action": LIST_ACTION,
        # The listing is an index: ids, labels, and availability only. Text
        # comes back one selected block at a time, so a picker cannot spill
        # every artifact into chat at once.
        "artifacts": [_listed(entry, shapes) for entry in manifest["artifacts"]],
    }


def build_work_artifact_show_shape_action(
    status_payload: Mapping[str, object],
    *,
    artifact_id: str = "",
    lens: str = "flow",
    format: str = "ascii",
    capabilities: ShowShapeCapabilities = ShowShapeCapabilities(),
) -> dict[str, object]:
    """Render one listed artifact's structural shape, or report it unavailable.

    Reads the same already-built ``wrapper_session_result/v1`` payload as the
    copy actions, passes the artifact's exact source schema, evidence state, and
    field-level refs into the committed show_shape facade, and leaves
    ``next_action`` at ``show_status``: showing a shape never advances the
    session toward dispatch or evidence.
    """

    briefing = _object(status_payload.get("coding_briefing"))
    return {
        "schema_version": WORK_ARTIFACT_COPY_MANIFEST_SCHEMA_VERSION,
        "session_id": str(status_payload.get("session_id", "")),
        "next_action": "show_status",
        "claim_boundary": SHAPE_CLAIM_BOUNDARY,
        "action": SHOW_SHAPE_ACTION,
        "artifact_id": artifact_id,
        "shape": show_work_artifact_shape(
            _shape_sources(status_payload, briefing),
            artifact_id,
            lens=lens,
            format=format,
            capabilities=capabilities,
        ),
    }


def _shape_sources(
    status_payload: Mapping[str, object], briefing: Mapping[str, object]
) -> dict[str, Mapping[str, object] | None]:
    return work_artifact_shape_sources(
        briefing,
        prompt_handoff=_object(status_payload.get("prompt_handoff")),
        runtime_handoff=_object(status_payload.get("runtime_handoff")),
        session_id=str(status_payload.get("session_id", "")),
    )


def _listed(
    entry: Mapping[str, object], shapes: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    listed = {key: value for key, value in entry.items() if key != "text"}
    # Shape availability rides the same index discipline: lenses and
    # availability only, never shape bodies.
    listed["shape"] = shapes[str(entry["artifact_id"])]
    return listed


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
