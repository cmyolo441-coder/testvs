"""Public shape-surface facade: rendering and measured lens availability.

``work_artifact_shape_sources`` (the typed parser module) projects the persisted
payloads; this module renders one artifact's shape through the committed
``show_shape`` facade and measures which lenses each artifact can render now.
The committed facade stays the single availability authority: unknown
artifacts, missing sources, unsupported lenses, formats, and source schemas,
missing refs, unsafe content, unobserved Mermaid capability, and budget
exhaustion all come back as explicit unavailable results with an empty body.
"""

from __future__ import annotations

from collections.abc import Mapping

from .show_shape import build_work_artifact_show_shape
from .show_shape_models import (
    UNCHANGED_SOURCE_CLAIM_BOUNDARY,
    ShowShapeCapabilities,
    WORK_ARTIFACT_SHOW_SHAPE_SCHEMA_VERSION,
)
from .work_artifact_shape_sources import work_artifact_shape_sources

__all__ = [
    "SHAPE_CLAIM_BOUNDARY",
    "show_work_artifact_shape",
    "work_artifact_shape_availability",
    "work_artifact_shape_sources",
]

SHAPE_CLAIM_BOUNDARY = (
    "Showing an artifact's shape is not dispatch, execution, verification, review, CI, "
    "merge-readiness, or merge evidence, and it does not advance the session."
)


def show_work_artifact_shape(
    sources: Mapping[str, Mapping[str, object] | None],
    artifact_id: str,
    *,
    lens: str = "flow",
    format: str = "ascii",
    capabilities: ShowShapeCapabilities = ShowShapeCapabilities(),
) -> dict[str, object]:
    """Render one artifact's shape through the committed facade, or say why not."""

    if artifact_id not in sources:
        return _shape_unavailable(artifact_id, "unknown_artifact_id", lens, format)
    source = sources[artifact_id]
    if source is None:
        return _shape_unavailable(artifact_id, "source_not_recorded", lens, format)
    return build_work_artifact_show_shape(
        source, lens=lens, format=format, capabilities=capabilities
    ).to_dict()


def work_artifact_shape_availability(
    sources: Mapping[str, Mapping[str, object] | None],
) -> dict[str, dict[str, object]]:
    """Advertise per artifact which lenses the committed facade can render now.

    Availability is measured, not assumed: each lens is rendered through the
    same facade call the show action uses, so an advertised lens is exactly one
    that would come back available. The reported reason for an artifact with no
    renderable lens prefers a data-level reason over a lens-specific one.
    """

    availability: dict[str, dict[str, object]] = {}
    for artifact_id, source in sources.items():
        if source is None:
            availability[artifact_id] = {
                "availability": "unavailable",
                "reason": "source_not_recorded",
                "lenses": [],
            }
            continue
        lenses: list[str] = []
        first_reason = ""
        shared_reason = ""
        for lens in _SHAPE_LENSES:
            result = show_work_artifact_shape(sources, artifact_id, lens=lens)
            if result["availability"] == "available":
                lenses.append(lens)
                continue
            reason = str(result["reason"])
            if not first_reason:
                first_reason = reason
            if not shared_reason and reason != "lens_not_supported_for_source_schema":
                shared_reason = reason
        availability[artifact_id] = {
            "availability": "available" if lenses else "unavailable",
            "reason": "" if lenses else (shared_reason or first_reason),
            "lenses": lenses,
        }
    return availability


_SHAPE_LENSES = ("flow", "structure", "change", "state", "ownership")
_SHAPE_UNAVAILABLE_LEGEND = (
    "Shape unavailable; no structure, causality, or evidence is inferred.",
)


def _shape_unavailable(
    artifact_id: str, reason: str, lens: str, format: str
) -> dict[str, object]:
    # Byte-identical to the facade result's to_dict() for an unavailable shape:
    # no source identity is invented for a source this surface never read.
    return {
        "schema_version": WORK_ARTIFACT_SHOW_SHAPE_SCHEMA_VERSION,
        "availability": "unavailable",
        "reason": reason,
        "source_artifact_id": artifact_id,
        "source_schema": "",
        "evidence_state": "",
        "lens": lens,
        "format": format,
        "body": "",
        "bullets": [],
        "legend": list(_SHAPE_UNAVAILABLE_LEGEND),
        "nodes": [],
        "edges": [],
        "omissions": [],
        "claim_boundary": UNCHANGED_SOURCE_CLAIM_BOUNDARY,
    }
