"""Metadata-only health journal path for observed paired-run execution."""

from __future__ import annotations

from pathlib import Path

from ..runtime.critical_path_health_models import CriticalPathHealthEvent
from ..system.local_store import append_jsonl_locked
from ..system.metadata_safety import require_opaque_metadata_ref
from ..system.paths import OmhPaths


def paired_run_health_events_path(paths: OmhPaths, decision_id: str) -> Path:
    """Return the contained journal path for one paired decision."""
    safe_id = require_opaque_metadata_ref(decision_id, field="decision_id")
    if safe_id in {".", ".."} or "/" in safe_id or "\\" in safe_id:
        raise ValueError("decision_id must be one safe path segment")
    return (
        paths.omh_home
        / "coding"
        / "paired-run"
        / safe_id
        / "critical_path_health_events.jsonl"
    )


def write_paired_run_health_event(
    paths: OmhPaths,
    decision_id: str,
    event: CriticalPathHealthEvent,
) -> None:
    """Append one validated paired-run lifecycle event privately."""
    append_jsonl_locked(
        paired_run_health_events_path(paths, decision_id),
        event.to_dict(),
        private=True,
    )
