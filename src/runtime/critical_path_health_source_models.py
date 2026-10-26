"""Read-only result shape for journal-backed critical-path health."""

from __future__ import annotations

from dataclasses import dataclass

from .critical_path_health_models import CriticalPathHealthEvent, CriticalPathHealthProjection


CRITICAL_PATH_HEALTH_SOURCES_SCHEMA_VERSION = "critical_path_health_sources/v1"
CRITICAL_PATH_HEALTH_SOURCES_CLAIM_BOUNDARY = (
    "This read-only projection maps complete dispatcher lifecycle observations to process timing. "
    "It does not claim verification, review, CI, merge, command execution details, or private payloads."
)


@dataclass(frozen=True)
class CriticalPathHealthSourceResult:
    """Committed event inputs and their unavailable-or-exact health record."""

    fanout_id: str
    events: tuple[CriticalPathHealthEvent, ...]
    record: CriticalPathHealthProjection
    evidence_gaps: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": CRITICAL_PATH_HEALTH_SOURCES_SCHEMA_VERSION,
            "privacy": "metadata_only",
            "fanout_id": self.fanout_id,
            "event_inputs": [event.to_dict() for event in self.events],
            "critical_path_health": self.record.to_dict(),
            "evidence_gaps": [
                {"task_id": task_id, "code": code} for task_id, code in self.evidence_gaps
            ],
            "claim_boundary": CRITICAL_PATH_HEALTH_SOURCES_CLAIM_BOUNDARY,
        }
