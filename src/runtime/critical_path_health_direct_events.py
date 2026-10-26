"""Read the dispatcher's typed critical-path journal conservatively."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..coding.fanout_health_events import fanout_health_events_path
from ..system.paths import OmhPaths
from ..system.secure_regular_file import SecureFileError, open_regular_read, read_bounded
from .critical_path_health_models import CriticalPathHealthEvent
from .critical_path_health_source_parsing import ObjectMapping, object_mapping, string_values


MAX_HEALTH_EVENT_JOURNAL_BYTES = 1_048_576
MAX_HEALTH_EVENT_LINE_BYTES = 16_384
MAX_HEALTH_EVENT_RECORDS = 1_024


def read_direct_health_events(
    paths: OmhPaths,
    fanout_id: str,
    gaps: set[tuple[str, str]],
) -> tuple[CriticalPathHealthEvent, ...] | None:
    """Read explicit lifecycle observations, or select legacy fallback when absent."""
    try:
        path = fanout_health_events_path(paths, fanout_id)
    except ValueError:
        gaps.add(("", "health_event_journal_invalid"))
        return ()
    if not path.exists():
        return None
    return read_health_events_path(path, gaps)


def read_health_events_path(
    path: Path,
    gaps: set[tuple[str, str]],
) -> tuple[CriticalPathHealthEvent, ...]:
    """Read one explicit lifecycle journal without projecting a partial file."""
    records, error = _read_bounded_event_records(path)
    if error:
        gaps.add(("", error))
        return ()
    events: list[CriticalPathHealthEvent] = []
    for record in records:
        event = _direct_health_event(object_mapping(record))
        if event is None:
            gaps.add(("", "health_event_journal_invalid"))
            continue
        events.append(event)
    return tuple(events)


def _read_bounded_event_records(path: Path) -> tuple[list[dict[str, object]], str]:
    """Read one no-follow journal within fixed byte, line, and record limits."""
    try:
        with open_regular_read(path) as descriptor:
            if os.fstat(descriptor).st_size > MAX_HEALTH_EVENT_JOURNAL_BYTES:
                return [], "health_event_journal_limit"
            try:
                payload = read_bounded(descriptor, MAX_HEALTH_EVENT_JOURNAL_BYTES)
            except SecureFileError:
                return [], "health_event_journal_limit"
    except (OSError, SecureFileError):
        return [], "health_event_journal_invalid"
    records: list[dict[str, object]] = []
    for line in payload.splitlines():
        if len(line) > MAX_HEALTH_EVENT_LINE_BYTES:
            return [], "health_event_journal_limit"
        if not line.strip():
            continue
        if len(records) == MAX_HEALTH_EVENT_RECORDS:
            return [], "health_event_journal_limit"
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, RecursionError, UnicodeError):
            return [], "health_event_journal_invalid"
        if not isinstance(record, dict) or not all(
            isinstance(key, str) for key in record
        ):
            return [], "health_event_journal_invalid"
        records.append(record)
    return records, ""


def _direct_health_event(record: ObjectMapping | None) -> CriticalPathHealthEvent | None:
    if record is None:
        return None
    dependencies = string_values(record.get("dependencies"))
    at_ms = record.get("at_ms")
    if dependencies is None or isinstance(at_ms, bool) or not isinstance(at_ms, int):
        return None
    try:
        return CriticalPathHealthEvent(
            task_id=str(record.get("task_id", "")),
            event=str(record.get("event", "")),
            at_ms=at_ms,
            revision=str(record.get("revision", "")),
            executor=str(record.get("executor", "")),
            model=str(record.get("model", "")),
            environment=str(record.get("environment", "")),
            dependencies=tuple(dependencies),
            resource_class=str(record.get("resource_class", "")),
            phase=str(record.get("phase", "")),
            retry=record.get("retry", 0),
            terminal_status=str(record.get("terminal_status", "")),
            reused=record.get("reused", False),
            schema_version=str(record.get("schema_version", "")),
            privacy=str(record.get("privacy", "")),
        )
    except ValueError:
        return None
