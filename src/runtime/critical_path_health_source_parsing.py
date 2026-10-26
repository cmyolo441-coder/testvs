"""Concrete JSON-shape narrowing for critical-path health sources."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timezone
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


ObjectMapping = Mapping[str, object]


def object_mapping(value: object) -> ObjectMapping | None:
    """Return a string-keyed mapping without letting untyped JSON cross the boundary."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return {key: item for key, item in value.items()}


def object_mappings(value: object) -> list[ObjectMapping] | None:
    """Return a complete list of string-keyed mappings, or reject its whole shape."""
    if not isinstance(value, list):
        return None
    values: list[ObjectMapping] = []
    for item in value:
        mapped = object_mapping(item)
        if mapped is None:
            return None
        values.append(mapped)
    return values


def string_values(value: object) -> list[str] | None:
    """Return a list's string values, preserving order and rejecting mixed JSON."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def timestamp_ms(value: object) -> int | None:
    """Convert one timezone-aware ISO-8601 observation timestamp to milliseconds."""
    try:
        parsed = strict_fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)
