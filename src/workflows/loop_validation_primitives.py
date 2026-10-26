from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Final
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


MAX_METADATA_TEXT: Final = 320
MAX_EVIDENCE_REFS: Final = 64
_STORAGE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_UTC_RFC3339 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|\+00:00)"
)
_FORBIDDEN_FIELD_PARTS: Final = frozenset(
    {
        "content", "log", "logs", "message", "prompt", "quota",
        "reasoning", "score", "scoring", "transcript",
    }
)


def key_errors(
    value: Mapping[object, object],
    allowed: frozenset[str],
    label: str,
    *,
    required: frozenset[str] | None = None,
) -> list[str]:
    """Return missing, unsupported, and non-string key errors."""
    non_string = [key for key in value if not isinstance(key, str)]
    present = {key for key in value if isinstance(key, str)}
    missing = sorted((required if required is not None else allowed) - present)
    unexpected = sorted(present - allowed)
    errors = [f"{label} has non-string keys"] if non_string else []
    if missing:
        errors.append(f"{label} is missing keys: {missing}")
    if unexpected:
        errors.append(f"{label} has unsupported keys: {unexpected}")
    return errors


def observation_reference_errors(value: object, label: str) -> list[str]:
    """Return metadata-only observation evidence errors."""
    if not isinstance(value, list) or not value:
        return [f"{label} must contain observed-progress evidence"]
    if len(value) > MAX_EVIDENCE_REFS:
        return [f"{label} must contain at most {MAX_EVIDENCE_REFS} refs"]
    if len(set(item for item in value if isinstance(item, str))) != len(value):
        return [f"{label} must contain unique metadata refs"]
    if not all(is_metadata_text(item) for item in value):
        return [
            f"{label} must contain single-line metadata refs of at most "
            f"{MAX_METADATA_TEXT} characters"
        ]
    return []


def forbidden_fields(value: object) -> set[str]:
    """Return recursively discovered fields forbidden from metadata records."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and _FORBIDDEN_FIELD_PARTS.intersection(
                re.split(r"[^a-z]+", key.lower())
            ):
                found.add(key)
            found.update(forbidden_fields(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            found.update(forbidden_fields(child))
    return found


def is_storage_id(value: object) -> bool:
    """Return whether a value is safe for use as a storage identifier."""
    return (
        isinstance(value, str)
        and _STORAGE_ID.fullmatch(value) is not None
        and ".." not in value
    )


def is_metadata_text(value: object) -> bool:
    """Return whether a value is bounded single-line metadata."""
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= MAX_METADATA_TEXT
        and "\n" not in value
        and "\r" not in value
    )


def is_utc_rfc3339(value: object) -> bool:
    """Return whether a value is a parseable UTC RFC3339 timestamp."""
    if not isinstance(value, str) or _UTC_RFC3339.fullmatch(value) is None:
        return False
    try:
        strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
