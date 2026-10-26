"""Small closed-shape checks shared by the paired-run parser."""

from __future__ import annotations

from typing import Final, TypeVar

from .paired_run_values import is_safe_metadata_ref, is_utc_z

_BoundaryValue = TypeVar("_BoundaryValue")
_BANNED_NAMES: Final = ("raw", "hidden", "score", "grade", "rank", "resource", "token")


def safe_ref_error(value: _BoundaryValue, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not is_safe_metadata_ref(value):
        errors.append(f"{field} must be a safe opaque metadata reference")


def utc_z_error(value: _BoundaryValue, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not is_utc_z(value):
        errors.append(f"{field} must be an exact UTC-Z timestamp")


def closed_shape_error(
    value: dict[str, _BoundaryValue],
    expected: set[str],
    label: str,
) -> str | None:
    return f"{label} keys are not closed" if set(value) != expected else None


def append_banned_key_errors(value: _BoundaryValue, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(name in str(key).casefold() for name in _BANNED_NAMES):
                errors.append(f"banned field name: {key}")
            append_banned_key_errors(item, errors)
        return
    if isinstance(value, list):
        for item in value:
            append_banned_key_errors(item, errors)
