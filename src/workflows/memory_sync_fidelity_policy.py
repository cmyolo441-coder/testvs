"""Closed, metadata-only input selection policies; no provider defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .memory_sync_fidelity_validation import (
    Assessment, assessments as _assessments, closed_value as _closed_value, metadata_mapping,
)


ROLES: Final = ("user", "assistant", "tool_call", "tool_result", "summary", "attachment", "metadata")
UNITS: Final = ("characters", "tokens", "bytes", "messages", "records", "unknown")
DIMENSIONS: Final = (
    "input_surface", "selection_policy", "limit", "truncation_policy",
    "omission_visibility", "backpressure_policy", "extraction",
)
# Closed field schemas are data, not provider-specific capability inference.
CHOICES: Final = {
    "selection_policy": {"mode": ("whole_turn", "selected_messages", "selected_fields", "approved_facts", "unknown")},
    "limit": {"unit": UNITS, "scope": ("per_turn", "per_message", "per_session", "unknown"), "origin": ("configured", "discovered", "unknown")},
    "truncation_policy": {"mode": ("head", "tail", "boundary", "sampled", "rejected", "unknown")},
    "omission_visibility": {"mode": ("counted_without_content", "not_counted", "unknown")},
    "backpressure_policy": {"mode": ("queue", "coalesce", "block", "retry", "skip", "unknown"), "duplicate_prevention": ("attempt_id", "input_digest", "none", "unknown")},
    "extraction": {"mode": ("local_visible", "hosted_opaque", "unknown")},
}
EXTRA_FIELDS: Final = {
    "input_surface": {"roles"}, "limit": {"value"},
    "truncation_policy": {"splits_structured_input"}, "backpressure_policy": {"max_wait_ms"},
}


@dataclass(frozen=True, slots=True)
class FidelityPolicy:
    """Normalized assessments and scalar policy metadata, never input bodies."""

    assessments: tuple[tuple[str, Assessment], ...]
    fields: tuple[tuple[str, tuple[tuple[str, str | int | bool | None], ...]], ...]
    roles: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, dict[str, object]]:
        result = {name: dict(values) for name, values in self.fields}
        output: dict[str, dict[str, object]] = {
            name: {**assessment.to_dict(), **result[name]}
            for name, assessment in self.assessments
        }
        output["input_surface"]["roles"] = dict(self.roles)
        return output

    @property
    def unknown_field_count(self) -> int:
        return (
            sum(assessment.status == "unknown" for _, assessment in self.assessments)
            + sum(value is None or value == "unknown" for _, values in self.fields for _, value in values)
            + sum(value == "unknown" for _, value in self.roles)
        )


def bounded_count(value: object, field: str) -> int | None:
    """Parse counts and waits without coercing bools, negatives, or huge integers."""
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{field} must be a nonnegative bounded integer or null")
    return value


def parse_policy(raw: dict[str, object]) -> FidelityPolicy:
    assessments: list[tuple[str, Assessment]] = []
    fields: list[tuple[str, tuple[tuple[str, str | int | bool | None], ...]]] = []
    roles: tuple[tuple[str, str], ...] = tuple((role, "unknown") for role in ROLES)
    for name in DIMENSIONS:
        item = metadata_mapping(raw.get(name, {}), name)
        choices = CHOICES.get(name, {})
        allowed = {"status", "evidence_class"} | set(choices) | EXTRA_FIELDS.get(name, set())
        if set(item) - allowed:
            raise ValueError(f"input_fidelity {name} must contain only supported metadata fields")
        assessment = _assessments({name: {
            "status": item.get("status", "unknown"), "evidence_class": item.get("evidence_class", "none"),
        }}, (name,), name)[name]
        assessments.append((name, assessment))
        values: dict[str, str | int | bool | None] = {
            field: _closed_value(item.get(field, "unknown"), vocabulary, f"{name}.{field}")
            for field, vocabulary in choices.items()
        }
        match name:
            case "input_surface":
                supplied = metadata_mapping(item.get("roles", {}), "roles")
                if set(supplied) - set(ROLES):
                    raise ValueError("input_surface.roles must contain supported roles only")
                roles = tuple((role, _closed_value(supplied.get(role, "unknown"), ("eligible", "excluded", "unknown"), role)) for role in ROLES)
            case "limit":
                values["value"] = bounded_count(item.get("value"), "limit.value")
                if values["value"] is not None and "unknown" in (values["unit"], values["scope"], values["origin"]):
                    raise ValueError("a known limit requires unit, scope, and origin")
            case "truncation_policy":
                split = item.get("splits_structured_input", "unknown")
                values["splits_structured_input"] = split if isinstance(split, bool) else _closed_value(split, ("unknown",), "splits_structured_input")
            case "backpressure_policy":
                values["max_wait_ms"] = bounded_count(item.get("max_wait_ms"), "max_wait_ms")
            case "selection_policy" | "omission_visibility" | "extraction":
                pass
        fields.append((name, tuple(values.items())))
    return FidelityPolicy(tuple(assessments), tuple(fields), roles)
