"""Bounded untrusted clarification metadata; never an authority grant."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Final, Literal, TypedDict

from ..plugin_bundle.omh._governance_safety import contains_credential_like_material
from .fanout_failure_diagnostics import is_string_map, is_object_list

MAX_CLARIFICATION_ROUNDS: Final = 2
_SLUG: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")


class Options(TypedDict):
    kind: Literal["options"]
    options: list[str]


class Text(TypedDict):
    kind: Literal["text"]
    max_chars: int


class InputRequired(TypedDict):
    decision_id: str
    question: str
    blocking_reason: str
    answer_shape: Options | Text
    affected_unit_ids: list[str]
    redacted_context: list[str]


class ClarificationError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason: str = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ClarificationAnswer:
    unit_id: str
    decision_id: str
    attempt_id: str
    round: int
    answer: str | None = None


def metadata_text(value: object, field: str, limit: int) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > limit
            or not value.isprintable() or contains_credential_like_material(value)
            or any(token in value for token in ("`", "$(", "${", "<script", ";", "&&", "||"))):
        raise ClarificationError(field + " must be bounded non-executable secret-free metadata")
    return value


def parse_input_required(value: object, unit_id: str) -> InputRequired:
    keys = {"decision_id", "question", "blocking_reason", "answer_shape", "affected_unit_ids", "redacted_context"}
    if not is_string_map(value) or set(value) != keys:
        raise ClarificationError("input_required fields are missing or unknown")
    decision = metadata_text(value["decision_id"], "decision_id", 64)
    if not _SLUG.fullmatch(decision):
        raise ClarificationError("decision_id must be a slug")
    affected = value["affected_unit_ids"]
    if (not is_object_list(affected) or not 1 <= len(affected) <= 16
            or any(not isinstance(item, str) or not _SLUG.fullmatch(item) for item in affected)
            or len(set(affected)) != len(affected) or unit_id not in affected):
        raise ClarificationError("affected_unit_ids must include the reporting unit without duplicates")
    context = value["redacted_context"]
    if not is_object_list(context) or len(context) > 8:
        raise ClarificationError("redacted_context must be a bounded list")
    shape = value["answer_shape"]
    parsed: Options | Text
    if not is_string_map(shape):
        raise ClarificationError("answer_shape must be an object")
    match shape:
        case {"kind": "options", "options": options} if set(shape) == {"kind", "options"}:
            if not is_object_list(options) or not 1 <= len(options) <= 8:
                raise ClarificationError("answer_shape options must contain 1..8 entries")
            choices = [metadata_text(item, "answer_shape.options", 80) for item in options]
            if len(set(choices)) != len(choices):
                raise ClarificationError("answer_shape options must be unique")
            parsed = {"kind": "options", "options": choices}
        case {"kind": "text", "max_chars": int(limit)} if set(shape) == {"kind", "max_chars"}:
            if isinstance(limit, bool) or not 1 <= limit <= 300:
                raise ClarificationError("answer_shape max_chars must be 1..300")
            parsed = {"kind": "text", "max_chars": limit}
        case _:
            raise ClarificationError("answer_shape must be options or bounded text")
    return {"decision_id": decision, "question": metadata_text(value["question"], "question", 300),
        "blocking_reason": metadata_text(value["blocking_reason"], "blocking_reason", 300),
        "answer_shape": parsed, "affected_unit_ids": [str(item) for item in affected],
        "redacted_context": [metadata_text(item, "redacted_context", 160) for item in context]}
