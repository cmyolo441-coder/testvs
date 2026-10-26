from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import re
from typing import Final, Never, TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
_HEX_40: Final = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64: Final = re.compile(r"[0-9a-f]{64}\Z")
_UNSAFE: Final = re.compile(
    r"(^/|/Users/|/home/|[A-Za-z]:\\|sk-[A-Za-z0-9]|api[_-]?key|"
    r"BEGIN PRIVATE KEY|ignore previous|<script)",
    re.IGNORECASE,
)


class BenchmarkValidationError(ValueError):
    __slots__: tuple[str, ...] = ("reason_codes",)
    reason_codes: tuple[str, ...]

    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        self.reason_codes = reason_codes
        super().__init__(*reason_codes)

    def __str__(self) -> str:
        return ",".join(self.reason_codes)


def corpus_digest(value: JsonValue | Mapping[str, JsonValue]) -> str:
    """Return the canonical SHA-256 digest for JSON-compatible metadata."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def shape(raw: Mapping[str, JsonValue], fields: set[str]) -> None:
    missing = fields - set(raw)
    extra = set(raw) - fields
    if missing:
        raise_validation("missing_fields")
    if extra:
        raise_validation("extra_fields")


def json_map(value: JsonValue | None) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise_validation("wrong_type")
    return value


def items(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise_validation("wrong_type")
    return value


def flat(value: JsonValue) -> Mapping[str, JsonScalar]:
    return {key: scalar(item) for key, item in json_map(value).items()}


def scalar(value: JsonValue) -> JsonScalar:
    if isinstance(value, (dict, list)):
        raise_validation("wrong_type")
    return value


def text(value: JsonValue | None) -> str:
    if not isinstance(value, str) or not value:
        raise_validation("wrong_type")
    return value


def integer(value: JsonValue) -> int:
    if type(value) is not int:
        raise_validation("wrong_type")
    return value


def commit(value: JsonValue) -> str:
    parsed = text(value)
    if not _HEX_40.fullmatch(parsed):
        raise_validation("invalid_commit")
    return parsed


def digest(value: JsonValue, reason: str) -> str:
    parsed = text(value)
    if not _HEX_64.fullmatch(parsed):
        raise_validation(reason)
    return parsed


def relative(value: JsonValue) -> str:
    parsed = text(value)
    if parsed.startswith("/") or ".." in parsed.split("/"):
        raise_validation("unsafe_metadata")
    return parsed


def safe(value: JsonValue | Mapping[str, JsonValue]) -> None:
    pending: list[JsonValue | Mapping[str, JsonValue]] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str) and _UNSAFE.search(current):
            raise_validation("unsafe_metadata")
        if isinstance(current, Mapping):
            pending.extend(current.values())
        if isinstance(current, list):
            pending.extend(current)


def unique(values: Iterable[str], reason: str) -> None:
    parsed = tuple(values)
    if len(parsed) != len(set(parsed)):
        raise_validation(reason)


def raise_validation(reason: str) -> Never:
    raise BenchmarkValidationError((reason,))
