from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Final, Never

from .cross_harness_benchmark_values import JsonValue


MAX_INPUT_BYTES: Final = 1_000_000
MAX_JSON_DEPTH: Final = 64
MAX_JSON_CONTAINERS: Final = 10_000
MAX_JSON_NODES: Final = 50_000


@dataclass(frozen=True, slots=True)
class BenchmarkJsonInputError(ValueError):
    reason_code: str

    def __str__(self) -> str:
        return self.reason_code


def decode_benchmark_json(text: str) -> JsonValue:
    """Decode strict JSON and reject inputs beyond fixed complexity limits."""
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise BenchmarkJsonInputError("input_too_large")
    try:
        value: JsonValue = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except BenchmarkJsonInputError:
        raise
    except ValueError as error:
        raise BenchmarkJsonInputError("invalid_json") from error
    except RecursionError as error:
        raise BenchmarkJsonInputError("input_too_complex") from error
    _validate_complexity(value)
    return value


def decode_benchmark_bytes(encoded: bytes) -> JsonValue:
    if len(encoded) > MAX_INPUT_BYTES:
        raise BenchmarkJsonInputError("input_too_large")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BenchmarkJsonInputError("invalid_utf8") from error
    return decode_benchmark_json(text)


def _validate_complexity(value: JsonValue) -> None:
    pending = [(value, 1)]
    nodes = 0
    containers = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise BenchmarkJsonInputError("input_too_complex")
        if isinstance(current, float) and not math.isfinite(current):
            raise BenchmarkJsonInputError("invalid_json")
        if isinstance(current, dict):
            containers += 1
            pending.extend((item, depth + 1) for item in current.values())
        if isinstance(current, list):
            containers += 1
            pending.extend((item, depth + 1) for item in current)
        if containers > MAX_JSON_CONTAINERS:
            raise BenchmarkJsonInputError("input_too_complex")


def _reject_constant(_value: str) -> Never:
    raise BenchmarkJsonInputError("invalid_json")


def _reject_duplicate_pairs(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    parsed: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in parsed:
            raise BenchmarkJsonInputError("duplicate_key")
        parsed[key] = value
    return parsed
