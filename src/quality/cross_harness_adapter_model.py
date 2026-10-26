from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
from typing import Final, Never, TypeVar

from .cross_harness_benchmark_values import JsonScalar, JsonValue


REQUEST_SCHEMA: Final = "cross_harness_adapter_request/v1"
RESULT_SCHEMA: Final = "cross_harness_adapter_result/v1"
PROTOCOL_VERSION: Final = "cross_harness_adapter_protocol/v1"
ARTIFACT_TYPE: Final = "cross_harness_adapter_fixture_result/v1"
MAX_ITEMS: Final = 128
_HEX_64: Final = re.compile(r"[0-9a-f]{64}")
_SECRET: Final = re.compile(r"(sk-[A-Za-z0-9]|ghp_[A-Za-z0-9]|AKIA[0-9A-Z]{16}|api[_-]?key|authorization|bearer\s+|password|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ignore previous|<script)", re.IGNORECASE)
_RAW_KEY: Final = re.compile(r"(^|_)(environment|env|body|prompt|response|stdout|stderr|skill_body|secret|token|credential|password|authorization)($|_)", re.IGNORECASE)
_EnumT = TypeVar("_EnumT", bound=StrEnum)


class Profile(StrEnum):
    CODEX = "codex"
    CLAUDE_CODE = "claude-code"
    PI = "pi"
    HERMES = "hermes"
    OMX = "omx"
    PI_FAMILY = "pi-family"
    OMC = "omc"


class Effort(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"
    ULTRA = "ultra"


class ObservationState(StrEnum):
    OBSERVED = "observed"
    PREPARED = "prepared_not_observed"
    NOT_APPLICABLE = "not_applicable"


class ProcessStatus(StrEnum):
    EXIT = "exit"
    TIMEOUT = "timeout"
    CRASH = "crash"
    NOT_STARTED = "not_started"


class AdapterContractError(ValueError):
    __slots__ = ("reason_code",)
    reason_code: str

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)

    def __str__(self) -> str:
        return self.reason_code


@dataclass(frozen=True, slots=True)
class NormalizedEvent:
    id: str
    result: str


@dataclass(frozen=True, slots=True)
class SideEffect:
    path: str
    change: str


@dataclass(frozen=True, slots=True)
class AdapterRequest:
    schema_version: str
    protocol_version: str
    corpus_digest: str
    fixture_binding_digest: str
    fixture_id: str
    adapter_id: str
    capability_id: str
    profile: Profile
    executable: str
    executable_version: str
    model: str
    effort: Effort
    capabilities: tuple[str, ...]
    argv_digest: str
    repetition: int
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class AdapterResult:
    schema_version: str
    request_digest: str
    fixture_id: str
    adapter_id: str
    capability_id: str
    evidence_class: str
    observation_state: ObservationState
    actual_machine: tuple[tuple[str, JsonScalar], ...]
    facts: tuple[tuple[str, JsonScalar], ...]
    skill_events: tuple[NormalizedEvent, ...]
    tool_events: tuple[NormalizedEvent, ...]
    child_results: tuple[NormalizedEvent, ...]
    artifact_type: str
    artifact_hash: str
    process_status: ProcessStatus
    exit_code: int | None
    side_effects: tuple[SideEffect, ...]


def canonical_digest(value: JsonValue | Mapping[str, JsonValue]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_adapter_request(raw: Mapping[str, JsonValue]) -> AdapterRequest:
    _shape(raw, {"schema_version", "protocol_version", "corpus_digest", "fixture_binding_digest", "fixture_id", "adapter_id", "capability_id", "profile", "executable", "executable_version", "model", "effort", "capabilities", "argv_digest", "repetition", "timeout_seconds"})
    _safe(raw)
    if _text(raw["schema_version"]) != REQUEST_SCHEMA or _text(raw["protocol_version"]) != PROTOCOL_VERSION:
        _raise("stale_adapter_version")
    executable = _text(raw["executable"])
    if "/" in executable or "\\" in executable:
        _raise("unsafe_path")
    capabilities = tuple(_text(item) for item in _items(raw["capabilities"]))
    _bounded_unique(capabilities, "duplicate_capability")
    repetition = _integer(raw["repetition"])
    timeout = _integer(raw["timeout_seconds"])
    if repetition < 1 or repetition > 100 or timeout < 1 or timeout > 3600:
        _raise("out_of_bounds")
    return AdapterRequest(
        REQUEST_SCHEMA, PROTOCOL_VERSION, _digest(raw["corpus_digest"]), _digest(raw["fixture_binding_digest"]), _text(raw["fixture_id"]), _text(raw["adapter_id"]), _text(raw["capability_id"]), _enum(Profile, raw["profile"]),
        executable, _text(raw["executable_version"]), _text(raw["model"]), _enum(Effort, raw["effort"]), capabilities, _digest(raw["argv_digest"]), repetition, timeout,
    )


def parse_adapter_result(raw: Mapping[str, JsonValue]) -> AdapterResult:
    _shape(raw, {"schema_version", "request_digest", "fixture_id", "adapter_id", "capability_id", "evidence_class", "observation_state", "actual_machine", "facts", "skill_events", "tool_events", "child_results", "artifact_type", "artifact_hash", "process_status", "exit_code", "side_effects"})
    _safe(raw)
    if _text(raw["schema_version"]) != RESULT_SCHEMA:
        _raise("stale_adapter_version")
    if _text(raw["artifact_type"]) != ARTIFACT_TYPE:
        _raise("wrong_artifact_type")
    artifact_hash = _digest(raw["artifact_hash"])
    evidence = _text(raw["evidence_class"])
    if evidence not in {"prepared", "static", "test", "runtime"}:
        _raise("invalid_evidence_state")
    children = _events(raw["child_results"], child=True)
    status = _enum(ProcessStatus, raw["process_status"])
    exit_value = raw["exit_code"]
    exit_code = None if exit_value is None else _integer(exit_value)
    if (status is ProcessStatus.EXIT) != (exit_code is not None):
        _raise("invalid_process_state")
    observed = _enum(ObservationState, raw["observation_state"])
    if (observed is ObservationState.OBSERVED) == (status is ProcessStatus.NOT_STARTED):
        _raise("invalid_evidence_state")
    return AdapterResult(
        RESULT_SCHEMA, _digest(raw["request_digest"]), _text(raw["fixture_id"]), _text(raw["adapter_id"]),
        _text(raw["capability_id"]), evidence, observed, _flat(raw["actual_machine"]), _flat(raw["facts"]),
        _events(raw["skill_events"]), _events(raw["tool_events"]), children, ARTIFACT_TYPE, artifact_hash, status, exit_code, _side_effects(raw["side_effects"]),
    )


def _events(value: JsonValue, *, child: bool = False) -> tuple[NormalizedEvent, ...]:
    parsed: list[NormalizedEvent] = []
    for item in _items(value):
        raw = _mapping(item)
        _shape(raw, {"id", "result"})
        result = _text(raw["result"])
        if child and result == "partial":
            _raise("partial_child_failure")
        if result not in {"pass", "fail"}:
            _raise("invalid_event_result")
        parsed.append(NormalizedEvent(_text(raw["id"]), result))
    _bounded_unique((item.id for item in parsed), "duplicate_event")
    return tuple(parsed)


def _side_effects(value: JsonValue) -> tuple[SideEffect, ...]:
    parsed: list[SideEffect] = []
    for item in _items(value):
        raw = _mapping(item)
        _shape(raw, {"path", "change"})
        change = _text(raw["change"])
        if change not in {"created", "modified", "deleted"}:
            _raise("invalid_side_effect")
        parsed.append(SideEffect(_relative(raw["path"]), change))
    _bounded_unique((item.path for item in parsed), "duplicate_side_effect")
    return tuple(parsed)


def _flat(value: JsonValue) -> tuple[tuple[str, JsonScalar], ...]:
    raw = _mapping(value)
    if len(raw) > MAX_ITEMS:
        _raise("collection_too_large")
    parsed: list[tuple[str, JsonScalar]] = []
    for key, item in sorted(raw.items()):
        if _RAW_KEY.search(key) or key in {"path", "raw_path", "executable_path"}:
            _raise("raw_field_forbidden")
        if isinstance(item, (dict, list)) or isinstance(item, float) and not math.isfinite(item):
            _raise("wrong_type")
        parsed.append((_text(key), item))
    return tuple(parsed)


def _shape(raw: Mapping[str, JsonValue], fields: set[str]) -> None:
    if fields - set(raw):
        _raise("missing_fields")
    if set(raw) - fields:
        _raise("extra_fields")


def _mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        _raise("wrong_type")
    return value


def _items(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        _raise("wrong_type")
    if len(value) > MAX_ITEMS:
        _raise("collection_too_large")
    return value


def _text(value: JsonValue) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        _raise("wrong_type")
    return value


def _integer(value: JsonValue) -> int:
    if type(value) is not int:
        _raise("wrong_type")
    return value


def _digest(value: JsonValue) -> str:
    parsed = _text(value)
    if not _HEX_64.fullmatch(parsed):
        _raise("invalid_digest")
    return parsed


def _relative(value: JsonValue) -> str:
    parsed = _text(value)
    path = PurePosixPath(parsed)
    if path.is_absolute() or ".." in path.parts or parsed.startswith("~"):
        _raise("unsafe_path")
    return parsed


def _safe(value: JsonValue | Mapping[str, JsonValue]) -> None:
    pending: list[tuple[JsonValue | Mapping[str, JsonValue], int]] = [(value, 1)]
    nodes = 0
    while pending:
        current, depth = pending.pop()
        nodes += 1
        if nodes > 4096 or depth > 32:
            _raise("input_too_complex")
        if isinstance(current, str):
            if current.startswith("/") or re.match(r"[A-Za-z]:[\\/]", current):
                _raise("unsafe_path")
            if _SECRET.search(current):
                _raise("secret_or_raw_data")
        if isinstance(current, Mapping):
            if any(_RAW_KEY.search(key) for key in current):
                _raise("raw_field_forbidden")
            pending.extend((item, depth + 1) for item in (*current.keys(), *current.values()))
        if isinstance(current, list):
            if len(current) > MAX_ITEMS:
                _raise("collection_too_large")
            pending.extend((item, depth + 1) for item in current)


def _bounded_unique(values: Iterable[str], reason: str) -> None:
    parsed = tuple(values)
    if len(parsed) != len(set(parsed)):
        _raise(reason)


def _enum(kind: type[_EnumT], value: JsonValue) -> _EnumT:
    try:
        return kind(_text(value))
    except ValueError as error:
        raise AdapterContractError("unknown_enum_value") from error


def _raise(reason: str) -> Never:
    raise AdapterContractError(reason)
