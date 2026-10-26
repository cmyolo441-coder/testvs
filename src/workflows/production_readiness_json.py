from __future__ import annotations

import json
import math
from typing import Any, cast

from .production_readiness_values import (
    READINESS_CANONICAL_JSON_MAX_BYTES,
    READINESS_CANONICAL_JSON_MAX_DEPTH,
    READINESS_CANONICAL_JSON_MAX_NODES,
    _CANONICAL_JSON_REJECTED,
)

def _canonical_json_snapshot(value: object) -> tuple[Any, bytes] | None:
    nodes = [0]
    bytes_used = [0]
    active_containers: set[int] = set()
    try:
        snapshot = _snapshot_json_value(
            value,
            depth=0,
            nodes=nodes,
            bytes_used=bytes_used,
            active_containers=active_containers,
        )
        if snapshot is _CANONICAL_JSON_REJECTED:
            return None
        encoded = json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (KeyError, RecursionError, RuntimeError, TypeError, UnicodeError, ValueError):
        return None
    if len(encoded) != bytes_used[0] or len(encoded) > READINESS_CANONICAL_JSON_MAX_BYTES:
        return None
    return snapshot, encoded


def _snapshot_json_value(
    value: object,
    *,
    depth: int,
    nodes: list[int],
    bytes_used: list[int],
    active_containers: set[int],
) -> Any:
    if depth > READINESS_CANONICAL_JSON_MAX_DEPTH:
        return _CANONICAL_JSON_REJECTED
    nodes[0] += 1
    if nodes[0] > READINESS_CANONICAL_JSON_MAX_NODES:
        return _CANONICAL_JSON_REJECTED
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(cast(float, value)):
            return _CANONICAL_JSON_REJECTED
        if type(value) is str and len(cast(str, value)) > READINESS_CANONICAL_JSON_MAX_BYTES:
            return _CANONICAL_JSON_REJECTED
        if type(value) is int and cast(int, value).bit_length() > READINESS_CANONICAL_JSON_MAX_BYTES * 4:
            return _CANONICAL_JSON_REJECTED
        primitive_bytes = len(json.dumps(value, allow_nan=False).encode())
        return value if _consume_json_bytes(primitive_bytes, bytes_used) else _CANONICAL_JSON_REJECTED
    if type(value) not in (dict, list):
        return _CANONICAL_JSON_REJECTED
    identity = id(value)
    if identity in active_containers:
        return _CANONICAL_JSON_REJECTED
    if not _consume_json_bytes(2, bytes_used):
        return _CANONICAL_JSON_REJECTED
    active_containers.add(identity)
    try:
        if type(value) is list:
            source = cast(list[object], value)
            detached = _copy_plain_list(source)
            items: list[Any] = []
            for index, item in enumerate(detached):
                if index and not _consume_json_bytes(1, bytes_used):
                    return _CANONICAL_JSON_REJECTED
                snapshot = _snapshot_json_value(
                    item,
                    depth=depth + 1,
                    nodes=nodes,
                    bytes_used=bytes_used,
                    active_containers=active_containers,
                )
                if snapshot is _CANONICAL_JSON_REJECTED:
                    return _CANONICAL_JSON_REJECTED
                items.append(snapshot)
            return items if _same_shallow_sequence(detached, _copy_plain_list(source)) else _CANONICAL_JSON_REJECTED
        source_mapping = cast(dict[object, object], value)
        detached_mapping = _copy_plain_dict(source_mapping)
        mapping: dict[str, Any] = {}
        for index, (key, item) in enumerate(detached_mapping.items()):
            if type(key) is not str or len(key) > READINESS_CANONICAL_JSON_MAX_BYTES:
                return _CANONICAL_JSON_REJECTED
            separator_bytes = (1 if index else 0) + len(json.dumps(key).encode()) + 1
            if not _consume_json_bytes(separator_bytes, bytes_used):
                return _CANONICAL_JSON_REJECTED
            snapshot = _snapshot_json_value(
                item,
                depth=depth + 1,
                nodes=nodes,
                bytes_used=bytes_used,
                active_containers=active_containers,
            )
            if snapshot is _CANONICAL_JSON_REJECTED:
                return _CANONICAL_JSON_REJECTED
            mapping[key] = snapshot
        return (
            mapping
            if _same_shallow_mapping(detached_mapping, _copy_plain_dict(source_mapping))
            else _CANONICAL_JSON_REJECTED
        )
    finally:
        active_containers.remove(identity)


def _copy_plain_dict(value: dict[object, object]) -> dict[object, object]:
    return dict.copy(value)


def _copy_plain_list(value: list[object]) -> list[object]:
    return list.copy(value)


def _same_shallow_mapping(before: dict[object, object], after: dict[object, object]) -> bool:
    if len(before) != len(after) or set(before) != set(after):
        return False
    return all(_same_json_slot(value, after[key]) for key, value in before.items())


def _same_shallow_sequence(before: list[object], after: list[object]) -> bool:
    return len(before) == len(after) and all(
        _same_json_slot(left, right) for left, right in zip(before, after, strict=True)
    )


def _same_json_slot(left: object, right: object) -> bool:
    if type(left) in (dict, list):
        return left is right
    if type(left) in (str, bool, int, float) or left is None:
        return type(left) is type(right) and left == right
    return left is right


def _consume_json_bytes(amount: int, bytes_used: list[int]) -> bool:
    bytes_used[0] += amount
    return bytes_used[0] <= READINESS_CANONICAL_JSON_MAX_BYTES
