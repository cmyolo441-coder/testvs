from __future__ import annotations

import json
from json import JSONDecodeError
import os


def read_limited_bytes(descriptor: int, maximum: int) -> bytes:
    remaining = maximum + 1
    chunks: list[bytes] = []
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_bounded_json_object(
    raw: bytes,
    *,
    max_depth: int,
    max_nodes: int,
) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except RecursionError as exc:
        raise ValueError("artifact_json_depth_exceeded") from exc
    except (UnicodeDecodeError, JSONDecodeError) as exc:
        raise ValueError("malformed_json") from exc
    if not isinstance(value, dict):
        raise ValueError("malformed_json")
    _validate_json_bounds(value, max_depth=max_depth, max_nodes=max_nodes)
    return value


def _validate_json_bounds(value: object, *, max_depth: int, max_nodes: int) -> None:
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError("artifact_json_nodes_exceeded")
        if depth > max_depth:
            raise ValueError("artifact_json_depth_exceeded")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
