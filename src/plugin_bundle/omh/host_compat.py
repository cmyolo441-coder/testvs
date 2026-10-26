"""Dependency-free runtime contract shared by the installed bundle and OMH."""
from __future__ import annotations

import operator
from pathlib import Path
import re

HERMES_RANGE_FIELD = "requires_hermes"
_OPERATORS = {">=": operator.ge, "<=": operator.le, ">": operator.gt,
              "<": operator.lt, "==": operator.eq, "!=": operator.ne}
_VERSION = r"[0-9]+\.[0-9]+\.[0-9]+"


def _version(value: str) -> tuple[int, ...]:
    if not re.fullmatch(_VERSION, value):
        raise ValueError("invalid Hermes version")
    return tuple(int(part) for part in value.split("."))


def parse_range(value: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    clauses = []
    for clause in value.split(","):
        match = re.fullmatch(rf"\s*(>=|<=|>|<|==|!=)\s*({_VERSION})\s*", clause)
        if match is None:
            raise ValueError("invalid requires_hermes range")
        clauses.append((match[1], _version(match[2])))
    return tuple(clauses)


def version_satisfies(version: str, requirement: str) -> bool:
    running = _version(version)
    return all(_OPERATORS[op](running, bound) for op, bound in parse_range(requirement))


def declared_range(plugin_yaml_path: Path) -> str:
    values = []
    for line in plugin_yaml_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(HERMES_RANGE_FIELD + ":"):
            values.append(line.split(":", 1)[1].strip().strip("\"'"))
    if len(values) != 1:
        raise ValueError("requires_hermes must be declared once")
    return values[0]


def admission_error(running_version: object, plugin_yaml_path: Path) -> str | None:
    # Invalid metadata is represented by sentinels, never echoed as a payload.
    requirement = "<invalid>"
    running = "<unknown>"
    try:
        candidate = declared_range(plugin_yaml_path)
        parse_range(candidate)
        requirement = candidate
    except (OSError, UnicodeError, ValueError):
        pass  # Fail closed below with metadata-only diagnostics.
    if isinstance(running_version, str) and re.fullmatch(_VERSION, running_version):
        running = running_version
    if requirement != "<invalid>" and running != "<unknown>" and version_satisfies(running, requirement):
        return None
    return f'omh plugin requires Hermes "{requirement[:200]}"; running Hermes {running[:40]}'
