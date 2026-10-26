"""Fail-closed parsing for recorded local fanout graph contracts."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
import json
import re
from typing import Final

from .run_records import (
    FANOUT_CONTRACT_PROVENANCE_SCHEMA_VERSION as _FANOUT_PROVENANCE_SCHEMA_VERSION,
    FANOUT_CONTRACT_SCHEMA_VERSION as _FANOUT_CONTRACT_SCHEMA_VERSION,
)

GRAPH_CONTRACT_UNIT_LIMIT: Final[int] = 64
GRAPH_DEPENDENCY_LIMIT: Final[int] = 32
GRAPH_FILE_SCOPE_LIMIT: Final[int] = 32

_FANOUT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^fanout-[0-9a-f]{12}$")
_UNIT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_PROVENANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "fanout_id",
        "contract_schema_version",
        "contract_sha256",
        "privacy",
    }
)


def normalized_graph_units(
    contract: Mapping[str, object],
) -> tuple[list[dict[str, object]], str]:
    """Parse the topology-bearing subset of one frozen v2 contract."""
    if contract.get("schema_version") != _FANOUT_CONTRACT_SCHEMA_VERSION:
        return [], "invalid_contract_schema"
    if contract.get("status") != "prepared_not_observed":
        return [], "invalid_contract_status"
    if not _FANOUT_ID_RE.fullmatch(str(contract.get("fanout_id", ""))):
        return [], "invalid_fanout_id"
    raw_units = contract.get("units")
    if not isinstance(raw_units, list):
        return [], "empty_graph"
    if len(raw_units) > GRAPH_CONTRACT_UNIT_LIMIT:
        return [], "graph_limit_exceeded"
    normalized: list[dict[str, object]] = []
    for raw in raw_units:
        if not isinstance(raw, Mapping):
            return [], "unknown_dependency"
        raw_unit_id = raw.get("unit_id")
        if not isinstance(raw_unit_id, str) or not _UNIT_ID_RE.fullmatch(raw_unit_id):
            return [], "unknown_dependency"
        raw_dependencies = raw.get("depends_on", [])
        if not isinstance(raw_dependencies, list):
            return [], "unknown_dependency"
        if len(raw_dependencies) > GRAPH_DEPENDENCY_LIMIT:
            return [], "graph_limit_exceeded"
        if any(
            not isinstance(dependency, str) or not _UNIT_ID_RE.fullmatch(dependency)
            for dependency in raw_dependencies
        ):
            return [], "unknown_dependency"
        if len(set(raw_dependencies)) != len(raw_dependencies):
            return [], "duplicate_dependency"
        boundary = raw.get("boundary")
        if not isinstance(boundary, Mapping):
            return [], "invalid_contract_boundary"
        raw_scope = boundary.get("file_scope")
        if (
            not isinstance(raw_scope, list)
            or not raw_scope
            or len(raw_scope) > GRAPH_FILE_SCOPE_LIMIT
            or any(
                not isinstance(path, str)
                or not path.strip()
                or len(path) > 512
                for path in raw_scope
            )
        ):
            return [], "invalid_contract_boundary"
        normalized.append(
            {
                "unit_id": raw_unit_id,
                "depends_on": tuple(raw_dependencies),
                "file_scope": tuple(raw_scope),
            }
        )
    return normalized, ""


def recorded_contract_blocker(
    contract: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    expected_fanout_id: str,
) -> str:
    """Return a fail-closed reason when one local contract record is not frozen."""
    if (
        set(provenance) != _PROVENANCE_KEYS
        or provenance.get("schema_version") != _FANOUT_PROVENANCE_SCHEMA_VERSION
        or provenance.get("fanout_id") != expected_fanout_id
        or contract.get("fanout_id") != expected_fanout_id
        or provenance.get("contract_schema_version") != contract.get("schema_version")
        or provenance.get("privacy") != "metadata_only"
    ):
        return "unverified_fanout_contract"
    digest = provenance.get("contract_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return "unverified_fanout_contract"
    try:
        encoded = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return "unverified_fanout_contract"
    return "" if sha256(encoded).hexdigest() == digest else "unverified_fanout_contract"
