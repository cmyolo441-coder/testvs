from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping

from ..system.metadata_safety import is_sensitive_metadata_text, require_opaque_metadata_ref


MCP_TOOL_NAME_COMPATIBILITY_SCHEMA_VERSION = "mcp_tool_name_compatibility/v1"
MCP_TOOL_NAME_SNAPSHOT_SCHEMA_VERSION = "mcp_tool_name_compatibility_snapshot/v1"
MAX_MCP_TOOL_NAME_SNAPSHOTS = 20
MAX_MCP_TOOL_NAME_SNAPSHOT_BYTES = 131_072
MAX_MCP_TOOL_NAME_ENTRIES = 256
MAX_MCP_TOOL_NAME_AUDIT_CHARS = 128
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class McpToolNameCompatibilityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NamingRules:
    target_harness: str
    separator_normalization: str
    punctuation_pattern: str
    maximum_name_length: int | None
    source: str

    def normalize(self, value: str) -> str:
        return re.sub(self.punctuation_pattern, "_", value)


# OpenCode's toolName joins sanitized server/tool names with '_'; sanitize
# preserves '-' and '_' and replaces each other punctuation character with '_'.
# The v1 audit contract additionally folds '-' to '_' so dash, dot and underscore
# spellings share a comparison key. This is not a host registration transform.
# Compare supplied whole names only: never strip/invent a server prefix. There
# is no name-length cap in this source; the separate audit cap is conservative.
# Claude Code, Codex and Cursor remain unsupported until their full naming
# contracts (including namespace/hash behavior) can be represented faithfully.
MCP_TOOL_NAMING_ADAPTERS: Mapping[tuple[str, str], NamingRules] = MappingProxyType({
    ("opencode-mcp-tool-naming", "v1"): NamingRules(
        target_harness="opencode",
        separator_normalization="hyphen_to_underscore",
        punctuation_pattern=r"[^a-zA-Z0-9_]",
        maximum_name_length=None,
        source=(
            "https://github.com/anomalyco/opencode/blob/"
            "95daf90670b7c039c436c85537da5fbfe2205b41/packages/opencode/src/mcp/catalog.ts#L117-L119"
        ),
    ),
})


def _object(value: object, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise McpToolNameCompatibilityError("malformed MCP tool-name snapshot object")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value) or is_sensitive_metadata_text(value):
        raise McpToolNameCompatibilityError("MCP tool-name snapshot requires safe bounded metadata")
    return value


def _evidence(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 32:
        raise McpToolNameCompatibilityError("MCP tool-name evidence must be a bounded list")
    try:
        return sorted({require_opaque_metadata_ref(ref, field="evidence ref") for ref in value})
    except ValueError as exc:
        raise McpToolNameCompatibilityError("MCP tool-name evidence must contain safe opaque references") from exc


def _tools(value: object, *, advertised: bool) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_MCP_TOOL_NAME_ENTRIES:
        raise McpToolNameCompatibilityError("MCP tool-name entries must be a bounded list")
    name_key = "name" if advertised else "logical_name"
    keys = {name_key, "evidence_refs"} | ({"observation"} if advertised else set())
    tools = []
    identities = set()
    for item in value:
        item = _object(item, keys)
        name = _name(item[name_key])
        tool = {name_key: name, "evidence_refs": _evidence(item["evidence_refs"])}
        if advertised:
            observation = _name(item["observation"])
            if observation not in ("registered", "config_only"):
                raise McpToolNameCompatibilityError("invalid MCP tool-name observation")
            tool["observation"] = observation
        identity = (name, tool.get("observation"))
        if identity in identities:
            raise McpToolNameCompatibilityError("duplicate MCP tool-name entry")
        identities.add(identity)
        tools.append(tool)
    return sorted(tools, key=lambda tool: (tool[name_key], tool.get("observation", "")))


def _snapshot(value: object) -> dict[str, Any]:
    value = _object(value, {"schema_version", "target_harness", "adapter", "required_tools", "advertised_tools"})
    if value["schema_version"] != MCP_TOOL_NAME_SNAPSHOT_SCHEMA_VERSION:
        raise McpToolNameCompatibilityError("unsupported MCP tool-name snapshot schema")
    adapter = _object(value["adapter"], {"id", "version"})
    return {
        "target_harness": _name(value["target_harness"]),
        "adapter": {"id": _name(adapter["id"]), "version": _name(adapter["version"])},
        "required_tools": _tools(value["required_tools"], advertised=False),
        "advertised_tools": _tools(value["advertised_tools"], advertised=True),
    }


def _row(required: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    logical_name = required["logical_name"]
    adapter = snapshot["adapter"]
    rules = MCP_TOOL_NAMING_ADAPTERS.get((adapter["id"], adapter["version"]))
    row = {
        "logical_name": logical_name,
        "target_harness": snapshot["target_harness"],
        "adapter": dict(adapter),
        "state": "unobserved",
        "reason": "unsupported_adapter",
        "candidates": [],
        "auto_selected": False,
        "evidence_refs": list(required["evidence_refs"]),
    }
    if rules is None or rules.target_harness != snapshot["target_harness"]:
        return row
    limit = min(rules.maximum_name_length or MAX_MCP_TOOL_NAME_AUDIT_CHARS, MAX_MCP_TOOL_NAME_AUDIT_CHARS)
    if len(logical_name) > limit:
        row["reason"] = "name_exceeds_audit_limit"
        return row
    advertised = snapshot["advertised_tools"]
    exact = [tool for tool in advertised if tool["name"] == logical_name and tool["observation"] == "registered"]
    if exact:
        state, reason, matches = "exact", "registered_exact", exact
    else:
        normalized = rules.normalize(logical_name)
        matches = [tool for tool in advertised if len(tool["name"]) <= limit and rules.normalize(tool["name"]) == normalized]
        registered = [tool for tool in matches if tool["observation"] == "registered"]
        if registered:
            matches = registered
            row["candidates"] = sorted(tool["name"] for tool in matches)
            state = "compatible_one_to_one" if len(matches) == 1 else "ambiguous"
            reason = "registered_adapter_candidate" if len(matches) == 1 else "multiple_registered_candidates"
        elif matches:
            state, reason = "unobserved", "config_only"
        else:
            state, reason = "missing", "not_advertised"
    row.update(state=state, reason=reason)
    row["evidence_refs"] = sorted(set(required["evidence_refs"]).union(*(tool["evidence_refs"] for tool in matches)))
    return row


def build_mcp_tool_name_compatibility_report(snapshots: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Audit only supplied metadata; no discovery, invocation or alias selection."""
    if not 1 <= len(snapshots) <= MAX_MCP_TOOL_NAME_SNAPSHOTS:
        raise McpToolNameCompatibilityError("MCP tool-name audit requires 1 to 20 snapshots")
    validated = sorted((_snapshot(value) for value in snapshots), key=lambda value: value["target_harness"])
    if len({value["target_harness"] for value in validated}) != len(validated):
        raise McpToolNameCompatibilityError("MCP tool-name snapshots must name unique target harnesses")
    rows = sorted(
        (_row(tool, value) for value in validated for tool in value["required_tools"]),
        key=lambda row: (row["logical_name"], row["target_harness"]),
    )
    return {
        "schema_version": MCP_TOOL_NAME_COMPATIBILITY_SCHEMA_VERSION,
        "adapter": dict(validated[0]["adapter"]) if len(validated) == 1 else None,
        "snapshots": [{"target_harness": value["target_harness"], "adapter": dict(value["adapter"])} for value in validated],
        "rows": rows,
        "summary": {"row_count": len(rows), "state_counts": dict(sorted(Counter(row["state"] for row in rows).items()))},
        "not_observed": {
            name: {"status": "not_observed"} for name in (
                "host_discovery", "first_party_tools", "recorded_sessions", "tool_invocation",
                "config_mutation", "alias_registration", "handoff_rewriting",
            )
        },
        "claim_boundary": (
            "Read-only comparison of explicitly supplied advertised-name metadata, not independently verified "
            "host registration, tool identity, callability, invocation, alias registration or handoff execution. "
            "Config-only presence and unknown adapters never prove callable-name compatibility."
        ),
    }


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise McpToolNameCompatibilityError("duplicate MCP tool-name snapshot key")
        result[key] = value
    return result


def _read_snapshot(path: Path) -> dict[str, Any]:
    # Nonblocking open + fstat before reading refuses a FIFO even if a regular
    # file is replaced after the caller chose it. Read at most the byte budget.
    if is_sensitive_metadata_text(str(path)):
        raise McpToolNameCompatibilityError("MCP tool-name snapshot path must be safe metadata")
    try:
        descriptor = os.open(path.expanduser(), os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise McpToolNameCompatibilityError("MCP tool-name snapshot must be a regular file")
            content = handle.read(MAX_MCP_TOOL_NAME_SNAPSHOT_BYTES + 1)
    except OSError as exc:
        raise McpToolNameCompatibilityError("MCP tool-name snapshot cannot be read") from exc
    if len(content) > MAX_MCP_TOOL_NAME_SNAPSHOT_BYTES:
        raise McpToolNameCompatibilityError("MCP tool-name snapshot exceeds 131072 bytes")
    try:
        return json.loads(content.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise McpToolNameCompatibilityError("malformed MCP tool-name snapshot JSON") from exc


def audit_mcp_tool_name_compatibility(snapshot_paths: tuple[Path, ...]) -> dict[str, Any]:
    if not 1 <= len(snapshot_paths) <= MAX_MCP_TOOL_NAME_SNAPSHOTS:
        raise McpToolNameCompatibilityError("MCP tool-name audit requires 1 to 20 snapshot paths")
    return build_mcp_tool_name_compatibility_report(tuple(_read_snapshot(path) for path in snapshot_paths))
