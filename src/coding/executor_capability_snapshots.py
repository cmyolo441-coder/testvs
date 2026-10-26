from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re
from typing import Final, TypeAlias

from ..local_store import atomic_write_json, read_json_object, utc_now
from .executors import canonical_executor_profile, legacy_executor_profile_ids
from .executor_local_workflow_selection import (
    environment_name,
    evidence_reference,
    is_workflow,
    observation_time_relation,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION: Final = "executor_capability_snapshot/v3"
MEDIA_EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION: Final = "executor_capability_snapshot/v2"
LEGACY_EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION: Final = "executor_capability_snapshot/v1"
PREPARED_CAPABILITY_SNAPSHOT_RECORDED_AT: Final = "1970-01-01T00:00:00Z"
CAPABILITY_STATUSES: Final = frozenset({"prepared", "host_observed", "unavailable", "unknown"})
LOCAL_WORKFLOW_CAPABILITY_NAME: Final = "local_workflow"
_OPERATIONAL_CAPABILITY_NAMES: Final = frozenset(
    {
        "parallel_agents",
        "background_work",
        "worktree_isolation",
        "visual_qa",
        "browser_or_computer_use",
        "long_running_continuation",
        "scheduled_or_recurring_work",
        "edit_format_hashline",
        "edit_format_str_replace",
        "edit_format_patch",
        "persistent_eval",
        "tool_reentry",
        "code_mode_batching",
    }
)
INPUT_MODALITY_CAPABILITY_NAMES: Final = frozenset(
    {
        "input_modality_text",
        "input_modality_image",
        "input_modality_audio",
        "input_modality_video",
        "input_modality_document",
    }
)
KNOWN_CAPABILITY_NAMES: Final = _OPERATIONAL_CAPABILITY_NAMES | INPUT_MODALITY_CAPABILITY_NAMES | {"resumable_goal"}
DESCRIPTIVE_CAPABILITY_NAMES: Final = frozenset(
    {
        "edit_format_hashline",
        "edit_format_str_replace",
        "edit_format_patch",
        "persistent_eval",
        "tool_reentry",
        "code_mode_batching",
    }
)
_ROOT_FIELDS: Final = frozenset({"schema_version", "executor", "recorded_at", "capabilities"})
_OBSERVED_CAPABILITY_FIELDS: Final = frozenset({"status", "scope", "evidence_ref", "observed_at"})
_STATUS_ONLY_CAPABILITY_FIELDS: Final = frozenset({"status"})
_FORBIDDEN_KEYS: Final = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "ci",
        "execution",
        "implementation",
        "log",
        "logs",
        "merge",
        "prompt",
        "raw",
        "raw_log",
        "raw_logs",
        "raw_output",
        "raw_prompt",
        "reasoning",
        "result",
        "review",
        "transcript",
        "verification",
    }
)
_MAX_EXECUTOR_LENGTH: Final = 80
_MAX_EVIDENCE_REF_LENGTH: Final = 240
_MAX_SCOPE_ITEMS: Final = 12
_MAX_SCOPE_TEXT_LENGTH: Final = 160
_LOCAL_WORKFLOW_SCOPE_KEYS: Final = frozenset({"profile", "skill_id", "environment"})
_MODALITY_SCOPE_KEYS: Final = frozenset({"provider", "wire_model", "endpoint_mode"})
_FORBIDDEN_SCOPE_KEY_TERMS: Final = ("raw", "prompt", "log", "transcript", "reasoning")
_SENSITIVE_METADATA_PATTERNS: Final = ("api_key", "apikey", "authorization:", "bearer ", "ghp_", "github_pat_", "password", "private-token", "secret", "token", "xoxb-", "xoxp-")
_SENSITIVE_METADATA_TOKEN_RE: Final = re.compile(r"(?:^|[\s=:,])(sk-|gh[opsu]_)", re.IGNORECASE)
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
SnapshotRecord: TypeAlias = dict[str, JsonValue]


class ExecutorCapabilitySnapshotError(ValueError):
    pass


def build_executor_capability_snapshot(
    *,
    executor: str,
    capabilities: Mapping[str, Mapping[str, JsonValue]],
    recorded_at: str | None = None,
) -> SnapshotRecord:
    snapshot: SnapshotRecord = {
        "schema_version": EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION,
        "executor": executor.strip(),
        "recorded_at": recorded_at or utc_now(),
        "capabilities": {name: _copy_snapshot(capability) for name, capability in capabilities.items()},
    }
    _raise_if_invalid(snapshot)
    return snapshot


def prepared_executor_capability_snapshot(
    executor: str,
    *,
    recorded_at: str | None = None,
) -> SnapshotRecord:
    """Prepared fallback for a handoff when no host observation was recorded."""
    capabilities = {name: {"status": "unknown"} for name in sorted(KNOWN_CAPABILITY_NAMES)}
    capabilities["worktree_isolation"] = {"status": "prepared"}
    return build_executor_capability_snapshot(
        executor=executor,
        capabilities=capabilities,
        # A prepared fallback records no observation. A wall-clock value would
        # make identical handoffs and fanout contracts differ for no evidentiary
        # reason, so the absence of an observation has one stable identity.
        recorded_at=recorded_at or PREPARED_CAPABILITY_SNAPSHOT_RECORDED_AT,
    )


def resolved_executor_capability_snapshot(executor: str, directory: Path | None) -> SnapshotRecord:
    """Recorded evidence for *executor*, otherwise an explicit prepared fallback."""
    snapshot = recorded_executor_capability_snapshot(executor, directory)
    if snapshot is not None:
        return complete_executor_capability_snapshot(snapshot)
    return prepared_executor_capability_snapshot(executor)


def recorded_executor_capability_snapshot(
    executor: str,
    directory: Path | None,
) -> SnapshotRecord | None:
    """Recorded evidence for *executor*, with no prepared fallback."""
    if directory is None:
        return None
    return read_matching_executor_capability_snapshot(
        executor_capability_snapshot_path(directory, executor),
        expected_executor=executor,
    )


def complete_executor_capability_snapshot(
    snapshot: Mapping[str, JsonValue],
) -> SnapshotRecord:
    """Project a valid sparse snapshot onto the v3 vocabulary.

    v1/v2 records cannot establish resumable goals; that row stays unknown.

    v1 records remain valid input, but cannot establish route-bound media
    support; every added modality row is therefore explicitly unknown.
    """
    errors = validate_executor_capability_snapshot(snapshot)
    if errors:
        raise ExecutorCapabilitySnapshotError("; ".join(errors))
    raw_capabilities = snapshot["capabilities"]
    assert isinstance(raw_capabilities, Mapping)
    capabilities = {name: {"status": "unknown"} for name in sorted(KNOWN_CAPABILITY_NAMES)}
    capabilities.update({str(name): _copy_snapshot(capability) for name, capability in raw_capabilities.items()})
    projected = _copy_snapshot(snapshot)
    projected["schema_version"] = EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION
    projected["capabilities"] = capabilities
    return projected


def executor_capability_snapshot_compatibility(snapshot: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """State whether a snapshot is readable and whether it required v1 projection."""
    errors = validate_executor_capability_snapshot(snapshot)
    if errors:
        return {"compatible": False, "reason": "; ".join(errors)}
    version = snapshot.get("schema_version")
    if version == LEGACY_EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION:
        return {"compatible": True, "projected_from": "v1", "modality_rows": "unknown"}
    return {"compatible": True, "projected_from": "", "modality_rows": "recorded"}


def validate_executor_capability_snapshot(snapshot: Mapping[str, JsonValue]) -> list[str]:
    errors = _forbidden_key_errors(snapshot) + _root_errors(snapshot)
    capabilities = snapshot.get("capabilities")
    if isinstance(capabilities, Mapping):
        errors.extend(_capability_errors(capabilities, recorded_at=snapshot.get("recorded_at")))
        goal = capabilities.get("resumable_goal")
        if goal is not None:
            if snapshot.get("schema_version") != EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION:
                errors.append("resumable_goal requires executor_capability_snapshot/v3")
            if isinstance(goal, Mapping) and goal.get("status") == "host_observed":
                scope = goal.get("scope")
                if not isinstance(scope, Mapping) or set(scope) != {"executor", "environment", "session_ref"}:
                    errors.append("resumable_goal scope requires executor, environment, session_ref")
                elif scope.get("executor") != snapshot.get("executor") or not all(
                    environment_name(scope.get(key)) for key in ("executor", "environment", "session_ref")
                ):
                    errors.append("resumable_goal scope must bind the executor and opaque environment/session")
                if not evidence_reference(goal.get("evidence_ref")):
                    errors.append("resumable_goal requires a safe evidence reference")
                if not observation_time_relation(snapshot.get("recorded_at"), goal.get("observed_at")):
                    errors.append("resumable_goal observation must precede recorded_at within 24 hours")
    bounded = [error[:240] for error in errors[:20]]
    if len(errors) > 20:
        bounded.append(f"... ({len(errors) - 20} more validation errors)")
    return bounded


def write_executor_capability_snapshot(path: Path, snapshot: Mapping[str, JsonValue]) -> SnapshotRecord:
    _raise_if_invalid(snapshot)
    persisted = _copy_snapshot(snapshot)
    atomic_write_json(path, persisted, private=True)
    return persisted


def read_executor_capability_snapshot(path: Path) -> SnapshotRecord | None:
    raw = read_json_object(path)
    if raw is None:
        return None
    snapshot = _canonical_executor_ids(_copy_snapshot(raw))
    _raise_if_invalid(snapshot)
    return snapshot


def executor_capability_snapshot_path(directory: Path, executor: str) -> Path:
    normalized = executor.strip()
    if not normalized or normalized in {".", ".."} or Path(normalized).name != normalized:
        raise ExecutorCapabilitySnapshotError("executor must be a safe snapshot filename")
    return directory / f"{normalized}.json"


def snapshot_file_exists(path: Path) -> bool:
    """`path.exists()`, False for a name the filesystem cannot stat.

    Python 3.11 raises `OSError` (ENAMETOOLONG) for an over-long file name,
    where 3.12 and later return False; an owner id from an untrusted contract
    can be that long.
    """
    try:
        return path.exists()
    except OSError:
        return False


def read_matching_executor_capability_snapshot(path: Path, *, expected_executor: str) -> SnapshotRecord | None:
    if not snapshot_file_exists(path):
        # A snapshot an older release recorded sits under the profile id it
        # used then; read it there, nothing is written back.
        for legacy_id in legacy_executor_profile_ids(expected_executor):
            legacy_path = path.with_name(f"{legacy_id}.json")
            if snapshot_file_exists(legacy_path):
                path = legacy_path
                break
    try:
        snapshot = read_executor_capability_snapshot(path)
    except (ExecutorCapabilitySnapshotError, OSError, ValueError):
        return None
    if snapshot is None or snapshot.get("executor") != expected_executor:
        return None
    return snapshot


def _canonical_executor_ids(snapshot: SnapshotRecord) -> SnapshotRecord:
    executor = snapshot.get("executor")
    if isinstance(executor, str):
        snapshot["executor"] = canonical_executor_profile(executor)
    capabilities = snapshot.get("capabilities")
    goal = capabilities.get("resumable_goal") if isinstance(capabilities, dict) else None
    scope = goal.get("scope") if isinstance(goal, dict) else None
    if isinstance(scope, dict) and isinstance(scope.get("executor"), str):
        scope["executor"] = canonical_executor_profile(scope["executor"])
    return snapshot


def _copy_snapshot(snapshot: Mapping[str, JsonValue]) -> SnapshotRecord:
    return {str(key): _copy_json_value(value) for key, value in snapshot.items()}


def _copy_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {str(key): _copy_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_value(item) for item in value]
    return value


def _raise_if_invalid(snapshot: Mapping[str, JsonValue]) -> None:
    errors = validate_executor_capability_snapshot(snapshot)
    if errors:
        raise ExecutorCapabilitySnapshotError("; ".join(errors))


def _root_errors(snapshot: Mapping[str, JsonValue]) -> list[str]:
    errors: list[str] = []
    unexpected = set(snapshot) - _ROOT_FIELDS
    if unexpected:
        names = sorted(str(key)[:80] for key in unexpected)
        rendered = ", ".join(names[:3])
        if len(names) > 3:
            rendered += f", ... ({len(names) - 3} more)"
        errors.append(f"snapshot contains unsupported fields: {rendered}")
    if snapshot.get("schema_version") not in {
        EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION,
        MEDIA_EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION,
        LEGACY_EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION,
    }:
        errors.append(
            "schema_version must be "
            f"{EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION} or "
            f"{LEGACY_EXECUTOR_CAPABILITY_SNAPSHOT_SCHEMA_VERSION}"
        )
    executor = snapshot.get("executor")
    if not isinstance(executor, str) or not executor.strip() or len(executor.strip()) > _MAX_EXECUTOR_LENGTH:
        errors.append("executor must be a nonempty bounded string")
    recorded_at = snapshot.get("recorded_at")
    if not _is_timestamp(recorded_at):
        errors.append("recorded_at must be an ISO-8601 timestamp with timezone")
    capabilities = snapshot.get("capabilities")
    if not isinstance(capabilities, Mapping) or not capabilities:
        errors.append("capabilities must be a nonempty mapping")
    return errors


def _capability_errors(
    capabilities: Mapping[str, JsonValue],
    *,
    recorded_at: JsonValue | None,
) -> list[str]:
    errors: list[str] = []
    for name, value in capabilities.items():
        if name == LOCAL_WORKFLOW_CAPABILITY_NAME:
            errors.extend(_local_workflow_errors(value, recorded_at=recorded_at))
            continue
        if name not in KNOWN_CAPABILITY_NAMES:
            errors.append(f"unsupported capability name: {str(name)[:80]}")
            continue
        if name in INPUT_MODALITY_CAPABILITY_NAMES:
            errors.extend(_modality_errors(name, value, recorded_at=recorded_at))
            continue
        if not isinstance(value, Mapping):
            errors.append(f"{name} capability must be a mapping")
            continue
        status = value.get("status")
        match status:
            case "host_observed":
                errors.extend(_observed_capability_errors(name, value, status="host_observed"))
            case "prepared" | "unavailable" | "unknown":
                errors.extend(_unexpected_capability_field_errors(name, value, _STATUS_ONLY_CAPABILITY_FIELDS))
            case _:
                errors.append(f"{name} capability status must be one of {', '.join(sorted(CAPABILITY_STATUSES))}")
    return errors


def _modality_errors(name: str, capability: JsonValue, *, recorded_at: JsonValue | None) -> list[str]:
    if not isinstance(capability, Mapping):
        return [f"{name} capability must be a mapping"]
    status = capability.get("status")
    if status in {"host_observed", "unavailable"}:
        errors = _observed_capability_errors(name, capability, status=str(status))
        scope = capability.get("scope")
        if isinstance(scope, Mapping) and set(scope) != _MODALITY_SCOPE_KEYS:
            errors.append(f"{name} scope must contain exactly provider, wire_model, and endpoint_mode")
        if not evidence_reference(capability.get("evidence_ref")):
            errors.append(f"{name} evidence_ref must be a safe opaque reference")
        if not observation_time_relation(recorded_at, capability.get("observed_at")):
            errors.append(f"{name} observed_at must be timezone-aware and within 24 hours before recorded_at")
        return errors
    if status == "unknown":
        return _unexpected_capability_field_errors(name, capability, _STATUS_ONLY_CAPABILITY_FIELDS)
    return [f"{name} capability status must be one of host_observed, unavailable, unknown"]


def _local_workflow_errors(capability: JsonValue, *, recorded_at: JsonValue | None) -> list[str]:
    name = LOCAL_WORKFLOW_CAPABILITY_NAME
    if not isinstance(capability, Mapping):
        return [f"{name} capability must be a mapping"]
    status = capability.get("status")
    match status:
        case "host_observed" | "unavailable":
            errors = _observed_capability_errors(name, capability, status=status)
            scope = capability.get("scope")
            if isinstance(scope, Mapping):
                if set(scope) != _LOCAL_WORKFLOW_SCOPE_KEYS:
                    errors.append(f"{name} scope must contain exactly profile, skill_id, and environment")
                if not is_workflow(scope.get("skill_id")):
                    errors.append(f"{name} skill_id must name a routable workflow")
                if not environment_name(scope.get("environment")):
                    errors.append(f"{name} environment must be a canonical local identifier")
            if not evidence_reference(capability.get("evidence_ref")):
                errors.append(f"{name} evidence_ref must be a safe opaque reference")
            if not observation_time_relation(recorded_at, capability.get("observed_at")):
                errors.append(f"{name} observed_at must be timezone-aware and within 24 hours before recorded_at")
            return errors
        case "unknown":
            return _unexpected_capability_field_errors(name, capability, _STATUS_ONLY_CAPABILITY_FIELDS)
        case _:
            return [f"{name} capability status must be one of host_observed, unavailable, unknown"]


def _observed_capability_errors(name: str, capability: Mapping[str, JsonValue], *, status: str) -> list[str]:
    errors = _unexpected_capability_field_errors(name, capability, _OBSERVED_CAPABILITY_FIELDS)
    scope = capability.get("scope")
    if not isinstance(scope, Mapping) or not scope:
        errors.append(f"{name} {status} capability requires a nonempty bounded scope mapping")
    else:
        errors.extend(_scope_errors(name, scope))
    evidence_ref = capability.get("evidence_ref")
    if not isinstance(evidence_ref, str) or not evidence_ref.strip() or len(evidence_ref.strip()) > _MAX_EVIDENCE_REF_LENGTH:
        errors.append(f"{name} {status} capability requires a nonempty evidence_ref")
    elif _looks_sensitive_metadata(evidence_ref):
        errors.append(f"{name} {status} capability evidence_ref must not contain sensitive metadata")
    if not _is_timestamp(capability.get("observed_at")):
        errors.append(f"{name} {status} capability requires an observed_at timestamp")
    return errors


def _unexpected_capability_field_errors(
    name: str,
    capability: Mapping[str, JsonValue],
    allowed: frozenset[str],
) -> list[str]:
    unexpected = set(capability) - allowed
    if not unexpected:
        return []
    return [f"{name} capability contains unsupported fields: {', '.join(sorted(str(key) for key in unexpected))}"]


def _scope_errors(name: str, scope: Mapping[str, JsonValue]) -> list[str]:
    errors: list[str] = []
    if len(scope) > _MAX_SCOPE_ITEMS:
        errors.append(f"{name} scope must contain at most {_MAX_SCOPE_ITEMS} items")
    for key, value in scope.items():
        if not isinstance(key, str) or not key.strip() or len(key.strip()) > _MAX_SCOPE_TEXT_LENGTH:
            errors.append(f"{name} scope keys must be nonempty bounded strings")
        elif any(term in key.casefold() for term in _FORBIDDEN_SCOPE_KEY_TERMS):
            errors.append(f"{name} scope keys must not contain raw or lifecycle material")
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > _MAX_SCOPE_TEXT_LENGTH:
            errors.append(f"{name} scope values must be nonempty bounded strings")
        elif _looks_sensitive_metadata(key) or _looks_sensitive_metadata(value):
            errors.append(f"{name} scope must not contain sensitive metadata")
    return errors


def _forbidden_key_errors(value: JsonValue | Mapping[str, JsonValue], path: str = "snapshot") -> list[str]:
    errors: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)[:80]
            if key_text.casefold() in _FORBIDDEN_KEYS:
                errors.append(f"{path}.{key_text} is forbidden metadata")
            errors.extend(_forbidden_key_errors(item, f"{path}.{key_text}"[:200]))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_forbidden_key_errors(item, f"{path}[{index}]"))
    return errors


def _is_timestamp(value: JsonValue | None) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _looks_sensitive_metadata(value: str) -> bool:
    return any(pattern in value.casefold() for pattern in _SENSITIVE_METADATA_PATTERNS) or bool(_SENSITIVE_METADATA_TOKEN_RE.search(value))
