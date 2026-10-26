"""Metadata-only lifecycle and portability posture for optional memory providers.

Generic connector readiness answers whether a provider is affordable, authorized,
and scoped. It cannot answer whether enabling one is *reversible*: whether the
provider writes on its own hooks, which identity it isolates records by, what
retention it applies, what a deletion actually deletes, whether an export exists,
and what switching away leaves behind.

This module is that missing block. It reads no credential, calls no provider,
changes no provider selection, and mutates no memory store. Every field it does
not receive stays `unknown` rather than inheriting an OMH or Hermes default, and
`unknown` deletion, isolation, or write semantics produce explicit blockers
rather than a quiet pass.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from .memory_sync_fidelity import MemorySyncFidelity, build_memory_sync_fidelity, parse_memory_sync_fidelity
from .memory_sync_fidelity_validation import (
    Assessment, EVIDENCE_CLASSES, FIELD_STATUSES,
    assessments as _assessments, closed_value as _closed_value,
    observed_at as _observed_at, provider_id as _provider_id,
)

from ..local_store import atomic_write_json, ensure_dir
from ..paths import OmhPaths
from ..system.metadata_safety import require_opaque_metadata_ref


MEMORY_PROVIDER_POSTURE_INPUT_SCHEMA_VERSION = "memory_provider_posture_input/v1"
MEMORY_PROVIDER_POSTURE_SCHEMA_VERSION = "memory_provider_posture/v1"

# Open package code is not the hosted service that runs it: package license,
# tests, and interfaces establish neither storage nor retention nor deletion.
STORAGE_BOUNDARIES: tuple[str, ...] = (
    "open_package_code",
    "local_runtime",
    "self_hosted_service",
    "hosted_api",
    "unknown",
)

IDENTITY_SCOPES: tuple[str, ...] = (
    "profile",
    "user",
    "agent",
    "project_or_workspace",
    "session",
    "task",
    "namespace",
    "tenant",
)

AUTOMATIC_BEHAVIORS: tuple[str, ...] = (
    "prefetch",
    "turn_start",
    "native_write",
    "pre_compression",
    "session_end",
    "shutdown",
    "recovery",
    "background_synchronization",
)

# Disabling, removing a local cache, deleting remote memory, and deleting an
# account are four different postconditions, so they are four different rows.
LIFECYCLE_OPERATIONS: tuple[str, ...] = (
    "pause",
    "disable",
    "retention",
    "edit_or_correction",
    "logical_expiry",
    "archive",
    "local_cache_removal",
    "provider_side_deletion",
    "account_deletion",
    "export",
    "import",
    "backup",
    "restore",
    "provider_switch",
)

SYNCHRONIZATION_DIMENSIONS: tuple[str, ...] = (
    "direction",
    "checkpoint_identity",
    "idempotency",
    "retry_and_queue",
    "last_observed_success",
    "partial_failure",
    "failure_mode",
)

# The generic connector-readiness fields this posture preserves rather than
# replaces. Dropping them would trade one blind spot for another.
GENERIC_READINESS_DIMENSIONS: tuple[str, ...] = (
    "credentials",
    "egress",
    "privacy",
    "pii",
    "cost",
    "quota",
    "permissions",
    "fallback",
    "observed_trials",
)

# The three uses a posture can block. Each is an adoption decision a user makes,
# not a provider capability.
ADOPTION_DECISIONS: tuple[str, ...] = (
    "automatic_write",
    "destructive_operation",
    "irreversible_adoption",
)

# Portability is what survives leaving: without an export, an import, and a
# switch story, adoption is one-way.
_PORTABILITY_OPERATIONS: tuple[str, ...] = ("export", "import", "provider_switch")

_SETTLED_STATUSES = frozenset({"ready", "missing", "risky"})
_UNSETTLED_STATUSES = frozenset({"unknown", "not_observed"})

_RECEIPT_KEYS = {"receipt_id", "provider_id", "scope", "operation", "observed_at", "postcondition"}
_INPUT_REQUIRED_KEYS = {"schema_version", "provider_id", "observed_version_boundary", "storage_boundary"}
_INPUT_OPTIONAL_KEYS = {
    "identity_scopes",
    "automatic_behaviors",
    "lifecycle_operations",
    "synchronization",
    "generic_readiness",
    "observed_trials",
    "input_fidelity",
}
_MAX_RECEIPTS = 24


@dataclass(frozen=True)
class TrialReceipt:
    """A provider-returned observation bound to one scoped operation."""

    receipt_id: str
    provider_id: str
    scope: str
    operation: str
    observed_at: str
    postcondition: str

    def to_dict(self) -> dict[str, str]:
        return {
            "receipt_id": self.receipt_id,
            "provider_id": self.provider_id,
            "scope": self.scope,
            "operation": self.operation,
            "observed_at": self.observed_at,
            "postcondition": self.postcondition,
        }


@dataclass(frozen=True)
class MemoryProviderPostureInput:
    provider_id: str
    observed_version_boundary: str
    storage_boundary: str
    identity_scopes: dict[str, Assessment]
    automatic_behaviors: dict[str, Assessment]
    lifecycle_operations: dict[str, Assessment]
    synchronization: dict[str, Assessment]
    generic_readiness: dict[str, Assessment]
    observed_trials: tuple[TrialReceipt, ...]
    input_fidelity: MemorySyncFidelity | None = None


def parse_memory_provider_posture_input(raw: object) -> MemoryProviderPostureInput:
    if not isinstance(raw, dict) or not _INPUT_REQUIRED_KEYS.issubset(raw):
        raise ValueError("memory provider posture input must use the supported metadata fields")
    if set(raw) - (_INPUT_REQUIRED_KEYS | _INPUT_OPTIONAL_KEYS):
        raise ValueError("memory provider posture input must use the supported metadata fields")
    if raw.get("schema_version") != MEMORY_PROVIDER_POSTURE_INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported memory provider posture input schema")
    return MemoryProviderPostureInput(
        _provider_id(raw.get("provider_id")),
        require_opaque_metadata_ref(raw.get("observed_version_boundary"), field="observed_version_boundary"),
        _closed_value(raw.get("storage_boundary"), STORAGE_BOUNDARIES, "storage_boundary"),
        _assessments(raw.get("identity_scopes", {}), IDENTITY_SCOPES, "identity_scopes"),
        _assessments(raw.get("automatic_behaviors", {}), AUTOMATIC_BEHAVIORS, "automatic_behaviors"),
        _assessments(raw.get("lifecycle_operations", {}), LIFECYCLE_OPERATIONS, "lifecycle_operations"),
        _assessments(raw.get("synchronization", {}), SYNCHRONIZATION_DIMENSIONS, "synchronization"),
        _assessments(raw.get("generic_readiness", {}), GENERIC_READINESS_DIMENSIONS, "generic_readiness"),
        _trial_receipts(raw.get("observed_trials", [])),
        parse_memory_sync_fidelity(raw["input_fidelity"]) if "input_fidelity" in raw else None,
    )


def build_memory_provider_posture(value: MemoryProviderPostureInput) -> dict[str, object]:
    fidelity = build_memory_sync_fidelity(value.input_fidelity, value.provider_id) if value.input_fidelity is not None else None
    accepted, rejected = _partition_receipts(value)
    lifecycle = _lifecycle_block(value, accepted)
    identity = _complete(value.identity_scopes, IDENTITY_SCOPES)
    automatic = _automatic_block(value.automatic_behaviors)
    synchronization = _complete(value.synchronization, SYNCHRONIZATION_DIMENSIONS)
    generic = _complete(value.generic_readiness, GENERIC_READINESS_DIMENSIONS)
    blockers = _blockers(identity, automatic, lifecycle)
    return {
        "schema_version": MEMORY_PROVIDER_POSTURE_SCHEMA_VERSION,
        "provider_id": value.provider_id,
        "observed_version_boundary": value.observed_version_boundary,
        "storage_boundary": value.storage_boundary,
        "state": "prepared_not_observed",
        "identity_scopes": identity,
        "automatic_behaviors": automatic,
        "lifecycle_operations": lifecycle,
        "synchronization": synchronization,
        **({"input_fidelity": fidelity} if fidelity is not None else {}),
        "generic_readiness": generic,
        "observed_trials": [receipt.to_dict() for receipt in accepted],
        "rejected_trials": rejected,
        "blockers": blockers,
        "adoption_verdicts": _adoption_verdicts(blockers),
        "unknown_field_count": _unknown_field_count(identity, automatic, lifecycle, synchronization, generic),
        "optionality": {
            "provider_required": False,
            "omh_memory_available_without_provider": True,
            "hosted_default_permitted": False,
            "unknown_provider_effect": "omh_memory_unaffected",
        },
        "memory_sync_handoff": {
            **({"input_fidelity_summary": {
                "readiness": fidelity["synchronization_readiness"],
                "unknown_count": fidelity["unknown_field_count"],
            }} if fidelity is not None else {}),
            "review_status": "not_omh_reviewed",
            "imports_provider_records": False,
            "authorizes_native_memory_mutation": False,
            "next_action": "prepare_memory_sync",
            "note": (
                "Provider posture reaches memory-sync as not_omh_reviewed context and a next-action handoff only. "
                "It never imports provider records into OMH review and never authorizes a native-memory write."
            ),
        },
        "allowed_actions": [
            "request_operator_lifecycle_declaration",
            "request_observed_trial_receipt",
            "review_external_connector_readiness",
            "hand_off_not_omh_reviewed_context_to_memory_sync",
        ],
        "prohibited_actions": [
            "read_secret_value",
            "call_provider",
            "install_provider_plugin",
            "enable_automatic_hook",
            "change_provider_selection",
            "synchronize_records",
            "mutate_memory_store",
            "delete_provider_memory",
            "delete_account",
            "import_provider_records_into_omh_review",
        ],
        "claim_boundary": (
            "A memory provider posture is OMH-local preparation metadata derived from declared and supplied "
            "observations. It is not provider connectivity, credential validation, hook activation, write "
            "authority, synchronization, export, deletion, or account evidence, and a declared contract is never "
            "an observed remote postcondition."
        ),
    }


def write_memory_provider_posture(paths: OmhPaths, posture: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(posture, sort_keys=True, separators=(",", ":"))
    posture_id = "memory_provider_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    directory = _managed_memory_provider_postures_dir(paths)
    path = directory / f"{posture_id}.json"
    ensure_dir(directory, private=True)
    atomic_write_json(path, {**posture, "posture_id": posture_id}, private=True)
    return {"written": True, "posture_id": posture_id, "path": str(path)}


def _lifecycle_block(
    value: MemoryProviderPostureInput,
    accepted: tuple[TrialReceipt, ...],
) -> dict[str, dict[str, object]]:
    bound: dict[str, TrialReceipt] = {}
    for receipt in accepted:
        bound.setdefault(receipt.operation, receipt)
    block: dict[str, dict[str, object]] = {}
    for operation in LIFECYCLE_OPERATIONS:
        assessment = value.lifecycle_operations.get(operation, Assessment("unknown", "none"))
        receipt = bound.get(operation)
        if receipt is None:
            block[operation] = {**assessment.to_dict(), "observed_receipt_id": "", "observed_scope": ""}
            continue
        # A bound receipt settles an operation nothing else established. It never
        # erases a status the operator already recorded as missing or risky.
        status = "ready" if assessment.status in _UNSETTLED_STATUSES else assessment.status
        block[operation] = {
            "status": status,
            "evidence_class": "observed_trial_receipt",
            "observed_receipt_id": receipt.receipt_id,
            "observed_scope": receipt.scope,
        }
    return block


def _automatic_block(supplied: dict[str, Assessment]) -> dict[str, dict[str, object]]:
    # A hook observation says where the provider can act, never that it may.
    return {
        behavior: {**supplied.get(behavior, Assessment("unknown", "none")).to_dict(), "grants_write_authority": False}
        for behavior in AUTOMATIC_BEHAVIORS
    }


def _complete(supplied: dict[str, Assessment], dimensions: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {
        dimension: supplied.get(dimension, Assessment("unknown", "none")).to_dict() for dimension in dimensions
    }


def _blockers(
    identity: dict[str, dict[str, str]],
    automatic: dict[str, dict[str, object]],
    lifecycle: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    unsettled_deletion = [
        operation
        for operation in ("provider_side_deletion", "account_deletion", "local_cache_removal")
        if lifecycle[operation]["status"] in _UNSETTLED_STATUSES
    ]
    if unsettled_deletion:
        blockers.append(
            {
                "id": "deletion_semantics_unknown",
                "reason": "Deletion postconditions are not established, so removal cannot be shown to be real.",
                "unresolved_fields": unsettled_deletion,
                "blocks": ["automatic_write", "destructive_operation", "irreversible_adoption"],
            }
        )
    unsettled_identity = [scope for scope in IDENTITY_SCOPES if identity[scope]["status"] in _UNSETTLED_STATUSES]
    if unsettled_identity:
        blockers.append(
            {
                "id": "isolation_semantics_unknown",
                "reason": "Record isolation is not established, so a write cannot be shown to stay in its scope.",
                "unresolved_fields": unsettled_identity,
                "blocks": ["automatic_write", "irreversible_adoption"],
            }
        )
    unsettled_write = [
        behavior
        for behavior in ("native_write", "background_synchronization")
        if automatic[behavior]["status"] in _UNSETTLED_STATUSES
    ]
    if unsettled_write:
        blockers.append(
            {
                "id": "write_semantics_unknown",
                "reason": "Automatic write and synchronization behaviour is not established.",
                "unresolved_fields": unsettled_write,
                "blocks": ["automatic_write"],
            }
        )
    unsettled_portability = [
        operation for operation in _PORTABILITY_OPERATIONS if lifecycle[operation]["status"] in _UNSETTLED_STATUSES
    ]
    if unsettled_portability:
        blockers.append(
            {
                "id": "portability_unknown",
                "reason": "Export, import, or provider switching is not established, so adoption may be one-way.",
                "unresolved_fields": unsettled_portability,
                "blocks": ["irreversible_adoption"],
            }
        )
    return blockers


def _adoption_verdicts(blockers: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    verdicts: dict[str, dict[str, object]] = {}
    for decision in ADOPTION_DECISIONS:
        blocking = [str(blocker["id"]) for blocker in blockers if decision in blocker["blocks"]]
        verdicts[decision] = {
            "verdict": "blocked" if blocking else "no_recorded_blocker",
            "blocking_ids": blocking,
        }
    return verdicts


def _unknown_field_count(*blocks: dict[str, dict[str, str]] | dict[str, dict[str, object]]) -> int:
    return sum(1 for block in blocks for field in block.values() if field["status"] == "unknown")


def _partition_receipts(
    value: MemoryProviderPostureInput,
) -> tuple[tuple[TrialReceipt, ...], list[dict[str, str]]]:
    accepted: list[TrialReceipt] = []
    rejected: list[dict[str, str]] = []
    for receipt in value.observed_trials:
        if receipt.provider_id != value.provider_id:
            # Mismatched evidence keeps the claim blocked; it never binds.
            rejected.append({"receipt_id": receipt.receipt_id, "reason": "provider_identity_mismatch"})
            continue
        accepted.append(receipt)
    return tuple(accepted), rejected


def _trial_receipts(raw: object) -> tuple[TrialReceipt, ...]:
    if not isinstance(raw, list) or len(raw) > _MAX_RECEIPTS:
        raise ValueError(f"observed_trials must contain at most {_MAX_RECEIPTS} items")
    receipts: list[TrialReceipt] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _RECEIPT_KEYS:
            raise ValueError(
                "observed trial receipt must bind receipt_id, provider_id, scope, operation, observed_at, "
                "and postcondition"
            )
        receipts.append(
            TrialReceipt(
                require_opaque_metadata_ref(item.get("receipt_id"), field="observed trial receipt_id"),
                _provider_id(item.get("provider_id")),
                _closed_value(item.get("scope"), IDENTITY_SCOPES, "observed trial scope"),
                _closed_value(item.get("operation"), LIFECYCLE_OPERATIONS, "observed trial operation"),
                _observed_at(item.get("observed_at")),
                require_opaque_metadata_ref(item.get("postcondition"), field="observed trial postcondition"),
            )
        )
    if len({receipt.receipt_id for receipt in receipts}) != len(receipts):
        raise ValueError("observed trial receipt ids must be unique")
    return tuple(receipts)


def _managed_memory_provider_postures_dir(paths: OmhPaths) -> Path:
    root = paths.memory_provider_postures_dir
    if root.is_symlink():
        raise ValueError("memory provider posture storage must not be a symlink")
    if not root.resolve(strict=False).is_relative_to(paths.omh_home.resolve(strict=False)):
        raise ValueError("memory provider posture storage must resolve under OMH home")
    return root


__all__ = [
    "ADOPTION_DECISIONS",
    "AUTOMATIC_BEHAVIORS",
    "Assessment",
    "EVIDENCE_CLASSES",
    "FIELD_STATUSES",
    "GENERIC_READINESS_DIMENSIONS",
    "IDENTITY_SCOPES",
    "LIFECYCLE_OPERATIONS",
    "MEMORY_PROVIDER_POSTURE_INPUT_SCHEMA_VERSION",
    "MEMORY_PROVIDER_POSTURE_SCHEMA_VERSION",
    "MemoryProviderPostureInput",
    "STORAGE_BOUNDARIES",
    "SYNCHRONIZATION_DIMENSIONS",
    "TrialReceipt",
    "build_memory_provider_posture",
    "parse_memory_provider_posture_input",
    "write_memory_provider_posture",
]
