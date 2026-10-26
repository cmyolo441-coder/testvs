"""Read-path validation of a persisted diagnostic-provider outcome record.

The derived fields are re-derived here rather than merely type-checked. A
record that reached a status surface with its `outcome` edited to a clean
one is the exact failure this contract exists to prevent, so the check that
catches it lives on the read path too.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..quality.language_diagnostic_evidence import (
    LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY,
    LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR,
)
from .diagnostic_provider_models import (
    DIAGNOSTIC_COMPATIBILITY_MARKERS,
    DIAGNOSTIC_ITEM_KEYS,
    DIAGNOSTIC_OUTCOMES,
    DIAGNOSTIC_OUTCOME_RECORD_KEYS,
    DIAGNOSTIC_PROVIDER_IDS,
    DIAGNOSTIC_PROVIDERS_PRIVACY,
    DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION,
    DIAGNOSTIC_TERMINAL_STATES,
    GLOBAL_MAX_DIAGNOSTICS_PER_CHECK,
    GLOBAL_MAX_FILES_PER_CHECK,
    DiagnosticProviderError,
    _ref_errors,
)
from .diagnostic_provider_parse import (
    DiagnosticItem,
    _derive_outcome,
    _normalized_workspace_path,
    _outcome_id,
    normalize_diagnostic_item,
)


def validate_diagnostic_outcome_record(record: object) -> list[str]:
    """Return every reason the payload is not a valid outcome record."""
    if not isinstance(record, Mapping):
        return ["diagnostic_providers outcome record must be an object"]
    errors: list[str] = []
    keys = {str(key) for key in record}
    extra = sorted(keys - set(DIAGNOSTIC_OUTCOME_RECORD_KEYS))
    missing = sorted(set(DIAGNOSTIC_OUTCOME_RECORD_KEYS) - keys)
    if extra:
        errors.append(f"diagnostic_providers outcome record has unsupported keys: {extra}")
    if missing:
        errors.append(f"diagnostic_providers outcome record is missing keys: {missing}")
    if missing:
        return errors
    if record["schema_version"] != DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION:
        errors.append("diagnostic_providers outcome record schema_version must be diagnostic_providers/v1")
    if record["privacy"] != DIAGNOSTIC_PROVIDERS_PRIVACY:
        errors.append("diagnostic_providers outcome record privacy must be metadata_only")
    if record["claim_boundary"] != LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY:
        errors.append(
            "diagnostic_providers outcome record claim_boundary must state the language-diagnostic boundary"
        )
    if list(record["not_evidence_for"] or ()) != list(LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR):
        errors.append(
            "diagnostic_providers outcome record not_evidence_for must list every claim this record cannot settle"
        )
    for field, vocabulary in (
        ("terminal_state", DIAGNOSTIC_TERMINAL_STATES),
        ("compatibility", DIAGNOSTIC_COMPATIBILITY_MARKERS),
        ("outcome", DIAGNOSTIC_OUTCOMES),
    ):
        if record[field] not in vocabulary:
            errors.append(f"diagnostic_providers outcome record {field} is unsupported: {record[field]!r}")
    if errors:
        return errors
    provider_id = str(record["provider_id"])
    compatibility = str(record["compatibility"])
    terminal_state = str(record["terminal_state"])
    if compatibility == "provider_disabled":
        if provider_id != "":
            errors.append("diagnostic_providers outcome record provider_disabled marker cannot name a provider")
        if terminal_state != "completed":
            errors.append(
                "diagnostic_providers outcome record provider_disabled marker cannot carry a terminal state"
            )
    elif provider_id not in DIAGNOSTIC_PROVIDER_IDS:
        errors.append(f"diagnostic_providers outcome record provider_id is not allowlisted: {provider_id!r}")
    errors.extend(_ref_errors(record["workspace_id"], field="workspace_id", required=True))
    errors.extend(_ref_errors(record["revision"], field="revision", required=True))
    errors.extend(_ref_errors(record["diagnostics_revision"], field="diagnostics_revision", required=False))
    errors.extend(_ref_errors(record["config_identity"], field="config_identity", required=False))
    errors.extend(_ref_errors(record["outcome_id"], field="outcome_id", required=True))
    errors.extend(_file_list_errors(record["in_scope_files"], field="in_scope_files"))
    errors.extend(_file_list_errors(record["out_of_scope_files"], field="out_of_scope_files"))
    errors.extend(_file_list_errors(record["diagnosed_files"], field="diagnosed_files"))
    errors.extend(_file_list_errors(record["changed_files"], field="changed_files"))
    items, item_errors = _item_list_errors(record["diagnostics"])
    errors.extend(item_errors)
    if errors:
        return errors
    in_scope = [str(value) for value in record["in_scope_files"]]
    out_of_scope = [str(value) for value in record["out_of_scope_files"]]
    diagnosed = [str(value) for value in record["diagnosed_files"]]
    if set(in_scope) & set(out_of_scope):
        errors.append("diagnostic_providers outcome record in_scope_files and out_of_scope_files must not overlap")
    if not set(diagnosed) <= set(in_scope):
        errors.append("diagnostic_providers outcome record diagnosed_files must stay inside in_scope_files")
    if list(record["changed_files"]) != sorted(set(in_scope) | set(out_of_scope)):
        errors.append("diagnostic_providers outcome record changed_files must be the sorted union of the scope files")
    if any(item.path not in set(in_scope) for item in items):
        errors.append("diagnostic_providers outcome record diagnostics must point inside in_scope_files")
    derived = _derive_outcome(
        compatibility,
        provider_id,
        terminal_state,
        str(record["revision"]),
        str(record["diagnostics_revision"]),
        tuple(in_scope),
        tuple(diagnosed),
    )
    if record["outcome"] != derived:
        errors.append(f"diagnostic_providers outcome record outcome must be derived as {derived!r}")
    expected_id = _outcome_id(
        workspace_id=str(record["workspace_id"]),
        revision=str(record["revision"]),
        diagnostics_revision=str(record["diagnostics_revision"]),
        provider_id=provider_id,
        terminal_state=terminal_state,
        outcome=derived,
        compatibility=compatibility,
        in_scope_files=tuple(in_scope),
        out_of_scope_files=tuple(out_of_scope),
        diagnosed_files=tuple(diagnosed),
        diagnostics=tuple(items),
        config_identity=str(record["config_identity"]),
    )
    if record["outcome_id"] != expected_id:
        errors.append("diagnostic_providers outcome record outcome_id must be derived from the record fields")
    return errors


def _file_list_errors(values: object, *, field: str) -> list[str]:
    if not isinstance(values, list):
        return [f"diagnostic_providers outcome record {field} must be a list"]
    errors: list[str] = []
    if len(values) > GLOBAL_MAX_FILES_PER_CHECK:
        errors.append(
            f"diagnostic_providers outcome record {field} must have at most {GLOBAL_MAX_FILES_PER_CHECK} entries"
        )
    for index, value in enumerate(values):
        try:
            normalized = _normalized_workspace_path(value, field=f"diagnostic_providers {field}[{index}]")
        except DiagnosticProviderError as error:
            errors.append(str(error))
            continue
        if normalized != value:
            errors.append(
                f"diagnostic_providers outcome record {field}[{index}] must be a normalized workspace-relative path"
            )
    if not all(isinstance(value, str) for value in values):
        errors.append(f"diagnostic_providers outcome record {field} entries must be strings")
    elif values != sorted(values) or len(set(values)) != len(values):
        errors.append(f"diagnostic_providers outcome record {field} must be sorted and unique")
    return errors


def _item_list_errors(values: object) -> tuple[list[DiagnosticItem], list[str]]:
    if not isinstance(values, list):
        return [], ["diagnostic_providers outcome record diagnostics must be a list"]
    errors: list[str] = []
    if len(values) > GLOBAL_MAX_DIAGNOSTICS_PER_CHECK:
        errors.append(
            f"diagnostic_providers outcome record diagnostics must have at most "
            f"{GLOBAL_MAX_DIAGNOSTICS_PER_CHECK} entries"
        )
    items: list[DiagnosticItem] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or {str(key) for key in value} != set(DIAGNOSTIC_ITEM_KEYS):
            errors.append(
                f"diagnostic_providers outcome record diagnostics[{index}] must carry exactly "
                f"{list(DIAGNOSTIC_ITEM_KEYS)}"
            )
            continue
        try:
            item = normalize_diagnostic_item(value)
        except DiagnosticProviderError as error:
            errors.append(str(error))
            continue
        if item.as_record() != dict(value):
            errors.append(f"diagnostic_providers outcome record diagnostics[{index}] must be normalized")
            continue
        items.append(item)
    keys = [(item.character, item.code, item.line, item.path, item.severity, item.source) for item in items]
    if len(keys) == len(values) and keys != sorted(set(keys)):
        errors.append("diagnostic_providers outcome record diagnostics must be sorted and unique")
    return items, errors
