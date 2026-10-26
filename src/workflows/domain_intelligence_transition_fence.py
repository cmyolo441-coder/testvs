from __future__ import annotations

from collections.abc import Callable

from ..paths import OmhPaths
from . import domain_intelligence_operation_store as journal
from .domain_intelligence_contracts import SAFE_PROFILE_ID
from .domain_intelligence_operations import (
    APPROVAL_OPERATION_SCHEMA_VERSION,
    validate_approval_operation,
)
from .domain_intelligence_rejection_operations import (
    REJECTION_OPERATION_SCHEMA_VERSION,
    validate_rejection_operation,
)
from .domain_intelligence_retirement_operations import (
    RETIREMENT_OPERATION_SCHEMA_VERSION,
    validate_retirement_operation,
)


OperationValidator = Callable[[OmhPaths | None, dict[str, object]], None]
_OPERATION_VALIDATORS: dict[str, OperationValidator] = {
    APPROVAL_OPERATION_SCHEMA_VERSION: validate_approval_operation,
    REJECTION_OPERATION_SCHEMA_VERSION: validate_rejection_operation,
    RETIREMENT_OPERATION_SCHEMA_VERSION: validate_retirement_operation,
}


def require_profile_transition(
    paths: OmhPaths, *, profile_id: str, operation_id: str
) -> None:
    if not SAFE_PROFILE_ID.fullmatch(profile_id):
        raise ValueError("unsafe_profile_id")
    claimed_operation_ids: list[str] = []
    seen_operation_ids: set[str] = set()
    for operation_path, operation in journal.scan_operations(paths):
        schema_version = operation.get("schema_version")
        validator = _OPERATION_VALIDATORS.get(str(schema_version))
        if validator is None:
            raise ValueError("domain_transition_operation_invalid")
        validator(paths, operation)
        recorded_id = str(operation["operation_id"])
        if operation_path.stem != recorded_id:
            raise ValueError("domain_transition_operation_identity_mismatch")
        if recorded_id in seen_operation_ids:
            raise ValueError("domain_transition_operation_duplicate")
        seen_operation_ids.add(recorded_id)
        recorded_profile = operation.get("profile_id")
        if (
            not isinstance(recorded_profile, str)
            or not SAFE_PROFILE_ID.fullmatch(recorded_profile)
        ):
            raise ValueError("domain_transition_operation_profile_mismatch")
        if recorded_profile == profile_id:
            claimed_operation_ids.append(recorded_id)
    if len(claimed_operation_ids) > 1:
        raise ValueError("domain_transition_operation_duplicate")
    if claimed_operation_ids and claimed_operation_ids[0] != operation_id:
        raise ValueError("domain_transition_in_progress")
