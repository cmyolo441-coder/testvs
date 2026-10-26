"""Stale override validation for memory governance (internal module)."""

from __future__ import annotations

from datetime import datetime, timedelta
from .iso_datetime import strict_fromisoformat


def validate_stale_override(
    override: dict[str, object],
    artifact: dict[str, object],
    reval_deadline: datetime,
    now: datetime,
    run_id: str | None,
) -> dict[str, object]:
    """Validate a stale override.
    
    B2 fix: Requires confirmed_at, expires_at, and revalidation_deadline.
    Valid only if:
    - artifact_identity matches exactly
    - run_id matches exactly
    - revalidation_deadline matches exactly
    - confirmed_at is present (timestamp of confirmation)
    - expires_at is present and <= 7 days from now
    """
    stable_artifact_identity = _governance_evaluation_module.stable_artifact_identity
    
    artifact_identity = stable_artifact_identity(artifact)
    override_identity = override.get("artifact_identity")
    
    if artifact_identity != override_identity:
        return {"valid": False, "reason": "stale_override_invalid"}
    
    override_run_id = override.get("run_id")
    if override_run_id != run_id:
        return {"valid": False, "reason": "stale_override_invalid"}
    
    # B2: Require confirmed_at
    confirmed_at = override.get("confirmed_at")
    if not confirmed_at:
        return {"valid": False, "reason": "stale_override_invalid"}
    
    # B2: Require revalidation_deadline match
    override_deadline_str = override.get("revalidation_deadline")
    if override_deadline_str:
        try:
            override_str = str(override_deadline_str).replace("Z", "+00:00")
            override_deadline = strict_fromisoformat(override_str)
            if override_deadline != reval_deadline:
                return {"valid": False, "reason": "stale_override_invalid"}
        except (ValueError, TypeError):
            return {"valid": False, "reason": "stale_override_invalid"}
    else:
        return {"valid": False, "reason": "stale_override_invalid"}
    
    # B2: Require and validate expires_at
    expires_at_str = override.get("expires_at")
    if not expires_at_str:
        return {"valid": False, "reason": "stale_override_invalid"}
    
    try:
        expires_str = str(expires_at_str).replace("Z", "+00:00")
        expires_at = strict_fromisoformat(expires_str)
        
        if now >= expires_at:
            return {"valid": False, "reason": "stale_override_expired"}
        
        # Bound check: override cannot last > 7 days
        delta = expires_at - now
        if delta > timedelta(days=7):
            return {"valid": False, "reason": "stale_override_invalid"}
    except (ValueError, TypeError):
        return {"valid": False, "reason": "stale_override_invalid"}
    
    return {"valid": True}


# Module scope, not call time: Hermes can evict the bundle from sys.modules
# while its callbacks stay live (#1979). Last, so an import cycle finds every name.
from . import _governance_evaluation as _governance_evaluation_module
