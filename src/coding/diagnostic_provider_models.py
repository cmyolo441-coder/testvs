"""Vocabulary, bounds, and shared reference rules for diagnostic providers.

This is the constants half of the `diagnostic_providers/v1` contract: the
closed vocabularies (providers, outcomes, terminal states, compatibility
markers), the global bounds that cap every capability, the error type every
refusal raises, the opaque-reference helpers every module normalizes through,
and the moving-HEAD equality predicate freshness is derived from.
"""

from __future__ import annotations

from ..quality.language_diagnostic_evidence import LANGUAGE_DIAGNOSTIC_ITEM_KEYS, MAX_REFERENCE_CHARS
from ..system.metadata_safety import require_opaque_metadata_ref


DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION = "diagnostic_providers/v1"
DIAGNOSTIC_PROVIDERS_PRIVACY = "metadata_only"

# The diagnostic item vocabulary is reused from the evidence record verbatim.
DIAGNOSTIC_ITEM_KEYS = LANGUAGE_DIAGNOSTIC_ITEM_KEYS

# The allowlist. Nothing outside it can be named as a provider, a capability,
# or a diagnostic source, so the contract never has to reason about a tool it
# did not vet.
DIAGNOSTIC_PROVIDER_IDS = ("pyright", "basedpyright", "mypy", "ruff")

# Every way one serial check can end. `ok` and `partial` describe completed
# checks; `timeout`, `cancelled`, and `crashed` name provider behavior;
# `unsupported` means no allowlisted provider could run at all; `stale` means
# the diagnostics cannot be pinned to the requested revision.
DIAGNOSTIC_OUTCOMES = ("ok", "timeout", "cancelled", "crashed", "partial", "unsupported", "stale")

# What the caller reports about the provider process. Only `completed` can
# produce `ok`, `partial`, or `stale`; the other three are the outcome.
DIAGNOSTIC_TERMINAL_STATES = ("completed", "timeout", "cancelled", "crashed")

# How the diagnostics (or their absence) relate to the allowlisted providers:
# the scheduler selected one, none was runnable (disabled or no match for the
# scope), or the caller ran the provider itself and recorded the result.
DIAGNOSTIC_COMPATIBILITY_MARKERS = ("provider_selected", "provider_disabled", "caller_supplied")

# Revisions that name where a pointer points, not what it points at.
MOVING_REVISION_REFS = ("HEAD", "FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD")

# Global bounds. A capability may declare less; nothing may declare more.
GLOBAL_MAX_TIMEOUT_MS = 120_000
MIN_TIMEOUT_MS = 1
GLOBAL_MAX_DIAGNOSTICS_PER_CHECK = 200
GLOBAL_MAX_FILES_PER_CHECK = 200

# Key names a raw payload would arrive under. An input diagnostic carrying
# one is refused by name, so a source body has nowhere to ride in.
RAW_PAYLOAD_FIELD_NAMES = ("message", "prompt", "content", "body", "snippet", "text", "diff", "log")

DIAGNOSTIC_OUTCOME_RECORD_KEYS = (
    "changed_files",
    "claim_boundary",
    "compatibility",
    "config_identity",
    "diagnosed_files",
    "diagnostics",
    "diagnostics_revision",
    "in_scope_files",
    "not_evidence_for",
    "outcome",
    "outcome_id",
    "out_of_scope_files",
    "privacy",
    "provider_id",
    "revision",
    "schema_version",
    "terminal_state",
    "workspace_id",
)


class DiagnosticProviderError(ValueError):
    """Raised when a diagnostic-provider contract input cannot become a record."""


def _checked_ref(value: object, *, field: str, required: bool) -> str:
    if not isinstance(value, str):
        raise DiagnosticProviderError(f"diagnostic_providers {field} must be a string")
    if value == "":
        if required:
            raise DiagnosticProviderError(f"diagnostic_providers {field} must not be empty")
        return ""
    try:
        text = require_opaque_metadata_ref(value, field=field)
    except ValueError as error:
        raise DiagnosticProviderError(str(error)) from error
    if len(text) > MAX_REFERENCE_CHARS:
        raise DiagnosticProviderError(f"diagnostic_providers {field} must be at most {MAX_REFERENCE_CHARS} characters")
    return text


def _ref_errors(value: object, *, field: str, required: bool) -> list[str]:
    if not isinstance(value, str):
        return [f"diagnostic_providers {field} must be a string"]
    if value == "":
        return [] if not required else [f"diagnostic_providers {field} must not be empty"]
    try:
        _checked_ref(value, field=field, required=required)
    except DiagnosticProviderError as error:
        return [str(error)]
    return []


def is_moving_revision(revision: str) -> bool:
    """True when the revision names where a pointer points, not a commit."""
    return revision in MOVING_REVISION_REFS


def revisions_identical(revision_a: str, revision_b: str) -> bool:
    """The equality predicate freshness is derived from.

    A moving ref is identical to nothing, including itself: between the read
    that produced one side and the read that produced the other, the ref may
    have moved, so even `HEAD == HEAD` proves nothing about the revisions the
    two sides actually observed.
    """
    if not revision_a or not revision_b:
        return False
    if is_moving_revision(revision_a) or is_moving_revision(revision_b):
        return False
    return revision_a == revision_b
