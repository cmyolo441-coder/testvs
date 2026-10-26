"""Allowlisted diagnostic-provider capabilities for one serial check.

`language_diagnostic_evidence/v1` records what one diagnostic check observed.
This package answers the questions that come before the observation: which
provider was allowed to run, under which declared bounds, over which changed
files, and what its terminal behavior was. It keeps the same discipline:

- Providers are an allowlist (`DIAGNOSTIC_PROVIDER_IDS`). A capability,
  outcome, or diagnostic item naming anything else is refused, and a
  diagnostic's `source` field must name an allowlisted provider, so the one
  `source` key can never carry file content.
- Every capability declares per-provider bounds, and global caps refuse any
  capability, request, or record that declares or carries more than the
  contract allows.
- Outcomes are a closed vocabulary -- ok, timeout, cancelled, crashed,
  partial, unsupported, stale -- derived from the terminal state, the scope,
  and the revisions. A caller cannot select one; a caller can only cause one.
- `revisions_identical` is the moving-HEAD equality predicate: a revision
  named `HEAD` (or any moving ref) is identical to nothing, not even itself,
  because it may have moved between the two reads it would be comparing. A
  check observed "at HEAD" is therefore always `stale`, never fresh.
- Every persistable shape is metadata only. There is no message, prompt,
  body, snippet, diff, or log field, and an input diagnostic carrying one is
  refused by key name.
- Like the v1 record, an outcome can settle only the claim
  `fresh_language_diagnostic_check`. Compilation, tests, review, CI, and
  merge are refused structurally, for every outcome including a clean one.

OMH observes nothing here either. The scheduler starts no language server,
and the outcome records only what a caller reports about a provider it ran
itself. The scheduler is stateful and serial: one check is active at a time,
selection falls back past disabled providers and providers whose last check
timed out or crashed, and a provider that recovers is re-selected only when
marked available again.

This module is the public facade; the contract lives in single-responsibility
modules behind it: `diagnostic_provider_models` (vocabularies, bounds,
reference rules, moving-revision predicates), `diagnostic_provider_parse`
(normalization and derivation), `diagnostic_provider_config` (capabilities and
config identity), `diagnostic_provider_scope` (request and ticket),
`diagnostic_provider_outcomes` (the outcome record and its builder),
`diagnostic_provider_claims` (the claim boundary), `diagnostic_provider_validate`
(read-path record validation), and `diagnostic_provider_scheduler` (the
serial scheduler with fallback).
"""

from __future__ import annotations

from .diagnostic_provider_claims import diagnostic_claim_support, diagnostic_outcome_supports_claim
from .diagnostic_provider_config import (
    DEFAULT_PROVIDER_CAPABILITIES,
    DiagnosticProviderConfig,
    ProviderCapability,
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
    GLOBAL_MAX_TIMEOUT_MS,
    MIN_TIMEOUT_MS,
    MOVING_REVISION_REFS,
    RAW_PAYLOAD_FIELD_NAMES,
    DiagnosticProviderError,
    is_moving_revision,
    revisions_identical,
)
from .diagnostic_provider_outcomes import DiagnosticCheckOutcome, build_diagnostic_check_outcome
from .diagnostic_provider_parse import DiagnosticItem, normalize_diagnostic_item
from .diagnostic_provider_scheduler import DiagnosticProviderScheduler
from .diagnostic_provider_scope import DiagnosticCheckRequest, DiagnosticCheckTicket
from .diagnostic_provider_validate import validate_diagnostic_outcome_record

__all__ = (
    "DEFAULT_PROVIDER_CAPABILITIES",
    "DIAGNOSTIC_COMPATIBILITY_MARKERS",
    "DIAGNOSTIC_ITEM_KEYS",
    "DIAGNOSTIC_OUTCOMES",
    "DIAGNOSTIC_OUTCOME_RECORD_KEYS",
    "DIAGNOSTIC_PROVIDER_IDS",
    "DIAGNOSTIC_PROVIDERS_PRIVACY",
    "DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION",
    "DIAGNOSTIC_TERMINAL_STATES",
    "GLOBAL_MAX_DIAGNOSTICS_PER_CHECK",
    "GLOBAL_MAX_FILES_PER_CHECK",
    "GLOBAL_MAX_TIMEOUT_MS",
    "MIN_TIMEOUT_MS",
    "MOVING_REVISION_REFS",
    "RAW_PAYLOAD_FIELD_NAMES",
    "DiagnosticCheckOutcome",
    "DiagnosticCheckRequest",
    "DiagnosticCheckTicket",
    "DiagnosticItem",
    "DiagnosticProviderConfig",
    "DiagnosticProviderError",
    "DiagnosticProviderScheduler",
    "ProviderCapability",
    "build_diagnostic_check_outcome",
    "diagnostic_claim_support",
    "diagnostic_outcome_supports_claim",
    "is_moving_revision",
    "normalize_diagnostic_item",
    "revisions_identical",
    "validate_diagnostic_outcome_record",
)
