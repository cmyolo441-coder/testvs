"""Which claims a diagnostic outcome can settle: one, and only one.

A diagnostic outcome can support only `fresh_language_diagnostic_check`.
Compilation, tests, review, CI, merge-readiness, and merge are refused
structurally for every outcome this contract can build, including a clean
one, because no outcome name reads as any of them and no argument selects
them.
"""

from __future__ import annotations

from ..quality.language_diagnostic_evidence import (
    LANGUAGE_DIAGNOSTIC_CLAIMS,
    LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY,
    LANGUAGE_DIAGNOSTIC_SUPPORTABLE_CLAIMS,
)
from .diagnostic_provider_models import DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION, revisions_identical
from .diagnostic_provider_outcomes import DiagnosticCheckOutcome


def diagnostic_outcome_supports_claim(outcome: DiagnosticCheckOutcome, claim: str) -> bool:
    """True only for a fresh, attributable check asked about its own claim.

    Every other question -- verification, compilation, tests, review, CI,
    merge-readiness, merge -- is False for every outcome this contract can
    build, and so is the one supportable claim for every outcome except a
    fresh `ok`.
    """
    if claim not in LANGUAGE_DIAGNOSTIC_SUPPORTABLE_CLAIMS:
        return False
    if outcome.outcome != "ok":
        return False
    if not (outcome.workspace_id and outcome.revision and outcome.provider_id):
        return False
    return revisions_identical(outcome.revision, outcome.diagnostics_revision)


def diagnostic_claim_support(outcome: DiagnosticCheckOutcome) -> dict[str, object]:
    """One reportable answer per claim, for a status surface to render."""
    return {
        "schema_version": DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION,
        "supported_claims": [
            claim for claim in LANGUAGE_DIAGNOSTIC_CLAIMS if diagnostic_outcome_supports_claim(outcome, claim)
        ],
        "unsupported_claims": [
            claim for claim in LANGUAGE_DIAGNOSTIC_CLAIMS if not diagnostic_outcome_supports_claim(outcome, claim)
        ],
        "claim_boundary": LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY,
    }
