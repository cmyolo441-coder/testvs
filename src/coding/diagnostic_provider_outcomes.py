"""The persistable outcome of one serial diagnostic check.

This is the outcome half of the `diagnostic_providers/v1` contract: the
metadata-only record shape, and the builder that derives `outcome` and
`outcome_id` from the compatibility marker, terminal state, revisions, and
scope, so a caller supplies what happened and never what it proves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..quality.language_diagnostic_evidence import (
    LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY,
    LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR,
)
from .diagnostic_provider_models import (
    DIAGNOSTIC_COMPATIBILITY_MARKERS,
    DIAGNOSTIC_PROVIDER_IDS,
    DIAGNOSTIC_PROVIDERS_PRIVACY,
    DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION,
    DIAGNOSTIC_TERMINAL_STATES,
    GLOBAL_MAX_FILES_PER_CHECK,
    DiagnosticProviderError,
    _checked_ref,
)
from .diagnostic_provider_parse import (
    DiagnosticItem,
    _derive_outcome,
    _normalized_diagnostics,
    _normalized_files,
    _outcome_id,
)


@dataclass(frozen=True)
class DiagnosticCheckOutcome:
    """The persistable result of one serial check, metadata only.

    `outcome` and `outcome_id` are derived, never supplied. The compatibility
    marker says how the check related to the allowlisted providers, and
    `language_diagnostic_check_state` projects the outcome into the
    `language_diagnostic_evidence/v1` check-state vocabulary so a reader of
    either contract sees the same classification.
    """

    workspace_id: str
    revision: str
    diagnostics_revision: str
    provider_id: str
    terminal_state: str
    outcome: str
    compatibility: str
    in_scope_files: tuple[str, ...]
    out_of_scope_files: tuple[str, ...]
    diagnosed_files: tuple[str, ...]
    diagnostics: tuple[DiagnosticItem, ...]
    config_identity: str
    outcome_id: str
    schema_version: str = DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION
    privacy: str = DIAGNOSTIC_PROVIDERS_PRIVACY
    claim_boundary: str = LANGUAGE_DIAGNOSTIC_CLAIM_BOUNDARY
    not_evidence_for: tuple[str, ...] = LANGUAGE_DIAGNOSTIC_NOT_EVIDENCE_FOR

    def language_diagnostic_check_state(self) -> str:
        if self.compatibility == "provider_disabled":
            return "unsupported"
        if self.outcome in ("timeout", "crashed"):
            return "failed"
        if self.outcome == "cancelled":
            return "not_observed"
        return "observed"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outcome_id": self.outcome_id,
            "workspace_id": self.workspace_id,
            "revision": self.revision,
            "diagnostics_revision": self.diagnostics_revision,
            "provider_id": self.provider_id,
            "terminal_state": self.terminal_state,
            "outcome": self.outcome,
            "compatibility": self.compatibility,
            "in_scope_files": list(self.in_scope_files),
            "out_of_scope_files": list(self.out_of_scope_files),
            "changed_files": sorted({*self.in_scope_files, *self.out_of_scope_files}),
            "diagnosed_files": list(self.diagnosed_files),
            "diagnostics": [item.as_record() for item in self.diagnostics],
            "config_identity": self.config_identity,
            "privacy": self.privacy,
            "claim_boundary": self.claim_boundary,
            "not_evidence_for": list(self.not_evidence_for),
        }


def build_diagnostic_check_outcome(
    *,
    workspace_id: str,
    revision: str,
    diagnostics_revision: str,
    provider_id: str = "",
    terminal_state: str = "completed",
    compatibility: str = "provider_selected",
    in_scope_files: Iterable[str] = (),
    out_of_scope_files: Iterable[str] = (),
    diagnosed_files: Iterable[str] = (),
    diagnostics: Iterable[Mapping[str, object]] = (),
    config_identity: str = "",
) -> DiagnosticCheckOutcome:
    """Build one outcome, or refuse.

    Nothing about the classification is a parameter: `outcome` is derived from
    the compatibility marker, the terminal state, the revisions, and the
    scope, so a caller supplies what happened and never what it proves.
    """
    if terminal_state not in DIAGNOSTIC_TERMINAL_STATES:
        raise DiagnosticProviderError(
            f"diagnostic_providers terminal_state is unsupported: {terminal_state!r}"
        )
    if compatibility not in DIAGNOSTIC_COMPATIBILITY_MARKERS:
        raise DiagnosticProviderError(
            f"diagnostic_providers compatibility is unsupported: {compatibility!r}"
        )
    if compatibility == "provider_disabled":
        if provider_id:
            raise DiagnosticProviderError(
                "diagnostic_providers provider_disabled marker cannot name a provider"
            )
        if terminal_state != "completed":
            raise DiagnosticProviderError(
                "diagnostic_providers provider_disabled marker cannot carry a terminal state"
            )
    elif provider_id not in DIAGNOSTIC_PROVIDER_IDS:
        raise DiagnosticProviderError(
            f"diagnostic_providers provider_id is not allowlisted: {provider_id!r}"
        )
    safe_workspace = _checked_ref(workspace_id, field="diagnostic_providers workspace_id", required=True)
    safe_revision = _checked_ref(revision, field="diagnostic_providers revision", required=True)
    safe_diagnostics_revision = _checked_ref(
        diagnostics_revision, field="diagnostic_providers diagnostics_revision", required=False
    )
    safe_config_identity = _checked_ref(
        config_identity, field="diagnostic_providers config_identity", required=False
    )
    scope_in = _normalized_files(in_scope_files, field="in_scope_files", bound=GLOBAL_MAX_FILES_PER_CHECK)
    scope_out = _normalized_files(out_of_scope_files, field="out_of_scope_files", bound=GLOBAL_MAX_FILES_PER_CHECK)
    diagnosed = _normalized_files(diagnosed_files, field="diagnosed_files", bound=GLOBAL_MAX_FILES_PER_CHECK)
    if set(scope_in) & set(scope_out):
        raise DiagnosticProviderError(
            "diagnostic_providers in_scope_files and out_of_scope_files must not overlap"
        )
    if not set(diagnosed) <= set(scope_in):
        raise DiagnosticProviderError(
            "diagnostic_providers diagnosed_files must stay inside in_scope_files"
        )
    items = _normalized_diagnostics(diagnostics, scope=set(scope_in))
    outcome = _derive_outcome(
        compatibility, provider_id, terminal_state, safe_revision, safe_diagnostics_revision, scope_in, diagnosed
    )
    return DiagnosticCheckOutcome(
        workspace_id=safe_workspace,
        revision=safe_revision,
        diagnostics_revision=safe_diagnostics_revision,
        provider_id=provider_id,
        terminal_state=terminal_state,
        outcome=outcome,
        compatibility=compatibility,
        in_scope_files=scope_in,
        out_of_scope_files=scope_out,
        diagnosed_files=diagnosed,
        diagnostics=items,
        config_identity=safe_config_identity,
        outcome_id=_outcome_id(
            workspace_id=safe_workspace,
            revision=safe_revision,
            diagnostics_revision=safe_diagnostics_revision,
            provider_id=provider_id,
            terminal_state=terminal_state,
            outcome=outcome,
            compatibility=compatibility,
            in_scope_files=scope_in,
            out_of_scope_files=scope_out,
            diagnosed_files=diagnosed,
            diagnostics=items,
            config_identity=safe_config_identity,
        ),
    )
