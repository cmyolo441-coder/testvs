"""Reviewed enrollment, not an assertion that all public documentation is covered.

Probe names are identifiers in a closed implementation allowlist, never commands
or expressions supplied by a document. Expected facts are machine values; prose
may be rewritten without changing a claim's ID or pinning a sentence in tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImplementationAnchor:
    path: str
    symbol: str


@dataclass(frozen=True)
class DocumentationClaim:
    claim_id: str
    question: str
    invariant: str
    expected_fact: bool | str
    pages: tuple[str, ...]
    anchors: tuple[ImplementationAnchor, ...]
    mode: str
    probe: str
    risk: str
    owner: str


def documentation_claims() -> tuple[DocumentationClaim, ...]:
    return (
        DocumentationClaim(
            "release.checklist-prepared",
            "Does rendering the release checklist observe release execution?",
            "The release checklist's top-level observed flag is false.",
            False, ("docs/RELEASE.md",),
            (ImplementationAnchor("src/maintenance/release.py", "release_readiness_checklist"),
             ImplementationAnchor("src/commands/release.py", "cmd_release_checklist")),
            "cli_probe", "release-checklist-observed", "high", "release-readiness",
        ),
        DocumentationClaim(
            "reporting.rate-schema",
            "Which schema identifies a reported rate with its denominator?",
            "Reported rates use the versioned omh_reported_rate/v1 contract.",
            "omh_reported_rate/v1", ("AGENTS.md", "docs/DOCUMENTATION-CLAIMS.md"),
            (ImplementationAnchor("src/quality/reported_rate.py", "REPORTED_RATE_SCHEMA_VERSION"),),
            "schema_assertion", "reported-rate-schema", "high", "docs-specialist",
        ),
        DocumentationClaim(
            "release.checklist-symbol",
            "Is the programmatic release checklist entry point available?",
            "release_readiness_checklist is an importable callable.",
            True, ("docs/RELEASE.md", "docs/DOCUMENTATION-CLAIMS.md"),
            (ImplementationAnchor("src/maintenance/release.py", "release_readiness_checklist"),),
            "symbol_check", "release-checklist-symbol", "medium", "release-readiness",
        ),
        DocumentationClaim(
            "reporting.empty-rate",
            "Does a rate with no observations avoid reporting zero percent?",
            "A zero-denominator reported rate has percent=null and basis=no_observations.",
            True, ("AGENTS.md",),
            (ImplementationAnchor("src/quality/reported_rate.py", "reported_rate"),),
            "fixture_behavior", "empty-reported-rate", "high", "docs-specialist",
        ),
        DocumentationClaim(
            "generated.roles-equality",
            "Does the shipped role reference equal the canonical renderer?",
            "docs/ROLES.md content, after CRLF-to-LF translation, equals roles_reference_markdown output.",
            True, ("docs/ROLES.md", "CLAUDE.md"),
            (ImplementationAnchor("src/catalogs/roles.py", "roles_reference_markdown"),),
            "render_equality", "roles-equality", "medium", "docs-specialist",
        ),
        DocumentationClaim(
            "docs.evidence-language",
            "Does the audit guide distinguish prepared documentation work from observed audit results?",
            "The guide does not represent catalog enrollment or a prepared checklist as observed verification.",
            True, ("docs/DOCUMENTATION-CLAIMS.md",),
            (ImplementationAnchor("src/commands/docs.py", "cmd_docs_claims"),),
            "model_assisted", "evidence-language", "low", "docs-specialist",
        ),
    )
