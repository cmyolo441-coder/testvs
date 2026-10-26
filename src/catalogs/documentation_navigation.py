"""Checked-in navigation policy for public documentation structure.

This declares three things and nothing else: which pages are entry points, which
pages must be reachable from one of them, and which pages are intentionally not
reachable and why. Every in-scope page is settled by exactly one of those, so an
accidentally orphaned page fails instead of blending into a silent allowlist.

Reachability here means discoverability through declared local Markdown links.
It is not information-architecture quality, accessibility, semantic accuracy, or
evidence that a page was published anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


PAGE_CLASSES = (
    "public",
    "generated",
    "maintainer_procedure",
    "repo_internal",
    "working_note",
)
REACHABILITY_STATES = ("required", "exempt")

# The publishing surface this policy scopes itself to. GitHub's blob view serves
# a page at its repository path, so two pages collide only when their paths
# differ by case alone -- which is exactly what breaks a clone on a
# case-insensitive filesystem. Static-site generators add slug semantics that
# this policy deliberately does not claim to model.
PUBLISHING_SURFACE = "github_repository_blob"


@dataclass(frozen=True)
class NavigationRoot:
    """A declared entry point. Traversal starts here; roots need no referrer."""

    path: str
    reason: str


@dataclass(frozen=True)
class PageClassification:
    """One in-scope page's declared class and reachability expectation.

    `reachability="required"` states the page must be reachable from a root, and
    is used to classify a page whose kind might otherwise invite a blanket
    exclusion -- a generated page is still a public page. `exempt` states the
    page is intentionally not on a public path, and `reason` must name where a
    reader actually enters it.
    """

    path: str
    page_class: str
    reachability: str
    reason: str
    owner: str


def documentation_navigation_roots() -> tuple[NavigationRoot, ...]:
    return (
        NavigationRoot(
            "README.md",
            "Repository front page; the entry point for someone arriving at the project.",
        ),
        NavigationRoot(
            "docs/README.md",
            "The public operating map; the entry point for someone looking for a contract.",
        ),
    )


def documentation_scope_patterns() -> tuple[str, ...]:
    """Pages whose reachability is required unless classified exempt.

    Top-level `docs/` only. Deeper trees, `skills/` output, and repository-root
    contracts other than the declared roots are traversed when a link reaches
    them and their own links are validated, but they are not required to be
    reachable, because they are not the public documentation surface.
    """
    return ("docs/*.md",)


def documentation_page_classifications() -> tuple[PageClassification, ...]:
    return (
        PageClassification(
            "docs/AGENT-SKILLS.md",
            "public",
            "required",
            "Portable projection installation and host capability boundaries, linked from docs/README.md.",
            "docs-specialist",
        ),
        PageClassification(
            "docs/ADDING-A-SKILL.md",
            "repo_internal",
            "exempt",
            "Contributor checklist for this repository's catalog. Entered from CLAUDE.md "
            "Common Pitfalls and REVIEW.md, not from a product reader path.",
            "docs-specialist",
        ),
        PageClassification(
            "docs/CI-TEST-SHARDING.md",
            "repo_internal",
            "exempt",
            "Describes this repository's own CI workflow shape. Entered from CONTRIBUTING.md; "
            "it documents nothing a user or integrator of OMH can run.",
            "release-readiness",
        ),
        PageClassification(
            "docs/REVIEW-SWEEP.md",
            "maintainer_procedure",
            "exempt",
            "Repository-side maintainer procedure. Indexed by the AGENTS.md Repository "
            "Maintenance Procedures table.",
            "docs-specialist",
        ),
        PageClassification(
            "docs/TRIAGE-SWEEP.md",
            "maintainer_procedure",
            "exempt",
            "Repository-side maintainer procedure. Indexed by the AGENTS.md Repository "
            "Maintenance Procedures table.",
            "docs-specialist",
        ),
        PageClassification(
            "docs/routing-quality.md",
            "repo_internal",
            "exempt",
            "Describes the golden evaluation in tests/test_routing_quality.py and its baseline. "
            "Entered from docs/ADDING-A-SKILL.md; it is a test-gate note, not a product contract.",
            "docs-specialist",
        ),
        PageClassification(
            "docs/WORKFLOWS.md",
            "generated",
            "required",
            "Generated from src/skills/catalog.py, and also the public skill catalog readers are "
            "sent to. Generated is a provenance fact here, not a reason to drop it from the map.",
            "docs-specialist",
        ),
        PageClassification(
            "docs/ROLES.md",
            "generated",
            "required",
            "Generated from roles_reference_markdown(), and also the public role reference. "
            "Classified rather than excluded for the same reason as docs/WORKFLOWS.md.",
            "docs-specialist",
        ),
    )
