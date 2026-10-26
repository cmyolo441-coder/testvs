"""Offline documentation-structure audit: reachability, local targets, collisions.

This is a separate evidence class from the two that already exist. Generated-file
equality (`docs workflows|roles|capability-families|ulw-*`) proves a projection
still matches its producer; the selected-claim audit (`docs claims`) proves a
sentence still matches implementation. Neither can see a page that quietly left
the map, and neither is repaired the same way, so this check does not fold into
them and does not repeat their assertions.

Everything here is local and deterministic: no network, no subprocess, no
provider, and no automatic edit to any documentation file. A page being
structurally reachable says nothing about whether its content is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import TypedDict
from urllib.parse import unquote

from ..catalogs.documentation_navigation import (
    PAGE_CLASSES,
    PUBLISHING_SURFACE,
    REACHABILITY_STATES,
    documentation_navigation_roots,
    documentation_page_classifications,
    documentation_scope_patterns,
)


DOCUMENTATION_NAVIGATION_SCHEMA = "documentation_navigation_audit/v1"

# docs/WORKFLOWS.md is ~1 MiB of generated catalog today. The ceiling is set
# above that with room to grow, and a page that crosses it is reported rather
# than skipped, so growth stays visible instead of silently losing coverage.
MAX_PAGE_BYTES = 4 * 1024 * 1024
MAX_TRAVERSED_PAGES = 500
MAX_LINKS_PER_PAGE = 4000
# Diagnostics name paths and link targets only, never document prose. A target
# is still document-controlled text, so it is length-capped and control
# characters are escaped before it reaches any output stream.
MAX_DIAGNOSTIC_CHARS = 200

FAILURE_CLASSES = (
    "absolute_link_target",
    "duplicate_classification",
    "duplicate_navigation_root",
    "duplicate_public_path",
    "invalid_classification",
    "link_escapes_repository",
    "missing_heading_anchor",
    "missing_link_target",
    "missing_navigation_root",
    "page_over_budget",
    "required_page_unreachable",
    "stale_classification",
    "traversal_budget_exceeded",
    "unclassified_orphan",
    "unnecessary_exclusion",
    "windows_separator_link",
)
DEFAULT_OWNER = "docs-specialist"

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_INLINE_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(<[^>\n]*>|[^)\s]*)(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?\s*\)")
_REFERENCE_DEFINITION = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*(<[^>\n]*>|\S+)")
_MARKDOWN_LINK_TEXT = re.compile(r"!?\[([^\]\n]*)\]\([^)\n]*\)")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_SLUG_DROP = re.compile(r"[^a-z0-9 _\-]")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class NavigationFinding(TypedDict):
    failure_class: str
    page: str
    referrer: str | None
    target: str | None
    detail: str
    owner: str


class NavigationPageRow(TypedDict):
    path: str
    page_class: str
    reachability: str
    reachable: bool
    referrers: list[str]
    owner: str
    reason: str


class DocumentationNavigationReport(TypedDict):
    schema_version: str
    mode: str
    observed: bool
    ok: bool
    publishing_surface: str
    roots: list[dict[str, str]]
    scope_patterns: list[str]
    pages: list[NavigationPageRow]
    findings: list[NavigationFinding]
    advisories: list[NavigationFinding]
    summary: dict[str, int]
    bounds: dict[str, object]
    anchor_scope: str
    structure_boundary: str


@dataclass(frozen=True)
class _Link:
    """One local link edge, already normalized to a repository-relative path."""

    page: str
    raw: str
    target: str
    fragment: str


def _safe_diagnostic(value: str) -> str:
    """Render document-controlled text as a bounded single-line diagnostic."""
    cleaned = _CONTROL.sub("�", value)
    if len(cleaned) > MAX_DIAGNOSTIC_CHARS:
        return cleaned[:MAX_DIAGNOSTIC_CHARS] + "..."
    return cleaned


def _strip_code(text: str) -> str:
    """Blank out fenced blocks and inline code spans, keeping line numbering.

    A shell comment inside a fence starts with `#` and would otherwise be read
    as a heading, and example paths inside fences are not navigation edges.
    """
    lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match:
            token = match.group(1)
            if fence is None:
                fence = token[0] * 3
                lines.append("")
                continue
            if line.lstrip().startswith(fence):
                fence = None
                lines.append("")
                continue
        lines.append("" if fence is not None else line)
    return _INLINE_CODE.sub("", "\n".join(lines))


def _heading_slugs(text: str) -> tuple[frozenset[str], bool]:
    """Return heading anchors this parser can settle, and whether all were settled.

    The declared rule for a settled heading: drop markdown link syntax, lowercase,
    remove every character outside `[a-z0-9 _-]`, turn spaces into hyphens, and
    disambiguate repeats with `-1`, `-2`. A heading holding any non-ASCII
    character is left unsettled, because renderers disagree on how to transliterate
    it and this parser will not guess on their behalf.
    """
    slugs: dict[str, int] = {}
    settled = True
    for line in _strip_code(text).splitlines():
        match = _HEADING.match(line)
        if not match:
            continue
        heading = _MARKDOWN_LINK_TEXT.sub(r"\1", match.group(1)).strip()
        if not heading.isascii():
            settled = False
            continue
        base = _SLUG_DROP.sub("", heading.lower()).replace(" ", "-")
        if not base:
            settled = False
            continue
        seen = slugs.get(base, 0)
        slugs[base] = seen + 1
    resolved = set()
    for base, count in slugs.items():
        resolved.add(base)
        resolved.update(f"{base}-{index}" for index in range(1, count))
    return frozenset(resolved), settled


def _link_targets(text: str) -> list[str]:
    """Collect inline-link, image, and reference-definition targets in order."""
    body = _strip_code(text)
    targets: list[str] = []
    for match in _INLINE_LINK.finditer(body):
        targets.append(match.group(1))
        if len(targets) >= MAX_LINKS_PER_PAGE:
            return targets
    for line in body.splitlines():
        match = _REFERENCE_DEFINITION.match(line)
        if match:
            targets.append(match.group(1))
            if len(targets) >= MAX_LINKS_PER_PAGE:
                return targets
    return targets


def _is_external(target: str) -> bool:
    """True for anything a local filesystem check must not attempt to resolve."""
    if target.startswith("//"):
        return True
    return bool(_SCHEME.match(target)) and not _WINDOWS_DRIVE.match(target)


def _normalize_relative(page: str, target: str) -> list[str] | None:
    """Resolve `target` against `page`'s directory purely lexically.

    Lexical resolution is the point: `Path.resolve()` consults the filesystem and
    follows symlinks, so it would answer differently per machine and could walk
    outside the repository through a link. `None` means the path escaped.
    """
    parts = page.split("/")[:-1]
    for segment in target.split("/"):
        if segment in ("", "."):
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    return parts


def _exists_exact(root: Path, parts: list[str], cache: dict[str, frozenset[str]]) -> bool:
    """Case-sensitive existence check, independent of the host filesystem.

    macOS and Windows resolve `docs/readme.md` to `docs/README.md`; Linux does
    not. Matching against each directory's real entry names keeps the verdict
    identical everywhere, so a link that only works on a case-insensitive
    checkout still fails here.
    """
    if not parts:
        return False
    current = ""
    for segment in parts:
        entries = cache.get(current)
        if entries is None:
            directory = root if not current else root.joinpath(*current.split("/"))
            try:
                with os.scandir(directory) as scan:
                    entries = frozenset(entry.name for entry in scan)
            except OSError:
                # Not a directory, or unreadable: nothing below it can resolve.
                entries = frozenset()
            cache[current] = entries
        if segment not in entries:
            return False
        current = segment if not current else f"{current}/{segment}"
    return True


def _read_page(root: Path, page: str) -> tuple[str | None, str | None]:
    """Read one page within budget. Returns (text, failure_class)."""
    path = root.joinpath(*page.split("/"))
    try:
        size = path.stat().st_size
    except OSError:
        return None, "missing_link_target"
    if size > MAX_PAGE_BYTES:
        return None, "page_over_budget"
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except OSError:
        return None, "missing_link_target"


def _classification_owner(by_path: dict[str, object], page: str) -> str:
    entry = by_path.get(page)
    return getattr(entry, "owner", DEFAULT_OWNER)


def _scope_pages(root: Path, patterns: tuple[str, ...]) -> list[str]:
    pages: set[str] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file():
                pages.add(path.relative_to(root).as_posix())
    return sorted(pages)


def public_path_collisions(paths: list[str]) -> list[NavigationFinding]:
    """Report pages whose public paths differ only by case.

    Exposed as its own function because the condition cannot be staged on a
    case-insensitive filesystem: macOS and Windows refuse to hold both files at
    once, which is precisely the breakage this catches when a page pair is
    created on Linux and then cloned somewhere else.
    """
    lowered: dict[str, set[str]] = {}
    for path in paths:
        lowered.setdefault(path.lower(), set()).add(path)
    collisions: list[NavigationFinding] = []
    for key, group in sorted(lowered.items()):
        if len(group) > 1:
            collisions.append({
                "failure_class": "duplicate_public_path", "page": sorted(group)[0],
                "referrer": None, "target": key, "owner": DEFAULT_OWNER,
                "detail": f"Pages differing only by case collide on the {PUBLISHING_SURFACE} "
                          "surface: " + ", ".join(sorted(group)),
            })
    return collisions


def documentation_navigation_report(*, root: Path | None = None) -> DocumentationNavigationReport:
    """Walk declared roots, validate every local link, and settle every in-scope page.

    Findings block; advisories do not. The only advisory class is an anchor this
    parser refuses to settle, which is a limit of the parser rather than a defect
    in the page.
    """
    resolved_root = (root or Path.cwd()).resolve()
    roots = documentation_navigation_roots()
    classifications = documentation_page_classifications()
    patterns = documentation_scope_patterns()

    findings: list[NavigationFinding] = []
    advisories: list[NavigationFinding] = []

    by_path: dict[str, object] = {}
    for entry in classifications:
        if entry.path in by_path:
            findings.append({
                "failure_class": "duplicate_classification", "page": entry.path,
                "referrer": None, "target": None, "owner": entry.owner,
                "detail": "Classified more than once; keep exactly one entry per page.",
            })
            continue
        if entry.page_class not in PAGE_CLASSES or entry.reachability not in REACHABILITY_STATES:
            findings.append({
                "failure_class": "invalid_classification", "page": entry.path,
                "referrer": None, "target": None, "owner": entry.owner,
                "detail": f"page_class must be one of {PAGE_CLASSES} and reachability one of {REACHABILITY_STATES}.",
            })
            continue
        by_path[entry.path] = entry

    root_paths: list[str] = []
    seen_roots: set[str] = set()
    dir_cache: dict[str, frozenset[str]] = {}
    for declared in roots:
        if declared.path in seen_roots:
            findings.append({
                "failure_class": "duplicate_navigation_root", "page": declared.path,
                "referrer": None, "target": None, "owner": DEFAULT_OWNER,
                "detail": "Declared as a navigation root more than once.",
            })
            continue
        seen_roots.add(declared.path)
        if not _exists_exact(resolved_root, declared.path.split("/"), dir_cache):
            findings.append({
                "failure_class": "missing_navigation_root", "page": declared.path,
                "referrer": None, "target": None, "owner": DEFAULT_OWNER,
                "detail": "Declared navigation root does not exist in the repository.",
            })
            continue
        root_paths.append(declared.path)

    scope = _scope_pages(resolved_root, patterns)
    reachable: dict[str, list[str]] = {path: [] for path in root_paths}
    queue: list[str] = list(root_paths)
    visited: set[str] = set()
    anchors: dict[str, tuple[frozenset[str], bool]] = {}
    pending_anchor_checks: list[_Link] = []
    budget_exceeded = False

    # Traversal reads reachable pages plus every in-scope page: an orphan's own
    # broken links are the ones nobody is looking at, so they are checked too.
    parse_order = list(root_paths) + [path for path in scope if path not in reachable]
    parsed: set[str] = set()

    while queue or parse_order:
        page = queue.pop(0) if queue else parse_order.pop(0)
        if page in visited:
            continue
        if len(visited) >= MAX_TRAVERSED_PAGES:
            budget_exceeded = True
            break
        visited.add(page)
        parsed.add(page)
        text, failure = _read_page(resolved_root, page)
        if failure == "page_over_budget":
            findings.append({
                "failure_class": "page_over_budget", "page": page,
                "referrer": None, "target": None,
                "owner": _classification_owner(by_path, page),
                "detail": f"Page exceeds the {MAX_PAGE_BYTES}-byte parse budget and was not parsed.",
            })
            continue
        if text is None:
            continue
        anchors[page] = _heading_slugs(text)
        for raw in _link_targets(text):
            target = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
            if not target or _is_external(target):
                continue
            path_part, _, fragment = target.partition("#")
            if "\\" in path_part:
                findings.append({
                    "failure_class": "windows_separator_link", "page": page,
                    "referrer": None, "target": _safe_diagnostic(target),
                    "owner": _classification_owner(by_path, page),
                    "detail": "Link uses a backslash separator; write local paths with '/'.",
                })
                continue
            if not path_part:
                pending_anchor_checks.append(_Link(page, target, page, unquote(fragment)))
                continue
            decoded = unquote(path_part)
            if decoded.startswith("/") or _WINDOWS_DRIVE.match(decoded):
                findings.append({
                    "failure_class": "absolute_link_target", "page": page,
                    "referrer": None, "target": _safe_diagnostic(target),
                    "owner": _classification_owner(by_path, page),
                    "detail": "Absolute local path does not resolve in a repository view; use a relative path.",
                })
                continue
            parts = _normalize_relative(page, decoded)
            if parts is None:
                findings.append({
                    "failure_class": "link_escapes_repository", "page": page,
                    "referrer": None, "target": _safe_diagnostic(target),
                    "owner": _classification_owner(by_path, page),
                    "detail": "Link resolves above the repository root.",
                })
                continue
            if not _exists_exact(resolved_root, parts, dir_cache):
                findings.append({
                    "failure_class": "missing_link_target", "page": page,
                    "referrer": None, "target": _safe_diagnostic(target),
                    "owner": _classification_owner(by_path, page),
                    "detail": "Local target does not exist (matched case-sensitively).",
                })
                continue
            resolved_target = "/".join(parts)
            if fragment and resolved_target.endswith(".md"):
                pending_anchor_checks.append(_Link(page, target, resolved_target, unquote(fragment)))
            if not resolved_target.endswith(".md"):
                continue
            if page in reachable or page in root_paths:
                referrers = reachable.setdefault(resolved_target, [])
                if page not in referrers and resolved_target not in root_paths:
                    referrers.append(page)
                if resolved_target not in visited:
                    queue.append(resolved_target)

    if budget_exceeded:
        findings.append({
            "failure_class": "traversal_budget_exceeded", "page": "",
            "referrer": None, "target": None, "owner": DEFAULT_OWNER,
            "detail": f"Traversal stopped after {MAX_TRAVERSED_PAGES} pages; results are incomplete.",
        })

    for link in pending_anchor_checks:
        known = anchors.get(link.target)
        if known is None:
            text, _ = _read_page(resolved_root, link.target)
            if text is None:
                continue
            known = _heading_slugs(text)
            anchors[link.target] = known
        slugs, settled = known
        if link.fragment in slugs:
            continue
        row: NavigationFinding = {
            "failure_class": "missing_heading_anchor", "page": link.target,
            "referrer": link.page, "target": _safe_diagnostic(link.raw),
            "owner": _classification_owner(by_path, link.target),
            "detail": "No heading on the target page produces this anchor.",
        }
        if settled:
            findings.append(row)
        else:
            advisories.append({
                **row,
                "detail": "Anchor unsettled: the target page has a heading this parser will not slug.",
            })

    findings.extend(public_path_collisions(scope + root_paths))

    for entry in classifications:
        if by_path.get(entry.path) is not entry:
            continue
        if entry.path not in scope and entry.path not in root_paths:
            findings.append({
                "failure_class": "stale_classification", "page": entry.path,
                "referrer": None, "target": None, "owner": entry.owner,
                "detail": "Classified page is not in scope; remove the entry or restore the page.",
            })

    pages: list[NavigationPageRow] = []
    for page in scope:
        entry = by_path.get(page)
        is_reachable = page in reachable or page in root_paths
        page_class = getattr(entry, "page_class", "public")
        expectation = getattr(entry, "reachability", "required")
        reason = getattr(entry, "reason", "")
        owner = getattr(entry, "owner", DEFAULT_OWNER)
        if expectation == "exempt" and is_reachable:
            findings.append({
                "failure_class": "unnecessary_exclusion", "page": page,
                "referrer": sorted(reachable.get(page, []))[0] if reachable.get(page) else None,
                "target": None, "owner": owner,
                "detail": "Page is reachable but still classified exempt; drop the classification entry.",
            })
        elif not is_reachable and expectation == "required":
            failure_class = "required_page_unreachable" if entry is not None else "unclassified_orphan"
            detail = (
                "Classified as reachability=required but no declared root reaches it; "
                "add the link or change the classification."
                if entry is not None else
                "Not reachable from any declared root and not classified; link it from a root "
                "or add a reasoned entry to documentation_page_classifications()."
            )
            findings.append({
                "failure_class": failure_class, "page": page,
                "referrer": None, "target": None, "owner": owner, "detail": detail,
            })
        pages.append({
            "path": page, "page_class": page_class, "reachability": expectation,
            "reachable": is_reachable, "referrers": sorted(reachable.get(page, [])),
            "owner": owner, "reason": reason,
        })

    findings.sort(key=lambda row: (row["failure_class"], row["page"], row["referrer"] or "", row["target"] or ""))
    advisories.sort(key=lambda row: (row["failure_class"], row["page"], row["referrer"] or "", row["target"] or ""))
    return {
        "schema_version": DOCUMENTATION_NAVIGATION_SCHEMA,
        "mode": "observed_structure",
        "observed": True,
        "ok": not findings,
        "publishing_surface": PUBLISHING_SURFACE,
        "roots": [{"path": item.path, "reason": item.reason} for item in roots],
        "scope_patterns": list(patterns),
        "pages": pages,
        "findings": findings,
        "advisories": advisories,
        "summary": {
            "scope_pages": len(pages),
            "reachable_pages": sum(row["reachable"] for row in pages),
            "exempt_pages": sum(row["reachability"] == "exempt" for row in pages),
            "parsed_pages": len(parsed),
            "finding_count": len(findings),
            "advisory_count": len(advisories),
        },
        "bounds": {
            "max_page_bytes": MAX_PAGE_BYTES,
            "max_traversed_pages": MAX_TRAVERSED_PAGES,
            "max_links_per_page": MAX_LINKS_PER_PAGE,
            "network": False,
            "subprocess": False,
            "automatic_doc_edits": False,
        },
        "anchor_scope": (
            "Local heading anchors are settled only against headings this parser slugs: ASCII "
            "heading text, lowercased, characters outside [a-z0-9 _-] removed, spaces hyphenated, "
            "repeats suffixed -1/-2. Anything else is reported as an advisory, never a failure. "
            "This claims no equivalence with GitHub, a static-site generator, or any other renderer."
        ),
        "structure_boundary": (
            "Structure only: reachability through declared local links, local target existence, "
            "navigation-root uniqueness, and case-collision on the declared publishing surface. "
            "A reachable page is not evidence that its content is current, accurate, or usable, "
            "and this is not generated-artifact equality, claim auditing, review, CI, or merge evidence."
        ),
    }


def format_documentation_navigation(report: DocumentationNavigationReport) -> str:
    summary = report["summary"]
    lines = ["Documentation navigation: " + ("PASS" if report["ok"] else "NEEDS ATTENTION")]
    lines.append(
        f"  roots={len(report['roots'])} scope={summary['scope_pages']} "
        f"reachable={summary['reachable_pages']} exempt={summary['exempt_pages']} "
        f"parsed={summary['parsed_pages']}"
    )
    for row in report["findings"]:
        lines.append(f"{row['failure_class']} {row['page']} owner={row['owner']}")
        if row["referrer"] or row["target"]:
            lines.append(f"  referrer={row['referrer'] or '-'} target={row['target'] or '-'}")
        lines.append(f"  {row['detail']}")
    for row in report["advisories"]:
        lines.append(f"advisory {row['failure_class']} {row['page']} referrer={row['referrer'] or '-'}")
        lines.append(f"  {row['detail']}")
    lines.append(report["anchor_scope"])
    lines.append(report["structure_boundary"])
    return "\n".join(lines)
