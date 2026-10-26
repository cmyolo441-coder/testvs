"""Import externally authored agent rules as OMH-managed imported skills.

Teams arrive with rule files other tools already own — Cursor's
``.cursorrules`` and ``.cursor/rules/*.mdc``, Cline's ``.clinerules``,
Windsurf's ``.windsurfrules``, Copilot's ``.github/copilot-instructions.md``.
Requiring a hand migration before Hermes can see any of them is a tax this
module removes: each discovered source converts deterministically into one
imported skill at ``<skills_dir>/imported/<slug>/SKILL.md``, which Hermes
already reads through the ``skills.external_dirs`` registration OMH setup
performed (the ``imported`` path segment becomes the Hermes category).

Boundaries:

- Discovery is explicit-root only: the caller names the repository root, and
  only the known relative locations under it are inspected. No walking, no
  symlinked sources, bounded source count and size.
- Conversion is a pure text transform with provenance (source path, sha256,
  format) recorded in the generated frontmatter and in an import manifest,
  so re-imports are idempotent and `omh update` — which prunes only
  dirs recorded in the managed-skill manifest — never touches imports.
- Trust checks mirror the prompt-compatibility surface: sources carrying
  prompt-injection phrasing or credential-value-shaped material are refused
  with reasons, never silently imported. Reads enforce repo containment:
  the resolved source must stay under the resolved repo root, and no path
  component below the root may be a symlink — so a symlinked rules
  directory cannot smuggle content from outside the repository. (Ancestor
  symlinks above the root, such as macOS's `/var -> private/var`, stay
  legal because they cannot change what is inside the root.)
- Importing registers content for Hermes to read. It is not evidence Hermes
  loaded the skill, followed it, or that the rule improved anything.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from ..install.manifest import IMPORTED_SKILLS_DIR_NAME
from ..system.local_store import atomic_write_json, atomic_write_text, ensure_dir, read_json_object_result, utc_now
from ..system.metadata_safety import is_secret_value_shaped

EXTERNAL_RULE_IMPORT_SCHEMA_VERSION: Final = "external_rule_import/v1"
IMPORT_MANIFEST_FILE: Final = "skills-import-manifest.json"
IMPORTED_CATEGORY_DIR: Final = IMPORTED_SKILLS_DIR_NAME
IMPORTED_NAME_PREFIX: Final = "omh-imported-"

MAX_IMPORT_SOURCES: Final = 40
MAX_SOURCE_BYTES: Final = 131_072
MAX_DESCRIPTION_CHARS: Final = 200
MAX_SLUG_CHARS: Final = 60

EXTERNAL_RULE_IMPORT_CLAIM_BOUNDARY: Final = (
    "An imported rule skill is registered content only. Import is not evidence that "
    "Hermes loaded the skill, followed it, or that the rule changed any outcome."
)

RuleFormat: TypeAlias = Literal[
    "cursorrules",
    "cursor-mdc",
    "clinerules",
    "windsurfrules",
    "copilot-instructions",
]

_PROMPT_INJECTION: Final = re.compile(
    r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions\b",
    re.IGNORECASE,
)
_SLUG_SANITIZE: Final = re.compile(r"[^a-z0-9]+")
_MDC_KEY: Final = re.compile(r"^(description|globs|alwaysApply)\s*:\s*(.*)$")


class ExternalRuleImportError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveredRuleSource:
    relative_path: str
    format_name: RuleFormat


@dataclass(frozen=True)
class RuleImportItem:
    slug: str
    source_relative_path: str
    format_name: RuleFormat
    source_sha256: str
    byte_count: int
    target_relative_path: str
    status: str  # planned | unchanged | refused
    refusal_reasons: tuple[str, ...]
    description: str
    body: str

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "source": self.source_relative_path,
            "format": self.format_name,
            "source_sha256": self.source_sha256,
            "byte_count": self.byte_count,
            "target": self.target_relative_path,
            "status": self.status,
            "refusal_reasons": list(self.refusal_reasons),
        }


def import_manifest_path(omh_home: Path) -> Path:
    return omh_home / IMPORT_MANIFEST_FILE


def discover_rule_sources(repo_root: Path) -> list[DiscoveredRuleSource]:
    """The known external-rule locations present under *repo_root*."""
    if not repo_root.is_dir():
        raise ExternalRuleImportError(f"repo root is not a directory: {repo_root}")
    sources: list[DiscoveredRuleSource] = []

    def _add(relative: str, format_name: RuleFormat) -> None:
        path = repo_root / relative
        if path.is_file() and not path.is_symlink():
            sources.append(DiscoveredRuleSource(relative, format_name))

    _add(".cursorrules", "cursorrules")
    cursor_rules_dir = repo_root / ".cursor" / "rules"
    if cursor_rules_dir.is_dir():
        for entry in sorted(cursor_rules_dir.glob("*.mdc")):
            if entry.is_file() and not entry.is_symlink():
                sources.append(
                    DiscoveredRuleSource(str(entry.relative_to(repo_root)), "cursor-mdc")
                )
    clinerules = repo_root / ".clinerules"
    if clinerules.is_file() and not clinerules.is_symlink():
        sources.append(DiscoveredRuleSource(".clinerules", "clinerules"))
    elif clinerules.is_dir():
        for entry in sorted(clinerules.glob("*.md")):
            if entry.is_file() and not entry.is_symlink():
                sources.append(
                    DiscoveredRuleSource(str(entry.relative_to(repo_root)), "clinerules")
                )
    _add(".windsurfrules", "windsurfrules")
    _add(".github/copilot-instructions.md", "copilot-instructions")

    if len(sources) > MAX_IMPORT_SOURCES:
        raise ExternalRuleImportError(
            f"{len(sources)} rule sources discovered; import is bounded to {MAX_IMPORT_SOURCES}"
        )
    for source in sources:
        if any(ord(char) < 32 or char == "\x7f" for char in source.relative_path):
            raise ExternalRuleImportError(
                "rule source path contains control characters; refusing the whole import: "
                f"{source.relative_path!r}"
            )
    return sources


def plan_rule_import(
    repo_root: Path,
    *,
    skills_dir: Path,
    omh_home: Path,
    only_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    """The full dry-run import plan; ``apply_rule_import`` executes it."""
    discovered = discover_rule_sources(repo_root)
    if only_sources:
        wanted = set(only_sources)
        discovered = [source for source in discovered if source.relative_path in wanted]
        missing = wanted - {source.relative_path for source in discovered}
        if missing:
            raise ExternalRuleImportError(
                f"--source paths not discovered under {repo_root}: {', '.join(sorted(missing))}"
            )
    manifest, manifest_error = read_json_object_result(import_manifest_path(omh_home))
    if manifest_error:
        raise ExternalRuleImportError(
            f"import manifest is unreadable ({manifest_error}); fix or delete it and re-run"
        )
    manifest = manifest or {}
    prior_entries = [entry for entry in manifest.get("entries", []) if isinstance(entry, dict)]
    recorded = {
        str(entry.get("source", "")): str(entry.get("source_sha256", ""))
        for entry in prior_entries
    }
    used_slugs: set[str] = set()
    items: list[RuleImportItem] = []
    for source in discovered:
        items.append(
            _plan_item(repo_root, source, skills_dir=skills_dir, recorded=recorded, used_slugs=used_slugs)
        )
    discovered_sources = {source.relative_path for source in discovered}
    orphaned = (
        []
        if only_sources
        else [entry for entry in prior_entries if str(entry.get("source", "")) not in discovered_sources]
    )
    return {
        "schema_version": EXTERNAL_RULE_IMPORT_SCHEMA_VERSION,
        "repo_root": str(repo_root),
        "skills_dir": str(skills_dir),
        "discovered_count": len(items),
        "planned": [item.to_public_dict() for item in items if item.status == "planned"],
        "unchanged": [item.to_public_dict() for item in items if item.status == "unchanged"],
        "refused": [item.to_public_dict() for item in items if item.status == "refused"],
        "orphaned": [
            {"slug": entry.get("slug", ""), "source": entry.get("source", ""), "target": entry.get("target", "")}
            for entry in orphaned
        ],
        "claim_boundary": EXTERNAL_RULE_IMPORT_CLAIM_BOUNDARY,
        "_items": items,  # stripped before printing; consumed by apply
        "_prior_entries": prior_entries,
        "_scoped": bool(only_sources),
    }


def apply_rule_import(plan: dict[str, Any], *, skills_dir: Path, omh_home: Path) -> dict[str, Any]:
    """Write planned items, retire orphans, and record provenance.

    A scoped run (--source) merges into the prior manifest so out-of-scope
    entries keep their provenance. A full run also retires orphans: an entry
    whose source no longer exists has its imported target removed, because a
    rule the user deleted must stop being served to Hermes — and `omh update`
    is barred from pruning imports, so this is the only cleanup path.
    A still-present source that is now refused keeps its prior entry (stale
    but accounted) and is reported under `refused`.
    """
    items = [item for item in plan.get("_items", []) if isinstance(item, RuleImportItem)]
    prior_entries = [entry for entry in plan.get("_prior_entries", []) if isinstance(entry, dict)]
    scoped = bool(plan.get("_scoped"))
    written: list[str] = []
    for item in items:
        if item.status != "planned":
            continue
        target = skills_dir / item.target_relative_path
        ensure_dir(target.parent)
        atomic_write_text(target, _render_skill_document(item))
        written.append(item.slug)
    removed: list[str] = []
    orphaned_sources = {str(row.get("source", "")) for row in plan.get("orphaned", [])}
    if not scoped:
        for entry in prior_entries:
            if str(entry.get("source", "")) not in orphaned_sources:
                continue
            target_rel = str(entry.get("target", ""))
            target_dir = (skills_dir / target_rel).parent if target_rel else None
            if (
                target_dir is not None
                and target_dir.is_dir()
                and not target_dir.is_symlink()
                and target_dir.parent == skills_dir / IMPORTED_CATEGORY_DIR
            ):
                shutil.rmtree(target_dir)
                removed.append(str(entry.get("slug", "")))
    handled_sources = {item.source_relative_path for item in items}
    kept_entries = [
        entry
        for entry in prior_entries
        if str(entry.get("source", "")) not in handled_sources
        and str(entry.get("source", "")) not in (orphaned_sources if not scoped else set())
    ]
    fresh_entries = [
        {
            "slug": item.slug,
            "source": item.source_relative_path,
            "source_sha256": item.source_sha256,
            "format": item.format_name,
            "target": item.target_relative_path,
            "imported_at": utc_now(),
        }
        for item in items
        if item.status in ("planned", "unchanged")
    ]
    # A now-refused source that was previously imported keeps its prior entry.
    refused_sources = {item.source_relative_path for item in items if item.status == "refused"}
    carried_refused = [
        entry for entry in prior_entries if str(entry.get("source", "")) in refused_sources
    ]
    manifest_entries = sorted(
        kept_entries + carried_refused + fresh_entries,
        key=lambda entry: str(entry.get("source", "")),
    )
    atomic_write_json(
        import_manifest_path(omh_home),
        {"schema_version": EXTERNAL_RULE_IMPORT_SCHEMA_VERSION, "entries": manifest_entries},
    )
    public = public_plan(plan)
    public["written"] = sorted(written)
    public["removed"] = sorted(removed)
    return public


def public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def _plan_item(
    repo_root: Path,
    source: DiscoveredRuleSource,
    *,
    skills_dir: Path,
    recorded: dict[str, str],
    used_slugs: set[str],
) -> RuleImportItem:
    refusals: list[str] = []
    text = ""
    try:
        raw = _read_contained_source(repo_root, source.relative_path)
    except ExternalRuleImportError as exc:
        refusals.append(str(exc))
        raw = b""
    sha256 = hashlib.sha256(raw).hexdigest()
    slug = _slug_for(source.relative_path, sha256, used_slugs)
    target_rel = f"{IMPORTED_CATEGORY_DIR}/{slug}/SKILL.md"
    if raw:
        text = raw.decode("utf-8", errors="replace")
        if _PROMPT_INJECTION.search(text):
            refusals.append("prompt-injection phrasing requires human review before import")
        for line_number, line in enumerate(text.splitlines(), start=1):
            # The credential-VALUE predicate, not the marker-word one: a rule
            # document legitimately says "never commit secrets", and refusing
            # it for the word would make most real inputs unimportable.
            if is_secret_value_shaped(line):
                refusals.append(
                    f"line {line_number} looks like an issued credential value; refused"
                )
                break
    description, body = _describe(source.format_name, source.relative_path, text)
    if refusals:
        status = "refused"
    elif recorded.get(source.relative_path) == sha256 and (skills_dir / target_rel).is_file():
        status = "unchanged"
    else:
        status = "planned"
    return RuleImportItem(
        slug=slug,
        source_relative_path=source.relative_path,
        format_name=source.format_name,
        source_sha256=sha256,
        byte_count=len(raw),
        target_relative_path=target_rel,
        status=status,
        refusal_reasons=tuple(refusals),
        description=description,
        body=body,
    )


def _read_contained_source(repo_root: Path, relative_path: str) -> bytes:
    """Read a source, refusing anything that leaves the repository root.

    Containment, not a descriptor walk: the resolved source must stay under
    the resolved root, and no component *below* the root may be a symlink.
    Ancestor symlinks above the root (macOS `/var -> private/var`) are legal
    because they cannot change what is inside the root. The stat/read pair is
    not atomic; the threat model is checked-out repository content, not a
    live attacker racing the import on the operator's own machine.
    """
    root = repo_root.resolve()
    current = root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ExternalRuleImportError(
                f"rule source path component is a symlink; refusing: {relative_path}"
            )
    resolved = current.resolve()
    if resolved != root and root not in resolved.parents:
        raise ExternalRuleImportError(
            f"rule source escapes the repository root; refusing: {relative_path}"
        )
    if not resolved.is_file():
        raise ExternalRuleImportError(f"rule source is not a regular file: {relative_path}")
    if resolved.stat().st_size > MAX_SOURCE_BYTES:
        raise ExternalRuleImportError(f"rule source exceeds {MAX_SOURCE_BYTES} bytes: {relative_path}")
    with open(resolved, "rb") as handle:
        return handle.read(MAX_SOURCE_BYTES + 1)


def _slug_for(relative_path: str, sha256: str, used_slugs: set[str]) -> str:
    stem = Path(relative_path).stem or Path(relative_path).name
    base = _SLUG_SANITIZE.sub("-", stem.lower()).strip("-") or "rule"
    slug = base[:MAX_SLUG_CHARS]
    suffix_width = 8
    while slug in used_slugs:
        slug = f"{base[: MAX_SLUG_CHARS - suffix_width - 1]}-{sha256[:suffix_width]}"
        suffix_width += 4
        if suffix_width > len(sha256):
            slug = f"{slug}-x"
    used_slugs.add(slug)
    return slug


def _describe(format_name: RuleFormat, relative_path: str, text: str) -> tuple[str, str]:
    body = text
    description = ""
    if format_name == "cursor-mdc":
        frontmatter, remainder = _split_frontmatter(text)
        for line in frontmatter.splitlines():
            match = _MDC_KEY.match(line.strip())
            if match and match.group(1) == "description":
                description = match.group(2).strip().strip("\"'")
        body = remainder if remainder.strip() else text
    if not description:
        for line in body.splitlines():
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                description = stripped
                break
    label = {
        "cursorrules": "Cursor rules",
        "cursor-mdc": "Cursor MDC rule",
        "clinerules": "Cline rules",
        "windsurfrules": "Windsurf rules",
        "copilot-instructions": "Copilot instructions",
    }[format_name]
    description = f"[imported {label}] {description}"[:MAX_DESCRIPTION_CHARS].rstrip()
    fallback = f"[imported {label}] {relative_path}"[:MAX_DESCRIPTION_CHARS].rstrip()
    return (description or fallback, body)


_FRONTMATTER_LINE: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\s*:")


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split a leading MDC frontmatter block, or return the text untouched.

    A leading `---` block counts as frontmatter only when every nonempty line
    inside it looks like `key: value`; a document that merely opens with a
    markdown horizontal rule keeps its full body, so the "imported verbatim"
    statement in the rendered skill stays true.
    """
    if not text.startswith("---"):
        return ("", text)
    parts = text.split("\n---", 2)
    if len(parts) < 2:
        return ("", text)
    frontmatter = parts[0].removeprefix("---")
    inner_lines = [line for line in frontmatter.splitlines() if line.strip()]
    if not inner_lines or not all(_FRONTMATTER_LINE.match(line.strip()) for line in inner_lines):
        return ("", text)
    remainder = parts[1]
    if remainder.startswith(("\n", "\r")):
        remainder = remainder[1:]
    return (frontmatter, remainder)


def _render_skill_document(item: RuleImportItem) -> str:
    # Every scalar that carries source-derived text is emitted as a JSON
    # string, which is also a valid YAML double-quoted scalar: a filename or
    # description containing newlines or `name:` lines cannot forge
    # frontmatter keys. The slug is sanitized to [a-z0-9-] and needs no quote.
    quoted_description = json.dumps(item.description, ensure_ascii=False)
    quoted_source = json.dumps(item.source_relative_path, ensure_ascii=False)
    return (
        "---\n"
        f"name: {IMPORTED_NAME_PREFIX}{item.slug}\n"
        f"description: {quoted_description}\n"
        "metadata:\n"
        "  omh:\n"
        "    imported:\n"
        f"      format: {item.format_name}\n"
        f"      source: {quoted_source}\n"
        f"      source_sha256: {item.source_sha256}\n"
        "---\n\n"
        f"# Imported rule: {item.slug}\n\n"
        f"{EXTERNAL_RULE_IMPORT_CLAIM_BOUNDARY}\n\n"
        "The content below was imported verbatim from "
        f"`{item.source_relative_path}` and is maintained there; re-run "
        "`omh ops rules-import --apply` after the source changes.\n\n"
        "---\n\n"
        f"{item.body.rstrip()}\n"
    )
