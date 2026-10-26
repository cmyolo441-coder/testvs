from __future__ import annotations

import re
from pathlib import Path

from .skill_pack import DESCRIPTIONS, SkillReferenceTemplate, SkillTemplate
from .skills.catalog import (
    OMH_SKILL_DISPLAY_NAME_OVERRIDES,
    OMH_SKILL_NAME_PREFIX,
    ULW_SKILL_NAME_PREFIX,
    omh_description,
    omh_skill_display_name,
)

FRONTMATTER_RE = re.compile(r"^---\n(?P<meta>.*?)\n---\n(?P<body>.*)$", re.DOTALL)


_CANONICAL_BY_OVERRIDE = {
    display: canonical for canonical, display in OMH_SKILL_DISPLAY_NAME_OVERRIDES.items()
}


def extract_name(raw: str, fallback: str) -> str:
    """Return the canonical install identity for a source SKILL.md.

    The frontmatter `name` is a rendered display label and may already carry a
    display prefix (re-importing OMH's own generated taps through
    `omh setup --source <dir>` hits exactly that). Strip it here so the install
    directory, the manifest key, and the curated `DESCRIPTIONS` lookup all key on
    the canonical name.

    Both prefixes have to be stripped: workflow-engine skills render `ulw-`, so
    stripping only `omh-` would round-trip `ultrawork` back as `ulw-ultrawork`
    and install it into the wrong directory under a name nothing routes to.
    """
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return fallback
    for line in match.group("meta").splitlines():
        if line.startswith("name:"):
            rendered = line.split(":", 1)[1].strip().strip("'\"")
            # A hand-picked label is not `prefix + canonical`, so stripping the
            # prefix invents a name that owns nothing: `omh-routing` became
            # `routing`, `ulw-work` became `work`. Reverse the override map first
            # and only fall back to stripping for the mechanical labels.
            canonical = _CANONICAL_BY_OVERRIDE.get(rendered)
            if canonical is not None:
                return canonical
            for prefix in (OMH_SKILL_NAME_PREFIX, ULW_SKILL_NAME_PREFIX):
                if rendered.startswith(prefix):
                    return rendered.removeprefix(prefix) or fallback
            return rendered or fallback
    return fallback


def convert_skill(raw: str, fallback_name: str) -> SkillTemplate:
    name = extract_name(raw, fallback_name)
    description = DESCRIPTIONS.get(name, omh_description(f"Hermes workflow skill for {name}."))
    # Imported skills are wrapper-installed too, so they render the same display
    # prefix as the generated catalog. `SkillTemplate.name` stays canonical and
    # keeps owning the `skills/<name>/` directory.
    content = _replace_frontmatter_description(
        raw, name=omh_skill_display_name(name), description=description
    ).rstrip() + """

## Hermes Compatibility Contract

This skill was imported by `omh` from a local skill source.

- Keep the upstream workflow intent, but adapt runtime behavior to Hermes Agent.
- Do not require runtime features that Hermes Agent does not expose.
- Use Hermes `skills_list`, `skill_view`, file tools, terminal tools, and Hermes delegation when available.
"""
    return SkillTemplate(name=name, content=content + "\n")


def _replace_frontmatter_description(raw: str, *, name: str, description: str) -> str:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return f"---\nname: {name}\ndescription: {description}\n---\n\n{raw}"
    lines = match.group("meta").splitlines()
    output: list[str] = []
    saw_name = False
    saw_description = False
    for line in lines:
        if line.startswith("name:"):
            output.append(f"name: {name}")
            saw_name = True
        elif line.startswith("description:"):
            output.append(f"description: {description}")
            saw_description = True
        else:
            output.append(line)
    if not saw_name:
        output.insert(0, f"name: {name}")
    if not saw_description:
        output.insert(1, f"description: {description}")
    return "---\n" + "\n".join(output) + "\n---\n" + match.group("body")


def discover_skill_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source does not exist: {source_dir}")
    # Agent artifacts, the separate Agent Skills projection and OMH's own
    # project state root (`.omh/skills` is the managed install OUTPUT, another
    # layout of these same skills) are not Hermes source inputs. Check relative
    # components so an explicitly selected pack root still works, without
    # mixing sibling projections during repo import.
    from .install.agent_skills_projection import MANIFEST_NAME, _read_manifest

    excluded = {".omc", ".omh", "agent-skills", ".agents"}
    claude_mirror = source_dir / ".claude/skills"
    manifest = _read_manifest(claude_mirror.absolute())
    # A shared host directory is not wholly ours. Use validated manifest paths,
    # never a directory-prefix exclusion that hides neighboring custom skills.
    owned_paths = {claude_mirror / relative for relative in (manifest or {}).get("files", {})}
    owned_paths.add(claude_mirror / MANIFEST_NAME)
    nested: dict[Path, bool] = {}
    return sorted(
        path for path in source_dir.rglob("SKILL.md")
        if ".git" not in path.parts and not excluded.intersection(path.relative_to(source_dir).parts)
        and path not in owned_paths
        and not _inside_nested_repository(path, source_dir, nested, exempt=claude_mirror)
    )


def _inside_nested_repository(path: Path, source_dir: Path, memo: dict[Path, bool], *, exempt: Path) -> bool:
    """True when another checkout sits between ``source_dir`` and ``path``.

    A linked worktree, a submodule, or a clone left under the source root is a
    different revision of the same skills; importing it beside the source's own
    copy made two templates claim one install path and the second write refuse
    as a local modification. A linked worktree marks its root with a ``.git``
    FILE, so the ``".git" in parts`` test above never sees it. The one
    repository root that is NOT another revision of this source is the host's
    own ``.claude/skills`` mirror: people keep their custom skills there under
    version control, and those are a deliberate import source.
    """
    for parent in path.parents:
        if parent == source_dir:
            return False
        if parent not in memo:
            memo[parent] = parent != exempt and (parent / ".git").exists()
        if memo[parent]:
            return True
    return False


def convert_from_dir(source_dir: Path) -> list[SkillTemplate]:
    templates: list[SkillTemplate] = []
    for skill_file in discover_skill_files(source_dir):
        raw = skill_file.read_text(encoding="utf-8")
        templates.append(convert_skill(raw, skill_file.parent.name))
    return templates


def convert_references_from_dir(source_dir: Path) -> list[SkillReferenceTemplate]:
    """Copy progressive references beside each imported skill into the managed pack."""
    templates: list[SkillReferenceTemplate] = []
    for skill_file in discover_skill_files(source_dir):
        raw = skill_file.read_text(encoding="utf-8")
        skill_name = extract_name(raw, skill_file.parent.name)
        references_dir = skill_file.parent / "references"
        if not references_dir.is_dir():
            continue
        for reference_file in sorted(references_dir.rglob("*.md")):
            if ".git" in reference_file.parts:
                continue
            templates.append(
                SkillReferenceTemplate(
                    skill_name=skill_name,
                    relative_path=reference_file.relative_to(skill_file.parent).as_posix(),
                    content=reference_file.read_text(encoding="utf-8"),
                )
            )
    return templates


def convert_support_from_dir(source_dir: Path):
    """Import only the fixed reviewed companion, as inert UTF-8 data.

    Source manifests never grant execution authority or widen the inventory.
    Custom skills without this companion remain a no-op.
    """
    from .skills.support_files import builtin_skill_support_templates, checked_support_path, validate_support_templates

    expected = builtin_skill_support_templates()
    owners = {t.skill_name for t in expected}
    templates = []
    for skill_file in discover_skill_files(source_dir):
        skill_name = extract_name(skill_file.read_text(encoding="utf-8"), skill_file.parent.name)
        if skill_name not in owners:
            continue
        if (skill_file.parent / "toolkit").is_symlink():
            raise ValueError("local-source support refuses symlink toolkit")
        if not (skill_file.parent / "toolkit").exists():
            raise ValueError(f"local-source support toolkit missing for {skill_name}")
        owned = {t.relative_path for t in expected if t.skill_name == skill_name}
        actual = {path.relative_to(skill_file.parent).as_posix()
                  for path in (skill_file.parent / "toolkit").rglob("*")
                  if path.is_file() or path.is_symlink()}
        if actual != owned:
            raise ValueError(f"local-source support inventory mismatch for {skill_name}")
        for template in expected:
            if template.skill_name != skill_name:
                continue
            path = checked_support_path(skill_file.parent, template.relative_path)
            if not path.is_file() or path.read_bytes() != template.content.encode("utf-8"):
                raise ValueError(f"local-source support missing/unreviewed bytes: {template.relative_path}")
            templates.append(template)
    return validate_support_templates(templates)
