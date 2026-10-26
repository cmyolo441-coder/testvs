from __future__ import annotations

from ..skill_pack import builtin_skill_templates
from ..skills.catalog import omh_skill_display_name, omh_skill_install_path

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from ..version import __version__
from ..hashutil import sha256_file
from ..local_store import atomic_write_json, read_json_object, utc_now
from .guidance_projection import catalog_revision


@dataclass(frozen=True)
class SkillRecord:
    name: str
    path: str
    sha256: str
    source: str


def new_manifest(source: str, skills_dir: Path, records: list[SkillRecord]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "package": "oh-my-hermes",
        "version": __version__,
        "source": source,
        "installed_at": utc_now(),
        "skills_dir": str(skills_dir),
        # What the projection was rendered from, not just which release wrote
        # it. A version string cannot distinguish two builds of the same
        # release with different catalog data, and per-file checksums detect
        # drift without naming what drifted. See `install.guidance_projection`.
        "catalog_revision": catalog_revision(),
        "skills": [record.__dict__ for record in sorted(records, key=lambda item: item.name)],
    }


def read_manifest(path: Path) -> dict[str, Any] | None:
    return read_json_object(path)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)


# The category directory `omh ops rules-import` writes under the skills dir.
# The managed manifest must not record it: recording would mark every imported
# skill as a managed artifact, and the next update's orphan pruning (imported
# names are never in the catalog) would delete what the user just imported.
# Provenance for these lives in the separate skills-import manifest instead.
IMPORTED_SKILLS_DIR_NAME = "imported"


def skill_records(skills_dir: Path, source: str) -> list[SkillRecord]:
    records: list[SkillRecord] = []
    if not skills_dir.exists():
        return records
    # `name` stays canonical and `path` carries the real directory. The two
    # diverged when installs moved to display-labelled directories, and callers
    # compare `name` against the catalog (CORE_PROFILE_SKILLS, capability ids),
    # so recording the label there made every lookup miss.
    canonical_by_directory = {
        omh_skill_display_name(template.name): template.name for template in builtin_skill_templates()
    }
    # A directory that is spelled in another letter case than the one an
    # install renders, yet IS that directory (a case-insensitive filesystem,
    # after someone renamed it), is recorded as the skill it holds, at the
    # path the install renders. Recorded by its on-disk spelling it became a
    # skill the catalog does not ship, and the orphan pruner deleted the
    # directory the same install had just rendered into.
    render_by_folded_path = {
        omh_skill_install_path(template.name).casefold(): (template.name, omh_skill_install_path(template.name))
        for template in builtin_skill_templates()
    }
    for skill_file in sorted(skills_dir.rglob("SKILL.md")):
        rel = skill_file.relative_to(skills_dir)
        if rel.parts and rel.parts[0] == IMPORTED_SKILLS_DIR_NAME:
            continue
        directory = skill_file.parent.name
        name, path = canonical_by_directory.get(directory, directory), rel.as_posix()
        rendered = render_by_folded_path.get(rel.parent.as_posix().casefold())
        if (
            rendered is not None
            and rel.parent.as_posix() != rendered[1]
            and same_directory(skills_dir / rendered[1], skill_file.parent)
        ):
            name, path = rendered[0], f"{rendered[1]}/SKILL.md"
        records.append(SkillRecord(name=name, path=path, sha256=sha256_file(skill_file), source=source))
    return records


def same_directory(first: Path, second: Path) -> bool:
    """Whether two paths are one directory on disk, however each is spelled; False when either cannot be read."""
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def local_modifications(
    manifest: dict[str, Any] | None,
    skills_dir: Path,
    *,
    incoming: dict[str, str] | None = None,
) -> list[str]:
    """Managed files whose bytes differ from what the manifest recorded.

    `incoming` maps a SKILL.md record path to the sha256 of the bytes the
    install about to run writes there. A file already holding exactly those
    bytes loses nothing to the install, so it is not a modification even when
    an older release recorded another hash (#2300). Support files keep the
    manifest-only rule.
    """
    if not manifest:
        return []
    incoming = incoming or {}
    modified: list[str] = []
    for record in manifest.get("skills", []):
        rel = record.get("path")
        expected = record.get("sha256")
        if not rel or not expected:
            continue
        path = skills_dir / rel
        if not path.exists():
            modified.append(str(rel))
            continue
        observed = sha256_file(path)
        if observed != expected and observed != incoming.get(str(rel)):
            modified.append(str(rel))
    from .support_materialization import support_records, support_path

    for relative, expected in support_records(manifest).items():
        path = support_path(skills_dir, relative)
        # Missing support is safe repair, not user-authored modification.
        if path.is_file() and sha256_file(path) != expected:
            modified.append(relative)
    return modified
