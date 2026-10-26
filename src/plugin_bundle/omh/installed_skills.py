"""Which OMH skills this home actually installed, as the per-turn hints need it.

The route hint and the skill-candidate line rank the whole catalog, and a
`--core` install holds ten of its skills. Unfiltered, both told the model to
load skills that are not on disk (#1954), and `skill_view` on a missing name
is an error the model has to recover from mid-answer.

The install manifest (`<omh_home>/manifest.json`) is the record read here. The
installer writes it after every install, update and reconcile from the
`SKILL.md` files it finds on disk, so it names full-only skills a core profile
kept as well as the core ones. Each record carries the canonical name and the
installed path, whose leaf directory is the label Hermes lists; both spellings
are kept, because the route hint emits canonical names and ULW labels while
the candidate line emits labels.

`pre_llm_call` runs this every turn, so the parse is cached on the file's
mtime and size: a turn costs one `stat`, and an install that rewrites the
manifest is seen on the next turn without a Hermes restart.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from .skill_shortlist import catalog_skill_names

MANIFEST_NAME = "manifest.json"

_cache: tuple[tuple[str, int, int], frozenset[str] | None] | None = None


def installed_skill_names(omh_home: str | Path) -> frozenset[str] | None:
    """Canonical names and labels of the skills this home's manifest records.

    `None` means the installed set is unknown -- no manifest, an unreadable
    one, or one that records no skill -- and every caller then fails OPEN, to
    the full catalog it named before this filter existed. Failing closed would
    silence the hint on every install whose manifest OMH cannot read: a
    checkout run without `omh setup`, a plugin bound to a home other than the
    one that installed, or a manifest mid-write. Naming a skill that may be
    missing is the older, recoverable cost; withholding every skill is not.
    """
    global _cache
    path = Path(omh_home) / MANIFEST_NAME
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _cache
    if cached is not None and cached[0] == key:
        return cached[1]
    names = _read_manifest_names(path)
    _cache = (key, names)
    return names


def _read_manifest_names(path: Path) -> frozenset[str] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    records = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return None
    names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if isinstance(name, str) and name:
            names.add(name)
        relative = record.get("path")
        if isinstance(relative, str) and relative:
            label = PurePosixPath(relative).parent.name
            if label:
                names.add(label)
    return frozenset(names) or None


def skill_not_installed(name: str, installed: frozenset[str] | None) -> bool:
    """True only for a catalog skill the known installed set does not hold.

    A name outside the catalog is left alone: the hint also names lanes and
    retired engines that no install ever holds, and a full install must emit
    exactly what it emitted before. An unknown installed set or an unreadable
    catalog sidecar holds nothing back, for the reason `installed_skill_names`
    gives.
    """
    if installed is None or name in installed:
        return False
    catalog = catalog_skill_names()
    return catalog is not None and name in catalog


def reset_installed_skill_cache() -> None:
    global _cache
    _cache = None


__all__ = [
    "MANIFEST_NAME",
    "installed_skill_names",
    "reset_installed_skill_cache",
    "skill_not_installed",
]
