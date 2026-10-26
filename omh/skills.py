"""Markdown skills: small, user-editable prompt packs.

A skill is a ``*.md`` file with optional frontmatter::

    ---
    name: review
    description: Review a diff like a senior engineer
    ---
    You are reviewing ... (instructions)

They are loaded from ``<workspace>/skills`` and ``~/.omh/skills``. Unlike the
upstream project's generated catalogue, this is four files you can read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .config import omh_home


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: str


def _parse(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    name = path.stem
    description = ""
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            header = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            for line in header.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key, value = key.strip().lower(), value.strip().strip("\"'")
                if key == "name" and value:
                    name = value
                elif key == "description":
                    description = value
    return Skill(name=name, description=description, body=body.strip(), path=str(path))


def skill_dirs(workspace: str | Path) -> List[Path]:
    return [Path(workspace) / "skills", omh_home() / "skills"]


def load_skills(workspace: str | Path) -> Dict[str, Skill]:
    found: Dict[str, Skill] = {}
    for directory in skill_dirs(workspace):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.md")):
            try:
                skill = _parse(path)
            except OSError:
                continue
            found[skill.name] = skill
    return found


def skill_prompt(skill: Skill) -> str:
    return (
        f"Active skill: {skill.name}\n"
        f"{skill.description}\n\n"
        f"{skill.body}"
    )
