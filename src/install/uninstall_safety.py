"""Fail-closed path guards shared by destructive uninstall surfaces."""
from __future__ import annotations

import os
from pathlib import Path

from ..local_store import is_directory_link


def unsafe_removal_reason(path: Path, *, hermes_home: Path | None = None) -> str | None:
    """Force never grants recursive ownership of a home or filesystem root."""
    if is_directory_link(path):
        return "refusing to remove a symlink or directory junction"
    try:
        target = path.expanduser().resolve()
        homes = [Path.home().resolve()]
        if hermes_home is not None:
            homes.append(hermes_home.resolve())
        configured_hermes = os.environ.get("HERMES_HOME")
        if configured_hermes:
            homes.append(Path(configured_hermes).expanduser().resolve())
        if target.parent == target or any(home == target or target in home.parents for home in homes):
            return "refusing to remove filesystem root, HOME, HERMES_HOME, or an ancestor of a home"
    except (OSError, RuntimeError, ValueError):
        return "removal target could not be safely resolved"
    return None


def linked_removal_reason(path: Path, boundary: Path) -> str | None:
    """Do not follow a directory link inside the installation boundary."""
    current = path.expanduser().absolute()
    boundary = boundary.expanduser().absolute()
    if not current.is_relative_to(boundary):
        return "removal target is outside its managed directory"
    while True:
        if is_directory_link(current):
            return "removal target traverses a symlink or directory junction"
        if current == boundary:
            return None
        current = current.parent


def configured_link_reason(path: Path, variable: str) -> str | None:
    """Path resolution must not erase a configured root's directory-link status."""
    raw = os.environ.get(variable)
    if raw:
        try:
            configured = Path(os.path.expandvars(raw)).expanduser()
            if configured.resolve() == path.resolve() and is_directory_link(configured):
                return f"refusing to remove symlink or directory junction configured by {variable}"
        except (OSError, RuntimeError, ValueError):
            return f"cannot safely resolve {variable} for removal"
    return None
