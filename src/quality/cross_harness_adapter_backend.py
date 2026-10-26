from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Final

from .cross_harness_adapter_io import read_regular_file


_TRUSTED_BWRAP_CANDIDATES: Final = tuple(Path(path) for path in ("/usr/bin/bwrap", "/usr/local/bin/bwrap"))


@dataclass(frozen=True, slots=True)
class ExecutableSnapshot:
    path: Path
    digest: str
    device: int
    inode: int
    size: int
    modified_ns: int


def _trusted_modes(path: Path) -> bool:
    for entry in (path, *path.parents):
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError:
            return False
        if metadata.st_uid != 0 or metadata.st_mode & 0o022:
            return False
    return True


def _snapshot(path: Path) -> ExecutableSnapshot | None:
    if not path.is_absolute() or not _trusted_modes(path):
        return None
    try:
        content, opened = read_regular_file(path)
        current = path.stat(follow_symlinks=False)
    except OSError:
        return None
    identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
    if identity != (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns):
        return None
    if not stat.S_ISREG(opened.st_mode) or not os.access(path, os.X_OK):
        return None
    return ExecutableSnapshot(path, hashlib.sha256(content).hexdigest(), *identity)


def trusted_bwrap(expected_digest: str | None = None) -> ExecutableSnapshot | None:
    for candidate in _TRUSTED_BWRAP_CANDIDATES:
        snapshot = _snapshot(candidate)
        if snapshot is not None and (expected_digest is None or snapshot.digest == expected_digest):
            return snapshot
    return None


def command_is_trusted(command: str, expected_digest: str | None) -> bool:
    path = Path(command)
    if path not in _TRUSTED_BWRAP_CANDIDATES:
        return True
    snapshot = trusted_bwrap(expected_digest)
    return snapshot is not None and snapshot.path == path
