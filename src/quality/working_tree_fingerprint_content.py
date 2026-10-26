"""Byte-streaming overlay construction for working-tree fingerprints."""

from __future__ import annotations

import hashlib
import errno
import os
import stat
from pathlib import Path

from . import working_tree_fingerprint_windows as windows


class ContentRace(RuntimeError):
    """A path changed while its final bytes were being observed."""


def overlay(
    root: Path,
    paths: list[bytes],
    head_entries: dict[bytes, tuple[bytes, bytes]],
) -> list[tuple[bytes, bytes, bytes]] | None:
    """Stream final changed paths and return only their tree-different overlay."""
    entries: list[tuple[bytes, bytes, bytes]] = []
    root_bytes = os.fsencode(root)
    for path in paths:
        candidate = os.path.join(root_bytes, path)
        try:
            before = _path_stat(candidate)
        except FileNotFoundError:
            if path in head_entries:
                entries.append((path, b"deleted", b""))
            continue
        if stat.S_ISDIR(before.st_mode):
            return None
        current = _entry_digest(candidate, before)
        if current is None:
            return None
        try:
            after = _path_stat(candidate)
        except FileNotFoundError:
            raise ContentRace
        if not _same_stat(before, after):
            raise ContentRace
        if head_entries.get(path) == current[:2]:
            continue
        entries.append((path, *current))
    return entries


def _path_stat(path: bytes) -> os.stat_result | windows.Metadata:
    metadata = os.lstat(path)
    if os.name == "nt" and stat.S_ISREG(metadata.st_mode):
        return windows.stat(path)
    return metadata


def _entry_digest(path: bytes, metadata: os.stat_result | windows.Metadata) -> tuple[bytes, bytes] | None:
    mode = metadata.st_mode
    if stat.S_ISREG(mode):
        git_mode = b"100755" if mode & stat.S_IXUSR else b"100644"
        digest = hashlib.sha1()
        digest.update(f"blob {metadata.st_size}\0".encode())
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = (
                windows.open(path) if os.name == "nt"
                else os.open(path, os.O_RDONLY | no_follow | getattr(os, "O_BINARY", 0))
            )
            observe_fd = windows.fstat if os.name == "nt" else os.fstat
            opened = observe_fd(descriptor)
            # Compare observations from the SAME API. CPython Windows path
            # stat and fstat disagree on identity, ctime and extension modes.
            if (
                not stat.S_ISREG(opened.st_mode)
                or not _same_stat(metadata, opened)
                or not _same_stat(metadata, _path_stat(path))
            ):
                raise ContentRace
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
            if not _same_stat(opened, observe_fd(descriptor)):
                raise ContentRace
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOENT}:
                raise ContentRace from error
            return None
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return git_mode, digest.hexdigest().encode()
    if stat.S_ISLNK(mode):
        try:
            target = os.readlink(path)
        except FileNotFoundError as error:
            raise ContentRace from error
        except OSError:
            return None
        raw_target = target.encode() if isinstance(target, str) else target
        return b"120000", hashlib.sha1(b"blob " + str(len(raw_target)).encode() + b"\0" + raw_target).hexdigest().encode()
    return None


def _same_stat(
    before: os.stat_result | windows.Metadata, after: os.stat_result | windows.Metadata,
) -> bool:
    if isinstance(before, windows.Metadata) or isinstance(after, windows.Metadata):
        return before == after
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
