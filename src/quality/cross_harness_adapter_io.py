from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys

from .cross_harness_benchmark_values import JsonValue


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    path: str
    sha256: str


class UnsafeRegularFileError(OSError):
    pass


_DIRECTORY_ONLY = getattr(os, "O_DIRECTORY", None)
_NO_FOLLOW = getattr(os, "O_NOFOLLOW", None)
_CLOSE_ON_EXEC = getattr(os, "O_CLOEXEC", None)
_DIRECTORY_FLAGS = os.O_RDONLY | (_CLOSE_ON_EXEC or 0) | (_DIRECTORY_ONLY or 0) | (_NO_FOLLOW or 0)


def _normalize_platform_alias(path: Path) -> Path:
    absolute = path.absolute()
    if sys.platform != "darwin" or absolute.parts[:2] != ("/", "var"):
        return absolute
    try:
        alias = Path("/var").lstat()
        private = Path("/private").stat(follow_symlinks=False)
        target = Path("/private/var").stat(follow_symlinks=False)
        link_target = os.readlink("/var")
    except OSError as error:
        raise UnsafeRegularFileError from error
    trusted = (
        stat.S_ISLNK(alias.st_mode) and alias.st_uid == 0 and not alias.st_mode & 0o022
        and link_target == "private/var"
        and all(stat.S_ISDIR(item.st_mode) and item.st_uid == 0 and not item.st_mode & 0o022 for item in (private, target))
    )
    if not trusted:
        raise UnsafeRegularFileError("untrusted platform alias")
    return Path("/private/var", *absolute.parts[2:])


def _open_directory(path: Path, *, create: bool) -> int:
    if _DIRECTORY_ONLY is None or _NO_FOLLOW is None or _CLOSE_ON_EXEC is None:
        raise UnsafeRegularFileError("safe directory traversal is unsupported")
    directory = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in _normalize_platform_alias(path).parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, mode=0o700, dir_fd=directory)
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory)
            except OSError as error:
                raise UnsafeRegularFileError from error
            os.close(directory)
            directory = child
        return directory
    except BaseException:
        os.close(directory)
        raise


def read_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    directory = _open_directory(path.parent, create=False)
    try:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NONBLOCK | (_CLOSE_ON_EXEC or 0) | (_NO_FOLLOW or 0),
                dir_fd=directory,
            )
        except FileNotFoundError:
            raise
        except OSError as error:
            raise UnsafeRegularFileError from error
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsafeRegularFileError
        return stream.read(), metadata


def inventory(root: Path, excluded: set[str]) -> tuple[InventoryEntry, ...]:
    directory = _open_directory(root, create=False)
    os.close(directory)
    entries: list[InventoryEntry] = []
    for directory_text, names, files in os.walk(root, followlinks=False):
        names.sort()
        files.sort()
        base = Path(directory_text)
        if any((base / name).is_symlink() for name in names):
            raise UnsafeRegularFileError
        for name in files:
            path = base / name
            relative = path.relative_to(root).as_posix()
            if relative in excluded:
                continue
            if path.is_symlink():
                raise UnsafeRegularFileError
            entries.append(InventoryEntry(relative, hashlib.sha256(read_regular_file(path)[0]).hexdigest()))
    return tuple(entries)


def unlink_file(path: Path) -> None:
    try:
        directory = _open_directory(path.parent, create=False)
    except FileNotFoundError:
        return
    try:
        try:
            os.unlink(path.name, dir_fd=directory)
        except FileNotFoundError:
            pass
    finally:
        os.close(directory)


def write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    directory = _open_directory(path.parent, create=True)
    temporary = f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | (_CLOSE_ON_EXEC or 0), 0o600, dir_fd=directory)
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)
