from __future__ import annotations

import errno
import os
from contextlib import contextmanager
from pathlib import Path
import stat
import time
from typing import Iterator

from ..system.local_store import FileLockTimeout
from .domain_intelligence_store_writer import (
    _managed_home_from_directory,
    _portable_directory_chain,
    _validate_portable_directory_chain,
)

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)
_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)


def open_domain_directory_at(
    domain_root_fd: int | Path,
    *relative_parts: str,
) -> int | Path:
    """Open existing descendants without leaving a bound domain-root descriptor."""
    if isinstance(domain_root_fd, Path):
        directory = domain_root_fd
        for part in relative_parts:
            if Path(part).name != part or part in {"", ".", ".."}:
                raise ValueError("domain-intelligence descriptor path is unsafe")
            directory /= part
        home = _managed_home_from_directory(directory)
        chain = _portable_directory_chain(directory, home=home, create=False)
        _validate_portable_directory_chain(chain)
        return directory
    if not _NOFOLLOW_FLAG or not _DIRECTORY_FLAG:
        raise ValueError("domain-intelligence safe reads are unavailable")
    flags = os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG | _NOFOLLOW_FLAG
    directory_fd = os.dup(domain_root_fd)
    os.set_inheritable(directory_fd, False)
    try:
        for part in relative_parts:
            if Path(part).name != part or part in {"", ".", ".."}:
                raise ValueError("domain-intelligence descriptor path is unsafe")
            next_directory_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_directory_fd
        return directory_fd
    except (OSError, ValueError) as exc:
        os.close(directory_fd)
        if isinstance(exc, OSError) and exc.errno in {
            errno.ELOOP,
            errno.EMLINK,
            errno.ENOTDIR,
        }:
            raise ValueError(
                "domain-intelligence descriptor path contains a symlink or non-directory"
            ) from exc
        raise


@contextmanager
def shared_domain_store_lock_at(
    domain_root_fd: int | Path,
    *,
    timeout_seconds: float = 0.25,
    poll_interval: float = 0.01,
) -> Iterator[dict[str, object]]:
    """Acquire the existing store lock through an already-bound root descriptor."""
    if isinstance(domain_root_fd, Path):
        with _shared_domain_store_lock_path(
            domain_root_fd,
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
        ) as state:
            yield state
        return
    if fcntl is None or not _NOFOLLOW_FLAG or not _NONBLOCK_FLAG:
        raise ValueError("shared_lock_unavailable")
    flags = os.O_RDONLY | _CLOEXEC_FLAG | _NOFOLLOW_FLAG | _NONBLOCK_FLAG
    try:
        descriptor = os.open(".store.lock", flags, dir_fd=domain_root_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(
                "domain-intelligence lock path must not be a symlink"
            ) from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("domain-intelligence lock path must be a regular file")
        deadline = time.monotonic() + min(max(timeout_seconds, 0.0), 0.25)
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(
                        "could not acquire bound domain store lock "
                        f"within {timeout_seconds}s"
                    ) from exc
                time.sleep(poll_interval)
        try:
            yield {"locked": True, "mode": "shared"}
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def _shared_domain_store_lock_path(
    domain_root: Path,
    *,
    timeout_seconds: float,
    poll_interval: float,
) -> Iterator[dict[str, object]]:
    chain = _portable_directory_chain(
        domain_root,
        home=_managed_home_from_directory(domain_root),
        create=False,
    )
    lock_path = domain_root / ".store.lock"
    before = os.stat(lock_path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("domain-intelligence lock path must be a regular file")
    descriptor = os.open(lock_path, os.O_RDWR | _CLOEXEC_FLAG)
    mechanism = ""
    try:
        opened = os.fstat(descriptor)
        after = os.stat(lock_path, follow_symlinks=False)
        if (
            _stable_identity(before) != _stable_identity(opened)
            or _stable_identity(opened) != _stable_identity(after)
        ):
            raise ValueError("domain-intelligence lock path changed while opening")
        deadline = time.monotonic() + min(max(timeout_seconds, 0.0), 0.25)
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    mechanism = "fcntl"
                elif msvcrt is not None:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    mechanism = "msvcrt"
                else:
                    raise ValueError("shared_lock_unavailable")
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(
                        "could not acquire bound domain store lock "
                        f"within {timeout_seconds}s"
                    ) from exc
                time.sleep(poll_interval)
        _validate_bound_lock_path(lock_path, descriptor)
        _validate_portable_directory_chain(chain)
        try:
            yield {"locked": True, "mode": "shared"}
        finally:
            _validate_bound_lock_path(lock_path, descriptor)
            _validate_portable_directory_chain(chain)
    finally:
        if mechanism == "fcntl":
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif mechanism == "msvcrt":
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)


def _validate_bound_lock_path(lock_path: Path, descriptor: int) -> None:
    named = os.stat(lock_path, follow_symlinks=False)
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(named.st_mode) or not stat.S_ISREG(opened.st_mode):
        raise ValueError("domain-intelligence lock path must be a regular file")
    if _stable_identity(named) != _stable_identity(opened):
        raise ValueError("domain-intelligence lock path changed while locking")


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino), stat.S_IFMT(metadata.st_mode))
