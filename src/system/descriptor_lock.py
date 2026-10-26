"""Bounded fail-closed advisory locking for an already-open file descriptor."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import errno
import os
from pathlib import Path
from threading import Event
import time
from typing import Final

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    msvcrt = None

_BUSY_ERRNOS: Final = frozenset({errno.EACCES, errno.EAGAIN, errno.EDEADLK})


class DescriptorLockError(Exception):
    """Raised when an opened store descriptor cannot be locked safely."""


def require_descriptor_lock_backend() -> None:
    """Fail before a caller creates or opens mutable store state."""
    if fcntl is None and msvcrt is None:
        raise DescriptorLockError("no supported descriptor lock backend is available")


@contextmanager
def locked_descriptor(
    descriptor: int,
    path: Path,
    *,
    timeout_seconds: float = 10.0,
    poll_interval: float = 0.05,
) -> Iterator[None]:
    """Lock the actual opened store inode or fail without entering the body."""
    require_descriptor_lock_backend()
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    mechanism = ""
    while not mechanism:
        try:
            posix_backend = fcntl
            windows_backend = msvcrt
            if posix_backend is not None:
                posix_backend.flock(
                    descriptor,
                    posix_backend.LOCK_EX | posix_backend.LOCK_NB,
                )
                mechanism = "fcntl"
            elif windows_backend is not None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                windows_backend.locking(
                    descriptor,
                    windows_backend.LK_NBLCK,
                    1,
                )
                mechanism = "msvcrt"
            else:
                raise DescriptorLockError(
                    "no supported descriptor lock backend is available"
                )
        except OSError as exc:
            if exc.errno not in _BUSY_ERRNOS:
                raise DescriptorLockError(f"could not lock store descriptor: {path}") from exc
            if time.monotonic() >= deadline:
                raise DescriptorLockError(f"timed out locking store descriptor: {path}") from exc
            Event().wait(poll_interval)
    try:
        yield
    finally:
        if mechanism == "fcntl":
            posix_backend = fcntl
            if posix_backend is None:
                raise DescriptorLockError("POSIX descriptor lock backend disappeared")
            posix_backend.flock(descriptor, posix_backend.LOCK_UN)
        elif mechanism == "msvcrt":
            windows_backend = msvcrt
            if windows_backend is None:
                raise DescriptorLockError("Windows descriptor lock backend disappeared")
            os.lseek(descriptor, 0, os.SEEK_SET)
            windows_backend.locking(descriptor, windows_backend.LK_UNLCK, 1)
        else:
            raise DescriptorLockError("descriptor lock mechanism was not recorded")
