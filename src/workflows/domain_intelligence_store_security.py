from __future__ import annotations

import errno
import os
from contextlib import contextmanager, ExitStack
from pathlib import Path
import stat
import threading
import time
from typing import Iterator

from ..paths import OmhPaths
from ..system.binary_io import open_binary, set_binary_mode
from ..system.local_store import FileLockTimeout
from .domain_intelligence_store_writer import (
    _managed_home_from_directory,
    _portable_directory_chain,
    _validate_portable_directory_chain,
    atomic_write_managed_json,
    open_domain_directory,
    open_domain_directory_path,
    read_managed_json_at,
)

__all__ = ("atomic_write_managed_json",)

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)
_MANAGED_DIRECTORIES = frozenset(
    {"candidates", "history", "operations", "profiles", "reviews"}
)
_LOCK_STATE = threading.local()

# These bounds keep local reviewed metadata cheap to inspect and diagnose.
MAX_DOMAIN_ARTIFACT_BYTES = 256 * 1024
MAX_DOMAIN_CANDIDATE_FILES = 256
MAX_DOMAIN_ARTIFACT_FILES = 1024
MAX_DOMAIN_JSON_DEPTH = 32
MAX_DOMAIN_JSON_NODES = 4096
MAX_DOMAIN_DIAGNOSTICS = 64


def ensure_new_artifact_capacity(
    directory: Path,
    target: Path,
    *,
    limit: int,
    reason: str,
) -> None:
    paths, overflow = bounded_json_paths(directory, limit=max(limit - 1, 0))
    if not target.exists() and (overflow or len(paths) >= limit):
        raise ValueError(reason)


def bounded_json_paths(directory: Path, *, limit: int) -> tuple[tuple[Path, ...], bool]:
    """Return at most ``limit + 1`` JSON paths without an unbounded scan."""
    paths: list[Path] = []
    scan_limit = max(limit * 2 + 1, 1)
    scanned = 0
    scan_overflow = False
    with anchored_directory_path(directory) as directory_fd:
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                scanned += 1
                if scanned > scan_limit:
                    scan_overflow = True
                    break
                if entry.name.endswith(".json"):
                    paths.append(directory / entry.name)
                    if len(paths) > limit:
                        break
    overflow = len(paths) > limit or scan_overflow
    return tuple(sorted(paths)), overflow


def secure_domain_root(paths: OmhPaths, *, create: bool = False) -> Path:
    root = paths.memory_dir / "domain-intelligence"
    try:
        descriptor = open_domain_directory(paths, create=create)
    except FileNotFoundError:
        if create:
            raise
        return root
    if isinstance(descriptor, int):
        os.close(descriptor)
    return root


def secure_managed_dir(paths: OmhPaths, name: str, *, create: bool = True) -> Path:
    if name not in _MANAGED_DIRECTORIES:
        raise ValueError("unsafe_domain_managed_directory")
    with anchored_managed_directory(paths, name, create=create):
        return paths.memory_dir / "domain-intelligence" / name


def secure_artifact_path(directory: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("domain-intelligence artifact path must remain managed")
    path = directory / filename
    with anchored_directory_path(directory) as directory_fd:
        try:
            file_stat = (
                os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
                if isinstance(directory_fd, int)
                else os.stat(path, follow_symlinks=False)
            )
        except FileNotFoundError:
            return path
        if stat.S_ISLNK(file_stat.st_mode):
            raise ValueError("domain-intelligence artifact path must not be a symlink")
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("domain-intelligence artifact path must be a regular file")
    return path


def secure_store_lock_target(paths: OmhPaths) -> Path:
    root = paths.memory_dir / "domain-intelligence"
    with _domain_root_descriptor(paths) as directory_fd:
        if isinstance(directory_fd, Path):
            try:
                lock_stat = _lock_path_stat(directory_fd / ".store.lock")
            except FileNotFoundError:
                pass
            else:
                _validate_lock_metadata(lock_stat)
        else:
            _validate_lock_entry(directory_fd)
    return root / "store"


@contextmanager
def anchored_managed_directory(
    paths: OmhPaths,
    name: str,
    *,
    create: bool = True,
) -> Iterator[int | Path]:
    if name not in _MANAGED_DIRECTORIES:
        raise ValueError("unsafe_domain_managed_directory")
    descriptor = open_domain_directory(paths, name, create=create)
    chain = (
        _portable_directory_chain(
            descriptor,
            home=Path(os.path.abspath(paths.omh_home)),
            create=False,
        )
        if isinstance(descriptor, Path)
        else None
    )
    try:
        yield descriptor
    finally:
        if chain is not None:
            _validate_portable_directory_chain(chain)
        else:
            os.close(descriptor)


@contextmanager
def anchored_directory_path(directory: Path) -> Iterator[int | Path]:
    descriptor = open_domain_directory_path(directory)
    chain = (
        _portable_directory_chain(
            descriptor,
            home=_managed_home_from_directory(descriptor),
            create=False,
        )
        if isinstance(descriptor, Path)
        else None
    )
    try:
        yield descriptor
    finally:
        if chain is not None:
            _validate_portable_directory_chain(chain)
        else:
            os.close(descriptor)


@contextmanager
def _domain_root_descriptor(paths: OmhPaths) -> Iterator[int | Path]:
    descriptor = open_domain_directory(paths, create=True)
    try:
        yield descriptor
    finally:
        if isinstance(descriptor, int):
            os.close(descriptor)


def _validate_lock_entry(directory_fd: int) -> None:
    try:
        lock_stat = os.stat(
            ".store.lock",
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if stat.S_ISLNK(lock_stat.st_mode):
        raise ValueError("domain-intelligence lock path must not be a symlink")
    if not stat.S_ISREG(lock_stat.st_mode):
        raise ValueError("domain-intelligence lock path must be a regular file")


def _lock_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _validate_lock_metadata(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("domain-intelligence lock path must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("domain-intelligence lock path must be a regular file")


def _lock_stat(directory_fd: int) -> os.stat_result:
    return os.stat(
        ".store.lock",
        dir_fd=directory_fd,
        follow_symlinks=False,
    )


def _lock_path_stat(lock_path: Path) -> os.stat_result:
    return os.stat(lock_path, follow_symlinks=False)


def _prepare_portable_domain_root(
    paths: OmhPaths,
) -> tuple[Path, tuple[tuple[Path, tuple[int, int, int]], ...]]:
    home = Path(os.path.abspath(paths.omh_home))
    root = home / "memory" / "domain-intelligence"
    chain = _portable_directory_chain(root, home=home, create=True)
    for name in ("profiles", "reviews", "history"):
        _portable_directory_chain(root / name, home=home, create=True)
    _validate_portable_directory_chain(chain)
    return root, chain


def _open_store_lock_path(lock_path: Path, flags: int) -> int:
    before: os.stat_result | None
    try:
        before = _lock_path_stat(lock_path)
    except FileNotFoundError:
        before = None
        try:
            descriptor = open_binary(
                lock_path,
                flags | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            before = _lock_path_stat(lock_path)
            _validate_lock_metadata(before)
            descriptor = open_binary(
                lock_path,
                flags & ~(os.O_CREAT | os.O_EXCL),
            )
    else:
        _validate_lock_metadata(before)
        descriptor = open_binary(
            lock_path,
            flags & ~(os.O_CREAT | os.O_EXCL),
        )

    with os.fdopen(descriptor, "r+b", closefd=True):
        opened = os.fstat(descriptor)
        _validate_lock_metadata(opened)
        try:
            after = _lock_path_stat(lock_path)
        except FileNotFoundError as exc:
            raise ValueError(
                "domain-intelligence lock path changed while opening"
            ) from exc
        _validate_lock_metadata(after)
        if _lock_identity(opened) != _lock_identity(after) or (
            before is not None
            and _lock_identity(before) != _lock_identity(opened)
        ):
            raise ValueError("domain-intelligence lock path changed while opening")
        return set_binary_mode(os.dup(descriptor))


def _validate_open_lock_path(lock_path: Path, descriptor: int) -> None:
    try:
        named = _lock_path_stat(lock_path)
    except FileNotFoundError as exc:
        raise ValueError("domain-intelligence lock path changed while locking") from exc
    _validate_lock_metadata(named)
    opened = os.fstat(descriptor)
    _validate_lock_metadata(opened)
    if _lock_identity(opened) != _lock_identity(named):
        raise ValueError("domain-intelligence lock path changed while locking")


def _open_store_lock_descriptor(directory_fd: int, flags: int) -> int:
    if _NOFOLLOW_FLAG:
        try:
            return open_binary(
                ".store.lock",
                flags | _NOFOLLOW_FLAG,
                0o600,
                dir_fd=directory_fd,
            )
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK}:
                raise ValueError(
                    "domain-intelligence lock path must not be a symlink"
                ) from exc
            raise

    before: os.stat_result | None
    try:
        before = _lock_stat(directory_fd)
    except FileNotFoundError:
        before = None
        try:
            descriptor = open_binary(
                ".store.lock",
                flags | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError:
            before = _lock_stat(directory_fd)
            _validate_lock_metadata(before)
            descriptor = open_binary(
                ".store.lock",
                flags & ~(os.O_CREAT | os.O_EXCL),
                dir_fd=directory_fd,
            )
    else:
        _validate_lock_metadata(before)
        descriptor = open_binary(
            ".store.lock",
            flags & ~(os.O_CREAT | os.O_EXCL),
            dir_fd=directory_fd,
        )

    try:
        opened = os.fstat(descriptor)
        _validate_lock_metadata(opened)
        try:
            after = _lock_stat(directory_fd)
        except FileNotFoundError as exc:
            raise ValueError(
                "domain-intelligence lock path changed while opening"
            ) from exc
        _validate_lock_metadata(after)
        if _lock_identity(opened) != _lock_identity(after) or (
            before is not None
            and _lock_identity(before) != _lock_identity(opened)
        ):
            raise ValueError("domain-intelligence lock path changed while opening")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _lock_descriptor(
    descriptor: int,
    target: Path,
    timeout_seconds: float,
    poll_interval: float,
) -> str | None:
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        raise ValueError("domain-intelligence lock path must be a regular file")
    if fcntl is not None:
        os.fchmod(descriptor, 0o600)
        mechanism = "fcntl"
    elif msvcrt is not None:
        # Native msvcrt materializes the locked byte as NUL when this sidecar
        # is empty. That byte belongs to synchronization, not store content.
        mechanism = "msvcrt"
    else:
        return None
    deadline = time.monotonic() + max(timeout_seconds, 0.0)
    while True:
        try:
            if mechanism == "fcntl":
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            return mechanism
        except OSError as exc:
            if exc.errno not in (
                errno.EACCES,
                errno.EAGAIN,
                errno.EDEADLK,
                getattr(errno, "EDEADLOCK", errno.EDEADLK),
            ):
                raise
            if time.monotonic() >= deadline:
                raise FileLockTimeout(
                    f"could not acquire lock on {target} within {timeout_seconds}s"
                ) from exc
            time.sleep(poll_interval)


def _unlock_descriptor(descriptor: int, mechanism: str) -> None:
    if mechanism == "fcntl":
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return
    os.lseek(descriptor, 0, os.SEEK_SET)
    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


@contextmanager
def domain_store_lock(
    paths: OmhPaths,
    *,
    timeout_seconds: float = 10.0,
    poll_interval: float = 0.05,
) -> Iterator[dict[str, object]]:
    target = paths.memory_dir / "domain-intelligence" / "store"
    lock_key = os.fspath(target.absolute())
    depths = getattr(_LOCK_STATE, "depths", None)
    if depths is None:
        depths = {}
        _LOCK_STATE.depths = depths
    if depths.get(lock_key, 0):
        depths[lock_key] += 1
        try:
            locked = fcntl is not None or msvcrt is not None
            yield {
                "locked": locked,
                "reason": "" if locked else "os_file_lock_unavailable",
            }
        finally:
            depths[lock_key] -= 1
        return

    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | _CLOEXEC_FLAG
    with ExitStack() as resources:
        root_chain: tuple[tuple[Path, tuple[int, int, int]], ...] | None = None
        root_fd = resources.enter_context(_domain_root_descriptor(paths))
        if isinstance(root_fd, Path):
            root = root_fd
            home = Path(os.path.abspath(paths.omh_home))
            root_chain = _portable_directory_chain(root, home=home, create=False)
            _validate_portable_directory_chain(root_chain)
            descriptor = _open_store_lock_path(root / ".store.lock", flags)
        else:
            descriptor = _open_store_lock_descriptor(root_fd, flags)
        resources.callback(os.close, descriptor)
        mechanism = _lock_descriptor(
            descriptor,
            target,
            timeout_seconds,
            poll_interval,
        )
        try:
            if root_chain is not None:
                _validate_open_lock_path(root / ".store.lock", descriptor)
                _validate_portable_directory_chain(root_chain)
            depths[lock_key] = 1
            try:
                yield {
                    "locked": mechanism is not None,
                    "reason": "" if mechanism else "os_file_lock_unavailable",
                }
            finally:
                depths.pop(lock_key, None)
        finally:
            if mechanism is not None:
                _unlock_descriptor(descriptor, mechanism)


def unlink_managed_json(
    paths: OmhPaths,
    managed_name: str,
    filename: str,
    *,
    expected: dict[str, object],
) -> None:
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("unsafe_domain_artifact_filename")
    with anchored_managed_directory(paths, managed_name) as directory:
        if read_bounded_json_at(directory, filename) != expected:
            raise ValueError("domain-intelligence managed target changed before deletion")
        if isinstance(directory, int):
            os.unlink(filename, dir_fd=directory)
            os.fsync(directory)
            return
        home = Path(os.path.abspath(paths.omh_home))
        chain = _portable_directory_chain(directory, home=home, create=False)
        target = directory / filename
        metadata = os.stat(target, follow_symlinks=False)
        _validate_lock_metadata(metadata)
        _validate_portable_directory_chain(chain)
        os.unlink(target)
        _validate_portable_directory_chain(chain)


def read_bounded_json(path: Path) -> dict[str, object] | None:
    with anchored_directory_path(path.parent) as directory_fd:
        return read_bounded_json_at(directory_fd, path.name)


def read_bounded_json_at(
    directory_fd: int | Path,
    filename: str,
) -> dict[str, object] | None:
    return read_managed_json_at(
        directory_fd,
        filename,
        max_bytes=MAX_DOMAIN_ARTIFACT_BYTES,
        max_depth=MAX_DOMAIN_JSON_DEPTH,
        max_nodes=MAX_DOMAIN_JSON_NODES,
    )
