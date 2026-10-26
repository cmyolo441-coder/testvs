from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat

from ..paths import OmhPaths
from ..system.binary_io import open_binary
from .domain_intelligence_bounded_json import (
    decode_bounded_json_object,
    read_limited_bytes,
)


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)
_WINDOWS_PATH_CTIME_IS_BIRTHTIME = os.name == "nt"
_MANAGED_DIRECTORIES = frozenset(
    {"candidates", "history", "operations", "profiles", "reviews"}
)
_HEALTH_DIRECTORIES = ("profiles", "reviews", "history")


def open_domain_directory(
    paths: OmhPaths,
    *relative_parts: str,
    create: bool,
) -> int | Path:
    """Open a domain directory through one anchored, no-follow descriptor chain."""
    home = Path(os.path.abspath(paths.omh_home))
    parts = ("memory", "domain-intelligence", *relative_parts)
    directory_fd = (
        _open_home_tree(home, parts, create=create)
        if _NOFOLLOW_FLAG and _DIRECTORY_FLAG
        else _open_domain_directory_portable(home, parts, create=create)
    )
    if create and not relative_parts:
        try:
            _ensure_health_directories(directory_fd)
        except (OSError, ValueError):
            if isinstance(directory_fd, int):
                os.close(directory_fd)
            raise
    return directory_fd


def open_domain_directory_path(directory: Path) -> int | Path:
    """Open an existing domain directory without re-resolving managed parents."""
    absolute = Path(os.path.abspath(directory))
    parts = absolute.parts
    domain_index = _domain_component_index(parts)
    home = Path(*parts[: domain_index - 1])
    relative_parts = parts[domain_index + 1 :]
    tree_parts = ("memory", "domain-intelligence", *relative_parts)
    if _NOFOLLOW_FLAG and _DIRECTORY_FLAG:
        return _open_home_tree(home, tree_parts, create=False)
    return _open_domain_directory_portable(home, tree_parts, create=False)


def read_managed_json_at(
    directory_fd: int | Path,
    filename: str,
    *,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
) -> dict[str, object] | None:
    if Path(filename).name != filename:
        raise ValueError("artifact_path_escape")
    if isinstance(directory_fd, Path):
        return _read_managed_json_at_portable(
            directory_fd,
            filename,
            max_bytes=max_bytes,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
    flags = os.O_RDONLY | _CLOEXEC_FLAG | _NOFOLLOW_FLAG
    try:
        descriptor = open_binary(filename, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError("artifact_symlink") from exc
        raise
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError("symlink_or_not_file")
        if file_stat.st_size > max_bytes:
            raise ValueError("artifact_too_large")
        raw = read_limited_bytes(descriptor, max_bytes)
    finally:
        os.close(descriptor)
    if len(raw) > max_bytes:
        raise ValueError("artifact_too_large")
    return decode_bounded_json_object(
        raw,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


def atomic_write_managed_json(
    paths: OmhPaths,
    managed_name: str,
    filename: str,
    data: dict[str, object],
) -> None:
    """Atomically write private JSON below one safely checked managed directory."""
    if managed_name not in _MANAGED_DIRECTORIES:
        raise ValueError("unsafe_domain_managed_directory")
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("unsafe_domain_artifact_filename")
    if not _NOFOLLOW_FLAG or not _DIRECTORY_FLAG:
        _atomic_write_managed_json_portable(paths, managed_name, filename, data)
        return
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_digest = hashlib.sha256(payload).digest()
    directory_fd = open_domain_directory(paths, managed_name, create=True)
    temporary_name = f".{filename}.{os.getpid()}-{secrets.token_hex(8)}.tmp"
    temporary_created = False
    try:
        target_before = _regular_target_at(directory_fd, filename)
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | _CLOEXEC_FLAG
            | _NOFOLLOW_FLAG
        )
        temporary_fd = open_binary(temporary_name, flags, 0o600, dir_fd=directory_fd)
        temporary_created = True
        try:
            temporary_identity = os.fstat(temporary_fd)
            _validate_portable_regular(temporary_identity, "temporary path")
            os.fchmod(temporary_fd, 0o600)
            _write_all(temporary_fd, payload)
            os.fsync(temporary_fd)
            _verify_temporary_descriptor(
                temporary_fd,
                expected_identity=_portable_identity(temporary_identity),
                expected_size=len(payload),
                expected_digest=expected_digest,
            )
        finally:
            os.close(temporary_fd)
        linked_temporary = os.stat(
            temporary_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _validate_portable_regular(linked_temporary, "temporary path")
        if _portable_identity(temporary_identity) != _portable_identity(linked_temporary):
            raise ValueError("domain-intelligence temporary path changed while writing")
        target_after = _regular_target_at(directory_fd, filename)
        if not _same_optional_portable_identity(target_before, target_after):
            raise ValueError("domain-intelligence managed target changed before replacement")
        _verify_temporary_at(
            directory_fd,
            temporary_name,
            expected_identity=_portable_identity(linked_temporary),
            expected_size=len(payload),
            expected_digest=expected_digest,
        )
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_created = False
        replaced = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
        _validate_portable_regular(replaced, "managed target")
        if _portable_identity(linked_temporary) != _portable_identity(replaced):
            raise ValueError("domain-intelligence managed target changed during replacement")
        os.fsync(directory_fd)
    except (TypeError, NotImplementedError) as exc:
        raise ValueError(
            "domain-intelligence safe managed writes are unavailable"
        ) from exc
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _read_managed_json_at_portable(
    directory_fd: Path,
    filename: str,
    *,
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
) -> dict[str, object] | None:
    directory = directory_fd
    home = _managed_home_from_directory(directory)
    chain = _portable_directory_chain(directory, home=home, create=False)
    path = directory / filename
    try:
        before = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _validate_portable_regular(before, "managed source")
    if before.st_size > max_bytes:
        raise ValueError("artifact_too_large")
    descriptor = open_binary(
        path,
        os.O_RDONLY | _CLOEXEC_FLAG,
    )
    try:
        opened = os.fstat(descriptor)
        _validate_portable_regular(opened, "managed source")
        if _portable_identity(before) != _portable_identity(opened):
            raise ValueError("domain-intelligence managed source changed while opening")
        raw = read_limited_bytes(descriptor, max_bytes)
        after = os.fstat(descriptor)
        named_after = os.stat(path, follow_symlinks=False)
        _validate_portable_regular(named_after, "managed source")
        if (
            _portable_snapshot(opened) != _portable_snapshot(after)
            or _portable_cross_api_snapshot(after)
            != _portable_cross_api_snapshot(named_after)
            or len(raw) != after.st_size
        ):
            raise ValueError("domain-intelligence managed source changed while reading")
        _validate_portable_directory_chain(chain)
    finally:
        os.close(descriptor)
    if len(raw) > max_bytes:
        raise ValueError("artifact_too_large")
    return decode_bounded_json_object(
        raw,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


def _atomic_write_managed_json_portable(
    paths: OmhPaths,
    managed_name: str,
    filename: str,
    data: dict[str, object],
) -> None:
    """Use lstat/open/fstat identity checks where no-follow dirfds are absent."""
    home = Path(os.path.abspath(paths.omh_home))
    directory = home / "memory" / "domain-intelligence" / managed_name
    directory_chain = _portable_directory_chain(directory, home=home, create=True)
    target = directory / filename
    target_before = _portable_regular_target(target)
    payload = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_digest = hashlib.sha256(payload).digest()
    temporary = directory / f".{filename}.{os.getpid()}-{secrets.token_hex(8)}.tmp"
    temporary_created = False
    descriptor = -1
    try:
        descriptor = open_binary(
            temporary,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | _CLOEXEC_FLAG,
            0o600,
        )
        temporary_created = True
        opened = os.fstat(descriptor)
        _validate_portable_regular(opened, "temporary path")
        linked = os.stat(temporary, follow_symlinks=False)
        _validate_portable_regular(linked, "temporary path")
        if _portable_identity(opened) != _portable_identity(linked):
            raise ValueError("domain-intelligence temporary path changed while opening")
        if hasattr(os, "fchmod"):
            os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        _verify_temporary_descriptor(
            descriptor,
            expected_identity=_portable_identity(opened),
            expected_size=len(payload),
            expected_digest=expected_digest,
        )
        os.close(descriptor)
        descriptor = -1

        _validate_portable_directory_chain(directory_chain)
        target_after = _portable_regular_target(target)
        if not _same_optional_portable_identity(target_before, target_after):
            raise ValueError("domain-intelligence managed target changed before replacement")
        _verify_portable_temporary(
            temporary,
            expected_identity=_portable_identity(linked),
            expected_size=len(payload),
            expected_digest=expected_digest,
        )
        os.replace(temporary, target)
        temporary_created = False
        replaced = os.stat(target, follow_symlinks=False)
        _validate_portable_regular(replaced, "managed target")
        if _portable_identity(linked) != _portable_identity(replaced):
            raise ValueError("domain-intelligence managed target changed during replacement")
        _validate_portable_directory_chain(directory_chain)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_created:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _verify_temporary_at(
    directory_fd: int,
    temporary_name: str,
    *,
    expected_identity: tuple[int, int, int],
    expected_size: int,
    expected_digest: bytes,
) -> None:
    before = os.stat(
        temporary_name,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    _validate_portable_regular(before, "temporary path")
    if _portable_identity(before) != expected_identity:
        raise ValueError("domain-intelligence temporary path changed before replacement")
    descriptor = open_binary(
        temporary_name,
        os.O_RDONLY | _CLOEXEC_FLAG | _NOFOLLOW_FLAG,
        dir_fd=directory_fd,
    )
    try:
        _verify_temporary_descriptor(
            descriptor,
            expected_identity=expected_identity,
            expected_size=expected_size,
            expected_digest=expected_digest,
        )
    finally:
        os.close(descriptor)
    after = os.stat(
        temporary_name,
        dir_fd=directory_fd,
        follow_symlinks=False,
    )
    _validate_portable_regular(after, "temporary path")
    if _portable_identity(after) != expected_identity:
        raise ValueError("domain-intelligence temporary path changed before replacement")


def _verify_portable_temporary(
    temporary: Path,
    *,
    expected_identity: tuple[int, int, int],
    expected_size: int,
    expected_digest: bytes,
) -> None:
    before = os.stat(temporary, follow_symlinks=False)
    _validate_portable_regular(before, "temporary path")
    if _portable_identity(before) != expected_identity:
        raise ValueError("domain-intelligence temporary path changed before replacement")
    descriptor = open_binary(temporary, os.O_RDONLY | _CLOEXEC_FLAG)
    try:
        _verify_temporary_descriptor(
            descriptor,
            expected_identity=expected_identity,
            expected_size=expected_size,
            expected_digest=expected_digest,
        )
    finally:
        os.close(descriptor)
    after = os.stat(temporary, follow_symlinks=False)
    _validate_portable_regular(after, "temporary path")
    if _portable_identity(after) != expected_identity:
        raise ValueError("domain-intelligence temporary path changed before replacement")


def _verify_temporary_descriptor(
    descriptor: int,
    *,
    expected_identity: tuple[int, int, int],
    expected_size: int,
    expected_digest: bytes,
) -> None:
    before = os.fstat(descriptor)
    _validate_portable_regular(before, "temporary path")
    if _portable_identity(before) != expected_identity:
        raise ValueError("domain-intelligence temporary path changed before replacement")
    if before.st_size != expected_size:
        raise ValueError("domain-intelligence temporary content changed before replacement")
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            raise ValueError(
                "domain-intelligence temporary content changed before replacement"
            )
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError("domain-intelligence temporary content changed before replacement")
    after = os.fstat(descriptor)
    if (
        _portable_identity(after) != expected_identity
        or after.st_size != expected_size
        or digest.digest() != expected_digest
    ):
        raise ValueError("domain-intelligence temporary content changed before replacement")


def _portable_directory_chain(
    directory: Path,
    *,
    home: Path,
    create: bool,
) -> tuple[tuple[Path, tuple[int, int, int]], ...]:
    if not directory.is_absolute() or not home.is_absolute():
        raise ValueError("domain-intelligence managed storage cannot be safely opened")
    current = Path(directory.anchor)
    chain: list[tuple[Path, tuple[int, int, int]]] = []
    for part in directory.parts[1:]:
        current /= part
        try:
            metadata = os.stat(current, follow_symlinks=False)
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            metadata = os.stat(current, follow_symlinks=False)
        _validate_portable_directory(metadata)
        if current == home or home in current.parents:
            try:
                os.chmod(current, 0o700, follow_symlinks=False)
            except (NotImplementedError, TypeError):
                os.chmod(current, 0o700)
        chain.append((current, _portable_identity(metadata)))
    return tuple(chain)


def _validate_portable_directory_chain(
    chain: tuple[tuple[Path, tuple[int, int, int]], ...],
) -> None:
    for path, expected in chain:
        try:
            current = os.stat(path, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError(
                "domain-intelligence managed directory changed while writing"
            ) from exc
        _validate_portable_directory(current)
        if _portable_identity(current) != expected:
            raise ValueError("domain-intelligence managed directory changed while writing")


def _regular_target_at(
    directory_fd: int,
    filename: str,
) -> os.stat_result | None:
    try:
        metadata = os.stat(filename, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _validate_portable_regular(metadata, "managed target")
    return metadata


def _portable_regular_target(path: Path) -> os.stat_result | None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return None
    _validate_portable_regular(metadata, "managed target")
    return metadata


def _validate_portable_directory(metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError("domain-intelligence managed directory must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("domain-intelligence managed directory must be a directory")


def _validate_portable_regular(metadata: os.stat_result, label: str) -> None:
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"domain-intelligence {label} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"domain-intelligence {label} must be a regular file")


def _portable_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _portable_snapshot(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        *_portable_identity(metadata),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _portable_cross_api_snapshot(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    snapshot = _portable_snapshot(metadata)
    if not _WINDOWS_PATH_CTIME_IS_BIRTHTIME:
        return snapshot
    return (*snapshot[:-1], 0)


def _same_optional_portable_identity(
    before: os.stat_result | None,
    after: os.stat_result | None,
) -> bool:
    if before is None or after is None:
        return before is after
    return _portable_identity(before) == _portable_identity(after)


def _open_domain_directory_portable(
    home: Path,
    relative_parts: tuple[str, ...],
    *,
    create: bool,
) -> Path:
    directory = home.joinpath(*relative_parts)
    chain = _portable_directory_chain(directory, home=home, create=create)
    _validate_portable_directory_chain(chain)
    return directory


def _managed_home_from_directory(directory: Path) -> Path:
    absolute = Path(os.path.abspath(directory))
    parts = absolute.parts
    domain_index = _domain_component_index(parts)
    return Path(*parts[: domain_index - 1])


def _open_home_tree(
    home: Path,
    relative_parts: tuple[str, ...],
    *,
    create: bool,
) -> int:
    if not _NOFOLLOW_FLAG or not _DIRECTORY_FLAG:
        raise ValueError("domain-intelligence safe managed writes are unavailable")
    if not home.is_absolute() or home.name in {"", ".", ".."}:
        raise ValueError("domain-intelligence managed storage cannot be safely opened")
    flags = os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG | _NOFOLLOW_FLAG
    directory_fd = -1
    try:
        directory_fd, missing_parts = _open_anchor(home.parent, flags, create=create)
        parts = (*missing_parts, home.name, *relative_parts)
        for index, part in enumerate(parts):
            if Path(part).name != part or part in {"", ".", ".."}:
                raise ValueError(
                    "domain-intelligence managed storage cannot be safely opened"
                )
            try:
                next_directory_fd = os.open(part, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o700, dir_fd=directory_fd)
                next_directory_fd = os.open(part, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_directory_fd
            if index:
                os.fchmod(directory_fd, 0o700)
        return directory_fd
    except (OSError, ValueError) as exc:
        if directory_fd >= 0:
            os.close(directory_fd)
        if isinstance(exc, OSError) and exc.errno in {
            errno.ELOOP,
            errno.EMLINK,
            errno.ENOTDIR,
        }:
            raise ValueError(
                "domain-intelligence managed storage cannot be safely opened: "
                "symlink or non-directory component"
            ) from exc
        raise


def _ensure_health_directories(domain_root_fd: int | Path) -> None:
    """Create the complete resolver health universe for a writable store."""
    if isinstance(domain_root_fd, Path):
        home = _managed_home_from_directory(domain_root_fd)
        for name in _HEALTH_DIRECTORIES:
            _portable_directory_chain(domain_root_fd / name, home=home, create=True)
        return
    flags = os.O_RDONLY | _DIRECTORY_FLAG | _CLOEXEC_FLAG | _NOFOLLOW_FLAG
    for name in _HEALTH_DIRECTORIES:
        try:
            os.mkdir(name, 0o700, dir_fd=domain_root_fd)
        except FileExistsError:
            pass
        try:
            descriptor = os.open(name, flags, dir_fd=domain_root_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EMLINK, errno.ENOTDIR}:
                raise ValueError(
                    "domain-intelligence health directory contains a symlink or non-directory"
                ) from exc
            raise
        try:
            os.fchmod(descriptor, 0o700)
        finally:
            os.close(descriptor)


def _domain_component_index(parts: tuple[str, ...]) -> int:
    for index in range(len(parts) - 1, 1, -1):
        if parts[index] == "domain-intelligence" and parts[index - 1] == "memory":
            return index
    raise ValueError("domain-intelligence path is outside managed storage")


def _open_anchor(path: Path, flags: int, *, create: bool) -> tuple[int, tuple[str, ...]]:
    missing: list[str] = []
    current = path
    while True:
        try:
            return os.open(current, flags), tuple(reversed(missing))
        except FileNotFoundError:
            if not create or current == current.parent:
                raise
            missing.append(current.name)
            current = current.parent


def _write_all(file_descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(file_descriptor, view)
        if written <= 0:
            raise OSError("domain-intelligence managed write made no progress")
        view = view[written:]
