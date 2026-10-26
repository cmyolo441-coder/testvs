from __future__ import annotations

import errno
import os
from pathlib import Path
import stat

from .domain_intelligence_bounded_json import (
    decode_bounded_json_object,
    read_limited_bytes,
)
from .domain_intelligence_store_security import (
    MAX_DOMAIN_ARTIFACT_BYTES,
    MAX_DOMAIN_JSON_DEPTH,
    MAX_DOMAIN_JSON_NODES,
)
from .domain_intelligence_store_writer import read_managed_json_at


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)


def read_stable_json_at(directory_fd: int | Path, filename: str) -> dict[str, object]:
    if isinstance(directory_fd, Path):
        value = read_managed_json_at(
            directory_fd,
            filename,
            max_bytes=MAX_DOMAIN_ARTIFACT_BYTES,
            max_depth=MAX_DOMAIN_JSON_DEPTH,
            max_nodes=MAX_DOMAIN_JSON_NODES,
        )
        if value is None:
            raise FileNotFoundError(filename)
        return value
    if not _NOFOLLOW_FLAG:
        raise ValueError("domain-intelligence safe reads require O_NOFOLLOW")
    flags = os.O_RDONLY | _CLOEXEC_FLAG | _NOFOLLOW_FLAG
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError("artifact_symlink") from exc
        raise
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("symlink_or_not_file")
        if before.st_size > MAX_DOMAIN_ARTIFACT_BYTES:
            raise ValueError("artifact_too_large")
        raw = read_limited_bytes(descriptor, MAX_DOMAIN_ARTIFACT_BYTES)
        after = os.fstat(descriptor)
        if stable_file_identity(before) != stable_file_identity(after):
            raise ValueError("artifact_changed_during_read")
    finally:
        os.close(descriptor)
    if len(raw) > MAX_DOMAIN_ARTIFACT_BYTES:
        raise ValueError("artifact_too_large")
    return decode_bounded_json_object(
        raw,
        max_depth=MAX_DOMAIN_JSON_DEPTH,
        max_nodes=MAX_DOMAIN_JSON_NODES,
    )


def stable_file_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
