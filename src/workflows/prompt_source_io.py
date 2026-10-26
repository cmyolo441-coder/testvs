from __future__ import annotations

import os
from pathlib import Path
import stat


_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", None)
_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)


class PromptSourceAccessError(ValueError):
    pass


def read_regular_prompt_source(path: Path, *, maximum_bytes: int) -> bytes:
    nofollow = _NOFOLLOW_FLAG
    if nofollow is None:
        raise PromptSourceAccessError("secure prompt audit file access is unavailable")
    try:
        descriptor = _open_without_symlinks(path, nofollow)
        with os.fdopen(descriptor, "rb") as handle:
            content = handle.read(maximum_bytes + 1)
    except OSError as exc:
        raise PromptSourceAccessError("prompt audit source cannot be safely opened") from exc
    if len(content) > maximum_bytes:
        raise PromptSourceAccessError(f"prompt audit source exceeds {maximum_bytes} bytes")
    return content


def _open_without_symlinks(path: Path, nofollow: int) -> int:
    descriptor = os.open(path.anchor, os.O_RDONLY | _DIRECTORY_FLAG)
    try:
        parts = path.parts[1:]
        for index, part in enumerate(parts):
            flags = os.O_RDONLY | nofollow | _NONBLOCK_FLAG
            if index != len(parts) - 1:
                flags |= _DIRECTORY_FLAG
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PromptSourceAccessError("prompt audit source must be a regular file")
        return descriptor
    except (OSError, PromptSourceAccessError):
        os.close(descriptor)
        raise
