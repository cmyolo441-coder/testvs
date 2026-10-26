from __future__ import annotations

import os
from typing import Any

try:
    import msvcrt as _msvcrt
except ImportError:
    _msvcrt = None


_BINARY_FLAG = getattr(os, "O_BINARY", 0)


def set_binary_mode(descriptor: int) -> int:
    """Make a CRT-backed descriptor byte-exact; leave POSIX descriptors unchanged."""
    if _msvcrt is not None and _BINARY_FLAG:
        _msvcrt.setmode(descriptor, _BINARY_FLAG)
    return descriptor


def open_binary(path: Any, flags: int, mode: int = 0o777, **kwargs: Any) -> int:
    """Open a descriptor in binary mode on Windows and with unchanged POSIX semantics."""
    descriptor = os.open(path, flags | _BINARY_FLAG, mode, **kwargs)
    configured = False
    try:
        set_binary_mode(descriptor)
        configured = True
        return descriptor
    finally:
        if not configured:
            os.close(descriptor)
