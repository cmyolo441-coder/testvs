"""Whether stdin is a terminal, answered without raising.

A shell that closes stdin (`<&-`) starts Python with `sys.stdin` set to None,
so `sys.stdin.isatty()` raises AttributeError; a stream closed underneath the
process raises ValueError. Every interactive entry point asks this question
before it prompts, and each of them used to ask it bare, so a closed stdin
became a traceback instead of the refusal the command already had for a pipe.
One home, so a new prompt cannot reintroduce the crash by copying the bare
probe.

On Windows `isatty()` is not enough: the CRT reports any character device as
a tty, and `NUL` (what `os.devnull` and `< NUL` open) is one, so a stdin from
the null device read as a terminal and the setup wizard ran on it. Only a
console handle has a console mode, so Windows also asks `GetConsoleMode`.
"""

from __future__ import annotations

import os
import sys

_IS_WINDOWS = os.name == "nt"


def stdin_is_tty() -> bool:
    """True only when stdin is open and attached to a terminal."""
    try:
        if not sys.stdin.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    return _has_console_mode(sys.stdin) if _IS_WINDOWS else True


def _has_console_mode(stream) -> bool:
    """Whether `stream` is a Windows console: `GetConsoleMode` succeeds only there.

    A stream with no OS descriptor (an in-process stand-in such as a
    StringIO) has no handle to ask about, so its own `isatty()` answer
    stands. For a real descriptor, every failure -- a bad descriptor, no
    msvcrt or kernel32 -- answers False, so the probe can never raise.
    """
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return True
    try:
        import ctypes
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = msvcrt.get_osfhandle(descriptor)
        mode = ctypes.c_uint32()
        return bool(kernel32.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)))
    except (AttributeError, ImportError, OSError, ValueError):
        return False
