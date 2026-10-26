"""Locked, atomic writes for the memory provider's own state files.

The provider keeps counters (``dreaming.json``) and the latest brief
(``consolidation.json``) per Hermes profile under
``<omh_home>/memory/profiles/<slot>``, and the served receipt and two bounded
journals under ``<omh_home>/memory``. A CLI session and a gateway session can run against the
same home at once, so a read-modify-write that is not serialized loses the
other writer's update, and a plain ``write_text`` that dies half-way leaves a
file no reader accepts. This module is the open-reminders ledger's pattern
(``memory_open_reminders.mark_open_reminder_asked``) made reusable: the
bundle's one sanctioned OS lock on a sidecar lock file beside the target, then
a temporary file in the same directory and ``os.replace``.

A lock timeout raises ``TimeoutError`` (an ``OSError``); the provider swallows
it through ``_safely`` and counts it, so a contended home costs one write and
says so, never the turn. Stdlib only; no import of the ``omh`` control plane.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Iterator

from .awareness_delivery import _awareness_delivery_lock

# Longer than the lock's telemetry default (0.1s), which is sized to drop a
# counter under contention. These counters decide when consolidation is due,
# and losing them under contention is the defect the lock exists to close, so
# a writer waits for the other session's write -- milliseconds -- rather than
# dropping its own. Still bounded: a stuck holder costs one counted write.
STATE_LOCK_TIMEOUT_SECONDS = 2.0


@contextmanager
def state_file_lock(path: Path) -> Iterator[str]:
    """Serialize every writer of ``path``; yields the lock mechanism that held it.

    Not re-entrant: an flock is per open file description, so taking it again
    for the same path inside the block waits on itself until the timeout.
    """
    with _awareness_delivery_lock(path, timeout_seconds=STATE_LOCK_TIMEOUT_SECONDS) as mechanism:
        yield mechanism


def write_text_atomic(path: Path, text: str) -> None:
    """Replace ``path`` whole: a crash leaves the previous file, never half of one.

    ``newline="\\n"`` keeps the bytes identical on Windows, where text mode
    would otherwise write CRLF. The temporary file is removed on any failure,
    including one that is not an ``OSError`` (an unencodable surrogate).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        replaced = True
    finally:
        if not replaced:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def write_text_locked(path: Path, text: str) -> None:
    """``write_text_atomic`` under the state lock, for whole-file writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with state_file_lock(path):
        write_text_atomic(path, text)


__all__ = ["STATE_LOCK_TIMEOUT_SECONDS", "state_file_lock", "write_text_atomic", "write_text_locked"]
