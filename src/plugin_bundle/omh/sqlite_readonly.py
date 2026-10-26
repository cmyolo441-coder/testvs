"""Open a SQLite store read-only through a URI whose path is quoted.

``sqlite3.connect(f"file:{path}?mode=ro", uri=True)`` hands the path to
SQLite's URI parser as it is. A ``#`` in the path starts the fragment and a
``?`` starts the query, so the open lands on the path cut there, loses
``mode=ro``, and creates a writable file at the cut. A ``%41`` decodes to
``A`` and opens another directory's file. ``quote`` escapes every byte but
``/`` and the unreserved ones, and SQLite decodes the escapes back, so the
open reads the file the path names and nothing else.

The path's file-system bytes are quoted, not its ``str``. A POSIX name that
is not valid UTF-8 decodes to lone surrogates, which ``quote`` of a ``str``
refuses with ``UnicodeEncodeError``. Its ``os.fsencode`` bytes are the ones
``sqlite3.connect`` hands SQLite for a plain path. For a valid UTF-8 name the
two quote to the same URI.

Stdlib only and no ``omh`` import: Hermes loads this directory with its own
interpreter, and core code imports this module from the bundle.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from urllib.parse import quote


def connect_readonly(path: str | Path, *, timeout: float, immutable: bool = False) -> sqlite3.Connection:
    """A read-only connection to the existing file at ``path``.

    Raises ``sqlite3.Error`` when the file cannot be opened; a missing file is
    an error, never created. ``immutable`` adds SQLite's ``immutable=1``: no
    lock is taken and no ``-wal`` or ``-shm`` file is created beside the store,
    which a ``mode=ro`` open of a WAL store does when they are missing. It is
    only for a file no connection is writing.
    """
    if immutable:
        return sqlite3.connect(f"file:{quote(os.fsencode(path))}?mode=ro&immutable=1", uri=True, timeout=timeout)
    return sqlite3.connect(f"file:{quote(os.fsencode(path))}?mode=ro", uri=True, timeout=timeout)
