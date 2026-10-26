"""Primitives for reading Hermes' own ``state.db``: the read-only open, the
``sessions.source`` filter, and the stamp parser.

Hermes keeps sessions and messages in ``<hermes_home>/state.db``; the OMH
readers observe it and none writes. The reply lint and the session-usage
report in this package open it through ``open_state_db_readonly``, so both
fail the same way on a missing or unreadable database. The menubar session
reader in ``surfaces.hermes_sessions`` keeps its own open (it reports
``unobserved`` rather than raising) but parses the same stamp column through
``hermes_epoch``, so there is one parser to change when Hermes changes the
stamp shape.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any
from urllib.parse import quote

from ..plugin_bundle.omh.memory_session_digest import stamp_epoch


NO_SOURCE_LABEL = "(none)"
HERMES_LATEST_SESSION = "latest"
# A compaction re-persists tool rows under new ids, so a tool call is keyed by
# its tool_call_id. An empty tool_call_id is not NULL, so without NULLIF every
# such row in a session would share one key and distinct calls would collapse
# into one; a row without one is keyed by its own id.
TOOL_CALL_KEY_SQL = "COALESCE(NULLIF(tool_call_id, ''), 'row:' || id)"


def open_state_db_readonly(
    hermes_home: str | Path, *, error: type[ValueError]
) -> tuple[Path, sqlite3.Connection]:
    """Open ``<hermes_home>/state.db`` with ``mode=ro`` and return its path and connection.

    A missing file or a refused open raises ``error`` with the path in the
    message; the caller closes the connection.
    """
    path = Path(hermes_home).expanduser() / "state.db"
    if not path.exists():
        raise error(f"no Hermes state database at {path}")
    try:
        connection = sqlite3.connect(f"file:{quote(str(path))}?mode=ro", uri=True, timeout=0.5)
    except sqlite3.Error as exc:
        raise error(f"could not open {path}: {exc}") from exc
    return path, connection


def session_source_clause(source: str) -> tuple[str, tuple[str, ...]]:
    """The ``WHERE`` predicate for a ``sessions.source`` filter and its parameters.

    ``NO_SOURCE_LABEL`` is what the readers print for a row without a tag, so
    as a filter it selects those rows; any other value must equal the tag.
    """
    if source == NO_SOURCE_LABEL:
        return "(source IS NULL OR source = '')", ()
    return "source = ?", (source,)


def hermes_epoch(value: Any) -> float | None:
    """Epoch seconds from a Hermes stamp, or ``None`` when it is not one.

    Hermes stores REAL epoch seconds; older rows and fixtures carry ISO-8601
    text. A non-finite or unrepresentable number is not a stamp in either
    form: ``nan`` text would otherwise disable a ``--since`` window, and a
    400-digit integer made ``math.isfinite`` raise. One parser for the
    reports and the memory provider's digest, which runs in the Hermes process
    and cannot import this package, so it lives in the bundle.
    """
    return stamp_epoch(value)


def resolve_session_id(
    connection: sqlite3.Connection, session_id: str, source: str | None, *, error: type[ValueError]
) -> str:
    """The session a ``--hermes-session`` value names, raising ``error`` when none does.

    ``HERMES_LATEST_SESSION`` resolves to the most recently active session row
    -- the most recent row whose ``source`` tag equals ``source`` when one is
    given (``NO_SOURCE_LABEL`` selects untagged rows). An explicit id is
    returned as given unless ``source`` is set, in which case its row must
    exist and carry that tag, so a filter never looks applied when it was not.
    """
    wanted = str(session_id or "").strip()
    if not wanted:
        raise error("a session id (or `latest`) is required")
    if wanted != HERMES_LATEST_SESSION:
        if source is not None:
            row = connection.execute("SELECT source FROM sessions WHERE id = ?", (wanted,)).fetchone()
            if row is None:
                raise error(f"no Hermes session {wanted} to check --source {source} against")
            actual = row[0] or NO_SOURCE_LABEL
            if actual != source:
                raise error(f"session {wanted} has source {actual}, not {source}")
        return wanted
    order = "ORDER BY COALESCE(last_activity_at, started_at) DESC, id DESC LIMIT 1"
    if source is None:
        row = connection.execute(f"SELECT id FROM sessions {order}").fetchone()
        if row is None:
            raise error("the Hermes state database has no sessions")
        return str(row[0])
    clause, params = session_source_clause(source)
    row = connection.execute(f"SELECT id FROM sessions WHERE {clause} {order}", params).fetchone()
    if row is None:
        raise error(f"no Hermes session with source {source}")
    return str(row[0])
