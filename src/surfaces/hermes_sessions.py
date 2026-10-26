from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from ..paths import OmhPaths
from ..quality.hermes_state import hermes_epoch


HERMES_SESSION_SCHEMA_VERSION = "hermes_session_observation/v1"
# An open session (no `ended_at`) counts as live only while Hermes has
# touched it recently. Hermes closes sessions on agent/TUI shutdown and reaps
# orphans on its next startup, so a crash, a killed terminal, or a long-lived
# TUI that never exits leaves rows open for days; on the owner machine 53
# such rows sat "live" for two days with zero activity, and the menu bar
# count never moved. Activity within this window is the honest reading.
LIVE_WINDOW_SECONDS = 900
_CLAIM_BOUNDARY = (
    "Session counts are a read-only observation of Hermes' own session store; "
    "they are not execution, review, CI, merge, or token-usage evidence."
)


def hermes_state_db_path(paths: OmhPaths) -> Path:
    return paths.hermes_home / "state.db"


def hermes_state_watch_paths(paths: OmhPaths) -> tuple[Path, ...]:
    """The files whose stamp moves when Hermes records session activity.

    The reader below opens `state.db`, but Hermes runs it in WAL mode, so an
    ordinary write lands in the `-wal` sidecar and leaves the database file's
    own mtime where it was until a checkpoint. On the owner machine the two
    stamps were four minutes apart while sessions were live, so a watcher
    that stats only the database goes blind for exactly the window it exists
    to cover.
    """
    db_path = hermes_state_db_path(paths)
    return (db_path, db_path.with_name(f"{db_path.name}-wal"))


def observe_hermes_sessions(paths: OmhPaths, *, now: datetime | str | None = None) -> dict[str, Any]:
    db_path = hermes_state_db_path(paths)
    if not db_path.exists():
        return _unobserved("state_db_missing")

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True, timeout=1.0)
        try:
            cursor = connection.cursor()
            total = int(
                cursor.execute(
                    "select count(*) from sessions where archived = 0 and hidden = 0"
                ).fetchone()[0]
            )
            open_rows = cursor.execute(
                "select model, model_config, coalesce(last_activity_at, started_at) from sessions "
                "where archived = 0 and hidden = 0 and ended_at is null"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return _unobserved("state_db_unreadable")

    reference = _coerce_now(now)
    live_rows: list[tuple[float, Any, Any]] = []
    for row in open_rows:
        seen_at = hermes_epoch(row[2])
        if seen_at is None or reference - seen_at > LIVE_WINDOW_SECONDS:
            continue
        live_rows.append((seen_at, row[0], row[1]))
    live_rows.sort(key=lambda entry: entry[0], reverse=True)
    current_row = (live_rows[0][1], live_rows[0][2]) if live_rows else None

    return {
        "schema_version": HERMES_SESSION_SCHEMA_VERSION,
        "observed": True,
        "reason": "",
        "live": len(live_rows),
        "open": len(open_rows),
        "stale": len(open_rows) - len(live_rows),
        "live_window_seconds": LIVE_WINDOW_SECONDS,
        "total": total,
        "current_model": _current_model(current_row),
        "source": "hermes_state_db_readonly",
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _coerce_now(now: datetime | str | None) -> float:
    if isinstance(now, datetime):
        return _aware(now).timestamp()
    if isinstance(now, str) and now.strip():
        parsed = hermes_epoch(now)
        if parsed is not None:
            return parsed
    return datetime.now(timezone.utc).timestamp()


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _current_model(row: tuple[Any, Any] | None) -> dict[str, Any]:
    if row is None:
        return _current_model_payload()

    value = row[0] if isinstance(row[0], str) else ""
    provider = ""
    effort = ""
    try:
        model_config = json.loads(row[1])
    except (TypeError, ValueError, JSONDecodeError):
        model_config = {}

    if isinstance(model_config, dict):
        configured_provider = model_config.get("provider")
        if isinstance(configured_provider, str):
            provider = configured_provider
        reasoning_config = model_config.get("reasoning_config")
        if isinstance(reasoning_config, dict):
            configured_effort = reasoning_config.get("effort")
            if isinstance(configured_effort, str):
                effort = configured_effort

    return _current_model_payload(value=value, effort=effort, provider=provider)


def _current_model_payload(*, value: str = "", effort: str = "", provider: str = "") -> dict[str, Any]:
    return {
        "observed": bool(value),
        "value": value,
        "effort": effort,
        "provider": provider,
        "label": f"{value}:{effort}" if value and effort else value or "not observed",
    }


def _unobserved(reason: str) -> dict[str, Any]:
    return {
        "schema_version": HERMES_SESSION_SCHEMA_VERSION,
        "observed": False,
        "reason": reason,
        "live": 0,
        "open": 0,
        "stale": 0,
        "live_window_seconds": LIVE_WINDOW_SECONDS,
        "total": 0,
        "current_model": _current_model_payload(),
        "source": "hermes_state_db_readonly",
        "claim_boundary": _CLAIM_BOUNDARY,
    }
