"""What Hermes' per-plugin load deadline left on disk for one plugin.

Hermes times the load of every plugin -- importing it, then calling its
``register()`` -- against ``plugins.load_timeout_seconds``: 10 seconds by
default, ``0`` for no limit, at most 600. Past the limit Hermes drops that
plugin and carries on with the others. The mechanism is
``run_with_load_deadline`` in Hermes' ``hermes_cli/plugins_loader.py`` (Hermes
0.21.5 and later, and canary builds from 2026-09-25); what follows was measured
against that code in an isolated ``HERMES_HOME`` rather than read off it:

- The skipped plugin is recorded in the running process only -- ``/plugins``
  and ``list_plugins()`` carry ``load timed out after <N>s (import +
  register() never returned)``. Nothing outside the process can read that.
- The one trace that outlives the process is a log line. Hermes logs
  ``Failed to load plugin '<name>': <that error>`` at WARNING, and its
  ``logs/errors.log`` keeps WARNING and above for the home that was loading.
- The load is not cancelled. It runs on a worker thread Hermes abandons, so
  the plugin's modules stay in ``sys.modules`` (some half-initialized until
  the worker finishes) and every ``register_*`` call the worker still makes
  is refused with its own WARNING. Those later lines are a consequence of the
  skip, not a second record of it, and are not counted here.

The deadline skips a plugin a second way, read from Hermes' source and not
replayed: once eight abandoned workers are still running, Hermes refuses every
further load with ``not loaded: 8 abandoned plugin loader thread(s) are still
running (plugins.load_timeout_seconds); restart Hermes to retry``, through the
same ``Failed to load plugin`` line. That plugin never ran, so it is the same
missing tools with a different cause, and it is counted with its cause named.

So this reads Hermes' error log for those two lines, and answers three ways. A
record is a past load: it says nothing about the load a running Hermes did
last. No record says only what the files named in ``files_read`` hold -- at
most ``READ_LIMIT_BYTES`` from the end of each, ``truncated_files`` naming the
ones cut short -- and a process that never set up file logging writes none, so
it is not evidence that the plugin loaded. No log is ``observed: False``.
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
from typing import Any

RECORD_SCHEMA_VERSION = "plugin_load_deadline_record/v1"

LOAD_TIMEOUT_RECORDED = "load_timeout_recorded"
NO_LOAD_TIMEOUT_RECORDED = "no_load_timeout_recorded"
ERROR_LOG_ABSENT = "hermes_error_log_absent"
ERROR_LOG_UNREADABLE = "hermes_error_log_unreadable"

# Why Hermes did not load the plugin, for the latest record.
LOAD_TIMED_OUT = "load_timed_out"
ABANDONED_LOADER_CAP = "abandoned_loader_cap"

# How much of each file is read, from its end. Hermes rotates errors.log at
# 2 MiB, but its Windows fallback handler never rolls over, so the file can be
# any size; the newest lines are the ones this check is asked about.
READ_LIMIT_BYTES = 4 * 1024 * 1024

# The current log and the two backups Hermes' rotation keeps, newest first.
_LOG_NAMES = ("errors.log", "errors.log.1", "errors.log.2")
_ABSENT = "absent"
_FIELD_LIMIT = 40


def load_deadline_skip_cause(line: str, plugin_name: str) -> str:
    """Which deadline-caused skip of `plugin_name` this log line records, or "".

    The quoted name and the reason are matched together: another plugin's
    timeout, this plugin's other load failures, and the refused late
    registrations all lack one or the other.
    """
    _, found, reason = line.partition(f"Failed to load plugin '{plugin_name}': ")
    if not found:
        return ""
    if reason.startswith("load timed out after "):
        return LOAD_TIMED_OUT
    if reason.startswith("not loaded: ") and "abandoned plugin loader thread(s)" in reason:
        return ABANDONED_LOADER_CAP
    return ""


def read_recorded_load_timeouts(hermes_home: Path, plugin_name: str) -> dict[str, Any]:
    """Whether `hermes_home`'s retained error logs record a deadline skip of `plugin_name`."""
    logs_dir = hermes_home / "logs"
    log_path = logs_dir / _LOG_NAMES[0]
    files_read: list[str] = []
    truncated_files: list[str] = []
    count = 0
    latest = ""
    for name in _LOG_NAMES:
        text, truncated, why_not = _read_log_tail(logs_dir / name)
        if text is None:
            if name != _LOG_NAMES[0]:
                continue  # a backup that is absent or unreadable is simply not in `files_read`
            if why_not == _ABSENT:
                return _record(log_path, observed=False, reason=ERROR_LOG_ABSENT)
            return _record(log_path, observed=False, reason=ERROR_LOG_UNREADABLE, detail=why_not)
        files_read.append(name)
        if truncated:
            truncated_files.append(name)
        matches = [line for line in text.splitlines() if load_deadline_skip_cause(line, plugin_name)]
        count += len(matches)
        if matches and not latest:
            latest = matches[-1]  # files run newest first, lines oldest first
    if not latest:
        return _record(
            log_path,
            observed=True,
            reason=NO_LOAD_TIMEOUT_RECORDED,
            files_read=files_read,
            truncated_files=truncated_files,
        )
    cause = load_deadline_skip_cause(latest, plugin_name)
    deadline = latest.partition("load timed out after ")[2].split(" ", 1)[0] if cause == LOAD_TIMED_OUT else ""
    return _record(
        log_path,
        observed=True,
        reason=LOAD_TIMEOUT_RECORDED,
        files_read=files_read,
        truncated_files=truncated_files,
        count=count,
        # Hermes' line opens with its own local-time stamp; it is carried as
        # written, never converted, because the log does not name a zone.
        latest_at=latest.partition(" WARNING")[0].strip()[:_FIELD_LIMIT],
        latest_cause=cause,
        latest_deadline=deadline[:_FIELD_LIMIT],
    )


def _read_log_tail(path: Path) -> tuple[str | None, bool, str]:
    """`(text, truncated, why_not)`; `text` is None when the file was not read."""
    try:
        if not stat.S_ISREG(path.stat().st_mode):
            # Opening a FIFO blocks until something writes to it, and a device
            # never ends: neither is a log, and neither may hang `omh doctor`.
            return None, False, "not_a_regular_file"
        with path.open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - READ_LIMIT_BYTES))
            data = handle.read(READ_LIMIT_BYTES)
    except FileNotFoundError:
        return None, False, _ABSENT
    except OSError as exc:
        return None, False, type(exc).__name__
    return data.decode("utf-8", errors="replace"), size > READ_LIMIT_BYTES, ""


def _record(
    log_path: Path,
    *,
    observed: bool,
    reason: str,
    detail: str = "",
    files_read: list[str] | None = None,
    truncated_files: list[str] | None = None,
    count: int = 0,
    latest_at: str = "",
    latest_cause: str = "",
    latest_deadline: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "observed": observed,
        "reason": reason,
        "detail": detail,
        "log_path": str(log_path),
        "files_read": list(files_read or []),
        "truncated_files": list(truncated_files or []),
        "truncated": bool(truncated_files),
        "read_limit_bytes": READ_LIMIT_BYTES,
        "count": count,
        "latest_at": latest_at,
        "latest_cause": latest_cause,
        "latest_deadline": latest_deadline,
    }


__all__ = [
    "ABANDONED_LOADER_CAP",
    "ERROR_LOG_ABSENT",
    "ERROR_LOG_UNREADABLE",
    "LOAD_TIMED_OUT",
    "LOAD_TIMEOUT_RECORDED",
    "NO_LOAD_TIMEOUT_RECORDED",
    "READ_LIMIT_BYTES",
    "load_deadline_skip_cause",
    "read_recorded_load_timeouts",
]
