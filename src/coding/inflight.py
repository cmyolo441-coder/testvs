"""Cross-session in-flight unit markers (`omh_inflight_marker/v1`).

Local dispatch of a fanout unit is blocking: `_dispatch_unit` calls the runner
and waits, so the dispatching process cannot report on itself and no second
session can see that a unit is currently occupying a worktree. A marker file
written before the spawn and removed after it is the whole mechanism — one
small JSON object per in-flight unit, under the fanout directory that already
holds that fanout's contract and dispatch summary.

Boundaries, in order of importance:

- Presence is not liveness. A marker only proves that some process wrote it
  and has not removed it. A crashed or killed dispatcher leaves the marker
  behind, and nothing here can tell "still running" from "died holding the
  file". `read_inflight_markers` therefore reports `started_at` and a fixed
  `liveness: "unknown"`, and never emits a "stale" or "running" verdict — age
  is the caller's arithmetic against its own threshold, not this module's
  guess.
- Never raise at the clear or read boundary. `clear_inflight_marker` is meant
  for a `finally:` block where raising would mask the real dispatch failure,
  so it swallows a missing file, a bad id, and any OS error and returns False.
  `read_inflight_markers` skips what it cannot parse instead of failing the
  whole listing.
- Metadata only, flat and scalar. Every persisted value is a string: owner,
  host, routed model and effort, run ref, worktree path, timestamps. No
  prompts, no output, no nested objects, no floats — the setup-profile
  correct/restore path drops non-scalar nested values.
- Reuse the existing fanout directory convention. Marker storage hangs off
  `fanout_artifacts._managed_fanout_dir`, the same containment-validated
  helper that resolves `paths.fanout_contracts_dir / <fanout_id>` for the
  contract and dispatch summary. No parallel layout, no second id validator.
- Reporting only. An in-flight marker is dispatch bookkeeping. It is not
  result, review, CI, merge-readiness, or merge evidence.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Final, Mapping

from ..system.local_store import atomic_write_json, ensure_dir, read_json_object_result, utc_now
from ..plugin_bundle.omh.run_records import INFLIGHT_MARKER_SCHEMA_VERSION
from ..system.paths import OmhPaths
from .fanout_artifacts import _managed_fanout_dir
from .fanout_contracts import FANOUT_ID_PATTERN

# Held at v1 through the 2026-09 progress-evidence fields below on purpose.
# Every reader of this file (`read_inflight_markers` here, `_merge_inflight` in
# `plugin_bundle/omh/status_board_reader.py`) compares the version for EXACT
# equality and counts a mismatch as `unreadable`, and the plugin copy on a live
# machine is refreshed by `omh update` rather than in lockstep with core. A bump
# for a purely additive field set would therefore blank the running-work board
# on every machine whose plugin lags its core by one update, which is a worse
# outcome than an older reader ignoring keys it does not know. Bump when a
# field's MEANING changes; not when one is added. The version itself is
# `INFLIGHT_MARKER_SCHEMA_VERSION` in `plugin_bundle/omh/run_records.py`.

INFLIGHT_CLAIM_BOUNDARY: Final[str] = (
    "An in-flight marker records that a dispatch wrote a file before spawning a unit and has not "
    "removed it. Presence is not liveness: a crashed dispatcher leaves the marker behind, so this "
    "is reporting-only dispatch bookkeeping, never result, review, CI, merge-readiness, or merge "
    "evidence."
)

# Directory name under the fanout's own directory. Sibling of
# `fanout_contract.json` and `dispatch_summary.json`.
INFLIGHT_DIR_NAME: Final[str] = "inflight"

# Same unit-id shape the fanout contract builder enforces
# (`_UNIT_ID_RE` in `fanout.py`). Repeated rather than imported because it is
# a filename guard here: a unit id becomes a path segment.
UNIT_ID_PATTERN: Final[str] = r"^[a-z0-9][a-z0-9-]{0,63}$"

# Statuses a listed marker can carry. Same vocabulary as the model inventory's
# source statuses: the file was read, the file was not there, or the file was
# there and could not be turned into a marker object.
INFLIGHT_MARKER_STATUSES: Final[tuple[str, ...]] = ("present", "absent", "unreadable")

# Every accepted `fields` key. Each one is persisted as a string, and each one
# is always present in the written payload so the shape is stable.
INFLIGHT_MARKER_FIELDS: Final[tuple[str, ...]] = (
    "owner",
    "owner_host",
    "model",
    "reasoning_effort",
    "run_ref",
    "worktree",
    "started_at",
    # Group-leader pid of the spawned unit, recorded by the signal-safe
    # runner. It does not upgrade the liveness claim — presence is still not
    # liveness — but it is the only pid the fanout reaper will ever touch.
    "pid",
    # What `unit_progress.assess_progress` made of the unit's stdout at the
    # last mid-run snapshot. This is the only field group on the marker that
    # says anything about the WORK; everything above it is dispatch
    # bookkeeping. Unset means no snapshot has been assessed yet — never that
    # the work is moving. `unit_state` carries a value of
    # `unit_execution_state.UNIT_EXECUTION_STATES`; `stalled_for_seconds` is
    # the derived elapsed since output last grew, recorded instead of the
    # assessment's own readings because those are monotonic and mean nothing
    # in a reading process; `phase_markers` is comma-joined because every
    # marker value is a scalar string.
    "unit_state",
    "state_reason",
    "last_new_output_at",
    "stalled_for_seconds",
    "repeat_count",
    "phase_markers",
)

DEFAULT_INFLIGHT_READ_LIMIT: Final[int] = 20

_FANOUT_ID_RE = re.compile(FANOUT_ID_PATTERN)
_UNIT_ID_RE = re.compile(UNIT_ID_PATTERN)


class InflightMarkerError(ValueError):
    """Rejected marker input: bad id, unknown field, or non-scalar value."""


def write_inflight_marker(paths: OmhPaths, fanout_id: str, unit_id: str, fields: Mapping[str, Any]) -> Path:
    """Write the marker for one unit atomically and return its path.

    Called immediately before the blocking spawn. Raises
    `InflightMarkerError` on a bad id, an unknown field name, or a non-scalar
    value — all caller-programming errors that must surface before the unit
    starts, never from a `finally:` block.
    """
    marker_path = _marker_path(paths, fanout_id, unit_id)
    payload: dict[str, Any] = {
        "schema_version": INFLIGHT_MARKER_SCHEMA_VERSION,
        "claim_boundary": INFLIGHT_CLAIM_BOUNDARY,
        "fanout_id": fanout_id,
        "unit_id": unit_id,
        "liveness": "unknown",
        "written_at": utc_now(),
    }
    payload.update(_normalized_fields(fields))
    if not payload["started_at"]:
        payload["started_at"] = payload["written_at"]
    ensure_dir(marker_path.parent, private=True)
    atomic_write_json(marker_path, payload, private=True)
    return marker_path


def clear_inflight_marker(paths: OmhPaths, fanout_id: str, unit_id: str) -> bool:
    """Remove the marker for one unit. Returns True only if a file was removed.

    Idempotent and total: this runs in the `finally:` of a dispatch, where any
    raised exception would replace the real dispatch failure. A missing file, a
    rejected id, and an OS error all return False.
    """
    try:
        marker_path = _marker_path(paths, fanout_id, unit_id)
    except (OSError, ValueError):
        return False
    try:
        marker_path.unlink()
    except (OSError, ValueError):
        return False
    return True


def read_inflight_markers(
    paths: OmhPaths, *, limit: int = DEFAULT_INFLIGHT_READ_LIMIT, fanout_id: str = ""
) -> list[dict[str, Any]]:
    """List markers across every fanout, newest-sorting last, never raising.

    Ordering is `(started_at, unit_id, fanout_id)` so repeated reads of the
    same directory produce the same list. Entries carry `marker_status` from
    `INFLIGHT_MARKER_STATUSES` — a marker that vanished between listing and
    reading is `absent`, one that could not be parsed is `unreadable`, and
    neither reports a path or an error string. No entry claims the unit is
    alive: compare `started_at` against your own threshold. `fanout_id`
    filters BEFORE the limit slice — a reader asking about one fanout must
    never lose its markers to older fanouts' backlog.
    """
    if limit <= 0:
        return []
    entries: list[dict[str, Any]] = []
    for marker_path, marker_fanout_id, unit_id in _marker_files(paths):
        if fanout_id and marker_fanout_id != fanout_id:
            continue
        entries.append(_entry_for(marker_path, marker_fanout_id, unit_id))
    entries.sort(key=lambda entry: (entry["started_at"], entry["unit_id"], entry["fanout_id"]))
    return entries[:limit]


def _marker_files(paths: OmhPaths) -> list[tuple[Path, str, str]]:
    """Every `<fanout>/inflight/<unit>.json` under the fanout contracts root."""
    root = paths.fanout_contracts_dir
    found: list[tuple[Path, str, str]] = []
    for fanout_dir in _sorted_children(root):
        fanout_id = fanout_dir.name
        if not _FANOUT_ID_RE.match(fanout_id):
            continue
        for marker_path in _sorted_children(fanout_dir / INFLIGHT_DIR_NAME):
            if marker_path.suffix != ".json":
                continue
            unit_id = marker_path.stem
            if not _UNIT_ID_RE.match(unit_id):
                continue
            found.append((marker_path, fanout_id, unit_id))
    return found


def _sorted_children(directory: Path) -> list[Path]:
    try:
        return sorted(directory.iterdir())
    except OSError:
        return []


def _entry_for(marker_path: Path, fanout_id: str, unit_id: str) -> dict[str, Any]:
    entry = _empty_entry(fanout_id, unit_id)
    payload, error = read_json_object_result(marker_path)
    if error is not None:
        entry["marker_status"] = "unreadable"
        return entry
    if payload is None:
        entry["marker_status"] = "absent"
        return entry
    if str(payload.get("schema_version", "")) != INFLIGHT_MARKER_SCHEMA_VERSION:
        entry["marker_status"] = "unreadable"
        return entry
    entry["marker_status"] = "present"
    entry["written_at"] = _scalar_text(payload.get("written_at"))
    for name in INFLIGHT_MARKER_FIELDS:
        entry[name] = _scalar_text(payload.get(name))
    return entry


def _empty_entry(fanout_id: str, unit_id: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "schema_version": INFLIGHT_MARKER_SCHEMA_VERSION,
        "claim_boundary": INFLIGHT_CLAIM_BOUNDARY,
        "fanout_id": fanout_id,
        "unit_id": unit_id,
        "marker_status": "absent",
        "liveness": "unknown",
        "written_at": "",
    }
    for name in INFLIGHT_MARKER_FIELDS:
        entry[name] = ""
    return entry


def _normalized_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    unknown = sorted(str(name) for name in fields if str(name) not in INFLIGHT_MARKER_FIELDS)
    if unknown:
        raise InflightMarkerError(f"unknown in-flight marker field(s): {', '.join(unknown)}")
    normalized: dict[str, str] = {}
    for name in INFLIGHT_MARKER_FIELDS:
        value = fields.get(name, "")
        if isinstance(value, (dict, list, tuple, set, float)):
            raise InflightMarkerError(f"in-flight marker field {name!r} must be a scalar string")
        normalized[name] = _scalar_text(value)
    return normalized


def _scalar_text(value: Any) -> str:
    if value is None or isinstance(value, (dict, list, tuple, set, float)):
        return ""
    return str(value)


def _marker_path(paths: OmhPaths, fanout_id: str, unit_id: str) -> Path:
    if not _FANOUT_ID_RE.match(str(fanout_id)):
        raise InflightMarkerError(f"invalid fanout_id: {fanout_id!r}")
    if not _UNIT_ID_RE.match(str(unit_id)):
        raise InflightMarkerError(f"invalid unit_id: {unit_id!r}")
    return _managed_fanout_dir(paths, fanout_id) / INFLIGHT_DIR_NAME / f"{unit_id}.json"
