"""Reap fanout unit process groups — marker-verified pids only.

`omh coding fanout dispatch` spawns each unit as its own process group and
records the leader pid in the unit's inflight marker. When the dispatcher
dies without running its cleanup (SIGKILL, power loss), those groups can
survive against the worktree with a marker left behind — including the
shape the pi-family runtime shipped as an incident, where the LEADER is already gone but its
grandchildren live on in the group. This reaper therefore judges liveness
at the GROUP level, and terminates ONLY pids an inflight marker names:

- a pid no marker names is refused, whatever its process name says — there
  is deliberately no pattern-matching kill (the pi-family runtime's `doctor --reap` carries
  the same constraint);
- a live leader that is no longer its own group leader is refused as a
  recycled pid;
- markers are cleared only once the whole group is gone.

The reaper does not check that the dispatcher is dead: every marker-named
group is fair game, including a live dispatcher's running units — verify
the dispatcher first. It adds remediation, not a liveness claim; the
marker's presence-is-not-liveness boundary is untouched. Known limit: a
dispatcher killed between spawn and the marker's pid write leaves an
orphan no marker names, which this module refuses by design.
"""

from __future__ import annotations

import os
import signal
import time
from typing import Any

from ..system.paths import OmhPaths
from ._hermes_child_process import group_absent
from .inflight import clear_inflight_marker, read_inflight_markers

REAP_SCHEMA_VERSION = "fanout_reap_report/v1"
REAP_GRACE_SECONDS = 10.0

REAP_CLAIM_BOUNDARY = (
    "A reap report records signals sent to marker-named process groups only. "
    "It is not execution, verification, or recovery evidence, and it never "
    "kills by process name."
)


def _leader_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _leader_recycled(pid: int) -> bool:
    """A live process that no longer leads its own group is a recycled pid."""
    if not _leader_alive(pid):
        return False
    try:
        return os.getpgid(pid) != pid
    except OSError:
        # The process vanished between the checks; not recycled, just gone.
        return False


def _terminate_group(pid: int, grace: float) -> list[str]:
    """SIGTERM then SIGKILL the group, polling GROUP absence between."""
    sent: list[str] = []
    for signum, label in ((signal.SIGTERM, "SIGTERM"), (getattr(signal, "SIGKILL", signal.SIGTERM), "SIGKILL")):
        try:
            os.killpg(pid, signum)
            sent.append(label)
        except PermissionError:
            sent.append(f"{label}:refused_permission")
            return sent
        except OSError:
            break
        deadline = time.monotonic() + max(0.1, grace)
        while time.monotonic() < deadline:
            if group_absent(pid):
                return sent
            time.sleep(0.05)
        if group_absent(pid):
            break
    return sent


def reap_fanout_units(
    paths: OmhPaths,
    fanout_id: str,
    *,
    pids: list[int] | None = None,
    grace: float = REAP_GRACE_SECONDS,
) -> dict[str, Any]:
    """Terminate marker-named unit groups for one fanout; refuse the rest."""
    report: dict[str, Any] = {
        "schema_version": REAP_SCHEMA_VERSION,
        "fanout_id": fanout_id,
        "candidates": [],
        "claim_boundary": REAP_CLAIM_BOUNDARY,
    }
    if os.name == "nt":
        # POSIX process groups are the verification mechanism; without them
        # the leader re-check cannot hold, so the reaper refuses outright.
        report["status"] = "unsupported_platform"
        return report
    marked: dict[int, dict[str, Any]] = {}
    for entry in read_inflight_markers(paths, limit=200, fanout_id=fanout_id):
        raw_pid = str(entry.get("pid", "") or "")
        if raw_pid.isdigit():
            marked[int(raw_pid)] = entry
    requested = pids if pids is not None else sorted(marked)
    for pid in requested:
        row: dict[str, Any] = {"pid": pid}
        entry = marked.get(pid)
        if entry is None:
            # The refusal, not the kill, is the point: a pid outside the
            # markers is somebody else's process however promising its name.
            row["status"] = "refused_not_marker_named"
        elif group_absent(pid):
            # Group-level absence, not leader absence: a dead leader with
            # live grandchildren keeps the group present and stays reapable.
            row["status"] = "already_gone"
            row["unit_id"] = entry.get("unit_id", "")
            clear_inflight_marker(paths, fanout_id, str(entry.get("unit_id", "")))
        elif _leader_recycled(pid):
            # Alive but led by someone else means the pid was recycled by an
            # unrelated process; killing its group would be a name-free
            # version of the pattern kill this module refuses.
            row["status"] = "refused_not_group_leader"
            row["unit_id"] = entry.get("unit_id", "")
        else:
            signals_sent = _terminate_group(pid, grace)
            row["signals_sent"] = signals_sent
            row["unit_id"] = entry.get("unit_id", "")
            if any(item.endswith("refused_permission") for item in signals_sent):
                row["status"] = "refused_permission"
            elif group_absent(pid):
                row["status"] = "reaped"
                clear_inflight_marker(paths, fanout_id, str(entry.get("unit_id", "")))
            else:
                row["status"] = "still_alive"
        report["candidates"].append(row)
    report["status"] = "observed"
    return report
