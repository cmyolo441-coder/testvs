"""Is the Hermes process a helper ran in still there? Hermes' own probes, imported when called.

A helper `delegate_task` starts runs inside the Hermes process that started
it, and a restart ends it without a `subagent_stop`: Hermes fires none for a
child a restart killed; its own recovery of such work is
`tools/async_delegation.py:recover_abandoned_delegations`. So the process is
recorded when the helper starts (`this_process`), and a later process asks
whether that one is still running (`process_alive`).

Both use the probes in Hermes' `gateway.status` that its own recovery uses
(`_owner_liveness` beside it): `_pid_exists`, which never signals the
process (`os.kill(pid, 0)` ends it on Windows), and the start-time
fingerprint with its drift tolerance, so a pid the OS gave to another process
is not read as the recorded one. The import is at call time and guarded
(#1623): outside Hermes, or on a Hermes without these probes, nothing is
recorded and every answer is unknown, which leaves each caller's own time
bound in charge. The guard catches any failure, as Hermes' own call sites
do, so a Hermes module that fails to load reads the same as no Hermes.

A pid means something only on the machine, and in the pid namespace, that
read it: a Hermes gateway in a container and a CLI on its host can share one
OMH home, and the host has no such pid. So the record also names where the
pid was read (`machine`), and a record from anywhere else is unknown, never
gone. Two machines must never read as one: a pid the other machine runs
does not exist here, so it would read as gone, and its work would run and
be paid for twice. On Linux the identity includes the boot, so a reboot,
`wsl --shutdown`, or a restarted or recreated container reads as unknown
too: "same machine" cannot be proven across a boot (machine ids are baked
into images, and every host's own pid namespace is the same constant).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Final
import uuid

# Linux: a random id per boot, and this process's pid namespace (a container has its own).
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
_PID_NAMESPACE = "/proc/self/ns/pid"
_PLATFORM: Final = sys.platform
_MACHINE_CHARS = 300
# `gethostuuid` waits this long at most; a zero wait would mean "wait indefinitely" (gethostuuid(2)).
_UUID_WAIT_SECONDS = 1


def this_process() -> dict[str, Any] | None:
    """``{"pid", "started", "machine"}`` for this process, or None when Hermes' probes are not here or `machine`
    cannot be read.

    ``started`` is Hermes' start-time fingerprint, None when it could not be
    read; ``machine`` is where the pid was read (`machine`).
    """
    try:
        from gateway.status import get_process_start_time
    except Exception:
        # Not Hermes (ImportError), or a Hermes whose module fails to load: nothing is recorded either way,
        # and the `subagent_start` observers calling this never raise.
        return None
    where = machine()
    if not where:
        return None
    pid = os.getpid()
    try:
        started = get_process_start_time(pid)
    except Exception:
        # Hermes' reader answers None itself for every failure it knows; any other is the same unread time.
        started = None
    return {"pid": pid, "started": started if _is_int(started) else None, "machine": where}


def machine() -> str:
    """Where a pid of this process means what it says.

    On Linux, the boot id and the pid namespace: another host, and another
    boot of this one, has another boot id, and a container has its own
    namespace. On macOS the hardware UUID (`gethostuuid`), on Windows the
    `MachineGuid` the OS keeps in the registry: neither follows the network,
    as a host name does (macOS renames itself on a DHCP or reverse-DNS
    change, or on a Bonjour conflict), and two machines may share a name.
    "" when that cannot be read, and on any other platform: nothing is
    recorded then, and no record reads as here, so the caller's own bound
    holds.
    """
    if not _PLATFORM.startswith("linux"):
        hardware = _darwin_uuid() if _PLATFORM == "darwin" else _windows_guid() if _PLATFORM == "win32" else ""
        return f"{_PLATFORM}:{hardware}"[:_MACHINE_CHARS] if hardware else ""
    try:
        boot = _BOOT_ID.read_text(encoding="ascii").strip()
        namespace = os.readlink(_PID_NAMESPACE)
    except (OSError, ValueError):
        return ""
    return f"linux:{boot}:{namespace}"[:_MACHINE_CHARS] if boot and namespace else ""


def _darwin_uuid() -> str:
    """This Mac's hardware UUID from libc's `gethostuuid`, or "" when it cannot be read. No process is spawned."""
    try:
        import ctypes

        class _Timespec(ctypes.Structure):
            _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

        found = (ctypes.c_ubyte * 16)()
        wait = _Timespec(_UUID_WAIT_SECONDS, 0)
        if ctypes.CDLL(None).gethostuuid(found, ctypes.byref(wait)) != 0:
            return ""
        return str(uuid.UUID(bytes=bytes(found)))
    except (ImportError, OSError, AttributeError, ValueError):
        return ""


def _windows_guid() -> str:
    """Windows' `MachineGuid` (HKLM\\SOFTWARE\\Microsoft\\Cryptography), or "" when it cannot be read."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography", 0,
                            winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            value, kind = winreg.QueryValueEx(key, "MachineGuid")
    except (ImportError, OSError):
        return ""
    return value.strip() if kind == winreg.REG_SZ and isinstance(value, str) else ""


def recorded_process(value: object) -> dict[str, Any] | None:
    """A process in the shape `this_process` writes, or None."""
    if not isinstance(value, dict) or not _is_int(value.get("pid")) or value["pid"] <= 0:
        return None
    started, where = value.get("started"), value.get("machine")
    if started is not None and not _is_int(started):
        return None
    if not isinstance(where, str) or not where or len(where) > _MACHINE_CHARS:
        return None
    return {"pid": value["pid"], "started": started, "machine": where}


def process_alive(value: object) -> bool | None:
    """True when the recorded process still runs, False when it is gone, None when that cannot be told.

    None: nothing recorded, a pid read on another machine or in another pid
    namespace, no Hermes probes, a pid that exists with no start time to
    compare it by, or a probe that raised. Only False says the process
    ended, so only False may hand its helper's work out again.
    """
    recorded = recorded_process(value)
    if recorded is None or recorded["machine"] != machine():
        return None
    try:
        from gateway.status import _pid_exists, get_process_start_time, start_time_fingerprints_match
    except Exception:
        # Hermes' own call sites guard this import the same way (`_owner_liveness`, `_persist_dispatch`).
        return None
    try:
        if not _pid_exists(recorded["pid"]):
            return False
        current = get_process_start_time(recorded["pid"])
        if recorded["started"] is None or not _is_int(current):
            return None
        # A pid that exists under another start time is another process: the recorded one ended.
        return bool(start_time_fingerprints_match(recorded["started"], current))
    except Exception:
        # A probe that raised has not said the process ended.
        return None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
