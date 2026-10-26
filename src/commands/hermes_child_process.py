"""Process identity checks for Hermes-child cancellation."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys

from ..coding.hermes_child_receipts import JsonValue
from ..core.errors import OmhError


def validate_active_record(active: dict[str, JsonValue], run_id: str) -> None:
    expected_fields = {
        "schema_version",
        "run_id",
        "run_nonce",
        "dispatcher_pid",
        "child_pid",
        "process_identity",
    }
    if set(active) != expected_fields or active.get("schema_version") != "hermes_child_active/v2":
        raise OmhError("Hermes child active record is invalid")
    if active.get("run_id") != run_id:
        raise OmhError("Hermes child active record is invalid")
    for field in ("dispatcher_pid", "child_pid"):
        value = active.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 1:
            raise OmhError("Hermes child active record is invalid")
    nonce = active.get("run_nonce")
    if not isinstance(nonce, str) or len(nonce) != 64:
        raise OmhError("Hermes child active record is invalid")
    try:
        int(nonce, 16)
    except ValueError as exc:
        raise OmhError("Hermes child active record is invalid") from exc
    identity = active.get("process_identity")
    if not isinstance(identity, dict) or set(identity) != {"start_time", "executable"}:
        raise OmhError("Hermes child active record is invalid")
    if not isinstance(identity.get("start_time"), str) or not identity["start_time"]:
        raise OmhError("Hermes child active record is invalid")
    if not isinstance(identity.get("executable"), str) or not identity["executable"]:
        raise OmhError("Hermes child active record is invalid")


def process_identity(pid: int) -> dict[str, str]:
    if sys.platform.startswith("linux"):
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(stat_fields) < 22:
            raise OSError("process metadata is incomplete")
        executable = str(Path(f"/proc/{pid}/exe").resolve(strict=True))
        return {"start_time": stat_fields[21], "executable": executable}
    if sys.platform == "darwin":
        return _darwin_process_identity(pid)
    if sys.platform == "win32":
        return _windows_process_identity(pid)
    raise OSError("process identity verification is unavailable on this platform")


def _darwin_process_identity(pid: int) -> dict[str, str]:
    class ProcBsdInfo(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint32), ("status", ctypes.c_uint32),
            ("xstatus", ctypes.c_int32), ("pid", ctypes.c_int32),
            ("ppid", ctypes.c_int32), ("uid", ctypes.c_uint32),
            ("gid", ctypes.c_uint32), ("ruid", ctypes.c_uint32),
            ("rgid", ctypes.c_uint32), ("svuid", ctypes.c_uint32),
            ("svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
            ("comm", ctypes.c_char * 16), ("name", ctypes.c_char * 32),
            ("nfiles", ctypes.c_uint32), ("pgid", ctypes.c_uint32),
            ("pjobc", ctypes.c_uint32), ("tdev", ctypes.c_int32),
            ("tpgid", ctypes.c_int32), ("nice", ctypes.c_int32),
            ("start_tvsec", ctypes.c_uint64), ("start_tvusec", ctypes.c_uint64),
        ]

    libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    info = ProcBsdInfo()
    if libproc.proc_pidinfo(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
        raise ProcessLookupError(pid)
    path_buffer = ctypes.create_string_buffer(4096)
    if libproc.proc_pidpath(pid, path_buffer, len(path_buffer)) <= 0:
        raise ProcessLookupError(pid)
    executable = os.fsdecode(path_buffer.value)
    return {
        "start_time": f"{info.start_tvsec}.{info.start_tvusec}",
        "executable": str(Path(executable).resolve()),
    }


def _windows_process_identity(pid: int) -> dict[str, str]:
    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        raise ProcessLookupError(pid)
    try:
        creation = FileTime()
        exit_time = FileTime()
        kernel = FileTime()
        user = FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel),
            ctypes.byref(user),
        ):
            raise ProcessLookupError(pid)
        capacity = ctypes.c_uint32(32_768)
        path_buffer = ctypes.create_unicode_buffer(capacity.value)
        if not kernel32.QueryFullProcessImageNameW(
            handle, 0, path_buffer, ctypes.byref(capacity)
        ):
            raise ProcessLookupError(pid)
        start_time = (creation.high << 32) | creation.low
        return {
            "start_time": str(start_time),
            "executable": str(Path(path_buffer.value).resolve()),
        }
    finally:
        kernel32.CloseHandle(handle)
