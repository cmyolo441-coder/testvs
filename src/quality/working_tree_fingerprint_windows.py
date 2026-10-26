"""Windows regular-file observations from one handle API, never mixed CRT stats.

CreateFileW OPEN_REPARSE_POINT refuses final-component traversal. FileIdInfo
provides the same volume/128-bit identity for both metadata and data handles;
FileBasicInfo preserves creation AND change time. No directory-find fallback.
"""

from __future__ import annotations

import ctypes
import errno
import os
import stat as stat_module
from dataclasses import dataclass
from functools import lru_cache


class _BasicInfo(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in (
        "CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime",
    )] + [("FileAttributes", ctypes.c_uint32)]


class _StandardInfo(ctypes.Structure):
    _fields_ = [
        ("AllocationSize", ctypes.c_int64), ("EndOfFile", ctypes.c_int64),
        ("NumberOfLinks", ctypes.c_uint32), ("DeletePending", ctypes.c_ubyte),
        ("Directory", ctypes.c_ubyte),
    ]


class _IdInfo(ctypes.Structure):
    _fields_ = [("VolumeSerialNumber", ctypes.c_uint64), ("FileId", ctypes.c_ubyte * 16)]


@dataclass(frozen=True, slots=True)
class Metadata:
    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int
    st_ctime_ns: int
    creation_ns: int
    attributes: int
    links: int

    @property
    def st_mode(self) -> int:
        # Git for Windows does not synthesize executable-extension bits.
        # Reparse points (including non-symlink types) are not regular files.
        if self.attributes & 0x400:
            return 0
        if self.attributes & 0x10:
            return stat_module.S_IFDIR
        return stat_module.S_IFREG | (0o444 if self.attributes & 1 else 0o666)


@lru_cache(maxsize=1)
def _api():
    # Lazy import/binding: POSIX collection never loads a Windows DLL.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.GetFileInformationByHandleEx.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ]
    kernel.GetFileInformationByHandleEx.restype = ctypes.c_int
    kernel.GetFileType.argtypes = [ctypes.c_void_p]
    kernel.GetFileType.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    return kernel


def _open_handle(path: bytes, access: int) -> int:
    # Metadata probes share all access. The data handle denies write/delete
    # access (including existing writers), since Windows may defer write times
    # until a writer closes. Named FILE_READ_ATTRIBUTES probes remain possible.
    handle = _api().CreateFileW(
        os.fsdecode(path), access, 0x7 if access == 0x80 else 0x1, None, 3,
        0x00200000 | 0x02000000, None,  # OPEN_REPARSE_POINT | BACKUP_SEMANTICS
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def _close_handle(handle: int) -> None:
    if not _api().CloseHandle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _metadata(handle: int) -> Metadata:
    api = _api()
    if api.GetFileType(handle) != 1:  # FILE_TYPE_DISK only
        raise OSError(errno.ENOTSUP, "not a disk file")
    identity, basic, standard = _IdInfo(), _BasicInfo(), _StandardInfo()
    for kind, value in ((18, identity), (0, basic), (1, standard)):
        if not api.GetFileInformationByHandleEx(handle, kind, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())
    inode = int.from_bytes(bytes(identity.FileId), "little")
    if not identity.VolumeSerialNumber or not inode or standard.DeletePending:
        raise OSError(errno.ENOTSUP, "file identity unavailable or deletion pending")
    attributes = basic.FileAttributes
    if bool(attributes & 0x10) != bool(standard.Directory):
        raise OSError(errno.ENOTSUP, "inconsistent file type")
    return Metadata(
        identity.VolumeSerialNumber, inode, standard.EndOfFile,
        basic.LastWriteTime * 100, basic.ChangeTime * 100,
        basic.CreationTime * 100, attributes, standard.NumberOfLinks,
    )


def stat(path: bytes) -> Metadata:
    """Observe a named object without reading bytes or following a reparse point."""
    handle = _open_handle(path, 0x80)  # FILE_READ_ATTRIBUTES
    try:
        return _metadata(handle)
    finally:
        _close_handle(handle)


def open(path: bytes) -> int:
    """Transfer a no-follow read handle to a non-inheritable binary CRT fd."""
    import msvcrt

    handle = _open_handle(path, 0x80000000)  # GENERIC_READ
    owns_handle = True
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_NOINHERIT)
        owns_handle = False
    finally:
        if owns_handle:
            _close_handle(handle)
    try:
        _ = msvcrt.setmode(descriptor, os.O_BINARY)
    except OSError:
        os.close(descriptor)
        raise
    return descriptor


def fstat(descriptor: int) -> Metadata:
    import msvcrt

    return _metadata(msvcrt.get_osfhandle(descriptor))
