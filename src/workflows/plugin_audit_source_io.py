from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from typing import Final

from ..system.metadata_safety import is_sensitive_metadata_text


MAX_PLUGIN_AUDIT_DEPTH: Final = 8
MAX_PLUGIN_AUDIT_DIRECTORIES: Final = 128
MAX_PLUGIN_AUDIT_FILES: Final = 128
MAX_PLUGIN_AUDIT_DIRECTORY_ENTRIES: Final = 128
MAX_PLUGIN_AUDIT_FILE_BYTES: Final = 131_072
_NOFOLLOW_FLAG: Final = getattr(os, "O_NOFOLLOW", None)
_DIRECTORY_FLAG: Final = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK_FLAG: Final = getattr(os, "O_NONBLOCK", 0)
_IGNORED_DIRECTORIES: Final = frozenset({".git", ".venv", "__pycache__", "node_modules"})
_AUDITED_FILENAMES: Final = frozenset({".env", "plugin.json", "pyproject.toml", "requirements.txt", "setup.cfg", "setup.py"})
_AUDITED_SUFFIXES: Final = frozenset(
    {".cjs", ".cts", ".js", ".json", ".mjs", ".mts", ".py", ".toml", ".ts", ".txt", ".yaml", ".yml"}
)


class PluginAuditSourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PluginAuditSource:
    name: str
    content: bytes
    is_root_file: bool


@dataclass(frozen=True, slots=True)
class _ScanState:
    sources: tuple[PluginAuditSource, ...]
    directory_count: int


def resolve_plugin_audit_root(plugin_root: Path) -> Path:
    provided_root = Path(os.path.abspath(plugin_root.expanduser()))
    if is_sensitive_metadata_text(str(provided_root)):
        raise PluginAuditSourceError("plugin audit root path must be safe metadata")
    if any(component.is_symlink() for component in (provided_root, *provided_root.parents)):
        raise PluginAuditSourceError("plugin audit root path must not contain a symlink")
    if provided_root == Path(provided_root.anchor):
        raise PluginAuditSourceError("plugin audit root must not be a filesystem root")
    if not provided_root.is_dir():
        raise PluginAuditSourceError("plugin audit root must be a directory")
    return provided_root


def read_static_plugin_sources(root: Path) -> tuple[PluginAuditSource, ...]:
    descriptor = _open_directory_without_symlinks(root)
    try:
        return _scan_directory(descriptor, 0, _ScanState((), 0)).sources
    finally:
        os.close(descriptor)


def _open_directory_without_symlinks(root: Path) -> int:
    nofollow = _NOFOLLOW_FLAG
    if nofollow is None:
        raise PluginAuditSourceError("secure plugin audit directory access is unavailable")
    descriptor = os.open(root.anchor, os.O_RDONLY | _DIRECTORY_FLAG)
    try:
        for part in root.parts[1:]:
            next_descriptor = os.open(
                part,
                os.O_RDONLY | _DIRECTORY_FLAG | _NONBLOCK_FLAG | nofollow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PluginAuditSourceError("plugin audit root must be a directory")
        return descriptor
    except (OSError, PluginAuditSourceError) as exc:
        os.close(descriptor)
        if isinstance(exc, PluginAuditSourceError):
            raise
        raise PluginAuditSourceError("plugin audit root cannot be safely opened") from exc


def _scan_directory(directory_fd: int, depth: int, state: _ScanState) -> _ScanState:
    if depth > MAX_PLUGIN_AUDIT_DEPTH:
        raise PluginAuditSourceError("plugin audit source tree exceeds the maximum depth")
    if state.directory_count >= MAX_PLUGIN_AUDIT_DIRECTORIES:
        raise PluginAuditSourceError(f"plugin audit scans at most {MAX_PLUGIN_AUDIT_DIRECTORIES} directories")
    next_state = _ScanState(state.sources, state.directory_count + 1)
    try:
        names = _bounded_directory_names(directory_fd)
    except OSError as exc:
        raise PluginAuditSourceError("plugin audit source directory cannot be read") from exc
    for name in names:
        next_state = _scan_entry(directory_fd, name, depth, next_state)
    return next_state


def _scan_entry(directory_fd: int, name: str, depth: int, state: _ScanState) -> _ScanState:
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise PluginAuditSourceError("plugin audit source entry cannot be inspected") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise PluginAuditSourceError("plugin audit source must not contain a symlink")
    if stat.S_ISDIR(metadata.st_mode):
        if name in _IGNORED_DIRECTORIES:
            return state
        return _scan_child_directory(directory_fd, name, depth, state)
    if _is_audited_file(name):
        if not stat.S_ISREG(metadata.st_mode):
            raise PluginAuditSourceError("plugin audit source must contain regular files")
        if len(state.sources) >= MAX_PLUGIN_AUDIT_FILES:
            raise PluginAuditSourceError(f"plugin audit scans at most {MAX_PLUGIN_AUDIT_FILES} files")
        return _ScanState(
            (*state.sources, PluginAuditSource(name, _read_regular_file(directory_fd, name), depth == 0)),
            state.directory_count,
        )
    return state


def _scan_child_directory(directory_fd: int, name: str, depth: int, state: _ScanState) -> _ScanState:
    nofollow = _NOFOLLOW_FLAG
    if nofollow is None:
        raise PluginAuditSourceError("secure plugin audit directory access is unavailable")
    try:
        child_descriptor = os.open(
            name,
            os.O_RDONLY | _DIRECTORY_FLAG | _NONBLOCK_FLAG | nofollow,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise PluginAuditSourceError("plugin audit source directory cannot be safely opened") from exc
    try:
        return _scan_directory(child_descriptor, depth + 1, state)
    finally:
        os.close(child_descriptor)


def _bounded_directory_names(directory_fd: int) -> tuple[str, ...]:
    names: list[str] = []
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            if len(names) >= MAX_PLUGIN_AUDIT_DIRECTORY_ENTRIES:
                raise PluginAuditSourceError(
                    f"plugin audit scans at most {MAX_PLUGIN_AUDIT_DIRECTORY_ENTRIES} entries per directory"
                )
            names.append(entry.name)
    return tuple(sorted(names))


def _is_audited_file(name: str) -> bool:
    return name in _AUDITED_FILENAMES or Path(name).suffix.lower() in _AUDITED_SUFFIXES


def _read_regular_file(directory_fd: int, name: str) -> bytes:
    nofollow = _NOFOLLOW_FLAG
    if nofollow is None:
        raise PluginAuditSourceError("secure plugin audit file access is unavailable")
    descriptor = -1
    try:
        descriptor = os.open(name, os.O_RDONLY | _NONBLOCK_FLAG | nofollow, dir_fd=directory_fd)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise PluginAuditSourceError("plugin audit source must contain regular files")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read(MAX_PLUGIN_AUDIT_FILE_BYTES + 1)
    except OSError as exc:
        raise PluginAuditSourceError("plugin audit source cannot be safely opened") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(content) > MAX_PLUGIN_AUDIT_FILE_BYTES:
        raise PluginAuditSourceError(f"plugin audit source exceeds {MAX_PLUGIN_AUDIT_FILE_BYTES} bytes")
    return content
