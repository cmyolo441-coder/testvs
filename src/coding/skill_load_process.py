"""Executable selection, snapshot binding, and process lifecycle for skill probes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from typing import Final

from ._hermes_child_process import start_pipe_drainers, terminate_process_group
from .skill_load_protocol import HERMES_SKILL_INVENTORY_SCHEMA_VERSION

SAFE_ENV_NAMES: Final = frozenset({
    "PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "TMP", "TEMP",
    "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC",
})


@dataclass(frozen=True, slots=True)
class SkillLoadProbeRequest:
    expected_skills: Sequence[str]
    hermes: str = "hermes"
    timeout_seconds: float = 10.0
    termination_grace_seconds: float = 0.25
    env: Mapping[str, str] | None = None
    nonce: str | None = None


@dataclass(slots=True)
class ResolvedTool:
    executable: str
    fingerprint: str
    _scratch: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        self._scratch.cleanup()


@dataclass(frozen=True, slots=True)
class ProcessOutcome:
    kind: str
    exit_code: int | None
    stdout: bytes


def run_inventory_process(
    request: SkillLoadProbeRequest,
    executable: str,
    env: Mapping[str, str],
    nonce: str,
    expected_digest: str,
    tool_fingerprint: str,
) -> ProcessOutcome:
    protocol_argv = (
        "--safe-mode", "--ignore-user-config", "--ignore-rules", "skills", "inventory",
        "--protocol", HERMES_SKILL_INVENTORY_SCHEMA_VERSION, "--nonce", nonce,
        "--expected-digest", expected_digest, "--tool-fingerprint", tool_fingerprint,
    )
    argv = inventory_argv(executable, protocol_argv, windows=os.name == "nt")
    process: subprocess.Popen[bytes] | None = None
    drainers = None
    kind = "spawn_error"
    cleanup_verified = True
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=env, text=False, close_fds=True, start_new_session=os.name != "nt",
        )
        drainers = start_pipe_drainers(process)
        try:
            process.wait(timeout=request.timeout_seconds)
            kind = "complete"
        except subprocess.TimeoutExpired:
            kind = "timeout"
    except OSError:
        pass
    finally:
        if process is not None:
            _signals, cleanup_verified = terminate_process_group(
                process, request.termination_grace_seconds, signal.SIGTERM
            )
            if drainers is None:
                for pipe in (process.stdout, process.stderr):
                    if pipe is not None:
                        pipe.close()
            else:
                for drainer in drainers:
                    drainer.done.wait(max(0.1, request.termination_grace_seconds))
                    drainer.thread.join(timeout=max(0.1, request.termination_grace_seconds))
    if not cleanup_verified or process is None or drainers is None:
        return ProcessOutcome("spawn_error", None, b"")
    stdout_capture = drainers[0].capture()
    if stdout_capture.truncated:
        return ProcessOutcome("complete", process.returncode, b"")
    return ProcessOutcome(kind, process.returncode, stdout_capture.data)


def probe_environment(source: Mapping[str, str] | None) -> dict[str, str]:
    values = os.environ if source is None else source
    folded = {name.casefold(): value for name, value in values.items()}
    env = {
        name: folded[name.casefold()]
        for name in SAFE_ENV_NAMES
        if folded.get(name.casefold())
    }
    if source is None:
        env.setdefault("PATH", os.defpath)
    env.update({
        "HERMES_SAFE_MODE": "1", "HERMES_IGNORE_USER_CONFIG": "1", "HERMES_IGNORE_RULES": "1",
        "OMH_ISOLATED_HERMES_ROUTING": "disabled", "OMH_ISOLATED_HERMES_MAX_DEPTH": "1",
    })
    return env


def inventory_argv(
    executable: str, protocol_argv: Sequence[str], *, windows: bool,
) -> tuple[str, ...]:
    if windows and Path(executable).suffix.casefold() == ".py":
        return (sys.executable, executable, *protocol_argv)
    return (executable, *protocol_argv)


def resolve_executable(
    executable: str, env: Mapping[str, str], *, windows: bool,
) -> Path | None:
    has_separator = "/" in executable or "\\" in executable
    if has_separator or Path(executable).is_absolute():
        return Path(executable).expanduser()
    folded = {name.casefold(): value for name, value in env.items()}
    path_value = folded.get("path", "")
    if not windows:
        found = shutil.which(executable, path=path_value)
        return None if found is None else Path(found)
    extensions = tuple(
        extension if extension.startswith(".") else f".{extension}"
        for extension in folded.get("pathext", ".COM;.EXE;.BAT;.CMD").split(";")
        if extension
    )
    supplied_suffix = Path(executable).suffix.casefold()
    names = (
        (executable,) if supplied_suffix in {extension.casefold() for extension in extensions}
        else tuple(f"{executable}{extension}" for extension in extensions)
    )
    for directory_name in path_value.split(";"):
        if not directory_name:
            continue
        directory = Path(directory_name)
        try:
            entries = {entry.name.casefold(): entry for entry in directory.iterdir()}
        except OSError:
            continue
        for name in names:
            selected = entries.get(name.casefold())
            if selected is not None and selected.is_file():
                return selected
    return None


def snapshot_resolved_tool(executable: str, env: Mapping[str, str]) -> ResolvedTool:
    """Copy one opened executable inode into a private per-probe snapshot."""
    selected = resolve_executable(executable, env, windows=os.name == "nt")
    if selected is None:
        raise FileNotFoundError(executable)
    resolved = selected.resolve(strict=True)
    source_fd = os.open(
        resolved, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    scratch: tempfile.TemporaryDirectory[str] | None = None
    snapshot_fd: int | None = None
    keep_snapshot = False
    try:
        source_identity = os.fstat(source_fd)
        if not stat.S_ISREG(source_identity.st_mode):
            raise PermissionError(executable)
        if os.name != "nt" and source_identity.st_mode & 0o111 == 0:
            raise PermissionError(executable)
        scratch = tempfile.TemporaryDirectory(prefix="omh-skill-load-tool-")
        snapshot = Path(scratch.name) / resolved.name
        snapshot_fd = os.open(
            snapshot, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o500,
        )
        source_content = hashlib.sha256()
        copied = 0
        while chunk := os.read(source_fd, 64 * 1024):
            source_content.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(snapshot_fd, view)
                if written <= 0:
                    raise OSError("snapshot write did not advance")
                view = view[written:]
        if copied != source_identity.st_size:
            raise OSError("executable size changed during snapshot")
        fchmod = getattr(os, "fchmod", None)
        if fchmod is None:
            os.chmod(snapshot, 0o500)
        else:
            fchmod(snapshot_fd, 0o500)
        os.fsync(snapshot_fd)
        snapshot_identity = os.fstat(snapshot_fd)
        if not stat.S_ISREG(snapshot_identity.st_mode) or snapshot_identity.st_size != copied:
            raise OSError("executable snapshot is incomplete")
        os.lseek(snapshot_fd, 0, os.SEEK_SET)
        snapshot_content = hashlib.sha256()
        while chunk := os.read(snapshot_fd, 64 * 1024):
            snapshot_content.update(chunk)
        if snapshot_content.digest() != source_content.digest():
            raise OSError("executable snapshot bytes do not match source descriptor")
        fingerprint = hashlib.sha256()
        fingerprint.update(b"omh-skill-load-tool/v2\0")
        fingerprint.update(os.fsencode(resolved))
        fingerprint.update(
            (f"\0{source_identity.st_dev}\0{source_identity.st_ino}\0{source_identity.st_mode}"
             f"\0{source_identity.st_size}\0").encode("ascii")
        )
        fingerprint.update(snapshot_content.digest())
        os.close(snapshot_fd)
        snapshot_fd = None
        os.close(source_fd)
        source_fd = -1
        keep_snapshot = True
        return ResolvedTool(str(snapshot), fingerprint.hexdigest(), scratch)
    finally:
        if snapshot_fd is not None:
            os.close(snapshot_fd)
        if source_fd >= 0:
            os.close(source_fd)
        if scratch is not None and not keep_snapshot:
            scratch.cleanup()
