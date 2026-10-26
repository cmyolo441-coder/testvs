"""Platform operations for staged command-package generation pointers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import subprocess
from typing import Literal, Protocol

try:
    from ..core.errors import OmhError
    from ..system.local_store import atomic_write_text, is_directory_link, is_junction
except ImportError:  # pragma: no cover - direct-source installer smoke.
    from core.errors import OmhError
    from system.local_store import atomic_write_text, is_directory_link, is_junction


JUNCTION_TIMEOUT_SECONDS = 15.0
JUNCTION_LINK_ENV = "OMH_JUNCTION_LINK"
JUNCTION_TARGET_ENV = "OMH_JUNCTION_TARGET"
WINDOWS_JUNCTION_COMMAND = (
    "New-Item -ItemType Junction -Path $env:OMH_JUNCTION_LINK "
    "-Target $env:OMH_JUNCTION_TARGET -ErrorAction Stop | Out-Null"
)

class Runner(Protocol):
    """The exact subprocess surface used to create a Windows junction."""

    def __call__(
        self,
        command: list[str],
        /,
        *,
        shell: Literal[False],
        cwd: str,
        env: dict[str, str],
        text: Literal[True],
        stdout: int,
        stderr: int,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...



@dataclass(frozen=True)
class SelfUpdatePlatform:
    """Explicit OS and subprocess seam for pointer and launcher operations."""

    is_windows: bool
    runner: Runner = subprocess.run

    @classmethod
    def host(cls, *, is_windows: bool | None = None) -> SelfUpdatePlatform:
        return cls(os.name == "nt" if is_windows is None else is_windows)

    @classmethod
    def windows(cls, runner: Runner) -> SelfUpdatePlatform:
        return cls(True, runner)

    def scripts_dir(self, venv: Path) -> Path:
        return venv / ("Scripts" if self.is_windows else "bin")

    def link_target(self, root: Path, link: Path) -> Path | None:
        if not is_directory_link(link):
            return None
        try:
            raw = os.readlink(link)
        except OSError:
            return None
        target = Path(_without_windows_namespace(raw) if self.is_windows else raw)
        return target if target.is_absolute() else root / target

    def create_directory_link(self, root: Path, link: Path, target: Path) -> None:
        """Create a relative POSIX symlink or absolute Windows junction without resolving its target."""
        if not self.is_windows:
            relative = os.path.relpath(target, link.parent)
            os.symlink(relative, link, target_is_directory=True)
            return
        command = [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            WINDOWS_JUNCTION_COMMAND,
        ]
        environment = os.environ.copy()
        environment[JUNCTION_LINK_ENV] = str(link)
        environment[JUNCTION_TARGET_ENV] = os.path.abspath(target)
        try:
            completed = self.runner(
                command,
                shell=False,
                cwd=str(link.parent),
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=JUNCTION_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OmhError(f"cannot create Windows directory junction: {exc}") from exc
        if completed.returncode:
            detail = str(completed.stderr or completed.stdout or "junction creation failed").strip()
            raise OmhError(f"cannot create Windows directory junction: {detail}")

    def replace_current(self, root: Path, target: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        current = root / "current"
        # Random names: both links are created exclusively, so nothing that
        # sits at a guessable name is removed or reused.
        token = secrets.token_hex(16)
        temporary = root / f".current.{token}.tmp"
        backup = root / f".current.{token}.previous"
        try:
            self.create_directory_link(root, temporary, target)
            if self.is_windows:
                _replace_windows_pointer(temporary, current, backup)
            else:
                os.replace(temporary, current)
        except (OSError, OmhError) as exc:
            _remove_directory_link(temporary)
            raise OmhError(f"cannot atomically replace current generation pointer: {exc}") from exc

    def rewrite_command_shim(self, launcher: Path, root: Path) -> None:
        target = self.scripts_dir(root / "current" / "venv") / "omh.exe"
        atomic_write_text(launcher, f'@echo off\r\n"{_windows_spelling(target)}" %*\r\n')


def _remove_directory_link(path: Path) -> None:
    # `is_junction` / `is_directory_link` come from `local_store` so that the
    # generation-pointer seam here and the plugin-install cleanup share one
    # answer to "is this entry a link?". Two answers is how a junction ends up
    # walked as a directory on Windows.
    junction = is_junction(path)
    if not path.exists() and not path.is_symlink() and not junction:
        return
    if junction:
        path.rmdir()
    else:
        path.unlink()


remove_directory_link = _remove_directory_link


def _replace_windows_pointer(temporary: Path, current: Path, backup: Path) -> None:
    """Switch a junction pair without replacing an existing junction in place."""
    had_current = is_directory_link(current)
    if current.exists() and not had_current:
        raise OSError(f"current generation pointer is not a directory junction: {current}")
    if had_current:
        os.replace(current, backup)
    try:
        os.replace(temporary, current)
    except OSError:
        if had_current and is_directory_link(backup) and not is_directory_link(current):
            os.replace(backup, current)
        raise
    if had_current:
        _remove_directory_link(backup)


def _windows_spelling(path: Path) -> str:
    return str(path).replace("/", "\\")


def _without_windows_namespace(value: str) -> str:
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    if value.startswith("\\\\?\\"):
        return value[4:]
    return value
