"""Closed execution environment and toolchain identity for reusable checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .verification_plan import VerificationNode

_SECRET_ENVIRONMENT_NAME_TOKENS = frozenset(
    {"AUTH", "AUTHORIZATION", "CREDENTIAL", "KEY", "PASS", "PASSWORD", "PIN", "SECRET", "TOKEN"}
)
_INHERITED_VERIFICATION_ENVIRONMENT_NAMES = frozenset(
    {
        "APPDATA",
        "CI",
        "COMSPEC",
        "GITHUB_ACTIONS",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOCALAPPDATA",
        "LOGNAME",
        "NO_COLOR",
        "NUMBER_OF_PROCESSORS",
        "OMH_FANOUT_DEPTH",
        "OMH_FANOUT_LINEAGE",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONIOENCODING",
        "PYTHONPATH",
        "PYTHONUTF8",
        "RUNNER_ARCH",
        "RUNNER_OS",
        "RUNNER_TEMP",
        "RUNNER_TOOL_CACHE",
        "SHELL",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)
_TOOLCHAIN_FILE_NAMES = (
    "uv.lock",
    "poetry.lock",
    "requirements.txt",
    "package-lock.json",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "go.sum",
    "pyproject.toml",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
)


def verification_execution_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Project the non-secret ambient environment structured checks inherit."""
    return {
        name: value
        for name, value in environment.items()
        if name in _INHERITED_VERIFICATION_ENVIRONMENT_NAMES
    }


def _effective_environment(
    node: VerificationNode, environment: Mapping[str, str] | None
) -> dict[str, str]:
    effective = dict(environment or {})
    effective.update(dict(node.env_overrides))
    return effective


def _environment_name_is_secret(name: str) -> bool:
    tokens = {token for token in re.split(r"[^A-Z0-9]+", name.upper()) if token}
    return bool(tokens & _SECRET_ENVIRONMENT_NAME_TOKENS)


def has_secret_execution_environment(
    node: VerificationNode, *, environment: Mapping[str, str] | None = None
) -> bool:
    """Whether this check explicitly adds a secret-bearing environment value."""
    return any(
        _environment_name_is_secret(name)
        for name in _effective_environment(node, environment)
    )


def toolchain_digest(
    node: VerificationNode,
    *,
    worktree: Path,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Hash executable, safe environment, and relevant config without storing values."""
    effective_environment = _effective_environment(node, environment)
    resolved = (
        shutil.which(node.argv[0], path=effective_environment.get("PATH"))
        or node.argv[0]
    )
    digest = hashlib.sha256()
    digest.update(resolved.encode("utf-8"))
    executable = Path(resolved)
    try:
        digest.update(executable.read_bytes())
    except OSError:
        digest.update(b"unreadable-executable")
    for name, value in sorted(effective_environment.items()):
        digest.update(b"\x00env\x00")
        digest.update(name.encode("utf-8"))
        if _environment_name_is_secret(name):
            digest.update(b"=secret-environment-present")
        else:
            digest.update(b"=")
            digest.update(value.encode("utf-8"))
    for name in _TOOLCHAIN_FILE_NAMES:
        candidate = worktree / name
        if not candidate.is_file():
            continue
        digest.update(b"\x00config\x00")
        digest.update(name.encode("utf-8"))
        try:
            digest.update(candidate.read_bytes())
        except OSError:
            digest.update(b"unreadable")
    return digest.hexdigest()
