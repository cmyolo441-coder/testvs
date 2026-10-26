"""Filesystem-only, opaque project-memory identity for the standalone bundle.

Resolution is read-only. Only the explicit operator mint function writes.
Git includes, commands, credential helpers and network services are never run.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
from urllib.parse import unquote, urlsplit

from .iso_datetime import strict_fromisoformat

RESOLVER_VERSION = "project_identity/v2"
EXPLICIT_SCHEMA_VERSION = "project_identity_file/v1"
LEGACY_BASENAME_STATE = "legacy_basename"
DIAGNOSTICS = frozenset({"remote_absent", "remote_ambiguous", "explicit_invalid", "git_metadata_unreadable", "outside_repository"})


@dataclass(frozen=True)
class ProjectIdentityResolution:
    resolver_version: str = RESOLVER_VERSION
    state: str = "unresolved"
    identity: str = ""
    evidence: str = ""
    diagnostics: tuple[str, ...] = ()


class ProjectIdentityUnresolvedError(ValueError):
    """Capture cannot invent project authority or silently widen its audience."""

    def __init__(self) -> None:
        super().__init__("scope_unresolved: run `omh memory project-identity init` in the project")


def project_identity_root(cwd: str | Path | None = None) -> Path | None:
    start = Path(cwd).expanduser().absolute() if cwd is not None else Path.cwd()
    for root in (start, *start.parents):
        if (root / ".git").exists() or (root / ".git").is_symlink() or (root / ".omh" / "project-identity.json").exists():
            return root
    return None


def _unresolved(reason: str) -> ProjectIdentityResolution:
    return ProjectIdentityResolution(diagnostics=(reason,))


def _explicit(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("explicit_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "identity", "created_at", "resolver_version"}:
        raise ValueError("explicit_invalid")
    if value["schema_version"] != EXPLICIT_SCHEMA_VERSION or value["resolver_version"] != RESOLVER_VERSION:
        raise ValueError("explicit_invalid")
    identity = value["identity"]
    if not isinstance(identity, str) or not re.fullmatch(r"prj:[0-9a-f]{64}", identity):
        raise ValueError("explicit_invalid")
    created = value["created_at"]
    if not isinstance(created, str) or strict_fromisoformat(created.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("explicit_invalid")
    return identity


def _git_config_value(raw: str) -> str:
    """Decode Git quotes/escapes and comments before interpreting a remote URL."""
    decoded: list[str] = []
    quoted = escaped = False
    escapes = {"n": "\n", "t": "\t", "b": "\b", '"': '"', "\\": "\\"}
    for char in raw:
        if escaped:
            if char not in escapes:
                raise ValueError("remote_invalid")
            decoded.append(escapes[char])
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif char in "#;" and not quoted:
            break
        else:
            decoded.append(char)
    if quoted or escaped:
        raise ValueError("remote_invalid")
    return "".join(decoded).strip()


def _local_remote(value: str, anchor: Path) -> str:
    # Local endpoints have no host authority. Qualify them against the common
    # Git directory's checkout, never hash an ambiguous relative spelling.
    if not value or (os.name != "nt" and re.match(r"^(?:[A-Za-z]:|\\\\)", value)):
        raise ValueError("remote_invalid")
    endpoint = (anchor / Path(value).expanduser()).resolve(strict=True)
    if not endpoint.is_dir() and not endpoint.is_file():
        raise ValueError("remote_invalid")
    return "local:" + endpoint.as_uri()


def _normalize_remote(raw: str, *, local_anchor: Path) -> str:
    value = _git_config_value(raw)
    if any(char in value for char in ("\n", "\r", "\x00")):
        raise ValueError("remote_invalid")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.scheme == "file":
            if parsed.query or parsed.fragment or parsed.hostname not in (None, "", "localhost"):
                raise ValueError("remote_invalid")
            local = unquote(parsed.path)
            if os.name == "nt" and re.match(r"^/[A-Za-z]:/", local):
                local = local[1:]
            return _local_remote(local, local_anchor)
        host = parsed.hostname or ""
        if not host:
            raise ValueError("remote_invalid")
        # Keep non-default ports: they can identify distinct repositories.
        port = f":{parsed.port}" if parsed.port is not None else ""
        value = host.lower() + port + parsed.path
    else:
        scp = re.fullmatch(r"(?:[^/@:]+@)?([^/:]+):(.+)", value)
        if scp and not re.match(r"^[A-Za-z]:[\\\\/]", value):
            value = scp[1].lower() + "/" + scp[2].lstrip("/")
        else:
            return _local_remote(value, local_anchor)
    value = value.rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if not value or any(char in value for char in ("\n", "\r", "\x00")):
        raise ValueError("remote_invalid")
    return value


def resolve_project_identity(cwd: str | Path | None = None) -> ProjectIdentityResolution:
    try:
        root = project_identity_root(cwd)
        if root is None:
            return _unresolved("outside_repository")
        explicit = root / ".omh" / "project-identity.json"
        if explicit.exists() or explicit.is_symlink():
            try:
                identity = _explicit(explicit)
            except (OSError, ValueError, TypeError, RecursionError):
                return _unresolved("explicit_invalid")
            return ProjectIdentityResolution(state="resolved", identity=identity, evidence="explicit")
        gitdir = root / ".git"
        if gitdir.is_file():
            marker = gitdir.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir: ") or "\n" in marker:
                return _unresolved("git_metadata_unreadable")
            gitdir = root / marker[8:].strip()
        common = gitdir / "commondir"
        if common.exists():
            relative = common.read_text(encoding="utf-8").strip()
            if not relative or "\n" in relative:
                return _unresolved("git_metadata_unreadable")
            gitdir = gitdir / relative
        gitdir = gitdir.resolve(strict=True)
        config = configparser.RawConfigParser(interpolation=None)
        config.read_string((gitdir / "config").read_text(encoding="utf-8"))
        remotes = {}
        for section in config.sections():
            match = re.fullmatch(r'remote "([^"]+)"', section)
            if match and config.has_option(section, "url"):
                remotes[match[1]] = _normalize_remote(config.get(section, "url"), local_anchor=gitdir.parent)
        if not remotes:
            return _unresolved("remote_absent")
        if "origin" in remotes:
            remote = remotes["origin"]
        elif len(set(remotes.values())) > 1:
            return _unresolved("remote_ambiguous")
        else:
            remote = remotes[sorted(remotes)[0]]
        identity = "repo:" + hashlib.sha256(remote.encode("utf-8")).hexdigest()[:32]
        return ProjectIdentityResolution(state="resolved", identity=identity, evidence="vcs_remote")
    except (OSError, ValueError, configparser.Error):
        return _unresolved("git_metadata_unreadable")


def require_project_identity(cwd: str | Path | None = None) -> str:
    resolution = resolve_project_identity(cwd)
    if resolution.state != "resolved":
        raise ProjectIdentityUnresolvedError()
    return resolution.identity


def mint_explicit_project_identity(root: str | Path) -> str:
    """Explicit local-only opt-in. Existing invalid evidence is never overwritten."""
    root = Path(root).expanduser()
    path = root / ".omh" / "project-identity.json"
    if path.exists() or path.is_symlink():
        try:
            return _explicit(path)
        except (OSError, ValueError, TypeError, RecursionError):
            raise ValueError("explicit_invalid") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = "prj:" + secrets.token_hex(32)
    value = {"schema_version": EXPLICIT_SCHEMA_VERSION, "identity": identity,
             "created_at": datetime.now(timezone.utc).isoformat(), "resolver_version": RESOLVER_VERSION}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return _explicit(path)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
    return identity


def legacy_project_scope(root: str | Path) -> dict[str, str]:
    """Inspection/migration only; never an automatic recall fallback."""
    return {"kind": "project", "ref": Path(root).name or "default"}
