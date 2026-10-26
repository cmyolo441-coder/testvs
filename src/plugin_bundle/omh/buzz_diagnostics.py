from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Protocol


BUZZ_PROBE_SCHEMA_VERSION = "omh_buzz_probe/v1"
_SECRET_KEYS = ("BUZZ_PRIVATE_KEY", "BUZZ_CREDENTIALS_FILE")


class _Paths(Protocol):
    hermes_home: Path
    hermes_config_path: Path


def probe_buzz(
    paths: _Paths,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Inspect local Buzz readiness without network access or state mutation."""

    source_env = os.environ if environ is None else environ
    config_text = _read_text(paths.hermes_config_path)
    dotenv_keys = _dotenv_keys(paths.hermes_home / ".env")
    configured = _buzz_enabled(config_text) or bool(source_env.get("BUZZ_RELAY_URL"))
    credential_present = any(bool(source_env.get(key)) or key in dotenv_keys for key in _SECRET_KEYS)
    executable = _resolve_cli(source_env, config_text, paths.hermes_home)

    if executable is None:
        return _payload(
            status="missing",
            reason_code="buzz_cli_missing",
            configured=configured,
            credential_present=credential_present,
            executable=None,
        )

    return _payload(
        status="unverified",
        reason_code="buzz_cli_present_not_executed",
        configured=configured,
        credential_present=credential_present,
        executable=str(executable),
    )


def _payload(
    *,
    status: str,
    reason_code: str,
    configured: bool,
    credential_present: bool,
    executable: str | None,
) -> dict[str, object]:
    return {
        "schema_version": BUZZ_PROBE_SCHEMA_VERSION,
        "status": status,
        "reason_code": reason_code,
        "configured": configured,
        "credential_present": credential_present,
        "installed": executable is not None,
        "executable": executable,
        "version": None,
        "observed": False,
        "read_only": True,
        "claim_boundary": (
            "This probe inspects local configuration and executable filesystem metadata only. It does not execute "
            "the Buzz CLI, contact a relay or model, or prove authentication, membership, message delivery, media "
            "rendering, or restore readiness."
        ),
    }


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _dotenv_keys(path: Path) -> frozenset[str]:
    keys: set[str] = set()
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        normalized_key = key.strip()
        if normalized_key.startswith("export "):
            normalized_key = normalized_key.removeprefix("export ").strip()
        if normalized_key and value.strip():
            keys.add(normalized_key)
    return frozenset(keys)


def _buzz_enabled(config_text: str) -> bool:
    lines = config_text.splitlines()
    buzz_indent: int | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "buzz:":
            buzz_indent = indent
            continue
        if buzz_indent is None:
            continue
        if indent <= buzz_indent:
            buzz_indent = None
            continue
        if indent == buzz_indent + 2 and _yaml_scalar(stripped.partition(":")[2]).casefold() == "true":
            return True
    return False


def _config_cli_path(config_text: str) -> str:
    buzz_indent: int | None = None
    extra_indent: int | None = None
    for raw_line in config_text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped == "buzz:":
            buzz_indent = indent
            extra_indent = None
            continue
        if buzz_indent is None:
            continue
        if indent <= buzz_indent:
            buzz_indent = None
            extra_indent = None
            continue
        if stripped == "extra:":
            extra_indent = indent
            continue
        if extra_indent is not None and indent == extra_indent + 2 and stripped.startswith("cli_path:"):
            return _yaml_scalar(stripped.partition(":")[2])
    return ""


def _yaml_scalar(value: str) -> str:
    scalar = value.strip()
    if len(scalar) >= 2 and scalar[0] == scalar[-1] and scalar[0] in {"'", '"'}:
        return scalar[1:-1]
    return scalar


def _resolve_cli(environ: Mapping[str, str], config_text: str, hermes_home: Path) -> Path | None:
    configured = environ.get("BUZZ_CLI_PATH", "").strip() or _config_cli_path(config_text)
    if configured:
        path = Path(os.path.expandvars(configured)).expanduser()
        return path.resolve() if path.is_file() and os.access(path, os.X_OK) else None
    fallback = hermes_home.parent / "bin" / "buzz"
    return fallback.resolve() if fallback.is_file() and os.access(fallback, os.X_OK) else None
