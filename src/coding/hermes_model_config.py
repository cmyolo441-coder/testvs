"""Local-only Hermes model-alias configuration adapter.

The adapter delegates all configuration interpretation and writes to Hermes'
own ``config`` and ``auth`` commands. It never invokes ``hermes model``, reads
Hermes' dotenv file, or performs network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from threading import Lock
from typing import Final, Mapping

from ..system.metadata_safety import is_sensitive_metadata_text


_ALIAS_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")
# `hermes auth list` heads each pool with `<id> (N credentials):`. A legacy
# `custom_providers:` entry's pool id is composed as `custom:<name>`
# (agent/credential_pool.py CUSTOM_POOL_PREFIX); only that one colon form is read.
_AUTH_PROVIDER_RE: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z0-9_-]+|custom:[A-Za-z0-9_.-]+) \("
)
_CUSTOM_PROVIDER_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.-]+$")
_CUSTOM_POOL_PREFIX: Final = "custom:"
_COMMAND_TIMEOUT_SECONDS: Final = 10
_APPLY_LOCK: Final = Lock()


class HermesModelConfigError(RuntimeError):
    """Base error for deterministic Hermes model configuration operations."""


class AliasCollisionError(HermesModelConfigError):
    """Raised when a preview would replace an existing alias by default."""


class ConfirmationRequiredError(HermesModelConfigError):
    """Raised when mutation was not explicitly confirmed."""


class ConfigDigestMismatchError(HermesModelConfigError):
    """Raised when compare-and-swap detects configuration drift."""


class HermesCommandError(HermesModelConfigError):
    """Raised when a required local Hermes command fails."""


class HermesConfigUnreadableError(HermesCommandError):
    """Raised when the config file Hermes names cannot be read.

    `missing` is True only when the file does not exist; a file that exists
    but cannot be read (permissions, a directory in its place) is not
    something a later setup run creates.
    """

    def __init__(self, config_path: Path, *, missing: bool) -> None:
        super().__init__(f"cannot read Hermes config at {config_path}")
        self.config_path = config_path
        self.missing = missing


class HermesUnavailableError(HermesCommandError):
    """Raised when the Hermes executable is not found, so no command could run."""


class HermesVerificationError(HermesModelConfigError):
    """Raised when post-write inspection does not match the preview."""


@dataclass(frozen=True, slots=True)
class ProviderPresence:
    provider_id: str
    auth_present: bool
    auth_status_ok: bool
    plugin_present: bool


@dataclass(frozen=True, slots=True)
class HermesModelConfigInspection:
    hermes: str
    config_path: Path
    config_digest: str
    config_check_ok: bool
    model_aliases: Mapping[str, str]
    model_dot_aliases: Mapping[str, str]
    providers: tuple[ProviderPresence, ...]
    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class HermesModelConfigPreview:
    hermes: str
    config_path: Path
    config_digest: str
    before_aliases: Mapping[str, str]
    changes: Mapping[str, str | None]
    commands: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class HermesModelConfigReceipt:
    verified: bool
    before_digest: str
    after_digest: str
    commands: tuple[tuple[str, ...], ...]
    inspection: HermesModelConfigInspection


def inspect_hermes_model_config(
    *,
    hermes: str = "hermes",
    env: Mapping[str, str] | None = None,
) -> HermesModelConfigInspection:
    """Inspect aliases, auth presence/status, and provider plugins locally."""
    commands: list[tuple[str, ...]] = []
    config_path = Path(_required_output((hermes, "config", "path"), env, commands)).expanduser()
    digest = _config_digest(config_path)
    model_aliases = _optional_json_mapping(
        (hermes, "config", "get", "model_aliases", "--json"), env, commands
    )
    model_dot_aliases = _optional_json_mapping(
        (hermes, "config", "get", "model.aliases", "--json"), env, commands
    )
    provider = _optional_json_value(
        (hermes, "config", "get", "model.provider", "--json"), env, commands
    )
    plugins = _optional_json_value(
        (hermes, "config", "get", "plugins", "--json"), env, commands
    )
    custom_providers = _optional_json_value(
        (hermes, "config", "get", "custom_providers", "--json"), env, commands
    )
    check = _run((hermes, "config", "check"), env, commands)
    auth_list = _run((hermes, "auth", "list"), env, commands)

    auth_providers = _configured_auth_provider_ids(
        _auth_provider_ids(auth_list.stdout if auth_list.returncode == 0 else ""),
        _custom_provider_names(custom_providers),
    )
    plugin_providers = _plugin_provider_ids(plugins)
    providers = set(auth_providers) | plugin_providers
    if isinstance(provider, str) and _safe_metadata(provider):
        providers.add(provider)
    providers.update(_alias_provider_ids(model_aliases))
    providers.update(_alias_provider_ids(model_dot_aliases))

    presence: list[ProviderPresence] = []
    for provider_id in sorted(providers):
        status = _run((hermes, "auth", "status", provider_id), env, commands)
        presence.append(
            ProviderPresence(
                provider_id=provider_id,
                auth_present=provider_id in auth_providers,
                auth_status_ok=status.returncode == 0,
                plugin_present=provider_id in plugin_providers,
            )
        )
    return HermesModelConfigInspection(
        hermes=hermes,
        config_path=config_path,
        config_digest=digest,
        config_check_ok=check.returncode == 0,
        model_aliases=model_aliases,
        model_dot_aliases=model_dot_aliases,
        providers=tuple(presence),
        commands=tuple(commands),
    )


def preview_hermes_model_config(
    inspection: HermesModelConfigInspection,
    changes: Mapping[str, str | None],
    *,
    allow_collisions: bool = False,
) -> HermesModelConfigPreview:
    """Build exact nested-alias Hermes commands without executing them."""
    if not inspection.config_check_ok:
        raise HermesModelConfigError("Hermes config check failed; mutation preview refused")
    commands: list[tuple[str, ...]] = []
    normalized: dict[str, str | None] = {}
    for alias in sorted(changes):
        if not _ALIAS_RE.fullmatch(alias) or not _safe_metadata(alias):
            raise HermesModelConfigError(f"unsupported Hermes model alias: {alias!r}")
        target = changes[alias]
        existing = inspection.model_dot_aliases.get(alias)
        if target is not None and existing is not None and existing != target and not allow_collisions:
            raise AliasCollisionError(
                f"Hermes model alias {alias!r} already points to {existing!r}; collision refused"
            )
        if target is not None:
            provider_id = target.split("/", 1)[0]
            if (
                _safe_metadata(target)
                and provider_id == "custom"
                and not _provider_has_auth(inspection, provider_id)
            ):
                raise HermesModelConfigError(
                    f"Hermes model target {target!r} uses bare 'custom', which names no "
                    "configured endpoint or credential pool; write it as '<name>/<model>' "
                    "with the custom provider's configured name"
                )
            if not _safe_metadata(target) or not _provider_has_auth(inspection, provider_id):
                raise HermesModelConfigError(
                    f"Hermes model target {target!r} lacks safe observed provider auth"
                )
        normalized[alias] = target
        key = f"model.aliases.{alias}"
        if target is None:
            commands.append((inspection.hermes, "config", "unset", key))
        elif existing != target:
            commands.append((inspection.hermes, "config", "set", key, target))
    return HermesModelConfigPreview(
        hermes=inspection.hermes,
        config_path=inspection.config_path,
        config_digest=inspection.config_digest,
        before_aliases=dict(inspection.model_dot_aliases),
        changes=normalized,
        commands=tuple(commands),
    )


def apply_hermes_model_config(
    preview: HermesModelConfigPreview,
    *,
    confirmed: bool,
    expected_config_digest: str,
    env: Mapping[str, str] | None = None,
) -> HermesModelConfigReceipt:
    """Apply a confirmed digest-bound preview and verify the resulting aliases."""
    if not confirmed:
        raise ConfirmationRequiredError("Hermes model config apply requires explicit confirmation")
    if expected_config_digest != preview.config_digest:
        raise ConfigDigestMismatchError("expected digest does not match the preview digest")
    with _APPLY_LOCK:
        inspection = _apply_locked(preview, env)
    for alias, target in preview.changes.items():
        observed = inspection.model_dot_aliases.get(alias)
        if target is None and observed is not None:
            raise HermesVerificationError(f"Hermes alias {alias!r} remained after unset")
        if target is not None and observed != target:
            raise HermesVerificationError(
                f"Hermes alias {alias!r} verification mismatch: expected {target!r}, got {observed!r}"
            )
    return HermesModelConfigReceipt(
        verified=True,
        before_digest=preview.config_digest,
        after_digest=inspection.config_digest,
        commands=preview.commands,
        inspection=inspection,
    )


def _apply_locked(
    preview: HermesModelConfigPreview,
    env: Mapping[str, str] | None,
) -> HermesModelConfigInspection:
    expected_digest = preview.config_digest
    attempted: list[str] = []
    try:
        for command in preview.commands:
            if _config_digest(preview.config_path) != expected_digest:
                raise ConfigDigestMismatchError(
                    "Hermes config changed during apply; remaining mutations refused"
                )
            alias = command[3].removeprefix("model.aliases.")
            result = _run(command, env)
            if result.returncode != 0:
                observed = _optional_json_mapping(
                    (preview.hermes, "config", "get", "model.aliases", "--json"),
                    env,
                    [],
                )
                target = preview.changes[alias]
                mutation_is_owned = (
                    alias not in observed
                    if target is None
                    else observed.get(alias) == target
                )
                if mutation_is_owned:
                    attempted.append(alias)
                    expected_digest = _config_digest(preview.config_path)
                raise HermesCommandError(
                    f"Hermes config command failed ({result.returncode}): {' '.join(command[1:])}"
                )
            attempted.append(alias)
            expected_digest = _config_digest(preview.config_path)
            _require_expected_alias_state(preview, attempted, env)
        return inspect_hermes_model_config(hermes=preview.hermes, env=env)
    except HermesModelConfigError:
        _rollback_aliases(
            preview,
            attempted,
            env,
            expected_config_digest=expected_digest,
        )
        raise


def _require_expected_alias_state(
    preview: HermesModelConfigPreview,
    attempted: list[str],
    env: Mapping[str, str] | None,
) -> None:
    observed = _optional_json_mapping(
        (preview.hermes, "config", "get", "model.aliases", "--json"),
        env,
        [],
    )
    expected = dict(preview.before_aliases)
    for alias in attempted:
        target = preview.changes[alias]
        if target is None:
            expected.pop(alias, None)
        else:
            expected[alias] = target
    if observed != expected:
        raise ConfigDigestMismatchError(
            "Hermes aliases changed concurrently during apply; rollback required"
        )


def _rollback_aliases(
    preview: HermesModelConfigPreview,
    attempted: list[str],
    env: Mapping[str, str] | None,
    *,
    expected_config_digest: str,
) -> None:
    observed = _optional_json_mapping(
        (preview.hermes, "config", "get", "model.aliases", "--json"),
        env,
        [],
    )
    if any(observed.get(alias) != preview.changes[alias] for alias in attempted):
        raise ConfigDigestMismatchError(
            "Hermes alias ownership changed during apply; rollback refused"
        )
    rollback_error: HermesModelConfigError | None = None
    for alias in reversed(attempted):
        if _config_digest(preview.config_path) != expected_config_digest:
            raise ConfigDigestMismatchError(
                "Hermes config changed during rollback; remaining rollback refused"
            )
        prior = preview.before_aliases.get(alias)
        key = f"model.aliases.{alias}"
        command = (
            (preview.hermes, "config", "unset", key)
            if prior is None
            else (preview.hermes, "config", "set", key, prior)
        )
        result = _run(command, env)
        if result.returncode != 0:
            rollback_error = HermesVerificationError(
                f"Hermes rollback command failed ({result.returncode}): {' '.join(command[1:])}"
            )
        expected_config_digest = _config_digest(preview.config_path)
    inspection = inspect_hermes_model_config(hermes=preview.hermes, env=env)
    for alias in attempted:
        if inspection.model_dot_aliases.get(alias) != preview.before_aliases.get(alias):
            rollback_error = HermesVerificationError(
                f"Hermes alias {alias!r} rollback verification mismatch"
            )
    if rollback_error is not None:
        raise rollback_error


def _run(
    command: tuple[str, ...],
    env: Mapping[str, str] | None,
    commands: list[tuple[str, ...]] | None = None,
) -> subprocess.CompletedProcess[str]:
    if commands is not None:
        commands.append(command)
    try:
        # Hermes forces its stdout to UTF-8 on every platform
        # (hermes_cli/__init__.py), and `auth list` marks the current
        # credential with `←`; the locale codec (cp1252 on Windows) cannot
        # decode those bytes.
        return subprocess.run(
            _platform_command(command),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=None if env is None else dict(env),
            timeout=_COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise HermesCommandError(
            f"Hermes command timed out after {_COMMAND_TIMEOUT_SECONDS}s: {' '.join(command[1:])}"
        ) from exc
    except FileNotFoundError as exc:
        # A machine may run OMH beside a hosted wrapper that manages Hermes,
        # with no `hermes` on PATH. That is a missing prerequisite the caller
        # reports, not a crash.
        raise HermesUnavailableError(f"Hermes command not found: {command[0]}") from exc


def _platform_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if os.name == "nt" and Path(command[0]).suffix.casefold() == ".py":
        return (sys.executable, *command)
    return command


def _required_output(
    command: tuple[str, ...],
    env: Mapping[str, str] | None,
    commands: list[tuple[str, ...]],
) -> str:
    result = _run(command, env, commands)
    if result.returncode != 0:
        raise HermesCommandError(f"Hermes command failed: {' '.join(command[1:])}")
    return result.stdout.strip()


def _optional_json_value(
    command: tuple[str, ...],
    env: Mapping[str, str] | None,
    commands: list[tuple[str, ...]],
) -> str | list[str] | dict[str, str | list[str] | dict[str, str]] | None:
    result = _run(command, env, commands)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise HermesCommandError(f"Hermes returned invalid JSON for {command[2]}") from exc
    if value is None or isinstance(value, (str, list, dict)):
        return value
    raise HermesCommandError(f"Hermes returned unsupported JSON for {command[2]}")


def _optional_json_mapping(
    command: tuple[str, ...],
    env: Mapping[str, str] | None,
    commands: list[tuple[str, ...]],
) -> dict[str, str]:
    value = _optional_json_value(command, env, commands)
    if not isinstance(value, dict):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, str) and _safe_metadata(str(key)) and _safe_metadata(item)
    }


def _config_digest(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError as exc:
        raise HermesConfigUnreadableError(path, missing=True) from exc
    except OSError as exc:
        raise HermesConfigUnreadableError(path, missing=False) from exc


def _auth_provider_ids(output: str) -> set[str]:
    providers: set[str] = set()
    for line in output.splitlines():
        match = _AUTH_PROVIDER_RE.match(line.strip())
        if match and _safe_metadata(match.group(1)):
            providers.add(match.group(1))
    return providers


def _custom_provider_names(
    value: str | list[str] | dict[str, str | list[str] | dict[str, str]] | None,
) -> set[str]:
    """Configured legacy custom provider names, normalized the way Hermes keys their pools."""
    if not isinstance(value, list):
        return set()
    names: set[str] = set()
    for entry in value:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str):
            continue
        normalized = name.strip().lower().replace(" ", "-")
        if _CUSTOM_PROVIDER_NAME_RE.fullmatch(normalized) and _safe_metadata(normalized):
            names.add(normalized)
    return names


def _configured_auth_provider_ids(auth_providers: set[str], custom_names: set[str]) -> set[str]:
    """Credentialed ids an alias target can name.

    Hermes resolves an alias prefix `<name>` or `custom:<name>` to a legacy
    custom provider only while `custom_providers` still configures `<name>`, so
    a `custom:<name>` pool counts for both spellings then, and for neither once
    the entry is gone.
    """
    providers = {
        provider
        for provider in auth_providers
        if not provider.startswith(_CUSTOM_POOL_PREFIX)
        or provider.removeprefix(_CUSTOM_POOL_PREFIX) in custom_names
    }
    providers.update(
        name for name in custom_names if f"{_CUSTOM_POOL_PREFIX}{name}" in auth_providers
    )
    return providers


def _plugin_provider_ids(
    value: str | list[str] | dict[str, str | list[str] | dict[str, str]] | None,
) -> set[str]:
    if not isinstance(value, dict):
        return set()
    providers: set[str] = set()
    enabled = value.get("enabled")
    if isinstance(enabled, list):
        providers.update(str(item) for item in enabled if _safe_metadata(str(item)))
    entries = value.get("entries")
    if isinstance(entries, dict):
        providers.update(str(item) for item in entries if _safe_metadata(str(item)))
    return providers


def _alias_provider_ids(aliases: Mapping[str, str]) -> set[str]:
    return {
        provider
        for target in aliases.values()
        if "/" in target and _safe_metadata(provider := target.split("/", 1)[0])
    }


def _provider_has_auth(inspection: HermesModelConfigInspection, provider_id: str) -> bool:
    return any(
        provider.provider_id == provider_id
        and provider.auth_present
        and provider.auth_status_ok
        for provider in inspection.providers
    )


def _safe_metadata(value: str) -> bool:
    return bool(value) and not is_sensitive_metadata_text(value)
