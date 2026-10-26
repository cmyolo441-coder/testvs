"""Allowlisted provider capabilities and the config identity of a check.

This is the capability half of the `diagnostic_providers/v1` contract: what
each allowlisted provider declares it can do (languages, file suffixes, and
per-provider bounds capped by the global bounds), and the derived identity of
one exact capability set, so a check names the configuration it ran under.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .diagnostic_provider_models import (
    DIAGNOSTIC_PROVIDER_IDS,
    GLOBAL_MAX_DIAGNOSTICS_PER_CHECK,
    GLOBAL_MAX_FILES_PER_CHECK,
    GLOBAL_MAX_TIMEOUT_MS,
    MIN_TIMEOUT_MS,
    DiagnosticProviderError,
    _checked_ref,
)


@dataclass(frozen=True)
class ProviderCapability:
    """One allowlisted provider's declared capability and bounds.

    The bounds are the provider's own ceiling, capped by the global bounds:
    `max_timeout_ms` is the longest a check may claim to have waited,
    `max_diagnostics_per_check` the most diagnostics one check may carry, and
    `max_files_per_check` the largest changed-file scope it accepts. A
    disabled capability is still part of the config identity, because which
    providers are off is part of what a check ran under.
    """

    provider_id: str
    languages: tuple[str, ...]
    file_suffixes: tuple[str, ...]
    max_timeout_ms: int
    max_diagnostics_per_check: int
    max_files_per_check: int
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.provider_id not in DIAGNOSTIC_PROVIDER_IDS:
            raise DiagnosticProviderError(
                f"diagnostic_providers provider_id is not allowlisted: {self.provider_id!r}"
            )
        for name, value, low, high in (
            ("max_timeout_ms", self.max_timeout_ms, MIN_TIMEOUT_MS, GLOBAL_MAX_TIMEOUT_MS),
            ("max_diagnostics_per_check", self.max_diagnostics_per_check, 1, GLOBAL_MAX_DIAGNOSTICS_PER_CHECK),
            ("max_files_per_check", self.max_files_per_check, 1, GLOBAL_MAX_FILES_PER_CHECK),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
                raise DiagnosticProviderError(
                    f"diagnostic_providers {name} must be an integer between {low} and {high}"
                )
        languages = tuple(sorted({language.strip().lower() for language in self.languages}))
        if not languages:
            raise DiagnosticProviderError("diagnostic_providers languages must name at least one language")
        for language in languages:
            _checked_ref(language, field="diagnostic_providers languages", required=True)
        suffixes = tuple(sorted({suffix.strip().lower() for suffix in self.file_suffixes}))
        if not suffixes:
            raise DiagnosticProviderError("diagnostic_providers file_suffixes must name at least one suffix")
        for suffix in suffixes:
            if (
                not suffix.startswith(".")
                or len(suffix) < 2
                or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789." for character in suffix)
            ):
                raise DiagnosticProviderError(
                    f"diagnostic_providers file_suffixes must be dotted suffixes like '.py': {suffix!r}"
                )
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "file_suffixes", suffixes)


DEFAULT_PROVIDER_CAPABILITIES: tuple[ProviderCapability, ...] = (
    ProviderCapability("pyright", ("python",), (".py", ".pyi"), 120_000, 200, 200),
    ProviderCapability("basedpyright", ("python",), (".py", ".pyi"), 60_000, 200, 200),
    ProviderCapability("mypy", ("python",), (".py", ".pyi"), 90_000, 100, 100),
    ProviderCapability("ruff", ("python",), (".py", ".pyi"), 30_000, 200, 200),
)


def _files_in_scope(files: tuple[str, ...], capability: ProviderCapability) -> tuple[str, ...]:
    return tuple(path for path in files if path.endswith(capability.file_suffixes))


@dataclass(frozen=True)
class DiagnosticProviderConfig:
    """A set of capabilities plus the identity of that exact set.

    `config_identity` is derived, never supplied: a check names the bounds,
    suffixes, and enabled state it ran under, and the same set in a different
    order is the same config because it selects the same provider.
    """

    capabilities: tuple[ProviderCapability, ...] = DEFAULT_PROVIDER_CAPABILITIES

    def __post_init__(self) -> None:
        if not self.capabilities:
            raise DiagnosticProviderError("diagnostic_providers config needs at least one provider capability")
        provider_ids = [capability.provider_id for capability in self.capabilities]
        if len(set(provider_ids)) != len(provider_ids):
            raise DiagnosticProviderError(
                "diagnostic_providers config cannot carry two capabilities for one provider"
            )
        object.__setattr__(self, "capabilities", tuple(self.capabilities))

    def capability_for(self, provider_id: str) -> ProviderCapability | None:
        for capability in self.capabilities:
            if capability.provider_id == provider_id:
                return capability
        return None

    def select_provider(
        self, changed_files: tuple[str, ...], unavailable: tuple[str, ...] = ()
    ) -> ProviderCapability | None:
        """The first allowlisted, enabled, in-bound provider for this scope.

        Selection order is the allowlist order, not the capabilities order, so
        a caller cannot reorder its way to a different provider. Providers
        that are disabled, marked unavailable, or whose file bound is smaller
        than the in-scope file count are skipped: that skip is the fallback.
        """
        for provider_id in DIAGNOSTIC_PROVIDER_IDS:
            capability = self.capability_for(provider_id)
            if capability is None or not capability.enabled:
                continue
            if provider_id in unavailable:
                continue
            in_scope = _files_in_scope(changed_files, capability)
            if not in_scope:
                continue
            if len(in_scope) > capability.max_files_per_check:
                continue
            return capability
        return None

    def config_identity(self) -> str:
        rows = []
        for capability in sorted(self.capabilities, key=lambda item: item.provider_id):
            rows.append(
                "|".join(
                    (
                        capability.provider_id,
                        ",".join(capability.languages),
                        ",".join(capability.file_suffixes),
                        str(capability.max_timeout_ms),
                        str(capability.max_diagnostics_per_check),
                        str(capability.max_files_per_check),
                        str(capability.enabled),
                    )
                )
            )
        digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()[:16]
        return f"provdiag-{digest}"
