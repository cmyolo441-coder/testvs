"""Normalization and derivation of diagnostic-provider values.

This is the input-processing half of the `diagnostic_providers/v1` contract:
one diagnostic item (metadata only, refused by key name if a raw payload
field arrives), the workspace-relative file-scope normalization, the
derivation that classifies a check into its outcome, and the derived outcome
identity. Nothing here runs a provider or reads a file; everything here turns
what a caller reports into the normalized values the rest of the contract
reasons over.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..quality.language_diagnostic_evidence import LANGUAGE_DIAGNOSTIC_SEVERITIES, MAX_POSITION
from .diagnostic_provider_models import (
    DIAGNOSTIC_ITEM_KEYS,
    DIAGNOSTIC_PROVIDER_IDS,
    DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION,
    GLOBAL_MAX_DIAGNOSTICS_PER_CHECK,
    DiagnosticProviderError,
    _checked_ref,
    revisions_identical,
)


@dataclass(frozen=True)
class DiagnosticItem:
    """One normalized diagnostic, metadata only.

    Reuses the `language_diagnostic_evidence/v1` item vocabulary exactly:
    severity, code, path, line, character, source. `source` must name an
    allowlisted provider -- it is the analyser that produced the item, never
    a source body -- and an input carrying any other key is refused by name.
    """

    severity: str
    code: str
    path: str
    line: int
    character: int
    source: str

    def __post_init__(self) -> None:
        severity = str(self.severity).strip().lower()
        if severity not in LANGUAGE_DIAGNOSTIC_SEVERITIES:
            raise DiagnosticProviderError(
                f"diagnostic_providers diagnostics severity is unsupported: {self.severity!r}"
            )
        code = _checked_ref(str(self.code), field="diagnostic_providers diagnostics code", required=False)
        path = _normalized_workspace_path(str(self.path), field="diagnostic_providers diagnostics path")
        for name, value in (("line", self.line), ("character", self.character)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_POSITION:
                raise DiagnosticProviderError(
                    f"diagnostic_providers diagnostics {name} must be an integer offset between 0 and {MAX_POSITION}"
                )
        if self.source not in DIAGNOSTIC_PROVIDER_IDS:
            raise DiagnosticProviderError(
                "diagnostic_providers diagnostics source must name an allowlisted provider, "
                f"never file content: {self.source!r}"
            )
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "path", path)

    def as_record(self) -> dict[str, object]:
        return {
            "character": self.character,
            "code": self.code,
            "line": self.line,
            "path": self.path,
            "severity": self.severity,
            "source": self.source,
        }


def normalize_diagnostic_item(value: Mapping[str, object]) -> DiagnosticItem:
    """Build one item, or refuse the keys a raw payload would arrive under."""
    extra = sorted(str(key) for key in value if str(key) not in DIAGNOSTIC_ITEM_KEYS)
    if extra:
        raise DiagnosticProviderError(
            f"diagnostic_providers diagnostics entries carry unsupported keys: {extra}; "
            f"a diagnostic is metadata only ({', '.join(DIAGNOSTIC_ITEM_KEYS)})"
        )
    return DiagnosticItem(
        severity=str(value.get("severity", "") or ""),
        code=str(value.get("code", "") or ""),
        path=str(value.get("path", "") or ""),
        line=_int_value(value.get("line", 0), field="diagnostic_providers diagnostics line"),
        character=_int_value(value.get("character", 0), field="diagnostic_providers diagnostics character"),
        source=str(value.get("source", "") or ""),
    )


def _normalized_workspace_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiagnosticProviderError(f"diagnostic_providers {field} must be a nonblank path")
    text = value.strip().replace("\\", "/")
    if text.startswith("/") or (len(text) > 2 and text[1] == ":" and text[0].isalpha()):
        raise DiagnosticProviderError(
            f"diagnostic_providers {field} must be workspace-relative, not absolute: {value!r}"
        )
    parts = [part for part in text.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise DiagnosticProviderError(f"diagnostic_providers {field} must stay inside the workspace: {value!r}")
    return _checked_ref("/".join(parts), field=f"diagnostic_providers {field}", required=True)


def _normalized_files(values: Iterable[str], *, field: str, bound: int) -> tuple[str, ...]:
    paths = sorted({_normalized_workspace_path(value, field=f"diagnostic_providers {field}") for value in values})
    if len(paths) > bound:
        raise DiagnosticProviderError(f"diagnostic_providers {field} must have at most {bound} entries")
    return tuple(paths)


def _int_value(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticProviderError(f"diagnostic_providers {field} must be an integer")
    return value


def _normalized_diagnostics(
    values: Iterable[Mapping[str, object]], *, scope: set[str]
) -> tuple[DiagnosticItem, ...]:
    items = [normalize_diagnostic_item(value) for value in values]
    unique = sorted(
        set(items),
        key=lambda item: (item.character, item.code, item.line, item.path, item.severity, item.source),
    )
    if len(unique) > GLOBAL_MAX_DIAGNOSTICS_PER_CHECK:
        raise DiagnosticProviderError(
            f"diagnostic_providers diagnostics must have at most {GLOBAL_MAX_DIAGNOSTICS_PER_CHECK} entries"
        )
    for item in unique:
        if item.path not in scope:
            raise DiagnosticProviderError(
                f"diagnostic_providers diagnostics must point inside in_scope_files: {item.path!r}"
            )
    return tuple(unique)


def _derive_outcome(
    compatibility: str,
    provider_id: str,
    terminal_state: str,
    revision: str,
    diagnostics_revision: str,
    in_scope_files: tuple[str, ...],
    diagnosed_files: tuple[str, ...],
) -> str:
    if compatibility == "provider_disabled":
        return "unsupported"
    if terminal_state != "completed":
        return terminal_state
    if not provider_id:
        return "unsupported"
    if not revisions_identical(revision, diagnostics_revision):
        return "stale"
    if set(diagnosed_files) != set(in_scope_files):
        return "partial"
    return "ok"


def _outcome_id(
    *,
    workspace_id: str,
    revision: str,
    diagnostics_revision: str,
    provider_id: str,
    terminal_state: str,
    outcome: str,
    compatibility: str,
    in_scope_files: tuple[str, ...],
    out_of_scope_files: tuple[str, ...],
    diagnosed_files: tuple[str, ...],
    diagnostics: tuple[DiagnosticItem, ...],
    config_identity: str,
) -> str:
    """Identity of which check, derived from the fields a reader can re-read."""
    parts = [
        DIAGNOSTIC_PROVIDERS_SCHEMA_VERSION,
        workspace_id,
        revision,
        diagnostics_revision,
        provider_id,
        terminal_state,
        outcome,
        compatibility,
        config_identity,
    ]
    for label, files in (("in", in_scope_files), ("out", out_of_scope_files), ("diagnosed", diagnosed_files)):
        parts.append(label)
        parts.extend(files)
    parts.append("diagnostics")
    parts.extend(
        ":".join((str(item.character), item.code, str(item.line), item.path, item.severity, item.source))
        for item in diagnostics
    )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"diagout-{digest}"
