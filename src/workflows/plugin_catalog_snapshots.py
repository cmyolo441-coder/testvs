"""Host plugin catalog snapshots (`plugin_catalog_snapshot/v1`, issue #1449).

OMH ships one packaged ecosystem catalog and a hand-curated outcome matrix.
Both are pinned to a repository revision, so a plugin the host admitted this
morning, a reviewed revision that changed underneath an identity, a removal,
or a host-version constraint the host refused is invisible until a maintainer
edits the repository. This contract is the input that makes those visible
without OMH acquiring anything.

A snapshot is produced by an authorized host or operator and handed to OMH as
a file. OMH validates it, reconciles it (see
`omh.workflows.plugin_catalog_coverage`), and reports. It fetches nothing,
reads no credential, imports no plugin, installs no dependency, and changes no
host configuration -- the file is the only input, and reading it is the only
thing that happens.

Three properties carry the contract:

- **Identity and content revision are separate fields.** `plugin_id` is the
  stable identity the host uses; `content_revision` is the reviewed content
  behind it at snapshot time. Keeping them apart is what lets a reviewed
  revision change read as `changed` for a known identity rather than as a new
  plugin, and it is why both revisions survive into the coverage row.
- **Every field is untrusted metadata.** Identifiers come from outside OMH, so
  each one is screened as a bounded opaque reference. `required_env_names`
  carries environment variable *names* -- the one field whose legitimate
  content contains the words `TOKEN` and `SECRET` -- so it is screened with
  `is_secret_value_shaped`, which flags issued credential material and never a
  name. There is no field a credential value, a file body, or a repository
  excerpt can arrive in.
- **A catalog entry is a review signal, not a security guarantee.** Declared
  capabilities are what the entry claims, recorded as claims. Nothing here is
  evidence that a plugin behaves the way its entry says, that the host loaded
  it, or that anyone ran it.

Diagnostics are bounded and positional. A violation names the entry index and
the field, never the offending value, so a malformed snapshot cannot use OMH's
own error path to print what it was carrying.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from ..system.metadata_safety import (
    is_body_shaped_metadata_text,
    is_raw_pii_shaped,
    is_secret_value_shaped,
    is_sensitive_metadata_text,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


PLUGIN_CATALOG_SNAPSHOT_SCHEMA_VERSION: Final = "plugin_catalog_snapshot/v1"

SNAPSHOT_CLAIM_BOUNDARY: Final = (
    "A plugin catalog snapshot is a bounded, point-in-time record an authorized host or operator "
    "produced about its own catalog. It is not plugin installation, admission, activation, permission, "
    "compatibility enforcement, or execution evidence, and OMH acquired none of it."
)

#: Who may produce a snapshot. Both are outside OMH: the host reports its own
#: catalog, or an operator exports it. OMH never produces one.
SNAPSHOT_PRODUCER_KINDS: Final = ("hermes_host", "authorized_operator")

#: What the host says about the entry's review tier. `unreviewed` is a real
#: value, not a missing one -- an admitted-but-unreviewed entry still routes.
ENTRY_TIERS: Final = ("official", "verified", "community", "unreviewed")

#: The host's own answer to its own constraint. OMH does not parse, compare, or
#: solve `host_version_constraint`; version enforcement is the host's job and
#: this field is where the host reports the result of doing it.
HOST_VERSION_STATUSES: Final = ("satisfied", "unsatisfied", "unknown")

#: `removed` means the catalog withdrew the entry. It says nothing about an
#: installed copy on the machine, which OMH cannot see.
REMOVAL_STATUSES: Final = ("active", "deprecated", "removed")

#: Platform shapes an entry declares it is limited to. Closed, because an
#: open platform vocabulary would make "desktop only" unrecognisable.
PLATFORM_LIMITS: Final = ("desktop", "headless", "linux", "macos", "windows")

#: Bounds. A snapshot is metadata about a catalog, not a catalog dump, and
#: every one of these is a refusal rather than a truncation: silently dropping
#: the tail of an oversized snapshot would report coverage over a subset while
#: claiming to cover the whole catalog.
MAX_SNAPSHOT_BYTES: Final = 512 * 1024
MAX_SNAPSHOT_ENTRIES: Final = 512
MAX_ENTRY_CAPABILITIES: Final = 16
MAX_ENTRY_ENV_NAMES: Final = 32
MAX_ENTRY_PLATFORM_LIMITS: Final = len(PLATFORM_LIMITS)
MAX_METADATA_TEXT: Final = 160
#: How many diagnostics a caller sees before the list is closed with a count.
#: A snapshot with a thousand broken entries must not print a thousand lines.
MAX_DIAGNOSTICS: Final = 20

#: A catalog moves when the host admits, revises, or withdraws an entry, which
#: is a slower clock than a live connector but not a still one. A day-old
#: snapshot is reported as stale rather than as the current catalog.
SNAPSHOT_STALE_AFTER_SECONDS: Final = 24 * 60 * 60

_ENVELOPE_KEYS: Final = (
    "schema_version",
    "producer",
    "profile_ref",
    "observed_at",
    "catalog_revision",
    "entry_count",
    "entries",
)
_PRODUCER_KEYS: Final = ("kind", "ref", "host_version")
_ENTRY_KEYS: Final = (
    "plugin_id",
    "content_revision",
    "tier",
    "declared_capabilities",
    "required_env_names",
    "platform_limits",
    "host_version_constraint",
    "host_version_status",
    "removal_status",
)

_OPAQUE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}$")
_CAPABILITY_TOKEN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class PluginCatalogSnapshotError(ValueError):
    """A snapshot could not be read at all, before any field was inspected."""


def validate_plugin_catalog_snapshot(raw: object) -> list[str]:
    """Bounded diagnostics for one candidate snapshot; empty means it parsed.

    Positional by design: every diagnostic names an entry index and a field
    name and never the value that failed, so an untrusted snapshot cannot
    route its own content through OMH's error output.
    """
    diagnostics: list[str] = []
    if not isinstance(raw, Mapping):
        return ["snapshot must be a JSON object"]
    _check_key_set(raw, _ENVELOPE_KEYS, "snapshot", diagnostics)
    if raw.get("schema_version") != PLUGIN_CATALOG_SNAPSHOT_SCHEMA_VERSION:
        diagnostics.append(f"snapshot.schema_version must be {PLUGIN_CATALOG_SNAPSHOT_SCHEMA_VERSION}")
    _validate_producer(raw.get("producer"), diagnostics)
    _check_opaque(raw.get("profile_ref"), "snapshot.profile_ref", diagnostics)
    _check_opaque(raw.get("catalog_revision"), "snapshot.catalog_revision", diagnostics)
    if _parse_stamp(raw.get("observed_at")) is None:
        diagnostics.append("snapshot.observed_at must be an ISO-8601 timestamp")
    diagnostics.extend(_validate_entries(raw.get("entries"), raw.get("entry_count")))
    return _bounded(diagnostics)


def load_plugin_catalog_snapshot(path: str | Path) -> dict[str, Any]:
    """Read one snapshot file, refusing an oversized or unparsable one.

    The byte cap is enforced on the file before `json.loads` sees it, so an
    oversized input is refused rather than parsed and then measured.
    """
    resolved = Path(path).expanduser()
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise PluginCatalogSnapshotError(f"snapshot file could not be read: {exc.strerror or 'unavailable'}") from exc
    if size > MAX_SNAPSHOT_BYTES:
        raise PluginCatalogSnapshotError(
            f"snapshot file exceeds the {MAX_SNAPSHOT_BYTES}-byte bound and was not parsed"
        )
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise PluginCatalogSnapshotError(f"snapshot file could not be read: {exc.strerror or 'unavailable'}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PluginCatalogSnapshotError(f"snapshot file is not valid JSON at line {exc.lineno}") from exc
    if not isinstance(parsed, dict):
        raise PluginCatalogSnapshotError("snapshot file must contain a JSON object")
    return parsed


def snapshot_entries(snapshot: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Entries of a validated snapshot, sorted by identity."""
    entries = snapshot.get("entries")
    if not isinstance(entries, Sequence):
        return ()
    return tuple(sorted((dict(entry) for entry in entries if isinstance(entry, Mapping)), key=_entry_identity))


def snapshot_freshness(snapshot: Mapping[str, Any], *, now: str = "") -> dict[str, Any]:
    """Whether the snapshot still describes the catalog it was taken from.

    An unreadable reference clock reports `unknown` rather than `fresh`: a
    snapshot whose age nobody could compute has not been shown to be current.
    """
    observed = _parse_stamp(snapshot.get("observed_at"))
    reference = _parse_stamp(now) if now.strip() else datetime.now(timezone.utc)
    if observed is None or reference is None:
        return {
            "state": "unknown",
            "age_seconds": None,
            "stale_after_seconds": SNAPSHOT_STALE_AFTER_SECONDS,
            "evaluated_at": _format_stamp(reference),
        }
    age = int((reference - observed).total_seconds())
    state = "stale" if age > SNAPSHOT_STALE_AFTER_SECONDS or age < 0 else "fresh"
    return {
        "state": state,
        "age_seconds": age,
        "stale_after_seconds": SNAPSHOT_STALE_AFTER_SECONDS,
        "evaluated_at": _format_stamp(reference),
    }


def snapshot_identity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """The producer, profile, time, and revision one coverage answer is bound to."""
    producer = snapshot.get("producer")
    producer_map = producer if isinstance(producer, Mapping) else {}
    return {
        "producer_kind": str(producer_map.get("kind", "")),
        "producer_ref": str(producer_map.get("ref", "")),
        "host_version": str(producer_map.get("host_version", "")),
        "profile_ref": str(snapshot.get("profile_ref", "")),
        "observed_at": str(snapshot.get("observed_at", "")),
        "catalog_revision": str(snapshot.get("catalog_revision", "")),
    }


def _validate_producer(raw: object, diagnostics: list[str]) -> None:
    if not isinstance(raw, Mapping):
        diagnostics.append("snapshot.producer must be an object")
        return
    _check_key_set(raw, _PRODUCER_KEYS, "snapshot.producer", diagnostics)
    if raw.get("kind") not in SNAPSHOT_PRODUCER_KINDS:
        diagnostics.append("snapshot.producer.kind must name an authorized host or operator")
    _check_opaque(raw.get("ref"), "snapshot.producer.ref", diagnostics)
    _check_opaque(raw.get("host_version"), "snapshot.producer.host_version", diagnostics)


def _validate_entries(raw: object, declared_count: object) -> list[str]:
    diagnostics: list[str] = []
    if not isinstance(raw, list):
        return ["snapshot.entries must be a list"]
    if len(raw) > MAX_SNAPSHOT_ENTRIES:
        return [f"snapshot.entries exceeds the {MAX_SNAPSHOT_ENTRIES}-entry bound"]
    if declared_count != len(raw):
        diagnostics.append("snapshot.entry_count does not match the number of entries")
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        diagnostics.extend(_validate_entry(entry, index))
        if isinstance(entry, Mapping):
            identity = entry.get("plugin_id")
            if isinstance(identity, str):
                if identity in seen:
                    diagnostics.append(f"entries[{index}].plugin_id repeats an identity already in this snapshot")
                seen.add(identity)
    return diagnostics


def _validate_entry(raw: object, index: int) -> list[str]:
    diagnostics: list[str] = []
    label = f"entries[{index}]"
    if not isinstance(raw, Mapping):
        return [f"{label} must be an object"]
    _check_key_set(raw, _ENTRY_KEYS, label, diagnostics)
    _check_opaque(raw.get("plugin_id"), f"{label}.plugin_id", diagnostics)
    _check_opaque(raw.get("content_revision"), f"{label}.content_revision", diagnostics)
    if raw.get("tier") not in ENTRY_TIERS:
        diagnostics.append(f"{label}.tier must be one of {', '.join(ENTRY_TIERS)}")
    if raw.get("host_version_status") not in HOST_VERSION_STATUSES:
        diagnostics.append(f"{label}.host_version_status must be one of {', '.join(HOST_VERSION_STATUSES)}")
    if raw.get("removal_status") not in REMOVAL_STATUSES:
        diagnostics.append(f"{label}.removal_status must be one of {', '.join(REMOVAL_STATUSES)}")
    _check_token_list(
        raw.get("declared_capabilities"),
        f"{label}.declared_capabilities",
        _CAPABILITY_TOKEN,
        MAX_ENTRY_CAPABILITIES,
        diagnostics,
    )
    _check_env_names(raw.get("required_env_names"), f"{label}.required_env_names", diagnostics)
    _check_platform_limits(raw.get("platform_limits"), f"{label}.platform_limits", diagnostics)
    _check_constraint(raw.get("host_version_constraint"), f"{label}.host_version_constraint", diagnostics)
    return diagnostics


def _check_key_set(raw: Mapping[str, Any], expected: tuple[str, ...], label: str, diagnostics: list[str]) -> None:
    present = set(raw)
    missing = sorted(set(expected) - present)
    unexpected = sorted(present - set(expected))
    if missing:
        diagnostics.append(f"{label} is missing required fields: {', '.join(missing)}")
    if unexpected:
        diagnostics.append(f"{label} carries unsupported fields: {', '.join(unexpected)}")


def _check_opaque(value: object, label: str, diagnostics: list[str]) -> None:
    if not isinstance(value, str) or not _OPAQUE_REF.fullmatch(value):
        diagnostics.append(f"{label} must be a bounded opaque reference")
        return
    if is_sensitive_metadata_text(value) or is_raw_pii_shaped(value):
        diagnostics.append(f"{label} must not carry credential or personal-identifier text")


def _check_constraint(value: object, label: str, diagnostics: list[str]) -> None:
    """The host's constraint expression, kept as opaque text OMH never parses."""
    if not isinstance(value, str):
        diagnostics.append(f"{label} must be a string")
        return
    if not value:
        return
    if is_body_shaped_metadata_text(value, limit=MAX_METADATA_TEXT):
        diagnostics.append(f"{label} must be one bounded line")
        return
    if is_sensitive_metadata_text(value) or is_raw_pii_shaped(value):
        diagnostics.append(f"{label} must not carry credential or personal-identifier text")


def _check_token_list(
    value: object,
    label: str,
    pattern: re.Pattern[str],
    bound: int,
    diagnostics: list[str],
) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        diagnostics.append(f"{label} must be a list of strings")
        return
    items = [str(item) for item in value]
    if len(items) > bound:
        diagnostics.append(f"{label} exceeds the {bound}-item bound")
        return
    if items != sorted(items) or len(items) != len(set(items)):
        diagnostics.append(f"{label} must be sorted and unique")
    if any(not pattern.fullmatch(item) for item in items):
        diagnostics.append(f"{label} contains an unsupported token shape")


def _check_env_names(value: object, label: str, diagnostics: list[str]) -> None:
    """Environment variable *names*, never values.

    A name legitimately reads as `GITHUB_TOKEN` or `OPENAI_API_KEY`, so the
    wider sensitive-text screen would refuse every honest snapshot here. The
    narrow screen flags issued credential material -- prefixed keys, AWS access
    key ids -- which a variable name can never match.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        diagnostics.append(f"{label} must be a list of strings")
        return
    items = [str(item) for item in value]
    if len(items) > MAX_ENTRY_ENV_NAMES:
        diagnostics.append(f"{label} exceeds the {MAX_ENTRY_ENV_NAMES}-item bound")
        return
    if items != sorted(items) or len(items) != len(set(items)):
        diagnostics.append(f"{label} must be sorted and unique")
    if any(not _ENV_NAME.fullmatch(item) for item in items):
        diagnostics.append(f"{label} must contain environment variable names only")
        return
    if any(is_secret_value_shaped(item) for item in items):
        diagnostics.append(f"{label} contains a credential value rather than a variable name")


def _check_platform_limits(value: object, label: str, diagnostics: list[str]) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        diagnostics.append(f"{label} must be a list of strings")
        return
    items = [str(item) for item in value]
    if len(items) > MAX_ENTRY_PLATFORM_LIMITS:
        diagnostics.append(f"{label} exceeds the {MAX_ENTRY_PLATFORM_LIMITS}-item bound")
        return
    if items != sorted(items) or len(items) != len(set(items)):
        diagnostics.append(f"{label} must be sorted and unique")
    if any(item not in PLATFORM_LIMITS for item in items):
        diagnostics.append(f"{label} must use platform limits from {', '.join(PLATFORM_LIMITS)}")


def _bounded(diagnostics: list[str]) -> list[str]:
    if len(diagnostics) <= MAX_DIAGNOSTICS:
        return diagnostics
    hidden = len(diagnostics) - MAX_DIAGNOSTICS
    return [*diagnostics[:MAX_DIAGNOSTICS], f"... {hidden} further diagnostics were not listed"]


def _entry_identity(entry: Mapping[str, Any]) -> str:
    identity = entry.get("plugin_id")
    return identity if isinstance(identity, str) else ""


def _parse_stamp(value: object) -> datetime | None:
    """An aware UTC datetime from an ISO-8601 timestamp, else None."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = strict_fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_stamp(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "ENTRY_TIERS",
    "HOST_VERSION_STATUSES",
    "MAX_DIAGNOSTICS",
    "MAX_ENTRY_CAPABILITIES",
    "MAX_ENTRY_ENV_NAMES",
    "MAX_SNAPSHOT_BYTES",
    "MAX_SNAPSHOT_ENTRIES",
    "PLATFORM_LIMITS",
    "PLUGIN_CATALOG_SNAPSHOT_SCHEMA_VERSION",
    "REMOVAL_STATUSES",
    "SNAPSHOT_CLAIM_BOUNDARY",
    "SNAPSHOT_PRODUCER_KINDS",
    "SNAPSHOT_STALE_AFTER_SECONDS",
    "PluginCatalogSnapshotError",
    "load_plugin_catalog_snapshot",
    "snapshot_entries",
    "snapshot_freshness",
    "snapshot_identity",
    "validate_plugin_catalog_snapshot",
]
