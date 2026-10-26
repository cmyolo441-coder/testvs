"""Operator-configured category->model chains for the Maestro dispatch lane.

The Hermes-native delegation lane already routes per work category through an
editable mixture (`omh_delegate_route`). This module gives the Maestro lane
the same dial: `~/.omh/routing/category-maestro.json` overrides individual
categories of `BUILTIN_CATEGORY_MODELS` for the dispatchable CLI profiles, and
`resolve_model_route` consults the merged table wherever it would consult the
built-in one — explicit categories, role chains, research depth, task scale.

Presence of the file IS the opt-in. The built-in table is a package default
that must stay byte-identical across machines; this file is the one sanctioned
per-machine divergence, and every route resolved against it records
`catalog_kind: "operator_category_config"` plus the config fingerprint so a
frozen contract names the exact basis it was resolved from.

Reading never raises: a missing, unreadable, or malformed document reads the
same as an absent one (mirroring the dispatch-model preference file), because
a broken config must not block a prepare or dispatch. Invalid pieces are
dropped and named in `rejected` so `omh coding category-maestro show` can
report them instead of silently swallowing a typo.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Final, Mapping

from ..system.local_store import atomic_write_json, read_json_object_result
from .model_routing import (
    BUILTIN_CATEGORY_MODELS,
    MODEL_CATEGORIES,
    NON_GENERATIVE_MODEL_CLASS,
    canonical_model_category,
    model_class,
)

CATEGORY_MAESTRO_SCHEMA_VERSION: Final[str] = "omh_category_maestro/v1"

# Only profiles with a built-in category table are configurable here: the
# catalogless profiles already have the local-inventory path, and layering a
# second override source over that one would make "which basis resolved this
# route" unanswerable.
CATEGORY_MAESTRO_PROFILES: Final[tuple[str, ...]] = tuple(sorted(BUILTIN_CATEGORY_MODELS))

# Effort values are embedded into executor argv composites, so they get the
# same closed shape gate the resolver applies to requested efforts.
_EFFORT_CHARSET: Final[frozenset[str]] = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
_MAX_CHAIN_ENTRIES: Final[int] = 8
_MAX_MODEL_ID_CHARS: Final[int] = 120
_MAX_EFFORT_CHARS: Final[int] = 32


def category_maestro_path(omh_home: Path) -> Path:
    return Path(omh_home) / "routing" / "category-maestro.json"


def _config_fingerprint(profiles: Mapping[str, object]) -> dict[str, str]:
    canonical = json.dumps(profiles, sort_keys=True, separators=(",", ":"))
    return {
        "source": "category-maestro.json",
        "digest": sha256(canonical.encode("utf-8")).hexdigest()[:16],
    }


def _validated_entry(entry: object, *, where: str, rejected: list[str]) -> dict[str, str] | None:
    if not isinstance(entry, Mapping):
        rejected.append(f"{where}: chain entry must be an object with model_id")
        return None
    model_id = str(entry.get("model_id", "") or "").strip()
    # A leading `-` is rejected because selected_model lands in the spawned
    # CLI's argv as its own token: never allow a "model id" that would parse
    # as a flag there.
    if (
        not model_id
        or len(model_id) > _MAX_MODEL_ID_CHARS
        or model_id.startswith("-")
        or any(char.isspace() for char in model_id)
    ):
        rejected.append(
            f"{where}: model_id must be a non-empty token without whitespace or a leading '-'"
        )
        return None
    # Shape alone cannot catch this one: the token is well formed and the
    # model is real, but a non-generative model answers typed questions and
    # will never write the unit a category chain hands it. Named in
    # `rejected` like every other dropped piece, so `omh coding
    # category-maestro show` reports it instead of the category silently
    # routing to a model that cannot do the work.
    if model_class(model_id) == NON_GENERATIVE_MODEL_CLASS:
        rejected.append(
            f"{where}: model_id {model_id!r} is a {NON_GENERATIVE_MODEL_CLASS} model "
            "(it answers typed questions and cannot write code)"
        )
        return None
    effort = str(entry.get("reasoning_effort", "") or "").strip().casefold()
    if effort and (len(effort) > _MAX_EFFORT_CHARS or not set(effort) <= _EFFORT_CHARSET):
        rejected.append(f"{where}: reasoning_effort {effort!r} is not an effort-shaped value")
        return None
    return {"model_id": model_id, "reasoning_effort": effort}


def read_category_maestro_config(omh_home: Path) -> dict[str, object] | None:
    """Return the validated operator category config, or None when absent.

    The returned payload carries only validated data: `profiles` maps a
    dispatchable profile to canonical categories with non-empty entry chains,
    `fingerprint` names the exact accepted basis, and `rejected` lists every
    dropped piece by name. A document that validates down to nothing returns
    None so callers treat it exactly like an absent file.
    """
    document, _error = read_json_object_result(category_maestro_path(omh_home))
    if not isinstance(document, dict):
        return None
    if document.get("schema_version") != CATEGORY_MAESTRO_SCHEMA_VERSION:
        return None
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        return None
    rejected: list[str] = []
    profiles: dict[str, dict[str, tuple[dict[str, str], ...]]] = {}
    for raw_profile, raw_categories in raw_profiles.items():
        profile = str(raw_profile or "").strip().casefold()
        if profile not in CATEGORY_MAESTRO_PROFILES:
            rejected.append(
                f"profile {raw_profile!r} is not configurable here; expected one of "
                f"{', '.join(CATEGORY_MAESTRO_PROFILES)}"
            )
            continue
        if not isinstance(raw_categories, Mapping):
            rejected.append(f"profile {profile!r}: categories must be an object")
            continue
        categories: dict[str, tuple[dict[str, str], ...]] = {}
        for raw_category, raw_entries in raw_categories.items():
            category = canonical_model_category(raw_category)
            if not category:
                rejected.append(
                    f"profile {profile!r}: category {raw_category!r} is not one of "
                    f"{', '.join(MODEL_CATEGORIES)}"
                )
                continue
            if not isinstance(raw_entries, (list, tuple)):
                rejected.append(f"profile {profile!r} category {category!r}: chain must be a list")
                continue
            if len(raw_entries) > _MAX_CHAIN_ENTRIES:
                rejected.append(
                    f"profile {profile!r} category {category!r}: chain longer than "
                    f"{_MAX_CHAIN_ENTRIES} entries"
                )
                continue
            entries = [
                validated
                for position, raw_entry in enumerate(raw_entries)
                if (
                    validated := _validated_entry(
                        raw_entry,
                        where=f"profile {profile!r} category {category!r} entry {position}",
                        rejected=rejected,
                    )
                )
                is not None
            ]
            if not entries:
                # An empty (or fully-rejected) chain is treated as unset, not
                # as "no chain": clearing a category back to the built-in is
                # `clear`, and a typo must not silently disable a category.
                rejected.append(
                    f"profile {profile!r} category {category!r}: no valid entries; built-in chain kept"
                )
                continue
            categories[category] = tuple(entries)
        if categories:
            profiles[profile] = categories
    if not profiles:
        return None
    return {
        "schema_version": CATEGORY_MAESTRO_SCHEMA_VERSION,
        "profiles": profiles,
        "fingerprint": _config_fingerprint(profiles),
        "rejected": rejected,
    }


def _raw_document(omh_home: Path) -> dict[str, object]:
    """Return the on-disk document for editing; raise on one this code does not own.

    An existing file the writers do not recognize (different schema_version,
    non-object profiles, unparseable JSON) is never overwritten — losing a
    hand-written or future-versioned config on a `set`/`clear` would be silent
    data loss, and the dispatch-preference seeder next door already documents
    the same stance ("an existing file, however it got there, is never
    overwritten"). Only the READ path is never-raise; editing is an explicit
    operator action that can and should refuse loudly.
    """
    path = category_maestro_path(omh_home)
    if not path.exists():
        return {"schema_version": CATEGORY_MAESTRO_SCHEMA_VERSION, "profiles": {}}
    document, error = read_json_object_result(path)
    if (
        isinstance(document, dict)
        and document.get("schema_version") == CATEGORY_MAESTRO_SCHEMA_VERSION
        and isinstance(document.get("profiles"), dict)
    ):
        return document
    raise ValueError(
        f"refusing to edit {path}: the existing file is not a recognized "
        f"{CATEGORY_MAESTRO_SCHEMA_VERSION} document"
        + (f" ({error})" if error else "")
        + "; fix or remove it first"
    )


def set_category_maestro_chain(
    omh_home: Path,
    profile: str,
    category: str,
    entries: list[Mapping[str, str]],
) -> dict[str, object]:
    """Write one category's chain for one profile; raises ValueError on bad input."""
    normalized_profile = str(profile or "").strip().casefold()
    if normalized_profile not in CATEGORY_MAESTRO_PROFILES:
        raise ValueError(
            f"profile {profile!r} is not configurable; expected one of "
            f"{', '.join(CATEGORY_MAESTRO_PROFILES)}"
        )
    normalized_category = canonical_model_category(category)
    if not normalized_category:
        raise ValueError(
            f"category {category!r} is not one of {', '.join(MODEL_CATEGORIES)}"
        )
    if not entries or len(entries) > _MAX_CHAIN_ENTRIES:
        raise ValueError(f"a chain needs 1..{_MAX_CHAIN_ENTRIES} entries")
    rejected: list[str] = []
    validated = [
        _validated_entry(entry, where=f"entry {position}", rejected=rejected)
        for position, entry in enumerate(entries)
    ]
    if any(entry is None for entry in validated):
        raise ValueError("; ".join(rejected))
    document = _raw_document(omh_home)
    profiles = document["profiles"]
    assert isinstance(profiles, dict)
    profile_categories = profiles.setdefault(normalized_profile, {})
    if not isinstance(profile_categories, dict):
        profile_categories = {}
        profiles[normalized_profile] = profile_categories
    profile_categories[normalized_category] = [dict(entry) for entry in validated if entry]
    atomic_write_json(category_maestro_path(omh_home), document)
    return {
        "profile": normalized_profile,
        "category": normalized_category,
        "chain": profile_categories[normalized_category],
        "path": str(category_maestro_path(omh_home)),
    }


def clear_category_maestro_chain(omh_home: Path, profile: str, category: str) -> dict[str, object]:
    """Remove one category override (back to the built-in chain)."""
    normalized_profile = str(profile or "").strip().casefold()
    if normalized_profile not in CATEGORY_MAESTRO_PROFILES:
        raise ValueError(
            f"profile {profile!r} is not configurable; expected one of "
            f"{', '.join(CATEGORY_MAESTRO_PROFILES)}"
        )
    normalized_category = canonical_model_category(category)
    if not normalized_category:
        raise ValueError(
            f"category {category!r} is not one of {', '.join(MODEL_CATEGORIES)}"
        )
    document = _raw_document(omh_home)
    profiles = document["profiles"]
    assert isinstance(profiles, dict)
    profile_categories = profiles.get(normalized_profile)
    removed = False
    if isinstance(profile_categories, dict) and normalized_category in profile_categories:
        del profile_categories[normalized_category]
        removed = True
        if not profile_categories:
            del profiles[normalized_profile]
    if removed:
        # A no-op clear writes nothing: it must not create the config file
        # (whose presence is the routing opt-in) as a side effect.
        atomic_write_json(category_maestro_path(omh_home), document)
    return {
        "profile": normalized_profile,
        "category": normalized_category,
        "removed": removed,
        "path": str(category_maestro_path(omh_home)),
    }
