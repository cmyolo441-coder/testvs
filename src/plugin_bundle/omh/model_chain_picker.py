"""The editing model behind the chain picker, shared by its two surfaces.

Bare `omh model-chains` on a terminal and `/omh-model` in the OMH TUI walk
the same rows: one per shipped mixture category, each showing the chain
in effect, whether it is a shipped default or an override, and whether a
provider this machine holds -- linked to Hermes, or recorded by `omh setup`
-- serves its head. Left and right step the head
model through the known aliases, minus and plus step the head's reasoning
effort, and a save composes the override document exactly the way
`omh model-chains set` composes it. This module holds every one of those
rules so neither surface carries chain logic of its own; it renders nothing
and reads no keys.

It lives in the plugin bundle, so it may depend only on siblings in this
package (`tests/test_plugin_bundle_standalone.py`). Display labels and
purpose phrases live in the repo's catalog module, which the bundle cannot
see; `picker_rows` takes them as optional mappings and falls back to the raw
alias — the name routing actually uses — so a surface without the catalog
still shows the truth, just less prettily.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import secrets
from typing import Any

from . import runtime_paths
from .hermes_delegation import (
    HERMES_MIXTURE_CATEGORY_CHAINS,
    MIXTURE_CHAIN_OVERRIDES_SCHEMA_VERSION,
    alias_is_served,
    effective_provider_entitlements,
    load_mixture_chain_overrides,
    load_model_provider_routes,
    mixture_chain_overrides_path,
    parse_mixture_chain_overrides,
    provider_entitlements_path,
)

Chain = tuple[tuple[str, str], ...]

PICKER_SCHEMA_VERSION = "model_chain_picker/v1"

# The rungs the picker steps through. Shipped chains only ever declare these
# four; `max` exists on some models but is never a default, so `+` on `max`
# holds and `-` on it re-enters the ring at `xhigh`.
PICKER_EFFORT_RING: tuple[str, ...] = ("low", "medium", "high", "xhigh")

# Mirror of `REASONING_EFFORT_LADDER` in `omh.coding.model_routing`, which
# the bundle cannot import; `tests/test_model_chain_picker.py` pins parity.
_EFFORT_LADDER: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "max")
_RING_LOW = _EFFORT_LADDER.index(PICKER_EFFORT_RING[0])
_RING_HIGH = _EFFORT_LADDER.index(PICKER_EFFORT_RING[-1])


def chain_text(chain: Chain) -> str:
    """`model[:effort], ...` — the form `omh model-chains set` accepts back."""
    return ", ".join(model + (f":{effort}" if effort else "") for model, effort in chain)


def chain_entries(chain: Chain) -> list[dict[str, str]]:
    return [{"model": model, "reasoning_effort": effort} for model, effort in chain]


def known_model_aliases(overrides: Mapping[str, Chain] | None = None) -> tuple[str, ...]:
    """Every alias a chain names today, alphabetical: each member of a shipped
    chain plus each member of the user's overrides.

    Deliberately not the price table. That table also carries superseded
    generations and recognition-only aliases, and a ring is an offer: a model
    it steps to reads as one OMH recommends. Anything a person wants beyond
    what a chain already names is one `omh model-chains set` away, and once
    written it joins the ring here.
    """
    aliases = {model for chain in HERMES_MIXTURE_CATEGORY_CHAINS.values() for model, _ in chain}
    for chain in (overrides or {}).values():
        aliases.update(model for model, _ in chain)
    return tuple(sorted(aliases))


def model_ring(
    aliases: tuple[str, ...],
    entitlements: Mapping[str, Any] | None,
    routes: Mapping[str, tuple[str, str]] | None,
) -> tuple[dict[str, Any], ...]:
    """The aliases left/right step through: served first, alphabetical within.

    Unserved aliases stay in the ring rather than vanishing — the picker marks
    them, and a person may want one anyway (an entitlement recorded before a
    provider was added) — but they sit behind every served one so the common
    walk never crosses a model this machine cannot reach.
    """
    rows = [
        {
            "alias": alias,
            "served": alias_is_served(alias, entitlements, routes) if entitlements is not None else True,
        }
        for alias in aliases
    ]
    return tuple(sorted(rows, key=lambda row: (not row["served"], row["alias"])))


def read_override_document(omh_home: str | Path | None) -> dict[str, Any]:
    """The override document as written, or an empty one for anything else.

    An unreadable or invalid file yields the empty document on purpose: the
    writer then replaces it with a valid one instead of preserving a shape the
    reader would keep ignoring.
    """
    path = mixture_chain_overrides_path(omh_home)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, ValueError):
        raw = None
    if isinstance(raw, dict) and isinstance(raw.get("categories"), dict):
        return raw
    return {"schema_version": MIXTURE_CHAIN_OVERRIDES_SCHEMA_VERSION, "categories": {}}


def picker_rows(
    omh_home: str | Path | None,
    *,
    hermes_home: str | Path | None = None,
    labels: Mapping[str, str] | None = None,
    purposes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Everything a picker surface renders, read once.

    Rows carry the chain as stored — the override or the shipped default —
    not the entitlement-reordered projection `omh model-chains show` prints,
    because the picker edits what will be written, and a save of the shaped
    order would silently freeze a reordering that was meant to follow the
    machine's providers. `providers` lists the providers counted for the
    served marks, each with where it was found, so the surface can show why
    a model is marked instead of only that it is.
    """
    labels = labels or {}
    purposes = purposes or {}
    omh_home = _bound_store(omh_home, hermes_home)
    overrides, status = load_mixture_chain_overrides(omh_home)
    entitlements, entitlement_status, providers = effective_provider_entitlements(omh_home, hermes_home)
    routes, _ = load_model_provider_routes(omh_home)

    def served(alias: str) -> bool:
        return alias_is_served(alias, entitlements, routes) if entitlements is not None else True

    categories = []
    for name, default_chain in HERMES_MIXTURE_CATEGORY_CHAINS.items():
        chain = overrides.get(name, default_chain)
        categories.append(
            {
                "category": name,
                "purpose": purposes.get(name, ""),
                "chain": [
                    {
                        "model": model,
                        "reasoning_effort": effort,
                        "label": labels.get(model, model),
                        "served": served(model),
                    }
                    for model, effort in chain
                ],
                "default_chain": chain_entries(default_chain),
                "origin": "override" if name in overrides else "default",
            }
        )
    return {
        "schema_version": PICKER_SCHEMA_VERSION,
        "path": str(mixture_chain_overrides_path(omh_home)),
        "document_status": status,
        "entitlements_path": str(provider_entitlements_path(omh_home)),
        "entitlements_status": entitlement_status,
        "providers": [dict(row) for row in providers],
        "categories": categories,
        "models": [
            {"alias": row["alias"], "label": labels.get(row["alias"], row["alias"]), "served": row["served"]}
            for row in model_ring(known_model_aliases(overrides), entitlements, routes)
        ],
        "efforts": list(PICKER_EFFORT_RING),
    }


def chain_from_entries(entries: list[dict[str, Any]]) -> Chain:
    return tuple((str(entry["model"]), str(entry.get("reasoning_effort", ""))) for entry in entries)


def step_head_model(chain: Chain, aliases: tuple[str, ...], direction: int) -> Chain:
    """Replace the head model with its ring neighbour, keeping its effort.

    The tail is preserved minus any entry now equal to the new head, so a
    chain never names one model twice. A head the ring does not know (a
    custom entry) is one step away from either end of the ring; it cannot be
    stepped back to, which is what `d` and cancel are for.
    """
    if not chain or not aliases:
        return chain
    head_model, head_effort = chain[0]
    if head_model in aliases:
        index = aliases.index(head_model)
        new_head = aliases[(index + direction) % len(aliases)]
    else:
        new_head = aliases[0] if direction > 0 else aliases[-1]
    tail = tuple(entry for entry in chain[1:] if entry[0] != new_head)
    return ((new_head, head_effort),) + tail


def step_effort(chain: Chain, direction: int) -> Chain:
    """Step the head's effort one rung along the ring, clamped at its ends.

    An effort below the ring or absent enters at `low` from either
    direction; `max` (above it) holds on `+` and re-enters at `xhigh` on `-`.
    """
    if not chain:
        return chain
    head_model, current = chain[0]
    index = _EFFORT_LADDER.index(current) if current in _EFFORT_LADDER else _RING_LOW - 1
    if index < _RING_LOW:
        new_index = _RING_LOW
    elif index > _RING_HIGH:
        new_index = _RING_HIGH if direction < 0 else index
    else:
        new_index = min(max(index + direction, _RING_LOW), _RING_HIGH)
    return ((head_model, _EFFORT_LADDER[new_index]),) + chain[1:]


def compose_override_document(
    document: Mapping[str, Any],
    changes: Mapping[str, Chain],
) -> dict[str, Any]:
    """A new document with `changes` applied, validated the way the reader
    validates.

    A category set back to its shipped default is removed from the document
    rather than written out, so the shipped default keeps following
    `omh update`; every category `changes` does not name is carried over
    byte-for-byte. Raises ValueError for an unknown category or a document
    the reader would reject.
    """
    categories = dict(document.get("categories") or {})
    for name, chain in changes.items():
        default_chain = HERMES_MIXTURE_CATEGORY_CHAINS.get(name)
        if default_chain is None:
            raise ValueError(f"unknown category {name!r}")
        if chain == default_chain:
            categories.pop(name, None)
        else:
            categories[name] = chain_entries(chain)
    composed = {**document, "schema_version": MIXTURE_CHAIN_OVERRIDES_SCHEMA_VERSION, "categories": categories}
    _, status = parse_mixture_chain_overrides(composed)
    if status.startswith("invalid"):
        raise ValueError(status)
    return composed


def write_override_document(omh_home: str | Path | None, document: Mapping[str, Any]) -> Path:
    """Write a composed document where the reader looks for it.

    The `omh model-chains` command keeps the repo-wide atomic writer; this is
    the bundle's own (the todo store's idiom: a fresh temporary beside the
    target, then `os.replace`), because a module in the plugin bundle may not
    import `omh.system` (`tests/test_plugin_bundle_standalone.py`). Validated once
    more before the bytes land, so an invalid document is never written by
    either surface.
    """
    _, status = parse_mixture_chain_overrides(document)
    if status.startswith("invalid"):
        raise ValueError(status)
    path = mixture_chain_overrides_path(omh_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return path


def _bound_store(omh_home: str | Path | None, hermes_home: str | Path | None) -> str | Path:
    """A store named by the caller, else the one `hermes_home` resolves to.

    Naming no store means "the store this Hermes home dispatches from": a
    profile's own `plugins.entries.omh.settings.omh_home`, then the
    standalone precedence (#1679). Bound here, once, so every read and the
    write below agree, and so `hermes_home` binds rather than reading as if
    it did while the ambient `HERMES_HOME` decided.
    """
    if omh_home is not None:
        return omh_home
    return runtime_paths.resolve_homes(None, hermes_home)[0]


def apply_picker_changes(
    omh_home: str | Path | None,
    raw_changes: object,
    *,
    hermes_home: str | Path | None = None,
) -> dict[str, Any]:
    """The save `/omh-model` in the OMH TUI makes, as one call.

    `raw_changes` is `{category: [{"model": ..., "reasoning_effort": ...}]}`
    for every edited row. A refusal comes back as `{"error": ...}` rather
    than a traceback, and the TUI shows the sentence.
    """
    if not isinstance(raw_changes, Mapping):
        return {"error": "changes must be a JSON object keyed by category"}
    try:
        omh_home = _bound_store(omh_home, hermes_home)
    except runtime_paths.RuntimeBindingError as exc:
        return {"error": str(exc)}
    changes: dict[str, Chain] = {}
    for name, entries in raw_changes.items():
        if not isinstance(entries, list) or not all(
            isinstance(entry, Mapping) and isinstance(entry.get("model"), str) for entry in entries
        ):
            return {"error": f"category {name!r} needs a list of model entries"}
        changes[str(name)] = tuple(
            (str(entry["model"]), str(entry.get("reasoning_effort", "") or "")) for entry in entries
        )
    try:
        document = compose_override_document(read_override_document(omh_home), changes)
        path = write_override_document(omh_home, document)
    except (ValueError, OSError) as exc:
        return {"error": str(exc)}
    return {
        "schema_version": "model_chain_picker_apply/v1",
        "path": str(path),
        "changed": sorted(changes),
    }
