"""Hermes-side model-routing preflight: does the config name one provider?

Hermes resolves a request from three independent ``config.yaml`` keys —
``model.default`` (often written as ``<family>/<model>``), ``model.provider``,
and ``model.base_url`` — plus whichever provider the user is authenticated as.
Nothing forces those to agree, and Hermes' model picker changes the *active
provider*, never ``model.default``. A user who is logged in to one provider and
has a different family pinned in ``model.default`` therefore keeps seeing that
pinned model in the picker and reads it as the wrapper hardcoding a model.

This module inspects the Hermes side read-only — no subprocess, no network, no
writes — and reports when the config does not name the provider that serves
``model.default``. Per the repo's fault-domain model this is a Hermes
USER-CONFIG fault: OMH reports it and never repairs it.

Scope, deliberately: a finding requires ``model.provider`` to be *unpinned*
(``auto``/unset, the merged Hermes default). An explicit pin is authoritative —
Hermes' own ``split_model_config_default`` treats it as such — and it also makes
a ``<family>/`` prefix a gateway namespace rather than a provider claim
(``provider: openrouter`` with ``model: anthropic/claude-sonnet-4`` is the shape
Hermes' own config comments document). Reporting a pinned config would flag
those correct setups, so a pin means there is no ambiguity left to report.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..config_adapter import model_scalar_selection
from ..paths import OmhPaths
from ..system.local_store import read_json_object_result
from ..system.metadata_safety import require_opaque_metadata_ref

HERMES_MODEL_ROUTING_SCHEMA_VERSION = "omh_hermes_model_routing/v1"

# Values that mean "no explicit provider pin"; mirrors the advisory lane's
# reading of the same key so the two surfaces cannot disagree about `auto`.
_UNPINNED_PROVIDER_MARKERS = frozenset({"", "auto", "null", "~", "default", "none"})

# Reading caps: config.yaml is tens of KB and auth.json a few KB today.
_MAX_INSPECT_BYTES = 512_000
# A credential pool has one entry per provider; a file claiming hundreds is not
# a pool this check understands, so it is reported as unobserved instead.
_MAX_CREDENTIAL_PROVIDERS = 32

_HOST_SHAPE = re.compile(r"^[A-Za-z0-9._:-]{1,253}$")


def _read_text_bounded(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_INSPECT_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _model_section_is_unambiguous(config_text: str) -> bool:
    """False when the ``model:`` block cannot be read by one canonical rule.

    Tab indentation and duplicate top-level ``model:`` blocks both make the
    tolerant scalar reader pick one shape out of several plausible ones. A
    guess here would produce a named finding about values the user never wrote,
    so an unreadable shape reports nothing at all.
    """
    if "\t" in config_text or "\x00" in config_text:
        return False
    return sum(1 for line in config_text.splitlines() if line.rstrip() == "model:") <= 1


def _safe_scalar(value: str) -> str:
    """The value when it is a safe opaque reference to echo back, else ""."""
    try:
        return require_opaque_metadata_ref(value, field="hermes model config value")
    except ValueError:
        return ""


def _model_family(default_model: str) -> str:
    """The ``<family>/`` prefix of a model pin, or "" when it carries none."""
    family, separator, rest = default_model.partition("/")
    if not separator or not rest:
        return ""
    return family.casefold()


def _base_url_host(base_url: str) -> str:
    """The host of ``model.base_url``, or "" when it cannot be read.

    Only the host leaves this module. A base URL can carry userinfo, and the
    rest of the URL is not needed to say which host serves the default model.
    """
    if not base_url:
        return ""
    candidate = base_url if "://" in base_url else f"//{base_url}"
    try:
        host = urlsplit(candidate).hostname or ""
    except ValueError:
        return ""
    return host if _HOST_SHAPE.fullmatch(host) else ""


def _host_core(host: str) -> str:
    """The host without its public suffix label, casefolded.

    ``openrouter.ai`` -> ``openrouter``; ``api.anthropic.com`` ->
    ``api.anthropic``. Dropping the last label is what keeps a two-letter TLD
    such as ``.ai`` from matching family names that merely contain those
    letters.
    """
    labels = host.casefold().rstrip(".").split(".")
    if len(labels) > 1:
        labels = labels[:-1]
    return ".".join(labels)


def _host_names_family(host: str, family: str) -> bool:
    return bool(host) and bool(family) and family in _host_core(host)


def _credential_providers(paths: OmhPaths) -> dict[str, Any]:
    """Which providers Hermes has stored credentials for. Names only.

    Reads the top-level ``credential_pool`` keys and ``active_provider`` from
    ``auth.json`` — provider ids, which Hermes also writes in the clear to
    ``provider_models_cache.json``. No nested value is read, so no issued
    credential material can reach a doctor message.
    """
    path = paths.hermes_home / "auth.json"
    try:
        if not path.is_file() or path.stat().st_size > _MAX_INSPECT_BYTES:
            return {"observed": False, "providers": [], "active": ""}
    except OSError:
        return {"observed": False, "providers": [], "active": ""}
    payload, error = read_json_object_result(path)
    if error or payload is None:
        return {"observed": False, "providers": [], "active": ""}
    pool = payload.get("credential_pool")
    if not isinstance(pool, dict) or len(pool) > _MAX_CREDENTIAL_PROVIDERS:
        return {"observed": False, "providers": [], "active": ""}
    providers = sorted({name for name in (_safe_scalar(str(key)) for key in pool) if name})
    active = _safe_scalar(str(payload.get("active_provider") or ""))
    return {"observed": True, "providers": providers, "active": active}


def hermes_model_routing_preflight(paths: OmhPaths) -> dict[str, Any]:
    """Inspect how Hermes' config names the provider for its default model."""
    config_path = paths.hermes_config_path
    config_text = _read_text_bounded(config_path)
    if config_text is None or not _model_section_is_unambiguous(config_text):
        return {
            "schema_version": HERMES_MODEL_ROUTING_SCHEMA_VERSION,
            "config": {
                "found": config_text is not None,
                "readable": False,
                "path": str(config_path),
            },
            "default_model": {"value": "", "family": ""},
            "provider": {"value": "", "pinned": False},
            "base_url": {"set": False, "host": ""},
            "credentials": {"observed": False, "providers": [], "active": ""},
        }
    default_model = _safe_scalar(model_scalar_selection(config_text, "default"))
    provider = _safe_scalar(model_scalar_selection(config_text, "provider")).casefold()
    base_url = model_scalar_selection(config_text, "base_url")
    return {
        "schema_version": HERMES_MODEL_ROUTING_SCHEMA_VERSION,
        "config": {"found": True, "readable": True, "path": str(config_path)},
        "default_model": {"value": default_model, "family": _model_family(default_model)},
        "provider": {
            "value": provider,
            "pinned": provider not in _UNPINNED_PROVIDER_MARKERS,
        },
        "base_url": {"set": bool(base_url), "host": _base_url_host(base_url)},
        "credentials": _credential_providers(paths),
    }


def model_routing_disagreements(preflight: dict[str, Any]) -> list[str]:
    """Plain-English clauses naming the observed disagreement, empty when none.

    The first clause is the finding; any further clause corroborates it. The
    credential clause never stands alone: a provider authenticated by
    environment variable is invisible to ``auth.json``, so an absent family
    there is evidence about an already-inconsistent config and never a claim
    on its own.
    """
    config = preflight.get("config", {})
    if not config.get("readable"):
        return []
    default_model = preflight.get("default_model", {})
    provider = preflight.get("provider", {})
    base_url = preflight.get("base_url", {})
    family = str(default_model.get("family") or "")
    host = str(base_url.get("host") or "")
    if not family or not host or provider.get("pinned"):
        return []
    if _host_names_family(host, family):
        return []

    provider_value = str(provider.get("value") or "")
    clauses = [
        f"`model.default` is `{default_model['value']}`, whose `{family}/` prefix names one provider family, "
        f"but `model.base_url` sends every request to `{host}`, and `model.provider` is "
        f"{f'`{provider_value}`' if provider_value else 'unset'}, which names no provider — so the provider you "
        f"pick in Hermes' model picker does not change `model.default`"
    ]
    credentials = preflight.get("credentials", {})
    stored = [str(name) for name in credentials.get("providers", [])]
    if credentials.get("observed") and stored and family not in stored:
        active = str(credentials.get("active") or "")
        suffix = f" (active: `{active}`)" if active else ""
        clauses.append(
            f"stored Hermes credentials cover {', '.join(f'`{name}`' for name in stored)}{suffix}, "
            f"not `{family}`"
        )
    return clauses


def model_routing_consistent_summary(preflight: dict[str, Any]) -> str:
    """Why there is nothing to reconcile, naming the reason that applies.

    A single "config looks consistent" line would over-claim: most of these
    states are "no ambiguity exists", not "all three keys were compared and
    agreed". The reader is told which one they are in.
    """
    default_model = str(preflight.get("default_model", {}).get("value") or "")
    family = str(preflight.get("default_model", {}).get("family") or "")
    provider = preflight.get("provider", {})
    host = str(preflight.get("base_url", {}).get("host") or "")
    if not default_model:
        return "no `model.default` is pinned in the Hermes config; nothing overrides the provider you select"
    if provider.get("pinned"):
        return (
            f"`model.provider` is pinned to `{provider['value']}`, which is authoritative for "
            f"`model.default` (`{default_model}`)"
        )
    if not family:
        return f"`model.default` is `{default_model}`, which names no provider family to disagree with"
    if not host:
        return f"`model.default` is `{default_model}` and no `model.base_url` redirects where it is served"
    return f"`model.default` is `{default_model}` and `model.base_url` (`{host}`) name the same provider family"


def model_routing_next_action(preflight: dict[str, Any]) -> str:
    """The user-owned repair for a disagreement; OMH never applies it."""
    host = str(preflight.get("base_url", {}).get("host") or "")
    return (
        f"Reconcile these in Hermes yourself — OMH never writes Hermes model configuration. Either pin the "
        f"provider that serves {host} (`hermes config set model.provider <name>`) or set `model.default` to a "
        f"model the provider you actually use serves. Then rerun `omh doctor`."
    )
