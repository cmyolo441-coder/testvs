from __future__ import annotations

import re
from typing import Any

from ..paths import OmhPaths


HERMES_MODEL_SETTINGS_SCHEMA_VERSION = "hermes_model_settings/v1"
# The `auxiliary:` task slots every supported Hermes reads, in Hermes' order:
# the task blocks of `auxiliary` in `hermes_cli/config_defaults.py` at the
# floor tag `v2026.9.7`, all of which upstream main still reads. Hermes stopped
# reading `web_extract` before the floor and ignores a leftover block, so it is
# not listed. A slot Hermes added after the floor is not listed either: the
# floor does not read it.
HERMES_AUX_ALIASES = (
    "vision",
    "compression",
    "skills_hub",
    "approval",
    "review",
    "mcp",
    "title_generation",
    "memory_query_rewrite",
    "tts_audio_tags",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "goal_judge",
    "curator",
    "monitor",
    "background_review",
    "moa_reference",
    "moa_aggregator",
)

_KEY = re.compile(r"^([a-z_]+):")
_NULL_SCALARS = {"null", "~"}
_UNSUPPORTED_SCALAR_PREFIXES = ("|", ">", "[", "{", "&", "*", "!")
_CLAIM_BOUNDARY = (
    "Model settings are read from Hermes configuration; a configured alias is not evidence "
    "that a request used that model."
)


def _scalar_value(line: str) -> str | None:
    value = line.partition(":")[2].strip()
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        quote = value[0]
        closing_index = 0
        for index in range(len(value) - 1, 0, -1):
            if value[index] == quote:
                remainder = value[index + 1 :].strip()
                if not remainder or remainder.startswith("#"):
                    closing_index = index
                    value = value[: index + 1]
                    break
        if not closing_index:
            return None
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() in _NULL_SCALARS:
        return ""
    if value.startswith(_UNSUPPORTED_SCALAR_PREFIXES):
        return None
    return value


def _parse_settings(config_text: str) -> tuple[dict[str, str], dict[str, dict[str, str]]] | None:
    settings: dict[str, str] = {}
    auxiliary: dict[str, dict[str, str]] = {}
    top_level = ""
    auxiliary_task = ""
    empty_scalar_indent: int | None = None
    for line in config_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indentation = line[: len(line) - len(line.lstrip(" \t"))]
        if "\t" in indentation:
            return None
        current_indent = len(indentation)
        if empty_scalar_indent is not None:
            if current_indent > empty_scalar_indent:
                return None
            empty_scalar_indent = None
        if not line.startswith(" "):
            match = _KEY.match(line)
            top_level = match.group(1) if match else ""
            auxiliary_task = ""
            if top_level in {"model", "agent", "auxiliary", "delegation"}:
                value = _scalar_value(line)
                if value is None or value:
                    return None
            continue
        if top_level == "auxiliary" and line.startswith("  ") and not line.startswith("    "):
            match = _KEY.match(line[2:])
            auxiliary_task = match.group(1) if match else ""
            if match:
                value = _scalar_value(line[2:])
                if value is None or value:
                    return None
            continue
        if top_level in {"model", "agent", "delegation"} and line.startswith("  ") and not line.startswith("    "):
            match = _KEY.match(line[2:])
            if match:
                key = match.group(1)
                if (top_level, key) in {
                    ("model", "default"),
                    ("model", "provider"),
                    ("agent", "reasoning_effort"),
                    ("delegation", "model"),
                    ("delegation", "provider"),
                    ("delegation", "reasoning_effort"),
                }:
                    value = _scalar_value(line)
                    if value is None:
                        return None
                    settings[f"{top_level}.{key}"] = value
                    if not line.partition(":")[2].strip():
                        empty_scalar_indent = current_indent
            continue
        if top_level == "auxiliary" and auxiliary_task and line.startswith("    ") and not line.startswith("      "):
            match = _KEY.match(line[4:])
            if match and match.group(1) in {"model", "reasoning_effort"}:
                value = _scalar_value(line)
                if value is None:
                    return None
                auxiliary.setdefault(auxiliary_task, {})[match.group(1)] = value
                if not line.partition(":")[2].strip():
                    empty_scalar_indent = current_indent
    return settings, auxiliary


def _alias_entry(*, alias: str, model: str, effort: str, source: str) -> dict[str, Any]:
    label = f"{model}:{effort}" if model and effort else model or "inherit"
    return {
        "alias": alias,
        "model": model,
        "effort": effort,
        "source": source,
        "configured": bool(model),
        "label": label,
    }


def _unreadable_result() -> dict[str, Any]:
    return {
        "schema_version": HERMES_MODEL_SETTINGS_SCHEMA_VERSION,
        "observed": False,
        "reason": "config_unreadable",
        "provider": "",
        "aliases": [],
        "delegation": _alias_entry(alias="delegation", model="", effort="", source="config.delegation.model"),
        "configured_count": 0,
        "inherit_count": 0,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def read_hermes_model_settings(paths: OmhPaths) -> dict[str, Any]:
    try:
        config_text = (paths.hermes_home / "config.yaml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return _unreadable_result()

    parsed_settings = _parse_settings(config_text)
    if parsed_settings is None:
        return _unreadable_result()
    settings, auxiliary = parsed_settings
    aliases = [
        _alias_entry(
            alias="main",
            model=settings.get("model.default", ""),
            effort=settings.get("agent.reasoning_effort", ""),
            source="config.model.default",
        )
    ]
    for alias in HERMES_AUX_ALIASES:
        alias_settings = auxiliary.get(alias, {})
        aliases.append(
            _alias_entry(
                alias=alias,
                model=alias_settings.get("model", ""),
                effort=alias_settings.get("reasoning_effort", ""),
                source=f"config.auxiliary.{alias}",
            )
        )
    configured_count = sum(1 for entry in aliases if entry["configured"])
    # `delegation:` is the model every delegate_task child inherits when no
    # OMH route is set, so it is what the fan-out actually runs on; it is
    # reported beside the aliases, not among them, because the alias tuple is
    # the auxiliary slots Hermes reads.
    delegation = _alias_entry(
        alias="delegation",
        model=settings.get("delegation.model", ""),
        effort=settings.get("delegation.reasoning_effort", ""),
        source="config.delegation.model",
    )
    delegation["provider"] = settings.get("delegation.provider", "")
    return {
        "schema_version": HERMES_MODEL_SETTINGS_SCHEMA_VERSION,
        "observed": True,
        "reason": "",
        "provider": settings.get("model.provider", ""),
        "aliases": aliases,
        "delegation": delegation,
        "configured_count": configured_count,
        "inherit_count": len(aliases) - configured_count,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
