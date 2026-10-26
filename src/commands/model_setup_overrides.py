from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Final

from ..coding.model_recommendations import (
    MODEL_RECOMMENDATION_OVERRIDE_SCHEMA_VERSION,
    load_recommendation_overrides,
)
from ..coding.model_routing import MODEL_CATEGORIES, model_family
from ..coding.pi_family_interop import RUNTIME_CONFIG_FILES, RUNTIME_HOME


_JSONC_LINE_COMMENT: Final = re.compile(r"^\s*//.*$", re.MULTILINE)


def load_category_overrides(home: Path) -> tuple[dict[str, object] | None, str]:
    for path in (home / RUNTIME_HOME / name for name in RUNTIME_CONFIG_FILES):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonc":
            text = _JSONC_LINE_COMMENT.sub("", text)
        payload = json.loads(text)
        categories = payload.get("categories", {})
        if not isinstance(categories, dict):
            return None, str(path)
        overrides: dict[str, list[dict[str, object]]] = {}
        for category in MODEL_CATEGORIES:
            raw = categories.get(category)
            if not isinstance(raw, dict):
                continue
            candidates = _category_override_candidates(raw)
            if candidates:
                overrides[category] = candidates
        if not overrides:
            return None, str(path)
        return load_recommendation_overrides(
            {
                "schema_version": MODEL_RECOMMENDATION_OVERRIDE_SCHEMA_VERSION,
                "categories": overrides,
            }
        ), str(path)
    return None, ""


def _category_override_candidates(raw: dict[str, object]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    primary = raw.get("model")
    if isinstance(primary, str) and primary.strip():
        candidates.append(_candidate(primary, raw.get("variant")))
    fallbacks = raw.get("fallback_models", [])
    if isinstance(fallbacks, list):
        for fallback in fallbacks:
            if isinstance(fallback, str):
                candidates.append(_candidate(fallback, None))
            elif isinstance(fallback, dict) and isinstance(fallback.get("model"), str):
                candidates.append(_candidate(str(fallback["model"]), fallback.get("variant")))
    return candidates


def _candidate(model: str, variant: object) -> dict[str, object]:
    provider, separator, model_alias = model.strip().partition("/")
    candidate: dict[str, object] = {
        "model_alias": model_alias if separator else provider,
        "model_family": model_family(model_alias if separator else provider),
        "preferred_provider_families": [provider] if separator else [],
        "reasoning": "Imported from the canonical user pi-family runtime category configuration.",
    }
    if isinstance(variant, str) and variant.strip():
        candidate["reasoning_effort"] = variant.strip()
    return candidate
