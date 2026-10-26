from __future__ import annotations

import argparse

from ..installer import OmhError


def validate_model_setup_args(args: argparse.Namespace) -> None:
    model_flags = (
        getattr(args, "confirm_model", []),
        getattr(args, "model_alias", []),
        getattr(args, "apply_model_config", False),
        getattr(args, "model_config_digest", ""),
        getattr(args, "allow_model_alias_collision", False),
        getattr(args, "import_category_overrides", False),
    )
    if any(model_flags) and not getattr(args, "model_setup", False):
        raise OmhError("model activation flags require --model-setup.")
    if getattr(args, "apply_model_config", False) and not getattr(args, "model_alias", []):
        raise OmhError("--apply-model-config requires at least one --model-alias ALIAS=MODEL.")


def confirmed_model_ids(values: list[str]) -> set[str]:
    confirmed: set[str] = set()
    for raw in values:
        model_id = str(raw or "").strip()
        if not model_id or any(character.isspace() for character in model_id):
            raise OmhError("--confirm-model requires a non-empty provider/model identifier.")
        confirmed.add(model_id)
    return confirmed


def classified_model_candidates(
    discovery: dict[str, object],
    confirmed: set[str],
) -> list[dict[str, str]]:
    candidates: dict[str, dict[str, str]] = {}
    observations = discovery.get("observations", [])
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            provider = str(observation.get("provider", "") or "")
            model_id = str(observation.get("model_id", "") or "")
            reference = f"{provider}/{model_id}" if provider else model_id
            if not reference:
                continue
            candidates[reference] = {
                "model": reference,
                "provider": provider,
                "model_id": model_id,
                "source": str(observation.get("source", "") or ""),
                "status": "confirmed_active" if reference in confirmed else str(
                    observation.get("status", "observed_before")
                ),
            }
    for reference in sorted(confirmed):
        provider, separator, model_id = reference.partition("/")
        candidates[reference] = {
            "model": reference,
            "provider": provider if separator else "",
            "model_id": model_id if separator else provider,
            "source": candidates.get(reference, {}).get("source", "user_confirmation"),
            "status": "confirmed_active",
        }
    return [candidates[key] for key in sorted(candidates)]


def model_alias_changes(values: list[str]) -> dict[str, str]:
    changes: dict[str, str] = {}
    for raw in values:
        alias, separator, model = str(raw or "").partition("=")
        alias = alias.strip()
        model = model.strip()
        if not separator or not alias or not model:
            raise OmhError("--model-alias requires ALIAS=MODEL.")
        changes[alias] = model
    return changes
