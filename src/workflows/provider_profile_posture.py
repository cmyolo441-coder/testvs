from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from ..local_store import atomic_write_json, ensure_dir
from ..paths import OmhPaths
from ..system.metadata_safety import require_opaque_metadata_ref
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


PROVIDER_PROFILE_POSTURE_INPUT_SCHEMA_VERSION = "provider_profile_posture_input/v1"
PROVIDER_PROFILE_POSTURE_SCHEMA_VERSION = "provider_profile_posture/v1"
_INPUT_REQUIRED_KEYS = {"schema_version", "provider_id", "profile_id", "requested_capabilities", "secret_requirements"}
_HOST_OBSERVATION_KEYS = {"kind", "reference", "observed_at"}
_SECRET_REQUIREMENT_KEYS = {"name", "present"}
_CAPABILITIES = frozenset({"chat", "embedding", "tool_call", "image"})
_PROFILE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SECRET_NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,95}$")


@dataclass(frozen=True)
class SecretRequirement:
    name: str
    present: bool | str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "present": self.present}


@dataclass(frozen=True)
class HostObservation:
    kind: str
    reference: str
    observed_at: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "reference": self.reference, "observed_at": self.observed_at}


@dataclass(frozen=True)
class ProviderProfilePostureInput:
    provider_id: str
    profile_id: str
    requested_capabilities: tuple[str, ...]
    secret_requirements: tuple[SecretRequirement, ...]
    host_observations: tuple[HostObservation, ...]


def parse_provider_profile_posture_input(raw: object) -> ProviderProfilePostureInput:
    if not isinstance(raw, dict) or not _INPUT_REQUIRED_KEYS.issubset(raw) or set(raw) - (_INPUT_REQUIRED_KEYS | {"host_observations"}):
        raise ValueError("provider profile posture input must use the supported metadata fields")
    if raw.get("schema_version") != PROVIDER_PROFILE_POSTURE_INPUT_SCHEMA_VERSION:
        raise ValueError("unsupported provider profile posture input schema")
    return ProviderProfilePostureInput(
        _profile_id(raw.get("provider_id"), "provider_id"),
        _profile_id(raw.get("profile_id"), "profile_id"),
        _capabilities(raw.get("requested_capabilities")),
        _secret_requirements(raw.get("secret_requirements")),
        _host_observations(raw.get("host_observations", [])),
    )


def build_provider_profile_posture(value: ProviderProfilePostureInput) -> dict[str, object]:
    return {
        "schema_version": PROVIDER_PROFILE_POSTURE_SCHEMA_VERSION,
        "provider_id": value.provider_id,
        "profile_id": value.profile_id,
        "state": "prepared_not_observed",
        "requested_capabilities": list(value.requested_capabilities),
        "secret_requirements": [requirement.to_dict() for requirement in value.secret_requirements],
        "host_observations": [observation.to_dict() for observation in value.host_observations],
        "allowed_actions": [
            "request_operator_secret_presence_confirmation",
            "request_host_observation_reference",
            "review_external_connector_readiness",
        ],
        "prohibited_actions": [
            "read_secret_value",
            "call_provider",
            "validate_credential",
            "launch_proxy",
            "route_model",
            "create_wallet",
            "execute_payment",
        ],
        "claim_boundary": (
            "Provider/profile posture is OMH-local preparation metadata; it is not credential validation, provider "
            "connectivity, model routing, payment/wallet, or host execution evidence."
        ),
    }


def write_provider_profile_posture(paths: OmhPaths, posture: dict[str, object]) -> dict[str, object]:
    encoded = json.dumps(posture, sort_keys=True, separators=(",", ":"))
    posture_id = "provider_profile_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    directory = _managed_provider_profile_postures_dir(paths)
    path = directory / f"{posture_id}.json"
    ensure_dir(directory, private=True)
    atomic_write_json(path, {**posture, "posture_id": posture_id}, private=True)
    return {"written": True, "posture_id": posture_id, "path": str(path)}


def _profile_id(value: object, field: str) -> str:
    if not isinstance(value, str) or not _PROFILE_ID.fullmatch(value):
        raise ValueError(f"{field} must match [a-z0-9][a-z0-9._-]{{0,63}}")
    return require_opaque_metadata_ref(value, field=field)


def _capabilities(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 12 or not all(item in _CAPABILITIES for item in raw):
        raise ValueError("requested_capabilities must contain 1 to 12 supported values")
    if len(raw) != len(set(raw)):
        raise ValueError("requested_capabilities must not contain duplicates")
    return tuple(raw)


def _secret_requirements(raw: object) -> tuple[SecretRequirement, ...]:
    if not isinstance(raw, list) or len(raw) > 12:
        raise ValueError("secret_requirements must contain at most 12 items")
    requirements: list[SecretRequirement] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _SECRET_REQUIREMENT_KEYS:
            raise ValueError("secret requirement must contain only name and present metadata")
        name = item.get("name")
        present = item.get("present")
        if not isinstance(name, str) or not _SECRET_NAME.fullmatch(name):
            raise ValueError("secret requirement name is invalid")
        if not isinstance(present, bool) and present != "unknown":
            raise ValueError("secret requirement present must be true, false, or unknown")
        requirements.append(SecretRequirement(name, present))
    if len({requirement.name for requirement in requirements}) != len(requirements):
        raise ValueError("secret requirement names must be unique")
    return tuple(requirements)


def _host_observations(raw: object) -> tuple[HostObservation, ...]:
    if not isinstance(raw, list) or len(raw) > 12:
        raise ValueError("host_observations must contain at most 12 items")
    observations: list[HostObservation] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _HOST_OBSERVATION_KEYS:
            raise ValueError("host observation must contain kind, reference, and observed_at")
        kind = require_opaque_metadata_ref(item.get("kind"), field="host observation kind")
        reference = require_opaque_metadata_ref(item.get("reference"), field="host observation reference")
        observed_at = item.get("observed_at")
        if not isinstance(observed_at, str) or not observed_at.endswith("Z"):
            raise ValueError("host observation observed_at must be an ISO-8601 UTC timestamp")
        try:
            strict_fromisoformat(observed_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("host observation observed_at must be an ISO-8601 UTC timestamp") from exc
        observations.append(HostObservation(kind, reference, observed_at))
    return tuple(observations)


def _managed_provider_profile_postures_dir(paths: OmhPaths) -> Path:
    root = paths.provider_profile_postures_dir
    if root.is_symlink():
        raise ValueError("provider profile posture storage must not be a symlink")
    if not root.resolve(strict=False).is_relative_to(paths.omh_home.resolve(strict=False)):
        raise ValueError("provider profile posture storage must resolve under OMH home")
    return root
