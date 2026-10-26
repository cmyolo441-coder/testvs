"""Closed configuration identity and immutable caller-carried observation seal."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Final, Literal, TypedDict, assert_never

from .lifecycle_growth_configuration_values import (
    CANONICAL_SCHEMA, CONFIGURATION_BOUNDARY, ConfigurationInputError, closed,
    configuration_digest, configuration_is_known, digest, optional_reference,
    record, reference, references, timestamp,
)

IDENTITY_SCHEMA: Final = "lifecycle_growth_configuration_identity/v1"
SEAL_SCHEMA: Final = "lifecycle_growth_configuration_seal/v1"
IdentityState = Literal["observed", "unknown", "drifted", "legacy"]


class ConfigurationIdentity(TypedDict):
    schema_version: str
    lifecycle_growth_id: str
    identity_state: IdentityState
    configuration_ref: str | None
    revision_ref: str | None
    canonical_schema: str
    digest_algorithm: str
    configuration_digest: str | None
    observed_at: str | None
    evidence_refs: list[str]
    claim_boundary: str


class ConfigurationObservation(TypedDict):
    kind: str
    observed_at: str
    evidence_ref: str
    configuration_digest: str
    revision_ref: str | None


class ConfigurationSeal(TypedDict):
    schema_version: str
    lifecycle_growth_id: str
    observation: ConfigurationObservation


def parse_identity(value: object) -> ConfigurationIdentity:
    source = closed(value, set(ConfigurationIdentity.__annotations__))
    state = source["identity_state"]
    if state not in ("observed", "unknown", "drifted", "legacy"):
        raise ConfigurationInputError("configuration_identity_state_invalid")
    # Literal matching narrows the boundary input without an unchecked cast.
    match state:
        case "observed":
            parsed_state: IdentityState = "observed"
        case "unknown":
            parsed_state = "unknown"
        case "drifted":
            parsed_state = "drifted"
        case "legacy":
            parsed_state = "legacy"
        case _:
            raise ConfigurationInputError("configuration_identity_state_invalid")
    result: ConfigurationIdentity = {
        "schema_version": IDENTITY_SCHEMA,
        "lifecycle_growth_id": reference(source["lifecycle_growth_id"]),
        "identity_state": parsed_state,
        "configuration_ref": optional_reference(source["configuration_ref"]),
        "revision_ref": optional_reference(source["revision_ref"]),
        "canonical_schema": CANONICAL_SCHEMA, "digest_algorithm": "sha256",
        "configuration_digest": None if source["configuration_digest"] is None else digest(source["configuration_digest"]),
        "observed_at": None if source["observed_at"] is None else timestamp(source["observed_at"]),
        "evidence_refs": references(source["evidence_refs"]), "claim_boundary": CONFIGURATION_BOUNDARY,
    }
    for key in ("schema_version", "canonical_schema", "digest_algorithm", "claim_boundary"):
        if source[key] != result[key]:
            raise ConfigurationInputError("configuration_identity_constants_invalid")
    match parsed_state:
        case "observed" | "drifted":
            if not result["configuration_digest"] or not result["observed_at"] or not result["evidence_refs"]:
                raise ConfigurationInputError("configuration_observation_required")
        case "unknown" | "legacy":
            if result["observed_at"] is not None or result["evidence_refs"]:
                raise ConfigurationInputError("configuration_observation_contradiction")
        case unreachable:
            assert_never(unreachable)
    return result


def build_configuration_identity(artifacts: Mapping[str, object], metadata: object) -> ConfigurationIdentity:
    """Compute reviewed digest; observations remain supplied by the adapter."""
    fields = closed(metadata, {"identity_state", "configuration_ref", "revision_ref", "observed_at", "evidence_refs"})
    identity = parse_identity({
        "schema_version": IDENTITY_SCHEMA,
        "lifecycle_growth_id": record(artifacts["experiment"])["lifecycle_growth_id"],
        **fields, "canonical_schema": CANONICAL_SCHEMA, "digest_algorithm": "sha256",
        "configuration_digest": configuration_digest(artifacts), "claim_boundary": CONFIGURATION_BOUNDARY,
    })
    if not configuration_is_known(artifacts, identity["revision_ref"]):
        identity.update(identity_state="unknown", observed_at=None, evidence_refs=[])
    return identity


def parse_observation(value: object) -> ConfigurationObservation:
    source = closed(value, set(ConfigurationObservation.__annotations__))
    if source["kind"] not in ("launch", "exposure"):
        raise ConfigurationInputError("configuration_observation_kind_invalid")
    return {"kind": reference(source["kind"]), "observed_at": timestamp(source["observed_at"]),
            "evidence_ref": reference(source["evidence_ref"]),
            "configuration_digest": digest(source["configuration_digest"]),
            "revision_ref": optional_reference(source["revision_ref"])}


def parse_observations(value: object) -> list[ConfigurationObservation]:
    if not isinstance(value, list) or len(value) > 8:
        raise ConfigurationInputError("configuration_observations_invalid")
    return [parse_observation(item) for item in value]


def parse_seal(value: object) -> ConfigurationSeal | None:
    if value is None:
        return None
    source = closed(value, set(ConfigurationSeal.__annotations__))
    if source["schema_version"] != SEAL_SCHEMA:
        raise ConfigurationInputError("configuration_seal_schema_invalid")
    return {"schema_version": SEAL_SCHEMA, "lifecycle_growth_id": reference(source["lifecycle_growth_id"]),
            "observation": parse_observation(source["observation"])}


def observation_order(value: ConfigurationObservation) -> tuple[datetime, str]:
    return datetime.fromisoformat(value["observed_at"].replace("Z", "+00:00")), value["evidence_ref"]


def seal_configuration(identity: ConfigurationIdentity, observations: list[ConfigurationObservation], predecessor: ConfigurationSeal | None) -> ConfigurationSeal | None:
    """Never replace a predecessor; reconciliation reports contradictory arrivals."""
    if predecessor is not None:
        if predecessor["lifecycle_growth_id"] != identity["lifecycle_growth_id"]:
            raise ConfigurationInputError("configuration_seal_lifecycle_mismatch")
        return predecessor
    if not observations:
        return None
    return {"schema_version": SEAL_SCHEMA, "lifecycle_growth_id": identity["lifecycle_growth_id"],
            "observation": min(observations, key=observation_order)}


def seal_reasons(identity: ConfigurationIdentity, seal: ConfigurationSeal | None, observations: list[ConfigurationObservation]) -> list[str]:
    if seal is None:
        return ["configuration_identity_unknown"]
    first = seal["observation"]
    reasons: list[str] = []
    if seal["lifecycle_growth_id"] != identity["lifecycle_growth_id"]:
        reasons.append("configuration_binding_mismatch")
    if (identity["configuration_digest"] is not None
            and (identity["configuration_digest"], identity["revision_ref"]) != (first["configuration_digest"], first["revision_ref"])):
        reasons.append("configuration_drift")
    for observation in observations:
        different = any(observation[key] != first[key] for key in ("configuration_digest", "revision_ref"))
        if different:
            reasons.append("configuration_seal_conflict" if observation_order(observation) <= observation_order(first) else "configuration_drift")
        if observation["evidence_ref"] == first["evidence_ref"] and observation != first:
            reasons.append("configuration_seal_conflict")
    return list(dict.fromkeys(reasons))
