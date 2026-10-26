"""Pure configuration companion construction and five-artifact reconciliation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypedDict, assert_never

from .lifecycle_growth_configuration_identity import (
    ConfigurationIdentity, ConfigurationObservation, ConfigurationSeal, IDENTITY_SCHEMA,
    build_configuration_identity, parse_identity, parse_observations, parse_seal,
    seal_configuration, seal_reasons,
)
from .lifecycle_growth_configuration_values import (
    CONFIGURATION_BOUNDARY, ConfigurationInputError, artifact_digest, closed,
    configuration_digest, configuration_is_known, digest, optional_reference,
    record, references,
)

BINDING_SCHEMA: Final = "lifecycle_growth_configuration_binding/v1"
LAUNCH_SLOTS: Final = {"experiment", "audience_review"}
EVALUATION_SLOTS: Final = {*LAUNCH_SLOTS, "exposure_evidence", "analysis_status", "readout"}


class ArtifactConfigurationBinding(TypedDict):
    artifact_digest: str
    configuration_digest: str | None
    revision_ref: str | None
    evidence_refs: list[str]


class ConfigurationBinding(TypedDict):
    schema_version: str
    identity: ConfigurationIdentity
    seal: ConfigurationSeal | None
    observations: list[ConfigurationObservation]
    bindings: dict[str, ArtifactConfigurationBinding]
    claim_boundary: str


def parse_binding(value: object) -> ConfigurationBinding:
    source = closed(value, set(ConfigurationBinding.__annotations__))
    if source["schema_version"] != BINDING_SCHEMA or source["claim_boundary"] != CONFIGURATION_BOUNDARY:
        raise ConfigurationInputError("configuration_binding_constants_invalid")
    supplied = record(source["bindings"])
    if set(supplied) not in (LAUNCH_SLOTS, EVALUATION_SLOTS):
        raise ConfigurationInputError("configuration_binding_slots_invalid")
    bindings: dict[str, ArtifactConfigurationBinding] = {}
    for slot, value in supplied.items():
        item = closed(value, set(ArtifactConfigurationBinding.__annotations__))
        bindings[slot] = {
            "artifact_digest": digest(item["artifact_digest"]),
            "configuration_digest": None if item["configuration_digest"] is None else digest(item["configuration_digest"]),
            "revision_ref": optional_reference(item["revision_ref"]),
            "evidence_refs": references(item["evidence_refs"]),
        }
    return {"schema_version": BINDING_SCHEMA, "identity": parse_identity(source["identity"]),
            "seal": parse_seal(source["seal"]), "observations": parse_observations(source["observations"]),
            "bindings": bindings, "claim_boundary": CONFIGURATION_BOUNDARY}


def configuration_artifacts(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Use the actual nested analysis object, never a separate asserted copy."""
    artifacts = {slot: record(payload[slot]) for slot in LAUNCH_SLOTS if slot in payload}
    if "readout" in payload:
        readout = record(payload["readout"])
        artifacts.update(readout=readout, analysis_status=record(readout.get("analysis_status")))
        if "exposure_evidence" in payload:
            artifacts["exposure_evidence"] = record(payload["exposure_evidence"])
    return artifacts


def build_configuration_binding(payload: Mapping[str, object]) -> ConfigurationBinding:
    """Return the predecessor seal unchanged; no hidden persistence authority."""
    closed(payload, {"artifacts", "metadata", "observations", "predecessor_seal"})
    supplied = record(payload["artifacts"])
    if set(supplied) not in (LAUNCH_SLOTS, EVALUATION_SLOTS - {"analysis_status"}):
        raise ConfigurationInputError("configuration_artifacts_invalid")
    artifacts = configuration_artifacts(supplied)
    identity = build_configuration_identity(artifacts, payload["metadata"])
    observations = parse_observations(payload["observations"])
    predecessor = parse_seal(payload["predecessor_seal"])
    bindings: dict[str, ArtifactConfigurationBinding] = {
        slot: {"artifact_digest": artifact_digest(artifact),
               "configuration_digest": identity["configuration_digest"],
               "revision_ref": identity["revision_ref"], "evidence_refs": identity["evidence_refs"].copy()}
        for slot, artifact in artifacts.items()
    }
    return {"schema_version": BINDING_SCHEMA, "identity": identity,
            "seal": seal_configuration(identity, observations, predecessor),
            "observations": observations, "bindings": bindings, "claim_boundary": CONFIGURATION_BOUNDARY}


def validate_configuration_artifact(value: object) -> list[str]:
    try:
        source = record(value)
        if source.get("schema_version") == IDENTITY_SCHEMA:
            parse_identity(source)
        else:
            parse_binding(source)
    except ValueError as exc:
        return [str(exc)]
    return []


def review_configuration(payload: Mapping[str, object], *, evaluation: bool = True) -> list[str]:
    """Reconcile digest, predecessor and hashes before promotion; no provider I/O."""
    raw = payload.get("configuration_binding")
    if raw is None:
        return ["configuration_identity_missing"]
    binding = parse_binding(raw)
    identity = binding["identity"]
    reasons: list[str] = []
    match identity["identity_state"]:
        case "unknown":
            reasons.append("configuration_identity_unknown")
        case "legacy":
            reasons.append("configuration_identity_legacy")
        case "drifted":
            reasons.append("configuration_drift")
        case "observed":
            pass
        case unreachable:
            assert_never(unreachable)
    reasons.extend(seal_reasons(identity, binding["seal"], binding["observations"]))
    receipts = binding["observations"] + ([binding["seal"]["observation"]] if binding["seal"] else [])
    if identity["identity_state"] == "observed" and not any(
        receipt["observed_at"] == identity["observed_at"]
        and receipt["evidence_ref"] in identity["evidence_refs"]
        and receipt["configuration_digest"] == identity["configuration_digest"]
        and receipt["revision_ref"] == identity["revision_ref"] for receipt in receipts
    ):
        reasons.append("configuration_identity_unknown")
    if not evaluation:
        reasons = [code for code in reasons if code not in (
            "configuration_identity_unknown", "configuration_identity_legacy",
        )]
    artifacts = configuration_artifacts(payload)
    if not LAUNCH_SLOTS <= artifacts.keys():
        return list(dict.fromkeys([*reasons, "configuration_binding_mismatch"]))
    actual_digest = configuration_digest(artifacts)
    if identity["configuration_digest"] is not None and actual_digest != identity["configuration_digest"]:
        reasons.append("configuration_drift")
    if not configuration_is_known(artifacts, identity["revision_ref"]):
        reasons.append("configuration_identity_unknown")
    expected_slots = EVALUATION_SLOTS if evaluation else LAUNCH_SLOTS
    if set(artifacts) != expected_slots or set(binding["bindings"]) != expected_slots:
        reasons.append("configuration_binding_mismatch")
    for slot, artifact in artifacts.items():
        asserted = binding["bindings"].get(slot)
        if slot != "analysis_status" and artifact.get("lifecycle_growth_id") != identity["lifecycle_growth_id"]:
            reasons.append("configuration_binding_mismatch")
        if (asserted is None or asserted["artifact_digest"] != artifact_digest(artifact)
                or asserted["configuration_digest"] != identity["configuration_digest"]
                or asserted["revision_ref"] != identity["revision_ref"]
                or (evaluation and identity["identity_state"] == "observed" and not asserted["evidence_refs"])):
            reasons.append("configuration_binding_mismatch")
    return list(dict.fromkeys(reasons))
