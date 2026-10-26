from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import re
from typing import Final

from .awesome_hermes_agent import (
    AwesomeHermesCatalogError,
    PACKAGED_COVERAGE_SCOPE,
    PACKAGED_COVERAGE_SCOPE_NOTE,
    UPSTREAM_SOURCE_COMMIT,
)


PLUGIN_OUTCOME_SCHEMA_VERSION: Final = "awesome_hermes_plugin_outcome_matrix/v1"
_OUTCOME_KEYS: Final = {
    "plugin_id",
    "upstream_outcome_summary",
    "implementation_state",
    "owner_boundary",
    "claim_status",
    "evidence_refs",
    "rationale",
    "native_capability_ids",
}
_ENVELOPE_KEYS: Final = {"schema_version", "source_commit", "outcomes", "claim_boundary"}
_EVIDENCE_KEYS: Final = {"kind", "ref"}
_IMPLEMENTATION_STATES: Final = {
    "absent",
    "reference_only",
    "contract_only",
    "omh_native",
    "host_integrated",
    "externally_observed",
}
_OWNER_BOUNDARIES: Final = {"omh", "hermes_host", "external_runtime"}
_CLAIM_STATUSES: Final = {"prepared_not_observed", "observed"}
_EVIDENCE_KINDS: Final = {"code", "test", "doc", "observed_artifact"}
_CAPABILITY_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


@dataclass(frozen=True)
class PluginOutcomeEvidence:
    kind: str
    ref: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "ref": self.ref}


@dataclass(frozen=True)
class AwesomeHermesPluginOutcome:
    plugin_id: str
    upstream_outcome_summary: str
    implementation_state: str
    owner_boundary: str
    claim_status: str
    evidence_refs: tuple[PluginOutcomeEvidence, ...]
    rationale: str
    native_capability_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plugin_id": self.plugin_id,
            "upstream_outcome_summary": self.upstream_outcome_summary,
            "implementation_state": self.implementation_state,
            "owner_boundary": self.owner_boundary,
            "claim_status": self.claim_status,
            "evidence_refs": [evidence.to_dict() for evidence in self.evidence_refs],
            "rationale": self.rationale,
            "native_capability_ids": list(self.native_capability_ids),
        }


def awesome_hermes_plugin_outcomes() -> dict[str, object]:
    source_commit, outcomes, claim_boundary = _plugin_outcomes_cached()
    return {
        "schema_version": PLUGIN_OUTCOME_SCHEMA_VERSION,
        "source_commit": source_commit,
        "coverage_scope": PACKAGED_COVERAGE_SCOPE,
        "coverage_scope_note": PACKAGED_COVERAGE_SCOPE_NOTE,
        "outcomes": [outcome.to_dict() for outcome in outcomes],
        "claim_boundary": claim_boundary,
    }


@lru_cache(maxsize=1)
def _plugin_outcomes_cached() -> tuple[str, tuple[AwesomeHermesPluginOutcome, ...], str]:
    resource = files("omh.catalogs").joinpath("awesome_hermes_agent_plugin_outcomes.json")
    raw = json.loads(resource.read_text(encoding="utf-8"))
    return _parse_plugin_outcomes(raw)


def _parse_plugin_outcomes(raw: object) -> tuple[str, tuple[AwesomeHermesPluginOutcome, ...], str]:
    if not isinstance(raw, dict) or set(raw) != _ENVELOPE_KEYS:
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcomes must have the exact matrix envelope")
    if raw.get("schema_version") != PLUGIN_OUTCOME_SCHEMA_VERSION:
        raise AwesomeHermesCatalogError("unsupported awesome Hermes plugin outcome schema")
    source_commit = _required_string(raw, "source_commit")
    if source_commit != UPSTREAM_SOURCE_COMMIT:
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcomes deviate from pinned upstream source")
    claim_boundary = _required_string(raw, "claim_boundary")
    raw_outcomes = raw.get("outcomes")
    if not isinstance(raw_outcomes, list):
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcomes must be a list")
    outcomes = tuple(_parse_outcome(item) for item in raw_outcomes)
    plugin_ids = [outcome.plugin_id for outcome in outcomes]
    if plugin_ids != sorted(plugin_ids) or len(plugin_ids) != len(set(plugin_ids)):
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcomes must use unique sorted plugin ids")
    return source_commit, outcomes, claim_boundary


def _parse_outcome(raw: object) -> AwesomeHermesPluginOutcome:
    if not isinstance(raw, dict) or set(raw) != _OUTCOME_KEYS:
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcome has unexpected fields")
    evidence_refs = _parse_evidence_refs(raw.get("evidence_refs"))
    native_capability_ids = _parse_native_capability_ids(raw.get("native_capability_ids"))
    outcome = AwesomeHermesPluginOutcome(
        plugin_id=_required_string(raw, "plugin_id"),
        upstream_outcome_summary=_required_string(raw, "upstream_outcome_summary"),
        implementation_state=_required_choice(raw, "implementation_state", _IMPLEMENTATION_STATES),
        owner_boundary=_required_choice(raw, "owner_boundary", _OWNER_BOUNDARIES),
        claim_status=_required_choice(raw, "claim_status", _CLAIM_STATUSES),
        evidence_refs=evidence_refs,
        rationale=_required_string(raw, "rationale"),
        native_capability_ids=native_capability_ids,
    )
    _validate_outcome_claim(outcome)
    return outcome


def _parse_evidence_refs(raw: object) -> tuple[PluginOutcomeEvidence, ...]:
    if not isinstance(raw, list):
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcome evidence_refs must be a list")
    evidence_refs: list[PluginOutcomeEvidence] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != _EVIDENCE_KEYS:
            raise AwesomeHermesCatalogError("awesome Hermes plugin outcome evidence reference has unexpected fields")
        evidence_refs.append(
            PluginOutcomeEvidence(
                kind=_required_choice(item, "kind", _EVIDENCE_KINDS),
                ref=_required_string(item, "ref"),
            )
        )
    return tuple(evidence_refs)


def _parse_native_capability_ids(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcome native_capability_ids must be a string list")
    if len(raw) != len(set(raw)) or any(not _CAPABILITY_ID.fullmatch(item) for item in raw):
        raise AwesomeHermesCatalogError("awesome Hermes plugin outcome native_capability_ids are invalid")
    return tuple(raw)


def _validate_outcome_claim(outcome: AwesomeHermesPluginOutcome) -> None:
    evidence_kinds = {evidence.kind for evidence in outcome.evidence_refs}
    if outcome.implementation_state == "omh_native":
        if outcome.owner_boundary != "omh" or outcome.claim_status != "prepared_not_observed":
            raise AwesomeHermesCatalogError("OMH-native outcomes must remain prepared OMH-local claims")
        if not evidence_kinds.intersection({"code", "test"}):
            raise AwesomeHermesCatalogError("OMH-native outcomes require code or test evidence")
        if not outcome.native_capability_ids:
            raise AwesomeHermesCatalogError("OMH-native outcomes require native capability ids")
    if outcome.implementation_state == "contract_only":
        if outcome.owner_boundary != "omh" or outcome.claim_status != "prepared_not_observed":
            raise AwesomeHermesCatalogError("contract-only outcomes must remain prepared OMH-local claims")
        if not evidence_kinds.intersection({"code", "doc"}) or "observed_artifact" in evidence_kinds:
            raise AwesomeHermesCatalogError("contract-only outcomes require code or doc evidence without observed artifacts")
    if outcome.implementation_state in {"absent", "reference_only"} and outcome.native_capability_ids:
        raise AwesomeHermesCatalogError("absent or reference-only outcomes cannot claim native capability ids")
    if outcome.owner_boundary in {"hermes_host", "external_runtime"}:
        if outcome.claim_status != "observed" or "observed_artifact" not in evidence_kinds:
            raise AwesomeHermesCatalogError("host or external outcomes require observed artifact evidence")


def _required_string(raw: dict[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AwesomeHermesCatalogError(f"awesome Hermes plugin outcome field must be a non-empty string: {key}")
    return value


def _required_choice(raw: dict[object, object], key: str, choices: set[str]) -> str:
    value = _required_string(raw, key)
    if value not in choices:
        raise AwesomeHermesCatalogError(f"awesome Hermes plugin outcome field has an unsupported value: {key}")
    return value
