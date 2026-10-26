from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from .cross_harness_adapter_model import (
    AdapterRequest,
    AdapterResult,
    NormalizedEvent,
    _digest,
    _integer,
    _items,
    _mapping,
    _raise,
    _relative,
    _safe,
    _shape,
    _text,
    canonical_digest,
    parse_adapter_request,
    parse_adapter_result,
)
from .cross_harness_benchmark import CommandBinding, Corpus, Fixture, Source
from .cross_harness_benchmark_identity import require_trusted_corpus
from .cross_harness_benchmark_values import JsonValue


EVIDENCE_SCHEMA: Final = "cross_harness_adapter_evidence/v1"
SUBMISSION_SCHEMA: Final = "cross_harness_benchmark_submission/v1"


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    source_id: str
    commit: str
    license: str
    path_metadata: str
    source_digest: str


@dataclass(frozen=True, slots=True)
class CommandEvidence:
    command_id: str
    harness: str
    argv: tuple[str, ...]
    cwd_class: str
    source_id: str
    source_commit: str
    expected_exit: int
    expected_semantic_result: str
    binding_digest: str
    observed_exit: int
    observed_semantic_result: str


@dataclass(frozen=True, slots=True)
class AdapterEvidenceCase:
    fixture_id: str
    request: AdapterRequest
    request_digest: str
    result: AdapterResult
    result_digest: str
    source_binding: SourceEvidence
    command_evidence: CommandEvidence


@dataclass(frozen=True, slots=True)
class AdapterEvidenceBundle:
    schema_version: str
    corpus_digest: str
    harness_id: str
    cases: tuple[AdapterEvidenceCase, ...]
    bundle_digest: str


def parse_adapter_evidence(raw: Mapping[str, JsonValue]) -> AdapterEvidenceBundle:
    _shape(raw, {"schema_version", "corpus_digest", "harness_id", "cases", "bundle_digest"})
    _safe(raw)
    if _text(raw["schema_version"]) != EVIDENCE_SCHEMA:
        _raise("stale_adapter_version")
    supplied_digest = _digest(raw["bundle_digest"])
    unsigned = {key: value for key, value in raw.items() if key != "bundle_digest"}
    if canonical_digest(unsigned) != supplied_digest:
        _raise("bundle_digest_mismatch")
    corpus_digest = _digest(raw["corpus_digest"])
    cases = tuple(_parse_case(item, corpus_digest) for item in _items(raw["cases"]))
    fixture_ids = {item.fixture_id for item in cases}
    if len(cases) != len(fixture_ids):
        _raise("duplicate_fixture_evidence")
    return AdapterEvidenceBundle(
        EVIDENCE_SCHEMA, corpus_digest, _text(raw["harness_id"]), cases,
        supplied_digest,
    )


def project_adapter_evidence(
    corpus: Corpus, bundle: AdapterEvidenceBundle
) -> dict[str, JsonValue]:
    require_trusted_corpus(corpus)
    if bundle.corpus_digest != corpus.digest:
        _raise("stale_corpus_digest")
    indexed = {item.fixture_id: item for item in bundle.cases}
    expected = {item.id for item in corpus.fixtures}
    if expected - indexed.keys():
        _raise("missing_fixture_evidence")
    if indexed.keys() - expected:
        _raise("mismatched_fixture_evidence")
    results: list[JsonValue] = []
    for fixture in corpus.fixtures:
        case = indexed[fixture.id]
        request = case.request
        result = case.result
        source = next(item for item in corpus.sources if item.source_id == fixture.source_id)
        command = next(item for item in corpus.commands if item.command_id == fixture.command_binding_id)
        if (
            request.fixture_id != fixture.id
            or result.fixture_id != fixture.id
            or request.fixture_binding_digest != corpus_fixture_binding_digest(fixture)
        ):
            _raise("fixture_binding_mismatch")
        if request.adapter_id != fixture.adapter_id or result.adapter_id != fixture.adapter_id:
            _raise("fixture_binding_mismatch")
        if request.capability_id != fixture.capability_id or result.capability_id != fixture.capability_id:
            _raise("fixture_binding_mismatch")
        if not _source_matches(case.source_binding, source) or not _command_matches(case.command_evidence, command):
            _raise("fixture_binding_mismatch")
        if not _satisfies(result, fixture) and any(
            _satisfies(result, other) for other in corpus.fixtures if other.id != fixture.id
        ):
            _raise("fixture_binding_mismatch")
        if result.artifact_hash != artifact_content_digest(adapter_result_payload(result)):
            _raise("artifact_hash_mismatch")
        results.append(_submission_result(case))
    return {
        "schema_version": SUBMISSION_SCHEMA,
        "corpus_digest": corpus.digest,
        "harness_id": bundle.harness_id,
        "results": results,
    }


def adapter_evidence_payload(value: AdapterEvidenceBundle) -> dict[str, JsonValue]:
    unsigned: dict[str, JsonValue] = {
        "schema_version": value.schema_version,
        "corpus_digest": value.corpus_digest,
        "harness_id": value.harness_id,
        "cases": [_case_payload(item) for item in value.cases],
    }
    return {**unsigned, "bundle_digest": value.bundle_digest}


def adapter_request_payload(value: AdapterRequest) -> dict[str, JsonValue]:
    return {"schema_version": value.schema_version, "protocol_version": value.protocol_version, "corpus_digest": value.corpus_digest, "fixture_binding_digest": value.fixture_binding_digest, "fixture_id": value.fixture_id, "adapter_id": value.adapter_id, "capability_id": value.capability_id, "profile": value.profile.value, "executable": value.executable, "executable_version": value.executable_version, "model": value.model, "effort": value.effort.value, "capabilities": list(value.capabilities), "argv_digest": value.argv_digest, "repetition": value.repetition, "timeout_seconds": value.timeout_seconds}


def adapter_result_payload(value: AdapterResult) -> dict[str, JsonValue]:
    return {"schema_version": value.schema_version, "request_digest": value.request_digest, "fixture_id": value.fixture_id, "adapter_id": value.adapter_id, "capability_id": value.capability_id, "evidence_class": value.evidence_class, "observation_state": value.observation_state.value, "actual_machine": dict(value.actual_machine), "facts": dict(value.facts), "skill_events": [_event_payload(item) for item in value.skill_events], "tool_events": [_event_payload(item) for item in value.tool_events], "child_results": [_event_payload(item) for item in value.child_results], "artifact_type": value.artifact_type, "artifact_hash": value.artifact_hash, "process_status": value.process_status.value, "exit_code": value.exit_code, "side_effects": [{"path": item.path, "change": item.change} for item in value.side_effects]}


def artifact_content_digest(raw: Mapping[str, JsonValue]) -> str:
    return canonical_digest({key: value for key, value in raw.items() if key != "artifact_hash"})


def corpus_fixture_binding_digest(fixture: Fixture) -> str:
    return canonical_digest({"fixture_id": fixture.id, "adapter_id": fixture.adapter_id, "capability_id": fixture.capability_id, "source_id": fixture.source_id, "command_binding_id": fixture.command_binding_id})


def _parse_case(value: JsonValue, corpus_digest: str) -> AdapterEvidenceCase:
    raw = _mapping(value)
    _shape(raw, {"fixture_id", "request", "request_digest", "result", "result_digest", "source_binding", "command_evidence"})
    request_raw = _mapping(raw["request"])
    result_raw = _mapping(raw["result"])
    request = parse_adapter_request(request_raw)
    result = parse_adapter_result(result_raw)
    request_digest = _digest(raw["request_digest"])
    result_digest = _digest(raw["result_digest"])
    fixture_id = _text(raw["fixture_id"])
    if canonical_digest(request_raw) != request_digest or result.request_digest != request_digest:
        _raise("request_digest_mismatch")
    if canonical_digest(result_raw) != result_digest:
        _raise("result_digest_mismatch")
    if request.corpus_digest != corpus_digest:
        _raise("stale_corpus_digest")
    if fixture_id != request.fixture_id or fixture_id != result.fixture_id:
        _raise("fixture_binding_mismatch")
    if request.adapter_id != result.adapter_id or request.capability_id != result.capability_id:
        _raise("fixture_binding_mismatch")
    source = _parse_source(raw["source_binding"])
    command = _parse_command(raw["command_evidence"])
    if result.process_status.value == "exit" and result.exit_code != command.observed_exit:
        _raise("process_evidence_mismatch")
    return AdapterEvidenceCase(
        fixture_id, request, request_digest, result, result_digest, source, command
    )


def _parse_source(value: JsonValue) -> SourceEvidence:
    raw = _mapping(value)
    _shape(raw, {"source_id", "commit", "license", "path_metadata", "source_digest"})
    return SourceEvidence(
        _text(raw["source_id"]), _text(raw["commit"]), _text(raw["license"]),
        _relative(raw["path_metadata"]), _digest(raw["source_digest"]),
    )


def _parse_command(value: JsonValue) -> CommandEvidence:
    raw = _mapping(value)
    _shape(raw, {"command_id", "harness", "argv", "cwd_class", "source_id", "source_commit", "expected_exit", "expected_semantic_result", "binding_digest", "observed_exit", "observed_semantic_result"})
    argv = tuple(_text(item) for item in _items(raw["argv"]))
    if not argv:
        _raise("missing_command_argv")
    return CommandEvidence(
        _text(raw["command_id"]), _text(raw["harness"]), argv,
        _text(raw["cwd_class"]), _text(raw["source_id"]),
        _text(raw["source_commit"]), _integer(raw["expected_exit"]),
        _text(raw["expected_semantic_result"]), _digest(raw["binding_digest"]),
        _integer(raw["observed_exit"]), _text(raw["observed_semantic_result"]),
    )


def _submission_result(case: AdapterEvidenceCase) -> JsonValue:
    result = case.result
    return {
        "fixture_id": case.fixture_id,
        "adapter_id": result.adapter_id,
        "capability_id": result.capability_id,
        "evidence_class": result.evidence_class,
        "runtime_observation": result.observation_state.value,
        "actual_machine": dict(result.actual_machine),
        "facts": dict(result.facts),
        "source_binding": _source_payload(case.source_binding),
        "command_evidence": _command_payload(case.command_evidence),
        "child_results": [
            {"id": item.id, "result": item.result} for item in result.child_results
        ],
    }


def _case_payload(value: AdapterEvidenceCase) -> JsonValue:
    return {"fixture_id": value.fixture_id, "request": adapter_request_payload(value.request), "request_digest": value.request_digest, "result": adapter_result_payload(value.result), "result_digest": value.result_digest, "source_binding": _source_payload(value.source_binding), "command_evidence": _command_payload(value.command_evidence)}


def _event_payload(value: NormalizedEvent) -> JsonValue:
    return {"id": value.id, "result": value.result}


def _source_payload(value: SourceEvidence) -> JsonValue:
    return {"source_id": value.source_id, "commit": value.commit, "license": value.license, "path_metadata": value.path_metadata, "source_digest": value.source_digest}


def _command_payload(value: CommandEvidence) -> JsonValue:
    return {"command_id": value.command_id, "harness": value.harness, "argv": list(value.argv), "cwd_class": value.cwd_class, "source_id": value.source_id, "source_commit": value.source_commit, "expected_exit": value.expected_exit, "expected_semantic_result": value.expected_semantic_result, "binding_digest": value.binding_digest, "observed_exit": value.observed_exit, "observed_semantic_result": value.observed_semantic_result}


def _source_matches(value: SourceEvidence, source: Source) -> bool:
    base: dict[str, JsonValue] = {"source_id": source.source_id, "commit": source.commit, "license": source.license, "path_metadata": source.path_metadata}
    return (value.source_id, value.commit, value.license, value.path_metadata, value.source_digest) == (source.source_id, source.commit, source.license, source.path_metadata, canonical_digest(base))


def _command_matches(value: CommandEvidence, command: CommandBinding) -> bool:
    base: dict[str, JsonValue] = {"command_id": command.command_id, "harness": command.harness, "argv": list(command.argv), "cwd_class": command.cwd_class, "source_id": command.source_id, "source_commit": command.source_commit, "expected_exit": command.expected_exit, "expected_semantic_result": command.expected_semantic_result}
    return (value.command_id, value.harness, value.argv, value.cwd_class, value.source_id, value.source_commit, value.expected_exit, value.expected_semantic_result, value.binding_digest) == (command.command_id, command.harness, command.argv, command.cwd_class, command.source_id, command.source_commit, command.expected_exit, command.expected_semantic_result, canonical_digest(base))


def _satisfies(result: AdapterResult, fixture: Fixture) -> bool:
    actual = dict(result.actual_machine)
    facts = dict(result.facts)
    return all(
        predicate.key in (values := actual if predicate.scope == "actual_machine" else facts)
        and type(values[predicate.key]) is type(predicate.value)
        and values[predicate.key] == predicate.value
        for predicate in fixture.predicates
    )
