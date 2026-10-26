"""Pure, independent cross-harness benchmark contract and scorer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeAlias

from . import cross_harness_benchmark_identity as _identity
from .cross_harness_benchmark_model import (
    CommandBinding as CommandBinding,
    Corpus as Corpus,
    Dimension as Dimension,
    DimensionScore as DimensionScore,
    EvaluationReport as EvaluationReport,
    Fixture as Fixture,
    FixtureOutcome as FixtureOutcome,
    Predicate as Predicate,
    ScoreReport as ScoreReport,
    Source as Source,
)
from .cross_harness_benchmark_values import (
    BenchmarkValidationError as BenchmarkValidationError,
    JsonValue,
    commit as _commit,
    corpus_digest,
    digest as _digest,
    flat as _flat,
    integer as _integer,
    items as _items,
    json_map as _map,
    raise_validation as _raise,
    relative as _relative,
    safe as _safe,
    scalar as _scalar,
    shape as _shape,
    text as _text,
    unique as _unique,
)


CANONICAL_CORPUS_DIGEST: Final = _identity.CANONICAL_CORPUS_DIGEST
_require_trusted_corpus = _identity.require_trusted_corpus
SCHEMA: Final = "cross_harness_benchmark/v1"
SUBMISSION_SCHEMA: Final = "cross_harness_benchmark_submission/v1"
EVIDENCE_RANK: Final = {"prepared": 0, "static": 1, "test": 2, "runtime": 3}
Submission: TypeAlias = Mapping[str, JsonValue]


def parse_corpus(raw: Mapping[str, JsonValue]) -> Corpus:
    """Parse the exact frozen corpus shape and reject semantic drift."""
    _shape(raw, {"schema_version", "corpus_id", "corpus_digest", "claim_boundary", "dimensions", "sources", "command_bindings", "fixtures"})
    if _text(raw["schema_version"]) != SCHEMA:
        _raise("unknown_schema")
    declared = _digest(raw["corpus_digest"], "invalid_corpus_digest")
    if declared != CANONICAL_CORPUS_DIGEST:
        _raise("untrusted_corpus_digest")
    if corpus_digest({key: value for key, value in raw.items() if key != "corpus_digest"}) != declared:
        _raise("corpus_digest_mismatch")
    _safe(raw)
    _ = _text(raw["claim_boundary"])
    dimensions = tuple(_parse_dimension(item) for item in _items(raw["dimensions"]))
    sources = tuple(_parse_source(item) for item in _items(raw["sources"]))
    commands = tuple(_parse_command(item) for item in _items(raw["command_bindings"]))
    fixtures = tuple(_parse_fixture(item) for item in _items(raw["fixtures"]))
    _unique((item.id for item in dimensions), "duplicate_dimension")
    _unique((item.source_id for item in sources), "duplicate_source")
    _unique((item.command_id for item in commands), "duplicate_command")
    _unique((item.id for item in fixtures), "duplicate_fixture")
    if len(dimensions) != 10 or sum(item.weight for item in dimensions) != 100:
        _raise("invalid_dimension_weights")
    dimension_ids = {item.id for item in dimensions}
    source_ids = {item.source_id for item in sources}
    command_ids = {item.command_id for item in commands}
    if len(fixtures) != 15 or any(item.dimension not in dimension_ids for item in fixtures):
        _raise("invalid_fixture_corpus")
    if any(item.source_id not in source_ids or item.command_binding_id not in command_ids for item in fixtures):
        _raise("unknown_fixture_binding")
    if any(command.source_id not in source_ids or next(source for source in sources if source.source_id == command.source_id).commit != command.source_commit for command in commands):
        _raise("invalid_command_source_binding")
    return Corpus(_text(raw["corpus_id"]), declared, dimensions, sources, commands, fixtures)


def evaluate_submission(raw: Submission, corpus: Corpus) -> EvaluationReport:
    """Derive fixture status solely from supplied machine facts and bindings."""
    _require_trusted_corpus(corpus)
    _shape(raw, {"schema_version", "corpus_digest", "harness_id", "results"})
    _safe(raw)
    if _text(raw["schema_version"]) != SUBMISSION_SCHEMA:
        _raise("unknown_submission_schema")
    if _digest(raw["corpus_digest"], "invalid_corpus_digest") != corpus.digest:
        _raise("stale_corpus_digest")
    harness = _text(raw["harness_id"])
    indexed: dict[str, Mapping[str, JsonValue]] = {}
    for item in _items(raw["results"]):
        result = _map(item)
        fixture_id = _text(result.get("fixture_id"))
        if fixture_id in indexed:
            _raise("duplicate_result")
        indexed[fixture_id] = result
    if any(result_id not in {item.id for item in corpus.fixtures} for result_id in indexed):
        _raise("unknown_fixture")
    outcomes = tuple(_evaluate_fixture(fixture, indexed.get(fixture.id), corpus, harness) for fixture in corpus.fixtures)
    return EvaluationReport("cross_harness_benchmark_evaluation/v1", corpus.digest, harness, outcomes)


def score_submission(submission: Submission, corpus: Corpus) -> ScoreReport:
    """Evaluate submission claims for deterministic contract compliance."""
    return _score_evaluation(evaluate_submission(submission, corpus), corpus)


def _score_evaluation(report: EvaluationReport, corpus: Corpus) -> ScoreReport:
    _require_trusted_corpus(corpus)
    scores: list[DimensionScore] = []
    for dimension in corpus.dimensions:
        outcomes = tuple(item for item in report.outcomes if item.dimension == dimension.id)
        supported = sum(item.status != "unsupported" for item in outcomes)
        all_pass = bool(outcomes) and all(item.status == "pass" for item in outcomes)
        partial = bool(outcomes) and all(item.status in {"pass", "partial"} for item in outcomes)
        if all_pass:
            earned = dimension.weight
        elif partial:
            earned = dimension.weight // 2
        else:
            earned = 0
        scores.append(DimensionScore(dimension.id, earned, dimension.weight, supported, len(outcomes)))
    total = sum(item.earned for item in scores)
    all_pass = all(item.status == "pass" for item in report.outcomes)
    all_dynamic_runtime_observations_claimed = all(not fixture.dynamic or next(item for item in report.outcomes if item.fixture_id == fixture.id).submission_claims_runtime_observed for fixture in corpus.fixtures)
    if all_pass:
        level = 5 if all_dynamic_runtime_observations_claimed else 4
    elif total >= 70:
        level = 3
    elif total >= 50:
        level = 2
    elif total:
        level = 1
    else:
        level = 0
    reasons: list[str] = []
    if any(item.priority == "P0" and item.status == "fail" for item in report.outcomes):
        reasons.append("p0_failure")
    if not all_pass:
        reasons.append("fixture_not_passed")
    if any(score.earned < dimension.minimum for score, dimension in zip(scores, corpus.dimensions, strict=True)):
        reasons.append("below_dimension_minimum")
    contract_certified = not reasons and level >= 4
    return ScoreReport(total, level, contract_certified, "unverified_submission", False, sum(item.supported for item in scores), len(report.outcomes), tuple(scores), tuple(reasons))


def _evaluate_fixture(fixture: Fixture, raw: Mapping[str, JsonValue] | None, corpus: Corpus, harness: str) -> FixtureOutcome:
    if raw is None:
        return _outcome(fixture, "unsupported", "missing_result", False)
    _shape(raw, {"fixture_id", "adapter_id", "capability_id", "evidence_class", "runtime_observation", "actual_machine", "facts", "source_binding", "command_evidence", "child_results"})
    adapter = _text(raw["adapter_id"])
    capability = _text(raw["capability_id"])
    runtime_observation_claim = _text(raw["runtime_observation"])
    source = next(item for item in corpus.sources if item.source_id == fixture.source_id)
    command = next(item for item in corpus.commands if item.command_id == fixture.command_binding_id)
    _verify_source(_map(raw["source_binding"]), source)
    command_ok = _verify_command(_map(raw["command_evidence"]), command, harness)
    children = tuple(_map(item) for item in _items(raw["child_results"]))
    _unique((_text(item.get("id")) for item in children), "duplicate_child")
    child_failed = any(_child_failed(item) for item in children)
    actual = _flat(raw["actual_machine"])
    facts = _flat(raw["facts"])
    predicates_ok = True
    for predicate in fixture.predicates:
        values = actual if predicate.scope == "actual_machine" else facts
        if predicate.key not in values or type(values[predicate.key]) is not type(predicate.value) or values[predicate.key] != predicate.value:
            predicates_ok = False
            break
    submission_claims_runtime_observed = runtime_observation_claim == "observed"
    evidence = _text(raw["evidence_class"])
    if evidence not in EVIDENCE_RANK or runtime_observation_claim not in {"observed", "prepared_not_observed", "not_applicable"}:
        _raise("invalid_evidence_state")
    if adapter != fixture.adapter_id:
        return _outcome(fixture, "unsupported", "adapter_unavailable", submission_claims_runtime_observed)
    if capability != fixture.capability_id:
        return _outcome(fixture, "unsupported", "capability_unavailable", submission_claims_runtime_observed)
    if child_failed:
        return _outcome(fixture, "fail", "child_failed", submission_claims_runtime_observed)
    if not command_ok:
        return _outcome(fixture, "fail", "command_result_mismatch", submission_claims_runtime_observed)
    if not predicates_ok:
        return _outcome(fixture, "fail", "predicate_mismatch", submission_claims_runtime_observed)
    if EVIDENCE_RANK[evidence] < EVIDENCE_RANK[fixture.required_evidence_class]:
        return _outcome(fixture, "partial", "insufficient_evidence_class", submission_claims_runtime_observed)
    if fixture.dynamic and evidence == "runtime" and not submission_claims_runtime_observed:
        return _outcome(fixture, "partial", "runtime_not_observed", submission_claims_runtime_observed)
    return _outcome(fixture, "pass", "predicate_satisfied", submission_claims_runtime_observed, no_reason=True)


def _parse_dimension(value: JsonValue) -> Dimension:
    raw = _map(value)
    _shape(raw, {"id", "weight", "minimum"})
    weight = _integer(raw["weight"])
    minimum = _integer(raw["minimum"])
    if weight <= 0 or minimum < 0 or minimum > weight:
        _raise("invalid_dimension")
    return Dimension(_text(raw["id"]), weight, minimum)


def _parse_source(value: JsonValue) -> Source:
    raw = _map(value)
    _shape(raw, {"source_id", "commit", "license", "path_metadata"})
    return Source(_text(raw["source_id"]), _commit(raw["commit"]), _text(raw["license"]), _relative(raw["path_metadata"]))


def _parse_command(value: JsonValue) -> CommandBinding:
    raw = _map(value)
    _shape(raw, {"command_id", "harness", "argv", "cwd_class", "source_id", "source_commit", "expected_exit", "expected_semantic_result"})
    argv = tuple(_text(item) for item in _items(raw["argv"]))
    if not argv:
        _raise("invalid_argv")
    return CommandBinding(_text(raw["command_id"]), _text(raw["harness"]), argv, _text(raw["cwd_class"]), _text(raw["source_id"]), _commit(raw["source_commit"]), _integer(raw["expected_exit"]), _text(raw["expected_semantic_result"]))


def _parse_fixture(value: JsonValue) -> Fixture:
    raw = _map(value)
    _shape(raw, {"id", "dimension", "priority", "dynamic", "prompt", "setup", "expected_machine", "required_evidence_class", "evaluator", "adapter_id", "capability_id", "source_id", "command_binding_id"})
    prompt = _map(raw["prompt"])
    setup = _map(raw["setup"])
    _shape(prompt, {"intent", "constraint"})
    _shape(setup, {"profile", "mode"})
    _ = tuple(_text(item) for item in (*prompt.values(), *setup.values()))
    if _text(raw["evaluator"]) != "predicate_subset/v1":
        _raise("unknown_evaluator")
    priority = _text(raw["priority"])
    dynamic = raw["dynamic"]
    if priority not in {"P0", "P1"} or not isinstance(dynamic, bool):
        _raise("wrong_type")
    predicates = tuple(_parse_predicate(item) for item in _items(raw["expected_machine"]))
    if not predicates:
        _raise("missing_predicate")
    evidence = _text(raw["required_evidence_class"])
    if evidence not in EVIDENCE_RANK:
        _raise("invalid_evidence_class")
    return Fixture(_text(raw["id"]), _text(raw["dimension"]), priority, dynamic, predicates, evidence, _text(raw["adapter_id"]), _text(raw["capability_id"]), _text(raw["source_id"]), _text(raw["command_binding_id"]))


def _parse_predicate(value: JsonValue) -> Predicate:
    raw = _map(value)
    _shape(raw, {"scope", "key", "operator", "value"})
    scope = _text(raw["scope"])
    if scope not in {"actual_machine", "facts"} or _text(raw["operator"]) != "eq":
        _raise("invalid_predicate")
    return Predicate(scope, _text(raw["key"]), _scalar(raw["value"]))


def _verify_source(raw: Mapping[str, JsonValue], source: Source) -> None:
    _shape(raw, {"source_id", "commit", "license", "path_metadata", "source_digest"})
    base: dict[str, JsonValue] = {"source_id": source.source_id, "commit": source.commit, "license": source.license, "path_metadata": source.path_metadata}
    supplied = {key: raw[key] for key in base}
    if supplied != base:
        _raise("source_binding_mismatch")
    if _digest(raw["source_digest"], "invalid_source_digest") != corpus_digest(base):
        _raise("stale_source_digest")


def _verify_command(raw: Mapping[str, JsonValue], command: CommandBinding, harness: str) -> bool:
    fields = {"command_id", "harness", "argv", "cwd_class", "source_id", "source_commit", "expected_exit", "expected_semantic_result", "binding_digest", "observed_exit", "observed_semantic_result"}
    _shape(raw, fields)
    base: dict[str, JsonValue] = {"command_id": command.command_id, "harness": command.harness, "argv": list(command.argv), "cwd_class": command.cwd_class, "source_id": command.source_id, "source_commit": command.source_commit, "expected_exit": command.expected_exit, "expected_semantic_result": command.expected_semantic_result}
    supplied: dict[str, JsonValue] = {"command_id": _text(raw["command_id"]), "harness": _text(raw["harness"]), "argv": [_text(item) for item in _items(raw["argv"])], "cwd_class": _text(raw["cwd_class"]), "source_id": _text(raw["source_id"]), "source_commit": _commit(raw["source_commit"]), "expected_exit": _integer(raw["expected_exit"]), "expected_semantic_result": _text(raw["expected_semantic_result"])}
    if harness != command.harness or supplied != base:
        _raise("command_binding_mismatch")
    if _digest(raw["binding_digest"], "invalid_binding_digest") != corpus_digest(base):
        _raise("stale_binding_digest")
    return _integer(raw["observed_exit"]) == command.expected_exit and _text(raw["observed_semantic_result"]) == command.expected_semantic_result


def _child_failed(raw: Mapping[str, JsonValue]) -> bool:
    _shape(raw, {"id", "result"})
    _ = _text(raw["id"])
    result = _text(raw["result"])
    if result not in {"pass", "fail"}:
        _raise("invalid_child_result")
    return result == "fail"


def _outcome(fixture: Fixture, status: str, reason: str, submission_claims_runtime_observed: bool, *, no_reason: bool = False) -> FixtureOutcome:
    return FixtureOutcome(fixture.id, fixture.dimension, fixture.priority, status, () if no_reason else (reason,), submission_claims_runtime_observed)
