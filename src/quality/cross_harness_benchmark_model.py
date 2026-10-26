from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .cross_harness_benchmark_values import JsonScalar


@dataclass(frozen=True, slots=True)
class Dimension:
    id: str
    weight: int
    minimum: int


@dataclass(frozen=True, slots=True)
class Source:
    source_id: str
    commit: str
    license: str
    path_metadata: str


@dataclass(frozen=True, slots=True)
class CommandBinding:
    command_id: str
    harness: str
    argv: tuple[str, ...]
    cwd_class: str
    source_id: str
    source_commit: str
    expected_exit: int
    expected_semantic_result: str


@dataclass(frozen=True, slots=True)
class Predicate:
    scope: str
    key: str
    value: JsonScalar


@dataclass(frozen=True, slots=True)
class Fixture:
    id: str
    dimension: str
    priority: str
    dynamic: bool
    predicates: tuple[Predicate, ...]
    required_evidence_class: str
    adapter_id: str
    capability_id: str
    source_id: str
    command_binding_id: str


@dataclass(frozen=True, slots=True)
class Corpus:
    corpus_id: str
    digest: str
    dimensions: tuple[Dimension, ...]
    sources: tuple[Source, ...]
    commands: tuple[CommandBinding, ...]
    fixtures: tuple[Fixture, ...]


@dataclass(frozen=True, slots=True)
class FixtureOutcome:
    fixture_id: str
    dimension: str
    priority: str
    status: str
    reason_codes: tuple[str, ...]
    submission_claims_runtime_observed: bool


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    schema_version: str
    corpus_digest: str
    harness_id: str
    outcomes: tuple[FixtureOutcome, ...]


@dataclass(frozen=True, slots=True)
class DimensionScore:
    dimension: str
    earned: int
    available: int
    supported: int
    fixtures: int


@dataclass(frozen=True, slots=True)
class ScoreReport:
    total: int
    level: int
    contract_certified: bool
    evidence_authenticity: Literal["unverified_submission"]
    execution_verified: Literal[False]
    coverage_supported: int
    coverage_total: int
    dimensions: tuple[DimensionScore, ...]
    reason_codes: tuple[str, ...]
