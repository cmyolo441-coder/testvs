from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Final, Literal, assert_never

from ..quality.cross_harness_benchmark import (
    BenchmarkValidationError,
    EvaluationReport,
    ScoreReport,
    evaluate_submission,
    parse_corpus,
    score_submission,
)
from ..quality.cross_harness_benchmark_input import (
    MAX_INPUT_BYTES,
    BenchmarkJsonInputError,
    decode_benchmark_bytes,
)
from ..quality.cross_harness_benchmark_values import JsonValue


INPUT_SCHEMA: Final = "cross_harness_benchmark_cli_input/v1"
ERROR_SCHEMA: Final = "cross_harness_benchmark_cli_error/v1"
VALIDATION_SCHEMA: Final = "cross_harness_benchmark_validation/v1"
SCORE_SCHEMA: Final = "cross_harness_benchmark_score/v1"
REPORT_SCHEMA: Final = "cross_harness_benchmark_report/v1"
CommandName = Literal["validate", "score", "report"]


@dataclass(frozen=True, slots=True)
class BenchmarkCliInputError(ValueError):
    reason_code: str


@dataclass(frozen=True, slots=True)
class BenchmarkCliInput:
    corpus: dict[str, JsonValue]
    submission: dict[str, JsonValue]
    claim_boundary: str


def cmd_benchmark_validate(args: argparse.Namespace) -> int:
    return _run_command(args, "validate")


def cmd_benchmark_score(args: argparse.Namespace) -> int:
    return _run_command(args, "score")


def cmd_benchmark_report(args: argparse.Namespace) -> int:
    return _run_command(args, "report")


def _run_command(args: argparse.Namespace, command: CommandName) -> int:
    try:
        benchmark_input = _read_input(args)
        corpus = parse_corpus(benchmark_input.corpus)
        match command:
            case "validate":
                evaluation = evaluate_submission(benchmark_input.submission, corpus)
                payload = _validation_payload(evaluation)
                status = 0
            case "score":
                score = score_submission(benchmark_input.submission, corpus)
                payload = _score_payload(score)
                status = 0 if score.contract_certified else 1
            case "report":
                evaluation = evaluate_submission(benchmark_input.submission, corpus)
                score = score_submission(benchmark_input.submission, corpus)
                payload = _report_payload(
                    benchmark_input.claim_boundary, evaluation, score
                )
                status = 0 if score.contract_certified else 1
            case unreachable:
                assert_never(unreachable)
    except (BenchmarkCliInputError, BenchmarkJsonInputError) as error:
        _print_json(_error_payloads((error.reason_code,)))
        return 2
    except BenchmarkValidationError as error:
        _print_json(_error_payloads(error.reason_codes))
        return 2
    _print_json(payload)
    return status


def _read_input(args: argparse.Namespace) -> BenchmarkCliInput:
    input_path: str | None = args.input_file
    use_stdin = bool(args.stdin)
    if input_path is not None and use_stdin:
        raise BenchmarkCliInputError("conflicting_input")
    if use_stdin:
        try:
            encoded = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        except AttributeError:
            try:
                encoded = sys.stdin.read(MAX_INPUT_BYTES + 1).encode("utf-8")
            except UnicodeEncodeError as error:
                raise BenchmarkJsonInputError("invalid_utf8") from error
    elif input_path is not None:
        try:
            with Path(input_path).open("rb") as stream:
                encoded = stream.read(MAX_INPUT_BYTES + 1)
        except OSError as error:
            raise BenchmarkCliInputError("input_file_unavailable") from error
    else:
        raise BenchmarkCliInputError("missing_input")
    value = decode_benchmark_bytes(encoded)
    if not isinstance(value, dict):
        raise BenchmarkCliInputError("input_must_be_object")
    if set(value) != {"schema_version", "corpus", "submission"}:
        raise BenchmarkCliInputError("invalid_input_shape")
    if value["schema_version"] != INPUT_SCHEMA:
        raise BenchmarkCliInputError("unknown_input_schema")
    corpus = value["corpus"]
    submission = value["submission"]
    if not isinstance(corpus, dict) or not isinstance(submission, dict):
        raise BenchmarkCliInputError("input_must_be_object")
    claim_boundary = corpus.get("claim_boundary")
    if not isinstance(claim_boundary, str):
        raise BenchmarkCliInputError("invalid_claim_boundary")
    return BenchmarkCliInput(corpus, submission, claim_boundary)


def _validation_payload(evaluation: EvaluationReport) -> dict[str, JsonValue]:
    return {
        "schema_version": VALIDATION_SCHEMA,
        "valid": True,
        "evaluation": _evaluation_payload(evaluation),
        "outcomes": _outcomes_payload(evaluation),
    }


def _report_payload(
    claim_boundary: str, evaluation: EvaluationReport, score: ScoreReport
) -> dict[str, JsonValue]:
    unsupported: list[JsonValue] = [
        outcome.fixture_id
        for outcome in evaluation.outcomes
        if outcome.status == "unsupported"
    ]
    unknowns: list[JsonValue] = [
        outcome.fixture_id
        for outcome in evaluation.outcomes
        if outcome.status in {"unsupported", "partial"}
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "claim_boundary": claim_boundary,
        "evaluation": _evaluation_payload(evaluation),
        "score": _score_payload(score),
        "coverage": {
            "supported": score.coverage_supported,
            "total": score.coverage_total,
            "unsupported": len(unsupported),
        },
        "unsupported": unsupported,
        "unknowns": unknowns,
    }


def _score_payload(score: ScoreReport) -> dict[str, JsonValue]:
    dimensions: list[JsonValue] = [
        {
            "dimension": dimension.dimension,
            "earned": dimension.earned,
            "available": dimension.available,
            "supported": dimension.supported,
            "fixtures": dimension.fixtures,
        }
        for dimension in score.dimensions
    ]
    return {
        "schema_version": SCORE_SCHEMA,
        "total": score.total,
        "level": score.level,
        "contract_certified": score.contract_certified,
        "evidence_authenticity": score.evidence_authenticity,
        "execution_verified": score.execution_verified,
        "coverage_supported": score.coverage_supported,
        "coverage_total": score.coverage_total,
        "dimensions": dimensions,
        "reason_codes": list(score.reason_codes),
    }


def _evaluation_payload(evaluation: EvaluationReport) -> dict[str, JsonValue]:
    return {
        "schema_version": evaluation.schema_version,
        "corpus_digest": evaluation.corpus_digest,
        "harness_id": evaluation.harness_id,
        "outcomes": _outcomes_payload(evaluation),
    }


def _outcomes_payload(evaluation: EvaluationReport) -> list[JsonValue]:
    outcomes: list[JsonValue] = [
        {
            "fixture_id": outcome.fixture_id,
            "dimension": outcome.dimension,
            "priority": outcome.priority,
            "status": outcome.status,
            "reason_codes": list(outcome.reason_codes),
            "submission_claims_runtime_observed": outcome.submission_claims_runtime_observed,
        }
        for outcome in evaluation.outcomes
    ]
    return outcomes


def _error_payloads(reason_codes: tuple[str, ...]) -> dict[str, JsonValue]:
    return {
        "schema_version": ERROR_SCHEMA,
        "valid": False,
        "reason_codes": list(reason_codes),
    }


def _print_json(value: dict[str, JsonValue]) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _add_benchmark_commands(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    benchmark = sub.add_parser(
        "benchmark",
        help="Agent/operator-facing offline benchmark validation and scoring (does not execute agents).",
        description=(
            "Agent/operator-facing offline benchmark assessment. It reads exactly one explicit "
            "JSON envelope and does not execute agents, dispatch work, read runtime state, or write artifacts."
        ),
    )
    commands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    _add_benchmark_command(
        commands,
        "validate",
        "Validate an explicit benchmark envelope without contract certification.",
        cmd_benchmark_validate,
    )
    _add_benchmark_command(
        commands,
        "score",
        "Score an explicit benchmark envelope; results without contract certification exit nonzero.",
        cmd_benchmark_score,
    )
    _add_benchmark_command(
        commands,
        "report",
        "Report score, coverage, unknowns, and claim boundary; results without contract certification exit nonzero.",
        cmd_benchmark_report,
    )


def _add_benchmark_command(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    name: CommandName,
    help_text: str,
    handler: Callable[[argparse.Namespace], int],
) -> None:
    command = commands.add_parser(name, help=help_text, description=help_text)
    command.add_argument(
        "--input", dest="input_file", help="Path to one benchmark JSON envelope."
    )
    command.add_argument(
        "--stdin",
        action="store_true",
        help="Read one benchmark JSON envelope from standard input.",
    )
    command.set_defaults(func=handler)
