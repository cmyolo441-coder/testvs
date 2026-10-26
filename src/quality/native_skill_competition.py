from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Mapping

from ..skills.catalog import builtin_definitions, omh_skill_display_name
from ..skills.render import frontmatter_description


_TOKEN_RE = re.compile(r"[0-9a-z]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
)


@dataclass(frozen=True)
class NativeCompetitionCase:
    case_id: str
    query: str
    omh_skill: str
    native_name: str
    native_description: str
    expected_winner: Literal["native", "omh"]


NATIVE_COMPETITION_CASES = (
    NativeCompetitionCase(
        "browser-native-default",
        "open this url in the browser click login and fill the form",
        "browser-operator",
        "browser",
        "Open a URL in the browser, click a login button, fill a form, and read page content.",
        "native",
    ),
    NativeCompetitionCase(
        "browser-policy-overlay",
        "browser login requires auth confirmation and an observed trace before the action",
        "browser-operator",
        "browser",
        "Open a URL in the browser, click a login button, fill a form, and read page content.",
        "omh",
    ),
    NativeCompetitionCase(
        "files-native-default",
        "list files in this folder search by name and rename one file",
        "workspace-file-operator",
        "files",
        "List files and folders, search by name, read files, copy files, move files, and rename files.",
        "native",
    ),
    NativeCompetitionCase(
        "files-policy-overlay",
        "delete a file only after path scoping destructive action confirmation and evidence",
        "workspace-file-operator",
        "files",
        "List files and folders, search by name, read files, copy files, move files, and rename files.",
        "omh",
    ),
    NativeCompetitionCase(
        "shell-native-default",
        "run the pytest command in the shell and show its output",
        "command-operator",
        "shell",
        "Run shell commands, CLI tools, package managers, and tests, then return command output.",
        "native",
    ),
    NativeCompetitionCase(
        "shell-policy-overlay",
        "run a production migration command with cwd environment safety and result evidence gates",
        "command-operator",
        "shell",
        "Run shell commands, CLI tools, package managers, and tests, then return command output.",
        "omh",
    ),
    NativeCompetitionCase(
        "live-info-native-default",
        "show the weather today in seoul",
        "live-info-operator",
        "weather",
        "Look up live weather today, forecasts, prices, sports scores, maps, and local time.",
        "native",
    ),
    NativeCompetitionCase(
        "live-info-policy-overlay",
        "exchange rate lookup requires provider freshness units and source quality evidence",
        "live-info-operator",
        "weather",
        "Look up live weather today, forecasts, prices, sports scores, maps, and local time.",
        "omh",
    ),
    NativeCompetitionCase(
        "context-explicit-label",
        "use ulw-context before planning",
        "context",
        "memory",
        "Store durable preferences and recall them across sessions.",
        "omh",
    ),
    NativeCompetitionCase(
        "context-fuzzy-project-language",
        "align and correct the terminology and words this repository uses before planning",
        "context",
        "memory",
        "Store durable preferences and recall them across sessions.",
        "omh",
    ),
    NativeCompetitionCase(
        "context-multilingual-explicit",
        "ulw-context로 이 프로젝트의 용어를 정리하고 합의하자",
        "context",
        "memory",
        "Store durable preferences and recall them across sessions.",
        "omh",
    ),
    NativeCompetitionCase(
        "context-memory-neighbor-negative",
        "remember this preference across sessions and recall it later",
        "context",
        "memory",
        "Store durable preferences and recall them across sessions.",
        "native",
    ),
    NativeCompetitionCase(
        "jit-learn-policy-overlay",
        (
            "find the highest-payoff thing to learn for my current blocker, confirm the target first, then rank "
            "source-backed books podcasts creators and courses by immediate application rather than popularity"
        ),
        "jit-learn",
        "recommendations",
        "Recommend popular books, podcasts, creators, and courses for a topic.",
        "omh",
    ),
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOP_WORDS)


def _lexical_score(query: str, name: str, description: str) -> tuple[int, tuple[str, ...]]:
    query_tokens = _tokens(query)
    name_overlap = query_tokens & _tokens(name)
    description_overlap = query_tokens & _tokens(description)
    matched = tuple(sorted(name_overlap | description_overlap))
    return (2 * len(name_overlap)) + len(description_overlap), matched


def build_native_skill_competition_report() -> dict[str, object]:
    definitions = {definition.name: definition for definition in builtin_definitions()}
    results: list[dict[str, object]] = []
    failures: list[str] = []
    for case in NATIVE_COMPETITION_CASES:
        definition = definitions[case.omh_skill]
        omh_score, omh_matches = _lexical_score(
            case.query,
            omh_skill_display_name(definition.name),
            frontmatter_description(definition),
        )
        native_score, native_matches = _lexical_score(
            case.query,
            case.native_name,
            case.native_description,
        )
        if omh_score == native_score:
            actual_winner = "tie"
            winner_score = omh_score
            loser_score = native_score
        elif omh_score > native_score:
            actual_winner = "omh"
            winner_score = omh_score
            loser_score = native_score
        else:
            actual_winner = "native"
            winner_score = native_score
            loser_score = omh_score
        passed = actual_winner == case.expected_winner
        if not passed:
            failures.append(case.case_id)
        results.append(
            {
                "case_id": case.case_id,
                "omh_skill": case.omh_skill,
                "native_name": case.native_name,
                "expected_winner": case.expected_winner,
                "actual_winner": actual_winner,
                "winner_score": winner_score,
                "loser_score": loser_score,
                "omh_score": omh_score,
                "native_score": native_score,
                "omh_matches": list(omh_matches),
                "native_matches": list(native_matches),
                "passed": passed,
                "picker_surface": "generated_frontmatter_name_description",
                "evidence_kind": "lexical_proxy",
            }
        )
    return {
        "schema_version": "omh_native_skill_competition/v1",
        "case_count": len(results),
        "passed_count": len(results) - len(failures),
        "failed_count": len(failures),
        "failures": failures,
        "all_passing": not failures,
        "results": results,
        "claim_boundary": (
            "Deterministic lexical comparison of checked-in representative native descriptions against "
            "generated OMH frontmatter name+description only; this is lexical_proxy evidence, not live Hermes picker, runtime, semantic nomination, or market evidence."
        ),
    }


def native_skill_competition_errors(payload: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    expected_cases = {case.case_id: case for case in NATIVE_COMPETITION_CASES}
    expected_case_ids = set(expected_cases)
    if payload.get("schema_version") != "omh_native_skill_competition/v1":
        errors.append("native competition schema_version is invalid")
    counts: dict[str, int] = {}
    for key in ("case_count", "passed_count", "failed_count"):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"native competition {key} is invalid")
        else:
            counts[key] = value
    failures = payload.get("failures")
    if not isinstance(failures, list) or not all(isinstance(item, str) and item for item in failures):
        errors.append("native competition failures must be a list of case ids")
        failure_count = -1
    else:
        failure_count = len(failures)
    results = payload.get("results")
    result_rows: list[Mapping[str, object]] = []
    if not isinstance(results, list):
        errors.append("native competition results must be a list")
    else:
        for row in results:
            if not isinstance(row, Mapping):
                errors.append("native competition result rows must be objects")
                continue
            case_id = row.get("case_id")
            passed = row.get("passed")
            if not isinstance(case_id, str) or case_id not in expected_case_ids:
                errors.append("native competition result case_id is invalid")
                continue
            if not isinstance(passed, bool):
                errors.append(f"native competition result {case_id} passed flag is invalid")
                continue
            expected = expected_cases[case_id]
            if (
                row.get("omh_skill") != expected.omh_skill
                or row.get("native_name") != expected.native_name
                or row.get("expected_winner") != expected.expected_winner
            ):
                errors.append(f"native competition result {case_id} disagrees with the fixed corpus")
                continue
            score_values: dict[str, int] = {}
            for key in ("winner_score", "loser_score", "omh_score", "native_score"):
                value = row.get(key)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    errors.append(f"native competition result {case_id} {key} is invalid")
                    break
                score_values[key] = value
            if len(score_values) != 4:
                continue
            if score_values["omh_score"] == score_values["native_score"]:
                derived_winner = "tie"
            elif score_values["omh_score"] > score_values["native_score"]:
                derived_winner = "omh"
            else:
                derived_winner = "native"
            if (
                row.get("actual_winner") != derived_winner
                or score_values["winner_score"] != max(score_values["omh_score"], score_values["native_score"])
                or score_values["loser_score"] != min(score_values["omh_score"], score_values["native_score"])
                or passed != (derived_winner == expected.expected_winner)
                or row.get("picker_surface") != "generated_frontmatter_name_description"
                or row.get("evidence_kind") != "lexical_proxy"
            ):
                errors.append(f"native competition result {case_id} winner evidence is inconsistent")
                continue
            result_rows.append(row)
    all_passing = payload.get("all_passing")
    if not isinstance(all_passing, bool):
        errors.append("native competition all_passing must be boolean")
    if len(counts) == 3:
        if counts["case_count"] != len(expected_case_ids):
            errors.append("native competition case_count does not cover the required corpus")
        if counts["passed_count"] + counts["failed_count"] != counts["case_count"]:
            errors.append("native competition counts are inconsistent")
        if failure_count >= 0 and counts["failed_count"] != failure_count:
            errors.append("native competition failure count is inconsistent")
        expected_all_passing = counts["failed_count"] == 0 and counts["passed_count"] == counts["case_count"]
        if isinstance(all_passing, bool) and all_passing != expected_all_passing:
            errors.append("native competition all_passing is inconsistent")
        if isinstance(results, list):
            result_ids = [str(row["case_id"]) for row in result_rows]
            if len(result_rows) != counts["case_count"] or set(result_ids) != expected_case_ids:
                errors.append("native competition results do not cover the required corpus")
            if len(result_ids) != len(set(result_ids)):
                errors.append("native competition results contain duplicate case ids")
            passed_ids = {str(row["case_id"]) for row in result_rows if row["passed"] is True}
            failed_ids = {str(row["case_id"]) for row in result_rows if row["passed"] is False}
            if len(passed_ids) != counts["passed_count"] or len(failed_ids) != counts["failed_count"]:
                errors.append("native competition result rows disagree with aggregate counts")
            if isinstance(failures, list) and set(failures) != failed_ids:
                errors.append("native competition failure ids disagree with result rows")
    return errors


def format_native_skill_competition_summary(payload: dict[str, object]) -> str:
    return "\n".join(
        (
            "Native skill competition",
            f"- cases: {payload['passed_count']}/{payload['case_count']} passing",
            f"- failures: {', '.join(payload['failures']) if payload['failures'] else 'none'}",
            f"- boundary: {payload['claim_boundary']}",
        )
    )
