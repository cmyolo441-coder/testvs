"""The two shipped routing corpora, projected as typed questions and scored.

`src/quality/routing_precision.py` holds the corpora OMH's deterministic router
is measured against, and it measures one arm: the router itself. This module
turns the same cases into a question any answerer can answer -- one relative
Choice over the router's own candidate shortlist plus `none`, and one absolute
yes/no per candidate -- so a second arm can be measured on exactly the cases
the first one is gated on.

Two rules hold this together, and both exist because breaking either publishes
a false number about OMH's own router:

- **Expected answers come from the corpora, never from a rule restated here.**
  A negative control expects no workflow; an intervention case expects what its
  own record says it expects, including the one case whose correct answer is
  "do not open a workflow".
- **The deterministic arm's verdicts come from the producer's evaluators.**
  `precision_case_verdict` and `intervention_case_verdict` are the accessors,
  and the deterministic arm must reproduce a zero-overroute, zero-missed
  reading: `routing_precision_errors(build_routing_precision_demo(...)) == []`
  is the authority, and a test pins that the two agree. A re-derived
  over-route predicate is the specific way this goes wrong, because `clarify`
  with a named candidate is a pass in the negative corpus and the expected
  intervention in the positive one. Every item also carries the producer's
  pass verdict, and `routing_question_score_errors` fails on it, so a report
  cannot read clean over a corpus the gate reads as red.

Scoring is offline and deterministic. Nothing here calls a model, reads a
credential, or reaches the network; answers arrive as rows somebody else
produced.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from ..coding.context_safety import compact_visible_text
from ..ingress import CHAT_SOURCES
from ..plugin_bundle.omh.route_answer_consistency import INVALID_ANSWER_VERDICT, invalid_answer_reasons
from ..routing.route_question import (
    FITS_CLARIFY_THRESHOLD,
    FITS_DISPATCH_THRESHOLD,
    FIT_QUESTION_PREFIX,
    NO_WORKFLOW_OPTION,
    ROUTE_CHOICE_KEY,
    build_route_question_for_candidate_handoff,
    build_route_question_from_candidates,
    clean_skill_description,
    fit_question_skill,
    message_digest,
    normalized_route_candidates,
)
from ..skills.catalog import builtin_definitions
from .reported_rate import reported_rate
from .routing_precision import (
    ROUTING_INTERVENTION_CASES,
    ROUTING_PRECISION_CASES,
    ROUTING_PRECISION_SCHEMA_VERSION,
    intervention_case_interaction,
    intervention_case_verdict,
    precision_case_interaction,
    precision_case_verdict,
)

ROUTING_QUESTION_CORPUS_SCHEMA_VERSION = "routing_question_corpus/v1"
ROUTING_QUESTION_ANSWERS_SCHEMA_VERSION = "routing_question_answers/v1"
ROUTING_QUESTION_SCORE_SCHEMA_VERSION = "routing_question_score/v1"
ROUTE_QUESTION_ANSWER_SCHEMA_VERSION = "route_question_answer/v1"

NEGATIVE_CONTROL_CORPUS = "negative_control"
INTERVENTION_CORPUS = "intervention"

DETERMINISTIC_ARM = "deterministic"
UNKNOWN_ARM = "unknown"

# Where an item's question came from, and whether a live route would ask it.
#
# A live route attaches a question only where it could not decide, and it
# builds that question from the candidate handoff. Every other case is one the
# router resolved, so live never asks about it: the corpus still carries a
# question for it -- an offline arm can answer the whole corpus -- but no
# recorded live answer can ever exist for it, and counting those cases against
# a live arm would report it as having failed to answer questions it was never
# asked.
HANDOFF_QUESTION_SOURCE = "candidate_handoff"
RECOMMENDATIONS_QUESTION_SOURCE = "recommendations"

# The action vocabulary an answer set resolves to. `none` covers both "no
# workflow applies" and the router's own `fallback`, which is the same
# outcome under a different name.
DISPATCH_ACTION = "dispatch"
CLARIFY_ACTION = "clarify"
NONE_ACTION = "none"

# Bounds on every untrusted read. A corpus is a file an operator points at, an
# answers file is written by a live model or by whoever ran an external arm,
# and a record directory is both. None of them is read whole before its size is
# known: a file over its cap is refused by name rather than loaded.
MAX_CORPUS_BYTES = 64 * 1024 * 1024
MAX_ANSWER_SOURCE_BYTES = 16 * 1024 * 1024
MAX_ANSWER_RECORD_BYTES = 1024 * 1024

# Bounds on every untrusted string that reaches a formatted report line.
# `format_routing_question_score` prints an arm name, a row reference and a fit
# question's skill suffix one per line, and all three come out of a file
# somebody else wrote: an embedded newline forges a standalone line in a report
# attached to a PR, and `\x1b[2K\r` repaints the line above it. The same
# two-step the repo already uses for captured child output -- drop the control
# characters, then bound the length -- is applied at the point each value is
# read, so the stored payload carries what the report prints.
MAX_REPORT_FIELD_CHARS = 120
_UNSAFE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# How many colliding case ids an ambiguous record names, and how many failed
# cases the deterministic arm names. Both lists exist to make a report
# actionable, not to reproduce the corpus.
MAX_NAMED_CASES = 5

_CORPUS_CLAIM_BOUNDARY = (
    "These items are prepared questions over the two shipped routing corpora. "
    "An exported corpus is not a measurement, and an answer recorded against "
    "it is a routing judgment by whoever answered, not execution, review, CI, "
    "or merge evidence."
)

_SCORE_CLAIM_BOUNDARY = (
    "Every count is over the cases this arm actually answered; unanswered and "
    "malformed rows are excluded and named, never counted as correct. The "
    "deterministic arm's over-route verdict is the routing-precision corpus's "
    "own, and `producer_failed` carries that corpus's pass verdict on every "
    "case, so this report cannot read clean while the gate reads red. "
    "`agree` is an exact match on both questions and is not a pass metric: the router "
    "answers `clarify` on negative controls where asking one question is the "
    "correct non-hijacking behaviour, and those count as disagreements here "
    "while staying passes there. An arm answering recorded live routes is "
    "scored only on the cases a live route asks about; the rest are named "
    "`not_live_joinable` and excluded from its denominators, never counted as "
    "questions it failed to answer. An `invalid_answer` -- a Choice distribution "
    "that omits an offered option, does not sum to one, or does not peak at the "
    "chosen option -- is no opinion: it is counted and named, and it leaves every "
    "accuracy denominator. A `digest_mismatch` is an answer about "
    "this request asked over a shortlist cut differently by the surface that "
    "asked it; it is scored and reported, not dropped. An arm's score "
    "describes these corpora at this revision and nothing beyond them."
)


class RoutingQuestionCorpusError(ValueError):
    """A corpus or answer source that cannot be read as what it claims to be."""


@dataclass(frozen=True)
class AnswerRecord:
    """One answer set as it arrived, with where it came from kept attached."""

    ref: str
    arm: str
    case_id: str
    question_digest: str
    answers: dict[str, Any]
    error: str = ""
    # The message this answer was about. It is the primary join key, because a
    # question digest also covers the candidate shortlist and the shortlist is
    # cut differently per surface -- a route-hint question carries two
    # candidates where a full route carries three -- so the same request
    # produces a different digest depending on which surface asked. The message
    # does not move.
    message_sha256: str = ""
    # True when this came from a `route_question_answer/v1` record, which is
    # what a live route writes. It decides which denominator the arm is read
    # against: a live arm can only ever answer the items a live route asks.
    from_live_record: bool = False


def report_safe_text(value: object, *, max_chars: int = MAX_REPORT_FIELD_CHARS) -> str:
    """Bound one untrusted string to something a single report line can carry."""
    return compact_visible_text(_UNSAFE_CONTROL_RE.sub(" ", str(value or "")), max_chars=max_chars)


def _read_bounded_text(path: Path, *, limit: int, label: str) -> str:
    """Read a file only when it is small enough to be read whole.

    The cap is applied to the bytes on disk rather than to a stat, so a file
    that reports a zero size and then yields gigabytes is refused like any
    other oversized file.
    """
    target = Path(path)
    try:
        with target.open("rb") as handle:
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise RoutingQuestionCorpusError(f"{label} is not readable: {report_safe_text(exc)}") from exc
    if len(raw) > limit:
        raise RoutingQuestionCorpusError(f"{label} exceeds the {limit}-byte cap: {target}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RoutingQuestionCorpusError(f"{label} is not UTF-8 text: {target}") from exc


def read_routing_question_corpus(path: Path) -> dict[str, Any]:
    """Read an exported corpus from a path, bounded and named on refusal."""
    text = _read_bounded_text(Path(path), limit=MAX_CORPUS_BYTES, label="corpus")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RoutingQuestionCorpusError(f"corpus is not readable JSON: {report_safe_text(exc.msg)}") from exc
    if not isinstance(document, dict):
        raise RoutingQuestionCorpusError(f"corpus is not an object: {Path(path)}")
    return document


def _skill_descriptions() -> dict[str, str]:
    return {definition.name: definition.description for definition in builtin_definitions()}


def _candidates_from_route(
    route: Mapping[str, Any],
    descriptions: Mapping[str, str],
    *,
    limit: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    recommendations = route.get("recommendations")
    if not isinstance(recommendations, list):
        return rows
    for entry in recommendations:
        if len(rows) >= limit:
            break
        if not isinstance(entry, Mapping):
            continue
        skill = str(entry.get("skill") or "").strip()
        if not skill:
            continue
        rows.append({"skill": skill, "description": clean_skill_description(descriptions.get(skill, ""))})
    return rows


def deterministic_route_reading(route: Mapping[str, Any]) -> tuple[str, str]:
    """The router's own answer to the two questions, from a route or a routing record.

    Public for the shadow report, which reads it off a recorded
    `routing.json`; both carry the same `action`, `selected_skill`, and
    `candidate_skill` fields, so one reading serves both.
    """
    return _deterministic_reading(route)


def _deterministic_reading(route: Mapping[str, Any]) -> tuple[str, str]:
    """The router's own answer to the same two questions.

    `fallback` is reported as `none`: the router declined to open a workflow,
    which is the outcome the `none` option exists to express.
    """
    action = str(route.get("action") or "")
    if action == DISPATCH_ACTION:
        return DISPATCH_ACTION, str(route.get("selected_skill") or NO_WORKFLOW_OPTION)
    if action == CLARIFY_ACTION:
        return CLARIFY_ACTION, str(route.get("candidate_skill") or NO_WORKFLOW_OPTION)
    return NONE_ACTION, NO_WORKFLOW_OPTION


def _expected_for_intervention(case: Any) -> dict[str, str]:
    """The expected answer a single intervention case already carries.

    `dispatch` names its workflow. `clarify` names its candidate when the case
    pins one, and `none` when it does not -- the corpus does not pin a skill
    there, so neither does this. `fallback` is an intervention case whose
    correct answer is to open nothing, so it maps to `none` on both fields.
    """
    expected_action = str(case.expected_route_action or "")
    if expected_action == DISPATCH_ACTION:
        return {"action": DISPATCH_ACTION, "choice": str(case.expected_workflow or NO_WORKFLOW_OPTION)}
    if expected_action == CLARIFY_ACTION:
        return {"action": CLARIFY_ACTION, "choice": str(case.expected_candidate or NO_WORKFLOW_OPTION)}
    return {"action": NONE_ACTION, "choice": NO_WORKFLOW_OPTION}


def routing_question_contract() -> dict[str, object]:
    """The question contract every exported corpus carries.

    It is its own function so a consumer that cannot import this package --
    the benchmark lane talks to the product as an executable -- has one
    producer to pin its own copy of these literals against.
    """
    return {
        "choice_key": ROUTE_CHOICE_KEY,
        "fit_prefix": FIT_QUESTION_PREFIX,
        "none_option": NO_WORKFLOW_OPTION,
        "thresholds": {
            "fits_dispatch": FITS_DISPATCH_THRESHOLD,
            "fits_clarify": FITS_CLARIFY_THRESHOLD,
        },
    }


def build_routing_question_corpus(*, source: str = "discord", limit: int = 3) -> dict[str, object]:
    """Project both shipped routing corpora into typed questions.

    `limit` caps how many of the router's own recommendations become Choice
    options; it does not change how a case is routed, so the verdicts below
    stay the producer's. No case is added to either corpus here: this is a
    projection, and the corpora's exact-count pins are the reason it has to
    stay one.
    """
    if source not in CHAT_SOURCES:
        raise RoutingQuestionCorpusError(f"unsupported corpus source: {source}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise RoutingQuestionCorpusError("corpus limit must be a positive integer")
    descriptions = _skill_descriptions()
    items: list[dict[str, object]] = []
    for case in ROUTING_PRECISION_CASES:
        interaction = precision_case_interaction(case, source=source)
        route = interaction.get("route")
        route = route if isinstance(route, Mapping) else {}
        verdict = precision_case_verdict(case, interaction, source=source)
        action, choice = _deterministic_reading(route)
        items.append(
            _corpus_item(
                case_id=case.id,
                corpus=NEGATIVE_CONTROL_CORPUS,
                message=case.message,
                message_sha256=message_digest(case.message),
                route=route,
                descriptions=descriptions,
                limit=limit,
                expected={"action": NONE_ACTION, "choice": NO_WORKFLOW_OPTION},
                deterministic={
                    "action": action,
                    "choice": choice,
                    "overrouted": bool(verdict["overrouted"]),
                    "case_passed": bool(verdict["passed"]),
                },
            )
        )
    for case in ROUTING_INTERVENTION_CASES:
        interaction = intervention_case_interaction(case, source=source)
        route = interaction.get("route")
        route = route if isinstance(route, Mapping) else {}
        verdict = intervention_case_verdict(case, interaction, source=source)
        action, choice = _deterministic_reading(route)
        items.append(
            _corpus_item(
                case_id=case.id,
                corpus=INTERVENTION_CORPUS,
                message=case.message,
                message_sha256=message_digest(case.message),
                route=route,
                descriptions=descriptions,
                limit=limit,
                expected=_expected_for_intervention(case),
                deterministic={
                    "action": action,
                    "choice": choice,
                    # An intervention case is one the router is supposed to act
                    # on, so the negative corpus's over-route reading is not
                    # defined for it; `case_passed` is this corpus's verdict.
                    "overrouted": False,
                    "case_passed": bool(verdict["passed"]),
                },
            )
        )
    return {
        "schema_version": ROUTING_QUESTION_CORPUS_SCHEMA_VERSION,
        "source": source,
        "generated_from": {
            "schema": ROUTING_PRECISION_SCHEMA_VERSION,
            "case_count": len(ROUTING_PRECISION_CASES),
            "intervention_case_count": len(ROUTING_INTERVENTION_CASES),
        },
        "question_contract": routing_question_contract(),
        # Both counts, because the difference is what a live arm can be scored
        # on. Only a handoff item's question is one a live route ever asks, so
        # a report that quoted the item count as a live denominator would be
        # quoting questions nobody was asked.
        "question_sources": {
            HANDOFF_QUESTION_SOURCE: sum(
                1 for item in items if item.get("question_source") == HANDOFF_QUESTION_SOURCE
            ),
            RECOMMENDATIONS_QUESTION_SOURCE: sum(
                1 for item in items if item.get("question_source") == RECOMMENDATIONS_QUESTION_SOURCE
            ),
            "live_joinable": sum(1 for item in items if item.get("live_joinable")),
        },
        "items": items,
        "claim_boundary": _CORPUS_CLAIM_BOUNDARY,
    }


def _corpus_item(
    *,
    case_id: str,
    corpus: str,
    message: str,
    message_sha256: str,
    route: Mapping[str, Any],
    descriptions: Mapping[str, str],
    limit: int,
    expected: Mapping[str, str],
    deterministic: Mapping[str, object],
) -> dict[str, object]:
    # An undecidable route carries a candidate handoff, and that handoff is
    # what a live route builds its question from. Mirroring it here -- same
    # candidates, same reasons, same builder call -- is what lets an answer
    # recorded on a live route be scored against this corpus at all. Reading
    # `route["recommendations"]` instead would ask about the same skills with
    # different text and different reasons, and every live record would land
    # as unmatched with nothing saying why.
    handoff = route.get("candidate_handoff")
    if isinstance(handoff, Mapping):
        question = build_route_question_for_candidate_handoff(handoff, message=message)
        candidates = normalized_route_candidates(
            [row for row in handoff.get("candidates", []) if isinstance(row, Mapping)]
        )
        question_source = HANDOFF_QUESTION_SOURCE
    else:
        # A decided route. Live never questions it, so this question exists for
        # the offline arms alone and `limit` is the only thing cutting it.
        candidates = normalized_route_candidates(_candidates_from_route(route, descriptions, limit=limit))
        reason = str(route.get("reason") or "")
        question = build_route_question_from_candidates(
            candidates,
            message_sha256=message_sha256,
            reasons=(reason,) if reason else (),
        )
        question_source = RECOMMENDATIONS_QUESTION_SOURCE
    return {
        "case_id": case_id,
        "corpus": corpus,
        "message": message,
        "message_sha256": message_sha256,
        "question_source": question_source,
        "live_joinable": question_source == HANDOFF_QUESTION_SOURCE,
        "candidates": candidates,
        "question": question,
        "expected": dict(expected),
        "deterministic": dict(deterministic),
    }


def corpus_shape_errors(corpus: object) -> tuple[str, ...]:
    """Return why a value is not a readable routing-question corpus."""
    if not isinstance(corpus, Mapping):
        return ("corpus is not an object",)
    errors: list[str] = []
    if corpus.get("schema_version") != ROUTING_QUESTION_CORPUS_SCHEMA_VERSION:
        errors.append("corpus schema_version is not " + ROUTING_QUESTION_CORPUS_SCHEMA_VERSION)
    items = corpus.get("items")
    if not isinstance(items, list) or not items:
        errors.append("corpus carries no items")
        return tuple(errors)
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"item {index} is not an object")
            continue
        if not str(item.get("case_id") or ""):
            errors.append(f"item {index} has no case_id")
        if not isinstance(item.get("expected"), Mapping):
            errors.append(f"item {index} has no expected answer")
        if not isinstance(item.get("question"), Mapping):
            errors.append(f"item {index} has no question block")
        # The message digest is the primary join key for a recorded answer, and
        # `live_joinable` decides whether a live arm is scored on this item at
        # all. A corpus missing either cannot be joined or denominated
        # correctly, and both failures are silent -- an arm reads as having
        # answered nothing -- so they are refused here instead.
        if not str(item.get("message_sha256") or ""):
            errors.append(f"item {index} has no message_sha256")
        if not isinstance(item.get("live_joinable"), bool):
            errors.append(f"item {index} does not say whether a live route asks it")
        # `case_passed` is the producer's own verdict on this case, and the
        # score report fails on it. A corpus that does not carry it cannot be
        # scored as a reading of the routing-precision gate, only as a
        # re-derivation of one -- which is the failure this module exists to
        # avoid -- so its absence is a shape error rather than a default.
        deterministic = item.get("deterministic")
        if not isinstance(deterministic, Mapping):
            errors.append(f"item {index} has no deterministic reading")
        elif not isinstance(deterministic.get("case_passed"), bool):
            errors.append(f"item {index} carries no case_passed verdict")
    return tuple(errors)


# --- answers -----------------------------------------------------------------


def _fit_values(answers: Mapping[str, Any]) -> tuple[dict[str, float], list[str]]:
    fits: dict[str, float] = {}
    problems: list[str] = []
    for key, value in answers.items():
        skill = fit_question_skill(str(key))
        if not skill:
            continue
        # The key after `fits::` is whatever the answers file put there, and a
        # problem string built from it is printed as a report line, so it is
        # bounded here rather than at the sink.
        named = report_safe_text(skill)
        if not isinstance(value, Mapping):
            problems.append(f"malformed_fit_answer:{named}")
            continue
        raw = value.get("noul")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            problems.append(f"malformed_fit_answer:{named}")
            continue
        number = float(raw)
        if not 0.0 <= number <= 1.0:
            problems.append(f"fit_answer_out_of_range:{named}")
            continue
        fits[skill] = number
    return fits, problems


def parse_answer_row(
    row: object,
    *,
    ref: str,
    arm_default: str = "",
    digest_default: str = "",
    message_default: str = "",
    from_live_record: bool = False,
) -> AnswerRecord:
    """Read one `routing_question_answers/v1` row, keeping why it failed.

    `arm_default` and `digest_default` are what a wrapping record already
    stated: a recorded answer names its answerer and the question digest on
    the record, so the row it embeds does not have to repeat either.

    Every field a report line carries -- the arm, the case id, the digest -- is
    bounded here, because the row was written by a model or by whoever ran an
    external arm and the report is an artifact operators attach to a PR.
    """
    ref = report_safe_text(ref)
    blank = AnswerRecord(
        ref=ref,
        arm=UNKNOWN_ARM,
        case_id="",
        question_digest="",
        answers={},
        from_live_record=from_live_record,
    )
    if not isinstance(row, Mapping):
        return _failed(blank, "row is not an object")
    if row.get("schema_version") != ROUTING_QUESTION_ANSWERS_SCHEMA_VERSION:
        # A row whose schema is not this one has no field worth trusting,
        # including the arm it names, so it is reported under `unknown` rather
        # than charged to an arm that may not have written it. It is still
        # counted and still named by its reference.
        return _failed(blank, "schema_version is not " + ROUTING_QUESTION_ANSWERS_SCHEMA_VERSION)
    arm = report_safe_text(row.get("arm") or arm_default or "")
    case_id = report_safe_text(row.get("case_id") or "")
    digest = report_safe_text(row.get("question_digest") or digest_default or "")
    message = report_safe_text(row.get("message_sha256") or message_default or "")
    record = AnswerRecord(
        ref=ref,
        arm=arm or UNKNOWN_ARM,
        case_id=case_id,
        question_digest=digest,
        answers={},
        message_sha256=message,
        from_live_record=from_live_record,
    )
    if not arm:
        return _failed(record, "row names no arm")
    if arm == DETERMINISTIC_ARM:
        # The deterministic arm is computed from the corpus and assigned after
        # every supplied row is tallied, so a row claiming that name would be
        # counted into a tally the report then replaces: its answers would
        # vanish while its malformed rows stayed visible, and the two halves of
        # one report would disagree. The name is refused instead, by name.
        reserved = AnswerRecord(
            ref=ref,
            arm=UNKNOWN_ARM,
            case_id=case_id,
            question_digest=digest,
            answers={},
            message_sha256=message,
            from_live_record=from_live_record,
        )
        return _failed(reserved, f"arm name '{DETERMINISTIC_ARM}' is reserved for the router's own reading")
    if not case_id and not digest and not message:
        return _failed(record, "row names no case_id, question_digest, or message_sha256")
    answers = row.get("answers")
    if not isinstance(answers, Mapping):
        return _failed(record, "row carries no answers object")
    choice_answer = answers.get(ROUTE_CHOICE_KEY)
    if not isinstance(choice_answer, Mapping) or not str(choice_answer.get("choice") or "").strip():
        return _failed(record, f"row carries no {ROUTE_CHOICE_KEY} answer")
    _, problems = _fit_values(answers)
    if problems:
        return _failed(record, "; ".join(sorted(problems)))
    return AnswerRecord(
        ref=ref,
        arm=arm,
        case_id=case_id,
        question_digest=digest,
        answers=dict(answers),
        message_sha256=message,
        from_live_record=from_live_record,
    )


def _failed(record: AnswerRecord, reason: str) -> AnswerRecord:
    return AnswerRecord(
        ref=record.ref,
        arm=record.arm or UNKNOWN_ARM,
        case_id=record.case_id,
        question_digest=record.question_digest,
        answers={},
        error=reason,
        message_sha256=record.message_sha256,
        from_live_record=record.from_live_record,
    )


def read_answer_rows_from_jsonl(path: Path) -> list[AnswerRecord]:
    """Read a JSONL answer file, naming every line that could not be read."""
    records: list[AnswerRecord] = []
    text = _read_bounded_text(Path(path), limit=MAX_ANSWER_SOURCE_BYTES, label="answer file")
    name = report_safe_text(Path(path).name)
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        ref = f"{name}:{number}"
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            records.append(
                AnswerRecord(
                    ref=ref,
                    arm=UNKNOWN_ARM,
                    case_id="",
                    question_digest="",
                    answers={},
                    error=f"line is not JSON: {report_safe_text(exc.msg)}",
                )
            )
            continue
        records.append(parse_answer_row(row, ref=ref))
    return records


def read_answer_records_from_directory(path: Path) -> list[AnswerRecord]:
    """Read a directory of `route_question_answer/v1` records.

    Each record embeds one answer row under `answer` and carries the digest of
    the question it answered, which is how a recorded answer joins back to a
    corpus item that was exported separately.

    Each record is bounded on its own, and the directory carries one budget
    across all of them: a directory is as untrusted as the files in it, and a
    per-file cap alone bounds nothing about reading a million files.
    """
    return [record for record, _ in read_route_answer_documents(path)]


def read_route_answer_documents(path: Path) -> list[tuple[AnswerRecord, Mapping[str, Any]]]:
    """Each recorded answer, parsed, beside the record document it came from.

    The document is what carries the fields a row has no place for -- the
    mode, when it was recorded, the verdict the recorder reached -- and a
    reader that needs them reads them from here rather than opening the files
    a second time. An unreadable file pairs its failed record with `{}`.
    """
    records: list[tuple[AnswerRecord, Mapping[str, Any]]] = []
    budget = MAX_ANSWER_SOURCE_BYTES
    for entry in sorted(Path(path).glob("*.json")):
        ref = report_safe_text(entry.name)
        try:
            text = _read_bounded_text(
                entry,
                limit=max(0, min(MAX_ANSWER_RECORD_BYTES, budget)),
                label=f"record {ref}",
            )
        except RoutingQuestionCorpusError as exc:
            records.append(
                (
                    AnswerRecord(
                        ref=ref,
                        arm=UNKNOWN_ARM,
                        case_id="",
                        question_digest="",
                        answers={},
                        error=report_safe_text(exc),
                    ),
                    {},
                )
            )
            continue
        budget -= len(text.encode("utf-8"))
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            records.append(
                (
                    AnswerRecord(
                        ref=ref,
                        arm=UNKNOWN_ARM,
                        case_id="",
                        question_digest="",
                        answers={},
                        error=f"record is not readable JSON: {report_safe_text(exc.msg)}",
                    ),
                    {},
                )
            )
            continue
        records.append(
            (_record_from_document(document, ref=ref), document if isinstance(document, Mapping) else {})
        )
    return records


def _record_from_document(document: object, *, ref: str) -> AnswerRecord:
    blank = AnswerRecord(
        ref=ref, arm=UNKNOWN_ARM, case_id="", question_digest="", answers={}, from_live_record=True
    )
    if not isinstance(document, Mapping):
        return _failed(blank, "record is not an object")
    if document.get("schema_version") != ROUTE_QUESTION_ANSWER_SCHEMA_VERSION:
        return _failed(blank, "schema_version is not " + ROUTE_QUESTION_ANSWER_SCHEMA_VERSION)
    answered_by = report_safe_text(document.get("answered_by") or "")
    return parse_answer_row(
        document.get("answer"),
        ref=ref,
        arm_default=answered_by,
        digest_default=report_safe_text(document.get("question_digest") or ""),
        message_default=report_safe_text(document.get("message_sha256") or ""),
        from_live_record=True,
    )


def read_answer_source(path: Path) -> list[AnswerRecord]:
    """Read either a JSONL answer file or a directory of recorded answers."""
    target = Path(path)
    if target.is_dir():
        return read_answer_records_from_directory(target)
    if not target.exists():
        raise RoutingQuestionCorpusError(f"answer source not found: {target}")
    return read_answer_rows_from_jsonl(target)


# --- scoring -----------------------------------------------------------------


def resolve_answer_action(
    answers: Mapping[str, Any],
    *,
    fits_dispatch: float,
    fits_clarify: float,
) -> tuple[str, str]:
    """Resolve one answer set into an (action, choice) pair.

    The yes/no answers decide WHETHER: the strongest fit against the two flat
    thresholds picks dispatch, clarify, or nothing. The Choice decides WHICH,
    and it is taken as given. The two are separate questions with no invariant
    between them, so an answer set can say "this fits strongly" while choosing
    `none`; that is reported as it stands rather than repaired here, and it
    scores as naming no workflow.
    """
    choice_answer = answers.get(ROUTE_CHOICE_KEY)
    choice = ""
    if isinstance(choice_answer, Mapping):
        choice = str(choice_answer.get("choice") or "").strip()
    choice = choice or NO_WORKFLOW_OPTION
    fits, _ = _fit_values(answers)
    if fits:
        strongest = max(fits.values())
        if strongest >= fits_dispatch:
            return DISPATCH_ACTION, choice
        if strongest >= fits_clarify:
            return CLARIFY_ACTION, choice
        return NONE_ACTION, choice
    if choice != NO_WORKFLOW_OPTION:
        return DISPATCH_ACTION, choice
    return NONE_ACTION, choice


@dataclass
class _ArmTally:
    answered: int = 0
    malformed: int = 0
    overroute: int = 0
    missed: int = 0
    wrong_workflow: int = 0
    correct_workflow: int = 0
    band_mismatch: int = 0
    digest_mismatch: int = 0
    invalid_answer: int = 0
    agree: int = 0
    no_dispatch_denominator: int = 0
    intervention_denominator: int = 0
    workflow_denominator: int = 0


def _tally_item(
    tally: _ArmTally,
    item: Mapping[str, Any],
    *,
    action: str,
    choice: str,
    overrouted: bool | None,
) -> None:
    expected = item.get("expected")
    expected = expected if isinstance(expected, Mapping) else {}
    expected_action = str(expected.get("action") or NONE_ACTION)
    expected_choice = str(expected.get("choice") or NO_WORKFLOW_OPTION)
    tally.answered += 1
    if action == expected_action and choice == expected_choice:
        tally.agree += 1
    if expected_action == NONE_ACTION:
        tally.no_dispatch_denominator += 1
        # `overrouted` is supplied only for the deterministic arm, where the
        # verdict is the routing-precision corpus's own. Every other arm is
        # read from its answers: a dispatch on a case that expects no workflow
        # is the over-route; a clarify is not, because asking one question
        # hijacks nothing.
        over = overrouted if overrouted is not None else action == DISPATCH_ACTION
        if over:
            tally.overroute += 1
        return
    tally.intervention_denominator += 1
    missed = action == NONE_ACTION
    if missed:
        tally.missed += 1
    if expected_choice != NO_WORKFLOW_OPTION:
        tally.workflow_denominator += 1
        if choice == expected_choice:
            tally.correct_workflow += 1
            if not missed and action != expected_action:
                tally.band_mismatch += 1
        else:
            tally.wrong_workflow += 1


def _arm_payload(tally: _ArmTally, *, case_count: int, live_only: bool = False) -> dict[str, object]:
    # An `invalid_answer` is "no opinion": it leaves every accuracy
    # denominator, and is named there, the same way a malformed row does.
    excluded = ("unanswered_cases", "malformed_answer_rows", INVALID_ANSWER_VERDICT)
    if live_only:
        # A live arm answers only the questions a live route asks, which is the
        # undecidable cases. Counting the decided ones against it would report
        # an arm that answered everything it was asked as having answered a
        # fraction, so they leave the denominator and the exclusion is named
        # rather than silent.
        excluded = excluded + ("not_live_joinable",)
    # `unanswered` counts cases and `malformed` counts rows, so a case whose
    # only row was malformed is in both: it is one row this arm wrote that
    # could not be read, and one case it therefore has no answer for. Neither
    # rate double-counts it -- `excluded` names both classes -- and the two
    # integers are not meant to sum to the case count.
    return {
        "answered": tally.answered,
        "unanswered": max(case_count - tally.answered, 0),
        "case_count": case_count,
        "malformed": tally.malformed,
        "overroute": tally.overroute,
        "missed": tally.missed,
        "wrong_workflow": tally.wrong_workflow,
        "band_mismatch": tally.band_mismatch,
        "digest_mismatch": tally.digest_mismatch,
        "invalid_answer": tally.invalid_answer,
        "agree": tally.agree,
        "invalid_answer_rate": reported_rate(
            numerator=tally.invalid_answer,
            denominator=tally.answered + tally.invalid_answer,
            numerator_of=(INVALID_ANSWER_VERDICT,),
            denominator_of="joined answer rows, invalid ones included",
            excluded=("malformed_answer_rows", "unmatched_answer_rows", "ambiguous_answer_rows"),
        ).to_payload(),
        "overroute_rate": reported_rate(
            numerator=tally.overroute,
            denominator=tally.no_dispatch_denominator,
            numerator_of=("overroute",),
            denominator_of="answered cases that expect no workflow",
            excluded=excluded,
        ).to_payload(),
        "missed_rate": reported_rate(
            numerator=tally.missed,
            denominator=tally.intervention_denominator,
            numerator_of=("missed_intervention",),
            # Every intervention case whose expected action is not `none`,
            # including the two that expect a clarification without naming a
            # workflow. `workflow_accuracy` below is the one over cases that
            # name an expected workflow, and its denominator is smaller.
            denominator_of="answered intervention cases that expect the router to act",
            excluded=excluded,
        ).to_payload(),
        "workflow_accuracy": reported_rate(
            numerator=tally.correct_workflow,
            denominator=tally.workflow_denominator,
            numerator_of=("correct_workflow",),
            denominator_of="answered intervention cases that name an expected workflow",
            excluded=excluded,
        ).to_payload(),
    }


def score_routing_question_answers(
    corpus: Mapping[str, Any],
    answer_records: Iterable[AnswerRecord] = (),
    *,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Score every arm present in `answer_records`, plus the deterministic arm.

    The deterministic arm is computed from the corpus itself and is always
    present, so any other arm is read next to the router it would replace.

    The three counts answer three different questions and none of them is a
    summary of the others. `overroute` is over cases that expect no workflow.
    `missed` is over cases that expect one and were answered with none.
    `workflow_accuracy` reads the Choice answer alone: it says whether the arm
    named the right workflow, not whether it was confident enough to act, so a
    case can be both a miss and a correct workflow.
    """
    errors = corpus_shape_errors(corpus)
    if errors:
        raise RoutingQuestionCorpusError("; ".join(errors))
    contract = corpus.get("question_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    declared = contract.get("thresholds")
    declared = declared if isinstance(declared, Mapping) else {}
    fits_dispatch = _threshold(thresholds, declared, "fits_dispatch", FITS_DISPATCH_THRESHOLD)
    fits_clarify = _threshold(thresholds, declared, "fits_clarify", FITS_CLARIFY_THRESHOLD)
    if fits_clarify > fits_dispatch:
        raise RoutingQuestionCorpusError("fits_clarify cannot exceed fits_dispatch")

    items = [item for item in corpus["items"] if isinstance(item, Mapping)]
    by_case: dict[str, Mapping[str, Any]] = {}
    for item in items:
        by_case.setdefault(str(item.get("case_id") or ""), item)
    by_message: dict[str, list[Mapping[str, Any]]] = {}
    by_digest: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        message = str(item.get("message_sha256") or "")
        if message:
            by_message.setdefault(message, []).append(item)
        question = item.get("question")
        digest = str(question.get("question_digest") or "") if isinstance(question, Mapping) else ""
        if digest:
            by_digest.setdefault(digest, []).append(item)

    tallies: dict[str, _ArmTally] = {}
    live_arms: set[str] = set()
    seen: dict[tuple[str, str], str] = {}
    malformed: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    ambiguous: list[dict[str, object]] = []
    matched: list[dict[str, object]] = []
    invalid_answers: list[dict[str, object]] = []
    for record in answer_records:
        tally = tallies.setdefault(record.arm or UNKNOWN_ARM, _ArmTally())
        if record.from_live_record:
            live_arms.add(record.arm or UNKNOWN_ARM)
        if record.error:
            tally.malformed += 1
            malformed.append({"ref": record.ref, "arm": record.arm, "reason": record.error})
            continue
        item = by_case.get(record.case_id) if record.case_id else None
        joined_by = "case_id" if item is not None else ""
        collision: list[Mapping[str, Any]] = []
        collision_by = ""
        # The message first, then the digest. A digest covers the candidate
        # shortlist as well as the request, and the shortlist is cut per
        # surface -- a route hint shows two candidates where a full route shows
        # three -- so the same request asked on two surfaces produces two
        # digests. The message does not move, so it is the key that joins an
        # answer to the case it was about; the digest then says whether the
        # shortlist was the same one.
        if item is None and record.message_sha256:
            matches = by_message.get(record.message_sha256, [])
            if len(matches) == 1:
                item, joined_by = matches[0], "message_sha256"
            elif matches:
                collision, collision_by = matches, "message_sha256"
        if item is None and record.question_digest:
            # Tried even when the message was ambiguous: the digest is the
            # narrower key, so two items carrying one request can still be told
            # apart when they were asked about different shortlists.
            matches = by_digest.get(record.question_digest, [])
            if len(matches) == 1:
                item, joined_by = matches[0], "question_digest"
            elif matches:
                collision, collision_by = matches, "question_digest"
        if item is None and collision:
            # More than one corpus item is the same question, on every key the
            # record carries. Scoring it against the first would charge this
            # arm with what some other case expected, and the report would say
            # nothing about it.
            ambiguous.append(
                {
                    "ref": record.ref,
                    "arm": record.arm,
                    "by": collision_by,
                    "question_digest": record.question_digest,
                    "message_sha256": record.message_sha256,
                    "case_count": len(collision),
                    "case_ids": [str(match.get("case_id") or "") for match in collision[:MAX_NAMED_CASES]],
                }
            )
            continue
        if item is None:
            unmatched.append(
                {
                    "ref": record.ref,
                    "arm": record.arm,
                    "case_id": record.case_id,
                    "question_digest": record.question_digest,
                    "message_sha256": record.message_sha256,
                }
            )
            continue
        key = (record.arm, str(item.get("case_id")))
        if key in seen:
            # The first answer for a case stands; a second is reported rather
            # than dropped, because silently keeping one of two disagreeing
            # answers is how an arm scores better than it answered.
            tally.malformed += 1
            malformed.append(
                {
                    "ref": record.ref,
                    "arm": record.arm,
                    "reason": f"duplicate answer for {item.get('case_id')}, first was {seen[key]}",
                }
            )
            continue
        seen[key] = record.ref
        question = item.get("question")
        item_digest = str(question.get("question_digest") or "") if isinstance(question, Mapping) else ""
        # After the duplicate check, so the first answer for a case stands
        # even when it is the invalid one. Judged against the joined item's own
        # options: that is the question this answer is scored as answering.
        contradictions = answer_contradictions(record.answers, question)
        if contradictions:
            tally.invalid_answer += 1
            invalid_answers.append(
                {
                    "ref": record.ref,
                    "arm": record.arm,
                    "case_id": str(item.get("case_id") or ""),
                    "reasons": list(contradictions),
                }
            )
            continue
        # A digest mismatch is a different shortlist cut of the same request,
        # not a different request. It is scored, because the answer is about
        # this case; it is reported, because an arm answering a two-candidate
        # shortlist is not being asked quite what a three-candidate corpus item
        # asks, and a reader comparing arms has to be able to see that.
        digest_match = bool(record.question_digest) and record.question_digest == item_digest
        if record.question_digest and not digest_match:
            tally.digest_mismatch += 1
        matched.append(
            {
                "ref": record.ref,
                "arm": record.arm,
                "case_id": str(item.get("case_id") or ""),
                "joined_by": joined_by,
                "digest_match": digest_match,
            }
        )
        action, choice = resolve_answer_action(
            record.answers,
            fits_dispatch=fits_dispatch,
            fits_clarify=fits_clarify,
        )
        _tally_item(tally, item, action=action, choice=choice, overrouted=None)

    deterministic = _ArmTally()
    producer_failed: list[str] = []
    for item in items:
        reading = item.get("deterministic")
        reading = reading if isinstance(reading, Mapping) else {}
        if not bool(reading.get("case_passed")):
            producer_failed.append(str(item.get("case_id") or ""))
        _tally_item(
            deterministic,
            item,
            action=str(reading.get("action") or NONE_ACTION),
            choice=str(reading.get("choice") or NO_WORKFLOW_OPTION),
            overrouted=bool(reading.get("overrouted")),
        )

    case_count = len(items)
    live_case_count = sum(1 for item in items if bool(item.get("live_joinable")))
    arms = {
        name: _arm_payload(
            tally,
            case_count=live_case_count if name in live_arms else case_count,
            live_only=name in live_arms,
        )
        for name, tally in sorted(tallies.items())
    }
    deterministic_payload = _arm_payload(deterministic, case_count=case_count)
    # The producer's verdict, carried beside the counts this module derives
    # from the same items. The two can disagree: a router that dispatches on
    # the one intervention case whose correct answer is to open nothing still
    # names a workflow, so it reads clean here while the gate records a
    # failure. `routing_question_score_errors` fails on this field, which is
    # what makes the deterministic arm a reading of the gate rather than a
    # second opinion about it.
    deterministic_payload["producer_failed"] = len(producer_failed)
    deterministic_payload["producer_failed_cases"] = producer_failed[:MAX_NAMED_CASES]
    arms[DETERMINISTIC_ARM] = deterministic_payload
    return {
        "schema_version": ROUTING_QUESTION_SCORE_SCHEMA_VERSION,
        "source": str(corpus.get("source") or ""),
        "case_count": case_count,
        "thresholds": {"fits_dispatch": fits_dispatch, "fits_clarify": fits_clarify},
        "live_joinable_case_count": live_case_count,
        "arms": arms,
        "matched": matched,
        "malformed": malformed,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "invalid_answers": invalid_answers,
        "claim_boundary": _SCORE_CLAIM_BOUNDARY,
    }


def question_choice_options(question: object) -> list[str] | None:
    """The Choice options a route question offers, or None when it names none."""
    questions = question.get("questions") if isinstance(question, Mapping) else None
    choice = questions.get(ROUTE_CHOICE_KEY) if isinstance(questions, Mapping) else None
    options = choice.get("options") if isinstance(choice, Mapping) else None
    return [str(option) for option in options] if isinstance(options, Mapping) else None


def answer_contradictions(answers: Mapping[str, Any], question: object) -> tuple[str, ...]:
    """`route_answer_consistency`'s verdict on one answer set, against its question.

    The one rule the live record and this scorer share: a record written by
    `omh_route_answer` and a row in an offline answer file are judged by the
    same function, so an answer cannot be valid in one and invalid in the
    other except where the recorder could not see the question's options.
    """
    choice_answer = answers.get(ROUTE_CHOICE_KEY)
    if not isinstance(choice_answer, Mapping):
        return ()
    probabilities = choice_answer.get("probabilities")
    return invalid_answer_reasons(
        choice_answer.get("choice"),
        probabilities if isinstance(probabilities, Mapping) else None,
        options=question_choice_options(question),
    )


def _threshold(
    override: Mapping[str, float] | None,
    declared: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    for source in (override or {}, declared):
        if key in source:
            value = source[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RoutingQuestionCorpusError(f"{key} must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise RoutingQuestionCorpusError(f"{key} must fall between 0 and 1")
            return float(value)
    return default


def format_routing_question_corpus(corpus: Mapping[str, Any]) -> str:
    """A compact human summary of an exported corpus."""
    generated = corpus.get("generated_from")
    generated = generated if isinstance(generated, Mapping) else {}
    items = corpus.get("items")
    items = items if isinstance(items, list) else []
    negative = sum(1 for item in items if isinstance(item, Mapping) and item.get("corpus") == NEGATIVE_CONTROL_CORPUS)
    intervention = sum(1 for item in items if isinstance(item, Mapping) and item.get("corpus") == INTERVENTION_CORPUS)
    sources = corpus.get("question_sources")
    sources = sources if isinstance(sources, Mapping) else {}
    lines = [
        f"Routing question corpus ({corpus.get('schema_version')}) from {generated.get('schema')}",
        f"Source: {corpus.get('source')}",
        f"Items: {len(items)} ({negative} negative-control, {intervention} intervention)",
        (
            f"Questions: {sources.get(HANDOFF_QUESTION_SOURCE, 0)} from the candidate handoff "
            f"(a live route asks these), {sources.get(RECOMMENDATIONS_QUESTION_SOURCE, 0)} from the "
            "router's recommendations (offline arms only)"
        ),
        f"Boundary: {corpus.get('claim_boundary', '')}",
    ]
    return "\n".join(lines)


def format_routing_question_score(score: Mapping[str, Any]) -> str:
    """A compact human summary of a score report."""
    arms = score.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    lines = [
        f"Routing question score ({score.get('schema_version')}) over {score.get('case_count')} cases",
    ]
    for name in sorted(arms):
        arm = arms[name]
        if not isinstance(arm, Mapping):
            continue
        lines.append(
            f"- {name}: answered {arm.get('answered')}/{arm.get('case_count')}, "
            f"unanswered {arm.get('unanswered')}, "
            f"malformed {arm.get('malformed')}, overroute {arm.get('overroute')}, "
            f"missed {arm.get('missed')}, wrong workflow {arm.get('wrong_workflow')}, "
            f"band mismatch {arm.get('band_mismatch')}, "
            f"digest mismatch {arm.get('digest_mismatch')}, "
            f"invalid answer {arm.get('invalid_answer', 0)}, agree {arm.get('agree')}"
        )
    malformed = score.get("malformed")
    if isinstance(malformed, list) and malformed:
        lines.append(f"Malformed rows: {len(malformed)}")
        for entry in malformed[:10]:
            if isinstance(entry, Mapping):
                lines.append(f"- {entry.get('ref')}: {entry.get('reason')}")
    invalid = score.get("invalid_answers")
    if isinstance(invalid, list) and invalid:
        lines.append(f"Invalid answers (no opinion): {len(invalid)}")
        for entry in invalid[:10]:
            if isinstance(entry, Mapping):
                reasons = entry.get("reasons")
                named = ", ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
                lines.append(f"- {entry.get('ref')}: {named}")
    unmatched = score.get("unmatched")
    if isinstance(unmatched, list) and unmatched:
        lines.append(f"Unmatched answer records: {len(unmatched)}")
    ambiguous = score.get("ambiguous")
    if isinstance(ambiguous, list) and ambiguous:
        lines.append(f"Ambiguous answer records (digest reaches more than one case): {len(ambiguous)}")
        for entry in ambiguous[:10]:
            if isinstance(entry, Mapping):
                lines.append(f"- {entry.get('ref')}: {entry.get('case_count')} cases share this question digest")
    mismatched = sum(
        1 for entry in (score.get("matched") or []) if isinstance(entry, Mapping) and not entry.get("digest_match")
    )
    if mismatched:
        lines.append(
            f"Answers joined by message with a different shortlist cut: {mismatched} "
            "(scored; the question asked about a different candidate list)"
        )
    arm = arms.get(DETERMINISTIC_ARM)
    if isinstance(arm, Mapping) and int(arm.get("producer_failed", 0) or 0):
        lines.append(
            f"Producer verdict: {arm.get('producer_failed')} case(s) fail the routing-precision gate"
        )
    lines.append(f"Boundary: {score.get('claim_boundary', '')}")
    return "\n".join(lines)


def routing_question_score_errors(score: Mapping[str, Any]) -> list[str]:
    """Return why a score report is not a clean deterministic-arm reading.

    Two different readings have to agree for the report to be clean: the counts
    this module derives from the items, and the producer's own verdict on each
    case. A corpus can carry a failed verdict that the derived counts read as a
    clean answer, so both are checked here.
    """
    errors: list[str] = []
    if score.get("schema_version") != ROUTING_QUESTION_SCORE_SCHEMA_VERSION:
        errors.append("unexpected_schema")
    arms = score.get("arms")
    arms = arms if isinstance(arms, Mapping) else {}
    arm = arms.get(DETERMINISTIC_ARM)
    if not isinstance(arm, Mapping):
        errors.append("deterministic_arm_missing")
        return errors
    if int(arm.get("overroute", 0) or 0):
        errors.append(f"deterministic_overroute: {arm.get('overroute')}")
    if int(arm.get("missed", 0) or 0):
        errors.append(f"deterministic_missed: {arm.get('missed')}")
    if int(arm.get("malformed", 0) or 0):
        errors.append(f"deterministic_malformed: {arm.get('malformed')}")
    failed = int(arm.get("producer_failed", 0) or 0)
    if failed:
        named = arm.get("producer_failed_cases")
        names = ", ".join(str(case) for case in named) if isinstance(named, list) and named else ""
        errors.append(f"deterministic_producer_failed: {failed}" + (f" ({names})" if names else ""))
    return errors


def answer_records_from_rows(rows: Sequence[Mapping[str, Any]], *, ref_prefix: str = "row") -> list[AnswerRecord]:
    """Parse in-memory rows, keeping each one's position as its reference."""
    return [parse_answer_row(row, ref=f"{ref_prefix}:{index}") for index, row in enumerate(rows, start=1)]


__all__ = [
    "CLARIFY_ACTION",
    "DETERMINISTIC_ARM",
    "DISPATCH_ACTION",
    "HANDOFF_QUESTION_SOURCE",
    "INTERVENTION_CORPUS",
    "MAX_ANSWER_RECORD_BYTES",
    "MAX_ANSWER_SOURCE_BYTES",
    "MAX_CORPUS_BYTES",
    "MAX_REPORT_FIELD_CHARS",
    "NEGATIVE_CONTROL_CORPUS",
    "NONE_ACTION",
    "RECOMMENDATIONS_QUESTION_SOURCE",
    "ROUTE_QUESTION_ANSWER_SCHEMA_VERSION",
    "ROUTING_QUESTION_ANSWERS_SCHEMA_VERSION",
    "ROUTING_QUESTION_CORPUS_SCHEMA_VERSION",
    "ROUTING_QUESTION_SCORE_SCHEMA_VERSION",
    "AnswerRecord",
    "RoutingQuestionCorpusError",
    "answer_contradictions",
    "answer_records_from_rows",
    "deterministic_route_reading",
    "question_choice_options",
    "read_route_answer_documents",
    "build_routing_question_corpus",
    "corpus_shape_errors",
    "format_routing_question_corpus",
    "format_routing_question_score",
    "parse_answer_row",
    "read_answer_records_from_directory",
    "read_answer_rows_from_jsonl",
    "read_answer_source",
    "read_routing_question_corpus",
    "report_safe_text",
    "resolve_answer_action",
    "routing_question_contract",
    "routing_question_score_errors",
    "score_routing_question_answers",
]
