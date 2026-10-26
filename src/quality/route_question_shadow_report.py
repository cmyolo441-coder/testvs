"""What the route question would have decided, read against the owner's own traffic.

The route question runs in `shadow` by default: the router attaches it to an
undecidable route, something answers it, `omh_route_answer` records the answer,
and nothing about the route changes. The corpus scorer
(`routing_question_corpus`) reads those answers against a fixed corpus. This
module reads them against the routes the operator's sessions actually took:
each recorded `route_question_answer/v1` is joined, by `message_sha256`, to the
`route_decision/v1` record `omh chat route --record` wrote for the same
request (`routing_log_calibration` reads those), and the window is reported as
four numbers:

- the **decline rate** -- of the recorded routes that built a question, how
  many the decline predicate says had nothing to decide. In `shadow` those
  questions were still asked; the rate is what a mode that acted on the
  predicate would have withheld.
- the **invalid_answer rate** -- recorded answers whose Choice distribution
  contradicts itself, which count as no opinion.
- the **agreement rate** -- of the valid answers joined to a recorded route,
  how many resolve to the same action and workflow as the deterministic route.
- a **per-turn cost estimate** over a declared turn shape, priced at a named
  model's list price. OMH has measured no turn shape of its own; the one used
  here is declared, with its source, and every line that prints it says so.

Every rate is built by `reported_rate`, so each names what it counted, what it
divided by, and what it excluded, and an empty window reports `percent: null`.

Nothing here calls a model or the network. Records carry `message_sha256`,
never message text.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from ..plugin_bundle.omh.hermes_delegation import APPROX_PRICE_PER_MTOK
from ..plugin_bundle.omh.route_answer_consistency import INVALID_ANSWER_VERDICT
from ..routing.route_question import (
    FITS_CLARIFY_THRESHOLD,
    FITS_DISPATCH_THRESHOLD,
    ROUTE_QUESTION_DECLINE_REASONS,
)
from .reported_rate import ReportedRate, format_reported_rate, reported_rate
from .routing_log_calibration import collect_routing_records, router_source_mtime
from ..system.local_store import read_jsonl_objects
from .routing_question_corpus import (
    answer_contradictions,
    deterministic_route_reading,
    read_route_answer_documents,
    report_safe_text,
    resolve_answer_action,
)

ROUTE_QUESTION_SHADOW_REPORT_SCHEMA_VERSION = "route_question_shadow_report/v1"

# The turn shape the per-turn cost is priced over. Declared, not measured: OMH
# has no turn-shape measurement of its own, so the report carries the shape it
# assumed and where the numbers came from, and never presents them as OMH's.
DECLARED_TURN_SHAPE: dict[str, object] = {
    "input_tokens_per_turn": 192_000,
    "output_tokens_per_turn": 5_600,
    "api_calls_per_turn": 8,
    "source": (
        "kerpopule/hermes-jev-skills (MIT), docs/measuring-a-router.md, read 2026-09-22: "
        "that project's median agent turn, self-reported from its own fleet"
    ),
    "observed_by_omh": False,
}

COST_BASIS_PRICED = "declared_turn_shape_times_list_price"
COST_BASIS_NO_MODEL = "no_model_named"
COST_BASIS_UNPRICED = "model_not_in_price_table"

# How an answer that is not an opinion or not in the window leaves the join.
UNRECORDED_MODE = "unrecorded"

LANE_ROUTE_RECORD = "route_record"
LANE_WRAPPER_SESSION = "wrapper_session"
# The wrapper-session event a live turn writes when it builds a question.
# Restated from `omh.wrapper.sessions`, which imports this package's
# neighbours; a test pins the two against each other.
ROUTE_QUESTION_OBSERVED_EVENT = "route_question_observed"

JOIN_RULES = (
    "An answer joins the newest route in the window with the same message_sha256, from either lane.",
    "The join ignores the route's source surface and whether the answer was recorded before or after the route.",
    "The decline rate counts every route that built a question, so one request routed twice counts twice.",
    "Answers whose coverage was not checked (no message to re-derive the question) are excluded from agreement.",
)

CLAIM_BOUNDARY = (
    "This report reads OMH-local records only: route_question_answer/v1 answers, the "
    "route_decision/v1 records `omh chat route --record` wrote, and the route_question_observed "
    "events a recorded wrapper session (omh_interact, omh chat session) writes on a turn that built "
    "a question. Live turns that built no question, and interactions run without a session, are "
    "not seen. It joins answers to routes by message_sha256 "
    "and never sees message text. In shadow the route question changes no route, so agreement "
    "is what the answers would have decided, not what happened, and the decline rate is what a "
    "mode acting on the predicate would have withheld. The per-turn cost is an estimate over a "
    "declared turn shape OMH did not measure, priced at a list price, and is not billing evidence. "
    "None of this is execution, review, CI, or merge evidence, or a claim about any model's quality."
)


def build_route_question_shadow_report(
    answers_dir: Path,
    runs_dir: Path,
    *,
    sessions_dir: Path | None = None,
    since: str | None = None,
    model: str = "",
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Join recorded answers to recorded routes over one window and report it.

    Routes come from two lanes, and the report counts each:

    - `route_record`: every `omh chat route --record` turn, one
      `runtime/runs/<id>/routing.json` each, decided routes included.
    - `wrapper_session`: every live interaction turn through a recorded wrapper
      session (`omh_interact`, `omh chat session start`) that BUILT a route
      question, one `route_question_observed` event in that session's
      `events.jsonl`. Decided live turns write nothing and are not seen.

    `omh chat interact` without a session and `omh_interact` with
    `record_session: false` record no route, so their answers read as
    unjoined. `since` follows `routing_log_calibration`: omitted, it is the
    router source's modification time when that file exists; an empty string
    means every record. The same bound applies to both sides of the join.
    """
    if since is None:
        resolved_since = router_source_mtime(repo_root)
        since_basis = "router_source_mtime" if resolved_since else "none"
    elif since:
        resolved_since, since_basis = since, "explicit"
    else:
        resolved_since, since_basis = "", "none"

    recorded, coverage = collect_routing_records(Path(runs_dir), since=resolved_since)
    live = collect_live_route_observations(Path(sessions_dir), since=resolved_since) if sessions_dir else []
    routes = [*recorded, *live]
    route_side = _route_side(routes)
    route_side["summary"]["by_lane"] = {LANE_ROUTE_RECORD: len(recorded), LANE_WRAPPER_SESSION: len(live)}

    # The newest recorded route per request, whichever lane wrote it. One
    # request routed twice in the window is the same request; the later record
    # is the router the answer was most recently read against. The join reads
    # neither the route's source nor whether the answer came before or after
    # it -- `JOIN_RULES` says so in the payload.
    latest: dict[str, Mapping[str, Any]] = {}
    for record in sorted(routes, key=lambda item: str(item.get("updated_at", ""))):
        digest = str(record.get("message_sha256") or "")
        if digest:
            latest[digest] = record

    answer_side = _answer_side(Path(answers_dir), resolved_since, latest)

    return {
        "schema_version": ROUTE_QUESTION_SHADOW_REPORT_SCHEMA_VERSION,
        "since": {"value": resolved_since, "basis": since_basis},
        "routes": route_side["summary"],
        "route_coverage": coverage.to_payload(),
        "answers": answer_side["summary"],
        "rates": {
            "decline_rate": route_side["decline_rate"],
            "invalid_answer_rate": answer_side["invalid_answer_rate"],
            "agreement_rate": answer_side["agreement_rate"],
        },
        "cost": _cost(model, turns=len(routes)),
        "join_rules": list(JOIN_RULES),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def collect_live_route_observations(sessions_dir: Path, *, since: str = "") -> list[Mapping[str, Any]]:
    """Every `route_question_observed` event in the window, shaped as a route record.

    Each event carries what a routing record carries for the join -- the
    request hash, the deterministic reading, the question summary -- under
    `data`, and its own timestamp. A line that is not such an event, or an
    unreadable log, contributes nothing.
    """
    routes: list[Mapping[str, Any]] = []
    if not sessions_dir.is_dir():
        return routes
    for events_path in sorted(sessions_dir.glob("*/events.jsonl")):
        events, _errors = read_jsonl_objects(events_path)
        for event in events:
            if event.get("event") != ROUTE_QUESTION_OBSERVED_EVENT:
                continue
            data = event.get("data")
            timestamp = str(event.get("timestamp") or "")
            if not isinstance(data, Mapping) or not timestamp:
                continue
            if since and timestamp < since:
                continue
            routes.append({**data, "updated_at": timestamp, "lane": LANE_WRAPPER_SESSION})
    return routes


def _route_side(routes: list[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = [record.get("route_question") for record in routes]
    recorded = [summary for summary in summaries if isinstance(summary, Mapping)]
    built = [summary for summary in recorded if summary.get("built") is True]
    declinable = [summary for summary in built if summary.get("decline_reason") in ROUTE_QUESTION_DECLINE_REASONS]
    return {
        "summary": {
            "recorded_routes": len(routes),
            "with_route_question_summary": len(recorded),
            "built": len(built),
            "asked": sum(1 for summary in built if summary.get("asked") is True),
            "mode": dict(sorted(Counter(str(summary.get("mode") or UNRECORDED_MODE) for summary in recorded).items())),
            "mode_source": dict(
                sorted(Counter(str(summary.get("mode_source") or UNRECORDED_MODE) for summary in recorded).items())
            ),
            "decline_reason": dict(sorted(Counter(str(summary.get("decline_reason")) for summary in declinable).items())),
        },
        "decline_rate": reported_rate(
            numerator=len(declinable),
            denominator=len(built),
            numerator_of=("declinable",),
            denominator_of="routes in the window that built a route question, from both lanes",
            # A routing record written before the summary existed says nothing
            # about the question either way, so it is outside both counts.
            excluded=("routes_recorded_without_route_question_summary", "decided_routes"),
        ).to_payload(),
    }


def _answer_side(
    answers_dir: Path,
    since: str,
    latest: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = read_route_answer_documents(answers_dir) if answers_dir.is_dir() else []
    malformed = 0
    before_since = 0
    readable = 0
    invalid = 0
    unjoined = 0
    coverage_unchecked = 0
    joined_valid = 0
    agree = 0
    modes: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    for record, document in pairs:
        if record.error:
            malformed += 1
            continue
        if since and str(document.get("recorded_at") or "") < since:
            before_since += 1
            continue
        readable += 1
        modes[report_safe_text(document.get("mode") or UNRECORDED_MODE)] += 1
        sources[report_safe_text(document.get("mode_source") or UNRECORDED_MODE)] += 1
        # The recorder's verdict, which saw the question's options when it
        # could verify the digest, and the shared rule over the answer's own
        # fields, which holds whether or not it could.
        if document.get("answer_verdict") == INVALID_ANSWER_VERDICT or answer_contradictions(record.answers, None):
            invalid += 1
            continue
        # `accepted` without coverage is a verdict on mass and argmax alone,
        # so it is kept out of agreement and named there. A record written
        # before the field existed says nothing either way and is treated the
        # same.
        if document.get("coverage_checked") is not True:
            coverage_unchecked += 1
            continue
        route = latest.get(record.message_sha256) if record.message_sha256 else None
        if route is None:
            unjoined += 1
            continue
        joined_valid += 1
        thresholds = document.get("action_thresholds")
        thresholds = thresholds if isinstance(thresholds, Mapping) else {}
        answered = resolve_answer_action(
            record.answers,
            fits_dispatch=_threshold(thresholds, "fits_dispatch", FITS_DISPATCH_THRESHOLD),
            fits_clarify=_threshold(thresholds, "fits_clarify", FITS_CLARIFY_THRESHOLD),
        )
        if answered == deterministic_route_reading(route):
            agree += 1
    return {
        "summary": {
            "records_read": len(pairs),
            "malformed": malformed,
            "recorded_before_since": before_since,
            "in_window": readable,
            "invalid_answer": invalid,
            "coverage_unchecked": coverage_unchecked,
            "unjoined": unjoined,
            "joined": joined_valid,
            "agree": agree,
            "mode": dict(sorted(modes.items())),
            "mode_source": dict(sorted(sources.items())),
        },
        "invalid_answer_rate": reported_rate(
            numerator=invalid,
            denominator=readable,
            numerator_of=(INVALID_ANSWER_VERDICT,),
            denominator_of="readable answer records in the window",
            excluded=("malformed_answer_records", "recorded_before_since"),
        ).to_payload(),
        "agreement_rate": reported_rate(
            numerator=agree,
            denominator=joined_valid,
            numerator_of=("agree",),
            denominator_of="valid answers joined by message_sha256 to a recorded route in the window",
            excluded=(
                "malformed_answer_records",
                "recorded_before_since",
                INVALID_ANSWER_VERDICT,
                "coverage_unchecked",
                "unjoined_answer_records",
            ),
        ).to_payload(),
    }


def _threshold(values: Mapping[str, Any], key: str, default: float) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        return default
    return float(value)


def _cost(model: str, *, turns: int) -> dict[str, object]:
    name = str(model or "").strip()
    prices = APPROX_PRICE_PER_MTOK.get(name) if name else None
    input_tokens = int(DECLARED_TURN_SHAPE["input_tokens_per_turn"])  # type: ignore[call-overload]
    output_tokens = int(DECLARED_TURN_SHAPE["output_tokens_per_turn"])  # type: ignore[call-overload]
    payload: dict[str, object] = {
        "estimate": True,
        "turn_shape": dict(DECLARED_TURN_SHAPE),
        "model": name,
        "turns_in_window": turns,
        "turns_counted_as": (
            "routes in the window, one per turn: every `omh chat route --record` turn plus every "
            "recorded wrapper-session turn that built a route question"
        ),
    }
    if not name:
        return {**payload, "basis": COST_BASIS_NO_MODEL, "price_per_mtok": None, "per_turn_usd": None, "window_usd": None}
    if not prices:
        return {**payload, "basis": COST_BASIS_UNPRICED, "price_per_mtok": None, "per_turn_usd": None, "window_usd": None}
    input_price, output_price = prices
    per_turn = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    return {
        **payload,
        "basis": COST_BASIS_PRICED,
        "price_per_mtok": {"input": input_price, "output": output_price},
        "per_turn_usd": round(per_turn, 6),
        "window_usd": round(per_turn * turns, 6),
    }


def format_route_question_shadow_report(report: Mapping[str, Any]) -> str:
    """A compact human reading; every rate line carries its denominator."""

    def rate_line(payload: object) -> str:
        if not isinstance(payload, Mapping):
            return "unavailable"
        return format_reported_rate(
            ReportedRate(
                numerator=int(payload.get("numerator", 0) or 0),
                denominator=int(payload.get("denominator", 0) or 0),
                numerator_of=tuple(payload.get("numerator_of") or ()),
                denominator_of=str(payload.get("denominator_of") or ""),
                excluded=tuple(payload.get("excluded") or ()),
                percent=payload.get("percent"),
            )
        )

    since = report.get("since") if isinstance(report.get("since"), Mapping) else {}
    routes = report.get("routes") if isinstance(report.get("routes"), Mapping) else {}
    answers = report.get("answers") if isinstance(report.get("answers"), Mapping) else {}
    rates = report.get("rates") if isinstance(report.get("rates"), Mapping) else {}
    cost = report.get("cost") if isinstance(report.get("cost"), Mapping) else {}
    shape = cost.get("turn_shape") if isinstance(cost.get("turn_shape"), Mapping) else {}
    lines = [
        f"Route question shadow report ({report.get('schema_version')})",
        f"Since: {since.get('value') or 'all records'} ({since.get('basis')})",
        (
            f"Routes: {routes.get('recorded_routes', 0)} recorded {routes.get('by_lane') or {}}, "
            f"{routes.get('built', 0)} built a question, {routes.get('asked', 0)} asked; mode {routes.get('mode') or {}}"
        ),
        (
            f"Answers: {answers.get('in_window', 0)} in window, {answers.get('malformed', 0)} malformed, "
            f"{answers.get('joined', 0)} joined; mode {answers.get('mode') or {}}"
        ),
        f"Decline rate: {rate_line(rates.get('decline_rate'))}",
        f"Invalid answer rate: {rate_line(rates.get('invalid_answer_rate'))}",
        f"Agreement rate: {rate_line(rates.get('agreement_rate'))}",
    ]
    per_turn = cost.get("per_turn_usd")
    shape_text = (
        f"{shape.get('input_tokens_per_turn')} input + {shape.get('output_tokens_per_turn')} output tokens "
        f"per turn, declared from {shape.get('source')}"
    )
    if per_turn is None:
        lines.append(f"Per-turn cost estimate: not priced ({cost.get('basis')}); turn shape assumed: {shape_text}")
    else:
        lines.append(
            f"Per-turn cost estimate: ~${per_turn} on {cost.get('model')} "
            f"(~${cost.get('window_usd')} over {cost.get('turns_in_window')} turns); "
            f"turn shape assumed: {shape_text}"
        )
    for rule in report.get("join_rules") or ():
        lines.append(f"Join rule: {rule}")
    lines.append(f"Boundary: {report.get('claim_boundary', '')}")
    return "\n".join(lines)


__all__ = [
    "CLAIM_BOUNDARY",
    "COST_BASIS_NO_MODEL",
    "COST_BASIS_PRICED",
    "COST_BASIS_UNPRICED",
    "DECLARED_TURN_SHAPE",
    "ROUTE_QUESTION_SHADOW_REPORT_SCHEMA_VERSION",
    "JOIN_RULES",
    "LANE_ROUTE_RECORD",
    "LANE_WRAPPER_SESSION",
    "ROUTE_QUESTION_OBSERVED_EVENT",
    "build_route_question_shadow_report",
    "collect_live_route_observations",
    "format_route_question_shadow_report",
]
