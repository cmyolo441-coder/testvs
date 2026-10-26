"""Calibrate the routing corpus against locally recorded routing decisions.

`ROUTING_PRECISION_CASES` and `ROUTING_INTERVENTION_CASES` are hand-written.
They are the right shape for guarding regressions, and no shape at all for
answering "does this corpus look like what the router actually sees?" Tuning a
threshold against a corpus nobody has compared to real traffic is tuning
against an assumption.

`omh chat route --record` already writes the raw material: one
`runtime/runs/<run-id>/routing.json` per recorded decision, carrying the
`route_decision/v1` contract — router stage, action, selected skill, confidence,
score, threshold, and the top-two score margin. This module aggregates those
files. No network, no model, no prompt text.

**What this cannot tell you.** A routing record stores `message_sha256` and
`message_length`, never the message. That is a deliberate privacy boundary, and
it means this surface calibrates *decision distributions* — which stages fire,
how often the router is confident, how thin its margins run — and never
phrasings. A new corpus phrasing still has to come from an operator's own
message; the numbers here say whether the corpus's *shape* is wrong, not which
sentence to add.

Two methodology rules, both copied deliberately:

- **Print parse coverage.** Every report names how many run directories were
  walked, how many carried a routing record, how many parsed, and why each
  skipped file was skipped. A format skew that silently drops a third of the
  corpus would otherwise bias every number below it, and nothing in the output
  would say so.
- **Normalize before taking a median.** A single heavy day of recording would
  otherwise dominate a flat median. The report carries both: the flat median
  over all decisions, and the median of per-day medians. OMH's routing runs
  carry no session id, so the recording day is the normalization unit, and the
  payload names it rather than implying a session.

`--since` defaults to the modification time of the router source being studied
(`src/routing/chat.py`), so the default report covers only decisions recorded
after the router last changed. Comparing a corpus against decisions made by a
router that no longer exists is the easiest way to calibrate against noise.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROUTING_LOG_CALIBRATION_SCHEMA_VERSION = "omh_routing_log_calibration/v1"

# Every reason a routing record can leave the denominator, named. A skipped
# file that is only counted is a file nobody can argue with.
SKIP_REASONS = (
    "no_routing_record",
    "unreadable_json",
    "not_an_object",
    "unexpected_schema",
    "missing_recorded_at",
    "recorded_before_since",
)

# The router source whose modification time bounds the default window.
ROUTER_SOURCE_RELATIVE_PATH = "src/routing/chat.py"

CLAIM_BOUNDARY = (
    "Routing log calibration reads locally recorded routing decisions only. Records carry "
    "message_sha256 and message_length, never message text, so this compares decision "
    "distributions and never phrasings. It is not live Hermes chat evidence, executor dispatch, "
    "implementation, verification, review, CI, or merge evidence."
)


@dataclass(slots=True)
class CalibrationCoverage:
    """How much of the local record actually reached the numbers below."""

    run_dirs_seen: int = 0
    routing_records_found: int = 0
    parsed: int = 0
    skipped: Counter[str] = field(default_factory=Counter)

    def to_payload(self) -> dict[str, object]:
        return {
            "run_dirs_seen": self.run_dirs_seen,
            "routing_records_found": self.routing_records_found,
            "parsed": self.parsed,
            "parse_coverage_percent": (
                round(self.parsed * 100 / self.routing_records_found, 1)
                if self.routing_records_found
                else None
            ),
            "parse_coverage_numerator": "parsed routing records",
            "parse_coverage_denominator": "routing records found on disk",
            "skipped": {reason: self.skipped[reason] for reason in SKIP_REASONS if self.skipped[reason]},
            "skipped_total": sum(self.skipped.values()),
        }


def router_source_mtime(repo_root: Path | None = None) -> str:
    """Return the router source's mtime as an ISO-8601 UTC timestamp, or ''.

    Defaulting the window to this is the point: a decision recorded by a router
    that has since changed is evidence about a router that no longer exists.
    """
    from datetime import datetime, timezone

    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    source = root / ROUTER_SOURCE_RELATIVE_PATH
    try:
        stamp = source.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _median(values: Sequence[float]) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return round((ordered[middle - 1] + ordered[middle]) / 2, 2)


def _recording_day(recorded_at: str) -> str:
    return recorded_at[:10]


def _read_routing_record(path: Path) -> tuple[Mapping[str, Any] | None, str]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "unreadable_json"
    if not isinstance(parsed, dict):
        return None, "not_an_object"
    decision = parsed.get("route_decision")
    if not isinstance(decision, dict) or not str(decision.get("schema_version", "")).startswith("route_decision/"):
        return None, "unexpected_schema"
    if not str(parsed.get("updated_at", "")):
        return None, "missing_recorded_at"
    return parsed, ""


def collect_routing_records(
    runs_dir: Path,
    *,
    since: str = "",
) -> tuple[list[Mapping[str, Any]], CalibrationCoverage]:
    """Read every recorded routing decision, counting what was skipped and why."""
    coverage = CalibrationCoverage()
    records: list[Mapping[str, Any]] = []
    if not runs_dir.exists():
        return records, coverage
    for run_dir in sorted(path for path in runs_dir.glob("*") if path.is_dir()):
        coverage.run_dirs_seen += 1
        routing_path = run_dir / "routing.json"
        if not routing_path.exists():
            coverage.skipped["no_routing_record"] += 1
            continue
        coverage.routing_records_found += 1
        record, reason = _read_routing_record(routing_path)
        if record is None:
            coverage.skipped[reason] += 1
            continue
        if since and str(record.get("updated_at", "")) < since:
            coverage.skipped["recorded_before_since"] += 1
            continue
        coverage.parsed += 1
        records.append(record)
    return records, coverage


def _corpus_shape() -> dict[str, object]:
    from .routing_precision import ROUTING_INTERVENTION_CASES, ROUTING_PRECISION_CASES

    return {
        "precision_case_count": len(ROUTING_PRECISION_CASES),
        "intervention_case_count": len(ROUTING_INTERVENTION_CASES),
        "total_case_count": len(ROUTING_PRECISION_CASES) + len(ROUTING_INTERVENTION_CASES),
        "note": (
            "The corpus is hand-written and holds message text; recorded decisions hold none. "
            "Compare the two on decision shape — stage mix, confidence mix, margin spread — "
            "and source any new phrasing from an operator's own message."
        ),
    }


def build_routing_log_calibration(
    runs_dir: Path,
    *,
    since: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Aggregate recorded routing decisions into a calibration report."""
    if since is None:
        resolved_since = router_source_mtime(repo_root)
        since_basis = "router_source_mtime" if resolved_since else "none"
    elif since:
        resolved_since = since
        since_basis = "explicit"
    else:
        resolved_since = ""
        since_basis = "none"

    records, coverage = collect_routing_records(runs_dir, since=resolved_since)
    decisions = [record["route_decision"] for record in records]

    margins = [
        float(decision["margin"])
        for decision in decisions
        if isinstance(decision.get("margin"), (int, float)) and not isinstance(decision.get("margin"), bool)
    ]
    by_day: dict[str, list[float]] = {}
    for record, decision in zip(records, decisions):
        margin = decision.get("margin")
        if isinstance(margin, (int, float)) and not isinstance(margin, bool):
            by_day.setdefault(_recording_day(str(record.get("updated_at", ""))), []).append(float(margin))
    per_day_medians = [value for value in (_median(day) for day in by_day.values()) if value is not None]

    return {
        "schema_version": ROUTING_LOG_CALIBRATION_SCHEMA_VERSION,
        "since": {"value": resolved_since, "basis": since_basis, "source": ROUTER_SOURCE_RELATIVE_PATH},
        "coverage": coverage.to_payload(),
        "observed": {
            "decision_count": len(decisions),
            "router_stage": dict(sorted(Counter(str(item.get("router_stage", "")) for item in decisions).items())),
            "action": dict(sorted(Counter(str(item.get("action", "")) for item in decisions).items())),
            "confidence": dict(sorted(Counter(str(item.get("confidence", "")) for item in decisions).items())),
            "selected_skill": dict(sorted(Counter(str(item.get("selected_skill", "")) for item in decisions).items())),
            "source": dict(sorted(Counter(str(record.get("source", "")) for record in records).items())),
        },
        "margin": {
            "reported_count": len(margins),
            "reported_of_decisions": len(decisions),
            "min": min(margins) if margins else None,
            "max": max(margins) if margins else None,
            "median": _median(margins),
            "normalized_median": _median(per_day_medians),
            "normalization_unit": "recording_day",
            "normalization_note": (
                "Recorded routing runs carry no session id, so the recording day is the "
                "normalization unit. normalized_median is the median of per-day medians, so a "
                "heavy day of recording cannot dominate."
            ),
            "day_count": len(by_day),
        },
        "corpus": _corpus_shape(),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def format_routing_log_calibration(payload: Mapping[str, object]) -> str:
    coverage = payload.get("coverage", {})
    observed = payload.get("observed", {})
    margin = payload.get("margin", {})
    corpus = payload.get("corpus", {})
    since = payload.get("since", {})
    assert isinstance(coverage, Mapping) and isinstance(observed, Mapping)
    assert isinstance(margin, Mapping) and isinstance(corpus, Mapping) and isinstance(since, Mapping)

    percent = coverage.get("parse_coverage_percent")
    lines = [
        "OMH routing log calibration",
        f"Since: {since.get('value') or 'all records'} ({since.get('basis')}; {since.get('source')})",
        (
            f"Parse coverage: {coverage.get('parsed', 0)}/{coverage.get('routing_records_found', 0)} "
            f"routing records parsed"
            + (f" ({percent}%)" if percent is not None else " (no records found)")
            + f"; {coverage.get('run_dirs_seen', 0)} run dir(s) walked"
        ),
    ]
    skipped = coverage.get("skipped", {})
    if isinstance(skipped, Mapping) and skipped:
        lines.append("Skipped: " + ", ".join(f"{reason}={count}" for reason, count in skipped.items()))
    lines.append(f"Decisions: {observed.get('decision_count', 0)}")
    for label in ("router_stage", "action", "confidence"):
        bucket = observed.get(label, {})
        if isinstance(bucket, Mapping) and bucket:
            lines.append(f"  {label}: " + ", ".join(f"{key}={value}" for key, value in bucket.items()))
    lines.append(
        f"Margin: median {margin.get('median')}, day-normalized median "
        f"{margin.get('normalized_median')} over {margin.get('day_count', 0)} day(s); "
        f"reported on {margin.get('reported_count', 0)}/{margin.get('reported_of_decisions', 0)} decisions"
    )
    lines.append(
        f"Corpus: {corpus.get('precision_case_count', 0)} precision + "
        f"{corpus.get('intervention_case_count', 0)} intervention case(s)"
    )
    lines.extend(["", f"Boundary: {payload.get('claim_boundary', '')}"])
    return "\n".join(lines)


__all__ = [
    "CLAIM_BOUNDARY",
    "ROUTER_SOURCE_RELATIVE_PATH",
    "ROUTING_LOG_CALIBRATION_SCHEMA_VERSION",
    "SKIP_REASONS",
    "CalibrationCoverage",
    "build_routing_log_calibration",
    "collect_routing_records",
    "format_routing_log_calibration",
    "router_source_mtime",
]
