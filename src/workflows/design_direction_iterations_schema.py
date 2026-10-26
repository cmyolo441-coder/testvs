from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Final

from .design_directions import validate_design_direction_set

DESIGN_DIRECTION_ITERATION_SCHEMA_VERSION: Final = "design_direction_iteration/v1"
MAX_ACTIVE_OPTIONS: Final = 4
MAX_REVISION_ROUNDS: Final = 4
MAX_SNAPSHOTS: Final = 5
MAX_MODEL_ATTEMPTS: Final = 8
SUCCESSOR_KINDS: Final = ("preserved", "revised", "combined", "introduced", "dropped")
TERMINAL_REASONS: Final = ("accepted", "threshold_reached", "no_improvement", "revision_cap_exhausted", "model_call_cap_exhausted", "blocked_evidence", "blocked_capability", "cancelled")
_FEEDBACK_TERMS: Final = ("hierarchy", "palette", "typography", "layout", "signature_element", "avoid_patterns")
_OPAQUE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_digest(value: object) -> str:
    """Hash a deterministic JSON projection without retaining source bytes."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_design_direction_iteration(
    direction_set: object,
    *,
    source_revision_digest: str,
    criteria_revision: str,
    criteria_dimensions: tuple[str, ...],
    score_threshold: float,
    scores: tuple[tuple[str, float, str, str, str], ...] = (),
) -> dict[str, object]:
    """Bind one closed direction set to an immutable root and declared stop policy."""
    _require_direction_set(direction_set)
    root_set_digest = canonical_digest(direction_set)
    criteria = _criteria(criteria_revision, criteria_dimensions, score_threshold)
    snapshot = _snapshot(
        index=0,
        parent_digest="",
        root_set_digest=root_set_digest,
        source_revision_digest=source_revision_digest,
        criteria=criteria,
        direction_set=direction_set,
        feedback_reference="",
        feedback_delta=(),
        successors=(),
        scores=scores,
        model_attempts=(),
        comparison_baseline="",
        comparable=False,
    )
    iteration_id = f"design-direction-iteration-{canonical_digest((root_set_digest, source_revision_digest, criteria['digest'], snapshot['revision_digest']))[:16]}"
    record: dict[str, object] = {
        "schema_version": DESIGN_DIRECTION_ITERATION_SCHEMA_VERSION,
        "iteration_id": iteration_id,
        "root_set_digest": root_set_digest,
        "source_revision_digest": _digest(source_revision_digest, "source_revision_digest"),
        "policy": {
            "max_active_options": MAX_ACTIVE_OPTIONS,
            "max_revision_rounds": MAX_REVISION_ROUNDS,
            "max_snapshots": MAX_SNAPSHOTS,
            "max_model_attempts": MAX_MODEL_ATTEMPTS,
            "score_threshold": score_threshold,
            "no_improvement_rule": "strict_increase_on_same_criteria",
        },
        "snapshots": [snapshot],
        "budget_usage": _budget_usage([snapshot]),
        "terminal": _terminal_open(),
        "memory_promotion": {"state": "not_requested", "request": None},
        "claim_boundary": "Direction-fit scores are advisory preference evidence only. This artifact makes no provider, browser, executor, implementation, accessibility, visual-QA, review, CI, deployment, or merge claim; missing or incomparable evidence never becomes PASS.",
    }
    return record


def _snapshot(*, index: int, parent_digest: str, root_set_digest: str, source_revision_digest: str, criteria: dict[str, object], direction_set: object, feedback_reference: str, feedback_delta: tuple[str, ...], successors: tuple[tuple[str, tuple[str, ...], str], ...], scores: tuple[tuple[str, float, str, str, str], ...], model_attempts: tuple[tuple[str, str, int | None, float | None, float | None, str | None], ...], comparison_baseline: str, comparable: bool) -> dict[str, object]:
    options = _option_ids(direction_set)
    attempts = _attempt_records(model_attempts)
    score_records = _score_records(scores, criteria)
    successor_records = _successor_records(successors, parent_digest, options)
    base = {
        "revision_index": index,
        "parent_revision_digest": parent_digest,
        "root_set_digest": root_set_digest,
        "source_revision_digest": _digest(source_revision_digest, "source_revision_digest"),
        "criteria": criteria,
        "direction_set": deepcopy(direction_set),
        "option_refs": options,
        "feedback": {
            "reference": _opaque_or_empty(feedback_reference, "feedback_reference"),
            "delta": list(_feedback_delta(feedback_delta)) if feedback_reference or feedback_delta else [],
        },
        "successors": successor_records,
        "scores": score_records,
        "model_attempts": attempts,
        "usage": _usage(attempts),
        "score_comparison": {"baseline_revision_digest": comparison_baseline, "comparable_with_parent": comparable},
        "idempotency_key": canonical_digest(
            (parent_digest, feedback_reference, _feedback_delta(feedback_delta) if feedback_reference or feedback_delta else ())
        ),
    }
    digest = canonical_digest(base)
    base["revision_digest"] = digest
    base["option_refs"] = [f"{digest}:{option_id}" for option_id in options]
    for successor in successor_records:
        target = str(successor["to_option_ref"])
        successor["to_option_ref"] = f"{digest}:{target}" if target else ""
    return base


def _criteria(revision: str, dimensions: tuple[str, ...], threshold: float) -> dict[str, object]:
    clean_revision = _opaque(revision, "criteria_revision")
    clean_dimensions = tuple(_opaque(item, "criteria_dimension") for item in dimensions)
    if not clean_dimensions or len(set(clean_dimensions)) != len(clean_dimensions):
        raise ValueError("criteria_dimensions must be a non-empty unique vocabulary")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 100:
        raise ValueError("score_threshold must be between 0 and 100")
    payload = {"revision": clean_revision, "dimensions": list(clean_dimensions), "score_threshold": threshold}
    return {**payload, "digest": canonical_digest(payload)}


def _attempt_records(attempts: tuple[tuple[str, str, int | None, float | None, float | None, str | None], ...]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for purpose, model_id, tokens, cost, latency, failure_ref in attempts:
        if purpose not in ("generation", "evaluation", "schema_repair"):
            raise ValueError("model attempt purpose is unsupported")
        records.append({"purpose": purpose, "model_id": _opaque_or_unknown(model_id, "model_id"), "tokens": _nonnegative_or_none(tokens, "tokens"), "cost": _nonnegative_or_none(cost, "cost"), "latency_ms": _nonnegative_or_none(latency, "latency_ms"), "failure_ref": _opaque_or_none(failure_ref, "failure_ref")})
    if sum(item["purpose"] == "schema_repair" for item in records) > 1:
        raise ValueError("schema repair is allowed at most once per revision")
    return records


def _score_records(scores: tuple[tuple[str, float, str, str, str], ...], criteria: dict[str, object]) -> list[dict[str, object]]:
    dimensions = criteria["dimensions"]
    if not isinstance(dimensions, list):
        raise ValueError("criteria dimensions are invalid")
    records: list[dict[str, object]] = []
    for dimension, score, evidence_ref, evaluator_id, rubric_revision in scores:
        if dimension not in dimensions or isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 100:
            raise ValueError("score must name a criteria dimension and be between 0 and 100")
        records.append({"dimension": dimension, "score": score, "evidence_ref": _opaque(evidence_ref, "score evidence_ref"), "evaluator_id": _opaque(evaluator_id, "evaluator_id"), "rubric_revision": _opaque(rubric_revision, "rubric_revision")})
    if len({str(item["dimension"]) for item in records}) != len(records):
        raise ValueError("scores must name each dimension at most once")
    return records


def _scores_are_comparable(left: object, right: object) -> bool:
    """Scores compare only when they cover the same dimensions under one judge/rubric."""
    if not isinstance(left, list) or not isinstance(right, list) or not left or not right:
        return False
    def identities(scores: list[object]) -> tuple[tuple[str, str, str], ...] | None:
        result: list[tuple[str, str, str]] = []
        for score in scores:
            if not isinstance(score, dict):
                return None
            dimension = score.get("dimension")
            evaluator_id = score.get("evaluator_id")
            rubric_revision = score.get("rubric_revision")
            if not isinstance(dimension, str) or not isinstance(evaluator_id, str) or not isinstance(rubric_revision, str):
                return None
            result.append((dimension, evaluator_id, rubric_revision))
        return tuple(sorted(result))
    left_identities = identities(left)
    right_identities = identities(right)
    return left_identities is not None and left_identities == right_identities


def _successor_records(successors: tuple[tuple[str, tuple[str, ...], str], ...], parent_digest: str, option_ids: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for kind, source_ids, target_id in successors:
        if kind not in SUCCESSOR_KINDS:
            raise ValueError("successor kind is unsupported")
        records.append({"kind": kind, "from_option_refs": [f"{parent_digest}:{option_id}" for option_id in source_ids] if parent_digest else [], "to_option_ref": target_id})
    return records


def _usage(attempts: list[dict[str, object]]) -> dict[str, object]:
    def total(key: str) -> int | float | None:
        values = [item[key] for item in attempts]
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        return None if not values or len(numeric) != len(values) else sum(numeric)
    return {"model_attempts": len(attempts), "observed_tokens": total("tokens"), "observed_cost": total("cost"), "observed_latency_ms": total("latency_ms"), "failure_count": None if not attempts else sum(item["failure_ref"] is not None for item in attempts)}


def _budget_usage(snapshots: list[dict[str, object]]) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    for snapshot in snapshots:
        model_attempts = snapshot.get("model_attempts")
        if isinstance(model_attempts, list):
            attempts.extend(attempt for attempt in model_attempts if isinstance(attempt, dict))
    return _usage(attempts)


def _option_ids(direction_set: object) -> list[str]:
    if not isinstance(direction_set, dict):
        raise ValueError("direction_set is invalid")
    options = direction_set.get("options")
    if not isinstance(options, list):
        raise ValueError("direction_set options are invalid")
    return [str(option["option_id"]) for option in options if isinstance(option, dict)]


def _feedback_delta(delta: tuple[str, ...]) -> tuple[str, ...]:
    if not delta or any(item not in _FEEDBACK_TERMS for item in delta) or len(set(delta)) != len(delta):
        raise ValueError("feedback_delta must be a non-empty unique structured design vocabulary")
    return delta


def _terminal_open() -> dict[str, object]:
    return {"outcome": "OPEN", "reason": None, "accepted_revision_digest": None, "accepted_option_ref": None}


def _require_direction_set(value: object) -> None:
    if not isinstance(value, dict) or validate_design_direction_set(value):
        raise ValueError("direction_set must be a valid design_direction_set/v1")


def _digest(value: str, label: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a sha256 digest")
    return value


def _opaque(value: str, label: str) -> str:
    if not _OPAQUE_REF.fullmatch(value):
        raise ValueError(f"{label} must be an opaque identifier")
    return value


def _opaque_or_empty(value: str, label: str) -> str:
    return "" if not value else _opaque(value, label)


def _opaque_or_none(value: str | None, label: str) -> str | None:
    return None if value is None else _opaque(value, label)


def _opaque_or_unknown(value: str, label: str) -> str:
    return "unknown" if not value else _opaque(value, label)


def _nonnegative_or_none(value: int | float | None, label: str) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{label} must be non-negative or null")
    return value
