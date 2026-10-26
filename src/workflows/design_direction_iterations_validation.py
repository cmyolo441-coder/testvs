from __future__ import annotations

from copy import deepcopy

from .design_direction_iterations_schema import (
    DESIGN_DIRECTION_ITERATION_SCHEMA_VERSION,
    MAX_MODEL_ATTEMPTS,
    MAX_SNAPSHOTS,
    SUCCESSOR_KINDS,
    TERMINAL_REASONS,
    _FEEDBACK_TERMS,
    _OPAQUE_REF,
    _SHA256,
    _budget_usage,
    _usage,
    _scores_are_comparable,
    canonical_digest,
)
from .design_directions import validate_design_direction_set


def validate_design_direction_iteration(value: object) -> list[str]:
    """Return contract violations; a valid record is fully lineage-verifiable offline."""
    if not isinstance(value, dict):
        return ["iteration must be an object"]
    required = {"schema_version", "iteration_id", "root_set_digest", "source_revision_digest", "policy", "snapshots", "budget_usage", "terminal", "memory_promotion", "claim_boundary"}
    if set(value) != required:
        return ["iteration keys are invalid"]
    errors: list[str] = []
    if value["schema_version"] != DESIGN_DIRECTION_ITERATION_SCHEMA_VERSION:
        errors.append("iteration schema_version is invalid")
    if not _OPAQUE_REF.fullmatch(str(value["iteration_id"])):
        errors.append("iteration_id is invalid")
    if not _digest(value["root_set_digest"]) or not _digest(value["source_revision_digest"]):
        errors.append("iteration root or source digest is invalid")
    policy = value["policy"]
    if not isinstance(policy, dict) or policy != {"max_active_options": 4, "max_revision_rounds": 4, "max_snapshots": 5, "max_model_attempts": 8, "score_threshold": policy.get("score_threshold") if isinstance(policy, dict) else None, "no_improvement_rule": "strict_increase_on_same_criteria"}:
        errors.append("iteration policy is invalid")
    elif not _number(policy["score_threshold"], minimum=0, maximum=100):
        errors.append("iteration policy score_threshold is invalid")
    snapshots = value["snapshots"]
    if not isinstance(snapshots, list) or not 1 <= len(snapshots) <= MAX_SNAPSHOTS:
        return [*errors, "iteration snapshots are invalid"]
    prior: dict[str, object] | None = None
    for index, snapshot in enumerate(snapshots):
        errors.extend(_snapshot_errors(snapshot, index, value, prior))
        prior = snapshot if isinstance(snapshot, dict) else None
    if isinstance(snapshots[0], dict) and snapshots[0].get("revision_digest") != value["root_set_digest"]:
        # Root-set identity commits the closed direction-set bytes, not the
        # containing iteration snapshot, so this condition is deliberately not
        # an equality requirement. It remains here as a no-op-free type guard.
        root_set = snapshots[0].get("direction_set")
        if canonical_digest(root_set) != value["root_set_digest"]:
            errors.append("root_set_digest does not match root direction set")
    if isinstance(snapshots[0], dict) and canonical_digest(snapshots[0].get("direction_set")) != value["root_set_digest"]:
        errors.append("root_set_digest does not match root direction set")
    expected_budget = _budget_usage([snapshot for snapshot in snapshots if isinstance(snapshot, dict)])
    if value["budget_usage"] != expected_budget:
        errors.append("budget_usage must match complete model attempt telemetry")
    if int(expected_budget["model_attempts"]) > MAX_MODEL_ATTEMPTS:
        errors.append("model attempts exceed the hard cap")
    repair_count = sum(
        attempt.get("purpose") == "schema_repair"
        for snapshot in snapshots
        if isinstance(snapshot, dict)
        for attempt in snapshot.get("model_attempts", [])
        if isinstance(attempt, dict)
    )
    if repair_count > 1:
        errors.append("schema repair exceeds the hard cap")
    errors.extend(_terminal_errors(value["terminal"], snapshots))
    errors.extend(_memory_errors(value["memory_promotion"], value["terminal"]))
    if not isinstance(value["claim_boundary"], str) or "visual-QA" not in value["claim_boundary"]:
        errors.append("claim_boundary must deny visual-QA proof")
    return errors


def _snapshot_errors(snapshot: object, index: int, iteration: dict[str, object], parent: dict[str, object] | None) -> list[str]:
    if not isinstance(snapshot, dict):
        return [f"snapshots[{index}] must be an object"]
    required = {"revision_index", "revision_digest", "parent_revision_digest", "root_set_digest", "source_revision_digest", "criteria", "direction_set", "option_refs", "feedback", "successors", "scores", "model_attempts", "usage", "score_comparison", "idempotency_key"}
    if set(snapshot) != required:
        return [f"snapshots[{index}] keys are invalid"]
    errors: list[str] = []
    if snapshot["revision_index"] != index or not _digest(snapshot["revision_digest"]):
        errors.append(f"snapshots[{index}] revision identity is invalid")
    parent_digest = "" if parent is None else parent.get("revision_digest")
    if snapshot["parent_revision_digest"] != parent_digest:
        errors.append(f"snapshots[{index}] parent revision is invalid")
    if snapshot["root_set_digest"] != iteration["root_set_digest"] or snapshot["source_revision_digest"] != iteration["source_revision_digest"]:
        errors.append(f"snapshots[{index}] root or source lineage drifted")
    if validate_design_direction_set(snapshot["direction_set"]):
        errors.append(f"snapshots[{index}] direction_set is invalid")
    errors.extend(_criteria_errors(snapshot["criteria"], index, parent))
    errors.extend(_option_ref_errors(snapshot))
    errors.extend(_feedback_errors(snapshot["feedback"], index))
    errors.extend(_attempt_errors(snapshot["model_attempts"], snapshot["usage"]))
    errors.extend(_score_errors(snapshot["scores"], snapshot["criteria"]))
    errors.extend(_comparison_errors(snapshot["score_comparison"], snapshot, parent))
    errors.extend(_successor_errors(snapshot["successors"], snapshot, parent))
    if not _digest(snapshot["idempotency_key"]):
        errors.append(f"snapshots[{index}] idempotency_key is invalid")
    if canonical_digest(_digest_projection(snapshot)) != snapshot["revision_digest"]:
        errors.append(f"snapshots[{index}] revision_digest does not match immutable contents")
    return errors


def _criteria_errors(criteria: object, index: int, parent: dict[str, object] | None) -> list[str]:
    if not isinstance(criteria, dict) or set(criteria) != {"revision", "dimensions", "score_threshold", "digest"}:
        return [f"snapshots[{index}] criteria is invalid"]
    payload = {"revision": criteria["revision"], "dimensions": criteria["dimensions"], "score_threshold": criteria["score_threshold"]}
    if not _OPAQUE_REF.fullmatch(str(criteria["revision"])) or not isinstance(criteria["dimensions"], list) or not criteria["dimensions"] or any(not _OPAQUE_REF.fullmatch(str(item)) for item in criteria["dimensions"]) or len(set(criteria["dimensions"])) != len(criteria["dimensions"]) or not _number(criteria["score_threshold"], minimum=0, maximum=100) or criteria["digest"] != canonical_digest(payload):
        return [f"snapshots[{index}] criteria is invalid"]
    if parent is not None and isinstance(parent.get("criteria"), dict) and criteria["revision"] == parent["criteria"]["revision"] and criteria != parent["criteria"]:
        return [f"snapshots[{index}] criteria revision changed without a new revision"]
    return []


def _option_ref_errors(snapshot: dict[str, object]) -> list[str]:
    direction_set = snapshot["direction_set"]
    options = direction_set.get("options") if isinstance(direction_set, dict) else None
    expected_ids = [str(option.get("option_id", "")) for option in options if isinstance(option, dict)] if isinstance(options, list) else []
    expected_refs = [f"{snapshot['revision_digest']}:{option_id}" for option_id in expected_ids]
    return [] if snapshot["option_refs"] == expected_refs else ["snapshot option_refs must be stable current option references"]


def _feedback_errors(feedback: object, index: int) -> list[str]:
    if not isinstance(feedback, dict) or set(feedback) != {"reference", "delta"} or not isinstance(feedback["reference"], str) or not isinstance(feedback["delta"], list):
        return [f"snapshots[{index}] feedback is invalid"]
    delta = feedback["delta"]
    if index == 0:
        return [] if feedback["reference"] == "" and delta == [] else ["root feedback must be empty"]
    if not _OPAQUE_REF.fullmatch(feedback["reference"]) or not delta or any(item not in _FEEDBACK_TERMS for item in delta) or len(set(delta)) != len(delta):
        return [f"snapshots[{index}] feedback must use opaque structured vocabulary"]
    return []


def _attempt_errors(attempts: object, usage: object) -> list[str]:
    if not isinstance(attempts, list) or not isinstance(usage, dict):
        return ["model telemetry is invalid"]
    required = {"purpose", "model_id", "tokens", "cost", "latency_ms", "failure_ref"}
    for attempt in attempts:
        if not isinstance(attempt, dict) or set(attempt) != required or attempt["purpose"] not in ("generation", "evaluation", "schema_repair") or not _OPAQUE_REF.fullmatch(str(attempt["model_id"])) or any(not _number_or_none(attempt[field]) for field in ("tokens", "cost", "latency_ms")) or (attempt["failure_ref"] is not None and not _OPAQUE_REF.fullmatch(str(attempt["failure_ref"]))):
            return ["model telemetry is invalid"]
    return [] if usage == _usage(attempts) else ["usage must match model telemetry"]


def _score_errors(scores: object, criteria: object) -> list[str]:
    if not isinstance(scores, list) or not isinstance(criteria, dict):
        return ["scores are invalid"]
    dimensions = criteria.get("dimensions")
    if not isinstance(dimensions, list):
        return ["scores are invalid"]
    names: list[str] = []
    for score in scores:
        if not isinstance(score, dict) or set(score) != {"dimension", "score", "evidence_ref", "evaluator_id", "rubric_revision"} or score["dimension"] not in dimensions or not _number(score["score"], minimum=0, maximum=100) or any(not _OPAQUE_REF.fullmatch(str(score[field])) for field in ("evidence_ref", "evaluator_id", "rubric_revision")):
            return ["scores are invalid"]
        names.append(str(score["dimension"]))
    return [] if len(set(names)) == len(names) else ["scores repeat a dimension"]


def _comparison_errors(comparison: object, snapshot: dict[str, object], parent: dict[str, object] | None) -> list[str]:
    if not isinstance(comparison, dict) or set(comparison) != {"baseline_revision_digest", "comparable_with_parent"} or not isinstance(comparison["comparable_with_parent"], bool):
        return ["score comparison is invalid"]
    if parent is None:
        return [] if comparison == {"baseline_revision_digest": "", "comparable_with_parent": False} else ["root cannot compare scores"]
    same = isinstance(parent.get("criteria"), dict) and snapshot["criteria"] == parent["criteria"]
    comparable = bool(same and _scores_are_comparable(parent["scores"], snapshot["scores"]))
    expected = {"baseline_revision_digest": parent["revision_digest"] if comparable else "", "comparable_with_parent": comparable}
    return [] if comparison == expected else ["score comparison must preserve criteria baseline separation"]


def _successor_errors(successors: object, snapshot: dict[str, object], parent: dict[str, object] | None) -> list[str]:
    if not isinstance(successors, list):
        return ["successors are invalid"]
    if parent is None:
        return [] if successors == [] else ["root cannot have successors"]
    parent_refs = set(parent["option_refs"])
    child_refs = set(snapshot["option_refs"])
    covered: set[str] = set()
    targets: set[str] = set()
    for successor in successors:
        if not isinstance(successor, dict) or set(successor) != {"kind", "from_option_refs", "to_option_ref"}:
            return ["successors are invalid"]
        kind, sources, target = successor["kind"], successor["from_option_refs"], successor["to_option_ref"]
        if kind not in SUCCESSOR_KINDS or not isinstance(sources, list) or not isinstance(target, str):
            return ["successors are invalid"]
        if (kind == "introduced" and sources) or (kind == "combined" and len(sources) < 2) or (kind in ("preserved", "revised", "dropped") and len(sources) != 1) or (kind == "dropped" and target) or (kind != "dropped" and target not in child_refs) or not set(sources).issubset(parent_refs) or covered.intersection(sources) or (target and target in targets):
            return ["successors do not preserve valid ancestry"]
        covered.update(sources)
        if target:
            targets.add(target)
    return [] if covered == parent_refs and targets == child_refs else ["successors must cover complete ancestry"]


def _terminal_errors(terminal: object, snapshots: list[object]) -> list[str]:
    if not isinstance(terminal, dict) or set(terminal) != {"outcome", "reason", "accepted_revision_digest", "accepted_option_ref"}:
        return ["terminal is invalid"]
    outcome, reason = terminal["outcome"], terminal["reason"]
    if outcome == "OPEN":
        return [] if reason is None and terminal["accepted_revision_digest"] is None and terminal["accepted_option_ref"] is None else ["open terminal is invalid"]
    if reason not in TERMINAL_REASONS:
        return ["terminal reason is invalid"]
    if outcome == "ACCEPT":
        current = snapshots[-1] if isinstance(snapshots[-1], dict) else {}
        valid = reason == "accepted" and terminal["accepted_revision_digest"] == current.get("revision_digest") and terminal["accepted_option_ref"] in current.get("option_refs", [])
        return [] if valid else ["accepted terminal must bind the current stable option"]
    expected = "CANCELLED" if reason == "cancelled" else "BLOCK/REVISE" if reason in ("no_improvement", "revision_cap_exhausted", "model_call_cap_exhausted", "blocked_evidence", "blocked_capability") else "STOPPED"
    return [] if outcome == expected and terminal["accepted_revision_digest"] is None and terminal["accepted_option_ref"] is None else ["terminal outcome is invalid"]


def _memory_errors(memory: object, terminal: object) -> list[str]:
    if not isinstance(memory, dict) or set(memory) != {"state", "request"}:
        return ["memory promotion is invalid"]
    if memory == {"state": "not_requested", "request": None}:
        return []
    request = memory["request"]
    if not isinstance(request, dict) or set(request) != {"action", "review_required", "accepted_revision_digest", "automatic_write", "global_promotion"} or not isinstance(terminal, dict):
        return ["memory promotion is invalid"]
    valid = memory["state"] == "requested_for_review" and request["action"] == "memory-new" and request["review_required"] is True and request["accepted_revision_digest"] == terminal.get("accepted_revision_digest") and request["automatic_write"] is False and request["global_promotion"] is False
    return [] if valid else ["memory promotion must remain a reviewed accepted-revision request"]


def _digest_projection(snapshot: dict[str, object]) -> dict[str, object]:
    projection = deepcopy(snapshot)
    projection.pop("revision_digest", None)
    projection["option_refs"] = [str(ref).rsplit(":", 1)[-1] for ref in projection["option_refs"]]
    for successor in projection["successors"]:
        if isinstance(successor, dict):
            successor["to_option_ref"] = str(successor["to_option_ref"]).rsplit(":", 1)[-1] if successor["to_option_ref"] else ""
    return projection


def _digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _number(value: object, *, minimum: float, maximum: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and minimum <= value <= maximum


def _number_or_none(value: object) -> bool:
    return value is None or _number(value, minimum=0, maximum=float("inf"))
