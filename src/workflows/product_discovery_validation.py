"""Executable product-discovery preparation, evaluation, and durable re-entry."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from ..system.append_only_store import append_store_line
from ..system.local_store import file_lock, read_jsonl_objects
from ..system.paths import OmhPaths
from .product_discovery_artifacts import (
    ASSUMPTION_TEST_PORTFOLIO_SCHEMA_VERSION,
    BUILD_BOUNDARY_ROUTES,
    CHANNEL_FEEDBACK_AFFECTED_TARGETS,
    CHANNEL_FEEDBACK_ENTRY_KEYS,
    CHANNEL_FEEDBACK_LEDGER_SCHEMA_VERSION,
    CHANNEL_FEEDBACK_DISPOSITION_SCHEMA_VERSION,
    _artifact,
    _channel_feedback_entry,
    _ref,
    _stamp as _feedback_stamp,
    build_channel_feedback_ledger,
    build_channel_feedback_disposition,
    CUSTOMER_DISCOVERY_PLAN_SCHEMA_VERSION,
    DISCOVERY_CONTINUATION_ROUTE,
    DISCOVERY_DECISION_FRAME_SCHEMA_VERSION,
    DISCOVERY_DECISION_RECEIPT_SCHEMA_VERSION,
    DISCOVERY_EVIDENCE_LEDGER_SCHEMA_VERSION,
    INITIAL_GTM_HYPOTHESIS_SCHEMA_VERSION,
    audience_is_defined,
    build_assumption_test_portfolio,
    build_customer_discovery_plan,
    build_discovery_decision_frame,
    build_discovery_decision_receipt,
    build_discovery_evidence_ledger,
    build_initial_gtm_hypothesis,
    missing_audience_evidence_refs,
)
from .product_discovery_artifact_validation import validate_product_discovery_artifact
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


PRODUCT_DISCOVERY_STORE_NAME: Final = "product_discovery_artifacts.jsonl"
DISCOVERY_AUDIENCE_GATE_SCHEMA_VERSION: Final = "discovery_audience_gate/v1"
_EXTERNAL_EVIDENCE_CLASSES: Final = ("external_human", "behavioral_data")
_AUDIENCE_GATE_CLAIM_BOUNDARY: Final = (
    "The audience gate reads one prepared decision frame. Evidence gathering may continue while the audience "
    "is undefined; solution work is permitted only by a validated decision receipt. This is not recruitment, "
    "customer research, a PRD, a prototype, code, execution, review, CI, or merge evidence."
)

__all__ = (
    "append_product_discovery_artifact",
    "build_assumption_test_portfolio",
    "build_customer_discovery_plan",
    "build_discovery_decision_frame",
    "build_discovery_evidence_ledger",
    "build_initial_gtm_hypothesis",
    "discovery_audience_gate",
    "discovery_audience_gate_with_feedback",
    "evaluate_channel_feedback",
    "channel_feedback_ledger_for_history",
    "product_brief_consumption_with_feedback",
    "evaluate_product_discovery",
    "prepare_product_discovery",
    "product_brief_consumption",
    "read_product_discovery_artifacts",
    "validate_product_discovery_artifact",
)


def prepare_product_discovery(
    *,
    frame: Mapping[str, Any],
    ledger: Mapping[str, Any],
    plan: Mapping[str, Any],
    portfolio: Mapping[str, Any],
    gtm: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate the five pre-decision artifacts as one discovery package."""
    artifacts = {"frame": frame, "ledger": ledger, "plan": plan, "portfolio": portfolio, "gtm": gtm}
    expected_schemas = {
        "frame": DISCOVERY_DECISION_FRAME_SCHEMA_VERSION,
        "ledger": DISCOVERY_EVIDENCE_LEDGER_SCHEMA_VERSION,
        "plan": CUSTOMER_DISCOVERY_PLAN_SCHEMA_VERSION,
        "portfolio": ASSUMPTION_TEST_PORTFOLIO_SCHEMA_VERSION,
        "gtm": INITIAL_GTM_HYPOTHESIS_SCHEMA_VERSION,
    }
    discovery_ids: set[str] = set()
    prepared: dict[str, dict[str, Any]] = {}
    for name, artifact in artifacts.items():
        errors = validate_product_discovery_artifact(artifact)
        if errors:
            raise ValueError(f"{name} is invalid: {errors[0]}")
        if artifact.get("schema_version") != expected_schemas[name]:
            raise ValueError(f"{name} has the wrong artifact type")
        discovery_ids.add(str(artifact["discovery_id"]))
        prepared[name] = dict(artifact)
    if len(discovery_ids) != 1:
        raise ValueError("all discovery artifacts must have the same discovery_id")
    frame_segment = str(prepared["frame"]["segment_ref"])
    if frame_segment != str(prepared["gtm"]["beachhead_segment_ref"]):
        raise ValueError("initial GTM beachhead must match the framed target segment")
    for assumption in prepared["portfolio"]["assumptions"]:
        if str(assumption["scope_segment_ref"]) != frame_segment:
            raise ValueError("assumption scope must match the framed target segment")
        if _time(str(assumption["deadline_at"])) > _time(str(prepared["frame"]["deadline_at"])):
            raise ValueError("assumption deadline must not exceed the decision-frame deadline")
    return prepared


def evaluate_product_discovery(package: Mapping[str, Mapping[str, Any]], *, now: str) -> dict[str, Any]:
    """Derive a conservative kill/pivot/persevere/inconclusive receipt."""
    required = {"frame", "ledger", "plan", "portfolio", "gtm"}
    if set(package) != required:
        raise ValueError("product discovery package keys are invalid")
    prepared = prepare_product_discovery(**package)
    frame = prepared["frame"]
    assumptions = prepared["portfolio"]["assumptions"]
    entries = prepared["ledger"]["entries"]
    if not isinstance(assumptions, list) or not isinstance(entries, list):
        raise ValueError("prepared discovery artifacts have invalid rows")
    _stamp(now, "now")
    evaluated_at = _time(now)
    segment_state = str(frame["segment_definition_state"])
    audience_defined = audience_is_defined(segment_state)
    outcomes = [_assumption_outcome(assumption, entries, evaluated_at=evaluated_at) for assumption in assumptions]
    failed = [outcome for outcome in outcomes if outcome["contradicted"]]
    rejected = [outcome["assumption_id"] for outcome in failed]
    if failed:
        decision = "kill" if any(outcome["failure_decision"] == "kill" for outcome in failed) else "pivot"
        problem_gate, route = "refuted", DISCOVERY_CONTINUATION_ROUTE
    elif outcomes and all(outcome["validated"] for outcome in outcomes) and audience_defined:
        decision, problem_gate, route = "persevere", "validated", "product-brief"
    else:
        decision, problem_gate, route = "inconclusive", "inconclusive", DISCOVERY_CONTINUATION_ROUTE
    eligible_refs = [reference for outcome in outcomes for reference in outcome["eligible_refs"]]
    residual = ["risk-evidence-limits"]
    if any(outcome["timed_out"] for outcome in outcomes):
        residual.append("risk-test-deadline-expired")
    if decision == "inconclusive":
        residual.append("risk-unresolved-assumptions")
    if not audience_defined:
        residual.append("risk-audience-undefined")
    return build_discovery_decision_receipt(
        discovery_id=str(frame["discovery_id"]),
        problem_ref=str(frame["problem_ref"]),
        segment_ref=str(frame["segment_ref"]),
        segment_definition_state=segment_state,
        decision=decision,
        problem_gate=problem_gate,
        precommitted_test_ids=[str(assumption["test_id"]) for assumption in assumptions],
        eligible_evidence_refs=eligible_refs,
        rejected_hypothesis_ids=rejected,
        residual_risk_refs=residual,
        next_route=route,
    )


def _channel_feedback_input(feedback_ledger: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"discovery_id", "gtm_artifact_id", "initial_channel_ref", "segment_ref", "entries"}
    if not isinstance(feedback_ledger, Mapping):
        raise ValueError("feedback_ledger must be an object")
    if "schema_version" in feedback_ledger:
        errors = validate_product_discovery_artifact(feedback_ledger)
        if errors:
            raise ValueError(errors[0])
        if feedback_ledger["schema_version"] != CHANNEL_FEEDBACK_LEDGER_SCHEMA_VERSION:
            raise ValueError("feedback_ledger has the wrong artifact type")
    elif set(feedback_ledger) != fields:
        raise ValueError("channel feedback semantic input keys are invalid")
    values = {field: _ref(feedback_ledger[field], field) for field in fields - {"entries"}}
    entries = feedback_ledger["entries"]
    if not isinstance(entries, list):
        raise ValueError("channel feedback entries must be a list")
    values["entries"] = [
        _channel_feedback_entry(entry, discovery_id=values["discovery_id"], allow_missing=True) for entry in entries
    ]
    return values


def channel_feedback_ledger_for_history(feedback_ledger: Mapping[str, Any]) -> dict[str, Any]:
    """Return an appendable ledger, or no ledger for incomplete/duplicate supplied rows.

    The evaluator still retains each refusal in the appendable disposition. A
    digest of the bounded submission identifies inputs that cannot be a ledger.
    """
    values = _channel_feedback_input(feedback_ledger)
    entries = values["entries"]
    if any(set(entry) != CHANNEL_FEEDBACK_ENTRY_KEYS for entry in entries):
        return {}
    ids = {entry["feedback_id"] for entry in entries}
    sources = {(entry["test_id"], entry["source_ref"], entry["affected_target"]) for entry in entries}
    if len(ids) != len(entries) or len(sources) != len(entries):
        return {}
    return build_channel_feedback_ledger(**values)


def evaluate_channel_feedback(*, frame: Mapping[str, Any], gtm: Mapping[str, Any],
        portfolio: Mapping[str, Any], feedback_ledger: Mapping[str, Any], now: str) -> dict[str, Any]:
    """Reconcile explicit targets using fixed precedence; no clock, I/O or receipt edits."""
    for name, artifact, schema in (("frame", frame, DISCOVERY_DECISION_FRAME_SCHEMA_VERSION),
                                  ("gtm", gtm, INITIAL_GTM_HYPOTHESIS_SCHEMA_VERSION),
                                  ("portfolio", portfolio, ASSUMPTION_TEST_PORTFOLIO_SCHEMA_VERSION)):
        errors = validate_product_discovery_artifact(artifact)
        if errors:
            raise ValueError(f"{name} is invalid: {errors[0]}")
        if artifact["schema_version"] != schema:
            raise ValueError(f"{name} has the wrong artifact type")
    values = _channel_feedback_input(feedback_ledger)
    if any(artifact["discovery_id"] != frame["discovery_id"] for artifact in (gtm, portfolio, values)):
        raise ValueError("all channel feedback artifacts must have the same discovery_id")
    if gtm["beachhead_segment_ref"] != frame["segment_ref"]:
        raise ValueError("initial GTM beachhead must match the framed target segment")
    tests = {row["test_id"]: row for row in portfolio["assumptions"]}
    for test in tests.values():
        if test["scope_segment_ref"] != frame["segment_ref"]:
            raise ValueError("assumption scope must match the framed target segment")
        if _time(test["deadline_at"]) > _time(frame["deadline_at"]):
            raise ValueError("assumption deadline must not exceed the decision-frame deadline")
    evaluated_at = _time(_feedback_stamp(now, "now"))
    entries = values["entries"]
    source_keys = Counter((row.get("test_id"), row.get("source_ref"), row.get("affected_target")) for row in entries)
    ids = Counter(row.get("feedback_id") for row in entries)
    source_effects: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    for row in entries:
        source_effects[(row.get("source_ref"), row.get("affected_target"))].add(row.get("effect", "unknown"))
    held = []
    admissible: dict[str, list[dict[str, Any]]] = {target: [] for target in CHANNEL_FEEDBACK_AFFECTED_TARGETS}
    invalid_targets: set[str] = set()
    for index, row in enumerate(entries):
        # 1. Inadmissible observations outrank every decision effect for their target.
        reason = _feedback_admissible(row, frame=frame, gtm=gtm, ledger=values, tests=tests,
            evaluated_at=evaluated_at, duplicate=ids[row.get("feedback_id")] > 1 or
            source_keys[(row.get("test_id"), row.get("source_ref"), row.get("affected_target"))] > 1,
            contradictory={"supports", "contradicts"} <= source_effects[(row.get("source_ref"), row.get("affected_target"))])
        targets = (row["affected_target"],) if "affected_target" in row else CHANNEL_FEEDBACK_AFFECTED_TARGETS
        if reason:
            held.append({"feedback_ref": row.get("feedback_id", f"feedback-missing-{index}"), "reason": reason})
            invalid_targets.update(targets)
        else:
            admissible[row["affected_target"]].append(row)
    effects = {}
    for target, rows in admissible.items():
        if target in invalid_targets:
            effects[target] = "unknown"
        elif not rows:
            effects[target] = "unknown"
            held.append({"feedback_ref": f"feedback-missing-{target}", "reason": "channel_feedback_missing"})
        elif any(row["effect"] == "contradicts" for row in rows):
            effects[target] = "contradicts"
        else:
            support = Counter()
            for row in rows:
                support[row["test_id"]] += row["sample_count"]
            complete = all(count >= tests[test_id]["sample_target"] for test_id, count in support.items())
            effects[target] = "supports" if complete else "unknown"
            if not complete:
                held.extend({"feedback_ref": row["feedback_id"], "reason": "channel_feedback_unresolved"} for row in rows)
    # 2. A rejected channel follows only the contradicted GTM test's precommit.
    rejected = []
    followup = "none"
    if effects["channel_reachability"] == "contradicts":
        rejected = [gtm["artifact_id"]]
        failed = [tests[row["test_id"]] for row in admissible["channel_reachability"] if row["effect"] == "contradicts"]
        followup = "reevaluate_discovery" if any(test["failure_decision"] == "kill" for test in failed) else "gtm_pivot"
    # 3. Opportunity contradiction is independent. 4/5. Gate and handoff consume
    # these effects without changing the frame or receipt. 6. Unknown never promotes.
    if "unknown" in effects.values() and followup == "none":
        followup = "bounded_channel_retest"
    input_ref = _artifact(CHANNEL_FEEDBACK_LEDGER_SCHEMA_VERSION, values["discovery_id"],
                          {key: value for key, value in values.items() if key != "discovery_id"}, status="reentered")["artifact_id"]
    return build_channel_feedback_disposition(discovery_id=frame["discovery_id"], frame_ref=frame["artifact_id"],
        gtm_artifact_id=gtm["artifact_id"], portfolio_ref=portfolio["artifact_id"], feedback_ledger_ref=input_ref,
        initial_channel_ref=gtm["initial_channel_ref"], segment_ref=frame["segment_ref"], evaluated_at=now,
        **effects, rejected_channel_hypothesis_refs=rejected, held_observations=held, proposed_followup=followup)


def _feedback_admissible(entry: Mapping[str, Any], *, frame: Mapping[str, Any], gtm: Mapping[str, Any],
        ledger: Mapping[str, Any], tests: Mapping[str, Any], evaluated_at: datetime,
        duplicate: bool, contradictory: bool) -> str | None:
    if set(entry) != CHANNEL_FEEDBACK_ENTRY_KEYS:
        return "channel_feedback_missing"
    test = tests.get(entry["test_id"])
    observed_at = _time(entry["observed_at"])
    if observed_at > evaluated_at or (test and not _time(test["precommitted_at"]) <= observed_at <= _time(test["deadline_at"])):
        return "channel_feedback_stale"
    # A conflicting copy is the specific duplicate refusal, not generic duplication.
    if contradictory:
        return "channel_feedback_contradictory"
    if duplicate:
        return "channel_feedback_duplicate"
    if entry["segment_ref"] != frame["segment_ref"] or ledger["segment_ref"] != frame["segment_ref"]:
        return "channel_feedback_foreign_segment"
    if (entry["channel_ref"] != gtm["initial_channel_ref"] or ledger["initial_channel_ref"] != gtm["initial_channel_ref"]
            or ledger["gtm_artifact_id"] != gtm["artifact_id"]):
        return "channel_feedback_wrong_channel"
    if test is None or (entry["affected_target"] == "channel_reachability" and test["category"] != "go_to_market"):
        return "channel_feedback_wrong_test"
    criteria = {"supports": test["success_criterion_ref"], "contradicts": test["failure_criterion_ref"],
                "unknown": test["inconclusive_criterion_ref"]}
    if entry["criterion_ref"] != criteria[entry["effect"]]:
        return "channel_feedback_wrong_test"
    if (entry["effect"] == "unknown" or entry["source_class"] not in _EXTERNAL_EVIDENCE_CLASSES
            or entry["source_class"] not in test["required_evidence_classes"] or entry["confidence_limit"] != "bounded"):
        return "channel_feedback_unresolved"
    return None


def discovery_audience_gate_with_feedback(frame: Mapping[str, Any], disposition: Mapping[str, Any]) -> dict[str, Any]:
    """Revise only the matching frame's gate; a new frame needs explicit new evaluation."""
    gate = discovery_audience_gate(frame)
    errors = validate_product_discovery_artifact(disposition)
    if errors:
        raise ValueError(errors[0])
    if (disposition["schema_version"] != CHANNEL_FEEDBACK_DISPOSITION_SCHEMA_VERSION
            or disposition["frame_ref"] != frame["artifact_id"]
            or disposition["discovery_id"] != frame["discovery_id"] or disposition["segment_ref"] != frame["segment_ref"]):
        raise ValueError("channel feedback disposition must bind the same decision frame")
    if disposition["channel_reachability"] == "contradicts":
        gate["audience_gate"] = "audience_reachability_contradicted"
        gate["blocked_outputs"] = list(BUILD_BOUNDARY_ROUTES)
    elif disposition["handoff_held"]:
        gate["blocked_outputs"] = list(BUILD_BOUNDARY_ROUTES)
    return gate


def product_brief_consumption_with_feedback(receipt: Mapping[str, Any], disposition: Mapping[str, Any]) -> dict[str, Any]:
    """A receipt cannot bypass supplied channel holds or borrow another segment's support."""
    context = product_brief_consumption(receipt)
    if not context or validate_product_discovery_artifact(disposition):
        return {}
    if (disposition["schema_version"] != CHANNEL_FEEDBACK_DISPOSITION_SCHEMA_VERSION
            or disposition["discovery_id"] != receipt["discovery_id"] or disposition["segment_ref"] != receipt["segment_ref"]
            or disposition["handoff_held"]):
        return {}
    return context


def _assumption_outcome(
    assumption: Mapping[str, Any], entries: Sequence[Mapping[str, Any]], *, evaluated_at: datetime
) -> dict[str, Any]:
    eligible: list[Mapping[str, Any]] = []
    seen_sources: set[str] = set()
    for entry in entries:
        if not _eligible(entry, assumption, evaluated_at=evaluated_at):
            continue
        source_ref = str(entry["source_ref"])
        if source_ref in seen_sources:
            continue
        seen_sources.add(source_ref)
        eligible.append(entry)
    contradiction = any(entry["direction"] == "contradicts" for entry in eligible)
    unresolved = any(entry["direction"] == "unresolved" for entry in eligible)
    support = sum(int(entry["sample_count"]) for entry in eligible if entry["direction"] == "supports")
    deadline = _time(str(assumption["deadline_at"]))
    return {
        "assumption_id": str(assumption["assumption_id"]),
        "contradicted": contradiction,
        "failure_decision": str(assumption["failure_decision"]),
        "timed_out": evaluated_at > deadline and not eligible,
        "validated": not contradiction and not unresolved and support >= int(assumption["sample_target"]),
        "eligible_refs": [str(entry["source_ref"]) for entry in eligible],
    }


def _eligible(entry: Mapping[str, Any], assumption: Mapping[str, Any], *, evaluated_at: datetime) -> bool:
    if str(entry["test_id"]) != str(assumption["test_id"]):
        return False
    if str(entry["segment_ref"]) != str(assumption["scope_segment_ref"]):
        return False
    if str(entry["source_class"]) not in _EXTERNAL_EVIDENCE_CLASSES:
        return False
    if str(entry["source_class"]) not in assumption["required_evidence_classes"]:
        return False
    if not bool(entry["representative"]) or entry["reentry"] != "reentered":
        return False
    if entry["confidence_limit"] != "bounded":
        return False
    if entry["observation_kind"] in {"source_pointer", "prototype_completion"}:
        return False
    criterion_by_direction = {
        "supports": assumption["success_criterion_ref"],
        "contradicts": assumption["failure_criterion_ref"],
        "unresolved": assumption["inconclusive_criterion_ref"],
    }
    if entry["criterion_ref"] != criterion_by_direction[entry["direction"]]:
        return False
    precommitted_at = _time(str(assumption["precommitted_at"]))
    observed_at = _time(str(entry["observed_at"]))
    deadline_at = _time(str(assumption["deadline_at"]))
    return precommitted_at <= observed_at <= deadline_at and observed_at <= evaluated_at


def discovery_audience_gate(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Report the audience-before-build gate for one prepared decision frame.

    Evidence work always continues. A frame alone never permits solution work;
    only a validated decision receipt can. When the audience is undefined the
    gate names the missing audience evidence and the outputs it blocks.
    """
    errors = validate_product_discovery_artifact(frame)
    if errors:
        raise ValueError(errors[0])
    if frame.get("schema_version") != DISCOVERY_DECISION_FRAME_SCHEMA_VERSION:
        raise ValueError("the audience gate reads a discovery decision frame")
    state = str(frame["segment_definition_state"])
    defined = audience_is_defined(state)
    return {
        "schema_version": DISCOVERY_AUDIENCE_GATE_SCHEMA_VERSION,
        "discovery_id": frame["discovery_id"],
        "segment_ref": frame["segment_ref"],
        "segment_definition_state": state,
        "audience_gate": "audience_defined" if defined else "audience_undefined",
        "evidence_work_permitted": True,
        "solution_work_permitted": False,
        "blocked_outputs": [] if defined else list(BUILD_BOUNDARY_ROUTES),
        "missing_audience_evidence_refs": missing_audience_evidence_refs(state),
        "next_route": DISCOVERY_CONTINUATION_ROUTE,
        "claim_boundary": _AUDIENCE_GATE_CLAIM_BOUNDARY,
    }


def product_brief_consumption(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only a validated receipt's compact, transcript-free handoff."""
    if validate_product_discovery_artifact(receipt):
        return {}
    if receipt.get("schema_version") != DISCOVERY_DECISION_RECEIPT_SCHEMA_VERSION:
        return {}
    if receipt.get("decision") != "persevere" or receipt.get("problem_gate") != "validated":
        return {}
    if receipt.get("solution_work_permitted") is not True:
        return {}
    return {
        "problem_ref": receipt["problem_ref"],
        "target_segment_ref": receipt["segment_ref"],
        "mvp_learning_boundary_ref": receipt["precommitted_test_ids"],
        "residual_risk_refs": receipt["residual_risk_refs"],
        "next_route": receipt["next_route"],
    }


def product_discovery_artifact_store_path(paths: OmhPaths) -> Path:
    """Locate the runtime-wide append-only artifact store without a new path primitive."""
    return paths.runtime_journal_dir / PRODUCT_DISCOVERY_STORE_NAME


def append_product_discovery_artifact(paths: OmhPaths, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Append one valid artifact so re-entry survives a process restart."""
    errors = validate_product_discovery_artifact(artifact)
    if errors:
        raise ValueError(errors[0])
    record = dict(artifact)
    path = product_discovery_artifact_store_path(paths)
    with file_lock(path, private=True):
        append_store_line(path, record)
    return record


def read_product_discovery_artifacts(paths: OmhPaths, *, discovery_id: str) -> list[dict[str, Any]]:
    """Read valid artifacts for one discovery identity in append order."""
    records, _ = read_jsonl_objects(product_discovery_artifact_store_path(paths))
    return [
        record
        for record in records
        if record.get("discovery_id") == discovery_id and not validate_product_discovery_artifact(record)
    ]


def _time(value: str) -> datetime:
    return strict_fromisoformat(value.replace("Z", "+00:00"))


def _stamp(value: str, field: str) -> None:
    try:
        parsed = _time(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
