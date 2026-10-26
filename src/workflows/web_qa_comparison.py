"""Pure comparison of re-admitted host-owned web-QA observations.

This module never launches a browser, contacts a host, deploys, or rolls back.
The optional deployment resolver is the sole trusted host boundary for a canary
claim; it must return a closed observation of an already-observed deployment.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import hashlib
import json
import re
from typing import Final, NoReturn

from .web_qa_observation import WebQaObservationError, build_web_qa_observation
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

WEB_QA_COMPARISON_SCHEMA_VERSION: Final = "web_qa_comparison/v1"
HOST_DEPLOYMENT_OBSERVATION_SCHEMA_VERSION: Final = "host_deployment_observation/v1"
_DIGEST: Final = re.compile(r"^[a-f0-9]{64}$")
_REF: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_DEPLOYMENT_EVIDENCE_DIGESTS: Final = 8


class WebQaComparisonError(ValueError):
    """Raised when a trusted deployment resolver returns an invalid value."""


TrustedDeploymentResolver = Callable[[str], object]


def compare_web_qa_observations(
    baseline: object,
    candidate: object,
    *,
    trusted_deployment_resolver: TrustedDeploymentResolver | None = None,
    trusted_trace_resolver: Callable[[str, str, str, tuple[str, ...]], bool] | None = None,
) -> dict[str, object]:
    """Compare two closed ``{plan, receipt}`` envelopes without I/O.

    Persisted verdicts are deliberately ignored: both envelopes are admitted
    again through :func:`build_web_qa_observation`. A canary receives no
    deployment authority from its plan or receipt; it needs a successful,
    matching observation returned by the explicitly supplied trusted resolver.
    """
    admitted_baseline, baseline_error = _admit_envelope(baseline, trusted_trace_resolver)
    admitted_candidate, candidate_error = _admit_envelope(candidate, trusted_trace_resolver)
    if admitted_baseline is None or admitted_candidate is None:
        return _inadmissible_result(admitted_baseline, admitted_candidate, baseline_error, candidate_error)

    differences = _condition_differences(admitted_baseline["condition"], admitted_candidate["condition"])
    if differences:
        return {
            **_core(admitted_baseline, admitted_candidate, differences),
            **_authority_contract(),
            "status": "not_comparable",
            "condition_differences": differences,
            "cells": [],
            "verdict": "BLOCK",
            "blockers": ["condition_digest_mismatch"],
            "revision_reasons": [],
            "recommendation": "Collect a new baseline under the candidate's exact condition.",
        }

    blockers: list[str] = []
    revisions: list[str] = []
    cells = _cell_comparisons(admitted_baseline, admitted_candidate, revisions)
    _baseline_integrity(admitted_baseline, blockers)
    _visual_round_requirements(admitted_baseline, admitted_candidate, blockers)
    if admitted_candidate["verdict"] == "BLOCK":
        blockers.append("candidate_observation_blocked")
    elif admitted_candidate["verdict"] == "REVISE":
        revisions.append("candidate_observation_requires_revision")

    deployment_observation: dict[str, object] | None = None
    if admitted_candidate["mode"] == "canary":
        deployment_observation = _canary_requirements(
            admitted_baseline,
            admitted_candidate,
            trusted_deployment_resolver,
            blockers,
        )
    elif admitted_baseline["mode"] == "canary":
        blockers.append("canary_baseline_requires_canary_candidate")

    verdict = "BLOCK" if blockers else "REVISE" if revisions else "PASS"
    return {
        **_core(admitted_baseline, admitted_candidate, differences, deployment_observation),
        **_authority_contract(),
        "status": "comparable",
        "condition_differences": [],
        "cells": cells,
        "verdict": verdict,
        "blockers": sorted(set(blockers)),
        "revision_reasons": sorted(set(revisions)),
        "recommendation": _recommendation(verdict, admitted_candidate["mode"]),
    }


def _admit_envelope(
    value: object,
    trace_resolver: Callable[[str, str, str, tuple[str, ...]], bool] | None,
) -> tuple[dict[str, object] | None, str]:
    if type(value) is not dict or set(value) != {"plan", "receipt"}:
        return None, "envelope_must_contain_exactly_plan_and_receipt"
    try:
        return build_web_qa_observation(
            value["plan"], value["receipt"], trusted_trace_resolver=trace_resolver
        ), ""
    except (WebQaObservationError, TypeError, ValueError):
        return None, "admission_rejected"


def _inadmissible_result(
    baseline: dict[str, object] | None,
    candidate: dict[str, object] | None,
    baseline_error: str,
    candidate_error: str,
) -> dict[str, object]:
    blockers = []
    if baseline is None:
        blockers.append(f"baseline_{baseline_error}")
    if candidate is None:
        blockers.append(f"candidate_{candidate_error}")
    known = candidate or baseline or {}
    return {
        "schema_version": WEB_QA_COMPARISON_SCHEMA_VERSION,
        "comparison_id": _digest({"baseline_error": baseline_error, "candidate_error": candidate_error}),
        **_authority_contract(),
        "mode": known.get("mode", ""),
        "round": known.get("round", {}),
        "limits": known.get("limits", {}),
        "subject": {"baseline": baseline.get("subject", {}) if baseline else {}, "candidate": candidate.get("subject", {}) if candidate else {}},
        "condition": known.get("condition", {}),
        "observation_window": {"baseline": baseline.get("observation_window", {}) if baseline else {}, "candidate": candidate.get("observation_window", {}) if candidate else {}},
        "baseline": _reference(baseline),
        "candidate": _reference(candidate),
        "status": "not_comparable",
        "condition_differences": [],
        "cells": [],
        "verdict": "BLOCK",
        "blockers": sorted(blockers),
        "revision_reasons": [],
        "recommendation": "Repair the invalid observation envelope and collect evidence again; do not execute or roll back from this comparison.",
    }


def _core(
    baseline: dict[str, object],
    candidate: dict[str, object],
    differences: list[dict[str, object]],
    deployment_observation: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = {
        "baseline_observation_digest": _digest(baseline),
        "candidate_observation_digest": _digest(candidate),
        "deployment_observation_digest": _digest(deployment_observation) if deployment_observation else "",
        "condition_differences": differences,
    }
    return {
        "schema_version": WEB_QA_COMPARISON_SCHEMA_VERSION,
        "comparison_id": _digest(identity),
        "mode": candidate["mode"],
        "round": {"baseline": baseline["round"], "candidate": candidate["round"]},
        "limits": candidate["limits"],
        "subject": {"baseline": baseline["subject"], "candidate": candidate["subject"]},
        "condition": candidate["condition"],
        "observation_window": {"baseline": baseline["observation_window"], "candidate": candidate["observation_window"]},
        "baseline": _reference(baseline),
        "candidate": _reference(candidate),
    }


def _reference(observation: dict[str, object] | None) -> dict[str, object]:
    if observation is None:
        return {}
    return {
        "run_id": observation["run_id"],
        "plan_digest": observation["plan_digest"],
        "subject_digest": observation["subject_digest"],
        "condition_digest": observation["condition_digest"],
    }


def _condition_differences(baseline: object, candidate: object, path: str = "condition") -> list[dict[str, object]]:
    if type(baseline) is dict and type(candidate) is dict:
        differences: list[dict[str, object]] = []
        for key in sorted(set(baseline) | set(candidate)):
            child = f"{path}.{key}"
            if key not in baseline or key not in candidate:
                differences.append({"field": child, "baseline": baseline.get(key), "candidate": candidate.get(key)})
            else:
                differences.extend(_condition_differences(baseline[key], candidate[key], child))
        return differences
    if type(baseline) is list and type(candidate) is list:
        differences = []
        for index in range(max(len(baseline), len(candidate))):
            child = f"{path}[{index}]"
            if index >= len(baseline) or index >= len(candidate):
                differences.append({"field": child, "baseline": baseline[index] if index < len(baseline) else None, "candidate": candidate[index] if index < len(candidate) else None})
            else:
                differences.extend(_condition_differences(baseline[index], candidate[index], child))
        return differences
    return [] if baseline == candidate else [{"field": path, "baseline": baseline, "candidate": candidate}]


def _cell_comparisons(baseline: dict[str, object], candidate: dict[str, object], revisions: list[str]) -> list[dict[str, object]]:
    previous = {str(cell["cell_id"]): cell for cell in _objects(baseline["cells"], "baseline cells")}
    current = {str(cell["cell_id"]): cell for cell in _objects(candidate["cells"], "candidate cells")}
    cells: list[dict[str, object]] = []
    for cell_id in sorted(set(previous) | set(current)):
        old = previous.get(cell_id)
        new = current.get(cell_id)
        if old is None or new is None:
            revisions.append(f"{cell_id}:matrix_cell_changed")
            cells.append({"cell_id": cell_id, "status": "not_comparable"})
            continue
        performance = _performance_regressions(old, new, candidate)
        revisions.extend(f"{cell_id}:{reason}" for reason in performance)
        cells.append(
            {
                "cell_id": cell_id,
                "status": "compared",
                "baseline_capture_sha256": _capture_digest(old),
                "candidate_capture_sha256": _capture_digest(new),
                "performance_regressions": performance,
            }
        )
    return cells


def _performance_regressions(old: dict[str, object], new: dict[str, object], candidate: dict[str, object]) -> list[str]:
    old_field = _field_metrics(old)
    new_field = _field_metrics(new)
    if old_field is None or new_field is None:
        return []
    budgets = _object(_object(candidate["condition"], "condition")["budgets"], "budgets")
    tolerances = _object(_object(budgets["performance"], "performance")["relative_tolerances"], "tolerances")
    regressions: list[str] = []
    if new_field["lcp_ms"] > old_field["lcp_ms"] * (1 + _number(tolerances["lcp_percent"]) / 100):
        regressions.append("field_lcp_regression_beyond_tolerance")
    if new_field["inp_ms"] > old_field["inp_ms"] * (1 + _number(tolerances["inp_percent"]) / 100):
        regressions.append("field_inp_regression_beyond_tolerance")
    if new_field["cls"] > old_field["cls"] + _number(tolerances["cls_absolute"]):
        regressions.append("field_cls_regression_beyond_tolerance")
    return regressions


def _field_metrics(cell: dict[str, object]) -> dict[str, int | float] | None:
    performance = _object(_object(cell["channels"], "channels")["performance"], "performance channel")
    if performance["status"] != "observed":
        return None
    evidence = _object(performance["evidence"], "performance evidence")
    if evidence["evidence_class"] != "field" or evidence["field"] is None:
        return None
    metrics = _object(_object(evidence["field"], "field evidence")["p75"], "field p75")
    return {
        "lcp_ms": _number(metrics["lcp_ms"]),
        "inp_ms": _number(metrics["inp_ms"]),
        "cls": _number(metrics["cls"]),
    }


def _baseline_integrity(baseline: dict[str, object], blockers: list[str]) -> None:
    non_visual = [reason for reason in _strings(baseline["blockers"], "baseline blockers") if not reason.endswith("visual_score_below_minimum_requires_fresh_capture")]
    if non_visual:
        blockers.append("baseline_observation_blocked")


def _visual_round_requirements(baseline: dict[str, object], candidate: dict[str, object], blockers: list[str]) -> None:
    failed = [
        cell
        for cell in _objects(baseline["cells"], "baseline cells")
        if (score := _visual_score(cell)) is not None and score < 90
    ]
    if not failed:
        return
    old_subject = _object(baseline["subject"], "baseline subject")
    new_subject = _object(candidate["subject"], "candidate subject")
    old_round = _object(baseline["round"], "baseline round")
    new_round = _object(candidate["round"], "candidate round")
    if old_subject["revision"] == new_subject["revision"]:
        blockers.append("visual_retest_requires_source_revision_change")
    if _integer(new_round["ordinal"]) != _integer(old_round["ordinal"]) + 1:
        blockers.append("visual_retest_requires_incremented_round")
    if _integer(new_round["ordinal"]) > _integer(_object(candidate["limits"], "candidate limits")["max_rounds"]):
        blockers.append("visual_retest_round_cap_exhausted")
    candidate_cells = {str(cell["cell_id"]): cell for cell in _objects(candidate["cells"], "candidate cells")}
    for old_cell in failed:
        cell_id = str(old_cell["cell_id"])
        new_cell = candidate_cells.get(cell_id)
        if new_cell is None:
            blockers.append(f"{cell_id}:visual_retest_cell_missing")
            continue
        if _capture_digest(old_cell) == _capture_digest(new_cell):
            blockers.append(f"{cell_id}:visual_retest_requires_changed_capture")
        old_capture = _capture_time(old_cell)
        new_capture = _capture_time(new_cell)
        if old_capture is None or new_capture is None or new_capture <= old_capture:
            blockers.append(f"{cell_id}:visual_retest_requires_newer_capture")


def _canary_requirements(
    baseline: dict[str, object],
    candidate: dict[str, object],
    resolver: TrustedDeploymentResolver | None,
    blockers: list[str],
) -> dict[str, object] | None:
    subject = _object(candidate["subject"], "candidate subject")
    deployment = _object(subject["deployment"], "candidate deployment")
    if resolver is None:
        blockers.append("canary_trusted_deployment_resolver_missing")
        return None
    try:
        deployment_observation = _deployment_observation(resolver(str(subject["observed_deploy_ref"])))
    except (WebQaComparisonError, TypeError, ValueError):
        blockers.append("canary_deployment_receipt_unresolved")
        return None
    except Exception:
        blockers.append("canary_deployment_resolver_failed")
        return None
    expected = {
        "deployment_ref": subject["observed_deploy_ref"],
        "repository": subject["repository"],
        "revision": subject["revision"],
        "deployment_id": deployment["deployment_id"],
        "environment": deployment["environment"],
        "rollout_phase": deployment["rollout_phase"],
    }
    if deployment_observation["status"] != "succeeded":
        blockers.append("canary_deployment_not_succeeded")
    if any(deployment_observation[key] != value for key, value in expected.items()):
        blockers.append("canary_deployment_binding_mismatch")
    planned_window = _object(deployment["window"], "candidate deployment window")
    if deployment_observation["window"] != planned_window:
        blockers.append("canary_deployment_window_mismatch")
    starts = _time(planned_window["starts_at"])
    ends = _time(planned_window["ends_at"])
    deployed_at = _time(deployment_observation["observed_at"])
    if deployed_at > starts:
        blockers.append("canary_deployment_observed_after_canary_window_start")
    candidate_window = _object(candidate["observation_window"], "candidate observation window")
    if _time(candidate_window["started_at"]) < starts or _time(candidate_window["ended_at"]) > ends:
        blockers.append("canary_candidate_observation_outside_deployment_window")
    for cell in _objects(candidate["cells"], "candidate cells"):
        captured = _capture_time(cell)
        if captured is None:
            blockers.append(f"{cell['cell_id']}:canary_capture_not_observed")
        elif captured < starts or captured > ends:
            blockers.append(f"{cell['cell_id']}:canary_capture_outside_deployment_window")
    baseline_window = _object(baseline["observation_window"], "baseline observation window")
    if _time(baseline_window["started_at"]) >= deployed_at or _time(baseline_window["ended_at"]) >= deployed_at:
        blockers.append("canary_baseline_observation_not_before_deployment")
    baseline_condition = _object(baseline["condition"], "baseline condition")
    if baseline_condition["environment"] != "production":
        blockers.append("canary_baseline_is_not_production")
    for cell in _objects(baseline["cells"], "baseline cells"):
        captured = _capture_time(cell)
        if captured is None:
            blockers.append(f"{cell['cell_id']}:canary_baseline_capture_not_observed")
        elif captured >= deployed_at:
            blockers.append(f"{cell['cell_id']}:canary_baseline_not_before_deployment")
    return deployment_observation


def _deployment_observation(value: object) -> dict[str, object]:
    keys = {
        "schema_version", "deployment_ref", "repository", "revision", "deployment_id", "environment", "rollout_phase",
        "status", "window", "observed_at", "observation_source_ref", "evidence_digests",
    }
    result = _object(value, "deployment observation")
    if set(result) != keys or result["schema_version"] != HOST_DEPLOYMENT_OBSERVATION_SCHEMA_VERSION:
        _fail("deployment observation must have the closed deployment-observation schema")
    if result["status"] not in ("succeeded", "failed", "unknown", "prepared"):
        _fail("deployment observation status is invalid")
    for key in ("deployment_ref", "repository", "revision", "deployment_id", "environment", "rollout_phase"):
        if not isinstance(result[key], str) or not result[key]:
            _fail(f"deployment observation {key} is invalid")
    if not isinstance(result["observation_source_ref"], str) or not _REF.fullmatch(result["observation_source_ref"]):
        _fail("deployment observation source reference is invalid")
    observed_at = _utc_timestamp(result["observed_at"])
    window = _object(result["window"], "deployment observation window")
    if set(window) != {"starts_at", "ends_at", "duration_seconds"}:
        _fail("deployment observation window is invalid")
    starts_at = _utc_timestamp(window["starts_at"])
    ends_at = _utc_timestamp(window["ends_at"])
    starts, ends = _time(starts_at), _time(ends_at)
    if (
        ends <= starts
        or type(window["duration_seconds"]) is not int
        or window["duration_seconds"] != int((ends - starts).total_seconds())
    ):
        _fail("deployment observation window is invalid")
    digests = result["evidence_digests"]
    if (
        type(digests) is not list
        or not digests
        or len(digests) > _MAX_DEPLOYMENT_EVIDENCE_DIGESTS
        or not all(isinstance(item, str) and _DIGEST.fullmatch(item) for item in digests)
    ):
        _fail("deployment observation evidence digests are invalid")
    return {
        **result,
        "observed_at": observed_at,
        "window": {"starts_at": starts_at, "ends_at": ends_at, "duration_seconds": window["duration_seconds"]},
        "evidence_digests": list(digests),
    }


def _visual_score(cell: dict[str, object]) -> int | float | None:
    screenshot = _object(_object(cell["channels"], "channels")["screenshot"], "screenshot")
    if screenshot["status"] != "observed":
        return None
    evidence = _object(screenshot["evidence"], "screenshot evidence")
    review = evidence.get("review")
    if type(review) is not dict or "score" not in review:
        return None
    return _number(review["score"])


def _capture_digest(cell: dict[str, object]) -> str:
    screenshot = _object(_object(cell["channels"], "channels")["screenshot"], "screenshot")
    if screenshot["status"] != "observed":
        return ""
    evidence = _object(screenshot["evidence"], "screenshot evidence")
    return str(evidence.get("capture_sha256", ""))


def _capture_time(cell: dict[str, object]) -> datetime | None:
    screenshot = _object(_object(cell["channels"], "channels")["screenshot"], "screenshot")
    if screenshot["status"] != "observed":
        return None
    evidence = _object(screenshot["evidence"], "screenshot evidence")
    captured = evidence.get("captured_at")
    return _time(captured) if captured is not None else None


def _authority_contract() -> dict[str, object]:
    return {
        "does_not_authorize": ("execution", "deployment", "rollback"),
        "rollback_authorized": False,
    }


def _recommendation(verdict: str, mode: object) -> str:
    if verdict == "PASS":
        return "Comparison passed; this is observation only and does not authorize execution, deployment, or rollback."
    if verdict == "REVISE":
        return "Revise the candidate and collect a fresh same-condition observation; do not execute or roll back from this comparison."
    if mode == "canary":
        return "Hand off the observed canary result to the existing deploy-and-monitor decision gate; this comparison does not authorize execution or rollback."
    return "Resolve the blockers and collect fresh evidence; this comparison does not authorize execution, deployment, or rollback."


def _object(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict:
        _fail(f"{field} must be an object")
    return value


def _objects(value: object, field: str) -> list[dict[str, object]]:
    if type(value) is not list or not all(type(item) is dict for item in value):
        _fail(f"{field} must be an object list")
    return value


def _strings(value: object, field: str) -> list[str]:
    if type(value) is not list or not all(isinstance(item, str) for item in value):
        _fail(f"{field} must be a string list")
    return value


def _integer(value: object) -> int:
    if type(value) is not int:
        _fail("value must be an integer")
    return value


def _number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("value must be numeric")
    return value


def _utc_timestamp(value: object) -> str:
    return _time(value).isoformat(timespec="seconds").replace("+00:00", "Z")


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        _fail("timestamp must be UTC")
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WebQaComparisonError("timestamp must be UTC") from exc
    if parsed.tzinfo is None:
        _fail("timestamp must have an offset")
    return parsed.astimezone(UTC)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def _fail(message: str) -> NoReturn:
    raise WebQaComparisonError(message)
