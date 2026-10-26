"""Pure closed admission and verdict logic for host-owned web-QA receipts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
import hashlib
import json
import math
import re
from typing import Final, NoReturn
from urllib.parse import urlsplit

from .web_qa_observation_plan import REQUIRED_CHANNELS, WebQaObservationPlanError, parse_normalized_web_qa_observation_plan
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

WEB_QA_OBSERVATION_SCHEMA_VERSION: Final = "web_qa_observation_run/v1"
HOST_RECEIPT_SCHEMA_VERSION: Final = "host_web_qa_adapter_receipt/v1"
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CHANNEL_STATES = ("observed", "not_observed", "blocked")
_ACTIONS = ("navigate", "read", "click", "extract", "submit")
_TRANSIENT = ("navigation", "timeout", "transport", "browser_crash")
_TERMINAL = ("auth", "http_4xx", "assertion", "regression", "mutation", "success")
_READ_ONLY_ACTIONS = ("navigate", "read", "extract")
_OPERATION_BINDINGS: Final[dict[str, tuple[str, ...]]] = {
    "screenshot": ("screenshot",), "console": ("console", "errors"),
    "network": ("network_requests",), "critical_flow": ("navigate",),
    "accessibility": ("accessibility_audit",), "keyboard": ("keyboard_tab", "focus_before", "focus_after"),
    "performance": ("lab_vitals", "field_vitals"),
}


class WebQaObservationError(ValueError):
    """The supplied object cannot be admitted as a closed host receipt."""


TrustedTraceResolver = Callable[[str, str, str, tuple[str, ...]], bool]


def build_web_qa_observation(
    plan: object,
    sanitized_host_receipt: object,
    *,
    trusted_trace_resolver: TrustedTraceResolver | None = None,
) -> dict[str, object]:
    """Return a closed `web_qa_observation_run/v1` without performing I/O.

    A resolver is an explicit trusted boundary supplied by the caller. A trace
    shaped dictionary in a receipt is never reusable evidence by itself.
    """
    normalized_plan = _normalize_plan(plan)
    receipt = _receipt(sanitized_host_receipt)
    for key in ("run_id", "subject_digest", "condition_digest"):
        if receipt[key] != normalized_plan[key]:
            _fail(f"receipt {key} does not match normalized plan")
    command_digests = _commands(receipt["execution"], _integer(_object(normalized_plan["limits"], "plan limits")["max_step_seconds"], "max step seconds"))
    window = _window(receipt["execution"], "execution")
    blockers: list[str] = []
    revisions: list[str] = []
    _execution_caps(normalized_plan, receipt["execution"], blockers)
    if _object(receipt["execution"], "execution")["session_close_observed"] is not True:
        blockers.append("session_cleanup_not_observed")
    if receipt["release_status"] != "COLLECTED" or receipt["release_blockers"]:
        blockers.append("host_receipt_not_collected")

    planned = {str(cell["cell_id"]): cell for cell in _objects(normalized_plan["matrix"], "normalized matrix")}
    supplied: dict[str, dict[str, object]] = {}
    for cell in _objects(receipt["cells"], "receipt.cells"):
        cell_id = _ref(cell.get("cell_id"), "receipt.cells[].cell_id")
        if cell_id in supplied:
            _fail("receipt.cells contains duplicate cell_id")
        if cell_id not in planned:
            _fail("receipt.cells contains unplanned cell_id")
        supplied[cell_id] = cell

    output_cells: list[dict[str, object]] = []
    for cell_id, expected in sorted(planned.items()):
        candidate = supplied.get(cell_id)
        if candidate is None:
            blockers.append(f"{cell_id}:host_cell_missing")
            output_cells.append(_missing_cell(expected))
            continue
        output, cell_blockers, cell_revisions = _admit_cell(
            normalized_plan, expected, candidate, window, command_digests, trusted_trace_resolver,
        )
        output_cells.append(output)
        blockers.extend(f"{cell_id}:{reason}" for reason in cell_blockers)
        revisions.extend(f"{cell_id}:{reason}" for reason in cell_revisions)

    return {
        "schema_version": WEB_QA_OBSERVATION_SCHEMA_VERSION,
        "run_id": normalized_plan["run_id"],
        "mode": normalized_plan["mode"],
        "round": normalized_plan["round"],
        "limits": normalized_plan["limits"],
        "plan_digest": normalized_plan["plan_digest"],
        "subject_digest": normalized_plan["subject_digest"],
        "condition_digest": normalized_plan["condition_digest"],
        "subject": normalized_plan["subject"],
        "condition": normalized_plan["condition"],
        "adapter": receipt["adapter"],
        "observation_window": window,
        "cells": output_cells,
        "verdict": "BLOCK" if blockers else "REVISE" if revisions else "PASS",
        "blockers": sorted(set(blockers)),
        "revision_reasons": sorted(set(revisions)),
        "does_not_authorize": ("browser_launch", "network_request", "deployment", "rollback", "storage_write"),
    }


def _normalize_plan(value: object) -> dict[str, object]:
    try:
        return parse_normalized_web_qa_observation_plan(value)
    except WebQaObservationPlanError as exc:
        raise WebQaObservationError(str(exc)) from exc


def _receipt(value: object) -> dict[str, object]:
    receipt = _closed(_snapshot(value), (
        "schema_version", "receipt_version", "run_id", "subject_digest", "condition_digest", "adapter", "execution",
        "cells", "release_status", "release_blockers", "redaction",
    ), "receipt")
    if receipt["schema_version"] != HOST_RECEIPT_SCHEMA_VERSION or receipt["receipt_version"] != 1:
        _fail("receipt schema is invalid")
    _closed(receipt["adapter"], ("adapter_id", "session_id_digest"), "receipt adapter")
    _adapter_id(_object(receipt["adapter"], "adapter")["adapter_id"])
    _digest_value(_object(receipt["adapter"], "adapter")["session_id_digest"], "session_id_digest")
    _closed(receipt["execution"], ("command_results", "artifact_bytes", "session_close_observed", "attempts", "cost_units", "peak_concurrency", "started_at", "ended_at"), "receipt execution")
    if receipt["release_status"] not in ("COLLECTED", "BLOCK") or type(receipt["release_blockers"]) is not list or any(not _valid_ref(item) for item in receipt["release_blockers"]):
        _fail("receipt release status is invalid")
    redaction = _closed(receipt["redaction"], ("status", "forbidden"), "receipt redaction")
    forbidden = ("headers", "cookies", "credentials", "query_values", "request_bodies", "response_bodies", "console_text", "accessibility_html")
    if redaction["status"] != "redacted_before_persistence" or type(redaction["forbidden"]) is not list or any(item not in forbidden for item in redaction["forbidden"]):
        _fail("receipt redaction is invalid")
    return receipt


def _commands(execution: object, max_step_seconds: int) -> dict[str, tuple[str, bool, int, str]]:
    source = _object(execution, "execution")
    commands = _objects(source["command_results"], "command_results")
    if not commands or len(commands) > 1024:
        _fail("command_results is missing or exceeds bound")
    output: dict[str, tuple[str, bool, int, str]] = {}
    for command in commands:
        item = _closed(command, ("cell_id", "operation", "success", "returncode", "duration_ms", "stdout_sha256", "stderr_sha256", "stdout_bytes", "stderr_bytes"), "command")
        _ref(item["operation"], "command operation"); _boolean(item["success"], "command success")
        _integer(item["returncode"], "command returncode")
        if _non_negative(item["duration_ms"], "command duration") > max_step_seconds * 1000:
            _fail("command duration exceeds max_step_seconds")
        _digest_value(item["stdout_sha256"], "command stdout"); _digest_value(item["stderr_sha256"], "command stderr")
        _non_negative(item["stdout_bytes"], "command stdout bytes"); _non_negative(item["stderr_bytes"], "command stderr bytes")
        output[_digest(item)] = (str(item["operation"]), _boolean(item["success"], "command success"), _integer(item["returncode"], "command returncode"), _ref(item["cell_id"], "command cell_id"))
    return output


def _execution_caps(plan: dict[str, object], execution: object, blockers: list[str]) -> None:
    source = _object(execution, "execution"); limits = _object(plan["limits"], "plan limits")
    _non_negative(source["attempts"], "execution attempts")
    _non_negative(source["artifact_bytes"], "execution artifact_bytes")
    _non_negative(source["peak_concurrency"], "execution peak_concurrency")
    if _integer(source["peak_concurrency"], "execution peak_concurrency") > _integer(limits["max_concurrency"], "max concurrency"):
        blockers.append("concurrency_cap_exhausted")
    if _integer(source["attempts"], "execution attempts") > _integer(limits["max_attempts_per_read"], "max attempts"):
        blockers.append("attempt_cap_exhausted")
    if _integer(source["artifact_bytes"], "execution artifact_bytes") > _integer(limits["max_artifact_bytes"], "max artifact bytes"):
        blockers.append("artifact_cap_exhausted")
    _non_negative(source["cost_units"], "execution cost_units")
    if _integer(source["cost_units"], "execution cost_units") > _integer(limits["max_cost_units"], "max cost units"):
        blockers.append("cost_cap_exhausted")
    if (_parse(_utc(source["ended_at"], "execution ended_at")) - _parse(_utc(source["started_at"], "execution started_at"))).total_seconds() > _integer(limits["max_run_seconds"], "max run seconds"):
        blockers.append("run_deadline_exhausted")


def _admit_cell(plan: dict[str, object], expected: dict[str, object], value: dict[str, object], window: dict[str, str], commands: dict[str, tuple[str, bool, int, str]], resolver: TrustedTraceResolver | None) -> tuple[dict[str, object], list[str], list[str]]:
    cell = _closed(value, ("cell_id", "route_id", "state_id", "terminal_status", "terminal_blocker_id", "actual", "channels"), "receipt cell")
    for key in ("cell_id", "route_id", "state_id"):
        if cell[key] != expected[key]:
            _fail(f"receipt cell {key} does not match plan")
    if cell["terminal_status"] not in _CHANNEL_STATES or not isinstance(cell["terminal_blocker_id"], str) or (cell["terminal_status"] == "observed") != (cell["terminal_blocker_id"] == ""):
        _fail("receipt terminal status is invalid")
    if cell["terminal_status"] == "observed":
        _actual_conditions(plan, expected, cell["actual"])
    elif cell["actual"] is not None:
        _fail("unobserved cell actual must be null")
    raw_channels = _closed(cell["channels"], REQUIRED_CHANNELS, "receipt channels")
    blockers: list[str] = []
    revisions: list[str] = []
    normalized: dict[str, object] = {}
    for name in REQUIRED_CHANNELS:
        channel = _channel(raw_channels[name], name, str(expected["cell_id"]), commands)
        normalized[name] = channel
        if channel["status"] != "observed":
            blockers.append(f"{name}:{channel['blocker_id']}")
    if cell["terminal_status"] != "observed":
        revisions.append("terminal_not_observed")
    screenshot = _object(normalized["screenshot"], "normalized screenshot")
    console = _object(normalized["console"], "normalized console")
    network = _object(normalized["network"], "normalized network")
    flow = _object(normalized["critical_flow"], "normalized critical flow")
    accessibility = _object(normalized["accessibility"], "normalized accessibility")
    keyboard = _object(normalized["keyboard"], "normalized keyboard")
    performance = _object(normalized["performance"], "normalized performance")
    if screenshot["status"] == "observed":
        _screenshot(plan, screenshot["evidence"], window, blockers)
    if console["status"] == "observed":
        _console(plan, console["evidence"], revisions)
    if network["status"] == "observed":
        _network(plan, network["evidence"], revisions)
    if flow["status"] == "observed":
        _flow(plan, expected, flow["evidence"], str(expected["cell_id"]), commands, resolver, blockers, revisions)
    if accessibility["status"] == "observed":
        _a11y(accessibility["evidence"], revisions)
    if keyboard["status"] == "observed":
        _keyboard(keyboard["evidence"], revisions)
    if performance["status"] == "observed":
        _performance(plan, performance["evidence"], blockers, revisions)
    return ({"cell_id": expected["cell_id"], "route_id": expected["route_id"], "state_id": expected["state_id"], "expected_terminal": expected["expected_terminal"], "terminal_status": cell["terminal_status"], "terminal_blocker_id": cell["terminal_blocker_id"], "actual": cell["actual"], "channels": normalized}, blockers, revisions)


def _actual_conditions(plan: dict[str, object], expected: dict[str, object], value: object) -> None:
    actual = _closed(value, ("origin", "navigation_digest", "viewport_id", "width", "height", "dpr", "browser_id", "engine", "version", "locale", "timezone", "profiles", "auth_fixture_ref", "terminal_state"), "actual conditions")
    condition = _object(plan["condition"], "condition")
    route = next(item for item in _objects(condition["routes"], "routes") if item["route_digest"] == expected["route_digest"])
    viewport = next(item for item in _objects(condition["viewports"], "viewports") if item["viewport_id"] == expected["viewport_id"])
    browser = next(item for item in _objects(condition["browsers"], "browsers") if item["browser_id"] == expected["browser_id"])
    expected_actual = {
        "origin": route["origin"], "navigation_digest": route["navigation_digest"], "viewport_id": viewport["viewport_id"],
        "width": viewport["width"], "height": viewport["height"], "dpr": viewport["dpr"], "browser_id": browser["browser_id"],
        "engine": browser["engine"], "version": browser["version"], "locale": condition["locale"], "timezone": condition["timezone"],
        "profiles": condition["profiles"], "auth_fixture_ref": condition["auth_fixture_ref"], "terminal_state": expected["expected_terminal"],
    }
    if actual != expected_actual:
        _fail("actual conditions do not match the planned cell")


def _channel(value: object, name: str, cell_id: str, commands: dict[str, tuple[str, bool, int, str]]) -> dict[str, object]:
    channel = _closed(value, ("status", "blocker_id", "evidence"), f"{name} channel")
    state = channel["status"]
    if state not in _CHANNEL_STATES or not isinstance(channel["blocker_id"], str):
        _fail(f"{name} channel state is invalid")
    evidence = _object(channel["evidence"], f"{name} evidence")
    if state == "observed":
        if channel["blocker_id"]:
            _fail(f"{name} observed channel cannot have blocker")
        bound = _strings(evidence.get("operation_digests"), f"{name} operation_digests")
        allowed_operations = _OPERATION_BINDINGS[name]
        if name == "performance":
            allowed_operations = ("field_vitals",) if evidence.get("evidence_class") == "field" else ("lab_vitals",)
        if not bound or any(item not in commands or commands[item][0] not in allowed_operations or commands[item][1] is not True or commands[item][2] != 0 or commands[item][3] != cell_id for item in bound):
            _fail(f"{name} observed evidence lacks a successful matching host operation binding")
        observed_operations = {commands[item][0] for item in bound}
        if name == "console" and not {"console", "errors"} <= observed_operations:
            _fail("console evidence requires console and errors operations")
        if name == "keyboard" and not {"keyboard_tab", "focus_before", "focus_after"} <= observed_operations:
            _fail("keyboard evidence requires keypress and focus observations")
    elif not channel["blocker_id"]:
        _fail(f"{name} missing channel requires named blocker")
    elif evidence:
        _fail(f"{name} missing evidence must be empty")
    return {"status": state, "blocker_id": channel["blocker_id"], "evidence": evidence}


def _screenshot(plan: dict[str, object], value: object, window: dict[str, str], blockers: list[str]) -> None:
    evidence = _closed(value, ("capture_sha256", "byte_size", "captured_at", "review", "operation_digests"), "screenshot evidence")
    capture = _digest_value(evidence["capture_sha256"], "capture")
    _positive(evidence["byte_size"], "capture byte_size")
    captured = _utc(evidence["captured_at"], "captured_at")
    if not _parse(window["started_at"]) <= _parse(captured) <= _parse(window["ended_at"]):
        blockers.append("stale_or_out_of_window_capture")
    if evidence["review"] is None:
        blockers.append("screenshot_review_missing")
        return
    review = _closed(evidence["review"], ("reviewer_id", "rubric_ref", "evidence_ref", "capture_sha256", "subject_digest", "condition_digest", "score"), "screenshot review")
    for key in ("reviewer_id", "rubric_ref", "evidence_ref"):
        _ref(review[key], f"review {key}")
    if review["capture_sha256"] != capture or review["subject_digest"] != plan["subject_digest"] or review["condition_digest"] != plan["condition_digest"]:
        blockers.append("screenshot_review_lineage_mismatch")
    budgets = _object(_object(plan["condition"], "condition")["budgets"], "budgets")
    minimum = _integer(_object(budgets["visual"], "visual budget")["minimum_score"], "minimum score")
    score = _number(review["score"], "review score")
    if score > 100:
        _fail("review score exceeds rubric maximum")
    if score < minimum:
        blockers.append("visual_score_below_minimum_requires_fresh_capture")


def _console(plan: dict[str, object], value: object, revisions: list[str]) -> None:
    evidence = _closed(value, ("exceptions", "operation_digests"), "console evidence")
    allowed = {item["noise_id"] for item in _objects(_object(plan["condition"], "condition")["expected_noise_allowlist"], "allowlist") if item["channel"] == "console"}
    for item in _bounded_objects(evidence["exceptions"], "console exceptions", 128):
        entry = _closed(item, ("exception_id", "classification", "allowlist_id"), "console exception")
        _ref(entry["exception_id"], "exception_id")
        if entry["classification"] != "allowlisted" or entry["allowlist_id"] not in allowed:
            revisions.append("unallowlisted_console_exception")


def _network(plan: dict[str, object], value: object, revisions: list[str]) -> None:
    evidence = _closed(value, ("requests", "operation_digests"), "network evidence")
    origins = {item["origin"] for item in _objects(_object(plan["condition"], "condition")["routes"], "routes")}
    allowed = {item["noise_id"] for item in _objects(_object(plan["condition"], "condition")["expected_noise_allowlist"], "allowlist") if item["channel"] == "network"}
    for item in _bounded_objects(evidence["requests"], "network requests", 128):
        request = _closed(item, ("origin", "path_digest", "method", "status", "duration_ms", "classification", "allowlist_id"), "network request")
        origin = _origin(request["origin"], "network origin")
        if origin != request["origin"]:
            _fail("network origin must be canonical")
        _digest_value(request["path_digest"], "network path_digest")
        status = _integer(request["status"], "network status")
        duration = _non_negative(request["duration_ms"], "network duration_ms")
        if request["method"] not in ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE") or request["classification"] not in ("success", "failure", "allowlisted", "transport_failure") or not isinstance(request["allowlist_id"], str):
            _fail("network request is invalid")
        if status == 0:
            if request["classification"] != "transport_failure":
                _fail("status zero must be transport_failure")
        elif not 100 <= status <= 599:
            _fail("network status is invalid")
        if request["allowlist_id"] and request["allowlist_id"] not in allowed:
            _fail("network allowlist_id is invalid")
        if origin in origins and (status == 0 or status >= 400) and request["allowlist_id"] not in allowed:
            revisions.append("first_party_request_failure")
        budgets = _object(_object(plan["condition"], "condition")["budgets"], "budgets")
        slow = _integer(_object(budgets["network"], "network budget")["slow_request_ms"], "slow request")
        if origin in origins and duration > slow:
            revisions.append("slow_first_party_request")


def _flow(plan: dict[str, object], expected: dict[str, object], value: object, cell_id: str, commands: dict[str, tuple[str, bool, int, str]], resolver: TrustedTraceResolver | None, blockers: list[str], revisions: list[str]) -> None:
    evidence = _closed(value, ("steps", "trace_reference", "operation_digests"), "critical flow evidence")
    steps = _bounded_objects(evidence["steps"], "critical flow steps", 32)
    if not steps:
        blockers.append("critical_flow_steps_missing")
    for step in steps:
        item = _closed(step, ("step_id", "action", "locator", "attempts"), "critical flow step")
        _ref(item["step_id"], "step_id")
        if item["action"] not in _ACTIONS:
            _fail("critical flow action is invalid")
        condition = _object(plan["condition"], "condition")
        if item["action"] == "submit" and (condition["interaction"] != "mutation" or condition["environment"] == "production"):
            _fail("observed submit is not authorized by the current plan")
        locator = _locator(item["locator"])
        if item["action"] == "navigate" and (locator["strategy"] != "route" or locator["value_digest"] != _route_navigation_digest(plan, expected)):
            _fail("navigate action must bind the planned route navigation digest")
        if item["action"] in ("click", "submit") and (condition["interaction"] == "read_only" or condition["environment"] == "production"):
            _fail("observed action is not authorized by the current plan")
        attempts = _bounded_objects(item["attempts"], "critical flow attempts", _integer(_object(plan["limits"], "plan limits")["max_attempts_per_read"], "max attempts"))
        if not attempts:
            blockers.append("critical_flow_attempts_missing")
            continue
        previous = ""
        identities: set[str] = set()
        for index, attempt in enumerate(attempts):
            current = _closed(attempt, ("query_identity", "operation_digest", "outcome", "failure_class"), "critical flow attempt")
            query = _digest_value(current["query_identity"], "attempt query identity")
            operation_digest = _digest_value(current["operation_digest"], "attempt operation_digest")
            command = commands.get(operation_digest)
            if command is None or command[0] != item["action"] or command[1] is not True or command[2] != 0 or command[3] != cell_id:
                _fail("critical flow attempt lacks matching successful command")
            if query in identities:
                _fail("critical flow retry query identity is not fresh")
            identities.add(query)
            if current["outcome"] not in ("success", "failure") or current["failure_class"] not in (*_TRANSIENT, *_TERMINAL, ""):
                _fail("critical flow attempt is invalid")
            if index:
                if _object(plan["condition"], "condition")["interaction"] != "read_only" or item["action"] not in _READ_ONLY_ACTIONS or previous not in _TRANSIENT:
                    _fail("critical flow retry is not an allowed transient read-only retry")
            if current["outcome"] == "success" and index != len(attempts) - 1:
                _fail("critical flow retry after success is invalid")
            if current["outcome"] == "failure" and current["failure_class"] in _TERMINAL and index != len(attempts) - 1:
                _fail("critical flow retry follows terminal failure")
            previous = str(current["failure_class"])
        if attempts[-1]["outcome"] != "success":
            if len(attempts) == _integer(_object(plan["limits"], "plan limits")["max_attempts_per_read"], "max attempts") and attempts[-1]["failure_class"] in _TRANSIENT:
                blockers.append("critical_flow_transient_attempts_exhausted")
            else:
                revisions.append("critical_flow_unsuccessful")
    reference = evidence["trace_reference"]
    if reference is not None:
        trace = _closed(reference, ("schema_version", "trace_id", "digest", "project_identity", "origins", "lifecycle_status"), "trace reference")
        origins = tuple(_origin(origin, "trace origin") for origin in _strings(trace["origins"], "trace origins"))
        if trace["schema_version"] != "browser_workflow_trace_reference/v1" or trace["lifecycle_status"] != "approved" or resolver is None or not resolver(str(trace["trace_id"]), str(trace["digest"]), str(trace["project_identity"]), origins):
            blockers.append("untrusted_reusable_trace_reference")


def _a11y(value: object, revisions: list[str]) -> None:
    evidence = _closed(value, ("findings", "operation_digests"), "accessibility evidence")
    for finding in _bounded_objects(evidence["findings"], "accessibility findings", 128):
        item = _closed(finding, ("rule_id", "impact", "node_count"), "accessibility finding")
        _ref(item["rule_id"], "a11y rule_id"); _non_negative(item["node_count"], "a11y node_count")
        if item["impact"] not in ("none", "minor", "moderate", "serious", "critical"):
            _fail("accessibility impact is invalid")
        if item["impact"] in ("serious", "critical"):
            revisions.append("serious_or_critical_accessibility_finding")


def _keyboard(value: object, revisions: list[str]) -> None:
    evidence = _closed(value, ("action", "focus_before", "focus_after", "focus_changed", "operation_digests"), "keyboard evidence")
    if evidence["action"] not in ("Tab", "Shift+Tab"):
        _fail("keyboard action is invalid")
    _locator(evidence["focus_before"]); _locator(evidence["focus_after"])
    if evidence["focus_before"] == evidence["focus_after"] or not _boolean(evidence["focus_changed"], "focus_changed"):
        revisions.append("keyboard_focus_not_observed")


def _performance(plan: dict[str, object], value: object, blockers: list[str], revisions: list[str]) -> None:
    evidence = _closed(value, ("evidence_class", "lab", "field", "operation_digests"), "performance evidence")
    _metrics(evidence["lab"], "lab", allow_null=True)
    field_gate = _object(_object(_object(plan["condition"], "condition")["budgets"], "budgets")["performance"], "performance")["field_gate"]
    if evidence["evidence_class"] == "lab":
        if evidence["field"] is not None:
            _fail("lab evidence cannot include field claim")
        if field_gate == "required":
            blockers.append("field_p75_not_observed")
        return
    if evidence["evidence_class"] != "field":
        _fail("performance evidence class is invalid")
    field = _closed(evidence["field"], ("source_class", "source_ref", "source_digest", "condition_digest", "sample_count", "window", "p75"), "field evidence")
    if field["source_class"] not in ("rum", "crux"):
        _fail("field source_class is invalid")
    _ref(field["source_ref"], "field source_ref"); _digest_value(field["source_digest"], "field source_digest")
    if field["condition_digest"] != plan["condition_digest"]:
        blockers.append("field_evidence_condition_mismatch")
    _positive(field["sample_count"], "field sample_count"); _window(field["window"], "field window")
    metrics = _metrics(field["p75"], "field p75", allow_null=False)
    bars = _object(_object(_object(_object(plan["condition"], "condition")["budgets"], "budgets")["performance"], "performance")["field_p75"], "field bars")
    if metrics["lcp_ms"] is None or metrics["inp_ms"] is None or metrics["cls"] is None:
        _fail("field metrics must be numeric")
    if metrics["lcp_ms"] >= _number(bars["lcp_ms_lt"], "lcp bar") or metrics["inp_ms"] >= _number(bars["inp_ms_lt"], "inp bar") or metrics["cls"] >= _number(bars["cls_lt"], "cls bar"):
        revisions.append("field_performance_budget_exceeded")


def _metrics(value: object, field: str, *, allow_null: bool) -> dict[str, int | float | None]:
    metrics = _closed(value, ("lcp_ms", "inp_ms", "cls"), field)
    output: dict[str, int | float | None] = {}
    for key, metric in metrics.items():
        if metric is None and allow_null:
            output[key] = None
        else:
            output[key] = _non_negative(metric, f"{field}.{key}")
    return output


def _missing_cell(expected: dict[str, object]) -> dict[str, object]:
    return {"cell_id": expected["cell_id"], "route_id": expected["route_id"], "state_id": expected["state_id"], "expected_terminal": expected["expected_terminal"], "terminal_status": "not_observed", "terminal_blocker_id": "host_cell_missing", "actual": None, "channels": {name: {"status": "blocked", "blocker_id": "host_cell_missing", "evidence": {}} for name in REQUIRED_CHANNELS}}


def _route_navigation_digest(plan: dict[str, object], expected: dict[str, object]) -> object:
    route = next(item for item in _objects(_object(plan["condition"], "condition")["routes"], "routes") if item["route_digest"] == expected["route_digest"])
    return route["navigation_digest"]


def _locator(value: object) -> dict[str, object]:
    locator = _closed(value, ("strategy", "value_digest"), "locator")
    if locator["strategy"] not in ("role", "label", "test_id", "attribute", "route"):
        _fail("locator strategy is invalid")
    _digest_value(locator["value_digest"], "locator value_digest")
    return locator


def _window(value: object, field: str) -> dict[str, str]:
    item = _object(value, field); started = _utc(item["started_at"], f"{field}.started_at"); ended = _utc(item["ended_at"], f"{field}.ended_at")
    if _parse(ended) <= _parse(started):
        _fail(f"{field} window must be positive")
    return {"started_at": started, "ended_at": ended}


def _origin(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be an origin")
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        _fail(f"{field} must be a canonical http(s) origin")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    suffix = "" if parsed.port is None or (parsed.scheme, parsed.port) in (("http", 80), ("https", 443)) else f":{parsed.port}"
    return f"{parsed.scheme}://{host}{suffix}"


def _snapshot(value: object) -> object:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise WebQaObservationError("receipt must contain finite JSON values") from exc
    if len(encoded) > 262_144:
        _fail("receipt exceeds byte bound")
    return value


def _closed(value: object, keys: tuple[str, ...], field: str) -> dict[str, object]:
    item = _object(value, field)
    if set(item) != set(keys):
        _fail(f"{field} must contain exactly its closed keys")
    return item


def _object(value: object, field: str) -> dict[str, object]:
    if type(value) is not dict or not all(isinstance(key, str) for key in value):
        _fail(f"{field} must be an object")
    return value


def _objects(value: object, field: str) -> list[dict[str, object]]:
    if type(value) is not list:
        _fail(f"{field} must be a list")
    return [_object(item, field) for item in value]


def _bounded_objects(value: object, field: str, maximum: int) -> list[dict[str, object]]:
    result = _objects(value, field)
    if len(result) > maximum:
        _fail(f"{field} exceeds bound")
    return result


def _strings(value: object, field: str) -> list[str]:
    if type(value) is not list or not all(isinstance(item, str) for item in value):
        _fail(f"{field} must be a string list")
    return value


def _valid_ref(value: object, *, empty: bool = False) -> bool:
    return isinstance(value, str) and (empty and value == "" or bool(_REF.fullmatch(value)))


def _adapter_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", value):
        _fail("adapter_id is invalid")
    return value


def _ref(value: object, field: str) -> str:
    if not _valid_ref(value):
        _fail(f"{field} is invalid")
    return str(value)


def _digest_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        _fail(f"{field} must be a SHA-256 digest")
    return value


def _revision(value: object) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or any(character not in "0123456789abcdef" for character in value):
        _fail("subject revision must be pinned hexadecimal")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        _fail(f"{field} must be an integer")
    return value


def _positive(value: object, field: str) -> int:
    number = _integer(value, field)
    if number < 1:
        _fail(f"{field} must be positive")
    return number


def _non_negative(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        _fail(f"{field} must be finite and non-negative")
    return value


def _number(value: object, field: str) -> int | float:
    return _non_negative(value, field)


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field} must be boolean")
    return value


def _utc(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be UTC")
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{field} must be UTC")
    if parsed.tzinfo is None:
        _fail(f"{field} must include an offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _fail(message: str) -> NoReturn:
    raise WebQaObservationError(message)
