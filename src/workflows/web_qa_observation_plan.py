"""Pure `web_qa_observation_plan/v1` construction.

`build_web_qa_observation_plan(raw)` accepts one bounded, closed JSON-shaped
request and returns a closed plan with: schema/version, mode, subject,
condition, all required channels, caps, authorization references, round,
sanitized matrix, independent subject/condition digests, plan digest, and run
ID. `subject_digest` covers pinned source/deployment lineage only.
`condition_digest` covers immutable test conditions only. `plan_digest` and
`run_id` additionally bind caps, authorization, and round identity.

This dependency-free module performs no I/O, browser launch, provider call, or
execution attestation. Raw route URLs are transient input only. Persisted route
records retain canonical origin plus opaque path/navigation digests, never URL
credentials, query values, fragments, request bodies, or response bodies.
"""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
import re
from typing import Final, NoReturn, TypeGuard
from urllib.parse import urlsplit, urlunsplit
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

WEB_QA_OBSERVATION_PLAN_SCHEMA_VERSION: Final = "web_qa_observation_plan/v1"
REQUIRED_CHANNELS: Final = (
    "screenshot",
    "console",
    "network",
    "critical_flow",
    "accessibility",
    "keyboard",
    "performance",
)
HARD_LIMITS: Final[dict[str, int]] = {
    "max_routes": 16,
    "max_viewports": 8,
    "max_browsers": 4,
    "max_cells": 64,
    "max_concurrency": 4,
    "max_step_seconds": 60,
    "max_run_seconds": 900,
    "max_rounds": 3,
    "max_attempts_per_read": 3,
    "max_artifact_bytes": 50_000_000,
    "max_cost_units": 100,
}
MAX_JSON_BYTES: Final = 65_536
MAX_JSON_DEPTH: Final = 16
MAX_JSON_NODES: Final = 2_048

_OPAQUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PINNED_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_LOCALE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_TIMEZONE = re.compile(r"^(?:UTC|[A-Za-z_]+/[A-Za-z_]+(?:/[A-Za-z_]+)?)$")
_VERSION = re.compile(r"^\d+(?:\.\d+){1,3}$")

_MODES: Final = ("matrix", "canary")
_ENVIRONMENTS: Final = ("development", "test", "staging", "production")
_INTERACTIONS: Final = ("read_only", "login", "mutation")
_TERMINALS: Final = ("document_ready", "same_origin", "authenticated_view")
_ENGINES: Final = ("chromium", "firefox", "webkit")
_CACHE_PROFILES: Final = ("cold", "warm", "disabled")
_LOAD_PROFILES: Final = ("normal", "slow_4g", "fast_3g")
_DEVICE_PROFILES: Final = ("desktop", "mobile", "tablet")
_CPU_PROFILES: Final = ("normal", "x2", "x4")
_NETWORK_PROFILES: Final = ("online", "offline", "high_latency")
_ROLLOUT_PHASES: Final = ("pre_deploy", "canary", "rolling", "complete")


class WebQaObservationPlanError(ValueError):
    """Raised when an input cannot become a bounded observation plan."""


def build_web_qa_observation_plan(raw: object) -> dict[str, object]:
    """Build one deterministic, host-neutral `web_qa_observation_plan/v1`.

    Inputs are strictly typed and closed. Unsupported keys, duplicate route or
    matrix identifiers, non-finite values, cap increases, and oversized input
    fail closed. Authorization references only describe prerequisites for a
    later host action; they neither grant authority nor attest execution.
    """
    request = _bounded_snapshot(raw)
    root = _closed(request, ("mode", "subject", "condition", "authorization", "limits", "round"), "request")
    mode = _enum(root["mode"], _MODES, "mode")
    limits = _limits(root["limits"])
    subject = _subject(root["subject"], mode)
    condition = _condition(root["condition"], limits)
    authorization = _authorization(root["authorization"], condition)
    round_identity = _round(root["round"], limits)

    if mode == "canary":
        deployment_value = subject["deployment"]
        if not _is_object(deployment_value) or not subject["observed_deploy_ref"]:
            _fail("canary requires observed deployment lineage")
        if condition["environment"] != "production" or deployment_value["environment"] != "production":
            _fail("canary requires production environment")
        window_value = deployment_value["window"]
        if not _is_object(window_value) or not _is_int(window_value["duration_seconds"]):
            _fail("canary deployment window is invalid")
        if window_value["duration_seconds"] > limits["max_run_seconds"]:
            _fail("canary window exceeds max_run_seconds")

    subject_digest = _digest(subject)
    condition_digest = _digest(condition)
    identity = {
        "schema_version": WEB_QA_OBSERVATION_PLAN_SCHEMA_VERSION,
        "mode": mode,
        "subject_digest": subject_digest,
        "condition_digest": condition_digest,
        "required_channels": list(REQUIRED_CHANNELS),
        "limits": limits,
        "authorization": authorization,
        "round": round_identity,
    }
    plan_digest = _digest(identity)
    return {
        "schema_version": WEB_QA_OBSERVATION_PLAN_SCHEMA_VERSION,
        "mode": mode,
        "subject": subject,
        "condition": condition,
        "required_channels": list(REQUIRED_CHANNELS),
        "limits": limits,
        "authorization": authorization,
        "round": round_identity,
        "matrix": _matrix(condition, limits),
        "subject_digest": subject_digest,
        "condition_digest": condition_digest,
        "plan_digest": plan_digest,
        "run_id": f"web-qa-{plan_digest[:24]}",
        "does_not_authorize": (
            "browser_launch",
            "network_request",
            "login",
            "mutation",
            "deployment",
            "rollback",
            "execution_attestation",
        ),
    }


def parse_normalized_web_qa_observation_plan(value: object) -> dict[str, object]:
    """Re-admit a persisted digest-only plan using the plan's own validators.

    Persisted routes retain digests rather than raw URLs, so this validates and
    recomputes their canonical digest form before reusing the normal condition,
    authorization, cap, round, and matrix validators.
    """
    root = _closed(
        _bounded_normalized_snapshot(value),
        (
            "schema_version", "mode", "subject", "condition", "required_channels", "limits", "authorization",
            "round", "matrix", "subject_digest", "condition_digest", "plan_digest", "run_id", "does_not_authorize",
        ),
        "normalized plan",
    )
    mode = _enum(root["mode"], _MODES, "mode")
    if root["schema_version"] != WEB_QA_OBSERVATION_PLAN_SCHEMA_VERSION or root["required_channels"] != list(REQUIRED_CHANNELS):
        _fail("normalized plan schema or required channels are invalid")
    supplied_limits = _closed(root["limits"], tuple(HARD_LIMITS), "normalized plan limits")
    limits = _limits(supplied_limits)
    condition = _normalized_condition(root["condition"], limits)
    authorization = _authorization(root["authorization"], condition)
    round_identity = _round(root["round"], limits)
    subject_input = _normalized_subject_input(root["subject"], mode)
    subject = _subject(subject_input, mode)
    if mode == "canary":
        deployment = _closed(subject["deployment"], ("deployment_id", "environment", "rollout_phase", "window"), "normalized canary deployment")
        window = _closed(deployment["window"], ("starts_at", "ends_at", "duration_seconds"), "normalized canary window")
        if not subject["observed_deploy_ref"] or condition["environment"] != "production" or deployment["environment"] != "production" or _int(window["duration_seconds"], "normalized canary duration") > limits["max_run_seconds"]:
            _fail("normalized canary cross-check is invalid")
    if root["subject"] != subject or root["condition"] != condition or root["limits"] != limits or root["authorization"] != authorization or root["round"] != round_identity:
        _fail("normalized plan contains noncanonical values")
    if root["subject_digest"] != _digest(subject) or root["condition_digest"] != _digest(condition):
        _fail("normalized plan subject or condition digest is invalid")
    identity = {
        "schema_version": WEB_QA_OBSERVATION_PLAN_SCHEMA_VERSION,
        "mode": mode,
        "subject_digest": root["subject_digest"],
        "condition_digest": root["condition_digest"],
        "required_channels": list(REQUIRED_CHANNELS),
        "limits": limits,
        "authorization": authorization,
        "round": round_identity,
    }
    plan_digest = _digest(identity)
    if root["plan_digest"] != plan_digest or root["run_id"] != f"web-qa-{plan_digest[:24]}":
        _fail("normalized plan identity is invalid")
    matrix = _matrix(condition, limits)
    if root["matrix"] != matrix:
        _fail("normalized plan matrix is invalid")
    does_not_authorize = ("browser_launch", "network_request", "login", "mutation", "deployment", "rollback", "execution_attestation")
    if root["does_not_authorize"] not in (list(does_not_authorize), does_not_authorize):
        _fail("normalized plan does_not_authorize is invalid")
    return {
        "schema_version": WEB_QA_OBSERVATION_PLAN_SCHEMA_VERSION, "mode": mode, "subject": subject, "condition": condition,
        "required_channels": list(REQUIRED_CHANNELS), "limits": limits, "authorization": authorization, "round": round_identity,
        "matrix": matrix, "subject_digest": root["subject_digest"], "condition_digest": root["condition_digest"],
        "plan_digest": plan_digest, "run_id": root["run_id"], "does_not_authorize": does_not_authorize,
    }


def _bounded_normalized_snapshot(value: object) -> object:
    """Bound persisted plan material without changing tuple-valued plan output."""
    nodes = 0

    def visit(current: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            _fail("normalized plan exceeds structural bound")
        if current is None or isinstance(current, bool) or _is_int(current):
            return current
        if isinstance(current, float):
            if not math.isfinite(current):
                _fail("normalized plan contains non-finite number")
            return current
        if isinstance(current, str):
            if len(current.encode("utf-8")) > MAX_JSON_BYTES:
                _fail("normalized plan exceeds byte bound")
            return current
        if _is_list(current) or isinstance(current, tuple):
            if len(current) > MAX_JSON_NODES - nodes:
                _fail("normalized plan exceeds node bound")
            for item in current:
                visit(item, depth + 1)
            return current
        if _is_object(current):
            if len(current) > MAX_JSON_NODES - nodes:
                _fail("normalized plan exceeds node bound")
            for key, item in current.items():
                if len(key.encode("utf-8")) > MAX_JSON_BYTES:
                    _fail("normalized plan exceeds byte bound")
                visit(item, depth + 1)
            return current
        _fail("normalized plan must contain JSON values only")

    result = visit(value, 0)
    try:
        if len(json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")) > MAX_JSON_BYTES:
            _fail("normalized plan exceeds byte bound")
    except (TypeError, ValueError) as exc:
        raise WebQaObservationPlanError("normalized plan must contain JSON values only") from exc
    return result


def _normalized_subject_input(value: object, mode: str) -> dict[str, object]:
    subject = _closed(value, ("repository", "revision", "observed_deploy_ref", "deployment"), "normalized subject")
    if mode == "matrix":
        return subject
    deployment = _closed(subject["deployment"], ("deployment_id", "environment", "rollout_phase", "window"), "normalized deployment")
    window = _closed(deployment["window"], ("starts_at", "ends_at", "duration_seconds"), "normalized deployment window")
    return {
        "repository": subject["repository"], "revision": subject["revision"], "observed_deploy_ref": subject["observed_deploy_ref"],
        "deployment": {"deployment_id": deployment["deployment_id"], "environment": deployment["environment"], "rollout_phase": deployment["rollout_phase"], "window": {"starts_at": window["starts_at"], "ends_at": window["ends_at"]}},
    }


def _normalized_condition(value: object, limits: dict[str, int]) -> dict[str, object]:
    condition = _closed(value, ("routes", "viewports", "browsers", "locale", "timezone", "auth_fixture_ref", "profiles", "feature_flags", "budgets", "expected_noise_allowlist", "environment", "interaction"), "normalized condition")
    return {
        "routes": _normalized_routes(condition["routes"], limits), "viewports": _viewports(condition["viewports"], limits),
        "browsers": _browsers(condition["browsers"], limits), "locale": _text(condition["locale"], "condition.locale", _LOCALE),
        "timezone": _text(condition["timezone"], "condition.timezone", _TIMEZONE), "auth_fixture_ref": _opaque(condition["auth_fixture_ref"], "condition.auth_fixture_ref"),
        "profiles": _profiles(condition["profiles"]), "feature_flags": _feature_flags(condition["feature_flags"]),
        "budgets": _budgets(condition["budgets"]), "expected_noise_allowlist": _noise_allowlist(condition["expected_noise_allowlist"]),
        "environment": _enum(condition["environment"], _ENVIRONMENTS, "condition.environment"), "interaction": _enum(condition["interaction"], _INTERACTIONS, "condition.interaction"),
    }


def _normalized_routes(value: object, limits: dict[str, int]) -> list[dict[str, object]]:
    items = _list(value, "normalized condition.routes")
    if not items or len(items) > limits["max_routes"]:
        _fail("normalized routes exceed max_routes")
    routes: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        route = _closed(item, ("route_id", "state_id", "expected_terminal", "origin", "path_digest", "navigation_digest", "route_digest"), f"normalized routes[{index}]")
        route_id = _opaque(route["route_id"], "normalized route_id")
        state_id = _opaque(route["state_id"], "normalized state_id")
        if (route_id, state_id) in seen:
            _fail("normalized routes contain duplicate route/state")
        seen.add((route_id, state_id))
        terminal = _enum(route["expected_terminal"], _TERMINALS, "normalized expected_terminal")
        origin = _normalized_origin(route["origin"])
        path_digest = _text(route["path_digest"], "normalized path_digest", _PINNED_REVISION)
        navigation_digest = _text(route["navigation_digest"], "normalized navigation_digest", _PINNED_REVISION)
        digest = _digest({"route_id": route_id, "state_id": state_id, "expected_terminal": terminal, "origin": origin, "path_digest": path_digest, "navigation_digest": navigation_digest})
        if route["route_digest"] != digest:
            _fail("normalized route_digest is invalid")
        routes.append({"route_id": route_id, "state_id": state_id, "expected_terminal": terminal, "origin": origin, "path_digest": path_digest, "navigation_digest": navigation_digest, "route_digest": digest})
    return sorted(routes, key=lambda item: (str(item["route_id"]), str(item["state_id"])))


def _normalized_origin(value: object) -> str:
    if not isinstance(value, str):
        _fail("normalized origin must be a string")
    try:
        parsed = urlsplit(value)
        origin = _canonical_origin(parsed.scheme, parsed.hostname, parsed.port)
    except (ValueError, UnicodeError):
        _fail("normalized origin is invalid")
    if origin != value or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        _fail("normalized origin is invalid")
    return origin


def _subject(value: object, mode: str) -> dict[str, object]:
    subject = _closed(value, ("repository", "revision", "observed_deploy_ref", "deployment"), "subject")
    base = {
        "repository": _repository(subject["repository"]),
        "revision": _text(subject["revision"], "subject.revision", _PINNED_REVISION),
        "observed_deploy_ref": _optional_opaque(subject["observed_deploy_ref"], "subject.observed_deploy_ref"),
    }
    if mode == "matrix":
        if subject["deployment"] is not None:
            _fail("matrix subject.deployment must be null")
        return {**base, "deployment": None}

    deployment = _closed(
        subject["deployment"],
        ("deployment_id", "environment", "rollout_phase", "window"),
        "subject.deployment",
    )
    window = _closed(deployment["window"], ("starts_at", "ends_at"), "subject.deployment.window")
    starts_at = _utc_timestamp(window["starts_at"], "subject.deployment.window.starts_at")
    ends_at = _utc_timestamp(window["ends_at"], "subject.deployment.window.ends_at")
    starts = datetime.fromisoformat(starts_at.replace("Z", "+00:00"))
    ends = datetime.fromisoformat(ends_at.replace("Z", "+00:00"))
    duration = int((ends - starts).total_seconds())
    if duration <= 0:
        _fail("canary window must have positive duration")
    return {
        **base,
        "deployment": {
            "deployment_id": _opaque(deployment["deployment_id"], "subject.deployment.deployment_id"),
            "environment": _enum(deployment["environment"], _ENVIRONMENTS, "subject.deployment.environment"),
            "rollout_phase": _enum(deployment["rollout_phase"], _ROLLOUT_PHASES, "subject.deployment.rollout_phase"),
            "window": {"starts_at": starts_at, "ends_at": ends_at, "duration_seconds": duration},
        },
    }


def _condition(value: object, limits: dict[str, int]) -> dict[str, object]:
    condition = _closed(
        value,
        (
            "routes",
            "viewports",
            "browsers",
            "locale",
            "timezone",
            "auth_fixture_ref",
            "profiles",
            "feature_flags",
            "budgets",
            "expected_noise_allowlist",
            "environment",
            "interaction",
        ),
        "condition",
    )
    return {
        "routes": _routes(condition["routes"], limits),
        "viewports": _viewports(condition["viewports"], limits),
        "browsers": _browsers(condition["browsers"], limits),
        "locale": _text(condition["locale"], "condition.locale", _LOCALE),
        "timezone": _text(condition["timezone"], "condition.timezone", _TIMEZONE),
        "auth_fixture_ref": _opaque(condition["auth_fixture_ref"], "condition.auth_fixture_ref"),
        "profiles": _profiles(condition["profiles"]),
        "feature_flags": _feature_flags(condition["feature_flags"]),
        "budgets": _budgets(condition["budgets"]),
        "expected_noise_allowlist": _noise_allowlist(condition["expected_noise_allowlist"]),
        "environment": _enum(condition["environment"], _ENVIRONMENTS, "condition.environment"),
        "interaction": _enum(condition["interaction"], _INTERACTIONS, "condition.interaction"),
    }


def _routes(value: object, limits: dict[str, int]) -> list[dict[str, object]]:
    items = _list(value, "condition.routes")
    if not items or len(items) > limits["max_routes"]:
        _fail("condition.routes exceeds max_routes")
    routes: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(items):
        route = _closed(item, ("route_id", "state_id", "expected_terminal", "url"), f"condition.routes[{index}]")
        route_id = _opaque(route["route_id"], f"condition.routes[{index}].route_id")
        state_id = _opaque(route["state_id"], f"condition.routes[{index}].state_id")
        identity = (route_id, state_id)
        if identity in seen:
            _fail("condition.routes contains duplicate route/state cell")
        seen.add(identity)
        terminal = _enum(route["expected_terminal"], _TERMINALS, f"condition.routes[{index}].expected_terminal")
        origin, path_digest, navigation_digest = _route_location(route["url"], f"condition.routes[{index}].url")
        route_digest = _digest(
            {
                "route_id": route_id,
                "state_id": state_id,
                "expected_terminal": terminal,
                "origin": origin,
                "path_digest": path_digest,
                "navigation_digest": navigation_digest,
            }
        )
        routes.append(
            {
                "route_id": route_id,
                "state_id": state_id,
                "expected_terminal": terminal,
                "origin": origin,
                "path_digest": path_digest,
                "navigation_digest": navigation_digest,
                "route_digest": route_digest,
            }
        )
    return sorted(routes, key=lambda item: (str(item["route_id"]), str(item["state_id"])))


def _viewports(value: object, limits: dict[str, int]) -> list[dict[str, object]]:
    items = _list(value, "condition.viewports")
    if not items or len(items) > limits["max_viewports"]:
        _fail("condition.viewports exceeds max_viewports")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        viewport = _closed(item, ("viewport_id", "width", "height", "dpr"), f"condition.viewports[{index}]")
        viewport_id = _opaque(viewport["viewport_id"], f"condition.viewports[{index}].viewport_id")
        if viewport_id in seen:
            _fail("condition.viewports contains duplicate viewport_id")
        seen.add(viewport_id)
        width = _int(viewport["width"], f"condition.viewports[{index}].width")
        height = _int(viewport["height"], f"condition.viewports[{index}].height")
        dpr = _number(viewport["dpr"], f"condition.viewports[{index}].dpr")
        if not 1 <= width <= 7680 or not 1 <= height <= 4320 or not 0.5 <= dpr <= 4:
            _fail("condition viewport is outside supported bounds")
        result.append({"viewport_id": viewport_id, "width": width, "height": height, "dpr": dpr})
    return sorted(result, key=lambda item: str(item["viewport_id"]))


def _browsers(value: object, limits: dict[str, int]) -> list[dict[str, object]]:
    items = _list(value, "condition.browsers")
    if not items or len(items) > limits["max_browsers"]:
        _fail("condition.browsers exceeds max_browsers")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        browser = _closed(item, ("browser_id", "engine", "version"), f"condition.browsers[{index}]")
        browser_id = _opaque(browser["browser_id"], f"condition.browsers[{index}].browser_id")
        if browser_id in seen:
            _fail("condition.browsers contains duplicate browser_id")
        seen.add(browser_id)
        result.append(
            {
                "browser_id": browser_id,
                "engine": _enum(browser["engine"], _ENGINES, f"condition.browsers[{index}].engine"),
                "version": _text(browser["version"], f"condition.browsers[{index}].version", _VERSION),
            }
        )
    return sorted(result, key=lambda item: str(item["browser_id"]))


def _profiles(value: object) -> dict[str, object]:
    profiles = _closed(value, ("cache", "load", "device", "cpu", "network"), "condition.profiles")
    return {
        "cache": _enum(profiles["cache"], _CACHE_PROFILES, "condition.profiles.cache"),
        "load": _enum(profiles["load"], _LOAD_PROFILES, "condition.profiles.load"),
        "device": _enum(profiles["device"], _DEVICE_PROFILES, "condition.profiles.device"),
        "cpu": _enum(profiles["cpu"], _CPU_PROFILES, "condition.profiles.cpu"),
        "network": _enum(profiles["network"], _NETWORK_PROFILES, "condition.profiles.network"),
    }


def _feature_flags(value: object) -> list[dict[str, object]]:
    items = _list(value, "condition.feature_flags")
    if len(items) > 64:
        _fail("condition.feature_flags exceeds maximum")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        flag = _closed(item, ("flag_id", "state"), f"condition.feature_flags[{index}]")
        flag_id = _opaque(flag["flag_id"], f"condition.feature_flags[{index}].flag_id")
        if flag_id in seen:
            _fail("condition.feature_flags contains duplicate flag_id")
        seen.add(flag_id)
        result.append({"flag_id": flag_id, "state": _enum(flag["state"], ("enabled", "disabled"), f"condition.feature_flags[{index}].state")})
    return sorted(result, key=lambda item: str(item["flag_id"]))


def _noise_allowlist(value: object) -> list[dict[str, object]]:
    items = _list(value, "condition.expected_noise_allowlist")
    if len(items) > 64:
        _fail("condition.expected_noise_allowlist exceeds maximum")
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        noise = _closed(item, ("noise_id", "channel"), f"condition.expected_noise_allowlist[{index}]")
        noise_id = _opaque(noise["noise_id"], f"condition.expected_noise_allowlist[{index}].noise_id")
        if noise_id in seen:
            _fail("condition.expected_noise_allowlist contains duplicate noise_id")
        seen.add(noise_id)
        result.append({"noise_id": noise_id, "channel": _enum(noise["channel"], ("console", "network"), f"condition.expected_noise_allowlist[{index}].channel")})
    return sorted(result, key=lambda item: str(item["noise_id"]))


def _budgets(value: object) -> dict[str, object]:
    budgets = _closed(value, ("visual", "functional", "accessibility", "console", "network", "performance"), "condition.budgets")
    visual = _closed(budgets["visual"], ("minimum_score",), "condition.budgets.visual")
    functional = _closed(budgets["functional"], ("maximum_terminal_failures",), "condition.budgets.functional")
    accessibility = _closed(budgets["accessibility"], ("maximum_serious_or_critical_findings",), "condition.budgets.accessibility")
    console = _closed(budgets["console"], ("maximum_unallowlisted_exceptions",), "condition.budgets.console")
    network = _closed(budgets["network"], ("maximum_first_party_failures", "slow_request_ms"), "condition.budgets.network")
    performance = _closed(budgets["performance"], ("field_gate", "lab_evidence", "field_p75", "relative_tolerances"), "condition.budgets.performance")
    field = _closed(performance["field_p75"], ("lcp_ms_lt", "inp_ms_lt", "cls_lt"), "condition.budgets.performance.field_p75")
    tolerance = _closed(performance["relative_tolerances"], ("lcp_percent", "inp_percent", "cls_absolute"), "condition.budgets.performance.relative_tolerances")
    minimum_score = _int(visual["minimum_score"], "condition.budgets.visual.minimum_score")
    if not 90 <= minimum_score <= 100:
        _fail("visual minimum_score must be between 90 and 100")
    strict = {
        "lcp_ms_lt": _positive_number(field["lcp_ms_lt"], "condition.budgets.performance.field_p75.lcp_ms_lt"),
        "inp_ms_lt": _positive_number(field["inp_ms_lt"], "condition.budgets.performance.field_p75.inp_ms_lt"),
        "cls_lt": _positive_number(field["cls_lt"], "condition.budgets.performance.field_p75.cls_lt"),
    }
    if strict["lcp_ms_lt"] > 2500 or strict["inp_ms_lt"] > 200 or strict["cls_lt"] > 0.1:
        _fail("published genuine-field bars must remain LCP<2500ms INP<200ms CLS<0.1")
    return {
        "visual": {"minimum_score": minimum_score},
        "functional": {"maximum_terminal_failures": _zero(functional["maximum_terminal_failures"], "condition.budgets.functional.maximum_terminal_failures")},
        "accessibility": {"maximum_serious_or_critical_findings": _zero(accessibility["maximum_serious_or_critical_findings"], "condition.budgets.accessibility.maximum_serious_or_critical_findings")},
        "console": {"maximum_unallowlisted_exceptions": _zero(console["maximum_unallowlisted_exceptions"], "condition.budgets.console.maximum_unallowlisted_exceptions")},
        "network": {
            "maximum_first_party_failures": _zero(network["maximum_first_party_failures"], "condition.budgets.network.maximum_first_party_failures"),
            "slow_request_ms": _positive_int(network["slow_request_ms"], "condition.budgets.network.slow_request_ms"),
        },
        "performance": {
            "field_gate": _enum(performance["field_gate"], ("not_required", "required"), "condition.budgets.performance.field_gate"),
            "lab_evidence": _enum(performance["lab_evidence"], ("diagnostic",), "condition.budgets.performance.lab_evidence"),
            "field_p75": strict,
            "relative_tolerances": {
                "lcp_percent": _non_negative_number(tolerance["lcp_percent"], "condition.budgets.performance.relative_tolerances.lcp_percent"),
                "inp_percent": _non_negative_number(tolerance["inp_percent"], "condition.budgets.performance.relative_tolerances.inp_percent"),
                "cls_absolute": _non_negative_number(tolerance["cls_absolute"], "condition.budgets.performance.relative_tolerances.cls_absolute"),
            },
        },
    }


def _authorization(value: object, condition: dict[str, object]) -> dict[str, object]:
    authorization = _closed(value, ("intent", "staging_test_authorization_ref"), "authorization")
    interaction = condition["interaction"]
    if not isinstance(interaction, str):
        _fail("condition interaction is invalid")
    intent = _enum(authorization["intent"], _INTERACTIONS, "authorization.intent")
    if intent != interaction:
        _fail("authorization.intent must match condition.interaction")
    reference = _optional_opaque(authorization["staging_test_authorization_ref"], "authorization.staging_test_authorization_ref")
    environment = condition["environment"]
    if environment == "production" and interaction != "read_only":
        _fail("production plans must be read_only")
    if interaction in ("login", "mutation") and (environment not in ("staging", "test") or not reference):
        _fail("login or mutation requires staging/test authorization reference")
    return {"intent": intent, "staging_test_authorization_ref": reference}


def _limits(value: object) -> dict[str, int]:
    supplied = _closed(value, tuple(HARD_LIMITS), "limits", required=False)
    limits = dict(HARD_LIMITS)
    for key, supplied_value in supplied.items():
        amount = _positive_int(supplied_value, f"limits.{key}")
        if amount > HARD_LIMITS[key]:
            _fail(f"limits.{key} may only tighten its hard maximum")
        limits[key] = amount
    return limits


def _round(value: object, limits: dict[str, int]) -> dict[str, object]:
    round_value = _closed(value, ("round_id", "ordinal"), "round")
    ordinal = _positive_int(round_value["ordinal"], "round.ordinal")
    if ordinal > limits["max_rounds"]:
        _fail("round.ordinal exceeds max_rounds")
    return {"round_id": _opaque(round_value["round_id"], "round.round_id"), "ordinal": ordinal}


def _matrix(condition: dict[str, object], limits: dict[str, int]) -> list[dict[str, object]]:
    routes = _list(condition["routes"], "planned routes")
    viewports = _list(condition["viewports"], "planned viewports")
    browsers = _list(condition["browsers"], "planned browsers")
    if len(routes) * len(viewports) * len(browsers) > limits["max_cells"]:
        _fail("matrix exceeds max_cells")
    cells: list[dict[str, object]] = []
    for route_value in routes:
        route = _closed(route_value, ("route_id", "state_id", "expected_terminal", "origin", "path_digest", "navigation_digest", "route_digest"), "planned route")
        for viewport_value in viewports:
            viewport = _closed(viewport_value, ("viewport_id", "width", "height", "dpr"), "planned viewport")
            for browser_value in browsers:
                browser = _closed(browser_value, ("browser_id", "engine", "version"), "planned browser")
                identity = {
                    "route_digest": route["route_digest"],
                    "viewport_id": viewport["viewport_id"],
                    "browser_id": browser["browser_id"],
                }
                cells.append(
                    {
                        "cell_id": f"cell-{_digest(identity)[:24]}",
                        "route_id": route["route_id"],
                        "state_id": route["state_id"],
                        "expected_terminal": route["expected_terminal"],
                        "route_digest": route["route_digest"],
                        "viewport_id": viewport["viewport_id"],
                        "browser_id": browser["browser_id"],
                    }
                )
    return cells


def _route_location(value: object, field: str) -> tuple[str, str, str]:
    if not isinstance(value, str):
        _fail(f"{field} must be a string")
    try:
        parsed = urlsplit(value)
        origin = _canonical_origin(parsed.scheme, parsed.hostname, parsed.port)
    except (ValueError, UnicodeError):
        _fail(f"{field} must be an absolute credential-free http(s) URL")
    if not origin or parsed.username or parsed.password:
        _fail(f"{field} must be an absolute credential-free http(s) URL")
    path = parsed.path or "/"
    return origin, _digest({"path": path}), _digest({"path": path, "query": parsed.query, "fragment": parsed.fragment})


def _repository(value: object) -> str:
    if not isinstance(value, str):
        _fail("subject.repository must be a string")
    try:
        parsed = urlsplit(value)
        origin = _canonical_origin(parsed.scheme, parsed.hostname, parsed.port)
    except ValueError:
        _fail("subject.repository must be an absolute credential-free URL")
    if not origin or parsed.username or parsed.password or parsed.query or parsed.fragment:
        _fail("subject.repository must be an absolute credential-free URL")
    return urlunsplit((parsed.scheme.lower(), origin.split("://", 1)[1], parsed.path.rstrip("/"), "", ""))


def _canonical_origin(scheme: str, hostname: str | None, port: int | None) -> str:
    if scheme not in ("http", "https") or not hostname:
        return ""
    if ":" in hostname:
        host = f"[{hostname.lower()}]"
    else:
        host = hostname.encode("idna").decode("ascii").lower()
    default_port = 80 if scheme == "http" else 443
    suffix = "" if port is None or port == default_port else f":{port}"
    return f"{scheme.lower()}://{host}{suffix}"


def _bounded_snapshot(value: object) -> dict[str, object]:
    nodes = 0
    bytes_used = 0

    def consume(amount: int) -> None:
        nonlocal bytes_used
        bytes_used += amount
        if bytes_used > MAX_JSON_BYTES:
            _fail("request JSON exceeds byte bound")

    def encoded_size(item: object) -> int:
        return len(json.dumps(item, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))

    def visit(current: object, depth: int) -> object:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_JSON_NODES:
            _fail("request JSON exceeds node bound")
        if depth > MAX_JSON_DEPTH:
            _fail("request JSON exceeds depth bound")
        if current is None or isinstance(current, bool) or _is_int(current):
            consume(encoded_size(current))
            return current
        if isinstance(current, float):
            if not math.isfinite(current):
                _fail("request JSON contains non-finite number")
            consume(encoded_size(current))
            return current
        if isinstance(current, str):
            if len(current) > MAX_JSON_BYTES:
                _fail("request JSON exceeds byte bound")
            consume(encoded_size(current))
            return current
        if _is_list(current):
            if len(current) > MAX_JSON_NODES - nodes:
                _fail("request JSON exceeds node bound")
            consume(2 + max(len(current) - 1, 0))
            return [visit(item, depth + 1) for item in current]
        if _is_object(current):
            if len(current) > MAX_JSON_NODES - nodes:
                _fail("request JSON exceeds node bound")
            consume(2 + max(len(current) - 1, 0))
            result: dict[str, object] = {}
            for key, item in current.items():
                if len(key) > MAX_JSON_BYTES:
                    _fail("request JSON exceeds byte bound")
                consume(encoded_size(key) + 1)
                result[key] = visit(item, depth + 1)
            return result
        _fail("request must contain JSON values only")

    snapshot = visit(value, 0)
    if not _is_object(snapshot):
        _fail("request must be an object")
    return snapshot


def _closed(value: object, keys: tuple[str, ...], field: str, *, required: bool = True) -> dict[str, object]:
    if not _is_object(value):
        _fail(f"{field} must be an object")
    actual = set(value)
    expected = set(keys)
    if actual - expected:
        _fail(f"{field} contains unsupported keys")
    if required and expected - actual:
        _fail(f"{field} is missing required keys")
    return value


def _is_object(value: object) -> TypeGuard[dict[str, object]]:
    return type(value) is dict and all(isinstance(key, str) for key in value)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return type(value) is list


def _is_int(value: object) -> TypeGuard[int]:
    return type(value) is int


def _list(value: object, field: str) -> list[object]:
    if not _is_list(value):
        _fail(f"{field} must be a list")
    return value


def _text(value: object, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        _fail(f"{field} has invalid format")
    return value


def _opaque(value: object, field: str) -> str:
    return _text(value, field, _OPAQUE)


def _optional_opaque(value: object, field: str) -> str:
    if value == "":
        return ""
    return _opaque(value, field)


def _enum(value: object, choices: tuple[str, ...], field: str) -> str:
    if not isinstance(value, str) or value not in choices:
        _fail(f"{field} must be an exact supported value")
    return value


def _int(value: object, field: str) -> int:
    if not _is_int(value):
        _fail(f"{field} must be an integer")
    return value


def _positive_int(value: object, field: str) -> int:
    number = _int(value, field)
    if number < 1:
        _fail(f"{field} must be positive")
    return number


def _number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(f"{field} must be a finite number")
    return value


def _positive_number(value: object, field: str) -> int | float:
    number = _number(value, field)
    if number <= 0:
        _fail(f"{field} must be positive")
    return number


def _non_negative_number(value: object, field: str) -> int | float:
    number = _number(value, field)
    if number < 0:
        _fail(f"{field} must be non-negative")
    return number


def _zero(value: object, field: str) -> int:
    number = _int(value, field)
    if number != 0:
        _fail(f"{field} must be zero")
    return number


def _utc_timestamp(value: object, field: str) -> str:
    if not isinstance(value, str):
        _fail(f"{field} must be a UTC timestamp")
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{field} must be a UTC timestamp")
    if parsed.tzinfo is None:
        _fail(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fail(message: str) -> NoReturn:
    raise WebQaObservationPlanError(message)
