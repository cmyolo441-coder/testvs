"""Opt-in host-owned campaigns; ordinary work never enters this module."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
from typing import Any

from .fanout import build_fanout_contract, merge_order, require_spawn_plan, validate_fanout_units
from .fanout_contracts import verification_command_argv
from ..system.append_only_store import is_unsafe_metadata_line
from ..system.metadata_safety import is_body_shaped_metadata_text, is_sensitive_metadata_text
from .work_campaign_routes import resolve_campaign_routes

WORK_CAMPAIGN_SCHEMA_VERSION = "work_campaign/v1"
MAX_UNITS = 16
MAX_RECORD_BYTES = 65536
CLAIM_BOUNDARY = "Prepared routes are not model execution, review, CI, PR readiness, or merge authorization."


class CampaignError(ValueError):
    """A named, fail-closed campaign boundary refusal."""


def digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def scope_path(value: str) -> str:
    """Canonical literal file scopes before the existing fanout overlap gate."""
    if not isinstance(value, str) or not value or len(value) > 240:
        raise CampaignError("invalid_scope")
    path = posixpath.normpath(value.replace("\\", "/"))
    if path in (".", "..") or path.startswith(("/", "../")) or re.search(r"[:*?\[\]\x00-\x1f]", path):
        raise CampaignError("scope_must_be_relative_literal")
    return path


def bounded_strings(values, name: str) -> list[str]:
    if not isinstance(values, list) or not 1 <= len(values) <= 16:
        raise CampaignError("invalid_" + name)
    if any(not isinstance(x, str) or not x.strip() or len(x) > 240 or "\n" in x
           or (name not in ("scope", "artifacts") and is_unsafe_metadata_line(x)) for x in values):
        raise CampaignError("invalid_" + name)
    return values.copy()


def safe_verification_command(command: str) -> tuple[dict[str, str], list[str]]:
    """Retain executable metadata, never credentials or a pasted body."""
    if (not isinstance(command, str) or not command.strip()
            or is_body_shaped_metadata_text(command, limit=240)
            or is_sensitive_metadata_text(command)):
        raise CampaignError("invalid_verification_command")
    env, argv = verification_command_argv(command)
    # Shell quoting can hide a sensitive name/value in the original string.
    if any(is_sensitive_metadata_text(x) for x in [*env, *env.values(), *argv]):
        raise CampaignError("invalid_verification_command")
    return env, argv


def normalize_units(units, broad_command: str, spawn_plan=None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(units, list) or not 2 <= len(units) <= MAX_UNITS:
        raise CampaignError("campaign_requires_two_to_sixteen_units")
    normalized = []
    broad_argv = safe_verification_command(broad_command)
    for unit in units:
        if not isinstance(unit, dict):
            raise CampaignError("invalid_unit")
        if unit.get("unit_id") == "orchestrator":
            raise CampaignError("reserved_unit_id")
        paths = sorted({scope_path(x) for x in bounded_strings(unit.get("file_scope"), "scope")})
        acceptance = bounded_strings(unit.get("acceptance"), "acceptance")
        artifacts = [scope_path(x) for x in bounded_strings(unit.get("artifacts"), "artifacts")]
        if any(x not in paths for x in artifacts):
            raise CampaignError("artifact_outside_scope")
        command = unit.get("verification_command", "")
        if safe_verification_command(command) == broad_argv:
            raise CampaignError("leaf_requires_focused_verification")
        normalized.append(dict(unit_id=unit.get("unit_id", ""), owner="hermes", file_scope=paths,
                               depends_on=unit.get("depends_on", []), acceptance=acceptance,
                               artifacts=artifacts, verification_command=command))
    validate_fanout_units(normalized)
    merge_order(normalized)
    require_spawn_plan(len(normalized), spawn_plan)
    conflicts = []
    for i, first in enumerate(normalized):
        for second in normalized[i + 1:]:
            shared = [a for a in first["file_scope"] for b in second["file_scope"]
                      if a == b or PurePosixPath(a) in PurePosixPath(b).parents
                      or PurePosixPath(b) in PurePosixPath(a).parents]
            if shared:
                conflicts.append(dict(units=[first["unit_id"], second["unit_id"]],
                                      invariant_digest=digest(shared), resolved=False))
    # Reuse the existing freeze gate; conflicts are retained rather than quietly
    # bypassed with synthetic dependency edges. A conflicted graph cannot spawn.
    if not conflicts:
        build_fanout_contract("campaign graph", normalized, spawn_plan=spawn_plan)
    return normalized, conflicts


def build_work_campaign(*, mode="ordinary", accepted=False, goal="", units=None,
                        parent_session_ref="", acceptance_criteria=None,
                        verification_command="", workspace: str | Path = ".", omh_home=None, hermes_home=None,
                        owner_model="", owner_provider="", owner_effort="",
                        worker_model="", worker_provider="", worker_effort="", spawn_plan=None):
    if mode != "campaign-orchestrator":
        return None
    if not accepted or not isinstance(goal, str) or not goal.strip() or len(goal) > 4096:
        raise CampaignError("accepted_objective_required")
    if not parent_session_ref:
        raise CampaignError("host_parent_identity_required")
    criteria = bounded_strings(acceptance_criteria, "acceptance_criteria")
    normalized, conflicts = normalize_units(units, verification_command, spawn_plan)
    accepted_plan = require_spawn_plan(len(normalized), spawn_plan)
    if accepted_plan and any(is_unsafe_metadata_line(str(value)) for value in accepted_plan.values()):
        raise CampaignError("invalid_spawn_plan_metadata")
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise CampaignError("workspace_required")
    routes = resolve_campaign_routes(omh_home=omh_home, hermes_home=hermes_home, owner_model=owner_model,
                                    owner_provider=owner_provider, owner_effort=owner_effort,
                                    worker_model=worker_model, worker_provider=worker_provider,
                                    worker_effort=worker_effort)
    campaign_id = "campaign-" + digest([parent_session_ref, digest(goal), normalized, criteria,
                                        verification_command, str(root), routes])[:24]
    for unit in normalized:
        unit.update(attempt_id=digest([campaign_id, "unit", unit["unit_id"], 1]), state="prepared",
                    owner_session_ref="", binding=None)
        unit["idempotency_key"] = unit["attempt_id"]
    for conflict in conflicts:
        conflict["integration_owner"] = parent_session_ref
    return dict(schema_version=WORK_CAMPAIGN_SCHEMA_VERSION, campaign_id=campaign_id,
                objective_digest=digest(goal), acceptance_criteria=criteria, workspace=str(root), spawn_plan=accepted_plan,
                parent_session_ref=parent_session_ref, owner_session_ref=parent_session_ref,
                state="conflicted" if conflicts else "prepared", units=normalized, routes=routes,
                root_attempt_id=digest([campaign_id, "root", 1]), root_binding=None,
                conflicts=conflicts, pending_routes=[], events=[],
                broad_suite=dict(command=verification_command, queued=False, executions=0,
                                 attempt_id=digest([campaign_id, "verification", "broad", 1]), observed=None),
                metrics=dict(duplicate_dispatch=0, duplicate_result=0, conflicts=len(conflicts),
                             resolutions=0, fallbacks=0, cleanup_failures=0, rework_rounds=0),
                cleanup=dict(delegation_route_touched=False, expired=False, ordinary_readback=None),
                claim_boundary=CLAIM_BOUNDARY)


def leaf_contract_text(campaign, unit) -> str:
    return (f"TASK: {unit['unit_id']} in {campaign['campaign_id']} ({campaign['objective_digest']})\n"
            f"DELIVERABLE: {', '.join(unit['artifacts'])}; {', '.join(unit['acceptance'])}\n"
            f"SCOPE: {', '.join(unit['file_scope'])}\nVERIFY: {unit['verification_command']}\n"
            f"STOP WHEN: Return evidence to {campaign['owner_session_ref']} under {unit['attempt_id']}. "
            "Leaf only; no delegation, scope expansion, continuation, or broad suite.")


def kanban_task_manifests(campaign, *, worker_profile="", owner_profile="") -> list[dict[str, Any]]:
    root = dict(campaign_id=campaign["campaign_id"], unit_id="orchestrator", role="orchestrator",
                attempt_id=campaign["root_attempt_id"], idempotency_key=campaign["root_attempt_id"],
                route=deepcopy(campaign["routes"]["root"]), parents=[], assignee=owner_profile,
                max_spawn_depth=1, max_workers=len(campaign["units"]), max_retries=0,
                goal_mode=True, goal_max_turns=32, max_runtime_seconds=1800,
                owns=["plan", "graph", "dispatch", "results", "conflicts", "broad_suite", "go_no_go"],
                result_recipient=campaign["parent_session_ref"])
    leaves = [dict(campaign_id=campaign["campaign_id"], unit_id=u["unit_id"], role="leaf",
                   attempt_id=u["attempt_id"], idempotency_key=u["idempotency_key"],
                   route=deepcopy(campaign["routes"]["worker"]), parents=u["depends_on"].copy(),
                   assignee=worker_profile, max_spawn_depth=0, max_retries=0, goal_mode=False,
                   max_runtime_seconds=900, file_scope=u["file_scope"].copy(),
                   result_recipient=campaign["owner_session_ref"], contract=leaf_contract_text(campaign, u))
              for u in campaign["units"]]
    return [root, *leaves]
