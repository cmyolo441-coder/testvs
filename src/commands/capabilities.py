from __future__ import annotations

import argparse
import json

from ..capabilities.projection import (
    CapabilityProjectionError,
    DEFAULT_PROJECTION_LIMIT,
    approve_capability_authority,
    authority_change_report,
    expand_capability,
    project_capabilities,
    request_digest,
)
from ..capabilities.registry import capability_summary, filtered_capability_snapshot, inspect_capability, list_capabilities
from ..capabilities.schema import CAPABILITY_SECTION_CHOICES
from ..capabilities.toggles import enabled_workflow_names, read_capability_policy
from ..installer import OmhError
from ..quality.capability_impact import build_capability_impact_report, format_capability_impact_summary
from ..runtime.context_budget import (
    CAPABILITY_PROJECTION_SURFACE,
    record_context_emission,
    run_context_budget,
)
from .common import _paths, _print_json, _wants_json


def cmd_capabilities_export(args: argparse.Namespace) -> int:
    try:
        _print_json(filtered_capability_snapshot(args.section))
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    return 0


def cmd_capabilities_list(args: argparse.Namespace) -> int:
    try:
        payload = list_capabilities(args.section)
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(payload)
        return 0
    print("OMH capabilities")
    for section in payload["sections"]:
        ids = section["ids"]
        print(f"- {section['section']}: {len(ids)}")
        if ids:
            print(f"  {', '.join(str(item) for item in ids[:12])}")
            if len(ids) > 12:
                print(f"  ... {len(ids) - 12} more")
    print("For machine-readable output, rerun with `--json`.")
    return 0


def cmd_capabilities_summary(args: argparse.Namespace) -> int:
    payload = capability_summary()
    if _wants_json(args):
        _print_json(payload)
        return 0
    print("OMH capability summary")
    print("Use this when Hermes needs to explain what OMH can do without shell catalog approval.")
    families = payload.get("capability_families", [])
    if isinstance(families, list) and families:
        print("Capability families")
        print("Families are the user-facing front door; legacy lanes remain compatibility context.")
        for family in families:
            if not isinstance(family, dict):
                continue
            workflows = family.get("primary_workflows", [])
            workflow_text = ", ".join(str(item) for item in workflows[:5]) if isinstance(workflows, list) else ""
            print(f"- {family.get('label', '')} ({family.get('owner_role', '')})")
            print(f"  Use for: {family.get('use_for', '')}")
            print(f"  Workflows: {workflow_text}")
        print("Legacy lanes")
    for lane in payload["lanes"]:
        skills = lane["primary_skills"]
        playbooks = lane["representative_playbooks"]
        print(f"- {lane['label']} ({lane['owner_role']})")
        print(f"  Use for: {lane['use_for']}")
        print(f"  Skills: {', '.join(str(item) for item in skills[:8])}")
        if playbooks:
            print(f"  Playbooks: {', '.join(str(item['id']) for item in playbooks[:4])}")
    print("Boundary: capability summary is routing context, not execution evidence.")
    print("For machine-readable output, rerun with `--json`.")
    return 0


def cmd_capabilities_inspect(args: argparse.Namespace) -> int:
    try:
        payload = inspect_capability(args.identifier, section=args.section)
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    if _wants_json(args):
        _print_json(payload)
        return 0
    capability = payload["capability"]
    print(f"OMH capability: {payload['id']}")
    print(f"Section: {payload['section']}")
    if isinstance(capability, dict):
        for key in ("schema_version", "display_name", "category", "phase", "runtime_claim", "evidence_boundary"):
            if capability.get(key):
                print(f"{key.replace('_', ' ').title()}: {capability[key]}")
    print("For machine-readable output, rerun with `--json`.")
    return 0


def cmd_capabilities_impact(args: argparse.Namespace) -> int:
    payload = build_capability_impact_report()
    if _wants_json(args):
        _print_json(payload)
        return 0
    print(format_capability_impact_summary(payload))
    return 0


def cmd_capabilities_project(args: argparse.Namespace) -> int:
    """Project only the capabilities one request needs, inside the task budget.

    The grant is re-derived from this install's capability policy on every call.
    `--authority-digest` is how a caller pins the grant it approved: a policy
    change between two projections produces a different digest and this command
    refuses rather than quietly serving a different view.
    """
    paths = _paths(args)
    request = str(getattr(args, "request", "") or "")
    task_id = str(getattr(args, "task_id", "") or "") or f"task-{request_digest(request)[:12]}"
    offered = enabled_workflow_names(read_capability_policy(paths))

    try:
        authority = approve_capability_authority(task_id=task_id, granted_capabilities=offered)
    except CapabilityProjectionError as exc:
        raise OmhError(str(exc)) from exc

    approved_digest = str(getattr(args, "authority_digest", "") or "")
    if approved_digest:
        change = authority_change_report(approved_digest, authority)
        if not change["unchanged"]:
            if _wants_json(args):
                _print_json(change)
            else:
                print(f"OMH capability authority changed for {task_id}")
                print(f"  Approved digest: {change['approved_digest']}")
                print(f"  Current digest:  {change['current_digest']}")
                print("  Re-approve the capability authority before reusing this projection.")
            return 2

    budget = run_context_budget(paths, task_id, surface=CAPABILITY_PROJECTION_SURFACE)
    try:
        projection = project_capabilities(
            request,
            authority=authority,
            budget=budget,
            offered_capabilities=offered,
            limit=int(getattr(args, "limit", DEFAULT_PROJECTION_LIMIT)),
        )
        expand = str(getattr(args, "expand", "") or "")
        payload = expand_capability(projection, expand) if expand else projection.to_dict()
    except CapabilityProjectionError as exc:
        raise OmhError(str(exc)) from exc
    except ValueError as exc:
        raise OmhError(str(exc)) from exc

    record_context_emission(
        paths,
        task_id,
        surface=CAPABILITY_PROJECTION_SURFACE,
        byte_count=len(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)),
    )

    if _wants_json(args):
        _print_json(payload)
        return 0
    if expand:
        _print_capability_expansion(payload)
    else:
        _print_capability_projection(payload)
    return 0


def _print_capability_projection(payload: dict[str, object]) -> None:
    authority = payload.get("authority", {})
    authority = authority if isinstance(authority, dict) else {}
    print(f"OMH capability projection for task {payload.get('task_id', '')}")
    if payload.get("degraded"):
        drop = payload.get("budget_drop", {})
        drop = drop if isinstance(drop, dict) else {}
        print("  Degraded: this task exhausted its context budget; no capability detail was emitted.")
        print(f"  Dropped: {', '.join(str(item) for item in drop.get('dropped_capabilities', []))}")
        print(f"  Bytes: {drop.get('projected_bytes', 0)} needed, {drop.get('remaining_bytes', 0)} remaining.")
    included = payload.get("included", [])
    if isinstance(included, list) and included:
        print("Relevant capabilities")
        for entry in included:
            if not isinstance(entry, dict):
                continue
            print(f"- {entry.get('capability', '')} ({entry.get('family', '')})")
            print(f"  {entry.get('summary', '')}")
            print(f"  Why: {entry.get('match_reason', '')}")
    summary = payload.get("exclusion_summary", {})
    if isinstance(summary, dict) and summary:
        print("Excluded")
        for reason, count in sorted(summary.items()):
            print(f"  {reason}: {count}")
    print(f"Offered: {payload.get('offered_count', 0)}; included: {payload.get('included_count', 0)}.")
    print("Expansion is explicit: rerun with `--expand <capability>` for exact detail.")
    print("For machine-readable output, rerun with `--json`.")


def _print_capability_expansion(payload: dict[str, object]) -> None:
    detail = payload.get("detail", {})
    detail = detail if isinstance(detail, dict) else {}
    print(f"OMH capability detail: {payload.get('capability', '')}")
    print(f"Family: {payload.get('family', '')}")
    for key in ("description", "category", "phase", "hermes_role", "primary_harness"):
        if detail.get(key):
            print(f"{key.replace('_', ' ').title()}: {detail[key]}")
    print("Boundary: expanded detail is catalog metadata, not execution evidence.")
    print("For machine-readable output, rerun with `--json`.")


def _add_capabilities_commands(sub) -> None:
    capabilities = sub.add_parser("capabilities", help="Inspect OMH capability manifests for Hermes/plugin/wrapper use.")
    capabilities.add_argument("--json", action="store_true", help="Print the default machine-readable capability summary.")
    capabilities.set_defaults(func=cmd_capabilities_summary)
    capabilities_sub = capabilities.add_subparsers(dest="capabilities_command")

    export = capabilities_sub.add_parser("export", help="Export the deterministic OMH capability manifest.")
    export.add_argument("--section", choices=CAPABILITY_SECTION_CHOICES, default=None)
    export.add_argument("--json", action="store_true", help="Accepted for consistency; export is always machine-readable JSON.")
    export.set_defaults(func=cmd_capabilities_export)

    list_cmd = capabilities_sub.add_parser("list", help="List capability ids by section.")
    list_cmd.add_argument("--section", choices=CAPABILITY_SECTION_CHOICES, default=None)
    list_cmd.add_argument("--json", action="store_true", help="Print machine-readable capability id lists.")
    list_cmd.set_defaults(func=cmd_capabilities_list)

    summary = capabilities_sub.add_parser("summary", help="Summarize OMH lanes, representative skills, and playbooks.")
    summary.add_argument("--json", action="store_true", help="Print machine-readable capability summary.")
    summary.set_defaults(func=cmd_capabilities_summary)

    impact = capabilities_sub.add_parser(
        "impact",
        help="Separate proven routing impact from host, provider, verification, and outcome claims.",
    )
    impact.add_argument("--json", action="store_true", help="Print the machine-readable impact report.")
    impact.set_defaults(func=cmd_capabilities_impact)

    project = capabilities_sub.add_parser(
        "project",
        help="Project only the capabilities one request needs, inside the declared context budget.",
    )
    project.add_argument("request", nargs="?", default="", help="The outcome the user asked for.")
    project.add_argument("--limit", type=int, default=DEFAULT_PROJECTION_LIMIT, help="Maximum capabilities to include.")
    project.add_argument("--task-id", default="", help="Ledger and authority id; defaults to a digest of the request.")
    project.add_argument("--expand", default="", help="Return exact detail for one already-projected capability.")
    project.add_argument(
        "--authority-digest",
        default="",
        help="Refuse the projection when the re-derived authority no longer matches this approved digest.",
    )
    project.add_argument("--json", action="store_true", help="Print the machine-readable projection payload.")
    project.set_defaults(func=cmd_capabilities_project)

    inspect = capabilities_sub.add_parser("inspect", help="Inspect one capability by id.")
    inspect.add_argument("identifier")
    inspect.add_argument("--section", choices=CAPABILITY_SECTION_CHOICES, default=None)
    inspect.add_argument("--json", action="store_true", help="Print the full machine-readable capability.")
    inspect.set_defaults(func=cmd_capabilities_inspect)
