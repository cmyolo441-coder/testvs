"""`omh capability-policy` -- inspect and change which families this install offers.

Named singular on purpose. `omh capabilities {list,summary,inspect,export}`
already exists and REPORTS what OMH can do; this group CHANGES what one local
install offers. Two plural/singular neighbours are easier to keep apart than
two verbs hidden inside one noun.

Plain text is the default and `--json` is the opt-in, matching every other
polled surface in this package.
"""

from __future__ import annotations

import argparse

from ..capabilities.families import capability_family_cards
from ..capabilities.toggles import (
    CapabilityPolicyError,
    build_capability_policy,
    disabled_family_ids,
    disabled_family_workflows,
    enabled_workflow_names,
    family_labels_by_id,
    normalize_family_id,
    org_rule_source_attestation_key_path,
    org_rule_source_enabled,
    org_rule_source_path,
    read_capability_policy,
    read_org_rule_source_policy,
    retained_core_skills,
    toggleable_family_ids,
    write_capability_policy,
)
from .common import _paths, _print_json, _wants_json

# From `quickstart`, not `setup`: `setup` registers this module's subparser, so
# importing it back here would close an import cycle. `quickstart` imports only
# `common`, which keeps the chain setup -> capability_policy -> quickstart open.
from .quickstart import _color, _use_color


def _family_use_for() -> dict[str, str]:
    return {str(card["id"]): str(card.get("use_for", "")) for card in capability_family_cards()}


def _policy_report(paths) -> dict[str, object]:
    policy = read_capability_policy(paths)
    disabled = disabled_family_ids(policy)
    labels = family_labels_by_id()
    use_for = _family_use_for()
    families = [
        {
            "id": family_id,
            "label": labels.get(family_id, family_id),
            "use_for": use_for.get(family_id, ""),
            "enabled": family_id not in disabled,
            "reverse_command": (
                f"omh capability-policy enable {family_id}"
                if family_id in disabled
                else f"omh capability-policy disable {family_id}"
            ),
        }
        for family_id in toggleable_family_ids()
    ]
    # The org safety rule source is the other setup-profile policy this group
    # owns. It is reported, never changed here: turning it on is a deliberate
    # local edit, and reporting it keeps the opt-in visible instead of hidden.
    org_policy = read_org_rule_source_policy(paths)
    return {
        "schema_version": "omh_capability_policy_report/v1",
        "policy": policy,
        "families": families,
        "disabled_families": list(disabled),
        "disabled_workflows": list(disabled_family_workflows(policy)),
        "enabled_workflow_count": len(enabled_workflow_names(policy)),
        "retained_core_skills": list(retained_core_skills()),
        "org_rule_source_policy": org_policy,
        "org_rule_source_enabled": org_rule_source_enabled(org_policy),
        "org_rule_source_path": org_rule_source_path(org_policy),
        "org_rule_source_attestation_key_path": org_rule_source_attestation_key_path(org_policy),
        "setup_profile_path": str(paths.setup_profile_path),
    }


def cmd_capability_policy_status(args: argparse.Namespace) -> int:
    payload = _policy_report(_paths(args))
    if _wants_json(args):
        _print_json(payload)
    else:
        _print_capability_policy_summary(payload)
    return 0


def cmd_capability_policy_disable(args: argparse.Namespace) -> int:
    return _apply(args, turn_on=False)


def cmd_capability_policy_enable(args: argparse.Namespace) -> int:
    return _apply(args, turn_on=True)


def _apply(args: argparse.Namespace, *, turn_on: bool) -> int:
    paths = _paths(args)
    try:
        family_id = normalize_family_id(str(getattr(args, "family", "") or ""))
    except CapabilityPolicyError as error:
        print(str(error))
        return 2

    current = read_capability_policy(paths)
    disabled = set(disabled_family_ids(current))
    already = (family_id not in disabled) if turn_on else (family_id in disabled)
    if turn_on:
        disabled.discard(family_id)
    else:
        disabled.add(family_id)

    preview = build_capability_policy(sorted(disabled), files_removed=False)
    dry_run = bool(getattr(args, "dry_run", False))
    payload = {
        "schema_version": "omh_capability_policy_change/v1",
        "family": family_id,
        "requested_state": "enabled" if turn_on else "disabled",
        "already_in_state": already,
        "dry_run": dry_run,
        "policy_before": current,
        "policy_after": preview,
        "affected_workflows": list(
            disabled_family_workflows(build_capability_policy([family_id]))
        ),
        "retained_core_skills": list(retained_core_skills()),
        "reverse_command": (
            f"omh capability-policy {'disable' if turn_on else 'enable'} {family_id}"
        ),
    }
    if not dry_run and not already:
        payload["policy_after"] = write_capability_policy(paths, sorted(disabled), files_removed=False)

    if _wants_json(args):
        _print_json(payload)
    else:
        _print_capability_policy_change(payload)
    return 0


def _print_capability_policy_summary(payload: dict[str, object]) -> None:
    use_color = _use_color()
    families = payload.get("families", [])
    disabled = payload.get("disabled_families", [])
    print(_color("OMH capability policy", "1;36", use_color))
    print(_color("Families", "1;32", use_color))
    if isinstance(families, list):
        for family in families:
            if not isinstance(family, dict):
                continue
            mark = "on " if family.get("enabled") else "OFF"
            print(f"  [{mark}] {family.get('id', '')} - {family.get('label', '')}")
    print(_color("Summary", "1;32", use_color))
    print(f"  Offered workflows: {payload.get('enabled_workflow_count', 0)}")
    if isinstance(disabled, list) and disabled:
        workflows = payload.get("disabled_workflows", [])
        count = len(workflows) if isinstance(workflows, list) else 0
        print(f"  Disabled families: {', '.join(str(item) for item in disabled)}")
        print(f"  Withheld workflows: {count}")
    else:
        print("  Disabled families: none")
    core = payload.get("retained_core_skills", [])
    if isinstance(core, list):
        print(f"  Always retained: {', '.join(str(item) for item in core)}")
    if payload.get("org_rule_source_enabled"):
        print(f"  Org safety rule source: on ({payload.get('org_rule_source_path', '')})")
        key_path = str(payload.get("org_rule_source_attestation_key_path", ""))
        # "Local attestation", never "signature": the key is symmetric, so this
        # line must not read as a provenance claim in a terminal either.
        if key_path:
            print(f"  Local rule attestation: required (key {key_path})")
        else:
            print("  Local rule attestation: not required")
    else:
        print("  Org safety rule source: off")
    print(_color("Next", "1;32", use_color))
    print("  Disable one with `omh capability-policy disable <family-id>`.")
    print(f"  Policy file: {payload.get('setup_profile_path', '')}")


def _print_capability_policy_change(payload: dict[str, object]) -> None:
    use_color = _use_color()
    family = str(payload.get("family", ""))
    state = str(payload.get("requested_state", ""))
    print(_color(f"OMH capability policy: {family} -> {state}", "1;36", use_color))
    if payload.get("already_in_state"):
        print(f"  No change: {family} is already {state}.")
    elif payload.get("dry_run"):
        print(f"  Dry run: {family} would be {state}; nothing was written.")
    else:
        print(f"  {family} is now {state}.")
    workflows = payload.get("affected_workflows", [])
    if isinstance(workflows, list) and workflows:
        print(_color("Workflows in this family", "1;32", use_color))
        for name in workflows:
            print(f"  - {name}")
    core = payload.get("retained_core_skills", [])
    if isinstance(core, list) and core:
        print(f"  Core skills always retained: {', '.join(str(item) for item in core)}")
    print(_color("Reverse", "1;32", use_color))
    print(f"  {payload.get('reverse_command', '')}")
