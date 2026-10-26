"""`omh doctor --profile-isolation A B`: the pairwise profile isolation sub-check.

A doctor sub-check rather than a top-level command or a skill. It answers an
operator question about the local install, which is doctor's subject, and it
fires only for the multi-profile operators who pass the flag -- the ordinary
`omh doctor` run is unchanged, as `AGENTS.md` requires of the three commands
most people ever type.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..installer import OmhError
from ..maintenance.profile_isolation import (
    VERDICT_ISOLATED,
    compare_profile_isolation,
    parse_profile_ref,
)
from .common import _print_json


def run_profile_isolation(args: argparse.Namespace) -> int:
    """Compare the two supplied profile references and print the report."""
    first_text, second_text = args.profile_isolation
    try:
        first = parse_profile_ref(first_text)
        second = parse_profile_ref(second_text)
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    report = compare_profile_isolation(first, second)
    if getattr(args, "json", False):
        _print_json(report)
    else:
        _print_profile_isolation(report)
    return _profile_isolation_exit_code(report)


def _print_profile_isolation(report: dict[str, Any]) -> None:
    profiles = report["profiles"]
    print("Profile isolation (pairwise):")
    for index, profile in enumerate(profiles):
        hermes = profile["hermes_home"] or "(not supplied; Hermes-owned surfaces stay unknown)"
        print(f"  {'AB'[index]}: omh={profile['omh_home']} hermes={hermes}")
    for surface in report["surfaces"]:
        print(f"  {surface['surface']}: {surface['verdict']}")
        for finding in surface["findings"]:
            if finding["verdict"] == VERDICT_ISOLATED:
                continue
            detail = finding.get("shared_path") or finding.get("value") or ""
            suffix = f" [{detail}]" if detail else ""
            print(f"    {finding['artifact']}: {finding['verdict']} ({finding['reason']}){suffix}")
    summary = report["summary"]
    print(
        f"  {summary['isolated']} isolated, {summary['leaked']} leaked, "
        f"{summary['unknown']} unknown, of {summary['surfaces']} surfaces"
    )
    print(str(report["claim_boundary"]))


def _profile_isolation_exit_code(report: Any) -> int:
    """0 only when every surface was inspected and none leaked.

    1 when a surface leaked. Two profiles sharing config, cache, plugin state,
    files, or the environment variable that selects a home is the concrete
    finding, and an operator scripting a profile switch must not read it as a
    clean pair.

    3 when nothing leaked but a surface could not be inspected. Not 0, because
    an uninspected surface is not an isolated one and this whole check exists
    so the safe-looking answer cannot stand in for a missing one; not 1,
    because it is a gap in what was supplied (usually a profile reference with
    no Hermes home) rather than a fault in the install. Not 2: `main` already
    returns 2 for an OmhError, so a caller reading 2 could not tell a
    reference it could not parse from a comparison that ran.

    Generic failure signals -- a refused or interrupted run, or a unit
    carrying a failure kind -- are never success either, so this mapper cannot
    be passed by ignoring them.
    """
    summary = report if isinstance(report, dict) else {}
    if summary.get("refused") or summary.get("interrupted"):
        return 1
    units = summary.get("units")
    if isinstance(units, list) and any(isinstance(unit, dict) and unit.get("failure_kind") for unit in units):
        return 1
    if summary.get("leaked_surfaces"):
        return 1
    return 3 if summary.get("unknown_surfaces") else 0


def add_doctor_profile_isolation_argument(doctor: argparse.ArgumentParser) -> None:
    doctor.add_argument(
        "--profile-isolation",
        nargs=2,
        metavar=("PROFILE_A", "PROFILE_B"),
        default=None,
        help=(
            "Compare two profiles instead of running the health checklist, and report per-surface "
            "isolation across config, cache, plugin state, files, and env. Each profile is "
            "`<omh-home>[,<hermes-home>]`; omit the Hermes home and its surfaces report `unknown` "
            "rather than isolated. Exits 1 on a leak, 3 when a surface could not be inspected."
        ),
    )


__all__ = ["add_doctor_profile_isolation_argument", "run_profile_isolation"]
