"""Verify pairwise profile isolation across the five surfaces that can leak.

`CONTEXT.md` states "Exactly one home is active per invocation" as an
invariant of the OMH home. Nothing checked it. An operator running two
profiles found a leak by watching one profile's behaviour change, which is the
slowest possible way to learn it.

A profile here is the pair a single invocation resolves: an OMH home and,
optionally, a Hermes home. That is what `resolve_paths` produces and what
every artifact below hangs off, so comparing two of them compares what two
invocations would actually touch.

Five surfaces, each with named artifacts rather than a whole-tree diff. A tree
diff answers "are these directories different", which is not the question: two
different directories still leak when one is nested inside the other, when
both resolve through a symlink to one target, or when an ambient environment
variable pins one of them for every invocation that forgets a flag. Each
surface asks about the artifacts a leak would actually travel through.

Three verdicts, and `unknown` is a real one. A surface whose artifacts live
under a Hermes home the caller did not supply is `unknown`, never `isolated`:
the check has not looked, and reporting the safe-looking answer for a surface
it could not inspect is the failure this module exists to prevent. `leaked`
outranks `unknown` at the surface level, because a leak the check DID find is
the concrete finding and an operator acts on it first.

Reads nothing but path metadata. Nothing here opens a file, writes one, or
starts a process; `Path.resolve()` is the only filesystem call, and it is the
one that makes two symlinks to one target report as the leak they are.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping

PROFILE_ISOLATION_SCHEMA_VERSION: Final[str] = "profile_isolation/v1"

VERDICT_ISOLATED: Final[str] = "isolated"
VERDICT_LEAKED: Final[str] = "leaked"
VERDICT_UNKNOWN: Final[str] = "unknown"
PROFILE_ISOLATION_VERDICTS: Final[tuple[str, ...]] = (
    VERDICT_ISOLATED,
    VERDICT_LEAKED,
    VERDICT_UNKNOWN,
)

SURFACE_CONFIG: Final[str] = "config"
SURFACE_CACHE: Final[str] = "cache"
SURFACE_PLUGIN_STATE: Final[str] = "plugin_state"
SURFACE_FILES: Final[str] = "files"
SURFACE_ENV: Final[str] = "env"
PROFILE_ISOLATION_SURFACES: Final[tuple[str, ...]] = (
    SURFACE_CONFIG,
    SURFACE_CACHE,
    SURFACE_PLUGIN_STATE,
    SURFACE_FILES,
    SURFACE_ENV,
)

REASON_DISTINCT_PATHS: Final[str] = "distinct_paths"
REASON_SAME_PATH: Final[str] = "same_resolved_path"
REASON_NESTED_PATH: Final[str] = "one_path_contains_the_other"
REASON_HOME_NOT_SUPPLIED: Final[str] = "home_not_supplied"
REASON_PATH_UNRESOLVABLE: Final[str] = "path_could_not_be_resolved"
REASON_ENV_UNSET: Final[str] = "environment_variable_unset"
REASON_ENV_PINS_PROFILE: Final[str] = "environment_variable_pins_one_profile"
REASON_ENV_PINS_NEITHER: Final[str] = "environment_variable_pins_neither_profile"

REASON_TEXT: Final[dict[str, str]] = {
    REASON_DISTINCT_PATHS: "The two profiles resolve this artifact to different paths, neither inside the other.",
    REASON_SAME_PATH: "Both profiles resolve this artifact to one path, so either can observe and overwrite the other's.",
    REASON_NESTED_PATH: "One profile's path lies inside the other's tree, so writes under the inner one land inside the outer profile.",
    REASON_HOME_NOT_SUPPLIED: (
        "This artifact lives under a Hermes home, and at least one profile reference supplied "
        "none. The check has not looked; this is not a finding of isolation."
    ),
    REASON_PATH_UNRESOLVABLE: (
        "The path could not be resolved, so the two could not be compared. The check has not "
        "looked; this is not a finding of isolation."
    ),
    REASON_ENV_UNSET: "The variable is unset in the inspected environment, so it pins neither profile.",
    REASON_ENV_PINS_PROFILE: (
        "The variable pins one of these two profiles for every invocation that does not override "
        "it, so the other profile is reachable only with an explicit flag."
    ),
    REASON_ENV_PINS_NEITHER: (
        "The variable is set to a home that is neither of these two, so neither profile can reach "
        "the other through it."
    ),
}

PROFILE_ISOLATION_CLAIM_BOUNDARY: Final[str] = (
    "An isolation report compares the paths two profile references resolve and the environment "
    "handed to this process. It is path metadata only: nothing was opened, written, or executed, "
    "and no invocation was observed running under either profile. An `isolated` verdict covers "
    "the named artifacts of that surface and nothing else, and a surface reported `unknown` was "
    "not inspected -- it is not a quieter way of saying isolated."
)

# Artifact paths per surface, relative to the profile's OMH or Hermes home.
# Relative segments rather than `OmhPaths` properties on purpose: the check
# must describe a profile the current process is not running under, and every
# property here would otherwise have to be re-derived from a second
# `OmhPaths`. Each entry is (artifact name, home, *segments).
_OMH: Final[str] = "omh_home"
_HERMES: Final[str] = "hermes_home"
_SURFACE_ARTIFACTS: Final[dict[str, tuple[tuple[str, str, tuple[str, ...]], ...]]] = {
    SURFACE_CONFIG: (
        ("setup_profile", _OMH, ("setup-profile.json",)),
        ("model_chain_overrides", _OMH, ("routing", "model-chains.json")),
        ("toolcall_rules", _OMH, ("rules", "toolcall-rules.json")),
        ("hermes_config", _HERMES, ("config.yaml",)),
    ),
    SURFACE_CACHE: (
        ("update_check_cache", _OMH, ("runtime", "update-check.json")),
        ("output_spills", _OMH, ("runtime", "output-spills")),
    ),
    SURFACE_PLUGIN_STATE: (
        ("plugin_host_observations", _OMH, ("runtime", "plugin-host-observations.jsonl")),
        ("plan_todo", _OMH, ("runtime", "todo.json")),
        ("session_plan_todos", _OMH, ("runtime", "todos")),
        ("installed_plugin_bundle", _HERMES, ("plugins", "omh")),
    ),
    SURFACE_FILES: (
        ("managed_skills", _OMH, ("skills",)),
        ("install_manifest", _OMH, ("manifest.json",)),
        ("memory_records", _OMH, ("memory",)),
        ("tui_widgets", _HERMES, ("tui-widgets",)),
    ),
}

# The variables that select a home when no flag does. `OMH_SCOPE` is not one:
# scope arrives as `--scope`, never from the environment.
_ENV_VARIABLES: Final[tuple[tuple[str, str], ...]] = (
    ("OMH_HOME", _OMH),
    ("HERMES_HOME", _HERMES),
)


@dataclass(frozen=True)
class ProfileRef:
    """One profile as an invocation would resolve it: a ref and its two homes."""

    ref: str
    omh_home: Path
    hermes_home: Path | None


def parse_profile_ref(text: str) -> ProfileRef:
    """Parse `<omh-home>[,<hermes-home>]` into a profile reference.

    An omitted Hermes home is not defaulted to `~/.hermes`. Guessing it would
    turn "the caller did not say" into a comparison between one real home and
    one invented one, and every Hermes-owned surface would report a verdict
    about a profile nobody described. It stays `None`, and those surfaces
    report `unknown`.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("a profile reference needs an OMH home path")
    omh_text, separator, hermes_text = raw.partition(",")
    omh_text = omh_text.strip()
    hermes_text = hermes_text.strip()
    if not omh_text:
        raise ValueError(f"profile reference {raw!r} names no OMH home")
    if separator and not hermes_text:
        raise ValueError(
            f"profile reference {raw!r} ends in a comma but names no Hermes home; "
            "drop the comma to leave the Hermes-owned surfaces unknown"
        )
    return ProfileRef(
        ref=raw,
        omh_home=Path(omh_text).expanduser(),
        hermes_home=Path(hermes_text).expanduser() if hermes_text else None,
    )


def compare_profile_isolation(
    first: ProfileRef,
    second: ProfileRef,
    *,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Report per-surface isolation between two profile references."""
    environment = dict(os.environ if env is None else env)
    surfaces = [
        _path_surface(surface, first, second)
        for surface in PROFILE_ISOLATION_SURFACES
        if surface in _SURFACE_ARTIFACTS
    ]
    surfaces.append(_env_surface(first, second, environment))
    counts = {verdict: 0 for verdict in PROFILE_ISOLATION_VERDICTS}
    for surface in surfaces:
        counts[str(surface["verdict"])] += 1
    return {
        "schema_version": PROFILE_ISOLATION_SCHEMA_VERSION,
        "profiles": [_profile_row(first), _profile_row(second)],
        "surfaces": surfaces,
        "leaked_surfaces": [str(row["surface"]) for row in surfaces if row["verdict"] == VERDICT_LEAKED],
        "unknown_surfaces": [str(row["surface"]) for row in surfaces if row["verdict"] == VERDICT_UNKNOWN],
        "summary": {
            "surfaces": len(surfaces),
            "isolated": counts[VERDICT_ISOLATED],
            "leaked": counts[VERDICT_LEAKED],
            "unknown": counts[VERDICT_UNKNOWN],
        },
        "reason_codes": dict(REASON_TEXT),
        "claim_boundary": PROFILE_ISOLATION_CLAIM_BOUNDARY,
    }


def _profile_row(profile: ProfileRef) -> dict[str, Any]:
    return {
        "ref": profile.ref,
        "omh_home": str(profile.omh_home),
        "hermes_home": str(profile.hermes_home) if profile.hermes_home is not None else "",
    }


def _path_surface(surface: str, first: ProfileRef, second: ProfileRef) -> dict[str, Any]:
    artifacts = _SURFACE_ARTIFACTS[surface]
    # The home roots come first, and they are not redundant with the
    # artifacts below. Two nested homes leak -- everything under the inner one
    # sits inside the outer profile's tree -- while every artifact pair
    # remains a pair of siblings that contains neither the other nor itself.
    # Comparing only artifacts reports that case isolated, which is how a
    # whole profile nested inside another would have passed.
    roots = [home for home in (_OMH, _HERMES) if any(entry[1] == home for entry in artifacts)]
    findings = [_compare_artifact(home, home, (), first, second) for home in roots]
    findings.extend(
        _compare_artifact(name, home, segments, first, second) for name, home, segments in artifacts
    )
    return {
        "surface": surface,
        "verdict": _roll_up([str(row["verdict"]) for row in findings]),
        "findings": findings,
    }


def _compare_artifact(
    name: str,
    home: str,
    segments: tuple[str, ...],
    first: ProfileRef,
    second: ProfileRef,
) -> dict[str, Any]:
    left = _artifact_path(first, home, segments)
    right = _artifact_path(second, home, segments)
    if left is None or right is None:
        return _finding(name, VERDICT_UNKNOWN, REASON_HOME_NOT_SUPPLIED, left, right)
    try:
        left_resolved = left.resolve()
        right_resolved = right.resolve()
    except (OSError, RuntimeError, ValueError):
        # A symlink loop, a path too long for the platform, or a filesystem
        # that refuses the lookup. Classified and surfaced as `unknown`: the
        # comparison did not happen, and saying `isolated` here would be a
        # verdict over an answer nobody got.
        return _finding(name, VERDICT_UNKNOWN, REASON_PATH_UNRESOLVABLE, left, right)
    if left_resolved == right_resolved:
        return _finding(
            name,
            VERDICT_LEAKED,
            REASON_SAME_PATH,
            left,
            right,
            shared_path=str(left_resolved),
        )
    container = _containing_path(left_resolved, right_resolved)
    if container is not None:
        return _finding(
            name,
            VERDICT_LEAKED,
            REASON_NESTED_PATH,
            left,
            right,
            shared_path=str(container),
        )
    return _finding(name, VERDICT_ISOLATED, REASON_DISTINCT_PATHS, left, right)


def _artifact_path(profile: ProfileRef, home: str, segments: tuple[str, ...]) -> Path | None:
    root = profile.omh_home if home == _OMH else profile.hermes_home
    return None if root is None else root.joinpath(*segments)


def _containing_path(left: Path, right: Path) -> Path | None:
    """The outer path when one contains the other, else None."""
    if left in right.parents:
        return left
    if right in left.parents:
        return right
    return None


def _env_surface(
    first: ProfileRef,
    second: ProfileRef,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for variable, home in _ENV_VARIABLES:
        raw = str(environment.get(variable, "") or "").strip()
        if not raw:
            findings.append(_env_finding(variable, VERDICT_ISOLATED, REASON_ENV_UNSET, ""))
            continue
        try:
            pinned = Path(raw).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            findings.append(_env_finding(variable, VERDICT_UNKNOWN, REASON_PATH_UNRESOLVABLE, raw))
            continue
        pins = [
            profile.ref
            for profile in (first, second)
            if _env_pins(pinned, _artifact_path(profile, home, ()))
        ]
        if pins:
            findings.append(
                _env_finding(variable, VERDICT_LEAKED, REASON_ENV_PINS_PROFILE, raw, pins=pins)
            )
        else:
            findings.append(_env_finding(variable, VERDICT_ISOLATED, REASON_ENV_PINS_NEITHER, raw))
    return {
        "surface": SURFACE_ENV,
        "verdict": _roll_up([str(row["verdict"]) for row in findings]),
        "findings": findings,
    }


def _env_pins(pinned: Path, home: Path | None) -> bool:
    if home is None:
        return False
    try:
        resolved = home.resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return pinned == resolved or resolved in pinned.parents


def _finding(
    artifact: str,
    verdict: str,
    reason: str,
    left: Path | None,
    right: Path | None,
    *,
    shared_path: str = "",
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "verdict": verdict,
        "reason": reason,
        "reason_text": REASON_TEXT[reason],
        "path_a": str(left) if left is not None else "",
        "path_b": str(right) if right is not None else "",
        "shared_path": shared_path,
    }


def _env_finding(
    variable: str,
    verdict: str,
    reason: str,
    value: str,
    *,
    pins: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "artifact": variable,
        "verdict": verdict,
        "reason": reason,
        "reason_text": REASON_TEXT[reason],
        "value": value,
        "pins": list(pins or []),
    }


def _roll_up(verdicts: list[str]) -> str:
    """A surface is leaked if anything leaked, else unknown if anything is unknown.

    Leak outranks unknown deliberately. A surface that found a real leak and
    could not inspect one artifact is still a leak an operator must close
    first; reporting it as `unknown` would bury the one fact the check has.
    """
    if VERDICT_LEAKED in verdicts:
        return VERDICT_LEAKED
    if VERDICT_UNKNOWN in verdicts or not verdicts:
        return VERDICT_UNKNOWN
    return VERDICT_ISOLATED


__all__ = [
    "PROFILE_ISOLATION_CLAIM_BOUNDARY",
    "PROFILE_ISOLATION_SCHEMA_VERSION",
    "PROFILE_ISOLATION_SURFACES",
    "PROFILE_ISOLATION_VERDICTS",
    "ProfileRef",
    "VERDICT_ISOLATED",
    "VERDICT_LEAKED",
    "VERDICT_UNKNOWN",
    "compare_profile_isolation",
    "parse_profile_ref",
]
