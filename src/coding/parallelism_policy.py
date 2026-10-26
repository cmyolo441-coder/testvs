"""Execution-concurrency policy for the opt-in fanout dispatch path.

Defaults mirror the pi-family runtime's task engine (per-lane default 5, global ceiling 8) on
owner direction. Every value is a setup-profile tunable under `parallelism`,
read with the same validated-override-plus-disclosure shape as the memory
policy's cadence block: an invalid stored value falls back to the default
and is named in `ignored_keys` instead of failing the read or silently
winning. OMH core still makes no LLM or network calls — this policy only
sizes the explicit `omh coding fanout dispatch` subprocess pool and its
per-owner lanes, and states the advisory lane budget a Hermes-native run
may read; it never patches Hermes and never enforces anything there.

The same block also carries the spawn guard — `max_depth` and
`run_spawn_ceiling` — because both bound the same one subprocess exception
from a different direction: the pool width bounds how many units run at once,
the ceiling bounds how many are ever started, and the depth cap bounds whether
a dispatched agent CLI may dispatch again at all. `max_retries` rides along
for the same reason, sized by `fanout_retry.FANOUT_MAX_RETRIES`.

`read_parallelism_policy` applies the active `OMH_SECURITY` posture last, on
top of the stored profile: `default` changes nothing, `strict` clamps every
tunable here (and `max_retries`) to the ceiling named in
`system.security_posture.POSTURE_MAPPING`, disclosed the same
clamp-and-name way an out-of-range profile value already is.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..profiles.setup import (
    PARALLELISM_DEFAULTS,
    PARALLELISM_POLICY_SCHEMA_VERSION,
    read_setup_profile,
)
from ..system.paths import OmhPaths
from ..system.security_posture import (
    DEFAULT_POSTURE,
    STRICT_POSTURE,
    resolve_security_posture,
    strict_override,
)
from .executors import canonical_executor_profile
from .fanout_retry import FANOUT_MAX_RETRIES

# One config typo must not turn the fanout pool into a fork bomb: 32 is far
# above any real local agent-CLI fleet while still catching a stray 500.
_PARALLELISM_MAX = 32

# Nested dispatch is the other fork-bomb shape, and the pool width does not
# bound it: a dispatched agent CLI that reads the word "fanout" can invoke
# `omh coding fanout dispatch` itself, and each of ITS units can do the same.
# Depth 0 is the operator's own invocation; `max_depth` 1 therefore means a
# dispatched child never dispatches again. The reader's ceiling of 4 exists for the same
# reason as `_PARALLELISM_MAX`: a typo must not authorize unbounded nesting.
FANOUT_MAX_DEPTH_DEFAULT = 1
_FANOUT_MAX_DEPTH_CEILING = 4

# Total real agent-CLI spawns ONE dispatch run may start, across every unit
# and every retry of the pool. The pool width bounds how many run at once;
# this bounds how many are ever started, which is the number that matters
# when a contract is larger than anyone intended.
FANOUT_RUN_SPAWN_CEILING_DEFAULT = 60
_FANOUT_RUN_SPAWN_CEILING_MAX = 512

_COUNT_KEYS = ("default_concurrency", "global_concurrency", "lane_budget_default")

# Every validated integer tunable and the inclusive bounds it is read within.
# Widths share `_PARALLELISM_MAX`; the two spawn-guard tunables have their own
# ranges because they count different things.
_BOUNDED_KEYS: dict[str, tuple[int, int]] = {
    **{key: (1, _PARALLELISM_MAX) for key in _COUNT_KEYS},
    "max_depth": (1, _FANOUT_MAX_DEPTH_CEILING),
    "run_spawn_ceiling": (1, _FANOUT_RUN_SPAWN_CEILING_MAX),
}

PARALLELISM_CLAIM_BOUNDARY = (
    "Parallelism policy sizes the explicit `omh coding fanout dispatch` "
    "subprocess pool only. Hermes-native lanes read `lane_budget_default` as "
    "advisory context; OMH never enforces a lane count inside Hermes."
)


def _bounded_count(policy: dict[str, Any], key: str) -> int | None:
    """One validated integer tunable from a policy mapping, or None."""
    low, high = _BOUNDED_KEYS[key]
    value = policy.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
        return value
    return None


def build_parallelism_policy() -> dict[str, Any]:
    return {
        "schema_version": PARALLELISM_POLICY_SCHEMA_VERSION,
        **dict(PARALLELISM_DEFAULTS),
        "max_depth": FANOUT_MAX_DEPTH_DEFAULT,
        "run_spawn_ceiling": FANOUT_RUN_SPAWN_CEILING_DEFAULT,
        "per_owner": {},
        "ignored_keys": [],
        "claim_boundary": PARALLELISM_CLAIM_BOUNDARY,
    }


def read_parallelism_policy(paths: OmhPaths, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The effective policy: stored `parallelism` overrides on the defaults,
    then the active security posture's ceilings.

    `env` is injected the same way `dispatch_fanout`'s own spawn-guard read
    is, so the posture resolution is exercised without mutating process
    environment in a test. Posture resolution can raise `ValueError` for an
    unrecognized `OMH_SECURITY` value; unlike an invalid stored profile value
    (which falls back and is named in `ignored_keys`), an invalid posture is
    a loud refusal, not a silent fallback -- callers that reach this from a
    CLI command must catch it and convert it to a clean CLI error.
    """
    policy = build_parallelism_policy()
    policy["max_retries"] = FANOUT_MAX_RETRIES
    posture = resolve_security_posture(env)
    setup = read_setup_profile(paths)
    stored = setup.get("parallelism") if isinstance(setup, dict) else None
    if not isinstance(stored, dict):
        return _apply_security_posture(policy, posture)
    ignored: list[str] = []
    for key in _BOUNDED_KEYS:
        if key not in stored:
            continue
        value = _bounded_count(stored, key)
        if value is None:
            ignored.append(key)
        else:
            policy[key] = value
    raw_owners = stored.get("per_owner")
    if isinstance(raw_owners, dict):
        for owner, value in raw_owners.items():
            # An owner an older release wrote reads as the current profile.
            label = canonical_executor_profile(owner.strip()) if isinstance(owner, str) else ""
            width = (
                value
                if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= _PARALLELISM_MAX
                else None
            )
            if label and width is not None:
                policy["per_owner"][label] = width
            else:
                ignored.append("per_owner." + (label or repr(owner)))
    elif "per_owner" in stored:
        ignored.append("per_owner")
    # A pool wider than the run's whole spawn budget would open lanes the
    # ceiling can never fill. Clamped first so the per-lane clamp below reads
    # the width the pool will actually run with. Same disclose-don't-refuse
    # posture as every other clamp here.
    if policy["global_concurrency"] > policy["run_spawn_ceiling"]:
        policy["global_concurrency_clamped_from"] = policy["global_concurrency"]
        policy["global_concurrency"] = policy["run_spawn_ceiling"]
    # A per-lane default above the global ceiling would promise lanes the
    # pool cannot grant; clamp and disclose rather than refuse the profile.
    if policy["default_concurrency"] > policy["global_concurrency"]:
        policy["default_concurrency_clamped_from"] = policy["default_concurrency"]
        policy["default_concurrency"] = policy["global_concurrency"]
    policy["ignored_keys"] = ignored
    return _apply_security_posture(policy, posture)


# Every numeric tunable the strict posture ceilings apply to, paired with its
# `security_posture` mapping-table key. Order matches the cascade above so a
# clamp report reads in the same "width, then guard, then retries" order an
# operator already sees the rest of this policy in.
_SECURITY_CLAMPED_KEYS: tuple[tuple[str, str], ...] = (
    ("default_concurrency", "fanout_default_concurrency"),
    ("global_concurrency", "fanout_global_concurrency"),
    ("lane_budget_default", "fanout_lane_budget_default"),
    ("max_depth", "fanout_max_depth"),
    ("run_spawn_ceiling", "fanout_run_spawn_ceiling"),
    ("max_retries", "fanout_max_retries"),
)


def _apply_security_posture(policy: dict[str, Any], posture: str) -> dict[str, Any]:
    """Tighten `policy` to the active posture's ceilings, in place.

    `default` posture returns `policy` completely unchanged beyond recording
    the label -- every strict ceiling in `_SECURITY_CLAMPED_KEYS` is defined
    to be at or below its surface's own default, so the clamp is a no-op
    against a default-posture read and only ever lowers a strict one.
    """
    policy["security_posture"] = posture
    if posture != STRICT_POSTURE:
        return policy
    for key, mapping_key in _SECURITY_CLAMPED_KEYS:
        ceiling = strict_override(mapping_key, posture, policy[key])
        if policy[key] > ceiling:
            policy[f"{key}_security_clamped_from"] = policy[key]
            policy[key] = ceiling
    return policy


def resolve_fanout_concurrency(policy: dict[str, Any], requested: int | None) -> dict[str, Any]:
    """The pool width one dispatch runs with, and where it came from.

    An explicit `--concurrency` flag wins over the policy default; the
    global ceiling always clamps, and a clamp is disclosed so the operator
    sees the requested and applied widths side by side.
    """
    global_cap = int(policy.get("global_concurrency", PARALLELISM_DEFAULTS["global_concurrency"]))
    if requested is None:
        wanted = int(policy.get("default_concurrency", PARALLELISM_DEFAULTS["default_concurrency"]))
        source = "policy_default"
    else:
        wanted = max(1, int(requested))
        source = "cli_flag"
    resolution: dict[str, Any] = {
        "schema_version": PARALLELISM_POLICY_SCHEMA_VERSION,
        "requested": requested,
        "applied": min(wanted, global_cap),
        "source": source,
        "global_concurrency": global_cap,
        "clamped": wanted > global_cap,
        # The reader's disclosures ride along so the dispatch record — the
        # one surface an operator reads after the fact — names an ignored
        # profile key or a clamped default instead of discarding the finding.
        "ignored_keys": list(policy.get("ignored_keys", [])),
        # The spawn guard rides in the same block the dispatch summary already
        # records, so the one surface an operator reads after the fact answers
        # "how wide", "how deep", and "how many in total" together.
        "max_depth": int(policy.get("max_depth", FANOUT_MAX_DEPTH_DEFAULT)),
        "run_spawn_ceiling": int(policy.get("run_spawn_ceiling", FANOUT_RUN_SPAWN_CEILING_DEFAULT)),
        "max_retries": int(policy.get("max_retries", FANOUT_MAX_RETRIES)),
        "security_posture": str(policy.get("security_posture", DEFAULT_POSTURE)),
    }
    global_clamped_from = policy.get("global_concurrency_clamped_from")
    if global_clamped_from is not None:
        resolution["global_concurrency_clamped_from"] = global_clamped_from
    clamped_from = policy.get("default_concurrency_clamped_from")
    if source == "policy_default" and clamped_from is not None:
        resolution["clamped"] = True
        resolution["policy_clamped_from"] = clamped_from
    return resolution
