"""Named strict security posture: one switch that bundles the tightened end of
OMH's already-existing safety knobs, instead of an operator hand-tuning each
one across the setup profile.

`OMH_SECURITY=strict` follows the same env-resolution idiom `OMH_LANG` and
`OMH_HOME` already use: unset or blank reads as the safe, unchanged default,
and an unrecognized value is a loud `ValueError` naming the valid choices --
never a silent fallback, because a security knob that fails open on a typo is
worse than no knob at all.

`POSTURE_MAPPING` is the single table every tightened surface reads through.
Each row names the surface module, the value strict applies there, and a
short rationale -- it deliberately does NOT carry the default value: every
consuming surface already owns its default as a pre-existing constant, and
asking this module to restate it would create a second copy that could drift
from the original. `default` posture therefore changes nothing anywhere:
every surface's own constant is untouched, and this module is never consulted
for a value beyond the posture label itself.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, NamedTuple

SECURITY_POSTURE_SCHEMA_VERSION = "omh_security_posture/v1"
SECURITY_POSTURE_ENV_VAR = "OMH_SECURITY"

DEFAULT_POSTURE = "default"
STRICT_POSTURE = "strict"
VALID_POSTURES: tuple[str, ...] = (DEFAULT_POSTURE, STRICT_POSTURE)


class PostureRow(NamedTuple):
    key: str
    surface: str
    strict_value: Any
    rationale: str


# One row per tightened surface. `key` is the lookup a surface module passes
# to `strict_override`/`security_posture_value`; `surface` is the file that
# reads it, for the doctor/status projection; `strict_value` is what
# `strict` applies -- the default lives only in the surface's own constant.
# Two rows (`completion_integrity_refusal_overridable`,
# `dispatch_confirmation_required`) tighten nothing: both gates are already
# non-overridable/always-required in `default` posture, and the row exists so
# the mapping table documents the invariant instead of leaving it unstated.
POSTURE_MAPPING: tuple[PostureRow, ...] = (
    PostureRow(
        "fanout_default_concurrency",
        "src/coding/parallelism_policy.py",
        2,
        "Fewer fanout units run at once by default, so a misjudged contract burns less "
        "provider budget before an operator notices.",
    ),
    PostureRow(
        "fanout_global_concurrency",
        "src/coding/parallelism_policy.py",
        3,
        "The pool-wide ceiling shrinks alongside the per-lane default and clamps any "
        "setup-profile override that would otherwise reopen the width strict just closed.",
    ),
    PostureRow(
        "fanout_lane_budget_default",
        "src/coding/parallelism_policy.py",
        2,
        "The advisory Hermes-native lane budget matches the tightened dispatch pool so the "
        "two numbers an operator reads never disagree.",
    ),
    PostureRow(
        "fanout_max_depth",
        "src/coding/parallelism_policy.py",
        1,
        "Already the minimum that permits one dispatch at all; strict pins it as a hard "
        "ceiling a setup-profile override cannot raise.",
    ),
    PostureRow(
        "fanout_run_spawn_ceiling",
        "src/coding/parallelism_policy.py",
        10,
        "One dispatch run may start far fewer real agent-CLI processes in total, bounding "
        "the worst case of a misjudged contract regardless of pool width.",
    ),
    PostureRow(
        "fanout_max_retries",
        "src/coding/fanout_retry.py",
        0,
        "Zero retries on transient (ambiguous) classifications: a strict operator wants a "
        "surfaced failure, not an automatic second attempt.",
    ),
    PostureRow(
        "verification_escalate_always",
        "src/quality/verification_tiering.py",
        True,
        "Every request escalates to the thorough verification lane, not only ones that "
        "touch a recognized sensitive-path pattern the classifier already knows.",
    ),
    PostureRow(
        "loop_no_progress_cap",
        "src/workflows/goal_loop.py",
        1,
        "The stop ladder's no-progress rung fires after one stalled tick instead of two, so "
        "a stuck loop surfaces sooner.",
    ),
    PostureRow(
        "completion_integrity_refusal_overridable",
        "src/quality/completion_integrity.py",
        False,
        "No override path exists in either posture -- completion-integrity refusals are "
        "already non-overridable; the row documents the invariant rather than changing it.",
    ),
    PostureRow(
        "dispatch_confirmation_required",
        "src/coding/hermes_child_dispatch.py",
        True,
        "Explicit `ask_before_dispatch` confirmation is already required in both postures "
        "(`require_hermes_child_dispatch_boundary`); the row documents the invariant.",
    ),
    PostureRow(
        "installer_confirmation_override_available",
        "src/install/installer.py",
        False,
        "A strict operator gets no `--force` override path at all for overwriting a "
        "locally-modified managed file or reclaiming an unowned plugin directory; only "
        "default posture honors the flag (`approval_tier.resolve_approval_tier`).",
    ),
)

_POSTURE_MAPPING_BY_KEY: dict[str, PostureRow] = {row.key: row for row in POSTURE_MAPPING}


def resolve_security_posture(env: Mapping[str, str] | None = None) -> str:
    """The active posture: `OMH_SECURITY` normalized, or `default` when unset.

    Matches the `OMH_LANG`/`normalize_language` idiom: an unset or blank
    value reads as the safe default, and an unrecognized value is a loud
    refusal naming the valid choices rather than a silent fallback.
    """
    source = os.environ if env is None else env
    raw = str(source.get(SECURITY_POSTURE_ENV_VAR, "") or "").strip().lower()
    if not raw:
        return DEFAULT_POSTURE
    if raw in VALID_POSTURES:
        return raw
    valid = ", ".join(VALID_POSTURES)
    raise ValueError(f"unsupported {SECURITY_POSTURE_ENV_VAR} value: {raw!r}; expected one of {valid}")


def security_posture_value(key: str) -> Any:
    """The `strict` value mapped for `key`. Raises `KeyError` for an unknown key."""
    return _POSTURE_MAPPING_BY_KEY[key].strict_value


def strict_override(key: str, posture: str, default: Any) -> Any:
    """`default` unless `posture` is `strict`, in which case the mapped strict value.

    Each surface passes its OWN pre-existing default constant in; this
    module never states that default itself, so `default` posture can never
    diverge from the value the surface already had -- there is no second
    copy of the number to drift out of sync.
    """
    if posture != STRICT_POSTURE:
        return default
    return security_posture_value(key)


def describe_security_posture(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The active posture plus the full mapping table, for a status/doctor surface."""
    posture = resolve_security_posture(env)
    return {
        "schema_version": SECURITY_POSTURE_SCHEMA_VERSION,
        "env_var": SECURITY_POSTURE_ENV_VAR,
        "posture": posture,
        "valid_postures": list(VALID_POSTURES),
        "rows": [
            {
                "key": row.key,
                "surface": row.surface,
                "strict_value": row.strict_value,
                "active_when_strict": posture == STRICT_POSTURE,
                "rationale": row.rationale,
            }
            for row in POSTURE_MAPPING
        ],
    }
