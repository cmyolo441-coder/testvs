"""A pure, testable resolver from a proposed OMH operation to an approval tier.

Before this module, the same shape -- "if not confirmed: raise", "kept unless
--force" -- was written out separately at each dispatch and install boundary:
`require_hermes_child_dispatch_boundary` (`coding/hermes_child_dispatch.py`),
the two `--confirm-dispatch` CLI guards (`commands/hermes_child.py`,
`commands/hermes_child_skill_load.py`), the fanout spawn-depth guard
(`coding/fanout_dispatch.py`), and the installer's overwrite/removal guards
(`install/installer.py`). Each copy had its own wording and its own chance to
drift from the others. `resolve_approval_tier` is the one DECISION function
those sites now ask; every one of them still does its own enforcement (raising
its own error, returning its own refusal payload) because the tier alone does
not know how a given call site should fail.

Three tiers, closed vocabulary: `auto_allowed`, `needs_confirmation`,
`refused`. Two absorbed rules, both from `docs/approval-mode.md`'s six-step
precedence for tool-call tiers (`oh-my-pi`; OMH absorption, P3):

- **Unknown defaults to most-restrictive.** An `operation_class` this table has
  never seen resolves to `refused`, mirroring "tools without an approval
  declaration ... are treated as exec" -- OMH's most restrictive tier is
  `refused`, not `auto_allowed`.
- **Headless rejects rather than hangs.** `needs_confirmation` without
  `confirmed=True` resolves straight to `refused`. This module has no live
  prompt loop to wait on; a caller that cannot supply confirmation now gets a
  refusal now, never a pending state.

Strict security posture (`security_posture.py`, #1196) composes through the
same `POSTURE_MAPPING` / `strict_override` pattern every other tightened
surface already uses: a rule's `posture_key`, when set, names a
`POSTURE_MAPPING` row whose strict value answers "is the confirmation override
even available here". `default` posture always reads `True` (unchanged
behavior); `strict` can read `False`, which turns `needs_confirmation` into
`refused` regardless of `confirmed` -- a strict operator gets no override path
at all for that operation class.
"""

from __future__ import annotations

from typing import NamedTuple

from .security_posture import DEFAULT_POSTURE, strict_override

APPROVAL_TIER_SCHEMA_VERSION = "omh_approval_tier/v1"

TIER_AUTO_ALLOWED = "auto_allowed"
TIER_NEEDS_CONFIRMATION = "needs_confirmation"
TIER_REFUSED = "refused"
APPROVAL_TIERS: tuple[str, ...] = (TIER_AUTO_ALLOWED, TIER_NEEDS_CONFIRMATION, TIER_REFUSED)

UNKNOWN_OPERATION_REASON = "unknown_operation_class"


class ApprovalRule(NamedTuple):
    operation_class: str
    base_tier: str
    # A `security_posture.POSTURE_MAPPING` key whose strict value is a bool:
    # whether the confirmation override is available at all in that posture.
    # None means posture never changes this row's tier.
    posture_key: str | None
    rationale: str


# One row per decision this resolver answers for. `operation_class` is the
# lookup every enforcement site passes; `base_tier` is what the table assigns
# before any posture or confirmation input is applied.
APPROVAL_RULE_TABLE: tuple[ApprovalRule, ...] = (
    ApprovalRule(
        "hermes_child_dispatch",
        TIER_NEEDS_CONFIRMATION,
        None,
        "Isolated local Hermes dispatch requires the fixed ask_before_dispatch policy plus an explicit "
        "operator confirmation (--confirm-dispatch). security_posture.py's dispatch_confirmation_required "
        "row documents that this requirement is already non-tightenable: both postures require it.",
    ),
    ApprovalRule(
        "hermes_child_recursion_depth",
        TIER_REFUSED,
        None,
        "A dispatched agent CLI must never start another isolated Hermes child (the depth-one boundary). "
        "Structural: no confirmation lifts it.",
    ),
    ApprovalRule(
        "fanout_recursion_depth",
        TIER_REFUSED,
        None,
        "Mirrors hermes_child_recursion_depth for the fanout spawn guard: a dispatched agent CLI must not "
        "start another fanout dispatch. Structural: no confirmation lifts it.",
    ),
    ApprovalRule(
        "installer_overwrite_managed_file",
        TIER_AUTO_ALLOWED,
        None,
        "A file OMH already manages is rewritten on every normal install/update; that is what 'managed' "
        "means, so no confirmation is asked for it.",
    ),
    ApprovalRule(
        "installer_overwrite_local_modification",
        TIER_NEEDS_CONFIRMATION,
        "installer_confirmation_override_available",
        "A local edit under a managed skill path is overwritten only with the operator's explicit --force.",
    ),
    ApprovalRule(
        "installer_remove_unowned_plugin_dir",
        TIER_NEEDS_CONFIRMATION,
        "installer_confirmation_override_available",
        "A plugin directory OMH cannot prove it manages is kept unless --force explicitly claims it.",
    ),
)

_RULES_BY_CLASS: dict[str, ApprovalRule] = {rule.operation_class: rule for rule in APPROVAL_RULE_TABLE}


def known_operation_classes() -> tuple[str, ...]:
    """Every `operation_class` the table can resolve, in table order."""
    return tuple(rule.operation_class for rule in APPROVAL_RULE_TABLE)


class ApprovalDecision(NamedTuple):
    operation_class: str
    tier: str
    reason_code: str
    posture: str


def resolve_approval_tier(
    operation_class: str,
    *,
    confirmed: bool = False,
    posture: str = DEFAULT_POSTURE,
) -> ApprovalDecision:
    """The DECISION only: which tier `operation_class` resolves to right now.

    Pure and side-effect free -- this never raises, writes, or spawns. A call
    site compares `decision.tier` against `TIER_AUTO_ALLOWED` and enforces its
    own refusal (its own exception type, its own message, its own refusal
    payload shape); this function only ever answers "what tier".
    """
    rule = _RULES_BY_CLASS.get(operation_class)
    if rule is None:
        return ApprovalDecision(operation_class, TIER_REFUSED, UNKNOWN_OPERATION_REASON, posture)

    if rule.base_tier == TIER_REFUSED:
        return ApprovalDecision(operation_class, TIER_REFUSED, "structural_refusal", posture)

    if rule.base_tier == TIER_AUTO_ALLOWED:
        return ApprovalDecision(operation_class, TIER_AUTO_ALLOWED, "table_default", posture)

    # base_tier is needs_confirmation.
    override_available = (
        True if rule.posture_key is None else strict_override(rule.posture_key, posture, True)
    )
    if not override_available:
        return ApprovalDecision(operation_class, TIER_REFUSED, "strict_posture_override_disabled", posture)
    if confirmed:
        return ApprovalDecision(operation_class, TIER_AUTO_ALLOWED, "confirmed", posture)
    # Headless-rejects-rather-than-hangs: no confirmation channel exists
    # inside a pure function, so an unconfirmed needs_confirmation operation
    # is refused now rather than left pending.
    return ApprovalDecision(operation_class, TIER_REFUSED, "unconfirmed", posture)


def describe_approval_tier_table() -> dict[str, object]:
    """The full rule table, for a status/doctor surface (mirrors `describe_security_posture`)."""
    return {
        "schema_version": APPROVAL_TIER_SCHEMA_VERSION,
        "tiers": list(APPROVAL_TIERS),
        "rows": [
            {
                "operation_class": rule.operation_class,
                "base_tier": rule.base_tier,
                "posture_key": rule.posture_key,
                "rationale": rule.rationale,
            }
            for rule in APPROVAL_RULE_TABLE
        ],
    }
