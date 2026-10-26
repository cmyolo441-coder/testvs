"""Per-family capability enable/disable policy (`omh_capability_policy/v1`).

OMH ships 101 installable skills behind one binary install lever: `core` (9
skills) or `full` (all 101). There is no middle, so a user who wants the
coding handoff surface but not the memory surface has to take both. This
module adds the middle at the unit users already see -- the six capability
families in `families.py` -- so one family can be turned off without
uninstalling OMH and without touching the other five.

Boundaries, in order of importance:

- `CORE_SKILLS` are never disableable. They are the floor `omh doctor` checks
  for; removing them turns a deliberate opt-out into a broken install.
- The policy records which surfaces this install OFFERS. It never disables
  Hermes itself, never claims a skill was uninstalled, and is not execution,
  review, CI, or merge evidence.
- Every persisted value is a scalar or a list of scalars. Nested dicts and
  floats are dropped by the setup-profile correct/restore path, so a policy
  that used them would silently lose fields on the next repair.
- Absent policy means all six families enabled. An install that predates this
  feature keeps working unchanged, and `build_capability_policy()` with no
  arguments is the same thing written down.

The org safety rule source opt-in at the bottom of this module is a second,
independent setup-profile policy that follows the same three rules: scalar
values only, absent means off, and the read rebuilds rather than trusting the
persisted file.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..local_store import atomic_write_json, read_json_object
from ..paths import OmhPaths
from ..skills.catalog import CORE_SKILLS, installable_skill_names
from .families import capability_family_cards, capability_family_projection

CAPABILITY_POLICY_SCHEMA_VERSION = "omh_capability_policy/v1"

CAPABILITY_POLICY_CLAIM_BOUNDARY = (
    "Capability policy records which OMH family surfaces are offered locally; it does not disable "
    "Hermes itself and is not execution, review, CI, or merge evidence."
)

# Casefolded aliases a person actually types, mapped to canonical family ids.
# Labels ("plan and decide") and ids ("plan_and_decide") are resolved
# mechanically in `normalize_family_id`; this table only carries the short
# words that are not derivable from either, so it stays small and reviewable.
_FAMILY_ALIASES: dict[str, str] = {
    "memory": "retain_knowledge",
    "memory management": "retain_knowledge",
    "knowledge": "retain_knowledge",
    "coding": "delegate_coding_and_ship",
    "coding orchestration": "delegate_coding_and_ship",
    "delegate coding": "delegate_coding_and_ship",
    "research": "learn_and_gather",
    "learn": "learn_and_gather",
    "gather": "learn_and_gather",
    "plan": "plan_and_decide",
    "planning": "plan_and_decide",
    "decide": "plan_and_decide",
    "materials": "create_materials_and_visuals",
    "visuals": "create_materials_and_visuals",
    "design": "create_materials_and_visuals",
    "ops": "operate_and_observe",
    "operate": "operate_and_observe",
    "status": "operate_and_observe",
    "observe": "operate_and_observe",
}


class CapabilityPolicyError(ValueError):
    """Raised for an unrecognized family id, with every valid id named."""


def toggleable_family_ids() -> tuple[str, ...]:
    """The six canonical family ids, derived -- never a second literal list."""
    return tuple(str(card["id"]) for card in capability_family_cards())


def family_labels_by_id() -> dict[str, str]:
    return {str(card["id"]): str(card["label"]) for card in capability_family_cards()}


def normalize_family_id(value: str) -> str:
    """Resolve a canonical id, a label, or a short alias to a family id.

    Never guesses: an unrecognized value raises with the full valid list
    rather than falling back to a default, because silently disabling the
    wrong family is worse than refusing.
    """
    raw = str(value or "").strip()
    if not raw:
        raise CapabilityPolicyError(
            f"capability family is required; expected one of {', '.join(toggleable_family_ids())}"
        )
    folded = raw.casefold()
    valid = toggleable_family_ids()
    if folded in {family_id.casefold() for family_id in valid}:
        return next(family_id for family_id in valid if family_id.casefold() == folded)
    underscored = folded.replace(" ", "_").replace("-", "_")
    if underscored in valid:
        return underscored
    for family_id, label in family_labels_by_id().items():
        if folded == label.casefold():
            return family_id
    alias = _FAMILY_ALIASES.get(folded) or _FAMILY_ALIASES.get(folded.replace("_", " "))
    if alias:
        return alias
    raise CapabilityPolicyError(
        f"unsupported capability family: {raw}; expected one of {', '.join(valid)}"
    )


def normalize_family_ids(values: Iterable[str] | None) -> tuple[str, ...]:
    """Normalize, dedupe, and sort so the persisted list is byte-stable."""
    if not values:
        return ()
    return tuple(sorted({normalize_family_id(value) for value in values}))


def build_capability_policy(
    disabled_families: Iterable[str] | None = (),
    *,
    files_removed: bool = False,
) -> dict[str, Any]:
    """Build the policy payload persisted inside setup-profile.json.

    No clock reading: the enclosing setup profile already stamps `updated_at`,
    and a second timestamp nested here would make the payload non-deterministic
    for the tests that compare built policies directly.
    """
    disabled = normalize_family_ids(disabled_families)
    enabled = tuple(family_id for family_id in toggleable_family_ids() if family_id not in disabled)
    return {
        "schema_version": CAPABILITY_POLICY_SCHEMA_VERSION,
        "disabled_families": list(disabled),
        "enabled_families": sorted(enabled),
        "retained_core_skills": sorted(CORE_SKILLS),
        "files_removed": bool(files_removed),
        "claim_boundary": CAPABILITY_POLICY_CLAIM_BOUNDARY,
    }


def read_capability_policy(paths: OmhPaths) -> dict[str, Any]:
    """Read the policy from setup-profile.json; absent means all enabled."""
    profile = read_json_object(paths.setup_profile_path) or {}
    policy = profile.get("capability_policy")
    if not isinstance(policy, dict):
        return build_capability_policy()
    return _coerce_policy(policy)


def write_capability_policy(
    paths: OmhPaths,
    disabled_families: Iterable[str] | None,
    *,
    files_removed: bool = False,
) -> dict[str, Any]:
    """Read-modify-write the policy, preserving every other profile field."""
    policy = build_capability_policy(disabled_families, files_removed=files_removed)
    profile = read_json_object(paths.setup_profile_path) or {}
    profile["capability_policy"] = policy
    atomic_write_json(paths.setup_profile_path, profile, private=True)
    return policy


def disabled_family_ids(policy: dict[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(policy, dict):
        return ()
    valid = set(toggleable_family_ids())
    return tuple(
        sorted(
            {
                str(value)
                for value in policy.get("disabled_families", [])
                if str(value) in valid
            }
        )
    )


def family_is_enabled(policy: dict[str, Any] | None, family_id: str) -> bool:
    return str(family_id) not in set(disabled_family_ids(policy))


def disabled_family_workflows(policy: dict[str, Any] | None) -> tuple[str, ...]:
    """Workflows a disabled family owns, minus the never-removable core."""
    disabled = set(disabled_family_ids(policy))
    if not disabled:
        return ()
    mapping = _workflow_to_family()
    core = set(CORE_SKILLS)
    return tuple(
        sorted(
            name
            for name in installable_skill_names()
            if name not in core and mapping.get(name, "") in disabled
        )
    )


def enabled_workflow_names(policy: dict[str, Any] | None) -> tuple[str, ...]:
    """The availability set handed to `capability_family_projection`."""
    disabled = set(disabled_family_ids(policy))
    names = sorted(installable_skill_names())
    if not disabled:
        return tuple(names)
    mapping = _workflow_to_family()
    core = set(CORE_SKILLS)
    return tuple(
        name for name in names if name in core or mapping.get(name, "") not in disabled
    )


def retained_core_skills(policy: dict[str, Any] | None = None) -> tuple[str, ...]:
    """CORE_SKILLS survive every disable; the policy cannot change this."""
    return tuple(sorted(CORE_SKILLS))


def _workflow_to_family() -> dict[str, str]:
    projection = capability_family_projection()
    mapping = projection.get("workflow_to_family", {})
    if not isinstance(mapping, dict):
        return {}
    return {str(key): str(value) for key, value in mapping.items()}


def _coerce_policy(policy: dict[str, Any]) -> dict[str, Any]:
    """Rebuild from the persisted disable list so an edited file cannot drift.

    A hand-edited `enabled_families` that contradicts `disabled_families` is
    resolved in favour of the disable list, which is the field the CLI writes
    and the only one enforcement reads.
    """
    return build_capability_policy(
        policy.get("disabled_families", []),
        files_removed=bool(policy.get("files_removed", False)),
    )


# --- Org safety rule source opt-in (issue #802) ------------------------------
#
# Same file and same contract as the capability policy above: a scalar-only
# payload inside setup-profile.json, absent meaning off, and a defensive read
# that rebuilds rather than trusting a hand-edited value. It lives here and not
# in `config.yaml` because that file is Hermes-owned; OMH policy belongs in the
# OMH setup profile.
#
# Off is the default because the org level can only DENY. Turning it on is a
# local decision to be bound by a local rule document, and an install that
# predates this key keeps behaving exactly as it did.
ORG_RULE_SOURCE_POLICY_SCHEMA_VERSION = "omh_org_rule_source_policy/v1"

ORG_RULE_SOURCE_POLICY_CLAIM_BOUNDARY = (
    "The org rule source policy records whether this install consults a local org safety rule "
    "document; it is not compliance, execution, review, CI, or merge evidence."
)


def build_org_rule_source_policy(
    *, enabled: bool = False, source_path: str = "", attestation_key_path: str = ""
) -> dict[str, Any]:
    """Build the opt-in payload persisted inside setup-profile.json.

    `attestation_key_path` (issue #805) is where the operator keeps the local
    HMAC-SHA256 key that `coding/safety_rule_attestation.py` verifies the rule
    source against. Blank is the default and means no attestation is required,
    which is why an install that predates the key behaves exactly as it did. A
    path, not a key: the secret itself never enters the setup profile.
    """
    return {
        "schema_version": ORG_RULE_SOURCE_POLICY_SCHEMA_VERSION,
        "enabled": bool(enabled),
        "source_path": str(source_path or "").strip(),
        "attestation_key_path": str(attestation_key_path or "").strip(),
        "claim_boundary": ORG_RULE_SOURCE_POLICY_CLAIM_BOUNDARY,
    }


def read_org_rule_source_policy(paths: OmhPaths) -> dict[str, Any]:
    """Read the opt-in from setup-profile.json; absent means off."""
    profile = read_json_object(paths.setup_profile_path) or {}
    policy = profile.get("org_rule_source_policy")
    if not isinstance(policy, dict):
        return build_org_rule_source_policy()
    return build_org_rule_source_policy(
        enabled=bool(policy.get("enabled", False)),
        source_path=str(policy.get("source_path", "") or ""),
        attestation_key_path=str(policy.get("attestation_key_path", "") or ""),
    )


def write_org_rule_source_policy(
    paths: OmhPaths, *, enabled: bool, source_path: str = "", attestation_key_path: str = ""
) -> dict[str, Any]:
    """Read-modify-write the opt-in, preserving every other profile field."""
    policy = build_org_rule_source_policy(
        enabled=enabled, source_path=source_path, attestation_key_path=attestation_key_path
    )
    profile = read_json_object(paths.setup_profile_path) or {}
    profile["org_rule_source_policy"] = policy
    atomic_write_json(paths.setup_profile_path, profile, private=True)
    return policy


def org_rule_source_enabled(policy: dict[str, Any] | None) -> bool:
    return isinstance(policy, dict) and policy.get("enabled") is True


def org_rule_source_path(policy: dict[str, Any] | None) -> str:
    if not isinstance(policy, dict):
        return ""
    return str(policy.get("source_path", "") or "").strip()


def org_rule_source_attestation_key_path(policy: dict[str, Any] | None) -> str:
    """Where the local attestation key lives, or empty when none is required."""
    if not isinstance(policy, dict):
        return ""
    return str(policy.get("attestation_key_path", "") or "").strip()
