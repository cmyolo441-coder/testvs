"""Freshness-aware readiness invalidation and the pre-handoff repair card.

The defect this closes (#837): a readiness decision was cached the first time a
coding owner was probed and then reused forever. `_cached_profile` returned any
entry carrying `observed_once`, `updated_at` was written and never read back,
and only `force=True` re-probed. A person who uninstalled `codex`, revoked a
permission, or moved to another checkout still got `ready`, and the handoff was
dispatched into a gap nobody had been told about.

Two derivations live here, both pure and both I/O-free:

* **Freshness.** Age is derived at READ time from the stored `updated_at` plus
  `READINESS_STALE_AFTER_SECONDS`, the way `action_gate._account_state` derives
  `expired` from `ACCOUNT_SIGNAL_STALE_AFTER_SECONDS`. No expiry deadline is
  ever written into the cached record, so shortening the horizon takes effect
  on records already on disk instead of only on records written afterwards.
* **Binding.** A readiness decision is bound to a digest per invalidating axis
  --- profile identity, tool set, permission profile, workspace --- so a
  changed prerequisite invalidates the decision and the verdict can name WHICH
  prerequisite changed. The binding carries no timestamp: it is compared for
  equality, and a wall clock inside a compared payload makes equality a race.

Nothing here decides that a repair happened. The repair card is
`prepared_not_observed`; it names the gap and the command a person may run.

Executor-neutrality (AC3): the card's key set is closed and identical for every
owner. Codex, Claude Code, the Hermes runtime, and generic profiles differ only
in the VALUES of the owner-specific fields (`profile`, `label`,
`missing_prerequisites`, `repair_steps[].command`, `verify_command`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final

from .executors import executor_label
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


PRE_HANDOFF_READINESS_SCHEMA_VERSION: Final = "pre_handoff_readiness/v1"
PRE_HANDOFF_REPAIR_CARD_SCHEMA_VERSION: Final = "pre_handoff_repair_card/v1"
READINESS_BINDING_SCHEMA_VERSION: Final = "executor_readiness_binding/v1"

# Six hours, the same horizon `action_gate.ACCOUNT_SIGNAL_STALE_AFTER_SECONDS`
# gives an account signal and the same one the limit-signal advisory already
# reports as `stale`. Readiness answers a question of the same kind --- "is the
# local environment still the one that was observed" --- so a second, different
# number would only make two surfaces disagree about the same machine.
READINESS_STALE_AFTER_SECONDS: Final = 6 * 60 * 60

# Twenty-four hours, mirroring the window
# `executor_capability_snapshots._local_workflow_errors` already enforces
# between `observed_at` and `recorded_at`. That check runs at RECORD time only;
# this one re-applies the same window at DISPATCH time, against the clock now
# rather than against the moment the snapshot was written.
CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS: Final = 24 * 60 * 60

READINESS_BINDING_AXES: Final = ("profile", "tool", "permission", "workspace")

PRE_HANDOFF_READINESS_STATES: Final = (
    "fresh",
    "never_observed",
    "unbound",
    "changed",
    "expired",
    "evidence_prepared_only",
    "evidence_expired",
)

PRE_HANDOFF_REPAIR_CARD_KEYS: Final = (
    "schema_version",
    "status",
    "profile",
    "label",
    "state",
    "reason_code",
    "reason",
    "changed_axes",
    "missing_prerequisites",
    "repair_steps",
    "verify_command",
    "blocked_action",
    "claim_boundary",
)
PRE_HANDOFF_REPAIR_STEP_KEYS: Final = ("id", "action", "command")
PRE_HANDOFF_REPAIR_STEP_IDS: Final = ("restore_prerequisite", "reobserve_readiness")

PRE_HANDOFF_READINESS_CLAIM_BOUNDARY: Final = (
    "This verdict says only whether a stored readiness observation may still be reused. It is not "
    "dispatch, execution, verification, review, CI, or merge evidence, and a fresh verdict is not "
    "proof that the executor would accept the handoff."
)
PRE_HANDOFF_REPAIR_CARD_CLAIM_BOUNDARY: Final = (
    "This card is prepared repair guidance, not a repair. OMH did not install a tool, change a "
    "permission, rebind a workspace, or re-probe anything; each step is a command a person may "
    "choose to run."
)
PRE_HANDOFF_BLOCKED_ACTION: Final = "coding_handoff_dispatch"

_MAX_REASON_LENGTH: Final = 400
_MAX_TEXT_LENGTH: Final = 240
_MAX_PREREQUISITES: Final = 6
_STATE_REASON_CODES: Final = {
    "fresh": "readiness_fresh",
    "never_observed": "readiness_never_observed",
    "unbound": "readiness_unbound",
    "changed": "readiness_inputs_changed",
    "expired": "readiness_expired",
    "evidence_prepared_only": "capability_evidence_prepared_only",
    "evidence_expired": "capability_evidence_expired",
}
_AXIS_PREREQUISITE_LABELS: Final = {
    "profile": "the executor profile identity this decision was observed against",
    "tool": "the executor command that was observed on PATH",
    "permission": "the safety permission profile that was in force",
    "workspace": "the workspace this decision was observed in",
}


class PreHandoffRepairCardError(ValueError):
    pass


def readiness_binding(
    *,
    profile: str,
    profile_revision: str,
    tool_paths: Sequence[str],
    permission_revision: str,
    workspace_ref: str,
) -> dict[str, Any]:
    """Bind a readiness decision to the inputs whose change invalidates it.

    One digest per axis rather than one digest overall: a single digest can say
    "something changed", and the person then has to guess what. Per-axis
    digests let the verdict name `tool` or `permission` by itself, which is the
    difference between a repair card and a shrug.
    """
    axes = {
        "profile": _axis_digest(profile, profile_revision),
        "tool": _axis_digest(*tool_paths),
        "permission": _axis_digest(permission_revision),
        "workspace": _axis_digest(workspace_ref),
    }
    return {
        "schema_version": READINESS_BINDING_SCHEMA_VERSION,
        "axes": axes,
        "digest": _axis_digest(*(axes[name] for name in READINESS_BINDING_AXES)),
    }


def changed_binding_axes(stored: object, live: Mapping[str, Any]) -> list[str]:
    """Axes whose digest differs between a stored binding and the live one."""
    stored_axes = stored.get("axes") if isinstance(stored, Mapping) else None
    if not isinstance(stored_axes, Mapping):
        return []
    live_axes = live.get("axes")
    live_map = live_axes if isinstance(live_axes, Mapping) else {}
    return [
        axis
        for axis in READINESS_BINDING_AXES
        if str(stored_axes.get(axis, "")) != str(live_map.get(axis, ""))
    ]


def capability_evidence_is_fresh(observed_at: str, now: str = "") -> bool:
    """Whether one capability observation is still inside the #837 evidence horizon.

    Exposed so the owner-fit matcher (#810) reuses this horizon instead of
    introducing a second one; two numbers for the same question would let two
    surfaces disagree about the same machine. Same rule
    `_capability_evidence_state` already applies: an empty, unreadable, or
    future stamp is older than the horizon, never fresher.
    """
    age = _stamp_age_seconds(observed_at, now, CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS)
    return age <= CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS


def evaluate_pre_handoff_readiness(
    *,
    profile: str,
    cached: Mapping[str, Any] | None,
    binding: Mapping[str, Any],
    capability_snapshot: Mapping[str, Any] | None = None,
    now: str = "",
) -> dict[str, Any]:
    """Whether a stored readiness decision may still be reused, and why not.

    `usable` is deliberately narrower than "the executor is ready": it says the
    stored observation still describes the current machine. The caller keeps
    the status it observed when `usable` is true and downgrades it otherwise;
    nothing here can promote a decision to `ready`.

    `now` is a parameter so the derivation is deterministic under test. An
    unreadable or future `updated_at` reads as older than every horizon, for
    the reason `action_gate._stamp_age_seconds` gives: neither can be shown to
    be fresh, and clamping would turn editing a stored timestamp into a way to
    widen the window from disk.

    `capability_snapshot` is optional because most readiness calls have no
    snapshot to consult. When one IS supplied it is authoritative: a snapshot
    with no host-observed capability is prepared-only evidence and can never
    yield a usable decision, which is the central evidence rule of this repo
    applied at dispatch time instead of only at record time.
    """
    state, changed, age = _readiness_state(
        cached=cached,
        binding=binding,
        capability_snapshot=capability_snapshot,
        now=now,
    )
    return {
        "schema_version": PRE_HANDOFF_READINESS_SCHEMA_VERSION,
        "profile": profile,
        "state": state,
        "usable": state == "fresh",
        "reason_code": _STATE_REASON_CODES[state],
        "reason": _reason_text(state, profile, changed),
        "changed_axes": changed,
        "age_seconds": age,
        "stale_after_seconds": READINESS_STALE_AFTER_SECONDS,
        "claim_boundary": PRE_HANDOFF_READINESS_CLAIM_BOUNDARY,
    }


def build_pre_handoff_repair_card(
    verdict: Mapping[str, Any],
    *,
    repair_command: str,
) -> dict[str, Any]:
    """The deterministic repair card for a verdict that blocked a handoff.

    `repair_command` is the one owner-specific value the caller must supply:
    what a person runs to put the missing prerequisite back. Everything else is
    derived from the verdict, so two owners blocked for the same reason produce
    cards with the same keys and the same shape.
    """
    state = str(verdict.get("state", ""))
    if state not in _STATE_REASON_CODES:
        raise PreHandoffRepairCardError(f"unsupported pre-handoff readiness state: {state}")
    if state == "fresh":
        raise PreHandoffRepairCardError("a fresh readiness verdict has no gap to repair")
    profile = str(verdict.get("profile", ""))
    changed = [str(axis) for axis in verdict.get("changed_axes", []) if isinstance(axis, str)]
    card = {
        "schema_version": PRE_HANDOFF_REPAIR_CARD_SCHEMA_VERSION,
        "status": "prepared_not_observed",
        "profile": profile,
        "label": executor_label(profile),
        "state": state,
        "reason_code": str(verdict.get("reason_code", "")),
        "reason": str(verdict.get("reason", ""))[:_MAX_REASON_LENGTH],
        "changed_axes": changed,
        "missing_prerequisites": _missing_prerequisites(state, changed),
        "repair_steps": [
            {
                "id": "restore_prerequisite",
                "action": _restore_action(state, changed),
                "command": repair_command[:_MAX_TEXT_LENGTH],
            },
            {
                "id": "reobserve_readiness",
                "action": (
                    "Re-observe readiness once the prerequisite is back. A stale decision is never "
                    "refreshed on its own; forcing the probe is the only way to replace it."
                ),
                "command": f"omh coding executor-readiness --executor {profile} --force",
            },
        ],
        "verify_command": f"omh coding executor-readiness --executor {profile}",
        "blocked_action": PRE_HANDOFF_BLOCKED_ACTION,
        "claim_boundary": PRE_HANDOFF_REPAIR_CARD_CLAIM_BOUNDARY,
    }
    errors = validate_pre_handoff_repair_card(card)
    if errors:
        raise PreHandoffRepairCardError("; ".join(errors))
    return card


def validate_pre_handoff_repair_card(card: Mapping[str, Any]) -> list[str]:
    """Both directions: no key of the closed set missing, no key outside it."""
    expected = set(PRE_HANDOFF_REPAIR_CARD_KEYS)
    present = {str(key) for key in card}
    errors: list[str] = []
    missing = expected - present
    if missing:
        errors.append(f"repair card is missing required keys: {', '.join(sorted(missing))}")
    unexpected = present - expected
    if unexpected:
        errors.append(f"repair card contains unsupported keys: {', '.join(sorted(unexpected))}")
    if card.get("schema_version") != PRE_HANDOFF_REPAIR_CARD_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PRE_HANDOFF_REPAIR_CARD_SCHEMA_VERSION}")
    if card.get("status") != "prepared_not_observed":
        errors.append("status must be prepared_not_observed; the card never claims a repair ran")
    if card.get("blocked_action") != PRE_HANDOFF_BLOCKED_ACTION:
        errors.append(f"blocked_action must be {PRE_HANDOFF_BLOCKED_ACTION}")
    state = card.get("state")
    if state not in PRE_HANDOFF_READINESS_STATES or state == "fresh":
        errors.append("state must name a blocking pre-handoff readiness state")
    elif card.get("reason_code") != _STATE_REASON_CODES[str(state)]:
        errors.append("reason_code must match the state it was derived from")
    errors.extend(_bounded_text_errors(card))
    errors.extend(_axis_list_errors(card.get("changed_axes")))
    errors.extend(_prerequisite_errors(card.get("missing_prerequisites")))
    errors.extend(_repair_step_errors(card.get("repair_steps")))
    return errors


def _bounded_text_errors(card: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field, limit in (
        ("profile", _MAX_TEXT_LENGTH),
        ("label", _MAX_TEXT_LENGTH),
        ("reason", _MAX_REASON_LENGTH),
        ("verify_command", _MAX_TEXT_LENGTH),
    ):
        value = card.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            errors.append(f"{field} must be a nonempty string of at most {limit} characters")
    return errors


def _axis_list_errors(value: object) -> list[str]:
    if not isinstance(value, list):
        return ["changed_axes must be a list"]
    unknown = [str(axis) for axis in value if axis not in READINESS_BINDING_AXES]
    if unknown:
        return [f"changed_axes contains unsupported axes: {', '.join(sorted(unknown))}"]
    if len(value) != len(set(value)):
        return ["changed_axes must not repeat an axis"]
    return []


def _prerequisite_errors(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["missing_prerequisites must be a nonempty list"]
    if len(value) > _MAX_PREREQUISITES:
        return [f"missing_prerequisites must hold at most {_MAX_PREREQUISITES} entries"]
    if any(not isinstance(entry, str) or not entry.strip() or len(entry) > _MAX_TEXT_LENGTH for entry in value):
        return ["missing_prerequisites entries must be nonempty bounded strings"]
    return []


def _repair_step_errors(value: object) -> list[str]:
    if not isinstance(value, list):
        return ["repair_steps must be a list"]
    if [step.get("id") if isinstance(step, Mapping) else None for step in value] != list(PRE_HANDOFF_REPAIR_STEP_IDS):
        return [f"repair_steps must be exactly {', '.join(PRE_HANDOFF_REPAIR_STEP_IDS)} in order"]
    errors: list[str] = []
    for step in value:
        unexpected = {str(key) for key in step} - set(PRE_HANDOFF_REPAIR_STEP_KEYS)
        if unexpected:
            errors.append(f"repair step contains unsupported keys: {', '.join(sorted(unexpected))}")
        for field in ("action", "command"):
            text = step.get(field)
            if not isinstance(text, str) or not text.strip() or len(text) > _MAX_TEXT_LENGTH:
                errors.append(f"repair step {field} must be a nonempty string of at most {_MAX_TEXT_LENGTH} characters")
    return errors


def _readiness_state(
    *,
    cached: Mapping[str, Any] | None,
    binding: Mapping[str, Any],
    capability_snapshot: Mapping[str, Any] | None,
    now: str,
) -> tuple[str, list[str], int]:
    """(state, changed axes, age in seconds) in a fixed precedence order.

    Precedence is deliberate: an absent observation is reported before an
    expired one, and a changed prerequisite before an expired clock, because a
    person repairs the named prerequisite and a fresh probe follows from that.
    Capability evidence is checked first of all --- prepared-only evidence is
    not weaker evidence, it is the absence of evidence, and no amount of
    freshness converts it.
    """
    if not isinstance(cached, Mapping) or cached.get("observed_once") is not True:
        return "never_observed", [], _UNKNOWN_AGE_SECONDS
    if capability_snapshot is not None:
        evidence_state, evidence_age = _capability_evidence_state(capability_snapshot, now)
        if evidence_state is not None:
            return evidence_state, [], evidence_age
    age = _stamp_age_seconds(str(cached.get("updated_at", "") or ""), now, READINESS_STALE_AFTER_SECONDS)
    stored_binding = cached.get("readiness_binding")
    if not isinstance(stored_binding, Mapping) or not isinstance(stored_binding.get("axes"), Mapping):
        return "unbound", [], age
    changed = changed_binding_axes(stored_binding, binding)
    if changed:
        return "changed", changed, age
    if age > READINESS_STALE_AFTER_SECONDS:
        return "expired", [], age
    return "fresh", [], age


def _capability_evidence_state(
    snapshot: Mapping[str, Any],
    now: str,
) -> tuple[str | None, int]:
    capabilities = snapshot.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return "evidence_prepared_only", _UNKNOWN_AGE_SECONDS
    observed = [
        value
        for value in capabilities.values()
        if isinstance(value, Mapping) and value.get("status") == "host_observed"
    ]
    if not observed:
        return "evidence_prepared_only", _UNKNOWN_AGE_SECONDS
    ages = [
        _stamp_age_seconds(
            str(value.get("observed_at", "") or ""),
            now,
            CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS,
        )
        for value in observed
    ]
    oldest = max(ages)
    if oldest > CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS:
        return "evidence_expired", oldest
    return None, oldest


def _reason_text(state: str, profile: str, changed: Sequence[str]) -> str:
    label = executor_label(profile)
    match state:
        case "fresh":
            return f"The stored readiness observation for {label} still matches this machine."
        case "never_observed":
            return f"Readiness for {label} has never been observed, so there is nothing to reuse."
        case "unbound":
            return (
                f"The stored readiness observation for {label} predates input binding, so no changed "
                "prerequisite can be ruled out."
            )
        case "changed":
            named = ", ".join(changed)
            return (
                f"A prerequisite changed since readiness for {label} was observed: {named}. The stored "
                "decision no longer describes this machine."
            )
        case "expired":
            hours = READINESS_STALE_AFTER_SECONDS // 3600
            return (
                f"The stored readiness observation for {label} is older than {hours} hours, so it is "
                "expired rather than current."
            )
        case "evidence_prepared_only":
            return (
                f"The capability snapshot for {label} carries no host-observed capability, so it is "
                "prepared evidence only and cannot support a ready decision."
            )
        case _:
            hours = CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS // 3600
            return (
                f"The capability evidence for {label} was observed more than {hours} hours ago, so it "
                "is expired rather than current."
            )


def _missing_prerequisites(state: str, changed: Sequence[str]) -> list[str]:
    if changed:
        return [_AXIS_PREREQUISITE_LABELS[axis] for axis in changed]
    match state:
        case "never_observed":
            return ["a first readiness observation for this executor"]
        case "unbound":
            return ["a readiness observation bound to the current profile, tool, permission, and workspace"]
        case "expired":
            return [f"a readiness observation newer than {READINESS_STALE_AFTER_SECONDS // 3600} hours"]
        case "evidence_prepared_only":
            return ["a host-observed capability snapshot instead of prepared-only capability evidence"]
        case _:
            return [
                "a capability observation newer than "
                f"{CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS // 3600} hours"
            ]


def _restore_action(state: str, changed: Sequence[str]) -> str:
    if changed:
        named = ", ".join(changed)
        return f"Put back what changed on this machine ({named}), then confirm it with the command below."
    if state.startswith("evidence_"):
        return "Record a host-observed capability snapshot for this executor before handing work to it."
    return "Confirm the executor prerequisite is in place on this machine before handing work to it."


_UNKNOWN_AGE_SECONDS: Final = -1


def _stamp_age_seconds(stamp: str, now: str, horizon: int) -> int:
    """Age of an ISO-8601 stamp in seconds, or a value past the given horizon."""

    def _parse(value: str) -> datetime | None:
        text = value.strip().replace("Z", "+00:00")
        if not text:
            return None
        try:
            parsed = strict_fromisoformat(text)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    observed = _parse(stamp)
    if observed is None:
        return horizon + 1
    reference = _parse(now) or datetime.now(timezone.utc)
    age = int((reference - observed).total_seconds())
    return horizon + 1 if age < 0 else age


def _axis_digest(*parts: str) -> str:
    # A joined string rather than JSON: this digest runs on the prepared
    # readiness path, where the efficiency contract forbids JSON serialization
    # (see `skill_governance.policy_decision_digest`). The unit separator
    # cannot appear in a path or a revision, so no two axis inputs can collide
    # by concatenation.
    return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()
