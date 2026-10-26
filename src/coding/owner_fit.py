"""Owner fit: what the accepted plan requires against what a coding owner can prove.

The defect this closes (#810). `routing.coding_route_actions.resolve_coding_route_decision`
picks a coding owner from four inputs and nothing else --- an owner named in the request
envelope, an owner named in the message text, a recorded setup preference, and a keyword
route-family cue. It never reads a capability snapshot, never consults a readiness probe,
and never looks at what the accepted plan needs; its payload carries no requirement, no
gap, and no unknown. A person could therefore name an owner, be told the handoff was
prepared, and only discover mid-execution that the owner cannot isolate a worktree, run
parallel lanes, or invoke the routed workflow at all.

Two derivations live here. Both are pure, both are deterministic, and neither reads free
text:

* **Requirements** come from the accepted plan's DECLARED fields only --- the routed
  workflow, the workspace-binding strategy, and the work-owner mode
  (`ACCEPTED_PLAN_FIELDS`). The message is never parsed here, so a requirement cannot
  appear because of a word somebody happened to use. The vocabulary is the one the
  capability snapshot already speaks (`KNOWN_CAPABILITY_NAMES` plus `local_workflow`), so a
  requirement and the evidence that answers it always share a name.
* **Classification** matches one requirement against one owner's recorded capability
  snapshot and lands in exactly one of three states, kept distinct on purpose:

  - `met` --- fresh host-observed evidence that the owner has the capability.
  - `unmet` --- fresh host-observed evidence that the owner does NOT have it.
  - `unknown` --- no snapshot, no record of that capability, prepared-only evidence,
    evidence scoped to something else, or evidence too old to describe this machine now.

  `unknown` is not a quiet `met` and not a quiet `unmet`. Its consequence is stated and
  tested: an owner carrying an unknown required capability is `unproven`. `unproven` is
  never recommended, which keeps AC1 strict, and it is never reported as a blocking gap,
  which keeps the report honest --- the person is told which observation is missing
  instead of being handed a guess.

Freshness reuses the #837 horizon through
`pre_handoff_readiness.capability_evidence_is_fresh`; adding a second horizon would let two
surfaces disagree about the same machine. Expired evidence classifies `unknown` whatever it
recorded, including a recorded `unavailable`. That is safe rather than permissive: an
expired observation is not knowledge of the present, and `unknown` still yields `unproven`,
which is still not recommended.

**An explicitly named owner is still honoured, and that is a deliberate decision.**
`resolve_coding_route_decision` is unchanged: naming an owner still selects it and still
prepares that owner's handoff. Owner fit is additive. When the named owner cannot do the
work the report says so --- `named_owner_honoured` stays true, `named_owner_gap` names the
capabilities --- and the owner is simply absent from `recommended_owners`. So Hermes never
RECOMMENDS an owner with a known unmet capability, and it never silently overrides a person
who named one. Silently rerouting was the rejected alternative: a stated preference
replaced without a word is an invisible decision, and the user would learn about the swap
from the handoff rather than from Hermes.

Executor-neutrality (AC3): classification reads the snapshot and never the owner id. The
only owner-identity fields in a fit record are `owner` and `label`
(`OWNER_FIT_OWNER_IDENTITY_KEYS`); `owner_fit_without_owner_identity` strips them, and two
owners with equal capability evidence produce equal projections. Every reason string is
brand-neutral for the same reason.

Nothing here is execution. Both artifacts are `prepared_not_observed`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from .executor_capability_snapshots import (
    KNOWN_CAPABILITY_NAMES,
    LOCAL_WORKFLOW_CAPABILITY_NAME,
    executor_capability_snapshot_path,
    read_matching_executor_capability_snapshot,
)
from .executors import executor_label
from .media_handoff_capabilities import modality_requirements
from .pre_handoff_readiness import (
    CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS,
    capability_evidence_is_fresh,
)


OWNER_FIT_SCHEMA_VERSION: Final = "coding_owner_fit/v1"
OWNER_FIT_REPORT_SCHEMA_VERSION: Final = "coding_owner_fit_report/v1"
OWNER_FIT_STATUS: Final = "prepared_not_observed"

# The declared plan fields the derivation reads. Three, and no more: every one
# of them is a value the delegation build already decided and wrote down, so a
# requirement can always be traced back to a field rather than to a phrase.
# `review_required`, `dispatch_policy`, and the acceptance/verification lists
# are deliberately NOT read --- none of them names a capability from the
# snapshot vocabulary, and mapping prose onto a capability is exactly the
# heuristic this module exists to avoid.
ACCEPTED_PLAN_FIELDS: Final = ("workflow", "work_owner_mode", "isolation_strategy", "input_representation", "input_route")

_WORKTREE_REQUIRED_STRATEGY: Final = "worktree_required"
_RUNTIME_HANDOFF_MODE: Final = "runtime_handoff"

# Routed workflow -> the capabilities that workflow's lane runs on. Only lanes
# whose meaning IS the capability appear here: a `team` lane is parallel agents,
# a `loop` lane is recurring work. Guarded by tests on both sides --- every key
# is a routable workflow, every value is in the snapshot vocabulary.
WORKFLOW_CAPABILITY_REQUIREMENTS: Final[dict[str, tuple[str, ...]]] = {
    "browser-operator": ("browser_or_computer_use",),
    "loop": ("scheduled_or_recurring_work", "long_running_continuation"),
    "parallel-tools": ("parallel_agents",),
    "ultrawork": ("parallel_agents", "background_work"),
    "visual-qa": ("visual_qa",),
}

OWNER_FIT_CLASSIFICATIONS: Final = ("met", "unmet", "unknown")
OWNER_FIT_VERDICTS: Final = ("ready", "unproven", "blocked")
OWNER_FIT_EVIDENCE_STATUSES: Final = (
    "no_snapshot",
    "not_recorded",
    "prepared",
    "unknown",
    "host_observed",
    "unavailable",
)

# One table, one direction: a reason code determines its classification. Nothing
# else may set a classification, so "why" and "what" cannot drift apart.
OWNER_FIT_REASON_CODES: Final[dict[str, str]] = {
    "no_capability_snapshot": "unknown",
    "capability_not_recorded": "unknown",
    "evidence_prepared_only": "unknown",
    "evidence_status_unknown": "unknown",
    "evidence_scope_mismatch": "unknown",
    "evidence_expired": "unknown",
    "host_observed_available": "met",
    "host_observed_unavailable": "unmet",
}

OWNER_FIT_REQUIREMENT_KEYS: Final = (
    "capability",
    "source_field",
    "source_value",
    "scope",
    "reason",
)
OWNER_FIT_CLASSIFICATION_KEYS: Final = (
    "capability",
    "source_field",
    "source_value",
    "classification",
    "reason_code",
    "reason",
    "evidence_status",
    "evidence_ref",
    "evidence_observed_at",
)
OWNER_FIT_KEYS: Final = (
    "schema_version",
    "status",
    "owner",
    "label",
    "verdict",
    "recommendable",
    "capabilities",
    "met",
    "unmet",
    "unknown",
    "reason",
    "next_action",
    "claim_boundary",
)
OWNER_FIT_REPORT_KEYS: Final = (
    "schema_version",
    "status",
    "requirements",
    "owners",
    "recommended_owners",
    "unproven_owners",
    "blocked_owners",
    "named_owner",
    "named_owner_honoured",
    "named_owner_gap",
    "next_action",
    "claim_boundary",
)

# The only fields of a fit record that name an owner. AC3 is the property that
# removing exactly these makes two equally-evidenced owners indistinguishable.
OWNER_FIT_OWNER_IDENTITY_KEYS: Final = ("owner", "label")

OWNER_FIT_NEXT_ACTIONS: Final[dict[str, str]] = {
    "ready": "prepare_handoff_for_this_owner",
    "unproven": "record_capability_evidence",
    "blocked": "close_the_capability_gap_or_choose_another_owner",
}
OWNER_FIT_REPORT_NEXT_ACTIONS: Final = (
    "prepare_handoff_for_named_owner",
    "confirm_named_owner_despite_capability_gap",
    "choose_from_recommended_owners",
    "record_capability_evidence",
)

OWNER_FIT_CLAIM_BOUNDARY: Final = (
    "Owner fit compares the accepted plan's declared requirements against recorded capability "
    "evidence. It is not dispatch, execution, verification, review, CI, or merge evidence, and a "
    "ready verdict is not proof that the owner would accept or finish the handoff."
)
OWNER_FIT_REPORT_CLAIM_BOUNDARY: Final = (
    "This report recommends prepared coding owners from recorded capability evidence only. It is "
    "not dispatch, execution, verification, review, CI, or merge evidence, it never removes an "
    "owner the user explicitly named, and it never claims a capability nobody observed."
)

# Canonical references for the cases where the snapshot supplies none. Every
# classification carries a nonempty `evidence_ref` so "which evidence decided
# this" is always answerable, including when the answer is "there was none".
_NO_SNAPSHOT_EVIDENCE_REF: Final = "capability_snapshot:none"
_NOT_RECORDED_EVIDENCE_REF: Final = "capability_snapshot:not_recorded"
_STATUS_ONLY_EVIDENCE_REF: Final = "capability_snapshot:status_only"

_MAX_TEXT_LENGTH: Final = 240
_MAX_REASON_LENGTH: Final = 400
_MAX_REQUIREMENTS: Final = 12
_MAX_OWNERS: Final = 12
_MAX_SCOPE_ITEMS: Final = 4


class OwnerFitError(ValueError):
    pass


def derive_plan_capability_requirements(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """The capabilities an accepted plan requires, derived from its declared fields.

    Deterministic and order-stable: workflow-derived requirements first in the
    catalog's own order, then the workspace binding, then the runtime workflow
    binding. A capability already required by an earlier rule is not repeated;
    the first rule that asked for it owns the explanation, so `source_field`
    always names one field rather than a set.
    """
    unexpected = sorted(str(key) for key in plan if str(key) not in ACCEPTED_PLAN_FIELDS)
    if unexpected:
        raise OwnerFitError(f"accepted plan contains unsupported fields: {', '.join(unexpected)}")
    workflow = str(plan.get("workflow", "") or "").strip()
    work_owner_mode = str(plan.get("work_owner_mode", "") or "").strip()
    isolation_strategy = str(plan.get("isolation_strategy", "") or "").strip()
    input_representation = plan.get("input_representation", "")
    input_route = plan.get("input_route")

    requirements: list[dict[str, Any]] = []
    claimed: set[str] = set()

    def _add(capability: str, source_field: str, source_value: str, scope: dict[str, str], reason: str) -> None:
        if capability in claimed:
            return
        claimed.add(capability)
        requirements.append(
            {
                "capability": capability,
                "source_field": source_field,
                "source_value": source_value,
                "scope": dict(scope),
                "reason": reason[:_MAX_REASON_LENGTH],
            }
        )

    for capability in WORKFLOW_CAPABILITY_REQUIREMENTS.get(workflow, ()):
        _add(
            capability,
            "workflow",
            workflow,
            {},
            f"The accepted plan routes this work to the `{workflow}` workflow, whose lane runs on this capability.",
        )
    if isolation_strategy == _WORKTREE_REQUIRED_STRATEGY:
        _add(
            "worktree_isolation",
            "isolation_strategy",
            isolation_strategy,
            {},
            "The accepted plan's workspace binding declares an isolated worktree as required before the "
            "owner opens a coding session.",
        )
    if work_owner_mode == _RUNTIME_HANDOFF_MODE and workflow:
        _add(
            LOCAL_WORKFLOW_CAPABILITY_NAME,
            "work_owner_mode",
            work_owner_mode,
            {"skill_id": workflow},
            f"The accepted plan hands the `{workflow}` workflow to the owner's own runtime, so the owner "
            "has to carry that workflow locally.",
        )
    try:
        media_requirements = (
            modality_requirements(input_representation, input_route if isinstance(input_route, Mapping) else None)
            if input_representation
            else ()
        )
    except ValueError as exc:
        raise OwnerFitError(str(exc)) from exc
    for media in media_requirements:
        _add(
            media["capability"],
            "input_representation",
            f"{media['representation']}:{media['modality']}",
            {key: media[key] for key in ("provider", "wire_model", "endpoint_mode") if media[key]},
            "The accepted plan declares this input representation, which needs fresh route-scoped modality evidence.",
        )
    return tuple(requirements)


def accepted_plan_from_delegation(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project a built coding-delegation payload onto the declared plan fields.

    A projection rather than a second decision: every value is read back from
    where the delegation build already wrote it, so owner fit and the handoff
    can never describe different plans.
    """
    delegation = payload.get("delegation")
    delegation = delegation if isinstance(delegation, Mapping) else {}
    isolation = payload.get("isolation_plan")
    isolation = isolation if isinstance(isolation, Mapping) else {}
    handoff = next(
        (candidate for candidate in (payload.get("executor_handoff"), payload.get("prompt_handoff"), payload.get("runtime_handoff")) if isinstance(candidate, Mapping)),
        {},
    )
    return {
        "workflow": str(delegation.get("recommended_workflow", "") or ""),
        "work_owner_mode": str(payload.get("work_owner_mode", "") or ""),
        "isolation_strategy": str(isolation.get("strategy", "") or ""),
        "input_representation": payload.get("input_representation", handoff.get("input_representation", "")),
        "input_route": handoff.get("model_route", {}),
    }


def evaluate_owner_fit(
    *,
    owner: str,
    requirements: Sequence[Mapping[str, Any]],
    capability_snapshot: Mapping[str, Any] | None,
    now: str = "",
) -> dict[str, Any]:
    """Whether one owner can prove it satisfies every capability the plan requires.

    This is the reusable matcher. #812 (retarget an accepted handoff to another
    owner) derives the requirement set once from the accepted plan and calls
    this per candidate; nothing here knows or cares which surface asked.

    `capability_snapshot` is an `executor_capability_snapshot/v1` record for THIS
    owner, or `None` when none was recorded --- absent evidence classifies every
    requirement `unknown` rather than inventing a verdict. `now` is a parameter
    so the freshness derivation is deterministic under test.
    """
    normalized_owner = str(owner or "").strip()
    if not normalized_owner:
        raise OwnerFitError("owner fit requires a nonempty owner id")
    classifications = [_classify(requirement, capability_snapshot, now) for requirement in requirements]
    met = [str(item["capability"]) for item in classifications if item["classification"] == "met"]
    unmet = [str(item["capability"]) for item in classifications if item["classification"] == "unmet"]
    unknown = [str(item["capability"]) for item in classifications if item["classification"] == "unknown"]
    verdict = "blocked" if unmet else "unproven" if unknown else "ready"
    fit = {
        "schema_version": OWNER_FIT_SCHEMA_VERSION,
        "status": OWNER_FIT_STATUS,
        "owner": normalized_owner,
        "label": executor_label(normalized_owner),
        "verdict": verdict,
        "recommendable": verdict == "ready",
        "capabilities": classifications,
        "met": met,
        "unmet": unmet,
        "unknown": unknown,
        "reason": _fit_reason(verdict, met, unmet, unknown),
        "next_action": OWNER_FIT_NEXT_ACTIONS[verdict],
        "claim_boundary": OWNER_FIT_CLAIM_BOUNDARY,
    }
    errors = validate_owner_fit(fit)
    if errors:
        raise OwnerFitError("; ".join(errors))
    return fit


def build_owner_fit_report(
    *,
    requirements: Sequence[Mapping[str, Any]],
    owners: Sequence[tuple[str, Mapping[str, Any] | None]],
    named_owner: str = "",
    now: str = "",
) -> dict[str, Any]:
    """The owner-fit recommendation surface for one accepted plan.

    `owners` pairs each candidate with its recorded snapshot, or `None`. A named
    owner must be among them: the report never drops the owner a person asked
    for, so a caller that cannot supply its evidence has a bug rather than a
    silent omission.
    """
    normalized_named = str(named_owner or "").strip()
    fits = [
        evaluate_owner_fit(owner=owner, requirements=requirements, capability_snapshot=snapshot, now=now)
        for owner, snapshot in owners
    ]
    owner_ids = [str(fit["owner"]) for fit in fits]
    if len(owner_ids) != len(set(owner_ids)):
        raise OwnerFitError("owner fit report must not list an owner twice")
    if normalized_named and normalized_named not in owner_ids:
        raise OwnerFitError(f"named owner is missing from the owner fit candidates: {normalized_named}")
    named_fit = next((fit for fit in fits if str(fit["owner"]) == normalized_named), None)
    named_gap = "" if named_fit is None or named_fit["verdict"] == "ready" else _named_owner_gap(named_fit)
    recommended = [str(fit["owner"]) for fit in fits if fit["verdict"] == "ready"]
    report = {
        "schema_version": OWNER_FIT_REPORT_SCHEMA_VERSION,
        "status": OWNER_FIT_STATUS,
        "requirements": [dict(requirement) for requirement in requirements],
        "owners": fits,
        "recommended_owners": recommended,
        "unproven_owners": [str(fit["owner"]) for fit in fits if fit["verdict"] == "unproven"],
        "blocked_owners": [str(fit["owner"]) for fit in fits if fit["verdict"] == "blocked"],
        "named_owner": normalized_named,
        "named_owner_honoured": bool(normalized_named),
        "named_owner_gap": named_gap,
        "next_action": _report_next_action(normalized_named, named_gap, recommended),
        "claim_boundary": OWNER_FIT_REPORT_CLAIM_BOUNDARY,
    }
    errors = validate_owner_fit_report(report)
    if errors:
        raise OwnerFitError("; ".join(errors))
    return report


def owner_capability_snapshots(
    directory: Path | None,
    owners: Sequence[str],
) -> tuple[tuple[str, dict[str, Any] | None], ...]:
    """Pair each owner with its recorded capability snapshot, or `None`.

    The only I/O in this module, kept here so both the delegation build and the
    choose-executor card read evidence the same way. An absent, unreadable, or
    mismatched snapshot reads as `None`, which classifies every requirement
    `unknown` rather than inventing evidence.
    """
    if directory is None:
        return tuple((owner, None) for owner in owners)
    paired: list[tuple[str, dict[str, Any] | None]] = []
    for owner in owners:
        try:
            path = executor_capability_snapshot_path(directory, owner)
        except ValueError:
            paired.append((owner, None))
            continue
        paired.append((owner, read_matching_executor_capability_snapshot(path, expected_executor=owner)))
    return tuple(paired)


def owner_fit_without_owner_identity(fit: Mapping[str, Any]) -> dict[str, Any]:
    """The fit record with every owner-identity field removed (AC3).

    Two owners holding equal capability evidence must produce equal projections.
    This is the executor-neutrality property in code rather than in prose: if a
    brand ever leaks into a verdict, a reason, or a classification, the two
    projections stop matching.
    """
    return {str(key): value for key, value in fit.items() if str(key) not in OWNER_FIT_OWNER_IDENTITY_KEYS}


def validate_owner_fit(fit: Mapping[str, Any]) -> list[str]:
    """Both directions: no key of the closed set missing, no key outside it."""
    errors = _closed_key_errors("owner fit", fit, OWNER_FIT_KEYS)
    if fit.get("schema_version") != OWNER_FIT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {OWNER_FIT_SCHEMA_VERSION}")
    if fit.get("status") != OWNER_FIT_STATUS:
        errors.append(f"status must be {OWNER_FIT_STATUS}; owner fit never claims an observation")
    if fit.get("claim_boundary") != OWNER_FIT_CLAIM_BOUNDARY:
        errors.append("claim_boundary must be the owner fit claim boundary")
    for field, limit in (("owner", _MAX_TEXT_LENGTH), ("label", _MAX_TEXT_LENGTH), ("reason", _MAX_REASON_LENGTH)):
        value = fit.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            errors.append(f"{field} must be a nonempty string of at most {limit} characters")
    verdict = fit.get("verdict")
    if verdict not in OWNER_FIT_VERDICTS:
        errors.append(f"verdict must be one of {', '.join(OWNER_FIT_VERDICTS)}")
        return errors
    if fit.get("recommendable") is not (verdict == "ready"):
        errors.append("recommendable must be true for a ready verdict and false otherwise")
    if fit.get("next_action") != OWNER_FIT_NEXT_ACTIONS[str(verdict)]:
        errors.append("next_action must match the verdict it was derived from")
    capabilities = fit.get("capabilities")
    if not isinstance(capabilities, list):
        errors.append("capabilities must be a list")
        return errors
    if len(capabilities) > _MAX_REQUIREMENTS:
        errors.append(f"capabilities must hold at most {_MAX_REQUIREMENTS} entries")
    grouped: dict[str, list[str]] = {name: [] for name in OWNER_FIT_CLASSIFICATIONS}
    for entry in capabilities:
        if not isinstance(entry, Mapping):
            errors.append("each classified capability must be a mapping")
            continue
        entry_errors = _classification_errors(entry)
        errors.extend(entry_errors)
        if not entry_errors:
            grouped[str(entry["classification"])].append(str(entry["capability"]))
    for name in OWNER_FIT_CLASSIFICATIONS:
        if fit.get(name) != grouped[name]:
            errors.append(f"{name} must list exactly the capabilities classified {name}, in order")
    expected_verdict = "blocked" if grouped["unmet"] else "unproven" if grouped["unknown"] else "ready"
    if verdict != expected_verdict:
        errors.append(f"verdict must be {expected_verdict} for this classification set")
    return errors


def validate_owner_fit_report(report: Mapping[str, Any]) -> list[str]:
    """Both directions, plus the two acceptance properties the report exists for.

    AC1 has teeth here: `recommended_owners` must be exactly the ready owners,
    so an owner carrying a known unmet capability cannot appear in it however it
    was assembled. AC2 has teeth here too: every owner must classify exactly the
    declared requirements, in order, so a report cannot quietly drop one.
    """
    errors = _closed_key_errors("owner fit report", report, OWNER_FIT_REPORT_KEYS)
    if report.get("schema_version") != OWNER_FIT_REPORT_SCHEMA_VERSION:
        errors.append(f"schema_version must be {OWNER_FIT_REPORT_SCHEMA_VERSION}")
    if report.get("status") != OWNER_FIT_STATUS:
        errors.append(f"status must be {OWNER_FIT_STATUS}; the report never claims an observation")
    if report.get("claim_boundary") != OWNER_FIT_REPORT_CLAIM_BOUNDARY:
        errors.append("claim_boundary must be the owner fit report claim boundary")
    if report.get("next_action") not in OWNER_FIT_REPORT_NEXT_ACTIONS:
        errors.append(f"next_action must be one of {', '.join(OWNER_FIT_REPORT_NEXT_ACTIONS)}")
    requirements = report.get("requirements")
    if not isinstance(requirements, list):
        errors.append("requirements must be a list")
        return errors
    if len(requirements) > _MAX_REQUIREMENTS:
        errors.append(f"requirements must hold at most {_MAX_REQUIREMENTS} entries")
    required_names: list[str] = []
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            errors.append("each requirement must be a mapping")
            continue
        requirement_errors = _requirement_errors(requirement)
        errors.extend(requirement_errors)
        if not requirement_errors:
            required_names.append(str(requirement["capability"]))
    if len(required_names) != len(set(required_names)):
        errors.append("requirements must not name the same capability twice")
    owners = report.get("owners")
    if not isinstance(owners, list) or not owners:
        errors.append("owners must be a nonempty list")
        return errors
    if len(owners) > _MAX_OWNERS:
        errors.append(f"owners must hold at most {_MAX_OWNERS} entries")
    by_verdict: dict[str, list[str]] = {name: [] for name in OWNER_FIT_VERDICTS}
    fits: dict[str, Mapping[str, Any]] = {}
    for fit in owners:
        if not isinstance(fit, Mapping):
            errors.append("each owner entry must be a mapping")
            continue
        fit_errors = validate_owner_fit(fit)
        errors.extend(fit_errors)
        if fit_errors:
            continue
        owner_id = str(fit["owner"])
        fits[owner_id] = fit
        by_verdict[str(fit["verdict"])].append(owner_id)
        classified = [str(entry["capability"]) for entry in fit["capabilities"]]
        if classified != required_names:
            errors.append(
                f"owner {owner_id} must classify exactly the declared required capabilities, in order"
            )
    if len(fits) != len(owners):
        return errors
    for field, verdict in (
        ("recommended_owners", "ready"),
        ("unproven_owners", "unproven"),
        ("blocked_owners", "blocked"),
    ):
        if report.get(field) != by_verdict[verdict]:
            errors.append(f"{field} must list exactly the owners whose verdict is {verdict}, in order")
    errors.extend(_named_owner_errors(report, fits))
    return errors


def _named_owner_errors(report: Mapping[str, Any], fits: Mapping[str, Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    named = report.get("named_owner")
    if not isinstance(named, str):
        return ["named_owner must be a string, empty when no owner was named"]
    if report.get("named_owner_honoured") is not bool(named):
        errors.append("named_owner_honoured must be true exactly when an owner was named")
    gap = report.get("named_owner_gap")
    if not isinstance(gap, str) or len(gap) > _MAX_REASON_LENGTH:
        return [*errors, f"named_owner_gap must be a string of at most {_MAX_REASON_LENGTH} characters"]
    if not named:
        if gap:
            errors.append("named_owner_gap must be empty when no owner was named")
        return errors
    if named not in fits:
        errors.append("named_owner must be one of the reported owners; the report never drops a named owner")
        return errors
    ready = fits[named]["verdict"] == "ready"
    if ready and gap:
        errors.append("named_owner_gap must be empty when the named owner is ready")
    if not ready and not gap:
        errors.append("named_owner_gap must name the gap when the named owner is not ready")
    return errors


def _classify(requirement: Mapping[str, Any], snapshot: Mapping[str, Any] | None, now: str) -> dict[str, Any]:
    """One requirement against one snapshot.

    Precedence is deliberate. Scope is checked before freshness: evidence
    recorded about a different workflow is not stale evidence about this one, it
    is evidence about something else, and saying "expired" would send a person
    to re-observe the wrong thing.
    """
    capability = str(requirement.get("capability", ""))
    source_field = str(requirement.get("source_field", ""))
    source_value = str(requirement.get("source_value", ""))
    required_scope = requirement.get("scope")
    required_scope = required_scope if isinstance(required_scope, Mapping) else {}

    def _record(evidence_status: str, evidence_ref: str, reason_code: str, observed_at: str) -> dict[str, Any]:
        return {
            "capability": capability,
            "source_field": source_field,
            "source_value": source_value,
            "classification": OWNER_FIT_REASON_CODES[reason_code],
            "reason_code": reason_code,
            "reason": _classification_reason(reason_code)[:_MAX_REASON_LENGTH],
            "evidence_status": evidence_status,
            "evidence_ref": evidence_ref[:_MAX_TEXT_LENGTH],
            "evidence_observed_at": observed_at[:_MAX_TEXT_LENGTH],
        }

    if not isinstance(snapshot, Mapping):
        return _record("no_snapshot", _NO_SNAPSHOT_EVIDENCE_REF, "no_capability_snapshot", "")
    capabilities = snapshot.get("capabilities")
    entry = capabilities.get(capability) if isinstance(capabilities, Mapping) else None
    if not isinstance(entry, Mapping):
        return _record("not_recorded", _NOT_RECORDED_EVIDENCE_REF, "capability_not_recorded", "")
    status = str(entry.get("status", ""))
    if status == "prepared":
        return _record("prepared", _STATUS_ONLY_EVIDENCE_REF, "evidence_prepared_only", "")
    if status not in {"host_observed", "unavailable"}:
        return _record("unknown", _STATUS_ONLY_EVIDENCE_REF, "evidence_status_unknown", "")
    evidence_ref = str(entry.get("evidence_ref", "") or "").strip() or _STATUS_ONLY_EVIDENCE_REF
    # `executor_capability_snapshot/v1` gives a per-capability `observed_at`
    # only to `host_observed` entries and to `local_workflow`; an `unavailable`
    # generic capability is status-only. The snapshot's own `recorded_at` is
    # then the honest observation time --- the evidence IS "this snapshot,
    # written then" --- and it keeps a recorded absence subject to the same
    # freshness rule as a recorded presence.
    observed_at = str(entry.get("observed_at", "") or "").strip() or str(snapshot.get("recorded_at", "") or "").strip()
    if required_scope and not _scope_covers(entry.get("scope"), required_scope):
        return _record(status, evidence_ref, "evidence_scope_mismatch", observed_at)
    if not capability_evidence_is_fresh(observed_at, now):
        return _record(status, evidence_ref, "evidence_expired", observed_at)
    reason_code = "host_observed_available" if status == "host_observed" else "host_observed_unavailable"
    return _record(status, evidence_ref, reason_code, observed_at)


def _scope_covers(observed: Any, required: Mapping[str, Any]) -> bool:
    if not isinstance(observed, Mapping):
        return False
    return all(str(observed.get(key, "")) == str(value) for key, value in required.items())


def _classification_reason(reason_code: str) -> str:
    hours = CAPABILITY_EVIDENCE_STALE_AFTER_SECONDS // 3600
    match reason_code:
        case "no_capability_snapshot":
            return (
                "No capability snapshot was recorded for this owner, so this requirement is unproven rather "
                "than met or missing."
            )
        case "capability_not_recorded":
            return (
                "The recorded capability snapshot says nothing about this capability, so it is unproven "
                "rather than met or missing."
            )
        case "evidence_prepared_only":
            return (
                "The snapshot carries prepared evidence for this capability, and prepared evidence is the "
                "absence of an observation rather than a weaker one."
            )
        case "evidence_status_unknown":
            return "The snapshot records this capability as unknown, which is not an observation either way."
        case "evidence_scope_mismatch":
            return (
                "The recorded observation covers a different scope than this requirement asks about, so it "
                "cannot answer this requirement."
            )
        case "evidence_expired":
            return (
                f"The recorded observation is older than {hours} hours, so it describes a machine state "
                "nobody has confirmed since."
            )
        case "host_observed_available":
            return "A host observation inside the freshness window records this capability as available."
        case _:
            return "A host observation inside the freshness window records this capability as unavailable."


def _capability_count(names: Sequence[str]) -> str:
    return f"{len(names)} required {'capability' if len(names) == 1 else 'capabilities'}"


def _fit_reason(verdict: str, met: Sequence[str], unmet: Sequence[str], unknown: Sequence[str]) -> str:
    if verdict == "blocked":
        return (
            f"{_capability_count(unmet)} recorded unavailable by fresh host observation: "
            f"{', '.join(unmet)}. This owner cannot complete the accepted plan."
        )
    if verdict == "unproven":
        return (
            f"{_capability_count(unknown)} with no fresh host observation: {', '.join(unknown)}. "
            "Nothing is known to be missing, and nothing is proven present."
        )
    if not met:
        return "The accepted plan declares no required capability, so there is nothing left to prove."
    return f"Every required capability is answered by fresh host observation: {', '.join(met)}."


def _named_owner_gap(fit: Mapping[str, Any]) -> str:
    unmet = [str(name) for name in fit.get("unmet", [])]
    unknown = [str(name) for name in fit.get("unknown", [])]
    if unmet:
        gap = (
            "The named owner stays selected, and fresh host observation records these required capabilities "
            f"as unavailable: {', '.join(unmet)}. It is not recommended, and it was not swapped for another "
            "owner without saying so."
        )
    else:
        gap = (
            "The named owner stays selected, and these required capabilities have no fresh host observation: "
            f"{', '.join(unknown)}. Record the evidence or accept the gap explicitly."
        )
    return gap[:_MAX_REASON_LENGTH]


def _report_next_action(named_owner: str, named_gap: str, recommended: Sequence[str]) -> str:
    if named_owner:
        return "confirm_named_owner_despite_capability_gap" if named_gap else "prepare_handoff_for_named_owner"
    if recommended:
        return "choose_from_recommended_owners"
    return "record_capability_evidence"


def _closed_key_errors(label: str, value: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    expected = set(keys)
    present = {str(key) for key in value}
    errors: list[str] = []
    missing = expected - present
    if missing:
        errors.append(f"{label} is missing required keys: {', '.join(sorted(missing))}")
    unexpected = present - expected
    if unexpected:
        errors.append(f"{label} contains unsupported keys: {', '.join(sorted(unexpected))}")
    return errors


def _requirement_errors(requirement: Mapping[str, Any]) -> list[str]:
    errors = _closed_key_errors("requirement", requirement, OWNER_FIT_REQUIREMENT_KEYS)
    capability = requirement.get("capability")
    if capability not in KNOWN_CAPABILITY_NAMES and capability != LOCAL_WORKFLOW_CAPABILITY_NAME:
        errors.append(f"requirement capability must come from the snapshot vocabulary: {capability}")
    for field, limit in (
        ("source_field", _MAX_TEXT_LENGTH),
        ("source_value", _MAX_TEXT_LENGTH),
        ("reason", _MAX_REASON_LENGTH),
    ):
        value = requirement.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            errors.append(f"requirement {field} must be a nonempty string of at most {limit} characters")
    if requirement.get("source_field") not in ACCEPTED_PLAN_FIELDS:
        errors.append(f"requirement source_field must be a declared plan field: {requirement.get('source_field')}")
    scope = requirement.get("scope")
    if not isinstance(scope, Mapping):
        errors.append("requirement scope must be a mapping, empty when the requirement is unscoped")
    elif len(scope) > _MAX_SCOPE_ITEMS:
        errors.append(f"requirement scope must hold at most {_MAX_SCOPE_ITEMS} items")
    elif any(
        not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip()
        for key, item in scope.items()
    ):
        errors.append("requirement scope keys and values must be nonempty strings")
    return errors


def _classification_errors(entry: Mapping[str, Any]) -> list[str]:
    errors = _closed_key_errors("classified capability", entry, OWNER_FIT_CLASSIFICATION_KEYS)
    if errors:
        return errors
    capability = entry.get("capability")
    if capability not in KNOWN_CAPABILITY_NAMES and capability != LOCAL_WORKFLOW_CAPABILITY_NAME:
        errors.append(f"classified capability must come from the snapshot vocabulary: {capability}")
    reason_code = entry.get("reason_code")
    if reason_code not in OWNER_FIT_REASON_CODES:
        errors.append(f"reason_code must be one of {', '.join(sorted(OWNER_FIT_REASON_CODES))}")
    elif entry.get("classification") != OWNER_FIT_REASON_CODES[str(reason_code)]:
        errors.append("classification must match the reason code it was derived from")
    if entry.get("evidence_status") not in OWNER_FIT_EVIDENCE_STATUSES:
        errors.append(f"evidence_status must be one of {', '.join(OWNER_FIT_EVIDENCE_STATUSES)}")
    for field, limit in (
        ("source_field", _MAX_TEXT_LENGTH),
        ("source_value", _MAX_TEXT_LENGTH),
        ("evidence_ref", _MAX_TEXT_LENGTH),
        ("reason", _MAX_REASON_LENGTH),
    ):
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > limit:
            errors.append(f"classified capability {field} must be a nonempty string of at most {limit} characters")
    observed_at = entry.get("evidence_observed_at")
    if not isinstance(observed_at, str) or len(observed_at) > _MAX_TEXT_LENGTH:
        errors.append("classified capability evidence_observed_at must be a string, empty when nothing was observed")
    return errors
