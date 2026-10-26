"""Move accepted coding work to another owner without replanning it (#812).

The gap this closes. A coding plan is accepted once. The accepted plan decides
the routed workflow, the intent, the acceptance criteria, and the verification
expectations, and none of those four readings depend on who ends up doing the
work --- `build_coding_delegation_payload` computes `payload["delegation"]`
before it ever looks at `executor_target`. What DID depend on the owner was the
only way to change one: `prepare_wrapper_session_handoff` refused every
executor change on a follow-up handoff and offered exactly one escape, "start a
new session". A new session has no accepted plan, so changing owner meant
replanning work the person had already approved and re-deriving criteria they
had already agreed to.

So retargeting is a **re-projection, not a decision**. The same owner-neutral
task contract is rendered through a different owner's handoff projection, and
the only thing Hermes has to explain is the part that genuinely moved.

Three artifacts, all pure and all deterministic (`now` is a parameter, and
nothing here reads a clock, a network, a credential, or a host configuration
file):

* `coding_task_contract/v1` --- the owner-neutral half of one accepted plan.
  Projected from `payload["delegation"]`, whose ten fields are computed with no
  reading of the owner, plus a digest of the equally owner-neutral
  `specialist_work_quality` bar. This is what AC1 protects: the task's scope
  (`intent`, `recommended_workflow`, `recommended_harness`, `work_role`), its
  non-goals and review boundary (`review_required`, `review_workflow`, and the
  `delegation_prompt_template` that states them to the owner), its
  `acceptance_criteria`, and its `verification`.

* `coding_owner_retarget/v1` --- the record of one move. It carries the two
  owners, the preserved contract and its digest, the enumerated owner-specific
  delta, and the capability delta.

* The refusal. `build_owner_retarget` compares the two contracts field by field
  and raises when any of them moved. **A retarget that would change the task
  contract is refused as a replan**, so AC1 is enforced by the builder rather
  than asserted by a test alone.

AC2 --- "enumerate unsupported or changed owner capability before handoff" ---
reuses #810's matcher rather than growing a second one. Requirements come from
`derive_plan_capability_requirements(accepted_plan_from_delegation(...))` and
each owner is classified by `evaluate_owner_fit`. Both owners are judged
against the TARGET plan's requirement set, because "what changed" is only
meaningful on one yardstick; the yardstick's own movement is reported
separately as `required_by_owner_change` / `dropped_by_owner_change`. That
movement is real and is the whole point of the field: retargeting from an
external-executor owner to a runtime owner turns the routed workflow into
something the new owner has to carry locally, which adds a `local_workflow`
requirement that the source plan never had.

AC3 --- no source-host configuration or credential is read. Capability
snapshots arrive as a parameter; absent snapshots classify every requirement
`unknown` (an unproven owner), never a guess. Retargeting is a local
re-projection of two values the caller already holds.

Nothing here is execution. A retarget is not dispatch and not evidence that the
new owner accepted or started anything.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..system.hashutil import sha256_text
from .executors import EXECUTOR_PROFILES, canonical_executor_profile, executor_label
from .owner_fit import (
    OWNER_FIT_CLASSIFICATIONS,
    OWNER_FIT_VERDICTS,
    accepted_plan_from_delegation,
    derive_plan_capability_requirements,
    evaluate_owner_fit,
)


CODING_TASK_CONTRACT_SCHEMA_VERSION: Final = "coding_task_contract/v1"
OWNER_RETARGET_SCHEMA_VERSION: Final = "coding_owner_retarget/v1"
OWNER_RETARGET_STATUS: Final = "prepared_not_observed"

# The three fields a built payload can carry a prepared handoff under. Exactly
# one is present per owner, and which one it is is itself part of the delta.
HANDOFF_PAYLOAD_FIELDS: Final = ("executor_handoff", "prompt_handoff", "runtime_handoff")

# The preserved half. `work_role` is `delegation.executor_profile` renamed on
# projection: it is the ROLE the work is scoped to (coding-agent, reviewer,
# planner, docs-writer), never the coding owner, and in a record whose subject
# is "the owner changed" the source name would read as the opposite of what it
# means.
TASK_CONTRACT_KEYS: Final = (
    "schema_version",
    "status",
    "task_source_sha256",
    "action",
    "intent",
    "recommended_workflow",
    "recommended_harness",
    "work_role",
    "acceptance_criteria",
    "verification",
    "review_required",
    "review_workflow",
    "delegation_prompt_template",
    "work_quality_sha256",
    "claim_boundary",
)

# The moved half. A closed list, compared field by field, so the delta names
# what changed instead of diffing two whole handoffs and reporting noise.
OWNER_SPECIFIC_FIELDS: Final = (
    "work_owner_mode",
    "handoff_field",
    "dispatch_policy",
    "dispatchable",
    "dispatch_contract",
    "recording_contract",
    "invocation_mode",
    "isolation_strategy",
)

OWNER_RETARGET_KEYS: Final = (
    "schema_version",
    "status",
    "from_owner",
    "from_owner_label",
    "to_owner",
    "to_owner_label",
    "preserved",
    "preserved_digest",
    "owner_delta",
    "capability_delta",
    "next_action",
    "reason",
    "claim_boundary",
)
OWNER_DELTA_KEYS: Final = (
    "changed",
    "unchanged",
    "observed_evidence_gained",
    "observed_evidence_lost",
    "wrapper_actions_gained",
    "wrapper_actions_lost",
)
OWNER_DELTA_CHANGE_KEYS: Final = ("field", "from", "to")
CAPABILITY_DELTA_KEYS: Final = (
    "requirements",
    "required_by_owner_change",
    "dropped_by_owner_change",
    "unsupported",
    "unproven",
    "changed",
    "from_verdict",
    "to_verdict",
    "statement",
)
CAPABILITY_CHANGE_KEYS: Final = ("capability", "from_classification", "to_classification")

OWNER_RETARGET_NEXT_ACTIONS: Final = (
    "prepare_handoff_for_new_owner",
    "confirm_owner_capability_gap",
    "record_capability_evidence",
)

TASK_CONTRACT_CLAIM_BOUNDARY: Final = (
    "This is the owner-neutral half of one accepted coding plan: what the work is, what it has to "
    "satisfy, and how it has to be verified. It is prepared context, not dispatch, execution, "
    "verification, review, CI, or merge evidence."
)
OWNER_RETARGET_CLAIM_BOUNDARY: Final = (
    "A retarget re-projects one already-accepted task contract onto another coding owner. It is not "
    "dispatch, execution, verification, review, CI, or merge evidence, it is not proof that the new "
    "owner accepted or started the work, and it never replans the task."
)

_MAX_TEXT_LENGTH: Final = 240
_MAX_REASON_LENGTH: Final = 400
_MAX_LIST_ITEMS: Final = 16
_MAX_CAPABILITIES: Final = 12


class OwnerRetargetError(ValueError):
    pass


def coding_task_contract(payload: Mapping[str, Any], *, task_source_sha256: str = "") -> dict[str, Any]:
    """The owner-neutral projection of one built coding-delegation payload.

    A projection rather than a second derivation: every value is read back from
    where the delegation build already wrote it, so the contract and the
    handoff can never describe different work. `task_source_sha256` is the
    caller's binding to the task text --- the payload never carries the raw
    message, and OMH does not persist it.
    """
    delegation = payload.get("delegation")
    if not isinstance(delegation, Mapping):
        raise OwnerRetargetError("a built coding delegation payload is required to project a task contract")
    work_quality = payload.get("specialist_work_quality")
    review_workflow = delegation.get("review_workflow")
    return {
        "schema_version": CODING_TASK_CONTRACT_SCHEMA_VERSION,
        "status": OWNER_RETARGET_STATUS,
        "task_source_sha256": str(task_source_sha256 or ""),
        "action": str(delegation.get("action", "")),
        "intent": str(delegation.get("intent", "")),
        "recommended_workflow": str(delegation.get("recommended_workflow", "")),
        "recommended_harness": str(delegation.get("recommended_harness", "")),
        "work_role": str(delegation.get("executor_profile", "")),
        "acceptance_criteria": _text_list(delegation.get("acceptance_criteria")),
        "verification": _text_list(delegation.get("verification")),
        "review_required": bool(delegation.get("review_required", False)),
        "review_workflow": "" if review_workflow is None else str(review_workflow),
        "delegation_prompt_template": str(delegation.get("delegation_prompt_template", "")),
        # The work-quality bar is owner-neutral too, but it is a nested contract
        # rather than a sentence. Digesting it keeps it inside `preserved_digest`
        # without turning a session event into a copy of the whole payload.
        "work_quality_sha256": _canonical_digest(work_quality if isinstance(work_quality, Mapping) else {}),
        "claim_boundary": TASK_CONTRACT_CLAIM_BOUNDARY,
    }


def coding_task_contract_digest(contract: Mapping[str, Any]) -> str:
    """sha256 over the canonical JSON of one task contract."""
    return _canonical_digest(dict(contract))


def build_owner_retarget(
    *,
    from_payload: Mapping[str, Any],
    to_payload: Mapping[str, Any],
    task_source_sha256: str = "",
    capability_snapshots: Mapping[str, Mapping[str, Any] | None] | None = None,
    now: str = "",
) -> dict[str, Any]:
    """Record one accepted plan moving from one coding owner to another.

    Both arguments are built coding-delegation payloads for the SAME task, one
    per owner. The builder refuses rather than repairs: an owner-neutral field
    that differs between them is a replan, not a retarget, and the caller is
    told which field moved.

    `capability_snapshots` maps an owner to its recorded
    `executor_capability_snapshot/v1`, or to `None`. It is a parameter and not
    a read so that retargeting stays a pure local re-projection (AC3); an owner
    with no snapshot classifies `unknown` and reads as unproven, never as fit.
    """
    from_owner = _accepted_owner(from_payload, "source")
    to_owner = _accepted_owner(to_payload, "target")
    if from_owner == to_owner:
        raise OwnerRetargetError(f"retargeting requires a different coding owner; both sides are {to_owner}")

    from_contract = coding_task_contract(from_payload, task_source_sha256=task_source_sha256)
    to_contract = coding_task_contract(to_payload, task_source_sha256=task_source_sha256)
    moved = [key for key in TASK_CONTRACT_KEYS if from_contract[key] != to_contract[key]]
    if moved:
        raise OwnerRetargetError(
            "retargeting must preserve the accepted task contract, and these owner-neutral fields moved, "
            f"which is a replan rather than a retarget: {', '.join(moved)}"
        )

    owner_delta = _owner_delta(from_payload, to_payload)
    capability_delta = _capability_delta(
        from_payload=from_payload,
        to_payload=to_payload,
        from_owner=from_owner,
        to_owner=to_owner,
        capability_snapshots=dict(capability_snapshots or {}),
        now=now,
    )
    record = {
        "schema_version": OWNER_RETARGET_SCHEMA_VERSION,
        "status": OWNER_RETARGET_STATUS,
        "from_owner": from_owner,
        "from_owner_label": executor_label(from_owner),
        "to_owner": to_owner,
        "to_owner_label": executor_label(to_owner),
        "preserved": to_contract,
        "preserved_digest": coding_task_contract_digest(to_contract),
        "owner_delta": owner_delta,
        "capability_delta": capability_delta,
        "next_action": _next_action(capability_delta),
        "reason": _retarget_reason(from_owner, to_owner, owner_delta, capability_delta),
        "claim_boundary": OWNER_RETARGET_CLAIM_BOUNDARY,
    }
    errors = validate_owner_retarget(record)
    if errors:
        raise OwnerRetargetError("; ".join(errors))
    return record


def validate_coding_task_contract(contract: Mapping[str, Any]) -> list[str]:
    """Both directions: no key of the closed set missing, no key outside it."""
    errors = _closed_key_errors("task contract", contract, TASK_CONTRACT_KEYS)
    if contract.get("schema_version") != CODING_TASK_CONTRACT_SCHEMA_VERSION:
        errors.append(f"task contract schema_version must be {CODING_TASK_CONTRACT_SCHEMA_VERSION}")
    if contract.get("status") != OWNER_RETARGET_STATUS:
        errors.append("task contract status must be prepared_not_observed; a contract is never an observation")
    if contract.get("claim_boundary") != TASK_CONTRACT_CLAIM_BOUNDARY:
        errors.append("task contract claim_boundary must be the task contract claim boundary")
    for field in ("action", "intent", "recommended_workflow", "recommended_harness", "work_role"):
        value = contract.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > _MAX_TEXT_LENGTH:
            errors.append(f"task contract {field} must be a nonempty string of at most {_MAX_TEXT_LENGTH} characters")
    for field in ("task_source_sha256", "review_workflow"):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) > _MAX_TEXT_LENGTH:
            errors.append(f"task contract {field} must be a string, empty when the plan declares none")
    template = contract.get("delegation_prompt_template")
    if not isinstance(template, str) or not template.strip():
        errors.append("task contract delegation_prompt_template must be a nonempty string")
    digest = contract.get("work_quality_sha256")
    if not _is_sha256(digest):
        errors.append("task contract work_quality_sha256 must be a sha256 hex digest")
    if not isinstance(contract.get("review_required"), bool):
        errors.append("task contract review_required must be a boolean")
    for field in ("acceptance_criteria", "verification"):
        errors.extend(_text_list_errors(f"task contract {field}", contract.get(field), required=True))
    return errors


def validate_owner_retarget(record: Mapping[str, Any]) -> list[str]:
    """Both directions, plus the three properties the record exists to carry.

    AC1 has teeth here: `preserved_digest` must be recomputable from the stored
    contract, so a record cannot claim a preserved contract it does not hold.
    AC2 has teeth here: `unsupported` and `unproven` must be subsets of the
    declared requirements, and `next_action` must follow from them, so a gap
    cannot be enumerated and then quietly dropped from the recommendation.
    """
    errors = _closed_key_errors("owner retarget", record, OWNER_RETARGET_KEYS)
    if record.get("schema_version") != OWNER_RETARGET_SCHEMA_VERSION:
        errors.append(f"owner retarget schema_version must be {OWNER_RETARGET_SCHEMA_VERSION}")
    if record.get("status") != OWNER_RETARGET_STATUS:
        errors.append("owner retarget status must be prepared_not_observed; a retarget is never an observation")
    if record.get("claim_boundary") != OWNER_RETARGET_CLAIM_BOUNDARY:
        errors.append("owner retarget claim_boundary must be the owner retarget claim boundary")
    errors.extend(_owner_identity_errors(record))
    reason = record.get("reason")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > _MAX_REASON_LENGTH:
        errors.append(f"owner retarget reason must be a nonempty string of at most {_MAX_REASON_LENGTH} characters")

    preserved = record.get("preserved")
    if not isinstance(preserved, Mapping):
        errors.append("owner retarget preserved must be the preserved task contract")
    else:
        errors.extend(validate_coding_task_contract(preserved))
        if record.get("preserved_digest") != coding_task_contract_digest(preserved):
            errors.append("owner retarget preserved_digest must be the digest of the preserved task contract")
    if not _is_sha256(record.get("preserved_digest")):
        errors.append("owner retarget preserved_digest must be a sha256 hex digest")

    errors.extend(_owner_delta_errors(record.get("owner_delta")))
    capability_errors, capability_delta = _capability_delta_errors(record.get("capability_delta"))
    errors.extend(capability_errors)
    next_action = record.get("next_action")
    if next_action not in OWNER_RETARGET_NEXT_ACTIONS:
        errors.append(f"owner retarget next_action must be one of {', '.join(OWNER_RETARGET_NEXT_ACTIONS)}")
    elif capability_delta is not None and next_action != _next_action(capability_delta):
        errors.append("owner retarget next_action must follow from the enumerated capability delta")
    return errors


def _accepted_owner(payload: Mapping[str, Any], side: str) -> str:
    delegation = payload.get("delegation")
    if not isinstance(delegation, Mapping) or str(delegation.get("action", "")) != "delegate":
        raise OwnerRetargetError(f"the {side} payload carries no accepted coding delegation to retarget")
    owner = canonical_executor_profile(str(payload.get("selected_executor_profile") or ""))
    if owner not in EXECUTOR_PROFILES:
        raise OwnerRetargetError(
            f"unsupported coding owner on the {side} side of a retarget: {owner or '<none selected>'}"
        )
    if not _handoff_field(payload):
        raise OwnerRetargetError(f"the {side} payload carries no prepared handoff to retarget")
    return owner


def _handoff_field(payload: Mapping[str, Any]) -> str:
    return next((name for name in HANDOFF_PAYLOAD_FIELDS if isinstance(payload.get(name), Mapping)), "")


def _handoff(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    handoff = payload.get(_handoff_field(payload))
    return handoff if isinstance(handoff, Mapping) else {}


def _owner_facts(payload: Mapping[str, Any]) -> dict[str, str]:
    handoff = _handoff(payload)
    invocation = handoff.get("invocation")
    invocation = invocation if isinstance(invocation, Mapping) else {}
    isolation = payload.get("isolation_plan")
    isolation = isolation if isinstance(isolation, Mapping) else {}
    return {
        "work_owner_mode": _scalar_text(payload.get("work_owner_mode", "")),
        "handoff_field": _handoff_field(payload),
        "dispatch_policy": _scalar_text(payload.get("dispatch_policy", "")),
        "dispatchable": _scalar_text(bool(payload.get("dispatchable", False))),
        "dispatch_contract": _scalar_text(handoff.get("dispatch_contract", "")),
        "recording_contract": _scalar_text(handoff.get("recording_contract", "")),
        # The codex projection names its invocation shape `handoff_mode` and the
        # prompt/runtime projections name it `invocation.mode`; both answer the
        # same question, so the delta reads one field rather than two.
        "invocation_mode": _scalar_text(invocation.get("mode", "") or handoff.get("handoff_mode", "")),
        "isolation_strategy": _scalar_text(isolation.get("strategy", "")),
    }


def _owner_delta(from_payload: Mapping[str, Any], to_payload: Mapping[str, Any]) -> dict[str, Any]:
    from_facts = _owner_facts(from_payload)
    to_facts = _owner_facts(to_payload)
    from_evidence = _observed_evidence(from_payload)
    to_evidence = _observed_evidence(to_payload)
    from_actions = _wrapper_actions(from_payload)
    to_actions = _wrapper_actions(to_payload)
    return {
        "changed": [
            {"field": field, "from": from_facts[field], "to": to_facts[field]}
            for field in OWNER_SPECIFIC_FIELDS
            if from_facts[field] != to_facts[field]
        ],
        "unchanged": [field for field in OWNER_SPECIFIC_FIELDS if from_facts[field] == to_facts[field]],
        "observed_evidence_gained": _missing_from(to_evidence, from_evidence),
        "observed_evidence_lost": _missing_from(from_evidence, to_evidence),
        "wrapper_actions_gained": _missing_from(to_actions, from_actions),
        "wrapper_actions_lost": _missing_from(from_actions, to_actions),
    }


def _observed_evidence(payload: Mapping[str, Any]) -> list[str]:
    contract = _handoff(payload).get("evidence_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    return _text_list(contract.get("observed_required_for"))


def _wrapper_actions(payload: Mapping[str, Any]) -> list[str]:
    # The payload-level harness quality, not the handoff's own copy: only the
    # payload-level one is narrowed per owner by `_public_harness_quality`, so
    # only it answers "which actions does this owner's handoff actually offer".
    harness = payload.get("harness_quality")
    harness = harness if isinstance(harness, Mapping) else {}
    return _text_list(harness.get("wrapper_actions"))


def _capability_delta(
    *,
    from_payload: Mapping[str, Any],
    to_payload: Mapping[str, Any],
    from_owner: str,
    to_owner: str,
    capability_snapshots: Mapping[str, Mapping[str, Any] | None],
    now: str,
) -> dict[str, Any]:
    to_requirements = derive_plan_capability_requirements(accepted_plan_from_delegation(to_payload))
    from_requirements = derive_plan_capability_requirements(accepted_plan_from_delegation(from_payload))
    to_names = [str(requirement["capability"]) for requirement in to_requirements]
    from_names = [str(requirement["capability"]) for requirement in from_requirements]
    from_fit = evaluate_owner_fit(
        owner=from_owner,
        requirements=to_requirements,
        capability_snapshot=capability_snapshots.get(from_owner),
        now=now,
    )
    to_fit = evaluate_owner_fit(
        owner=to_owner,
        requirements=to_requirements,
        capability_snapshot=capability_snapshots.get(to_owner),
        now=now,
    )
    from_classification = _classifications(from_fit)
    to_classification = _classifications(to_fit)
    changed = [
        {
            "capability": name,
            "from_classification": from_classification[name],
            "to_classification": to_classification[name],
        }
        for name in to_names
        if from_classification[name] != to_classification[name]
    ]
    required_by_change = _missing_from(to_names, from_names)
    dropped_by_change = _missing_from(from_names, to_names)
    unsupported = [str(name) for name in to_fit["unmet"]]
    unproven = [str(name) for name in to_fit["unknown"]]
    return {
        "requirements": to_names,
        "required_by_owner_change": required_by_change,
        "dropped_by_owner_change": dropped_by_change,
        "unsupported": unsupported,
        "unproven": unproven,
        "changed": changed,
        "from_verdict": str(from_fit["verdict"]),
        "to_verdict": str(to_fit["verdict"]),
        "statement": _capability_statement(unsupported, unproven, required_by_change, changed),
    }


def _classifications(fit: Mapping[str, Any]) -> dict[str, str]:
    entries = fit.get("capabilities")
    entries = entries if isinstance(entries, Sequence) else ()
    return {
        str(entry["capability"]): str(entry["classification"])
        for entry in entries
        if isinstance(entry, Mapping)
    }


def _capability_statement(
    unsupported: Sequence[str],
    unproven: Sequence[str],
    required_by_change: Sequence[str],
    changed: Sequence[Mapping[str, Any]],
) -> str:
    if unsupported:
        statement = (
            f"Fresh host observation records {_capability_count(unsupported)} as unavailable for the new owner: "
            f"{', '.join(unsupported)}. Name this before the new handoff is used."
        )
    elif unproven:
        statement = (
            f"{_capability_count(unproven)} {_has_have(unproven)} no fresh host observation for the new owner: "
            f"{', '.join(unproven)}. Nothing is known to be missing, and nothing is proven present."
        )
    elif required_by_change:
        statement = (
            f"The owner change adds {_capability_count(required_by_change)} that the previous owner's plan did "
            f"not need: {', '.join(required_by_change)}. Fresh host observation answers all of them."
        )
    elif changed:
        names = ", ".join(str(entry["capability"]) for entry in changed)
        statement = f"No required capability is unsupported, and the evidence behind these moved: {names}."
    else:
        statement = "No required capability is unsupported, unproven, or changed by this owner change."
    return statement[:_MAX_REASON_LENGTH]


def _capability_count(names: Sequence[str]) -> str:
    return f"{len(names)} required {'capability' if len(names) == 1 else 'capabilities'}"


def _has_have(names: Sequence[str]) -> str:
    return "has" if len(names) == 1 else "have"


def _next_action(capability_delta: Mapping[str, Any]) -> str:
    if capability_delta.get("unsupported"):
        return "confirm_owner_capability_gap"
    if capability_delta.get("unproven"):
        return "record_capability_evidence"
    return "prepare_handoff_for_new_owner"


def _retarget_reason(
    from_owner: str,
    to_owner: str,
    owner_delta: Mapping[str, Any],
    capability_delta: Mapping[str, Any],
) -> str:
    changed = owner_delta.get("changed")
    moved = len(changed) if isinstance(changed, Sequence) else 0
    reason = (
        f"The accepted plan moved from {executor_label(from_owner)} to {executor_label(to_owner)} without "
        f"replanning: every owner-neutral field of the task contract is unchanged and "
        f"{moved} owner-specific {'field was' if moved == 1 else 'fields were'} re-projected. "
        f"{capability_delta.get('statement', '')}"
    )
    return reason[:_MAX_REASON_LENGTH]


def _owner_identity_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    owners: dict[str, str] = {}
    for field, label_field in (("from_owner", "from_owner_label"), ("to_owner", "to_owner_label")):
        owner = record.get(field)
        if owner not in EXECUTOR_PROFILES:
            errors.append(f"owner retarget {field} must be a known coding owner profile")
            continue
        owners[field] = str(owner)
        if record.get(label_field) != executor_label(str(owner)):
            errors.append(f"owner retarget {label_field} must be the label of {field}")
    if len(owners) == 2 and owners["from_owner"] == owners["to_owner"]:
        errors.append("owner retarget from_owner and to_owner must differ; a retarget moves the work")
    return errors


def _owner_delta_errors(delta: Any) -> list[str]:
    if not isinstance(delta, Mapping):
        return ["owner retarget owner_delta must be a mapping"]
    errors = _closed_key_errors("owner delta", delta, OWNER_DELTA_KEYS)
    for field in ("observed_evidence_gained", "observed_evidence_lost", "wrapper_actions_gained", "wrapper_actions_lost"):
        errors.extend(_text_list_errors(f"owner delta {field}", delta.get(field), required=False))
    changed = delta.get("changed")
    unchanged = delta.get("unchanged")
    if not isinstance(changed, list) or not isinstance(unchanged, list):
        errors.append("owner delta changed and unchanged must be lists")
        return errors
    changed_fields: list[str] = []
    for entry in changed:
        if not isinstance(entry, Mapping):
            errors.append("each owner delta change must be a mapping")
            continue
        entry_errors = _closed_key_errors("owner delta change", entry, OWNER_DELTA_CHANGE_KEYS)
        field = entry.get("field")
        if field not in OWNER_SPECIFIC_FIELDS:
            entry_errors.append(f"owner delta change field must be one of {', '.join(OWNER_SPECIFIC_FIELDS)}")
        for side in ("from", "to"):
            value = entry.get(side)
            if not isinstance(value, str) or len(value) > _MAX_TEXT_LENGTH:
                entry_errors.append(f"owner delta change {side} must be a string of at most {_MAX_TEXT_LENGTH} characters")
        if not entry_errors and entry["from"] == entry["to"]:
            entry_errors.append("owner delta change must record a field that actually moved")
        errors.extend(entry_errors)
        if not entry_errors:
            changed_fields.append(str(field))
    # Every owner-specific field lands on exactly one side. A field that is
    # neither changed nor unchanged would be a difference nobody was told about.
    partition = sorted([*changed_fields, *(str(field) for field in unchanged)])
    if partition != sorted(OWNER_SPECIFIC_FIELDS):
        errors.append("owner delta must place every owner-specific field in exactly one of changed or unchanged")
    return errors


def _capability_delta_errors(delta: Any) -> tuple[list[str], Mapping[str, Any] | None]:
    if not isinstance(delta, Mapping):
        return ["owner retarget capability_delta must be a mapping"], None
    errors = _closed_key_errors("capability delta", delta, CAPABILITY_DELTA_KEYS)
    for field in ("requirements", "required_by_owner_change", "dropped_by_owner_change", "unsupported", "unproven"):
        errors.extend(_text_list_errors(f"capability delta {field}", delta.get(field), required=False))
    for field in ("from_verdict", "to_verdict"):
        if delta.get(field) not in OWNER_FIT_VERDICTS:
            errors.append(f"capability delta {field} must be one of {', '.join(OWNER_FIT_VERDICTS)}")
    statement = delta.get("statement")
    if not isinstance(statement, str) or not statement.strip() or len(statement) > _MAX_REASON_LENGTH:
        errors.append("capability delta statement must be a nonempty string naming the gap or its absence")
    requirements = delta.get("requirements")
    declared = {str(name) for name in requirements} if isinstance(requirements, list) else set()
    for field in ("required_by_owner_change", "unsupported", "unproven"):
        values = delta.get(field)
        if isinstance(values, list) and not {str(name) for name in values} <= declared:
            errors.append(f"capability delta {field} must only name declared required capabilities")
    changed = delta.get("changed")
    if not isinstance(changed, list):
        errors.append("capability delta changed must be a list")
        return errors, delta
    if len(changed) > _MAX_CAPABILITIES:
        errors.append(f"capability delta changed must hold at most {_MAX_CAPABILITIES} entries")
    for entry in changed:
        if not isinstance(entry, Mapping):
            errors.append("each capability change must be a mapping")
            continue
        entry_errors = _closed_key_errors("capability change", entry, CAPABILITY_CHANGE_KEYS)
        if str(entry.get("capability", "")) not in declared:
            entry_errors.append("capability change must name a declared required capability")
        for side in ("from_classification", "to_classification"):
            if entry.get(side) not in OWNER_FIT_CLASSIFICATIONS:
                entry_errors.append(f"capability change {side} must be one of {', '.join(OWNER_FIT_CLASSIFICATIONS)}")
        if not entry_errors and entry["from_classification"] == entry["to_classification"]:
            entry_errors.append("capability change must record a classification that actually moved")
        errors.extend(entry_errors)
    return errors, delta


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


def _text_list_errors(label: str, value: Any, *, required: bool) -> list[str]:
    if not isinstance(value, list):
        return [f"{label} must be a list"]
    if required and not value:
        return [f"{label} must not be empty"]
    if len(value) > _MAX_LIST_ITEMS:
        return [f"{label} must hold at most {_MAX_LIST_ITEMS} entries"]
    if any(not isinstance(item, str) or not item.strip() or len(item) > _MAX_TEXT_LENGTH for item in value):
        return [f"{label} entries must be nonempty strings of at most {_MAX_TEXT_LENGTH} characters"]
    return []


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value]


def _missing_from(values: Sequence[str], other: Sequence[str]) -> list[str]:
    absent = set(other)
    return [value for value in values if value not in absent]


def _scalar_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_digest(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
