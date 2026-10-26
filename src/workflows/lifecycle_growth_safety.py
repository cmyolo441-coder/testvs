"""Provider-neutral lifecycle safety records (issue #1400).

Three bounded records that ride inside the existing lifecycle-growth
artifacts, plus one pure routing decision:

- a throttle grouping record that keeps the configured key separate from the
  value it resolved to, so distinct dynamic values never collapse into one
  window and a legacy static path keeps the identity it always had;
- a per-step outcome record that names matched and skipped steps with a
  distinct reason and status, carries no evaluated values, and folds any
  secret-shaped or body-shaped reason into a digest handle;
- a workflow-content mutation route that treats production content as
  view-only and sends every edit through a development draft, an explicit
  promotion decision, and an observed provider result.

Concept-level prior art only (Novu `c7bc772f`, MIT community code; no storage
keys, flags, enums, dashboard state, or code adopted).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final

from ..system.append_only_store import redacted_ref
from .lifecycle_growth_values import (
    MAX_REFERENCES,
    metadata_ref,
    ref_errors,
    require_state,
    state_errors,
)


THROTTLE_KEY_KINDS: Final = ("static_path", "dynamic_expression")
THROTTLE_SCOPES: Final = ("recipient", "tenant")
THROTTLE_RESOLVED_STATES: Final = ("present", "missing", "empty")
THROTTLE_WINDOW_RESETS: Final = ("none", "one_time_reset")
THROTTLE_GROUPING_INPUT_KEYS: Final = frozenset(
    {"key_kind", "configured_key_ref", "scope", "resolved_value_state", "resolved_value_ref", "window_reset_consequence"}
)
THROTTLE_GROUPING_KEYS: Final = frozenset(
    {
        "key_kind", "configured_key_ref", "scope", "resolved_value_state", "resolved_value_ref",
        "fallback_state", "group_identity", "window_reset_consequence",
    }
)

STEP_OUTCOMES: Final = ("matched", "skipped")
STEP_STATUSES: Final = {"matched": "step_proceeded", "skipped": "step_skipped"}
STEP_TRACE_STATES: Final = ("recorded", "write_failed", "not_attempted")
STEP_OUTCOME_KEYS: Final = frozenset({"step_ref", "outcome", "status", "reason_code", "evaluated_values_state"})
EVALUATED_VALUES_STATE: Final = "redacted"

WORKFLOW_CONTENT_STATES: Final = ("production_read_only", "development_draft")
MUTATION_ROUTES: Final = ("development_draft_then_promotion",)
PROMOTION_DECISION_STATES: Final = ("not_requested", "pending", "approved")
PROMOTION_RESULT_STATES: Final = ("not_observed", "observed")
LIFECYCLE_WORKFLOW_MUTATION_SCHEMA_VERSION: Final = "lifecycle_workflow_mutation/v1"


def build_throttle_grouping(*, key_kind: str, configured_key_ref: str, scope: str, resolved_value_state: str, resolved_value_ref: str, window_reset_consequence: str) -> dict[str, object]:
    """One throttle grouping record whose identity is derived, never caller-authored."""
    kind = require_state(key_kind, field="key_kind", allowed=THROTTLE_KEY_KINDS)
    key = metadata_ref(configured_key_ref, field="configured_key_ref")
    state = require_state(resolved_value_state, field="resolved_value_state", allowed=_resolved_states_for(kind))
    value = metadata_ref(resolved_value_ref, field="resolved_value_ref") if state == "present" else ""
    if state != "present" and str(resolved_value_ref or ""):
        raise ValueError("resolved_value_ref must be empty unless resolved_value_state is present")
    record = {
        "key_kind": kind,
        "configured_key_ref": key,
        "scope": require_state(scope, field="scope", allowed=THROTTLE_SCOPES),
        "resolved_value_state": state,
        "resolved_value_ref": value,
        "fallback_state": _fallback_state(state),
        "group_identity": "",
        "window_reset_consequence": require_state(window_reset_consequence, field="window_reset_consequence", allowed=THROTTLE_WINDOW_RESETS),
    }
    record["group_identity"] = derive_throttle_group_identity(record)
    return record


def derive_throttle_group_identity(record: Mapping[str, Any]) -> str:
    """Group identity from the configured key and the resolved value, kept apart.

    A resolved value is only ever a value: it is never re-read as a second
    configured key, so two dynamic values yield two identities and a value
    that happens to look like a path still groups under its own expression.
    """
    kind = record.get("key_kind")
    key = record.get("configured_key_ref")
    state = record.get("resolved_value_state")
    if state == "present":
        return f"{kind}:{key}={record.get('resolved_value_ref')}"
    if state == "missing":
        return "ungrouped"
    return f"default_window:{key}"


def throttle_grouping_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["throttle_grouping must be an object"]
    errors: list[str] = []
    keys = {str(key) for key in value}
    missing = sorted(THROTTLE_GROUPING_KEYS - keys)
    unexpected = sorted(keys - THROTTLE_GROUPING_KEYS)
    if missing:
        errors.append(f"throttle_grouping missing keys: {missing}")
    if unexpected:
        errors.append(f"throttle_grouping has unsupported keys: {unexpected}")
    kind = value.get("key_kind")
    errors.extend(state_errors(kind, field="throttle_grouping.key_kind", allowed=THROTTLE_KEY_KINDS))
    errors.extend(ref_errors(value.get("configured_key_ref"), field="throttle_grouping.configured_key_ref", required=True))
    errors.extend(state_errors(value.get("scope"), field="throttle_grouping.scope", allowed=THROTTLE_SCOPES))
    state = value.get("resolved_value_state")
    errors.extend(state_errors(state, field="throttle_grouping.resolved_value_state", allowed=_resolved_states_for(kind)))
    errors.extend(ref_errors(value.get("resolved_value_ref"), field="throttle_grouping.resolved_value_ref", required=state == "present"))
    if state != "present" and value.get("resolved_value_ref"):
        errors.append("throttle_grouping.resolved_value_ref must be empty unless resolved_value_state is present")
    if value.get("fallback_state") != _fallback_state(state):
        errors.append("throttle_grouping.fallback_state must match resolved_value_state")
    errors.extend(state_errors(value.get("window_reset_consequence"), field="throttle_grouping.window_reset_consequence", allowed=THROTTLE_WINDOW_RESETS))
    if value.get("group_identity") != derive_throttle_group_identity(value):
        errors.append("throttle_grouping.group_identity must match derived identity")
    return errors


def build_step_outcome(*, step_ref: str, outcome: str, reason_code: str) -> dict[str, object]:
    """One conditional step's matched or skipped outcome, with no evaluated values.

    The reason is folded through `redacted_ref`: a secret-shaped value, a
    whole-context blob, or anything longer than one bounded identifier becomes
    a digest handle, so the record cannot carry environment secrets.
    """
    result = require_state(outcome, field="outcome", allowed=STEP_OUTCOMES)
    reason = redacted_ref(reason_code, field="reason_code")
    if not reason:
        raise ValueError("reason_code is required")
    return {
        "step_ref": metadata_ref(step_ref, field="step_ref"),
        "outcome": result,
        "status": STEP_STATUSES[result],
        "reason_code": reason,
        "evaluated_values_state": EVALUATED_VALUES_STATE,
    }


def step_outcomes_errors(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCES:
        return [f"step_outcomes must be a list of at most {MAX_REFERENCES} step outcome records"]
    errors: list[str] = []
    for index, entry in enumerate(value):
        field = f"step_outcomes[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(f"{field} must be an object")
            continue
        keys = {str(key) for key in entry}
        if keys != STEP_OUTCOME_KEYS:
            errors.append(f"{field} must carry exactly {sorted(STEP_OUTCOME_KEYS)}")
        errors.extend(ref_errors(entry.get("step_ref"), field=f"{field}.step_ref", required=True))
        outcome = entry.get("outcome")
        errors.extend(state_errors(outcome, field=f"{field}.outcome", allowed=STEP_OUTCOMES))
        if entry.get("status") != STEP_STATUSES.get(outcome):
            errors.append(f"{field}.status must match outcome")
        errors.extend(ref_errors(entry.get("reason_code"), field=f"{field}.reason_code", required=True))
        if entry.get("evaluated_values_state") != EVALUATED_VALUES_STATE:
            errors.append(f"{field}.evaluated_values_state must be redacted")
    return errors


def route_lifecycle_workflow_mutation(safety: Mapping[str, Any]) -> dict[str, object]:
    """Where an edit to workflow content may go, derived from the safety policy only."""
    content_state = safety.get("workflow_content_state")
    decision = safety.get("promotion_decision_state")
    result = safety.get("promotion_result_state")
    reasons: list[str] = []
    if content_state not in WORKFLOW_CONTENT_STATES:
        reasons.append("workflow_content_state is invalid")
    elif content_state == "production_read_only":
        reasons.append("production workflow content is view-only; route the edit to a development draft")
    if safety.get("mutation_route") not in MUTATION_ROUTES:
        reasons.append("mutation_route is invalid")
    if decision not in PROMOTION_DECISION_STATES:
        reasons.append("promotion_decision_state is invalid")
    elif decision != "approved":
        reasons.append("promotion decision is not approved")
    if result not in PROMOTION_RESULT_STATES:
        reasons.append("promotion_result_state is invalid")
    return {
        "schema_version": LIFECYCLE_WORKFLOW_MUTATION_SCHEMA_VERSION,
        "verdict": "HOLD" if reasons else "READY",
        "content_view_state": "view_only" if content_state == "production_read_only" else "editable_draft",
        "mutation_target": "development_draft",
        "promotion_decision_state": decision if decision in PROMOTION_DECISION_STATES else "",
        "promotion_result_state": result if result in PROMOTION_RESULT_STATES else "",
        "reason_codes": reasons,
        "claim_boundary": (
            "Read-only guidance is not proof that a provider blocked a mutation, and an approved "
            "promotion decision is not an observed provider result."
        ),
    }


def workflow_mutation_hold_reasons(safety: Mapping[str, Any]) -> list[str]:
    return list(route_lifecycle_workflow_mutation(safety)["reason_codes"])


def _resolved_states_for(kind: object) -> tuple[str, ...]:
    if kind == "static_path":
        return ("present", "missing")
    if kind == "dynamic_expression":
        return ("present", "empty")
    return THROTTLE_RESOLVED_STATES


def _fallback_state(state: object) -> str:
    if state == "missing":
        return "ungrouped"
    if state == "empty":
        return "default_window"
    return "none"


def step_outcome_records(values: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    """Rebuild caller-supplied step outcomes through the builder so nothing else gets in."""
    if isinstance(values, (str, Mapping)) or len(values) > MAX_REFERENCES:
        raise ValueError(f"step_outcomes must contain at most {MAX_REFERENCES} step outcome records")
    records: list[dict[str, object]] = []
    for value in values:
        if not isinstance(value, Mapping) or {str(key) for key in value} != {"step_ref", "outcome", "reason_code"}:
            raise ValueError("step_outcomes entries must carry exactly step_ref, outcome, and reason_code")
        records.append(build_step_outcome(step_ref=str(value["step_ref"]), outcome=str(value["outcome"]), reason_code=str(value["reason_code"])))
    return records
