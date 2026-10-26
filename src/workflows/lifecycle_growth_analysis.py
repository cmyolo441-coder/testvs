"""Provider-neutral lifecycle analysis-run state (issue #1429).

A lifecycle readout used to say only what the numbers were, so a missing
readout collapsed into "review" or "insufficient_data" whether the analysis had
never been asked for, was sitting in a queue, was still running, had failed,
had been canceled, or had finished and returned nothing. Those are six
different situations with six different next actions, and the contract could
not tell them apart.

Four bounded pieces close that gap, and none of them reads a clock, queries a
provider, or starts anything:

- an analysis status record that carries one explicit run state, a safe run
  reference, the time the state was observed, and the run's elapsed time;
- a delay state derived from a caller-supplied service expectation, kept
  strictly beside the run state so age can raise a warning but can never
  rewrite a queued run into a failed or absent one, and no fixed cutoff (a
  twenty-four-hour rule included) is written into the contract;
- a selection over several observed run records that prefers the newest
  in-flight run, so a request routed against it holds instead of duplicating
  work that is already queued or running;
- a cancellation handoff that keeps three things apart: the prepared request,
  which is always `prepared_not_observed`; an observed provider
  acknowledgement; and an observed terminal cancellation. Only the third one
  may back a `canceled` run state.

Concept-level prior art only (GrowthBook `095f6164`, MIT community code; no
model, controller, endpoint, schema, query state, storage key, or timeout
adopted).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Final

from .lifecycle_growth_values import (
    MAX_REFERENCES,
    PREPARED_STATUS,
    count_errors,
    metadata_ref,
    metadata_refs,
    positive_count,
    ref_errors,
    refs_errors,
    require_nonnegative_count,
    require_state,
    state_errors,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


ANALYSIS_RUN_STATES: Final = ("not_started", "queued", "running", "completed", "failed", "canceled", "unknown")
IN_FLIGHT_ANALYSIS_STATES: Final = ("queued", "running")
TERMINAL_ANALYSIS_STATES: Final = ("completed", "failed", "canceled")
# States that only exist because a provider reported something, so a bare
# assertion without an evidence handle is not enough to record them.
OBSERVED_ANALYSIS_STATES: Final = IN_FLIGHT_ANALYSIS_STATES + TERMINAL_ANALYSIS_STATES
ANALYSIS_DELAY_STATES: Final = ("not_applicable", "within_expectation", "delayed", "unknown")
ANALYSIS_STATUS_INPUT_KEYS: Final = frozenset(
    {"run_state", "run_ref", "observed_at", "elapsed_minutes", "service_expectation_minutes", "evidence_refs"}
)
ANALYSIS_STATUS_KEYS: Final = ANALYSIS_STATUS_INPUT_KEYS | {"delay_state"}

CANCELLATION_REQUEST_STATES: Final = ("not_requested", "prepared")
CANCELLATION_SCOPES: Final = ("single_run", "target_runs")
CANCELLATION_ACKNOWLEDGEMENT_STATES: Final = ("not_observed", "observed_accepted", "observed_rejected")
CANCELLATION_RESULT_STATES: Final = ("not_observed", "observed_canceled", "observed_not_canceled")
ANALYSIS_CANCELLATION_INPUT_KEYS: Final = frozenset(
    {"request_state", "run_ref", "cancel_scope", "acknowledgement_state", "result_state", "evidence_refs"}
)
ANALYSIS_CANCELLATION_KEYS: Final = ANALYSIS_CANCELLATION_INPUT_KEYS | {"handoff_status"}

ANALYSIS_RECONCILIATION_STATES: Final = ("not_requested", "pending", "reconciled")
LIFECYCLE_ANALYSIS_SELECTION_SCHEMA_VERSION: Final = "lifecycle_analysis_selection/v1"
LIFECYCLE_ANALYSIS_REQUEST_SCHEMA_VERSION: Final = "lifecycle_analysis_request/v1"

_SELECTION_CLAIM_BOUNDARY: Final = (
    "Selection reads caller-supplied provider observations only. It does not query a provider, prove the "
    "selected run still exists, or start, cancel, or complete any analysis."
)
_REQUEST_CLAIM_BOUNDARY: Final = (
    "A READY verdict means no observed in-flight run blocks preparing another analysis. It is not "
    "authorization, a provider request, or a started run."
)
_EPOCH: Final = datetime.min.replace(tzinfo=timezone.utc)


def build_analysis_status(*, run_state: str, run_ref: str = "", observed_at: str = "", elapsed_minutes: int = 0, service_expectation_minutes: int = 0, evidence_refs: Sequence[str] = ()) -> dict[str, object]:
    """One observed analysis run, with its delay state derived rather than told.

    A `not_started` record may still carry `observed_at`: "nothing was running
    when we looked" is an observation with a time. It may not carry a run
    reference, elapsed time, or evidence, because there is no run to reference.
    """
    state = require_state(run_state, field="run_state", allowed=ANALYSIS_RUN_STATES)
    started = state != "not_started"
    if not started and (str(run_ref or "") or elapsed_minutes or tuple(evidence_refs)):
        raise ValueError("a not_started analysis carries no run reference, elapsed time, or evidence")
    record: dict[str, object] = {
        "run_ref": metadata_ref(run_ref, field="run_ref") if started else "",
        "run_state": state,
        "observed_at": require_observed_at(observed_at, field="observed_at", required=started),
        "elapsed_minutes": require_nonnegative_count(elapsed_minutes, field="elapsed_minutes"),
        "service_expectation_minutes": require_nonnegative_count(service_expectation_minutes, field="service_expectation_minutes"),
        "delay_state": "",
        "evidence_refs": metadata_refs(evidence_refs, field="evidence_refs", required=state in OBSERVED_ANALYSIS_STATES),
    }
    record["delay_state"] = derive_analysis_delay_state(record)
    return record


def derive_analysis_delay_state(record: Mapping[str, Any]) -> str:
    """Whether in-flight work has outrun a supplied expectation, and nothing more.

    Age is a warning about work that is taking longer than the caller expected.
    It is never evidence that a run failed, was canceled, or disappeared, so
    this answers a separate question from `run_state` and no caller may derive
    one from the other. A settled run has no delay to report; an in-flight run
    with no supplied expectation has nothing to compare against.
    """
    if record.get("run_state") not in IN_FLIGHT_ANALYSIS_STATES:
        return "not_applicable"
    expectation = record.get("service_expectation_minutes")
    elapsed = record.get("elapsed_minutes")
    if not positive_count(expectation) or not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0:
        return "unknown"
    return "delayed" if elapsed > expectation else "within_expectation"


def analysis_status_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["analysis_status must be an object"]
    errors: list[str] = []
    keys = {str(key) for key in value}
    missing = sorted(ANALYSIS_STATUS_KEYS - keys)
    unexpected = sorted(keys - ANALYSIS_STATUS_KEYS)
    if missing:
        errors.append(f"analysis_status missing keys: {missing}")
    if unexpected:
        errors.append(f"analysis_status has unsupported keys: {unexpected}")
    state = value.get("run_state")
    errors.extend(state_errors(state, field="analysis_status.run_state", allowed=ANALYSIS_RUN_STATES))
    started = state in ANALYSIS_RUN_STATES and state != "not_started"
    errors.extend(ref_errors(value.get("run_ref"), field="analysis_status.run_ref", required=started))
    errors.extend(observed_at_errors(value.get("observed_at"), field="analysis_status.observed_at", required=started))
    errors.extend(count_errors(value.get("elapsed_minutes"), field="analysis_status.elapsed_minutes"))
    errors.extend(count_errors(value.get("service_expectation_minutes"), field="analysis_status.service_expectation_minutes"))
    errors.extend(refs_errors(value.get("evidence_refs"), field="analysis_status.evidence_refs", required=state in OBSERVED_ANALYSIS_STATES))
    if state == "not_started":
        for field in ("run_ref", "elapsed_minutes", "evidence_refs"):
            if value.get(field):
                errors.append(f"analysis_status.{field} must be empty when no analysis has started")
    if value.get("delay_state") != derive_analysis_delay_state(value):
        errors.append("analysis_status.delay_state must match the derived delay state")
    return errors


def analysis_status_record(value: Mapping[str, Any]) -> dict[str, object]:
    """Rebuild a caller-supplied analysis status through the builder."""
    if not isinstance(value, Mapping) or {str(key) for key in value} != ANALYSIS_STATUS_INPUT_KEYS:
        raise ValueError(f"analysis_status must carry exactly {sorted(ANALYSIS_STATUS_INPUT_KEYS)}")
    return build_analysis_status(
        run_state=str(value["run_state"]),
        run_ref=str(value["run_ref"]),
        observed_at=str(value["observed_at"]),
        elapsed_minutes=value["elapsed_minutes"],
        service_expectation_minutes=value["service_expectation_minutes"],
        evidence_refs=value["evidence_refs"],
    )


def select_latest_analysis_run(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """The newest in-flight run among observed records, or the newest settled one.

    In-flight beats settled regardless of age: an old queued run is still the
    work that a new request would duplicate. Ordering is by observed time, with
    the supplied position as the tiebreak, so the selection is deterministic
    for records that share a timestamp.
    """
    if isinstance(records, (str, Mapping)) or not isinstance(records, Sequence):
        return _selection(None, 0, ["analysis run records must be a sequence"])
    if len(records) > MAX_REFERENCES:
        return _selection(None, 0, [f"analysis run records must number at most {MAX_REFERENCES}"])
    reasons: list[str] = []
    candidates: list[tuple[int, Mapping[str, Any]]] = []
    for index, record in enumerate(records):
        if analysis_status_errors(record):
            reasons.append(f"analysis run record {index} is invalid")
            continue
        if record.get("run_state") != "not_started":
            candidates.append((index, record))
    in_flight = [item for item in candidates if item[1].get("run_state") in IN_FLIGHT_ANALYSIS_STATES]
    pool = in_flight or candidates
    selected = max(pool, key=_selection_order)[1] if pool else None
    return _selection(selected, len(in_flight), reasons)


def route_lifecycle_analysis_request(records: Sequence[Mapping[str, Any]], *, reconciliation_state: str) -> dict[str, object]:
    """Whether another analysis may be prepared, given what is already in flight."""
    selection = select_latest_analysis_run(records)
    reasons = list(selection["reason_codes"])
    known_state = reconciliation_state in ANALYSIS_RECONCILIATION_STATES
    if not known_state:
        reasons.append("reconciliation_state is invalid")
    duplicate = selection["selection_state"] == "in_flight"
    if duplicate and reconciliation_state != "reconciled":
        reasons.append(_IN_FLIGHT_HOLD_REASON)
    return {
        "schema_version": LIFECYCLE_ANALYSIS_REQUEST_SCHEMA_VERSION,
        "verdict": "HOLD" if reasons else "READY",
        "selection_state": selection["selection_state"],
        "selected_run_ref": selection["selected_run_ref"],
        "selected_run_state": selection["selected_run_state"],
        "selected_observed_at": selection["selected_observed_at"],
        "selected_delay_state": selection["selected_delay_state"],
        "in_flight_count": selection["in_flight_count"],
        "duplicate_risk_state": "in_flight_run_present" if duplicate else "none",
        "reconciliation_state": reconciliation_state if known_state else "",
        "reason_codes": reasons,
        "claim_boundary": _REQUEST_CLAIM_BOUNDARY,
    }


def build_analysis_cancellation(*, request_state: str, run_ref: str = "", cancel_scope: str = "", acknowledgement_state: str = "not_observed", result_state: str = "not_observed", evidence_refs: Sequence[str] = ()) -> dict[str, object]:
    """A cancellation handoff whose prepared request never becomes a provider action.

    `handoff_status` is `prepared_not_observed` for every record, including one
    that carries an observed acknowledgement or an observed terminal
    cancellation: OMH prepared the request either way, and the observation
    belongs to the provider, not to the handoff.
    """
    state = require_state(request_state, field="request_state", allowed=CANCELLATION_REQUEST_STATES)
    acknowledgement = require_state(acknowledgement_state, field="acknowledgement_state", allowed=CANCELLATION_ACKNOWLEDGEMENT_STATES)
    result = require_state(result_state, field="result_state", allowed=CANCELLATION_RESULT_STATES)
    prepared = state == "prepared"
    if not prepared and (str(run_ref or "") or str(cancel_scope or "") or acknowledgement != "not_observed" or result != "not_observed" or tuple(evidence_refs)):
        raise ValueError("an unrequested cancellation carries no run reference, scope, observation, or evidence")
    if result != "not_observed" and acknowledgement == "not_observed":
        raise ValueError("an observed cancellation result requires an observed provider acknowledgement")
    if result == "observed_canceled" and acknowledgement != "observed_accepted":
        raise ValueError("an observed terminal cancellation requires an accepted cancellation request")
    return {
        "request_state": state,
        "handoff_status": PREPARED_STATUS,
        "run_ref": metadata_ref(run_ref, field="cancellation run_ref") if prepared else "",
        "cancel_scope": require_state(cancel_scope, field="cancel_scope", allowed=CANCELLATION_SCOPES) if prepared else "",
        "acknowledgement_state": acknowledgement,
        "result_state": result,
        "evidence_refs": metadata_refs(evidence_refs, field="cancellation evidence_refs", required=acknowledgement != "not_observed" or result != "not_observed"),
    }


def analysis_cancellation_errors(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return ["analysis_cancellation must be an object"]
    errors: list[str] = []
    keys = {str(key) for key in value}
    missing = sorted(ANALYSIS_CANCELLATION_KEYS - keys)
    unexpected = sorted(keys - ANALYSIS_CANCELLATION_KEYS)
    if missing:
        errors.append(f"analysis_cancellation missing keys: {missing}")
    if unexpected:
        errors.append(f"analysis_cancellation has unsupported keys: {unexpected}")
    state = value.get("request_state")
    acknowledgement = value.get("acknowledgement_state")
    result = value.get("result_state")
    errors.extend(state_errors(state, field="analysis_cancellation.request_state", allowed=CANCELLATION_REQUEST_STATES))
    errors.extend(state_errors(acknowledgement, field="analysis_cancellation.acknowledgement_state", allowed=CANCELLATION_ACKNOWLEDGEMENT_STATES))
    errors.extend(state_errors(result, field="analysis_cancellation.result_state", allowed=CANCELLATION_RESULT_STATES))
    prepared = state == "prepared"
    errors.extend(ref_errors(value.get("run_ref"), field="analysis_cancellation.run_ref", required=prepared))
    errors.extend(state_errors(value.get("cancel_scope"), field="analysis_cancellation.cancel_scope", allowed=CANCELLATION_SCOPES if prepared else ("",)))
    errors.extend(refs_errors(value.get("evidence_refs"), field="analysis_cancellation.evidence_refs", required=_is_observed(acknowledgement) or _is_observed(result)))
    if value.get("handoff_status") != PREPARED_STATUS:
        errors.append("analysis_cancellation.handoff_status must be prepared_not_observed")
    if not prepared:
        for field in ("run_ref", "acknowledgement_state", "result_state", "evidence_refs"):
            if value.get(field) not in ("", "not_observed", [], (), None):
                errors.append(f"analysis_cancellation.{field} must be empty when no cancellation is requested")
    if _is_observed(result) and not _is_observed(acknowledgement):
        errors.append("analysis_cancellation.result_state requires an observed provider acknowledgement")
    if result == "observed_canceled" and acknowledgement != "observed_accepted":
        errors.append("analysis_cancellation.result_state observed_canceled requires an accepted request")
    return errors


def analysis_cancellation_record(value: Mapping[str, Any]) -> dict[str, object]:
    """Rebuild a caller-supplied cancellation handoff through the builder."""
    if not isinstance(value, Mapping) or {str(key) for key in value} != ANALYSIS_CANCELLATION_INPUT_KEYS:
        raise ValueError(f"analysis_cancellation must carry exactly {sorted(ANALYSIS_CANCELLATION_INPUT_KEYS)}")
    return build_analysis_cancellation(
        request_state=str(value["request_state"]),
        run_ref=str(value["run_ref"]),
        cancel_scope=str(value["cancel_scope"]),
        acknowledgement_state=str(value["acknowledgement_state"]),
        result_state=str(value["result_state"]),
        evidence_refs=value["evidence_refs"],
    )


def analysis_cancellation_hold_reasons(status: Any, cancellation: Any) -> list[str]:
    """Where an observed run state and a cancellation handoff contradict each other."""
    if analysis_status_errors(status) or analysis_cancellation_errors(cancellation):
        return []
    run_state = status.get("run_state")
    result = cancellation.get("result_state")
    reasons: list[str] = []
    if run_state == "canceled" and result != "observed_canceled":
        reasons.append("a canceled analysis state requires an observed cancellation result")
    if result == "observed_canceled" and run_state in IN_FLIGHT_ANALYSIS_STATES:
        reasons.append("an observed cancellation result contradicts an in-flight analysis state")
    if cancellation.get("request_state") == "prepared" and cancellation.get("cancel_scope") == "single_run" and run_state != "not_started" and cancellation.get("run_ref") != status.get("run_ref"):
        reasons.append("the prepared cancellation targets a different run than the observed analysis")
    return reasons


def analysis_run_state(record: Mapping[str, Any]) -> str:
    """The readout's observed analysis state, or an empty string when it has none."""
    status = record.get("analysis_status")
    state = status.get("run_state") if isinstance(status, Mapping) else None
    return state if state in ANALYSIS_RUN_STATES else ""


def analysis_status_field(record: Mapping[str, Any], field: str) -> str:
    status = record.get("analysis_status")
    value = status.get(field) if isinstance(status, Mapping) else None
    return value if isinstance(value, str) else ""


def require_observed_at(value: str, *, field: str, required: bool) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError(f"{field} is required")
        return ""
    if _observed_time(text) is None:
        raise ValueError(f"{field} must be an ISO-8601 timestamp with a time-zone offset")
    return text


def observed_at_errors(value: Any, *, field: str, required: bool) -> list[str]:
    if not isinstance(value, str):
        return [f"{field} must be a string"]
    if not value:
        return [f"{field} is required"] if required else []
    if _observed_time(value) is None:
        return [f"{field} must be an ISO-8601 timestamp with a time-zone offset"]
    return []


_IN_FLIGHT_HOLD_REASON: Final = "the latest analysis run is still in flight; reconcile it before preparing another"


def _selection(selected: Mapping[str, Any] | None, in_flight_count: int, reasons: list[str]) -> dict[str, object]:
    in_flight = bool(selected) and selected.get("run_state") in IN_FLIGHT_ANALYSIS_STATES
    return {
        "schema_version": LIFECYCLE_ANALYSIS_SELECTION_SCHEMA_VERSION,
        "selection_state": ("in_flight" if in_flight else "settled") if selected else "none",
        "selected_run_ref": str(selected.get("run_ref", "")) if selected else "",
        "selected_run_state": str(selected.get("run_state", "")) if selected else "",
        "selected_observed_at": str(selected.get("observed_at", "")) if selected else "",
        "selected_delay_state": str(selected.get("delay_state", "")) if selected else "not_applicable",
        "selected_evidence_refs": list(selected.get("evidence_refs", [])) if selected else [],
        "in_flight_count": in_flight_count,
        "reason_codes": reasons,
        "claim_boundary": _SELECTION_CLAIM_BOUNDARY,
    }


def _selection_order(item: tuple[int, Mapping[str, Any]]) -> tuple[datetime, int]:
    index, record = item
    return _observed_time(record.get("observed_at")) or _EPOCH, index


def _is_observed(value: object) -> bool:
    return isinstance(value, str) and value not in ("", "not_observed")


def _observed_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = strict_fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed
