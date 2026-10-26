"""Parent-owned decision ledger beside one frozen fanout contract.

Bounded question/answer metadata is explicit, not raw prompt retention. OS file
locks serialize answer and redispatch claims; child depth never grants authority.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import assert_never

from ..runtime.artifacts import append_journal_observation
from ..system.local_store import atomic_write_json, file_lock, is_directory_link
from ..system.paths import OmhPaths
from .fanout_artifacts import unit_result_path, read_fanout_contract
from .fanout_unit_results import read_unit_result_input
from .fanout_clarification_schema import (
    ClarificationAnswer, ClarificationError, MAX_CLARIFICATION_ROUNDS,
    metadata_text,
)
from .fanout_clarification_records import (
    ClarificationRecord, ClarificationInput, ClarificationView, RenderableQuestion,
    parse_clarification_record, history_receipt,
)
from .fanout_failure_diagnostics import is_string_map, is_object_list
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


def clarification_path(paths: OmhPaths, fanout_id: str, unit_id: str) -> Path:
    directory = unit_result_path(paths, fanout_id, unit_id).parent.parent / "clarifications"
    if is_directory_link(directory):
        raise ClarificationError("clarification_directory_link")
    return directory / (unit_id + ".json")


def read_clarification(path: Path) -> ClarificationRecord | None:
    if not path.exists():
        return None
    return parse_clarification_record(read_unit_result_input(path))


def clarification_state(record: ClarificationRecord, now: datetime | None = None) -> str:
    state = record["state"]
    if state == "prepared_not_answered" and (now or datetime.now(timezone.utc)) >= strict_fromisoformat(record["expires_at"]):
        return "expired"
    return state


def _event_refs(record: ClarificationRecord, phase: str) -> list[str]:
    """Bind observations to decision content, not mutable state or event pointers."""
    payload: dict[str, object] = dict(record)
    for field in ("state", "history", "request_event_ref", "answer_event_ref", "redispatch_event_ref"):
        payload.pop(field)
    if phase not in ("answer", "redispatch"):
        payload.pop("answer")
    payload["observation_phase"] = phase
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return [
        f"clarification:{record['decision_id']}:{record['round']}",
        f"clarification-sha256:{digest}",
    ]


def clarification_event(paths: OmhPaths, record: ClarificationRecord, phase: str) -> str:
    event = append_journal_observation(paths, {
        "target_type": "run", "target_id": record["run_ref"], "run_id": record["run_ref"],
        "worker_ref": record["unit_id"], "attempt_id": record["attempt_id"],
        "event": {"request": "blocked", "answer": "plan_artifact_created", "cancel": "cancelled",
                  "redispatch": "executor_dispatch_observed"}[phase],
        "phase": "clarification_" + phase, "status": "observed",
        "summary": "Parent clarification " + phase,
        "evidence_refs": _event_refs(record, phase)})
    return str(event["event_id"])


def prepare_clarification(paths: OmhPaths, record: ClarificationInput) -> ClarificationRecord:
    """Bind already parsed child metadata to dispatcher-owned lineage and workspace."""
    path = clarification_path(paths, record["fanout_id"], record["unit_id"])
    with file_lock(path, private=True) as lock:
        if not lock["enforced"]:
            raise ClarificationError("clarification_lock_unavailable")
        previous = read_clarification(path)
        if previous and previous["goal_attempt_id"] != record["goal_attempt_id"]:
            raise ClarificationError("clarification_goal_attempt_changed")
        history = [*previous["history"], history_receipt(previous)][-2:] if previous else []
        round_number = min((previous["round"] if previous else 0) + 1, MAX_CLARIFICATION_ROUNDS + 1)
        now = datetime.now(timezone.utc)
        request: ClarificationRecord = {**record, "schema_version": "fanout_clarification_request/v1",
            "decision_id": record["input_required"]["decision_id"], "round": round_number,
            "requested_at": now.isoformat(), "expires_at": (now + timedelta(hours=24)).isoformat(),
            "state": "exhausted" if round_number > MAX_CLARIFICATION_ROUNDS else "prepared_not_answered",
            "history": history, "privacy": "bounded_decision_metadata", "answer": None,
            "redispatch_event_ref": None, "answer_event_ref": None, "request_event_ref": ""}
        request["request_event_ref"] = clarification_event(paths, request, "request")
        atomic_write_json(path, dict(request), private=True)
        return request


def project_clarifications(paths: OmhPaths, fanout_id: str) -> ClarificationView:
    contract = read_fanout_contract(paths, fanout_id)
    merge = contract.get("merge_plan")
    order = merge.get("merge_order") if is_string_map(merge) else None
    if not is_object_list(order) or not all(isinstance(unit, str) for unit in order):
        raise ClarificationError("clarification_contract_order_invalid")
    records = [read_clarification(clarification_path(paths, fanout_id, str(unit))) for unit in order]
    pending = [row for row in records if row and clarification_state(row) == "prepared_not_answered"]
    question: RenderableQuestion | None = None
    if pending:
        row = pending[0]
        question = {**row["input_required"], "fanout_id": row["fanout_id"], "unit_id": row["unit_id"],
            "run_ref": row["run_ref"], "base_sha": row["base_sha"], "attempt_id": row["attempt_id"],
            "goal_attempt_id": row["goal_attempt_id"], "round": row["round"],
            "authority": "question_only_no_scope_or_approval_change"}
    return {"schema_version": "fanout_clarification_view/v1", "renderable": question,
        "queued_units": [row["unit_id"] for row in pending[1:]],
        "units": [{"unit_id": row["unit_id"], "state": clarification_state(row)} for row in records if row]}


def answer_clarification(paths: OmhPaths, fanout_id: str, answer: ClarificationAnswer,
                         *, now: datetime | None = None) -> dict[str, object]:
    if os.environ.get("OMH_FANOUT_DEPTH", "0") != "0":
        raise ClarificationError("root_session_required")
    path = clarification_path(paths, fanout_id, answer.unit_id)
    with file_lock(path, private=True) as lock:
        if not lock["enforced"]:
            raise ClarificationError("clarification_lock_unavailable")
        record = read_clarification(path)
        if record is None:
            raise ClarificationError("stale_decision_id")
        for key, expected in (("decision_id", answer.decision_id), ("attempt_id", answer.attempt_id), ("round", answer.round)):
            if record.get(key) != expected:
                raise ClarificationError("stale_" + key)
        state = clarification_state(record, now)
        if state != "prepared_not_answered":
            raise ClarificationError("stale_" + state)
        if answer.answer is not None:
            value = metadata_text(answer.answer, "answer", 300)
            shape = record["input_required"]["answer_shape"]
            match shape["kind"]:
                case "options":
                    if value not in shape["options"]:
                        raise ClarificationError("answer_not_allowed_option")
                case "text":
                    if len(value) > shape["max_chars"]:
                        raise ClarificationError("answer_oversized")
                case unreachable:
                    assert_never(unreachable)
            record["answer"] = {"schema_version": "fanout_clarification_answer/v1",
                "decision_id": answer.decision_id, "attempt_id": answer.attempt_id,
                "round": answer.round, "answer": value, "source": "root_session_cli",
                "observed_at": (now or datetime.now(timezone.utc)).isoformat()}
        record["state"] = "cancelled" if answer.answer is None else "answered"
        record["answer_event_ref"] = clarification_event(paths, record, "cancel" if answer.answer is None else "answer")
        atomic_write_json(path, dict(record), private=True)
        return {"unit_id": answer.unit_id, "decision_id": answer.decision_id,
                "round": answer.round, "state": record["state"], "dispatch": "not_observed"}


def clarification_evidence(record: ClarificationRecord | None,
                           events: Sequence[Mapping[str, object]]) -> dict[str, str] | None:
    if record is None:
        return None
    expected_refs = {
        "clarification_" + phase: _event_refs(record, phase)
        for phase in ("request", "answer", "cancel", "redispatch")
    }
    phases = {str(event.get("phase")): event for event in events
        if event.get("status") == "observed" and event.get("run_id") == record["run_ref"]
        and event.get("attempt_id") == record["attempt_id"]
        and event.get("evidence_refs") == expected_refs.get(str(event.get("phase")))}
    return {"request": "prepared",
        "answer": "observed" if record["answer"] is not None
            and phases.get("clarification_answer", {}).get("event_id") == record["answer_event_ref"]
            and record["answer_event_ref"] is not None else "none",
        "redispatch": "observed" if record["redispatch_event_ref"] is not None
            and phases.get("clarification_redispatch", {}).get("event_id") == record["redispatch_event_ref"] else "none"}
