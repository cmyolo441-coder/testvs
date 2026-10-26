"""Typed parent decision records and their on-disk boundary parser."""
from __future__ import annotations

from typing import TypedDict, TypeGuard

from .fanout_clarification_schema import InputRequired, ClarificationError, parse_input_required, metadata_text
from .fanout_failure_diagnostics import is_string_map, is_object_list
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


class ClarificationIdentity(TypedDict):
    fanout_id: str
    unit_id: str
    run_ref: str
    base_sha: str
    contract_digest: str
    goal_attempt_id: str
    worktree_path: str


class ClarificationLineage(ClarificationIdentity):
    attempt_id: str


class ClarificationInput(ClarificationLineage):
    input_required: InputRequired
    workspace: dict[str, object]


class RenderableQuestion(InputRequired):
    fanout_id: str
    unit_id: str
    run_ref: str
    base_sha: str
    attempt_id: str
    goal_attempt_id: str
    round: int
    authority: str


class ClarificationView(TypedDict):
    schema_version: str
    renderable: RenderableQuestion | None
    queued_units: list[str]
    units: list[dict[str, str]]


class AnswerReceipt(TypedDict):
    schema_version: str
    decision_id: str
    attempt_id: str
    round: int
    answer: str
    source: str
    observed_at: str


class HistoryReceipt(TypedDict):
    decision_id: str
    attempt_id: str
    round: int
    answer: AnswerReceipt | None
    request_event_ref: str
    answer_event_ref: str | None
    redispatch_event_ref: str | None


class ClarificationRecord(ClarificationInput, HistoryReceipt):
    schema_version: str
    requested_at: str
    expires_at: str
    state: str
    privacy: str
    history: list[HistoryReceipt]


def is_answer_receipt(value: object) -> TypeGuard[AnswerReceipt]:
    return (is_string_map(value) and set(value) == set(AnswerReceipt.__annotations__)
        and value.get("schema_version") == "fanout_clarification_answer/v1"
        and value.get("source") == "root_session_cli"
        and all(isinstance(value.get(key), str) for key in
                ("schema_version", "decision_id", "attempt_id", "answer", "source", "observed_at"))
        and type(value.get("round")) is int)


def is_history_receipt(value: object) -> TypeGuard[HistoryReceipt]:
    return (is_string_map(value) and set(value) == set(HistoryReceipt.__annotations__)
        and all(isinstance(value.get(key), str) for key in ("decision_id", "attempt_id", "request_event_ref"))
        and all(value.get(key) is None or isinstance(value.get(key), str)
                for key in ("answer_event_ref", "redispatch_event_ref"))
        and type(value.get("round")) is int
        and (value.get("answer") is None or is_answer_receipt(value.get("answer"))))


def is_clarification_record(value: object) -> TypeGuard[ClarificationRecord]:
    """Check every persisted field before typed ledger operations."""
    if not is_string_map(value) or set(value) != set(ClarificationRecord.__annotations__):
        return False
    strings = (*ClarificationLineage.__annotations__, "schema_version", "requested_at", "expires_at", "state", "privacy")
    if not all(isinstance(value.get(key), str) for key in strings):
        return False
    if value["schema_version"] != "fanout_clarification_request/v1" or value["state"] not in (
            "prepared_not_answered", "answered", "cancelled", "exhausted", "redispatch_reserved", "redispatched"):
        return False
    history = value.get("history")
    if not is_object_list(history) or len(history) > 2 or not all(is_history_receipt(row) for row in history):
        return False
    receipt = {key: value.get(key) for key in HistoryReceipt.__annotations__}
    return is_history_receipt(receipt) and is_string_map(value.get("workspace"))


def parse_clarification_record(value: object) -> ClarificationRecord:
    if not is_clarification_record(value):
        raise ClarificationError("clarification_record_invalid")
    request = parse_input_required(value["input_required"], value["unit_id"])
    if request["decision_id"] != value["decision_id"] or not 1 <= value["round"] <= 3:
        raise ClarificationError("clarification_record_identity_invalid")
    for timestamp in (value["requested_at"], value["expires_at"]):
        if strict_fromisoformat(timestamp).tzinfo is None:
            raise ClarificationError("clarification_timestamp_invalid")
    answer = value["answer"]
    if answer is not None:
        _ = metadata_text(answer["answer"], "answer", 300)
        if (answer["decision_id"], answer["attempt_id"], answer["round"]) != (
                value["decision_id"], value["attempt_id"], value["round"]):
            raise ClarificationError("clarification_answer_identity_invalid")
    return value


def history_receipt(record: ClarificationRecord) -> HistoryReceipt:
    return {"decision_id": record["decision_id"], "attempt_id": record["attempt_id"],
        "round": record["round"], "answer": record["answer"],
        "request_event_ref": record["request_event_ref"], "answer_event_ref": record["answer_event_ref"],
        "redispatch_event_ref": record["redispatch_event_ref"]}
