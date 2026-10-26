"""Clarification binding and explicit same-worktree redispatch admission."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

from ..system.local_store import atomic_write_json, file_lock
from ..system.paths import OmhPaths
from ..workflows.observation_journal import read_observation_events
from .fanout_artifacts import fanout_contract_digest
from .fanout_clarification import (
    clarification_path, clarification_state, read_clarification, prepare_clarification,
    clarification_event, clarification_evidence,
)
from .fanout_clarification_schema import ClarificationError, parse_input_required
from .fanout_executor_sessions import observe_session_workspace
from .fanout_clarification_records import ClarificationRecord, ClarificationLineage
from .fanout_failure_diagnostics import is_object_list, is_string_map

if TYPE_CHECKING:
    from .fanout_confinement import FanoutFilesystemConfinement


class ResumeContext(TypedDict):
    journal: Mapping[str, object] | None
    base_sha: str
    goal_attempt_id: str
    only_units: list[str]


def bind_clarification(paths: OmhPaths, result: Mapping[str, object], lineage: ClarificationLineage,
                       *, confinement: FanoutFilesystemConfinement | None = None) -> ClarificationRecord:
    """Bind a unit's decision request to its worktree, observed inside `confinement` (#1999)."""
    request = parse_input_required(result["input_required"], lineage["unit_id"])
    # A child can describe its own decision, never select sibling execution.
    if request["affected_unit_ids"] != [lineage["unit_id"]]:
        raise ClarificationError("affected_unit_ids outside reporting unit authority")
    workspace = observe_session_workspace(lineage["worktree_path"], confinement=confinement)
    if workspace is None:
        raise ClarificationError("clarification_workspace_unavailable")
    return prepare_clarification(paths, {**lineage, "input_required": request,
        "workspace": asdict(workspace)})


def clarification_resume_journal(paths: OmhPaths, contract: Mapping[str, object], context: ResumeContext) -> dict[str, object] | None:
    """Only authoritative records can make an input-required journal row eligible."""
    journal = context["journal"]
    units = contract.get("units")
    if not is_object_list(units) or not all(is_string_map(unit) for unit in units):
        raise ClarificationError("clarification_contract_units_invalid")
    records = {str(unit["unit_id"]): read_clarification(clarification_path(paths, str(contract["fanout_id"]), str(unit["unit_id"])))
               for unit in units if is_string_map(unit)}
    if journal is None:
        if any(row and clarification_state(row) != "redispatched" for row in records.values()):
            raise ClarificationError("clarification_requires_explicit_resume_journal")
        return None
    originals = journal.get("units")
    if not is_object_list(originals) or not all(is_string_map(row) for row in originals):
        raise ClarificationError("clarification_journal_units_invalid")
    events = read_observation_events(paths)
    rows: list[dict[str, object]] = []
    for original in originals:
        if not is_string_map(original):
            raise ClarificationError("clarification_journal_row_invalid")
        row = dict(original)
        record = records.get(str(row["unit_id"]))
        # A pre-spawn refusal must not erase the decision lineage or its explicit-unit gate.
        if record is not None and row["terminal_state"] == "not_attempted" and not row.get("attempt_id"):
            row.update(terminal_state="input_required", attempt_id=record["attempt_id"])
        if row["terminal_state"] == "input_required":
            valid = record is not None and all(record.get(key) == expected for key, expected in (
                ("fanout_id", contract["fanout_id"]), ("run_ref", row["run_ref"]),
                ("attempt_id", row.get("attempt_id")), ("base_sha", context["base_sha"]),
                ("contract_digest", fanout_contract_digest(contract)),
                ("goal_attempt_id", context["goal_attempt_id"])))
            row["clarification_state"] = clarification_state(record) if valid and record is not None else "stale"
            evidence = clarification_evidence(record, events)
            if row["clarification_state"] == "answered" and (evidence is None or evidence["answer"] != "observed"):
                row["clarification_state"] = "stale"
            if row["clarification_state"] == "answered" and context["only_units"] != [row["unit_id"]]:
                row["clarification_state"] = "explicit_unit_required"
        rows.append(row)
    return {**journal, "units": rows}


def claim_answered_worktree(paths: OmhPaths, fanout_id: str, identity: Mapping[str, str],
                            *, confinement: FanoutFilesystemConfinement | None = None) -> ClarificationRecord:
    """Reserve an answer once; unchanged dirty work is preserved, never rebuilt.

    The worktree is the one the asking attempt wrote, observed before this
    attempt's own fence exists, so `confinement` is a fence for that probe (#1999).
    """
    path = clarification_path(paths, fanout_id, identity["unit_id"])
    with file_lock(path, private=True) as lock:
        if not lock["enforced"]:
            raise ClarificationError("clarification_lock_unavailable")
        record = read_clarification(path)
        if record is None:
            raise ClarificationError("clarification_record_missing")
        if clarification_state(record) != "answered":
            raise ClarificationError("clarification_not_answered")
        evidence = clarification_evidence(record, read_observation_events(paths, run_id=record["run_ref"]))
        if evidence is None or evidence["answer"] != "observed":
            raise ClarificationError("clarification_answer_unobserved")
        if any(record.get(key) != value for key, value in identity.items()):
            raise ClarificationError("clarification_lineage_changed")
        snapshot = observe_session_workspace(record["worktree_path"], confinement=confinement)
        if snapshot is None or asdict(snapshot) != record["workspace"]:
            raise ClarificationError("clarification_workspace_changed")
        record["state"] = "redispatch_reserved"
        atomic_write_json(path, dict(record), private=True)
        return record


def record_answer_redispatch(paths: OmhPaths, record: ClarificationRecord) -> None:
    path = clarification_path(paths, record["fanout_id"], record["unit_id"])
    with file_lock(path, private=True) as lock:
        if not lock["enforced"]:
            raise ClarificationError("clarification_lock_unavailable")
        current = read_clarification(path)
        if current is None or current["state"] != "redispatch_reserved" or current["attempt_id"] != record["attempt_id"]:
            raise ClarificationError("clarification_reservation_changed")
        current["redispatch_event_ref"] = clarification_event(paths, record, "redispatch")
        current["state"] = "redispatched"
        atomic_write_json(path, dict(current), private=True)


def parent_decision_prompt(record: ClarificationRecord) -> str:
    answer = record["answer"]
    if answer is None:
        raise ClarificationError("clarification_answer_missing")
    return "\n[Parent decision]\n" + json.dumps({
        **answer, "authority": "answer_only_original_scope_and_approval_boundaries_unchanged",
        "goal_attempt_id": record["goal_attempt_id"]}, sort_keys=True)


def reported_producer_head(_result: Mapping[str, object], worktree: Path,
                           *, confinement: FanoutFilesystemConfinement | None = None) -> str | None:
    """A decision request may preserve uncommitted work; success still requires a clean HEAD."""
    workspace = observe_session_workspace(str(worktree), confinement=confinement)
    return workspace.head if workspace is not None else None
