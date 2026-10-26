"""Read complete fanout evidence into metadata-only critical-path event inputs."""

from __future__ import annotations

import json

from ..coding.fanout_artifacts import fanout_run_journal_path, read_fanout_contract
from ..coding.fanout_contracts import FANOUT_CONTRACT_SCHEMA_VERSION
from ..coding.fanout_journal import FanoutJournalError, read_fanout_run_journal
from ..system.paths import OmhPaths
from ..workflows.observation_journal import read_observation_events_result, validate_observation_event
from .critical_path_health import project_critical_path_health
from .critical_path_health_direct_events import read_direct_health_events
from .critical_path_health_models import (
    CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
    CriticalPathEvidenceGap,
    CriticalPathHealthEvent,
    CriticalPathHealthProjection,
)
from .critical_path_health_source_models import CriticalPathHealthSourceResult
from .critical_path_health_source_parsing import ObjectMapping, object_mapping, object_mappings, string_values, timestamp_ms


_DISPATCH_EVENT = "executor_dispatch_observed"
_RESULT_EVENT = "executor_result_observed"
_CLEANUP_EVENT = "worktree_cleanup"
_TERMINAL_STATES = {
    "succeeded": "succeeded",
    "failed": "failed",
    "declined": "failed",
    "skipped_by_dependency": "skipped",
}


def project_fanout_critical_path_health(paths: OmhPaths, fanout_id: str) -> CriticalPathHealthSourceResult:
    """Project one frozen fanout only when its complete journals agree.

    The observation journal is read without a limit. Dispatch is the only
    timestamped process-start evidence available, so it supplies both queued
    and started timestamps; verification and review receipts are deliberately
    not lifecycle substitutes.
    """
    gaps: set[tuple[str, str]] = set()
    contract = _read_contract(paths, fanout_id, gaps)
    if contract is None:
        return _result(fanout_id, (), gaps)
    units, order = _contract_units(contract, fanout_id, gaps)
    journal = _read_journal(paths, fanout_id, gaps)
    if journal is None or not units:
        return _result(fanout_id, (), gaps)
    revision = str(journal.get("base_sha", ""))
    if not revision:
        gaps.add(("", "missing_revision"))
    rows = _journal_rows(journal, units, order, gaps)
    direct_events = read_direct_health_events(paths, fanout_id, gaps)
    if direct_events is not None:
        return _result(fanout_id, direct_events, gaps)
    observations = _observations(paths, gaps)
    events: list[CriticalPathHealthEvent] = []
    for unit_id, unit in units.items():
        row = rows.get(unit_id)
        if row is None or not revision:
            continue
        events.extend(_unit_events(unit_id, unit, row, observations, revision, gaps))
    return _result(fanout_id, tuple(events), gaps)


def _read_contract(paths: OmhPaths, fanout_id: str, gaps: set[tuple[str, str]]) -> ObjectMapping | None:
    try:
        raw_contract = read_fanout_contract(paths, fanout_id)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        gaps.add(("", "fanout_contract_unavailable"))
        return None
    contract = object_mapping(raw_contract)
    if contract is None:
        gaps.add(("", "fanout_contract_invalid"))
        return None
    if contract.get("schema_version") != FANOUT_CONTRACT_SCHEMA_VERSION:
        gaps.add(("", "fanout_contract_schema_unsupported"))
    if str(contract.get("fanout_id", "")) != fanout_id:
        gaps.add(("", "fanout_contract_mismatch"))
    return contract


def _contract_units(contract: ObjectMapping, fanout_id: str, gaps: set[tuple[str, str]]) -> tuple[dict[str, ObjectMapping], list[str]]:
    raw_units = object_mappings(contract.get("units"))
    if raw_units is None:
        gaps.add(("", "fanout_contract_invalid"))
        return {}, []
    units = {str(unit.get("unit_id", "")): unit for unit in raw_units}
    if not units or len(units) != len(raw_units) or "" in units:
        gaps.add(("", "fanout_contract_invalid"))
        return {}, []
    if any(not str(unit.get("run_ref", "")) for unit in units.values()):
        gaps.add(("", "fanout_contract_invalid"))
    if len({str(unit.get("run_ref", "")) for unit in units.values()}) != len(units):
        gaps.add(("", "fanout_contract_run_ref_mismatch"))
    merge_plan = object_mapping(contract.get("merge_plan"))
    order = string_values(merge_plan.get("merge_order")) if merge_plan is not None else None
    normalized_order = order if order is not None else []
    if len(normalized_order) != len(units) or set(normalized_order) != set(units):
        gaps.add(("", "fanout_contract_order_mismatch"))
    if str(contract.get("fanout_id", "")) != fanout_id:
        gaps.add(("", "fanout_contract_mismatch"))
    return units, normalized_order


def _read_journal(paths: OmhPaths, fanout_id: str, gaps: set[tuple[str, str]]) -> ObjectMapping | None:
    try:
        raw_journal = read_fanout_run_journal(fanout_run_journal_path(paths, fanout_id), expected_fanout_id=fanout_id)
    except (FanoutJournalError, OSError, ValueError) as exc:
        code = exc.reason_code if isinstance(exc, FanoutJournalError) else "unavailable"
        gaps.add(("", f"fanout_journal_{code}"))
        return None
    journal = object_mapping(raw_journal)
    if journal is None:
        gaps.add(("", "fanout_journal_invalid"))
    return journal


def _journal_rows(journal: ObjectMapping, units: dict[str, ObjectMapping], expected_order: list[str], gaps: set[tuple[str, str]]) -> dict[str, ObjectMapping]:
    raw_rows = object_mappings(journal.get("units"))
    if raw_rows is None:
        gaps.add(("", "fanout_journal_invalid"))
        return {}
    rows = {str(row.get("unit_id", "")): row for row in raw_rows}
    if set(rows) != set(units) or len(rows) != len(raw_rows):
        gaps.add(("", "fanout_journal_unit_mismatch"))
    order = journal.get("merge_order")
    if not isinstance(order, list) or [str(item) for item in order] != expected_order:
        gaps.add(("", "fanout_journal_order_mismatch"))
    for unit_id, unit in units.items():
        if unit_id in rows and str(rows[unit_id].get("run_ref", "")) != str(unit.get("run_ref", "")):
            gaps.add((unit_id, "fanout_journal_run_ref_mismatch"))
    return rows


def _observations(paths: OmhPaths, gaps: set[tuple[str, str]]) -> dict[str, list[ObjectMapping]]:
    try:
        events, errors = read_observation_events_result(paths)
    except (OSError, ValueError, KeyError):
        gaps.add(("", "observation_journal_unavailable"))
        return {}
    if errors:
        gaps.add(("", "observation_journal_invalid"))
    by_run: dict[str, list[ObjectMapping]] = {}
    for raw_event in events:
        event = object_mapping(raw_event)
        if event is None or validate_observation_event(dict(event)):
            gaps.add((str(event.get("run_id", "")) if event is not None else "", "observation_journal_invalid"))
            continue
        by_run.setdefault(str(event.get("run_id", "")), []).append(event)
    return by_run


def _unit_events(
    unit_id: str,
    unit: ObjectMapping,
    row: ObjectMapping,
    observations: dict[str, list[ObjectMapping]],
    revision: str,
    gaps: set[tuple[str, str]],
) -> list[CriticalPathHealthEvent]:
    run_ref = str(unit.get("run_ref", ""))
    observed = observations.get(run_ref, [])
    if any(str(event.get("event", "")) == _CLEANUP_EVENT for event in observed):
        gaps.add((unit_id, "cleanup_evidence_unavailable"))
    dispatches = [event for event in observed if str(event.get("event", "")) == _DISPATCH_EVENT]
    results = [event for event in observed if str(event.get("event", "")) == _RESULT_EVENT]
    if str(row.get("status", "")) == "already_completed":
        gaps.add((unit_id, "reuse_evidence_unavailable"))
    if not dispatches:
        gaps.add((unit_id, "missing_worker_dispatch"))
    if not results:
        gaps.add((unit_id, "missing_worker_result"))
    if dispatches:
        gaps.add((unit_id, "queue_evidence_unavailable"))
    if len(dispatches) != len(results):
        gaps.add((unit_id, "retry_evidence_incomplete"))
        return []
    terminal = _TERMINAL_STATES.get(str(row.get("terminal_state", "")))
    if terminal is None:
        gaps.add((unit_id, "terminal_evidence_unavailable"))
        return []
    resource = _resource_class(unit, unit_id, gaps)
    if not resource:
        return []
    dependencies = tuple(string_values(unit.get("depends_on")) or [])
    events: list[CriticalPathHealthEvent] = []
    for retry, (dispatch, result) in enumerate(zip(dispatches, results)):
        if not _matches_unit(dispatch, unit_id) or not _matches_unit(result, unit_id):
            gaps.add((unit_id, "worker_ref_mismatch"))
            return []
        queued_at = _at_ms(dispatch, unit_id, gaps)
        finished_at = _at_ms(result, unit_id, gaps)
        status = _result_status(result)
        if queued_at is None or finished_at is None or status is None or queued_at > finished_at:
            gaps.add((unit_id, "lifecycle_timestamp_invalid"))
            return []
        if retry + 1 == len(results) and status != terminal:
            gaps.add((unit_id, "terminal_mismatch"))
            return []
        events.extend(
            (
                CriticalPathHealthEvent(unit_id, "started", queued_at, revision, "fanout_dispatch", "frozen_contract", "omh", dependencies, resource, "execution", retry),
                CriticalPathHealthEvent(unit_id, "finished", finished_at, revision, "fanout_dispatch", "frozen_contract", "omh", dependencies, resource, "execution", retry, status),
            )
        )
    return events


def _resource_class(unit: ObjectMapping, unit_id: str, gaps: set[tuple[str, str]]) -> str:
    handoff = object_mapping(unit.get("handoff"))
    value = handoff.get("executor_target", "") if handoff is not None else unit.get("owner", "")
    resource = str(value or "")
    if not resource:
        gaps.add((unit_id, "resource_evidence_unavailable"))
    return resource


def _matches_unit(event: ObjectMapping, unit_id: str) -> bool:
    worker_ref = str(event.get("worker_ref", ""))
    return not worker_ref or worker_ref == unit_id


def _at_ms(event: ObjectMapping, unit_id: str, gaps: set[tuple[str, str]]) -> int | None:
    observed_at = timestamp_ms(event.get("observed_at"))
    if observed_at is None:
        gaps.add((unit_id, "observation_timestamp_invalid"))
    return observed_at


def _result_status(event: ObjectMapping) -> str | None:
    return {"observed": "succeeded", "failed": "failed", "cancelled": "cancelled"}.get(str(event.get("status", "")))


def _result(
    fanout_id: str, events: tuple[CriticalPathHealthEvent, ...], source_gaps: set[tuple[str, str]]
) -> CriticalPathHealthSourceResult:
    projected = project_critical_path_health(events)
    gaps = tuple(sorted(source_gaps | {(gap.task_id, gap.code) for gap in projected.evidence_gaps}))
    if source_gaps:
        projected = CriticalPathHealthProjection(
            CRITICAL_PATH_HEALTH_SCHEMA_VERSION,
            projected.executor,
            projected.model,
            projected.environment,
            projected.task_revisions,
            None,
            (),
            (),
            tuple(CriticalPathEvidenceGap(task_id, code) for task_id, code in gaps),
        )
    return CriticalPathHealthSourceResult(fanout_id, events, projected, gaps)
