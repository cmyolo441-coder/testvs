"""The incident layer of ``agent-debug``: capture, competing hypotheses, contained recovery, export.

``agent_debug_report/v1`` lists what one session recorded going wrong, each
row cited. This module builds the three artifacts the ``agent-debug`` skill
declares on top of that report, and the one way to share them:

- ``agent_failure_capture/v1`` -- what was observed. It binds the case to one
  identity (session, source, selection, turn range, source snapshot, report
  digest), lists the report's finding ids, everything that was *unavailable*
  (an unchecked kind, an oversized row, a reached row limit, host state OMH
  cannot read), and the receipts it admitted or refused.
- ``agent_failure_pattern_hypothesis/v1`` -- what might explain it. At least
  two competing hypotheses from a closed pattern table, each with typed
  evidence for and against, the evidence it could not see, a confidence, and
  the observation that would decide it. A hypothesis is ruled out only by an
  observed absence in a complete reading; one hypothesis -- that the cause
  lies outside the recorded evidence -- can never be ruled out from records,
  so an unresolved case stays unresolved.
- ``contained_recovery_action/v1`` -- the smallest reversible next step for
  the leading hypothesis, as a proposal: it requires approval, is never
  executed here, and lists what diagnosis did not do.

Receipts. A ``fanout_dispatch_summary/v1`` unit is admitted as evidence only
when it binds all five identities -- the report's session (its
``origin_session_id``), a run (``run_ref``), a unit (``unit_id``), a
configuration (the summary's ``contract_digest``), and freshness (an
``observed_at`` no earlier than the session's start; a session with no
recorded start binds no freshness). A unit that cannot bind one is refused
with the identity it lacked; two units claiming one run and
unit under different configurations are both refused.

Export is a separate action. ``build_agent_debug_export`` refuses a payload
with any key outside the artifacts' closed shapes, re-validates the
artifacts, re-checks the source against the report (refusing a missing,
foreign, stale, or mismatched reference, or a source that moved under the
report's snapshot), drops the absolute source path and the typed session
selector, and runs a deterministic leak scan; ``write_agent_debug_export``
writes the package only to a new file outside the source, never through an
existing name or link. Nothing here files an issue,
posts a comment, resets an executor, or touches the session.
"""

from __future__ import annotations

from datetime import timezone
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from ..plugin_bundle.omh.run_records import FANOUT_DISPATCH_SCHEMA_VERSION
from ..system.metadata_safety import is_raw_pii_shaped, is_secret_value_shaped
from .agent_debug_report import (
    BACKGROUND_WITHOUT_NOTIFY,
    CITATION_KEYS,
    COMPACTION_BOUNDARY,
    FINDING_KINDS,
    IDENTICAL_RETRY_AFTER_ERROR,
    TOOL_ERROR,
    agent_debug_reference_errors,
    agent_debug_report_errors,
)
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat


AGENT_FAILURE_CAPTURE_SCHEMA_VERSION = "agent_failure_capture/v1"
AGENT_FAILURE_PATTERN_HYPOTHESIS_SCHEMA_VERSION = "agent_failure_pattern_hypothesis/v1"
CONTAINED_RECOVERY_ACTION_SCHEMA_VERSION = "contained_recovery_action/v1"
AGENT_DEBUG_EXPORT_SCHEMA_VERSION = "agent_debug_export/v1"

OBSERVABLES: tuple[str, ...] = (
    "looping",
    "repeated_work",
    "goal_drift",
    "context_loss",
    "tool_stall",
    "unexpected_cost",
    "unspecified",
)

TOOL_ERROR_RETRY_LOOP = "tool_error_retry_loop"
CONTEXT_LOSS_AFTER_COMPACTION = "context_loss_after_compaction"
UNWATCHED_BACKGROUND_WORK = "unwatched_background_work"
OUTSIDE_RECORDED_EVIDENCE = "outside_recorded_evidence"
# pattern -> (the kind whose absence rules it out, the observation that decides it)
PATTERNS: dict[str, tuple[str | None, str]] = {
    TOOL_ERROR_RETRY_LOOP: (
        IDENTICAL_RETRY_AFTER_ERROR,
        "a retry with changed arguments that succeeds, or the same arguments failing again after a change elsewhere",
    ),
    CONTEXT_LOSS_AFTER_COMPACTION: (
        COMPACTION_BOUNDARY,
        "the same failure in a turn range with no compaction before it",
    ),
    UNWATCHED_BACKGROUND_WORK: (
        BACKGROUND_WITHOUT_NOTIFY,
        "the run waiting on a process whose completion it was notified of",
    ),
    OUTSIDE_RECORDED_EVIDENCE: (
        None,
        "a replay of the cited turns, or host evidence (provider, model, or runtime state) the record does not hold",
    ),
}
_OBSERVABLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "looping": (TOOL_ERROR_RETRY_LOOP, CONTEXT_LOSS_AFTER_COMPACTION, OUTSIDE_RECORDED_EVIDENCE),
    "repeated_work": (TOOL_ERROR_RETRY_LOOP, CONTEXT_LOSS_AFTER_COMPACTION, OUTSIDE_RECORDED_EVIDENCE),
    "goal_drift": (CONTEXT_LOSS_AFTER_COMPACTION, TOOL_ERROR_RETRY_LOOP, OUTSIDE_RECORDED_EVIDENCE),
    "context_loss": (CONTEXT_LOSS_AFTER_COMPACTION, OUTSIDE_RECORDED_EVIDENCE),
    "tool_stall": (UNWATCHED_BACKGROUND_WORK, TOOL_ERROR_RETRY_LOOP, OUTSIDE_RECORDED_EVIDENCE),
    "unexpected_cost": (
        TOOL_ERROR_RETRY_LOOP,
        CONTEXT_LOSS_AFTER_COMPACTION,
        UNWATCHED_BACKGROUND_WORK,
        OUTSIDE_RECORDED_EVIDENCE,
    ),
    "unspecified": tuple(PATTERNS),
}
STATUSES: tuple[str, ...] = ("supported", "unresolved", "ruled_out")
CONFIDENCES: tuple[str, ...] = ("none", "low", "medium", "high")
RESOLUTIONS: tuple[str, ...] = ("unresolved", "single_candidate")

# leading pattern (or none) -> the smallest reversible next step
RECOVERY_ACTIONS: dict[str, str] = {
    TOOL_ERROR_RETRY_LOOP: "change_arguments_before_next_retry",
    CONTEXT_LOSS_AFTER_COMPACTION: "restate_goal_after_compaction",
    UNWATCHED_BACKGROUND_WORK: "poll_or_notify_background_process",
    OUTSIDE_RECORDED_EVIDENCE: "collect_discriminating_evidence",
}
RECOVERY_ACTION_STEPS: dict[str, str] = {
    "change_arguments_before_next_retry": (
        "Pause the run before its next call of the cited tool, and change the arguments or the approach the "
        "cited error points at before it retries."
    ),
    "restate_goal_after_compaction": (
        "Restate the goal and the open plan steps in the next turn after the cited compaction, then let the run continue."
    ),
    "poll_or_notify_background_process": (
        "Poll the cited background process, or start it again with completion notification, before the run "
        "waits on its result."
    ),
    "collect_discriminating_evidence": (
        "Change nothing in the run yet: replay the cited turn range or supply the host evidence the decisive "
        "observation needs."
    ),
}
# Everything diagnosis never does. Each is its own approved action elsewhere.
NOT_PERFORMED: tuple[str, ...] = (
    "recovery",
    "executor_reset",
    "session_mutation",
    "runtime_repair",
    "export",
    "archive",
    "github_issue",
    "comment",
)

RECEIPT_SCHEMA_FANOUT_DISPATCH = FANOUT_DISPATCH_SCHEMA_VERSION
RECEIPT_IDENTITIES: tuple[str, ...] = ("session", "run", "unit", "configuration", "freshness")
RECEIPT_REJECTIONS: tuple[str, ...] = (
    "unreadable",
    "oversized",
    "ineligible_schema",
    "session_unbound",
    "foreign_session",
    "run_unbound",
    "unit_unbound",
    "configuration_unbound",
    "freshness_unbound",
    "stale",
    "identity_conflict",
)
MAX_RECEIPT_BYTES = 256 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

HOST_RUNTIME_UNAVAILABLE = {
    "evidence": "host_runtime",
    "reason": "live Hermes process, provider, and model state is not in the record; OMH reads persisted records only",
}

INCIDENT_CLAIM_BOUNDARY = (
    "An agent-debug incident is built from one cited report and the receipts it admitted. Hypotheses are "
    "inferred, not observed; a supported hypothesis is a correlation the record shows, not a proven cause, "
    "and the recovery action is a proposal that requires approval and was not executed. None of it is "
    "execution, review, CI, or merge evidence, or proof that a future run is fixed."
)
EXPORT_CLAIM_BOUNDARY = (
    "An agent-debug export is the redacted incident package a person reviewed and chose to write. It carries "
    "ids, digests, counts, timestamps, the source and receipt file names, closed-vocabulary values, and OMH's "
    "own fixed wording only; it was not uploaded, filed, or shared by OMH."
)


class AgentDebugIncidentError(ValueError):
    """An artifact failed validation, a receipt could not be read, or an export was refused."""


# --- receipts --------------------------------------------------------------


def read_receipt(path: str | Path) -> tuple[str, Any]:
    """``(label, payload)`` for a supplied receipt file; the label is the file name, never the path."""
    receipt_path = Path(path)
    label = receipt_path.name
    try:
        size = receipt_path.stat().st_size
        if size > MAX_RECEIPT_BYTES:
            return label, _Refused("oversized")
        with receipt_path.open("rb") as handle:
            raw = handle.read(MAX_RECEIPT_BYTES + 1)
    except OSError:
        return label, _Refused("unreadable")
    if len(raw) > MAX_RECEIPT_BYTES:
        return label, _Refused("oversized")
    try:
        return label, json.loads(raw.decode("utf-8"))
    except ValueError:
        return label, _Refused("unreadable")


class _Refused:
    def __init__(self, reason: str) -> None:
        self.reason = reason


def bind_receipts(report: Mapping[str, Any], receipts: Sequence[tuple[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Admit each receipt unit that binds all five identities to ``report``; refuse the rest by name."""
    session = report.get("session") or {}
    session_id = str(session.get("id") or "")
    started_at = session.get("started_at")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for label, payload in receipts:
        if isinstance(payload, _Refused):
            rejected.append({"receipt": label, "unit_id": None, "reason": payload.reason})
            continue
        if not isinstance(payload, Mapping) or payload.get("schema_version") != RECEIPT_SCHEMA_FANOUT_DISPATCH:
            rejected.append({"receipt": label, "unit_id": None, "reason": "ineligible_schema"})
            continue
        configuration = payload.get("contract_digest")
        observed_at = _epoch(payload.get("observed_at"))
        fanout_id = _safe_id(payload.get("fanout_id"))
        for unit in payload.get("units") or ():
            if not isinstance(unit, Mapping):
                continue
            unit_id = _safe_id(unit.get("unit_id"))
            origin = unit.get("origin_session_id") or payload.get("origin_session_id")
            reason = None
            if not isinstance(origin, str) or not origin:
                reason = "session_unbound"
            elif origin != session_id:
                reason = "foreign_session"
            elif _safe_id(unit.get("run_ref")) is None:
                reason = "run_unbound"
            elif unit_id is None or fanout_id is None:
                reason = "unit_unbound"
            elif not isinstance(configuration, str) or not _SHA256.match(configuration):
                reason = "configuration_unbound"
            elif observed_at is None or not isinstance(started_at, (int, float)) or isinstance(started_at, bool):
                # Freshness binds the receipt to the session's start; a
                # session with no recorded start leaves nothing to bind to.
                reason = "freshness_unbound"
            elif observed_at < started_at:
                reason = "stale"
            if reason is not None:
                rejected.append({"receipt": label, "unit_id": unit_id, "reason": reason})
                continue
            exit_code = unit.get("exit_code")
            accepted.append(
                {
                    "ref": f"receipt:{RECEIPT_SCHEMA_FANOUT_DISPATCH}#{fanout_id}/{unit_id}",
                    "receipt": label,
                    "schema_version": RECEIPT_SCHEMA_FANOUT_DISPATCH,
                    "session_id": session_id,
                    "run_id": str(unit["run_ref"]),
                    "unit_id": unit_id,
                    "configuration_digest": configuration,
                    "observed_at": observed_at,
                    "status": _safe_id(unit.get("status")),
                    "exit_code": exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None,
                }
            )
    configurations: dict[tuple[str, str], set[str]] = {}
    for item in accepted:
        configurations.setdefault((item["run_id"], item["unit_id"]), set()).add(item["configuration_digest"])
    conflicted = {key for key, digests in configurations.items() if len(digests) > 1}
    kept: list[dict[str, Any]] = []
    seen_refs: set[str] = set()
    for item in accepted:
        if (item["run_id"], item["unit_id"]) in conflicted:
            rejected.append({"receipt": item["receipt"], "unit_id": item["unit_id"], "reason": "identity_conflict"})
        elif item["ref"] not in seen_refs:
            seen_refs.add(item["ref"])
            kept.append(item)
    return {"accepted": kept, "rejected": rejected}


# --- capture ---------------------------------------------------------------


def build_agent_failure_capture(
    report: Mapping[str, Any],
    *,
    observable: str = "unspecified",
    receipts: Sequence[tuple[str, Any]] = (),
) -> dict[str, Any]:
    """``agent_failure_capture/v1`` for a valid report: identity, observed ids, unavailable evidence, receipts."""
    errors = agent_debug_report_errors(report)
    if errors:
        raise AgentDebugIncidentError("the report is not a valid agent_debug_report/v1: " + "; ".join(errors))
    if observable not in OBSERVABLES:
        raise AgentDebugIncidentError(f"observable must be one of {', '.join(OBSERVABLES)}")
    capture = {
        "schema_version": AGENT_FAILURE_CAPTURE_SCHEMA_VERSION,
        "identity": _capture_identity(report),
        "observable": observable,
        "observed_findings": [
            {"finding_id": item["finding_id"], "kind": item["kind"]} for item in report.get("findings") or ()
        ],
        "finding_counts": dict(report.get("finding_counts") or {}),
        "checked_kinds": list(report.get("checked_kinds") or ()),
        "reading_complete": bool((report.get("budget") or {}).get("complete", False)),
        "unavailable": _capture_unavailable(report),
        "receipts": bind_receipts(report, receipts),
        "observed": True,
        "claim_boundary": INCIDENT_CLAIM_BOUNDARY,
    }
    errors = agent_failure_capture_errors(capture, report)
    if errors:
        raise AgentDebugIncidentError("agent_failure_capture/v1 failed validation: " + "; ".join(errors))
    return capture


def _capture_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    source = report.get("source") or {}
    return {
        "session_id": str((report.get("session") or {}).get("id")),
        "source_kind": source.get("kind"),
        "source_label": source.get("label"),
        "selection": source.get("selection"),
        "turns": dict(report.get("turns") or {}),
        "snapshot": dict(source.get("snapshot") or {}),
        "report_sha256": report_digest(report),
    }


def _capture_unavailable(report: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every piece of evidence the report could not see: its unchecked kinds, unread rows, host state."""
    budget = report.get("budget") or {}
    unavailable = [{"evidence": item["kind"], "reason": item["reason"]} for item in report.get("unavailable") or ()]
    if budget.get("oversized_rows"):
        unavailable.append(
            {"evidence": "oversized_rows", "reason": f"{budget['oversized_rows']} rows over the byte budget were not read"}
        )
    if budget.get("row_limit_reached"):
        unavailable.append({"evidence": "rows_after_limit", "reason": "rows after the row budget were not read"})
    unavailable.append(dict(HOST_RUNTIME_UNAVAILABLE))
    return unavailable


def report_digest(report: Mapping[str, Any]) -> str:
    """sha256 over the report's findings and identity, independent of where it was read from."""
    body = {
        "session": report.get("session"),
        "turns": report.get("turns"),
        "findings": report.get("findings"),
        "unavailable": report.get("unavailable"),
        "budget": report.get("budget"),
    }
    return _digest(body)


def agent_failure_capture_errors(capture: Mapping[str, Any], report: Mapping[str, Any]) -> list[str]:
    """Every reason ``capture`` is not the capture of ``report``.

    What the capture says was checked and what it says was unavailable decide
    whether an absence can rule a hypothesis out, so both are compared with
    the report, not trusted: ``checked_kinds`` and ``finding_counts`` must
    equal the report's, and ``unavailable`` must be exactly what the report
    could not see (its unchecked kinds, unread rows, and host state). The
    report is validated first, which holds its ``checked_kinds`` to every kind
    it does not list as unavailable.
    """
    errors = [f"report: {problem}" for problem in agent_debug_report_errors(report)]
    if capture.get("schema_version") != AGENT_FAILURE_CAPTURE_SCHEMA_VERSION:
        errors.append(f"schema_version must be {AGENT_FAILURE_CAPTURE_SCHEMA_VERSION}")
    identity = capture.get("identity") or {}
    session_id = str((report.get("session") or {}).get("id"))
    if identity.get("session_id") != session_id:
        errors.append("identity.session_id names another session than the report")
    if identity.get("report_sha256") != report_digest(report):
        errors.append("identity.report_sha256 does not match the report: the capture is stale or for another report")
    elif identity != _capture_identity(report):
        errors.append("identity must be the report's source, selection, turn range, and snapshot")
    if capture.get("checked_kinds") != list(report.get("checked_kinds") or ()):
        errors.append("checked_kinds must equal the report's checked_kinds")
    if capture.get("unavailable") != _capture_unavailable(report):
        errors.append("unavailable must list exactly the evidence the report could not see")
    if capture.get("finding_counts") != dict(report.get("finding_counts") or {}):
        errors.append("finding_counts must equal the report's finding_counts")
    if capture.get("observed") is not True or capture.get("claim_boundary") != INCIDENT_CLAIM_BOUNDARY:
        errors.append("a capture is observed and carries the incident claim boundary")
    if capture.get("observable") not in OBSERVABLES:
        errors.append("observable is not in the closed vocabulary")
    report_ids = {item.get("finding_id"): item.get("kind") for item in report.get("findings") or ()}
    observed = capture.get("observed_findings") or []
    if {item.get("finding_id"): item.get("kind") for item in observed} != report_ids or len(observed) != len(report_ids):
        errors.append("observed_findings must list exactly the report's findings")
    if capture.get("reading_complete") is not bool((report.get("budget") or {}).get("complete", False)):
        errors.append("reading_complete must equal the report's budget.complete")
    receipts = capture.get("receipts") or {}
    for item in receipts.get("accepted") or ():
        missing = [
            name
            for name, key in (
                ("session", "session_id"),
                ("run", "run_id"),
                ("unit", "unit_id"),
                ("configuration", "configuration_digest"),
                ("freshness", "observed_at"),
            )
            if item.get(key) in (None, "")
        ]
        if missing:
            errors.append(f"receipt {item.get('ref')} was admitted without binding {', '.join(missing)}")
        elif item.get("session_id") != session_id:
            errors.append(f"receipt {item.get('ref')} binds another session")
        elif not _SHA256.match(str(item.get("configuration_digest"))):
            errors.append(f"receipt {item.get('ref')} binds no configuration digest")
    for item in receipts.get("rejected") or ():
        if item.get("reason") not in RECEIPT_REJECTIONS:
            errors.append(f"receipt {item.get('receipt')} was refused for an unknown reason")
    return errors


# --- hypotheses ------------------------------------------------------------


def build_agent_failure_pattern_hypothesis(capture: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """``agent_failure_pattern_hypothesis/v1``: competing hypotheses, each with typed evidence for and against.

    The capture is validated against the report first: what it says was
    checked decides whether an absence rules a hypothesis out.
    """
    errors = agent_failure_capture_errors(capture, report)
    if errors:
        raise AgentDebugIncidentError("agent_failure_capture/v1 failed validation: " + "; ".join(errors))
    findings = list(report.get("findings") or ())
    checked = set(capture.get("checked_kinds") or ())
    complete = bool(capture.get("reading_complete"))
    hypotheses: list[dict[str, Any]] = []
    for index, pattern in enumerate(_OBSERVABLE_PATTERNS[str(capture.get("observable"))], start=1):
        evidence_for, evidence_against, unavailable = _evidence(pattern, findings, checked, complete)
        if evidence_for:
            status = "supported"
            confidence = "medium" if len(evidence_for) >= 2 and not evidence_against else "low"
        elif any(ref.startswith("absence:") for ref in evidence_against):
            status, confidence = "ruled_out", "none"
        else:
            status, confidence = "unresolved", "low"
        hypotheses.append(
            {
                "hypothesis_id": f"H{index}",
                "pattern": pattern,
                "status": status,
                "confidence": confidence,
                "evidence_for": evidence_for,
                "evidence_against": evidence_against,
                "unavailable_evidence": unavailable,
                "discriminator": PATTERNS[pattern][1],
                "observed": False,
            }
        )
    open_ = [item for item in hypotheses if item["status"] != "ruled_out"]
    supported = [item for item in hypotheses if item["status"] == "supported"]
    leading = max(supported, key=lambda item: len(item["evidence_for"]), default=None)
    artifact = {
        "schema_version": AGENT_FAILURE_PATTERN_HYPOTHESIS_SCHEMA_VERSION,
        "session_id": capture["identity"]["session_id"],
        "capture_sha256": _digest(capture),
        "observable": capture.get("observable"),
        "hypotheses": hypotheses,
        "resolution": "unresolved" if len(open_) > 1 else "single_candidate",
        "leading_hypothesis": None if leading is None else leading["hypothesis_id"],
        "claim_boundary": INCIDENT_CLAIM_BOUNDARY,
    }
    errors = agent_failure_pattern_hypothesis_errors(artifact, capture)
    if errors:
        raise AgentDebugIncidentError("agent_failure_pattern_hypothesis/v1 failed validation: " + "; ".join(errors))
    return artifact


def _evidence(
    pattern: str, findings: list[Mapping[str, Any]], checked: set[str], complete: bool
) -> tuple[list[str], list[str], list[str]]:
    """Typed references for and against one pattern, and the evidence it could not see."""
    by_kind = {kind: [item for item in findings if item["kind"] == kind] for kind in FINDING_KINDS}
    absence_kind = PATTERNS[pattern][0]
    evidence_for: list[str] = []
    evidence_against: list[str] = []
    unavailable: list[str] = []
    if pattern == OUTSIDE_RECORDED_EVIDENCE:
        return [], [], ["unavailable:host_runtime"]
    if absence_kind not in checked:
        unavailable.append(f"unavailable:{absence_kind}")
    if not complete:
        unavailable.append("unavailable:unread_rows")
    error_starts = {item["citation"]["message_ids"][0] for item in by_kind[TOOL_ERROR]}
    if pattern == TOOL_ERROR_RETRY_LOOP:
        for retry in by_kind[IDENTICAL_RETRY_AFTER_ERROR]:
            # The retry failing again supports a loop; a retry that succeeded broke it.
            target = evidence_for if retry["citation"]["message_ids"][1] in error_starts else evidence_against
            target.append(f"finding:{retry['finding_id']}")
    elif pattern == CONTEXT_LOSS_AFTER_COMPACTION:
        failure_refs = [item["citation"]["message_ids"][0] for item in by_kind[TOOL_ERROR]]
        for boundary in by_kind[COMPACTION_BOUNDARY]:
            at = boundary["citation"]["message_ids"][0]
            target = evidence_for if any(ref > at for ref in failure_refs) else evidence_against
            target.append(f"finding:{boundary['finding_id']}")
    elif pattern == UNWATCHED_BACKGROUND_WORK:
        evidence_for.extend(f"finding:{item['finding_id']}" for item in by_kind[BACKGROUND_WITHOUT_NOTIFY])
    if not by_kind[absence_kind] and absence_kind in checked and complete:
        evidence_against.append(f"absence:{absence_kind}")
    return evidence_for, evidence_against, unavailable


def agent_failure_pattern_hypothesis_errors(artifact: Mapping[str, Any], capture: Mapping[str, Any]) -> list[str]:
    """Every reason the hypothesis artifact is not valid against its capture.

    Refused: fewer than two hypotheses; an unknown pattern, status, or
    confidence; a reference to a finding the capture did not observe; an
    absence the capture cannot vouch for (the kind was unchecked, the reading
    was incomplete, or the kind has findings); a supported hypothesis with no
    evidence for it; a ruled-out one without an absence; ``high`` confidence
    while a competitor is still open; a resolution that hides an open
    competitor; and a capture digest that does not match.
    """
    errors: list[str] = []
    if artifact.get("schema_version") != AGENT_FAILURE_PATTERN_HYPOTHESIS_SCHEMA_VERSION:
        errors.append(f"schema_version must be {AGENT_FAILURE_PATTERN_HYPOTHESIS_SCHEMA_VERSION}")
    if artifact.get("capture_sha256") != _digest(capture):
        errors.append("capture_sha256 does not match the capture")
    if artifact.get("session_id") != (capture.get("identity") or {}).get("session_id"):
        errors.append("session_id names another session than the capture")
    if artifact.get("claim_boundary") != INCIDENT_CLAIM_BOUNDARY:
        errors.append("claim_boundary must be the incident claim boundary")
    hypotheses = artifact.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:
        return errors + ["at least two competing hypotheses are required"]
    observed = {item.get("finding_id"): item.get("kind") for item in capture.get("observed_findings") or ()}
    checked = set(capture.get("checked_kinds") or ())
    complete = bool(capture.get("reading_complete"))
    ids: set[str] = set()
    open_ids: list[str] = []
    for item in hypotheses:
        label = str(item.get("hypothesis_id"))
        if label in ids:
            errors.append(f"{label} is repeated")
        ids.add(label)
        if item.get("pattern") not in PATTERNS:
            errors.append(f"{label} names an unknown pattern")
            continue
        status = item.get("status")
        if status not in STATUSES or item.get("confidence") not in CONFIDENCES:
            errors.append(f"{label} has an unknown status or confidence")
            continue
        if item.get("observed") is not False:
            errors.append(f"{label} must be marked observed: false; a hypothesis is inferred")
        if item.get("discriminator") != PATTERNS[item["pattern"]][1]:
            errors.append(f"{label} must carry its pattern's own discriminator")
        for ref in [*(item.get("evidence_for") or ()), *(item.get("evidence_against") or ())]:
            problem = _reference_problem(str(ref), observed, checked, complete)
            if problem:
                errors.append(f"{label} {problem}")
        if status == "supported" and not item.get("evidence_for"):
            errors.append(f"{label} is supported with no evidence for it")
        if status == "ruled_out":
            if item.get("evidence_for") or not any(str(ref).startswith("absence:") for ref in item.get("evidence_against") or ()):
                errors.append(f"{label} is ruled out without an observed absence and with nothing for it")
            if item.get("pattern") == OUTSIDE_RECORDED_EVIDENCE:
                errors.append(f"{label} rules out evidence outside the record, which records cannot do")
            if item.get("confidence") != "none":
                errors.append(f"{label} is ruled out and must carry confidence none")
        else:
            open_ids.append(label)
            if item.get("confidence") == "none":
                errors.append(f"{label} is open and cannot carry confidence none")
    for item in hypotheses:
        if item.get("confidence") == "high" and len(open_ids) > 1:
            errors.append(f"{item.get('hypothesis_id')} claims high confidence while a competing hypothesis is open")
    expected_resolution = "unresolved" if len(open_ids) > 1 else "single_candidate"
    if artifact.get("resolution") != expected_resolution:
        errors.append(f"resolution must be {expected_resolution} with {len(open_ids)} open hypotheses")
    leading = artifact.get("leading_hypothesis")
    if leading is not None and not any(
        item.get("hypothesis_id") == leading and item.get("status") == "supported" for item in hypotheses
    ):
        errors.append("leading_hypothesis must name a supported hypothesis")
    return errors


def _reference_problem(ref: str, observed: Mapping[str, Any], checked: set[str], complete: bool) -> str | None:
    kind, _, value = ref.partition(":")
    if kind == "finding":
        return None if value in observed else f"cites {ref}, which the capture did not observe"
    if kind == "absence":
        if value not in FINDING_KINDS or value not in checked or not complete:
            return f"cites {ref}, which an unchecked or incomplete reading cannot show"
        if value in observed.values():
            return f"cites {ref}, but the capture observed that kind"
        return None
    return f"cites {ref}, which is not a finding or absence reference"


# --- recovery --------------------------------------------------------------


def build_contained_recovery_action(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    """``contained_recovery_action/v1``: the smallest reversible step for the leading hypothesis, not executed."""
    leading = next(
        (item for item in hypothesis["hypotheses"] if item["hypothesis_id"] == hypothesis.get("leading_hypothesis")),
        None,
    )
    pattern = leading["pattern"] if leading is not None else OUTSIDE_RECORDED_EVIDENCE
    action = RECOVERY_ACTIONS[pattern]
    artifact = {
        "schema_version": CONTAINED_RECOVERY_ACTION_SCHEMA_VERSION,
        "session_id": hypothesis["session_id"],
        "hypothesis_sha256": _digest(hypothesis),
        "for_hypothesis": None if leading is None else leading["hypothesis_id"],
        "action": action,
        "step": RECOVERY_ACTION_STEPS[action],
        "targets": [] if leading is None else list(leading["evidence_for"]),
        "reversible": True,
        "requires_approval": True,
        "executed": False,
        "not_performed": list(NOT_PERFORMED),
        "claim_boundary": INCIDENT_CLAIM_BOUNDARY,
    }
    errors = contained_recovery_action_errors(artifact, hypothesis)
    if errors:
        raise AgentDebugIncidentError("contained_recovery_action/v1 failed validation: " + "; ".join(errors))
    return artifact


def contained_recovery_action_errors(artifact: Mapping[str, Any], hypothesis: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if artifact.get("schema_version") != CONTAINED_RECOVERY_ACTION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CONTAINED_RECOVERY_ACTION_SCHEMA_VERSION}")
    if artifact.get("hypothesis_sha256") != _digest(hypothesis):
        errors.append("hypothesis_sha256 does not match the hypothesis artifact")
    if artifact.get("executed") is not False or artifact.get("requires_approval") is not True:
        errors.append("a contained recovery action is never executed here and always requires approval")
    if artifact.get("reversible") is not True:
        errors.append("a contained recovery action must be reversible")
    if artifact.get("action") not in RECOVERY_ACTION_STEPS:
        errors.append("action is not in the closed recovery vocabulary")
    elif artifact.get("step") != RECOVERY_ACTION_STEPS[artifact["action"]]:
        errors.append("step must be the recovery action's own step")
    if artifact.get("claim_boundary") != INCIDENT_CLAIM_BOUNDARY:
        errors.append("claim_boundary must be the incident claim boundary")
    if set(NOT_PERFORMED) - set(artifact.get("not_performed") or ()):
        errors.append("not_performed must list every action diagnosis does not take")
    leading = hypothesis.get("leading_hypothesis")
    if artifact.get("for_hypothesis") != leading:
        errors.append("for_hypothesis must be the hypothesis artifact's leading hypothesis")
    lead = next((item for item in hypothesis.get("hypotheses") or () if item.get("hypothesis_id") == leading), None)
    allowed = set(lead.get("evidence_for") or ()) if lead else set()
    if set(artifact.get("targets") or ()) - allowed:
        errors.append("targets must be evidence for the leading hypothesis")
    expected = RECOVERY_ACTIONS[lead["pattern"]] if lead else RECOVERY_ACTIONS[OUTSIDE_RECORDED_EVIDENCE]
    if artifact.get("action") != expected:
        errors.append(f"action must be {expected} for the leading hypothesis")
    return errors


# --- incident --------------------------------------------------------------


def build_agent_debug_incident(
    report: Mapping[str, Any],
    *,
    observable: str = "unspecified",
    receipts: Sequence[tuple[str, Any]] = (),
) -> dict[str, Any]:
    """The three artifacts, each validated against the one it is built on."""
    capture = build_agent_failure_capture(report, observable=observable, receipts=receipts)
    hypothesis = build_agent_failure_pattern_hypothesis(capture, report)
    recovery = build_contained_recovery_action(hypothesis)
    return {
        "agent_failure_capture": capture,
        "agent_failure_pattern_hypothesis": hypothesis,
        "contained_recovery_action": recovery,
    }


def agent_debug_incident_errors(report: Mapping[str, Any], incident: Mapping[str, Any]) -> list[str]:
    capture = incident.get("agent_failure_capture")
    hypothesis = incident.get("agent_failure_pattern_hypothesis")
    recovery = incident.get("contained_recovery_action")
    if not isinstance(capture, Mapping) or not isinstance(hypothesis, Mapping) or not isinstance(recovery, Mapping):
        return list(agent_debug_report_errors(report)) + [
            "the incident must carry a capture, a hypothesis artifact, and a recovery action"
        ]
    # The capture check validates the report first.
    errors = agent_failure_capture_errors(capture, report)
    errors.extend(agent_failure_pattern_hypothesis_errors(hypothesis, capture))
    errors.extend(contained_recovery_action_errors(recovery, hypothesis))
    return errors


def format_agent_debug_incident(incident: Mapping[str, Any]) -> str:
    capture = incident["agent_failure_capture"]
    hypothesis = incident["agent_failure_pattern_hypothesis"]
    recovery = incident["contained_recovery_action"]
    out = [f"Incident (observable {capture['observable']})"]
    receipts = capture["receipts"]
    out.append(f"  receipts admitted {len(receipts['accepted'])}    refused {len(receipts['rejected'])}")
    out.extend(f"    refused {item['receipt']} {item['unit_id'] or ''} {item['reason']}".rstrip() for item in receipts["rejected"])
    out.append(f"  unavailable: {', '.join(item['evidence'] for item in capture['unavailable'])}")
    out.append(f"Hypotheses ({hypothesis['resolution']})")
    for item in hypothesis["hypotheses"]:
        out.append(f"  {item['hypothesis_id']} {item['pattern']}: {item['status']}, confidence {item['confidence']}")
        out.append(f"    for {', '.join(item['evidence_for']) or '(none)'}")
        out.append(f"    against {', '.join(item['evidence_against']) or '(none)'}")
        if item["unavailable_evidence"]:
            out.append(f"    could not see {', '.join(item['unavailable_evidence'])}")
        out.append(f"    decided by {item['discriminator']}")
    out.append("Contained recovery (proposed, requires approval, not executed)")
    out.append(f"  {recovery['action']}: {recovery['step']}")
    if recovery["targets"]:
        out.append(f"  targets {', '.join(recovery['targets'])}")
    out.append(f"  not performed: {', '.join(recovery['not_performed'])}")
    return "\n".join(out)


# --- export ----------------------------------------------------------------

_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'=(])(?:/[A-Za-z0-9._-]+/|~/|[A-Za-z]:\\)")
# Keys raw session material arrives under. The package has none by
# construction; the scan refuses one that does.
_RAW_KEYS = frozenset(
    {"content", "prompt", "prompts", "messages", "arguments", "output", "result", "results", "transcript", "path", "text"}
)
_MAX_EXPORT_STRING = 600
# The closed key shape of every artifact the export carries: a dict maps each
# allowed key to the shape of its value (None for a scalar or a list of
# scalars), and a one-item list is a list of that shape. A key outside it is
# refused, so a note added to a reviewed payload never rides into the package.
_UNAVAILABLE_SHAPE = [{"kind": None, "reason": None}]
_REPORT_SHAPE: dict[str, Any] = {
    "schema_version": None,
    "source": {
        "kind": None,
        "path": None,
        "label": None,
        "locator": None,
        "requested_session": None,
        "selection": None,
        "snapshot": {"session_rows": None, "max_message_id": None, "bytes": None, "mtime_ns": None},
    },
    "session": {"id": None, "source": None, "started_at": None, "ended_at": None, "end_reason": None},
    "turns": {"start": None, "end": None, "turn_count": None},
    "budget": {
        "max_rows": None,
        "max_row_bytes": None,
        "rows_read": None,
        "bytes_read": None,
        "bytes_skipped": None,
        "row_limit_reached": None,
        "oversized_rows": None,
        "oversized_refs": None,
        "complete": None,
    },
    "counts": {"tool_calls": None, "tool_calls_without_id": None, "tool_calls_with_arguments": None},
    "checked_kinds": None,
    "unavailable": _UNAVAILABLE_SHAPE,
    "findings": [{"finding_id": None, "kind": None, "citation": {key: None for key in CITATION_KEYS}}],
    "finding_counts": {kind: None for kind in FINDING_KINDS},
    "observed": None,
    "claim_boundary": None,
}
_INCIDENT_SHAPE: dict[str, Any] = {
    "agent_failure_capture": {
        "schema_version": None,
        "identity": {
            "session_id": None,
            "source_kind": None,
            "source_label": None,
            "selection": None,
            "turns": _REPORT_SHAPE["turns"],
            "snapshot": _REPORT_SHAPE["source"]["snapshot"],
            "report_sha256": None,
        },
        "observable": None,
        "observed_findings": [{"finding_id": None, "kind": None}],
        "finding_counts": _REPORT_SHAPE["finding_counts"],
        "checked_kinds": None,
        "reading_complete": None,
        "unavailable": [{"evidence": None, "reason": None}],
        "receipts": {
            "accepted": [
                {
                    "ref": None,
                    "receipt": None,
                    "schema_version": None,
                    "session_id": None,
                    "run_id": None,
                    "unit_id": None,
                    "configuration_digest": None,
                    "observed_at": None,
                    "status": None,
                    "exit_code": None,
                }
            ],
            "rejected": [{"receipt": None, "unit_id": None, "reason": None}],
        },
        "observed": None,
        "claim_boundary": None,
    },
    "agent_failure_pattern_hypothesis": {
        "schema_version": None,
        "session_id": None,
        "capture_sha256": None,
        "observable": None,
        "hypotheses": [
            {
                "hypothesis_id": None,
                "pattern": None,
                "status": None,
                "confidence": None,
                "evidence_for": None,
                "evidence_against": None,
                "unavailable_evidence": None,
                "discriminator": None,
                "observed": None,
            }
        ],
        "resolution": None,
        "leading_hypothesis": None,
        "claim_boundary": None,
    },
    "contained_recovery_action": {
        "schema_version": None,
        "session_id": None,
        "hypothesis_sha256": None,
        "for_hypothesis": None,
        "action": None,
        "step": None,
        "targets": None,
        "reversible": None,
        "requires_approval": None,
        "executed": None,
        "not_performed": None,
        "claim_boundary": None,
    },
}
_PAYLOAD_SHAPE: dict[str, Any] = {**_REPORT_SHAPE, "incident": _INCIDENT_SHAPE}
# Report fields that never leave the machine: the absolute source path, and
# the selector the user typed (free text that only located the session).
_EXPORT_DROPPED_SOURCE_KEYS = ("path", "requested_session")


def _keys_outside(value: Any, shape: Any, where: str) -> list[str]:
    """Every key in ``value`` that ``shape`` does not allow, by path."""
    if isinstance(shape, dict):
        if not isinstance(value, Mapping):
            return [] if value is None else [f"{where} must be an object"]
        outside = [f"{where}.{key}" for key in value if key not in shape]
        for key, sub in shape.items():
            if sub is not None and key in value:
                outside.extend(_keys_outside(value[key], sub, f"{where}.{key}"))
        return outside
    if isinstance(shape, list):
        if not isinstance(value, list):
            return [] if value is None else [f"{where} must be a list"]
        outside = []
        for index, item in enumerate(value):
            outside.extend(_keys_outside(item, shape[0], f"{where}[{index}]"))
        return outside
    if isinstance(value, Mapping) or (isinstance(value, list) and any(isinstance(item, (Mapping, list)) for item in value)):
        return [f"{where} must be a value, not a structure"]
    return []


def build_agent_debug_export(
    payload: Mapping[str, Any],
    *,
    hermes_home: str | Path | None = None,
    session_record: str | Path | None = None,
) -> dict[str, Any]:
    """The redacted package for a reviewed ``agent-debug --json`` payload, after re-checking its references.

    The payload must keep the closed key shape the reader and the incident
    builders produce; a key outside it (a reviewer's note, a memo on the
    recovery) is refused by path rather than carried or silently dropped.
    """
    incident = payload.get("incident")
    if not isinstance(incident, Mapping):
        raise AgentDebugIncidentError("the payload carries no incident; build it with agent-debug --json first")
    report = {key: value for key, value in payload.items() if key != "incident"}
    errors = agent_debug_incident_errors(report, incident)
    if errors:
        raise AgentDebugIncidentError("the payload failed validation: " + "; ".join(errors))
    outside = _keys_outside(payload, _PAYLOAD_SHAPE, "payload")
    if outside:
        raise AgentDebugIncidentError("the payload carries keys outside the export shape: " + ", ".join(outside))
    reference_errors = agent_debug_reference_errors(report, hermes_home=hermes_home, session_record=session_record)
    if reference_errors:
        raise AgentDebugIncidentError("the report's references no longer hold: " + "; ".join(reference_errors))
    source = {key: value for key, value in (report.get("source") or {}).items() if key not in _EXPORT_DROPPED_SOURCE_KEYS}
    redacted_report = {**report, "source": source}
    package = {
        "schema_version": AGENT_DEBUG_EXPORT_SCHEMA_VERSION,
        "report": redacted_report,
        "incident": {key: incident[key] for key in ("agent_failure_capture", "agent_failure_pattern_hypothesis", "contained_recovery_action")},
        "redaction": {
            "dropped": [f"source.{key}" for key in _EXPORT_DROPPED_SOURCE_KEYS],
            "never_read": ["prompts", "replies", "tool arguments", "tool output", "other sessions' rows"],
        },
        "approval": {"requires_user_review": True, "uploaded": False, "issue_filed": False},
        "claim_boundary": EXPORT_CLAIM_BOUNDARY,
    }
    leaks = agent_debug_export_leaks(package)
    if leaks:
        raise AgentDebugIncidentError("the export failed its leak scan: " + "; ".join(leaks))
    return package


def agent_debug_export_leaks(package: Any) -> list[str]:
    """Every string or key in ``package`` that could carry private material; deterministic."""
    leaks: list[str] = []

    def walk(value: Any, where: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key) in _RAW_KEYS:
                    leaks.append(f"{where}.{key} is a raw-material key")
                walk(item, f"{where}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")
        elif isinstance(value, str):
            if _ABSOLUTE_PATH.search(value):
                leaks.append(f"{where} carries a private absolute path")
            if is_secret_value_shaped(value):
                leaks.append(f"{where} carries a credential-shaped value")
            if is_raw_pii_shaped(value):
                leaks.append(f"{where} carries a phone number or email address")
            if len(value) > _MAX_EXPORT_STRING or "\n" in value:
                leaks.append(f"{where} carries a body-length string")

    walk(package, "package")
    return leaks


def agent_debug_export_target_problem(
    output: str | Path,
    *,
    hermes_home: str | Path | None = None,
    session_record: str | Path | None = None,
) -> str | None:
    """Why ``output`` cannot take the export, found without writing; None when it can.

    The parent directory is resolved (so a symlinked directory into the Hermes
    home is seen for what it is) and the file name is not: whatever already
    sits at that name -- a file, or a symlink, dangling or not -- is
    "already exists", never something to follow.
    """
    named = Path(output).expanduser()
    parent = named.parent.resolve()
    target = parent / named.name
    if not parent.is_dir():
        return f"the export directory {parent.name} does not exist"
    if session_record is not None and target == Path(session_record).expanduser().resolve():
        return "the export may not replace the session record"
    protected = [Path(item).expanduser().resolve() for item in (hermes_home,) if item is not None]
    if any(target == root or root in target.parents for root in protected):
        return "the export may not be written inside the Hermes home it was read from"
    if os.path.lexists(target):
        return f"{target.name} already exists; an export never overwrites"
    return None


def write_agent_debug_export(
    package: Mapping[str, Any],
    output: str | Path,
    *,
    hermes_home: str | Path | None = None,
    session_record: str | Path | None = None,
) -> Path:
    """Write the package to a new private file; refuse to overwrite, to follow a link, or to write inside the source.

    The file is created with ``O_EXCL`` (and ``O_NOFOLLOW`` where the platform
    has it), so an existing name -- including a dangling symlink -- is refused
    rather than written through. A write that fails removes the partial file.
    """
    leaks = agent_debug_export_leaks(package)
    if leaks:
        raise AgentDebugIncidentError("the export failed its leak scan: " + "; ".join(leaks))
    problem = agent_debug_export_target_problem(output, hermes_home=hermes_home, session_record=session_record)
    if problem is not None:
        raise AgentDebugIncidentError(problem)
    named = Path(output).expanduser()
    target = named.parent.resolve() / named.name
    text = json.dumps(package, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        raise AgentDebugIncidentError(f"{target.name} already exists; an export never overwrites") from None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except OSError:
        with contextlib.suppress(OSError):
            target.unlink()
        raise
    return target


# --- helpers ---------------------------------------------------------------


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _safe_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_ID.match(value) else None


def _epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = strict_fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()
