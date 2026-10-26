"""Deterministic declared-progress checks, not proof of research execution.

``research_progress/v1`` requires ``budget: {unit, limit, spent}``, ``status``,
``stop_reason``, ``axes``, ``waves`` and ``saturation_rule``. Each axis declares
``id``, ``required_coverage`` (a count of distinct evidence identifiers),
``covered_evidence_ids`` and ``unresolved_leads: [{id, required: bool}]``.
Waves are ordered ``{id, new_evidence_ids}`` records; evidence may be introduced
only once, but may cover multiple axes. Identifiers are compared literally.

The policy is explicit: ``saturation_rule.minimum_quiet_waves`` declares a
positive count of consecutive trailing waves introducing no new evidence.
Both ``saturated`` and ``completed`` require complete coverage, no required
open leads, that quiet-wave criterion, and unspent budget. Other stop reasons
are partial. At or beyond the limit even quiet research remains partial;
``budget_exhausted`` cannot be used while budget remains. ``cancelled`` and
``blocked`` may occur with or without remaining budget. Nothing is fetched,
executed, inferred from prose, or changed in the supplied record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def research_progress_errors(value: Mapping[str, Any]) -> list[str]:
    """Return deterministic structural errors; an empty list is not observation."""
    if not isinstance(value, Mapping):
        return ["research_progress must be a mapping"]
    errors = []
    for key in (
        "schema_version", "budget", "status", "stop_reason", "axes", "saturation_rule", "waves"
    ):
        if key not in value:
            errors.append(f"research_progress.{key} is required")
    if value.get("schema_version") != "research_progress/v1":
        errors.append("research_progress.schema_version must be research_progress/v1")
    if value.get("status") not in ("complete", "partial"):
        errors.append("research_progress.status must be complete or partial")
    reason = value.get("stop_reason")
    if reason not in (
        "saturated", "completed", "budget_exhausted", "cancelled", "blocked", "in_progress"
    ):
        errors.append("research_progress.stop_reason is invalid")
    finishing = reason in ("saturated", "completed")
    expected_status = "complete" if finishing else "partial"
    if value.get("status") != expected_status:
        errors.append("research_progress.status is inconsistent with stop_reason")
    errors.extend(_budget_errors(value.get("budget"), reason))
    axis_errors, coverage = _axes_errors(value.get("axes"), finishing)
    errors.extend(axis_errors)
    rule = value.get("saturation_rule")
    quiet_required = rule.get("minimum_quiet_waves") if isinstance(rule, Mapping) else None
    if type(quiet_required) is not int or quiet_required <= 0:
        errors.append("research_progress.saturation_rule.minimum_quiet_waves must be a positive integer")
    wave_errors, evidence, quiet = _waves_errors(value.get("waves"))
    errors.extend(wave_errors)
    if coverage - evidence:
        errors.append("research_progress coverage references evidence not declared in waves")
    if finishing and type(quiet_required) is int and quiet < quiet_required:
        errors.append("research_progress does not meet declared trailing quiet-wave criterion")
    return errors


def _budget_errors(budget: Any, reason: Any) -> list[str]:
    if not isinstance(budget, Mapping):
        return ["research_progress.budget must be a mapping"]
    errors = []
    if not _text(budget.get("unit")):
        errors.append("research_progress.budget.unit must be nonempty text")
    for key in ("limit", "spent"):
        if not _quantity(budget.get(key)):
            errors.append(f"research_progress.budget.{key} must be finite and nonnegative")
    if all(_quantity(budget.get(k)) for k in ("limit", "spent")):
        exhausted = budget["spent"] >= budget["limit"]
        if reason == "budget_exhausted" and not exhausted:
            errors.append("research_progress.budget_exhausted requires spent >= limit")
        if exhausted and reason in ("saturated", "completed", "in_progress"):
            errors.append("research_progress exhausted budget requires a partial stop")
    return errors


def _axes_errors(axes: Any, finishing: bool) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    coverage: set[str] = set()
    if not isinstance(axes, list) or not axes:
        return ["research_progress.axes must be a nonempty list"], coverage
    axis_ids = set()
    for index, axis in enumerate(axes):
        path = f"research_progress.axes[{index}]"
        if not isinstance(axis, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        ident = axis.get("id")
        if not _text(ident) or ident in axis_ids:
            errors.append(f"{path}.id must be unique nonempty text")
        else:
            axis_ids.add(ident)
        required = axis.get("required_coverage")
        if type(required) is not int or required < 0:
            errors.append(f"{path}.required_coverage must be a nonnegative integer")
        covered = axis.get("covered_evidence_ids")
        if not isinstance(covered, list) or not all(_text(e) for e in covered):
            errors.append(f"{path}.covered_evidence_ids must be a list of identifiers")
        else:
            coverage.update(covered)
            if len(covered) != len(set(covered)):
                errors.append(f"{path}.covered_evidence_ids contains duplicates")
            if finishing and type(required) is int and len(set(covered)) < required:
                errors.append(f"{path} does not meet required coverage")
        errors.extend(_lead_errors(axis.get("unresolved_leads"), finishing, path))
    return errors, coverage


def _lead_errors(leads: Any, finishing: bool, path: str) -> list[str]:
    if not isinstance(leads, list):
        return [f"{path}.unresolved_leads must be a list"]
    errors = []
    lead_ids = set()
    for lead in leads:
        if (
            not isinstance(lead, Mapping)
            or not _text(lead.get("id"))
            or type(lead.get("required")) is not bool
        ):
            errors.append(f"{path}.unresolved_leads requires id and boolean required")
        elif finishing and lead["required"]:
            errors.append(f"{path} has a required unresolved lead")
        if isinstance(lead, Mapping) and _text(lead.get("id")):
            if lead["id"] in lead_ids:
                errors.append(f"{path}.unresolved_leads contains duplicate id")
            lead_ids.add(lead["id"])
    return errors


def _waves_errors(waves: Any) -> tuple[list[str], set[str], int]:
    errors: list[str] = []
    evidence_ids: set[str] = set()
    quiet = 0
    if not isinstance(waves, list):
        return ["research_progress.waves must be a list"], evidence_ids, quiet
    wave_ids = set()
    for index, wave in enumerate(waves):
        path = f"research_progress.waves[{index}]"
        if not isinstance(wave, Mapping) or not _text(wave.get("id")):
            errors.append(f"{path} requires a nonempty id")
            quiet = 0
            continue
        if wave["id"] in wave_ids:
            errors.append(f"{path}.id is duplicated")
        wave_ids.add(wave["id"])
        evidence = wave.get("new_evidence_ids")
        if not isinstance(evidence, list) or not all(_text(e) for e in evidence):
            errors.append(f"{path}.new_evidence_ids must be a list of identifiers")
            quiet = 0
            continue
        for ident in evidence:
            if ident in evidence_ids:
                errors.append(f"{path}.new_evidence_ids repeats previously declared evidence")
            evidence_ids.add(ident)
        quiet = 0 if evidence else quiet + 1
    return errors, evidence_ids, quiet


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _quantity(value: Any) -> bool:
    # Integers are always finite; do not coerce arbitrary-sized integers to float.
    return (type(value) is int and value >= 0) or (
        type(value) is float and value >= 0 and value < float("inf")
    )
