"""One fanout unit per document range, derived from a `document_chunk_plan/v1` file.

`omh_document_plan` splits a long document into numbered read ranges with a
`read_file` window each. Fanout splits work into units with a file boundary
each. This module joins the two: every range becomes one unit whose input
budget is the range, whose file scope is the report file that range's reader
writes, and whose title is the per-range brief the plan already carries. The
derivation is deterministic and reads only the plan file; it opens no
document, dispatches nothing, and leaves verification to the operator's
flags.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..plugin_bundle.omh.document_chunk_plan import (
    CHARS_PER_TOKEN,
    DocumentPlanError,
    fill_brief,
    validate_document_plan,
)
from .fanout_contracts import (
    FANOUT_SPAWN_PLAN_THRESHOLD,
    FanoutContractError,
    MAX_SPAWN_PLAN_FIELD_CHARS,
)

DOCUMENT_UNIT_ID_PREFIX = "range"
# The report a range's reader commits; one file per unit so the boundaries
# are disjoint by construction.
DEFAULT_REPORT_DIR_TEMPLATE = "reports/document-plan-{plan_id}"
_MAX_PLAN_FILE_BYTES = 4 * 1024 * 1024
_LABEL_CHARS_IN_PROSE = 60


def read_document_plan_file(path: str | Path) -> dict[str, Any]:
    """Load and validate a plan file; every failure is a refusal with a reason."""
    plan_path = Path(path).expanduser()
    try:
        raw = plan_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FanoutContractError(f"document plan file not found: {plan_path}") from None
    except OSError as error:
        raise FanoutContractError(f"document plan file is unreadable: {error.__class__.__name__}") from error
    if len(raw) > _MAX_PLAN_FILE_BYTES:
        raise FanoutContractError("document plan file exceeds the plan file size cap")
    try:
        data = json.loads(raw)
    except ValueError:
        raise FanoutContractError(f"document plan file is not valid JSON: {plan_path}") from None
    try:
        return validate_document_plan(data)
    except DocumentPlanError as error:
        raise FanoutContractError(f"document plan refused: {error}") from error


def _short_label(label: str) -> str:
    text = " ".join(label.split())
    if len(text) <= _LABEL_CHARS_IN_PROSE:
        return text
    return text[: _LABEL_CHARS_IN_PROSE - 3] + "..."


def _span_text(entry: Mapping[str, Any]) -> str:
    pages = entry.get("pages")
    if isinstance(pages, Mapping):
        text = f"pages {pages['start']}-{pages['end']}"
    else:
        chars = entry["chars"]
        text = f"chars {chars['start']}-{chars['end']}"
    sections = entry.get("sections") or []
    if sections:
        text += ": " + "; ".join(str(section) for section in sections)
    return text


def _range_brief(plan: Mapping[str, Any], entry: Mapping[str, Any]) -> str:
    # A range from a tool result carries its own filled brief; a range read
    # straight out of the stored plan carries only the template, which the
    # plan file holds once so a 1,000-range plan does not repeat it. The
    # hint used to carry a `range_briefs` list; it was dropped when results
    # were bounded, so there is nothing to read there.
    filled = str(entry.get("brief") or "")
    if filled.strip():
        return filled
    hint = plan.get("delegation_hint") or {}
    template = str(hint.get("per_range_brief_template") or "")
    if template:
        return fill_brief(template, dict(plan), dict(entry))
    return f"Read {plan['source']['label']} range {entry['chunk']}/{plan['range_count']} ({_span_text(entry)})"


def _derived_spawn_plan(plan: Mapping[str, Any], unit_count: int, report_dir: str) -> dict[str, str]:
    label = _short_label(str(plan["source"]["label"]))
    budget = int(plan["inputs"]["budget_chars"])
    fields = {
        "why_parallel": (
            f"Derived from document_chunk_plan/v1 {plan['plan_id']}: {unit_count} ranges of {label} "
            f"each fit one read_file window ({budget} chars), so each unit reads exactly one range."
        ),
        "why_not_single_unit": (
            f"One reader would hold {unit_count} windows of about {budget} chars in one context and "
            "lose the earlier ranges to compression; one unit per range keeps each context to one window."
        ),
        "independence": (
            "Ranges are disjoint page or character spans; each unit reads its own range and writes "
            f"only its own report file under {report_dir}."
        ),
        "expected_evidence_shape": (
            "Per-unit run record plus a committed range report at the unit's file_scope stating "
            "covered / next / missing for its range."
        ),
    }
    for field, text in fields.items():
        if len(text) > MAX_SPAWN_PLAN_FIELD_CHARS:
            fields[field] = text[: MAX_SPAWN_PLAN_FIELD_CHARS - 3] + "..."
    return fields


def fanout_units_from_document_plan(
    plan: Mapping[str, Any],
    *,
    report_dir: str | None = None,
    owner: str | None = None,
) -> dict[str, Any]:
    """Derive the `fanout prepare` payload (units plus spawn plan) from a plan.

    Unit ids number the ranges (`range-01`), the title is the plan's per-range
    brief, the file scope is that range's report file, and the input budget is
    the plan's per-range character budget with the range's read window as its
    one source range. Above the spawn-plan threshold the payload carries a
    justification derived from the plan's own facts, so the operator sees the
    split's reason rather than an unanswered gate.
    """
    try:
        plan = validate_document_plan(dict(plan))
    except DocumentPlanError as error:
        raise FanoutContractError(f"document plan refused: {error}") from error
    ranges = plan["ranges"]
    plan_id = str(plan["plan_id"])
    directory = (report_dir or DEFAULT_REPORT_DIR_TEMPLATE.format(plan_id=plan_id)).strip().rstrip("/")
    if not directory:
        raise FanoutContractError("report_dir must name a directory for the per-range reports")
    width = len(str(len(ranges)))
    budget_chars = int(plan["inputs"]["budget_chars"])
    label = str(plan["source"]["label"])
    units: list[dict[str, Any]] = []
    for entry in ranges:
        chunk = int(entry["chunk"])
        unit_id = f"{DOCUMENT_UNIT_ID_PREFIX}-{chunk:0{width}d}"
        estimated_chars = int(entry["estimated_chars"])
        # The ceiling a unit may read is the plan's per-range budget; a single
        # range denser than the budget (one very long page) raises its own
        # ceiling rather than being frozen with a budget it cannot meet.
        chars = max(budget_chars, estimated_chars, 1)
        window = entry["read_window"]
        source_range: dict[str, Any] = {
            "source": label,
            "span": _span_text(entry),
            "offset": int(window["offset"]),
            "limit": int(window["limit"]),
            "estimated_chars": estimated_chars,
            "digest": str(entry["digest"]),
        }
        if isinstance(window.get("end_line"), int):
            source_range["end_line"] = int(window["end_line"])
        units.append(
            {
                "unit_id": unit_id,
                "title": _range_brief(plan, entry),
                "owner": owner,
                "file_scope": [f"{directory}/{unit_id}.md"],
                "depends_on": [],
                "input_budget": {
                    "chars": chars,
                    "tokens": math.ceil(chars / CHARS_PER_TOKEN),
                    "source_ranges": [source_range],
                },
            }
        )
    spawn_plan = _derived_spawn_plan(plan, len(units), directory) if len(units) > FANOUT_SPAWN_PLAN_THRESHOLD else None
    return {
        "schema_version": "fanout_document_units/v1",
        "plan_id": plan_id,
        "source": label,
        "report_dir": directory,
        "default_goal": f"Read {_short_label(label)} in {len(units)} ranges per document plan {plan_id}",
        "units": units,
        "spawn_plan": spawn_plan,
    }
