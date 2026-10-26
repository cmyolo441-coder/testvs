"""Evaluator-backed block selection with metadata-only omissions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .memory_governance import MEMORY_BLOCK_SCHEMA_VERSION, evaluate_memory_replay, stable_artifact_identity

if TYPE_CHECKING:
    from .memory_blocks import MemoryBlock


@dataclass(frozen=True)
class MemoryBlockSelection:
    eligible: tuple[MemoryBlock, ...]
    evaluations: dict[str, dict[str, object]]
    omissions: tuple[dict[str, object], ...]


def select_memory_blocks(
    blocks: tuple[MemoryBlock, ...] | list[MemoryBlock],
    *,
    now: datetime | None = None,
    omh_home: str | Path | None = None,
    requested_scope: dict[str, object] | None = None,
    tombstoned: bool = False,
) -> MemoryBlockSelection:
    """Run the shared evaluator before any render, index, or block read."""
    reviews = _reviews(omh_home)
    tombstones = _tombstones(omh_home)
    moment = now or datetime.now(timezone.utc)
    eligible: list[MemoryBlock] = []
    evaluations: dict[str, dict[str, object]] = {}
    omissions: list[dict[str, object]] = []
    for block in blocks:
        evaluation = _evaluate(
            block,
            now=moment,
            requested_scope=requested_scope,
            reviews=reviews,
            require_review=omh_home is not None,
            tombstoned=tombstoned or _is_tombstoned(block, tombstones),
        )
        evaluations[block.block_id] = evaluation
        if evaluation["eligible"]:
            eligible.append(block)
        else:
            omissions.append(_omission(block, evaluation))
    return MemoryBlockSelection(tuple(eligible), evaluations, tuple(omissions))


def _evaluate(
    block: MemoryBlock,
    *,
    now: datetime,
    requested_scope: dict[str, object] | None,
    reviews: dict[str, dict[str, object]],
    require_review: bool,
    tombstoned: bool,
) -> dict[str, object]:
    if block.schema_version != MEMORY_BLOCK_SCHEMA_VERSION:
        result = _legacy_result(now)
    else:
        result = evaluate_memory_replay(
            block.to_dict(),
            now=now,
            requested_scope=requested_scope,
            review_resolver=reviews or None,
            tombstoned=tombstoned,
        )
        admission = block.admission
        review_id = admission.get("review_id") if isinstance(admission, dict) else None
        approved = admission.get("state") in {"approved_manual", "approved_auto_safe"} if isinstance(admission, dict) else False
        missing_review = require_review and approved and review_id not in reviews
        if missing_review and result.get("reason_code") != "payload_digest_mismatch":
            result["eligible"] = False
            result["reason_code"] = "review_not_found"
    result["source_class"] = block.source_class
    result["retention_class"] = block.retention.get("class")
    result["artifact_identity"] = stable_artifact_identity(block.to_dict()) if block.schema_version == MEMORY_BLOCK_SCHEMA_VERSION else {}
    return result


def _legacy_result(now: datetime) -> dict[str, object]:
    return {
        "evaluated_at": now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "eligible": False,
        "reason_code": "review_required_legacy",
        "admission_state": "pending_review",
        "admission_mode": None,
        "payload_digest": None,
        "source_class": "omh_local",
        "retention_class": None,
    }


def _omission(block: MemoryBlock, evaluation: dict[str, object]) -> dict[str, object]:
    return {
        "block_id": block.block_id,
        "revision": block.revision,
        "reason_code": evaluation.get("reason_code", "ineligible"),
    }


def _reviews(omh_home: str | Path | None) -> dict[str, dict[str, object]]:
    if omh_home is None:
        return {}
    directory = Path(omh_home).expanduser() / "memory" / "block_reviews"
    result: dict[str, dict[str, object]] = {}
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return result
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            review = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(review, dict) and isinstance(review.get("review_id"), str):
            result[review["review_id"]] = review
    return result


def _tombstones(omh_home: str | Path | None) -> set[tuple[str, int]]:
    if omh_home is None:
        return set()
    directory = Path(omh_home).expanduser() / "memory" / "tombstones"
    found: set[tuple[str, int]] = set()
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return found
    for path in paths:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("block_id"), str) and isinstance(data.get("revision"), int):
            found.add((data["block_id"], data["revision"]))
    return found


def _is_tombstoned(block: MemoryBlock, tombstones: set[tuple[str, int]]) -> bool:
    return (block.block_id, block.revision) in tombstones
