"""Boundary-safe scalar and digest values for paired-run decisions."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Final, Protocol, Sequence

from ..system.metadata_safety import require_opaque_metadata_ref
from ..plugin_bundle.omh.iso_datetime import strict_fromisoformat

_UTC_Z: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_DIGEST: Final = re.compile(r"^[0-9a-f]{64}$")
MAX_DISPATCH_SECONDS: Final = 3_600
_RECEIPT_REF: Final = re.compile(
    r"^hermes-child:([A-Za-z0-9][A-Za-z0-9._-]{0,159}):[0-9a-f]{64}$",
)


class TaskValue(Protocol):
    @property
    def task_id(self) -> str: ...

    @property
    def acceptance_criteria_ref(self) -> str: ...

    @property
    def input_digest(self) -> str: ...


def canonical_digest(value: list[dict[str, str]]) -> str:
    material = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(material.encode("ascii")).hexdigest()


def exposure_digest(skills: tuple[str, ...]) -> str:
    return canonical_digest([{"skill": skill} for skill in skills])


def task_set_digest(tasks: Sequence[TaskValue]) -> str:
    return canonical_digest([
        {
            "task_id": task.task_id,
            "acceptance_criteria_ref": task.acceptance_criteria_ref,
            "input_digest": task.input_digest,
        }
        for task in sorted(
            tasks,
            key=lambda item: (
                item.task_id,
                item.acceptance_criteria_ref,
                item.input_digest,
            ),
        )
    ])


def is_safe_metadata_ref(value: str | None) -> bool:
    if value is None or "://" in value:
        return False
    try:
        require_opaque_metadata_ref(value, field="value")
    except ValueError:
        return False
    return True


def is_utc_z(value: str | None) -> bool:
    if value is None or _UTC_Z.fullmatch(value) is None:
        return False
    try:
        strict_fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def is_digest(value: str | None) -> bool:
    return value is not None and _DIGEST.fullmatch(value) is not None


def is_receipt_provenance(ref: str, run_id: str, verified_at: str) -> bool:
    match = _RECEIPT_REF.fullmatch(ref)
    return (
        match is not None
        and match.group(1) == run_id
        and is_safe_metadata_ref(run_id)
        and is_utc_z(verified_at)
    )
