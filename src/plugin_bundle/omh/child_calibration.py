"""Model-matched guidance for every native child, separate from protocol bytes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from .calibration import (
    HIGH_EFFORT_CALIBRATIONS,
    MODEL_HIGH_EFFORT_CALIBRATIONS,
    calibration_entry_for_model,
)
from .delegation_protocol import CHILD_PLATFORM

CHILD_CALIBRATION_SECTION_ID: Final[str] = "omh.child-calibration"
# The host refuses oversized sections rather than truncating them. Include the
# largest exact override or family block; headings are host-owned framing.
CHILD_CALIBRATION_SECTION_MAX_CHARS: Final[int] = max(
    len(text) for table in (HIGH_EFFORT_CALIBRATIONS, MODEL_HIGH_EFFORT_CALIBRATIONS)
    for text in table.values()
)


def child_calibration_section(session_info: object) -> str:
    """Use this child's model, once, even when effort is absent, low, or unknown.

    Hermes' section metadata does not reliably include reasoning effort, so it
    is not an admission gate here. Do not read parent config, routed intent, or
    remembered main-agent identity. Every non-child session renders nothing.
    """
    if not isinstance(session_info, Mapping) or session_info.get("platform") != CHILD_PLATFORM:
        return ""
    return calibration_entry_for_model(str(session_info.get("model", "") or ""))[1]
