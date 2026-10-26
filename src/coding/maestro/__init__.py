"""Typed facade for external coding handoff coordination."""

from .contracts import (
    ExternalHandoffCapability,
    ExternalHandoffRequest,
    ExternalProfile,
    HandoffField,
    PreparedExternalHandoff,
    PreparedHandoffObservationAdapter,
    PreparedHandoffStatusAdapter,
)
from .facade import HermesNativeSelectionError, Maestro, build_external_handoff

__all__ = [
    "ExternalHandoffCapability",
    "ExternalHandoffRequest",
    "ExternalProfile",
    "HandoffField",
    "HermesNativeSelectionError",
    "Maestro",
    "PreparedExternalHandoff",
    "PreparedHandoffObservationAdapter",
    "PreparedHandoffStatusAdapter",
    "build_external_handoff",
]
