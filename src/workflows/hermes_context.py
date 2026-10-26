"""Prepared context consumers for Hermes-owned downstream workflows."""

from __future__ import annotations

from .decision_receipt_handoffs import (
    DecisionArtifact,
    DecisionReceiptHandoff,
    build_decision_receipt_handoff,
)


def build_product_brief_context(discovery_receipt: DecisionArtifact) -> DecisionReceiptHandoff:
    """Consume one discovery receipt as prepared product-brief context."""
    return build_decision_receipt_handoff(discovery_receipt, target_workflow="product-brief")
