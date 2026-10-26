"""Public compatibility facade for revision-bound release evidence identity."""

from __future__ import annotations

from .release_evidence_verification import (
    RELEASE_EVIDENCE_VERIFICATION_SCHEMA,
    VERIFY_DIRTY,
    VERIFY_LEGACY_SCHEMA,
    VERIFY_MATCHING,
    VERIFY_MISMATCHED_REVISION,
    VERIFY_MISSING,
    VERIFY_STALE,
    VERIFY_UNVERIFIABLE,
    VERIFY_VERDICTS,
    verify_release_evidence_bundle,
)
from .release_source_identity import (
    RELEASE_EVIDENCE_BUNDLE_SCHEMA_V2,
    RELEASE_INPUT_MANIFEST_SCHEMA,
    RELEASE_SOURCE_IDENTITY_SCHEMA,
    SOURCE_ORIGINS,
    build_input_manifest,
    probe_source_identity,
)

__all__ = (
    "RELEASE_EVIDENCE_BUNDLE_SCHEMA_V2",
    "RELEASE_SOURCE_IDENTITY_SCHEMA",
    "RELEASE_INPUT_MANIFEST_SCHEMA",
    "RELEASE_EVIDENCE_VERIFICATION_SCHEMA",
    "VERIFY_MATCHING",
    "VERIFY_DIRTY",
    "VERIFY_MISMATCHED_REVISION",
    "VERIFY_STALE",
    "VERIFY_UNVERIFIABLE",
    "VERIFY_LEGACY_SCHEMA",
    "VERIFY_MISSING",
    "VERIFY_VERDICTS",
    "SOURCE_ORIGINS",
    "probe_source_identity",
    "build_input_manifest",
    "verify_release_evidence_bundle",
)
