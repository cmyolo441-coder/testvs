"""Pure verification and verdict payload construction for release evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from ..system.local_store import utc_now
from .release_notes_verification import verify_notes_binding
from .release_source_identity import (
    RELEASE_EVIDENCE_BUNDLE_SCHEMA_V2,
    _DEFAULT_GIT_RUNNER,
    _DIGEST_RE,
    build_input_manifest,
    probe_source_identity,
)

RELEASE_EVIDENCE_VERIFICATION_SCHEMA = "omh_release_evidence_verification/v1"
RELEASE_EVIDENCE_SOURCE_BINDING_SCHEMA = "omh_release_evidence_verification_source_binding/v1"

VERIFY_MATCHING = "matching"
VERIFY_DIRTY = "dirty"
VERIFY_MISMATCHED_REVISION = "mismatched_revision"
VERIFY_STALE = "stale"
VERIFY_UNVERIFIABLE = "unverifiable"
VERIFY_LEGACY_SCHEMA = "legacy_schema"
VERIFY_MISSING = "missing"
VERIFY_VERDICTS = (
    VERIFY_MATCHING,
    VERIFY_DIRTY,
    VERIFY_MISMATCHED_REVISION,
    VERIFY_STALE,
    VERIFY_UNVERIFIABLE,
    VERIFY_LEGACY_SCHEMA,
    VERIFY_MISSING,
)


def verify_release_evidence_bundle(
    bundle: Mapping[str, object] | None,
    *,
    version: str = "",
    repo_root: str | Path | None = None,
    paths: Any = None,
    artifact: str | Path | None = None,
    archive_digest: str = "",
    artifact_digest: str = "",
    runner: Callable[..., Any] = _DEFAULT_GIT_RUNNER,
    notes_file: str | Path | None = None,
) -> dict[str, object]:
    """Re-compute the recorded binding and return a verdict; never writes.

    The verdict vocabulary is closed: `matching`, `dirty` (the current
    worktree has uncommitted changes; checked first because it dominates
    staleness), `mismatched_revision` (the recorded commit differs from the
    current HEAD), `stale` (same revision, but the tree, a declared input
    digest, or the artifact digest changed), `unverifiable` (identity is
    unavailable on either side, or a recorded artifact digest has nothing
    supplied to check against), `legacy_schema` (the bundle predates the v2
    binding contract), and `missing` (no bundle was recorded at all).
    """
    checked_at = utc_now()
    if bundle is None:
        return _verification_payload(
            VERIFY_MISSING,
            version=version,
            checked_at=checked_at,
            reasons=["no release evidence bundle is recorded for the requested version"],
        )
    if isinstance(bundle, Mapping):
        version = str(bundle.get("version") or version)
    else:
        return _verification_payload(
            VERIFY_UNVERIFIABLE,
            version=version,
            checked_at=checked_at,
            reasons=["the recorded bundle is not a JSON object"],
        )
    if bundle.get("schema_version") != RELEASE_EVIDENCE_BUNDLE_SCHEMA_V2:
        return _verification_payload(
            VERIFY_LEGACY_SCHEMA,
            version=version,
            checked_at=checked_at,
            reasons=[
                "the recorded bundle predates omh_release_evidence_bundle/v2 and carries no revision binding; "
                "regenerate it with `omh release evidence-bundle --write --repo-root <root>`"
            ],
        )
    identity = bundle.get("source_identity")
    if not isinstance(identity, Mapping) or identity.get("identity_status") != "available":
        return _verification_payload(
            VERIFY_UNVERIFIABLE,
            version=version,
            checked_at=checked_at,
            reasons=["the recorded bundle carries no available source identity"],
        )
    recorded_manifest = identity.get("input_manifest")
    if not isinstance(recorded_manifest, Mapping):
        return _verification_payload(
            VERIFY_UNVERIFIABLE,
            version=version,
            checked_at=checked_at,
            reasons=["the recorded bundle carries no input manifest"],
        )
    current = probe_source_identity(
        repo_root,
        runner=runner,
        archive_digest=archive_digest,
        artifact_digest=artifact_digest,
    )
    if current["identity_status"] != "available":
        return _verification_payload(
            VERIFY_UNVERIFIABLE,
            version=version,
            checked_at=checked_at,
            reasons=[
                "the current source identity is unavailable; pass --repo-root, --archive-digest, or "
                "--artifact-digest so the recorded binding can be checked"
            ],
        )
    if current["origin"] != identity.get("origin"):
        return _verification_payload(
            VERIFY_UNVERIFIABLE,
            version=version,
            checked_at=checked_at,
            reasons=[
                f"the bundle was recorded from origin {identity.get('origin')} but the current source "
                f"origin is {current['origin']}"
            ],
        )
    origin = str(current["origin"])
    if origin == "git_checkout":
        if current["dirty"]:
            return _verification_payload(
                VERIFY_DIRTY,
                version=version,
                checked_at=checked_at,
                reasons=[
                    f"the worktree has {current['dirty_file_count']} uncommitted change(s); "
                    "evidence must be bound to a clean checkout"
                ],
            )
        if current["commit_sha"] != identity.get("commit_sha"):
            return _verification_payload(
                VERIFY_MISMATCHED_REVISION,
                version=version,
                checked_at=checked_at,
                reasons=["the recorded commit differs from the current HEAD commit"],
            )
        if current["tree_sha"] != identity.get("tree_sha"):
            return _verification_payload(
                VERIFY_STALE,
                version=version,
                checked_at=checked_at,
                reasons=["the tracked tree differs from the recorded tree at the same commit"],
            )
    elif origin == "source_archive":
        if not _DIGEST_RE.match(archive_digest or ""):
            return _verification_payload(
                VERIFY_UNVERIFIABLE,
                version=version,
                checked_at=checked_at,
                reasons=["a source-archive bundle can only be checked against an explicit --archive-digest"],
            )
        if archive_digest != identity.get("archive_digest"):
            return _verification_payload(
                VERIFY_STALE,
                version=version,
                checked_at=checked_at,
                reasons=["the supplied archive digest differs from the recorded archive digest"],
            )
    elif origin == "installed_package":
        if not _DIGEST_RE.match(artifact_digest or ""):
            return _verification_payload(
                VERIFY_UNVERIFIABLE,
                version=version,
                checked_at=checked_at,
                reasons=[
                    "an installed-package bundle can only be checked against an explicit --artifact-digest"
                ],
            )
        if artifact_digest != identity.get("artifact_digest"):
            return _verification_payload(
                VERIFY_STALE,
                version=version,
                checked_at=checked_at,
                reasons=["the supplied artifact digest differs from the recorded artifact digest"],
            )
    current_manifest = build_input_manifest(source_identity=current, paths=paths, artifact=artifact)
    recorded_store = recorded_manifest.get("local_artifact_store")
    recorded_store_digest = recorded_store.get("digest") if isinstance(recorded_store, Mapping) else None
    current_store = current_manifest.get("local_artifact_store")
    current_store_digest = current_store.get("digest") if isinstance(current_store, Mapping) else None
    if recorded_store_digest != current_store_digest:
        return _verification_payload(
            VERIFY_STALE,
            version=version,
            checked_at=checked_at,
            reasons=["the local artifact store digest changed since the bundle was recorded"],
        )
    recorded_artifact = recorded_manifest.get("artifact")
    recorded_artifact = recorded_artifact if isinstance(recorded_artifact, Mapping) else {}
    if recorded_artifact.get("binding") == "recorded":
        current_artifact = current_manifest.get("artifact")
        current_artifact = current_artifact if isinstance(current_artifact, Mapping) else {}
        if artifact is not None:
            current_artifact_sha = current_artifact.get("sha256")
        elif _DIGEST_RE.match(artifact_digest or ""):
            current_artifact_sha = artifact_digest
        else:
            return _verification_payload(
                VERIFY_UNVERIFIABLE,
                version=version,
                checked_at=checked_at,
                reasons=[
                    "the bundle records an artifact digest; pass --artifact <file> or --artifact-digest "
                    "so it can be checked"
                ],
            )
        if current_artifact_sha != recorded_artifact.get("sha256"):
            return _verification_payload(
                VERIFY_STALE,
                version=version,
                checked_at=checked_at,
                reasons=["the artifact digest differs from the recorded artifact digest"],
            )
    verdict, notes_binding = verify_notes_binding(bundle, recorded_manifest, notes_file)
    result = _verification_payload(verdict, version=version, checked_at=checked_at,
                                   reasons=[] if verdict == VERIFY_MATCHING else [notes_binding])
    result['notes_binding'] = notes_binding
    result['publication_ready'] = notes_binding == 'matching' and bundle.get('publication_ready') is True
    return result


def _verification_payload(
    verdict: str,
    *,
    version: str,
    checked_at: str,
    reasons: list[str],
) -> dict[str, object]:
    return {
        "schema_version": RELEASE_EVIDENCE_VERIFICATION_SCHEMA,
        "verification": verdict,
        "source_binding": {
            "schema_version": RELEASE_EVIDENCE_SOURCE_BINDING_SCHEMA,
            "verification_status": verdict,
        },
        "version": version,
        "checked_at": checked_at,
        "reasons": reasons,
        "claim_boundary": (
            "Release evidence verification re-computes the recorded source identity and declared input "
            "digests and compares them with the current environment. It never writes, regenerates, or "
            "upgrades evidence, and a matching verdict proves evidence provenance for the recorded "
            "revision only; it does not prove deployment, adoption, or runtime behavior outside the "
            "executed gates."
        ),
    }
