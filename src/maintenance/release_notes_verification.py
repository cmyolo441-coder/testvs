"""Reconcile the optional notes extension of legacy v2 JSON evidence."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping

from .changelog import ChangelogError
from .release_notes import notes_metadata, validate_notes_metadata
from .release_source_identity import _canonical_digest


def verify_notes_binding(
    bundle: Mapping[str, object], manifest: Mapping[str, object], notes_file: str | Path | None,
) -> tuple[Literal['matching', 'stale', 'unverifiable'], str]:
    """Parse the existing v2 JSON boundary, returning only bounded verdict codes."""
    recorded = manifest.get('release_notes')
    if recorded is None and bundle.get('release_notes') is None:
        return 'matching', 'not_recorded'
    try:
        manifest_body = {key: value for key, value in manifest.items() if key != 'digest'}
        if _canonical_digest(manifest_body) != manifest.get('digest'):
            raise ChangelogError('notes_manifest_digest_invalid')
        if not isinstance(recorded, dict) or notes_file is None:
            raise ChangelogError('notes_artifact_required')
        expected = validate_notes_metadata(recorded)
        if expected != bundle.get('release_notes') or expected['version'] != bundle.get('version'):
            raise ChangelogError('notes_binding_inconsistent')
        current = notes_metadata(Path(notes_file), expected['version'])
    except ChangelogError as exc:
        return 'unverifiable', exc.code
    except OSError:
        return 'unverifiable', 'notes_artifact_unreadable'
    if current != expected:
        return 'stale', 'release_notes_digest_changed'
    return 'matching', 'matching'
