"""Local release-note files and their closed publication evidence binding."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Literal, TypedDict
from collections.abc import Mapping

from ..system.local_store import atomic_write_text
from .changelog import (
    ChangelogError, MAX_CHANGELOG_BYTES, MAX_NOTES_BYTES,
    bound_release_body, extract_notes, release_version, stamp_changelog,
)


class NotesMetadata(TypedDict):
    schema_version: Literal['omh_release_notes/v1']
    path: str
    version: str
    sha256: str
    byte_length: int


def read_bounded(path: Path, limit: int) -> bytes:
    with path.open('rb') as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ChangelogError('input_too_large')
    return raw


def read_notes(path: Path) -> bytes:
    raw = read_bounded(path, MAX_NOTES_BYTES)
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise ChangelogError('invalid_utf8') from exc
    if not text.strip() or not text.endswith('\n') or '\r' in text:
        raise ChangelogError('notes_not_canonical')
    return raw


def notes_metadata(path: Path, version: str) -> NotesMetadata:
    return _metadata(path, version, read_notes(path))


def _metadata(path: Path, version: str, raw: bytes) -> NotesMetadata:
    return validate_notes_metadata(dict(schema_version='omh_release_notes/v1', path=path.resolve().name,
                                       version=release_version(version),
                                       sha256='sha256:' + hashlib.sha256(raw).hexdigest(), byte_length=len(raw)))


def validate_notes_metadata(value: Mapping[str, object]) -> NotesMetadata:
    """Parse a legacy JSON boundary without expanding the closed schema."""
    if set(value) != {'schema_version', 'path', 'version', 'sha256', 'byte_length'}:
        raise ChangelogError('invalid_notes_metadata')
    if value['schema_version'] != 'omh_release_notes/v1':
        raise ChangelogError('invalid_notes_metadata')
    path, version, digest, size = value['path'], value['version'], value['sha256'], value['byte_length']
    if not isinstance(path, str) or not path or any(char in path for char in '/\\:\x00') or path in ('.', '..'):
        raise ChangelogError('invalid_notes_path')
    if not isinstance(version, str) or release_version(version) != version:
        raise ChangelogError('invalid_version')
    if not isinstance(digest, str) or re.fullmatch(r'sha256:[0-9a-f]{64}', digest) is None:
        raise ChangelogError('invalid_notes_digest')
    if type(size) is not int or not 0 < size <= MAX_NOTES_BYTES:
        raise ChangelogError('invalid_notes_length')
    return NotesMetadata(schema_version='omh_release_notes/v1', path=path, version=version,
                         sha256=digest, byte_length=size)


def prepare_notes(repo_root: Path, version: str, notes_file: Path, *, stamp: bool = False) -> NotesMetadata:
    """Stamp locally only on explicit preparation; interrupted notes writes are recoverable."""
    changelog = repo_root / 'CHANGELOG.md'
    destination = notes_file.resolve()
    if changelog.resolve() == destination:
        raise ChangelogError('notes_overwrite_changelog')
    original = read_bounded(changelog, MAX_CHANGELOG_BYTES)
    if stamp:
        updated, notes = stamp_changelog(original, version, datetime.now(timezone.utc).date())
    else:
        updated, notes = original, extract_notes(original, version)
    # The file written here IS the published body -- the workflow passes it to
    # `gh release create --notes-file` and then requires the remote body to
    # equal it byte for byte. So the bound belongs on the artifact, before its
    # digest is taken, not on the API call: a body shortened anywhere else
    # would fail its own verification.
    notes = bound_release_body(notes, version)
    metadata = _metadata(destination, version, notes)
    if updated != original:
        atomic_write_text(changelog, updated.decode('utf-8'))
    atomic_write_text(destination, notes.decode('utf-8'))
    return metadata


def verify_body(notes_file: Path, body_json: Path) -> bool:
    """Compare decoded GitHub JSON to canonical artifact bytes, without trimming."""
    notes = read_notes(notes_file)
    try:
        value = json.loads(read_bounded(body_json, MAX_CHANGELOG_BYTES))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ChangelogError('invalid_body_json') from exc
    if not isinstance(value, dict) or set(value) != {'body'} or not isinstance(value['body'], str):
        raise ChangelogError('invalid_body_json')
    try:
        body = value['body'].encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ChangelogError('invalid_body_utf8') from exc
    return body == notes
