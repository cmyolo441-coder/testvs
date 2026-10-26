"""Bounded native continuation dossiers, not an executor or approval gate.

A checkpoint freezes the declaring session's accepted todo scope. Later sessions
can recall it; result rows are append-only, attributed MODEL declarations. No
input can mint a host observation or upgrade a receipt reference into proof.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time

from . import runtime_paths
from ._governance_safety import contains_credential_like_material
from .awareness_delivery import _awareness_delivery_lock, _write_delivery_record
from .project_identity import project_identity_root
from .todo_store import (
    TODO_DONE_BINDING_KEYS, TODO_SCHEMA_VERSION, TODO_STALE_SECONDS, MAX_TODO_RECORD_BYTES,
    TodoStoreError, _reject_symlink_ancestry, todo_path, validate_todo_items,
)

SCHEMA = 'native_completion/v1'
MAX_BYTES = 524288
MAX_CHECKPOINTS = 32
MAX_RESULTS = 64
STALE_SECONDS = 30 * 86400
KINDS = ('verification', 'review', 'qa')
READ_SCHEMA = 'native_completion_read/v1'
STORE_STATES = ('absent', 'empty', 'present')
SOURCE_STATES = ('absent', 'stale', 'declared_no_findings', 'declared_findings')
FRESHNESS_UNBOUND = 'unbound'
CLAIM_BOUNDARY = (
    'Scope, acceptance, verdicts, findings and QA are model declarations, not permission '
    'or observed execution. Claimed host exit, independent review and CI remain distinct '
    'unverified provenance. Receipt references do not prove freshness or validity. '
    'Read original evidence before claiming verified completion, review, CI or merge.'
)
_ID = re.compile(r'[0-9a-f]{32}')


class CompletionValidationError(ValueError):
    """An actionable static refusal, containing no caller or filesystem data."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _text(value, *, maximum=200, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise CompletionValidationError('Expected a bounded nonblank metadata summary')
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise CompletionValidationError('Metadata summaries must not contain control characters or transcripts')
    return '[redacted]' if contains_credential_like_material(value) else value


def _binding_text(value):
    safe = _text(value, maximum=128)
    if safe != value or safe == '[redacted]':
        raise CompletionValidationError('Fingerprints and identifiers must be non-secret metadata; they are never normalized')
    return safe


def _texts(value, maximum=20):
    if not isinstance(value, list) or len(value) > maximum:
        raise CompletionValidationError('Metadata list exceeds its bound or is not a list')
    return [_text(v) for v in value]


def _identity():
    cwd = runtime_paths.runtime_cwd()
    if cwd is None:
        raise CompletionValidationError('A host-bound logical project is required')
    root = project_identity_root(cwd) or cwd
    return {'profile': _digest(str(runtime_paths.default_hermes_home())),
            'project': _digest(str(root.resolve()))}


def _path(home: Path) -> Path:
    path = home / 'runtime' / 'completion' / 'records.json'
    _reject_symlink_ancestry(path, root=home)
    _reject_symlink_ancestry(path.with_name('.records.json.lock'), root=home)
    return path


def _read_metadata(path, maximum):
    # Reject special files before open on every platform, then recheck the opened
    # descriptor. O_NONBLOCK prevents a swapped FIFO from holding the global lock.
    if not stat.S_ISREG(path.lstat().st_mode):
        raise CompletionValidationError('Metadata must be a regular file')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise CompletionValidationError('Malformed or oversized metadata')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(maximum + 1)
        if len(raw) > maximum:
            raise CompletionValidationError('Metadata exceeds its byte limit')
        return json.loads(raw)
    finally:
        os.close(fd)


def _strict_items(value):
    # The writer normalizes user input. Persisted scope must already be canonical:
    # never stringify objects, invent missing states or drop unknown constraints.
    if not isinstance(value, list):
        raise CompletionValidationError('Malformed persisted scope')
    for item in value:
        if (not isinstance(item, dict) or not {'text', 'state'} <= set(item)
                or set(item) - {'text', 'state', 'phase', 'depth', 'blocked_reason'}
                or any(not isinstance(v, str) for k, v in item.items() if k != 'depth')):
            raise CompletionValidationError('Malformed persisted scope')
    validated = validate_todo_items(value)
    if validated != value:
        raise CompletionValidationError('Persisted scope must not require normalization')
    return validated


def _read(path):
    try:
        data = _read_metadata(path, MAX_BYTES)
    except FileNotFoundError:
        return {'schema_version': SCHEMA, 'checkpoints': []}
    if (not isinstance(data, dict) or set(data) != {'schema_version', 'checkpoints'}
            or data.get('schema_version') != SCHEMA
            or not isinstance(data.get('checkpoints'), list)
            or len(data['checkpoints']) > MAX_CHECKPOINTS):
        raise CompletionValidationError('Malformed completion store')
    for checkpoint in data['checkpoints']:
        _validate_checkpoint(checkpoint)
        checkpoint['title'] = _text(checkpoint['title'], maximum=80, empty=True)
        for field in ('revision', 'environment'):
            checkpoint[field] = _binding_text(checkpoint[field])
        checkpoint['rejected'] = _texts(checkpoint['rejected'])
        checkpoint['items'] = [{k: _text(v, empty=True) if isinstance(v, str) else v
                                for k, v in i.items()} for i in checkpoint['items']]
        for row in checkpoint['results']:
            for field in ('revision', 'environment'):
                row[field] = _binding_text(row[field])
            row.update(_result({k: row[k] for k in RESULT_FIELDS}, len(checkpoint['items'])))
    return data


def _validate_checkpoint(c):
    if not isinstance(c, dict) or set(c) != {
        'checkpoint_id', 'profile', 'project', 'author', 'created_at', 'revision',
        'environment', 'items', 'rejected', 'results', 'acceptance', 'title',
    }:
        raise CompletionValidationError('Malformed checkpoint')
    if not isinstance(c['checkpoint_id'], str) or not _ID.fullmatch(c['checkpoint_id']):
        raise CompletionValidationError('Malformed checkpoint identity')
    for key in ('profile', 'project', 'author'):
        if not isinstance(c[key], str) or not re.fullmatch(r'[0-9a-f]{64}', c[key]):
            raise CompletionValidationError('Malformed binding')
    _timestamp(c['created_at'])
    for key in ('revision', 'environment'):
        _binding_text(c[key])
    _text(c['title'], maximum=80, empty=True)
    if c['acceptance'] != 'model_declared_accepted':
        raise CompletionValidationError('Malformed acceptance standing')
    c['items'] = _strict_items(c['items'])
    _texts(c['rejected'])
    if not isinstance(c['results'], list) or len(c['results']) > MAX_RESULTS:
        raise CompletionValidationError('Malformed result list')
    for row in c['results']:
        if not isinstance(row, dict) or set(row) != {
            'kind', 'item', 'verdict', 'summary', 'findings', 'claimed_source',
            'claimed_evidence_state', 'references', 'revision', 'environment',
            'author', 'created_at', 'standing', 'observed',
        }:
            raise CompletionValidationError('Malformed result')
        if row['standing'] != 'model_declaration' or row['observed'] is not False:
            raise CompletionValidationError('Invalid result standing')
        _result({k: row[k] for k in RESULT_FIELDS}, len(c['items']))
        for key in ('revision', 'environment'):
            _binding_text(row[key])
        if not isinstance(row['author'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['author']):
            raise CompletionValidationError('Malformed result author')
        _timestamp(row['created_at'])


def _timestamp(value):
    if type(value) not in (int, float) or not 0 < value <= time.time() + 60:
        raise CompletionValidationError('Malformed timestamp')


RESULT_FIELDS = ('kind', 'item', 'verdict', 'summary', 'findings', 'claimed_source',
                 'claimed_evidence_state', 'references')


def _result(value, count):
    if not isinstance(value, dict) or set(value) != set(RESULT_FIELDS):
        raise CompletionValidationError('Result requires exactly the documented metadata fields')
    if value['kind'] not in KINDS or value['verdict'] not in ('PASS', 'HOLD', 'BLOCK'):
        raise CompletionValidationError('Unknown result kind or verdict')
    if type(value['item']) is not int or not 1 <= value['item'] <= count:
        raise CompletionValidationError('Result item is outside the frozen accepted scope')
    if value['claimed_source'] not in ('model', 'host_exit', 'independent_review', 'ci'):
        raise CompletionValidationError('Unknown claimed source')
    if value['claimed_evidence_state'] not in ('prepared_not_observed', 'observed'):
        raise CompletionValidationError('Unknown claimed evidence state')
    refs = value['references']
    if not isinstance(refs, list) or len(refs) > 8:
        raise CompletionValidationError('At most eight references are allowed')
    safe_refs = []
    for ref in refs:
        if (not isinstance(ref, dict) or set(ref) != {'type', 'id'}
                or ref['type'] not in ('verification_receipt/v1', 'artifact', 'ci_run')):
            raise CompletionValidationError('Invalid evidence reference')
        ref_id = _text(ref['id'], maximum=128)
        if ref['type'] == 'verification_receipt/v1' and not re.fullmatch(r'[0-9a-f]{64}', ref['id']):
            raise CompletionValidationError('Receipt key must be exactly 64 lowercase hexadecimal characters')
        if ref['type'] != 'verification_receipt/v1':
            ref_id = _binding_text(ref['id'])
        safe_refs.append({'type': ref['type'], 'id': ref['id'] if ref['type'] == 'verification_receipt/v1' else ref_id})
    return {**value, 'summary': _text(value['summary']),
            'findings': _texts(value['findings']), 'references': safe_refs}


def _checkpoint(home, session, args, binding):
    if args.get('accepted') is not True:
        raise CompletionValidationError('Checkpoint only after the person accepts this exact scope')
    path = todo_path(home, session)
    _reject_symlink_ancestry(path, root=home)
    todo = _read_metadata(path, MAX_TODO_RECORD_BYTES)
    if (not isinstance(todo, dict) or todo.get('schema_version') != TODO_SCHEMA_VERSION
            or todo.get('session_ref') != session
            or time.time() - path.stat().st_mtime > TODO_STALE_SECONDS):
        raise CompletionValidationError('A fresh, session-owned todo record is required')
    updated = datetime.fromisoformat(_text(todo.get('updated_at'), maximum=64)).timestamp()
    if not 0 <= time.time() - updated <= TODO_STALE_SECONDS:
        raise CompletionValidationError('A fresh todo timestamp is required')
    if todo.get('plan_stage') not in (None, '', 'accepted') or todo.get('deferred_reason'):
        raise CompletionValidationError('The plan is awaiting acceptance or deferred; do not resume it implicitly')
    # The done-item bindings OMH stamps (`todo_evidence`) say what closed an
    # item, which is evidence about the run, not part of the accepted scope, so
    # a checkpoint freezes the declaration without them.
    raw_items = todo.get('items')
    if isinstance(raw_items, list):
        raw_items = [{k: v for k, v in item.items() if k not in TODO_DONE_BINDING_KEYS}
                     if isinstance(item, dict) else item for item in raw_items]
    items = _strict_items(raw_items)
    # Copy only the todo contract, never unknown keys or a raw conversation.
    items = [{k: _text(v, empty=True) if isinstance(v, str) else v
              for k, v in item.items()} for item in items]
    return {'checkpoint_id': secrets.token_hex(16), **binding,
            'author': _digest(session), 'created_at': time.time(),
            'revision': _binding_text(args.get('revision')),
            'environment': _binding_text(args.get('environment')),
            'acceptance': 'model_declared_accepted',
            'title': _text(todo.get('title', ''), maximum=80, empty=True),
            'items': items, 'rejected': _texts(args.get('rejected', [])), 'results': []}


def _freshness(record, revision, environment):
    # Bound: a checkpoint or result row is current only for the exact revision
    # and environment it was declared against, and only inside the age cap. A
    # verdict is attached to a revision, not to a calendar: the revision moving
    # makes it stale at once, and the cap only bounds a dossier nobody moves.
    # Unbound (`revision is None`; the reader stated no binding): nothing can be
    # called current or stale, so it is neither, and no completion judgment is
    # made. The tool never reaches here unbound; its actions validate the
    # binding first.
    if revision is None:
        return FRESHNESS_UNBOUND
    current = (record['revision'] == revision and record['environment'] == environment
               and time.time() - record['created_at'] <= STALE_SECONDS)
    return 'current' if current else 'stale'


def _projection(c, revision, environment):
    rows = [{**row, 'freshness': _freshness(row, revision, environment)} for row in c['results']]
    considered = [r for r in rows if r['freshness'] != 'stale']
    latest = {(r['item'], r['kind']): r for r in considered}
    sources = {}
    for kind in KINDS:
        relevant = [r for r in latest.values() if r['kind'] == kind]
        sources[kind] = ('declared_findings' if any(r['findings'] for r in relevant)
                         else 'declared_no_findings') if relevant else (
                             'stale' if any(r['kind'] == kind for r in rows) else 'absent')
    projection = {'checkpoint': {**c, 'results': rows}, 'sources': sources}
    if revision is None:
        return projection
    missing = [i for i in range(1, len(c['items']) + 1)
               if (i, 'verification') not in latest]
    blockers = sorted({r['item'] for r in latest.values() if r['verdict'] != 'PASS' or r['findings']})
    projection['completion'] = {'status': 'not_verified', 'missing': missing, 'blockers': blockers,
                                'declarations_complete': not missing and not blockers}
    return projection


def read_completion_dossiers(home, *, session_ref='', checkpoint_id='', revision=None, environment=None):
    """Read every dossier in one OMH home for an operator or agent; never a write.

    The tool's `recall` is bound by the host to one profile and project. This
    reader takes the home the caller named and returns each checkpoint with its
    profile and project digests as stored, so a later session, a wrapper or a
    person can read what an earlier session declared. `store_state` separates a
    store that does not exist from one with no checkpoints from one with some;
    `sources` per kind separates absent from stale from declared-empty from
    declared-findings. Rows keep `standing=model_declaration` and
    `observed=false`, and the payload itself says the same at the top level on
    every status, read or malformed: the read adds freshness and source states,
    never evidence.
    """
    base = {'schema_version': READ_SCHEMA, 'standing': 'model_declaration',
            'claim_boundary': CLAIM_BOUNDARY}
    if (revision is None) != (environment is None):
        raise CompletionValidationError('revision and environment bind together; pass both or neither')
    if revision is not None:
        revision = _binding_text(revision)
        environment = _binding_text(environment)
    if checkpoint_id and (not isinstance(checkpoint_id, str) or not _ID.fullmatch(checkpoint_id)):
        raise CompletionValidationError('Invalid checkpoint id')
    try:
        path = _path(home)
        existed = path.exists()
        data = _read(path)
    except (ValueError, OSError, TodoStoreError, RecursionError) as error:
        # Same shape as the tool's recall: a store that cannot be read is
        # `malformed`, never an empty one, and raw stored content stays inside.
        return {**base, 'status': 'malformed',
                'error': str(error) if isinstance(error, CompletionValidationError) else 'Completion metadata unavailable or invalid',
                'error_type': type(error).__name__}
    checkpoints = data['checkpoints']
    author = _digest(session_ref) if session_ref else ''
    dossiers = []
    for c in checkpoints:
        if checkpoint_id and c['checkpoint_id'] != checkpoint_id:
            continue
        if author and c['author'] != author and all(r['author'] != author for r in c['results']):
            continue
        dossiers.append({'freshness': _freshness(c, revision, environment),
                         **_projection(c, revision, environment)})
    return {**base, 'status': 'read',
            'store_state': 'absent' if not existed else ('empty' if not checkpoints else 'present'),
            'path': str(path), 'session_ref': session_ref,
            'binding': None if revision is None else {'revision': revision, 'environment': environment},
            'checkpoint_count': len(checkpoints), 'dossiers': dossiers}


def completion_action(args, *, session):
    """Model-selected actions on the existing native todo tool, never router I/O."""
    action = args['action']
    base = {'schema_version': SCHEMA, 'claim_boundary': CLAIM_BOUNDARY}
    if any(k in args for k in ('omh_home', 'hermes_home', 'project_root', 'session_ref')):
        return {**base, 'status': 'invalid_completion', 'error': 'Runtime binding overrides are forbidden'}
    try:
        if not session:
            raise CompletionValidationError('A host-attributed session is required')
        binding = _identity()
        home = runtime_paths.default_omh_home()
        path = _path(home)
        if action == 'recall':
            data = _read(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _path(home)
            with _awareness_delivery_lock(path, timeout_seconds=2) as mechanism:
                if mechanism == 'none':
                    raise CompletionValidationError('Completion store requires a supported lock backend')
                data = _read(path)
                if action == 'checkpoint':
                    if len(data['checkpoints']) >= MAX_CHECKPOINTS:
                        raise CompletionValidationError('Completion checkpoint capacity reached; archive offline before adding more')
                    c = _checkpoint(home, session, args, binding)
                    data['checkpoints'].append(c)
                else:
                    c = _select(data, args.get('checkpoint_id'), binding)
                    if c is None:
                        return {**base, 'status': 'absent'}
                    if len(c['results']) >= MAX_RESULTS:
                        raise CompletionValidationError('Checkpoint result capacity reached; preserve this dossier and create a new one')
                    row = _result(args.get('result'), len(c['items']))
                    c['results'].append({**row, 'revision': _binding_text(args.get('revision')),
                                         'environment': _binding_text(args.get('environment')),
                                         'author': _digest(session), 'created_at': time.time(),
                                         'standing': 'model_declaration', 'observed': False})
                if len(json.dumps(data).encode('utf-8')) + 1 > MAX_BYTES:
                    raise CompletionValidationError('Completion store byte capacity reached')
                _path(home)
                _write_delivery_record(path, data, compact=True)
            # Read back the exact target rather than echoing the write payload.
            c = _select(_read(path), c['checkpoint_id'], binding)
            return {**base, 'status': 'written', **_projection(c, args.get('revision'), args.get('environment'))}
        if not args.get('checkpoint_id'):
            return {**base, 'status': 'index', 'checkpoints': [
                {k: c[k] for k in ('checkpoint_id', 'title', 'created_at', 'revision', 'environment')}
                for c in data['checkpoints'] if all(c[k] == v for k, v in binding.items())]}
        c = _select(data, args['checkpoint_id'], binding)
        if c is None:
            return {**base, 'status': 'absent'}
        revision = _binding_text(args.get('revision'))
        environment = _binding_text(args.get('environment'))
        status = 'current' if (c['revision'] == revision and c['environment'] == environment
                               and time.time() - c['created_at'] <= STALE_SECONDS) else 'stale'
        return {**base, 'status': status, **_projection(c, revision, environment)}
    except (ValueError, OSError, TodoStoreError, RecursionError) as error:
        # Never leak filesystem paths or raw malformed stored content.
        return {**base, 'status': 'invalid_completion' if action != 'recall' else 'malformed',
                'error': str(error) if isinstance(error, CompletionValidationError) else 'Completion metadata unavailable or invalid',
                'error_type': type(error).__name__}


def _select(data, key, binding):
    if not isinstance(key, str) or not _ID.fullmatch(key):
        raise CompletionValidationError('Invalid checkpoint id')
    return next((c for c in data['checkpoints'] if c['checkpoint_id'] == key
                 and all(c[k] == v for k, v in binding.items())), None)
