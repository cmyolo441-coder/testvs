"""Closed metadata for the injected last-mile consumer; never browser execution."""
from __future__ import annotations

import re

from .browser_adapter import BrowserContractError, digest, number, text

SCHEMA = 'browser_effect_attempt/v1'
OPERATIONS = frozenset({'submit', 'send', 'publish', 'upload', 'purchase', 'payment',
                        'credential_change', 'destructive'})
CLAIM_BOUNDARY = 'One approved browser attempt only; not delivery without readback, or review, CI, or merge evidence.'


def classify(operation, *, enabled=True):
    """Only the adapter's inert read is read-only; GET/click labels prove nothing."""
    if not enabled:
        return 'disabled'
    if operation == 'read':
        return 'read_only'
    if isinstance(operation, str) and operation in OPERATIONS:
        return 'sensitive' if operation in {'payment', 'purchase', 'credential_change'} else 'mutating'
    return 'blocked'


def hex_ref(value) -> str:
    if not isinstance(value, str) or not re.fullmatch('[a-f0-9]{64}', value):
        raise BrowserContractError('invalid_digest')
    return value


def effect_request(request):
    fields = {'lease_id', 'tab_id', 'revision', 'handle', 'operation', 'trace_revision', 'expected_postcondition'}
    if not isinstance(request, dict) or set(request) != fields:
        raise BrowserContractError('invalid_effect_request')
    if classify(request['operation']) not in {'sensitive', 'mutating'}:
        raise BrowserContractError('unclassified_action')
    if request['expected_postcondition'] != 'confirmation':
        raise BrowserContractError('unsupported_postcondition')
    hex_ref(request['lease_id'])
    hex_ref(request['handle'])
    text(request['tab_id'], 128)
    number(request['revision'], 2**53)
    if request['trace_revision'] is not None:
        hex_ref(request['trace_revision'])
    return dict(request)


def intent_for(lease, page, element, request, pending):
    if pending['payload_shape'] == 'opaque':
        raise BrowserContractError('opaque_payload')
    if (pending['origin'] != page['observed_origin'] or pending['target_role'] != element['role']
            or pending['target_name_digest'] != element['name_digest']):
        raise BrowserContractError('preview_binding_changed')
    return {'schema_version': SCHEMA, 'owner_ref': lease['owner_ref'],
            'lease_id': lease['lease_id'], 'tab_ref': digest(page['tab_id']),
            'revision': page['revision'], 'host_revision': page['host_revision'],
            'state_digest': page['fingerprint'], 'url_digest': page['observed_url_digest'],
            'handle': element['handle'], 'target_key_digest': element['key_digest'],
            'operation': request['operation'], 'trace_revision': request['trace_revision'],
            'expected_postcondition': request['expected_postcondition'],
            'adapter_ref': digest([lease['adapter_id'], lease['adapter_version']]),
            'pending': pending, 'privacy': 'metadata_only', 'claim_boundary': CLAIM_BOUNDARY}


def approval_scope(intent_digest, intent):
    """Existing approval vocabulary; tool scope is the exact browser intent hash."""
    return dict(approved_action='external_posting', scope_class='tool',
                scope_ref=intent_digest, owner=intent['owner_ref'],
                run_id=intent['lease_id'], safety_profile_revision=intent['state_digest'])


def observed_confirmation(raw, intent, attempt_id):
    """Readback is separate from resume's returned tool text, closed and bounded."""
    fields = {'schema_version', 'observed', 'attempt_id', 'preview_ref', 'postcondition', 'object_digest'}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise BrowserContractError('readback_unknown')
    if (raw['schema_version'] != 'browser_effect_readback/v1' or raw['observed'] is not True
            or raw['attempt_id'] != attempt_id or raw['preview_ref'] != intent['pending']['preview_ref']
            or raw['postcondition'] != intent['expected_postcondition']):
        raise BrowserContractError('readback_unknown')
    return hex_ref(raw['object_digest'])
