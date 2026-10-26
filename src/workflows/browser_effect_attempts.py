"""Host-injected last-mile orchestration. Core never drives a browser or network.

lease_source(owner, lease_id) returns the current verified #1391 lease view.
approval_source(intent_digest) returns its current host-owned approval head,
not model arguments. Both lookups must be indexed/bounded. Host callbacks own
deadlines and atomic live-state rechecks. Exceptions never authorize a retry.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import time

from ..plugin_bundle.omh.egress_attempt_receipts import AttemptStore
from ..system.paths import OmhPaths
from .approval_receipts import approval_satisfies_request_in, validate_approval_receipt
from .browser_adapter import (BrowserContractError, digest, page_state, pending_effect,
                              resolve_handle, text, timestamp)
from .browser_effect_attempts_contract import (approval_scope, classify, effect_request,
                                             hex_ref, intent_for, observed_confirmation)
from .browser_effect_attempts_store import EffectBindingStore
from .external_effect_receipts import append_external_effect_receipt, build_external_effect_receipt


class BrowserEffectEngine:
    def __init__(self, home, adapter, lease_source, approval_source, *, clock=time.time, enabled=True,
                 reserve_action=None):
        self.adapter, self.lease_source, self.approval_source = adapter, lease_source, approval_source
        self.reserve_action = reserve_action
        self.clock, self.enabled = clock, enabled
        self.store, self.attempts = EffectBindingStore(home), AttemptStore(home)
        self.paths = OmhPaths(Path(home), Path(home))

    def _lease(self, owner, lease_id):
        lease = self.lease_source(owner, lease_id)
        now = timestamp(self.clock())
        if lease['lease_id'] != lease_id or lease['owner_ref'] != digest(text(owner, 256)):
            raise BrowserContractError('foreign_lease')
        if lease['adapter_id'] != self.adapter.adapter_id or lease['adapter_version'] != self.adapter.adapter_version:
            raise BrowserContractError('foreign_adapter')
        if lease['status'] != 'active' or now >= min(lease['expires_at'], lease['capabilities']['expires_at']):
            raise BrowserContractError('lease_unavailable')
        if lease['capabilities']['mutation_interception'] != 'last_mile':
            raise BrowserContractError('adapter_cannot_intercept')
        return lease

    def _live_intent(self, lease, request):
        tab = request['tab_id']
        if tab not in lease['tabs']:
            raise BrowserContractError('foreign_tab')
        needed = 'upload' if request['operation'] == 'upload' else 'click'
        if needed not in lease['actions']:
            raise BrowserContractError('action_out_of_scope')
        page = lease['pages'][tab]
        element = resolve_handle(page, request)
        raw = self.adapter.observe(lease['lease_id'], tab, lease['expires_at'])
        fresh = page_state(raw, lease, tab, page)
        if fresh['fingerprint'] != page['fingerprint']:
            raise BrowserContractError('stale_state')
        pending = pending_effect(self.adapter.preview(lease['lease_id'], element['handle'], request['operation']))
        return intent_for(lease, page, element, request, pending)

    def preview(self, owner, request):
        mode = classify(request.get('operation'), enabled=self.enabled)
        if mode in {'disabled', 'read_only'}:
            return {'status': mode}  # No store, lease lookup, or host callback.
        request = effect_request(request)
        lease = self._lease(owner, request['lease_id'])
        intent = self._live_intent(lease, request)
        key = digest(intent)
        self.store.prepare(key, intent, request)
        return {'status': 'awaiting_approval', 'intent_digest': key, 'intent': deepcopy(intent)}

    def _approve(self, owner, lease, key, intent):
        receipt = self.approval_source(key)
        # Approval lookup may outlive or revoke the lease/context we observed.
        if self._lease(owner, intent['lease_id']) != lease:
            raise BrowserContractError('intent_changed')
        if not isinstance(receipt, dict) or validate_approval_receipt(receipt):
            raise BrowserContractError('approval_absent')
        now = timestamp(self.clock())
        if now >= min(lease['expires_at'], lease['capabilities']['expires_at']):
            raise BrowserContractError('lease_unavailable')
        verdict = approval_satisfies_request_in([receipt], **approval_scope(key, intent),
            now=datetime.fromtimestamp(now, timezone.utc).isoformat())
        if verdict['satisfied'] is not True:
            raise BrowserContractError(verdict['reason_code'])
        return digest(receipt)

    def execute(self, owner, intent_digest, event_id):
        if not self.enabled:
            return {'status': 'disabled'}
        key = hex_ref(intent_digest)
        owner_ref = digest(text(owner, 256))
        event_ref = digest([owner_ref, text(event_id, 256)])
        with self.store.transaction() as db:
            intent, request, result = self.store.get(db, key)
            if intent['owner_ref'] != owner_ref:
                raise BrowserContractError('foreign_lease')
            self.store.bind_event(db, event_ref, key)
            if result:
                return result  # Replays never reobserve, reapprove, or resend.
            lease = self._lease(owner, intent['lease_id'])
            tabs = [tab for tab in lease['tabs'] if digest(tab) == intent['tab_ref']]
            if len(tabs) != 1:
                raise BrowserContractError('foreign_tab')
            request['tab_id'] = tabs[0]
            if self._live_intent(lease, request) != intent:
                raise BrowserContractError('intent_changed')
            try:
                approval_ref = self._approve(owner, lease, key, intent)
            except BrowserContractError as exc:
                if str(exc) in {'approval_expired', 'lease_unavailable', 'intent_changed'}:
                    self.adapter.abort(intent['lease_id'], intent['pending']['preview_ref'])
                raise
            attempt = self.attempts.open_attempt(
                session_id=owner_ref, tool_call_id=key, tool_name='browser_effect',
                action_class='external_write', destination_class='endpoint',
                request_fingerprint=key, effect_id=key,
                destination_digest=digest(intent['pending']['origin']),
                payload_digest=intent['pending']['payload_digest'],
                payload_bytes=intent['pending']['payload_bytes'], approval_ref=approval_ref)
            if attempt['disposition'] == 'created' and self.reserve_action is not None:
                lease = self.reserve_action(owner, request)
            result = {'status': 'unknown', 'attempt_id': attempt['attempt_id'],
                      'intent_digest': key, 'receipt_id': '', 'retry_allowed': False}
            db.execute('UPDATE bindings SET result=? WHERE digest=?', (self.store.encode(result), key))
        if attempt['disposition'] != 'created':
            return result
        # FULL generic commit AND browser reservation commit precede resume.
        return self._resume(owner, lease, request, intent, result, approval_ref)

    def _resume(self, owner, lease, request, intent, result, approval_ref):
        attempt_id, key = result['attempt_id'], result['intent_digest']
        terminal, receipt = 'unknown', None
        try:
            try:
                # Neither durable write may extend authorization or replace its head.
                if self._approve(owner, lease, key, intent) != approval_ref:
                    raise BrowserContractError('approval_changed')
            except BrowserContractError:
                self.adapter.abort(intent['lease_id'], intent['pending']['preview_ref'])
                raise
            # Host atomically rechecks live state/target/bytes and permits ONE exact request.
            # Return text is deliberately discarded and cannot establish delivery.
            self.adapter.resume(lease['lease_id'], intent['pending']['preview_ref'], attempt_id)
            raw = self.adapter.observe(lease['lease_id'], request['tab_id'], lease['expires_at'])
            page = page_state(raw, lease, request['tab_id'], lease['pages'][request['tab_id']])
            if self.clock() >= min(lease['expires_at'], lease['capabilities']['expires_at']):
                raise BrowserContractError('readback_unknown')
            object_ref = observed_confirmation(raw.get('effect_readback'), intent, attempt_id)
            receipt = build_external_effect_receipt(
                effect_id=key, action='browser_mutation', acting_surface='browser_adapter_readback',
                observed_result='succeeded', external_ref=object_ref,
                evidence_refs=[key, digest(attempt_id), hex_ref(page['fingerprint'])],
                observed_at=datetime.fromtimestamp(self.clock(), timezone.utc).isoformat().replace('+00:00', 'Z'))
            # Append directly: never scan legacy JSONL history during reconciliation.
            append_external_effect_receipt(self.paths, receipt)
            result = {**result, 'status': 'succeeded', 'receipt_id': receipt['receipt_id']}
            terminal = 'returned'
        except (OSError, RuntimeError, ValueError, TypeError, KeyError):
            # Closed unknown: host errors may contain secrets and are never persisted.
            result = {**result, 'status': 'unknown', 'receipt_id': ''}
            receipt = None
        finally:
            self.attempts.record_terminal(attempt_id, terminal)
            self.store.finish(key, result, receipt)
        return result

    def abort(self, owner, intent_digest):
        if not self.enabled:
            return {'status': 'disabled'}
        key = hex_ref(intent_digest)
        with self.store.transaction() as db:
            intent, _, result = self.store.get(db, key)
            if intent['owner_ref'] != digest(text(owner, 256)):
                raise BrowserContractError('foreign_lease')
            if result:
                return result
            result = {'status': 'cancelled', 'intent_digest': key, 'retry_allowed': False}
            db.execute('UPDATE bindings SET result=? WHERE digest=?', (self.store.encode(result), key))
        self.adapter.abort(intent['lease_id'], intent['pending']['preview_ref'])
        return result
