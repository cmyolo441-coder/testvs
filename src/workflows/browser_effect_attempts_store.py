"""Bounded indexed browser bindings over the unchanged generic attempt journal."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .browser_adapter import BrowserContractError

MAX_BINDINGS = 256
MAX_ROW_BYTES = 4096


class EffectBindingStore:
    def __init__(self, home):
        self.path = Path(home) / 'runtime' / 'browser' / 'effects.sqlite3'

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        connection = sqlite3.connect(self.path, timeout=2, isolation_level=None)
        try:
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute('CREATE TABLE IF NOT EXISTS bindings (digest TEXT PRIMARY KEY, '
                               'preview_ref TEXT UNIQUE NOT NULL, intent TEXT NOT NULL, '
                               'request TEXT NOT NULL, result TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS events (event_ref TEXT PRIMARY KEY, '
                               'digest TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS receipts (attempt_id TEXT PRIMARY KEY, '
                               'receipt TEXT NOT NULL)')
            connection.execute('BEGIN IMMEDIATE')
            yield connection
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def encode(value):
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'))
        if len(encoded.encode()) > MAX_ROW_BYTES:
            raise BrowserContractError('effect_metadata_capped')
        return encoded

    def prepare(self, key, intent, request):
        encoded = self.encode(intent)
        # Tab ids are only transient request data. Persist the digest, never raw host ids.
        safe_request = {**request, 'tab_id': intent['tab_ref']}
        with self.transaction() as db:
            prior = db.execute('SELECT intent FROM bindings WHERE digest=?', (key,)).fetchone()
            if prior:
                if prior[0] != encoded:
                    raise BrowserContractError('intent_conflict')
                return
            if db.execute('SELECT count(*) FROM bindings').fetchone()[0] >= MAX_BINDINGS:
                raise BrowserContractError('effect_metadata_capped')
            try:
                db.execute('INSERT INTO bindings VALUES (?, ?, ?, ?, ?)',
                           (key, intent['pending']['preview_ref'], encoded,
                            self.encode(safe_request), '{}'))
            except sqlite3.IntegrityError as exc:
                raise BrowserContractError('preview_reused') from exc

    @staticmethod
    def get(db, key):
        row = db.execute('SELECT intent, request, result FROM bindings WHERE digest=?', (key,)).fetchone()
        if row is None:
            raise BrowserContractError('unknown_intent')
        return tuple(json.loads(value) for value in row)

    def result(self, key):
        with self.transaction() as db:
            return self.get(db, key)[2]

    def bind_event(self, db, event_ref, key):
        prior = db.execute('SELECT digest FROM events WHERE event_ref=?', (event_ref,)).fetchone()
        if prior:
            if prior[0] != key:
                raise BrowserContractError('event_binding_changed')
            return
        # Refuse exhaustion instead of evicting deduplication history.
        if db.execute('SELECT count(*) FROM events').fetchone()[0] >= MAX_BINDINGS:
            raise BrowserContractError('effect_metadata_capped')
        db.execute('INSERT INTO events VALUES (?, ?)', (event_ref, key))

    def finish(self, key, result, receipt=None):
        with self.transaction() as db:
            if receipt:
                db.execute('INSERT INTO receipts VALUES (?, ?)',
                           (result['attempt_id'], self.encode(receipt)))
            db.execute('UPDATE bindings SET result=? WHERE digest=?', (self.encode(result), key))

    def receipt(self, attempt_id):
        with self.transaction() as db:
            row = db.execute('SELECT receipt FROM receipts WHERE attempt_id=?', (attempt_id,)).fetchone()
            return json.loads(row[0]) if row else None
