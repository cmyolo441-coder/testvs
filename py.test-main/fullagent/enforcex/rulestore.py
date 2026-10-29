"""rulestore — Persistent, versioned JSON store for PromptSpecs.

A RuleStore owns one JSON document on disk:

    {"schema": 2, "specs": {signature: {…PromptSpec…}}, "history": [...]}

Durability properties:

  * ATOMIC WRITES  — the document is written to a tempfile in the same
    directory and moved into place with os.replace (atomic on POSIX and
    on same-volume Windows). A crash can lose the newest write but can
    never leave a torn file.
  * LOCKING        — a cross-process lockfile is taken with
    os.O_CREAT|os.O_EXCL (atomic create) and released with os.unlink.
    Stale locks (from crashed writers) are stolen after a timeout.
  * BOUNDED HISTORY— every append_history trims the list to the last
    500 events; older events are dropped silently.
  * SCHEMA GATE    — documents with an unknown schema version are
    readable in degraded mode (specs preserved) but writes re-stamp
    the current schema.

Versioning:

  * diff_specs(a, b) → DiffResult with added / removed / changed rule
    ids; two rules are "changed" when any enforcement-relevant field
    differs (text, category, severity, priority, keywords, patterns,
    immutable, source_line).
  * to_dict / from_dict round-trips for Rule and PromptSpec come from
    ruleextract (with a graceful local fallback so the store still
    works if ruleextract is broken).

Stdlib only, Python 3.9+, thread-safe via a re-entrant lock, never
raises on malformed on-disk data (read paths fail soft).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field

try:  # graceful import of the extractor module
    from . import ruleextract as _rx
except Exception:  # pragma: no cover - sibling missing in some installs
    _rx = None

__all__ = [
    "RuleStore",
    "DiffResult",
    "diff_specs",
    "StoreError",
    "SCHEMA_VERSION",
    "HISTORY_LIMIT",
    "LOCK_TIMEOUT",
]


# ── constants ────────────────────────────────────────────────────────────

SCHEMA_VERSION = 2
HISTORY_LIMIT = 500
LOCK_TIMEOUT = 5.0        # seconds before a stale lockfile is stolen
LOCK_RETRY_DELAY = 0.05   # seconds between O_EXCL retries
LOCK_STALE_AGE = 30.0     # lockfiles older than this are always stolen
CORRUPT_BACKUP_SUFFIX = ".corrupt"


class StoreError(Exception):
    """Raised on unrecoverable store problems (lock timeout, bad path)."""


# ── spec conversion with graceful fallback ───────────────────────────────

def _spec_to_dict(spec) -> dict:
    if _rx is not None and hasattr(_rx, "PromptSpec"):
        try:
            return spec.to_dict()
        except Exception:
            pass
    # fallback: duck-typed serialization
    return {
        "signature": getattr(spec, "signature", ""),
        "source_sha256": getattr(spec, "source_sha256", ""),
        "rules": [dict(getattr(r, "__dict__", {})) for
                  r in getattr(spec, "rules", [])],
        "created_at": getattr(spec, "created_at", ""),
        "meta": dict(getattr(spec, "meta", {}) or {}),
    }


def _spec_from_dict(d: dict):
    if _rx is not None and hasattr(_rx, "PromptSpec"):
        try:
            return _rx.PromptSpec.from_dict(d)
        except Exception:
            pass
    if _rx is not None and hasattr(_rx, "Rule"):
        rules = [_rx.Rule.from_dict(r) for r in d.get("rules", ())
                 if isinstance(r, dict)]
    else:  # last-resort plain dict rules
        rules = [r for r in d.get("rules", ()) if isinstance(r, dict)]
    # build via kwargs so both the dataclass and fallbacks work
    cls = _rx.PromptSpec if _rx is not None else None
    if cls is None:
        return dict(d)
    try:
        return cls(
            signature=str(d.get("signature", "")),
            source_sha256=str(d.get("source_sha256", "")),
            rules=rules,
            created_at=str(d.get("created_at", "")),
            meta=d.get("meta") if isinstance(d.get("meta"), dict) else {},
        )
    except Exception:
        return dict(d)


def _rules_digest(rules) -> str:
    if _rx is not None and hasattr(_rx, "rules_digest"):
        try:
            return _rx.rules_digest(rules)
        except Exception:
            pass
    import hashlib

    payload = json.dumps(
        [_spec_to_dict_rule(r) for r in (rules or [])],
        sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _spec_to_dict_rule(r):
    if hasattr(r, "to_dict"):
        return r.to_dict()
    return {k: v for k, v in getattr(r, "__dict__", {}).items()
            if not k.startswith("_")}


def _spec_signature(text: str) -> str:
    if _rx is not None and hasattr(_rx, "spec_signature"):
        try:
            return _rx.spec_signature(text)
        except Exception:
            pass
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]


# ── versioning: spec diff ────────────────────────────────────────────────

@dataclass
class DiffResult:
    """Differences between two PromptSpecs.

    `changed` maps rule id → (old Rule, new Rule) pairs so callers can
    render precise upgrade notes.
    """

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: dict = field(default_factory=dict)
    same: bool = True

    def summary(self) -> str:
        if self.same:
            return "specs identical"
        return (f"+{len(self.added)} -{len(self.removed)} "
                f"~{len(self.changed)}")

    def to_dict(self) -> dict:
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": sorted(self.changed),
            "same": self.same,
        }


def _rule_fingerprint(rule) -> dict:
    """Enforcement-relevant fields only — meta noise is ignored."""
    d = _spec_to_dict_rule(rule)
    return {k: d.get(k) for k in (
        "text", "category", "severity", "priority", "keywords",
        "patterns", "immutable", "source_line")}


def diff_specs(a, b) -> DiffResult:
    """Diff two PromptSpecs (or dicts) by rule id."""
    a_rules = {r.id: r for r in getattr(a, "rules", [])
               if getattr(r, "id", "")}
    b_rules = {r.id: r for r in getattr(b, "rules", [])
               if getattr(r, "id", "")}
    out = DiffResult()
    out.added = sorted(set(b_rules) - set(a_rules))
    out.removed = sorted(set(a_rules) - set(b_rules))
    for rid in sorted(set(a_rules) & set(b_rules)):
        if _rule_fingerprint(a_rules[rid]) != _rule_fingerprint(b_rules[rid]):
            out.changed[rid] = (a_rules[rid], b_rules[rid])
    out.same = not out.added and not out.removed and not out.changed
    return out


# ── the store ────────────────────────────────────────────────────────────

class RuleStore:
    """Persistent JSON store of PromptSpecs, keyed by prompt signature."""

    def __init__(self, path, *, lock_timeout: float = LOCK_TIMEOUT):
        self.path = os.fspath(path)
        self.lock_path = self.path + ".lock"
        self.lock_timeout = float(lock_timeout)
        self._lock = threading.RLock()
        self._doc: dict | None = None
        self._loaded = False
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and not os.path.isdir(parent):
            raise StoreError(f"store directory missing: {parent}")

    # ── document handling ────────────────────────────────────────────

    def _empty_doc(self) -> dict:
        return {"schema": SCHEMA_VERSION, "specs": {}, "history": []}

    def _load_doc(self) -> dict:
        """Read the document; malformed JSON → backup + fresh doc."""
        if self._loaded and self._doc is not None:
            return self._doc
        doc = self._empty_doc()
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            raw = None
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            raw = None
            self._quarantine_corrupt()
        if isinstance(raw, dict):
            doc["specs"] = (raw.get("specs")
                            if isinstance(raw.get("specs"), dict) else {})
            doc["history"] = (raw.get("history")
                              if isinstance(raw.get("history"), list)
                              else [])
            doc["schema"] = raw.get("schema", SCHEMA_VERSION)
        self._doc = doc
        self._loaded = True
        return doc

    def _quarantine_corrupt(self) -> None:
        """Keep unreadable bytes aside instead of destroying them."""
        try:
            if os.path.exists(self.path):
                backup = self.path + CORRUPT_BACKUP_SUFFIX
                os.replace(self.path, backup)
        except OSError:
            pass

    def _write_doc(self, doc: dict) -> None:
        """Atomic write: tempfile in same dir → os.replace."""
        doc["schema"] = SCHEMA_VERSION
        doc["history"] = doc["history"][-HISTORY_LIMIT:]
        payload = json.dumps(doc, indent=2, sort_keys=True,
                             ensure_ascii=False)
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=".enforcex-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ── lockfile (cross-process, O_EXCL, stale-stealing) ─────────────

    def _acquire_lock(self) -> None:
        """Create the lockfile atomically; steal it if stale.

        Raises StoreError only after `lock_timeout` of a live
        competitor's lock — in that case the caller must skip writing.
        """
        deadline = time.monotonic() + self.lock_timeout
        while True:
            try:
                fd = os.open(self.lock_path,
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(f"{os.getpid()}\n{time.time()}\n")
                return
            except FileExistsError:
                self._steal_stale_lock()
                if time.monotonic() >= deadline:
                    raise StoreError(
                        f"lock timeout: {self.lock_path} held")
                time.sleep(LOCK_RETRY_DELAY)
            except OSError as exc:
                raise StoreError(f"lock failed: {exc}") from exc

    def _steal_stale_lock(self) -> None:
        try:
            age = time.time() - os.path.getmtime(self.lock_path)
            if age > LOCK_STALE_AGE:
                os.unlink(self.lock_path)
        except OSError:
            pass

    def _release_lock(self) -> None:
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass

    # ── public API ───────────────────────────────────────────────────

    def save_spec(self, spec, *, event: str = "save") -> str:
        """Store a PromptSpec under its signature; returns signature."""
        sig = str(getattr(spec, "signature", "")) or _spec_signature(
            _first_rule_text(spec))
        if not sig:
            raise StoreError("spec has no signature and none derivable")
        with self._lock:
            self._acquire_lock()
            try:
                doc = self._load_doc()
                prev = doc["specs"].get(sig)
                doc["specs"][sig] = _spec_to_dict(spec)
                diff_note = ""
                if prev is not None:
                    a = _spec_from_dict(prev)
                    d = diff_specs(a, spec)
                    diff_note = d.summary()
                self._append_history_locked(doc, {
                    "ts": round(time.time(), 3),
                    "event": event,
                    "signature": sig,
                    "rules": len(getattr(spec, "rules", []) or []),
                    "note": diff_note,
                })
                self._write_doc(doc)
            finally:
                self._release_lock()
        return sig

    def load_spec(self, sig: str):
        """Fetch one spec by signature, or None."""
        with self._lock:
            doc = self._load_doc()
            raw = doc["specs"].get(str(sig))
        return _spec_from_dict(raw) if isinstance(raw, dict) else None

    def latest(self):
        """The most recently saved spec, or None when the store is empty."""
        with self._lock:
            doc = self._load_doc()
            last = next((ev for ev in reversed(doc["history"])
                         if isinstance(ev, dict)
                         and ev.get("event") in ("save", "update")
                         and ev.get("signature") in doc["specs"]), None)
        if last is None:
            # fall back to file mtimes of the stored dicts is impossible
            # (dicts have none); use insertion order of the specs dict.
            with self._lock:
                doc = self._load_doc()
                for raw in reversed(list(doc["specs"].values())):
                    if isinstance(raw, dict):
                        return _spec_from_dict(raw)
            return None
        return self.load_spec(last["signature"])

    def list_specs(self) -> list[str]:
        with self._lock:
            doc = self._load_doc()
            return sorted(str(k) for k in doc["specs"])

    def delete_spec(self, sig: str) -> bool:
        with self._lock:
            self._acquire_lock()
            try:
                doc = self._load_doc()
                if str(sig) not in doc["specs"]:
                    return False
                del doc["specs"][str(sig)]
                self._append_history_locked(doc, {
                    "ts": round(time.time(), 3),
                    "event": "delete",
                    "signature": str(sig),
                })
                self._write_doc(doc)
            finally:
                self._release_lock()
        return True

    def append_history(self, event: dict) -> None:
        """Append one event dict (bounded to the last HISTORY_LIMIT)."""
        with self._lock:
            self._acquire_lock()
            try:
                doc = self._load_doc()
                self._append_history_locked(doc, event)
                self._write_doc(doc)
            finally:
                self._release_lock()

    def iter_history(self, limit: int = 0):
        """Yield history events newest-last; `limit` > 0 trims the tail."""
        with self._lock:
            doc = self._load_doc()
            events = list(doc["history"])
        if limit and limit > 0:
            events = events[-limit:]
        for ev in events:
            if isinstance(ev, dict):
                yield ev

    def stats(self) -> dict:
        with self._lock:
            doc = self._load_doc()
            return {
                "schema": doc.get("schema"),
                "specs": len(doc["specs"]),
                "history": len(doc["history"]),
                "path": self.path,
            }

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _append_history_locked(doc: dict, event: dict) -> None:
        if not isinstance(event, dict):
            event = {"event": str(event)}
        event.setdefault("ts", round(time.time(), 3))
        doc["history"].append(event)
        del doc["history"][:-HISTORY_LIMIT]

    # context-manager sugar: exclusive batch updates
    def __enter__(self) -> "RuleStore":
        self._lock.acquire()
        self._acquire_lock()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if exc_type is None:
                self._write_doc(self._load_doc())
        finally:
            self._release_lock()
            self._lock.release()


def _first_rule_text(spec) -> str:
    rules = getattr(spec, "rules", None) or []
    for r in rules:
        t = getattr(r, "text", "")
        if t:
            return t
    return getattr(spec, "source_sha256", "") or "empty"


# ── self-test ────────────────────────────────────────────────────────────

_PROMPT_V1 = """# RULES:
1. Always answer in Hindi.
2. Never use bullet lists unless asked.
You must cite file paths when editing code.
"""

_PROMPT_V2 = """# RULES:
1. Always answer in Hindi.
2. Never use bullet lists unless asked.
3. Keep every reply under 150 words.
You must cite file paths when editing code.
"""


def _self_test() -> int:
    import shutil

    if _rx is None:
        print("RULESTORE SELF-TEST SKIP (ruleextract unavailable)")
        return 0

    tmp = tempfile.mkdtemp(prefix="enforcex-store-")
    try:
        path = os.path.join(tmp, "rules.json")
        store = RuleStore(path)

        # empty store reads soft
        assert store.list_specs() == []
        assert store.load_spec("nope") is None
        assert store.latest() is None
        assert store.stats()["specs"] == 0

        # save + reload round-trip
        spec1 = _rx.build_spec(_PROMPT_V1)
        sig = store.save_spec(spec1)
        assert sig == spec1.signature
        assert store.list_specs() == [sig]
        loaded = store.load_spec(sig)
        assert loaded is not None
        assert _rules_digest(loaded.rules) == _rules_digest(spec1.rules)
        assert loaded.signature == sig
        assert latest_ok(loaded, store)

        # persistence: a brand-new store instance sees the same data
        store2 = RuleStore(path)
        assert store2.load_spec(sig) is not None
        assert store2.stats()["specs"] == 1

        # versioning: v2 adds a rule → diff detects it
        spec2 = _rx.build_spec(_PROMPT_V2)
        assert spec2.signature != sig or spec2.source_sha256 != spec1.source_sha256
        d = diff_specs(spec1, spec2)
        assert not d.same
        assert d.added and not d.removed, d.to_dict()
        assert d.summary().startswith("+1")

        # changed rule → changed diff
        spec_mod = _rx.PromptSpec.from_dict(spec1.to_dict())
        spec_mod.rules[0].severity = "critical"
        d2 = diff_specs(spec1, spec_mod)
        assert not d2.same and not d2.added and not d2.removed
        assert len(d2.changed) == 1 and d2.changed[spec1.rules[0].id]

        # identical specs → same
        assert diff_specs(spec1, spec1).same
        assert diff_specs(spec1, _rx.PromptSpec.from_dict(spec1.to_dict())).same

        # update flow: saving v2 under a new sig; latest() returns it
        sig2 = store.save_spec(spec2, event="update")
        assert sig2 != sig
        assert store.list_specs() == sorted([sig, sig2])
        assert store.latest().signature == sig2

        # delete
        assert store.delete_spec(sig) is True
        assert store.delete_spec(sig) is False
        assert store.list_specs() == [sig2]
        hist_events = list(store.iter_history())
        assert hist_events[-1]["event"] == "delete"

        # history bounded to 500
        for i in range(600):
            store.append_history({"event": f"tick{i}", "i": i})
        events = list(store.iter_history())
        assert len(events) == 500, len(events)
        assert events[-1]["event"] == "tick599"
        assert events[0]["event"] != "tick0"  # old ticks dropped
        # iter_history limit
        assert len(list(store.iter_history(limit=3))) == 3

        # corrupt file → quarantined, store recovers
        store.append_history({"event": "marker"})
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("{not json at all")
        store3 = RuleStore(path)
        assert store3.list_specs() == []  # fresh doc, no crash
        assert os.path.exists(path + CORRUPT_BACKUP_SUFFIX)

        # export/import round-trip via to_dict/from_dict
        raw = _spec_to_dict(spec2)
        back = _spec_from_dict(raw)
        assert _rules_digest(back.rules) == _rules_digest(spec2.rules)

        # schema stamped on the rewritten document
        store4 = RuleStore(path)
        store4.save_spec(spec2)
        with open(store4.path, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["schema"] == SCHEMA_VERSION
        assert sig2 in on_disk["specs"]

        # concurrent writers: lockfile prevents interleaving damage
        s1 = RuleStore(path)
        s2 = RuleStore(path)
        s1._acquire_lock()
        try:
            try:
                s2._acquire_lock()
                got = False
            except StoreError:
                got = True
            assert got, "second writer must hit the lock"
        finally:
            s1._release_lock()
        s2._acquire_lock()  # lock free now → succeeds
        s2._release_lock()

        # lockfile stealing after stale age
        with open(s2.lock_path, "w", encoding="utf-8") as fh:
            fh.write("dead\n")
        os.utime(s2.lock_path, (time.time() - 999, time.time() - 999))
        s1._acquire_lock()
        s1._release_lock()
        assert not os.path.exists(s1.lock_path)

        print(f"RULESTORE SELF-TEST PASS "
              f"(specs={store4.stats()['specs']}, "
              f"history bounded at 500, diff + lock verified)")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def latest_ok(loaded, store) -> bool:
    dig = _rules_digest(loaded.rules)
    dig2 = _rules_digest(store.latest().rules)
    return dig == dig2


if __name__ == "__main__":
    raise SystemExit(_self_test())
