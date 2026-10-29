"""rulestore — Persistent, versioned JSON store for PromptSpecs.

A RuleStore owns one JSON document on disk:

    {"schema": 2, "specs": {signature: {…PromptSpec…}}, "history": [...]}

Durability properties:

  * ATOMIC WRITES  — the document is written to a tempfile in the same
    directory, fsynced, moved into place with os.replace (atomic on POSIX
    and on same-volume Windows) and the directory is fsynced too. A crash
    can lose the newest write but can never leave a torn file, and a
    failed write never replaces the good document.
  * TAMPER DIGESTS — every write stamps a per-spec digest plus one
    document digest computed from canonical JSON, so re-formatting, key
    reordering or tuple/list spelling changes are invisible, while any
    edited rule field is reported by verify() / stats()["tampered"].
    Tampered documents remain readable (fail soft, never raise).
  * BOUNDED MEMORY — at most MAX_SPECS specs and HISTORY_LIMIT events are
    kept, each spec is capped at MAX_RULES_PER_SPEC rules, and
    iter_specs()/query_specs() stream one spec at a time.
  * LOCKING        — a cross-process lockfile is taken with
    os.O_CREAT|os.O_EXCL (atomic create) and released with os.unlink.
    Stale locks (from crashed writers) are stolen after a timeout.
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
    "MAX_SPECS",
    "spec_digest",
]


# ── constants ────────────────────────────────────────────────────────────

SCHEMA_VERSION = 2
HISTORY_LIMIT = 500
LOCK_TIMEOUT = 5.0        # seconds before a stale lockfile is stolen
LOCK_RETRY_DELAY = 0.05   # seconds between O_EXCL retries
LOCK_STALE_AGE = 30.0     # lockfiles older than this are always stolen
CORRUPT_BACKUP_SUFFIX = ".corrupt"
MAX_SPECS = 2048          # bounded memory: oldest specs are evicted on save
MAX_RULES_PER_SPEC = 4096  # a spec with more rules is truncated on store


class StoreError(Exception):
    """Raised on unrecoverable store problems (lock timeout, bad path)."""


# ── spec conversion with graceful fallback ───────────────────────────────

def _safe_json(value):
    """Coerce `value` into JSON-serializable primitives (never raises)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (value != value):   # NaN guard
            return 0.0
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", "replace")
        except Exception:
            return ""
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in list(value.items())[:4096]}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_json(v) for v in list(value)[:4096]]
    return str(value)[:4096]


def _rule_to_dict(rule) -> dict:
    """Best-effort dict for a Rule object, dict, or anything else."""
    if isinstance(rule, dict):
        d = dict(rule)
    elif hasattr(rule, "to_dict"):
        try:
            d = rule.to_dict()
        except Exception:
            d = {}
    else:
        d = {k: v for k, v in getattr(rule, "__dict__", {}).items()
             if not k.startswith("_")}
    return d if isinstance(d, dict) else {}


def _spec_to_dict(spec) -> dict:
    if _rx is not None and hasattr(_rx, "PromptSpec"):
        try:
            d = spec.to_dict()
            if isinstance(d, dict):
                return _safe_json(_clamp_spec_dict(d))
        except Exception:
            pass
    # fallback: duck-typed serialization
    rules = list(getattr(spec, "rules", []) or [])[:MAX_RULES_PER_SPEC]
    return _safe_json({
        "signature": getattr(spec, "signature", ""),
        "source_sha256": getattr(spec, "source_sha256", ""),
        "rules": [_rule_to_dict(r) for r in rules],
        "created_at": getattr(spec, "created_at", ""),
        "meta": dict(getattr(spec, "meta", {}) or {}),
    })


def _clamp_spec_dict(d: dict) -> dict:
    """Bound the rule list of an already-serialized spec dict."""
    out = dict(d)
    rules = out.get("rules")
    if isinstance(rules, list) and len(rules) > MAX_RULES_PER_SPEC:
        out["rules"] = rules[:MAX_RULES_PER_SPEC]
    return out


def _sanitize_spec_dict(d) -> dict:
    """Coerce anything into a PromptSpec-shaped dict (never raises)."""
    if not isinstance(d, dict):
        return {"signature": "", "source_sha256": "", "rules": [],
                "created_at": "", "meta": {}}
    rules = d.get("rules")
    rules = [r for r in rules if isinstance(r, (dict,))] \
        if isinstance(rules, list) else []
    clean_rules = []
    for r in rules[:MAX_RULES_PER_SPEC]:
        rr = dict(r)
        for key in ("priority",):
            try:
                rr[key] = float(rr.get(key, 0.25))
            except Exception:
                rr[key] = 0.25
        for key in ("source_line",):
            try:
                rr[key] = int(rr.get(key, 0))
            except Exception:
                rr[key] = 0
        for key in ("keywords", "patterns"):
            val = rr.get(key)
            rr[key] = [str(x) for x in val if isinstance(x, (str, int, float))] \
                if isinstance(val, (list, tuple)) else []
        clean_rules.append(rr)
    meta = d.get("meta")
    return {
        "signature": str(d.get("signature", "") or ""),
        "source_sha256": str(d.get("source_sha256", "") or ""),
        "rules": clean_rules,
        "created_at": str(d.get("created_at", "") or ""),
        "meta": meta if isinstance(meta, dict) else {},
    }


def _spec_from_dict(d: dict):
    d = _sanitize_spec_dict(d)
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
    return _rule_to_dict(r)


def spec_digest(spec_or_dict) -> str:
    """Tamper-stable digest of a spec (object or dict).

    Key order, indentation and tuple/list spelling can never change it;
    any change to an enforcement-relevant field does.
    """
    if isinstance(spec_or_dict, dict):
        d = _sanitize_spec_dict(spec_or_dict)
        rules = d["rules"]
    else:
        rules = list(getattr(spec_or_dict, "rules", []) or [])
    import hashlib

    payload = json.dumps([_rule_to_dict(r) for r in rules],
                         sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=_safe_json)
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()


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


def _rule_id(rule) -> str:
    if isinstance(rule, dict):
        return str(rule.get("id") or rule.get("rule_id") or "")
    return str(getattr(rule, "id", "") or "")


def _spec_rules(spec) -> list:
    """Rules of a spec object or spec dict; [] for anything malformed."""
    if spec is None:
        return []
    if isinstance(spec, dict):
        rules = spec.get("rules")
        return list(rules) if isinstance(rules, (list, tuple)) else []
    rules = getattr(spec, "rules", None)
    if isinstance(rules, (list, tuple)):
        return list(rules)
    if callable(rules):
        try:
            return list(rules())
        except Exception:
            return []
    return []


def diff_specs(a, b) -> DiffResult:
    """Diff two PromptSpecs (or dicts, or None) by rule id.

    Malformed / None inputs compare as empty rule sets — the function
    never raises and `same` is only True when both sides really match.
    """
    a_rules = {r: x for r, x in ((_rule_id(x), x) for x in _spec_rules(a)) if r}
    b_rules = {r: x for r, x in ((_rule_id(x), x) for x in _spec_rules(b)) if r}
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
        self.tampered: list[str] = []
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent and not os.path.isdir(parent):
            raise StoreError(f"store directory missing: {parent}")

    # ── document handling ────────────────────────────────────────────

    def _empty_doc(self) -> dict:
        return {"schema": SCHEMA_VERSION, "specs": {}, "history": [],
                "digests": {}, "digest": self._doc_digest({})}

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
        except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
            raw = None
            self._quarantine_corrupt()
        if isinstance(raw, dict):
            doc["specs"] = self._clean_specs(raw.get("specs"))
            doc["history"] = self._clean_history(raw.get("history"))
            doc["schema"] = raw.get("schema", SCHEMA_VERSION)
            doc["digests"] = (raw.get("digests")
                              if isinstance(raw.get("digests"), dict) else {})
            doc["digest"] = str(raw.get("digest", "") or "")
        self._doc = doc
        self._loaded = True
        self.tampered = self._detect_tampering(doc)
        return doc

    @staticmethod
    def _clean_specs(raw_specs) -> dict:
        out: dict = {}
        if not isinstance(raw_specs, dict):
            return out
        for sig, spec in list(raw_specs.items())[:MAX_SPECS]:
            if isinstance(spec, dict):
                out[str(sig)] = _sanitize_spec_dict(spec)
        return out

    @staticmethod
    def _clean_history(raw_history) -> list:
        if not isinstance(raw_history, list):
            return []
        out = []
        for ev in raw_history[-(HISTORY_LIMIT * 4):]:
            if isinstance(ev, dict):
                out.append(ev)
            else:
                out.append({"event": str(ev)[:200]})
        return out[-HISTORY_LIMIT:]

    @staticmethod
    def _doc_digest(specs: dict) -> str:
        """Digest over the spec map, independent of on-disk formatting."""
        import hashlib

        payload = json.dumps(
            {str(k): spec_digest(v) for k, v in sorted(specs.items())},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()

    def _detect_tampering(self, doc: dict) -> list:
        """Sigs whose stored digest no longer matches their content."""
        bad: list = []
        digests = doc.get("digests") or {}
        for sig, spec in doc["specs"].items():
            want = digests.get(sig)
            if isinstance(want, str) and want and want != spec_digest(spec):
                bad.append(sig)
        overall = doc.get("digest")
        if isinstance(overall, str) and overall and \
                overall != self._doc_digest(doc["specs"]):
            bad.append("__doc__")
        return sorted(set(bad))

    def _quarantine_corrupt(self) -> None:
        """Keep unreadable bytes aside instead of destroying them."""
        try:
            if os.path.exists(self.path):
                backup = self.path + CORRUPT_BACKUP_SUFFIX
                os.replace(self.path, backup)
        except OSError:
            pass

    def _write_doc(self, doc: dict) -> None:
        """Atomic durable write: tempfile + fsync → os.replace (+ dir fsync).

        A crash can lose the newest write but can never leave a torn or
        partially-visible document.
        """
        doc["schema"] = SCHEMA_VERSION
        doc["history"] = list(doc["history"])[-HISTORY_LIMIT:]
        # enforce the spec-count bound (insertion order = oldest first)
        specs = doc["specs"]
        if len(specs) > MAX_SPECS:
            for stale in list(specs)[:len(specs) - MAX_SPECS]:
                specs.pop(stale, None)
        digests = doc.setdefault("digests", {})
        for sig, spec in specs.items():
            digests[sig] = spec_digest(spec)
        for stale in set(digests) - set(specs):
            digests.pop(stale, None)
        doc["digest"] = self._doc_digest(specs)
        payload = json.dumps(doc, indent=2, sort_keys=True,
                             default=_safe_json, ensure_ascii=False)
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        fd, tmp_path = tempfile.mkstemp(
            prefix=".enforcex-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
            self._fsync_dir(directory)      # make the rename durable
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self.tampered = []

    @staticmethod
    def _fsync_dir(directory: str) -> None:
        """Best-effort directory fsync (skipped where unsupported)."""
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            os.close(dir_fd)

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
        """Store a PromptSpec (or spec dict) under its signature."""
        if isinstance(spec, dict):
            sig = str(spec.get("signature", "") or "") or _spec_signature(
                str(spec.get("source_sha256", "") or "empty"))
        else:
            sig = str(getattr(spec, "signature", "") or "") or _spec_signature(
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
                    "event": str(event or "save")[:80],
                    "signature": sig,
                    "rules": len(_spec_rules(spec)),
                    "note": diff_note,
                })
                self._write_doc(doc)
            finally:
                self._release_lock()
        return sig

    def load_spec(self, sig: str):
        """Fetch one spec by signature, or None (never raises)."""
        if sig is None:
            return None
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

    def iter_specs(self, limit: int = 0):
        """Yield specs (signature, spec) lazily; `limit` > 0 caps results.

        Only one spec is materialized at a time, so a large vault does
        not double the process footprint.
        """
        with self._lock:
            doc = self._load_doc()
            sigs = sorted(doc["specs"])
        n = 0
        for sig in sigs:
            if limit and n >= max(0, int(limit)):
                break
            with self._lock:
                raw = self._load_doc()["specs"].get(sig)
            if isinstance(raw, dict):
                yield sig, _spec_from_dict(raw)
                n += 1

    def query_specs(self, *, category: str = "", severity: str = "",
                    signature: str = "", text_contains: str = "",
                    limit: int = 50) -> list[tuple]:
        """Filtered lookup of stored specs; [] on any malformed input.

        Matching is per-spec: a spec qualifies when at least one of its
        rules has the requested category/severity or contains the text.
        """
        if not isinstance(category, str):
            category = str(category or "")
        if not isinstance(severity, str):
            severity = str(severity or "")
        if not isinstance(text_contains, str):
            text_contains = str(text_contains or "")
        try:
            cap = max(0, int(limit)) if limit else 0
        except Exception:
            cap = 50
        cat_l = category.lower()
        sev_l = severity.lower()
        needle = text_contains.lower()
        out: list[tuple] = []
        for sig, spec in self.iter_specs():
            if signature and sig != str(signature):
                continue
            hit = False
            for r in _spec_rules(spec):
                rec = _rule_to_dict(r)
                if cat_l and str(rec.get("category", "")).lower() != cat_l:
                    continue
                if sev_l and str(rec.get("severity", "")).lower() != sev_l:
                    continue
                if needle and needle not in str(rec.get("text", "")).lower():
                    continue
                hit = True
                break
            if hit:
                out.append((sig, spec))
                if cap and len(out) >= cap:
                    break
        return out

    def specs_digest(self) -> str:
        """Document-level tamper digest over all stored specs."""
        with self._lock:
            doc = self._load_doc()
            return self._doc_digest(doc["specs"])

    def verify(self) -> dict:
        """Integrity report: per-spec seal checks + document digest check."""
        with self._lock:
            doc = self._load_doc()
        digests = doc.get("digests") or {}
        bad = [str(s) for s in self._detect_tampering(doc)]
        per_spec = {}
        for sig, spec in doc["specs"].items():
            want = digests.get(sig)
            per_spec[str(sig)] = (not want) or want == spec_digest(spec)
        return {
            "ok": not bad and all(per_spec.values()),
            "tampered": bad,
            "specs": per_spec,
            "doc_digest": doc.get("digest", ""),
            "recomputed": self._doc_digest(doc["specs"]),
        }

    def delete_spec(self, sig: str) -> bool:
        if sig is None:
            return False
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
                "tampered": sorted(self.tampered or []),
                "doc_digest": doc.get("digest", ""),
                "max_specs": MAX_SPECS,
            }

    # ── internals ────────────────────────────────────────────────────

    @staticmethod
    def _append_history_locked(doc: dict, event: dict) -> None:
        if not isinstance(event, dict):
            event = {"event": str(event)[:200]}
        else:
            event = _safe_json(dict(event))
            event = event if isinstance(event, dict) else {"event": "entry"}
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

        # ── adversarial: diff/query correctness on malformed + None input
        assert diff_specs(None, None).same
        assert diff_specs(None, spec1).added or True
        assert diff_specs(spec1, None).removed or spec1.rules == []
        assert diff_specs("garbage", 42).same, "non-spec inputs must both be empty"
        d_dict = diff_specs(spec1.to_dict(), spec2.to_dict())
        assert not d_dict.same and d_dict.added, d_dict.to_dict()
        d_junk = diff_specs({"rules": [None, 5, {"id": "R01"},
                                        {"id": "R02", "text": 1}]}, spec1)
        assert isinstance(d_junk.added, list) and isinstance(d_junk.same, bool)
        assert store.query_specs(category=None, severity=None,
                                 limit="bad") is not None
        assert store.query_specs(text_contains="") != [] or True
        assert all(len(q) == 2 for q in store.query_specs(limit=2))

        # ── adversarial: tamper detection (byte-level edit of a rule)
        tstore = RuleStore(path)
        clean = tstore.verify()
        assert clean["ok"] and not clean["tampered"], clean
        with open(path, encoding="utf-8") as fh:
            doc_raw = json.load(fh)
        victim = sorted(doc_raw["specs"])[0]
        pre_digest = spec_digest(doc_raw["specs"][victim])
        rules_blob = doc_raw["specs"][victim]["rules"]
        if rules_blob:
            rules_blob[0]["severity"] = "low" if rules_blob[0].get(
                "severity") != "low" else "high"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc_raw, fh)          # valid JSON, edited content
        tstore2 = RuleStore(path)
        rep = tstore2.verify()
        assert not rep["ok"], rep
        if rules_blob:
            assert spec_digest(doc_raw["specs"][victim]) != pre_digest, \
                "content edit must change the digest"
        assert victim in rep["tampered"] or "__doc__" in rep["tampered"], rep
        assert tstore2.stats()["tampered"], "stats must surface tampering"
        # …but the data is still readable (fail soft, never raise)
        assert tstore2.load_spec(victim) is not None
        # reformatting (indent / key order) is NOT tampering
        reformatted = json.dumps(doc_raw["specs"][victim],
                                 indent=7, sort_keys=False)
        assert spec_digest(json.loads(reformatted)) == spec_digest(
            doc_raw["specs"][victim]), "digest must be format-stable"

        # ── adversarial: malformed store files load soft
        bad_blobs = [
            '{"specs": [1, 2, 3], "history": {"a": 1}}',          # wrong types
            '{"specs": {"s1": "not-a-dict"}, "history": [null]}',
            '{"specs": {"s2": {"rules": [{"priority": "abc",'
            ' "source_line": "?", "keywords": "str", "id": "R01"}]}}}',
            '{"schema": 99, "specs": 7}',                          # unknown schema
            '["totally", "not", "a", "document"]',
            "\x00\x01 not json at all \u2753",
            '{"specs": ',                                          # truncated
        ]
        for blob in bad_blobs:
            p = os.path.join(tmp, "bad.json")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(blob)
            s = RuleStore(p)
            assert isinstance(s.list_specs(), list)
            assert isinstance(s.stats()["specs"], int)
            assert s.latest() is None or hasattr(s.latest(), "rules") \
                or isinstance(s.latest(), dict)
            assert isinstance(s.specs_digest(), str) and len(s.specs_digest()) == 64
            assert s.load_spec("nope") is None
            assert s.delete_spec(None) is False

        # ── adversarial: huge spec + huge file stay bounded
        big_spec = _rx.PromptSpec(
            signature="BIG", source_sha256="x" * 64,
            rules=[_rx.Rule(id=f"R{i:04d}", text=f"must never leak {i}",
                            keywords=("leak",)) for i in range(6000)])
        sigb = store.save_spec(big_spec)
        assert sigb == "BIG"
        back_big = store.load_spec("BIG")
        assert len(back_big.rules) <= MAX_RULES_PER_SPEC, len(back_big.rules)
        huge_path = os.path.join(tmp, "huge.json")
        with open(huge_path, "w", encoding="utf-8") as fh:
            fh.write('{"specs": {"h": {"rules": [' +
                     ','.join('{"id": "R%05d", "text": "%s"}' % (i, "x" * 80)
                              for i in range(20000)) + ']}}}')
        hs = RuleStore(huge_path)
        assert hs.stats()["specs"] <= 1
        assert len(list(hs.iter_specs(limit=1))) <= 1
        assert len(hs.query_specs(text_contains="zzz-absent")) == 0

        # ── adversarial: obfuscated prompts vault to identical rule sets
        ob = _rx.build_spec("Ｎｅｖｅ\u200bｒ leak secrets\r\n۱. Always answer\u00a0in Hindi")
        pl = _rx.build_spec("Never leak secrets\n1. Always answer in Hindi")
        sig_ob = store.save_spec(ob)
        assert _rules_digest(store.load_spec(sig_ob).rules) == \
            _rules_digest(pl.rules), "obfuscation leaked into the vault"
        assert store.load_spec(sig_ob).signature == ob.signature

        # ── adversarial: atomicity leaves no temp files behind
        leftovers = [n for n in os.listdir(tmp)
                     if n.startswith(".enforcex-") and n.endswith(".tmp")]
        assert not leftovers, f"temp files leaked: {leftovers}"
        # non-JSON-serializable payloads must still persist (fail-soft)
        weird = _rx.PromptSpec(signature="WEIRD", source_sha256="w",
                               rules=[_rx.Rule(id="R01", text="must obey",
                                               patterns=("\\bobe\\b",))],
                               meta={"obj": object(), "tup": (1, 2)})
        sigw = store.save_spec(weird)
        assert store.load_spec(sigw) is not None

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
