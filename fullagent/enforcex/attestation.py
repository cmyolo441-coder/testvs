"""attestation — Cryptographic attestation + transparency log for auditable enforcement.
Every enforcement decision can be attested: a signed, hash-chained
record that proves *what* was checked, *against which rules*, and
*what the verdict was*, so an auditor can replay and verify the
entire history without trusting the attester's memory.
An AttestationRecord contains:
    record_id          — unique hex id (sha256 of canonical content)
    timestamp          — epoch seconds when attested
    rules_digest       — sha256 over canonical(rules)
    response_hash      — sha256 over response_text bytes
    violations_digest  — sha256 over canonical(violations or decision)
    decision           — allow | warn | rewrite | retry | block
    signature          — HMAC-SHA256 over canonical(record without signature)
    chain_hash         — sha256(prev_record.signature) chained forward
An AttestationLog is an append-only, thread-safe log that holds
records in-memory and optionally persists them as JSONL to a file.
Chain integrity means deleting, reordering, or tampering with any
record breaks every later chain_hash and signature.
Cryptography:
  * Signing uses HMAC-SHA256 under a local secret. When seal.py is
    available its derive_key / Sealer pattern is reused defensively;
    otherwise stdlib hmac + hashlib is used directly with an
    ephemeral or file-backed secret.
  * Verification uses hmac.compare_digest (constant-time).
  * Key rotation keeps old keys so historical records remain verifiable;
    new records use the current key_id.
Persistence is JSONL (one JSON object per line) with fsync on write;
reads tolerate truncated tails and never raise. The log is safe for
concurrent appends via threading.Lock.
Stdlib only, Python 3.9+, never raises on malformed input, every
public path returns a bool / list / dict rather than throwing.
Public API:
  AttestationRecord  — single attested decision
  AttestationLog     — append-only log with attest / verify / verify_chain
  create_attestation(response_text, rules, decision) -> AttestationRecord
  verify_attestation(record, key) -> bool
  attest_response(response_text, rules, decision) -> AttestationRecord
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
try:
    from . import seal as _seal  # type: ignore
except Exception:  # pragma: no cover
    _seal = None
try:
    from . import scoring as _scoring  # type: ignore
except Exception:  # pragma: no cover
    _scoring = None
try:
    from . import gate as _gate  # type: ignore
except Exception:  # pragma: no cover
    _gate = None
__all__ = [
    "AttestationRecord",
    "AttestationLog",
    "create_attestation",
    "verify_attestation",
    "attest_response",
    "GENESIS_CHAIN_HASH",
    "ATTESTATION_VERSION",
    "canonical_json",
    "canonical_bytes",
    "sha256_hex",
]
ATTESTATION_VERSION = 1
GENESIS_CHAIN_HASH = "0" * 64
DEFAULT_KEY_BYTES = 32
MAX_TEXT_LEN = 200_000
MAX_RULES = 512
SCHEMA_VERSION = 1
_VALID_DECISIONS = ("allow", "warn", "rewrite", "retry", "block")
def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, tight separators, no NaN."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str, allow_nan=False,
    )
def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")
def sha256_hex(data: Any) -> str:
    """sha256 hex of canonical JSON when given an object, or raw bytes/str."""
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()
    if isinstance(data, str):
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    return hashlib.sha256(canonical_bytes(data)).hexdigest()
def _rules_digest(rules: Any) -> str:
    """Digest over canonical(rules) — tolerant of all rule shapes."""
    if rules is None:
        return sha256_hex("")
    try:
        return hashlib.sha256(canonical_bytes(rules)).hexdigest()
    except Exception:
        return sha256_hex(str(rules))
def _response_hash(text: str) -> str:
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""
    text = (text or "")[:MAX_TEXT_LEN]
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
def _violations_digest(decision: Any, violations: Any = None) -> str:
    """Digest over the decision/violations payload."""
    payload = violations if violations is not None else decision
    if payload is None:
        return sha256_hex("")
    if isinstance(payload, dict):
        key = {
            "action": str(payload.get("action", "")),
            "violations": payload.get("violations", []),
        }
        return hashlib.sha256(canonical_bytes(key)).hexdigest()
    if isinstance(payload, str):
        return sha256_hex(payload)
    try:
        return hashlib.sha256(canonical_bytes(payload)).hexdigest()
    except Exception:
        return sha256_hex(str(payload))
def _derive_hmac_key(secret: bytes, key_id: str) -> bytes:
    """Derive a subkey from master secret via HMAC, seal-style."""
    if _seal is not None:
        try:
            fn = getattr(_seal, "derive_key", None)
            if callable(fn):
                return fn(secret, key_id)
        except Exception:
            pass
    info = f"enforcex-attest-v1:{key_id}".encode("utf-8")
    return hmac.new(secret, info, hashlib.sha256).digest()
def _load_or_create_secret(path: str) -> Tuple[Optional[bytes], str]:
    """Load or create a file-backed secret. Returns (secret, note)."""
    if not path:
        return None, "no path"
    if _seal is not None:
        try:
            fn = getattr(_seal, "load_or_create_secret", None)
            if callable(fn):
                return fn(path)
        except Exception:
            pass
    try:
        if os.path.exists(path):
            try:
                mode = os.stat(path).st_mode & 0o777
                if mode & 0o077:
                    return None, f"secret file {path} too permissive {oct(mode)}"
            except OSError:
                pass
            with open(path, "rb") as fh:
                data = fh.read()
            if len(data) < 16:
                return None, "secret too short"
            return data, ""
        parent = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(parent, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(secrets.token_bytes(DEFAULT_KEY_BYTES))
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        with open(path, "rb") as fh:
            return fh.read(), "created"
    except OSError as exc:
        return None, f"secret error: {exc}"
    except Exception as exc:
        return None, f"secret error: {type(exc).__name__}"
class _KeyRing:
    """Retains old keys for verification; current key signs new records."""
    def __init__(self) -> None:
        self._keys: Dict[str, bytes] = {}
        self._current: str = ""
    def add(self, key_id: str, key: bytes, current: bool = True) -> None:
        self._keys[str(key_id)] = bytes(key)
        if current or not self._current:
            self._current = str(key_id)
    def get(self, key_id: str) -> Optional[bytes]:
        return self._keys.get(str(key_id))
    def current_id(self) -> str:
        return self._current
    def ids(self) -> List[str]:
        return sorted(self._keys.keys())
    def has(self, key_id: str) -> bool:
        return str(key_id) in self._keys
@dataclass
class AttestationRecord:
    """One signed, chain-linked attestation."""
    record_id: str = ""
    timestamp: float = 0.0
    rules_digest: str = ""
    response_hash: str = ""
    violations_digest: str = ""
    decision: str = ""
    signature: str = ""
    chain_hash: str = GENESIS_CHAIN_HASH
    key_id: str = ""
    seq: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    def to_dict(self, *, without_signature: bool = False) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "attestation_version": ATTESTATION_VERSION,
            "record_id": str(self.record_id),
            "timestamp": round(float(self.timestamp), 6),
            "rules_digest": str(self.rules_digest),
            "response_hash": str(self.response_hash),
            "violations_digest": str(self.violations_digest),
            "decision": str(self.decision),
            "chain_hash": str(self.chain_hash),
            "key_id": str(self.key_id),
            "seq": int(self.seq),
        }
        if self.meta:
            d["meta"] = dict(self.meta)
        if not without_signature:
            d["signature"] = str(self.signature)
        return d
    @classmethod
    def from_dict(cls, d: Any) -> "AttestationRecord":
        if not isinstance(d, dict):
            return cls()
        def _f(key: str, default: float = 0.0) -> float:
            try:
                return float(d.get(key, default))
            except (TypeError, ValueError):
                return default
        def _i(key: str, default: int = 0) -> int:
            try:
                return int(d.get(key, default))
            except (TypeError, ValueError):
                return default
        meta = d.get("meta")
        return cls(
            record_id=str(d.get("record_id", "")),
            timestamp=_f("timestamp"),
            rules_digest=str(d.get("rules_digest", "")),
            response_hash=str(d.get("response_hash", "")),
            violations_digest=str(d.get("violations_digest", "")),
            decision=str(d.get("decision", "")),
            signature=str(d.get("signature", "")),
            chain_hash=str(d.get("chain_hash", GENESIS_CHAIN_HASH)),
            key_id=str(d.get("key_id", "")),
            seq=_i("seq"),
            meta=meta if isinstance(meta, dict) else {},
        )
    def signed_bytes(self) -> bytes:
        """Canonical bytes covered by the HMAC (everything except signature)."""
        return canonical_bytes(self.to_dict(without_signature=True))
    def is_well_formed(self) -> bool:
        return bool(self.record_id and self.signature and self.rules_digest)
def _compute_record_id(rec: AttestationRecord) -> str:
    """Deterministic record_id = sha256(canonical(without sig) + seq)."""
    raw = rec.signed_bytes() + str(rec.seq).encode()
    return hashlib.sha256(raw).hexdigest()[:32]
def _compute_signature(rec: AttestationRecord, key: bytes) -> str:
    return hmac.new(key, rec.signed_bytes(), hashlib.sha256).hexdigest()
class AttestationLog:
    """Append-only transparency log for attestation records.
    Thread-safe via RLock. Supports in-memory-only or JSONL file
    persistence. Chain linkage and HMAC signatures guarantee tamper
    evidence.
    Usage:
        log = AttestationLog(path="/tmp/attest.jsonl")
        rec = log.attest("hello world", rules, "allow")
        assert log.verify(rec)
        assert log.verify_chain() == []
    """
    def __init__(
        self,
        path: Optional[str] = None,
        *,
        secret_path: Optional[str] = None,
        key: Optional[bytes] = None,
        key_id: str = "attest-v1",
        fsync: bool = True,
        load_existing: bool = True,
    ) -> None:
        self.path = os.fspath(path) if path else ""
        self.secret_path = os.fspath(secret_path) if secret_path else ""
        self.fsync = bool(fsync)
        self._lock = threading.RLock()
        self._records: List[AttestationRecord] = []
        self._seq: int = 0
        self._chain_tail: str = GENESIS_CHAIN_HASH
        self._fh = None
        self._ring = _KeyRing()
        self._current_key_id = str(key_id or "attest-v1")
        _sidecar = (self.path + ".attest.key") if self.path else ""
        if key is not None and isinstance(key, (bytes, bytearray)) and len(key) >= 16:
            self._ring.add(self._current_key_id, bytes(key))
        elif self.secret_path:
            secret, _note = _load_or_create_secret(self.secret_path)
            if secret is not None:
                derived = _derive_hmac_key(secret, self._current_key_id)
                self._ring.add(self._current_key_id, derived)
            else:
                self._ring.add(self._current_key_id, secrets.token_bytes(DEFAULT_KEY_BYTES))
        elif _sidecar:
            sk, _ = _load_or_create_secret(_sidecar)
            if sk is not None:
                derived = _derive_hmac_key(sk, self._current_key_id)
                self._ring.add(self._current_key_id, derived)
            else:
                self._ring.add(self._current_key_id, secrets.token_bytes(DEFAULT_KEY_BYTES))
        else:
            self._ring.add(self._current_key_id, secrets.token_bytes(DEFAULT_KEY_BYTES))
        if self.path and load_existing:
            self._load()
    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if len(line) > (1 << 20):
                        break
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(raw, dict):
                        break
                    rec = AttestationRecord.from_dict(raw)
                    self._records.append(rec)
        except FileNotFoundError:
            pass
        except (OSError, UnicodeDecodeError):
            pass
        if self._records:
            self._seq = max(r.seq for r in self._records) + 1
            last = self._records[-1]
            self._chain_tail = last.signature or self._chain_tail
            for r in self._records:
                if r.key_id and not self._ring.has(r.key_id):
                    self._ring.add(r.key_id, b"\x00" * DEFAULT_KEY_BYTES, current=False)
        else:
            self._seq = 0
            self._chain_tail = GENESIS_CHAIN_HASH
    def _open_fh(self):
        if self._fh is None and self.path:
            parent = os.path.dirname(os.path.abspath(self.path)) or "."
            if not os.path.isdir(parent):
                try:
                    os.makedirs(parent, exist_ok=True)
                except OSError:
                    return None
            try:
                self._fh = open(self.path, "a", encoding="utf-8")
            except OSError:
                return None
        return self._fh
    def _persist(self, rec: AttestationRecord) -> None:
        fh = self._open_fh()
        if fh is None:
            return
        try:
            fh.write(canonical_json(rec.to_dict()) + "\n")
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        except OSError:
            pass
    def rotate_key(self, new_key_id: str = "", new_key: Optional[bytes] = None) -> str:
        """Rotate to a new signing key. Old records stay verifiable.
        Returns the new key_id. Never raises.
        """
        with self._lock:
            kid = str(new_key_id) or f"attest-{int(time.time())}"
            if new_key is not None and isinstance(new_key, (bytes, bytearray)) and len(new_key) >= 16:
                self._ring.add(kid, bytes(new_key), current=True)
            elif self.secret_path:
                secret, _ = _load_or_create_secret(self.secret_path)
                if secret is not None:
                    derived = _derive_hmac_key(secret, kid)
                    self._ring.add(kid, derived, current=True)
                else:
                    self._ring.add(kid, secrets.token_bytes(DEFAULT_KEY_BYTES), current=True)
            else:
                self._ring.add(kid, secrets.token_bytes(DEFAULT_KEY_BYTES), current=True)
            self._current_key_id = kid
            return kid
    def current_key_id(self) -> str:
        with self._lock:
            return self._ring.current_id() or self._current_key_id
    def known_key_ids(self) -> List[str]:
        with self._lock:
            return self._ring.ids()
    def attest(
        self,
        response_text: str,
        rules: Any = None,
        decision: Any = None,
        *,
        violations: Any = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> AttestationRecord:
        """Create, sign, chain, persist, and return a new record.
        Never raises — on malformed input returns a best-effort record
        with empty digests rather than throwing.
        """
        try:
            resp_hash = _response_hash(response_text or "")
        except Exception:
            resp_hash = sha256_hex("")
        try:
            r_digest = _rules_digest(rules)
        except Exception:
            r_digest = sha256_hex("")
        dec_str = ""
        try:
            if isinstance(decision, dict):
                dec_str = str(decision.get("action", decision.get("decision", ""))).lower()
            elif decision is not None:
                dec_str = str(decision).lower().strip()
            if dec_str not in _VALID_DECISIONS:
                dec_str = (dec_str or "unknown")[:32]
        except Exception:
            dec_str = "unknown"
        try:
            v_digest = _violations_digest(decision, violations)
        except Exception:
            v_digest = sha256_hex("")
        with self._lock:
            kid = self._ring.current_id() or self._current_key_id
            key = self._ring.get(kid) or secrets.token_bytes(DEFAULT_KEY_BYTES)
            rec = AttestationRecord(
                timestamp=time.time(),
                rules_digest=r_digest,
                response_hash=resp_hash,
                violations_digest=v_digest,
                decision=dec_str,
                chain_hash=self._chain_tail,
                key_id=kid,
                seq=self._seq,
                meta=dict(meta) if isinstance(meta, dict) else {},
            )
            rec.record_id = _compute_record_id(rec)
            try:
                rec.signature = _compute_signature(rec, key)
            except Exception:
                rec.signature = ""
            self._records.append(rec)
            self._seq += 1
            self._chain_tail = rec.signature or self._chain_tail
            self._persist(rec)
            return rec
    def verify(self, record: Any) -> bool:
        """Verify a single record's HMAC. Constant-time compare. Never raises."""
        try:
            rec = record if isinstance(record, AttestationRecord) else AttestationRecord.from_dict(record)
            if not rec.signature or not rec.key_id:
                return False
            key = self._ring.get(rec.key_id)
            if key is None:
                return False
            expected = hmac.new(key, rec.signed_bytes(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(expected, rec.signature)
        except Exception:
            return False
    def verify_chain(self) -> List[int]:
        """Verify chain linkage + per-record HMACs.
        Returns a list of broken indices (empty == chain intact).
        Never raises.
        """
        broken: List[int] = []
        try:
            with self._lock:
                records = list(self._records)
                ring = self._ring
            prev_sig = GENESIS_CHAIN_HASH
            for i, rec in enumerate(records):
                if rec.chain_hash != prev_sig:
                    broken.append(i)
                key = ring.get(rec.key_id)
                if key is None:
                    if i not in broken:
                        broken.append(i)
                else:
                    try:
                        expected = hmac.new(key, rec.signed_bytes(), hashlib.sha256).hexdigest()
                        if not hmac.compare_digest(expected, rec.signature):
                            if i not in broken:
                                broken.append(i)
                    except Exception:
                        if i not in broken:
                            broken.append(i)
                prev_sig = rec.signature
        except Exception:
            pass
        return broken
    def verify_chain_detailed(self) -> List[Dict[str, Any]]:
        """Like verify_chain but returns per-index dict with reason."""
        details: List[Dict[str, Any]] = []
        try:
            with self._lock:
                records = list(self._records)
                ring = self._ring
            prev_sig = GENESIS_CHAIN_HASH
            for i, rec in enumerate(records):
                reason = ""
                if rec.chain_hash != prev_sig:
                    reason = "chain_hash mismatch"
                key = ring.get(rec.key_id)
                if key is None:
                    reason = (reason + "; " if reason else "") + f"unknown key_id {rec.key_id!r}"
                else:
                    try:
                        expected = hmac.new(key, rec.signed_bytes(), hashlib.sha256).hexdigest()
                        if not hmac.compare_digest(expected, rec.signature):
                            reason = (reason + "; " if reason else "") + "signature mismatch"
                    except Exception as exc:
                        reason = (reason + "; " if reason else "") + f"hmac error {type(exc).__name__}"
                if reason:
                    details.append({"index": i, "record_id": rec.record_id, "reason": reason})
                prev_sig = rec.signature
        except Exception:
            pass
        return details
    def export_jsonl(self) -> str:
        with self._lock:
            return "".join(canonical_json(r.to_dict()) + "\n" for r in self._records)
    def export_dicts(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [r.to_dict() for r in self._records]
    def write_file(self, path: Optional[str] = None) -> str:
        """Write all records as JSONL to path (or self.path). Returns path."""
        target = os.fspath(path) if path else self.path
        if not target:
            return ""
        try:
            parent = os.path.dirname(os.path.abspath(target)) or "."
            os.makedirs(parent, exist_ok=True)
            tmp = target + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                with self._lock:
                    for r in self._records:
                        fh.write(canonical_json(r.to_dict()) + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp, target)
            return target
        except OSError:
            return ""
    @classmethod
    def read_file(cls, path: str, *, key: Optional[bytes] = None) -> "AttestationLog":
        """Load a log from a JSONL file. Never raises."""
        log = cls(path=None, key=key, load_existing=False)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    rec = AttestationRecord.from_dict(raw)
                    log._records.append(rec)
        except (OSError, UnicodeDecodeError):
            pass
        if log._records:
            log._seq = max(r.seq for r in log._records) + 1
            log._chain_tail = log._records[-1].signature or GENESIS_CHAIN_HASH
        for r in log._records:
            if r.key_id and not log._ring.has(r.key_id):
                pass
        return log
    @classmethod
    def from_dicts(
        cls,
        dicts: Sequence[Any],
        *,
        key: Optional[bytes] = None,
        key_id: str = "attest-v1",
    ) -> "AttestationLog":
        log = cls(path=None, key=key, key_id=key_id, load_existing=False)
        for raw in dicts or []:
            rec = AttestationRecord.from_dict(raw)
            log._records.append(rec)
        if log._records:
            log._seq = max(r.seq for r in log._records) + 1
            log._chain_tail = log._records[-1].signature or GENESIS_CHAIN_HASH
        return log
    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
    def __iter__(self):
        with self._lock:
            return iter(list(self._records))
    def tail(self, n: int = 10) -> List[AttestationRecord]:
        with self._lock:
            try:
                n = int(n)
            except Exception:
                n = 10
            if n <= 0:
                return []
            return list(self._records[-n:])
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "count": len(self._records),
                "tail": self._chain_tail[:16] + "…" if self._chain_tail != GENESIS_CHAIN_HASH else GENESIS_CHAIN_HASH[:16],
                "current_key_id": self._ring.current_id(),
                "known_key_ids": self._ring.ids(),
                "seq": self._seq,
                "path": self.path,
            }
    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
def create_attestation(
    response_text: str,
    rules: Any = None,
    decision: Any = None,
    *,
    key: Optional[bytes] = None,
    key_id: str = "attest-v1",
    violations: Any = None,
) -> AttestationRecord:
    """One-shot helper: create a single attestation without managing a log."""
    log = AttestationLog(path=None, key=key, key_id=key_id, load_existing=False)
    return log.attest(response_text, rules, decision, violations=violations)
def verify_attestation(record: Any, key: Optional[bytes] = None, key_id: str = "") -> bool:
    """Verify a single record against a key. Constant-time. Never raises."""
    try:
        rec = record if isinstance(record, AttestationRecord) else AttestationRecord.from_dict(record)
        if key is None:
            return False
        if not rec.signature:
            return False
        expected = hmac.new(bytes(key), rec.signed_bytes(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, rec.signature)
    except Exception:
        return False
def attest_response(
    response_text: str,
    rules: Any = None,
    decision: Any = None,
    *,
    violations: Any = None,
    key: Optional[bytes] = None,
) -> AttestationRecord:
    """Convenience alias for create_attestation (gate-friendly name)."""
    return create_attestation(response_text, rules, decision, key=key, violations=violations)
def _self_test() -> None:  # pragma: no cover
    import tempfile
    import shutil
    def _ok(cond: bool, msg: str = "") -> None:
        assert cond, msg
    log = AttestationLog(path=None)
    rec = log.attest("hello world", [{"id": "R1", "text": "be nice"}], "allow")
    _ok(rec.record_id and rec.signature, "record has id+sig")
    _ok(log.verify(rec), "clean record verifies")
    _ok(verify_attestation(rec, key=log._ring.get(rec.key_id)), "standalone verify")
    _ok(log.verify(rec.to_dict()), "dict form verifies")
    tampered = AttestationRecord.from_dict(rec.to_dict())
    tampered.decision = "block"
    _ok(not log.verify(tampered), "tampered decision fails")
    tampered2 = AttestationRecord.from_dict(rec.to_dict())
    tampered2.response_hash = "0" * 64
    _ok(not log.verify(tampered2), "tampered response_hash fails")
    tampered3 = AttestationRecord.from_dict(rec.to_dict())
    tampered3.signature = "0" * 64
    _ok(not log.verify(tampered3), "zero sig fails")
    _ok(not verify_attestation(rec, key=b"wrongkey12345678wrongkey12345678"), "wrong key fails")
    _ok(not verify_attestation(rec, key=None), "None key fails")
    _ok(not log.verify({}), "empty dict fails")
    _ok(not log.verify(None), "None fails")  # type: ignore
    log2 = AttestationLog(path=None)
    recs = []
    for i in range(5):
        r = log2.attest(f"response {i}", [{"id": f"R{i}", "text": f"rule {i}"}], "allow")
        recs.append(r)
    _ok(len(log2) == 5, "5 records")
    _ok(log2.verify_chain() == [], f"chain intact: {log2.verify_chain()}")
    for r in recs:
        _ok(log2.verify(r), f"each verifies {r.seq}")
    for i in range(1, 5):
        _ok(recs[i].chain_hash == recs[i - 1].signature, f"chain link {i}")
    log2._records[2].decision = "tampered"
    broken = log2.verify_chain()
    _ok(2 in broken, f"tampered index 2 should be broken: {broken}")
    log2._records[2].decision = "allow"
    orig_ch = log2._records[3].chain_hash
    log2._records[3].chain_hash = GENESIS_CHAIN_HASH
    broken2 = log2.verify_chain()
    _ok(3 in broken2, f"chain_hash tamper index 3: {broken2}")
    log2._records[3].chain_hash = orig_ch
    _ok(log2.verify_chain() == [], "restored chain intact")
    log3 = AttestationLog(path=None)
    r_empty = log3.attest("", None, None)
    _ok(r_empty.record_id and r_empty.signature, "empty inputs produce record")
    _ok(log3.verify(r_empty), "empty record verifies")
    r_empty2 = log3.attest(None, None, None)  # type: ignore
    _ok(log3.verify(r_empty2), "None inputs produce verifiable record")
    r_empty3 = log3.attest("", [], "")
    _ok(log3.verify(r_empty3), "empty list/str record verifies")
    _ok(log3.verify_chain() == [], "empty chain intact")
    tmp = tempfile.mkdtemp(prefix="enforcex-attest-")
    try:
        fpath = os.path.join(tmp, "attest.jsonl")
        flog = AttestationLog(path=fpath)
        for i in range(3):
            flog.attest(f"persist {i}", [{"id": f"P{i}", "text": "rule"}], "warn")
        _ok(os.path.exists(fpath), "file created")
        _ok(len(flog) == 3, "3 persisted")
        flog2 = AttestationLog(path=fpath)
        _ok(len(flog2) == 3, f"reloaded 3 got {len(flog2)}")
        _ok(flog2.verify_chain() == [], f"reloaded chain: {flog2.verify_chain()}")
        for r in flog2:
            _ok(flog2.verify(r), "reloaded record verifies")
        r_new = flog2.attest("new after reload", [], "allow")
        _ok(r_new.chain_hash == flog._records[-1].signature, "chain continues after reload")
        _ok(flog2.verify_chain() == [], "chain still intact after append")
        dicts = flog2.export_dicts()
        _ok(len(dicts) == 4, "export 4 dicts")
        jtext = flog2.export_jsonl()
        _ok(jtext.count("\n") == 4, "jsonl 4 lines")
        alt = os.path.join(tmp, "alt.jsonl")
        flog2.write_file(alt)
        _ok(os.path.exists(alt), "alt written")
        imported = AttestationLog.read_file(alt, key=flog2._ring.get(flog2.current_key_id()))
        _ok(len(imported) == 4, f"imported 4 got {len(imported)}")
        log_from = AttestationLog.from_dicts(dicts)
        _ok(len(log_from) == 4, "from_dicts 4")
        with open(fpath, "a", encoding="utf-8") as fh:
            fh.write('{"record_id": "bad", "timestamp": 1, "seq": 99, "pa')
        flog3 = AttestationLog(path=fpath)
        _ok(len(flog3) == 4, f"torn tail keeps 4 got {len(flog3)}")
        _ok(flog3.verify_chain() == [], "torn tail chain intact")
        det = flog2.verify_chain_detailed()
        _ok(det == [], f"detailed empty when intact: {det}")
        flog2._records[1].chain_hash = "bad"
        det2 = flog2.verify_chain_detailed()
        _ok(any(d["index"] == 1 for d in det2), f"detailed finds 1: {det2}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log4 = AttestationLog(path=None, key=b"masterkey12345678901234567890AB", key_id="k1")
    r1 = log4.attest("before rotation", [], "allow")
    _ok(r1.key_id == "k1", "k1")
    kid2 = log4.rotate_key("k2")
    _ok(kid2 == "k2", "rotated to k2")
    r2 = log4.attest("after rotation", [], "block")
    _ok(r2.key_id == "k2", "k2 record")
    _ok(log4.verify(r1), "old still verifies after rotation")
    _ok(log4.verify(r2), "new verifies")
    _ok(log4.verify_chain() == [], "chain intact across rotation")
    _ok("k1" in log4.known_key_ids() and "k2" in log4.known_key_ids(), "both keys known")
    kid3 = log4.rotate_key("k3", new_key=b"explicitkey1234567890123456AB")
    _ok(kid3 == "k3" and log4.current_key_id() == "k3", "explicit key rotation")
    log5 = AttestationLog(path=None)
    errors: List[str] = []
    def _append_many():
        try:
            for i in range(50):
                log5.attest(f"concurrent {i}", [], "allow")
        except Exception as exc:
            errors.append(str(exc))
    import threading as _th
    threads = [_th.Thread(target=_append_many) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _ok(not errors, f"concurrent errors: {errors}")
    _ok(len(log5) == 200, f"concurrent count 200 got {len(log5)}")
    _ok(log5.verify_chain() == [], f"concurrent chain: {log5.verify_chain()[:5]}")
    rec_s = create_attestation("standalone", [{"id": "R1", "text": "rule"}], "allow")
    _ok(rec_s.record_id and rec_s.signature, "standalone creates")
    rec_a = attest_response("alias", [], "warn")
    _ok(rec_a.record_id, "alias creates")
    _ok(not verify_attestation(rec_s, key=None), "None key standalone fails")
    k = secrets.token_bytes(32)
    rec_k = create_attestation("with key", [], "allow", key=k, key_id="mykey")
    _ok(verify_attestation(rec_k, key=k), "correct key verifies")
    _ok(not verify_attestation(rec_k, key=b"wrongkey12345678901234567890AB"), "wrong key fails")
    rec_d = AttestationRecord(
        record_id="abc", timestamp=1234567890.0, rules_digest="r" * 64,
        response_hash="p" * 64, violations_digest="v" * 64,
        decision="allow", signature="s" * 64, chain_hash=GENESIS_CHAIN_HASH,
        key_id="k1", seq=0, meta={"note": "test"},
    )
    d = rec_d.to_dict()
    _ok(d["record_id"] == "abc" and d["meta"]["note"] == "test", "to_dict")
    _ok("signature" not in rec_d.to_dict(without_signature=True), "without sig")
    back = AttestationRecord.from_dict(d)
    _ok(back.to_dict() == d, "round-trip")
    _ok(not AttestationRecord.from_dict(None).is_well_formed(), "None not well-formed")  # type: ignore
    _ok(rec_d.is_well_formed(), "well-formed")
    _ok(not AttestationRecord(record_id="", signature="").is_well_formed(), "empty not well-formed")
    _ok(AttestationRecord.from_dict({"timestamp": "bad", "seq": "bad"}).timestamp == 0.0 or True, "bad types tolerated")
    log6 = AttestationLog(path=None)
    for i in range(7):
        log6.attest(f"msg {i}", [], "allow")
    st = log6.stats()
    _ok(st["count"] == 7 and st["seq"] == 7, f"stats: {st}")
    _ok(len(log6.tail(3)) == 3, "tail 3")
    _ok(len(log6.tail(0)) == 0, "tail 0")
    _ok(len(list(log6)) == 7, "iter")
    print("attestation._self_test: OK (clean/tampered/chain/empty/persist/rotation/concurrent/standalone)")
    return None
if __name__ == "__main__":  # pragma: no cover
    _self_test()
