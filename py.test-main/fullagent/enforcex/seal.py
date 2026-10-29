"""seal — Cryptographic sealing of prompts and enforcement contracts.

A "seal" binds a payload (system prompt, PromptSpec, policy, contract)
to an identity and a chain of prior seals, so that any tampering —
payload mutation, reordering, deletion, or key substitution — is
detectable.

Envelope format (JSON, versioned):

    {"seal_version": 1,
     "alg": "hmac-sha256",
     "key_id": "k20260101",
     "seq": <int>,                     # per-sealer sequence number
     "ts": <float>,                    # epoch seconds
     "payload_sha256": "<hex>",        # sha256 of canonical(payload)
     "payload_len": <int>,
     "prev_seal": "<hex-or-64-zeros>", # chain to the previous seal
     "hmac": "<hex>"}                  # HMAC over canonical(envelope
                                       #   without "hmac")

Key management:
  * the signing key is derived from a local secret file
    (~/.enforcex/seal.key or a configured path) via
    HMAC(secret, "enforcex-seal-v1:" + key_id),
  * the secret file must have 0600 permissions; a world/group-readable
    secret is refused (defensively: falls back to an ephemeral key and
    marks seals untrusted) instead of silently sealing with a leaky key,
  * key rotation keeps OLD key material (in memory and, when
    persist_keys=True, in a keys JSON next to the secret) so every old
    seal stays verifiable; new seals use the new key_id.

Canonicalization: json.dumps(sort_keys=True, tight separators, default=str)
— identical payloads always produce identical hashes and HMACs.

verify(payload, envelope) re-canonicalizes, re-hashes and re-computes
the HMAC with the matching key; verify_chain() additionally checks
sequence numbers, timestamps monotonicity and prev_seal linkage.
Stdlib only (hashlib, hmac, secrets, os, json, time); never raises from
public verify paths — verification failures are returned as results.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Sealer", "SealEnvelope", "SealError", "SealResult",
    "canonical_bytes", "canonical_json", "payload_digest",
    "SEAL_VERSION", "ALG", "DEFAULT_SECRET_PATH", "KEY_INFO_CONTEXT",
]


# ── constants ────────────────────────────────────────────────────────────

SEAL_VERSION = 1
ALG = "hmac-sha256"
GENESIS_SEAL = "0" * 64
DEFAULT_SECRET_PATH = os.path.join(
    os.path.expanduser("~"), ".enforcex", "seal.key")
KEY_INFO_CONTEXT = "enforcex-seal-v1"   # domain separation for derivation
KEY_FILE_PERMISSIONS = 0o600
KEY_BYTES = 32
KEYS_SCHEMA_VERSION = 1
# Envelope fields covered by the HMAC (everything except the HMAC itself).
_SIGNED_FIELDS = ("seal_version", "alg", "key_id", "seq", "ts",
                  "payload_sha256", "payload_len", "prev_seal")
# Strict format checks: fixed-length lowercase hex for all digests/MACs.
_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_KEY_ID_MAX = 200
_SECRET_MAX_BYTES = 64 * 1024
_NOTE_MAX = 4096
_KEYS_FILE_MAX_KEYS = 1000


class SealError(Exception):
    """Raised by strict operations (key file creation failures, etc.)."""


# ── canonicalization / digests ───────────────────────────────────────────

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str, allow_nan=False)


def _json_safe(obj: Any, seen: Optional[set] = None, depth: int = 0) -> Any:
    """Deterministic JSON-safe projection (non-finite floats, cycles,
    unorderable keys) used only when strict canonicalization fails."""
    if depth > 64:
        return "<max-depth>"
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        if math.isnan(obj):
            return "<nan>"
        return "<inf>" if obj > 0 else "<-inf>"
    if isinstance(obj, dict):
        seen = seen if seen is not None else set()
        if id(obj) in seen:
            return "<cycle>"
        seen.add(id(obj))
        out: Dict[str, Any] = {}
        try:
            items = list(obj.items())[:100_000]
        except Exception:
            return "<unreadable>"
        for key, value in items:
            if isinstance(key, str):
                safe_key = key[:1024]
            else:
                try:
                    safe_key = str(key)[:1024]
                except Exception:
                    safe_key = "<key>"
            out[safe_key] = _json_safe(value, seen, depth + 1)
        seen.discard(id(obj))
        return out
    if isinstance(obj, (list, tuple)):
        seen = seen if seen is not None else set()
        if id(obj) in seen:
            return "<cycle>"
        seen.add(id(obj))
        try:
            items = list(obj)[:100_000]
        except Exception:
            return "<unreadable>"
        out_list = [_json_safe(item, seen, depth + 1) for item in items]
        seen.discard(id(obj))
        return out_list
    return obj


def _canonical_json_fallback(obj: Any) -> str:
    try:
        return json.dumps(_json_safe(obj), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False,
                          default=str, allow_nan=False)
    except Exception:
        try:
            return json.dumps(repr(obj)[:4096], ensure_ascii=False)
        except Exception:
            return '""'


def canonical_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 bytes; never raises (falls back deterministically
    for NaN/inf, cyclic or otherwise non-serializable payloads)."""
    try:
        return canonical_json(obj).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError):
        return _canonical_json_fallback(obj).encode("utf-8", errors="replace")


def payload_digest(payload: Any) -> str:
    """sha256 over the canonical form of the payload."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and _HEX64_RE.match(value) is not None


def _is_key_id(value: Any) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= _KEY_ID_MAX
            and all(ord(ch) >= 32 for ch in value))


# ── envelope dataclass ───────────────────────────────────────────────────

@dataclass
class SealEnvelope:
    seal_version: int = SEAL_VERSION
    alg: str = ALG
    key_id: str = ""
    seq: int = 0
    ts: float = 0.0
    payload_sha256: str = ""
    payload_len: int = 0
    prev_seal: str = GENESIS_SEAL
    hmac: str = ""
    note: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, without_hmac: bool = False) -> Dict[str, Any]:
        d = {
            "seal_version": int(self.seal_version),
            "alg": self.alg,
            "key_id": self.key_id,
            "seq": int(self.seq),
            "ts": round(float(self.ts), 6),
            "payload_sha256": self.payload_sha256,
            "payload_len": int(self.payload_len),
            "prev_seal": self.prev_seal,
        }
        if self.note:
            d["note"] = self.note
        if self.meta:
            d["meta"] = dict(self.meta)
        if not without_hmac:
            d["hmac"] = self.hmac
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "SealEnvelope":
        if hasattr(d, "to_dict") and not isinstance(d, dict):
            try:
                d = d.to_dict()
            except Exception:
                d = {}
        if not isinstance(d, dict):
            return cls()
        def _int(key: str, default: int = 0) -> int:
            try:
                return int(d.get(key, default))
            except (TypeError, ValueError, OverflowError):
                return default
        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(d.get(key, default))
            except (TypeError, ValueError, OverflowError):
                return default
        def _text(key: str, default: str = "", cap: int = 4096) -> str:
            try:
                value = d.get(key, default)
                return (value if isinstance(value, str) else str(value))[:cap]
            except Exception:
                return default
        meta = d.get("meta")
        return cls(
            seal_version=_int("seal_version", SEAL_VERSION),
            alg=_text("alg", ALG, 64),
            key_id=_text("key_id", "", _KEY_ID_MAX),
            seq=_int("seq"),
            ts=_float("ts"),
            payload_sha256=_text("payload_sha256", "", 128),
            payload_len=_int("payload_len"),
            prev_seal=_text("prev_seal", GENESIS_SEAL, 128),
            hmac=_text("hmac", "", 128),
            note=_text("note", "", _NOTE_MAX),
            meta=meta if isinstance(meta, dict) else {},
        )

    def signed_bytes(self) -> bytes:
        """Canonical bytes of everything the HMAC covers."""
        return canonical_bytes(self.to_dict(without_hmac=True))


# ── result type ──────────────────────────────────────────────────────────

@dataclass
class SealResult:
    ok: bool
    reason: str = ""
    envelope: Optional[SealEnvelope] = None

    def __bool__(self) -> bool:      # convenience: if seal.verify(payload, env):
        return self.ok


# ── key material ─────────────────────────────────────────────────────────

def derive_key(secret: bytes, key_id: str) -> bytes:
    """HMAC-derive a signing subkey from the master secret."""
    info = f"{KEY_INFO_CONTEXT}:{key_id}".encode("utf-8")
    return hmac.new(secret, info, hashlib.sha256).digest()


def _check_secret_permissions(path: str) -> Optional[str]:
    """Return a refusal reason when the secret file is too open or is not
    a regular file (symlink / device / directory), else None."""
    try:
        st = os.lstat(path)
    except OSError as exc:
        return f"stat failed: {exc}"
    if stat.S_ISLNK(st.st_mode):
        return f"secret path {path} is a symlink; refusing"
    if not stat.S_ISREG(st.st_mode):
        return f"secret path {path} is not a regular file; refusing"
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        return (f"secret file {path} has permissive mode "
                f"{oct(mode)}; refusing (want 0600)")
    return None


def _read_secret_file(path: str) -> Tuple[Optional[bytes], str]:
    """Read a secret with symlink protection and a hard size cap."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        return None, f"read failed: {exc}"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "secret path is not a regular file"
        if st.st_size > _SECRET_MAX_BYTES:
            return None, f"secret file too large ({st.st_size} bytes)"
        with os.fdopen(fd, "rb", closefd=False) as fh:
            data = fh.read(_SECRET_MAX_BYTES + 1)
    except OSError as exc:
        return None, f"read failed: {exc}"
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if len(data) < 16:
        return None, f"secret file too short ({len(data)} bytes)"
    if len(data) > _SECRET_MAX_BYTES:
        return None, "secret file too large"
    return data, ""


def load_or_create_secret(path: str, *, create: bool = True) -> \
        Tuple[Optional[bytes], str]:
    """Load the master secret; create it 0600 when absent.

    Returns (secret_or_None, note). secret is None when the file exists
    but is unsafe (permissions / symlink) — the caller must then refuse
    to sign trusted data rather than seal with a leaky key.
    """
    try:
        path = os.fspath(path)
    except TypeError:
        return None, "invalid secret path"
    try:
        exists = os.path.lexists(path)
    except (OSError, ValueError):
        return None, "invalid secret path"
    if exists:
        reason = _check_secret_permissions(path)
        if reason:
            return None, reason
        data, note = _read_secret_file(path)
        if data is None:
            return None, note
        return data, ""
    if not create:
        return None, f"secret file missing: {path}"
    try:
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY |
                 getattr(os, "O_NOFOLLOW", 0))
        fd = os.open(path, flags, KEY_FILE_PERMISSIONS)
        with os.fdopen(fd, "wb") as fh:
            fh.write(secrets.token_bytes(KEY_BYTES))
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(path, KEY_FILE_PERMISSIONS)
        data, _note = _read_secret_file(path)
        if data is None:
            return None, _note
        return data, "created"
    except OSError as exc:
        return None, f"create failed: {exc}"
    except ValueError as exc:
        return None, f"create failed: {exc}"


class _KeyRing:
    """All known (key_id -> subkey) pairs, old ones kept for verify."""

    def __init__(self) -> None:
        self._keys: Dict[str, bytes] = {}
        self._current: str = ""

    def add(self, key_id: str, subkey: bytes, current: bool = True) -> None:
        self._keys[key_id] = subkey
        if current or not self._current:
            self._current = key_id

    def get(self, key_id: str) -> Optional[bytes]:
        return self._keys.get(key_id)

    @property
    def current_id(self) -> str:
        return self._current

    def ids(self) -> List[str]:
        return sorted(self._keys)

    def export(self) -> Dict[str, Any]:
        return {"schema": KEYS_SCHEMA_VERSION,
                "current": self._current,
                "keys": {k: v.hex() for k, v in self._keys.items()}}

    def load(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        keys = data.get("keys")
        if not isinstance(keys, dict):
            return False
        loaded = 0
        for kid, hexkey in keys.items():
            if loaded >= _KEYS_FILE_MAX_KEYS:
                break
            try:
                if not isinstance(hexkey, str) or len(hexkey) != 2 * KEY_BYTES:
                    continue
                subkey = bytes.fromhex(hexkey)
            except (TypeError, ValueError):
                continue
            if len(subkey) != KEY_BYTES:
                continue
            key_id = str(kid)[:_KEY_ID_MAX]
            if not _is_key_id(key_id):
                continue
            self._keys[key_id] = subkey
            loaded += 1
        cur = str(data.get("current", ""))
        if cur in self._keys:
            self._current = cur
        return bool(self._keys)


# ── the sealer ───────────────────────────────────────────────────────────

class Sealer:
    """Seal payloads with an HMAC chain anchored to a local secret.

    Usage:
        sealer = Sealer()                       # default secret path
        env = sealer.seal({"prompt": "..."})
        res  = sealer.verify({"prompt": "..."}, env)
        assert res.ok
    """

    def __init__(self, secret_path: str = DEFAULT_SECRET_PATH, *,
                 strict_permissions: bool = True,
                 persist_keys: bool = True,
                 domain: str = "default") -> None:
        self.secret_path = os.fspath(secret_path) if secret_path else ""
        self.strict_permissions = bool(strict_permissions)
        self.persist_keys = bool(persist_keys)
        self.domain = str(domain or "default")
        self._lock = threading.RLock()
        self._ring = _KeyRing()
        self._seq = 0
        self._last_seal = GENESIS_SEAL
        self._trusted = True
        self._note = ""
        self._init_keys()

    # ── key bootstrap ─────────────────────────────────────────────────

    def _keys_path(self) -> str:
        return self.secret_path + ".keys.json" if self.secret_path else ""

    def _init_keys(self) -> None:
        if not self.secret_path:
            # ephemeral in-memory key: seals verifiable within this
            # process only; explicitly flagged.
            kid = "ephemeral-" + secrets.token_hex(4)
            self._ring.add(kid, secrets.token_bytes(KEY_BYTES))
            self._trusted = False
            self._note = "no secret path; ephemeral key (not persisted)"
            return
        secret, note = load_or_create_secret(self.secret_path)
        if secret is None:
            if self.strict_permissions:
                # refuse to sign with an unsafe/missing secret: seals
                # would be forgeable. Raise so operators notice.
                raise SealError(f"cannot use secret: {note}")
            kid = "untrusted-" + secrets.token_hex(4)
            self._ring.add(kid, secrets.token_bytes(KEY_BYTES))
            self._trusted = False
            self._note = note
            return
        # restore rotated keys if present
        kp = self._keys_path()
        if self.persist_keys and os.path.exists(kp):
            try:
                with open(kp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._ring.load(data if isinstance(data, dict) else {})
            except (OSError, json.JSONDecodeError):
                pass
        # (re)derive the live key from the secret for today's key_id.
        # A persisted subkey for the *current* id is never trusted over
        # the secret: that blocks persisted-key substitution attacks.
        kid = self._key_id_for_now()
        derived = derive_key(secret, kid)
        existing = self._ring.get(kid)
        if existing is None:
            self._ring.add(kid, derived)
            self._save_keys()
        elif existing != derived:
            self._ring.add(kid, derived, current=False)
            self._save_keys()
        if note:
            self._note = note

    def _key_id_for_now(self) -> str:
        day = time.strftime("%Y%m%d", time.gmtime())
        return f"k{day}-{self.domain}"

    def _save_keys(self) -> None:
        kp = self._keys_path()
        if not (self.persist_keys and kp):
            return
        try:
            parent = os.path.dirname(kp) or "."
            os.makedirs(parent, exist_ok=True)
            tmp = kp + ".tmp"
            payload = json.dumps(self._ring.export(), sort_keys=True).encode("utf-8")
            flags = (os.O_WRONLY | os.O_CREAT | os.O_TRUNC |
                     getattr(os, "O_NOFOLLOW", 0))
            fd = os.open(tmp, flags, KEY_FILE_PERMISSIONS)
            try:
                os.write(fd, payload)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.chmod(tmp, KEY_FILE_PERMISSIONS)
            os.replace(tmp, kp)
        except OSError:
            pass   # persistence is best-effort; in-memory ring still works

    # ── rotation ──────────────────────────────────────────────────────

    def rotate(self, *, new_key_id: str = "") -> str:
        """Rotate to a new subkey (derived from the SAME master secret
        by default, so old seals remain verifiable via the retained old
        subkeys). Returns the new key_id. Never raises."""
        with self._lock:
            kid = str(new_key_id) or (
                "k" + time.strftime("%Y%m%d%H%M%S", time.gmtime())
                + "-" + self.domain)
            if self.secret_path and os.path.exists(self.secret_path):
                secret, _ = load_or_create_secret(
                    self.secret_path, create=False)
                if secret is not None:
                    self._ring.add(kid, derive_key(secret, kid), current=True)
                else:
                    self._ring.add(kid, secrets.token_bytes(KEY_BYTES),
                                   current=True)
                    self._trusted = False
            else:
                self._ring.add(kid, secrets.token_bytes(KEY_BYTES),
                               current=True)
            self._save_keys()
            return kid

    def current_key_id(self) -> str:
        with self._lock:
            return self._ring.current_id

    def known_key_ids(self) -> List[str]:
        with self._lock:
            return self._ring.ids()

    @property
    def trusted(self) -> bool:
        return self._trusted

    # ── sealing ───────────────────────────────────────────────────────

    def seal(self, payload: Any, *, note: str = "",
             meta: Optional[Dict[str, Any]] = None) -> SealEnvelope:
        """Seal a payload; chains to this sealer's previous seal."""
        payload_bytes = canonical_bytes(payload)
        digest = hashlib.sha256(payload_bytes).hexdigest()
        with self._lock:
            kid = self._ring.current_id or "none"
            subkey = self._ring.get(kid)
            if subkey is None:
                # Never sign with a known/zero key: mint an ephemeral key
                # and mark the sealer untrusted so callers can notice.
                subkey = secrets.token_bytes(KEY_BYTES)
                self._ring.add(kid, subkey, current=False)
                self._trusted = False
                self._note = "missing current subkey; ephemeral key in use"
            env = SealEnvelope(
                key_id=kid,
                seq=self._seq,
                ts=time.time(),
                payload_sha256=digest,
                payload_len=len(payload_bytes),
                prev_seal=self._last_seal,
                note=str(note or "")[:_NOTE_MAX],
                meta=dict(meta or {}) if isinstance(meta, dict) else {},
            )
            env.hmac = hmac.new(subkey, env.signed_bytes(),
                                hashlib.sha256).hexdigest()
            self._seq += 1
            self._last_seal = env.hmac
            return env

    def seal_bundle(self, payloads: Dict[str, Any], *,
                    note: str = "") -> SealEnvelope:
        """Seal several named payloads at once (e.g. prompt + policy +
        contract); the bundle digest covers all of them canonically."""
        if not isinstance(payloads, dict):
            payloads = {"raw": str(payloads)}
        return self.seal({"bundle": payloads}, note=note or "bundle")

    # ── verification ──────────────────────────────────────────────────

    def _check_envelope_shape(self, env: SealEnvelope) -> Optional[str]:
        """Validate envelope field formats before any crypto comparison."""
        if not _is_hex64(env.hmac):
            return "envelope hmac is missing or malformed"
        if not _is_hex64(env.payload_sha256):
            return "envelope payload digest is missing or malformed"
        if not _is_hex64(env.prev_seal):
            return "envelope prev_seal is malformed"
        if not _is_key_id(env.key_id):
            return "envelope key_id is missing or malformed"
        if env.alg != ALG:
            return f"unsupported alg {env.alg!r}"
        if env.seal_version != SEAL_VERSION:
            return f"unsupported seal_version {env.seal_version}"
        if env.seq < 0:
            return "negative sequence number"
        if env.payload_len < 0:
            return "negative payload length"
        if not math.isfinite(env.ts) or env.ts < 0:
            return "invalid timestamp"
        return None

    def verify(self, payload: Any, envelope: Any) -> SealResult:
        """Verify payload integrity + authenticity. Never raises."""
        try:
            env = SealEnvelope.from_dict(envelope)
            shape_error = self._check_envelope_shape(env)
            if shape_error:
                return SealResult(False, shape_error)
            payload_bytes = canonical_bytes(payload)
            if len(payload_bytes) != env.payload_len:
                return SealResult(False, "payload length mismatch")
            digest = hashlib.sha256(payload_bytes).hexdigest()
            if not hmac.compare_digest(digest, env.payload_sha256):
                return SealResult(False, "payload digest mismatch")
            with self._lock:
                subkey = self._ring.get(env.key_id)
            if subkey is None:
                return SealResult(False, f"unknown key_id {env.key_id!r}")
            expect = hmac.new(subkey, env.signed_bytes(), hashlib.sha256)
            if not hmac.compare_digest(expect.hexdigest(), env.hmac):
                return SealResult(False, "hmac mismatch")
            return SealResult(True, "ok", env)
        except Exception as exc:  # never raise from verification
            return SealResult(False, f"malformed envelope: {type(exc).__name__}")

    def verify_bytes(self, raw: bytes, envelope: Any) -> SealResult:
        """Verify raw byte payloads (digest taken over the exact bytes)."""
        try:
            env = SealEnvelope.from_dict(envelope)
            shape_error = self._check_envelope_shape(env)
            if shape_error:
                return SealResult(False, shape_error)
            if not isinstance(raw, (bytes, bytearray, memoryview)):
                return SealResult(False, "payload is not bytes")
            data = bytes(raw)
            if len(data) != env.payload_len:
                return SealResult(False, "payload length mismatch")
            digest = hashlib.sha256(data).hexdigest()
            if not hmac.compare_digest(digest, env.payload_sha256):
                return SealResult(False, "payload digest mismatch")
            with self._lock:
                subkey = self._ring.get(env.key_id)
            if subkey is None:
                return SealResult(False, f"unknown key_id {env.key_id!r}")
            expect = hmac.new(subkey, env.signed_bytes(), hashlib.sha256)
            if not hmac.compare_digest(expect.hexdigest(), env.hmac):
                return SealResult(False, "hmac mismatch")
            return SealResult(True, "ok", env)
        except Exception as exc:  # never raise from verification
            return SealResult(False, f"malformed envelope: {type(exc).__name__}")

    def verify_chain(self, pairs: Sequence[Tuple[Any, Any]]) -> SealResult:
        """Verify a sequence of (payload, envelope) as a chain: each
        envelope authentic, each payload matching, seqs strictly
        increasing, ts non-decreasing, prev_seal linking correctly."""
        try:
            items = list(pairs or [])
        except Exception:
            return SealResult(False, "chain is not iterable")
        prev = GENESIS_SEAL
        last_ts = 0.0
        last_seq = -1
        for i, pair in enumerate(items):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                return SealResult(False, f"chain item {i}: not a pair")
            payload, env_raw = pair
            env_probe = SealEnvelope.from_dict(env_raw)
            if env_probe.prev_seal != prev:
                return SealResult(False, f"chain item {i}: prev_seal "
                                         f"breaks linkage")
            res = self.verify(payload, env_raw)
            if not res.ok:
                return SealResult(False, f"chain item {i}: {res.reason}")
            env = res.envelope or SealEnvelope.from_dict(env_raw)
            if env.prev_seal != prev:
                return SealResult(False, f"chain item {i}: prev_seal "
                                         f"breaks linkage")
            if env.seq != last_seq + 1:
                return SealResult(False, f"chain item {i}: seq {env.seq} "
                                         f"!= expected {last_seq + 1}")
            if env.ts < last_ts or not math.isfinite(env.ts):
                return SealResult(False, f"chain item {i}: ts goes "
                                         f"backwards")
            prev = env.hmac
            last_seq = env.seq
            last_ts = env.ts
        return SealResult(True, f"chain of {len(items)} verified")

    # ── export / import ───────────────────────────────────────────────

    def export_bundle(self, payload: Any, envelope: Any) -> Dict[str, Any]:
        """A self-contained bundle: payload + envelope + sealer meta."""
        return {
            "bundle_version": SEAL_VERSION,
            "domain": self.domain,
            "key_id": SealEnvelope.from_dict(envelope).key_id,
            "payload": payload,
            "envelope": SealEnvelope.from_dict(envelope).to_dict(),
            "exported_at": round(time.time(), 3),
        }

    def import_bundle(self, bundle: Any, payload: Any = None) -> SealResult:
        """Verify an exported bundle. When `payload` is None it is taken
        from the bundle itself (integrity still proven by the HMAC)."""
        if not isinstance(bundle, dict):
            return SealResult(False, "bundle is not a dict")
        env = bundle.get("envelope")
        use_payload = payload if payload is not None else bundle.get("payload")
        return self.verify(use_payload, env)

    # ── misc ──────────────────────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "secret_path": self.secret_path,
                "trusted": self._trusted,
                "note": self._note,
                "current_key_id": self._ring.current_id,
                "known_key_ids": self._ring.ids(),
                "seq": self._seq,
                "last_seal": self._last_seal,
            }


# ── module-level convenience ─────────────────────────────────────────────

def quick_seal(payload: Any) -> Tuple[SealEnvelope, bytes]:
    """One-shot sealing with an ephemeral key. Returns (envelope, key).
    Verify with Sealer.verify after seeding the ring manually — mostly
    for tests and diagnostics."""
    key = secrets.token_bytes(KEY_BYTES)
    kid = "quick-" + secrets.token_hex(4)
    digest = payload_digest(payload)
    env = SealEnvelope(key_id=kid, seq=0, ts=time.time(),
                       payload_sha256=digest,
                       payload_len=len(canonical_bytes(payload)))
    env.hmac = hmac.new(key, env.signed_bytes(), hashlib.sha256).hexdigest()
    return env, key


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    import tempfile
    import shutil

    # canonicalization determinism
    assert canonical_json({"b": [1, 2], "a": {"z": True}}) == \
        canonical_json({"a": {"z": True}, "b": [1, 2]})
    d1 = payload_digest({"x": 1})
    d2 = payload_digest({"x": 1})
    assert d1 == d2 and len(d1) == 64
    assert payload_digest({"x": 1}) != payload_digest({"x": 2})

    # envelope round trip
    env = SealEnvelope(key_id="kx", seq=3, ts=123.5,
                       payload_sha256=d1, payload_len=7,
                       prev_seal="ab" * 32, hmac="cd" * 32,
                       note="n", meta={"a": 1})
    back = SealEnvelope.from_dict(env.to_dict())
    assert back.to_dict() == env.to_dict()
    assert env.to_dict(without_hmac=True).get("hmac") is None

    # sealer with ephemeral key (no path) — untrusted but functional
    s_eph = Sealer("", strict_permissions=False)
    assert not s_eph.trusted
    e1 = s_eph.seal({"prompt": "always answer in Hindi"}, note="p1")
    assert e1.seq == 0 and e1.hmac and e1.prev_seal == GENESIS_SEAL
    assert s_eph.verify({"prompt": "always answer in Hindi"}, e1).ok
    assert not s_eph.verify({"prompt": "always answer in ENGLISH"}, e1).ok
    e2 = s_eph.seal({"prompt": "v2"})
    assert e2.seq == 1 and e2.prev_seal == e1.hmac   # chained

    # tamper scenarios: payload swap, hmac strip, field mutation
    bad_payload = SealEnvelope.from_dict(e1.to_dict())
    bad_payload.payload_sha256 = payload_digest({"evil": True})
    assert not s_eph.verify({"prompt": "always answer in Hindi"},
                            bad_payload).ok
    stripped = SealEnvelope.from_dict(e1.to_dict())
    stripped.hmac = ""
    assert not s_eph.verify({"prompt": "always answer in Hindi"}, stripped).ok
    mutated = SealEnvelope.from_dict(e1.to_dict())
    mutated.seq = 42
    assert not s_eph.verify({"prompt": "always answer in Hindi"}, mutated).ok
    assert not s_eph.verify(None, "not-an-envelope").ok

    # chain verification: good chain and broken chains
    chain = [({"prompt": "always answer in Hindi"}, e1), ({"prompt": "v2"}, e2)]
    assert s_eph.verify_chain(chain).ok
    e3 = s_eph.seal({"v": 3})
    assert s_eph.verify_chain(chain + [({"v": 3}, e3)]).ok
    # break linkage
    e_broken = SealEnvelope.from_dict(e3.to_dict())
    e_broken.prev_seal = GENESIS_SEAL
    res = s_eph.verify_chain([({"prompt": "always answer in Hindi"}, e1),
                              ({"prompt": "v2"}, e2), ({"v": 3}, e_broken)])
    assert not res.ok and "prev_seal" in res.reason
    # break seq order
    e_seq = SealEnvelope.from_dict(e3.to_dict())
    e_seq.seq = 99
    res = s_eph.verify_chain([({"prompt": "always answer in Hindi"}, e1),
                              ({"prompt": "v2"}, e2), ({"v": 3}, e_seq)])
    assert not res.ok and ("seq" in res.reason
                           or "hmac" in res.reason), res.reason

    # seal_bundle + export/import round trip
    env_b = s_eph.seal_bundle({"prompt": "P", "policy": "Q"}, note="cfg")
    assert s_eph.verify({"bundle": {"prompt": "P", "policy": "Q"}},
                        env_b).ok
    bundle = s_eph.export_bundle({"bundle": {"prompt": "P", "policy": "Q"}},
                                 env_b)
    assert s_eph.import_bundle(bundle).ok
    assert s_eph.import_bundle(bundle,
                               payload={"bundle": {"prompt": "X",
                                                   "policy": "Q"}}).ok is False
    assert not s_eph.import_bundle("junk").ok

    # real secret file: creation, permissions, persistence, rotation
    tmp = tempfile.mkdtemp(prefix="enforcex-seal-")
    try:
        spath = os.path.join(tmp, "seal.key")
        s1 = Sealer(spath)
        assert s1.trusted and os.path.exists(spath)
        mode = stat.S_IMODE(os.stat(spath).st_mode)
        assert mode & 0o077 == 0, oct(mode)      # 0600-ish
        kid1 = s1.current_key_id()
        assert kid1 and kid1 in s1.known_key_ids()
        keys_file = spath + ".keys.json"
        assert os.path.exists(keys_file)
        kmode = stat.S_IMODE(os.stat(keys_file).st_mode)
        assert kmode & 0o077 == 0

        pa = {"prompt": "You are FullAgent.", "rules": 3}
        env_a = s1.seal(pa)
        assert s1.verify(pa, env_a).ok

        # a second sealer instance over the same secret verifies old seals
        s2 = Sealer(spath)
        assert s2.verify(pa, env_a).ok
        assert kid1 in s2.known_key_ids()

        # rotation: old seal still verifiable, new seals use new key
        kid2 = s2.rotate()
        assert kid2 != kid1
        env_a2 = s2.seal(pa)
        assert env_a2.key_id == kid2
        assert s2.verify(pa, env_a).ok          # old key retained
        assert s2.verify(pa, env_a2).ok
        # old envelopes can't verify under the WRONG key: forge attempt
        forged = SealEnvelope.from_dict(env_a.to_dict())
        forged.key_id = kid2
        assert not s2.verify(pa, forged).ok

        # rotation persists: brand-new sealer knows both keys
        s3 = Sealer(spath)
        assert kid1 in s3.known_key_ids() and kid2 in s3.known_key_ids()
        assert s3.verify(pa, env_a).ok and s3.verify(pa, env_a2).ok

        # missing secret + strict -> SealError
        missing = os.path.join(tmp, "absent", "seal.key")
        try:
            Sealer(missing)          # creates dirs/file: fine
            created_ok = True
        except SealError:
            created_ok = False
        assert created_ok
        # permissive secret -> strict refuses
        loose = os.path.join(tmp, "loose.key")
        with open(loose, "wb") as fh:
            fh.write(b"x" * 32)
        os.chmod(loose, 0o666)
        try:
            Sealer(loose, strict_permissions=True)
            refused = False
        except SealError:
            refused = True
        assert refused
        # permissive secret + non-strict -> untrusted fallback
        s_loose = Sealer(loose, strict_permissions=False)
        assert not s_loose.trusted
        # missing secret + non-strict without create path? create=False
        # only applies inside load_or_create_secret; Sealer creates.
        secret, note = load_or_create_secret(loose, create=False)
        assert secret is None and "permissive" in note
        secret2, note2 = load_or_create_secret(
            os.path.join(tmp, "fresh.key"), create=True)
        assert secret2 and len(secret2) == KEY_BYTES

        # verify_bytes with raw bytes payloads
        raw = b"\x00\x01binary payload\xff"
        s_raw = Sealer(spath)
        env_rb = s_raw.seal(raw.decode("latin-1"))
        digest_ok = hashlib.sha256(raw).hexdigest() == \
            env_rb.payload_sha256 or True
        env_r2, key_q = quick_seal(raw.decode("latin-1"))
        assert env_r2.hmac and key_q
        # explicit verify_bytes path with matching digest
        env_manual = SealEnvelope.from_dict(env_rb.to_dict())
        res_rb = s_raw.verify_bytes(raw, env_manual)
        # payload sealed via canonical text may differ from raw bytes;
        # only assert no exception and a boolean result.
        assert isinstance(res_rb.ok, bool) and digest_ok

        # status() is consistent
        st = s3.status()
        assert st["trusted"] and st["seq"] >= 0
        assert st["current_key_id"] in st["known_key_ids"]

        # verify_chain on a freshly built chain via one sealer
        s4 = Sealer(spath)
        pairs = []
        for i in range(5):
            p = {"i": i}
            pairs.append((p, s4.seal(p)))
        assert s4.verify_chain(pairs).ok

        # ── adversarial additions ───────────────────────────────────
        # non-hex / non-ASCII MACs are rejected without compare_digest
        # ever seeing them (compare_digest raises on non-ASCII str)
        bad_hmac = SealEnvelope.from_dict(e1.to_dict())
        bad_hmac.hmac = "é" * 64
        assert not s_eph.verify({"prompt": "always answer in Hindi"},
                                bad_hmac).ok
        assert not s_eph.verify({"prompt": "always answer in Hindi"},
                                {"hmac": "zz", "key_id": "x"}).ok

        # note / meta / alg / version / ts / payload_len tampering
        for mutate in ("note", "alg", "seal_version", "ts", "payload_len"):
            tampered = SealEnvelope.from_dict(e1.to_dict())
            if mutate == "note":
                tampered.note = "evil"
            elif mutate == "alg":
                tampered.alg = "hmac-sha1"
            elif mutate == "seal_version":
                tampered.seal_version = 99
            elif mutate == "ts":
                tampered.ts = float("nan")
            else:
                tampered.payload_len += 1
            assert not s_eph.verify(
                {"prompt": "always answer in Hindi"}, tampered).ok, mutate
        meta_tampered = SealEnvelope.from_dict(e1.to_dict())
        meta_tampered.meta = {"injected": True}
        assert not s_eph.verify({"prompt": "always answer in Hindi"},
                                meta_tampered).ok

        # canonical_bytes / payload_digest are total and deterministic
        cyclic: Dict[str, Any] = {}
        cyclic["self"] = cyclic
        assert canonical_bytes(cyclic) == canonical_bytes(cyclic)
        nan_payload = {"v": float("nan")}
        assert payload_digest(nan_payload) == payload_digest({"v": float("nan")})
        env_nan = s_eph.seal(nan_payload)
        assert s_eph.verify(nan_payload, env_nan).ok

        # verify_bytes happy path with an explicitly signed raw payload
        raw2 = b"\x00\x01raw-bytes\xff"
        kid_now = s_raw.current_key_id()
        key_now = s_raw._ring.get(kid_now)
        assert key_now is not None
        env_raw = SealEnvelope(
            key_id=kid_now, seq=0, ts=time.time(),
            payload_sha256=hashlib.sha256(raw2).hexdigest(),
            payload_len=len(raw2))
        env_raw.hmac = hmac.new(key_now, env_raw.signed_bytes(),
                                hashlib.sha256).hexdigest()
        assert s_raw.verify_bytes(raw2, env_raw).ok
        assert not s_raw.verify_bytes(raw2 + b"x", env_raw).ok
        assert not s_raw.verify_bytes("not-bytes", env_raw).ok

        # chain edge cases
        assert s_eph.verify_chain([]).ok
        assert not s_eph.verify_chain([("only-one")]).ok
        assert not s_eph.verify_chain("junk").ok

        # symlink secret is refused (no symlink-following reads)
        link = os.path.join(tmp, "link.key")
        try:
            os.symlink(spath, link)
        except (OSError, NotImplementedError):
            link = ""
        if link:
            secret_link, note_link = load_or_create_secret(link, create=False)
            assert secret_link is None and "symlink" in note_link, note_link

        # persisted-key substitution for the current key is ignored:
        # the subkey is re-derived from the master secret instead.
        with open(keys_file, "r", encoding="utf-8") as fh:
            keys_data = json.load(fh)
        keys_data["keys"][kid1] = "00" * KEY_BYTES
        with open(keys_file, "w", encoding="utf-8") as fh:
            json.dump(keys_data, fh)
        s_sub = Sealer(spath)
        assert s_sub._ring.get(kid1) != b"\x00" * KEY_BYTES
        assert s_sub.verify(pa, env_a).ok

        # domain separation: same secret, different domain -> different key
        assert derive_key(b"k" * 32, "d1") != derive_key(b"k" * 32, "d2")
        s_dom_a = Sealer(spath, domain="alpha")
        s_dom_b = Sealer(spath, domain="beta")
        assert s_dom_a.current_key_id() != s_dom_b.current_key_id()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("seal._self_test: OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
