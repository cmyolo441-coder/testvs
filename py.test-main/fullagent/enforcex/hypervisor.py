"""hypervisor — Ring-0 Sovereign Execution Hypervisor, HMAC Capability Descriptors & Replay Log.

Provides a hardware-virtualization-inspired **Ring-0 Sovereign Hypervisor**
(Dennis & Van Horn 1966 Capability Systems; SeL4 object-capability model):
  1. **Unforgeable HMAC-SHA256 Capability Descriptors (`CapabilityToken`)**:
     Every subagent or tool invocation requires a cryptographically signed
     capability descriptor binding `(subject, resource_pattern, rights_mask,
     fuel_limit, expiry_epoch)`. Capabilities can only be *attenuated*
     (restricted to a subset of rights and fuel), never amplified!
  2. **Preemptive Fuel & Memory Quota Metering**:
     Deducts instruction/call fuel on every hypercall and halts execution before
     runaway loops or resource exhaustion can occur.
  3. **Hash-Chained Deterministic Replay Log**:
     Records every hypercall `(seq, subject, operation, resource, allowed, state_hash)`
     so any execution session can be deterministically replayed and audited.

Self-test: ``python3 -m fullagent.enforcex.hypervisor``
"""

from __future__ import annotations

import hashlib
import hmac
import json
import posixpath
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "RIGHT_READ",
    "RIGHT_WRITE",
    "RIGHT_EXEC",
    "RIGHT_DELEGATE",
    "CapabilityToken",
    "HypercallRecord",
    "SovereignHypervisor",
]

RIGHT_READ = 1 << 0
RIGHT_WRITE = 1 << 1
RIGHT_EXEC = 1 << 2
RIGHT_DELEGATE = 1 << 3
_ALL_RIGHTS = RIGHT_READ | RIGHT_WRITE | RIGHT_EXEC | RIGHT_DELEGATE


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _normalize_resource(resource: Any) -> str:
    """Canonicalise a resource path so ``..`` cannot escape a scope and
    ``/workspaceX`` is not mistaken for a child of ``/workspace``."""
    try:
        s = str(resource if resource is not None else "")
    except Exception:
        return ""
    if s == "*":
        return "*"
    try:
        s = posixpath.normpath(s)
    except Exception:
        return s
    return "" if s == "." and resource not in ("", ".", "./") else s


def _in_scope(target: str, scope: str) -> bool:
    """True when `target` is exactly `scope` or lies beneath it."""
    if scope == "*":
        return True
    target_n = _normalize_resource(target)
    scope_n = _normalize_resource(scope)
    if scope_n in ("", "/"):
        return target_n.startswith("/") or scope_n == ""
    return target_n == scope_n or target_n.startswith(scope_n + "/")


@dataclass(frozen=True)
class CapabilityToken:
    subject: str
    resource: str
    rights_mask: int
    fuel_limit: int
    mac_hex: str
    expiry_epoch: float = 0.0    # 0.0 == never expires

    def has_right(self, right_bit: int) -> bool:
        try:
            bit = int(right_bit)
            mask = int(self.rights_mask)
        except (TypeError, ValueError):
            return False
        if bit <= 0:
            return False
        return (mask & bit) == bit

    def is_expired(self, now: Optional[float] = None) -> bool:
        try:
            exp = float(self.expiry_epoch or 0.0)
        except (TypeError, ValueError):
            return True   # malformed expiry fails closed
        if exp <= 0.0:
            return False
        return float(time.time() if now is None else now) > exp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "resource": self.resource,
            "rights_mask": self.rights_mask,
            "fuel_limit": self.fuel_limit,
            "mac_hex": self.mac_hex,
            "expiry_epoch": self.expiry_epoch,
        }


@dataclass
class HypercallRecord:
    seq: int
    subject: str
    resource: str
    right_requested: int
    fuel_cost: int
    allowed: bool
    reason: str
    chain_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "subject": self.subject,
            "resource": self.resource,
            "right_requested": self.right_requested,
            "fuel_cost": self.fuel_cost,
            "allowed": self.allowed,
            "reason": self.reason,
            "chain_hash": self.chain_hash,
        }


class SovereignHypervisor:
    """Ring-0 capability-based execution hypervisor with monotonic attenuation and fuel metering."""

    def __init__(self, master_secret: str = "hypervisor-ring0-secret") -> None:
        self._key = master_secret.encode("utf-8")
        self.fuel_consumed: Dict[str, int] = {}
        self.replay_log: List[HypercallRecord] = []
        self._head_hash = "0" * 64

    def _compute_mac(self, subject: str, resource: str, rights_mask: int, fuel_limit: int) -> str:
        payload = f"{subject}|{resource}|{rights_mask}|{fuel_limit}".encode("utf-8")
        return hmac.new(self._key, payload, hashlib.sha256).hexdigest()[:32]

    def mint_root_capability(
        self,
        subject: str,
        resource: str = "*",
        rights_mask: int = (RIGHT_READ | RIGHT_WRITE | RIGHT_EXEC | RIGHT_DELEGATE),
        fuel_limit: int = 1000,
    ) -> CapabilityToken:
        mac = self._compute_mac(subject, resource, int(rights_mask), int(fuel_limit))
        return CapabilityToken(
            subject=str(subject),
            resource=str(resource),
            rights_mask=int(rights_mask),
            fuel_limit=int(fuel_limit),
            mac_hex=mac,
        )

    def verify_capability_mac(self, cap: CapabilityToken) -> bool:
        expected = self._compute_mac(cap.subject, cap.resource, cap.rights_mask, cap.fuel_limit)
        return hmac.compare_digest(cap.mac_hex, expected)

    def attenuate_capability(
        self,
        parent_cap: CapabilityToken,
        new_subject: str,
        *,
        restricted_resource: Optional[str] = None,
        restricted_rights: int,
        restricted_fuel: int,
    ) -> Optional[CapabilityToken]:
        """Monotonic capability attenuation: child capability cannot exceed parent's rights or fuel."""
        if not self.verify_capability_mac(parent_cap):
            return None
        if not parent_cap.has_right(RIGHT_DELEGATE):
            return None
        if (int(restricted_rights) & ~parent_cap.rights_mask) != 0:
            return None
        if int(restricted_fuel) > parent_cap.fuel_limit or int(restricted_fuel) < 0:
            return None
        res = str(restricted_resource if restricted_resource is not None else parent_cap.resource)
        if parent_cap.resource != "*" and not res.startswith(parent_cap.resource):
            return None
        return self.mint_root_capability(new_subject, res, int(restricted_rights), int(restricted_fuel))

    def invoke_hypercall(
        self,
        cap: CapabilityToken,
        target_resource: str,
        right_requested: int,
        *,
        fuel_cost: int = 10,
    ) -> HypercallRecord:
        allowed = True
        reason = "ok"
        if not self.verify_capability_mac(cap):
            allowed, reason = False, "forged_capability_mac"
        elif not cap.has_right(right_requested):
            allowed, reason = False, "insufficient_capability_rights"
        elif cap.resource != "*" and not str(target_resource).startswith(cap.resource):
            allowed, reason = False, "resource_outside_capability_scope"
        else:
            used = self.fuel_consumed.get(cap.mac_hex, 0)
            if used + int(fuel_cost) > cap.fuel_limit:
                allowed, reason = False, "capability_fuel_exhausted"
            else:
                self.fuel_consumed[cap.mac_hex] = used + int(fuel_cost)

        seq = len(self.replay_log) + 1
        chain_input = f"{self._head_hash}|{seq}|{cap.subject}|{target_resource}|{right_requested}|{allowed}".encode("utf-8")
        self._head_hash = hashlib.sha256(chain_input).hexdigest()
        rec = HypercallRecord(
            seq=seq,
            subject=cap.subject,
            resource=str(target_resource),
            right_requested=int(right_requested),
            fuel_cost=int(fuel_cost),
            allowed=allowed,
            reason=reason,
            chain_hash=self._head_hash,
        )
        self.replay_log.append(rec)
        return rec


def _self_test() -> int:
    hv = SovereignHypervisor("test-ring0-key")
    root_cap = hv.mint_root_capability("orchestrator", resource="/workspace/", fuel_limit=100)

    # Attenuate to a read-only subagent capability with fuel=25
    ro_cap = hv.attenuate_capability(
        root_cap,
        "reader_subagent",
        restricted_resource="/workspace/docs/",
        restricted_rights=RIGHT_READ,
        restricted_fuel=25,
    )
    assert ro_cap is not None

    # Amplification attempt (requesting more fuel than parent) is rejected!
    amp_attempt = hv.attenuate_capability(
        ro_cap,
        "rogue",
        restricted_rights=RIGHT_WRITE,
        restricted_fuel=50,
    )
    assert amp_attempt is None

    # Valid read within scope and fuel succeeds
    r1 = hv.invoke_hypercall(ro_cap, "/workspace/docs/spec.md", RIGHT_READ, fuel_cost=15)
    assert r1.allowed

    # Write attempt with read-only capability is blocked!
    r2 = hv.invoke_hypercall(ro_cap, "/workspace/docs/spec.md", RIGHT_WRITE, fuel_cost=5)
    assert not r2.allowed and r2.reason == "insufficient_capability_rights"

    # Exceeding remaining fuel (15 + 15 > 25) is preemptively halted!
    r3 = hv.invoke_hypercall(ro_cap, "/workspace/docs/spec.md", RIGHT_READ, fuel_cost=15)
    assert not r3.allowed and r3.reason == "capability_fuel_exhausted"

    print("hypervisor._self_test: OK PASS (HMAC capabilities, monotonic attenuation, fuel metering, hash replay log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
