"""policy — Declarative enforcement policy engine for enforcex.

The gate (gate.py) makes hard-coded decisions; this module makes the
decision matrix *configurable*. A Policy is a declarative description of
HOW each detected violation should be enforced:

  PolicyRule   — one rule's enforcement intent:
                   * mode        : BLOCK | WARN | LOG | REWRITE | ESCALATE
                   * thresholds  : score bands that downgrade/upgrade
                   * stages      : conversation stages where it applies
                   * precedence  : override resolution order
                   * ttl / schedule / cooldown / rate-limit
  PolicySet    — an ordered collection of PolicyRules plus defaults,
                   with lookup, merge and override resolution.
  Policy       — the top-level object: PolicySet(s) + global config +
                   hot-reload from JSON + Policy.decide(detection).

A "detection" is the payload the gate produces before acting:
{"rule_id", "category", "severity", "confidence", "score", "stage",
"session_id", "violations", ...}.  decide() returns a Decision:

  {"action": allow|warn|log|rewrite|retry|block|escalate,
   "mode": <resolved mode>, "rule_id", "reason", "notes",
   "precedence_chain": [...], "matched_rule": {...}}

Resolution order (most specific wins):
  1. exact rule_id match (highest precedence value wins on ties,
     later-declared wins on equal precedence),
  2. category match,
  3. severity match,
  4. default rule ("*").
Stage gating: a rule that declares `stages` only applies when the
detection's stage is in it (or the stage is unknown); `not_stages`
excludes. Cooldowns/rate limits suppress repeated escalation noise per
session. Hot-reload is atomic: a policy that fails validation keeps the
previous one. Thread-safe via an RLock. Stdlib only; never raises on
malformed input — invalid policies degrade to permissive WARN.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Mode", "Stage", "PolicyRule", "PolicySet", "Policy", "Decision",
    "PolicyError", "parse_mode", "policy_from_json", "SCHEMA_VERSION",
]


# ── constants ────────────────────────────────────────────────────────────

SCHEMA_VERSION = 1

class Mode:
    """Enforcement modes, ordered by strictness."""
    ALLOW = "allow"
    LOG = "log"
    WARN = "warn"
    REWRITE = "rewrite"
    RETRY = "retry"
    BLOCK = "block"
    ESCALATE = "escalate"

    ALL = (ALLOW, LOG, WARN, REWRITE, RETRY, BLOCK, ESCALATE)
    STRICTNESS = {ALLOW: 0, LOG: 1, WARN: 2, REWRITE: 3,
                  RETRY: 4, BLOCK: 5, ESCALATE: 6}


class Stage:
    """Canonical conversation stages used for context sensitivity."""
    INGEST = "ingest"          # user input arriving
    PRE_CALL = "pre_call"      # about to call the model
    POST_CALL = "post_call"    # model response received
    TOOL = "tool"              # tool-call round-trips
    FINAL = "final"            # final answer delivery
    AUDIT = "audit"            # offline re-checks

    ALL = (INGEST, PRE_CALL, POST_CALL, TOOL, FINAL, AUDIT)


ACTION_FOR_MODE: Dict[str, str] = {
    Mode.ALLOW: "allow",
    Mode.LOG: "log",
    Mode.WARN: "warn",
    Mode.REWRITE: "rewrite",
    Mode.RETRY: "retry",
    Mode.BLOCK: "block",
    Mode.ESCALATE: "escalate",
}


class PolicyError(Exception):
    """Raised only by strict constructors (from_file strict=True)."""


# ── parsing helpers (defensive) ──────────────────────────────────────────

def parse_mode(value: Any, default: str = Mode.WARN) -> str:
    """Coerce anything into a known mode; unknown -> default."""
    if isinstance(value, str):
        v = value.strip().upper()
        if v in Mode.ALL:
            return v
        v = v.lower()
        if v in Mode.ALL:
            return v
    return default


def _to_float(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def _to_int(value: Any, default: int, lo: int = 0) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, n)


def _to_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


# ── dataclasses ──────────────────────────────────────────────────────────

@dataclass
class PolicyRule:
    """Enforcement intent for one slice of the rule space.

    match fields (rule_id / category / severity) select detections;
    "*" matches everything. Empty match fields mean "no constraint".
    """
    rule_id: str = "*"
    category: str = ""
    severity: str = ""
    mode: str = Mode.WARN
    precedence: int = 100              # higher wins on rule_id ties
    stages: Tuple[str, ...] = ()       # applicable conversation stages
    not_stages: Tuple[str, ...] = ()   # excluded stages
    min_confidence: float = 0.0        # detections below -> downgrade
    downgrade_mode: str = ""           # when confidence/min_score low
    score_floor: float = 0.0           # detection score below -> downgrade
    escalate_after: int = 0            # N prior hits in window -> escalate
    cooldown_s: float = 0.0            # suppress repeat actions per session
    rate_limit: int = 0                # max actions per window (0 = off)
    window_s: float = 60.0             # window for rate limiting
    ttl_s: float = 0.0                 # 0 = never expires
    created_at: float = 0.0            # epoch seconds; 0 = now on first use
    reason: str = ""                   # human-readable rationale
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- validation / normalization ---------------------------------------

    def normalized(self) -> "PolicyRule":
        """Return a copy with defaults filled and fields coerced."""
        r = copy.deepcopy(self)
        r.rule_id = str(r.rule_id or "*").strip() or "*"
        r.category = str(r.category or "").strip().lower()
        r.severity = str(r.severity or "").strip().lower()
        r.mode = parse_mode(r.mode)
        r.downgrade_mode = parse_mode(
            r.downgrade_mode, default=r.mode) if r.downgrade_mode else ""
        r.precedence = int(r.precedence)
        r.stages = tuple(s.strip().lower() for s in _to_str_list(r.stages))
        r.not_stages = tuple(s.strip().lower() for s in _to_str_list(r.not_stages))
        r.min_confidence = _to_float(r.min_confidence, 0.0, 0.0, 1.0)
        r.score_floor = _to_float(r.score_floor, 0.0, 0.0, 100.0)
        r.escalate_after = _to_int(r.escalate_after, 0)
        r.cooldown_s = _to_float(r.cooldown_s, 0.0, 0.0, 86400 * 7)
        r.rate_limit = _to_int(r.rate_limit, 0)
        r.window_s = _to_float(r.window_s, 60.0, 1.0, 86400 * 30)
        r.ttl_s = _to_float(r.ttl_s, 0.0, 0.0, 86400 * 365)
        if not r.created_at:
            r.created_at = time.time()
        if not isinstance(r.meta, dict):
            r.meta = {}
        return r

    def is_valid(self) -> bool:
        try:
            self.normalized()
            return True
        except Exception:
            return False

    # -- matching ----------------------------------------------------------

    def matches_target(self, rule_id: str, category: str,
                       severity: str) -> int:
        """Return specificity (higher = more specific) or 0 if no match.

        specificity: rule_id (4) + category (2) + severity (1) bits.
        """
        rid = str(rule_id or "").strip()
        cat = str(category or "").strip().lower()
        sev = str(severity or "").strip().lower()
        spec = 0
        if self.rule_id != "*":
            if rid != self.rule_id:
                return 0
            spec += 4
        if self.category:
            if cat != self.category:
                return 0
            spec += 2
        if self.severity:
            if sev != self.severity:
                return 0
            spec += 1
        return spec

    def applies_in_stage(self, stage: str) -> bool:
        st = str(stage or "").strip().lower()
        if self.not_stages and st in self.not_stages:
            return False
        if self.stages and st not in self.stages:
            return False
        return True

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_s <= 0:
            return False
        created = self.created_at or time.time()
        return (now if now is not None else time.time()) - created > self.ttl_s

    # -- downgrade path ----------------------------------------------------

    def effective_mode(self, detection: Dict[str, Any]) -> str:
        """Resolve mode considering confidence and score floors."""
        mode = self.mode
        try:
            conf = _to_float(detection.get("confidence"), 1.0, 0.0, 1.0)
            score = _to_float(detection.get("score"), 100.0, 0.0, 100.0)
        except Exception:
            return mode
        weak = (self.min_confidence > 0 and conf < self.min_confidence) or \
               (self.score_floor > 0 and score < self.score_floor)
        if weak and self.downgrade_mode:
            return self.downgrade_mode
        return mode

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity,
            "mode": self.mode,
            "precedence": self.precedence,
            "stages": list(self.stages),
            "not_stages": list(self.not_stages),
            "min_confidence": self.min_confidence,
            "downgrade_mode": self.downgrade_mode,
            "score_floor": self.score_floor,
            "escalate_after": self.escalate_after,
            "cooldown_s": self.cooldown_s,
            "rate_limit": self.rate_limit,
            "window_s": self.window_s,
            "ttl_s": self.ttl_s,
            "created_at": self.created_at,
            "reason": self.reason,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "PolicyRule":
        if not isinstance(d, dict):
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in d.items() if k in known}
        try:
            return cls(**kwargs)
        except TypeError:
            # tolerate bad types field-by-field
            base = cls()
            for k, v in kwargs.items():
                try:
                    setattr(base, k, v)
                except Exception:
                    continue
            return base


@dataclass
class PolicySet:
    """An ordered collection of PolicyRules with defaults."""
    name: str = "default"
    rules: List[PolicyRule] = field(default_factory=list)
    default_mode: str = Mode.WARN
    version: str = ""
    description: str = ""

    def normalized(self) -> "PolicySet":
        ps = PolicySet(
            name=str(self.name or "default"),
            rules=[r.normalized() if isinstance(r, PolicyRule) else
                   PolicyRule.from_dict(r).normalized()
                   for r in (self.rules or [])],
            default_mode=parse_mode(self.default_mode),
            version=str(self.version or ""),
            description=str(self.description or ""),
        )
        return ps

    def add(self, rule: PolicyRule) -> "PolicySet":
        self.rules.append(rule)
        return self

    def active_rules(self, now: Optional[float] = None) -> List[PolicyRule]:
        return [r for r in self.rules if not r.is_expired(now)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "default_mode": self.default_mode,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, d: Any) -> "PolicySet":
        if not isinstance(d, dict):
            return cls()
        raw_rules = d.get("rules") if isinstance(d.get("rules"), list) else []
        rules = [PolicyRule.from_dict(r) for r in raw_rules
                 if isinstance(r, dict)]
        return cls(
            name=str(d.get("name", "default")),
            rules=rules,
            default_mode=parse_mode(d.get("default_mode"), Mode.WARN),
            version=str(d.get("version", "")),
            description=str(d.get("description", "")),
        )

    # -- merge semantics ---------------------------------------------------

    def merged_with(self, other: "PolicySet") -> "PolicySet":
        """Overlay `other` on top: other's rules are appended (they win
        by declaration order at equal precedence); other's default_mode
        replaces when it is stricter or explicitly set."""
        merged = PolicySet(
            name=f"{self.name}+{other.name}",
            version=other.version or self.version,
            description=self.description or other.description,
            rules=list(self.rules) + list(other.rules),
            default_mode=other.default_mode,
        )
        return merged


# ── decision ─────────────────────────────────────────────────────────────

@dataclass
class Decision:
    action: str                       # allow|log|warn|rewrite|retry|block|escalate
    mode: str = Mode.WARN
    rule_id: str = ""
    matched_rule: Optional[Dict[str, Any]] = None
    precedence_chain: List[str] = field(default_factory=list)
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    suppressed: bool = False          # cooldown/rate-limit applied
    ts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "mode": self.mode,
            "rule_id": self.rule_id,
            "matched_rule": dict(self.matched_rule or {}),
            "precedence_chain": list(self.precedence_chain),
            "reason": self.reason,
            "notes": list(self.notes),
            "suppressed": self.suppressed,
            "ts": self.ts,
        }


def _allow(reason: str, notes: Optional[List[str]] = None) -> Decision:
    return Decision(action="allow", mode=Mode.ALLOW, reason=reason,
                    notes=list(notes or []), ts=time.time())


# ── the policy engine ────────────────────────────────────────────────────

class Policy:
    """Top-level policy engine: decide(detection) -> Decision.

    Usage:
        pol = Policy(PolicySet.from_dict(json.load(fh)))
        d = pol.decide({"rule_id": "R03", "category": "leak",
                        "severity": "high", "score": 55.0, "stage": "post_call"})
    """

    def __init__(self, policy_set: Optional[PolicySet] = None, *,
                 path: Optional[str] = None,
                 auto_reload: bool = False,
                 reload_poll_s: float = 5.0) -> None:
        self._lock = threading.RLock()
        self._set = (policy_set or PolicySet()).normalized()
        self.path = os.fspath(path) if path else ""
        self._mtime = self._path_mtime()
        self.auto_reload = bool(auto_reload)
        self.reload_poll_s = _to_float(reload_poll_s, 5.0, 0.5, 3600.0)
        self._last_reload_check = 0.0
        # per (rule_id, session) hit tracking for escalate/cooldown/rate
        self._hits: Dict[Tuple[str, str], List[float]] = {}
        self._last_action_ts: Dict[Tuple[str, str], float] = {}
        self._generations: int = 0

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_file(cls, path: str, *, strict: bool = False,
                  auto_reload: bool = True,
                  reload_poll_s: float = 5.0) -> "Policy":
        """Load a policy from JSON; on failure return a permissive
        default (or raise PolicyError when strict)."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            ps = PolicySet.from_dict(data).normalized()
            return cls(ps, path=path, auto_reload=auto_reload,
                       reload_poll_s=reload_poll_s)
        except Exception as exc:
            if strict:
                raise PolicyError(f"cannot load policy {path}: {exc}") from exc
            return cls(PolicySet(name=f"fallback:{os.path.basename(path)}"),
                       path=path, auto_reload=auto_reload)

    def _path_mtime(self) -> float:
        try:
            return os.path.getmtime(self.path) if self.path else 0.0
        except OSError:
            return 0.0

    def maybe_reload(self, force: bool = False) -> bool:
        """Hot-reload from disk if the file changed. Returns True when a
        new policy generation was installed. Never raises."""
        if not self.path:
            return False
        with self._lock:
            now = time.time()
            if not force:
                if now - self._last_reload_check < self.reload_poll_s:
                    return False
            self._last_reload_check = now
            mtime = self._path_mtime()
            if not force and mtime == self._mtime and self._generations > 0:
                return False
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                candidate = PolicySet.from_dict(data).normalized()
            except Exception:
                return False   # keep old policy; file may be mid-write
            self._set = candidate
            self._mtime = self._path_mtime()
            self._generations += 1
            self._hits.clear()
            self._last_action_ts.clear()
            return True

    # -- introspection -----------------------------------------------------

    @property
    def policy_set(self) -> PolicySet:
        with self._lock:
            return self._set

    @property
    def generations(self) -> int:
        with self._lock:
            return self._generations

    def replace_set(self, policy_set: PolicySet) -> None:
        with self._lock:
            self._set = policy_set.normalized()
            self._generations += 1
            self._hits.clear()
            self._last_action_ts.clear()

    # -- matching internals --------------------------------------------------

    def _resolve(self, detection: Dict[str, Any]) -> Tuple[PolicyRule, int, List[str]]:
        """Find the most specific applicable rule. Returns
        (rule, specificity, precedence_chain)."""
        rule_id = str(detection.get("rule_id", "") or "").strip()
        category = str(detection.get("category", "") or "").strip().lower()
        severity = str(detection.get("severity", "") or "").strip().lower()
        stage = str(detection.get("stage", "") or "").strip().lower()
        now = time.time()

        chain: List[str] = []
        candidates: List[Tuple[int, int, int, PolicyRule]] = []
        rules = self._set.active_rules(now)
        for idx, rule in enumerate(rules):
            spec = rule.matches_target(rule_id, category, severity)
            if spec == 0:
                continue
            if not rule.applies_in_stage(stage):
                chain.append(f"skip(stage) {rule.rule_id}")
                continue
            candidates.append((spec, rule.precedence, -idx, rule))
        if candidates:
            # most specific; then highest precedence; then latest declared
            candidates.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
            best = candidates[0][3]
            chain.append("resolved:" + best.rule_id)
            return best, candidates[0][0], chain
        # default rule
        default = PolicyRule(rule_id="*", mode=self._set.default_mode,
                             reason="policy default")
        chain.append("default:*")
        return default, 0, chain

    # -- hit bookkeeping -----------------------------------------------------

    def _key(self, rule_id: str, session_id: str) -> Tuple[str, str]:
        return (rule_id or "*", session_id or "default")

    def _prune_hits(self, key: Tuple[str, str], window_s: float,
                    now: float) -> List[float]:
        stamps = [t for t in self._hits.get(key, [])
                  if now - t <= max(window_s, 1.0)]
        self._hits[key] = stamps
        return stamps

    def _record_hit(self, key: Tuple[str, str], now: float) -> None:
        self._hits.setdefault(key, []).append(now)
        del self._hits[key][:-1000]      # hard bound per key

    # -- main entry ----------------------------------------------------------

    def decide(self, detection: Optional[Dict[str, Any]]) -> Decision:
        """Resolve enforcement for one detection. Never raises; a None or
        garbage detection yields allow with a note."""
        if not isinstance(detection, dict):
            return _allow("no detection payload", ["empty or non-dict detection"])

        # hot reload check (cheap, bounded)
        if self.auto_reload and self.path:
            try:
                self.maybe_reload()
            except Exception:
                pass

        with self._lock:
            rule, spec, chain = self._resolve(detection)
            session_id = str(detection.get("session_id", "") or "default")
            key = self._key(rule.rule_id, session_id)
            now = time.time()
            notes: List[str] = []

            mode = rule.effective_mode(detection)
            if mode != rule.mode:
                notes.append(f"downgraded {rule.mode}->{mode} "
                             "(low confidence or score below floor)")
                chain.append("downgrade")

            # expiry check on the matched rule
            if rule.is_expired(now):
                notes.append("matched rule expired; using default")
                mode = self._set.default_mode
                chain.append("ttl-expired")

            # cooldown: same (rule, session) actioned too recently?
            if rule.cooldown_s > 0:
                last = self._last_action_ts.get(key, 0.0)
                if last and now - last < rule.cooldown_s:
                    notes.append(f"cooldown active "
                                 f"({now - last:.1f}s < {rule.cooldown_s:.1f}s)")
                    return Decision(
                        action="log", mode=Mode.LOG, rule_id=rule.rule_id,
                        matched_rule=rule.to_dict(),
                        precedence_chain=chain, reason="cooldown suppression",
                        notes=notes, suppressed=True, ts=now)

            # rate limit within window
            if rule.rate_limit > 0:
                stamps = self._prune_hits(key, rule.window_s, now)
                if len(stamps) >= rule.rate_limit:
                    notes.append(f"rate limit {rule.rate_limit}/"
                                 f"{rule.window_s:.0f}s reached")
                    return Decision(
                        action="log", mode=Mode.LOG, rule_id=rule.rule_id,
                        matched_rule=rule.to_dict(),
                        precedence_chain=chain, reason="rate limit reached",
                        notes=notes, suppressed=True, ts=now)

            # escalation after repeated hits
            stamps = self._prune_hits(key, rule.window_s, now)
            if rule.escalate_after > 0 and len(stamps) >= rule.escalate_after \
                    and mode not in (Mode.BLOCK, Mode.ESCALATE):
                notes.append(f"{len(stamps)} prior hits >= "
                             f"escalate_after={rule.escalate_after}; escalating")
                mode = Mode.ESCALATE
                chain.append("escalated")

            action = ACTION_FOR_MODE.get(mode, "warn")
            if action not in ("allow", "log"):
                self._record_hit(key, now)
                self._last_action_ts[key] = now

            reason = rule.reason or (f"matched {rule.rule_id} "
                                     f"(specificity={spec})")
            return Decision(
                action=action, mode=mode, rule_id=rule.rule_id,
                matched_rule=rule.to_dict(), precedence_chain=chain,
                reason=reason, notes=notes, suppressed=False, ts=now)

    def decide_batch(self, detections: Sequence[Any]) -> List[Decision]:
        """Decide a list of detections; non-dicts degrade to allow."""
        return [self.decide(d if isinstance(d, dict) else None)
                for d in (detections or [])]

    def strictest(self, decisions: Sequence[Decision]) -> Decision:
        """Pick the strictest decision of a batch (for aggregated gating)."""
        best: Optional[Decision] = None
        for d in (decisions or []):
            if not isinstance(d, Decision):
                continue
            if best is None or Mode.STRICTNESS.get(d.mode, 0) > \
                    Mode.STRICTNESS.get(best.mode, 0):
                best = d
        return best or _allow("empty batch")

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self._set.name,
                "rules": len(self._set.rules),
                "default_mode": self._set.default_mode,
                "generations": self._generations,
                "tracked_keys": len(self._hits),
                "path": self.path,
            }


# ── JSON helpers ─────────────────────────────────────────────────────────

def policy_from_json(text: str) -> PolicySet:
    """Parse a policy from a JSON string; raises PolicyError on garbage
    so callers *choosing* strictness can catch it."""
    try:
        data = json.loads(text)
    except Exception as exc:
        raise PolicyError(f"invalid policy JSON: {exc}") from exc
    return PolicySet.from_dict(data).normalized()


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    # mode parsing
    assert parse_mode("BLOCK") == Mode.BLOCK
    assert parse_mode("escalate") == Mode.ESCALATE
    assert parse_mode("nonsense") == Mode.WARN
    assert parse_mode(None) == Mode.WARN
    assert parse_mode(42) == Mode.WARN

    # rule construction + normalization
    r = PolicyRule(rule_id="R01", mode="block", precedence=200,
                   stages=["post_call", "final"]).normalized()
    assert r.mode == Mode.BLOCK and r.stages == ("post_call", "final")
    assert r.created_at > 0

    # matching + specificity
    assert r.matches_target("R01", "leak", "high") == 4
    assert r.matches_target("R02", "", "") == 0
    assert r.applies_in_stage("post_call")
    assert not r.applies_in_stage("ingest")
    assert not r.applies_in_stage("")
    assert PolicyRule().normalized().applies_in_stage("")   # no stages: any

    cat_rule = PolicyRule(category="leak", mode=Mode.BLOCK).normalized()
    assert cat_rule.matches_target("R99", "leak", "low") == 2
    sev_rule = PolicyRule(severity="critical", mode=Mode.BLOCK).normalized()
    assert sev_rule.matches_target("R99", "format", "critical") == 1

    # specificity resolution: exact id beats category beats default
    ps = PolicySet(rules=[
        PolicyRule(rule_id="*", mode=Mode.WARN),
        PolicyRule(category="leak", mode=Mode.BLOCK, precedence=10),
        PolicyRule(rule_id="R03", mode=Mode.LOG, precedence=1),
    ])
    pol = Policy(ps)
    d = pol.decide({"rule_id": "R03", "category": "leak", "severity": "high"})
    assert d.action == "log", d.to_dict()
    d = pol.decide({"rule_id": "R07", "category": "leak"})
    assert d.action == "block", d.to_dict()
    d = pol.decide({"rule_id": "R07", "category": "format"})
    assert d.action == "warn", d.to_dict()

    # precedence overrides specificity order within same rule_id
    ps2 = PolicySet(rules=[
        PolicyRule(rule_id="R03", mode=Mode.WARN, precedence=1),
        PolicyRule(rule_id="R03", mode=Mode.BLOCK, precedence=99),
    ])
    d = Policy(ps2).decide({"rule_id": "R03"})
    assert d.action == "block", d.to_dict()
    assert d.precedence_chain and "resolved:R03" in d.precedence_chain

    # stage gating
    ps3 = PolicySet(rules=[
        PolicyRule(rule_id="R01", mode=Mode.BLOCK, not_stages=["audit"]),
    ])
    d = Policy(ps3).decide({"rule_id": "R01", "stage": "audit"})
    assert d.action == "warn", d.to_dict()      # falls to default
    d = Policy(ps3).decide({"rule_id": "R01", "stage": "post_call"})
    assert d.action == "block"

    # confidence downgrade
    ps4 = PolicySet(rules=[
        PolicyRule(rule_id="R05", mode=Mode.BLOCK, min_confidence=0.8,
                   downgrade_mode=Mode.WARN),
    ])
    d = Policy(ps4).decide({"rule_id": "R05", "confidence": 0.4})
    assert d.action == "warn", d.to_dict()
    d = Policy(ps4).decide({"rule_id": "R05", "confidence": 0.95})
    assert d.action == "block", d.to_dict()

    # score floor downgrade
    ps5 = PolicySet(rules=[
        PolicyRule(rule_id="R06", mode=Mode.BLOCK, score_floor=30.0,
                   downgrade_mode=Mode.LOG),
    ])
    d = Policy(ps5).decide({"rule_id": "R06", "score": 10.0})
    assert d.action == "log", d.to_dict()

    # cooldown suppression
    ps6 = PolicySet(rules=[
        PolicyRule(rule_id="R07", mode=Mode.BLOCK, cooldown_s=60.0),
    ])
    p6 = Policy(ps6)
    d1 = p6.decide({"rule_id": "R07", "session_id": "s1"})
    d2 = p6.decide({"rule_id": "R07", "session_id": "s1"})
    assert d1.action == "block" and d2.suppressed and d2.action == "log"
    d3 = p6.decide({"rule_id": "R07", "session_id": "s2"})  # other session
    assert d3.action == "block"

    # rate limiting
    ps7 = PolicySet(rules=[
        PolicyRule(rule_id="R08", mode=Mode.WARN, rate_limit=2, window_s=60),
    ])
    p7 = Policy(ps7)
    acts = [p7.decide({"rule_id": "R08"}).action for _ in range(4)]
    assert acts[:2] == ["warn", "warn"] and acts[2] == "log", acts

    # escalation after repeated hits
    ps8 = PolicySet(rules=[
        PolicyRule(rule_id="R09", mode=Mode.WARN, escalate_after=2,
                   window_s=60),
    ])
    p8 = Policy(ps8)
    a1 = p8.decide({"rule_id": "R09"}).action
    a2 = p8.decide({"rule_id": "R09"}).action
    a3 = p8.decide({"rule_id": "R09"})
    assert a1 == "warn" and a2 == "warn" and a3.mode == Mode.ESCALATE

    # TTL expiry
    rule_ttl = PolicyRule(rule_id="R10", mode=Mode.BLOCK, ttl_s=1.0)
    rule_ttl.created_at = time.time() - 10.0
    d = Policy(PolicySet(rules=[rule_ttl])).decide({"rule_id": "R10"})
    # expired rules are filtered before matching -> default applies
    assert d.action == "warn" and "default:*" in d.precedence_chain

    # garbage input never raises
    pol_g = Policy(PolicySet())
    assert pol_g.decide(None).action == "allow"
    assert pol_g.decide("junk").action == "allow"
    assert pol_g.decide({"severity": {"evil": 1}}).action in Mode.ALL

    # decide_batch + strictest
    ds = pol.decide_batch([{"rule_id": "R03"}, None,
                           {"rule_id": "R07", "category": "leak"}])
    assert len(ds) == 3
    assert pol.strictest(ds).mode in Mode.ALL

    # JSON round trip
    js = json.dumps(ps2.to_dict())
    ps_rt = policy_from_json(js)
    assert len(ps_rt.rules) == 2 and ps_rt.rules[1].mode == Mode.BLOCK
    try:
        policy_from_json("{nope")
        raised = False
    except PolicyError:
        raised = True
    assert raised

    # hot reload from file
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp(prefix="enforcex-policy-")
    try:
        ppath = os.path.join(tmp, "policy.json")
        with open(ppath, "w", encoding="utf-8") as fh:
            json.dump({"name": "v1", "default_mode": "warn",
                       "rules": [{"rule_id": "R01", "mode": "log"}]}, fh)
        p = Policy.from_file(ppath, auto_reload=True, reload_poll_s=0.1)
        assert p.decide({"rule_id": "R01"}).action == "log"
        # rewrite policy: R01 becomes block
        time.sleep(0.05)
        with open(ppath, "w", encoding="utf-8") as fh:
            json.dump({"name": "v2", "default_mode": "warn",
                       "rules": [{"rule_id": "R01", "mode": "block"}]}, fh)
        reloaded = p.maybe_reload(force=True)
        assert reloaded and p.generations >= 1
        assert p.decide({"rule_id": "R01"}).action == "block"
        # corrupt file keeps old policy
        with open(ppath, "w", encoding="utf-8") as fh:
            fh.write("{broken")
        assert p.maybe_reload(force=True) is False
        assert p.decide({"rule_id": "R01"}).action == "block"
        # missing file at load -> fallback, no raise
        pf = Policy.from_file(os.path.join(tmp, "missing.json"))
        assert pf.policy_set.name.startswith("fallback")
        # strict load raises
        try:
            Policy.from_file(os.path.join(tmp, "missing.json"), strict=True)
            strict_raised = False
        except PolicyError:
            strict_raised = True
        assert strict_raised
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("policy._self_test: OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
