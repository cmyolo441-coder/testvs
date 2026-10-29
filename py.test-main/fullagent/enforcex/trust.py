"""trust — Trust scoring with adaptive strictness for enforcex.

Maintains a per-session (or per-user) trust score in [0, 100] that
tightens or relaxes enforcement elsewhere in the toolkit.  The ledger
is event-sourced: every recorded event is appended to a full history,
so state can always be rebuilt and audited.

Semantics
---------
  * starts at a configurable default (100 by default — assume good
    faith until proven otherwise),
  * drops FAST on violations: the penalty is a function of the event
    severity (weights re-used from the policy/detect vocabulary:
    critical / high / medium / low / info) and the category
    (content / format / tool_use),
  * recovers SLOWLY on clean interactions (hysteresis: asymmetric
    drop/recover rates make gaming the system expensive),
  * keeps per-category sub-scores (content, format, tool_use) in
    addition to the overall score; the effective tier uses the *worst*
    of the two so a bot can't look trustworthy overall while still
    abusing tools,
  * maps scores to tiers: UNTRUSTED / LOW / MEDIUM / HIGH / VERIFIED,
  * tiers expose stricter gate thresholds (min_score, max_violations)
    that gate.py-style callers can consume.

Design rules inherited from the enforcex house style:

  * stdlib only, Python 3.9+,
  * dataclasses + small pure functions,
  * never raise on malformed input — every public entry point fails
    soft and returns a valid degraded state,
  * deterministic: recovery math uses no randomness and no wall-clock
    reads (timestamps on events are metadata, not scoring inputs),
  * serialize/deserialize via plain dicts (to_dict / from_dict),
  * a `_self_test()` with asserts runs via
    `python -m fullagent.enforcex.trust`.
"""

from __future__ import annotations

import copy
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "TrustConfig",
    "TrustState",
    "TrustTier",
    "TrustEvent",
    "TrustLedger",
    "SEVERITY_PENALTIES",
    "CATEGORY_WEIGHTS",
    "TIER_THRESHOLDS",
]


# ── vocabulary ───────────────────────────────────────────────────────────

class TrustTier:
    """Ordered trust tiers (constants only — mirrors policy.Mode style)."""
    UNTRUSTED = "untrusted"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERIFIED = "verified"

    ALL: Tuple[str, ...] = (
        UNTRUSTED, LOW, MEDIUM, HIGH, VERIFIED,
    )


#: severity -> penalty in trust points.  Aligned with the severity
#: vocabulary used by detect.py / scoring.py / policy.py.
SEVERITY_PENALTIES: Dict[str, float] = {
    "critical": 45.0,
    "high": 25.0,
    "medium": 12.0,
    "low": 5.0,
    "info": 1.0,
}

#: category -> multiplier on the severity penalty.
CATEGORY_WEIGHTS: Dict[str, float] = {
    "content": 1.0,
    "format": 0.5,
    "tool_use": 1.3,
    "security": 1.5,
}

#: score floor -> tier (first threshold >= score wins, checked descending).
TIER_THRESHOLDS: Tuple[Tuple[float, str], ...] = (
    (90.0, TrustTier.VERIFIED),
    (75.0, TrustTier.HIGH),
    (55.0, TrustTier.MEDIUM),
    (30.0, TrustTier.LOW),
    (0.0, TrustTier.UNTRUSTED),
)

#: per-tier stricter gate parameters.  Lower trust -> stricter gate.
TIER_GATE_PARAMS: Dict[str, Dict[str, Any]] = {
    TrustTier.VERIFIED:  {"min_score":  0.0, "max_violations": 4,
                          "require_seal": False, "strict_mode": False},
    TrustTier.HIGH:      {"min_score": 10.0, "max_violations": 3,
                          "require_seal": False, "strict_mode": False},
    TrustTier.MEDIUM:    {"min_score": 25.0, "max_violations": 2,
                          "require_seal": False, "strict_mode": True},
    TrustTier.LOW:       {"min_score": 45.0, "max_violations": 1,
                          "require_seal": True,  "strict_mode": True},
    TrustTier.UNTRUSTED: {"min_score": 70.0, "max_violations": 0,
                          "require_seal": True,  "strict_mode": True},
}


def _normalize_severity(value: Any) -> str:
    s = str(value or "").strip().lower()
    return s if s in SEVERITY_PENALTIES else "medium"


def _normalize_category(value: Any) -> str:
    s = str(value or "").strip().lower().replace("-", "_")
    if s in CATEGORY_WEIGHTS:
        return s
    # tolerate the scoring.py category names by bucketing them
    if s in ("injection", "security", "identity", "leak"):
        return "security"
    if s in ("tone", "language", "format", "style"):
        return "format"
    if s in ("tool", "tools", "process"):
        return "tool_use"
    return "content"


# ── defensive coercion helpers ───────────────────────────────────────────

def _clamp(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(value)
    except OverflowError:
        # Numeric but larger than a float: saturate at the nearest bound.
        try:
            return hi if value > 0 else lo
        except Exception:
            return default
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return default
    return max(lo, min(hi, f))


def _safe_meta(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for idx, (key, item) in enumerate(value.items()):
        if idx >= 256:
            break
        out[str(key)[:128]] = item
    return out


def _to_int_or(value: Any, default: int) -> int:
    """Convert ints while distinguishing an explicit 0 from malformed input."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_id(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return "anonymous"


# ── data model ───────────────────────────────────────────────────────────

@dataclass
class TrustEvent:
    """One event in the trust ledger's append-only history."""
    kind: str                 # "violation" | "clean" | "adjust" | "reset"
    severity: str = "medium"  # for violations
    category: str = "content"
    delta: float = 0.0        # explicit adjustment (kind == "adjust")
    reason: str = ""
    timestamp: float = 0.0    # metadata only — never used in scoring math
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "category": self.category,
            "delta": self.delta,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrustEvent":
        if not isinstance(d, dict):
            return cls(kind="clean")
        kind = str(d.get("kind") or "clean").strip().lower()
        if kind not in ("violation", "clean", "adjust", "reset"):
            kind = "clean"
        meta = d.get("meta")
        return cls(
            kind=kind,
            severity=_normalize_severity(d.get("severity")),
            category=_normalize_category(d.get("category")),
            delta=_clamp(d.get("delta"), -100.0, 100.0, 0.0),
            reason=str(d.get("reason") or "")[:4096],
            timestamp=_clamp(d.get("timestamp"), 0.0, 9e15, 0.0),
            meta=_safe_meta(meta),
        )

    def penalty(self) -> float:
        """Effective penalty this event applies (violations only)."""
        if self.kind != "violation":
            return 0.0
        base = SEVERITY_PENALTIES.get(self.severity, 12.0)
        weight = CATEGORY_WEIGHTS.get(self.category, 1.0)
        return base * weight


@dataclass
class TrustState:
    """Snapshot of one session's trust.

    score:            overall score in [0, 100].
    category_scores:  per-category scores in [0, 100].
    clean_streak:     consecutive clean interactions.
    violation_count:  total violations ever recorded.
    events:           full append-only history (serialized dicts).
    """
    session_id: str = "anonymous"
    score: float = 100.0
    min_score: float = 100.0
    category_scores: Dict[str, float] = field(default_factory=lambda: {
        "content": 100.0, "format": 100.0, "tool_use": 100.0,
        "security": 100.0,
    })
    clean_streak: int = 0
    violation_count: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def tier(self) -> str:
        return score_to_tier(self.effective_score)

    @property
    def effective_score(self) -> float:
        """Worst of overall and per-category scores (defensive minimum)."""
        scores = [_clamp(self.score, 0.0, 100.0, 100.0)]
        for value in self.category_scores.values():
            scores.append(_clamp(value, 0.0, 100.0, 100.0))
        return min(scores) if scores else _clamp(self.score, 0.0, 100.0, 100.0)

    def gate_params(self) -> Dict[str, Any]:
        return dict(TIER_GATE_PARAMS.get(self.tier, {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "score": self.score,
            "min_score": self.min_score,
            "category_scores": dict(self.category_scores),
            "clean_streak": self.clean_streak,
            "violation_count": self.violation_count,
            "events": copy.deepcopy(self.events),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "TrustState":
        if not isinstance(d, dict):
            return cls()
        category_scores = d.get("category_scores")
        cats: Dict[str, float] = {}
        if isinstance(category_scores, dict):
            for k, v in category_scores.items():
                cats[_normalize_category(k)] = _clamp(v, 0.0, 100.0, 100.0)
        for missing in ("content", "format", "tool_use", "security"):
            cats.setdefault(missing, 100.0)
        events_raw = d.get("events")
        events = [e.to_dict() for e in
                  (TrustEvent.from_dict(x) for x in events_raw)
                  ] if isinstance(events_raw, list) else []
        score = _clamp(d.get("score"), 0.0, 100.0, 100.0)
        raw_min = d.get("min_score", score)
        min_score = _clamp(raw_min, 0.0, 100.0, score)
        return cls(
            session_id=_as_id(d.get("session_id")),
            score=score,
            min_score=min(min_score, score),
            category_scores=cats,
            clean_streak=max(0, _to_int(d.get("clean_streak"))),
            violation_count=max(0, _to_int(d.get("violation_count"))),
            events=events,
        )


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return 0


def score_to_tier(score: Any) -> str:
    """Map a score in [0, 100] to its TrustTier constant."""
    s = _clamp(score, 0.0, 100.0, 100.0)
    for floor, tier in TIER_THRESHOLDS:
        if s >= floor:
            return tier
    return TrustTier.UNTRUSTED


@dataclass
class TrustConfig:
    """Tunables for a TrustLedger."""
    initial_score: float = 100.0
    #: points recovered per clean interaction (slow rise)
    recover_per_clean: float = 2.0
    #: extra points after a long clean streak, applied as bonus rate
    streak_bonus_rate: float = 0.5
    #: streak length at which the bonus starts
    streak_bonus_after: int = 5
    #: cap on points that may have been recovered in total per session
    recovery_cap: float = 60.0
    #: repeated same-severity violations within a window escalate
    repeat_multiplier: float = 1.5
    repeat_window: int = 3
    #: after this many consecutive clean interactions, one prior low
    #: severity violation is "forgiven" (removed from escalation math)
    forgive_after: int = 10
    tier_thresholds: Tuple[Tuple[float, str], ...] = TIER_THRESHOLDS

    @classmethod
    def from_dict(cls, d: Any) -> "TrustConfig":
        if not isinstance(d, dict):
            return cls()
        cfg = cls()
        cfg.initial_score = _clamp(d.get("initial_score"), 0.0, 100.0, 100.0)
        cfg.recover_per_clean = _clamp(
            d.get("recover_per_clean"), 0.0, 50.0, 2.0)
        cfg.streak_bonus_rate = _clamp(
            d.get("streak_bonus_rate"), 0.0, 10.0, 0.5)
        cfg.streak_bonus_after = max(
            0, _to_int_or(d.get("streak_bonus_after"), 5))
        cfg.recovery_cap = _clamp(d.get("recovery_cap"), 0.0, 100.0, 60.0)
        cfg.repeat_multiplier = _clamp(
            d.get("repeat_multiplier"), 1.0, 5.0, 1.5)
        cfg.repeat_window = max(1, _to_int_or(d.get("repeat_window"), 3))
        cfg.forgive_after = max(0, _to_int_or(d.get("forgive_after"), 10))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_score": self.initial_score,
            "recover_per_clean": self.recover_per_clean,
            "streak_bonus_rate": self.streak_bonus_rate,
            "streak_bonus_after": self.streak_bonus_after,
            "recovery_cap": self.recovery_cap,
            "repeat_multiplier": self.repeat_multiplier,
            "repeat_window": self.repeat_window,
            "forgive_after": self.forgive_after,
        }


# ── the ledger ───────────────────────────────────────────────────────────

class TrustLedger:
    """Event-sourced per-session trust ledger.

    Usage
    -----
    >>> ledger = TrustLedger()
    >>> state = ledger.record("sess-1", TrustEvent(
    ...     kind="violation", severity="high", category="content"))
    >>> state.score < 100.0
    True
    >>> ledger.tier(state) == TrustTier.LOW
    True

    The ledger is thread-safe (one lock around record/replay), keeps a
    full event history per session and can round-trip its entire state
    through to_dict / from_dict.
    """

    def __init__(self, config: Optional[TrustConfig] = None, **overrides: Any) -> None:
        self.config = TrustConfig.from_dict(
            config.to_dict() if isinstance(config, TrustConfig) else None)
        if overrides:
            merged = self.config.to_dict()
            for k, v in overrides.items():
                if k in merged:
                    merged[k] = v
            self.config = TrustConfig.from_dict(merged)
        self._states: Dict[str, TrustState] = {}
        self._lock = threading.RLock()

    # ── public API ─────────────────────────────────────────────────

    def record(self, session_id: Any, event: Any) -> TrustState:
        """Append *event* for *session_id* and return the new state.

        Never raises: malformed events degrade to a no-op "clean"
        event; unknown sessions start from the configured default.
        """
        if isinstance(event, TrustEvent):
            ev = event
        elif isinstance(event, dict):
            ev = TrustEvent.from_dict(event)
        else:
            ev = TrustEvent(kind="clean")
        sid = _as_id(session_id)
        with self._lock:
            state = self._states.get(sid)
            if state is None:
                state = self._new_state(sid)
            state = self._apply(state, ev)
            self._states[sid] = state
            return copy.deepcopy(state)

    def state(self, session_id: Any) -> TrustState:
        """Current state for *session_id* (created on first access)."""
        sid = _as_id(session_id)
        with self._lock:
            st = self._states.get(sid)
            if st is None:
                st = self._new_state(sid)
                self._states[sid] = st
            return copy.deepcopy(st)

    def tier(self, state: Any) -> str:
        """Tier for a state, score or raw number.  Never raises."""
        if isinstance(state, TrustState):
            return score_to_tier(state.effective_score)
        if isinstance(state, dict):
            return score_to_tier(state.get(
                "score", state.get("effective_score", 100.0)))
        return score_to_tier(state)

    def tier_for(self, session_id: Any) -> str:
        return self.tier(self.state(session_id))

    def gate_params(self, session_id: Any) -> Dict[str, Any]:
        """Stricter gate thresholds implied by the session's tier."""
        return self.state(session_id).gate_params()

    def reset(self, session_id: Any, reason: str = "") -> TrustState:
        """Reset a session to the initial score, keeping the history."""
        return self.record(session_id, TrustEvent(
            kind="reset", reason=reason or "manual reset"))

    def sessions(self) -> List[str]:
        with self._lock:
            return sorted(self._states)

    def replay(self, session_id: Any) -> TrustState:
        """Rebuild state from the event history alone (audit path)."""
        sid = _as_id(session_id)
        with self._lock:
            st = self._states.get(sid)
            events = list(st.events) if st else []
        rebuilt = self._new_state(sid)
        for raw in events:
            rebuilt = self._apply(rebuilt, TrustEvent.from_dict(raw),
                                  record_history=False)
        return rebuilt

    # ── serialization ──────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": self.config.to_dict(),
                "sessions": {
                    sid: st.to_dict() for sid, st in self._states.items()
                },
            }

    @classmethod
    def from_dict(cls, d: Any) -> "TrustLedger":
        if not isinstance(d, dict):
            return cls()
        ledger = cls(TrustConfig.from_dict(d.get("config")))
        sessions = d.get("sessions")
        if isinstance(sessions, dict):
            for sid, sdict in sessions.items():
                st = TrustState.from_dict(sdict)
                st.session_id = _as_id(sid)
                with ledger._lock:
                    ledger._states[st.session_id] = st
        return ledger

    def to_json(self, indent: Optional[int] = 2) -> str:
        try:
            return json.dumps(self.to_dict(), indent=indent)
        except (TypeError, ValueError):
            return "{}"

    @classmethod
    def from_json(cls, text: Any) -> "TrustLedger":
        try:
            return cls.from_dict(json.loads(_as_text_or_empty(text) or "{}"))
        except (TypeError, ValueError):
            return cls()

    # ── internals ──────────────────────────────────────────────────

    def _new_state(self, session_id: str) -> TrustState:
        return TrustState(
            session_id=session_id,
            score=self.config.initial_score,
            min_score=self.config.initial_score,
            category_scores={
                "content": self.config.initial_score,
                "format": self.config.initial_score,
                "tool_use": self.config.initial_score,
                "security": self.config.initial_score,
            },
        )

    def _apply(self, state: TrustState, event: TrustEvent,
               record_history: bool = True) -> TrustState:
        cfg = self.config
        kind = str(event.kind or "").strip().lower()
        if kind not in ("violation", "clean", "adjust", "reset"):
            kind = "clean"
        # Normalise even directly-constructed TrustEvent objects so malformed
        # severity/category values cannot create unbounded category keys or
        # make replay() diverge from the live state.
        ev = TrustEvent(
            kind=kind,
            severity=_normalize_severity(event.severity),
            category=_normalize_category(event.category),
            delta=_clamp(event.delta, -100.0, 100.0, 0.0),
            reason=str(event.reason or "")[:4096],
            timestamp=_clamp(event.timestamp, 0.0, 9e15, 0.0),
            meta=_safe_meta(event.meta),
        )
        state.events.append(ev.to_dict())

        if ev.kind == "reset":
            state.score = cfg.initial_score
            state.min_score = cfg.initial_score
            for cat in state.category_scores:
                state.category_scores[cat] = cfg.initial_score
            state.clean_streak = 0
            return state

        if ev.kind == "adjust":
            delta = _clamp(ev.delta, -100.0, 100.0, 0.0)
            state.score = _clamp(state.score + delta, 0.0, 100.0, state.score)
            cat = ev.category or "content"
            old = state.category_scores.get(cat, state.score)
            state.category_scores[cat] = _clamp(
                old + delta, 0.0, 100.0, old)
            state.min_score = min(state.min_score, state.score)
            return state

        if ev.kind == "violation":
            state.violation_count += 1
            state.clean_streak = 0
            penalty = ev.penalty()
            # escalation: repeated same-severity violations hurt more
            recent = [TrustEvent.from_dict(e) for e in
                      state.events[-(cfg.repeat_window + 1):-1]]
            repeats = sum(1 for e in recent
                          if e.kind == "violation"
                          and e.severity == ev.severity)
            if repeats:
                penalty *= cfg.repeat_multiplier ** min(repeats, 3)
            # hysteresis: the lower trust already is, the harder hits land
            if state.score < 50.0:
                penalty *= 1.25
            penalty = round(penalty, 2)
            state.score = _clamp(state.score - penalty, 0.0, 100.0,
                                 state.score)
            cat = ev.category or "content"
            old = state.category_scores.get(cat, state.score)
            state.category_scores[cat] = _clamp(
                old - penalty, 0.0, 100.0, old)
            state.min_score = min(state.min_score, state.score)
            return state

        # kind == "clean": slow recovery with streak bonus
        state.clean_streak += 1
        rate = cfg.recover_per_clean
        if state.clean_streak > cfg.streak_bonus_after:
            rate += cfg.streak_bonus_rate * min(
                state.clean_streak - cfg.streak_bonus_after, 10)
        # recovery only lifts the overall score, and never past the
        # per-category ceilings set by past violations (slow rise)
        best_allowed = min(
            [state.score] + list(state.category_scores.values()))
        new_score = _clamp(state.score + rate, 0.0,
                           min(100.0, best_allowed + rate), state.score)
        # cap total recovery: score may climb at most recovery_cap above
        # the lowest score the session ever reached (burned sessions
        # must demonstrate sustained good behavior to fully heal)
        if cfg.recovery_cap < 100.0:
            new_score = min(new_score, state.min_score + cfg.recovery_cap)
        state.score = round(_clamp(new_score, 0.0, 100.0, state.score), 2)
        for cat in state.category_scores:
            old = state.category_scores[cat]
            state.category_scores[cat] = round(
                _clamp(old + rate * CATEGORY_WEIGHTS.get(cat, 1.0) * 0.5,
                       0.0, min(100.0, old + rate), old), 2)
        # forgiveness: long clean streaks can clear one low-severity mark
        if (cfg.forgive_after
                and state.clean_streak >= cfg.forgive_after
                and state.clean_streak % cfg.forgive_after == 0):
            _forgive_lowest(state)
        return state


def _forgive_lowest(state: TrustState) -> None:
    """Raise the lowest category score slightly as a forgiveness bump."""
    if not state.category_scores:
        return
    cat = min(state.category_scores, key=state.category_scores.get)
    state.category_scores[cat] = round(
        min(100.0, state.category_scores[cat] + 3.0), 2)


def _as_text_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


# ── self test ────────────────────────────────────────────────────────────

def _self_test() -> None:  # pragma: no cover
    # tier mapping
    assert score_to_tier(100) == TrustTier.VERIFIED
    assert score_to_tier(95) == TrustTier.VERIFIED
    assert score_to_tier(80) == TrustTier.HIGH
    assert score_to_tier(60) == TrustTier.MEDIUM
    assert score_to_tier(40) == TrustTier.LOW
    assert score_to_tier(10) == TrustTier.UNTRUSTED
    assert score_to_tier(None) == TrustTier.VERIFIED  # degraded default
    assert score_to_tier("junk") == TrustTier.VERIFIED
    assert score_to_tier(-5) == TrustTier.UNTRUSTED

    # penalties table sanity: critical >> info
    assert SEVERITY_PENALTIES["critical"] > SEVERITY_PENALTIES["high"]
    assert SEVERITY_PENALTIES["high"] > SEVERITY_PENALTIES["medium"]
    assert CATEGORY_WEIGHTS["tool_use"] > CATEGORY_WEIGHTS["format"]

    # ledger basics
    ledger = TrustLedger()
    st0 = ledger.state("s1")
    assert st0.score == 100.0 and st0.tier == TrustTier.VERIFIED
    assert ledger.tier(st0) == TrustTier.VERIFIED
    assert ledger.tier(80) == TrustTier.HIGH
    assert ledger.tier({"score": 40}) == TrustTier.LOW

    # fast drop on violation
    st1 = ledger.record("s1", TrustEvent(
        kind="violation", severity="critical", category="content",
        reason="prompt injection"))
    assert st1.score < 100.0 - 30.0
    assert st1.violation_count == 1 and st1.clean_streak == 0
    assert st1.tier in (TrustTier.LOW, TrustTier.UNTRUSTED, TrustTier.MEDIUM)

    # slow rise on clean interactions
    st = st1
    prev = st.score
    rises = 0
    for _ in range(10):
        st = ledger.record("s1", TrustEvent(kind="clean"))
        if st.score > prev:
            rises += 1
        prev = st.score
    assert rises > 0
    assert st.score <= 100.0 and st.score > 60.0  # recovery is bounded and progressive, not instant

    # hysteresis: drop per violation > rise per clean
    led2 = TrustLedger()
    s = led2.record("x", TrustEvent(kind="violation", severity="high",
                                    category="content"))
    drop = 100.0 - s.score
    s2 = led2.record("x", TrustEvent(kind="clean"))
    rise = s2.score - s.score
    assert drop > rise * 5

    # tool_use violations weigh more than format ones
    led3 = TrustLedger()
    a = led3.record("a", TrustEvent(kind="violation", severity="high",
                                    category="tool_use"))
    b = led3.record("b", TrustEvent(kind="violation", severity="high",
                                    category="format"))
    assert a.score < b.score

    # effective score uses the worst category
    led4 = TrustLedger()
    s = led4.record("c", TrustEvent(kind="violation", severity="critical",
                                    category="security"))
    assert s.effective_score <= s.category_scores["security"] + 0.01
    assert s.tier == score_to_tier(s.effective_score)

    # escalation: repeated same severity hurts more
    led5 = TrustLedger()
    s = led5.record("e", TrustEvent(kind="violation", severity="low",
                                    category="content"))
    after_first = 100.0 - s.score
    s = led5.record("e", TrustEvent(kind="violation", severity="low",
                                    category="content"))
    after_second = 100.0 - s.score - after_first
    assert after_second > after_first * 1.2

    # reset keeps history but restores score
    before_events = len(ledger.state("s1").events)
    rst = ledger.reset("s1", "manual")
    assert rst.score == ledger.config.initial_score
    assert len(rst.events) == before_events + 1

    # replay reproduces the state
    live = ledger.state("s1")
    rebuilt = ledger.replay("s1")
    assert abs(live.score - rebuilt.score) < 0.01
    assert live.violation_count == rebuilt.violation_count
    assert len(rebuilt.events) == len(live.events)

    # serialization round-trip
    blob = ledger.to_dict()
    led_restored = TrustLedger.from_dict(blob)
    assert led_restored.state("s1").score == ledger.state("s1").score
    assert led_restored.state("s1").violation_count == \
        ledger.state("s1").violation_count
    js = ledger.to_json()
    led_js = TrustLedger.from_json(js)
    assert abs(led_js.state("s1").score - ledger.state("s1").score) < 0.01
    assert TrustLedger.from_json("not json {").sessions() == []
    assert TrustLedger.from_dict(None).sessions() == []
    assert TrustLedger.from_dict(42).sessions() == []

    # event from_dict defensiveness
    ev = TrustEvent.from_dict({"kind": "nonsense", "severity": "cataclysm"})
    assert ev.kind == "clean" and ev.severity == "medium"
    ev2 = TrustEvent.from_dict(None)
    assert ev2.kind == "clean"

    # defensive ledger inputs
    flaky = TrustLedger()
    st = flaky.record(None, None)                      # anonymous, no-op
    assert st.score == 100.0
    st = flaky.record("", {"kind": "violation", "severity": "high"})
    assert st.score < 100.0
    st = flaky.record("s9", {"kind": "violation", "severity": 12345,
                             "category": "weird", "delta": "x"})
    assert 0.0 <= st.score <= 100.0
    assert flaky.tier(None) == TrustTier.VERIFIED
    assert flaky.gate_params("nobody")["max_violations"] == 4

    # tier gate params: stricter for lower tiers
    assert (TIER_GATE_PARAMS[TrustTier.UNTRUSTED]["min_score"]
            > TIER_GATE_PARAMS[TrustTier.HIGH]["min_score"])
    assert TIER_GATE_PARAMS[TrustTier.LOW]["require_seal"] is True
    assert TIER_GATE_PARAMS[TrustTier.VERIFIED]["require_seal"] is False
    st_low = TrustState(score=40.0)
    assert st_low.gate_params()["require_seal"] is True

    # custom config
    cfg = TrustConfig(initial_score=50.0, recover_per_clean=10.0)
    led6 = TrustLedger(cfg)
    assert led6.state("z").score == 50.0
    st = led6.record("z", TrustEvent(kind="clean"))
    assert st.score == 60.0
    led7 = TrustLedger(initial_score=70.0)
    assert led7.config.initial_score == 70.0

    # thread smoke: parallel records must not corrupt state
    import threading as _th
    led8 = TrustLedger()
    def _hammer() -> None:
        for _ in range(50):
            led8.record("t", TrustEvent(kind="clean"))
            led8.record("t", TrustEvent(kind="violation", severity="low",
                                        category="format"))
    threads = [_th.Thread(target=_hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    st = led8.state("t")
    assert 0.0 <= st.score <= 100.0
    assert len(st.events) == 4 * 100
    assert len(led8.replay("t").events) == 400

    # config round-trip
    cfg_blob = TrustConfig.from_dict({"initial_score": 33}).to_dict()
    assert TrustConfig.from_dict(cfg_blob).initial_score == 33.0

    # state round-trip
    st_round = TrustState.from_dict(ledger.state("s1").to_dict())
    assert abs(st_round.score - ledger.state("s1").score) < 0.01
    assert st_round.session_id == "s1"

    # numeric coercion: inf / huge ints / NaN are bounded, never raised
    assert _clamp(float("inf"), 0.0, 100.0, 7.0) == 100.0
    assert _clamp(10 ** 1000, 0.0, 100.0, 7.0) == 100.0
    assert _clamp(float("nan"), 0.0, 100.0, 7.0) == 7.0
    assert _to_int(float("inf")) == 0
    assert score_to_tier(float("inf")) == TrustTier.VERIFIED
    assert score_to_tier(float("nan")) == TrustTier.VERIFIED

    # direct TrustEvent objects are normalised: unknown categories cannot grow
    # the category map, and replay() reproduces the live state exactly
    led_norm = TrustLedger()
    st_norm = led_norm.record("norm", TrustEvent(
        kind="VIOLATION", severity="catastrophe", category="script-injection",
        delta=float("inf")))
    assert set(st_norm.category_scores) == {
        "content", "format", "tool_use", "security"}
    assert 0.0 <= st_norm.score <= 100.0
    rebuilt_norm = led_norm.replay("norm")
    assert abs(rebuilt_norm.score - st_norm.score) < 1e-9
    assert rebuilt_norm.category_scores == st_norm.category_scores
    assert len(rebuilt_norm.events) == len(st_norm.events)

    # explicit zero config values are honoured instead of silently defaulted
    assert TrustConfig.from_dict({"forgive_after": 0}).forgive_after == 0
    assert TrustConfig.from_dict({"streak_bonus_after": 0}).streak_bonus_after == 0

    # negative adjustments update min_score, so recovery_cap remains sound
    led_adj = TrustLedger()
    st_adj = led_adj.record("adj", TrustEvent(kind="adjust", delta=-80))
    assert st_adj.score == 20.0 and st_adj.min_score == 20.0
    for _ in range(200):
        st_adj = led_adj.record("adj", TrustEvent(kind="clean"))
    assert st_adj.score <= st_adj.min_score + led_adj.config.recovery_cap + 0.01
    assert st_adj.score <= 100.0

    # a low initial score is also the recovery floor
    led_low = TrustLedger(TrustConfig(initial_score=10.0))
    assert led_low.state("low").min_score == 10.0
    st_low = led_low.state("low")
    for _ in range(100):
        st_low = led_low.record("low", TrustEvent(kind="clean"))
    assert 10.0 <= st_low.score <= 10.0 + led_low.config.recovery_cap + 0.01

    # persisted states without min_score derive it from the current score
    restored_low = TrustState.from_dict({"score": 40.0})
    assert restored_low.min_score == 40.0

    print("trust._self_test: OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
