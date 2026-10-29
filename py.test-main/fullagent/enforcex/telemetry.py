"""telemetry — Metrics, rates, per-rule adherence and an event bus.

An in-memory + file-backed metrics registry for the enforcement gate:

  * counters      — monotonically increasing ints (inc / dec)
  * gauges        — last-value-wins floats (set)
  * histograms    — value observations with bucket math and percentiles
  * per-rule adherence — hits / checks per rule id -> adherence rate
  * per-session stats  — evaluations, violations, retries per session
  * sliding-window rates — events/second over a bounded time window
  * snapshots     — full export to a plain dict / JSON file
  * event bus     — subscribe(topic, fn) / emit(topic, payload) so
    sibling modules (gate, dashboard, cli) can hook into live events

Design notes
------------
* Everything is thread-safe: one re-entrant lock guards the registry,
  subscriber lists are guarded separately so an emit can never
  dead-lock against a snapshot.
* Subscriber callbacks are fire-and-forget: an exception in a
  subscriber is swallowed (recorded as a counter) and never propagates
  to the emitter.
* File persistence is opt-in and best-effort: save_snapshot() writes
  atomically (tempfile + os.replace), load_snapshot() fails soft and
  merges what it can read.
* All public functions tolerate garbage input and never raise.
Stdlib only, Python 3.9+.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "Histogram", "RateWindow", "EventBus", "Telemetry",
    "normalize_topic",
]

_SCHEMA_VERSION = 1


# ── small helpers ────────────────────────────────────────────────────────

def normalize_topic(topic: Any) -> str:
    """Coerce anything into a clean topic string ('gate.decision')."""
    try:
        t = str(topic or "").strip().lower()
        if len(t) > 256:
            t = t[:256]
        parts = [p for p in t.split() if p]
        return ".".join(parts) if parts else "generic"
    except Exception:
        return "generic"


def _atomic_write_json(path: str, payload: Any) -> str:
    """Write JSON atomically; returns path. Raises OSError on failure."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tele-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


# ── histogram ────────────────────────────────────────────────────────────

@dataclass
class Histogram:
    """Bounded histogram with configurable bucket edges.

    Values below the lowest edge land in an implicit underflow bucket,
    values above the highest edge in an overflow bucket; percentile
    math interpolates linearly between bucket edges so p50/p95 stay
    meaningful even with coarse buckets.
    """
    name: str
    edges: Tuple[float, ...] = (10.0, 25.0, 50.0, 75.0, 90.0, 100.0)
    max_samples: int = 10_000
    buckets: Dict[str, int] = field(default_factory=dict)
    samples: List[float] = field(default_factory=list)
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def __post_init__(self) -> None:
        self.name = str(self.name or "h")[:256]
        cleaned: List[float] = []
        for edge in (self.edges or (10.0,)):
            try:
                value = float(edge)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                cleaned.append(value)
        if not cleaned:
            cleaned = [10.0]
        cleaned = sorted(set(cleaned))[:4_096]
        self.edges = tuple(cleaned)
        try:
            samples_cap = int(self.max_samples)
        except (TypeError, ValueError, OverflowError):
            samples_cap = 10_000
        self.max_samples = max(1, min(samples_cap, 1_000_000))
        self._lock = threading.RLock()

    # -- recording ---------------------------------------------------------

    def observe(self, value: Any) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if math.isnan(v) or math.isinf(v):
            return
        with self._lock:
            self.count += 1
            self.total += v
            self.minimum = min(self.minimum, v)
            self.maximum = max(self.maximum, v)
            label = self._bucket_label(v)
            self.buckets[label] = self.buckets.get(label, 0) + 1
            self.samples.append(v)
            if len(self.samples) > self.max_samples:
                # trim oldest; aggregate totals remain correct
                drop = len(self.samples) - self.max_samples
                del self.samples[:drop]

    def _bucket_label(self, v: float) -> str:
        if not self.edges:
            return "all"
        if v < self.edges[0]:
            return f"<{self.edges[0]:g}"
        for i in range(len(self.edges) - 1):
            if self.edges[i] <= v < self.edges[i + 1]:
                return f"[{self.edges[i]:g},{self.edges[i + 1]:g})"
        return f">={self.edges[-1]:g}"

    # -- statistics --------------------------------------------------------

    def mean(self) -> Optional[float]:
        with self._lock:
            if not self.count:
                return None
            return self.total / self.count

    def min(self) -> Optional[float]:
        with self._lock:
            return None if self.count == 0 else self.minimum

    def max(self) -> Optional[float]:
        with self._lock:
            return None if self.count == 0 else self.maximum

    def percentile(self, p: float) -> Optional[float]:
        """Interpolated percentile from the kept samples. p in [0,100]."""
        try:
            pf = float(p)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(pf):
            return None
        pf = max(0.0, min(100.0, pf))
        with self._lock:
            data = sorted(self.samples)
        if not data:
            return None
        if len(data) == 1:
            return data[0]
        rank = (pf / 100.0) * (len(data) - 1)
        lo = int(math.floor(rank))
        hi = min(lo + 1, len(data) - 1)
        frac = rank - lo
        return data[lo] * (1.0 - frac) + data[hi] * frac

    def percentiles(self, ps: Sequence[float] = (50, 90, 95, 99)
                    ) -> Dict[str, Optional[float]]:
        try:
            source = list(ps) if ps is not None else [50]
        except TypeError:
            source = [50]
        out: Dict[str, Optional[float]] = {}
        for p in source[:64]:
            try:
                key = f"p{int(p)}"
            except (TypeError, ValueError, OverflowError):
                continue
            out[key] = self.percentile(p)
        return out or {"p50": self.percentile(50)}

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "count": self.count,
                "mean": round(self.total / self.count, 3) if self.count else None,
                "min": self.minimum if self.count else None,
                "max": self.maximum if self.count else None,
                "buckets": dict(sorted(self.buckets.items())),
                "percentiles": {
                    k: (round(v, 3) if v is not None else None)
                    for k, v in self.percentiles().items()
                },
            }

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "edges": list(self.edges),
                "max_samples": self.max_samples,
                "buckets": dict(self.buckets),
                "samples": [round(s, 4) for s in self.samples[-1000:]],
                "count": self.count,
                "total": round(self.total, 4),
                "minimum": self.minimum if self.count else None,
                "maximum": self.maximum if self.count else None,
            }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Histogram":
        if not isinstance(d, dict):
            return cls(name="h")
        try:
            edges = tuple(float(e) for e in d.get("edges", ()))
        except (TypeError, ValueError, OverflowError):
            edges = ()
        try:
            max_samples = int(d.get("max_samples", 10_000))
        except (TypeError, ValueError, OverflowError):
            max_samples = 10_000
        h = cls(
            name=str(d.get("name", "h")),
            edges=edges,
            max_samples=max_samples,
        )
        saved_buckets: Dict[str, int] = {}
        for k, v in (d.get("buckets") or {}).items():
            try:
                saved_buckets[str(k)] = int(v)
            except (TypeError, ValueError, OverflowError):
                continue
        samples = d.get("samples")
        if isinstance(samples, (list, tuple)):
            for s in samples[:h.max_samples]:
                h.observe(s)
        # restore the true aggregates beyond the trimmed sample window
        try:
            h.count = max(h.count, int(d.get("count", 0)))
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            if "total" in d:
                h.total = float(d.get("total", 0.0))
            if not math.isfinite(h.total):
                h.total = sum(h.samples)
        except (TypeError, ValueError, OverflowError):
            h.total = sum(h.samples)
        try:
            if h.count:
                low = float(d.get("minimum"))
                high = float(d.get("maximum"))
                if math.isfinite(low):
                    h.minimum = low
                if math.isfinite(high):
                    h.maximum = high
        except (TypeError, ValueError, OverflowError):
            pass
        if not h.count:
            h.minimum = math.inf
            h.maximum = -math.inf
        # Snapshot bucket counts are authoritative; replaying the samples above
        # must not double-count them.
        if saved_buckets:
            h.buckets = saved_buckets
        return h


# ── sliding-window rate ──────────────────────────────────────────────────

class RateWindow:
    """Events-per-second over a sliding time window."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window = max(1.0, float(window_seconds))
        self._events: List[float] = []       # timestamps
        self._lock = threading.RLock()

    def record(self, ts: Optional[float] = None, n: int = 1) -> None:
        now = float(ts) if ts is not None else time.time()
        n = max(1, min(1000, int(n)))
        with self._lock:
            for _ in range(n):
                self._events.append(now)
            self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window
        if self._events and cutoff > self._events[0]:
            # events are appended in order; drop the prefix
            i = 0
            events = self._events
            while i < len(events) and events[i] < cutoff:
                i += 1
            self._events = events[i:]

    def rate(self) -> float:
        """Events per second over the window (0.0 when empty)."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if not self._events:
                return 0.0
            span = max(self.window, now - self._events[0])
            return len(self._events) / max(span, 1e-9)

    def count(self) -> int:
        with self._lock:
            self._prune(time.time())
            return len(self._events)

    def clear(self) -> None:
        with self._lock:
            self._events = []

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"window": self.window, "events": list(self._events)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RateWindow":
        rw = cls(window_seconds=float(d.get("window", 60.0)))
        try:
            rw._events = [float(t) for t in d.get("events", ())]
        except (TypeError, ValueError):
            rw._events = []
        return rw


# ── event bus ────────────────────────────────────────────────────────────

Subscriber = Callable[[Dict[str, Any]], None]


class EventBus:
    """Minimal synchronous pub/sub. Topics are dotted strings; subscribing
    to ``"gate"`` also receives ``"gate.decision"`` events (prefix match
    on whole segments). A subscribe of ``""`` or ``"*"`` receives all."""

    def __init__(self) -> None:
        self._subs: "OrderedDict[int, Tuple[str, Subscriber]]" = OrderedDict()
        self._next_id = 0
        self._lock = threading.RLock()
        self._dropped = 0

    def subscribe(self, topic: str, fn: Subscriber) -> int:
        """Register fn for a topic; returns a handle for unsubscribe."""
        if not callable(fn):
            return -1
        with self._lock:
            self._next_id += 1
            self._subs[self._next_id] = (normalize_topic(topic), fn)
            return self._next_id

    def unsubscribe(self, handle: int) -> bool:
        with self._lock:
            return self._subs.pop(int(handle), None) is not None

    def emit(self, topic: str, payload: Any = None) -> int:
        """Deliver an event to all matching subscribers. Returns number
        of subscribers actually called. Never raises."""
        t = normalize_topic(topic)
        event = {"topic": t, "payload": payload, "ts": time.time()}
        with self._lock:
            matches: List[Subscriber] = []
            for sub_topic, fn in self._subs.values():
                if sub_topic in ("", "*") or \
                        t == sub_topic or t.startswith(sub_topic + "."):
                    matches.append(fn)
        called = 0
        for fn in matches:
            try:
                fn(event)
                called += 1
            except Exception:
                with self._lock:
                    self._dropped += 1
        return called

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)


# ── the registry ─────────────────────────────────────────────────────────

class Telemetry:
    """Central metrics registry for one enforcement process.

    Usage::

        t = Telemetry()
        t.inc("gate.evaluations")
        t.observe("gate.score", 87.5)
        t.record_rule_check("SEC-1", violation=False)
        t.record_session("s1", action="retry", violations=2)
        t.rate.record()
        bus_handle = t.subscribe("gate.decision", my_hook)
        json.dump(t.snapshot(), open("tele.json", "w"))
    """

    def __init__(self, name: str = "enforcex") -> None:
        self.name = name
        self._lock = threading.RLock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._rule_stats: Dict[str, Dict[str, Any]] = {}
        self._sessions: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.rate = RateWindow(60.0)
        self.bus = EventBus()
        self._started = time.time()
        self._default_hist_edges = (10.0, 25.0, 50.0, 75.0, 90.0, 100.0)
        self._max_sessions = 512

    # -- counters ----------------------------------------------------------

    def inc(self, key: str, amount: Any = 1) -> float:
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            amt = 1.0
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + amt
            return self._counters[key]

    def dec(self, key: str, amount: Any = 1) -> float:
        try:
            return self.inc(key, -float(amount))
        except (TypeError, ValueError):
            return self.inc(key, -1.0)

    def counter(self, key: str) -> float:
        with self._lock:
            return self._counters.get(key, 0.0)

    # -- gauges ------------------------------------------------------------

    def set_gauge(self, key: str, value: Any) -> Optional[float]:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        with self._lock:
            self._gauges[key] = v
            return v

    def gauge(self, key: str) -> Optional[float]:
        with self._lock:
            return self._gauges.get(key)

    # -- histograms ---------------------------------------------------------

    def histogram(self, name: str,
                  edges: Optional[Sequence[float]] = None) -> Histogram:
        edges = tuple(edges) if edges else self._default_hist_edges
        with self._lock:
            h = self._histograms.get(name)
            if h is None:
                h = Histogram(name=name, edges=edges)
                self._histograms[name] = h
            return h

    def observe(self, name: str, value: Any) -> None:
        self.histogram(name).observe(value)

    # -- per-rule adherence -------------------------------------------------

    def record_rule_check(self, rule_id: str, violation: bool,
                          severity: str = "") -> None:
        rid = str(rule_id or "unknown")
        with self._lock:
            st = self._rule_stats.setdefault(rid, {
                "checks": 0, "violations": 0, "severities": {},
                "last_seen": 0.0,
            })
            st["checks"] += 1
            if violation:
                st["violations"] += 1
                if severity:
                    st["severities"][severity] = \
                        st["severities"].get(severity, 0) + 1
            st["last_seen"] = time.time()

    def record_rule_checks(self, violations: Iterable[Any]) -> None:
        """Bulk-record from normalized violation dicts: every dict counts
        as a violation for its rule_id; ids absent from the batch get no
        implicit check (use record_rule_check for clean passes)."""
        for v in (violations or []):
            try:
                if isinstance(v, dict):
                    self.record_rule_check(
                        str(v.get("rule_id") or "unknown"), True,
                        severity=str(v.get("severity", "")))
            except Exception:
                continue

    def rule_adherence(self, rule_id: str) -> Optional[float]:
        """0..100, higher = rule followed more often."""
        with self._lock:
            st = self._rule_stats.get(str(rule_id or ""))
        if not st or not st.get("checks"):
            return None
        return 100.0 * (1.0 - st["violations"] / st["checks"])

    def worst_rules(self, top: int = 10) -> List[Dict[str, Any]]:
        """Rules sorted by most violations first, with adherence."""
        with self._lock:
            items = list(self._rule_stats.items())
        out: List[Dict[str, Any]] = []
        for rid, st in items:
            checks = st.get("checks", 0)
            out.append({
                "rule_id": rid,
                "checks": checks,
                "violations": st.get("violations", 0),
                "adherence": (100.0 * (1.0 - st.get("violations", 0) / checks)
                              if checks else None),
                "severities": dict(st.get("severities", {})),
            })
        out.sort(key=lambda r: (-r["violations"], r["rule_id"]))
        try:
            top = max(1, int(top))
        except (TypeError, ValueError):
            top = 10
        return out[:top]

    # -- per-session stats ---------------------------------------------------

    def record_session(self, session_id: str, action: str = "allow",
                       violations: int = 0, score: Any = None) -> Dict[str, Any]:
        sid = str(session_id or "default")
        with self._lock:
            st = self._sessions.get(sid)
            if st is None:
                st = {"evaluations": 0, "violations": 0, "actions": {},
                      "scores": [], "first_seen": time.time()}
                self._sessions[sid] = st
            # keep the LRU bounded
            self._sessions.move_to_end(sid)
            while len(self._sessions) > max(16, self._max_sessions):
                self._sessions.popitem(last=False)
        st["evaluations"] += 1
        st["violations"] += max(0, int(violations or 0))
        a = str(action or "allow").lower()
        st["actions"][a] = st["actions"].get(a, 0) + 1
        try:
            sc = float(getattr(score, "total", score))
            if not math.isnan(sc) and not math.isinf(sc):
                st["scores"].append(sc)
                if len(st["scores"]) > 200:
                    st["scores"] = st["scores"][-200:]
        except (TypeError, ValueError):
            pass
        return self.session_summary(sid)

    def reset_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(str(session_id or ""), None) is not None

    def session_summary(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            st = self._sessions.get(str(session_id or ""))
        if not st:
            return {"session_id": session_id, "evaluations": 0}
        scores = st.get("scores", [])
        mean = round(sum(scores) / len(scores), 2) if scores else None
        return {
            "session_id": session_id,
            "evaluations": st.get("evaluations", 0),
            "violations": st.get("violations", 0),
            "actions": dict(st.get("actions", {})),
            "score_mean": mean,
            "score_last": scores[-1] if scores else None,
        }

    def sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            ids = list(self._sessions.keys())
        return [self.session_summary(sid) for sid in ids]

    # -- event convenience ----------------------------------------------------

    def subscribe(self, topic: str, fn: Subscriber) -> int:
        return self.bus.subscribe(topic, fn)

    def unsubscribe(self, handle: int) -> bool:
        return self.bus.unsubscribe(handle)

    def emit(self, topic: str, payload: Any = None) -> int:
        self.inc(f"events.emitted.{normalize_topic(topic)}")
        return self.bus.emit(topic, payload)

    # -- snapshot / persistence -----------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counters = {k: round(v, 4) for k, v in
                        sorted(self._counters.items())}
            gauges = {k: round(v, 4) for k, v in
                      sorted(self._gauges.items())}
            hists = {k: h.to_dict() for k, h in
                     sorted(self._histograms.items())}
            rules = {rid: {
                "checks": st.get("checks", 0),
                "violations": st.get("violations", 0),
                "severities": dict(st.get("severities", {})),
                "adherence": (round(100.0 * (1.0 - st.get("violations", 0) /
                            st["checks"]), 2) if st.get("checks") else None),
                "last_seen": st.get("last_seen", 0.0),
            } for rid, st in sorted(self._rule_stats.items())}
            sessions = {sid: {
                "evaluations": st.get("evaluations", 0),
                "violations": st.get("violations", 0),
                "actions": dict(st.get("actions", {})),
                "scores": [round(s, 2) for s in st.get("scores", [])[-100:]],
            } for sid, st in self._sessions.items()}
            return {
                "schema": _SCHEMA_VERSION,
                "name": self.name,
                "created_at": self._started,
                "uptime_seconds": round(time.time() - self._started, 3),
                "counters": counters,
                "gauges": gauges,
                "histograms": hists,
                "rules": rules,
                "sessions": sessions,
                "rate_per_second": round(self.rate.rate(), 4),
            }

    def save_snapshot(self, path: str) -> str:
        try:
            return _atomic_write_json(path, self.snapshot())
        except Exception:
            return ""

    def load_snapshot(self, path: str) -> bool:
        """Merge a snapshot file into this registry. Fails soft."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return False
        if not isinstance(data, dict):
            return False
        return self.merge(data)

    def merge(self, data: Dict[str, Any]) -> bool:
        """Merge a snapshot-shaped dict. Never raises."""
        if not isinstance(data, dict):
            return False
        try:
            for k, v in (data.get("counters") or {}).items():
                try:
                    self.inc(str(k), float(v))
                except (TypeError, ValueError):
                    continue
            for k, v in (data.get("gauges") or {}).items():
                self.set_gauge(str(k), v)
            for k, v in (data.get("histograms") or {}).items():
                if isinstance(v, dict):
                    try:
                        h = Histogram.from_dict(
                            {**v, "name": str(k)})
                        with self._lock:
                            self._histograms[str(k)] = h
                    except Exception:
                        continue
            for rid, v in (data.get("rules") or {}).items():
                if not isinstance(v, dict):
                    continue
                with self._lock:
                    st = self._rule_stats.setdefault(str(rid), {
                        "checks": 0, "violations": 0, "severities": {},
                        "last_seen": 0.0})
                    st["checks"] = max(st["checks"], int(v.get("checks", 0) or 0))
                    st["violations"] = max(st["violations"],
                                           int(v.get("violations", 0) or 0))
                    for sev, n in (v.get("severities") or {}).items():
                        st["severities"][sev] = max(
                            st["severities"].get(sev, 0), int(n or 0))
            for sid, v in (data.get("sessions") or {}).items():
                if not isinstance(v, dict):
                    continue
                for _ in range(min(int(v.get("evaluations", 0) or 0), 1000)):
                    self.record_session(str(sid), action="allow",
                                        violations=0)
                for action, n in (v.get("actions") or {}).items():
                    for _ in range(min(int(n or 0), 1000)):
                        self.record_session(str(sid), action=str(action))
                # restore score series (latest 100)
                with self._lock:
                    st = self._sessions.get(str(sid))
                    if st is not None:
                        try:
                            st["scores"] = [float(s) for s in
                                            (v.get("scores") or ())][-200:]
                        except (TypeError, ValueError):
                            pass
            return True
        except Exception:
            return False

    # -- reset ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all metrics (event bus subscribers are kept)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._rule_stats.clear()
            self._sessions.clear()
        self.rate.clear()


# ── module-level default instance ────────────────────────────────────────

_default: Optional[Telemetry] = None
_default_lock = threading.Lock()


def get_default() -> Telemetry:
    """Process-wide default Telemetry (lazily created)."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Telemetry()
    return _default


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> None:  # pragma: no cover
    t = Telemetry("test")

    # counters
    t.inc("gate.evaluations")
    t.inc("gate.evaluations", 2)
    t.dec("gate.evaluations")
    assert t.counter("gate.evaluations") == 2.0, t.counter("gate.evaluations")
    t.inc("bad", "not-a-number")            # garbage amount tolerated
    assert t.counter("bad") == 1.0

    # gauges
    t.set_gauge("score", 87.5)
    assert t.gauge("score") == 87.5
    assert t.set_gauge("score", "x") is None and t.gauge("score") == 87.5

    # histogram + percentiles
    h = t.histogram("gate.score")
    for v in list(range(0, 101)):
        h.observe(v)
    assert h.count == 101
    p50 = h.percentile(50)
    assert p50 is not None and 49.0 <= p50 <= 51.0, p50
    p95 = h.percentile(95)
    assert p50 < p95 <= 100.0
    s = h.summary()
    assert s["count"] == 101 and s["percentiles"]["p50"] is not None
    assert h.mean() is not None and abs(h.mean() - 50.0) < 0.01
    h.observe("garbage")                     # tolerated
    assert h.count == 101

    # per-rule adherence
    t.record_rule_check("SEC-1", violation=False)
    t.record_rule_check("SEC-1", violation=False)
    t.record_rule_check("SEC-1", violation=True, severity="high")
    assert abs(t.rule_adherence("SEC-1") - 66.67) < 0.1
    assert t.rule_adherence("NOPE") is None
    t.record_rule_checks([
        {"rule_id": "SEC-1", "severity": "critical"},
        {"rule_id": "FMT-2", "severity": "low"},
    ])
    worst = t.worst_rules(5)
    assert worst[0]["rule_id"] == "SEC-1" and worst[0]["violations"] == 2
    assert worst[0]["adherence"] is not None

    # sessions
    t.record_session("s1", action="retry", violations=2, score=55.0)
    t.record_session("s1", action="allow", violations=0, score=90.0)
    summ = t.session_summary("s1")
    assert summ["evaluations"] == 2 and summ["violations"] == 2
    assert summ["actions"] == {"retry": 1, "allow": 1}
    assert summ["score_mean"] == 72.5
    t.record_session("s2", action="block", violations=3)
    assert len(t.sessions()) == 2
    assert t.reset_session("s2") and not t.reset_session("s2")

    # rate window
    rw = RateWindow(window_seconds=5)
    rw.record(n=10)
    assert rw.count() == 10 and rw.rate() > 0.0
    rw.clear()
    assert rw.rate() == 0.0

    # event bus
    t2 = Telemetry("bus")
    got: List[Dict[str, Any]] = []
    h1 = t2.subscribe("gate", got.append)
    assert h1 > 0
    assert t2.emit("gate.decision", {"action": "allow"}) == 1
    assert got and got[0]["payload"] == {"action": "allow"}
    assert t2.emit("other", None) == 0       # 'gate' sub must not match 'other'
    def _boom(_ev: Any) -> None:
        raise RuntimeError("boom")
    t2.subscribe("gate", _boom)
    assert t2.emit("gate.decision") == 1     # boom swallowed, good sub still called
    assert t2.bus.dropped_count == 1
    assert t2.unsubscribe(h1) and not t2.unsubscribe(h1)
    assert t2.emit("gate.decision") == 0     # only the boom sub remains; swallowed

    # snapshot round trip via merge
    snap = t.snapshot()
    assert snap["schema"] == _SCHEMA_VERSION
    assert snap["counters"]["gate.evaluations"] == 2.0
    t3 = Telemetry("restored")
    assert t3.merge(snap) is True
    assert t3.counter("gate.evaluations") == 2.0
    assert abs(t3.rule_adherence("SEC-1") - t.rule_adherence("SEC-1")) < 0.01
    restored_h = t3.histogram("gate.score")
    assert restored_h.count == 101

    # file round trip
    import tempfile as _tf
    import os as _os
    with _tf.TemporaryDirectory() as tmp:
        p = _os.path.join(tmp, "tele.json")
        assert t.save_snapshot(p) == p
        t4 = Telemetry("from-file")
        assert t4.load_snapshot(p) is True
        assert t4.counter("gate.evaluations") == 2.0
        assert t4.load_snapshot(_os.path.join(tmp, "nope.json")) is False
        assert t4.merge([1, 2]) is False

    # normalize + default instance
    assert normalize_topic("Gate DECISION.x ") == "gate.decision.x"
    assert normalize_topic("") == "generic"
    d = get_default()
    assert d is get_default()

    t.reset()
    assert t.counter("gate.evaluations") == 0.0 and not t.sessions()

    print("telemetry._self_test: OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
