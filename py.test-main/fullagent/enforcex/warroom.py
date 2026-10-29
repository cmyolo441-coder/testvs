"""warroom — Incident console for enforcement outcomes (Tier-2).

Turns EnforcementOutcome-style detection payloads into manageable
incidents that humans can triage:

  * CLUSTERING   — related detections (same rule, same session, close
                   in time) are folded into one incident.
  * LIFECYCLE    — open -> acknowledged -> mitigated -> closed, with a
                   validated state machine (illegal transitions are
                   rejected with a reason, never an exception).
  * POSTMORTEM   — timeline, root-cause heuristics (dominant rule,
                   dominant category, most common evidence keyword,
                   burst analysis), affected rules and repair actions.
  * ESCALATION   — queue sorted by severity x frequency priority.
  * DIGEST       — top-N incidents for the day.

Persistence: IncidentStore keeps an append-only JSONL journal (one JSON
object per line: either an incident snapshot or a lifecycle event) and
replays it on startup, so the store survives restarts and concurrent
appends are cheap.

Design inherited from the enforcex house style:
  * stdlib only, Python 3.9+,
  * dataclasses + small pure functions,
  * never raises on malformed input,
  * `_self_test()` runnable via `python -m`.

Typical usage::

    from fullagent.enforcex import warroom

    store = warroom.IncidentStore("warroom.jsonl")
    ids = store.ingest(detections=[{...}, {...}], session_id="s1")
    store.acknowledge(ids[0], operator="maya")
    store.mitigate(ids[0], repair_action="stripped fake tags")
    store.close(ids[0], note="fixed upstream")
    print(json.dumps(store.digest(top_n=3), indent=2))
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "STATES", "TRANSITIONS", "SEVERITY_WEIGHTS",
    "IncidentEvent", "Incident", "IncidentStore",
    "cluster_detections",
]

# ── lifecycle model ──────────────────────────────────────────────────────

STATES: Tuple[str, ...] = ("open", "acknowledged", "mitigated", "closed")

TRANSITIONS: Dict[str, Tuple[str, ...]] = {
    "open": ("acknowledged", "mitigated", "closed"),     # fast paths allowed
    "acknowledged": ("mitigated", "closed"),
    "mitigated": ("closed",),
    "closed": (),                                        # terminal
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 4.0, "high": 3.0, "medium": 2.0, "low": 1.0, "info": 0.5,
}

DEFAULT_CLUSTER_WINDOW_S = 900.0        # 15 minutes of time proximity
DEFAULT_REOPEN_WINDOW_S = 3600.0

_TERM_RE = re.compile(r"[a-z][a-z0-9_\-']{3,}")


def _now() -> float:
    return time.time()


def _clip(text: Any, n: int = 200) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _sev_weight(sev: Any) -> float:
    return SEVERITY_WEIGHTS.get(str(sev or "").lower(), 1.0)


def _new_incident_id(rule_id: str, session_id: str, ts: float,
                     salt: int = 0) -> str:
    """Deterministic, human-greppable id: INC-<rule>-<hash6>."""
    import hashlib
    raw = f"{rule_id}|{session_id}|{ts:.6f}|{salt}"
    h = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:6]
    tag = re.sub(r"[^A-Za-z0-9\-]+", "", str(rule_id or "GEN")).strip("-")
    tag = tag[:10] or "GEN"
    return f"INC-{tag.upper()}-{h}"


# ── clustering ───────────────────────────────────────────────────────────

def cluster_detections(detections: List[Dict[str, Any]],
                       window_s: float = DEFAULT_CLUSTER_WINDOW_S,
                       ts_field: str = "ts") -> List[List[Dict[str, Any]]]:
    """Group detections into clusters: same (rule_id, session_id) and
    each member within ``window_s`` of the cluster's first detection.
    Detections without a usable rule_id all fall into one "general"
    bucket per session. Never raises."""
    clusters: List[List[Dict[str, Any]]] = []
    open_keys: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for det in (detections or []):
        if not isinstance(det, dict):
            continue
        rule_id = str(det.get("rule_id", "") or "GENERAL")
        session_id = str(det.get("session_id", "") or "default")
        try:
            ts = float(det.get(ts_field, 0.0) or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        key = (rule_id, session_id)
        bucket = open_keys.get(key)
        if bucket is None or (ts and bucket and
                              ts - float(bucket[0].get(ts_field, ts) or ts)
                              > window_s):
            bucket = [det]
            clusters.append(bucket)
            open_keys[key] = bucket
        else:
            bucket.append(det)
    return clusters


# ── data model ───────────────────────────────────────────────────────────

@dataclass
class IncidentEvent:
    """One append-only entry in an incident timeline."""
    ts: float
    kind: str                        # detection | acknowledge | mitigate | close | note | open
    detail: str = ""
    operator: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": round(float(self.ts), 6),
            "kind": str(self.kind),
            "detail": self.detail,
            "operator": self.operator,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "IncidentEvent":
        if not isinstance(d, dict):
            return cls(ts=0.0, kind="note", detail="corrupt entry")
        try:
            ts = float(d.get("ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            ts = 0.0
        return cls(
            ts=ts,
            kind=str(d.get("kind", "note") or "note"),
            detail=str(d.get("detail", "") or ""),
            operator=str(d.get("operator", "") or ""),
            data=dict(d.get("data") or {}),
        )


@dataclass
class Incident:
    """One manageable enforcement incident."""
    incident_id: str
    title: str = ""
    state: str = "open"
    created_ts: float = 0.0
    updated_ts: float = 0.0
    closed_ts: float = 0.0
    severity: str = "medium"
    rule_ids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[IncidentEvent] = field(default_factory=list)
    repair_actions: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    operator: str = ""
    notes: List[str] = field(default_factory=list)

    # -- derived ------------------------------------------------------------

    @property
    def frequency(self) -> int:
        return len(self.detections)

    def priority(self) -> float:
        """severity x frequency — the escalation-queue sort key."""
        return _sev_weight(self.severity) * max(1, self.frequency)

    def worst_severity(self) -> str:
        best, best_w = "info", -1.0
        for d in self.detections:
            sev = str(d.get("severity", "medium") or "medium").lower()
            w = _sev_weight(sev)
            if w > best_w:
                best, best_w = sev, w
        return best if best_w >= 0 else self.severity

    def last_activity(self) -> float:
        return self.updated_ts or self.created_ts

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "state": self.state,
            "created_ts": round(self.created_ts, 6),
            "updated_ts": round(self.updated_ts, 6),
            "closed_ts": round(self.closed_ts, 6),
            "severity": self.severity,
            "rule_ids": list(self.rule_ids),
            "session_ids": list(self.session_ids),
            "detections": list(self.detections),
            "timeline": [e.to_dict() for e in self.timeline],
            "repair_actions": list(self.repair_actions),
            "tags": list(self.tags),
            "operator": self.operator,
            "notes": list(self.notes),
            "frequency": self.frequency,
            "priority": round(self.priority(), 3),
        }

    @classmethod
    def from_dict(cls, d: Any) -> "Incident":
        if not isinstance(d, dict):
            return cls(incident_id="INC-CORRUPT-000000")
        try:
            created = float(d.get("created_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            created = 0.0
        inc = cls(
            incident_id=str(d.get("incident_id", "") or
                             "INC-CORRUPT-000000"),
            title=str(d.get("title", "") or ""),
            state=str(d.get("state", "open") or "open"),
            created_ts=created,
            severity=str(d.get("severity", "medium") or "medium"),
            rule_ids=[str(r) for r in (d.get("rule_ids") or [])],
            session_ids=[str(s) for s in (d.get("session_ids") or [])],
            detections=[x for x in (d.get("detections") or [])
                        if isinstance(x, dict)],
            repair_actions=[str(a) for a in (d.get("repair_actions") or [])],
            tags=[str(t) for t in (d.get("tags") or [])],
            operator=str(d.get("operator", "") or ""),
            notes=[str(n) for n in (d.get("notes") or [])],
        )
        inc.timeline = [IncidentEvent.from_dict(e)
                        for e in (d.get("timeline") or [])]
        try:
            inc.updated_ts = float(d.get("updated_ts", 0.0) or 0.0)
            inc.closed_ts = float(d.get("closed_ts", 0.0) or 0.0)
        except (TypeError, ValueError):
            pass
        if inc.state not in STATES:
            inc.state = "open"
        return inc


# ── store ────────────────────────────────────────────────────────────────

class IncidentStore:
    """JSONL-persisted incident store with lifecycle + reporting."""

    def __init__(self, path: Optional[str] = None, *,
                 cluster_window_s: float = DEFAULT_CLUSTER_WINDOW_S,
                 reopen_window_s: float = DEFAULT_REOPEN_WINDOW_S) -> None:
        self.path = path or ""
        self.cluster_window_s = max(1.0, float(cluster_window_s))
        self.reopen_window_s = max(1.0, float(reopen_window_s))
        self._incidents: Dict[str, Incident] = {}
        self._lock = threading.RLock()
        self._fh = None
        self._load()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    self._apply_record(rec, persist=False)
        except OSError:
            pass

    def _persist(self, rec: Dict[str, Any]) -> None:
        if not self.path:
            return
        try:
            if self._fh is None:
                self._fh = open(self.path, "a", encoding="utf-8")
            self._fh.write(json.dumps(rec, ensure_ascii=False,
                                      sort_keys=True) + "\n")
            self._fh.flush()
        except OSError:
            pass

    def _apply_record(self, rec: Any, *, persist: bool = True) -> None:
        if not isinstance(rec, dict):
            return
        rtype = str(rec.get("type", ""))
        data = rec.get("data") if isinstance(rec.get("data"), dict) else rec
        if rtype in ("incident", "snapshot") or (not rtype and
                                                 "incident_id" in data):
            inc = Incident.from_dict(data)
            if inc.incident_id:
                self._incidents[inc.incident_id] = inc
        elif rtype == "event":
            iid = str(rec.get("incident_id", "") or "")
            inc = self._incidents.get(iid)
            if inc is not None:
                self._apply_event(inc, IncidentEvent.from_dict(
                    rec.get("event")))

    def _apply_event(self, inc: Incident, ev: IncidentEvent) -> None:
        inc.timeline.append(ev)
        inc.updated_ts = ev.ts or _now()
        if ev.kind == "acknowledge":
            inc.state = "acknowledged"
            inc.operator = ev.operator or inc.operator
        elif ev.kind == "mitigate":
            inc.state = "mitigated"
            if ev.detail:
                inc.repair_actions.append(ev.detail)
        elif ev.kind == "close":
            inc.state = "closed"
            inc.closed_ts = ev.ts or _now()
            if ev.detail:
                inc.notes.append(ev.detail)
        elif ev.kind == "note":
            if ev.detail:
                inc.notes.append(ev.detail)

    def shutdown(self) -> None:
        """Close the JSONL file handle (lifecycle close is separate)."""
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    # -- ingest / clustering ---------------------------------------------------

    def ingest(self, detections: List[Dict[str, Any]], *,
               session_id: str = "default",
               ts: Optional[float] = None,
               default_severity: str = "medium") -> List[str]:
        """Fold detections into incidents (new or reopened). Returns the
        list of incident ids touched. Never raises."""
        now = float(ts) if ts is not None else _now()
        touched: List[str] = []
        prepared: List[Dict[str, Any]] = []
        for d in (detections or []):
            if not isinstance(d, dict):
                continue
            det = dict(d)
            det.setdefault("session_id", session_id)
            try:
                det.setdefault("ts", now)
            except Exception:
                det["ts"] = now
            prepared.append(det)
        clusters = cluster_detections(prepared,
                                      window_s=self.cluster_window_s)
        with self._lock:
            for cluster in clusters:
                iid = self._match_or_create(cluster, now,
                                            default_severity)
                if iid:
                    touched.append(iid)
        return touched

    def _match_or_create(self, cluster: List[Dict[str, Any]],
                         now: float, default_severity: str) -> str:
        first = cluster[0] if cluster else {}
        rule_id = str(first.get("rule_id", "") or "GENERAL")
        session_id = str(first.get("session_id", "") or "default")
        # try to attach to an existing, not-closed incident for the
        # same rule+session within the reopen window
        best: Optional[Incident] = None
        for inc in self._incidents.values():
            if inc.state == "closed":
                continue
            if rule_id not in inc.rule_ids:
                continue
            if session_id not in inc.session_ids:
                continue
            if now and inc.last_activity() and \
                    now - inc.last_activity() > self.reopen_window_s:
                continue
            if best is None or inc.last_activity() > best.last_activity():
                best = inc
        if best is None:
            iid = _new_incident_id(rule_id, session_id, now,
                                   salt=len(self._incidents))
            sev = str(first.get("severity", default_severity) or
                      default_severity).lower()
            inc = Incident(
                incident_id=iid,
                title=_clip(first.get("message") or
                            f"{rule_id} violations", 80),
                state="open",
                created_ts=now,
                updated_ts=now,
                severity=sev,
                rule_ids=[rule_id],
                session_ids=[session_id],
                tags=[sev],
            )
            inc.timeline.append(IncidentEvent(
                ts=now, kind="open",
                detail=f"opened from {len(cluster)} detection(s)",
                data={"rule_id": rule_id, "session_id": session_id}))
            self._incidents[iid] = inc
            self._persist({"type": "incident", "ts": now,
                           "data": inc.to_dict()})
            target = inc
        else:
            target = best
        for det in cluster:
            target.detections.append(det)
            target.timeline.append(IncidentEvent(
                ts=float(det.get("ts", now) or now), kind="detection",
                detail=_clip(det.get("evidence") or
                             det.get("message") or "", 120),
                data={"rule_id": str(det.get("rule_id", "")),
                      "severity": str(det.get("severity", ""))}))
            target.updated_ts = now
            for rid in {str(det.get("rule_id", "") or rule_id)}:
                if rid and rid not in target.rule_ids:
                    target.rule_ids.append(rid)
            sid = str(det.get("session_id", "") or session_id)
            if sid and sid not in target.session_ids:
                target.session_ids.append(sid)
        # severity = worst seen so far
        target.severity = target.worst_severity()
        self._persist({"type": "incident", "ts": now,
                       "data": target.to_dict()})
        return target.incident_id

    # -- lifecycle -------------------------------------------------------------

    def _transition(self, incident_id: str, action: str, *,
                    operator: str = "", detail: str = "",
                    ts: Optional[float] = None,
                    extra: Optional[Dict[str, Any]] = None
                    ) -> Tuple[bool, str]:
        now = float(ts) if ts is not None else _now()
        with self._lock:
            inc = self._incidents.get(str(incident_id or ""))
            if inc is None:
                return False, f"unknown incident {incident_id!r}"
            action_state = {"acknowledge": "acknowledged",
                            "mitigate": "mitigated",
                            "close": "closed"}.get(action, action)
            if inc.state == "closed":
                return False, "incident already closed"
            allowed = TRANSITIONS.get(inc.state, ())
            if action_state not in allowed:
                return False, (f"illegal transition {inc.state} -> "
                               f"{action}; allowed: {list(allowed)}")
            ev = IncidentEvent(ts=now, kind=action,
                               detail=_clip(detail, 200),
                               operator=str(operator or ""),
                               data=dict(extra or {}))
            self._apply_event(inc, ev)
            self._persist({"type": "event", "ts": now,
                           "incident_id": inc.incident_id,
                           "event": ev.to_dict()})
            return True, inc.state

    def acknowledge(self, incident_id: str, *, operator: str = "",
                    detail: str = "", ts: Optional[float] = None
                    ) -> Tuple[bool, str]:
        return self._transition(incident_id, "acknowledge",
                                operator=operator, detail=detail, ts=ts)

    def mitigate(self, incident_id: str, *, repair_action: str = "",
                 operator: str = "", ts: Optional[float] = None
                 ) -> Tuple[bool, str]:
        return self._transition(incident_id, "mitigate",
                                operator=operator,
                                detail=repair_action, ts=ts)

    def close(self, incident_id: str, *, note: str = "",
              operator: str = "", ts: Optional[float] = None
              ) -> Tuple[bool, str]:
        return self._transition(incident_id, "close",
                                operator=operator, detail=note, ts=ts)

    def add_note(self, incident_id: str, note: str, *,
                 operator: str = "") -> Tuple[bool, str]:
        with self._lock:
            inc = self._incidents.get(str(incident_id or ""))
            if inc is None:
                return False, f"unknown incident {incident_id!r}"
            ev = IncidentEvent(ts=_now(), kind="note",
                               detail=_clip(note, 200),
                               operator=str(operator or ""))
            self._apply_event(inc, ev)
            self._persist({"type": "event", "ts": ev.ts,
                           "incident_id": inc.incident_id,
                           "event": ev.to_dict()})
            return True, "noted"

    # -- queries -----------------------------------------------------------------

    def get(self, incident_id: str) -> Optional[Incident]:
        with self._lock:
            inc = self._incidents.get(str(incident_id or ""))
            return inc if inc is None else self._copy(inc)

    @staticmethod
    def _copy(inc: Incident) -> Incident:
        return Incident.from_dict(inc.to_dict())

    def incidents(self, state: Optional[str] = None, *,
                  rule_id: str = "", session_id: str = "",
                  since_ts: float = 0.0) -> List[Incident]:
        with self._lock:
            out = []
            for inc in self._incidents.values():
                if state and inc.state != str(state).lower():
                    continue
                if rule_id and rule_id not in inc.rule_ids:
                    continue
                if session_id and session_id not in inc.session_ids:
                    continue
                if since_ts and inc.last_activity() < since_ts:
                    continue
                out.append(self._copy(inc))
        out.sort(key=lambda i: (-i.priority(), i.incident_id))
        return out

    def open_incidents(self) -> List[Incident]:
        return self.incidents(state="open") + \
            self.incidents(state="acknowledged") + \
            self.incidents(state="mitigated")

    def escalation_queue(self, top: int = 0) -> List[Incident]:
        """Not-closed incidents sorted by severity x frequency, desc."""
        q = [i for i in self.incidents() if i.state != "closed"]
        q.sort(key=lambda i: (-i.priority(), i.incident_id))
        return q[:top] if top > 0 else q

    def digest(self, top_n: int = 5, *, day: Optional[float] = None,
               now: Optional[float] = None) -> Dict[str, Any]:
        """Top-N incidents for the day + headline counts."""
        t = float(now) if now is not None else _now()
        if day is not None:
            day_start = float(day)
        else:
            day_start = t - 86400.0
        with self._lock:
            all_inc = [self._copy(i) for i in self._incidents.values()]
        day_inc = [i for i in all_inc
                   if i.created_ts >= day_start or
                   (i.timeline and
                    max(e.ts for e in i.timeline) >= day_start)]
        day_inc.sort(key=lambda i: (-i.priority(), i.incident_id))
        by_state: Dict[str, int] = {}
        for s in STATES:
            by_state[s] = sum(1 for i in all_inc if i.state == s)
        return {
            "generated_ts": round(t, 6),
            "window_start": round(day_start, 6),
            "total_incidents": len(all_inc),
            "by_state": by_state,
            "open_count": by_state.get("open", 0) +
            by_state.get("acknowledged", 0) +
            by_state.get("mitigated", 0),
            "top": [i.to_dict() for i in day_inc[:max(0, int(top_n))]],
        }

    # -- postmortem ----------------------------------------------------------------

    def postmortem(self, incident_id: str) -> Dict[str, Any]:
        """Full report: timeline, root-cause heuristics, affected rules,
        repair actions. Never raises; unknown ids yield an error doc."""
        with self._lock:
            inc = self._incidents.get(str(incident_id or ""))
            if inc is None:
                return {"error": f"unknown incident {incident_id!r}",
                        "incident_id": str(incident_id or "")}
            snap = self._copy(inc).to_dict()
        return _build_postmortem(snap)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            incs = list(self._incidents.values())
        by_state = {s: sum(1 for i in incs if i.state == s) for s in STATES}
        total_dets = sum(i.frequency for i in incs)
        return {
            "incidents": len(incs),
            "by_state": by_state,
            "detections_clustered": total_dets,
            "path": self.path,
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._incidents)


# ── postmortem heuristics (pure function) ────────────────────────────────

def _top_term(texts: List[str]) -> str:
    """Most common informative token across evidence strings."""
    counts: Dict[str, int] = {}
    for t in texts:
        for m in _TERM_RE.findall((t or "").lower()):
            counts[m] = counts.get(m, 0) + 1
    if not counts:
        return ""
    return max(sorted(counts), key=lambda k: counts[k])


def _build_postmortem(snap: Dict[str, Any]) -> Dict[str, Any]:
    dets = [d for d in (snap.get("detections") or [])
            if isinstance(d, dict)]
    timeline = snap.get("timeline") or []
    severities = [str(d.get("severity", "medium")).lower() for d in dets]
    categories: Dict[str, int] = {}
    rules: Dict[str, int] = {}
    for d in dets:
        c = str(d.get("category", "other") or "other")
        categories[c] = categories.get(c, 0) + 1
        r = str(d.get("rule_id", "") or "GENERAL")
        rules[r] = rules.get(r, 0) + 1
    evidence = [str(d.get("evidence", "") or "") for d in dets]

    # burst analysis: biggest gap between consecutive detections
    tss = sorted(float(d.get("ts", 0.0) or 0.0) for d in dets if d.get("ts"))
    gaps = [b - a for a, b in zip(tss, tss[1:])]
    burst = {
        "detection_count": len(tss),
        "span_s": round(tss[-1] - tss[0], 3) if len(tss) > 1 else 0.0,
        "max_gap_s": round(max(gaps), 3) if gaps else 0.0,
        "bursty": bool(gaps and max(gaps) > 0 and
                       (tss[-1] - tss[0]) / max(gaps) < 3.0),
    }

    # root-cause heuristics
    dominant_rule = max(sorted(rules), key=lambda k: rules[k]) if rules else ""
    dominant_category = (max(sorted(categories),
                             key=lambda k: categories[k])
                         if categories else "")
    signature = _top_term(evidence)
    causes: List[str] = []
    if dominant_category == "injection":
        causes.append("hostile user/tool input attempting prompt injection")
    if dominant_category == "leak":
        causes.append("model output leaked internal/fake structural markup")
    if dominant_category == "refusal":
        causes.append("model drifted into refusal behavior")
    if dominant_category == "sensitive":
        causes.append("sensitive/secret-like content reached the output")
    if signature:
        causes.append(f"recurring evidence token: {signature!r}")
    if burst.get("bursty"):
        causes.append("bursty pattern: detections arrived in a tight window "
                      "(likely a single adversarial exchange)")
    else:
        causes.append("spread-out pattern: repeated drift over time "
                      "(likely a systemic prompt gap)")
    if not causes:
        causes.append("insufficient data for a root-cause estimate")

    return {
        "incident_id": snap.get("incident_id", ""),
        "title": snap.get("title", ""),
        "state": snap.get("state", ""),
        "severity": snap.get("severity", ""),
        "created_ts": snap.get("created_ts", 0.0),
        "closed_ts": snap.get("closed_ts", 0.0),
        "duration_s": round(
            (snap.get("closed_ts") or snap.get("updated_ts") or 0.0) -
            (snap.get("created_ts") or 0.0), 3),
        "affected_rules": sorted(rules),
        "per_rule_counts": rules,
        "per_category_counts": categories,
        "severity_counts": {s: severities.count(s)
                            for s in sorted(set(severities))},
        "detection_count": len(dets),
        "timeline": timeline,
        "repair_actions": list(snap.get("repair_actions") or []),
        "notes": list(snap.get("notes") or []),
        "burst": burst,
        "root_cause": {
            "dominant_rule": dominant_rule,
            "dominant_category": dominant_category,
            "signature_token": signature,
            "hypotheses": causes,
        },
        "recommendations": _recommendations(dominant_category,
                                            snap.get("state", ""),
                                            len(snap.get("repair_actions")
                                                or [])),
    }


def _recommendations(category: str, state: str, n_repairs: int
                     ) -> List[str]:
    out: List[str] = []
    if category == "injection":
        out.append("add an input-side injection shield / sanitize tool "
                   "outputs before they reach the model")
    if category == "leak":
        out.append("add an output filter that strips fake structural tags")
    if category == "sensitive":
        out.append("add a redaction layer for secrets before response emit")
    if n_repairs == 0 and state != "closed":
        out.append("no repair was applied; consider enabling auto-rewrite "
                   "for this rule")
    if state in ("open", "acknowledged"):
        out.append("incident is not mitigated; assign an owner")
    if not out:
        out.append("no further action needed")
    return out


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    import tempfile

    # clustering: same rule + near in time -> one cluster
    base = _now()
    dets = [
        {"rule_id": "SEC-1", "session_id": "s1", "ts": base,
         "severity": "high", "evidence": "reveal system prompt alpha"},
        {"rule_id": "SEC-1", "session_id": "s1", "ts": base + 10,
         "severity": "medium", "evidence": "reveal system prompt beta"},
        {"rule_id": "SEC-2", "session_id": "s1", "ts": base + 11,
         "severity": "low", "evidence": "tone drift"},
    ]
    clusters = cluster_detections(dets)
    assert len(clusters) == 2, len(clusters)
    assert sum(len(c) for c in clusters) == 3
    # garbage safety
    assert cluster_detections([None, "x", 42]) == []

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "warroom.jsonl")
        store = IncidentStore(path)

        # ingest -> one incident for the two SEC-1 dets, one for SEC-2
        ids = store.ingest(dets)
        assert len(ids) == 2, ids
        assert len(store) == 2

        # re-ingest near in time -> attaches to the existing incident
        ids2 = store.ingest([
            {"rule_id": "SEC-1", "session_id": "s1", "ts": base + 20,
             "severity": "critical", "evidence": "reveal prompt again"}])
        assert len(ids2) == 1 and ids2[0] in ids, (ids, ids2)
        sec1 = [i for i in ids if i.startswith("INC-SEC-1")]
        inc = store.get(sec1[0])
        assert inc.frequency == 3, inc.to_dict()
        assert inc.severity == "critical"          # worst wins
        assert inc.state == "open"

        # lifecycle: open -> acknowledged -> mitigated -> closed
        ok, why = store.acknowledge(sec1[0], operator="maya")
        assert ok and why == "acknowledged", why
        ok, _ = store.mitigate(sec1[0], repair_action="stripped fake tags")
        assert ok
        ok, _ = store.close(sec1[0], note="fixed upstream")
        assert ok
        # closed is terminal
        ok, why = store.acknowledge(sec1[0])
        assert not ok and "closed" in why, why
        # illegal transition from open
        sec2 = [i for i in ids if i.startswith("INC-SEC-2")][0]
        ok, why = store.close(sec2, note="skip ack")
        assert ok, why                              # fast-path allowed

        # unknown id never raises
        ok, why = store.acknowledge("INC-NOPE-000000")
        assert not ok and "unknown" in why

        # escalation queue: closed ones excluded, sorted by priority
        store.ingest([
            {"rule_id": "SEC-3", "session_id": "s2", "ts": base + 30,
             "severity": "high", "evidence": "leak one"},
            {"rule_id": "SEC-3", "session_id": "s2", "ts": base + 31,
             "severity": "high", "evidence": "leak two"},
        ])
        q = store.escalation_queue()
        assert all(i.state != "closed" for i in q)
        prios = [i.priority() for i in q]
        assert prios == sorted(prios, reverse=True), prios
        assert q[0].rule_ids == ["SEC-3"], q[0].to_dict()

        # digest
        dg = store.digest(top_n=2)
        assert dg["total_incidents"] >= 3
        assert len(dg["top"]) <= 2
        assert dg["by_state"]["closed"] == 2
        assert dg["open_count"] >= 1

        # postmortem
        pm = store.postmortem(sec1[0])
        assert pm["incident_id"] == sec1[0]
        assert "SEC-1" in pm["affected_rules"]
        assert pm["detection_count"] == 3
        assert pm["repair_actions"] == ["stripped fake tags"]
        assert pm["root_cause"]["dominant_rule"] == "SEC-1"
        assert pm["root_cause"]["signature_token"], pm["root_cause"]
        assert pm["recommendations"]
        assert "timeline" in pm and len(pm["timeline"]) >= 5
        # unknown -> error doc, no raise
        assert "error" in store.postmortem("INC-NOPE-000000")

        # add_note + stats
        ok, _ = store.add_note(sec2, "watch this one", operator="bo")
        assert ok
        st = store.stats()
        assert st["incidents"] == 3 and st["detections_clustered"] == 6

        # persistence: reload from disk into a fresh store
        store.shutdown()
        store2 = IncidentStore(path)
        assert len(store2) == 3, len(store2)
        inc_r = store2.get(sec1[0])
        assert inc_r is not None and inc_r.state == "closed"
        assert inc_r.frequency == 3
        assert "stripped fake tags" in inc_r.repair_actions

        # corrupt line safety
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("{{{not json\n")
        store3 = IncidentStore(path)
        assert len(store3) == 3

    # in-memory store (no path) works too
    mem = IncidentStore()
    ids = mem.ingest([{"rule_id": "R1", "severity": "low"}])
    assert len(mem) == 1 and ids
    assert mem.get(ids[0]).state == "open"

    # serialization round trips on garbage
    inc = Incident.from_dict(None)
    assert inc.state == "open"
    ev = IncidentEvent.from_dict("junk")
    assert ev.kind == "note"

    print("warroom._self_test: OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
