"""choreography — Vector Clocks, Lamport Timestamps & Chandy-Lamport Consistent Cut Verifier.

Tracks causal partial order (`e1 -> e2`, happens-before) across distributed or
multi-threaded subagents (Lamport 1978, Fidge/Mattern 1988, Chandy-Lamport 1985):
  * `VectorClock`: maps process ID `p -> logical_tick`.
    - `V1 <= V2` iff `forall p: V1[p] <= V2[p]`
    - `V1 < V2` (causal happens-before) iff `V1 <= V2` and `V1 != V2`
    - `V1 || V2` (causally concurrent) iff `not (V1 <= V2)` and `not (V2 <= V1)`
  * Concurrent Write-Write Race Detection: flags any two events `(e1, e2)` that
    mutate the same state key `k` with `V(e1) || V(e2)`.
  * Consistent Global Snapshot Cut (`C`): verifies whether a set of per-process
    checkpoint events forms a causally consistent frontier (if `e in C` and
    `e' -> e`, then `e'` precedes or belongs to the cut).

Self-test: ``python3 -m fullagent.enforcex.choreography``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "VectorClock",
    "DistributedEvent",
    "ConcurrencyRace",
    "ChoreographyReport",
    "CausalChoreographyEngine",
]


class VectorClock:
    """Fidge/Mattern Vector Clock over process identifiers."""

    def __init__(self, clocks: Optional[Dict[str, int]] = None) -> None:
        clean: Dict[str, int] = {}
        if isinstance(clocks, dict):
            for k, v in clocks.items():
                try:
                    tick = int(v)
                except (TypeError, ValueError, OverflowError):
                    continue
                if tick > 0:
                    clean[str(k)] = tick
        self.clocks: Dict[str, int] = clean

    def copy(self) -> "VectorClock":
        return VectorClock(self.clocks)

    def tick(self, proc: str) -> "VectorClock":
        key = str(proc)
        nxt = dict(self.clocks)
        nxt[key] = nxt.get(key, 0) + 1
        return VectorClock(nxt)

    def merge(self, other: "VectorClock") -> "VectorClock":
        if not isinstance(other, VectorClock):
            other = VectorClock(other if isinstance(other, dict) else None)
        all_keys = set(self.clocks.keys()) | set(other.clocks.keys())
        return VectorClock({k: max(self.clocks.get(k, 0), other.clocks.get(k, 0)) for k in all_keys})

    def leq(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            other = VectorClock(other if isinstance(other, dict) else None)
        return all(v <= other.clocks.get(k, 0) for k, v in self.clocks.items())

    def happens_before(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            other = VectorClock(other if isinstance(other, dict) else None)
        return self.leq(other) and self.clocks != other.clocks

    def concurrent_with(self, other: "VectorClock") -> bool:
        if not isinstance(other, VectorClock):
            other = VectorClock(other if isinstance(other, dict) else None)
        return (not self.leq(other)) and (not other.leq(self))

    def to_dict(self) -> Dict[str, int]:
        return dict(sorted(self.clocks.items()))


@dataclass
class DistributedEvent:
    event_id: str
    proc: str
    kind: str  # local | send | recv | write
    vc: VectorClock
    lamport_ts: int
    target_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "proc": self.proc,
            "kind": self.kind,
            "vc": self.vc.to_dict(),
            "lamport_ts": self.lamport_ts,
            "target_key": self.target_key,
        }


@dataclass
class ConcurrencyRace:
    event_a: str
    event_b: str
    proc_a: str
    proc_b: str
    key: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_a": self.event_a,
            "event_b": self.event_b,
            "proc_a": self.proc_a,
            "proc_b": self.proc_b,
            "key": self.key,
        }


@dataclass
class ChoreographyReport:
    race_free: bool
    consistent_cut: bool
    events_count: int
    races: List[ConcurrencyRace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "race_free": self.race_free,
            "consistent_cut": self.consistent_cut,
            "events_count": self.events_count,
            "races": [r.to_dict() for r in self.races],
        }


class CausalChoreographyEngine:
    """Tracks vector clocks across processes, detects data races, and verifies Chandy-Lamport cuts."""

    def __init__(self) -> None:
        self.proc_vc: Dict[str, VectorClock] = {}
        self.proc_lamport: Dict[str, int] = {}
        self.events: List[DistributedEvent] = []

    def record_local(self, proc: str, event_id: str, *, write_key: str = "") -> DistributedEvent:
        p = str(proc)
        vc = self.proc_vc.get(p, VectorClock()).tick(p)
        lts = self.proc_lamport.get(p, 0) + 1
        self.proc_vc[p] = vc
        self.proc_lamport[p] = lts
        key = "" if write_key is None else str(write_key)
        ev = DistributedEvent(
            event_id=str(event_id),
            proc=p,
            kind="write" if key else "local",
            vc=vc,
            lamport_ts=lts,
            target_key=key,
        )
        self.events.append(ev)
        return ev

    def record_message(self, sender: str, receiver: str, send_id: str, recv_id: str) -> Tuple[DistributedEvent, DistributedEvent]:
        s_name = str(sender)
        r_name = str(receiver)
        s_vc = self.proc_vc.get(s_name, VectorClock()).tick(s_name)
        s_lts = self.proc_lamport.get(s_name, 0) + 1
        self.proc_vc[s_name] = s_vc
        self.proc_lamport[s_name] = s_lts
        ev_send = DistributedEvent(str(send_id), s_name, "send", s_vc, s_lts)
        self.events.append(ev_send)

        r_base = self.proc_vc.get(r_name, VectorClock()).merge(s_vc).tick(r_name)
        r_lts = max(self.proc_lamport.get(r_name, 0), s_lts) + 1
        self.proc_vc[r_name] = r_base
        self.proc_lamport[r_name] = r_lts
        ev_recv = DistributedEvent(str(recv_id), r_name, "recv", r_base, r_lts)
        self.events.append(ev_recv)
        return ev_send, ev_recv

    def detect_write_races(self) -> List[ConcurrencyRace]:
        """Flag concurrent writes to the same key (grouped by key for O(k^2) cost)."""
        index = {id(ev): pos for pos, ev in enumerate(self.events)}
        by_key: Dict[str, List[DistributedEvent]] = {}
        for ev in self.events:
            if ev.kind == "write" and ev.target_key:
                by_key.setdefault(ev.target_key, []).append(ev)
        pairs: List[Tuple[int, int, DistributedEvent, DistributedEvent, str]] = []
        for key, writes in by_key.items():
            for i in range(len(writes)):
                for j in range(i + 1, len(writes)):
                    if writes[i].vc.concurrent_with(writes[j].vc):
                        ia, ib = index[id(writes[i])], index[id(writes[j])]
                        if ia > ib:
                            ia, ib = ib, ia
                            pairs.append((ia, ib, writes[j], writes[i], key))
                        else:
                            pairs.append((ia, ib, writes[i], writes[j], key))
        pairs.sort(key=lambda t: (t[0], t[1]))
        return [ConcurrencyRace(ea.event_id, eb.event_id, ea.proc, eb.proc, key)
                for _, _, ea, eb, key in pairs]

    @staticmethod
    def is_consistent_cut(cut_clocks: Dict[str, VectorClock]) -> bool:
        """Mattern's Consistent Cut Theorem: a frontier `{V_p}` is causally consistent
        iff for all processes `p, q`: `V_p[p] >= V_q[p]` (no orphan received message)."""
        if not isinstance(cut_clocks, dict) or not cut_clocks:
            return True
        clocks: Dict[str, VectorClock] = {}
        for p, vc in cut_clocks.items():
            clocks[str(p)] = vc if isinstance(vc, VectorClock) else VectorClock(
                vc if isinstance(vc, dict) else None
            )
        for p in clocks:
            vp_self = clocks[p].clocks.get(p, 0)
            for q in clocks:
                if clocks[q].clocks.get(p, 0) > vp_self:
                    return False
        return True


def _self_test() -> int:
    eng = CausalChoreographyEngine()
    e1 = eng.record_local("w1", "e1_write", write_key="shared_verdict")
    e2 = eng.record_local("w2", "e2_write", write_key="shared_verdict")
    # w1 and w2 have not synchronized via message -> concurrent write race!
    assert e1.vc.concurrent_with(e2.vc)
    races = eng.detect_write_races()
    assert len(races) == 1 and races[0].key == "shared_verdict"

    # Synchronize w1 -> w2 so subsequent write is causally ordered
    s_ev, r_ev = eng.record_message("w1", "w2", "s1", "r1")
    assert e1.vc.happens_before(r_ev.vc)
    assert eng.is_consistent_cut({"w1": s_ev.vc, "w2": r_ev.vc}) is True
    # Inconsistent cut: w2 includes r_ev (which received s_ev) but w1 cut is before s_ev!
    assert eng.is_consistent_cut({"w1": VectorClock(), "w2": r_ev.vc}) is False

    # Malformed / empty clocks are tolerated and normalized.
    vc_bad = VectorClock({"a": "not-an-int", "b": -3, "c": 2, "d": None})  # type: ignore[dict-item]
    assert vc_bad.to_dict() == {"c": 2}
    assert VectorClock(None).to_dict() == {}
    assert VectorClock(None).tick(None).to_dict() == {"None": 1}  # type: ignore[arg-type]
    assert VectorClock({"a": 1}).merge(None).to_dict() == {"a": 1}  # type: ignore[arg-type]
    assert not VectorClock({"a": 1}).leq(None)  # type: ignore[arg-type]
    assert VectorClock().concurrent_with(None) is False  # type: ignore[arg-type]
    assert eng.is_consistent_cut({}) is True
    assert eng.is_consistent_cut(None) is True  # type: ignore[arg-type]
    assert eng.is_consistent_cut({"w1": {"w1": 1}, "w2": {"w2": 1}}) is True

    # Repeated writes from one process are causally ordered (never a race).
    eng2 = CausalChoreographyEngine()
    a1 = eng2.record_local("p1", "a1", write_key="k")
    a2 = eng2.record_local("p1", "a2", write_key="k")
    assert a1.vc.happens_before(a2.vc)
    assert eng2.detect_write_races() == []
    # record_local tolerates None / non-str identifiers.
    ev_none = eng2.record_local(None, None)  # type: ignore[arg-type]
    assert ev_none.proc == "None" and ev_none.event_id == "None" and ev_none.kind == "local"

    # Three mutually concurrent writes to one key produce three race pairs.
    eng3 = CausalChoreographyEngine()
    for p in ("p1", "p2", "p3"):
        eng3.record_local(p, f"{p}_w", write_key="shared")
    races3 = eng3.detect_write_races()
    assert len(races3) == 3 and all(r.key == "shared" for r in races3)
    assert [(r.proc_a, r.proc_b) for r in races3] == [
        ("p1", "p2"), ("p1", "p3"), ("p2", "p3")], races3

    # Lamport timestamps are strictly monotone across a message handoff.
    assert r_ev.lamport_ts > s_ev.lamport_ts
    assert eng.is_consistent_cut({"w1": s_ev.vc, "w2": r_ev.vc}) is True

    # Determinism and chained causality: a later w2 write after recv is
    # causally after e1, so it must not race with it.
    eng4 = CausalChoreographyEngine()
    w1 = eng4.record_local("w1", "w1_write", write_key="k")
    eng4.record_message("w1", "w2", "s", "r")
    w2 = eng4.record_local("w2", "w2_write", write_key="k")
    assert w1.vc.happens_before(w2.vc)
    assert eng4.detect_write_races() == []
    assert eng4.detect_write_races() == eng4.detect_write_races()

    print("choreography._self_test: OK PASS (Fidge/Mattern vector clocks, write-race detection, consistent cuts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
