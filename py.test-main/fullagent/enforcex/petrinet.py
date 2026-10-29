"""petrinet — Place/Transition Petri Net Workflow & Concurrency Safety Verifier.

Models concurrent agent & tool workflows as a Petri Net $N = (P, T, Pre, Post, M_0)$
and explores the reachability graph via breadth-first search to verify:
  * Deadlock Freedom: whether any non-terminal reachable marking has zero
    enabled transitions.
  * k-Boundedness: whether any place ever accumulates more than $k$ tokens
    (resource leak / unbounded queue).
  * Mutual Exclusion: whether two critical-section places are ever simultaneously
    marked (`M[p1] > 0 and M[p2] > 0`).

The exploration is bounded by ``max_states`` (reported via ``truncated``) and
all weights/markings are sanitised, so malformed nets degrade to empty
transitions instead of raising. ``can_reach()`` performs a bounded
coverability search for a target marking, ``enabled_transitions()`` /
``fire_transition()`` expose single-step reachability edges.

Self-test: ``python3 -m fullagent.enforcex.petrinet``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Transition",
    "PetriReachabilityReport",
    "PetriNetVerifier",
]

Marking = Tuple[Tuple[str, int], ...]

_MAX_LISTED_MARKINGS = 100
_MAX_WEIGHT = 1_000_000


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clean_weights(raw: Any) -> Dict[str, int]:
    """Coerce a transition's arc weights; drop malformed/zero entries."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, int] = {}
    for place, weight in raw.items():
        try:
            key = str(place)
        except Exception:
            continue
        if not key:
            continue
        w = _safe_int(weight, 0)
        if w <= 0:
            continue
        out[key] = min(w, _MAX_WEIGHT)
    return out


@dataclass(frozen=True)
class Transition:
    name: str
    pre: Dict[str, int]
    post: Dict[str, int]


@dataclass
class PetriReachabilityReport:
    reachable_states: int = 0
    deadlock_free: bool = True
    bounded: bool = True
    mutex_safe: bool = True
    max_tokens_seen: int = 0
    deadlock_markings: List[Dict[str, int]] = field(default_factory=list)
    mutex_violations: List[Dict[str, int]] = field(default_factory=list)
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reachable_states": self.reachable_states,
            "deadlock_free": self.deadlock_free,
            "bounded": self.bounded,
            "mutex_safe": self.mutex_safe,
            "max_tokens_seen": self.max_tokens_seen,
            "deadlock_markings": list(self.deadlock_markings),
            "mutex_violations": list(self.mutex_violations),
            "truncated": self.truncated,
        }


class PetriNetVerifier:
    def __init__(self, initial_marking: Dict[str, int]) -> None:
        clean = _clean_weights(initial_marking)
        self.initial: Dict[str, int] = clean
        self.transitions: List[Transition] = []

    def add_transition(self, name: str, pre: Dict[str, int], post: Dict[str, int]) -> None:
        try:
            safe_name = str(name)
        except Exception:
            safe_name = ""
        self.transitions.append(Transition(
            name=safe_name,
            pre=_clean_weights(pre),
            post=_clean_weights(post),
        ))

    @staticmethod
    def _freeze(m: Dict[str, int]) -> Marking:
        return tuple(sorted((k, v) for k, v in m.items() if v > 0))

    @staticmethod
    def _enabled(m: Dict[str, int], t: Transition) -> bool:
        return all(m.get(p, 0) >= w for p, w in t.pre.items())

    @staticmethod
    def _fire(m: Dict[str, int], t: Transition) -> Dict[str, int]:
        nxt = dict(m)
        for p, w in t.pre.items():
            nxt[p] = nxt.get(p, 0) - w
            if nxt[p] <= 0:
                nxt.pop(p, None)
        for p, w in t.post.items():
            nxt[p] = nxt.get(p, 0) + w
        return nxt

    def enabled_transitions(self, marking: Dict[str, int]) -> List[str]:
        """Names of transitions enabled in ``marking`` (malformed -> [])."""
        try:
            current = _clean_weights(marking)
        except Exception:
            return []
        return [t.name for t in self.transitions if self._enabled(current, t)]

    def fire_transition(self, marking: Dict[str, int],
                        name: str) -> Optional[Dict[str, int]]:
        """Fire the first transition named ``name``; None if not enabled."""
        try:
            current = _clean_weights(marking)
        except Exception:
            return None
        target = str(name)
        for t in self.transitions:
            if t.name == target and self._enabled(current, t):
                return self._fire(current, t)
        return None

    def _normalized_transitions(self) -> List[Transition]:
        out: List[Transition] = []
        for t in self.transitions:
            if not isinstance(t, Transition):
                continue
            try:
                name = str(t.name)
            except Exception:
                name = ""
            out.append(Transition(name=name, pre=_clean_weights(t.pre),
                                  post=_clean_weights(t.post)))
        return out

    def verify(
        self,
        *,
        terminal_places: Sequence[str] = ("done",),
        mutex_pairs: Sequence[Tuple[str, str]] = (),
        bound_k: int = 10,
        max_states: int = 500,
    ) -> PetriReachabilityReport:
        # Sanitise every argument: verify() must never raise on bad input.
        try:
            if isinstance(terminal_places, str):
                term_set = {terminal_places}
            else:
                term_set = {str(p) for p in (terminal_places or ())}
        except Exception:
            term_set = set()
        pairs: List[Tuple[str, str]] = []
        try:
            for pair in (mutex_pairs or ()):
                if isinstance(pair, (tuple, list)) and len(pair) == 2:
                    p1, p2 = str(pair[0]), str(pair[1])
                    if p1 and p2 and p1 != p2:
                        pairs.append((p1, p2))
        except Exception:
            pairs = []
        bound_k = max(0, _safe_int(bound_k, 10))
        max_states = max(1, _safe_int(max_states, 500))
        transitions = self._normalized_transitions()

        start = dict(self.initial)
        start_f = self._freeze(start)
        visited: Set[Marking] = {start_f}
        queue: Deque[Dict[str, int]] = deque([start])
        deadlocks: List[Dict[str, int]] = []
        mutex_viols: List[Dict[str, int]] = []
        mutex_seen: Set[Marking] = set()
        max_tok = max(start.values(), default=0)
        truncated = False

        def _mutex_check(marking: Dict[str, int],
                         frozen: Marking) -> None:
            if not pairs or len(mutex_viols) >= _MAX_LISTED_MARKINGS:
                return
            for p1, p2 in pairs:
                if marking.get(p1, 0) > 0 and marking.get(p2, 0) > 0:
                    if frozen not in mutex_seen:
                        mutex_seen.add(frozen)
                        mutex_viols.append(dict(marking))
                    return

        while queue and len(visited) < max_states:
            cur = queue.popleft()
            cur_f = self._freeze(cur)
            if cur:
                cur_max = max(cur.values())
                if cur_max > max_tok:
                    max_tok = cur_max
            _mutex_check(cur, cur_f)

            enabled_ts = [t for t in transitions if self._enabled(cur, t)]
            if not enabled_ts:
                if not any(cur.get(tp, 0) > 0 for tp in term_set):
                    if len(deadlocks) < _MAX_LISTED_MARKINGS:
                        deadlocks.append(dict(cur))
                continue

            for t in enabled_ts:
                nxt = self._fire(cur, t)
                if nxt:
                    nxt_max = max(nxt.values())
                    if nxt_max > max_tok:
                        max_tok = nxt_max
                nxt_f = self._freeze(nxt)
                # Mutex is a safety property: check even over-bound states.
                _mutex_check(nxt, nxt_f)
                if any(v > bound_k for v in nxt.values()):
                    continue
                if nxt_f not in visited:
                    if len(visited) >= max_states:
                        truncated = True
                        continue
                    visited.add(nxt_f)
                    queue.append(nxt)

        if queue:
            # The frontier still holds states that were never expanded.
            truncated = True

        return PetriReachabilityReport(
            reachable_states=len(visited),
            deadlock_free=(len(deadlocks) == 0),
            bounded=(max_tok <= bound_k),
            mutex_safe=(len(mutex_viols) == 0),
            max_tokens_seen=max_tok,
            deadlock_markings=deadlocks,
            mutex_violations=mutex_viols,
            truncated=truncated,
        )

    def can_reach(
        self,
        target: Dict[str, int],
        *,
        max_states: int = 5_000,
    ) -> bool:
        """Bounded coverability check: can some reachable marking cover
        ``target`` component-wise? Malformed targets return False.

        Token counts are saturated at a cap derived from the target, the
        initial marking and the largest arc weight, which keeps the search
        finite on unbounded nets while preserving coverability of ``target``.
        """
        try:
            goal = _clean_weights(target)
        except Exception:
            return False
        if not goal:
            return True
        max_states = max(1, _safe_int(max_states, 5_000))
        transitions = self._normalized_transitions()
        cap = max(
            [w for t in transitions for w in list(t.pre.values()) +
             list(t.post.values())] + list(self.initial.values()) +
            list(goal.values()) + [1]
        )
        cap = min(cap, _MAX_WEIGHT)

        start = dict(self.initial)
        visited: Set[Marking] = {self._freeze(start)}
        queue: Deque[Dict[str, int]] = deque([start])
        while queue and len(visited) < max_states:
            cur = queue.popleft()
            if all(cur.get(p, 0) >= w for p, w in goal.items()):
                return True
            for t in transitions:
                if not self._enabled(cur, t):
                    continue
                nxt = self._fire(cur, t)
                # Saturate: more than `cap` tokens in a place never helps
                # enable a transition or cover the target.
                for p in list(nxt):
                    if nxt[p] > cap:
                        nxt[p] = cap
                nf = self._freeze(nxt)
                if nf not in visited:
                    visited.add(nf)
                    queue.append(nxt)
        return False


def _self_test() -> int:
    # Two workers sharing 1 lock mutex -> verify mutual exclusion & deadlock freedom
    net = PetriNetVerifier({"w1_idle": 1, "w2_idle": 1, "lock": 1})
    net.add_transition("w1_acquire", {"w1_idle": 1, "lock": 1}, {"w1_crit": 1})
    net.add_transition("w1_release", {"w1_crit": 1}, {"w1_done": 1, "lock": 1})
    net.add_transition("w2_acquire", {"w2_idle": 1, "lock": 1}, {"w2_crit": 1})
    net.add_transition("w2_release", {"w2_crit": 1}, {"w2_done": 1, "lock": 1})
    net.add_transition("join", {"w1_done": 1, "w2_done": 1}, {"done": 1})

    rep = net.verify(terminal_places=("done",), mutex_pairs=[("w1_crit", "w2_crit")])
    assert rep.deadlock_free and rep.mutex_safe and rep.bounded, rep.to_dict()
    assert not rep.truncated

    # 1. single-step reachability edges
    assert "w1_acquire" in net.enabled_transitions({"w1_idle": 1, "lock": 1})
    assert "w1_release" not in net.enabled_transitions({"w1_idle": 1, "lock": 1})
    fired = net.fire_transition({"w1_idle": 1, "lock": 1}, "w1_acquire")
    assert fired == {"w1_crit": 1}, fired
    assert net.fire_transition({"w1_idle": 1}, "w1_acquire") is None
    assert net.fire_transition(None, "w1_acquire") is None  # type: ignore[arg-type]

    # 2. deadlock detection on a blocked workflow
    dead = PetriNetVerifier({"start": 1})
    dead.add_transition("wait_for_missing", {"token": 1}, {"done": 1})
    rep_dead = dead.verify(terminal_places=("done",))
    assert not rep_dead.deadlock_free, rep_dead.to_dict()
    assert rep_dead.deadlock_markings and rep_dead.deadlock_markings[0] == {"start": 1}

    # 3. unbounded producer is flagged, growth is capped by bound_k
    grow = PetriNetVerifier({"p": 1})
    grow.add_transition("grow", {"p": 1}, {"p": 1, "q": 1})
    rep_grow = grow.verify(bound_k=3, terminal_places=())
    assert not rep_grow.bounded and rep_grow.max_tokens_seen == 4, rep_grow.to_dict()
    assert rep_grow.reachable_states == 4

    # 4. coverability: q can grow past any target; a missing place cannot
    assert grow.can_reach({"q": 2})
    assert grow.can_reach({"q": 10})
    assert not grow.can_reach({"absent": 1})
    assert grow.can_reach({})

    # 5. mutex violations are detected even in over-bound states that are
    #    not enqueued (safety property must not be missed by the bound)
    both = PetriNetVerifier({"s": 1})
    both.add_transition("both", {"s": 1}, {"c1": 1, "c2": 1, "big": 50})
    rep_both = both.verify(mutex_pairs=[("c1", "c2")], bound_k=10)
    assert not rep_both.mutex_safe and not rep_both.bounded, rep_both.to_dict()

    # 6. state cap is reported instead of silently claiming full coverage
    chain = PetriNetVerifier({"s0": 1})
    for i in range(5):
        chain.add_transition(f"t{i}", {f"s{i}": 1}, {f"s{i + 1}": 1})
    rep_chain = chain.verify(max_states=2, terminal_places=("s5",))
    assert rep_chain.truncated and rep_chain.reachable_states == 2

    # 7. malformed nets / arguments never raise
    mal = PetriNetVerifier(None)  # type: ignore[arg-type]
    mal.add_transition(None, None, {"x": "bad"})  # type: ignore[arg-type]
    mal.add_transition("neg", {"x": -5}, {"y": 0})
    rep_mal = mal.verify(
        terminal_places=None,  # type: ignore[arg-type]
        mutex_pairs=[None, ("a",), ("a", "b")],  # type: ignore[list-item]
        bound_k="junk",  # type: ignore[arg-type]
        max_states=None,  # type: ignore[arg-type]
    )
    assert isinstance(rep_mal.to_dict(), dict)
    assert not mal.can_reach({"x": 1})
    assert mal.can_reach(None)  # empty goal is trivially covered

    print("petrinet._self_test: OK PASS (reachability BFS, mutex verification, deadlock freedom, "
          "bounded coverability, truncation reporting)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
