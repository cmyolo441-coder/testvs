"""automata — Nondeterministic & Deterministic Finite Automata (NFA -> DFA) Protocol Compiler.

Implements classical automata theory in pure Python for O(N) sequence verification:
  * `NFA`: Nondeterministic Finite Automaton with epsilon-transitions
  * `epsilon_closure()` & `subset_construct()`: Powerset construction converting
    any NFA into an equivalent Deterministic Finite Automaton (`DFA`)
  * `minimize_dfa()`: Hopcroft-style partition refinement minimization
  * `intersect_dfa()`: Product automaton construction for conjunctive rules
  * Protocol sequence verification over token/section/action streams.

Self-test: ``python3 -m fullagent.enforcex.automata``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

__all__ = [
    "NFA",
    "DFA",
    "AutomataRunReport",
    "subset_construct",
    "minimize_dfa",
    "intersect_dfa",
]


@dataclass
class NFA:
    states: Set[int] = field(default_factory=set)
    alphabet: Set[str] = field(default_factory=set)
    transitions: Dict[Tuple[int, str], Set[int]] = field(default_factory=dict)  # "" = epsilon
    start_state: int = 0
    accept_states: Set[int] = field(default_factory=set)

    def add_transition(self, src: int, symbol: str, dst: int) -> None:
        try:
            self.states.add(src)
            self.states.add(dst)
        except TypeError:  # unhashable state ids — ignore, never raise
            return
        sym = "" if symbol is None else str(symbol)
        if sym:
            self.alphabet.add(sym)
        try:
            self.transitions.setdefault((src, sym), set()).add(dst)
        except TypeError:
            return

    def epsilon_closure(self, state_set: Iterable[int]) -> FrozenSet[int]:
        try:
            stack = list(state_set)
            closure: Set[int] = set(stack)
        except TypeError:
            return frozenset()
        while stack:
            s = stack.pop()
            for nxt in self.transitions.get((s, ""), ()):
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)
        return frozenset(closure)


@dataclass
class DFA:
    states: Set[int] = field(default_factory=set)
    alphabet: Set[str] = field(default_factory=set)
    transitions: Dict[Tuple[int, str], int] = field(default_factory=dict)
    start_state: int = 0
    accept_states: Set[int] = field(default_factory=set)

    def accepts(self, sequence: Sequence[str]) -> Tuple[bool, List[int]]:
        cur = self.start_state
        trace = [cur]
        try:
            symbols = list(sequence)
        except TypeError:
            return False, trace
        for sym in symbols:
            try:
                nxt = self.transitions.get((cur, sym))
            except TypeError:  # unhashable symbol
                return False, trace
            if nxt is None:
                return False, trace
            cur = nxt
            trace.append(cur)
        return (cur in self.accept_states), trace


@dataclass
class AutomataRunReport:
    accepted: bool
    trace: List[int]
    steps: int
    final_state: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "trace": list(self.trace),
            "steps": self.steps,
            "final_state": self.final_state,
        }


def subset_construct(nfa: NFA) -> DFA:
    """Convert an NFA (with epsilon transitions) to an equivalent DFA via powerset construction."""
    # Include symbols that only appear in the transition relation (hand-built
    # NFA instances may forget to populate `alphabet`).
    alphabet = set(nfa.alphabet) | {
        sym for (_, sym) in nfa.transitions.keys() if sym
    }
    start_set = nfa.epsilon_closure([nfa.start_state])
    state_map: Dict[FrozenSet[int], int] = {start_set: 0}
    worklist: List[FrozenSet[int]] = [start_set]
    dfa_trans: Dict[Tuple[int, str], int] = {}
    dfa_accept: Set[int] = set()
    if start_set & nfa.accept_states:
        dfa_accept.add(0)
    closure_cache: Dict[FrozenSet[int], FrozenSet[int]] = {start_set: start_set}

    while worklist:
        cur_set = worklist.pop(0)
        cur_id = state_map[cur_set]
        for sym in sorted(alphabet):
            move_targets: Set[int] = set()
            for s in cur_set:
                move_targets |= nfa.transitions.get((s, sym), set())
            if not move_targets:
                continue
            frozen_move = frozenset(move_targets)
            nxt_set = closure_cache.get(frozen_move)
            if nxt_set is None:
                nxt_set = nfa.epsilon_closure(move_targets)
                closure_cache[frozen_move] = nxt_set
            if nxt_set not in state_map:
                nxt_id = len(state_map)
                state_map[nxt_set] = nxt_id
                worklist.append(nxt_set)
                if nxt_set & nfa.accept_states:
                    dfa_accept.add(nxt_id)
            dfa_trans[(cur_id, sym)] = state_map[nxt_set]

    return DFA(
        states=set(state_map.values()),
        alphabet=alphabet,
        transitions=dfa_trans,
        start_state=0,
        accept_states=dfa_accept,
    )


def _reachable_states(dfa: DFA) -> Set[int]:
    """States reachable from the start state (with transition symbols)."""
    reachable: Set[int] = set()
    frontier = [dfa.start_state]
    while frontier:
        s = frontier.pop()
        if s in reachable:
            continue
        reachable.add(s)
        for (src, _sym), dst in dfa.transitions.items():
            if src == s and dst not in reachable:
                frontier.append(dst)
    return reachable


def minimize_dfa(dfa: DFA) -> DFA:
    """Minimize a DFA via partition refinement (Hopcroft/Moore equivalence classes).

    Unreachable states are removed first (they are not part of the language
    and otherwise survive minimization as phantom equivalence classes).
    """
    if not dfa.states:
        return dfa
    reachable = _reachable_states(dfa)
    reachable &= set(dfa.states) | {dfa.start_state}
    if not reachable:
        return dfa
    accept = set(dfa.accept_states) & reachable
    non_accept = reachable - accept
    partitions: List[Set[int]] = [p for p in (accept, non_accept) if p]
    if not partitions:
        return dfa
    if len(partitions) == 1:
        transitions = {
            (0, sym): 0
            for (s, sym), _dst in dfa.transitions.items() if s in reachable
        }
        return DFA(
            states={0},
            alphabet=set(dfa.alphabet),
            transitions=transitions,
            start_state=0,
            accept_states={0} if dfa.start_state in dfa.accept_states else set(),
        )

    while True:
        state_to_part = {s: idx for idx, part in enumerate(partitions) for s in part}
        new_parts: List[Set[int]] = []
        for part in partitions:
            buckets: Dict[Tuple[Tuple[str, int], ...], Set[int]] = {}
            for s in part:
                sig = tuple(
                    (sym, state_to_part.get(dfa.transitions[(s, sym)], -1))
                    for sym in sorted(dfa.alphabet)
                    if (s, sym) in dfa.transitions
                )
                buckets.setdefault(sig, set()).add(s)
            new_parts.extend(buckets.values())
        if len(new_parts) == len(partitions):
            break
        partitions = new_parts

    rep_map = {s: idx for idx, part in enumerate(partitions) for s in part}
    min_trans: Dict[Tuple[int, str], int] = {}
    for (s, sym), dst in dfa.transitions.items():
        if s in rep_map and dst in rep_map:
            min_trans[(rep_map[s], sym)] = rep_map[dst]
    start = dfa.start_state if dfa.start_state in rep_map else next(iter(reachable))
    return DFA(
        states=set(range(len(partitions))),
        alphabet=set(dfa.alphabet),
        transitions=min_trans,
        start_state=rep_map[start],
        accept_states={rep_map[s] for s in accept},
    )


def intersect_dfa(dfa1: DFA, dfa2: DFA) -> DFA:
    """Construct the product DFA accepting `L(dfa1) intersect L(dfa2)`."""
    if not isinstance(dfa1, DFA) or not isinstance(dfa2, DFA):
        return DFA()
    alpha = set(dfa1.alphabet) & set(dfa2.alphabet)
    start_pair = (dfa1.start_state, dfa2.start_state)
    pair_map: Dict[Tuple[int, int], int] = {start_pair: 0}
    queue = [start_pair]
    trans: Dict[Tuple[int, str], int] = {}
    accepts: Set[int] = set()
    if start_pair[0] in dfa1.accept_states and start_pair[1] in dfa2.accept_states:
        accepts.add(0)

    while queue:
        s1, s2 = queue.pop(0)
        cid = pair_map[(s1, s2)]
        for sym in sorted(alpha):
            n1 = dfa1.transitions.get((s1, sym))
            n2 = dfa2.transitions.get((s2, sym))
            if n1 is not None and n2 is not None:
                np = (n1, n2)
                if np not in pair_map:
                    nid = len(pair_map)
                    pair_map[np] = nid
                    queue.append(np)
                    if n1 in dfa1.accept_states and n2 in dfa2.accept_states:
                        accepts.add(nid)
                trans[(cid, sym)] = pair_map[np]

    return DFA(
        states=set(pair_map.values()),
        alphabet=alpha,
        transitions=trans,
        start_state=0,
        accept_states=accepts,
    )


def _self_test() -> int:
    # Build NFA for protocol: "plan" -> ("read" | eps) -> "edit" -> "test"
    nfa = NFA(start_state=0, accept_states={3})
    nfa.add_transition(0, "plan", 1)
    nfa.add_transition(1, "read", 2)
    nfa.add_transition(1, "", 2)      # epsilon transition
    nfa.add_transition(2, "edit", 2)
    nfa.add_transition(2, "test", 3)

    dfa = minimize_dfa(subset_construct(nfa))
    ok1, _ = dfa.accepts(["plan", "read", "edit", "test"])
    ok2, _ = dfa.accepts(["plan", "edit", "edit", "test"])
    bad1, _ = dfa.accepts(["edit", "test"])
    assert ok1 and ok2 and not bad1

    # Determinization must preserve the language exactly (brute force <= 3).
    raw_dfa = subset_construct(nfa)
    alphabet = ["plan", "read", "edit", "test"]

    def _nfa_accepts(machine: NFA, word: List[str]) -> bool:
        cur = machine.epsilon_closure({machine.start_state})
        for sym in word:
            nxt = set()
            for s in cur:
                nxt |= machine.transitions.get((s, sym), set())
            cur = machine.epsilon_closure(nxt)
        return bool(cur & machine.accept_states)

    def _enumerate(prefix: List[str], depth: int):
        yield list(prefix)
        if depth:
            for sym in alphabet:
                yield from _enumerate(prefix + [sym], depth - 1)

    for w in _enumerate([], 3):
        assert raw_dfa.accepts(w)[0] == _nfa_accepts(nfa, w), w
        assert dfa.accepts(w)[0] == _nfa_accepts(nfa, w), w

    # Epsilon cycles terminate and are fully closed.
    cyc = NFA(start_state=0, accept_states={3})
    cyc.add_transition(0, "", 1)
    cyc.add_transition(1, "", 0)
    cyc.add_transition(1, "", 3)
    cyc.add_transition(0, "go", 2)
    cyc.add_transition(2, "go", 3)
    assert cyc.epsilon_closure([0]) == frozenset({0, 1, 3})
    cdfa = subset_construct(cyc)
    assert cdfa.accepts([])[0] is True and cdfa.accepts(["go"])[0] is False
    assert cdfa.accepts(["go", "go"])[0] is True

    # Symbols present only in `transitions` are still determinized.
    hand = NFA(start_state=0, accept_states={1})
    hand.transitions[(0, "x")] = {1}
    assert subset_construct(hand).accepts(["x"])[0] is True

    # Unreachable states are pruned by minimization; language is preserved.
    unrich = DFA(
        states={0, 1, 2}, alphabet={"a"}, transitions={(0, "a"): 1, (1, "a"): 1, (2, "a"): 2},
        start_state=0, accept_states={1},
    )
    m = minimize_dfa(unrich)
    assert len(m.states) == 2 and 2 not in m.states, m
    assert m.accepts(["a"])[0] and m.accepts(["a", "a"])[0] and not m.accepts([])[0]

    # Empty / malformed automata never raise and accept nothing.
    empty = DFA()
    assert empty.accepts([])[0] is False
    assert minimize_dfa(DFA()).states == set()
    assert minimize_dfa(DFA(states=set(), start_state=0)).states == set()
    assert empty.accepts(None)[0] is False  # type: ignore[arg-type]
    assert empty.accepts(42)[0] is False  # type: ignore[arg-type]
    assert not intersect_dfa(None, empty).states  # type: ignore[arg-type]

    # Product construction intersects languages (starts with a AND ends with b).
    starts_a = DFA(states={0, 1, 2}, alphabet={"a", "b"},
                   transitions={(0, "a"): 1, (1, "a"): 1, (1, "b"): 1,
                                (0, "b"): 2, (2, "a"): 2, (2, "b"): 2},
                   start_state=0, accept_states={1})
    ends_b = DFA(states={0, 1}, alphabet={"a", "b"},
                 transitions={(0, "a"): 0, (0, "b"): 1, (1, "a"): 0, (1, "b"): 1},
                 start_state=0, accept_states={1})
    prod = intersect_dfa(starts_a, ends_b)
    assert prod.accepts(["a", "b"])[0]
    assert not prod.accepts(["b", "b"])[0]
    assert not prod.accepts(["a", "a"])[0]
    # Determinism of construction and minimization.
    assert subset_construct(nfa).transitions == subset_construct(nfa).transitions
    assert minimize_dfa(unrich).transitions == minimize_dfa(unrich).transitions

    print("automata._self_test: OK PASS (NFA epsilon-closure, subset construction, Hopcroft minimization)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
