"""cegar — Counterexample-Guided Abstraction Refinement (CEGAR) Loop & Predicate Splitter.

Implements Clarke, Grumberg, Jha, Lu & Veith's **CEGAR** algorithm for
verifying safety properties over large or infinite state spaces:
  1. **Predicate Abstraction**: projects concrete states `s` onto bitvectors
     over a current predicate set `P = {p_1, ..., p_k}`.
  2. **Abstract Reachability**: searches for an abstract error path
     `a_0 -> a_1 -> ... -> a_m` from initial states to an unsafe state.
  3. **Feasibility / Spuriousness Check**: simulates the abstract trace on the
     concrete transition system. If feasible, returns a real concrete
     counterexample.
  4. **Craig-Style Predicate Discovery & Refinement**: if the abstract path is
     *spurious* at step `i` (the concrete post-image of `a_i` does not intersect
     `a_{i+1}`), automatically adds a separating predicate `p_{new}` that splits
     the spurious abstract state and repeats the loop!

Self-test: ``python3 -m fullagent.enforcex.cegar``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "Predicate",
    "CEGARReport",
    "CEGARVerifier",
]


@dataclass(frozen=True)
class Predicate:
    name: str
    fn: Callable[[Dict[str, Any]], bool]


@dataclass
class CEGARReport:
    safe: bool
    iterations: int
    predicates_learned: List[str] = field(default_factory=list)
    concrete_counterexample: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "iterations": self.iterations,
            "predicates_learned": list(self.predicates_learned),
            "concrete_counterexample": list(self.concrete_counterexample),
        }


def _as_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _eval_pred(pred: "Predicate", state: Dict[str, Any]) -> bool:
    """Evaluate a predicate; a crashing predicate is treated as True (fail closed)."""
    try:
        return bool(pred.fn(state))
    except Exception:
        return True


class CEGARVerifier:
    """Predicate-abstraction model checker with automatic spurious-path refinement."""

    def verify(
        self,
        concrete_states: Sequence[Dict[str, Any]],
        initial_indices: Sequence[int],
        transitions: Dict[int, Sequence[int]],
        is_unsafe: Callable[[Dict[str, Any]], bool],
        candidate_predicates: Sequence[Predicate],
        *,
        max_iters: int = 12,
    ) -> CEGARReport:
        try:
            states = list(concrete_states)
        except TypeError:
            states = []
        if not callable(is_unsafe):
            return CEGARReport(safe=False, iterations=0)
        iters = max(0, _as_int(max_iters, 12))
        n_states = len(states)

        init_list: List[int] = []
        try:
            for raw in (initial_indices or ()):
                idx = _as_int(raw, -1)
                if 0 <= idx < n_states and idx not in init_list:
                    init_list.append(idx)
        except TypeError:
            init_list = []

        succ: Dict[int, List[int]] = {}
        if isinstance(transitions, dict):
            for raw_u, raw_vs in transitions.items():
                u = _as_int(raw_u, -1)
                if not (0 <= u < n_states):
                    continue
                targets: List[int] = []
                try:
                    vs_iter = list(raw_vs)
                except TypeError:
                    continue
                for raw_v in vs_iter:
                    v = _as_int(raw_v, -1)
                    if 0 <= v < n_states and v not in targets:
                        targets.append(v)
                succ[u] = targets

        unsafe_flags: List[bool] = []
        for st in states:
            try:
                unsafe_flags.append(bool(is_unsafe(st)))
            except Exception:
                unsafe_flags.append(True)  # fail closed

        active_preds: List[Predicate] = []
        remaining_preds: List[Predicate] = []
        try:
            for cand in (candidate_predicates or ()):
                if isinstance(cand, Predicate) and callable(cand.fn):
                    remaining_preds.append(cand)
                elif hasattr(cand, "fn") and callable(getattr(cand, "fn")):
                    remaining_preds.append(Predicate(str(getattr(cand, "name", "p")),
                                                     getattr(cand, "fn")))
        except TypeError:
            remaining_preds = []

        def abstract_sig(idx: int) -> Tuple[bool, ...]:
            st = states[idx]
            return tuple(_eval_pred(p, st) for p in active_preds)

        for it in range(1, iters + 1):
            if not states:
                return CEGARReport(safe=True, iterations=it,
                                   predicates_learned=[p.name for p in active_preds])
            # 1. Partition concrete state indices by their abstract predicate bitvector
            sig_to_concretes: Dict[Tuple[bool, ...], Set[int]] = {}
            idx_to_sig: Dict[int, Tuple[bool, ...]] = {}
            for idx in range(n_states):
                sig = abstract_sig(idx)
                idx_to_sig[idx] = sig
                sig_to_concretes.setdefault(sig, set()).add(idx)

            init_sigs = {idx_to_sig[i] for i in init_list}
            unsafe_sigs = {idx_to_sig[i] for i in range(n_states) if unsafe_flags[i]}

            # Existential abstract transition relation: A -> B iff exists u in A, v in B with (u, v) in R
            abs_trans: Dict[Tuple[bool, ...], Set[Tuple[bool, ...]]] = {
                s: set() for s in sig_to_concretes
            }
            for u, succs in succ.items():
                su = idx_to_sig[u]
                for v in succs:
                    abs_trans[su].add(idx_to_sig[v])

            # 2. BFS in abstract graph from init_sigs to unsafe_sigs
            q: deque = deque([[s] for s in sorted(init_sigs)])
            seen: Set[Tuple[bool, ...]] = set(init_sigs)
            abs_path: Optional[List[Tuple[bool, ...]]] = None
            while q:
                path = q.popleft()
                last = path[-1]
                if last in unsafe_sigs:
                    abs_path = path
                    break
                for nxt in sorted(abs_trans.get(last, ())):
                    if nxt not in seen:
                        seen.add(nxt)
                        q.append(path + [nxt])

            if abs_path is None:
                # Abstract over-approximation has NO path to unsafe -> system is provably SAFE!
                return CEGARReport(
                    safe=True,
                    iterations=it,
                    predicates_learned=[p.name for p in active_preds],
                )

            # 3. Check if abstract path corresponds to a real concrete path
            frontier: Dict[int, List[int]] = {
                i: [i] for i in init_list if idx_to_sig[i] == abs_path[0]
            }
            spurious_step = -1
            for step_idx in range(len(abs_path) - 1):
                next_sig = abs_path[step_idx + 1]
                next_frontier: Dict[int, List[int]] = {}
                for u, c_path in frontier.items():
                    for v in succ.get(u, ()):
                        if idx_to_sig[v] == next_sig and v not in next_frontier:
                            next_frontier[v] = c_path + [v]
                if not next_frontier:
                    spurious_step = step_idx
                    break
                frontier = next_frontier

            if spurious_step == -1:
                # Check if any reached concrete state in final bucket is genuinely unsafe
                for u, c_path in frontier.items():
                    if unsafe_flags[u]:
                        return CEGARReport(
                            safe=False,
                            iterations=it,
                            predicates_learned=[p.name for p in active_preds],
                            concrete_counterexample=[states[i] for i in c_path],
                        )
                # The abstract path can collapse to length 1 when the initial
                # signature is already "unsafe" (it merges an unsafe state).
                # Search the final bucket concretely so genuine multi-step
                # counterexamples are still reported instead of being lost.
                final_sig = abs_path[-1]
                seen_states: Set[int] = set(frontier.keys())
                queue_c: deque = deque((u, c_path) for u, c_path in frontier.items())
                found_path: Optional[List[int]] = None
                while queue_c:
                    u, c_path = queue_c.popleft()
                    if unsafe_flags[u]:
                        found_path = c_path
                        break
                    for v in succ.get(u, ()):
                        if v in seen_states or idx_to_sig.get(v) != final_sig:
                            continue
                        seen_states.add(v)
                        queue_c.append((v, c_path + [v]))
                if found_path is not None:
                    return CEGARReport(
                        safe=False,
                        iterations=it,
                        predicates_learned=[p.name for p in active_preds],
                        concrete_counterexample=[states[i] for i in found_path],
                    )
                # The BFS reached the *signature* of an unsafe state, but the
                # concretely reachable states are all safe: the whole path is
                # spurious and refinement must split the final bucket, not
                # the initial one.
                spurious_step = len(abs_path) - 1

            # 4. Path is spurious! Refine abstraction by selecting a predicate
            # that separates reachable concrete states in the failing bucket
            # from the remaining (dead-end) states in that same bucket.
            failing_sig = abs_path[max(0, min(spurious_step, len(abs_path) - 1))]
            bucket_states = sig_to_concretes.get(failing_sig, set())
            reachable_in_bucket = set(frontier.keys()) & bucket_states
            others_in_bucket = bucket_states - reachable_in_bucket
            chosen_pred: Optional[Predicate] = None
            best_score = -1
            for cand in remaining_preds:
                vals_r = {_eval_pred(cand, states[i]) for i in reachable_in_bucket}
                vals_o = {_eval_pred(cand, states[i]) for i in others_in_bucket}
                if not vals_r and not vals_o:
                    continue
                separates = bool(vals_r and vals_o) and not (vals_r & vals_o)
                changes = vals_r != vals_o or len(vals_r | vals_o) > 1
                score = 2 if separates else (1 if changes else 0)
                if score > best_score:
                    best_score = score
                    chosen_pred = cand
                    if score == 2:
                        break
            if chosen_pred is None and remaining_preds:
                chosen_pred = remaining_preds[0]
            if chosen_pred is None:
                break
            remaining_preds.remove(chosen_pred)
            active_preds.append(chosen_pred)

        return CEGARReport(
            safe=False,
            iterations=max(0, iters),
            predicates_learned=[p.name for p in active_preds],
        )


def _self_test() -> int:
    # Concrete transition system where s0 -> s1 (validated=True), and s2 (validated=False) -> s3 (unsafe)
    # Without predicate `validated`, s1 and s2 merge into one abstract state, creating a spurious path s0 -> {s1,s2} -> s3!
    states = [
        {"id": "s0_init", "validated": True, "leaked": False},
        {"id": "s1_safe_done", "validated": True, "leaked": False},
        {"id": "s2_unvalidated", "validated": False, "leaked": False},
        {"id": "s3_unsafe_leak", "validated": False, "leaked": True},
    ]
    transitions = {
        0: [1],
        1: [1],
        2: [3],
        3: [3],
    }
    preds = [
        Predicate("is_leaked", lambda s: s["leaked"]),
        Predicate("is_validated", lambda s: s["validated"]),
    ]
    v = CEGARVerifier()
    rep = v.verify(
        concrete_states=states,
        initial_indices=[0],
        transitions=transitions,
        is_unsafe=lambda s: s["leaked"],
        candidate_predicates=preds,
    )
    assert rep.safe is True and rep.iterations >= 2
    assert "is_validated" in rep.predicates_learned

    # A genuine concrete counterexample is returned with its full path,
    # even when predicate abstraction collapses everything into one bucket.
    states2 = [
        {"id": "start", "leaked": False},
        {"id": "leak", "leaked": True},
    ]
    v2 = CEGARVerifier()
    rep2 = v2.verify(
        concrete_states=states2,
        initial_indices=[0],
        transitions={0: [1], 1: [1]},
        is_unsafe=lambda s: s["leaked"],
        candidate_predicates=[],
    )
    assert rep2.safe is False
    assert [s["id"] for s in rep2.concrete_counterexample] == ["start", "leak"], rep2.to_dict()

    # Initial signature merging an unreachable unsafe state is spurious and
    # must be refined to prove safety (regression for final-bucket refinement).
    states3 = [
        {"id": "s0", "leaked": False},
        {"id": "s1", "leaked": False},
        {"id": "s2_dead", "leaked": True},
    ]
    rep3 = CEGARVerifier().verify(
        concrete_states=states3,
        initial_indices=[0],
        transitions={0: [1], 1: [1], 2: [2]},
        is_unsafe=lambda s: s["leaked"],
        candidate_predicates=[Predicate("is_leaked", lambda s: s["leaked"])],
    )
    assert rep3.safe is True and "is_leaked" in rep3.predicates_learned, rep3.to_dict()

    # Empty systems are vacuously safe; malformed graphs are tolerated.
    empty_rep = CEGARVerifier().verify([], [0], {}, lambda s: True, [])
    assert empty_rep.safe is True
    assert CEGARVerifier().verify(
        [{"x": 1}], [99], {99: [0]}, lambda s: False, []
    ).safe is True
    assert CEGARVerifier().verify(
        [{"x": 1}], [0], {0: [99, "bad", 0]}, lambda s: False, []
    ).safe is True
    assert CEGARVerifier().verify(
        None, None, None, None, None  # type: ignore[arg-type]
    ).safe is False
    # max_iters=0 is conservative (cannot prove safety).
    assert CEGARVerifier().verify(
        [{"x": 1}], [0], {}, lambda s: False, [], max_iters=0
    ).safe is False

    # A crashing predicate must never escape; evaluation fails closed.
    class _BadPredicate:
        name = "bad"
        @staticmethod
        def fn(_state):
            raise RuntimeError("predicate exploded")

    rep_bad = CEGARVerifier().verify(
        [{"x": 1}, {"x": 2}], [0], {0: [1]}, lambda s: False, [_BadPredicate()]
    )
    assert isinstance(rep_bad.iterations, int) and isinstance(rep_bad.safe, bool)

    # Determinism: identical inputs produce identical reports.
    def _run_det():
        return CEGARVerifier().verify(
            states3, [0], {0: [1], 1: [1], 2: [2]},
            lambda s: s["leaked"],
            [Predicate("is_leaked", lambda s: s["leaked"])],
        ).to_dict()

    assert _run_det() == _run_det()

    print("cegar._self_test: OK PASS (spurious counterexample elimination & automatic predicate refinement)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
