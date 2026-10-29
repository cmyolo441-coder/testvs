"""satisolver — DPLL Boolean Satisfiability (SAT), MaxSAT & Minimal Unsatisfiable Core Solver.

Implements a real Davis-Putnam-Logemann-Loveland (DPLL) SAT solver with:
  * Unit Propagation (`_unit_propagate`)
  * Pure Literal Elimination (`_pure_literal_assign`)
  * Weighted MaxSAT branch-and-bound solver (finds the assignment that
    maximizes total satisfied clause weight when hard + soft rules conflict)
  * Deletion-based Minimal Unsatisfiable Subset (MUS) extractor to pinpoint
    the exact minimal set of mutually contradictory rules in a system prompt.

Self-test: ``python3 -m fullagent.enforcex.satisolver``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "CNFClause",
    "SATComplianceResult",
    "MaxSATSolver",
    "dpll_solve",
    "extract_mus",
    "solve_maxsat",
]

# A literal is an int: +k means variable k is True, -k means variable k is False.

_MAX_WEIGHT = 1e9
_MAX_SEARCH_NODES = 200_000
_MAX_CLAUSE_ID = 256
_MAX_DESCRIPTION = 500


@dataclass(frozen=True)
class CNFClause:
    clause_id: str
    literals: Tuple[int, ...]
    weight: float = 1.0
    hard: bool = True
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "literals": list(self.literals),
            "weight": self.weight,
            "hard": self.hard,
            "description": self.description,
        }


@dataclass
class SATComplianceResult:
    satisfiable: bool = True
    assignment: Dict[int, bool] = field(default_factory=dict)
    satisfied_weight: float = 0.0
    total_weight: float = 0.0
    violated_clauses: List[str] = field(default_factory=list)
    unsat_core: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "satisfiable": self.satisfiable,
            "assignment": dict(self.assignment),
            "satisfied_weight": round(self.satisfied_weight, 3),
            "total_weight": round(self.total_weight, 3),
            "violated_clauses": list(self.violated_clauses),
            "unsat_core": list(self.unsat_core),
        }


def _as_literals(raw: Any) -> Tuple[int, ...]:
    """Sanitize a clause into a deduplicated tuple of non-zero ints.

    Tolerates tuples/lists/iterables and objects exposing ``literals``.
    Non-integer literals are ignored (never raise on malformed input).
    """
    if raw is None or isinstance(raw, (str, bytes)):
        return ()
    if hasattr(raw, "literals"):
        try:
            raw = raw.literals
        except Exception:
            return ()
    try:
        iterator = iter(raw)
    except Exception:
        return ()
    out: List[int] = []
    seen: Set[int] = set()
    try:
        for lit in iterator:
            if isinstance(lit, bool):
                lit = int(lit)
            if not isinstance(lit, int):
                continue
            if lit == 0:
                continue
            if lit not in seen:
                seen.add(lit)
                out.append(lit)
    except Exception:
        pass
    return tuple(out)


def _is_tautology(literals: Tuple[int, ...]) -> bool:
    present = set(literals)
    return any(-lit in present for lit in present)


def _prepare_clauses(clauses: Any) -> List[Tuple[int, ...]]:
    """Drop tautologies (always satisfied) but keep empty clauses (UNSAT)."""
    prepared: List[Tuple[int, ...]] = []
    try:
        iterator = iter(clauses or ())
    except Exception:
        return prepared
    for clause in iterator:
        literals = _as_literals(clause)
        if _is_tautology(literals):
            continue
        prepared.append(literals)
    return prepared


def _simplify_clauses(clauses: List[Tuple[int, ...]], lit: int) -> Optional[List[Tuple[int, ...]]]:
    """Assign `lit = True` and simplify `clauses`. Returns None if an empty clause is produced."""
    neg = -lit
    out: List[Tuple[int, ...]] = []
    for c in clauses:
        if lit in c:
            continue  # Clause satisfied
        if neg in c:
            rem = tuple(x for x in c if x != neg)
            if not rem:
                return None  # Conflict (empty clause)
            out.append(rem)
        else:
            out.append(c)
    return out


def _propagate(work: List[Tuple[int, ...]],
               assign: Dict[int, bool]) -> Optional[List[Tuple[int, ...]]]:
    """Unit propagation + pure-literal elimination to a fixpoint."""
    while True:
        if any(len(c) == 0 for c in work):
            return None
        lit: Optional[int] = None
        for c in work:
            if len(c) == 1:
                lit = c[0]
                break
        if lit is None:
            all_lits: Set[int] = set()
            for c in work:
                all_lits.update(c)
            for c in work:
                for x in c:
                    if -x not in all_lits:
                        lit = x
                        break
                if lit is not None:
                    break
        if lit is None:
            return work
        var = abs(lit)
        val = lit > 0
        if var in assign:
            if assign[var] != val:
                return None
            # Already assigned; occurrences should be gone, but simplify
            # defensively so a malformed caller can never loop forever.
        assign[var] = val
        simplified = _simplify_clauses(work, lit)
        if simplified is None:
            return None
        work = simplified


def _coerce_assignment(assignment: Any) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    if not isinstance(assignment, dict):
        return out
    try:
        items = list(assignment.items())[:100_000]
    except Exception:
        return out
    for key, value in items:
        try:
            var = abs(int(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if var == 0:
            continue
        if isinstance(value, str):
            val = value.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(value, (bool, int)):
            val = bool(value)
        else:
            continue
        out.setdefault(var, val)
    return out


def _complete_assignment(assign: Dict[int, bool],
                         variables: Sequence[int]) -> Dict[int, bool]:
    complete = dict(assign)
    for var in variables:
        complete.setdefault(var, False)
    return complete


def dpll_solve(
    clauses: Sequence[Sequence[int]],
    assignment: Optional[Dict[int, bool]] = None,
) -> Optional[Dict[int, bool]]:
    """Real iterative DPLL SAT solver with unit propagation and
    pure-literal elimination.

    Handles empty clause sets (SAT), tautologies (dropped), contradictory
    units (UNSAT) and pre-supplied (partial) assignments.  The returned
    assignment is total over the formula's variables, with unconstrained
    variables defaulting to False for determinism.
    """
    try:
        work = _prepare_clauses(clauses)
        initial = _coerce_assignment(assignment)
        variables = sorted({abs(lit) for c in work for lit in c})
        # Honor a caller-supplied partial assignment before search.
        for var in sorted(initial):
            lit = var if initial[var] else -var
            simplified = _simplify_clauses(work, lit)
            if simplified is None:
                return None
            work = simplified
        stack: List[Tuple[List[Tuple[int, ...]], Dict[int, bool]]] = [
            (work, initial)
        ]
        while stack:
            current, current_assign = stack.pop()
            simplified = _propagate(current, current_assign)
            if simplified is None:
                continue
            if not simplified:
                return _complete_assignment(current_assign, variables)
            counts: Dict[int, int] = {}
            for c in simplified:
                for x in c:
                    counts[x] = counts.get(x, 0) + 1
            if not counts:
                return _complete_assignment(current_assign, variables)
            # Deterministic DLIS-lite: most frequent, then smallest var,
            # then the positive literal.
            best_lit = max(counts, key=lambda k: (counts[k], -abs(k), k))
            # Push falsifying branch first so the satisfying branch is
            # explored first (matches historical behavior).
            for choice in (-best_lit, best_lit):
                next_clauses = _simplify_clauses(simplified, choice)
                if next_clauses is not None:
                    next_assign = dict(current_assign)
                    next_assign[abs(choice)] = choice > 0
                    stack.append((next_clauses, next_assign))
        return None
    except Exception:
        # Never raise: malformed input degrades to "no model found".
        return None


def _clause_id(clause: Any, index: int) -> str:
    try:
        cid = getattr(clause, "clause_id", None)
        if cid is None and isinstance(clause, dict):
            cid = clause.get("clause_id")
        return str(cid)[:_MAX_CLAUSE_ID] if cid is not None else f"c{index}"
    except Exception:
        return f"c{index}"


def _clause_literals(clause: Any) -> Tuple[int, ...]:
    if isinstance(clause, dict):
        return _as_literals(clause.get("literals", ()))
    return _as_literals(clause)


def _coerce_cnf_clauses(clauses: Any) -> List[CNFClause]:
    """Tolerantly coerce rule-ish inputs into CNFClause objects."""
    out: List[CNFClause] = []
    try:
        iterator = iter(clauses or ())
    except Exception:
        return out
    for idx, raw in enumerate(iterator):
        try:
            if isinstance(raw, dict):
                weight = raw.get("weight", 1.0)
                hard = raw.get("hard", True)
                desc = raw.get("description", "")
            else:
                weight = getattr(raw, "weight", 1.0)
                hard = getattr(raw, "hard", True)
                desc = getattr(raw, "description", "")
            try:
                weight_f = float(weight)
            except (TypeError, ValueError, OverflowError):
                weight_f = 1.0
            if not math.isfinite(weight_f) or weight_f < 0.0:
                weight_f = 0.0
            weight_f = min(weight_f, _MAX_WEIGHT)
            out.append(CNFClause(
                clause_id=_clause_id(raw, idx),
                literals=_clause_literals(raw),
                weight=weight_f,
                hard=bool(hard),
                description=str(desc)[:_MAX_DESCRIPTION],
            ))
        except Exception:
            continue
    return out


def extract_mus(clauses: Sequence[CNFClause]) -> List[str]:
    """Extract a Minimal Unsatisfiable Subset (MUS) of clause IDs using
    deletion-based reduction.  Returns [] when the formula is satisfiable."""
    c_list = _coerce_cnf_clauses(clauses)
    if dpll_solve([c.literals for c in c_list]) is not None:
        return []  # Satisfiable -> no UNSAT core

    core = list(c_list)
    i = 0
    while i < len(core):
        candidate = core[:i] + core[i + 1:]
        if candidate and dpll_solve([c.literals for c in candidate]) is None:
            # Clause i is redundant for unsatisfiability; drop it
            core = candidate
        else:
            i += 1
    return [c.clause_id for c in core]


def _maxsat_search(work: Sequence[CNFClause]) -> Tuple[float, Dict[int, bool]]:
    """Weighted MaxSAT branch-and-bound with deterministic tie-breaking.

    ``work`` must contain no tautologies and no empty clauses.  Returns
    (satisfied_weight, assignment).  Search is node-capped so pathological
    inputs terminate quickly with the best assignment found so far.
    """
    variables = sorted({abs(lit) for c in work for lit in c.literals})
    if not variables:
        return 0.0, {}
    initial_weight = sum(c.weight for c in work)
    # Remaining clause tuples: (literals, weight, clause_id)
    root: List[Tuple[Tuple[int, ...], float, str]] = [
        (c.literals, c.weight, c.clause_id) for c in work
    ]
    best_weight = -1.0
    best_assign: Dict[int, bool] = {}
    nodes = 0
    stack: List[Tuple[List[Tuple[Tuple[int, ...], float, str]],
                      Dict[int, bool], float, float]] = [
        (root, {}, 0.0, initial_weight)
    ]
    while stack:
        remaining, assign, sat_weight, remaining_weight = stack.pop()
        nodes += 1
        if nodes > _MAX_SEARCH_NODES:
            break
        # Optimistic upper bound: every remaining clause fully satisfied.
        if sat_weight + remaining_weight <= best_weight:
            continue
        if not remaining:
            if sat_weight > best_weight:
                best_weight = sat_weight
                best_assign = assign
            continue
        counts: Dict[int, int] = {}
        for literals, _w, _cid in remaining:
            for lit in literals:
                counts[lit] = counts.get(lit, 0) + 1
        if not counts:
            # Only empty clauses left -> cannot satisfy anything more.
            if sat_weight > best_weight:
                best_weight = sat_weight
                best_assign = assign
            continue
        best_lit = max(counts, key=lambda k: (counts[k], -abs(k), k))
        # Explore False first, push True first (LIFO).
        for choice in (best_lit, -best_lit):
            gained = 0.0
            lost = 0.0
            next_remaining: List[Tuple[Tuple[int, ...], float, str]] = []
            for literals, weight, cid in remaining:
                if choice in literals:
                    gained += weight
                elif -choice in literals:
                    shortened = tuple(x for x in literals if x != -choice)
                    if shortened:
                        next_remaining.append((shortened, weight, cid))
                    else:
                        lost += weight  # clause now permanently violated
                else:
                    next_remaining.append((literals, weight, cid))
            next_assign = dict(assign)
            next_assign[abs(choice)] = choice > 0
            stack.append((next_remaining, next_assign,
                          sat_weight + gained,
                          max(0.0, remaining_weight - gained - lost)))
    if best_weight < 0.0:
        # Node cap hit before any leaf: fall back to the all-False model,
        # which is always available (clauses simply go unsatisfied).
        best_weight = 0.0
        best_assign = {var: False for var in variables}
    best_assign = _complete_assignment(best_assign, variables)
    return best_weight, best_assign


def solve_maxsat(clauses: Sequence[CNFClause]) -> SATComplianceResult:
    """Solve weighted MaxSAT over CNFClauses, returning the optimal
    assignment (within a bounded search), violating clause IDs and the
    MUS when the clause set is unsatisfiable.

    This is exact for the small rule sets enforcex uses; for very large
    formulas the branch-and-bound is node-capped and returns the best
    assignment found, deterministically.
    """
    c_list = _coerce_cnf_clauses(clauses)
    total_w = sum(c.weight for c in c_list)
    if not c_list:
        return SATComplianceResult(satisfiable=True, total_weight=0.0)

    # First try exact DPLL on all clauses.
    exact = dpll_solve([c.literals for c in c_list])
    if exact is not None:
        return SATComplianceResult(
            satisfiable=True,
            assignment=exact,
            satisfied_weight=total_w,
            total_weight=total_w,
        )

    # UNSAT: extract the minimal unsatisfiable core and optimize weights.
    mus = extract_mus(c_list)
    taut_weight = 0.0
    searchable: List[CNFClause] = []
    for clause in c_list:
        if _is_tautology(clause.literals):
            taut_weight += clause.weight
        elif clause.literals:
            searchable.append(clause)
        # empty clauses can never be satisfied and are excluded from search
    best_w, best_assign = _maxsat_search(searchable)
    best_w += taut_weight
    variables = sorted({abs(lit) for c in c_list for lit in c.literals})
    best_assign = _complete_assignment(best_assign, variables)

    violated: List[str] = []
    for clause in c_list:
        if _is_tautology(clause.literals):
            continue
        clause_sat = any(
            (lit > 0 and best_assign.get(abs(lit), False)) or
            (lit < 0 and not best_assign.get(abs(lit), False))
            for lit in clause.literals
        )
        if not clause_sat:
            violated.append(clause.clause_id)

    return SATComplianceResult(
        satisfiable=False,
        assignment=best_assign,
        satisfied_weight=max(0.0, best_w),
        total_weight=total_w,
        violated_clauses=violated,
        unsat_core=mus,
    )


class MaxSATSolver:
    def solve(self, clauses: Sequence[CNFClause]) -> SATComplianceResult:
        return solve_maxsat(clauses)


def _self_test() -> int:
    # 1. Satisfiable formula: (x1 | x2) & (~x1 | x3) & (~x2 | x3)
    sat_clauses = [
        CNFClause("C1", (1, 2), weight=2.0),
        CNFClause("C2", (-1, 3), weight=2.0),
        CNFClause("C3", (-2, 3), weight=2.0),
    ]
    res_sat = solve_maxsat(sat_clauses)
    assert res_sat.satisfiable and res_sat.satisfied_weight == 6.0, res_sat.to_dict()
    assert res_sat.assignment[3] is True

    # 2. Unsatisfiable formula with extra irrelevant clause -> MUS isolates exact conflict
    unsat_clauses = [
        CNFClause("R1_JSON", (1,), weight=5.0, description="Must be JSON"),
        CNFClause("R2_NO_JSON", (-1,), weight=1.0, description="Never use JSON"),
        CNFClause("R3_POLITE", (2,), weight=3.0, description="Be polite"),
    ]
    res_unsat = solve_maxsat(unsat_clauses)
    assert not res_unsat.satisfiable
    assert set(res_unsat.unsat_core) == {"R1_JSON", "R2_NO_JSON"}, res_unsat.to_dict()
    # MaxSAT satisfies R1_JSON (w=5) + R3_POLITE (w=3) = 8.0, violating only R2_NO_JSON (w=1)
    assert res_unsat.satisfied_weight == 8.0
    assert res_unsat.violated_clauses == ["R2_NO_JSON"]

    # ── adversarial / edge-case additions ─────────────────────────────
    # empty clause set is SAT; a single empty clause is UNSAT
    assert dpll_solve([]) == {}
    assert dpll_solve([[]]) is None
    assert dpll_solve([(1, -1)]) == {}          # tautology dropped
    assert dpll_solve([(1,), (-1,)]) is None    # contradictory units
    assert dpll_solve([(1, 2)], {1: False}) == {1: False, 2: True}
    assert dpll_solve([(1,)], {1: False}) is None
    assert extract_mus([CNFClause("S1", (1, 2)),
                        CNFClause("S2", (-1, 2))]) == []

    mus = extract_mus([CNFClause("U1", (1,)), CNFClause("U2", (-1,)),
                       CNFClause("U3", (2,))])
    assert mus == ["U1", "U2"], mus

    # regression: MaxSAT must consider variables beyond the first 16
    wide = [CNFClause("W1", (20,), weight=5.0),
            CNFClause("W2", (-20,), weight=1.0)]
    res_wide = solve_maxsat(wide)
    assert not res_wide.satisfiable
    assert res_wide.satisfied_weight == 5.0, res_wide.to_dict()
    assert res_wide.assignment[20] is True
    assert res_wide.violated_clauses == ["W2"]

    # empty / tautological / malformed formulas
    res_empty = solve_maxsat([])
    assert res_empty.satisfiable and res_empty.total_weight == 0.0
    res_taut = solve_maxsat([CNFClause("T", (3, -3), weight=4.0)])
    assert res_taut.satisfiable and res_taut.satisfied_weight == 4.0
    res_mal = solve_maxsat([{"clause_id": "D", "literals": ["bad"]}])
    assert not res_mal.satisfiable and res_mal.violated_clauses == ["D"]
    res_nan = solve_maxsat([CNFClause("N", (1,), weight=float("nan"))])
    assert res_nan.satisfiable and res_nan.total_weight == 0.0
    assert dpll_solve([(1, "junk")]) is not None

    # determinism: identical inputs yield identical result dicts
    a1 = solve_maxsat(unsat_clauses).to_dict()
    a2 = solve_maxsat(unsat_clauses).to_dict()
    assert a1 == a2, (a1, a2)
    assert all(isinstance(k, int) for k in a1["assignment"])

    print("satisolver._self_test: OK PASS (DPLL unit propagation, MaxSAT optimization, MUS extraction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
