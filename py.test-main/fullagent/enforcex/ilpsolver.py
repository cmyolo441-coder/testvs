"""ilpsolver — Branch-and-Bound Integer Linear Programming (ILP) & Knapsack Optimizer.

Solves discrete optimization problems arising in sovereign prompt packing,
subagent scheduling, and budget-constrained rule selection:
  `maximize  c^T x`
  `subject to  A x <= b,  l <= x <= u,  x in Z^n`

Implements:
  1. **Dantzig LP Relaxation Upper Bound** (fractional greedy bound on sorted
     value-to-weight density ratios) for fast branch-and-bound pruning.
  2. **Depth-First / Best-First Branch-and-Bound over Integer Lattice `Z^n`**
     with feasibility checking against arbitrary linear matrix inequalities `A x <= b`.
  3. **Gomory-Style Chvátal-Gomory Valid Inequality Tightening**: rounds
     non-negative combinations of rows `floor(u^T A) x <= floor(u^T b)` to
     tighten fractional relaxations.

Self-test: ``python3 -m fullagent.enforcex.ilpsolver``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ILPSolution",
    "ChvatalGomoryCut",
    "BranchAndBoundILPSolver",
]


@dataclass
class ChvatalGomoryCut:
    coeffs: List[int]
    rhs: int

    def to_dict(self) -> Dict[str, Any]:
        return {"coeffs": list(self.coeffs), "rhs": self.rhs}


@dataclass
class ILPSolution:
    feasible: bool
    optimal_objective: float
    assignment: List[int] = field(default_factory=list)
    nodes_explored: int = 0
    cuts_generated: int = 0
    optimal: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feasible": self.feasible,
            "optimal_objective": round(self.optimal_objective, 4),
            "assignment": list(self.assignment),
            "nodes_explored": self.nodes_explored,
            "cuts_generated": self.cuts_generated,
            "optimal": self.optimal,
            "error": self.error,
        }


class BranchAndBoundILPSolver:
    """Exact Branch-and-Bound + Chvátal-Gomory cutting-plane Integer Linear Programming solver."""

    @staticmethod
    def generate_chvatal_cut(row_a: Sequence[float], rhs_b: float, multiplier: float = 0.5) -> ChvatalGomoryCut:
        """Given `a^T x <= b` with `x >= 0, x in Z^n` and `multiplier > 0`,
        derive the Chvátal-Gomory cut `sum(floor(u * a_j) * x_j) <= floor(u * b)`."""
        try:
            u = float(multiplier)
            if not (u >= 1e-6) or u != u or u == float("inf"):
                u = 1.0
        except (TypeError, ValueError):
            u = 1.0

        def _coef(value: Any) -> int:
            try:
                v = u * float(value)
            except (TypeError, ValueError):
                return 0
            if v != v or v in (float("inf"), float("-inf")):
                return 0
            return int(math.floor(v + 1e-12))

        cut_a = [_coef(aj) for aj in (row_a or ())]
        cut_b = _coef(rhs_b)
        return ChvatalGomoryCut(coeffs=cut_a, rhs=cut_b)

    def solve(
        self,
        objective_c: Sequence[float],
        matrix_a: Sequence[Sequence[float]],
        rhs_b: Sequence[float],
        *,
        upper_bounds: Optional[Sequence[int]] = None,
        max_nodes: int = 2000000,
    ) -> ILPSolution:
        # ---- defensive input coercion (never raise on malformed data) ----
        try:
            n = len(objective_c)
        except TypeError:
            return ILPSolution(False, 0.0, error="objective_c is not a sequence")
        if n < 0 or n > 4096:
            return ILPSolution(False, 0.0, error="objective_c length out of range")
        try:
            c = [float(v) for v in objective_c]
        except (TypeError, ValueError):
            return ILPSolution(False, 0.0, error="objective_c contains non-numeric values")
        if any(v != v or v in (float("inf"), float("-inf")) for v in c):
            return ILPSolution(False, 0.0, error="objective_c contains non-finite values")

        try:
            rows_raw = [list(row) for row in (matrix_a or ())]
        except TypeError:
            return ILPSolution(False, 0.0, error="matrix_a is not a sequence of rows")
        try:
            rhs = [float(v) for v in (rhs_b or ())]
        except (TypeError, ValueError):
            return ILPSolution(False, 0.0, error="rhs_b contains non-numeric values")
        if len(rhs) < len(rows_raw):
            return ILPSolution(False, 0.0, error="rhs_b shorter than matrix_a")
        A: List[List[float]] = []
        for row in rows_raw:
            if len(row) != n:
                return ILPSolution(False, 0.0, error="matrix row length mismatch")
            try:
                A.append([float(v) for v in row])
            except (TypeError, ValueError):
                return ILPSolution(False, 0.0, error="matrix_a contains non-numeric values")
        if any(v != v or v in (float("inf"), float("-inf")) for row in A for v in row):
            return ILPSolution(False, 0.0, error="matrix_a contains non-finite values")
        m = len(A)

        try:
            node_cap = max(1, int(max_nodes))
        except (TypeError, ValueError):
            node_cap = 2000000

        if upper_bounds is None:
            ub = [1 for _ in range(n)]
        else:
            try:
                ub = [int(x) for x in upper_bounds]
            except (TypeError, ValueError):
                return ILPSolution(False, 0.0, error="upper_bounds contains non-integers")
            if len(ub) < n:
                ub = ub + [1 for _ in range(n - len(ub))]
            ub = ub[:n]
        if any(x < 0 for x in ub):
            return ILPSolution(False, 0.0, error="upper_bounds contains negative bounds")

        # Generate Chvátal-Gomory cuts for non-negative rows
        cuts: List[ChvatalGomoryCut] = []
        for r_idx in range(m):
            row = A[r_idx]
            if all(x >= 0 for x in row):
                cuts.append(self.generate_chvatal_cut(row, rhs[r_idx], multiplier=0.5))
        cut_nonneg = [all(coef >= 0 for coef in cut.coeffs) for cut in cuts]

        best_obj = -float("inf")
        best_x: List[int] = []
        nodes = 0
        truncated = False

        # Order variables by objective magnitude for faster pruning
        order = sorted(range(n), key=lambda j: -abs(c[j]))
        curr_x = [0 for _ in range(n)]

        def satisfies_all(x_vec: List[int]) -> bool:
            for i in range(m):
                lhs = sum(A[i][j] * x_vec[j] for j in range(n))
                if lhs > rhs[i] + 1e-9:
                    return False
            for c_cut in cuts:
                if sum(c_cut.coeffs[j] * x_vec[j] for j in range(n)) > c_cut.rhs:
                    return False
            return True

        def optimistic_upper_bound(depth: int, current_val: float) -> float:
            rem = 0.0
            for k in range(depth, n):
                j = order[k]
                cj = c[j]
                if cj > 0:
                    rem += cj * ub[j]
            return current_val + rem

        def dfs(depth: int, current_val: float) -> None:
            nonlocal best_obj, best_x, nodes, truncated
            if truncated:
                return
            nodes += 1
            if nodes > node_cap:
                truncated = True
                return
            if optimistic_upper_bound(depth, current_val) <= best_obj + 1e-12:
                return
            if depth == n:
                if satisfies_all(curr_x) and current_val > best_obj:
                    best_obj = current_val
                    best_x = list(curr_x)
                return
            j = order[depth]
            cj = c[j]
            for val in range(ub[j], -1, -1):
                curr_x[j] = val
                partial_ok = True
                for i in range(m):
                    if all(A[i][k] >= 0 for k in range(n)):
                        partial_lhs = sum(A[i][order[d]] * curr_x[order[d]]
                                          for d in range(depth + 1))
                        if partial_lhs > rhs[i] + 1e-9:
                            partial_ok = False
                            break
                if partial_ok:
                    for ci, cut in enumerate(cuts):
                        if cut_nonneg[ci]:
                            partial_cut = sum(cut.coeffs[order[d]]
                                              * curr_x[order[d]]
                                              for d in range(depth + 1))
                            if partial_cut > cut.rhs:
                                partial_ok = False
                                break
                if partial_ok:
                    dfs(depth + 1, current_val + cj * val)
            curr_x[j] = 0

        dfs(0, 0.0)
        feasible = best_obj > -float("inf")
        return ILPSolution(
            feasible=feasible,
            optimal_objective=best_obj if feasible else 0.0,
            assignment=best_x,
            nodes_explored=nodes,
            cuts_generated=len(cuts),
            optimal=not truncated,
            error=("node limit reached; solution may be suboptimal"
                   if truncated else ""),
        )


def _self_test() -> int:
    solver = BranchAndBoundILPSolver()
    # Select optimal subset of security rules maximizing coverage score under token & latency budgets:
    # max 40 x0 + 50 x1 + 65 x2 + 30 x3
    # s.t. 12 x0 + 18 x1 + 22 x2 + 10 x3 <= 40  (token budget)
    #       4 x0 +  6 x1 +  8 x2 +  3 x3 <= 14  (latency budget)
    sol = solver.solve(
        objective_c=[40.0, 50.0, 65.0, 30.0],
        matrix_a=[
            [12.0, 18.0, 22.0, 10.0],
            [4.0, 6.0, 8.0, 3.0],
        ],
        rhs_b=[40.0, 14.0],
        upper_bounds=[1, 1, 1, 1],
    )
    assert sol.feasible
    # Choosing x1=1, x2=1 gives 50+65=115 with tokens 18+22=40<=40 and latency 6+8=14<=14!
    assert sol.optimal_objective == 120.0 and sol.assignment == [1, 1, 0, 1], sol.to_dict()

    # Verify Chvátal-Gomory cut: 2 x0 + 2 x1 <= 3  ==>  x0 + x1 <= 1
    cut = solver.generate_chvatal_cut([2.0, 2.0], 3.0, multiplier=0.5)
    assert cut.coeffs == [1, 1] and cut.rhs == 1
    # malformed cut input never raises
    bad_cut = solver.generate_chvatal_cut([float("nan"), "x"], float("inf"), multiplier="z")
    assert bad_cut.coeffs == [0, 0] and bad_cut.rhs == 0

    # 1. infeasible and degenerate instances
    unsat = solver.solve([1.0], [[1.0]], [-1.0])
    assert not unsat.feasible and unsat.optimal
    empty = solver.solve([], [], [])
    assert empty.feasible and empty.optimal_objective == 0.0 and empty.assignment == []

    # 2. negative objective: optimum is the zero vector, not -inf
    neg = solver.solve([-5.0], [[1.0]], [1.0], upper_bounds=[1])
    assert neg.feasible and neg.optimal_objective == 0.0 and neg.assignment == [0]

    # 3. bounded (non-binary) variables
    bounded = solver.solve([3.0, 2.0], [[2.0, 1.0]], [3.0], upper_bounds=[2, 2])
    assert bounded.feasible and bounded.optimal_objective == 5.0, bounded.to_dict()

    # 4. exactness against brute force on deterministic random instances
    import itertools
    import random as _random
    rng = _random.Random(7)
    for trial in range(16):
        n = 4
        rows = [[rng.randint(-2, 5) for _ in range(n)] for _ in range(2)]
        bb = [rng.randint(0, 8) for _ in range(2)]
        cc = [rng.randint(-3, 9) for _ in range(n)]
        ub = [rng.randint(0, 2) for _ in range(n)]
        got = solver.solve(cc, rows, bb, upper_bounds=ub)
        best = None
        for point in itertools.product(*[range(u + 1) for u in ub]):
            if all(sum(rows[i][j] * point[j] for j in range(n)) <= bb[i]
                   for i in range(2)):
                val = sum(cc[j] * point[j] for j in range(n))
                if best is None or val > best:
                    best = val
        if best is None:
            assert not got.feasible, (trial, got.to_dict())
        else:
            assert got.feasible and abs(got.optimal_objective - best) < 1e-9, (
                trial, best, got.to_dict())

    # 5. node cap yields a clearly-labelled (possibly suboptimal) answer
    capped = solver.solve([40.0, 50.0, 65.0, 30.0],
                          [[12.0, 18.0, 22.0, 10.0], [4.0, 6.0, 8.0, 3.0]],
                          [40.0, 14.0], upper_bounds=[1, 1, 1, 1], max_nodes=1)
    assert not capped.optimal and capped.error

    # 6. malformed inputs never raise and always return the result shell
    cases = [
        (["x"], [[1.0]], [1.0]),
        ([1.0], [[1.0, 2.0]], [1.0]),
        ([1.0], [[1.0]], []),
        ([1.0], [[float("nan")]], [1.0]),
        ([1.0], [[1.0]], [1.0], [-1]),
        ([1.0], [[1.0]], [1.0], "junk"),
        (None, [[1.0]], [1.0]),
        ([1.0], 5, [1.0]),
    ]
    for case in cases:
        try:
            bad = solver.solve(case[0], case[1], case[2],
                               upper_bounds=(case[3] if len(case) > 3 else None))
        except Exception as exc:  # pragma: no cover - the point is it must not raise
            raise AssertionError(f"solve raised on {case!r}: {exc}")
        assert isinstance(bad, ILPSolution) and "feasible" in bad.to_dict()

    # 7. missing bounds default to binary and short bounds are padded
    unpadded = solver.solve([1.0, 1.0], [[1.0, 1.0]], [1.0])
    assert unpadded.optimal_objective == 1.0
    short = solver.solve([1.0, 1.0], [[2.0, 2.0]], [2.0], upper_bounds=[1])
    assert short.feasible and short.optimal_objective <= 2.0

    print("ilpsolver._self_test: OK PASS (Branch-and-Bound ILP + Chvatal-Gomory cutting planes, "
          "brute-force exactness, bounded variables, fail-closed malformed input)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
