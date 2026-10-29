"""symbolicexec — AST Symbolic Execution & Path-Condition Witness Generator.

Symbolically explores branching paths (`if`/`else`, `assert`) in Python functions,
maintaining interval bounds on symbolic parameters `[lo, hi]` along each path
to:
  * Prune infeasible paths where `lo > hi`
  * Enumerate all feasible execution paths
  * Generate concrete witness inputs that trigger `assert` failures or `raise`
    statements.

Self-test: ``python3 -m fullagent.enforcex.symbolicexec``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "SymbolicPath",
    "SymbolicExplorationReport",
    "SymbolicExecutionEngine",
    "explore_function",
]


@dataclass
class SymbolicPath:
    path_id: int
    constraints: List[str] = field(default_factory=list)
    bounds: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    outcome: str = "return"   # return | assert_fail | raise_error
    witness: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "constraints": list(self.constraints),
            "outcome": self.outcome,
            "witness": dict(self.witness),
        }


@dataclass
class SymbolicExplorationReport:
    paths_explored: int = 0
    bug_paths: List[SymbolicPath] = field(default_factory=list)
    all_paths: List[SymbolicPath] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paths_explored": self.paths_explored,
            "bug_paths_count": len(self.bug_paths),
            "bug_paths": [p.to_dict() for p in self.bug_paths],
            "all_paths": [p.to_dict() for p in self.all_paths],
        }


def _refine_bounds(
    bounds: Dict[str, Tuple[int, int]],
    test: ast.AST,
    negate: bool = False,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """Refine `[lo, hi]` integer bounds given a comparison `var <op> const`."""
    b = dict(bounds)
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return b
    left, op, right = test.left, test.ops[0], test.comparators[0]
    if not (isinstance(left, ast.Name) and isinstance(right, ast.Constant) and isinstance(right.value, int)):
        return b
    var = left.id
    val = int(right.value)
    lo, hi = b.get(var, (-10_000, 10_000))

    if isinstance(op, ast.Gt):
        lo, hi = (lo, min(hi, val)) if negate else (max(lo, val + 1), hi)
    elif isinstance(op, ast.GtE):
        lo, hi = (lo, min(hi, val - 1)) if negate else (max(lo, val), hi)
    elif isinstance(op, ast.Lt):
        lo, hi = (max(lo, val), hi) if negate else (lo, min(hi, val - 1))
    elif isinstance(op, ast.LtE):
        lo, hi = (max(lo, val + 1), hi) if negate else (lo, min(hi, val))
    elif isinstance(op, ast.Eq):
        if not negate:
            lo, hi = max(lo, val), min(hi, val)
        elif lo == val and hi == val:
            return None
        elif lo == val:
            lo += 1
    if lo > hi:
        return None
    b[var] = (lo, hi)
    return b


def _witness_from_bounds(bounds: Dict[str, Tuple[int, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for k, (lo, hi) in bounds.items():
        if lo <= 0 <= hi:
            out[k] = 0
        elif lo > -10_000:
            out[k] = lo
        elif hi < 10_000:
            out[k] = hi
        else:
            out[k] = 1
    return out


def explore_function(code: str) -> SymbolicExplorationReport:
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        return SymbolicExplorationReport()

    fn_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    if fn_node is None:
        return SymbolicExplorationReport()

    init_bounds = {arg.arg: (-1000, 1000) for arg in fn_node.args.args}
    paths: List[SymbolicPath] = []

    def _walk(stmts: List[ast.stmt], bounds: Dict[str, Tuple[int, int]], conds: List[str]) -> None:
        if not stmts:
            paths.append(
                SymbolicPath(
                    path_id=len(paths) + 1,
                    constraints=list(conds),
                    bounds=dict(bounds),
                    outcome="return",
                    witness=_witness_from_bounds(bounds),
                )
            )
            return
        head, rest = stmts[0], stmts[1:]
        if isinstance(head, ast.If):
            c_str = ast.unparse(head.test) if hasattr(ast, "unparse") else "cond"
            b_true = _refine_bounds(bounds, head.test, negate=False)
            if b_true is not None:
                _walk(head.body + rest, b_true, conds + [c_str])
            b_false = _refine_bounds(bounds, head.test, negate=True)
            if b_false is not None:
                _walk(head.orelse + rest, b_false, conds + [f"not ({c_str})"])
        elif isinstance(head, ast.Assert):
            c_str = ast.unparse(head.test) if hasattr(ast, "unparse") else "assert"
            b_fail = _refine_bounds(bounds, head.test, negate=True)
            if b_fail is not None:
                paths.append(
                    SymbolicPath(
                        path_id=len(paths) + 1,
                        constraints=conds + [f"ASSERT_FAIL: not ({c_str})"],
                        bounds=b_fail,
                        outcome="assert_fail",
                        witness=_witness_from_bounds(b_fail),
                    )
                )
            b_pass = _refine_bounds(bounds, head.test, negate=False)
            if b_pass is not None:
                _walk(rest, b_pass, conds + [c_str])
        elif isinstance(head, ast.Raise):
            paths.append(
                SymbolicPath(
                    path_id=len(paths) + 1,
                    constraints=list(conds),
                    bounds=dict(bounds),
                    outcome="raise_error",
                    witness=_witness_from_bounds(bounds),
                )
            )
        elif isinstance(head, ast.Return):
            paths.append(
                SymbolicPath(
                    path_id=len(paths) + 1,
                    constraints=list(conds),
                    bounds=dict(bounds),
                    outcome="return",
                    witness=_witness_from_bounds(bounds),
                )
            )
        else:
            _walk(rest, bounds, conds)

    _walk(fn_node.body, init_bounds, [])
    bugs = [p for p in paths if p.outcome in ("assert_fail", "raise_error")]
    return SymbolicExplorationReport(paths_explored=len(paths), bug_paths=bugs, all_paths=paths)


class SymbolicExecutionEngine:
    def explore(self, code: str) -> SymbolicExplorationReport:
        return explore_function(code)


def _self_test() -> int:
    src = """
def withdraw(amount, balance):
    if amount > 100:
        if balance < 50:
            assert balance >= 50
        return balance - amount
    return balance
"""
    rep = explore_function(src)
    assert rep.paths_explored >= 3
    assert len(rep.bug_paths) == 1, rep.to_dict()
    w = rep.bug_paths[0].witness
    assert w["amount"] > 100 and w["balance"] < 50, w

    print("symbolicexec._self_test: OK PASS (branch forking, infeasible pruning, counterexample witness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
