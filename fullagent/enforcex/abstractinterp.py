"""abstractinterp — Interval `[lo, hi]` & Taint Abstract Interpretation Engine.

Executes static abstract interpretation over Python AST snippets using the
product domain `Interval x Taint`:
  * `Interval(lo, hi)`: exact interval arithmetic (`+`, `-`, `*`, `//`) with
    Galois widening (`widen`) for loops and division-by-zero / negative-index
    detection (`0 in rhs_interval`).
  * `tainted: bool`: tracks whether an abstract value depends on untrusted input.

Self-test: ``python3 -m fullagent.enforcex.abstractinterp``
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "Interval",
    "AbstractValue",
    "AbstractFixpointReport",
    "AbstractInterpreter",
    "analyze_ast_intervals",
]

INF = float("inf")
NEG_INF = float("-inf")

#: Hard cap so untrusted multi-megabyte submissions cannot exhaust the parser.
MAX_CODE_CHARS = 262_144

_TAINT_PREFIXES = ("user_", "untrusted_", "raw_", "extern_", "input_")

_TAINT_SOURCE_CALLS = frozenset({
    "input", "raw_input", "fileinput.input",
    "os.getenv", "os.environ.get", "sys.stdin.read",
    "request.form.get", "request.args.get", "request.json",
})

_TAINT_SINK_CALLS = frozenset({
    "eval", "exec", "compile", "__import__",
    "os.system", "os.popen", "os.execv", "os.execve", "os.spawnl",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_output", "subprocess.check_call",
    "pickle.loads", "pickle.load", "marshal.loads", "shelve.open",
    "yaml.load", "yaml.unsafe_load", "open",
})


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        cur = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


@dataclass(frozen=True)
class Interval:
    lo: float = NEG_INF
    hi: float = INF

    def is_bottom(self) -> bool:
        return self.lo > self.hi

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def join(self, other: "Interval") -> "Interval":
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def widen(self, nxt: "Interval") -> "Interval":
        lo = self.lo if nxt.lo >= self.lo else NEG_INF
        hi = self.hi if nxt.hi <= self.hi else INF
        return Interval(lo, hi)

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def sub(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def mul(self, other: "Interval") -> "Interval":
        prods = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        valid = [p for p in prods if not math.isnan(p)]
        return Interval(min(valid), max(valid)) if valid else Interval()

    def div(self, other: "Interval") -> "Interval":
        """Sound real-division image; top when the divisor may be zero."""
        if other.is_bottom() or other.contains(0.0):
            return Interval()
        quotients = (
            self.lo / other.lo, self.lo / other.hi,
            self.hi / other.lo, self.hi / other.hi,
        )
        valid = [q for q in quotients if not math.isnan(q)]
        return Interval(min(valid), max(valid)) if valid else Interval()

    def floor_div(self, other: "Interval") -> "Interval":
        """Sound floor-division image for finite operands; top otherwise."""
        if other.is_bottom() or other.contains(0.0):
            return Interval()
        if not all(math.isfinite(v) for v in (self.lo, self.hi, other.lo, other.hi)):
            return Interval()
        quotients = (
            self.lo // other.lo, self.lo // other.hi,
            self.hi // other.lo, self.hi // other.hi,
        )
        # Python floor division is monotone in the numerator and antitone in
        # the denominator, so endpoint quotients bracket the image.
        return Interval(float(min(quotients)), float(max(quotients)))

    def to_dict(self) -> Dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}


@dataclass
class AbstractValue:
    interval: Interval = field(default_factory=Interval)
    tainted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"interval": self.interval.to_dict(), "tainted": self.tainted}


@dataclass
class AbstractFixpointReport:
    safe: bool = True
    env: Dict[str, AbstractValue] = field(default_factory=dict)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "safe": self.safe,
            "env": {k: v.to_dict() for k, v in self.env.items()},
            "warnings_count": len(self.warnings),
            "warnings": list(self.warnings),
        }


class AbstractInterpreter:
    """Evaluates Python AST statements over the Interval x Taint abstract domain."""

    def __init__(self, initial_env: Optional[Dict[str, AbstractValue]] = None) -> None:
        self.env: Dict[str, AbstractValue] = dict(initial_env) if initial_env else {}
        self.warnings: List[Dict[str, Any]] = []
        self._depth = 0

    # -- expression evaluation ------------------------------------------------

    def _warn(self, rule_id: str, node: ast.AST, severity: str, message: str) -> None:
        self.warnings.append({
            "rule_id": rule_id,
            "lineno": getattr(node, "lineno", 1),
            "severity": severity,
            "message": message,
        })

    def _eval_expr(self, node: ast.AST) -> AbstractValue:
        if node is None:
            return AbstractValue()
        if isinstance(node, ast.Constant):
            val = node.value
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                try:
                    v = float(val)
                except (OverflowError, ValueError):
                    return AbstractValue()
                if math.isfinite(v):
                    return AbstractValue(Interval(v, v), False)
                return AbstractValue(Interval(v, v) if not math.isnan(v) else Interval(), False)
            if isinstance(val, bool):
                return AbstractValue(Interval(0.0, 1.0), False)
            return AbstractValue(Interval(), False)
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            tainted = node.id.startswith(_TAINT_PREFIXES)
            return AbstractValue(Interval(NEG_INF, INF), tainted)
        if isinstance(node, ast.UnaryOp):
            sub = self._eval_expr(node.operand)
            if isinstance(node.op, ast.USub):
                return AbstractValue(Interval(-sub.interval.hi, -sub.interval.lo), sub.tainted)
            if isinstance(node.op, ast.UAdd):
                return sub
            if isinstance(node.op, ast.Not):
                return AbstractValue(Interval(0.0, 1.0), sub.tainted)
            return AbstractValue(Interval(), sub.tainted)
        if isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            tainted = left.tainted or right.tainted
            if isinstance(node.op, ast.Add):
                return AbstractValue(left.interval.add(right.interval), tainted)
            if isinstance(node.op, ast.Sub):
                return AbstractValue(left.interval.sub(right.interval), tainted)
            if isinstance(node.op, ast.Mult):
                return AbstractValue(left.interval.mul(right.interval), tainted)
            if isinstance(node.op, ast.Div):
                if right.interval.contains(0.0):
                    self._warn(
                        "ABS-DIV-ZERO", node, "high",
                        f"Possible division by zero: divisor interval "
                        f"[{right.interval.lo}, {right.interval.hi}] contains 0",
                    )
                    return AbstractValue(Interval(), tainted)
                return AbstractValue(left.interval.div(right.interval), tainted)
            if isinstance(node.op, ast.FloorDiv):
                if right.interval.contains(0.0):
                    self._warn(
                        "ABS-DIV-ZERO", node, "high",
                        f"Possible floor division by zero: divisor interval "
                        f"[{right.interval.lo}, {right.interval.hi}] contains 0",
                    )
                    return AbstractValue(Interval(), tainted)
                return AbstractValue(left.interval.floor_div(right.interval), tainted)
            if isinstance(node.op, ast.Mod):
                if right.interval.contains(0.0):
                    self._warn(
                        "ABS-DIV-ZERO", node, "high",
                        f"Possible modulo by zero: divisor interval "
                        f"[{right.interval.lo}, {right.interval.hi}] contains 0",
                    )
                return AbstractValue(Interval(), tainted)
            if isinstance(node.op, ast.Pow):
                return AbstractValue(Interval(), tainted)
            return AbstractValue(Interval(), tainted)
        if isinstance(node, ast.Compare):
            left = self._eval_expr(node.left)
            tainted = left.tainted
            for comp in node.comparators:
                tainted = tainted or self._eval_expr(comp).tainted
            return AbstractValue(Interval(0.0, 1.0), tainted)
        if isinstance(node, ast.BoolOp):
            vals = [self._eval_expr(v) for v in node.values]
            if not vals:
                return AbstractValue(Interval(0.0, 1.0), False)
            merged = vals[0].interval
            tainted = any(v.tainted for v in vals)
            for v in vals[1:]:
                merged = merged.join(v.interval)
            return AbstractValue(merged, tainted)
        if isinstance(node, ast.IfExp):
            body = self._eval_expr(node.body)
            orelse = self._eval_expr(node.orelse)
            self._eval_expr(node.test)
            return AbstractValue(body.interval.join(orelse.interval),
                                 body.tainted or orelse.tainted)
        if isinstance(node, ast.Attribute):
            base = self._eval_expr(node.value)
            return AbstractValue(Interval(), base.tainted)
        if isinstance(node, ast.Subscript):
            base = self._eval_expr(node.value)
            idx = self._eval_expr(node.slice)
            return AbstractValue(Interval(), base.tainted or idx.tainted)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            tainted = False
            interval = Interval(NEG_INF, INF)
            for elt in node.elts:
                av = self._eval_expr(elt)
                tainted = tainted or av.tainted
            return AbstractValue(interval, tainted)
        if isinstance(node, ast.Dict):
            tainted = False
            for key, val in zip(node.keys, node.values):
                if key is not None:
                    tainted = tainted or self._eval_expr(key).tainted
                tainted = tainted or self._eval_expr(val).tainted
            return AbstractValue(Interval(), tainted)
        if isinstance(node, ast.JoinedStr):
            tainted = False
            for value in node.values:
                if isinstance(value, ast.FormattedValue):
                    tainted = tainted or self._eval_expr(value.value).tainted
            return AbstractValue(Interval(), tainted)
        if isinstance(node, ast.NamedExpr):
            val = self._eval_expr(node.value)
            if isinstance(node.target, ast.Name):
                self.env[node.target.id] = val
            return val
        if isinstance(node, ast.Call):
            func = self._eval_expr(node.func)
            tainted = func.tainted
            for arg in node.args:
                tainted = tainted or self._eval_expr(arg).tainted
            for kw in node.keywords:
                if kw.arg != "usedforsecurity":
                    tainted = tainted or self._eval_expr(kw.value).tainted
            cname = _call_name(node)
            if cname in _TAINT_SOURCE_CALLS:
                tainted = True
            if tainted and (cname in _TAINT_SINK_CALLS or func.tainted):
                self._warn(
                    "ABS-TAINT-SINK", node, "critical",
                    f"Tainted data reaches sensitive sink '{cname or '<call>'}()'",
                )
            return AbstractValue(Interval(), tainted)
        if isinstance(node, ast.Lambda):
            return AbstractValue(Interval(), False)
        return AbstractValue(Interval(), False)

    # -- condition refinement -------------------------------------------------

    def _refine(self, test: Optional[ast.AST], truth: bool) -> Dict[str, AbstractValue]:
        """Return a copy of the environment refined by `test == truth`."""
        env = dict(self.env)
        if test is None:
            return env
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            return self._refine_with_env(test.operand, not truth, env)
        if isinstance(test, ast.BoolOp) and isinstance(test.op, (ast.And, ast.Or)):
            conjunctive = isinstance(test.op, ast.And)
            if truth == conjunctive:
                for value in test.values:
                    env = self._refine_with_env(value, truth, env)
            return env
        return self._refine_with_env(test, truth, env)

    def _refine_with_env(self, test: Optional[ast.AST], truth: bool,
                         env: Dict[str, AbstractValue]) -> Dict[str, AbstractValue]:
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return env
        op = test.ops[0]
        left, right = test.left, test.comparators[0]
        name: Optional[str] = None
        const: Optional[float] = None
        flipped = False
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant) and \
                isinstance(right.value, (int, float)) and not isinstance(right.value, bool):
            name, const = left.id, float(right.value)
        elif isinstance(right, ast.Name) and isinstance(left, ast.Constant) and \
                isinstance(left.value, (int, float)) and not isinstance(left.value, bool):
            name, const = right.id, float(left.value)
            flipped = True
        if name is None or const is None or not math.isfinite(const):
            return env
        if flipped:
            op_map = {
                ast.Lt: ast.Gt, ast.LtE: ast.GtE,
                ast.Gt: ast.Lt, ast.GtE: ast.LtE,
                ast.Eq: ast.Eq, ast.NotEq: ast.NotEq,
            }
            op = op_map.get(type(op), type(op))
        cur = env.get(name, AbstractValue())
        lo, hi = cur.interval.lo, cur.interval.hi
        if isinstance(op, ast.Lt):
            if truth:
                hi = min(hi, math.nextafter(const, NEG_INF))
            else:
                lo = max(lo, const)
        elif isinstance(op, ast.LtE):
            if truth:
                hi = min(hi, const)
            else:
                lo = max(lo, math.nextafter(const, INF))
        elif isinstance(op, ast.Gt):
            if truth:
                lo = max(lo, math.nextafter(const, INF))
            else:
                hi = min(hi, const)
        elif isinstance(op, ast.GtE):
            if truth:
                lo = max(lo, const)
            else:
                hi = min(hi, math.nextafter(const, NEG_INF))
        elif isinstance(op, ast.Eq):
            if truth:
                lo = hi = const
            else:
                return env
        else:
            return env
        new_interval = Interval(lo, hi)
        if new_interval.is_bottom():
            return env
        env = dict(env)
        env[name] = AbstractValue(new_interval, cur.tainted)
        return env

    # -- statements -----------------------------------------------------------

    @staticmethod
    def _merge_envs(base: Dict[str, AbstractValue],
                    a: Dict[str, AbstractValue],
                    b: Dict[str, AbstractValue]) -> Dict[str, AbstractValue]:
        merged: Dict[str, AbstractValue] = {}
        for key in set(base) | set(a) | set(b):
            default = base.get(key, AbstractValue())
            va = a.get(key, default)
            vb = b.get(key, default)
            merged[key] = AbstractValue(
                va.interval.join(vb.interval),
                va.tainted or vb.tainted,
            )
        return merged

    def _exec_block(self, body: List[ast.stmt], env: Optional[Dict[str, AbstractValue]] = None) -> Dict[str, AbstractValue]:
        saved = self.env
        if env is not None:
            self.env = dict(env)
        try:
            for stmt in body:
                self._exec_stmt(stmt)
            return dict(self.env)
        finally:
            self.env = saved

    def _exec_loop_body(self, body: List[ast.stmt], entry: Dict[str, AbstractValue],
                        base: Dict[str, AbstractValue], rounds: int = 4) -> Dict[str, AbstractValue]:
        env = dict(entry)
        for _ in range(rounds):
            after = self._exec_block(body, env)
            # Join with the entry environment, then widen against the stable
            # pre-loop values so the iteration converges after finitely many
            # rounds even for unbounded counters.
            merged: Dict[str, AbstractValue] = {}
            for key in set(env) | set(after):
                av = env.get(key, base.get(key, AbstractValue()))
                bv = after.get(key, base.get(key, AbstractValue()))
                pre = base.get(key, AbstractValue())
                joined = AbstractValue(av.interval.join(bv.interval), av.tainted or bv.tainted)
                if key in base:
                    joined = AbstractValue(pre.interval.widen(joined.interval), joined.tainted)
                merged[key] = joined
            if merged == env:
                return merged
            env = merged
        return env

    def _exec_stmt(self, stmt: ast.AST) -> None:
        if stmt is None or self._depth > 60:
            return
        self._depth += 1
        try:
            self._exec_stmt_inner(stmt)
        finally:
            self._depth -= 1

    def _exec_stmt_inner(self, stmt: ast.AST) -> None:
        if isinstance(stmt, ast.Assign):
            val = self._eval_expr(stmt.value)
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    self.env[t.id] = val
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            self.env[elt.id] = AbstractValue(Interval(), val.tainted)
                elif isinstance(t, (ast.Subscript, ast.Attribute)):
                    self._eval_expr(t)
        elif isinstance(stmt, ast.AnnAssign):
            val = self._eval_expr(stmt.value) if stmt.value is not None else AbstractValue()
            if isinstance(stmt.target, ast.Name):
                self.env[stmt.target.id] = val
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            fake_bin = ast.BinOp(left=ast.Name(id=stmt.target.id, ctx=ast.Load()),
                                 op=stmt.op, right=stmt.value)
            ast.copy_location(fake_bin, stmt)
            self.env[stmt.target.id] = self._eval_expr(fake_bin)
        elif isinstance(stmt, ast.Expr):
            self._eval_expr(stmt.value)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._eval_expr(stmt.value)
        elif isinstance(stmt, ast.Assert):
            self._eval_expr(stmt.test)
        elif isinstance(stmt, ast.Raise):
            if stmt.exc is not None:
                self._eval_expr(stmt.exc)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            saved = self.env
            local = dict(self.env)
            args = stmt.args
            all_args = list(getattr(args, "posonlyargs", [])) + list(args.args) + \
                list(args.kwonlyargs)
            if args.vararg is not None:
                all_args.append(args.vararg)
            if args.kwarg is not None:
                all_args.append(args.kwarg)
            for arg in all_args:
                local[arg.arg] = AbstractValue(Interval(),
                                               arg.arg.startswith(_TAINT_PREFIXES))
            self.env = local
            try:
                for s in stmt.body:
                    self._exec_stmt(s)
            finally:
                self.env = saved
        elif isinstance(stmt, ast.ClassDef):
            self._exec_block(stmt.body)
        elif isinstance(stmt, ast.If):
            self._eval_expr(stmt.test)
            base = dict(self.env)
            then_env = self._exec_block(stmt.body, self._refine_with_env(stmt.test, True, base))
            else_env = self._exec_block(stmt.orelse, self._refine_with_env(stmt.test, False, base))
            self.env = self._merge_envs(base, then_env, else_env)
        elif isinstance(stmt, ast.While):
            self._eval_expr(stmt.test)
            base = dict(self.env)
            entry = self._refine_with_env(stmt.test, True, base)
            loop_env = self._exec_loop_body(stmt.body, entry, base)
            exit_env = self._refine_with_env(stmt.test, False, loop_env)
            if stmt.orelse:
                orelse_env = self._exec_block(stmt.orelse, exit_env)
                exit_env = self._merge_envs(exit_env, exit_env, orelse_env)
            self.env = exit_env
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            iter_av = self._eval_expr(stmt.iter)
            body = list(stmt.body)
            base = dict(self.env)
            entry = dict(base)
            if isinstance(stmt.target, ast.Name):
                entry[stmt.target.id] = AbstractValue(Interval(), iter_av.tainted)
            loop_env = self._exec_loop_body(body, entry, base, rounds=3)
            if stmt.orelse:
                orelse_env = self._exec_block(stmt.orelse, loop_env)
                loop_env = self._merge_envs(base, loop_env, orelse_env)
            self.env = self._merge_envs(base, loop_env, loop_env)
        elif isinstance(stmt, ast.Try):
            base = dict(self.env)
            body_env = self._exec_block(stmt.body, base)
            if stmt.orelse:
                body_env = self._exec_block(stmt.orelse, body_env)
            merged = body_env
            for handler in stmt.handlers:
                handler_env = self._exec_block(handler.body, base)
                merged = self._merge_envs(base, merged, handler_env)
            if stmt.finalbody:
                merged = self._exec_block(stmt.finalbody, merged)
            self.env = merged
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    self._eval_expr(item.context_expr)
                else:
                    self._eval_expr(item.context_expr)
            self._exec_block(stmt.body)

    # -- driver ---------------------------------------------------------------

    def run(self, code: str) -> AbstractFixpointReport:
        if isinstance(code, (bytes, bytearray)):
            try:
                code = bytes(code).decode("utf-8", "replace")
            except Exception:
                code = ""
        if not isinstance(code, str):
            return AbstractFixpointReport(
                safe=False,
                warnings=[{"rule_id": "ABS-INPUT", "lineno": 0, "severity": "medium",
                           "message": "Input is not textual; analysis refused"}],
            )
        if len(code) > MAX_CODE_CHARS:
            return AbstractFixpointReport(
                safe=False,
                warnings=[{"rule_id": "ABS-SIZE-LIMIT", "lineno": 0, "severity": "high",
                           "message": f"Source exceeds {MAX_CODE_CHARS} characters; analysis refused"}],
            )
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return AbstractFixpointReport(safe=False, warnings=[{
                "rule_id": "ABS-SYNTAX", "lineno": getattr(exc, "lineno", 0),
                "severity": "high", "message": str(exc.msg if hasattr(exc, "msg") else exc),
            }])
        except (ValueError, TypeError, MemoryError, RecursionError) as exc:
            return AbstractFixpointReport(safe=False, warnings=[{
                "rule_id": "ABS-PARSE", "lineno": 0, "severity": "high",
                "message": f"Unparseable source ({type(exc).__name__})",
            }])

        for stmt in tree.body:
            self._exec_stmt(stmt)

        # Deduplicate warnings deterministically (loop bodies may be analyzed
        # several times during fixpoint iteration).
        seen: set = set()
        deduped: List[Dict[str, Any]] = []
        for w in self.warnings:
            key = (w.get("rule_id"), w.get("lineno"), w.get("message"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(w)
        self.warnings = deduped
        return AbstractFixpointReport(
            safe=(len(self.warnings) == 0),
            env=dict(self.env),
            warnings=list(self.warnings),
        )


def analyze_ast_intervals(code: str) -> AbstractFixpointReport:
    return AbstractInterpreter().run(code)


def _self_test() -> int:
    code = """
a = 10
b = 20
c = a + b
d = a - 10
risky = c / d
"""
    rep = analyze_ast_intervals(code)
    assert not rep.safe
    assert rep.env["c"].interval == Interval(30.0, 30.0)
    assert rep.env["d"].interval == Interval(0.0, 0.0)
    assert any(w["rule_id"] == "ABS-DIV-ZERO" for w in rep.warnings), rep.to_dict()

    # Refined division: real division and floor division produce exact bounds.
    rep_div = analyze_ast_intervals("a = 100\nb = 4\nc = a / b\ne = 17 // 5\n")
    assert rep_div.safe, rep_div.to_dict()
    assert rep_div.env["c"].interval == Interval(25.0, 25.0)
    assert rep_div.env["e"].interval == Interval(3.0, 3.0)

    # Branch refinement proves the divisor positive in the `x > 10` branch.
    rep_ref = analyze_ast_intervals(
        "x = unknown_value\nif x > 10:\n    y = 1 / (x - 10)\n"
    )
    assert rep_ref.safe, rep_ref.to_dict()

    # ... and correctly keeps zero in the `else` of `x > 0`.
    rep_else = analyze_ast_intervals(
        "x = unknown_value\nif x > 0:\n    pass\nelse:\n    y = 1 / x\n"
    )
    assert not rep_else.safe
    assert any(w["rule_id"] == "ABS-DIV-ZERO" for w in rep_else.warnings)

    # While-loop fixpoint converges, deduplicates warnings, and the exit
    # refinement proves the final divisor non-zero.
    rep_while = analyze_ast_intervals(
        "n = 0\nwhile n < 10:\n    n = n + 1\nsafe = 10 / n\n"
    )
    assert rep_while.safe, rep_while.to_dict()
    rep_loop_div = analyze_ast_intervals(
        "total = 0\nwhile total < 3:\n    total = total + 1\n    z = 1 / 0\n"
    )
    divw = [w for w in rep_loop_div.warnings if w["rule_id"] == "ABS-DIV-ZERO"]
    assert len(divw) == 1, divw

    # Taint tracking: untrusted input reaching eval() is a critical sink.
    rep_taint = analyze_ast_intervals("payload = input()\neval(payload)\n")
    assert not rep_taint.safe
    assert any(w["rule_id"] == "ABS-TAINT-SINK" for w in rep_taint.warnings), rep_taint.to_dict()
    assert rep_taint.env["payload"].tainted

    # Function bodies are analyzed in an isolated scope (no env leakage).
    rep_scope = analyze_ast_intervals("def f():\n    hidden = 1\nleaked = hidden\n")
    assert "hidden" not in rep_scope.env and "leaked" in rep_scope.env

    # For-loop and Try/With constructs are analyzed, not crashed on.
    rep_for = analyze_ast_intervals("for i in range(3):\n    y = 1 / i\n")
    assert any(w["rule_id"] == "ABS-DIV-ZERO" for w in rep_for.warnings)
    rep_try = analyze_ast_intervals(
        "try:\n    a = 1\nexcept Exception:\n    a = 2\nfinally:\n    b = a + 1\n"
    )
    assert rep_try.env["b"].interval.lo <= rep_try.env["b"].interval.hi

    # Malformed / huge inputs fail closed but never raise.
    for bad in (None, 42, 3.14, ["a = 1"], {"x": 1}):
        r = analyze_ast_intervals(bad)  # type: ignore[arg-type]
        assert isinstance(r, AbstractFixpointReport) and not r.safe
        assert r.warnings[0]["rule_id"] == "ABS-INPUT"
    assert analyze_ast_intervals(b"a = 1\n").safe is True
    assert analyze_ast_intervals("").safe is True
    huge = ("x = 1\n" * 50_000)
    rep_huge = analyze_ast_intervals(huge)
    assert not rep_huge.safe and rep_huge.warnings[0]["rule_id"] == "ABS-SIZE-LIMIT"
    rep_syntax = analyze_ast_intervals("def broken(:\n")
    assert not rep_syntax.safe and rep_syntax.warnings[0]["rule_id"] == "ABS-SYNTAX"

    # Deterministic, repeatable reports.
    assert analyze_ast_intervals(code).to_dict() == analyze_ast_intervals(code).to_dict()

    print("abstractinterp._self_test: OK PASS (interval arithmetic, join/widen, div-by-zero detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
