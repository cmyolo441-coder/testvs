"""contracts — Hoare-Triple `{Pre} Turn {Post}` Design-by-Contract (DbC) Enforcer.

Provides executable precondition, postcondition, and invariant contracts with
formal blame assignment (`CALLER_BLAME` when a precondition fails, `CALLEE_BLAME`
when a postcondition or invariant fails given a valid precondition).

Self-test: ``python3 -m fullagent.enforcex.contracts``
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

__all__ = [
    "ContractViolationError",
    "ContractAuditReport",
    "ContractCompiler",
    "requires",
    "ensures",
]


def _safe_predicate(pred: Any, kind: str, blame: str, message: str,
                    *args: Any, **kwargs: Any) -> bool:
    """Evaluate a contract predicate; malformed predicates count as violations."""
    if not callable(pred):
        raise ContractViolationError(kind, blame, f"{message} (predicate is not callable)")
    try:
        return bool(pred(*args, **kwargs))
    except ContractViolationError:
        raise
    except Exception as exc:
        raise ContractViolationError(
            kind, blame, f"{message} (predicate raised {type(exc).__name__})") from exc


class ContractViolationError(AssertionError):
    def __init__(self, kind: str, blame: str, message: str) -> None:
        self.kind = kind          # precondition | postcondition | invariant
        self.blame = blame        # caller | callee
        super().__init__(f"[{kind.upper()}:{blame.upper()}] {message}")


@dataclass
class ContractAuditReport:
    passed: bool = True
    blame: str = "none"       # none | caller | callee
    failed_contracts: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "blame": self.blame,
            "failed_contracts": list(self.failed_contracts),
        }


def requires(pred: Callable[..., bool], message: str = "precondition failed") -> Callable:
    """Decorator enforcing a caller precondition before function execution."""
    def _dec(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            if not _safe_predicate(pred, "precondition", "caller", message, *args, **kwargs):
                raise ContractViolationError("precondition", "caller", message)
            return fn(*args, **kwargs)
        return _wrap
    return _dec


def ensures(pred: Callable[[Any], bool], message: str = "postcondition failed") -> Callable:
    """Decorator enforcing a callee postcondition on the function return value."""
    def _dec(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def _wrap(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            if not _safe_predicate(pred, "postcondition", "callee", message, result):
                raise ContractViolationError("postcondition", "callee", message)
            return result
        return _wrap
    return _dec


class ContractCompiler:
    """Evaluates Hoare-triple `{Pre} action {Post}` contracts over turn dictionaries."""

    def __init__(self) -> None:
        self._pre: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = []
        self._post: List[Tuple[str, Callable[[Dict[str, Any], Dict[str, Any]], bool]]] = []
        self._inv: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = []

    def add_precondition(self, name: str, fn: Callable[[Dict[str, Any]], bool]) -> None:
        self._pre.append((name, fn))

    def add_postcondition(self, name: str, fn: Callable[[Dict[str, Any], Dict[str, Any]], bool]) -> None:
        self._post.append((name, fn))

    def add_invariant(self, name: str, fn: Callable[[Dict[str, Any]], bool]) -> None:
        """Add a state invariant checked after preconditions and after the turn."""
        self._inv.append((name, fn))

    @staticmethod
    def _holds(fn: Any, *args: Any) -> bool:
        try:
            return bool(fn(*args))
        except Exception:
            return False

    def verify_turn(self, pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> ContractAuditReport:
        failures: List[Dict[str, str]] = []
        for name, pfn in self._pre:
            if not self._holds(pfn, pre_state):
                failures.append({"name": name, "kind": "precondition", "blame": "caller"})
        if failures:
            return ContractAuditReport(passed=False, blame="caller", failed_contracts=failures)

        for name, ifn in self._inv:
            if not self._holds(ifn, pre_state) or not self._holds(ifn, post_state):
                failures.append({"name": name, "kind": "invariant", "blame": "callee"})
        if failures:
            return ContractAuditReport(passed=False, blame="callee", failed_contracts=failures)

        for name, qfn in self._post:
            if not self._holds(qfn, pre_state, post_state):
                failures.append({"name": name, "kind": "postcondition", "blame": "callee"})
        if failures:
            return ContractAuditReport(passed=False, blame="callee", failed_contracts=failures)

        return ContractAuditReport(passed=True, blame="none")


def _self_test() -> int:
    @requires(lambda x: x > 0, "x must be positive")
    @ensures(lambda res: res >= 2, "result must be >= 2")
    def double_pos(x: int) -> int:
        return x * 2

    assert double_pos(3) == 6
    try:
        double_pos(-1)
        raise AssertionError("expected caller blame")
    except ContractViolationError as exc:
        assert exc.blame == "caller"

    cc = ContractCompiler()
    cc.add_precondition("has_system_prompt", lambda s: bool(s.get("system_prompt")))
    cc.add_postcondition("monotonic_seq", lambda old, new: new.get("seq", 0) > old.get("seq", 0))
    rep = cc.verify_turn({"system_prompt": "ok", "seq": 1}, {"system_prompt": "ok", "seq": 2})
    assert rep.passed and rep.blame == "none"

    # A raising predicate is a contract violation, not an uncaught crash
    def _boom(_x):
        raise RuntimeError("bad predicate")
    try:
        requires(_boom, "must not raise")(lambda x: x)(1)
        raise AssertionError("expected caller blame")
    except ContractViolationError as exc:
        assert exc.kind == "precondition" and exc.blame == "caller"
        assert "RuntimeError" in str(exc)

    # Compiler: raising / non-callable predicates become blame, not tracebacks
    cc_bad = ContractCompiler()
    cc_bad.add_precondition("raising", _boom)
    cc_bad.add_postcondition("never", lambda old, new: True)
    rep_bad = cc_bad.verify_turn({}, {})
    assert not rep_bad.passed and rep_bad.blame == "caller"
    assert rep_bad.failed_contracts[0]["kind"] == "precondition"
    cc_none = ContractCompiler()
    cc_none.add_precondition("not_callable", None)  # type: ignore[arg-type]
    assert cc_none.verify_turn(None, None).blame == "caller"  # type: ignore[arg-type]

    # Invariants are checked on both states; failure attributed to the callee
    cc_inv = ContractCompiler()
    cc_inv.add_invariant("non_negative_seq", lambda s: int(s.get("seq", 0)) >= 0)
    cc_inv.add_postcondition("advance", lambda old, new: new.get("seq") == old.get("seq") + 1)
    assert cc_inv.verify_turn({"seq": 1}, {"seq": 2}).passed
    rep_inv = cc_inv.verify_turn({"seq": 1}, {"seq": -1})
    assert not rep_inv.passed and rep_inv.blame == "callee"
    assert rep_inv.failed_contracts[0]["kind"] == "invariant"

    # Postconditions are not evaluated when a precondition already failed
    calls = {"post": 0}
    cc_order = ContractCompiler()

    def _post(old, new):
        calls["post"] += 1
        return True
    cc_order.add_precondition("fail", lambda s: False)
    cc_order.add_postcondition("should_not_run", _post)
    assert cc_order.verify_turn({"a": 1}, {"a": 2}).blame == "caller"
    assert calls["post"] == 0

    # Empty compiler accepts any turn; decorator metadata is preserved
    assert ContractCompiler().verify_turn({}, {}).passed
    assert double_pos.__name__ == "double_pos"

    print("contracts._self_test: OK PASS (Hoare-triple @requires/@ensures + caller/callee blame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
