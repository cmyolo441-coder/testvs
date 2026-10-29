"""earleyparser — Jay Earley's O(n^3) General Context-Free Grammar (CFG) Chart Parser.

Parses arbitrary Context-Free Grammars `G = (V, Sigma, P, S)` — including
left-recursive, right-recursive, nullable (`epsilon`), and ambiguous grammars —
without requiring Chomsky Normal Form conversion (Earley 1970):
  * Earley Item `(lhs -> rhs, dot_pos, origin_index)` stored in chart sets
    `S[0], S[1], ..., S[n]`.
  * **Predictor**: expands non-terminals to the right of the dot `• B`.
  * **Scanner**: matches terminal tokens `a` against input token `tokens[k]`.
  * **Completer**: advances parent items in `S[origin]` when `A -> alpha •`
    reaches the end of its production.

Used to validate complex recursive output structures (nested XML/HTML blocks,
recursive SQL/arithmetic queries, nested Markdown hierarchies) demanded by
system prompts.

Self-test: ``python3 -m fullagent.enforcex.earleyparser``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

__all__ = [
    "CFGProduction",
    "EarleyItem",
    "EarleyParseReport",
    "EarleyCFGParser",
]

_MAX_TOKENS = 4096


@dataclass(frozen=True)
class CFGProduction:
    lhs: str
    rhs: Tuple[str, ...]  # empty tuple () represents epsilon production


@dataclass(frozen=True)
class EarleyItem:
    prod: CFGProduction
    dot: int
    origin: int

    @property
    def completed(self) -> bool:
        return self.dot >= len(self.prod.rhs)

    @property
    def next_symbol(self) -> Optional[str]:
        if self.completed:
            return None
        return self.prod.rhs[self.dot]

    def advance(self) -> "EarleyItem":
        return EarleyItem(self.prod, self.dot + 1, self.origin)


@dataclass
class EarleyParseReport:
    accepted: bool
    tokens_consumed: int
    total_tokens: int
    chart_sizes: List[int] = field(default_factory=list)
    expected_next: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "tokens_consumed": self.tokens_consumed,
            "total_tokens": self.total_tokens,
            "chart_sizes": list(self.chart_sizes),
            "expected_next": list(self.expected_next),
        }


class EarleyCFGParser:
    """General Context-Free Grammar chart parser supporting left recursion & nullable rules."""

    def __init__(self, start_symbol: str = "S", *, max_tokens: int = _MAX_TOKENS) -> None:
        self.start_symbol = str(start_symbol)
        self.productions: List[CFGProduction] = []
        self.nonterminals: Set[str] = {self.start_symbol}
        self.max_tokens = max(1, int(max_tokens))
        self._by_lhs: Dict[str, Tuple[CFGProduction, ...]] = {}

    def add_rule(self, lhs: str, *rhs_alternatives: Sequence[str]) -> None:
        nt = str(lhs)
        self.nonterminals.add(nt)
        for alt in rhs_alternatives:
            self.productions.append(CFGProduction(nt, tuple(str(x) for x in alt)))
        self._by_lhs = {}

    def _productions_for(self, lhs: str) -> Tuple[CFGProduction, ...]:
        cached = self._by_lhs.get(lhs)
        if cached is None:
            cached = tuple(p for p in self.productions if p.lhs == lhs)
            self._by_lhs[lhs] = cached
        return cached

    def parse(self, tokens: Sequence[str]) -> EarleyParseReport:
        try:
            tok_list = [str(t) for t in (tokens or ())]
        except (TypeError, ValueError):
            return EarleyParseReport(accepted=False, tokens_consumed=0,
                                     total_tokens=0, chart_sizes=[],
                                     expected_next=[])
        total_n = len(tok_list)
        truncated = total_n > self.max_tokens
        if truncated:
            # Conservative: never claim acceptance for an unparsed suffix.
            tok_list = tok_list[: self.max_tokens]
        n = len(tok_list)
        chart: List[List[EarleyItem]] = [[] for _ in range(n + 1)]
        chart_sets: List[Set[EarleyItem]] = [set() for _ in range(n + 1)]

        def add_to_chart(k: int, item: EarleyItem) -> None:
            if item not in chart_sets[k]:
                chart_sets[k].add(item)
                chart[k].append(item)

        # Seed S[0] with start productions
        for prod in self._productions_for(self.start_symbol):
            add_to_chart(0, EarleyItem(prod, 0, 0))

        furthest_k = 0
        for k in range(n + 1):
            idx = 0
            while idx < len(chart[k]):
                item = chart[k][idx]
                idx += 1
                if not item.completed:
                    sym = item.next_symbol
                    if sym is None:
                        continue
                    if sym in self.nonterminals:
                        # Predictor
                        for prod in self._productions_for(sym):
                            add_to_chart(k, EarleyItem(prod, 0, k))
                        # Aycock-Horspool nullable advancement if sym already completed at (k, k)
                        if any(it.completed and it.origin == k and it.prod.lhs == sym for it in chart[k]):
                            add_to_chart(k, item.advance())
                    else:
                        # Scanner
                        if k < n and tok_list[k] == sym:
                            add_to_chart(k + 1, item.advance())
                            furthest_k = max(furthest_k, k + 1)
                else:
                    # Completer
                    for prev_item in list(chart[item.origin]):
                        if not prev_item.completed and prev_item.next_symbol == item.prod.lhs:
                            add_to_chart(k, prev_item.advance())

        accepted = (not truncated) and any(
            it.completed and it.origin == 0 and it.prod.lhs == self.start_symbol
            for it in chart[n]
        )
        expected = sorted({
            it.next_symbol
            for it in chart[furthest_k]
            if not it.completed and it.next_symbol is not None and it.next_symbol not in self.nonterminals
        })
        return EarleyParseReport(
            accepted=accepted,
            tokens_consumed=furthest_k,
            total_tokens=total_n,
            chart_sizes=[len(c) for c in chart],
            expected_next=expected,
        )


def _self_test() -> int:
    # Left-recursive arithmetic & parenthesized expression grammar:
    # E -> E + T | T
    # T -> T * F | F
    # F -> ( E ) | id
    p = EarleyCFGParser("E")
    p.add_rule("E", ("E", "+", "T"), ("T",))
    p.add_rule("T", ("T", "*", "F"), ("F",))
    p.add_rule("F", ("(", "E", ")"), ("id",))

    rep_ok = p.parse(["id", "+", "id", "*", "(", "id", "+", "id", ")"])
    assert rep_ok.accepted and rep_ok.tokens_consumed == 9, rep_ok.to_dict()

    # Unclosed parenthesis rejected with expected_next diagnostics
    rep_bad = p.parse(["(", "id", "+", "id"])
    assert not rep_bad.accepted and ")" in rep_bad.expected_next

    # ── new adversarial coverage ────────────────────────────────────────
    # 1. epsilon / nullable chains accept the empty and middle-epsilon strings
    eps = EarleyCFGParser("S")
    eps.add_rule("S", ("a", "A", "b"))
    eps.add_rule("A", ())
    assert eps.parse([]).accepted is False
    assert eps.parse(["a", "b"]).accepted is True
    eps2 = EarleyCFGParser("S")
    eps2.add_rule("S", ("A", "B"))
    eps2.add_rule("A", ())
    eps2.add_rule("B", ())
    assert eps2.parse([]).accepted is True

    # 2. left-recursive nullable cycle terminates and accepts epsilon
    cyc = EarleyCFGParser("S")
    cyc.add_rule("S", ("A",))
    cyc.add_rule("A", ("A",), ())
    assert cyc.parse([]).accepted is True
    assert cyc.parse(["x"]).accepted is False

    # 3. ambiguous grammar (S -> S S | a) is recognized without parse explosion
    amb = EarleyCFGParser("S")
    amb.add_rule("S", ("S", "S"), ("a",))
    rep_amb = amb.parse(["a", "a", "a", "a"])
    assert rep_amb.accepted and len(rep_amb.chart_sizes) == 5, rep_amb.to_dict()

    # 4. duplicate productions are hashconsed (no duplicate chart items)
    dup = EarleyCFGParser("S")
    dup.add_rule("S", ("x",))
    dup.add_rule("S", ("x",))
    assert dup.parse(["x"]).chart_sizes == [1, 1]

    # 5. huge token streams are conservatively rejected, never falsely accepted
    big = EarleyCFGParser("S", max_tokens=3)
    big.add_rule("S", ("S", "a"), ("a",))
    rep_big = big.parse(["a"] * 10)
    assert rep_big.accepted is False and rep_big.total_tokens == 10
    assert len(rep_big.chart_sizes) == 4

    # 6. malformed token sequences / empty grammar fail soft
    assert p.parse(None).accepted is False  # type: ignore[arg-type]
    assert p.parse(12345).accepted is False  # type: ignore[arg-type]
    empty = EarleyCFGParser("S")
    assert empty.parse(["x"]).accepted is False

    print("earleyparser._self_test: OK PASS (Jay Earley Predictor/Scanner/Completer on left-recursive CFG)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
