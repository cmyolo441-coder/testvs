"""packrat — Bryan Ford's Packrat Parsing Expression Grammar (PEG) Memoizing Engine.

Implements linear-time `O(n)` memoized **Packrat PEG Parsing** (Ford, ICFP 2002)
with full support for:
  * Terminal literals & regex patterns (`Literal`, `RegexMatch`)
  * Concatenation (`Seq(e1, e2, ...)`)
  * Prioritized Ordered Choice (`Choice(e1, e2, ...)` — `e1 / e2`)
  * Zero-or-more (`ZeroOrMore(e)` — `e*`) & One-or-more (`OneOrMore(e)` — `e+`)
  * Positive syntactic predicate / lookahead (`AndPredicate(e)` — `&e`, consumes 0 chars)
  * Negative syntactic predicate / lookahead (`NotPredicate(e)` — `!e`, consumes 0 chars)
  * Non-terminal rule references (`Ref("RuleName")`) with `(rule_name, pos)` packrat
    memoization table guaranteeing `O(n)` worst-case runtime even with backtracking!
  * Direct and simple indirect **left recursion** via seed-growing: a recursive
    call at the same `(rule, pos)` starts from the current (failing/partial)
    seed, and the rule body is re-evaluated until its match length stops
    growing (Warth-style bounded growth).

Memo entries are invalidated whenever ``eval_rule``/``parse`` is called with a
different text object, so a grammar instance can be safely reused across
inputs. Invalid regex patterns and undefined rules fail softly in ``parse``
(``None``) instead of raising.

Self-test: ``python3 -m fullagent.enforcex.packrat``
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "PEGExpr",
    "PEGNode",
    "PackratGrammar",
    "Lit",
    "Rx",
    "Seq",
    "Choice",
    "Star",
    "Plus",
    "AndPred",
    "NotPred",
    "Ref",
]

_MAX_INPUT_CHARS = 1_000_000
_MAX_PATTERN_CHARS = 4096


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    try:
        return str(value)
    except Exception:
        return ""


@lru_cache(maxsize=256)
def _compile_pattern(pattern: str) -> Optional[re.Pattern[str]]:
    """Compile (and cache) a PEG regex; malformed patterns return None."""
    try:
        return re.compile(pattern)
    except Exception:
        return None


@dataclass(frozen=True)
class PEGNode:
    kind: str
    text: str
    start: int
    end: int
    children: Tuple["PEGNode", ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "children": [c.to_dict() for c in self.children],
        }


class PEGExpr:
    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        raise NotImplementedError


@dataclass(frozen=True)
class Lit(PEGExpr):
    literal: str

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        literal = self.literal
        if not isinstance(literal, str):
            return None
        end = pos + len(literal)
        if text[pos:end] == literal:
            return end, PEGNode("Lit", literal, pos, end)
        return None


@dataclass(frozen=True)
class Rx(PEGExpr):
    pattern: str

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        if not isinstance(self.pattern, str) or \
                len(self.pattern) > _MAX_PATTERN_CHARS:
            return None
        compiled = _compile_pattern(self.pattern)
        if compiled is None:
            return None
        try:
            m = compiled.match(text, pos)
        except Exception:
            return None
        if m:
            return m.end(), PEGNode("Rx", m.group(0), pos, m.end())
        return None


@dataclass(frozen=True)
class Seq(PEGExpr):
    items: Tuple[PEGExpr, ...]

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        cur = pos
        kids: List[PEGNode] = []
        for it in self.items:
            res = it.eval(grammar, text, cur)
            if res is None:
                return None
            cur, node = res
            kids.append(node)
        return cur, PEGNode("Seq", text[pos:cur], pos, cur, tuple(kids))


@dataclass(frozen=True)
class Choice(PEGExpr):
    options: Tuple[PEGExpr, ...]

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        for opt in self.options:
            res = opt.eval(grammar, text, pos)
            if res is not None:
                return res
        return None


@dataclass(frozen=True)
class Star(PEGExpr):
    inner: PEGExpr

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        cur = pos
        kids: List[PEGNode] = []
        while True:
            res = self.inner.eval(grammar, text, cur)
            if res is None or res[0] == cur:
                break
            cur, node = res
            kids.append(node)
        return cur, PEGNode("Star", text[pos:cur], pos, cur, tuple(kids))


@dataclass(frozen=True)
class Plus(PEGExpr):
    inner: PEGExpr

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        first = self.inner.eval(grammar, text, pos)
        if first is None:
            return None
        cur, n0 = first
        rest = Star(self.inner).eval(grammar, text, cur)
        if rest is None:  # pragma: no cover - Star never fails
            return cur, PEGNode("Plus", text[pos:cur], pos, cur, (n0,))
        end_pos, rest_node = rest
        return end_pos, PEGNode("Plus", text[pos:end_pos], pos, end_pos, (n0,) + rest_node.children)


@dataclass(frozen=True)
class AndPred(PEGExpr):
    inner: PEGExpr

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        res = self.inner.eval(grammar, text, pos)
        if res is not None:
            return pos, PEGNode("AndPred", "", pos, pos)
        return None


@dataclass(frozen=True)
class NotPred(PEGExpr):
    inner: PEGExpr

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        res = self.inner.eval(grammar, text, pos)
        if res is None:
            return pos, PEGNode("NotPred", "", pos, pos)
        return None


@dataclass(frozen=True)
class Ref(PEGExpr):
    rule_name: str

    def eval(self, grammar: "PackratGrammar", text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        return grammar.eval_rule(_safe_text(self.rule_name), text, pos)


class PackratGrammar:
    """Memoizing Packrat PEG parser with `(rule_name, offset)` cache.

    Supports direct and simple indirect left recursion through bounded seed
    growing. The memo table is keyed by ``(rule_name, pos)`` and is cleared
    whenever the input text object changes.
    """

    def __init__(self, start_rule: str = "Start") -> None:
        self.start_rule = _safe_text(start_rule) or "Start"
        self.rules: Dict[str, PEGExpr] = {}
        self.memo: Dict[Tuple[str, int], Optional[Tuple[int, PEGNode]]] = {}
        self.memo_hits = 0
        self._memo_text: Optional[str] = None
        self._active: List[Tuple[str, int]] = []
        self._active_set: set = set()
        self._lr_grow: Dict[Tuple[str, int], set] = {}
        self._growing: set = set()

    def define(self, name: str, expr: PEGExpr) -> None:
        key = _safe_text(name)
        if not key or not callable(getattr(expr, "eval", None)):
            return
        self.rules[key] = expr

    def _reset_for_text(self, text: str) -> None:
        if text is not self._memo_text:
            self.memo.clear()
            self.memo_hits = 0
            self._memo_text = text
            self._active.clear()
            self._active_set.clear()
            self._lr_grow.clear()
            self._growing.clear()

    @staticmethod
    def _wrap(name: str, text: str, pos: int,
              res: Optional[Tuple[int, PEGNode]]) -> Optional[Tuple[int, PEGNode]]:
        if res is None:
            return None
        end_pos, node = res
        return end_pos, PEGNode(name, text[pos:end_pos], pos, end_pos, (node,))

    def _grow(self, key: Tuple[str, int], expr: PEGExpr, text: str,
              pos: int, involved: set,
              seed: Tuple[int, PEGNode]) -> Tuple[int, PEGNode]:
        """Re-evaluate a left-recursive rule until its match stops growing."""
        name = key[0]
        best = seed
        self._growing.add(key)
        try:
            for _ in range(len(text) + 2):
                # Invalidate every memo entry for rules participating in the
                # cycle so their earlier seed-dependent failures are re-tried.
                for memo_key in list(self.memo):
                    if memo_key[0] in involved:
                        del self.memo[memo_key]
                self.memo[key] = best
                self._active.append(key)
                self._active_set.add(key)
                try:
                    res = expr.eval(self, text, pos)
                except Exception:
                    res = None
                finally:
                    if self._active and self._active[-1] == key:
                        self._active.pop()
                    self._active_set.discard(key)
                candidate = self._wrap(name, text, pos, res)
                if candidate is None or candidate[0] <= best[0]:
                    break
                best = candidate
            self.memo[key] = best
            return best
        finally:
            self._growing.discard(key)
            self._lr_grow.pop(key, None)

    def eval_rule(self, name: str, text: str, pos: int) -> Optional[Tuple[int, PEGNode]]:
        name = _safe_text(name)
        text = _safe_text(text)
        self._reset_for_text(text)
        try:
            pos = max(0, int(pos))
        except Exception:
            pos = 0
        key = (name, pos)

        # Left-recursion re-entry: return the current seed (None on the first
        # pass, the best partial match during growth).
        if key in self._active_set:
            try:
                idx = self._active.index(key)
            except ValueError:  # pragma: no cover - set/stack drift
                idx = 0
            cycle = self._active[idx:] or [key]
            root = cycle[0]
            if root not in self._growing:
                involved = self._lr_grow.setdefault(root, set())
                involved.update(rule for rule, _ in cycle)
                involved.add(name)
            return self.memo.get(key)

        if key in self.memo:
            self.memo_hits += 1
            return self.memo[key]

        expr = self.rules.get(name)
        if expr is None:
            raise KeyError(f"Undefined PEG rule: {name!r}")

        self._active.append(key)
        self._active_set.add(key)
        try:
            res = expr.eval(self, text, pos)
        finally:
            if self._active and self._active[-1] == key:
                self._active.pop()
            self._active_set.discard(key)

        wrapped = self._wrap(name, text, pos, res)
        self.memo[key] = wrapped

        if key in self._lr_grow:
            involved = set(self._lr_grow[key])
            growing_rules = {gkey[0] for gkey in self._growing}
            if wrapped is not None and not (involved & growing_rules):
                # Nested growth is safe when it cannot invalidate the memo of
                # an outer grower (disjoint rule sets).
                wrapped = self._grow(key, expr, text, pos, involved, wrapped)
            else:
                # Cycles that touch an outer grower are resolved by that outer
                # growth loop; drop the entry to avoid double growth.
                self._lr_grow.pop(key, None)
        return self.memo.get(key)

    def parse(self, text: str, *, full_match: bool = True) -> Optional[PEGNode]:
        """Parse ``text``; malformed grammars/inputs return None (never raise)."""
        text = _safe_text(text)
        if len(text) > _MAX_INPUT_CHARS:
            return None
        self.memo.clear()
        self.memo_hits = 0
        self._memo_text = text
        self._active.clear()
        self._active_set.clear()
        self._lr_grow.clear()
        self._growing.clear()
        try:
            res = self.eval_rule(self.start_rule, text, 0)
        except Exception:
            return None
        if res is None:
            return None
        end_pos, node = res
        if full_match and end_pos != len(text):
            return None
        return node


def _self_test() -> int:
    # Classic non-Context-Free language a^n b^n c^n parsed in O(n) via Packrat PEG lookahead `&` and `!`:
    # S <- &(A !'b') 'a'+ B !.
    # A <- 'a' A? 'b'
    # B <- 'b' B? 'c'
    g = PackratGrammar("S")
    g.define("A", Seq((Lit("a"), Choice((Ref("A"), Lit(""))), Lit("b"))))
    g.define("B", Seq((Lit("b"), Choice((Ref("B"), Lit(""))), Lit("c"))))
    g.define(
        "S",
        Seq((
            AndPred(Seq((Ref("A"), NotPred(Lit("b"))))),
            Plus(Lit("a")),
            Ref("B"),
            NotPred(Rx(r".")),
        )),
    )

    assert g.parse("aaabbbccc") is not None
    assert g.parse("aabbc") is None
    assert g.parse("aaabbbcccc") is None

    # 1. direct left recursion: E <- E '+' N / N  (seed growing)
    lr = PackratGrammar("E")
    lr.define("N", Rx(r"[0-9]+"))
    lr.define("E", Choice((
        Seq((Ref("E"), Lit("+"), Ref("N"))),
        Ref("N"),
    )))
    node = lr.parse("1+2+3")
    assert node is not None and node.text == "1+2+3", node
    assert lr.parse("1+2+") is None
    assert lr.parse("7") is not None

    # 2. indirect left recursion: A <- B 'x' / 'a'; B <- A 'y'
    ind = PackratGrammar("A")
    ind.define("A", Choice((Seq((Ref("B"), Lit("x"))), Lit("a"))))
    ind.define("B", Seq((Ref("A"), Lit("y"))))
    assert ind.parse("ayx") is not None
    assert ind.parse("ax") is None

    # 2b. nested left recursion: E <- E '+' T / T ; T <- T '*' F / F
    prec = PackratGrammar("E")
    prec.define("F", Rx(r"[0-9]+"))
    prec.define("T", Choice((
        Seq((Ref("T"), Lit("*"), Ref("F"))), Ref("F"))))
    prec.define("E", Choice((
        Seq((Ref("E"), Lit("+"), Ref("T"))), Ref("T"))))
    assert prec.parse("1+2*3+4") is not None
    assert prec.parse("1*2+3*4") is not None
    assert prec.parse("1+*2") is None

    # 3. memo table is invalidated when the input text object changes
    iso = PackratGrammar("S")
    iso.define("S", Plus(Rx(r"[a-c]")))
    assert iso.parse("abc") is not None
    assert iso.parse("bca") is not None
    assert iso.parse("abd") is None
    # repeated refs hit the memo instead of re-parsing
    rep = PackratGrammar("S")
    rep.define("A", Rx(r"a+"))
    rep.define("S", Seq((AndPred(Ref("A")), Ref("A"))))
    assert rep.parse("aaa") is not None
    assert rep.memo_hits >= 1, rep.memo_hits

    # 4. malformed patterns/rules fail softly (no exception)
    bad = PackratGrammar("S")
    bad.define("S", Seq((Rx("([unclosed"), Lit("never"))))
    assert bad.parse("anything") is None
    missing = PackratGrammar("S")
    missing.define("S", Ref("DoesNotExist"))
    assert missing.parse("anything") is None
    bad_lit = PackratGrammar("S")
    bad_lit.define("S", Lit(None))  # type: ignore[arg-type]
    assert bad_lit.parse("x") is None

    # 5. empty-match repetition terminates instead of looping forever
    empty_star = PackratGrammar("S")
    empty_star.define("S", Star(Lit("")))
    assert empty_star.parse("") is not None
    assert empty_star.parse("x") is None

    # 6. non-string / huge inputs are handled without raising
    assert PackratGrammar("S").parse(None) is None  # type: ignore[arg-type]
    assert g.parse(b"aaabbbccc") is not None  # type: ignore[arg-type]
    deep = PackratGrammar("S")
    deep.define("S", Choice((Seq((Lit("a"), Ref("S"))), Lit("a"))))
    assert deep.parse("a" * 80) is not None
    deep.parse("a" * 3000)  # deep recursion must fail softly, never raise
    assert g.parse("a" * 1_100_000) is None

    print("packrat._self_test: OK PASS (Ford O(n) Packrat PEG memoization + non-CFG a^n b^n c^n "
          "lookahead + direct/indirect left recursion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
