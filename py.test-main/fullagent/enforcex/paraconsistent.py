"""paraconsistent — Belnap 4-Valued Logic (FOUR) & Dunn-Belnap Bilattice Reasoner.

When an agent ingests conflicting evidence from multiple sources (e.g. a system
prompt, a stale RAG document, and a user message), classical logic suffers from
the Principle of Explosion (`p /\\ ~p |- q`).

Implements Nuel Belnap's four-valued paraconsistent logic `FOUR = {T, F, B, N}`:
  * `N` (Neither / Unknown):   `{}`          — no evidence for or against
  * `T` (True only):           `{1}`         — positive evidence only
  * `F` (False only):          `{0}`         — negative evidence only
  * `B` (Both / Contradictory):`{0, 1}`      — conflicting positive & negative evidence

Equipped with two orthogonal lattice orderings (Ginsberg Bilattice):
  1. **Truth Ordering (`<=_t`)**: `F <=_t N <=_t T` and `F <=_t B <=_t T`
     (operations: `t_meet` `/\\`, `t_join` `\\/`, negation `~T=F, ~F=T, ~B=B, ~N=N`).
  2. **Knowledge / Information Ordering (`<=_k`)**: `N <=_k T <=_k B` and `N <=_k F <=_k B`
     (operations: `k_meet` consensus `(x @ y)`, `k_join` gullibility `(x + y)`).

Contradiction detection: `ParaconsistentReasoner.observe_claims()` fuses
per-atom evidence from any number of sources and reports atoms that received
both positive and negative evidence (`B`). `detect_contradictions()` is the
stateless convenience wrapper. Fusion is a no-op on malformed values, so a
guard can feed it untrusted evidence without ever raising.

Self-test: ``python3 -m fullagent.enforcex.paraconsistent``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "BelnapValue",
    "BilatticeState",
    "ParaconsistentReasoner",
    "N",
    "F",
    "T",
    "B",
    "symbol_to_value",
    "detect_contradictions",
]


@dataclass(frozen=True)
class BelnapValue:
    """Element of Belnap's 4-valued bilattice represented by `(pos: bool, neg: bool)`."""
    pos: bool
    neg: bool

    @property
    def symbol(self) -> str:
        if self.pos and self.neg:
            return "B"  # Both (inconsistent / conflicting)
        if self.pos:
            return "T"  # True
        if self.neg:
            return "F"  # False
        return "N"      # Neither (unknown)

    def __invert__(self) -> "BelnapValue":
        """De Morgan negation swap `(pos, neg) -> (neg, pos)`; preserves `B` and `N`."""
        return BelnapValue(self.neg, self.pos)

    def t_meet(self, other: "BelnapValue") -> "BelnapValue":
        """Truth-lattice conjunction `x /\\_t y`: positive iff both positive; negative iff either negative."""
        return BelnapValue(self.pos and other.pos, self.neg or other.neg)

    def t_join(self, other: "BelnapValue") -> "BelnapValue":
        """Truth-lattice disjunction `x \\/_t y`: positive iff either positive; negative iff both negative."""
        return BelnapValue(self.pos or other.pos, self.neg and other.neg)

    def k_meet(self, other: "BelnapValue") -> "BelnapValue":
        """Knowledge-lattice consensus `x *_k y`: retains only agreed evidence."""
        return BelnapValue(self.pos and other.pos, self.neg and other.neg)

    def k_join(self, other: "BelnapValue") -> "BelnapValue":
        """Knowledge-lattice gullibility/fusion `x +_k y`: pools all positive and negative evidence."""
        return BelnapValue(self.pos or other.pos, self.neg or other.neg)

    def designated(self) -> bool:
        """In Priest's LP / Belnap designated values `{T, B}`, positive evidence is present."""
        return self.pos

    def classical_true(self) -> bool:
        """Strictly true without contradiction (`T` only)."""
        return self.pos and not self.neg

    def is_contradictory(self) -> bool:
        """True iff both positive and negative evidence are present (`B`)."""
        return bool(self.pos and self.neg)

    def is_unknown(self) -> bool:
        """True iff neither polarity is present (`N`)."""
        return not self.pos and not self.neg

    def t_le(self, other: "BelnapValue") -> bool:
        """Truth-order preorder: `self <=_t other`."""
        return (self.pos <= other.pos) and (self.neg >= other.neg)

    def k_le(self, other: "BelnapValue") -> bool:
        """Knowledge-order preorder: `self <=_k other`."""
        return (self.pos <= other.pos) and (self.neg <= other.neg)

    def to_dict(self) -> Dict[str, Any]:
        return {"pos": bool(self.pos), "neg": bool(self.neg),
                "symbol": self.symbol}


N = BelnapValue(False, False)
F = BelnapValue(False, True)
T = BelnapValue(True, False)
B = BelnapValue(True, True)

_SYMBOL_TABLE: Dict[str, BelnapValue] = {
    "T": T, "TRUE": T, "1": T, "YES": T, "POS": T, "POSITIVE": T,
    "F": F, "FALSE": F, "0": F, "NO": F, "NEG": F, "NEGATIVE": F,
    "B": B, "BOTH": B, "CONFLICT": B, "CONTRADICTION": B, "INCONSISTENT": B,
    "N": N, "NEITHER": N, "UNKNOWN": N, "": N, "NONE": N,
}
_MAX_ATOM_CHARS = 4096


def _safe_text(value: Any) -> str:
    """String coercion that never raises."""
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


def symbol_to_value(value: Any, default: BelnapValue = N) -> BelnapValue:
    """Coerce a symbol / bool / pair / BelnapValue into a `BelnapValue`.

    Unknown or malformed representations degrade to ``default`` (``N`` by
    default) instead of raising.
    """
    if isinstance(value, BelnapValue):
        return value
    if value is None:
        return default
    if isinstance(value, bool):
        return T if value else F
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return BelnapValue(bool(value[0]), bool(value[1]))
        except Exception:
            return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return T
        if value == 0:
            return F
        return default
    text = _safe_text(value).strip().upper()
    return _SYMBOL_TABLE.get(text, default)


@dataclass
class BilatticeState:
    valuations: Dict[str, str] = field(default_factory=dict)
    contradictory_atoms: List[str] = field(default_factory=list)
    unknown_atoms: List[str] = field(default_factory=list)
    consistent_true: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valuations": dict(self.valuations),
            "contradictory_atoms": list(self.contradictory_atoms),
            "unknown_atoms": list(self.unknown_atoms),
            "consistent_true": list(self.consistent_true),
        }


class ParaconsistentReasoner:
    """Fuses multi-source evidence in `FOUR` without explosion on conflicting atoms."""

    def __init__(self, max_atoms: int = 100_000) -> None:
        self.env: Dict[str, BelnapValue] = {}
        try:
            self.max_atoms = max(1, int(max_atoms))
        except Exception:
            self.max_atoms = 100_000

    @staticmethod
    def _atom_key(atom: Any) -> Optional[str]:
        key = _safe_text(atom).strip()
        if not key:
            return None
        if len(key) > _MAX_ATOM_CHARS:
            key = key[:_MAX_ATOM_CHARS]
        return key

    def observe(self, atom: str, value: BelnapValue) -> BelnapValue:
        """Fuse one observation. Malformed atoms/values are ignored."""
        key = self._atom_key(atom)
        if key is None:
            return N
        fused_value = symbol_to_value(value, default=N)
        cur = self.env.get(key, N)
        if cur == N and key not in self.env and len(self.env) >= self.max_atoms:
            # Bound memory: still report the observation without storing it.
            return fused_value
        fused = cur.k_join(fused_value)
        self.env[key] = fused
        return fused

    def observe_symbol(self, atom: str, symbol: Any) -> BelnapValue:
        """Convenience wrapper accepting a symbol string / bool / pair."""
        return self.observe(atom, symbol_to_value(symbol))

    def observe_claims(self, claims: Iterable[Any]) -> List[str]:
        """Fuse `(atom, polarity)` claims and return newly contradictory atoms.

        ``polarity`` may be a bool, a FOUR symbol, or a `BelnapValue`.
        Malformed claims are skipped; the caller never sees an exception.
        """
        before = {k for k, v in self.env.items() if v.is_contradictory()}
        try:
            items = list(claims or [])
        except Exception:
            items = []
        for claim in items:
            try:
                atom, value = claim
            except Exception:
                continue
            self.observe(atom, symbol_to_value(value))
        after = {k for k, v in self.env.items() if v.is_contradictory()}
        return sorted(after - before)

    def contradictions(self) -> List[str]:
        """Atoms currently carrying both positive and negative evidence."""
        return sorted(k for k, v in self.env.items() if v.is_contradictory())

    def is_consistent(self) -> bool:
        """True when no atom is in the contradictory `B` state."""
        return not any(v.is_contradictory() for v in self.env.values())

    def merge(self, other: "ParaconsistentReasoner") -> "ParaconsistentReasoner":
        """Knowledge-order fusion of another reasoner's evidence into this one."""
        if not isinstance(other, ParaconsistentReasoner):
            return self
        for atom in sorted(other.env):
            self.observe(atom, other.env[atom])
        return self

    def decide(self, atom: str, *, on_contradiction: str = "review",
               default: str = "unknown") -> str:
        """Map an atom's FOUR value to a guard-friendly decision string.

        `T` -> ``accept``, `F` -> ``reject``, `B` -> ``on_contradiction``
        (``review`` by default), `N` -> ``default`` (``unknown``).
        """
        value = self.value_of(atom)
        if value == T:
            return "accept"
        if value == F:
            return "reject"
        if value == B:
            mode = _safe_text(on_contradiction).strip().lower()
            return mode if mode in ("review", "accept", "reject") else "review"
        mode = _safe_text(default).strip().lower()
        return mode if mode in ("unknown", "review", "accept", "reject") \
            else "unknown"

    def value_of(self, atom: str) -> BelnapValue:
        key = self._atom_key(atom)
        if key is None:
            return N
        return self.env.get(key, N)

    def summarize(self) -> BilatticeState:
        vals: Dict[str, str] = {}
        both: List[str] = []
        neither: List[str] = []
        strict_t: List[str] = []
        for atom in sorted(self.env):
            value = self.env[atom]
            vals[atom] = value.symbol
            if value == B:
                both.append(atom)
            elif value == N:
                neither.append(atom)
            elif value == T:
                strict_t.append(atom)
        return BilatticeState(
            valuations=vals,
            contradictory_atoms=both,
            unknown_atoms=neither,
            consistent_true=strict_t,
        )


def detect_contradictions(claims: Iterable[Any]) -> List[str]:
    """Stateless contradiction scan over ``(atom, polarity)`` evidence claims.

    Returns the sorted atoms that received both positive and negative
    evidence. Never raises; malformed claims are ignored.
    """
    reasoner = ParaconsistentReasoner()
    reasoner.observe_claims(claims)
    return reasoner.contradictions()


def _self_test() -> int:
    # 1. Verify Bilattice identities: ~B == B, ~N == N, F /\_t T == F, F +_k T == B, B *_k T == T
    assert (~B) == B and (~N) == N and (~T) == F and (~F) == T
    assert F.t_meet(T) == F and F.t_join(T) == T
    assert N.t_meet(B) == F and N.t_join(B) == T
    assert F.k_join(T) == B and B.k_meet(T) == T

    # 2. Verify non-explosion: contradictory atom `B` does not make unrelated atom `q` true
    pr = ParaconsistentReasoner()
    pr.observe("allow_markdown", T)
    pr.observe("allow_markdown", F)  # conflicting source -> B
    pr.observe("json_valid", T)
    state = pr.summarize()
    assert state.valuations["allow_markdown"] == "B"
    assert "allow_markdown" in state.contradictory_atoms
    assert state.valuations["json_valid"] == "T"
    assert pr.value_of("unrelated_claim") == N

    # 3. symbol coercion accepts bools, symbols and pairs; garbage -> N
    assert symbol_to_value(True) == T and symbol_to_value(False) == F
    assert symbol_to_value("both") == B and symbol_to_value("unknown") == N
    assert symbol_to_value((True, True)) == B
    assert symbol_to_value(object()) == N
    assert symbol_to_value(None, default=T) == T

    # 4. explicit contradiction detection over multi-source claims
    claims = [
        ("allow_markdown", True),
        ("allow_markdown", "F"),        # conflicting negative evidence
        ("json_valid", "T"),
        ("json_valid", "T"),
        ("tool_use", False),
    ]
    pr2 = ParaconsistentReasoner()
    new_bad = pr2.observe_claims(claims)
    assert new_bad == ["allow_markdown"], new_bad
    assert pr2.contradictions() == ["allow_markdown"]
    assert not pr2.is_consistent()
    assert pr2.value_of("json_valid") == T
    assert pr2.decide("allow_markdown") == "review"
    assert pr2.decide("allow_markdown", on_contradiction="reject") == "reject"
    assert pr2.decide("json_valid") == "accept"
    assert pr2.decide("tool_use") == "reject"
    assert pr2.decide("never_seen") == "unknown"
    assert detect_contradictions(claims) == ["allow_markdown"]

    # 5. malformed observations and claims are ignored, never raise
    for atom, val in [(None, T), ("", T), ("x", object()),
                      (["unhashable"], T), (object(), "T")]:
        fused = pr2.observe(atom, val)  # type: ignore[arg-type]
        assert isinstance(fused, BelnapValue)
    assert pr2.observe_claims([None, ("only-one",), 42, ("ok", "T")]) == []
    assert pr2.value_of("ok") == T
    huge_atom = "a" * 10_000
    pr2.observe(huge_atom, T)
    assert pr2.value_of(huge_atom) == T

    # 6. merge is knowledge-order fusion and preserves contradictions
    left = ParaconsistentReasoner()
    left.observe("p", T)
    right = ParaconsistentReasoner()
    right.observe("p", F)
    right.observe("q", F)
    left.merge(right)
    assert left.value_of("p") == B and left.value_of("q") == F
    assert left.summarize().contradictory_atoms == ["p"]
    left.merge(None)  # type: ignore[arg-type]
    assert left.value_of("p") == B

    # 7. lattice order relations are consistent with the four values
    assert F.t_le(N) and N.t_le(T) and F.t_le(B) and B.t_le(T)
    assert not T.t_le(B)
    assert N.k_le(T) and T.k_le(B) and N.k_le(F) and F.k_le(B)
    assert not B.k_le(T)

    print("paraconsistent._self_test: OK PASS (Belnap FOUR truth/knowledge bilattices, "
          "non-explosion, contradiction detection, defensive fusion)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
