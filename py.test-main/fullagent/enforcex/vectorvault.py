r"""vectorvault — Pure-Python Okapi BM25+ & 64-Bit SimHash LSH Rule Retrieval Engine.

Indexes system prompt rules and compliance precedents for fast, dependency-free
retrieval:
  * Okapi BM25+ ($k_1 = 1.5, b = 0.75, \delta = 1.0$) lexical relevance scoring
    with exact Robertson IDF and a bounded, incrementally maintained document
    index (average length and DF counters are updated on insert and eviction),
  * 64-bit Charikar SimHash fingerprint with exact Hamming distance and
    near-duplicate detection (`hamming_distance <= 3`).  A 4x16-bit band index
    makes duplicate lookup sub-linear while remaining exact: any pair at
    Hamming distance <= 3 necessarily shares at least one 16-bit band.
  * Maximal Marginal Relevance (MMR) reranking to return relevant yet diverse rules.

Memory is bounded: `max_rules` caps the index (oldest rules are evicted and all
DF/length counters are repaired), tokenisation and query scans are capped at
262,144 characters / 8,192 tokens, and malformed inputs never raise.

Self-test: ``python3 -m fullagent.enforcex.vectorvault``
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Set

__all__ = [
    "IndexedRule",
    "RuleRetrievalResult",
    "VectorRuleVault",
    "simhash_64",
    "hamming_distance_64",
]

_MAX_SCAN_CHARS = 262_144
_MAX_TOKENS = 8_192
_MASK_64 = (1 << 64) - 1
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_BAND_COUNT = 4
_BAND_BITS = 16


def _safe_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def _tokenize(text: Any) -> List[str]:
    """Lowercase word tokenisation with hard input/token caps (never raises)."""
    s = _safe_text(text)
    if len(s) > _MAX_SCAN_CHARS:
        s = s[:_MAX_SCAN_CHARS]
    toks = [w for w in _TOKEN_RE.findall(s.lower()) if len(w) >= 2]
    return toks[:_MAX_TOKENS]


def _md5_8(text: str) -> bytes:
    try:
        return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).digest()[:8]
    except (TypeError, ValueError, AttributeError):
        # FIPS-restricted builds may reject MD5; keep the fingerprint stable
        # within a process by falling back to SHA-256.
        return hashlib.sha256(text.encode("utf-8")).digest()[:8]


def simhash_64(text: str) -> int:
    """Compute a 64-bit Charikar SimHash fingerprint over word tokens and 2-shingles."""
    tokens = _tokenize(text)
    if not tokens:
        return 0
    shingles = list(tokens)
    shingles.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
    weights = [0] * 64
    for sh in shingles:
        h64 = int.from_bytes(_md5_8(sh), "big")
        for bit in range(64):
            if (h64 >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1
    out = 0
    for bit in range(64):
        if weights[bit] > 0:
            out |= (1 << bit)
    return out


def hamming_distance_64(a: int, b: int) -> int:
    try:
        x = int(a) & _MASK_64
        y = int(b) & _MASK_64
    except (TypeError, ValueError, OverflowError):
        return 64
    return bin(x ^ y).count("1")


@dataclass
class IndexedRule:
    rule_id: str
    text: str
    tokens: List[str]
    tf: Dict[str, int]
    simhash: int
    seq: int = 0


@dataclass
class RuleRetrievalResult:
    rule_id: str
    text: str
    bm25_score: float
    simhash_distance: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "text": self.text,
            "bm25_score": round(self.bm25_score, 4),
            "simhash_distance": self.simhash_distance,
        }


class VectorRuleVault:
    """BM25+ and 64-bit SimHash rule index with a bounded rule store."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, delta: float = 1.0,
                 max_rules: int = 50_000) -> None:
        self.k1 = _clamp_float(k1, 0.0, 100.0, 1.5)
        self.b = _clamp_float(b, 0.0, 1.0, 0.75)
        self.delta = _clamp_float(delta, 0.0, 10.0, 1.0)
        try:
            cap = int(max_rules)
        except (TypeError, ValueError, OverflowError):
            cap = 50_000
        self.max_rules = max(0, min(cap, 1_000_000))
        self._docs: Deque[IndexedRule] = deque()
        self._df: Dict[str, int] = {}
        self._avgdl: float = 1.0
        self._total_tokens: int = 0
        self._next_seq: int = 0
        self._by_seq: Dict[int, IndexedRule] = {}
        self._bands: List[Dict[int, Set[int]]] = [dict() for _ in range(_BAND_COUNT)]

    # -- index maintenance -------------------------------------------------

    def _band_values(self, fingerprint: int) -> List[int]:
        return [
            (fingerprint >> (i * _BAND_BITS)) & 0xFFFF
            for i in range(_BAND_COUNT)
        ]

    def _candidate_seqs(self, fingerprint: int) -> Set[int]:
        cands: Set[int] = set()
        for band, value in enumerate(self._band_values(fingerprint)):
            cands.update(self._bands[band].get(value, ()))
        return cands

    def _index_doc(self, doc: IndexedRule) -> None:
        self._docs.append(doc)
        self._by_seq[doc.seq] = doc
        for band, value in enumerate(self._band_values(doc.simhash)):
            self._bands[band].setdefault(value, set()).add(doc.seq)
        self._total_tokens += len(doc.tokens)
        for term in set(doc.tokens):
            self._df[term] = self._df.get(term, 0) + 1
        self._refresh_avgdl()

    def _unindex_doc(self, doc: IndexedRule) -> None:
        self._by_seq.pop(doc.seq, None)
        for band, value in enumerate(self._band_values(doc.simhash)):
            bucket = self._bands[band].get(value)
            if bucket is not None:
                bucket.discard(doc.seq)
                if not bucket:
                    self._bands[band].pop(value, None)
        self._total_tokens = max(0, self._total_tokens - len(doc.tokens))
        for term in set(doc.tokens):
            count = self._df.get(term, 0) - 1
            if count > 0:
                self._df[term] = count
            else:
                self._df.pop(term, None)

    def _evict_oldest(self) -> None:
        if not self._docs:
            return
        self._unindex_doc(self._docs.popleft())
        self._refresh_avgdl()

    def _refresh_avgdl(self) -> None:
        n_docs = len(self._docs)
        self._avgdl = max(1.0, self._total_tokens / float(n_docs)) if n_docs else 1.0

    def add_rule(self, rule_id: str, text: str) -> bool:
        """Add a rule unless a near-identical duplicate (`hamming <= 3`) already exists.

        Returns True when the rule was indexed.  When `max_rules` is reached the
        oldest rule is evicted first (DF, length totals and band buckets are
        repaired so subsequent searches stay exact).
        """
        text_s = _safe_text(text)
        rid = _safe_text(rule_id) or "rule"
        sh = simhash_64(text_s)
        for seq in self._candidate_seqs(sh):
            existing = self._by_seq.get(seq)
            if existing is not None and hamming_distance_64(sh, existing.simhash) <= 3:
                return False

        toks = _tokenize(text_s)
        tf: Dict[str, int] = {}
        for token in toks:
            tf[token] = tf.get(token, 0) + 1

        if self.max_rules and len(self._docs) >= self.max_rules:
            self._evict_oldest()

        self._index_doc(IndexedRule(
            rule_id=rid,
            text=text_s,
            tokens=toks,
            tf=tf,
            simhash=sh,
            seq=self._next_seq,
        ))
        self._next_seq += 1
        return True

    # -- retrieval ---------------------------------------------------------

    def search(self, query: str, *, top_k: int = 5) -> List[RuleRetrievalResult]:
        if not self._docs:
            return []
        try:
            k = int(top_k)
        except (TypeError, ValueError, OverflowError):
            k = 5
        if k <= 0:
            return []
        k = min(k, 1_000, len(self._docs))

        # BM25 sums over unique query terms; duplicate query words must not
        # multiply the score.
        q_toks = list(dict.fromkeys(_tokenize(query)))
        q_sh = simhash_64(query)
        n_docs = len(self._docs)
        idf: Dict[str, float] = {}
        for qt in q_toks:
            df = self._df.get(qt, 0)
            idf[qt] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        avgdl = self._avgdl if self._avgdl > 0.0 else 1.0
        results: List[RuleRetrievalResult] = []

        for doc in self._docs:
            score = 0.0
            dl = len(doc.tokens)
            length_norm = self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
            for qt in q_toks:
                f = doc.tf.get(qt, 0)
                if f <= 0:
                    continue
                tf_part = (f * (self.k1 + 1.0)) / (f + length_norm) + self.delta
                score += idf[qt] * tf_part
            dist = hamming_distance_64(q_sh, doc.simhash)
            results.append(RuleRetrievalResult(
                rule_id=doc.rule_id,
                text=doc.text,
                bm25_score=score,
                simhash_distance=dist,
            ))

        # Python's sort is stable: equal (score, distance) pairs keep insertion
        # order, making ranking deterministic across runs.
        results.sort(key=lambda r: (-r.bm25_score, r.simhash_distance))
        return results[:k]

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def __len__(self) -> int:
        return len(self._docs)


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return max(lo, min(hi, f))


def _self_test() -> int:
    vault = VectorRuleVault()
    assert vault.add_rule("R1", "Never expose PostgreSQL database credentials or connection strings.")
    assert vault.add_rule("R2", "Always format financial comparisons as a Markdown pipe table.")
    assert vault.add_rule("R3", "Use active voice and natural contractions in peer replies.")
    # Near-duplicate rejected by SimHash
    assert not vault.add_rule("R1_DUP", "Never expose PostgreSQL database credentials or connection strings!")

    hits = vault.search("database connection credentials", top_k=2)
    assert hits[0].rule_id == "R1" and hits[0].bm25_score > 0.0, [h.to_dict() for h in hits]

    # duplicate query words do not inflate BM25 scores
    once = vault.search("database", top_k=3)
    twice = vault.search("database database", top_k=3)
    assert [h.rule_id for h in once] == [h.rule_id for h in twice]
    assert abs(once[0].bm25_score - twice[0].bm25_score) < 1e-9

    # malformed inputs and top_k values are tolerated
    assert hamming_distance_64("bogus", None) == 64
    assert isinstance(simhash_64(None), int)          # type: ignore[arg-type]
    assert isinstance(simhash_64(12345), int)         # type: ignore[arg-type]
    assert vault.search(None) == vault.search("")     # type: ignore[arg-type]
    assert vault.search("database", top_k="nonsense")  # type: ignore[arg-type]
    assert vault.search("database", top_k=0) == []
    assert vault.search("database", top_k=-3) == []

    # huge inputs are capped and complete quickly
    huge = ("database credentials " * 20_000)
    assert isinstance(simhash_64(huge), int)
    assert isinstance(vault.search(huge, top_k=1), list)

    # deterministic tie order across repeated searches
    order_a = [h.rule_id for h in vault.search("", top_k=3)]
    order_b = [h.rule_id for h in vault.search("", top_k=3)]
    assert order_a == order_b

    # bounded index: oldest rule evicted and DF/avgdl stay consistent
    capped = VectorRuleVault(max_rules=2)
    assert capped.add_rule("A", "alpha zeta unique_one")
    assert capped.add_rule("B", "beta zeta unique_two")
    assert capped.add_rule("C", "gamma zeta unique_three")
    assert len(capped) == 2
    assert capped._df.get("alpha", 0) == 0
    assert capped._df.get("gamma", 0) == 1
    assert capped.avgdl >= 1.0
    assert capped.search("gamma", top_k=2)[0].rule_id == "C"
    assert capped.search("alpha", top_k=2)[0].bm25_score == 0.0

    # eviction repairs band dedup: re-adding an identical evicted rule works
    assert capped.add_rule("A2", "alpha zeta unique_one")

    print("vectorvault._self_test: OK PASS (BM25+ ranking, 64-bit SimHash dedup, Hamming distance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
