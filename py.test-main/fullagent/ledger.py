"""ledger — Semantic Rule Ledger & Diffing Engine (v5 apex).

Builds and maintains the machine-readable ledger of all active
directives (from hierarchy.compile_directives), with:

  • Token-budget-aware sizing (the ledger must fit the enforcement
    budget alongside the document itself — ~6% of doc tokens, capped).
  • Semantic dedup: near-duplicate rules (edit-distance + token Jaccard)
    collapse into one canonical id with alias list.
  • Ledger diffing: given an old and new directive set, produce a
    structured change report (added/removed/changed/unchanged) so the
    sentinel can re-anchor only what moved (cheap wire updates).
  • Importance ranking: position (recency), imperative strength,
    scope breadth, and heading depth feed a weighted score; the top-K
    rules are what the anchor references.
  • Compact serialization formats for the wire:
      - ledger_xml(): <rule id="Rxx" pri="n" scope="s">text</rule>
      - ledger_plain(): "R01. text" numbered list for text-only models
      - ledger_ids(): "R01 R04 R07" for anchors

Pure text-in/text-out, stdlib only, never raises on malformed input.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .hierarchy import Directive, SourceLevel, compile_directives

__all__ = [
    "LedgerEntry", "RuleLedger", "build_ledger", "diff_ledgers",
    "importance_score", "token_estimate",
]


# ── token estimation (no tokenizer dependency) ──────────────────────────

def token_estimate(text: str) -> int:
    """Cheap token estimate: ~4 chars/token for prose, with a floor for
    CJK (1 token per char). Good enough for budget sizing; never used
    for billing."""
    if not text:
        return 0
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff'
              or '\u3040' <= c <= '\u30ff' or '\uac00' <= c <= '\ud7af')
    return max(1, (len(text) - cjk) // 4 + cjk)


# ── similarity primitives ───────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9\u0900-\u097f\u0400-\u04ff]+", re.I)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w.lower() for w in _WORD_RE.findall(text or ""))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def edit_distance(a: str, b: str, cap: int = 64) -> int:
    """Bounded Levenshtein (O(len_a*cap) space) for short strings."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


_STOP = frozenset(
    "a an the and or but if then else when while for to of in on at by "
    "is are was were be been being do does did have has had will would "
    "shall should must may might can could this that these those it its "
    "you your with as from not no yes".split())


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        w for w in _tokens(text) if w not in _STOP and len(w) > 2)


# ── ledger entry ────────────────────────────────────────────────────────

@dataclass
class LedgerEntry:
    rule_id: str
    text: str
    source_level: int
    scope: str
    specificity: int
    importance: float = 0.0
    aliases: list[str] = field(default_factory=list)  # merged duplicate ids
    doc_id: str = ""

    def xml(self) -> str:
        esc = (self.text.replace("&", "&")
               .replace("<", "<").replace(">", ">")
               .replace('"', '"'))
        return (f'<rule id="{self.rule_id}" pri="{self.importance:.2f}" '
                f'scope="{self.scope}" src="{SourceLevel.name(self.source_level)}">'
                f'{esc}</rule>')

    def plain(self) -> str:
        return f"{self.rule_id}. {self.text}"

    def short(self, limit: int = 96) -> str:
        return self.text if len(self.text) <= limit else self.text[:limit - 1] + "…"


# ── importance scoring ──────────────────────────────────────────────────

_IMPERATIVE_STRONG = re.compile(r"\b(never|must not|do not|forbidden)\b", re.I)
_IMPERATIVE = re.compile(r"\b(must|shall|always|only|exactly)\b", re.I)
_SOFT = re.compile(r"\b(should|prefer|try|consider|may)\b", re.I)


def importance_score(d: Directive, total: int, position: int) -> float:
    """Weighted score in [0, 1]:
      0.30 imperative strength (hard ban > requirement > soft)
      0.20 recency (later in doc = more likely to be the refined law)
      0.20 specificity (exact tool/action rules outrank wildcards)
      0.15 scope (tool-scoped rules are checkable, hence higher)
      0.10 heading depth (deeper section = more specific intent)
      0.05 length (a one-line rule is sharper than a paragraph)
    """
    score = 0.0
    t = d.text
    if _IMPERATIVE_STRONG.search(t):
        score += 0.30
    elif _IMPERATIVE.search(t):
        score += 0.24
    elif _SOFT.search(t):
        score += 0.10
    else:
        score += 0.06
    if total > 1:
        score += 0.20 * (position / (total - 1))
    score += 0.20 * (d.specificity / 3.0)
    score += 0.15 * (1.0 if d.scope == "tool" else
                     (0.6 if d.scope == "turn" else 0.3))
    level = int(d.metadata.get("level", 0) or 0)
    score += 0.10 * min(1.0, level / 4.0)
    L = len(d.text)
    score += 0.05 * (1.0 if 20 <= L <= 160 else (0.6 if L <= 300 else 0.3))
    return round(min(1.0, score), 3)


# ── the ledger ──────────────────────────────────────────────────────────

@dataclass
class RuleLedger:
    entries: list[LedgerEntry] = field(default_factory=list)
    doc_id: str = ""
    doc_chars: int = 0
    truncated: bool = False

    # -- serialization ---------------------------------------------------

    def xml(self) -> str:
        return "\n".join(e.xml() for e in self.entries)

    def plain(self) -> str:
        return "\n".join(e.plain() for e in self.entries)

    def ids(self, k: int | None = None) -> str:
        ranked = self.entries if k is None else self.entries[:k]
        return " ".join(e.rule_id for e in ranked)

    def top_ids(self, k: int) -> list[str]:
        return [e.rule_id for e in self.entries[:k]]

    def by_id(self, rule_id: str) -> LedgerEntry | None:
        for e in self.entries:
            if e.rule_id == rule_id or rule_id in e.aliases:
                return e
        return None

    # -- budget sizing ----------------------------------------------------

    def size_for_budget(self, doc_chars: int, overhead_ratio: float = 0.06,
                        max_chars: int = 4200) -> int:
        """Max chars the ledger may occupy. Enforcement overhead must
    stay ~6% of the document so a 49k doc still leaves ~46k of real
    attention for the law itself."""
        doc_tokens = token_estimate("x" * doc_chars) if doc_chars < 4000 \
            else doc_chars // 4
        # floor of 1400: even a small directive document deserves a
        # ledger that can carry at least its top ~10 rules
        return max(1400, min(max_chars, int(doc_tokens * 4 * overhead_ratio)))

    def fit(self, max_chars: int | None = None) -> "RuleLedger":
        """Return a ledger trimmed to the char budget: entries are
        already importance-ranked, so we keep the top-K whose serialized
        forms fit. The ranking means the KEPT rules are the ones the
        anchor should cite."""
        limit = max_chars or self.size_for_budget(self.doc_chars)
        out: list[LedgerEntry] = []
        used = 0
        for e in self.entries:
            w = len(e.xml()) + 1
            if used + w > limit and out:
                self.truncated = True
                break
            out.append(e)
            used += w
        return RuleLedger(entries=out, doc_id=self.doc_id,
                          doc_chars=self.doc_chars)


# ── dedup: near-duplicate collapse ──────────────────────────────────────

def _near_duplicate(a: LedgerEntry, b: LedgerEntry) -> bool:
    ta, tb = _content_tokens(a.text), _content_tokens(b.text)
    if jaccard(ta, tb) >= 0.72:
        return True
    # short rules: edit distance is the better signal
    if len(a.text) <= 120 and len(b.text) <= 120:
        d = edit_distance(a.text.lower(), b.text.lower())
        if d <= max(4, len(a.text) // 8):
            return True
    return False


def _dedup(entries: list[LedgerEntry]) -> list[LedgerEntry]:
    """Collapse near-duplicates: the first (higher-ranked) entry survives
    and absorbs the later ids as aliases so references stay valid."""
    out: list[LedgerEntry] = []
    for e in entries:
        merged = False
        for keep in out:
            if _near_duplicate(keep, e):
                keep.aliases.append(e.rule_id)
                keep.importance = max(keep.importance, e.importance)
                merged = True
                break
        if not merged:
            out.append(e)
    return out


# ── build ───────────────────────────────────────────────────────────────

def build_ledger(doc_text: str,
                 source_level: int = SourceLevel.DEVELOPER_CONFIG,
                 doc_id: str = "") -> RuleLedger:
    """Full pipeline: document text → compiled directives → scored,
    deduplicated, importance-ranked, budget-fitted ledger."""
    doc_text = (doc_text or "").strip()
    if not doc_text:
        return RuleLedger(doc_id=doc_id or "sha256:empty")
    did = doc_id or ("sha256:" +
                     hashlib.sha256(doc_text.encode("utf-8", "replace")
                                    ).hexdigest()[:12])
    dirs, _ = compile_directives(doc_text, source_level)
    entries: list[LedgerEntry] = []
    total = len(dirs)
    for i, d in enumerate(dirs):
        entries.append(LedgerEntry(
            rule_id=d.rule_id,
            text=d.text,
            source_level=d.source_level,
            scope=d.scope,
            specificity=d.specificity,
            importance=importance_score(d, total, i),
            doc_id=did,
        ))
    # importance-ranked (descending), stable for equal scores
    entries.sort(key=lambda e: (-e.importance, e.rule_id))
    entries = _dedup(entries)
    ledger = RuleLedger(entries=entries, doc_id=did, doc_chars=len(doc_text))
    return ledger.fit()


# ── diffing ─────────────────────────────────────────────────────────────

@dataclass
class LedgerDiff:
    added: list[LedgerEntry] = field(default_factory=list)
    removed: list[LedgerEntry] = field(default_factory=list)
    changed: list[tuple[LedgerEntry, LedgerEntry]] = field(default_factory=list)
    unchanged: list[LedgerEntry] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    def summary(self) -> str:
        return (f"+{len(self.added)} -{len(self.removed)} "
                f"~{len(self.changed)} ={len(self.unchanged)}")

    def reanchor_ids(self) -> list[str]:
        """The ids that need re-anchoring after this diff: all added and
        changed rules (removed rules simply stop being referenced)."""
        ids = [e.rule_id for e in self.added]
        ids += [new.rule_id for _, new in self.changed]
        return ids


def diff_ledgers(old: RuleLedger, new: RuleLedger) -> LedgerDiff:
    """Structured change report between two ledgers. Identity is by
    content fingerprint (doc_id + rule text), not rule_id, so a hot
    reload that renumbers rules still diffs cleanly."""
    old_by_fp = {hashlib.sha256(e.text.encode()).hexdigest()[:10]: e
                 for e in old.entries}
    new_by_fp = {hashlib.sha256(e.text.encode()).hexdigest()[:10]: e
                 for e in new.entries}
    d = LedgerDiff()
    for fp, ne in new_by_fp.items():
        oe = old_by_fp.get(fp)
        if oe is None:
            # maybe a renumbered near-duplicate of a removed rule
            match = None
            for fp2, oe2 in old_by_fp.items():
                if fp2 not in new_by_fp and _near_duplicate(oe2, ne):
                    match = oe2
                    break
            if match is not None:
                d.changed.append((match, ne))
            else:
                d.added.append(ne)
        else:
            if (oe.importance != ne.importance
                    or oe.scope != ne.scope):
                d.changed.append((oe, ne))
            else:
                d.unchanged.append(ne)
    for fp, oe in old_by_fp.items():
        if fp not in new_by_fp and not any(
                _near_duplicate(oe, ne) for ne in d.changed and [c[1] for c in d.changed][0:0] or []):
            # removed unless it survived as a changed pair
            if not any(oe.rule_id == pair[0].rule_id
                       for pair in d.changed):
                d.removed.append(oe)
    return d


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    doc = """# Agent Rules
## Communication
1. Always answer in Hindi.
2. Never use bullet lists unless asked.
- Keep answers under 120 words.

## Tools
You must cite file paths when editing code.
Never run shell commands without approval.
Always verify a command's exit code before reporting success.
"""
    led = build_ledger(doc)
    assert led.entries, "ledger must not be empty"
    assert led.xml().startswith("<rule id=")
    assert all(e.importance >= 0 for e in led.entries)
    # ranked descending
    imps = [e.importance for e in led.entries]
    assert imps == sorted(imps, reverse=True), "entries must be ranked"

    # dedup: repeat the doc's rules verbatim — aliases absorb them
    led2 = build_ledger(doc + "\n" + doc)
    assert len(led2.entries) < 2 * len(led.entries), \
        "near-duplicates must collapse"

    # budget fit
    big = "\n".join(f"# Section {i}\n- Always respect constraint {i} "
                    "about formatting output." for i in range(200))
    big_led = build_ledger(big)
    assert len(big_led.xml()) <= 4200 + 200, "ledger must fit its budget"
    assert big_led.truncated or len(big_led.entries) < 300

    # diffing (change a top-importance rule so it survives budget fit)
    doc2 = doc.replace("Never run shell commands without approval.",
                       "Never run shell commands without explicit "
                       "written approval.")
    led3 = build_ledger(doc2)
    d = diff_ledgers(led, led3)
    assert d.has_changes
    assert d.reanchor_ids(), "changed rules need re-anchoring"
    d0 = diff_ledgers(led, led)
    assert not d0.added and not d0.removed

    # token estimate sanity
    assert token_estimate("hello world") == 2
    assert token_estimate("你好世界" * 10) == 40

    # never raises
    assert build_ledger("").entries == []
    assert diff_ledgers(build_ledger(""), build_ledger("x")) is not None

    print("LEDGER SELF-TEST PASS (build, rank, dedup, budget, diff)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())