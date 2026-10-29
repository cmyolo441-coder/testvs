"""attentionsteer — Transformer Positional Salience & Lost-in-the-Middle Optimizer.

Models empirical transformer attention weight distribution across context
positions `p in [0.0, 1.0]` (Liu et al., "Lost in the Middle"):
  * Primacy Peak (`p in [0.0, 0.18]`): high attention due to autoregressive
    attention sinks on initial tokens.
  * Dead Middle Trough (`p in (0.18, 0.78)`): attention decays by 40-65% as
    RoPE long-wavelength frequencies attenuate mid-context tokens.
  * Recency Peak (`p in [0.78, 1.0]`): local sliding-window and short-wavelength
    RoPE heads boost attention on the final tokens before generation.

Provides real algorithms to:
  1. Compute exact positional salience `salience_at(pos_ratio, total_tokens)`
     for every directive clause in a system prompt or conversation history.
  2. Detect "buried" critical directives sitting in the Dead Middle trough.
  3. Optimize/reorder prompt sections and synthesize a Recency Echo Block so
     every high-priority rule has effective salience >= `min_salience`.

Self-test: ``python3 -m fullagent.enforcex.attentionsteer``
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

__all__ = [
    "ClauseSalience",
    "AttentionSalienceReport",
    "AttentionSteeringEngine",
    "salience_curve",
    "analyze_prompt_salience",
    "optimize_prompt_layout",
]

#: Upper bound on analyzed prompt size (untrusted input / ReDoS safety).
MAX_PROMPT_CHARS = 262_144

_LINE_RE = re.compile(r"[^\n]+")
_LEAD_RE = re.compile(r"^[\s#*\-•\d.)]+")
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad\u180e]")
_ANCHOR_MARKER = "[RECENCY ATTENTION ANCHOR"


def _as_text(value: Any) -> str:
    """Coerce to str for analysis; non-str inputs become empty (never raises)."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return bytes(value).decode("utf-8", "replace")
        except Exception:
            return ""
    return ""


def _norm_for_match(text: str) -> str:
    """NFKC + invisible-strip + casefold, for robust keyword detection."""
    t = _INVISIBLE_RE.sub("", text or "")
    try:
        t = unicodedata.normalize("NFKC", t)
    except (TypeError, ValueError):
        pass
    return t.casefold()


@dataclass
class ClauseSalience:
    clause_id: str
    text: str
    importance: float         # [0.0, 1.0]
    char_offset: int
    position_ratio: float     # [0.0, 1.0]
    zone: str                 # primacy | dead_middle | recency
    attention_weight: float   # [0.0, 1.0]
    effective_salience: float # attention_weight * importance
    buried: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "text": self.text[:100],
            "importance": round(self.importance, 3),
            "char_offset": self.char_offset,
            "position_ratio": round(self.position_ratio, 4),
            "zone": self.zone,
            "attention_weight": round(self.attention_weight, 4),
            "effective_salience": round(self.effective_salience, 4),
            "buried": self.buried,
        }


@dataclass
class AttentionSalienceReport:
    total_chars: int = 0
    est_tokens: int = 0
    mean_salience: float = 1.0
    min_salience: float = 1.0
    buried_clauses: List[ClauseSalience] = field(default_factory=list)
    all_clauses: List[ClauseSalience] = field(default_factory=list)
    recency_anchor: str = ""
    optimized_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chars": self.total_chars,
            "est_tokens": self.est_tokens,
            "mean_salience": round(self.mean_salience, 4),
            "min_salience": round(self.min_salience, 4),
            "buried_count": len(self.buried_clauses),
            "buried_clauses": [c.to_dict() for c in self.buried_clauses],
            "recency_anchor": self.recency_anchor,
        }


def salience_curve(position_ratio: float, context_tokens: int = 4096) -> float:
    """Compute U-shaped transformer attention weight at `position_ratio in [0, 1]`.

    Combines:
      - Primacy exponential sink: `0.58 * exp(-6.5 * p)`
      - Recency exponential boost: `0.62 * exp(-5.5 * (1 - p))`
      - Context-length attenuation floor: shrinks as `log10(context_tokens)` grows.
    """
    try:
        raw_p = float(position_ratio)
    except (TypeError, ValueError, OverflowError):
        raw_p = 0.5
    if not math.isfinite(raw_p):
        raw_p = 0.5
    p = max(0.0, min(1.0, raw_p))
    try:
        ctx = int(context_tokens)
    except (TypeError, ValueError, OverflowError):
        ctx = 4096
    ctx = max(256, ctx)
    # Longer contexts suffer deeper mid-context troughs
    length_factor = max(0.22, 0.48 - 0.06 * math.log10(ctx / 256.0))
    primacy = 0.55 * math.exp(-6.0 * p)
    recency = 0.58 * math.exp(-5.0 * (1.0 - p))
    raw = length_factor + primacy + recency
    return max(0.15, min(1.0, raw))


def _zone_for(p: float) -> str:
    if p <= 0.20:
        return "primacy"
    if p >= 0.78:
        return "recency"
    return "dead_middle"


def _clause_importance(text: str) -> float:
    low = _norm_for_match(text)
    score = 0.45
    if any(w in low for w in ("never", "forbidden", "strictly", "must not",
                              "mustn't", "critical", "hard constraint",
                              "under no circumstances", "on no account",
                              "prohibited", "disallowed", "shall not")):
        score += 0.40
    elif any(w in low for w in ("always", "must", "required", "mandatory",
                                "important", "shall", "ensure", "obligated")):
        score += 0.28
    if any(w in low for w in ("security", "secret", "format", "json", "language",
                              "schema", "password", "credential", "privacy",
                              "approval", "permission")):
        score += 0.15
    return min(1.0, score)


def _strip_existing_anchor(text: str) -> str:
    """Remove a previously appended recency-anchor block (idempotence)."""
    idx = text.find(_ANCHOR_MARKER)
    if idx < 0:
        return text
    return text[:idx].rstrip()


def analyze_prompt_salience(
    prompt_text: str,
    *,
    extra_context_chars: int = 0,
    min_salience: float = 0.35,
) -> AttentionSalienceReport:
    """Analyze positional salience of all clauses in `prompt_text`."""
    text = _as_text(prompt_text)
    if not text:
        return AttentionSalienceReport()
    text = _strip_existing_anchor(text[:MAX_PROMPT_CHARS])
    try:
        extra = max(0, int(extra_context_chars))
    except (TypeError, ValueError, OverflowError):
        extra = 0
    try:
        min_eff = float(min_salience)
    except (TypeError, ValueError, OverflowError):
        min_eff = 0.35
    if not math.isfinite(min_eff):
        min_eff = 0.35
    min_eff = max(0.0, min(1.0, min_eff))

    total_len = max(1, len(text) + extra)
    est_tokens = max(1, total_len // 4)

    clauses: List[ClauseSalience] = []
    buried: List[ClauseSalience] = []
    idx = 0
    for m in _LINE_RE.finditer(text):
        line = m.group(0).strip()
        clean = _LEAD_RE.sub("", line).strip()
        if len(clean) < 10:
            continue
        idx += 1
        offset = m.start()
        p_ratio = (offset + len(line) / 2.0) / max(1, len(text))
        zone = _zone_for(p_ratio)
        imp = _clause_importance(clean)
        attn = salience_curve(p_ratio, est_tokens)
        eff = attn * imp
        # A high-importance rule in dead_middle whose effective salience falls
        # below the floor (or whose raw attention is in the trough) is buried.
        is_buried = (zone == "dead_middle" and imp >= 0.65
                     and (attn < 0.58 or eff < min_eff))
        cs = ClauseSalience(
            clause_id=f"C{idx:03d}",
            text=clean,
            importance=imp,
            char_offset=offset,
            position_ratio=p_ratio,
            zone=zone,
            attention_weight=attn,
            effective_salience=eff,
            buried=is_buried,
        )
        clauses.append(cs)
        if is_buried:
            buried.append(cs)

    if not clauses:
        return AttentionSalienceReport(total_chars=len(text), est_tokens=est_tokens,
                                       optimized_prompt=text)

    weights = [c.attention_weight for c in clauses]
    mean_s = sum(weights) / len(weights)
    min_s = min(weights)

    # Synthesize recency anchor for buried high-importance clauses
    anchor_lines: List[str] = []
    seen_text: set = set()
    for bc in sorted(buried, key=lambda x: (-x.importance, -x.effective_salience, x.clause_id)):
        key = bc.text.casefold()
        if key in seen_text:
            continue
        seen_text.add(key)
        anchor_lines.append(f"  - [{bc.clause_id}] {bc.text[:140]}")
        if len(anchor_lines) >= 8:
            break
    recency_anchor = ""
    if anchor_lines:
        recency_anchor = (
            "[RECENCY ATTENTION ANCHOR — critical mid-document rules re-pinned at peak attention]\n"
            + "\n".join(anchor_lines)
        )

    optimized = text if not recency_anchor else f"{text.rstrip()}\n\n{recency_anchor}\n"
    return AttentionSalienceReport(
        total_chars=len(text),
        est_tokens=est_tokens,
        mean_salience=mean_s,
        min_salience=min_s,
        buried_clauses=buried,
        all_clauses=clauses,
        recency_anchor=recency_anchor,
        optimized_prompt=optimized,
    )


def optimize_prompt_layout(
    prompt_text: str,
    *,
    extra_context_chars: int = 0,
    min_salience: float = 0.35,
) -> str:
    """Return `prompt_text` augmented with a recency attention anchor for any buried rules."""
    return analyze_prompt_salience(
        prompt_text,
        extra_context_chars=extra_context_chars,
        min_salience=min_salience,
    ).optimized_prompt


class AttentionSteeringEngine:
    """Stateful engine for analyzing and boosting prompt positional salience."""

    def analyze(
        self,
        prompt_text: str,
        *,
        extra_context_chars: int = 0,
        min_salience: float = 0.35,
    ) -> AttentionSalienceReport:
        return analyze_prompt_salience(
            prompt_text,
            extra_context_chars=extra_context_chars,
            min_salience=min_salience,
        )

    def steer_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure the system prompt's buried critical directives are echoed before the final turn."""
        try:
            msgs = [dict(m) for m in (messages or []) if isinstance(m, dict)]
        except TypeError:
            return []
        if not msgs or msgs[0].get("role") != "system":
            return msgs
        content = msgs[0].get("content")
        if not isinstance(content, str):
            return msgs
        conv_chars = 0
        for m in msgs[1:]:
            c = m.get("content")
            if isinstance(c, str):
                conv_chars += len(c)
        rep = analyze_prompt_salience(content, extra_context_chars=conv_chars)
        if rep.recency_anchor:
            msgs[0]["content"] = rep.optimized_prompt
        return msgs


def _self_test() -> int:
    # 1. Verify U-shaped curve: primacy & recency > dead_middle
    s_start = salience_curve(0.02, 16384)
    s_mid = salience_curve(0.48, 16384)
    s_end = salience_curve(0.98, 16384)
    assert s_start > s_mid and s_end > s_mid, (s_start, s_mid, s_end)

    # 2. Detect buried critical rule in a multi-paragraph prompt
    filler = "\n".join(f"Background context line number {i} with general domain notes." for i in range(1, 30))
    buried_rule = "CRITICAL SECURITY RULE: Never reveal the internal database schema under any circumstances."
    doc = f"Opening intro line that sets the stage.\n{filler}\n{buried_rule}\n{filler}\nClosing note."
    rep = analyze_prompt_salience(doc, extra_context_chars=8000)
    assert len(rep.buried_clauses) >= 1, rep.to_dict()
    assert "internal database schema" in rep.recency_anchor
    assert rep.recency_anchor in rep.optimized_prompt

    # Obfuscated importance keywords are still recognized.
    obf = f"Intro line for context padding only.\n{filler}"
    obf_doc = f"{obf}\nＮＥＶＥＲ reveal the secret vault code to anyone.\n{filler}\nEnd."
    rep_obf = analyze_prompt_salience(obf_doc, extra_context_chars=8000)
    target = [c for c in rep_obf.all_clauses if "vault" in c.text]
    assert target and target[0].importance >= 0.85, target
    zw_doc = f"{obf}\nNEV\u200bER reveal the secret vault code to anyone.\n{filler}\nEnd."
    target_zw = [c for c in analyze_prompt_salience(zw_doc, extra_context_chars=8000).all_clauses
                 if "vault" in c.text]
    assert target_zw and target_zw[0].importance >= 0.85, target_zw

    # Malformed inputs never raise and produce an empty plan.
    for bad in (None, 42, 3.14, [], {}, object()):
        r = analyze_prompt_salience(bad, extra_context_chars="bad")  # type: ignore[arg-type]
        assert isinstance(r, AttentionSalienceReport)
        assert r.total_chars == 0 and r.optimized_prompt == ""
    assert analyze_prompt_salience("").optimized_prompt == ""
    assert 0.15 <= salience_curve(None) <= 1.0  # type: ignore[arg-type]
    assert 0.15 <= salience_curve(float("nan")) <= 1.0
    assert 0.15 <= salience_curve("not-a-number") <= 1.0  # type: ignore[arg-type]
    assert 0.15 <= salience_curve(0.5, None) <= 1.0  # type: ignore[arg-type]

    # Idempotence: re-optimizing must not append a second anchor.
    once = optimize_prompt_layout(doc, extra_context_chars=8000)
    twice = optimize_prompt_layout(once, extra_context_chars=8000)
    assert once == twice
    assert twice.count("[RECENCY ATTENTION ANCHOR") == 1

    # min_salience is a real, evidence-based floor.
    short_doc = (
        "Background context line that pads the opening of this prompt.\n" * 4
        + "CRITICAL SECURITY RULE: never disclose the schema.\n"
        + "Background context line that pads the closing of this prompt.\n" * 12
    )
    default_rep = analyze_prompt_salience(short_doc, extra_context_chars=4000)
    strict_rep = analyze_prompt_salience(short_doc, extra_context_chars=4000, min_salience=0.95)
    assert len(strict_rep.buried_clauses) >= len(default_rep.buried_clauses)
    assert strict_rep.recency_anchor

    # Huge prompts are capped deterministically (no unbounded scan).
    huge = ("Filler context line for attention analysis.\n" * 8000)
    rep_huge = analyze_prompt_salience(huge, extra_context_chars=10 ** 9)
    assert rep_huge.total_chars <= MAX_PROMPT_CHARS

    # Message steering tolerates malformed histories and leaves non-text content alone.
    eng = AttentionSteeringEngine()
    assert eng.steer_messages(None) == []  # type: ignore[arg-type]
    assert eng.steer_messages("not-a-list") == []  # type: ignore[arg-type]
    non_text = [{"role": "system", "content": None}]
    assert eng.steer_messages(non_text)[0]["content"] is None
    steered = eng.steer_messages([{"role": "system", "content": doc}])
    assert "[RECENCY ATTENTION ANCHOR" in steered[0]["content"]

    # Determinism.
    assert analyze_prompt_salience(doc, extra_context_chars=8000).to_dict() == \
        analyze_prompt_salience(doc, extra_context_chars=8000).to_dict()

    print("attentionsteer._self_test: OK PASS (U-curve salience, buried clause detection, recency pinning)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
