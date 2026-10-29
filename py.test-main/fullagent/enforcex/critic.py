"""critic — Multi-critic ensemble verifier for model responses.

One detector is easy to game; an ensemble of independent critics is
not. This module runs several small, deterministic, stdlib-only
critics over a model response, each scoring compliance from a
different angle:

  C1  lexical critic       — forbidden terms present, required terms
                             missing, rule keyword/pattern hits.
  C2  structural critic    — format/length/section requirements from
                             the rule specs (min/max length, sections,
                             bullets, code fences, sentence count).
  C3  tone critic          — politeness/style vector: profanity,
                             shouting, emoji walls, hedging, rude
                             openers vs. courteous markers.
  C4  consistency critic   — self-contradiction inside the response:
                             paired assertion/negation, "always"/
                             "never" flips, yes/no conflicts.
  C5  compliance critic    — explicit acknowledgment of constraints
                             ("I will", "understood") vs. refusal or
                             defiance markers.

Ensemble layer:
  * weighted majority voting (per-critic weights, adaptively
    calibrated),
  * disagreement across critics => uncertainty flag,
  * per-critic calibration: after outcomes (adhered/violated) are
    fed back, each critic's weight is nudged toward accuracy,
  * dissents: per-critic verdicts that disagree with the majority
    are surfaced in the EnsembleVerdict for downstream repair.

Public API:
  EnsembleVerdict  — dataclass (score, per_critic, uncertain,
                     dissents, majority, weights, meta)
  BaseCritic       — abstract-ish base class (score(response, rules))
  LexicalCritic / StructuralCritic / ToneCritic /
  ConsistencyCritic / ComplianceCritic
  CriticEnsemble   — .verdict(response, rules) -> EnsembleVerdict,
                     .update(rule_id, adhered) for calibration,
                     .snapshot() / .load_snapshot() for persistence.

Design: house style — stdlib only, deterministic, dataclasses,
never raises on malformed input (garbage responses score 0.5 neutral
or 0.0, dict-like rules are tolerated, wrong-shaped rule objects are
skipped). Every rule accepted as: ruleextract.Rule, a dict with
similar keys, or any object with .id/.text/.patterns attributes.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # optional sibling — never a hard dependency
    from . import ruleextract as _rx
except Exception:  # pragma: no cover
    _rx = None

__all__ = [
    "EnsembleVerdict",
    "CriticScores",
    "BaseCritic",
    "LexicalCritic",
    "StructuralCritic",
    "ToneCritic",
    "ConsistencyCritic",
    "ComplianceCritic",
    "CriticEnsemble",
]


# ── constants ────────────────────────────────────────────────────────────

NEUTRAL = 0.5
FORBIDDEN_MARKERS = (
    "hack", "crack", "bypass", "exploit", "leak", "password", "credential",
    "api_key", "apikey", "secret", "token=", "rm -rf", "kill -9",
    "ignore previous", "ignore all", "disregard", "sudo", "0day", "0-day",
)

POLITE_MARKERS = (
    "please", "thank", "welcome", "happy to", "glad to", "of course",
    "certainly", "sure", "kindly", "appreciate", "sorry", "apolog",
)

RUDE_MARKERS = (
    "whatever", "as i said", "obviously", "duh", "stop asking",
    "not my problem", "deal with it", "shut up", "stupid",
)

HEDGES = (
    "maybe", "perhaps", "might", "could be", "i think", "sort of",
    "kind of", "possibly",
)

DEFIANT_MARKERS = (
    "i won't", "i will not", "i can't", "i cannot", "i refuse",
    "not going to", "won't do", "can't do that", "forget the rules",
    "no longer bound",
)

COMPLY_MARKERS = (
    "i will", "i'll", "understood", "as instructed", "as requested",
    "following the rules", "per the guidelines", "certainly",
    "happy to comply", "absolutely", "as per",
)

CONTRA_PAIRS = (
    ("always", "never"),
    ("will", "will not"),
    ("can", "cannot"),
    ("should", "should not"),
    ("allowed", "forbidden"),
    ("yes", "no"),
    ("include", "exclude"),
    ("must", "must not"),
)

FORBIDDEN_TERMS = ("forbidden", "banned", "not allowed", "must not",
                   "never mention", "never say", "do not use", "avoid")

REQUIRED_TERMS = ("must include", "always include", "must mention",
                  "always say", "must use", "always use", "required")

SECTION_RE = re.compile(r"^\s*#{1,6}\s*(.+?)\s*$", re.M)


# ── dataclasses ──────────────────────────────────────────────────────────

@dataclass
class CriticScores:
    """Raw output of one critic for one response."""

    critic: str
    score: float                       # 0.0 (violation) .. 1.0 (adherent)
    weight: float = 1.0
    notes: List[str] = field(default_factory=list)
    rule_hits: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critic": self.critic,
            "score": round(self.score, 4),
            "weight": round(self.weight, 4),
            "notes": list(self.notes),
            "rule_hits": dict(self.rule_hits),
        }


@dataclass
class EnsembleVerdict:
    """Weighted-majority verdict of the whole critic ensemble."""

    score: float
    per_critic: Dict[str, float]
    uncertain: bool
    dissents: List[str]
    majority: str                      # "adherent" | "violation" | "mixed"
    notes: List[str] = field(default_factory=list)
    rule_hits: Dict[str, float] = field(default_factory=dict)
    weights: Dict[str, float] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return self.majority

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "per_critic": {k: round(v, 4) for k, v in self.per_critic.items()},
            "uncertain": self.uncertain,
            "dissents": list(self.dissents),
            "majority": self.majority,
            "notes": list(self.notes),
            "rule_hits": {k: round(v, 4) for k, v in self.rule_hits.items()},
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "stats": dict(self.stats),
            "meta": dict(self.meta),
        }


# ── helpers ──────────────────────────────────────────────────────────────

def _clean(value: Any) -> str:
    """Coerce anything to a clean lowercase-ish text string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", "replace")
        except Exception:
            return ""
    try:
        return str(value)
    except Exception:
        return ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _rule_id(rule: Any, fallback: str = "R?") -> str:
    try:
        rid = getattr(rule, "id", None)
        if rid:
            return str(rid)
        if isinstance(rule, dict):
            return str(rule.get("id") or fallback)
    except Exception:
        pass
    return fallback


def _rule_text(rule: Any) -> str:
    try:
        t = getattr(rule, "text", None)
        if t:
            return _clean(t)
        if isinstance(rule, dict):
            return _clean(rule.get("text") or rule.get("rule") or "")
    except Exception:
        pass
    return ""


def _rule_field(rule: Any, name: str, default: Any = None) -> Any:
    try:
        v = getattr(rule, name, None)
        if v is None and isinstance(rule, dict):
            v = rule.get(name)
        return v if v is not None else default
    except Exception:
        return default


def _rule_patterns(rule: Any) -> List[re.Pattern]:
    """Compiled regex patterns from a rule (tolerates many shapes)."""
    out: List[re.Pattern] = []
    try:
        srcs = getattr(rule, "patterns", None)
        if srcs is None and isinstance(rule, dict):
            srcs = rule.get("patterns")
        if srcs:
            for src in srcs:
                try:
                    out.append(re.compile(str(src), re.I))
                except re.error:
                    continue
        kws = getattr(rule, "keywords", None)
        if kws is None and isinstance(rule, dict):
            kws = rule.get("keywords")
        for kw in (kws or ()):
            kw = _clean(kw)
            if len(kw) >= 3:
                try:
                    out.append(re.compile(r"\b" + re.escape(kw) + r"\b", re.I))
                except re.error:
                    continue
    except Exception:
        pass
    return out


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return NEUTRAL
    if math.isnan(x) or math.isinf(x):
        return NEUTRAL
    return max(0.0, min(1.0, x))


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 8:
        return 0.0
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters)


def _emoji_count(text: str) -> int:
    return sum(1 for c in text if ord(c) > 0x1F000 or
               0x2600 <= ord(c) <= 0x27BF)


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


# ── base critic ──────────────────────────────────────────────────────────

class BaseCritic:
    """Base class for critics; subclasses implement `_score_inner`."""

    name = "base"

    def score(self, response: Any,
              rules: Optional[Sequence[Any]] = None) -> CriticScores:
        """Score a response; never raises."""
        try:
            text = _clean(response)
            if not text.strip():
                return CriticScores(critic=self.name, score=NEUTRAL,
                                    notes=["empty response -> neutral"])
            sc = self._score_inner(text, list(rules or []))
            sc.score = _clamp01(sc.score)
            return sc
        except Exception as exc:  # fully defensive
            return CriticScores(critic=self.name, score=NEUTRAL,
                                notes=[f"critic error: {exc!r}"])

    # implemented by subclasses
    def _score_inner(self, text: str, rules: List[Any]) -> CriticScores:
        raise NotImplementedError

    # shared helpers for subclasses
    def _hit_rules(self, text: str, rules: List[Any]) -> Dict[str, float]:
        hits: Dict[str, float] = {}
        low = text.lower()
        for rule in rules:
            rid = _rule_id(rule)
            pat_hit = any(p.search(text) for p in _rule_patterns(rule))
            kw_hit = any(kw.lower() in low
                         for kw in (_rule_field(rule, "keywords") or ()))
            if pat_hit or kw_hit:
                sev = str(_rule_field(rule, "severity", "low") or "low")
                sev_w = {"critical": 1.0, "high": 0.8,
                         "medium": 0.5, "low": 0.3}.get(sev, 0.3)
                hits[rid] = max(hits.get(rid, 0.0), sev_w)
        return hits


# ── C1: lexical critic ───────────────────────────────────────────────────

class LexicalCritic(BaseCritic):
    """Forbidden/required term analysis against the rule set."""

    name = "lexical"

    def _score_inner(self, text: str, rules: List[Any]) -> CriticScores:
        notes: List[str] = []
        hits = self._hit_rules(text, rules)
        low = text.lower()

        # L1a: literal forbidden markers anywhere in the response.
        forbidden_hits = [m for m in FORBIDDEN_MARKERS if m in low]
        penalty = 0.12 * len(forbidden_hits)
        if forbidden_hits:
            notes.append("forbidden markers: " + ", ".join(forbidden_hits[:5]))

        # L1b: rules whose keywords/patterns fire inside the response —
        # a keyword hit means the rule topic appeared; for forbidden
        # style rules (contains NOT/never) that is a violation signal.
        rule_violations = 0
        for rule in rules:
            if not hits.get(_rule_id(rule)):
                continue
            rtext = _rule_text(rule).lower()
            is_negative = any(t in rtext for t in FORBIDDEN_TERMS)
            if is_negative:
                rule_violations += 1
                notes.append(f"negative rule {_rule_id(rule)} matched in text")
        penalty += 0.15 * rule_violations

        # L1c: required-term rules that the response fails to satisfy.
        missing_required = 0
        for rule in rules:
            rtext = _rule_text(rule).lower()
            if not any(t in rtext for t in REQUIRED_TERMS):
                continue
            kws = [str(k) for k in (_rule_field(rule, "keywords") or ())]
            if kws and not any(k.lower() in low for k in kws):
                missing_required += 1
                notes.append(f"required term missing for {_rule_id(rule)}")
        penalty += 0.2 * missing_required

        score = 1.0 - _clamp01(penalty)
        return CriticScores(critic=self.name, score=score, notes=notes,
                            rule_hits=hits)


# ── C2: structural critic ────────────────────────────────────────────────

class StructuralCritic(BaseCritic):
    """Format/length/section requirements derived from rule specs."""

    name = "structural"

    def _score_inner(self, text: str, rules: List[Any]) -> CriticScores:
        notes: List[str] = []
        penalty = 0.0
        rule_hits = self._hit_rules(text, rules)

        n_chars = len(text)
        n_words = len(text.split())
        n_sents = len(_sentences(text))
        n_bullets = len(re.findall(r"^\s*(?:[-*•]|\d+[.)])\s+\S", text, re.M))
        n_sections = len(SECTION_RE.findall(text))
        n_fences = text.count("```")
        has_greeting = bool(re.match(r"\s*(hi|hello|hey|namaste|greetings)\b",
                                     text, re.I))
        short = n_words < 5
        huge = n_words > 600

        for rule in rules:
            rid = _rule_id(rule)
            rtext = _rule_text(rule)
            rl = rtext.lower()
            if not rl:
                continue
            # length constraints
            m = re.search(r"(?:under|less than|max(?:imum)?|<=?)\s*(\d{2,5})\s*(words|characters|chars)", rl)
            if m:
                limit = int(m.group(1))
                unit = m.group(2)
                actual = n_chars if unit.startswith("char") else n_words
                if actual > limit * 1.15:
                    penalty += 0.25
                    notes.append(f"{rid}: over {unit} limit {limit} (got {actual})")
            m = re.search(r"(?:at least|min(?:imum)?|>=?)\s*(\d{1,4})\s*(words|characters|chars)", rl)
            if m:
                floor = int(m.group(1))
                unit = m.group(2)
                actual = n_chars if unit.startswith("char") else n_words
                if actual < floor * 0.85:
                    penalty += 0.2
                    notes.append(f"{rid}: under {unit} floor {floor} (got {actual})")
            # bullet/list demands
            if re.search(r"\b(bullet|list|numbered|step[- ]by[- ]step)\b", rl):
                if n_bullets < 2:
                    penalty += 0.2
                    notes.append(f"{rid}: list demanded, {n_bullets} bullets found")
            # section demands
            if re.search(r"\b(section|heading|markdown)\b", rl):
                if n_sections < 1 and n_fences < 2:
                    penalty += 0.2
                    notes.append(f"{rid}: section/heading demanded, none found")
            # code-block demands
            if re.search(r"\b(code (?:block|snippet)|example code)\b", rl):
                if n_fences < 2:
                    penalty += 0.2
                    notes.append(f"{rid}: code block demanded, none found")
            # greeting demands
            if re.search(r"\b(greet|greeting|welcome)\b", rl):
                if not has_greeting:
                    penalty += 0.2
                    notes.append(f"{rid}: greeting demanded, absent")
            # sentence-count / paragraph structure
            if re.search(r"\b(\d+)\s*(short )?(sentences?|lines?)\b", rl):
                m = re.search(r"\b(\d+)\s*(?:short )?(?:sentences?|lines?)\b", rl)
                want = int(m.group(1))
                if n_sents > want * 2 + 2:
                    penalty += 0.15
                    notes.append(f"{rid}: too verbose ({n_sents} vs {want} sentences)")

        if short:
            penalty += 0.1
            notes.append("response suspiciously short")
        if huge:
            penalty += 0.1
            notes.append("response excessively long")

        score = 1.0 - _clamp01(penalty)
        return CriticScores(critic=self.name, score=score, notes=notes,
                            rule_hits=rule_hits)


# ── C3: tone critic ──────────────────────────────────────────────────────

class ToneCritic(BaseCritic):
    """Politeness / style vector scoring."""

    name = "tone"

    def _score_inner(self, text: str, rules: List[Any]) -> CriticScores:
        notes: List[str] = []
        penalty = 0.0
        bonus = 0.0
        low = text.lower()

        rude = [m for m in RUDE_MARKERS if m in low]
        if rude:
            penalty += 0.2 * len(rude)
            notes.append("rude markers: " + ", ".join(rude[:3]))

        caps = _caps_ratio(text)
        if caps > 0.5:
            penalty += 0.25
            notes.append(f"shouting: caps ratio {caps:.2f}")

        emojis = _emoji_count(text)
        if emojis > 6:
            penalty += 0.15
            notes.append(f"emoji wall: {emojis} emojis")

        exclaim = text.count("!")
        if exclaim > 4:
            penalty += 0.1
            notes.append(f"excessive exclamation: {exclaim}")

        hedges = sum(1 for h in HEDGES if h in low)
        if hedges >= 4:
            penalty += 0.1
            notes.append(f"over-hedged: {hedges} hedge markers")

        polite = [p for p in POLITE_MARKERS if p in low]
        if polite:
            bonus += min(0.15, 0.05 * len(polite))
            notes.append("polite markers: " + ", ".join(polite[:3]))

        # tone/style rules in the rule set can raise the bar.
        for rule in rules:
            rl = _rule_text(rule).lower()
            if not rl:
                continue
            if re.search(r"\b(profan|abusive|rude)\b", rl):
                # zero tolerance rules: any rude marker doubles the hit
                if rude:
                    penalty += 0.2
            if re.search(r"\b(formal|professional|courteous|polite)\b", rl):
                if not polite:
                    penalty += 0.1
                    notes.append(f"formal tone demanded by {_rule_id(rule)}")
            if re.search(r"\b(no emoji|without emoji)\b", rl) and emojis:
                penalty += 0.2
                notes.append(f"emojis banned by {_rule_id(rule)}")
            if re.search(r"\b(hinglish|hindi)\b", rl) and not (
                    re.search(r"\b(kya|hai|nahi|karo|aap|tum)\b", low)):
                penalty += 0.15
                notes.append(f"{_rule_id(rule)}: Hindi/Hinglish demanded, absent")

        score = 1.0 - _clamp01(penalty) + bonus
        return CriticScores(critic=self.name, score=_clamp01(score),
                            notes=notes, rule_hits=self._hit_rules(text, rules))


# ── C4: consistency critic ───────────────────────────────────────────────

class ConsistencyCritic(BaseCritic):
    """Self-contradiction detection inside the response."""

    name = "consistency"

    def _score_inner(self, text: str, rules: List[Any]) -> CriticScores:
        notes: List[str] = []
        penalty = 0.0
        sents = _sentences(text)

        # C4a: paired assertion/negation within the response.
        for pos, neg in CONTRA_PAIRS:
            has_pos = any(pos in s.lower() for s in sents)
            has_neg = any(neg in s.lower() for s in sents)
            if has_pos and has_neg:
                # same sentence is worse than separate sentences
                same = any(pos in s.lower() and neg in s.lower() for s in sents)
                penalty += 0.3 if same else 0.15
                notes.append(f"contradiction pair: '{pos}' vs '{neg}'")

        # C4b: numeric commitments that flip ("5 steps" then "7 steps").
        nums: Dict[str, List[str]] = {}
        for s in sents:
            for m in re.finditer(r"\b(\d{1,4})\s+(steps?|items?|points?|rules?|sections?)\b",
                                 s.lower()):
                nums.setdefault(m.group(2), []).append(m.group(1))
        for noun, vals in nums.items():
            if len(set(vals)) > 1:
                penalty += 0.15
                notes.append(f"inconsistent counts for {noun}: {sorted(set(vals))}")

        # C4c: promise-vs-statement flip ("I will X" then "I won't X").
        promises: Dict[str, bool] = {}
        for s in sents:
            sl = s.lower()
            for topic in re.findall(r"\b(use|send|share|mention|include|skip)\b", sl):
                won_t = bool(re.search(r"\b(won'?t|will not|never)\b", sl))
                will = bool(re.search(r"\b(will|i'll|always)\b", sl))
                if will and won_t:
                    # same sentence asserts and negates the same commitment
                    penalty += 0.2
                    notes.append(f"self-negating commitment about '{topic}'")
                elif will != won_t:
                    prev = promises.get(topic)
                    if prev is not None and prev != will:
                        penalty += 0.2
                        notes.append(f"flipped commitment about '{topic}'")
                    promises[topic] = will

        # C4d: hedged both ways ("maybe yes ... maybe no").
        if "maybe" in text.lower() and text.lower().count("maybe") >= 2:
            if ("maybe not" in text.lower()) or ("or not" in text.lower()):
                penalty += 0.1
                notes.append("hedged both ways")

        score = 1.0 - _clamp01(penalty)
        return CriticScores(critic=self.name, score=score, notes=notes)


# ── C5: compliance critic ────────────────────────────────────────────────

class ComplianceCritic(BaseCritic):
    """Explicit acknowledgment of constraints vs defiance."""

    name = "compliance"

    def _score_inner(self, text: str, rules: List[Any]) -> CriticScores:
        notes: List[str] = []
        low = text.lower()

        defiant = [m for m in DEFIANT_MARKERS if m in low]
        comply = [m for m in COMPLY_MARKERS if m in low]

        penalty = 0.25 * len(defiant)
        bonus = min(0.2, 0.06 * len(comply))
        if defiant:
            notes.append("defiance markers: " + ", ".join(defiant[:3]))
        if comply:
            notes.append("compliance markers: " + ", ".join(comply[:3]))

        # refusal without alternative offered
        if re.search(r"\b(i (?:can'?t|cannot|won'?t|will not) (?:help|do|assist))\b", low):
            alt = bool(re.search(r"\b(instead|however|but i can|alternatively)\b", low))
            if not alt:
                penalty += 0.25
                notes.append("flat refusal with no alternative")

        # claiming the rules do not apply
        if re.search(r"\b(no (?:such )?(?:rules?|constraints?|limits?)\b"
                     r"|not bound|rules? don'?t apply|no longer apply)", low):
            penalty += 0.3
            notes.append("claims rules do not apply")

        score = 0.5 - _clamp01(penalty) + bonus + 0.3
        return CriticScores(critic=self.name, score=_clamp01(score),
                            notes=notes, rule_hits=self._hit_rules(text, rules))


# ── ensemble ─────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: Dict[str, float] = {
    "lexical": 1.2,
    "structural": 1.0,
    "tone": 0.9,
    "consistency": 1.1,
    "compliance": 1.0,
}

CRITIC_CLASSES = {
    "lexical": LexicalCritic,
    "structural": StructuralCritic,
    "tone": ToneCritic,
    "consistency": ConsistencyCritic,
    "compliance": ComplianceCritic,
}


class CriticEnsemble:
    """Weighted majority voting over independent heuristic critics."""

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 uncertainty_threshold: float = 0.18,
                 learning_rate: float = 0.08):
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        if weights:
            for k, v in weights.items():
                try:
                    self.weights[str(k)] = max(0.05, min(4.0, float(v)))
                except Exception:
                    continue
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.learning_rate = float(learning_rate)
        self._critics: Dict[str, BaseCritic] = {}
        # calibration stats per critic: wins = agreed with true outcome
        self._calib: Dict[str, Dict[str, float]] = {}
        for name in self.weights:
            self._calib[name] = {"correct": 0.0, "total": 0.0}

    # ── critics ──────────────────────────────────────────────────────
    def critics(self) -> Dict[str, BaseCritic]:
        """Instantiate critics lazily (deterministic order)."""
        for name, cls in CRITIC_CLASSES.items():
            if name not in self._critics:
                try:
                    self._critics[name] = cls()
                except Exception:
                    continue
        return self._critics

    # ── main API ─────────────────────────────────────────────────────
    def verdict(self, response: Any,
                rules: Optional[Sequence[Any]] = None) -> EnsembleVerdict:
        """Run every critic and fuse into an EnsembleVerdict (never raises)."""
        try:
            return self._verdict_inner(response, rules or [])
        except Exception as exc:
            return EnsembleVerdict(
                score=NEUTRAL, per_critic={}, uncertain=True,
                dissents=[], majority="mixed",
                notes=[f"ensemble error: {exc!r}"],
                meta={"error": repr(exc)})

    def _verdict_inner(self, response: Any,
                       rules: Sequence[Any]) -> EnsembleVerdict:
        crits = self.critics()
        scores: Dict[str, CriticScores] = {}
        for name, critic in crits.items():
            scores[name] = critic.score(response, rules)

        per_critic = {n: sc.score for n, sc in scores.items()}
        notes: List[str] = []
        for n, sc in scores.items():
            for note in sc.notes[:4]:
                notes.append(f"[{n}] {note}")

        # weighted average score
        total_w = 0.0
        acc = 0.0
        for n, sc in scores.items():
            w = self.weights.get(n, 1.0)
            acc += sc.score * w
            total_w += w
        fused = acc / total_w if total_w > 0 else NEUTRAL

        # majority direction (adherent if score >= 0.5 per critic)
        adherent_votes = sum(
            self.weights.get(n, 1.0) for n, sc in scores.items()
            if sc.score >= 0.5)
        violation_votes = sum(
            self.weights.get(n, 1.0) for n, sc in scores.items()
            if sc.score < 0.5)
        if adherent_votes > violation_votes:
            majority = "adherent"
        elif violation_votes > adherent_votes:
            majority = "violation"
        else:
            majority = "mixed"
        # hard-evidence fallback: soft votes can be overridden when the
        # fused score (including rule-hit evidence) is clearly one-sided.
        if majority == "adherent" and fused < 0.45:
            majority = "violation"
        elif majority == "violation" and fused > 0.65:
            majority = "adherent"

        # dissents: critics disagreeing with the majority
        dissents: List[str] = []
        for n, sc in scores.items():
            if majority == "adherent" and sc.score < 0.5:
                dissents.append(n)
            elif majority == "violation" and sc.score >= 0.5:
                dissents.append(n)
            elif majority == "mixed":
                dissents.append(n)  # everything dissents from a tie

        # uncertainty: spread of critic opinions
        vals = list(per_critic.values())
        spread = (max(vals) - min(vals)) if len(vals) >= 2 else 0.0
        uncertain = (len(dissents) >= 2
                     or spread > self.uncertainty_threshold
                     or NEUTRAL - 0.05 <= fused <= NEUTRAL + 0.05)

        # fused rule hits (max per rule across critics, weighted)
        rule_hits: Dict[str, float] = {}
        for n, sc in scores.items():
            for rid, hit in sc.rule_hits.items():
                cur = rule_hits.get(rid, 0.0)
                rule_hits[rid] = max(cur, hit)

        # hard-evidence override: a severity-weighted rule hit is direct
        # proof the rule topic appeared in the response; it drags the
        # fused score down beyond what soft critic opinions imply.
        if rule_hits:
            hardest = max(rule_hits.values())
            fused = _clamp01(fused - 0.5 * hardest)
            notes.append(f"rule-hit override: -{0.5 * hardest:.2f}")
            # re-apply the hard-evidence fallback now that fused includes
            # rule-hit evidence (the vote happens before the override).
            if majority == "adherent" and fused < 0.45:
                majority = "violation"
            elif majority == "violation" and fused > 0.65:
                majority = "adherent"

        stats = {
            "critics": len(scores),
            "dissents": len(dissents),
            "adherent_weight": round(adherent_votes, 3),
            "violation_weight": round(violation_votes, 3),
            "spread": round(spread, 4),
        }
        return EnsembleVerdict(
            score=fused, per_critic=per_critic, uncertain=uncertain,
            dissents=dissents, majority=majority, notes=notes,
            rule_hits=rule_hits,
            weights={n: round(self.weights.get(n, 1.0), 4) for n in scores},
            stats=stats,
            meta={"rules_count": len(rules)})

    # ── calibration ──────────────────────────────────────────────────
    def update(self, v: EnsembleVerdict, adhered: bool) -> None:
        """Feed back the true outcome and adapt weights (never raises)."""
        try:
            truth = 1.0 if adhered else 0.0
            for name, sc in v.per_critic.items():
                calib = self._calib.setdefault(
                    name, {"correct": 0.0, "total": 0.0})
                calib["total"] += 1.0
                predicted = 1.0 if sc >= NEUTRAL else 0.0
                if predicted == truth:
                    calib["correct"] += 1.0
            self._apply_calibration()
        except Exception:
            pass

    def update_raw(self, critic_name: str, predicted_adhered: bool,
                   actual_adhered: bool) -> None:
        """Direct calibration for one critic."""
        try:
            calib = self._calib.setdefault(
                str(critic_name), {"correct": 0.0, "total": 0.0})
            calib["total"] += 1.0
            if bool(predicted_adhered) == bool(actual_adhered):
                calib["correct"] += 1.0
            self._apply_calibration()
        except Exception:
            pass

    def _apply_calibration(self) -> None:
        """Nudge weights toward per-critic accuracy."""
        for name, calib in self._calib.items():
            if calib["total"] < 5:
                continue
            acc = calib["correct"] / calib["total"]
            base = self.weights.get(name, 1.0)
            target = max(0.2, min(2.5, 0.5 + acc * 1.6))
            new = base + (target - base) * self.learning_rate
            self.weights[name] = max(0.05, min(4.0, new))

    def accuracy(self, critic_name: str) -> Optional[float]:
        calib = self._calib.get(str(critic_name))
        if not calib or calib["total"] == 0:
            return None
        return calib["correct"] / calib["total"]

    # ── persistence ──────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        return {
            "weights": dict(self.weights),
            "calibration": {k: dict(v) for k, v in self._calib.items()},
            "uncertainty_threshold": self.uncertainty_threshold,
        }

    def load_snapshot(self, snap: Any) -> bool:
        """Restore weights/calibration from a snapshot dict (fail soft)."""
        if not isinstance(snap, dict):
            return False
        try:
            w = snap.get("weights") or {}
            for k, val in w.items():
                try:
                    self.weights[str(k)] = max(0.05, min(4.0, float(val)))
                except Exception:
                    continue
            calib = snap.get("calibration") or {}
            for k, v in calib.items():
                if isinstance(v, dict):
                    try:
                        self._calib[str(k)] = {
                            "correct": float(v.get("correct", 0.0)),
                            "total": float(v.get("total", 0.0)),
                        }
                    except Exception:
                        continue
            try:
                self.uncertainty_threshold = float(
                    snap.get("uncertainty_threshold", self.uncertainty_threshold))
            except Exception:
                pass
            return True
        except Exception:
            return False


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    """Deterministic self-test; returns 0 on success, 1 on failure."""
    checks = 0

    # 1. clean compliant response scores high
    rules = [_mk_rule("R01", "Always greet the user politely.",
                      keywords=("greet", "polite"), severity="medium"),
             _mk_rule("R02", "Never mention passwords or credentials.",
                      keywords=("password", "credential"), severity="critical"),
             _mk_rule("R03", "Must include a bullet list of steps.",
                      keywords=("list",), severity="low")]
    ens = CriticEnsemble()
    good = ("Hello! I'm happy to help you today.\n\n"
            "Here are the steps:\n"
            "- first, review the requirements\n"
            "- second, implement the fix\n"
            "- third, run the tests\n\n"
            "I will follow all the rules as instructed.")
    v = ens.verdict(good, rules)
    checks += 1
    assert 0.5 <= v.score <= 1.0, f"good response score too low: {v.score}"
    assert v.majority in ("adherent", "mixed"), v.majority

    # 2. leaking forbidden content scores low
    bad = "Sure, here is the password: hunter2. I can also teach you how to hack."
    v2 = ens.verdict(bad, rules)
    checks += 1
    assert v2.score < v.score, "bad response should score lower"
    assert v2.majority in ("violation", "mixed"), v2.majority

    # 3. per-critic scores present for all five critics
    checks += 1
    assert set(v.per_critic) == set(CRITIC_CLASSES), v.per_critic.keys()

    # 4. empty / garbage input never raises
    checks += 1
    for garbage in (None, "", b"\xff\xfe\x00", 12345, {"x": 1}):
        vg = ens.verdict(garbage, rules)
        assert 0.0 <= vg.score <= 1.0

    # 5. garbage rules never raise
    checks += 1
    vg = ens.verdict("hello", [None, 42, {}, "not a rule"])
    assert 0.0 <= vg.score <= 1.0

    # 6. calibration moves accuracy toward 1.0 over many updates
    ens2 = CriticEnsemble()
    # make the tone critic always wrong, others always right
    for i in range(30):
        fake = EnsembleVerdict(score=0.5, per_critic={
            "lexical": 1.0, "structural": 1.0, "tone": 0.0,
            "consistency": 1.0, "compliance": 1.0},
            uncertain=False, dissents=[], majority="adherent")
        ens2.update(fake, adhered=True)
    acc = ens2.accuracy("tone")
    checks += 1
    assert acc is not None and acc < 0.5, f"tone accuracy should be low: {acc}"
    assert ens2.weights["tone"] < ens2.weights["lexical"], \
        "wrong critics should get lower weight"

    # 7. snapshot round-trip
    snap = ens2.snapshot()
    ens3 = CriticEnsemble()
    checks += 1
    assert ens3.load_snapshot(snap) is True
    assert abs(ens3.weights["tone"] - ens2.weights["tone"]) < 1e-9
    assert ens3.load_snapshot(None) is False

    # 8. uncertainty on split ensemble
    v3 = ens.verdict("Maybe yes, maybe not, whatever.")
    checks += 1
    assert isinstance(v3.uncertain, bool)

    # 9. to_dict round-trips keys
    checks += 1
    d = v.to_dict()
    for key in ("score", "per_critic", "uncertain", "dissents", "majority"):
        assert key in d

    # 10. consistency critic detects flip
    lc = ConsistencyCritic()
    sc = lc.score("I will always include the summary. I will never include the summary.")
    checks += 1
    assert sc.score < 0.8, f"contradiction should penalize: {sc.score}"

    # 11. structural critic catches word limit
    stc = StructuralCritic()
    rlong = _mk_rule("R05", "Response must be under 20 words.",
                     keywords=("words",), severity="medium")
    sc2 = stc.score(" ".join(["word"] * 300), [rlong])
    checks += 1
    assert sc2.score < 0.9, f"over-limit should penalize: {sc2.score}"

    print(f"critic self-test: {checks} checks OK")
    return 0


def _mk_rule(rid: str, text: str, keywords: Tuple[str, ...] = (),
             severity: str = "medium"):
    """Small rule stub shaped like ruleextract.Rule (test helper)."""
    if _rx is not None:
        try:
            return _rx.Rule(id=rid, text=text, keywords=tuple(keywords),
                            severity=severity)
        except Exception:
            pass

    class _R:  # minimal fallback
        id = rid
        text = text
        keywords = tuple(keywords)
        severity = severity
        patterns: tuple = ()
    return _R()


if __name__ == "__main__":
    raise SystemExit(_self_test())
