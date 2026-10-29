"""comply — Multi-Stage Compliance Pipeline (v5 apex).

The L4 compliance gate, rebuilt as a RUBRIC-DRIVEN PIPELINE instead of
a single PASS/FAIL call. Stages:

  S1  EXTRACT    — build the checkable rubric from the directive ledger:
                   every ledger rule becomes a Check with a checkable
                   claim type (style/format/scope/process/factual).
  S2  PLAN       — decide the verification strategy per reply: full
                   audit (expensive, high-stakes) vs spot-check (cheap,
                   low-stakes) vs skip (short replies, low rule count).
  S3  VERIFY     — render the verification prompt: the reply, the
                   selected checks, and the output contract (JSON with
                   per-check verdicts + evidence quotes).
  S4  SCORE      — parse the verdicts, weight by rule importance,
                   produce a ComplianceScore in [0,1] with a repair
                   plan (which rules, what fix, retry or not).
  S5  DECIDE     — policy: pass / soft-warn / hard-fail (retry with
                   corrective feedback / fail loudly).

The pipeline is model-agnostic: S3's prompt works with any chat model,
S4/S5 are pure functions. All text operations are deterministic and
testable without a live model (S1/S2/S4/S5 fully; S3 renders only).

Thread-safe; stdlib only; never raises on malformed input.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .ledger import LedgerEntry, RuleLedger, token_estimate

__all__ = [
    "Check", "CheckType", "build_rubric", "verify_prompt",
    "parse_verdicts", "ComplianceScore", "decide",
    "run_pipeline_plan", "REPAIR_PROMPT",
]


# ── S1: rubric extraction ────────────────────────────────────────────────

class CheckType:
    STYLE = "style"        # tone, voice, person
    FORMAT = "format"      # structure, length, lists, headings
    SCOPE = "scope"        # what the reply must cover / not cover
    PROCESS = "process"    # tool usage, citation, verification steps
    LANGUAGE = "language"  # output language rules
    FACTUAL = "factual"    # evidence, no-unverified-claims rules


_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    (r"\b(hindi|english|language|bhasha)\b", CheckType.LANGUAGE),
    (r"\b(tone|voice|persona|style|register)\b", CheckType.STYLE),
    (r"\b(words?|chars?|length|lines?|paragraph|short|long|bullet|list|"
     r"heading|structure|format)\b", CheckType.FORMAT),
    (r"\b(cite|citation|evidence|proof|verify|test|file path|source)\b",
     CheckType.PROCESS),
    (r"\b(only|scope|cover|exclude|avoid|never mention)\b", CheckType.SCOPE),
    (r"\b(fact|true|false|claim|accurate)\b", CheckType.FACTUAL),
)


@dataclass
class Check:
    rule_id: str
    text: str
    check_type: str
    weight: float              # from ledger importance
    hard: bool                 # hard rules fail the reply; soft only warn
    evidence_hint: str = ""    # what to quote as evidence

    def prompt_line(self) -> str:
        hard = "HARD (violation fails the audit)" if self.hard \
            else "SOFT (violation is a warning)"
        hint = f" Evidence to look for: {self.evidence_hint}." \
            if self.evidence_hint else ""
        return (f"- [{self.rule_id}] ({self.check_type}, {hard}) "
                f"{self.text}{hint}")


_IMPERATIVE_HARD = re.compile(
    r"\b(never|must not|do not|forbidden|prohibited)\b", re.I)


def classify(text: str) -> str:
    t = (text or "").lower()
    for pattern, ctype in _TYPE_HINTS:
        if re.search(pattern, t):
            return ctype
    return CheckType.SCOPE


def _evidence_hint(e: LedgerEntry) -> str:
    t = e.text.lower()
    if "hindi" in t:
        return "the reply's language"
    if "bullet" in t or "list" in t:
        return "list markers (-, *, 1.) in the reply"
    if "words" in t or "length" in t:
        return "the reply's total length"
    if "file path" in t or "cite" in t:
        return "quoted paths/references in the reply"
    if "shell" in t or "command" in t:
        return "any executed/claimed commands"
    return ""


def build_rubric(ledger: RuleLedger, max_checks: int = 16) -> list[Check]:
    """Ledger → ordered list of Check objects (highest weight first)."""
    checks: list[Check] = []
    for e in ledger.entries[:max_checks]:
        checks.append(Check(
            rule_id=e.rule_id,
            text=e.text,
            check_type=classify(e.text),
            weight=max(0.05, e.importance),
            hard=e.importance >= 0.55
            or _IMPERATIVE_HARD.search(e.text) is not None,
            evidence_hint=_evidence_hint(e),
        ))
    return checks


# ── S2: strategy planning ────────────────────────────────────────────────

@dataclass
class VerifyPlan:
    mode: str                  # "full" | "spot" | "skip"
    checks: list[Check]
    reason: str
    est_tokens: int = 0


def run_pipeline_plan(rubric: list[Check], reply: str,
                      *, first_turn: bool = False,
                      after_compaction: bool = False,
                      budget_tokens: int = 4000) -> VerifyPlan:
    """S2: pick the audit strategy. High-stakes turns (first turn after
    startup or compaction, long replies, many hard rules) get the full
    audit; everything else gets a spot-check of the top hard rules;
    trivial replies skip the gate entirely (cost control)."""
    reply = reply or ""
    n_hard = sum(1 for c in rubric if c.hard)
    reply_tokens = token_estimate(reply)
    high_stakes = first_turn or after_compaction or reply_tokens > 600 \
        or n_hard >= 6

    if high_stakes:
        return VerifyPlan(mode="full", checks=rubric,
                          reason="high-stakes turn",
                          est_tokens=reply_tokens + 800)
    if not rubric or reply_tokens < 25:
        return VerifyPlan(mode="skip", checks=[], reason="low-stakes reply")
    # spot-check: top hard rules within budget
    spot: list[Check] = []
    used = 200
    for c in rubric:
        w = 60 + len(c.text) // 2
        if c.hard and used + w <= budget_tokens:
            spot.append(c)
            used += w
    if not spot:
        spot = rubric[:2]
    return VerifyPlan(mode="spot", checks=spot,
                      reason="routine turn (spot-check)",
                      est_tokens=used)


# ── S3: verification prompt ──────────────────────────────────────────────

_VERIFY_TMPL = """\
[CONFIG_LAYER: developer_policy | task=verification]
You are the compliance verifier for the developer configuration.

CHECKS (verify each against the reply; evidence must be quoted):
{checks}

REPLY UNDER TEST:
---
{reply}
---

For EVERY check output a verdict. Output ONLY this JSON (no prose):
{{"checks": [{{"rule_id": "R01", "verdict": "pass"|"fail"|"unclear",
  "evidence": "<short quote from the reply, or empty>",
  "note": "<one line, why>"}}]}}
Judge only what each check says. "unclear" when the reply neither
follows nor violates the rule. Do not invent checks.
"""


def verify_prompt(checks: list[Check], reply: str) -> str:
    """S3: render the verification prompt for any chat model."""
    lines = "\n".join(c.prompt_line() for c in checks) or "- (no checks)"
    return _VERIFY_TMPL.format(checks=lines,
                               reply=(reply or "")[:6000])


# repair prompt — fed back to the model when S5 decides to retry
REPAIR_PROMPT = """\
Your previous reply failed the compliance gate.

VIOLATIONS:
{violations}

Rewrite the reply so every listed rule passes. Keep all correct content;
change only what the violations require. The developer configuration
remains fully binding.
"""


# ── S4: scoring ──────────────────────────────────────────────────────────

@dataclass
class ComplianceScore:
    score: float                    # weighted pass ratio in [0, 1]
    verdict: str                    # "PASS" | "WARN" | "FAIL" | "UNKNOWN"
    failures: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    unclear: list[dict] = field(default_factory=list)
    repair_hint: str = ""

    def summary(self) -> str:
        return (f"{self.verdict} score={self.score:.2f} "
                f"fail={len(self.failures)} warn={len(self.warnings)} "
                f"unclear={len(self.unclear)}")


def parse_verdicts(raw: str, checks: list[Check]) -> ComplianceScore:
    """S4: parse the verifier's JSON, weight failures by rule weight."""
    data: dict = {}
    try:
        m = re.search(r"\{.*\}", raw or "", re.S)
        if m:
            data = json.loads(m.group(0))
    except Exception:
        data = {}
    rows = data.get("checks")
    if not isinstance(rows, list):
        return ComplianceScore(score=0.0, verdict="UNKNOWN",
                               repair_hint="verifier returned no checks")
    by_id = {c.rule_id: c for c in checks}
    total_w = 0.0
    pass_w = 0.0
    score = ComplianceScore(score=0.0, verdict="UNKNOWN")
    for row in rows:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("rule_id", ""))
        c = by_id.get(rid)
        if c is None:
            # tolerate renumbered/alias ids by prefix match
            for cid, cc in by_id.items():
                if rid.startswith(cid) or cid.startswith(rid):
                    c = cc
                    break
        if c is None:
            continue
        w = c.weight
        total_w += w
        v = str(row.get("verdict", "unclear")).lower()
        entry = {"rule_id": rid,
                 "evidence": str(row.get("evidence", ""))[:200],
                 "note": str(row.get("note", ""))[:200],
                 "weight": w, "hard": c.hard}
        if v == "pass":
            pass_w += w
        elif v == "fail":
            (score.failures if c.hard else score.warnings).append(entry)
        else:
            score.unclear.append(entry)
    if total_w <= 0:
        return ComplianceScore(score=0.0, verdict="UNKNOWN",
                               repair_hint="no recognized checks")
    score.score = round(pass_w / total_w, 3)
    if score.failures:
        score.verdict = "FAIL"
        score.repair_hint = "; ".join(
            f"{f['rule_id']}: {f['note'] or f['evidence']}"
            for f in score.failures[:3])[:400]
    elif score.warnings:
        score.verdict = "WARN"
    else:
        score.verdict = "PASS"
    return score


# ── S5: decision policy ──────────────────────────────────────────────────

@dataclass
class GateDecision:
    action: str                 # "pass" | "warn" | "retry" | "fail_loud"
    score: ComplianceScore
    reason: str = ""
    repair_prompt: str = ""


def decide(score: ComplianceScore, *, max_retries: int = 1,
           retries_used: int = 0) -> GateDecision:
    """S5: convert a score into an action. Policy:
      PASS              → pass
      WARN              → warn (deliver, but surface the drift)
      FAIL, retries left → retry with corrective feedback
      FAIL, no retries  → fail_loud (deliver + flag prominently)
    UNKNOWN verdicts fail OPEN: the gate must never deadlock the agent.
    """
    if score.verdict == "UNKNOWN":
        return GateDecision(action="pass", score=score,
                            reason="verifier output unparseable — "
                                   "fail-open")
    if score.verdict == "PASS":
        return GateDecision(action="pass", score=score,
                            reason="all checks passed")
    if score.verdict == "WARN":
        return GateDecision(action="warn", score=score,
                            reason="soft violations only")
    if retries_used < max_retries:
        return GateDecision(
            action="retry", score=score,
            reason=f"hard violations: {score.repair_hint}",
            repair_prompt=REPAIR_PROMPT.format(
                violations="\n".join(
                    f"- [{f['rule_id']}] {f['note'] or f['evidence']}"
                    for f in score.failures)))
    return GateDecision(action="fail_loud", score=score,
                        reason=f"hard violations persist after retry: "
                               f"{score.repair_hint}")


# ── self-test ────────────────────────────────────────────────────────────

def _self_test() -> int:
    from .ledger import build_ledger

    doc = """# Rules
1. Always answer in Hindi.
2. Never use bullet lists unless asked.
- Keep answers under 120 words.
You must cite file paths when editing code.
"""
    led = build_ledger(doc)
    rubric = build_rubric(led)
    assert rubric, "rubric must not be empty"
    assert any(c.check_type == CheckType.LANGUAGE for c in rubric), \
        [c.check_type for c in rubric]
    assert all(0 < c.weight <= 1 for c in rubric)

    # S2 planning
    plan = run_pipeline_plan(rubric, "x" * 50, first_turn=True)
    assert plan.mode == "full", plan.reason
    plan2 = run_pipeline_plan(rubric, "ok", first_turn=False)
    assert plan2.mode == "skip", plan2.reason
    long_reply = "word " * 300
    plan3 = run_pipeline_plan(rubric, long_reply)
    assert plan3.mode in ("full", "spot")

    # S3 prompt renders
    vp = verify_prompt(rubric[:3], "some reply")
    assert "R0" in vp and "some reply" in vp and "JSON" in vp

    # S4 scoring: a clean verdict
    good = json.dumps({"checks": [
        {"rule_id": c.rule_id, "verdict": "pass", "evidence": "", "note": ""}
        for c in rubric]})
    s_good = parse_verdicts(good, rubric)
    assert s_good.verdict == "PASS" and s_good.score == 1.0, s_good.summary()

    # a hard failure
    hard = [c for c in rubric if c.hard]
    assert hard, "expected at least one hard rule"
    bad = json.dumps({"checks": [
        {"rule_id": c.rule_id,
         "verdict": "fail" if c.rule_id == hard[0].rule_id else "pass",
         "evidence": "quote", "note": "used English"}
        for c in rubric]})
    s_bad = parse_verdicts(bad, rubric)
    assert s_bad.verdict == "FAIL" and s_bad.failures, s_bad.summary()
    assert s_bad.repair_hint

    # malformed verifier output → UNKNOWN, never raises
    s_junk = parse_verdicts("garbage {{{", rubric)
    assert s_junk.verdict == "UNKNOWN"

    # S5 decisions
    assert decide(s_good).action == "pass"
    assert decide(s_bad, retries_used=0).action == "retry"
    assert decide(s_bad, retries_used=1).action == "fail_loud"
    d = decide(s_bad, retries_used=0)
    assert "violations" in d.repair_prompt
    assert hard[0].rule_id in json.dumps(s_bad.failures)

    # warn path: only soft rules fail
    soft_only = json.dumps({"checks": [
        {"rule_id": c.rule_id,
         "verdict": "fail" if not c.hard else "pass",
         "evidence": "", "note": "minor"}
        for c in rubric]})
    s_warn = parse_verdicts(soft_only, rubric)
    assert s_warn.verdict == "WARN", s_warn.summary()

    print("COMPLY SELF-TEST PASS (rubric, plan, prompt, score, decide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())