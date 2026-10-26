"""What the words after an explicit skill invocation ask for, when they ask for anything but the skill's work.

The person chose the skill ("/ulw-work …", "$code-review …", and every slash skill Hermes sends), so the default is
to run it. It is answered instead only when the instruction is about the skill or its run, never a task:

- a stop with no task object (`dag_ulw_cues.stops_the_run`: "stop", "멈춰", "이거 취소", "停止", "止めて", "取消",
  "别做了", "그만해 비싸"), the skill's own run named included ("/ulw-loop stop the loop", "/omh-code-review 리뷰 취소",
  "/ulw-work stop ultrawork": `_own_run`). A stop word with a task object is work: "kill the zombie processes", "stop
  the modal from flickering", "로그 폭주 멈춰줘", "abort the rebase". After the cancel skill a stop is its work
  (`STOP_IS_THE_WORK`: "/omh-cancel stop");
- a request to see how the run stands (`dag_ulw_cues.asks_how_the_run_stands`: "status", "show me the run", "상태
  보여줘", "끝났어?", "is it running?", "状態"), except after a skill whose own work is that status
  (`STATUS_IS_THE_WORK`: "/omh-running-work-board status", "/omh-verification-gate is it done?");
- a question about what the skill is (`asks_after_the_name` without its run-state words: "what is it?", "what does
  this do?", "이게 뭐야?", "어떻게 써?", "何これ", "这是什么"; "what is omh-code-review?", "what does this skill do?",
  "이 스킬 뭐야?"), or about dag-ulw ("/ulw-loop dag-ulw가 뭐야?"; a report on it, "/ulw-work dag-ulw is broken", is
  work). After a skill whose work is to explain or judge what the person points at (`MATERIAL_IS_THE_WORK`), a
  question about "this" or "it" is about that material, so only a question that names the skill is about the skill
  ("/omh-code-review how is it?" reviews, "/omh-code-review what is omh-code-review?" is answered). A question
  about using the skill is about the skill after any skill ("/omh-code-review 어떻게 써?").

Every other instruction runs the skill, a question with a task object included: "/omh-code-review is it safe to
merge?", "/ulw-plan how should we migrate billing?", "/research is rust faster than go?", "/omh-app-debugging why did
the run stop?". dag-ulw reads its own sigil (`dag_ulw_cues`). The router (`policy.bare_name_reports_or_asks`) and the
route hint (`awareness._invoked_name_stops_or_asks`) both read this one function. Stdlib only: the plugin bundle loads
this file with Hermes' interpreter (#1623).
"""

from __future__ import annotations

from functools import lru_cache
import re
import unicodedata
from typing import Final

from .dag_ulw_cues import (
    asks_about_dag_ulw,
    asks_how_the_run_stands,
    ja_zh_asks_what_it_is,
    reports_on_dag_ulw,
    stops_the_run,
)
from .name_report_cues import COINED_NAMES, asks_after_the_name
from . import skill_shortlist as _skill_shortlist

# The skills whose job is reporting the state of work or systems, so a status request after their sigil runs them
# ("/omh-running-work-board status"). After any other skill a status request gets OMH's status card, a skill whose
# work is diagnosing, triaging, auditing or ending a run included ("/omh-cancel 상태 보여줘", "/omh-app-debugging is it
# running?"). Decided from each skill's description; `QUESTION_IS_THE_WORK` is the bare-name reading's, and wider.
# doctor's own work is the install's health, which the status card does not show: on a drifted install the card says
# "OMH setup gaps: 0" while doctor reports local_modifications and guidance_projection (#2221 L4), so "/omh-doctor
# status" runs doctor.
STATUS_IS_THE_WORK: Final = frozenset(
    {"running-work-board", "deploy-and-monitor", "verification-gate", "ops-observability-card",
     "live-incident-response", "context-budget-review", "harness-session-inventory", "skill-health", "run-efficiency",
     "agent-ops-review", "doctor"}
)
# The skill whose own work is the stop, so a stop after its sigil runs it, as on origin/main ("/omh-cancel stop").
STOP_IS_THE_WORK: Final = frozenset({"cancel"})
# The skills whose work is to explain or judge the material the person points at: after one of these, "what is it?"
# or "how is it?" asks about that material.
MATERIAL_IS_THE_WORK: Final = frozenset({"code-review", "codebase-onboarding", "paper-learning"})

# The words for a skill's own run, by a word of its name or label: "/ulw-loop stop the loop", "/omh-plan stop
# planning", "/omh-app-debugging stop the investigation", "/ulw-work stop ultrawork". A word that names what the skill
# works on is no run of it ("/omh-data-pipelines stop the pipelines" is work), so only these words count.
# Korean, Japanese and Chinese forms ride along, and a Korean compound only where it names the skill itself ("코드
# 리뷰"): "무한 루프 멈춰" stops an infinite loop in the code, which is work.
_DEBUGGING: Final = ("debugging", "debug", "investigation", "디버깅", "デバッグ", "调试")
_LEARNING: Final = ("learning", "학습")
_RUN_WORDS: Final = {
    "loop": ("loop", "looping", "루프", "ループ", "循环"),
    "review": ("review", "reviewing", "리뷰", "코드 리뷰", "코드리뷰", "レビュー", "审查"),
    "plan": ("plan", "planning", "계획", "플랜", "計画", "プラン", "计划"),
    "interview": ("interview", "interviewing", "인터뷰", "インタビュー", "访谈"),
    "debugging": _DEBUGGING,
    "debug": _DEBUGGING,
    "work": ("work", "working"),
    "ulw": ("ulw", "ultrawork"),
    "ultrawork": ("ultrawork", "ulw"),
    "research": ("research", "researching", "리서치", "조사", "リサーチ", "调研"),
    "audit": ("audit", "auditing", "감사", "監査", "审计"),
    "qa": ("qa",),
    "analysis": ("analysis", "analyzing", "분석", "分析"),
    "triage": ("triage", "triaging", "트리아지"),
    "onboarding": ("onboarding", "tour", "온보딩", "투어"),
    "learning": _LEARNING,
    "learn": _LEARNING,
}
# A skill's short name, which the router reads as an invocation of it too (`policy._EXPLICIT_SKILL_ALIASES`):
# "/ulw-work what is ulw?" asks about the skill.
_SHORT_NAMES: Final = {"ultrawork": ("ulw",)}
# The skill named as "this skill" or "이 스킬", which a question about the skill reads as "it". A skill with no
# determiner, or with "a", is any skill ("/omh-docs what is a skill?", "스킬이 뭐야?").
_THE_SKILL: Final = re.compile(r"\b(?:this|the|that|your)\s+skill\b|(?:이|그|저)\s*스킬")
# A question about using the skill, which is about the skill after any skill ("/omh-code-review 어떻게 써?").
_USAGE: Final = re.compile(
    r"^(?:how\s+(?:do|can|should)\s+(?:i|we)\s+use\s+(?:it|this)|how\s+to\s+use\s+(?:it|this)"
    r"|(?:이거\s*)?어떻게\s*(?:써|쓰는\s*거야|사용해|사용하는\s*거야)|사용법(?:\s*알려\s*줘)?"
    r"|使い方(?:は)?|どうやって使う(?:の)?|怎么用)\s*[?？。]*$"
)

STOP: Final = "stop"
STATUS: Final = "status"
META: Final = "meta"


@lru_cache(maxsize=256)
def _own_run(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The invoked skill's names (``name`` and the label Hermes lists it under, from the catalog's sidecar) and the
    words for its own run (`_RUN_WORDS`, by each word of a name, and a name spaced when it ends in one: "code
    review")."""
    names = tuple(sorted({name, _skill_shortlist.catalog_skill_label(name), *_SHORT_NAMES.get(name, ())} - {""}))
    nouns: set[str] = set()
    for written in names:
        parts = written.split("-")
        for part in parts:
            nouns.update(_RUN_WORDS.get(part, ()))
        if len(parts) > 1 and parts[-1] in _RUN_WORDS:
            nouns.add(" ".join(parts))
    return names, tuple(sorted(nouns))


def _names_as_it(folded: str, names: tuple[str, ...]) -> str:
    """``folded`` with the skill's names and "this skill" read as "it" ("이거" in Korean), so a question about the
    skill by name reads as one about it ("what is omh-code-review?" as "what is it?"). Only a name in its skill form
    counts, hyphenated or coined: a single common word that is also a skill's name names what it is about ("/omh-rust
    what is rust?", "/omh-frontend how does frontend routing work?")."""
    it = "이거" if re.search(r"[가-힣]", folded) else "it"
    for written in sorted((name for name in names if "-" in name or name in COINED_NAMES), key=len, reverse=True):
        # After an article the word names a thing, not the skill ("what does the frontend do?", "the research").
        folded = re.sub(
            rf"(?<![a-z0-9_-])(?<!\bthe )(?<!\ban )(?<!\ba ){re.escape(written)}(?![a-z0-9_-])", it, folded
        )
    return _THE_SKILL.sub(it, folded)


def invoked_instruction(name: str, rest: str) -> str:
    """`STOP`, `STATUS` or `META` when ``rest``, the words after an explicitly invoked skill (``name``, its canonical
    name), are about the skill or its run, else "": run the skill."""
    names, nouns = _own_run(name)
    if stops_the_run(rest, names=names, nouns=nouns):
        return "" if name in STOP_IS_THE_WORK else STOP
    if asks_how_the_run_stands(rest):
        return "" if name in STATUS_IS_THE_WORK else STATUS
    folded = unicodedata.normalize("NFKC", rest).casefold().strip(" ,:")
    if _USAGE.match(folded):
        return META
    named = _names_as_it(folded, names)
    if named != folded and asks_after_the_name(named, run_state=False):
        return META  # "what is omh-code-review?", "what does this skill do?", "이 스킬 뭐야?"
    if name not in MATERIAL_IS_THE_WORK and (
        asks_after_the_name(folded, run_state=False) or ja_zh_asks_what_it_is(folded)
    ):
        return META
    if asks_about_dag_ulw(folded) and not reports_on_dag_ulw(folded):
        return META  # "/ulw-loop dag-ulw가 뭐야?"; "/ulw-work dag-ulw is broken" is work for ulw-work
    return ""


__all__ = ["MATERIAL_IS_THE_WORK", "META", "STATUS", "STATUS_IS_THE_WORK", "STOP", "STOP_IS_THE_WORK", "invoked_instruction"]
