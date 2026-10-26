"""Measure routing accuracy, not routing regression.

Every existing routing gate answers "did behaviour change?". None answers "is it
right?". That is how a suite of routing tests stayed green while ordinary
requests misrouted: the expectations encoded whatever the router already did.

This gate asks the other question. It runs the same six intents in six
languages and scores each turn against the workflow a competent reader would
pick, so a change can be shown to have moved accuracy rather than merely
preserved behaviour. Running one intent set across languages is deliberate: it
makes the coverage gap legible instead of averaging it away.

Outcomes per case:

- `resolved`      dispatched to the expected workflow
- `handed_off`    undecided, but the candidate handoff shortlists the expected
                  workflow, so model selection can still land it
- `missed`        neither
- `high_confidence_misroute` dispatched to the wrong workflow at high
                  confidence. This is the worst failure mode: no picker and no
                  handoff rescues the user, so it carries its own ceiling.

Note that Spanish is Latin script and therefore reported as trigger-backed by
`routing/input_language.py`, while the trigger tables are English. Latin script
is not the same thing as English, and this corpus is what makes that visible.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..routing.chat import route_chat_message
from .reported_rate import ReportedRate, reported_rate


ROUTING_ACCURACY_SCHEMA_VERSION = "routing_accuracy/v1"

OUTCOME_RESOLVED = "resolved"
OUTCOME_HANDED_OFF = "handed_off"
OUTCOME_MISSED = "missed"


@dataclass(frozen=True)
class RoutingAccuracyCase:
    id: str
    language: str
    intent: str
    message: str
    expected_skill: str


# Six intents, six languages, one row per pair. Keep the intents identical
# across languages; that is what makes the per-language comparison meaningful.
ROUTING_ACCURACY_CASES: tuple[RoutingAccuracyCase, ...] = (
    # English
    RoutingAccuracyCase("en-build-failure", "en", "build-failure", "why is the build failing on main?", "build-failure-triage"),
    RoutingAccuracyCase("en-code-review", "en", "code-review", "code review this diff", "code-review"),
    RoutingAccuracyCase("en-code-change", "en", "code-change", "add dark mode to the settings page", "ultrawork"),
    RoutingAccuracyCase("en-onboarding", "en", "onboarding", "I am new to this repo, explain its structure", "codebase-onboarding"),
    RoutingAccuracyCase("en-release-notes", "en", "release-notes", "write release notes for 1.0.4", "content-operator"),
    RoutingAccuracyCase("en-paper", "en", "paper", "summarize this paper", "paper-learning"),
    # Korean
    RoutingAccuracyCase("ko-build-failure", "ko", "build-failure", "빌드 실패 원인 봐줘", "build-failure-triage"),
    RoutingAccuracyCase("ko-code-review", "ko", "code-review", "이 diff 코드 리뷰 해줘", "code-review"),
    RoutingAccuracyCase("ko-code-change", "ko", "code-change", "다크모드 추가해줘", "ultrawork"),
    RoutingAccuracyCase("ko-onboarding", "ko", "onboarding", "이 레포 처음 보는데 구조 좀 알려줘", "codebase-onboarding"),
    RoutingAccuracyCase("ko-release-notes", "ko", "release-notes", "1.0.4 릴리즈 노트 써줘", "content-operator"),
    RoutingAccuracyCase("ko-paper", "ko", "paper", "이 논문 요약해줘", "paper-learning"),
    # Japanese
    RoutingAccuracyCase("ja-build-failure", "ja", "build-failure", "ビルドが失敗した理由を教えて", "build-failure-triage"),
    RoutingAccuracyCase("ja-code-review", "ja", "code-review", "このコードをレビューして", "code-review"),
    RoutingAccuracyCase("ja-code-change", "ja", "code-change", "ダークモードを追加して", "ultrawork"),
    RoutingAccuracyCase("ja-onboarding", "ja", "onboarding", "このリポジトリの構造を教えて", "codebase-onboarding"),
    RoutingAccuracyCase("ja-release-notes", "ja", "release-notes", "リリースノートを書いて", "content-operator"),
    RoutingAccuracyCase("ja-paper", "ja", "paper", "この論文を要約して", "paper-learning"),
    # Chinese
    RoutingAccuracyCase("zh-build-failure", "zh", "build-failure", "为什么构建失败了", "build-failure-triage"),
    RoutingAccuracyCase("zh-code-review", "zh", "code-review", "帮我做代码审查", "code-review"),
    RoutingAccuracyCase("zh-code-change", "zh", "code-change", "添加深色模式", "ultrawork"),
    RoutingAccuracyCase("zh-onboarding", "zh", "onboarding", "介绍一下这个仓库的结构", "codebase-onboarding"),
    RoutingAccuracyCase("zh-release-notes", "zh", "release-notes", "写一份发布说明", "content-operator"),
    RoutingAccuracyCase("zh-paper", "zh", "paper", "总结这篇论文", "paper-learning"),
    # Spanish
    RoutingAccuracyCase("es-build-failure", "es", "build-failure", "por qué falla la compilación en main", "build-failure-triage"),
    RoutingAccuracyCase("es-code-review", "es", "code-review", "revisa mi código en este diff", "code-review"),
    RoutingAccuracyCase("es-code-change", "es", "code-change", "añade modo oscuro a la página de ajustes", "ultrawork"),
    RoutingAccuracyCase("es-onboarding", "es", "onboarding", "soy nuevo en este repositorio, explícame su estructura", "codebase-onboarding"),
    RoutingAccuracyCase("es-release-notes", "es", "release-notes", "escribe las notas de la versión 1.0.4", "content-operator"),
    RoutingAccuracyCase("es-paper", "es", "paper", "resume este artículo científico", "paper-learning"),
    # Hindi
    RoutingAccuracyCase("hi-build-failure", "hi", "build-failure", "बिल्ड क्यों फेल हो रहा है", "build-failure-triage"),
    RoutingAccuracyCase("hi-code-review", "hi", "code-review", "मेरे कोड की समीक्षा करो", "code-review"),
    RoutingAccuracyCase("hi-code-change", "hi", "code-change", "डार्क मोड जोड़ो", "ultrawork"),
    RoutingAccuracyCase("hi-onboarding", "hi", "onboarding", "इस रिपॉजिटरी की संरचना समझाओ", "codebase-onboarding"),
    RoutingAccuracyCase("hi-release-notes", "hi", "release-notes", "रिलीज़ नोट्स लिखो", "content-operator"),
    RoutingAccuracyCase("hi-paper", "hi", "paper", "इस शोध पत्र का सारांश दो", "paper-learning"),
)

CLAIM_BOUNDARY = (
    "Routing accuracy measures deterministic local routing against expected workflows. "
    "A handed_off outcome credits the shortlist, not a model decision: it proves the "
    "expected workflow was offered, never that Hermes picked it. This gate does not "
    "prove live Hermes chat rendering, execution, review, CI, or merge."
)


def evaluate_routing_accuracy_case(case: RoutingAccuracyCase, *, source: str = "generic") -> dict[str, object]:
    route = route_chat_message(case.message, source=source, limit=4)
    action = str(route.get("action", ""))
    selected = str(route.get("selected_skill", ""))
    confidence = str(route.get("confidence", ""))
    handoff = route.get("candidate_handoff")
    shortlist = (
        [str(candidate.get("skill")) for candidate in handoff.get("candidates", [])]
        if isinstance(handoff, dict)
        else []
    )
    input_language = route.get("input_language")
    script = str(input_language.get("script")) if isinstance(input_language, dict) else "unknown"

    if action == "dispatch" and selected == case.expected_skill:
        outcome = OUTCOME_RESOLVED
    elif case.expected_skill in shortlist:
        outcome = OUTCOME_HANDED_OFF
    else:
        outcome = OUTCOME_MISSED

    high_confidence_misroute = action == "dispatch" and confidence == "high" and selected != case.expected_skill

    return {
        "id": case.id,
        "language": case.language,
        "intent": case.intent,
        "expected_skill": case.expected_skill,
        "observed_skill": selected,
        "route_action": action,
        "confidence": confidence,
        "script": script,
        "shortlist": shortlist,
        "outcome": outcome,
        "high_confidence_misroute": high_confidence_misroute,
    }


def _resolved_rate(resolved: int, count: int, scope: object) -> ReportedRate:
    return reported_rate(
        numerator=resolved,
        denominator=count,
        numerator_of=(OUTCOME_RESOLVED,),
        denominator_of=f"routing accuracy cases ({scope})",
    )


def _covered_rate(covered: int, count: int, scope: object) -> ReportedRate:
    """Coverage counts model-selection handoffs as covered, so it says so.

    `covered_percent` sums two outcome buckets. The text formatter has always
    explained that; the JSON payload did not, which left every non-human
    consumer reading a composed numerator as a single one.
    """
    return reported_rate(
        numerator=covered,
        denominator=count,
        numerator_of=(OUTCOME_RESOLVED, OUTCOME_HANDED_OFF),
        denominator_of=f"routing accuracy cases ({scope})",
    )


def build_routing_accuracy_demo(*, source: str = "generic") -> dict[str, object]:
    rows = [evaluate_routing_accuracy_case(case, source=source) for case in ROUTING_ACCURACY_CASES]

    languages: dict[str, dict[str, object]] = {}
    for row in rows:
        language = str(row["language"])
        bucket = languages.setdefault(
            language,
            {"language": language, "case_count": 0, "resolved": 0, "handed_off": 0, "missed": 0, "high_confidence_misroutes": 0},
        )
        bucket["case_count"] = int(bucket["case_count"]) + 1
        bucket[str(row["outcome"])] = int(bucket[str(row["outcome"])]) + 1
        if row["high_confidence_misroute"]:
            bucket["high_confidence_misroutes"] = int(bucket["high_confidence_misroutes"]) + 1

    for bucket in languages.values():
        count = int(bucket["case_count"])
        resolved = int(bucket["resolved"])
        covered = resolved + int(bucket["handed_off"])
        resolved_rate = _resolved_rate(resolved, count, bucket["language"])
        covered_rate = _covered_rate(covered, count, bucket["language"])
        bucket["resolved_percent"] = resolved_rate.percent
        bucket["covered_percent"] = covered_rate.percent
        bucket["resolved_rate"] = resolved_rate.to_payload()
        bucket["covered_rate"] = covered_rate.to_payload()

    total = len(rows)
    resolved_total = sum(1 for row in rows if row["outcome"] == OUTCOME_RESOLVED)
    handed_total = sum(1 for row in rows if row["outcome"] == OUTCOME_HANDED_OFF)
    missed_total = sum(1 for row in rows if row["outcome"] == OUTCOME_MISSED)
    misroute_total = sum(1 for row in rows if row["high_confidence_misroute"])
    resolved_rate = _resolved_rate(resolved_total, total, "all languages")
    covered_rate = _covered_rate(resolved_total + handed_total, total, "all languages")

    return {
        "schema_version": ROUTING_ACCURACY_SCHEMA_VERSION,
        "source": source,
        "summary": {
            "case_count": total,
            "language_count": len(languages),
            "resolved": resolved_total,
            "handed_off": handed_total,
            "missed": missed_total,
            "high_confidence_misroutes": misroute_total,
            "resolved_percent": resolved_rate.percent,
            "covered_percent": covered_rate.percent,
            "resolved_rate": resolved_rate.to_payload(),
            "covered_rate": covered_rate.to_payload(),
        },
        "languages": [languages[key] for key in sorted(languages)],
        "cases": rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def format_routing_accuracy_summary(payload: dict[str, object]) -> str:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return "OMH routing accuracy: unavailable"
    lines = [
        "OMH routing accuracy",
        f"Source: {payload.get('source')}",
        (
            f"Result: {summary.get('resolved')}/{summary.get('case_count')} resolved "
            f"({summary.get('resolved_percent')}%); "
            f"{summary.get('covered_percent')}% covered once model-selection handoffs count"
        ),
        f"High-confidence misroutes: {summary.get('high_confidence_misroutes')}",
        "",
        "Per language:",
    ]
    languages = payload.get("languages", [])
    if isinstance(languages, list):
        for bucket in languages:
            if not isinstance(bucket, dict):
                continue
            lines.append(
                f"- {bucket.get('language')}: resolved {bucket.get('resolved')}/{bucket.get('case_count')} "
                f"({bucket.get('resolved_percent')}%), covered {bucket.get('covered_percent')}%, "
                f"high-confidence misroutes {bucket.get('high_confidence_misroutes')}"
            )
    lines.extend(["", f"Boundary: {payload.get('claim_boundary')}"])
    return "\n".join(lines)
