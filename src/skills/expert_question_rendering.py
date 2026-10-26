from __future__ import annotations

from .catalog import SkillDefinition


def expert_questions_markdown(definition: SkillDefinition, *, limit: int | None = None) -> str:
    questions = definition.expert_questions if limit is None else definition.expert_questions[:limit]
    if not questions:
        return ""
    lines = ["Expert clarification questions:"]
    for question in questions:
        lines.extend(
            [
                f"- `{question.required_input}`",
                f"  - English: {question.en}",
                f"  - Korean: {question.ko}",
            ]
        )
    return "\n".join(lines)


def expert_question_reference_lines(definition: SkillDefinition) -> list[str]:
    if not definition.expert_questions:
        return []
    return [
        "- Expert clarification questions:",
        *[
            line
            for question in definition.expert_questions
            for line in (
                f"  - `{question.required_input}`",
                f"    - English: {question.en}",
                f"    - Korean: {question.ko}",
            )
        ],
    ]


def expert_question_payloads(definition: SkillDefinition) -> list[dict[str, object]]:
    return [
        {
            "required_input": question.required_input,
            "questions": {"en": question.en, "ko": question.ko},
        }
        for question in definition.expert_questions
    ]


def copy_expert_question_payloads(payloads: object) -> list[dict[str, object]]:
    if not isinstance(payloads, list):
        raise TypeError("expert question payloads must be a list")
    copied: list[dict[str, object]] = []
    for item in payloads:
        if not isinstance(item, dict):
            raise TypeError("expert question payload must be an object")
        required_input = item.get("required_input")
        questions = item.get("questions")
        if not isinstance(required_input, str) or not isinstance(questions, dict):
            raise TypeError("expert question payload has an invalid shape")
        copied.append({"required_input": required_input, "questions": dict(questions)})
    return copied


def domain_expert_question_body(context: dict[str, object] | None) -> str | None:
    if not isinstance(context, dict):
        return None
    workflow = context.get("workflow_hint")
    required_input = context.get("required_input")
    question = context.get("question")
    if not isinstance(workflow, str) or not workflow:
        return None
    if not isinstance(required_input, str) or not required_input:
        return None
    if not isinstance(question, dict):
        return None
    text = question.get("text")
    if not isinstance(text, str) or not text:
        return None
    return text
