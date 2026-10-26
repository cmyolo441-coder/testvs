from __future__ import annotations

from .catalog import SkillDefinition


def validate_expert_questions(
    definition: SkillDefinition,
    label: str,
    errors: list[str],
) -> None:
    questions = definition.expert_questions
    if not isinstance(questions, (tuple, list)):
        errors.append(f"{label} expert_questions must be a list")
        return
    required_inputs = (
        tuple(item for item in definition.required_inputs if isinstance(item, str))
        if isinstance(definition.required_inputs, (tuple, list))
        else ()
    )
    seen_inputs: set[str] = set()
    for index, question in enumerate(questions):
        question_label = f"{label} expert_questions[{index}]"
        required_input = getattr(question, "required_input", None)
        if (
            not isinstance(required_input, str)
            or required_input not in required_inputs
        ):
            errors.append(
                f"{question_label} required_input must exactly match a declared required_inputs member"
            )
        elif len(required_input) > 120:
            errors.append(
                f"{question_label} required_input must be at most 120 code points"
            )
        if isinstance(required_input, str):
            if required_input in seen_inputs:
                errors.append(f"{question_label} required_input must not be duplicated")
            else:
                seen_inputs.add(required_input)
        for locale in ("en", "ko"):
            text = getattr(question, locale, None)
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{question_label} {locale} must be a non-empty string")
                continue
            if len(text) > 240:
                errors.append(
                    f"{question_label} {locale} must be at most 240 code points"
                )
            if text.count("?") != 1 or not text.endswith("?"):
                errors.append(
                    f"{question_label} {locale} must contain exactly one final question mark"
                )
            if any(marker in text for marker in ("\n", "\r", "|")):
                errors.append(f"{question_label} {locale} must be plain non-table text")
