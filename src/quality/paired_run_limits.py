"""Resource limits applied before paired-run semantic parsing."""

from __future__ import annotations

from typing import assert_never, Final, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

MAX_DOCUMENT_BYTES: Final = 1_048_576
MAX_NESTING: Final = 16
MAX_TASKS: Final = 128
MAX_SKILLS_PER_ARM: Final = 128
MAX_RESULTS: Final = MAX_TASKS * 2
MAX_STRING_LENGTH: Final = 512


def document_limit_errors(document: str) -> list[str]:
    """Reject oversized text and excessive JSON container nesting pre-decode."""
    errors: list[str] = []
    if len(document) > MAX_DOCUMENT_BYTES:
        return ["document exceeds the byte limit"]
    try:
        size = len(document.encode("utf-8"))
    except UnicodeEncodeError:
        return ["document must be valid UTF-8 text"]
    if size > MAX_DOCUMENT_BYTES:
        return ["document exceeds the byte limit"]
    depth = 0
    quoted = False
    escaped = False
    for character in document:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "[{":
            depth += 1
            if depth > MAX_NESTING:
                errors.append("document exceeds the nesting limit")
                break
        elif character in "]}":
            depth -= 1
    return errors


def payload_limit_errors(payload: JsonValue) -> list[str]:
    """Check collection and decoded-string bounds without recursive traversal."""
    errors: list[str] = []
    stack: list[tuple[JsonValue, int]] = [(payload, 1)]
    while stack:
        value, depth = stack.pop()
        if depth > MAX_NESTING:
            return ["document exceeds the nesting limit"]
        match value:
            case str() as text:
                if len(text) > MAX_STRING_LENGTH:
                    errors.append("document contains a string exceeding the length limit")
                    return errors
            case list() as items:
                stack.extend((item, depth + 1) for item in items)
            case dict() as mapping:
                for key, item in mapping.items():
                    if len(key) > MAX_STRING_LENGTH:
                        errors.append("document contains a string exceeding the length limit")
                        return errors
                    stack.append((item, depth + 1))
            case int() | float() | bool() | None:
                continue
            case unreachable:
                assert_never(unreachable)
    if not isinstance(payload, dict):
        return errors
    tasks = payload.get("tasks")
    if isinstance(tasks, list) and len(tasks) > MAX_TASKS:
        errors.append("tasks exceed the item limit")
    results = payload.get("results")
    if isinstance(results, list) and len(results) > MAX_RESULTS:
        errors.append("results exceed the item limit")
    for arm_name in ("baseline", "variant"):
        arm = payload.get(arm_name)
        if isinstance(arm, dict):
            skills = arm.get("exposed_skills")
            if isinstance(skills, list) and len(skills) > MAX_SKILLS_PER_ARM:
                errors.append(f"{arm_name}.exposed_skills exceed the item limit")
    return errors
