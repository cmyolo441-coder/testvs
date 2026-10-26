"""Explicit input-script detection for the router contract.

OMH targets a global audience with English as the primary language. Its trigger
tables grew in two scripts and stopped there, and for a long time that was
stated as a policy: per-language trigger tables do not scale, so non-English
intent resolution belongs to model selection. The measurement behind it was
real; the conclusion was not, because it treated "a language costs a source
edit" as a law of nature rather than as a property of where the phrases were
stored.

`routing/trigger_language_packs.py` changes that property. Trigger phrases for
a language are validated data now, shipped as `<lang>.json` and merged into the
catalog, so adding a language is authoring a file rather than editing the
router. Which scripts are trigger-backed therefore stops being a constant
somebody remembered to update and becomes a measurement of the packs actually
installed -- `TRIGGER_BACKED_SCRIPTS` below re-derives it from the catalog on
every process.

What has not changed is the honesty of the boundary. A script with no pack
entries is not unsupported: it means a deterministic trigger score is not
evidence of intent for that message, and the decision belongs to model
selection over supplied candidates. OMH itself still makes no LLM call, so
`docs/DIRECTION.md`'s "not an LLM router" boundary holds.
"""

from __future__ import annotations

from functools import lru_cache


INPUT_LANGUAGE_SCHEMA_VERSION = "routing_input_language/v1"

SCRIPT_LATIN = "latin"
SCRIPT_HANGUL = "hangul"
SCRIPT_KANA = "kana"
SCRIPT_HAN = "han"
SCRIPT_DEVANAGARI = "devanagari"
SCRIPT_ARABIC = "arabic"
SCRIPT_CYRILLIC = "cyrillic"
SCRIPT_UNKNOWN = "unknown"

# A script is trigger-backed when the catalog carries enough phrases in it to
# resolve ordinary requests -- not when one incidental token happens to be
# written in it. The floor is what separates "this language has a pack" from
# "a product name in this script leaked into a trigger": before any pack
# shipped, Han and Kana sat at 5 and 1 entries respectively, which could never
# resolve a Japanese or Chinese sentence and would have been a false claim of
# support.
MIN_TRIGGER_BACKED_PHRASES = 25

SUPPORT_TRIGGER_BACKED = "trigger_backed"
SUPPORT_MODEL_SELECTION_REQUIRED = "model_selection_required"

_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x1100, 0x11FF, SCRIPT_HANGUL),
    (0x3040, 0x30FF, SCRIPT_KANA),
    (0x3130, 0x318F, SCRIPT_HANGUL),
    (0x4E00, 0x9FFF, SCRIPT_HAN),
    (0xAC00, 0xD7AF, SCRIPT_HANGUL),
    (0x0400, 0x04FF, SCRIPT_CYRILLIC),
    (0x0600, 0x06FF, SCRIPT_ARABIC),
    (0x0900, 0x097F, SCRIPT_DEVANAGARI),
)


def script_of_character(character: str) -> str:
    """Classify one character, or `unknown` for digits, punctuation, and symbols."""
    if not character:
        return SCRIPT_UNKNOWN
    if character.isascii():
        return SCRIPT_LATIN if character.isalpha() else SCRIPT_UNKNOWN
    codepoint = ord(character)
    for start, end, script in _SCRIPT_RANGES:
        if start <= codepoint <= end:
            return script
    return SCRIPT_UNKNOWN


@lru_cache(maxsize=8192)
def detect_input_script(message: str) -> str:
    """Return the dominant script of `message`.

    Latin only wins when no other script is present, because product names,
    command tokens, and code identifiers stay Latin inside otherwise non-Latin
    sentences: "Claude Code로 바로 열어줘" is a Korean request, not a Latin one.
    """
    counts: dict[str, int] = {}
    for character in message:
        script = script_of_character(character)
        if script == SCRIPT_UNKNOWN:
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return SCRIPT_UNKNOWN
    non_latin = {script: count for script, count in counts.items() if script != SCRIPT_LATIN}
    if non_latin:
        return max(non_latin.items(), key=lambda item: (item[1], item[0]))[0]
    return SCRIPT_LATIN


@lru_cache(maxsize=1)
def trigger_backed_scripts() -> tuple[str, ...]:
    """Scripts the live catalog carries a usable trigger table for.

    Measured, not declared. The catalog is the merge of the authored triggers
    and every shipped trigger language pack, so shipping `ja.json` is what makes
    Kana trigger-backed -- no constant to remember, and no way for the claim to
    outlive the phrases behind it.
    """
    from ..skills.catalog import routable_definitions

    counts: dict[str, int] = {}
    for definition in routable_definitions():
        for trigger in definition.triggers:
            script = detect_input_script(trigger)
            if script == SCRIPT_UNKNOWN:
                continue
            counts[script] = counts.get(script, 0) + 1
    return tuple(
        sorted(script for script, count in counts.items() if count >= MIN_TRIGGER_BACKED_PHRASES)
    )


def routing_language_support(script: str) -> str:
    """State whether trigger tables can be expected to resolve this script."""
    return SUPPORT_TRIGGER_BACKED if script in trigger_backed_scripts() else SUPPORT_MODEL_SELECTION_REQUIRED


def routing_input_language(message: str) -> dict[str, object]:
    """Describe the input language as an explicit routing input."""
    script = detect_input_script(message)
    support = routing_language_support(script)
    return {
        "schema_version": INPUT_LANGUAGE_SCHEMA_VERSION,
        "script": script,
        "trigger_support": support,
        "trigger_backed_scripts": list(trigger_backed_scripts()),
        "boundary": (
            "Trigger tables carry the scripts the shipped language packs cover. A script "
            "marked model_selection_required is not unsupported; it means a deterministic "
            "trigger score is not evidence of intent and the decision belongs to model "
            "selection over supplied candidates."
        ),
    }
