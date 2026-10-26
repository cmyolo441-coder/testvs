"""Word-boundary phrase matching, owned once by the bundle.

`contains_boundary_phrase` decides whether an executor or runtime name is
mentioned in a message rather than merely contained in another word or a file
name. The control plane (`omh.routing.executor_cues`) imports it from here;
the bundle's awareness route hints read it directly. Until 2026-10-06 the
bundle carried a vendored copy for standalone hosts that lacked the rule
rejecting punctuation followed by more alphanumerics, so "see claude-code.md"
and "codex-utils.py" counted as agent mentions on the live plugin while the
control plane rejected them. One definition, no copy. Stdlib only.
"""

from __future__ import annotations


def contains_boundary_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    """True when any phrase occurs in `text` at a word boundary.

    Plain containment is wrong for the pi-family-runtime phrase family: "api한테"
    contains "pi한테", a word ending in the runtime's short name contains
    `RUNTIME_USER_PHRASE`, and "raspi
    status" contains "pi status". An occurrence only counts here when the
    character immediately before the match is absent or non-alphanumeric --
    `str.isalnum()` is True for Hangul (and its NFKD jamo), so "라즈베리pi한테"
    is rejected the same way "raspi" is -- and the run of ASCII punctuation
    immediately after the match (if any) is not itself followed by another
    ASCII alphanumeric character. A lone trailing ASCII punctuation run
    ("pi:", "claude-code.") is a sentence separator or a sentence-ending
    period, so it still counts as a boundary; a Hangul continuation right
    after the phrase ("opencode로", "codex가", "claude code에게") is a particle
    naming the agent, so that counts too. But punctuation that leads straight
    into more ASCII alphanumeric text without a space ("claudecode-notes",
    "claude-code.md", "codex-utils.py") is a hyphenated word or filename that
    merely starts with the phrase, never a mention of it.

    Callers must pass `text` and `phrases` through the same fold
    (`normalized_phrase` on routing surfaces, plain lowering in coding
    delegation); this helper compares them verbatim.
    """
    return any(phrase and _occurs_at_boundary(text, phrase) for phrase in phrases)


def _occurs_at_boundary(text: str, phrase: str) -> bool:
    start = text.find(phrase)
    while start != -1:
        end = start + len(phrase)
        before_is_boundary = start == 0 or not text[start - 1].isalnum()
        if before_is_boundary and _after_is_boundary(text, end):
            return True
        start = text.find(phrase, start + 1)
    return False


def _after_is_boundary(text: str, end: int) -> bool:
    position = end
    while position < len(text) and _is_ascii_punctuation(text[position]):
        position += 1
    if position >= len(text):
        return True
    following = text[position]
    return not (following.isascii() and following.isalnum())


def _is_ascii_punctuation(character: str) -> bool:
    return character.isascii() and not character.isalnum() and not character.isspace()
