"""Terminal escape sequences inside text the TUI did not write: remove them whole.

Gateway text (assistant replies, tool summaries, inline diffs, notices) can carry
ANSI colour codes, OSC hyperlinks or cursor movement. Dropping only the ESC byte
would leave `[38;2;…m` debris on screen, and keeping the sequence would let text
move the cursor or clear the screen. `strip_escapes` removes complete and cut-off
sequences with their parameters; the renderer applies its own styles afterwards.
"""

from __future__ import annotations

ESC = "\x1b"
BEL = "\x07"
CSI_8BIT = "\x9b"
ST_8BIT = "\x9c"
_STRING_INTRODUCERS = frozenset("]PX^_")  # after ESC: OSC, DCS, SOS, PM, APC
_STRING_8BIT = frozenset("\x9d\x90\x98\x9e\x9f")  # the same five as single 8-bit characters
_INTRODUCERS = (ESC, CSI_8BIT, *_STRING_8BIT)


def _csi_end(text: str, index: int) -> int:
    """The index after a CSI whose parameters start at `index` (ECMA-48: params, intermediates, final).

    A sequence cut off or broken by a character outside that grammar ends where
    it broke: what was read is dropped and the rest of the text stays.
    """
    while index < len(text) and "0" <= text[index] <= "?":
        index += 1
    while index < len(text) and " " <= text[index] <= "/":
        index += 1
    if index < len(text) and "@" <= text[index] <= "~":
        return index + 1
    return index


def _control_string_end(text: str, index: int) -> int:
    """The index after a control string (OSC, DCS, ...) whose body starts at `index`.

    BEL or ST (ESC \\ or the 8-bit ST) ends it; an unterminated one runs to the
    end of the text, which is what a terminal would swallow too.
    """
    while index < len(text):
        ch = text[index]
        if ch in (BEL, ST_8BIT):
            return index + 1
        if ch == ESC and text[index + 1 : index + 2] == "\\":
            return index + 2
        index += 1
    return index


def _escape_end(text: str, index: int) -> int:
    """The index after the escape sequence whose ESC is at `index`; a lone ESC is one character."""
    introducer = text[index + 1 : index + 2]
    if introducer == "[":
        return _csi_end(text, index + 2)
    if introducer and introducer in _STRING_INTRODUCERS:
        return _control_string_end(text, index + 2)
    after = index + 1
    while after < len(text) and " " <= text[after] <= "/":
        after += 1  # nF intermediates, as in ESC ( B
    if after < len(text) and "0" <= text[after] <= "~":
        return after + 1
    return index + 1


def strip_escapes(text: str) -> str:
    """`text` without any complete or cut-off terminal escape sequence (7-bit or 8-bit introducers)."""
    if not any(mark in text for mark in _INTRODUCERS):
        return text
    kept: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == ESC:
            index = _escape_end(text, index)
        elif ch == CSI_8BIT:
            index = _csi_end(text, index + 1)
        elif ch in _STRING_8BIT:
            index = _control_string_end(text, index + 1)
        else:
            kept.append(ch)
            index += 1
    return "".join(kept)
