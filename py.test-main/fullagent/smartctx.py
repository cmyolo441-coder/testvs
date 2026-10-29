"""Smart context: ``@file`` attachments and auto-included files.

Pre-processes the user's raw input line before the turn starts:

- ``extract_mentions`` finds ``@path`` tokens and bare tokens that are
  existing files relative to ``cwd``.
- ``read_attachments`` reads them into a plain-text block (with per-file
  and total size caps, binary skip).
- ``expand_input`` returns ``(cleaned_text, attachments_block)``; the
  coordinator appends the block to the user message right before the
  ``run_turn`` call in ``tui.py``.

Public API:
    - :func:`register` -- attach :func:`expand_input` to an agent.
    - :func:`extract_mentions` -- ``@`` / bare-file token extraction.
    - :func:`read_attachments` -- bounded file-content block builder.
    - :func:`expand_input` -- ``(text, attachments_block)`` for the turn.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Tuple

# @<path> token, e.g. @src/main.py, @./notes.md, @~/todo.txt
_MENTION_RE = re.compile(r"@([\w./-][\w./~+-]*)")

_QUOTES = "'\"`"

#: per-file content cap
MAX_FILE_BYTES = 50 * 1024
#: total content cap across all attachments
MAX_TOTAL_BYTES = 200 * 1024


def _candidate_path(token: str, cwd: str) -> str | None:
    """Return the token if it names an existing file, else None."""
    token = token.strip().strip(_QUOTES)
    if not token:
        return None
    token = os.path.expanduser(token)
    full = token if os.path.isabs(token) else os.path.join(cwd, token)
    full = os.path.expanduser(full)
    if os.path.isfile(full):
        return token
    return None


def extract_mentions(text: str, cwd: str = ".") -> List[str]:
    """Find file mentions in ``text``.

    Collects ``@path`` tokens (regex ``@([\\w./-][\\w./~+-]*)``) and bare
    whitespace-separated tokens that are existing files relative to
    ``cwd`` (quotes stripped). Dedupes, preserving first-seen order.
    """
    seen: List[str] = []
    seen_set = set()

    def _add(token: str) -> None:
        if token and token not in seen_set:
            seen_set.add(token)
            seen.append(token)

    # 1. @path tokens — kept as written (no existence check here)
    for match in _MENTION_RE.finditer(text or ""):
        _add(match.group(1))

    # 2. bare tokens that are existing files
    for token in (text or "").split():
        if token.startswith("@"):
            continue  # already handled by the @-pass above
        path = _candidate_path(token, cwd)
        if path is not None:
            _add(path)

    return seen


def read_attachments(paths: List[str], cwd: str = ".") -> str:
    """Read ``paths`` into a ``--- <path> ---`` content block.

    Skips files larger than 50KB ("[skipped: too large]") and files with
    null bytes ("[skipped: binary]"); missing/unreadable files get
    "[skipped: unreadable]". Total content across all files is capped at
    200KB — files beyond the cap are marked "[skipped: total cap]".
    """
    parts: List[str] = []
    total = 0

    for path in paths:
        header = f"--- {path} ---\n"
        path = os.path.expanduser(path)
        full = path if os.path.isabs(path) else os.path.join(cwd, path)
        full = os.path.expanduser(full)
        try:
            size = os.path.getsize(full)
        except OSError:
            parts.append(header + "[skipped: unreadable]\n")
            continue
        if size > MAX_FILE_BYTES:
            parts.append(header + "[skipped: too large]\n")
            continue
        if total + size > MAX_TOTAL_BYTES:
            parts.append(header + "[skipped: total cap]\n")
            continue
        try:
            with open(full, "rb") as fh:
                raw = fh.read()
        except OSError:
            parts.append(header + "[skipped: unreadable]\n")
            continue
        if b"\x00" in raw:
            parts.append(header + "[skipped: binary]\n")
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")
        total += len(raw)
        parts.append(header + content + "\n")

    return "".join(parts)


def expand_input(text: str, cwd: str = ".") -> Tuple[str, str]:
    """Return ``(text_with_@refs_kept, attachments_block)``.

    ``attachments_block`` is ``""`` when no files are mentioned, else
    ``"\\n\\n[ATTACHED FILES]\\n" + read_attachments(...)``. The text
    itself is returned unchanged (the ``@`` refs stay visible).
    """
    mentions = extract_mentions(text, cwd=cwd)
    if not mentions:
        return text, ""
    block = "\n\n[ATTACHED FILES]\n" + read_attachments(mentions, cwd=cwd)
    return text, block


def register(agent: Any) -> None:
    """Wire smart context into an agent (duck-typed, no imports)."""
    agent.expand_input = expand_input


if __name__ == "__main__":
    import tempfile

    tmp = tempfile.mkdtemp(prefix="smartctx_selftest_")

    # 2 text files, 1 binary, 1 large file
    notes = os.path.join(tmp, "notes.txt")
    with open(notes, "w", encoding="utf-8") as fh:
        fh.write("hello notes\n")
    big_text = os.path.join(tmp, "big.txt")
    with open(big_text, "w", encoding="utf-8") as fh:
        fh.write("x" * 60 * 1024)  # 60KB > 50KB cap
    binfile = os.path.join(tmp, "blob.bin")
    with open(binfile, "wb") as fh:
        fh.write(b"\x00\x01\x02binary\x00data")

    # 1. @-syntax extraction
    m = extract_mentions("please read @notes.txt and fix", cwd=tmp)
    assert m == ["notes.txt"], m

    # 2. bare-path extraction (quoted too), dedupe, order preserved
    m = extract_mentions('check "notes.txt" and @big.txt notes.txt', cwd=tmp)
    assert m == ["big.txt", "notes.txt"], m

    # 3. non-files are ignored for bare tokens; @ refs kept regardless
    m = extract_mentions("notafile @ghost.md", cwd=tmp)
    assert m == ["ghost.md"], m

    # 4. read_attachments: text emitted, large/binary skipped
    block = read_attachments(["notes.txt", "big.txt", "blob.bin"], cwd=tmp)
    assert "--- notes.txt ---\nhello notes" in block, block
    assert "--- big.txt ---\n[skipped: too large]" in block, block
    assert "--- blob.bin ---\n[skipped: binary]" in block, block
    assert "--- missing.txt ---\n[skipped: unreadable]" in read_attachments(
        ["missing.txt"], cwd=tmp)

    # 5. total 200KB cap: 5 x 45KB files -> 4 fit (180KB), 5th capped
    fits = []
    for i in range(5):
        p = os.path.join(tmp, f"fit{i}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("z" * 45 * 1024)  # 45KB each: 4*45=180 fit, 5th over
        fits.append(f"fit{i}.txt")
    block = read_attachments(fits, cwd=tmp)
    assert block.count("--- fit") == 5, block
    assert block.count("[skipped: total cap]") == 1, block

    # 6. expand_input output shape
    text, attach = expand_input("look at @notes.txt now", cwd=tmp)
    assert text == "look at @notes.txt now"  # refs kept, text unchanged
    assert attach.startswith("\n\n[ATTACHED FILES]\n")
    assert "hello notes" in attach
    text2, attach2 = expand_input("no files here", cwd=tmp)
    assert text2 == "no files here" and attach2 == ""

    # 7. register attaches the helper
    class FakeAgent:
        pass
    a = FakeAgent()
    register(a)
    assert a.expand_input is expand_input
    assert a.expand_input("@notes.txt", cwd=tmp)[1].startswith(
        "\n\n[ATTACHED FILES]\n")

    print("smartctx self-test PASSED")
