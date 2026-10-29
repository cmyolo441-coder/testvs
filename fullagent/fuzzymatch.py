"""Fuzzy old_string matching for edit tools.

The 2026-10-10 incident: a model asked MultiEdit to change a 2079-line
file, got "old_string not found in file", and retried the identical failing
edit for 512 seconds. The usual cause is a whitespace/indentation mismatch —
the model reproduces code from memory with different leading spaces, tabs
vs spaces, or collapsed blank lines — so the edit is *semantically* right
but *byte-identical* matching rejects it. The retry loop then burns the
whole turn because the error gives the model nothing actionable.

Strategy chain in :func:`find_best_match`:

(a) exact ``str.find`` match;
(b) whitespace-normalized match (every whitespace run in both haystack and
    needle collapses to a single space), mapped back to original offsets;
(c) indentation-insensitive line-based match (same line count, each line
    equal after per-line whitespace collapse);
(d) when nothing matches confidently, difflib close-match suggestions —
    the closest real snippets with line numbers — so the caller (or the
    model) can fix the old_string instead of retrying blindly.

A fuzzy match is only "confident" when it occurs exactly once; ambiguous
matches fall through to suggestions. The return contract is a 3-tuple::

    (match_start, match_end, suggestions)

``match_start``/``match_end`` are offsets into the original content, or
``None`` when nothing matched confidently. ``suggestions`` is a list of
human-readable ``"line N: <text>"`` strings, empty on a confident match.
"""

from __future__ import annotations

import difflib

__all__ = ["find_best_match"]


def _collapse_runs(s: str) -> tuple[str, list[int]]:
    """Collapse every whitespace run in ``s`` to a single space.

    Returns (collapsed_string, index_map) where ``index_map[i]`` is the
    offset in the original ``s`` of the first character that produced
    ``collapsed_string[i]``.
    """
    out: list[str] = []
    mapping: list[int] = []
    i, n = 0, len(s)
    while i < n:
        if s[i].isspace():
            out.append(" ")
            mapping.append(i)
            i += 1
            while i < n and s[i].isspace():
                i += 1
        else:
            out.append(s[i])
            mapping.append(i)
            i += 1
    return "".join(out), mapping


def _line_key(line: str) -> str:
    """Whitespace-insensitive comparison key for one line."""
    return " ".join(line.split())


def _whitespace_normalized_match(
    content: str, old: str
) -> tuple[int, int] | None:
    """Strategy (b): match with all whitespace runs collapsed.

    Returns (start, end) offsets into the original ``content``, or None
    when there is no match or the match is ambiguous.
    """
    norm_content, content_map = _collapse_runs(content)
    norm_old, _ = _collapse_runs(old)
    if not norm_old.strip():
        # A whitespace-only needle would match anywhere; never confident.
        return None
    if norm_content.count(norm_old) != 1:
        return None
    start_n = norm_content.find(norm_old)
    end_n = start_n + len(norm_old)
    start = content_map[start_n]
    # The last normalized char maps to the first original char of the last
    # run/token, so the original span ends one past that char.
    end = content_map[end_n - 1] + 1
    return start, end


def _indentation_insensitive_match(
    content: str, old: str
) -> tuple[int, int] | None:
    """Strategy (c): line-based match ignoring indentation/whitespace.

    The needle must span the same number of lines as the matched block,
    and every line must agree after per-line whitespace collapse.
    Returns (start, end) offsets into ``content``, or None when there is
    no match or the match is ambiguous.
    """
    content_lines = content.splitlines(keepends=True)
    needle_lines = old.splitlines()
    if not needle_lines:
        return None
    needle_keys = [_line_key(line) for line in needle_lines]
    # A needle whose lines are all blank would match any blank stretch.
    if not any(needle_keys):
        return None
    candidates: list[int] = []
    span = len(needle_lines)
    for i in range(len(content_lines) - span + 1):
        if all(
            _line_key(content_lines[i + j]) == needle_keys[j]
            for j in range(span)
        ):
            candidates.append(i)
            if len(candidates) > 1:
                return None  # ambiguous
    if not candidates:
        return None
    i = candidates[0]
    start = sum(len(content_lines[k]) for k in range(i))
    end = start + sum(len(content_lines[k]) for k in range(i, i + span))
    return start, end


def _suggestions(
    content: str, old: str, max_suggestions: int = 6
) -> list[str]:
    """Strategy (d): closest real snippets with line numbers.

    Always returns something actionable: proper difflib close matches
    first, then a best-effort pass of the highest-similarity lines so a
    totally-off needle still gets "line N: ..." hints instead of a bare
    "not found" that invites a blind retry loop.
    """
    lines = content.splitlines()
    stripped = [line.strip() for line in lines]
    out: list[str] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if idx not in seen and stripped[idx]:
            seen.add(idx)
            out.append(f"line {idx + 1}: {lines[idx].strip()[:120]}")

    # Pass 1: proper close matches.
    for needle_line in old.splitlines():
        key = needle_line.strip()
        if not key:
            continue
        for match in difflib.get_close_matches(
            key, stripped, n=2, cutoff=0.55
        ):
            for idx, text in enumerate(stripped):
                if text == match:
                    add(idx)
                    break
        if len(out) >= max_suggestions:
            return out[:max_suggestions]

    # Pass 2: best-effort — highest similarity lines regardless of cutoff.
    if len(out) < 3:
        scored: list[tuple[float, int]] = []
        for needle_line in old.splitlines():
            key = needle_line.strip()
            if not key:
                continue
            for idx, text in enumerate(stripped):
                if text and idx not in seen:
                    ratio = difflib.SequenceMatcher(
                        None, key, text).ratio()
                    scored.append((ratio, idx))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for _, idx in scored:
            add(idx)
            if len(out) >= max_suggestions:
                break
    return out[:max_suggestions]


def find_best_match(
    content: str, old_string: str
) -> tuple[int | None, int | None, list[str]]:
    """Find ``old_string`` in ``content`` using the strategy chain.

    Returns (match_start, match_end, suggestions): offsets into the
    original ``content`` on a confident match (suggestions empty), or
    (None, None, suggestions) with close-match hints when nothing
    matches confidently.
    """
    if not isinstance(old_string, str) or not old_string:
        return None, None, []
    if not isinstance(content, str):
        return None, None, []

    # (a) exact match
    idx = content.find(old_string)
    if idx >= 0:
        return idx, idx + len(old_string), []

    # (b) whitespace-normalized match
    res = _whitespace_normalized_match(content, old_string)
    if res is not None:
        return res[0], res[1], []

    # (c) indentation-insensitive line-based match
    res = _indentation_insensitive_match(content, old_string)
    if res is not None:
        return res[0], res[1], []

    # (d) close-match suggestions so the caller can fix old_string
    return None, None, _suggestions(content, old_string)


def register(agent) -> None:
    """Coordinator hook: fuzzymatch is a helper library, no tool to add.

    Kept so ``fullagent.fuzzymatch`` can sit in the feature-module tuple
    without breaking ``_register_feature_modules``.
    """
    agent.__dict__.setdefault("_fuzzymatch_available", True)


if __name__ == "__main__":
    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    CONTENT = (
        "def foo():\n"
        "    x = 1\n"
        "    return x + 1\n"
        "\n"
        "def bar():\n"
        "    y = 2\n"
        "    return y * 2\n"
    )

    # (i) exact match still works and returns empty suggestions
    s, e, sugg = find_best_match(CONTENT, "    x = 1\n")
    check("exact match offsets", (s, e) == (11, 21))
    check("exact match span", CONTENT[s:e] == "    x = 1\n")
    check("exact match no suggestions", sugg == [])

    # (ii) different indentation / whitespace succeeds via fuzzy
    s, e, sugg = find_best_match(CONTENT, "def foo():\n  x = 1\n\treturn x + 1\n")
    check("fuzzy indentation-insensitive match found", s is not None and e is not None)
    check("fuzzy span covers the real block",
          CONTENT[s:e] == "def foo():\n    x = 1\n    return x + 1\n")
    check("fuzzy match no suggestions", sugg == [])

    # (ii-b) whitespace-collapsed single-line match
    s, e, sugg = find_best_match(CONTENT, "return    y   *   2")
    check("fuzzy whitespace-collapsed match found", s is not None)
    check("fuzzy collapsed span", CONTENT[s:e] == "return y * 2")

    # (ii-c) ambiguous fuzzy match must NOT be confident
    dup = "a = 1\nb = 2\n" * 2
    s, e, sugg = find_best_match(dup, "a  =  1")
    check("ambiguous fuzzy match rejected", s is None and e is None)

    # (iii) garbage old_string returns suggestions with line numbers
    s, e, sugg = find_best_match(CONTENT, "definitely not in the file xyzzy")
    check("garbage returns no match", s is None and e is None)
    check("garbage returns suggestions", len(sugg) > 0)
    check("suggestions carry line numbers",
          all(sug.startswith("line ") and ":" in sug for sug in sugg))
    print("sample suggestion:", sugg[0])

    # empty / wrong-type inputs are safe
    check("empty needle safe", find_best_match(CONTENT, "") == (None, None, []))
    check("whitespace-only needle not confident",
          find_best_match(CONTENT, "   \n  ")[:2] == (None, None))

    print("ALL SELF-TESTS PASSED")
