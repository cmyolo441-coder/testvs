"""Edit fallback chain: apply an old_string/new_string edit with graceful
degradation instead of a single exact-match attempt.

Incident that motivated this: MultiEdit failed on an exact old_string
match and the model had no fallback -- it just retried the identical
failing call for 512s.  This module tries, in order:

  (a) exact match (old_string occurs exactly once);
  (b) fuzzy match via fullagent.fuzzymatch (whitespace-normalized, then
      indentation-insensitive line match), reported under its winning
      sub-strategy name;
  (c) line-number-anchored edit (start_line/end_line given, region must
      approximately match old_string);
  (d) structured failure: {"ok": False, "tried": [...],
      "context": <file excerpt with line numbers around best guess>,
      "hint": "re-read the region and retry with exact text"}.

Pure-text helpers (find_edit_span / apply_edit_to_text) let callers like
multiedit.py keep their own atomic read/validate/write flow; the
file-level apply_edit_with_fallbacks re-reads fresh, py_compiles .py
results, and writes atomically.
"""

from __future__ import annotations

import difflib
import os
import py_compile
import tempfile
from pathlib import Path
from typing import Any

from .fuzzymatch import find_best_match

STRATEGIES = ("exact", "whitespace-normalized", "indentation-insensitive",
              "line-anchored")

_WS_CHARS = " \t\r\n\v\f"

#: Minimum SequenceMatcher ratio between old_string and a line range for
#: the line-anchored strategy to accept the anchor.
_LINE_ANCHOR_MIN_SIMILARITY = 0.60

#: Lines of context shown around the best-guess line on total failure.
_CONTEXT_RADIUS = 8


def _collapse_runs(s: str) -> str:
    """Collapse every whitespace run to a single space (no stripping).

    Mirrors fuzzymatch._collapse_runs so the winning fuzzy sub-strategy
    can be classified: fuzzymatch tries its whitespace-normalized match
    before its indentation-insensitive one, so equal collapsed forms mean
    the whitespace-normalized strategy won.
    """
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i].isspace():
            out.append(" ")
            i += 1
            while i < n and s[i].isspace():
                i += 1
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _normalize_ws(s: str) -> str:
    """Collapse every whitespace run to a single space and strip ends."""
    out: list[str] = []
    prev_space = True  # leading whitespace is stripped
    for ch in s:
        if ch in _WS_CHARS:
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            out.append(ch)
            prev_space = False
    if out and out[-1] == " ":
        out.pop()
    return "".join(out)


def _best_guess_context(text: str, old_string: str) -> str:
    """File excerpt with line numbers around the line that best matches
    old_string's first line -- the most useful place to re-read."""
    lines = text.splitlines()
    if not lines:
        return "<file is empty>"
    norm_old = _normalize_ws(old_string)
    first = norm_old.split("\n")[0] if norm_old else ""
    best_i, best_r = 0, -1.0
    if first:
        for i, ln in enumerate(lines):
            r = difflib.SequenceMatcher(None, first, _normalize_ws(ln)).ratio()
            if r > best_r:
                best_r, best_i = r, i
    else:
        best_r = 0.0
    lo = max(0, best_i - _CONTEXT_RADIUS)
    hi = min(len(lines), best_i + _CONTEXT_RADIUS + 1)
    numbered = [f"{j + 1:>6}: {lines[j]}" for j in range(lo, hi)]
    header = (f"best-guess region around line {best_i + 1} "
              f"(similarity {best_r:.2f}, file has {len(lines)} lines):")
    return header + "\n" + "\n".join(numbered)


def _fail(tried: list[str], text: str, old_string: str) -> dict[str, Any]:
    return {
        "ok": False,
        "tried": tried or ["exact (no match)"],
        "context": _best_guess_context(text, old_string),
        "hint": "re-read the region and retry with exact text",
    }


def find_edit_span(text: str, old_string: str,
                   start_line: int | None = None,
                   end_line: int | None = None) -> dict[str, Any]:
    """Locate where old_string applies in text.

    Success: {"ok": True, "strategy": <name>, "start": int, "end": int,
              "tried": [...]} where text[start:end] is the span to replace.
    Failure: {"ok": False, "tried": [...], "context": ..., "hint": ...}.
    """
    tried: list[str] = []
    if not isinstance(old_string, str) or not old_string:
        return _fail(["old_string is empty"], text, old_string or "")

    # (a) exact match -- must be unique
    count = text.count(old_string)
    if count == 1:
        s = text.index(old_string)
        return {"ok": True, "strategy": "exact", "start": s,
                "end": s + len(old_string), "tried": ["exact"]}
    tried.append("exact (%s)" % ("no match" if count == 0
                                 else f"{count} matches, ambiguous"))

    # (b) fuzzy match (fuzzymatch.py): whitespace-normalized, then
    # indentation-insensitive. Only when exact found nothing -- with an
    # ambiguous exact match the fuzzy winner would be a guess, so the
    # chain skips straight to the explicit line anchor instead.
    if count == 0:
        s, e, _suggestions = find_best_match(text, old_string)
        if s is not None:
            strategy = ("whitespace-normalized"
                        if _collapse_runs(text[s:e]) == _collapse_runs(old_string)
                        else "indentation-insensitive")
            return {"ok": True, "strategy": strategy, "start": s, "end": e,
                    "tried": tried + [strategy]}
        tried.append("whitespace-normalized / indentation-insensitive "
                     "(no confident match)")
    else:
        tried.append("fuzzy match skipped (exact match ambiguous)")

    # (c) line-number-anchored edit
    if start_line is not None or end_line is not None:
        lines = text.splitlines(keepends=True)
        n = len(lines)
        s = start_line if start_line is not None else 1
        e = end_line if end_line is not None else s
        if 1 <= s <= e <= n and n > 0:
            region = "".join(lines[s - 1:e])
            sim = difflib.SequenceMatcher(
                None, _normalize_ws(old_string),
                _normalize_ws(region)).ratio()
            if sim >= _LINE_ANCHOR_MIN_SIMILARITY:
                ostart = sum(len(ln) for ln in lines[:s - 1])
                return {"ok": True, "strategy": "line-anchored",
                        "start": ostart, "end": ostart + len(region),
                        "tried": tried + [f"line-anchored (similarity "
                                          f"{sim:.2f}, lines {s}-{e})"],
                        "lines": (s, e)}
            tried.append(f"line-anchored (similarity {sim:.2f} < "
                         f"{_LINE_ANCHOR_MIN_SIMILARITY:.2f}, lines {s}-{e})")
        else:
            tried.append(f"line-anchored (invalid range {s}-{e}, file has "
                         f"{n} lines)")

    # (d) structured failure
    return _fail(tried, text, old_string)


def apply_edit_to_text(text: str, old_string: str, new_string: str,
                       start_line: int | None = None,
                       end_line: int | None = None
                       ) -> tuple[bool, str, dict[str, Any]]:
    """Apply one edit to text via the fallback chain.

    Returns (ok, resulting_text, result).  On failure resulting_text is the
    input text unchanged and result is the structured failure dict.
    """
    res = find_edit_span(text, old_string, start_line, end_line)
    if not res["ok"]:
        return False, text, res
    s, e = res["start"], res["end"]
    new = new_string
    if res["strategy"] == "line-anchored":
        # The anchor span is whole lines; preserve the region's trailing
        # newline convention so we don't glue lines together.
        region = text[s:e]
        if region.endswith("\n") and new and not new.endswith("\n"):
            new += "\n"
    return True, text[:s] + new + text[e:], res


def check_python_syntax(text: str) -> str | None:
    """Return an error string if text is not valid Python, else None.
    Uses py_compile against a temp copy so no __pycache__ is left behind."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as tf:
        tf.write(text)
        tmp = tf.name
    # An explicit cfile keeps py_compile from creating a __pycache__
    # directory next to the temp copy (i.e. littering /tmp forever).
    pyc = tmp + ".pyc"
    try:
        py_compile.compile(tmp, cfile=pyc, doraise=True)
    except py_compile.PyCompileError as exc:
        return str(exc)
    finally:
        for leftover in (tmp, pyc):
            try:
                os.unlink(leftover)
            except OSError:
                pass
    return None


def _atomic_write(p: Path, text: str) -> None:
    try:
        # mkstemp creates 0600; os.replace would then silently downgrade
        # an existing 0644/0755 source file. Keep the target's mode bits.
        mode = p.stat().st_mode & 0o777
    except OSError:
        mode = None
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".editfb-",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def apply_edit_with_fallbacks(path: str, old_string: str, new_string: str,
                               start_line: int | None = None,
                               end_line: int | None = None) -> dict[str, Any]:
    """File-level edit with the full fallback chain.

    Re-reads the file fresh, applies the chain, py_compiles the result for
    .py files, and writes atomically.  Returns a result dict that always
    carries "ok" plus, on success, "strategy", and on failure "tried",
    "context", and "hint".
    """
    p = Path(path)
    if not p.exists():
        return {"ok": False, "tried": ["file not found"],
                "context": f"path does not exist: {p}",
                "hint": "check the file path and retry"}
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "tried": ["read file"],
                "context": f"{p} is not valid UTF-8 text",
                "hint": "refusing to edit: a lossy rewrite would corrupt "
                        "unrelated bytes"}
    except OSError as exc:
        return {"ok": False, "tried": ["read file"],
                "context": f"cannot read {p}: {exc}",
                "hint": "check permissions and retry"}

    ok, new_text, res = apply_edit_to_text(text, old_string, new_string,
                                           start_line, end_line)
    if not ok:
        res["path"] = str(p)
        return res

    if p.suffix == ".py":
        syn_err = check_python_syntax(new_text)
        if syn_err:
            res = dict(res)
            res["ok"] = False
            res["tried"] = res["tried"] + ["py-compile"]
            res["context"] = syn_err
            res["hint"] = ("the edit parses as invalid Python; adjust "
                           "new_string and retry (file left unchanged)")
            return res

    try:
        _atomic_write(p, new_text)
    except OSError as exc:
        res = dict(res)
        res["ok"] = False
        res["tried"] = res["tried"] + ["write file"]
        res["context"] = f"cannot write {p}: {exc}"
        res["hint"] = "check permissions/disk and retry"
        return res

    res = dict(res)
    res["path"] = str(p)
    return res


if __name__ == "__main__":
    _passed = 0

    def check(name: str, cond: bool) -> None:
        global _passed
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")
        _passed += 1

    with tempfile.TemporaryDirectory() as d:
        # (a) exact match works
        f = Path(d) / "a.txt"
        f.write_text("hello world\nsecond line\n", encoding="utf-8")
        r = apply_edit_with_fallbacks(str(f), "hello world", "hi there")
        check("(a) exact: ok", r["ok"] is True)
        check("(a) exact: strategy reported",
              r["strategy"] == "exact")
        check("(a) exact: content",
              f.read_text(encoding="utf-8") == "hi there\nsecond line\n")

        # (b) whitespace-differing old_string succeeds via normalization
        # (delegated to fullagent.fuzzymatch)
        g = Path(d) / "b.py"
        g.write_text("def  foo():\n    x  =  1\n    return x\n",
                     encoding="utf-8")
        r = apply_edit_with_fallbacks(
            str(g), "def foo():\n  x = 1", "def foo():\n    x = 2")
        check("(b) whitespace-normalized: ok", r["ok"] is True)
        check("(b) whitespace-normalized: strategy reported",
              r["strategy"] == "whitespace-normalized")
        check("(b) whitespace-normalized: content",
              g.read_text(encoding="utf-8")
              == "def foo():\n    x = 2\n    return x\n")

        # (b2) indentation-only difference resolves via fuzzymatch too
        g2 = Path(d) / "b2.txt"
        g2.write_text("def foo():\n    x = 1\n    return x\n",
                      encoding="utf-8")
        r = apply_edit_with_fallbacks(
            str(g2), "def foo():\n\tx = 1", "def foo():\n    x = 9")
        check("(b2) fuzzy: ok", r["ok"] is True)
        check("(b2) fuzzy: strategy is a fuzzy one",
              r["strategy"] in ("whitespace-normalized",
                                "indentation-insensitive"))
        check("(b2) fuzzy: content",
              g2.read_text(encoding="utf-8")
              == "def foo():\n    x = 9\n    return x\n")

        # (c) line-anchored edit works when text differs but lines given
        h = Path(d) / "c.txt"
        h.write_text("line one\nline two here\nline three\n",
                     encoding="utf-8")
        r = apply_edit_with_fallbacks(
            str(h), "line 2 here!\nline three?",
            "REPLACED", start_line=2, end_line=3)
        check("(c) line-anchored: ok", r["ok"] is True)
        check("(c) line-anchored: strategy reported",
              r["strategy"] == "line-anchored")
        check("(c) line-anchored: content",
              h.read_text(encoding="utf-8") == "line one\nREPLACED\n")

        # (c2) line-anchored rejected when region does not resemble old
        r = apply_edit_with_fallbacks(
            str(h), "something totally unrelated zzz",
            "X", start_line=1, end_line=1)
        check("(c2) line-anchored mismatch: not ok", r["ok"] is False)
        check("(c2) tried names line-anchored",
              any("line-anchored" in t for t in r["tried"]))
        check("(c2) file unchanged",
              h.read_text(encoding="utf-8") == "line one\nREPLACED\n")

        # (d) total failure returns structured context + hints
        r = apply_edit_with_fallbacks(str(f), "zzz-nope-qqq", "whatever")
        check("(d) failure: not ok", r["ok"] is False)
        check("(d) failure: tried lists strategies",
              isinstance(r["tried"], list) and len(r["tried"]) >= 2
              and any("exact" in t for t in r["tried"]))
        check("(d) failure: context has line numbers",
              "1:" in r["context"] and "2:" in r["context"])
        check("(d) failure: hint present",
              r["hint"] == "re-read the region and retry with exact text")
        check("(d) failure: file untouched",
              f.read_text(encoding="utf-8") == "hi there\nsecond line\n")

        # (d2) ambiguous match is a structured failure, not a crash
        amb = Path(d) / "amb.txt"
        amb.write_text("dup\ndup\n", encoding="utf-8")
        r = apply_edit_with_fallbacks(str(amb), "dup", "x")
        check("(d2) ambiguous: not ok", r["ok"] is False)
        check("(d2) ambiguous: named in tried",
              any("ambiguous" in t for t in r["tried"]))
        check("(d2) ambiguous: file untouched",
              amb.read_text(encoding="utf-8") == "dup\ndup\n")

        # (e) py_compile guard: broken Python is refused, file unchanged
        bad = Path(d) / "bad.py"
        bad.write_text("x = 1\n", encoding="utf-8")
        r = apply_edit_with_fallbacks(str(bad), "x = 1", "def broken(:")
        check("(e) py-compile: not ok", r["ok"] is False)
        check("(e) py-compile: named in tried",
              any("py-compile" in t for t in r["tried"]))
        check("(e) py-compile: file unchanged",
              bad.read_text(encoding="utf-8") == "x = 1\n")

        # (f) missing file is a structured failure
        r = apply_edit_with_fallbacks(str(Path(d) / "nope.txt"), "a", "b")
        check("(f) missing file: not ok", r["ok"] is False)

    print(f"ALL SELF-TESTS PASSED ({_passed} checks)")
