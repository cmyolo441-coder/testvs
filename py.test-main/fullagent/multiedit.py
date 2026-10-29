"""MultiEdit tool: apply several find/replace edits to one file atomically.

All edits are validated against the original file content first through a
fallback chain (exact match, then whitespace-normalized /
indentation-insensitive match, then a line-number-anchored edit when
start_line/end_line are given); only if every ``old_string`` resolves
unambiguously is the file rewritten (once). Any validation failure leaves
the file untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import filestat  # Worker 8/20: mtime+size change detection
from .editfallback import (  # Worker 9/20: fallback edit chain
    check_python_syntax,
    find_edit_span,
)
from .editretry import smart_edit_retry  # Worker 19/20: bounded smart retry
from .tools import (
    RISK_CONFIRM,
    RISK_SAFE,
    Tool,
    _atomic_write_text,
    _checked,
    _rich_edit_miss_error,
    _preview_edit,
)

_DESCRIPTION = (
    "Apply multiple find/replace edits to a single file ATOMICALLY. "
    "Every old_string is validated first against a fallback chain: exact "
    "match, then whitespace-normalized / indentation-insensitive match, "
    "then a line-number-anchored edit when start_line/end_line are given. "
    "A match must be unambiguous. If ANY edit fails validation (missing, "
    "ambiguous, or malformed), the file is left completely unchanged and "
    "the error names the failing edit index plus what was tried. Prefer "
    "this over several sequential edit_file calls — the file never ends up "
    "half-edited."
)

_PARAMETERS = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path of the file to edit.",
        },
        "edits": {
            "type": "array",
            "description": "List of {old_string, new_string} replacements, "
                           "applied in order.",
            "items": {
                "type": "object",
                "properties": {
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to find "
                                       "(must match exactly once).",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-based first line anchoring "
                                       "the edit: when the old_string text "
                                       "does not match, the line range "
                                       "start_line..end_line is used instead "
                                       "if it approximately matches.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-based last line of the "
                                       "anchor range (defaults to "
                                       "start_line).",
                    },
                },
                "required": ["old_string", "new_string"],
            },
        },
        "preview_only": {
            "type": "boolean",
            "description": "Dry-run: validate every edit and show the "
                           "unified diff of what WOULD change (with matched "
                           "line numbers) without modifying the file.",
        },
    },
    "required": ["file_path", "edits"],
}


# [worker 7] Edit-confirmation grounding for large files. When the target
# file has more than _LARGE_FILE_LINES lines, each successful edit reports
# a TARGET REGION block (path, 1-based line range, and the actual current
# lines about to be replaced) so the model sees the verified target region
# before its next edit attempt. Informational only — the edit still applies
# atomically exactly as before.
_LARGE_FILE_LINES = 500


def _locate_fuzzy_region(text: str, start: int, end: int):
    """Return the 1-based (start_line, end_line) span of a fuzzy match
    from its character offsets into `text` (the fuzzy matcher's winning
    span — the region the edit actually replaced), or None when the
    offsets are invalid."""
    if (not isinstance(start, int) or not isinstance(end, int)
            or not 0 <= start < end <= len(text)):
        return None
    start_line = text.count("\n", 0, start) + 1
    # end-1 so a span ending right after a newline doesn't claim the
    # next (empty) line.
    end_line = text.count("\n", 0, end - 1) + 1
    return start_line, end_line


# (not-found misses are already self-diagnosing: _rich_edit_miss_error
# reports the closest actual lines via difflib — see tools.py.)


def _target_region_block(path_str: str, total_lines: int, edit_idx: int,
                         start: int, end: int, region_lines: list) -> str:
    """Format one edit's TARGET REGION info block."""
    numbered = [f"  {n:>6} | {ln}"
                for n, ln in zip(range(start, end + 1), region_lines)]
    return (f"\nTARGET REGION (edit #{edit_idx}): {path_str}, "
            f"lines {start}\u2013{end} of {total_lines}:\n"
            + "\n".join(numbered))


def _stat_sig(p: Path) -> tuple[int, int] | None:
    """(mtime_ns, size) snapshot of a file, or None when it can't be read."""
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _apply_edits(p: Path, edits: list[dict[str, Any]], text: str,
                 preview_only: bool = False):
    """Validate every edit against ``text``, then apply atomically.

    Returns ``(ok, message, fail_index, new_text)``:

    * ``ok`` — True when the edits applied (or the preview rendered).
    * ``message`` — success/preview string, or the ERROR string.
    * ``fail_index`` — the failing edit's index for content-dependent
      failures (miss / ambiguous: a fresh re-read might resolve them);
      None for success and for non-retryable failures (malformed, I/O).
    * ``new_text`` — post-edit content on success, else None.
    """
    n_lines = len(text.splitlines())

    # Validate every edit against the ORIGINAL content before touching
    # anything. Sequential replacement would let one edit shift another's
    # match, so each old_string is matched in `text` as it stood on disk.
    # [worker 1] When an old_string is not byte-identical, fuzzy matching
    # is tried before failing: a confident whitespace/indentation-
    # insensitive match is recorded and applied; otherwise the edit falls
    # through to the self-diagnosing miss error (which already carries
    # close-match suggestions).
    resolved: list[dict[str, Any]] = []  # find_edit_span results
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return (False,
                    f"ERROR: edit #{i} is malformed; expected "
                    f"{{\"old_string\", \"new_string\"}}",
                    None, None)
        old = edit.get("old_string")
        if not isinstance(old, str) or not old:
            return (False,
                    f"ERROR: edit #{i}: old_string must be a non-empty string",
                    None, None)
        if "new_string" not in edit or not isinstance(edit["new_string"], str):
            return (False,
                    f"ERROR: edit #{i}: new_string must be a string",
                    None, None)
        sl = edit.get("start_line")
        el = edit.get("end_line")
        for _name, _val in (("start_line", sl), ("end_line", el)):
            if (_val is not None
                    and (isinstance(_val, bool) or not isinstance(_val, int)
                         or _val < 1)):
                return (False,
                        f"ERROR: edit #{i}: {_name} must be a positive "
                        f"integer (no changes were made)",
                        None, None)
        # An ambiguous exact match stays a hard error unless the caller
        # disambiguates with an explicit line anchor.
        if text.count(old) > 1 and sl is None and el is None:
            return (False,
                    f"ERROR: edit #{i} failed validation: old_string matches "
                    f"{text.count(old)} places; add more context to make it "
                    f"unique (no changes were made)",
                    i, None)
        res = find_edit_span(text, old, sl, el)
        if not res["ok"]:
            tried = "; ".join(res["tried"])
            return (False,
                    _rich_edit_miss_error(p, text, old,
                                          label=f"edit #{i}: ")
                    + "\n(no changes were made)"
                    + f"\nfallback strategies tried: {tried}"
                    + f"\nhint: {res['hint']}",
                    i, None)
        resolved.append(res)

    # Validate that resolved spans do not overlap in the original text.
    spans = sorted((resolved[i]["start"], resolved[i]["end"]) for i in range(len(edits)))
    for a in range(1, len(spans)):
        if spans[a][0] < spans[a - 1][1]:
            return (False,
                    "ERROR: edits overlap in the original text "
                    "(no changes were made)", None, None)
    new_text = text
    detail_lines = []
    shift = 0  # cumulative offset delta from earlier edits in this batch
    strategies: list[str] = []  # [worker 9] winning chain strategy per edit
    fuzzy_indices: list[int] = []  # [worker 1] fuzzy-applied edit indices
    order = sorted(range(len(edits)), key=lambda i: resolved[i]["start"])
    for i in order:
        edit = edits[i]
        res = resolved[i]
        # Each old_string resolved to one confident span in `text`
        # (offsets into the ORIGINAL text, adjusted by `shift` from
        # earlier edits in this batch).
        strategy = res["strategy"]
        strategies.append(f"#{i} {strategy}")
        s, e = res["start"] + shift, res["end"] + shift
        new = edit["new_string"]
        if strategy == "line-anchored":
            # Whole-line anchor span: preserve the original region's
            # trailing newline so lines don't glue together.
            orig_region = text[res["start"]:res["end"]]
            if orig_region.endswith("\n") and new and not new.endswith("\n"):
                new += "\n"
        start_line = new_text.count("\n", 0, s) + 1
        if strategy == "exact":
            detail_lines.append(
                f"  edit #{i}: 1 occurrence(s) WOULD be replaced "
                f"at line(s) [{start_line}]")
        elif strategy == "line-anchored":
            a, b = res["lines"]
            detail_lines.append(
                f"  edit #{i}: line-anchored (lines {a}-{b}) span "
                f"WOULD be replaced at line(s) [{start_line}]")
        else:  # whitespace-normalized / indentation-insensitive
            fuzzy_indices.append(i)
            detail_lines.append(
                f"  edit #{i}: fuzzy-matched (whitespace differed) span "
                f"WOULD be replaced at line(s) [{start_line}]")
        new_text = new_text[:s] + new + new_text[e:]
        shift += len(new) - (e - s)
    if preview_only:
        return True, _preview_edit(p, text, new_text, detail_lines), None, \
            new_text
    # [worker 9] Never write a syntactically broken .py file: py_compile
    # the result first; on failure the file is left completely unchanged.
    # This gate also covers the smart-retry path (it calls _apply_edits).
    if p.suffix == ".py":
        syn_err = check_python_syntax(new_text)
        if syn_err:
            return (False,
                    f"ERROR: resulting file would have Python syntax "
                    f"errors; no changes were made: {syn_err}",
                    None, None)
    try:
        _atomic_write_text(p, new_text)
    except OSError as e:
        return False, f"ERROR: {e}", None, None
    # Worker 8/20: record post-edit stat so a chained edit doesn't warn
    # about our own change.
    filestat.note_written(None, str(p))
    result = f"Applied {len(edits)} edits to {p}"
    # [worker 1] Note which edits applied via fuzzy matching so the caller
    # knows the old_string was not byte-identical.
    if fuzzy_indices:
        nums = ", ".join(f"#{i}" for i in fuzzy_indices)
        result += (f" (fuzzy-matched (whitespace differed): edits {nums})")
    # [worker 9] Report the winning fallback-chain strategy per edit, so a
    # caller can see exactly how each old_string was resolved.
    result += " [match: " + ", ".join(strategies) + "]"
    # [worker 7] TARGET REGION grounding for large files: resolve each
    # edit's verified location against the ORIGINAL content (which is what
    # validation matched) and report it, so the model sees the true region
    # and line numbers. Informational only — the edit already applied.
    if n_lines > _LARGE_FILE_LINES:
        file_lines = text.splitlines()
        for i, res in enumerate(resolved):
            span = _locate_fuzzy_region(text, res["start"], res["end"])
            if span is None:  # cannot happen — validation passed above
                continue
            start, end = span
            result += _target_region_block(str(p), n_lines, i, start, end,
                                           file_lines[start - 1:end])
    return True, result, None, new_text


def _smart_retry_multiedit(p: Path, edits: list[dict[str, Any]],
                           fail_idx: int, preview_only: bool,
                           read_sig: tuple[int, int] | None) -> str | None:
    """Bounded fresh-content retry for a stale MultiEdit miss.

    The 512s incident was a blind immediate retry of the same stale
    content. This retry is the opposite: it only runs when the file
    actually changed since our read (stat gate — an unchanged file
    returns None immediately: no waiting, no extra reads), then waits
    1s/2s, re-reads fresh from disk, fuzzy re-matches, and re-validates
    ALL edits atomically. At most 2 smart retries, then None — the caller
    returns the original error. A retry-driven apply to a .py file is
    syntax-checked with py_compile inside _apply_edits and refused
    (never written) on failure.
    """
    if _stat_sig(p) == read_sig:
        return None  # unchanged since our read: retrying is the blind loop

    def attempt(path: str, old: str, new: str) -> str:
        # `old` is the fuzzy-resolved old_string for the failing edit
        # (or the original when fuzzy found nothing new). Re-validate
        # ALL edits against fresh disk content and apply atomically —
        # never a partial write.
        try:
            fresh = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            return f"ERROR: re-read failed during smart retry: {e}"
        trial = [dict(e) for e in edits]
        trial[fail_idx] = {"old_string": old, "new_string": new}
        # The core py_compile gate fires inside _apply_edits: a
        # syntax-breaking retry makes no write and reports nothing.
        ok, message, _fi, _new_text = _apply_edits(p, trial, fresh,
                                                   preview_only)
        if not ok:
            return message
        return message + " [smart retry: re-read fresh content]"

    failing = edits[fail_idx]
    outcome = smart_edit_retry(str(p), failing["old_string"],
                               failing["new_string"], attempt)
    return outcome["message"] if outcome["success"] else None


def multi_edit(file_path: str, edits: list[dict[str, Any]],
               preview_only: bool = False) -> str:
    """Apply all edits atomically; validate everything before writing.

    When preview_only=true the file is left untouched and a unified
    before/after diff of exactly what WOULD change is returned instead,
    including the matched line number of each edit — so the caller can
    verify every match before committing.

    On a content-mismatch failure (miss / ambiguous old_string) a bounded
    smart retry runs when the file changed since it was read: brief
    backoff, fresh re-read, fuzzy re-match, full re-validation — at most 2
    retries, then the original error is returned.
    """
    p, err = _checked(file_path)
    if err:
        return err
    if not p.exists():
        return f"ERROR: file not found: {p}"
    if not isinstance(edits, list) or not edits:
        return "ERROR: edits must be a non-empty list of {old_string, new_string}"
    # Worker 8/20 (512s incident): if the file changed since the agent last
    # read it, say so plainly instead of failing validation cryptically.
    warn = filestat.check_stale(None, str(p))
    if warn:
        return warn
    try:
        text = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ("ERROR: file is not valid UTF-8 text; refusing to edit "
                "(lossy rewrite would corrupt unrelated bytes)")
    except OSError as e:
        return f"ERROR: {e}"
    read_sig = _stat_sig(p)  # Worker 19/20: stat gate for the smart retry

    ok, message, fail_index, _new_text = _apply_edits(p, edits, text,
                                                      preview_only)
    if ok or fail_index is None:
        return message
    retried = _smart_retry_multiedit(p, edits, fail_index, preview_only,
                                     read_sig)
    return retried if retried is not None else message


def build_tool() -> Tool:
    risk = RISK_CONFIRM if isinstance(RISK_CONFIRM, str) else RISK_SAFE
    return Tool("MultiEdit", _DESCRIPTION, _PARAMETERS, multi_edit,
                risk=risk)


def register(agent) -> None:
    """Register the MultiEdit tool on an agent (``agent.tools`` dict)."""
    agent.tools["MultiEdit"] = build_tool()


if __name__ == "__main__":
    import tempfile

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "sample.txt"
        original = "alpha one\nbeta two\ngamma three\n"
        f.write_text(original, encoding="utf-8")

        # 1. happy path: 3 edits
        out = multi_edit(
            str(f),
            [{"old_string": "alpha", "new_string": "ALPHA"},
             {"old_string": "beta two", "new_string": "BETA TWO"},
             {"old_string": "gamma three", "new_string": "gamma 3"}],
        )
        check("happy-path result message",
              out.startswith(f"Applied 3 edits to {f}")
              and "[match: #0 exact, #1 exact, #2 exact]" in out)
        check("happy-path content",
              f.read_text(encoding="utf-8") == "ALPHA one\nBETA TWO\ngamma 3\n")

        # 2. failing middle edit leaves file unchanged
        before = f.read_text(encoding="utf-8")
        out = multi_edit(
            str(f),
            [{"old_string": "ALPHA", "new_string": "alpha"},
             {"old_string": "NOPE-NOT-THERE", "new_string": "x"},
             {"old_string": "gamma 3", "new_string": "gamma three"}],
        )
        check("middle-failure names edit index",
              "edit #1" in out and "no changes were made" in out.lower() or
              "no changes were made" in out.lower())
        check("file unchanged after middle failure",
              f.read_text(encoding="utf-8") == before)

        # 3. duplicate old_string error
        g = Path(d) / "dup.txt"
        g.write_text("aaa\naaa\n", encoding="utf-8")
        out = multi_edit(str(g),
                         [{"old_string": "aaa", "new_string": "b"}])
        check("duplicate old_string rejected",
              out.startswith("ERROR") and "matches 2 places" in out)
        check("file unchanged after duplicate failure",
              g.read_text(encoding="utf-8") == "aaa\naaa\n")

        # 4. missing file refused
        out = multi_edit(str(Path(d) / "missing.txt"),
                         [{"old_string": "a", "new_string": "b"}])
        check("missing file refused", out.startswith("ERROR"))

    # 5. register() wires into agent.tools with confirm risk
    class FakeAgent:
        def __init__(self):
            self.tools = {}

    a = FakeAgent()
    register(a)
    t = a.tools.get("MultiEdit")
    check("register sets agent.tools['MultiEdit']", t is not None)
    check("tool risk is confirm", t.risk == "confirm")
    check("handler works through registry",
          t.handler(str(Path(d) / "missing.txt"),
                    [{"old_string": "a", "new_string": "b"}]).startswith("ERROR"))

    # 6. failing edit is self-diagnosing (the incident case): path, line
    # count, numbered excerpt, and a suggestion — so one retry suffices
    # instead of a blind retry loop. (Own temp dir: the one above is
    # already closed.)
    # [worker 1] NOTE: a *whitespace-only* miss like "line  two" no longer
    # reaches this path — it now fuzzy-applies (see test 9). A genuine
    # content miss still self-diagnoses.
    with tempfile.TemporaryDirectory() as d2:
        h = Path(d2) / "diag.txt"
        h.write_text("line one\nline two\nline three\n", encoding="utf-8")
        out = multi_edit(str(h), [{"old_string": "line  twoo",
                                   "new_string": "x"}])
        check("failing edit names edit index and stays an ERROR",
              out.startswith("ERROR:") and "edit #0" in out)
        check("failing edit names the file", str(h) in out)
        check("failing edit gives the line count", "3 line(s) total" in out)
        check("failing edit shows a line-numbered excerpt",
              "→" in out and "line 2" in out)
        check("failing edit includes suggestions", "SUGGESTIONS" in out)
        check("failing edit leaves the file untouched",
              h.read_text(encoding="utf-8")
              == "line one\nline two\nline three\n")

    # 7. [worker 7] TARGET REGION grounding: >500-line file gets the block
    # with correct line numbers and the true current content.
    with tempfile.TemporaryDirectory() as d3:
        big = Path(d3) / "big.txt"
        lines = [f"filler line {i}" for i in range(1, 601)]
        lines[249] = "TARGET-UNIQUE-ABC marker here"
        lines[399] = "multi-A"
        lines[400] = "multi-B"
        big.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = multi_edit(
            str(big),
            [{"old_string": "TARGET-UNIQUE-ABC marker here",
              "new_string": "TARGET-UNIQUE-ABC replaced"},
             {"old_string": "multi-A\nmulti-B",
              "new_string": "multi-C\nmulti-D"}])
        check("large file: TARGET REGION block present",
              "TARGET REGION (edit #0)" in out
              and "TARGET REGION (edit #1)" in out)
        check("large file: correct single-line range of 600",
              "lines 250\u2013250 of 600" in out)
        check("large file: correct multi-line range of 600",
              "lines 400\u2013401 of 600" in out)
        check("large file: block shows the true pre-edit content",
              "TARGET-UNIQUE-ABC marker here" in out
              and "multi-A" in out and "multi-B" in out
              and "| TARGET-UNIQUE-ABC marker here" in out)
        check("large file: edit still applied",
              "TARGET-UNIQUE-ABC replaced" in
              big.read_text(encoding="utf-8"))
        # miss on a large file stays a self-diagnosing ERROR
        out2 = multi_edit(str(big), [{"old_string": "NOPE-NOT-HERE",
                                      "new_string": "x"}])
        check("large file: miss is a self-diagnosing ERROR",
              out2.startswith("ERROR:") and "TARGET REGION" not in out2)

    # 8. [worker 7] small file (<500 lines) does NOT get the block
    with tempfile.TemporaryDirectory() as d4:
        small = Path(d4) / "small.txt"
        small.write_text("\n".join(f"row {i}" for i in range(1, 101))
                         + "\n", encoding="utf-8")
        out = multi_edit(str(small), [{"old_string": "row 50",
                                       "new_string": "ROW 50"}])
        check("small file: no TARGET REGION block",
              out.startswith(f"Applied 1 edits to {small}")
              and "[match: #0 exact]" in out
              and "TARGET REGION" not in out)

    # 9. [worker 1] fuzzy matching: old_string with different
    # indentation/whitespace applies instead of erroring, and the result
    # notes the fuzzy match. The 512s incident case: model reproduces code
    # from memory with wrong indentation.
    with tempfile.TemporaryDirectory() as d5:
        z = Path(d5) / "fuzzy.txt"
        z.write_text("def foo():\n    x = 1\n    return x + 1\n",
                     encoding="utf-8")
        out = multi_edit(
            str(z),
            [{"old_string": "def foo():\n  x = 1\n\treturn x + 1\n",
              "new_string": "def foo():\n    x = 2\n    return x + 1\n"}])
        check("fuzzy edit applies with whitespace-differ note",
              out.startswith(f"Applied 1 edits to {z}")
              and "fuzzy-matched (whitespace differed)" in out
              and "edits #0" in out)
        check("fuzzy edit replaced the true span",
              z.read_text(encoding="utf-8")
              == "def foo():\n    x = 2\n    return x + 1\n")
        # preview_only also resolves the fuzzy span
        out = multi_edit(
            str(z),
            [{"old_string": "def foo():\n        x = 2\n    return x + 1\n",
              "new_string": "def foo():\n    x = 3\n    return x + 1\n"}],
            preview_only=True)
        check("fuzzy preview shows the matched line",
              "fuzzy-matched (whitespace differed)" in out
              and "line(s) [1]" in out)
        check("fuzzy preview leaves file untouched",
              z.read_text(encoding="utf-8")
              == "def foo():\n    x = 2\n    return x + 1\n")

    # 9. [worker 19/20] smart retry: stat gate + bounded fresh-content
    # retry + py_compile check after a retry-driven .py apply.
    with tempfile.TemporaryDirectory() as d5:
        # 9a. unchanged file -> stat gate returns None immediately
        # (no waiting, no extra reads — no blind loop).
        s = Path(d5) / "stable.txt"
        s.write_text("aaa\n", encoding="utf-8")
        check("retry stat gate: unchanged file -> None immediately",
              _smart_retry_multiedit(
                  s, [{"old_string": "zzz", "new_string": "q"}],
                  0, False, _stat_sig(s)) is None)
        check("retry stat gate: file untouched",
              s.read_text(encoding="utf-8") == "aaa\n")
        # 9b. changed file -> retry re-reads fresh, fuzzy-resolves the
        # drifted old_string, and applies. (Bogus read_sig forces the
        # retry path, simulating "file changed after our read".)
        r = Path(d5) / "retryme.py"
        r.write_text("def f():\n\treturn 1\n", encoding="utf-8")
        out = _smart_retry_multiedit(
            r, [{"old_string": "def f():\n    return 1",
                 "new_string": "def f():\n    return 2"}],
            0, False, (0, 0))
        check("retry applies fuzzy-resolved edit",
              out is not None and "Applied 1 edits" in out
              and "smart retry" in out)
        check("retry wrote the replacement",
              "return 2" in r.read_text(encoding="utf-8"))
        # 9c. retry that would introduce a syntax error is refused by the
        # core py_compile gate (unconditional): no write happens.
        r2 = Path(d5) / "bad.py"
        r2.write_text("x = 1\n", encoding="utf-8")
        out = _smart_retry_multiedit(
            r2, [{"old_string": "x = 1", "new_string": "x = "}],
            0, False, (0, 0))
        check("retry refuses syntax-breaking edit",
              out is None)
        check("retry leaves the file untouched",
              r2.read_text(encoding="utf-8") == "x = 1\n")
        # 9d. full multi_edit path: genuine miss on an unchanged file
        # returns the original self-diagnosing error with no retry delay.
        import time as _t
        t0 = _t.time()
        out = multi_edit(str(s), [{"old_string": "zzz-not-there",
                                   "new_string": "q"}])
        dt = _t.time() - t0
        check("multi_edit genuine miss: original error preserved",
              out.startswith("ERROR:") and "no changes were made" in out)
        check(f"multi_edit genuine miss: no retry delay ({dt:.2f}s < 1s)",
              dt < 1.0)
        check("multi_edit genuine miss: file untouched",
              s.read_text(encoding="utf-8") == "aaa\n")

    # 10. [worker 9] fallback chain: line-number-anchored edit. The
    # old_string text differs from the file, but start_line/end_line pin
    # the region and it approximately matches, so the edit applies and
    # the result reports the winning strategy.
    with tempfile.TemporaryDirectory() as d6:
        la = Path(d6) / "anchored.txt"
        la.write_text("line one\nline two here\nline three\n",
                      encoding="utf-8")
        out = multi_edit(
            str(la),
            [{"old_string": "line 2 here!\nline three?",
              "new_string": "REPLACED",
              "start_line": 2, "end_line": 3}])
        check("line-anchored edit applies",
              out.startswith(f"Applied 1 edits to {la}")
              and "[match: #0 line-anchored]" in out)
        check("line-anchored edit replaced the true lines",
              la.read_text(encoding="utf-8") == "line one\nREPLACED\n")
        # line anchor that does NOT resemble old_string fails safe
        out = multi_edit(
            str(la),
            [{"old_string": "something totally unrelated zzz",
              "new_string": "X", "start_line": 1, "end_line": 1}])
        check("line-anchored mismatch stays an ERROR",
              out.startswith("ERROR:") and "no changes were made" in out)
        check("line-anchored mismatch names the strategy tried",
              "line-anchored" in out and "fallback strategies tried" in out
              and "hint:" in out)
        check("line-anchored mismatch leaves file untouched",
              la.read_text(encoding="utf-8") == "line one\nREPLACED\n")
        # invalid line args are rejected without touching the file
        out = multi_edit(
            str(la),
            [{"old_string": "line one", "new_string": "x",
              "start_line": 0}])
        check("invalid start_line rejected",
              out.startswith("ERROR:") and "start_line" in out)
        check("invalid start_line leaves file untouched",
              la.read_text(encoding="utf-8") == "line one\nREPLACED\n")

    # 11. [worker 9] py_compile guard: an edit that would break a .py
    # file is refused BEFORE writing; the file is left unchanged.
    with tempfile.TemporaryDirectory() as d7:
        py = Path(d7) / "mod.py"
        py.write_text("x = 1\ny = 2\n", encoding="utf-8")
        out = multi_edit(str(py), [{"old_string": "x = 1",
                                    "new_string": "def broken(:"}])
        check("py_compile refusal is an ERROR",
              out.startswith("ERROR:") and "syntax" in out.lower()
              and "no changes were made" in out)
        check("py_compile refusal leaves the file unchanged",
              py.read_text(encoding="utf-8") == "x = 1\ny = 2\n")
        # a valid .py edit still applies
        out = multi_edit(str(py), [{"old_string": "x = 1",
                                    "new_string": "x = 10"}])
        check("valid .py edit applies",
              out.startswith(f"Applied 1 edits to {py}")
              and "[match: #0 exact]" in out)
        check("valid .py edit content",
              py.read_text(encoding="utf-8") == "x = 10\ny = 2\n")

    # 12. [worker 9] mixed strategies in one atomic batch are each
    # reported: exact + fuzzy + line-anchored.
    with tempfile.TemporaryDirectory() as d8:
        mx = Path(d8) / "mixed.txt"
        mx.write_text("aaa one\nbbb  two\nccc three\n", encoding="utf-8")
        out = multi_edit(
            str(mx),
            [{"old_string": "aaa one", "new_string": "AAA ONE"},
             {"old_string": "bbb two", "new_string": "BBB TWO"},
             {"old_string": "see three~", "new_string": "CCC",
              "start_line": 3, "end_line": 3}])
        check("mixed batch reports every strategy",
              out.startswith(f"Applied 3 edits to {mx}")
              and "[match: #0 exact, #1 whitespace-normalized, "
                  "#2 line-anchored]" in out)
        check("mixed batch content",
              mx.read_text(encoding="utf-8")
              == "AAA ONE\nBBB TWO\nCCC\n")

    print("ALL SELF-TESTS PASSED")
