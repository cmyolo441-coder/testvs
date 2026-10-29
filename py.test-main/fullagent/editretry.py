"""Smart edit retry: bounded, fresh-content, fuzzy re-match retries for edits.

Incident (512s turn): the model retried a failing edit IMMEDIATELY with the
same stale ``old_string``, dozens of times — blind hammering that could
never succeed because the file on disk had changed underneath it.

This module does the opposite. :func:`smart_edit_retry` wraps an edit
function ``edit_fn(path, old_string, new_string) -> str`` (the fullagent
convention: a result that does not start with ``"ERROR"`` is success):

1. Attempt 1 runs ``edit_fn`` as-is.
2. On a *content-mismatch* failure (``old_string`` not found, or ambiguous
   match — both depend on what's on disk right now), it waits a brief
   backoff (1s, then 2s, reusing :mod:`smartretry`'s exponential schedule),
   RE-READS the file fresh from disk, fuzzy-matches ``old_string`` against
   the fresh content, and re-attempts with the fuzzy-resolved match.
3. At most **2** smart retries (3 ``edit_fn`` invocations total, hard cap),
   then a final structured failure carrying the fresh file context.

Fail-fast (no retry) on sandbox/path errors, UTF-8 errors, malformed
params, or an ``edit_fn`` that raises for a non-content reason — re-reading
the file cannot fix those.

Only stdlib is used. No ``_EMBEDDED_KEYS`` touched.
"""

from __future__ import annotations

import difflib
import time
from pathlib import Path
from typing import Any, Callable

from .fuzzymatch import find_best_match
from .smartretry import backoff_delay

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SMART_RETRIES = 2          # hard cap on smart retries (task requirement)
BASE_DELAY = 1.0               # first retry waits ~1s, second ~2s
RETRY_MAX_DELAY = 8.0          # cap per backoff wait
CONTEXT_LINES = 25             # lines of fresh context in the final failure

EditFn = Callable[[str, str, str], str]


# ---------------------------------------------------------------------------
# Fuzzy matching against fresh content
# ---------------------------------------------------------------------------


def fuzzy_find(old_string: str, fresh_text: str) -> str | None:
    """Resolve ``old_string`` against fresh file content.

    Delegates to :mod:`fuzzymatch`'s strategy chain (exact, then
    whitespace-normalized, then indentation-insensitive), with one extra
    guard: the resolved span must occur exactly once in the fresh content,
    otherwise the retry would be ambiguous and we return ``None``.
    """
    if not old_string or not fresh_text:
        return None
    if fresh_text.count(old_string) == 1:
        return old_string
    start, end, _suggestions = find_best_match(fresh_text, old_string)
    if start is None or end is None:
        return None
    cand = fresh_text[start:end]
    if fresh_text.count(cand) != 1:
        return None
    return cand


def _is_retryable_error(message: Any) -> bool:
    """True only for content-mismatch failures worth a smart retry.

    "not found" (file or old_string) and "matches N places" both depend on
    current disk content, so a re-read may resolve them. Everything else —
    sandbox/path errors, UTF-8 errors, malformed params — fails fast.
    """
    if not isinstance(message, str) or not message.startswith("ERROR"):
        return False
    m = message.lower()
    return ("not found" in m) or ("matches" in m and "places" in m)


def fresh_context_snippet(fresh_text: str, old_string: str,
                          max_lines: int = CONTEXT_LINES) -> str:
    """Short fresh-file excerpt for the final failure, centered on the
    closest fuzzy region when one exists, else the head of the file."""
    lines = fresh_text.splitlines()
    total = len(lines)
    center: int | None = None
    if old_string:
        old_lines = old_string.splitlines()
        n = len(old_lines)
        best_ratio, best_i = 0.0, None
        for i in range(max(0, len(lines) - n + 1)):
            cand = "\n".join(lines[i:i + n]) if n else ""
            r = difflib.SequenceMatcher(None, old_string, cand).ratio()
            if r > best_ratio:
                best_ratio, best_i = r, i
        if best_i is not None and best_ratio >= 0.4:
            center = best_i
    if center is None:
        start = 0
    else:
        start = max(0, center - max_lines // 2)
    excerpt = lines[start:start + max_lines]
    head = (f"fresh file content ({total} line(s) total"
            + (f", showing lines {start + 1}-{start + len(excerpt)}"
               if total > len(excerpt) else "") + "):")
    return head + "\n" + "\n".join(excerpt)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def smart_edit_retry(path: str, old_string: str, new_string: str,
                     edit_fn: EditFn, *, max_retries: int = MAX_SMART_RETRIES,
                     backoff_base: float = BASE_DELAY,
                     sleep_fn: Callable[[float], None] = time.sleep) -> dict:
    """Retry a failing edit with backoff + fresh re-read + fuzzy re-match.

    ``edit_fn(path, old, new)`` follows the fullagent convention: it
    returns a string, success iff it does not start with ``"ERROR"``.
    Never invokes ``edit_fn`` more than ``1 + max_retries`` times
    (default: 3 — 1 initial attempt + 2 smart retries).

    Returns a dict: ``success``, ``attempts`` (edit_fn invocations),
    ``message`` (final human-readable message), ``attempts_log``
    (per-attempt records), and ``fresh_context`` (fresh file excerpt on
    final failure, else ``None``).
    """
    max_retries = max(0, min(int(max_retries), MAX_SMART_RETRIES))
    attempts_log: list[dict] = []
    invocations = 0
    last_error: str | None = None
    fresh_text: str | None = None

    def _attempt(old: str, waited_before: float,
                note: str) -> tuple[bool, str]:
        nonlocal invocations
        invocations += 1
        try:
            msg = edit_fn(path, old, new_string)
        except Exception as e:  # an exploding edit_fn is a failure, not a crash
            msg = f"ERROR: edit_fn raised {type(e).__name__}: {e}"
        ok = isinstance(msg, str) and not msg.startswith("ERROR")
        attempts_log.append({
            "attempt": invocations,
            "waited_before_s": round(waited_before, 3),
            "matched_old": None if old == old_string else old,
            "error": None if ok else msg,
            "note": note,
        })
        return ok, msg if isinstance(msg, str) else str(msg)

    # --- attempt 1: as-is, no waiting --------------------------------------
    ok, msg = _attempt(old_string, 0.0, "initial attempt")
    if ok:
        return _result(True, invocations, msg, attempts_log, None)
    last_error = msg
    if not _is_retryable_error(msg):
        attempts_log.append({"attempt": "fail-fast",
                             "note": "non-content error — no retry"})
        return _result(False, invocations, msg, attempts_log, None)

    # --- smart retries: backoff, fresh re-read, fuzzy re-match --------------
    for retry in range(1, max_retries + 1):
        wait = backoff_delay(retry - 1, backoff_base, RETRY_MAX_DELAY)
        sleep_fn(wait)
        try:
            fresh_text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as e:
            fresh_text = None
            note = f"re-read failed ({e}); retrying with original old_string"
            use_old = old_string
        else:
            matched = fuzzy_find(old_string, fresh_text)
            use_old = matched if matched is not None else old_string
            if matched is None:
                note = ("fresh re-read done; no unique fuzzy match — "
                        "retrying with original old_string")
            elif matched == old_string:
                note = "fresh re-read done; exact match present — retrying"
            else:
                note = ("fresh re-read done; fuzzy match resolved "
                        "(similarity drift) — retrying with resolved text")
        ok, msg = _attempt(use_old, wait, note)
        if ok:
            return _result(True, invocations, msg, attempts_log, None)
        last_error = msg
        if not _is_retryable_error(msg):
            attempts_log.append({"attempt": "fail-fast",
                                 "note": "non-content error — no retry"})
            break

    context = (fresh_context_snippet(fresh_text, old_string)
               if fresh_text is not None else None)
    final_msg = (
        f"ERROR: edit failed after {invocations} attempt(s) "
        f"(1 initial + {invocations - 1} smart retr"
        f"{'y' if invocations - 1 == 1 else 'ies'}); giving up. "
        f"Last error: {last_error}"
        + (f"\n{context}" if context else
           "\n(no fresh file content available — the file could not be re-read)")
    )
    return _result(False, invocations, final_msg, attempts_log, context)


def _result(success: bool, attempts: int, message: str,
            attempts_log: list[dict], fresh_context: str | None) -> dict:
    return {
        "success": success,
        "attempts": attempts,
        "message": message,
        "attempts_log": attempts_log,
        "fresh_context": fresh_context,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest() -> int:
    """Incident simulation + budget proofs. Prints PASS lines, exit code."""
    import tempfile

    failures: list[str] = []

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL") + f" — {name}")
        if not cond:
            failures.append(name)

    no_sleep = lambda s: None  # noqa: E731 — deterministic, no real waiting

    # --- fuzzy_find unit checks (delegates to fuzzymatch chain) -----------
    check("fuzzy exact-unique short-circuits",
          fuzzy_find("abc", "xxabcxx") == "abc")
    check("fuzzy resolves tab-vs-spaces drift",
          fuzzy_find("\tx = 1", "y = 0\n    x = 1\nz = 2") == "\n    x = 1")
    check("fuzzy no candidate -> None",
          fuzzy_find("qqq-not-there", "aaa\nbbb\n") is None)
    check("fuzzy ambiguous -> None",
          fuzzy_find("x = 1", "x = 1\nx = 1 \n") is None)
    check("fuzzy multi-line indentation drift",
          fuzzy_find("def f():\n\treturn 1",
                     "def f():\n    return 1\n")
          == "def f():\n    return 1")

    # --- 1. incident simulation: content changes between attempts ----------
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "victim.txt"
        stale = "header\n"
        f.write_text(stale, encoding="utf-8")
        old = "def compute(x):\n    return x * 2"
        new = "def compute(x):\n    return x * 3"
        calls: list[tuple[str, str]] = []  # (old_seen, file_text_seen)

        def incident_edit_fn(path: str, o: str, n: str) -> str:
            text = Path(path).read_text(encoding="utf-8")
            calls.append((o, text))
            if text.count(o) != 1:
                return ("ERROR: old_string not found in file "
                        "(it must match exactly, including indentation)")
            Path(path).write_text(text.replace(o, n, 1),
                                   encoding="utf-8")
            return f"Updated {path} — 1 occurrence(s) replaced"

        # The external change lands exactly while we back off between
        # attempt 1 and the first smart retry — like the other tool/process
        # in the 512s incident.
        def sleep_and_change(s: float) -> None:
            if len(calls) == 1:  # only between attempt 1 and retry 1
                f.write_text("header\ndef compute(x):\n"
                             "        return x * 2  \nfooter\n",
                             encoding="utf-8")

        r = smart_edit_retry(str(f), old, new, incident_edit_fn,
                             sleep_fn=sleep_and_change)
        check("incident: retry succeeds on fresh content",
              r["success"] is True)
        check("incident: exactly 2 edit_fn invocations",
              r["attempts"] == 2 and len(calls) == 2)
        check("incident: attempt 1 saw STALE content",
              calls[0][1] == stale)
        check("incident: retry re-read FRESH content from disk",
              calls[1][1] != stale
              and "return x * 2" in calls[1][1])
        check("incident: retry used fuzzy-resolved old_string",
              calls[1][0] != old
              and "return x * 2" in calls[1][0]
              and calls[1][0] in calls[1][1])
        check("incident: file now contains the replacement",
              "return x * 3" in f.read_text(encoding="utf-8"))
        check("incident: log records backoff + fuzzy note",
              r["attempts_log"][1]["waited_before_s"] > 0
              and "fuzzy" in r["attempts_log"][1]["note"])

        # --- 2. persistently failing edit: strict budget -------------------
        g = Path(d) / "stubborn.txt"
        g.write_text("nothing relevant here\n", encoding="utf-8")
        invocations = {"n": 0}

        def always_fails(path: str, o: str, n: str) -> str:
            invocations["n"] += 1
            return "ERROR: old_string not found in file"

        r2 = smart_edit_retry(str(g), "zzz-never-there", "q",
                              always_fails, sleep_fn=no_sleep)
        check("budget: stops after exactly 3 invocations",
              invocations["n"] == 3 and r2["attempts"] == 3)
        check("budget: reports failure", r2["success"] is False)
        check("budget: final message names the attempt count",
              "3 attempt(s)" in r2["message"])
        check("budget: final failure carries fresh file context",
              r2["fresh_context"] is not None
              and "nothing relevant here" in r2["fresh_context"])
        waits = [e["waited_before_s"] for e in r2["attempts_log"]
                 if isinstance(e["attempt"], int)][1:]
        check("budget: backoff waits follow 1s, 2s schedule (+jitter)",
              len(waits) == 2
              and 1.0 <= waits[0] <= 1.0 * 1.20 + 0.001
              and 2.0 <= waits[1] <= 2.0 * 1.20 + 0.001
              and waits[1] > waits[0])

        # --- 3. fail-fast on non-content errors -----------------------------
        def bad_path(path: str, o: str, n: str) -> str:
            return "ERROR: path outside the allowed workspace"

        r3 = smart_edit_retry(str(g), "a", "b", bad_path,
                              sleep_fn=no_sleep)
        check("fail-fast: single invocation on sandbox error",
              r3["attempts"] == 1 and r3["success"] is False)
        check("fail-fast: no fresh context needed",
              r3["fresh_context"] is None)

        # --- 4. immediate success: no retry at all ---------------------------
        h = Path(d) / "fine.txt"
        h.write_text("keep this\n", encoding="utf-8")

        def fine(path: str, o: str, n: str) -> str:
            return "Updated (no change needed)"

        r4 = smart_edit_retry(str(h), "keep this", "keep this", fine,
                              sleep_fn=no_sleep)
        check("success-first-try: 1 invocation, success",
              r4["success"] is True and r4["attempts"] == 1)

        # --- 5. raising edit_fn is a failure, still bounded ------------------
        def raiser(path: str, o: str, n: str) -> str:
            raise ValueError("boom")

        r5 = smart_edit_retry(str(h), "keep this", "x", raiser,
                              sleep_fn=no_sleep)
        check("raising edit_fn: bounded failure, no crash",
              r5["success"] is False and r5["attempts"] == 1
              and "raised ValueError" in r5["message"])

    print("PASS" if not failures else f"{len(failures)} FAILURES")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
