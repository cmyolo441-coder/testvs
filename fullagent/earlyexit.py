"""Early-exit nudge (permanent-fix sprint, worker 10/20).

Why: the agent burns all 25 tool calls even when it already has enough
information to answer after ~8. Stopping the model from calling tools
happens naturally IF the model stops on its own — but some models keep
"verifying" compulsively: re-reading files they already read, re-running
the same successful command, re-listing the same directory.

This module tracks a cheap per-result *novelty score* (0.0 = nothing new,
1.0 = all new). When the last 3 tool calls all produced near-zero novelty
AND the turn has already gathered substantial information (>= ``min_calls``
calls), ``note()`` returns a soft system hint urging the model to answer
now instead of making more tool calls. It never hard-stops — the model is
free to keep going; the hint is a nudge, not a gate. At most one hint per
turn, so it can never become token bloat of its own.

Signals (cheap, deterministic, stdlib-only):
  1. exact-duplicate result text (sha256 of normalised text) -> 0.0
  2. same read-type tool + same normalised args (re-read)     -> 0.0
  3. token containment: fraction of this result's significant tokens
     already seen in earlier results -> novelty = 1 - that fraction
Empty or failed results are ignored for novelty purposes (loopdetect owns
failures; an empty write result is not "redundant").

Integration: agent.py's turn loop instantiates one ``EarlyExitTracker``
per turn (next to the loopdetect detector) and calls ``note()`` after
every tool result inside ``_finish_one``. A non-None return is appended
as a tagged **user** runtime note — never a second system message — so the
model sees it before the next completion without it competing with
systemprompt.py.

Only stdlib. Safe to import at module level (no heavy deps). Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque

__all__ = ["EarlyExitTracker", "novelty_score", "EARLY_EXIT_HINT"]

EARLY_EXIT_HINT = (
    "You have gathered substantial information. "
    "If you can answer now, do so instead of more tool calls."
)

# Read-only, idempotent tools: calling one twice with the same arguments
# can never produce anything new on an unchanged filesystem, so a repeat is
# always redundant regardless of what the result text looks like.
_READ_TOOLS = frozenset({
    "read_file", "list_dir", "file_info", "search_files", "glob_files",
    "web_fetch", "web_search",
})

# Results shorter than this many significant tokens are not judged by
# overlap — tiny outputs ("OK", "3 files") collide by chance. Exact
# duplicates and same-args re-reads still count.
_MIN_TOKENS_FOR_OVERLAP = 40

# Bound the token memory so a pathological turn cannot grow it unbounded.
_MAX_SEEN_TOKENS = 50_000

_WS_RE = re.compile(r"\s+")

# Arg keys that change on every call but carry no meaning — a re-read that
# differs only in these is still the same read and must still count.
_VOLATILE_ARG_KEYS = frozenset({
    "timestamp", "ts", "_ts", "time", "elapsed", "elapsed_ms", "duration",
    "duration_ms", "request_id", "run_id", "trace_id", "session_id", "nonce",
    "uuid", "attempt", "retry_count",
})


def _normalise_text(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


def _normalise_args(args) -> str:
    if not isinstance(args, dict):
        return json.dumps(args, sort_keys=True, ensure_ascii=False,
                          default=str)
    clean = {k: v for k, v in args.items()
             if str(k).lower() not in _VOLATILE_ARG_KEYS}
    return json.dumps(clean, sort_keys=True, ensure_ascii=False,
                      default=str)


def _significant_tokens(norm_text: str) -> list[str]:
    """Content-bearing tokens: words of 4+ chars. Drops glue words and
    single-char noise so overlap measures meaning, not formatting."""
    return [w for w in norm_text.split(" ") if len(w) >= 4]


def novelty_score(result: str, seen_hashes: set[str],
                  seen_arg_keys: set[tuple[str, str]],
                  seen_tokens: set[str], tool_name: str = "",
                  args=None) -> float:
    """Novelty of one tool result against what was already seen.

    Returns 0.0 (nothing new) .. 1.0 (all new). ``seen_*`` are the
    tracker's accumulators — this function only reads them; the tracker
    updates them after scoring.
    """
    if not isinstance(result, str) or not result.strip():
        # Empty output: a read that found nothing is genuinely no new
        # information; a write that printed nothing is neutral. Score by
        # tool class so the tracker can tell them apart.
        return 0.0 if tool_name in _READ_TOOLS else 1.0
    norm = _normalise_text(result)
    digest = hashlib.sha256(norm.encode("utf-8", "replace")).hexdigest()
    if digest in seen_hashes:
        return 0.0  # byte-identical result seen before
    if tool_name in _READ_TOOLS and isinstance(args, dict):
        arg_key = (tool_name, _normalise_args(args))
        if arg_key in seen_arg_keys:
            return 0.0  # same read, same arguments: a re-read
    toks = _significant_tokens(norm)
    if len(toks) < _MIN_TOKENS_FOR_OVERLAP:
        # too short for overlap statistics — only exact dup / re-read
        # count, and those already returned above
        return 1.0
    uniq = set(toks)
    seen = sum(1 for w in uniq if w in seen_tokens)
    return max(0.0, 1.0 - seen / len(uniq))


class EarlyExitTracker:
    """Per-turn novelty tracker. ``note()`` returns the nudge hint (or
    None) after each successful tool result.

    Parameters: ``window`` — how many trailing calls must ALL be
    low-novelty; ``threshold`` — novelty below this counts as "no new
    information"; ``min_calls`` — the turn must have gathered this much
    before any nudge (never nag a turn that just started).
    """

    def __init__(self, window: int = 3, threshold: float = 0.35,
                 min_calls: int = 6) -> None:
        self.window = max(2, int(window))
        self.threshold = float(threshold)
        self.min_calls = max(1, int(min_calls))
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._hashes: set[str] = set()
            self._arg_keys: set[tuple[str, str]] = set()
            self._seen_tokens: set[str] = set()
            self._recent: deque[float] = deque(maxlen=self.window)
            self._calls = 0
            self._hinted = False

    def note(self, tool_name: str, args, result: str,
             status: str = "done") -> str | None:
        """Record one tool result. Returns the nudge hint when the last
        ``window`` calls all carried near-zero novelty after ``min_calls``
        total calls — at most once per turn. Never raises."""
        try:
            if status != "done":
                # Failures / denials are loopdetect's territory, not ours.
                return None
            with self._lock:
                self._calls += 1
                score = novelty_score(result, self._hashes,
                                      self._arg_keys, self._seen_tokens,
                                      tool_name, args)
                self._record(tool_name, args, result)
                self._recent.append(score)
                if (not self._hinted
                        and self._calls >= self.min_calls
                        and len(self._recent) == self.window
                        and all(s < self.threshold
                                for s in self._recent)):
                    self._hinted = True
                    return EARLY_EXIT_HINT
                return None
        except Exception:
            return None

    def _record(self, tool_name: str, args, result: str) -> None:
        """Fold this result into the seen-accumulators (called with the
        lock held, after scoring)."""
        if isinstance(result, str) and result.strip():
            norm = _normalise_text(result)
            self._hashes.add(
                hashlib.sha256(norm.encode("utf-8", "replace")).hexdigest())
            toks = _significant_tokens(norm)
            if len(toks) >= _MIN_TOKENS_FOR_OVERLAP:
                self._seen_tokens.update(toks)
                if len(self._seen_tokens) > _MAX_SEEN_TOKENS:
                    # drop arbitrary entries — overlap only needs a
                    # representative sample, not the full history
                    for _ in range(len(self._seen_tokens)
                                   - _MAX_SEEN_TOKENS):
                        self._seen_tokens.pop()
        if tool_name in _READ_TOOLS and isinstance(args, dict):
            self._arg_keys.add((tool_name, _normalise_args(args)))

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls


if __name__ == "__main__":
    # self-test — run with: python -m fullagent.earlyexit
    def _check(label, got, want_hint):
        ok = (got is not None) == want_hint
        extra = f" -> {got[:60]!r}..." if got else ""
        print(f"{'PASS' if ok else 'FAIL'}: {label}{extra}")
        return ok

    def _file_content(n):
        return (f"# module alpha{n}\nimport os\nimport sys\n"
                f"def handler_{n}(request):\n"
                f"    return process_{n}(request.payload)\n" * 6)

    all_ok = True

    # 1. redundant re-reads after a productive run -> hint fires exactly once
    t = EarlyExitTracker()
    for i in range(5):  # productive: 5 distinct reads, all novel
        r = t.note("read_file", {"path": f"src/a{i}.py"},
                   _file_content(i))
        all_ok &= _check(f"productive read {i} -> no hint", r, False)
    r1 = t.note("read_file", {"path": "src/a0.py"}, _file_content(0))
    r2 = t.note("read_file", {"path": "src/a0.py"}, _file_content(0))
    r3 = t.note("read_file", {"path": "src/a0.py"}, _file_content(0))
    all_ok &= _check("re-read x1 (novelty dip) -> no hint", r1, False)
    all_ok &= _check("re-read x2 -> no hint", r2, False)
    all_ok &= _check("re-read x3 -> hint FIRES", r3, True)
    assert r3 and "substantial information" in r3, \
        "hint must carry the mandated wording"
    r4 = t.note("read_file", {"path": "src/a0.py"}, _file_content(0))
    all_ok &= _check("hint never repeats in a turn", r4, False)

    # 2. productive sequence never fires, even past 25 calls
    t2 = EarlyExitTracker()
    fired = False
    tools = ["read_file", "list_dir", "run_command", "search_files",
             "web_fetch", "file_info", "glob_files"]
    for i in range(25):
        body = (f"distinct content block {i}: " +
                " ".join(f"token{i}_{k}" for k in range(60)))
        r = t2.note(tools[i % len(tools)], {"path": f"src/x{i}.py",
                                            "q": f"query{i}"}, body)
        fired |= r is not None
    ok2 = not fired
    print(f"{'PASS' if ok2 else 'FAIL'}: 25 productive calls -> no hint")
    all_ok &= ok2

    # 3. early re-reads don't nag: min_calls gate
    t3 = EarlyExitTracker()
    r = t3.note("read_file", {"path": "a.py"}, _file_content(0))
    r = t3.note("read_file", {"path": "a.py"}, _file_content(0))
    r = t3.note("read_file", {"path": "a.py"}, _file_content(0))
    all_ok &= _check("3 dupes before min_calls -> no hint", r, False)

    # 4. near-duplicate content (same file, one line changed) -> low novelty
    t4 = EarlyExitTracker()
    for i in range(6):
        t4.note("read_file", {"path": f"f{i}.py"}, _file_content(i))
    base = _file_content(0)
    tweaked = base.replace("import os", "import os  # tweaked comment")
    r = t4.note("read_file", {"path": "other.py"}, tweaked)
    all_ok &= _check("near-dupe alone -> no hint yet", r, False)
    # the novel tokens ("identical constant output") reset the streak once;
    # the exact duplicates after that pile up three zero-novelty calls
    for k in range(3):
        r = t4.note("run_command", {"cmd": "true"},
                    "identical constant output " * 40)
        all_ok &= _check(f"identical output x{k + 1} -> no hint yet", r,
                         False)
    r = t4.note("run_command", {"cmd": "true"},
                "identical constant output " * 40)
    all_ok &= _check("identical output x4 -> hint FIRES", r, True)

    # 5. errors never feed the novelty window (loopdetect owns them)
    t5 = EarlyExitTracker()
    for i in range(6):
        t5.note("read_file", {"path": f"g{i}.py"}, _file_content(i))
    for _ in range(5):
        r = t5.note("run_command", {"cmd": "boom"},
                    "ERROR: something failed", status="error")
        assert r is None
    all_ok &= _check("5 errors -> no hint", r, False)

    # 6. volatile arg keys do not hide a re-read — even with genuinely
    # NEW content, same read-tool + same normalised args scores 0.0
    t6 = EarlyExitTracker()
    for i in range(6):
        t6.note("read_file", {"path": f"h{i}.py"}, _file_content(i))
    for k in range(3):
        r = t6.note("read_file",
                    {"path": "h0.py",
                     "timestamp": f"2026-10-10T05:4{k}:00", "attempt": k},
                    _file_content(700 + k))  # novel tokens, same args
    all_ok &= _check("volatile-arg re-reads -> hint FIRES", r, True)

    # 7. different content same path (file changed between reads) -> novel
    t7 = EarlyExitTracker()
    for i in range(6):
        t7.note("read_file", {"path": f"k{i}.py"}, _file_content(i))
    r = t7.note("read_file", {"path": "evolved.py"}, _file_content(900))
    r = t7.note("read_file", {"path": "evolved.py"}, _file_content(901))
    r = t7.note("read_file", {"path": "evolved.py"}, _file_content(902))
    all_ok &= _check("genuinely new content -> no hint", r, False)

    # 8. reset() starts a fresh turn
    t8 = EarlyExitTracker()
    for i in range(5):
        t8.note("read_file", {"path": f"m{i}.py"}, _file_content(i))
    for _ in range(3):
        t8.note("read_file", {"path": "m0.py"}, _file_content(0))
    assert t8._hinted
    t8.reset()
    r = t8.note("read_file", {"path": "m0.py"}, _file_content(0))
    all_ok &= _check("reset clears hint + history", r, False)

    # 9. never raises on hostile input
    t9 = EarlyExitTracker()
    # 5 bad inputs < min_calls so no hint can fire — this only proves
    # nothing raises
    for bad in (None, 123, "", "   ", b"bytes"):
        r = t9.note("read_file", "not-a-dict", bad)  # type: ignore
        assert r is None, f"must not raise on {bad!r}"
    print("PASS: hostile inputs never raise")

    # 10. thread-safe under concurrent notes
    import threading as _th
    t10 = EarlyExitTracker()
    errs = []
    def _hammer(n):
        try:
            for i in range(50):
                t10.note("read_file", {"path": f"t{n}_{i}.py"},
                         _file_content(n * 50 + i))
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ths = [_th.Thread(target=_hammer, args=(n,)) for n in range(8)]
    [th.start() for th in ths]
    [th.join() for th in ths]
    ok = not errs and t10.calls == 400
    print(f"{'PASS' if ok else 'FAIL'}: 8 threads x 50 notes, no errors")
    all_ok &= ok

    print("ALL SELF-TESTS PASSED" if all_ok else "SELF-TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)
