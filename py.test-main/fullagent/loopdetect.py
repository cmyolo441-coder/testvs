"""Semantic retry-loop detection (deep-reliability, worker 4/20).

Why: a model can retry the same failing tool call with the same arguments
and the same error for minutes (the 512s MultiEdit incident). A hard
tool-call cap is a blunt instrument — this detector recognises the SEMANTIC
loop (identical tool + normalised args + normalised error) and breaks the
turn early with actionable guidance, so the model stops wasting budget and
tries a genuinely different approach instead.

Integration: agent.py's turn loop calls ``LoopDetector.note(...)`` after
every tool error; a non-None return means "break the turn now".

Only stdlib. Safe to import at module level (no heavy deps).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading

__all__ = ["LoopDetector", "normalize_args", "normalize_error"]

# Arg keys that change on every call but carry no meaning — a retry that
# differs only in these is still the same approach and must still count.
_VOLATILE_ARG_KEYS = frozenset({
    "timestamp", "ts", "_ts", "time", "elapsed", "elapsed_ms", "duration",
    "duration_ms", "request_id", "run_id", "trace_id", "session_id", "nonce",
    "uuid", "attempt", "retry_count",
})

_HEX_RE = re.compile(r"\b(?:0x)?[0-9a-fA-F]{6,}\b")
_NUM_RE = re.compile(r"\d[\d,._:/-]*\d|\d")
_WS_RE = re.compile(r"\s+")
_QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")


def normalize_args(args) -> str:
    """Canonical form of tool args: keys sorted, volatile fields dropped.

    Two calls that differ only in volatile noise (timestamps, attempt ids)
    normalise to the same string — they are the same approach.
    """
    if not isinstance(args, dict):
        try:
            return json.dumps(args, sort_keys=True, ensure_ascii=False,
                              default=str)
        except (TypeError, ValueError):
            return json.dumps(args, ensure_ascii=False, default=str)
    clean = {k: v for k, v in args.items()
             if str(k).lower() not in _VOLATILE_ARG_KEYS}
    try:
        return json.dumps(clean, sort_keys=True, ensure_ascii=False,
                          default=str)
    except (TypeError, ValueError):
        return json.dumps(clean, ensure_ascii=False, default=str)


def normalize_error(error_text) -> str:
    """Error signature: stable across volatile noise in the error text.

    Strips the leading "ERROR: " chain, collapses hex ids / numbers /
    quoted fragments / whitespace so the same failure with different
    timestamps or ids hashes identically. Different *kinds* of failure
    (different wording) still hash differently.
    """
    if not isinstance(error_text, str):
        error_text = "" if error_text is None else str(error_text)
    text = error_text.splitlines()[0] if error_text else ""
    text = re.sub(r"^(ERROR:\s*)+", "", text).strip()
    text = _HEX_RE.sub("<HEX>", text)
    text = _NUM_RE.sub("<N>", text)
    text = _QUOTED_RE.sub("<Q>", text)
    text = _WS_RE.sub(" ", text).strip()
    return text[:220]


class LoopDetector:
    """Tracks consecutive identical (tool, args, error) failure triples.

    ``note(tool_name, args, error_signature) -> str | None`` — returns a
    break message when the same failure triple has repeated ``repeat_limit``
    times in a row; otherwise returns None. A different error, different
    args, or a non-consecutive failure resets the streak, so only a true
    retry loop trips it. Thread-safe (the agent may finish tool calls from
    worker threads).
    """

    def __init__(self, repeat_limit: int = 3):
        self.repeat_limit = max(2, repeat_limit)
        self._lock = threading.Lock()
        self._last_key: str | None = None
        self._streak = 0

    def note(self, tool_name: str, args, error_signature: str) -> str | None:
        key = hashlib.sha256(
            ("\x00".join([
                str(tool_name),
                normalize_args(args),
                str(error_signature),
            ])).encode("utf-8", "replace")
        ).hexdigest()[:16]
        with self._lock:
            if key == self._last_key:
                self._streak += 1
            else:
                self._last_key = key
                self._streak = 1
            if self._streak >= self.repeat_limit:
                msg = (
                    f"Loop detected: {tool_name} failed {self._streak}x "
                    f"with the same error ({error_signature[:120]}). "
                    "Stop retrying. Alternatives: re-read the file, try a "
                    "different old_string, use line numbers."
                )
                # reset so a resumed turn starts a fresh streak instead of
                # re-firing instantly
                self._last_key = None
                self._streak = 0
                return msg
            return None

    def reset(self) -> None:
        with self._lock:
            self._last_key = None
            self._streak = 0


if __name__ == "__main__":
    # self-test — run with: python -m fullagent.loopdetect
    def _check(label, got, want_break):
        ok = (got is not None) == want_break
        extra = f" -> {got[:60]!r}..." if got else ""
        print(f"{'PASS' if ok else 'FAIL'}: {label}{extra}")
        return ok

    all_ok = True

    # 1. three identical failures -> break signal with helpful text
    d = LoopDetector()
    args = {"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}
    sig = normalize_error("ERROR: MultiEdit failed: old_string not found")
    r1 = d.note("multiedit", args, sig)
    r2 = d.note("multiedit", args, sig)
    r3 = d.note("multiedit", args, sig)
    all_ok &= _check("2 identical failures -> no break", r1, False)
    all_ok &= _check("2 identical failures -> no break", r2, False)
    all_ok &= _check("3 identical failures -> break", r3, True)
    assert r3 and "multiedit" in r3 and "old_string" in r3, \
        "break message must name the tool and give alternatives"

    # 2. volatile noise in error must still count as identical
    d2 = LoopDetector()
    s_a = normalize_error("ERROR: MultiEdit failed at 14:03:22: old_string not found")
    s_b = normalize_error("ERROR: MultiEdit failed at 14:03:41: old_string not found")
    assert s_a == s_b, "volatile timestamps must normalise away"
    d2.note("multiedit", args, s_a)
    d2.note("multiedit", args, s_b)
    r = d2.note("multiedit", args, s_a)
    all_ok &= _check("volatile-noise repeats -> break", r, True)

    # 3. same tool, different error -> no break
    d3 = LoopDetector()
    d3.note("multiedit", args, normalize_error("ERROR: old_string not found"))
    d3.note("multiedit", args, normalize_error("ERROR: old_string not found"))
    r = d3.note("multiedit", args,
                normalize_error("ERROR: file is binary, cannot edit"))
    all_ok &= _check("different error resets streak -> no break", r, False)

    # 4. different args -> no break
    d4 = LoopDetector()
    d4.note("multiedit", args, sig)
    d4.note("multiedit", args, sig)
    r = d4.note("multiedit",
                {"path": "a.py", "old_string": "y = 1", "new_string": "y = 2"},
                sig)
    all_ok &= _check("different args reset streak -> no break", r, False)

    # 5. volatile arg keys do not hide the loop
    d5 = LoopDetector()
    a1 = dict(args, timestamp="2026-10-10T05:40:00", attempt=1)
    a2 = dict(args, timestamp="2026-10-10T05:41:00", attempt=2)
    a3 = dict(args, timestamp="2026-10-10T05:42:00", attempt=3)
    d5.note("multiedit", a1, sig)
    d5.note("multiedit", a2, sig)
    r = d5.note("multiedit", a3, sig)
    all_ok &= _check("volatile args still count -> break", r, True)

    # 6. interleaved different failure breaks the streak
    d6 = LoopDetector()
    d6.note("multiedit", args, sig)
    d6.note("multiedit", args, normalize_error("ERROR: permission denied"))
    r = d6.note("multiedit", args, sig)
    all_ok &= _check("non-consecutive repeat -> no break", r, False)

    print("ALL SELF-TESTS PASSED" if all_ok else "SELF-TESTS FAILED")
    raise SystemExit(0 if all_ok else 1)
