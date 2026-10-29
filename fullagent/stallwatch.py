"""Stuck-stream detection for SSE model calls.

INCIDENT: a model call produced no visible tokens for 512s while the UI
sat silently on "reasoning...". The hard stall watchdog in client.py only
fires when NO data at all arrives for 60s — a stream that keeps dribbling
reasoning chunks (or empty events) never trips it, so the user sees
nothing and gets no warning.

StallWatcher fills that gap: it tracks the last time a VISIBLE token
(content or tool-call activity) was received. If the stream is still open
but no tokens have arrived for `threshold` seconds, check() returns a
warning string the caller surfaces to the user (via on_status /
on_reasoning in client.py). Warning fires once per stall episode; the
timer resets as soon as tokens flow again.

Design notes:
- Injectable clock (time.monotonic-compatible callable) for testability.
- Reasoning-only chunks do NOT reset the timer: reasoning pieces are not
  visible tokens to the user (the TUI only shows a "reasoning..." status
  for them), and the 512s incident was exactly a stream that kept
  producing reasoning while no content tokens arrived.
- Zero dependencies beyond the stdlib; safe to import anywhere (leaf
  module, no import cycles).
"""

from __future__ import annotations

import time
from typing import Callable


DEFAULT_STALL_THRESHOLD = 60.0


class StallWatcher:
    """Warns when an open stream goes too long without visible tokens."""

    def __init__(self,
                 threshold: float = DEFAULT_STALL_THRESHOLD,
                 clock: Callable[[], float] | None = None) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self._clock = clock or time.monotonic
        self._last_token = self._clock()
        self._warned = False

    def token_received(self) -> None:
        """A visible token (content chunk / tool-call activity) arrived:
        reset the stall timer. Also re-arms the warning so a later stall
        episode warns again."""
        self._last_token = self._clock()
        self._warned = False

    def check(self) -> str | None:
        """Return the stall warning if no tokens arrived for >= threshold
        seconds, else None. Fires at most once per stall episode — the
        caller must keep calling check() (e.g. each SSE loop iteration);
        a None return means "all quiet, keep going"."""
        if self._warned:
            return None
        if self._clock() - self._last_token >= self.threshold:
            self._warned = True
            return (
                "Model is taking unusually long "
                f"(no output for {self.threshold:g}s). Press Esc to cancel."
            )
        return None


if __name__ == "__main__":
    # Self-test with an injectable clock — no network, no sleeps.
    class FakeClock:
        def __init__(self):
            self.now = 1000.0

        def __call__(self):
            return self.now

    def _expect(cond, label):
        print(("PASS" if cond else "FAIL"), "-", label)
        if not cond:
            raise SystemExit(1)

    # 1. Stalled stream: clock advanced 61s, no tokens -> warning fires.
    clk = FakeClock()
    w = StallWatcher(clock=clk)
    _expect(w.check() is None, "no warning at t=0")
    clk.now += 59
    _expect(w.check() is None, "no warning at 59s (below threshold)")
    clk.now += 2  # 61s total
    warn = w.check()
    _expect(warn is not None, "warning fires at 61s")
    _expect(warn == "Model is taking unusually long (no output for 60s). "
                    "Press Esc to cancel.", "warning text exact match")
    _expect(w.check() is None, "warning fires only once per episode")

    # 2. Normal token flow: token at 30s, then 30s more -> no warning.
    clk2 = FakeClock()
    w2 = StallWatcher(clock=clk2)
    clk2.now += 30
    w2.token_received()
    clk2.now += 30
    _expect(w2.check() is None, "steady token flow never warns")

    # 3. Tokens resume after a warning: timer resets, later stall warns again.
    clk3 = FakeClock()
    w3 = StallWatcher(clock=clk3)
    clk3.now += 61
    _expect(w3.check() is not None, "first stall warns")
    w3.token_received()  # tokens resume
    clk3.now += 30
    _expect(w3.check() is None, "no warning 30s after tokens resume")
    clk3.now += 31  # 61s since last token again
    _expect(w3.check() is not None, "second stall episode warns again")

    # 4. Custom threshold is honoured.
    clk4 = FakeClock()
    w4 = StallWatcher(threshold=10, clock=clk4)
    clk4.now += 11
    got = w4.check()
    _expect(got is not None and "no output for 10s" in got,
            "custom 10s threshold warned at 11s")

    print("all stallwatch self-tests passed")
