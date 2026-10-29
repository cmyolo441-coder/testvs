"""First-token-safe render throttle for the streaming TUI.

Pure logic — no UI imports — so unit/latency tests can exercise the real
throttle without prompt_toolkit. Re-exported as ``tui._StreamThrottle``
(the TUI instantiates one per turn in ``_run_turn``).
"""
from __future__ import annotations

import time
from typing import Callable


class StreamThrottle:
    """Rate-limit per-token UI updates to at most one frame per `interval`.

    Token ingestion is NEVER blocked: every update() buffers the latest
    preview text instantly. maybe_flush() returns the buffered frame only
    when >= interval seconds have passed since the previous frame, so a
    fast stream (thousands of tokens/sec) triggers at most 10 screen
    invalidations/sec instead of one full-screen re-render per token.
    flush() forces the pending frame out — call it when the stream ends so
    the final render always reflects the complete output.

    FIRST-TOKEN GUARANTEE: the first update() is always flushed
    immediately by maybe_flush(), no matter what the clock says — the
    10fps throttle must never delay the first visible token. (The old
    code achieved this only implicitly via a ``_last_emit = 0.0`` epoch
    trick; the explicit flag survives any clock behaviour.)

    CLOCK SAFETY: the clock defaults to time.monotonic. With wall-clock
    time.time(), an NTP step *backward* made due() return False for the
    whole jump duration — freezing ALL frames, first token included,
    for many seconds. Monotonic time can never jump backward, so the
    throttle can only ever delay a frame by `interval`.

    `frames` / `suppressed` counters make the before/after measurable.
    """

    def __init__(self, interval: float = 0.1,
                 clock: Callable[[], float] | None = None):
        self.interval = interval
        self._clock = clock or time.monotonic
        self._last_emit = 0.0
        self._pending: str | None = None
        self._first = True  # first update() always flushes immediately
        self.frames = 0       # frames actually emitted
        self.suppressed = 0   # updates absorbed without emitting a frame

    def update(self, text: str) -> None:
        """Buffer the latest text. Never blocks, never drops text."""
        self._pending = text

    def due(self) -> bool:
        """True when at least `interval` seconds passed since last frame."""
        return (self._clock() - self._last_emit) >= self.interval

    def maybe_flush(self) -> str | None:
        """Return the buffered frame if one is due, else None.

        The very first update is ALWAYS flushed immediately — the
        throttle must not delay the first visible token.
        """
        if self._pending is None:
            return None
        if self._first:
            return self.flush()
        if not self.due():
            self.suppressed += 1
            return None
        return self.flush()

    def flush(self) -> str | None:
        """Force the buffered frame out (stream end / final render)."""
        if self._pending is None:
            return None
        text = self._pending
        self._pending = None
        self._last_emit = self._clock()
        self._first = False
        self.frames += 1
        return text


if __name__ == "__main__":
    # Self-test: throttle behaviour incl. the first-token guarantee.
    def _expect(cond, label):
        print(("PASS" if cond else "FAIL"), "-", label)
        if not cond:
            raise SystemExit(1)

    # 1. burst: first update flushes immediately, rest absorbed, no loss
    now = [1000.0]
    th = StreamThrottle(0.1, clock=lambda: now[0])
    for i in range(1000):
        th.update(f"tok{i}")
        frame = th.maybe_flush()
        if i == 0:
            _expect(frame == "tok0",
                    f"first update must flush immediately, got {frame!r}")
        else:
            _expect(frame is None, "burst updates must be absorbed")
    _expect(th.frames == 1, f"1 frame emitted, got {th.frames}")
    _expect(th.suppressed == 999, f"999 absorbed, got {th.suppressed}")
    _expect(th.flush() == "tok999", "flush returns latest text")
    _expect(th.flush() is None, "empty flush returns None")

    # 2. first frame is immediate even when the clock barely moved
    now2 = [5000.0]
    th2 = StreamThrottle(0.1, clock=lambda: now2[0])
    th2.update("hello")
    _expect(th2.maybe_flush() == "hello",
            "first frame immediate regardless of clock")
    # ...but the second frame still honours the 10fps throttle
    th2.update("world")
    _expect(th2.maybe_flush() is None, "second frame throttled")
    now2[0] += 0.2
    _expect(th2.maybe_flush() == "world", "frame due after interval")

    # 3. a backward wall-clock jump must not freeze the throttle:
    # the default clock is monotonic, which never jumps backward
    th3 = StreamThrottle(0.1)
    _expect(th3._clock is time.monotonic,
            "default clock is time.monotonic (NTP-jump safe)")
    th3.update("first")
    _expect(th3.maybe_flush() == "first",
            "first frame immediate on the default clock too")

    print("ALL STREAMTHROTTLE SELF-TESTS PASS")
