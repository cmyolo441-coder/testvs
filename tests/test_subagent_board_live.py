"""Live TUI agent board/footer driven by a REAL parallel swarm.

Proves the chrome is wired to reality, not to a fixture: eight tasks are
spawned into a real ``fullagent.subagents.ParallelSubagents`` manager and all
eight are genuinely inside the model call at the same time (a threading.Event
holds them there). While they run, the REAL ``fullagent.tui.UI`` fragment
builders must render the board (``≡ 8 Agents (8 running)`` + role chips) and
the footer badge (``◎ 8 agents | 8 running``). Once the swarm settles, the
board must vanish and the footer badge must disappear with it.

Offline: the injected ``chat`` callable returns a canned response, so there is
no network. No mocks of the code under test — only the provider boundary.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import io
import threading
import time
import unittest
from types import SimpleNamespace

from rich.console import Console

from fullagent.config import DEFAULT_MODEL_ID
from fullagent.subagents import (ACTIVE_STATES, MAX_GROUP_SIZE,
                                 TERMINAL_STATES, ParallelSubagents)
from fullagent.tui import UI

SWARM_SIZE = 8
START_DEADLINE = 5.0
SETTLE_DEADLINE = 15.0
CHAT_HOLD_TIMEOUT = 20.0  # bounded: a stuck worker can never hang the suite


class _Log:
    """Minimal event log satisfying ParallelSubagents' log contract."""

    def __init__(self):
        self.events = []

    def append(self, kind, data, **kwargs):
        event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
        self.events.append((kind, data, kwargs))
        return event


class _ConcurrencyProbe:
    """Offline ``chat`` that blocks until released and records real overlap.

    ``peak`` is the largest number of calls ever executing at the same time,
    taken under a lock at entry — the honest proof of real parallelism.
    """

    def __init__(self, release):
        self._release = release
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        with self._lock:
            self.active += 1
            self.calls += 1
            if self.active > self.peak:
                self.peak = self.active
        try:
            self._release.wait(CHAT_HOLD_TIMEOUT)
        finally:
            with self._lock:
                self.active -= 1
        return SimpleNamespace(
            content="offline result",
            tool_calls=[],
            usage={"prompt_tokens": 1, "completion_tokens": 1},
        )


def _make_manager(chat, max_workers=SWARM_SIZE, max_active=SWARM_SIZE):
    """A REAL ParallelSubagents wired to an offline provider boundary."""
    provider = SimpleNamespace(id="offline")
    model = SimpleNamespace(id="offline-model", supports_tools=True)
    effort = SimpleNamespace(key="high")
    return ParallelSubagents(
        _Log(), {},
        lambda _model_id: (provider, model, effort),
        max_workers=max_workers, max_active=max_active, chat=chat)


def _make_ui(manager):
    """UI stub carrying the REAL manager (no fake snapshot here)."""
    ui = UI.__new__(UI)
    ui.cfg = SimpleNamespace(extra={}, model_id=DEFAULT_MODEL_ID,
                             effort="high", show_reasoning=False)
    ui._busy = True
    ui._cancel_flag = threading.Event()
    ui._width = lambda: 120
    ui._invalidate = lambda: None
    ui._start_spinner = lambda: None
    ui._stop_spinner = lambda: None
    ui._set_flash = lambda text, color=None: None
    ui._focus_remaining = 0
    ui._status_text = ""
    ui._approve_request = None
    ui._auto_prev = 3
    ui._detail_expanded = False
    ui._last_detail_event = None
    ui._agents_cache = None
    ui._agents_cache_ts = 0.0

    ui.output = io.StringIO()
    ui.console = Console(file=ui.output, width=120, color_system=None)

    ui.agent = SimpleNamespace(
        autonomy=3,
        skill_forge=SimpleNamespace(registry={}),
        session_id="live1234",
        messages=[],
        model=SimpleNamespace(id=DEFAULT_MODEL_ID, context_window=262144,
                              label="Live Model"),
    )
    ui.agent.subagents = manager  # the real swarm, not a stub
    return ui


def _wait_until(predicate, deadline_seconds):
    """Poll *predicate* with a hard deadline; returns (ok, waited_seconds)."""
    start = time.monotonic()
    while time.monotonic() - start < deadline_seconds:
        if predicate():
            return True, time.monotonic() - start
        time.sleep(0.02)
    return predicate(), time.monotonic() - start


class LiveAgentBoardTests(unittest.TestCase):
    def test_board_and_footer_track_a_real_eight_worker_swarm(self):
        started = time.monotonic()
        release = threading.Event()
        probe = _ConcurrencyProbe(release)
        manager = _make_manager(probe)
        try:
            workers = [
                manager.spawn(f"live investigation {idx}", role="researcher")
                for idx in range(SWARM_SIZE)
            ]
            self.assertEqual(len(workers), SWARM_SIZE)
            self.assertEqual(len({w.id for w in workers}), SWARM_SIZE)
            self.assertEqual(len(manager.snapshot()["workers"]), SWARM_SIZE)

            # All 8 must reach "running" AND overlap inside chat simultaneously.
            def in_flight():
                return (all(w.state in ACTIVE_STATES for w in workers)
                        and probe.peak >= SWARM_SIZE)

            ok, waited = _wait_until(in_flight, START_DEADLINE)
            states = {w.id: w.state for w in workers}
            self.assertTrue(
                ok,
                f"swarm never reached {SWARM_SIZE} concurrent workers in "
                f"{waited:.2f}s (states={states}, peak={probe.peak})")
            self.assertGreaterEqual(probe.peak, SWARM_SIZE,
                                    f"only {probe.peak} concurrent chats")
            self.assertGreaterEqual(manager.snapshot()["peak_concurrency"],
                                    SWARM_SIZE)

            ui = _make_ui(manager)
            ui._invalidate_agents()  # drop any pre-run cache

            # Live board: header counts, "running" count and a real role chip.
            self.assertTrue(ui._agents_visible(),
                            "board not visible while 8 workers are running")
            board = "".join(text for _, text in ui._agents_fragments())
            self.assertNotEqual(board, "")
            self.assertIn(f"{SWARM_SIZE} Agents", board)
            self.assertIn(f"{SWARM_SIZE} running", board)
            self.assertIn("Explore", board)  # researcher role chip
            self.assertIn("◉", board)  # running-state icon on every row
            self.assertIn("↓ to manage", board)

            # Live footer: the agent badge is rendered from real snapshot data.
            footer = "".join(text for _, text in ui._meta_fragments())
            self.assertIn(f"{SWARM_SIZE} agents | {SWARM_SIZE} running", footer)
            self.assertIn(ui._model().label, footer)
            self.assertIn("ctx", footer)

            # Release: every worker must finish on its own.
            release.set()

            def settled():
                return all(w.state in TERMINAL_STATES for w in workers)

            ok, waited = _wait_until(settled, SETTLE_DEADLINE)
            states = {w.id: w.state for w in workers}
            self.assertTrue(ok, f"workers never settled in {waited:.2f}s: "
                                f"{states}")
            self.assertEqual({s for s in states.values()}, {"done"}, states)
            self.assertEqual(probe.calls, SWARM_SIZE,
                             "each worker should call chat exactly once")

            # The board and the badge must disappear once nothing is running.
            def idle_chrome():
                ui._invalidate_agents()
                return not ui._agents_visible()

            ok, waited = _wait_until(idle_chrome, START_DEADLINE)
            self.assertTrue(ok, f"board never hid after the swarm settled "
                                f"({waited:.2f}s)")
            self.assertFalse(ui._agents_visible())
            self.assertEqual(ui._agents_fragments(), [])

            footer = "".join(text for _, text in ui._meta_fragments())
            self.assertNotIn("agents |", footer,
                             f"stale agent badge in footer: {footer!r}")
            self.assertNotIn(" running", footer,
                             f"stale running count in footer: {footer!r}")
            self.assertIn(ui._model().label, footer)
            self.assertIn("ctx", footer)

            # The whole proof stays inside the bounded budget.
            elapsed = time.monotonic() - started
            self.assertLess(elapsed, 30.0,
                            f"test took {elapsed:.1f}s (must stay under 30s)")
        finally:
            release.set()
            manager.shutdown()


class DefaultPoolTests(unittest.TestCase):
    """The production pool must be able to run a full 8-task batch at once.

    Regression guard: the pool used to default to 4 workers while TaskGroup
    accepted 8, so half the batch silently sat in "queued" — and the old
    board labelled all 8 as "running", which was simply false.
    """

    def _default_manager(self, chat):
        provider = SimpleNamespace(id="offline")
        model = SimpleNamespace(id="offline-model", supports_tools=True)
        effort = SimpleNamespace(key="high")
        return ParallelSubagents(
            _Log(), {}, lambda _model_id: (provider, model, effort), chat=chat)

    def test_default_capacity_matches_a_full_batch(self):
        manager = self._default_manager(lambda *a, **k: None)
        try:
            self.assertEqual(manager.max_workers, MAX_GROUP_SIZE)
            self.assertGreaterEqual(manager.max_workers, manager.max_active)
        finally:
            manager.shutdown()

    def test_default_pool_really_runs_eight_at_once(self):
        release = threading.Event()
        probe = _ConcurrencyProbe(release)
        manager = self._default_manager(probe)
        try:
            workers = manager.spawn_many(
                [f"slice {i}" for i in range(SWARM_SIZE)],
                role="researcher", allow_writes=False)

            def all_eight_inside():
                return probe.peak >= SWARM_SIZE

            ok, waited = _wait_until(all_eight_inside, START_DEADLINE)
            states = {w.id: w.state for w in workers}
            self.assertTrue(
                ok, f"default pool ran only {probe.peak} concurrently in "
                    f"{waited:.2f}s (states={states})")
            release.set()

            def settled():
                return all(w.state in TERMINAL_STATES for w in workers)

            ok, waited = _wait_until(settled, SETTLE_DEADLINE)
            self.assertTrue(ok, f"workers never settled: {waited:.2f}s")
        finally:
            release.set()
            manager.shutdown()


if __name__ == "__main__":
    unittest.main()
