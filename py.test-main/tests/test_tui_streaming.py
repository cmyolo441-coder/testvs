
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
from unittest.mock import patch

from rich.console import Console

from fullagent.agent import ToolEvent, Turn
from fullagent.tui import UI, _StreamThrottle


def _ticking_clock(start=100.0, step=0.2):
    """Fake clock advancing 0.2s per call: every token passes the 10fps
    stream throttle, so per-token visibility assertions keep their
    original meaning."""
    t = [start]

    def now():
        v = t[0]
        t[0] += step
        return v

    return now


class StreamingTests(unittest.TestCase):
    def make_ui(self, render=False):
        ui = UI.__new__(UI)
        ui.cfg = SimpleNamespace(extra={"render_markdown": render}, show_reasoning=False)
        ui._busy = False
        ui._cancel_flag = threading.Event()
        ui._focus_remaining = 0
        ui._status_text = ""
        ui._width = lambda: 80
        ui._invalidate = lambda: None
        ui._start_spinner = lambda: None
        ui._stop_spinner = lambda: None
        ui._emit_user = lambda text: None
        ui._set_flash = lambda *args: None
        ui._print_turn_stats = lambda turn: None
        ui.output = io.StringIO()
        ui.console = Console(file=ui.output, width=80, color_system=None)
        ui.agent = SimpleNamespace(save_session=lambda: None)
        return ui

    def test_first_partial_token_renders_before_second_token(self):
        ui = self.make_ui()
        ui._approve_request = None
        ui._spinner_i = 0
        frames = []
        def run(text, on_token, *args, **kwargs):
            on_token("first ")
            frames.append("".join(text for _, text in ui._bottom_fragments()))
            on_token("second ")
            frames.append("".join(text for _, text in ui._bottom_fragments()))
            return Turn(text, assistant_text="first second ")
        ui.agent.run_turn = run
        with patch("fullagent.tui.time.monotonic", side_effect=_ticking_clock()):
            ui._run_turn_thread("hello")
        self.assertIn("first", frames[0])
        self.assertNotIn("second", frames[0])
        self.assertIn("first second", frames[1])

    def test_burst_line_is_visible_before_next_token(self):
        ui = self.make_ui()
        snapshots = []
        def run(text, on_token, *args, **kwargs):
            on_token("first ")
            on_token("line\n")
            snapshots.append(ui.output.getvalue())
            on_token("last")
            snapshots.append(ui._status_text)
            return Turn(text, assistant_text="first line\nlast")
        ui.agent.run_turn = run
        with patch("fullagent.tui.time.monotonic", side_effect=_ticking_clock()):
            ui._run_turn_thread("hello")
        self.assertIn("first line\n", snapshots[0])
        self.assertIn("last", snapshots[1])
        self.assertEqual(ui.output.getvalue().count("first line"), 1)
        self.assertEqual(ui.output.getvalue().count("last"), 1)

    def test_markdown_burst_preview_updates_without_another_token(self):
        ui = self.make_ui(render=True)
        snapshots = []
        def run(text, on_token, *args, **kwargs):
            on_token("first ")
            on_token("last")
            snapshots.extend([ui._status_text, ui.output.getvalue()])
            return Turn(text, assistant_text="first last")
        ui.agent.run_turn = run
        with patch("fullagent.tui.time.monotonic", side_effect=_ticking_clock()):
            ui._run_turn_thread("hello")
        self.assertIn("last", snapshots[0])
        self.assertNotIn("first last", snapshots[1])
        self.assertEqual(ui.output.getvalue().count("first last"), 1)

    def test_rapid_submit_starts_only_one_worker(self):
        ui = self.make_ui()
        with patch("fullagent.tui.threading.Thread") as worker:
            ui._dispatch("one")
            ui._dispatch("two")
        self.assertEqual(worker.return_value.start.call_count, 1)

    def test_busy_until_session_is_saved(self):
        ui = self.make_ui()
        ui.agent.run_turn = lambda text, *args, **kwargs: Turn(text)
        starts_during_save = []

        def save():
            ui._dispatch("overlapping request")
            starts_during_save.append(worker.return_value.start.call_count)

        ui.agent.save_session = save
        with patch("fullagent.tui.threading.Thread") as worker:
            ui._run_turn_thread("hello")
            # The turn was still busy inside save_session: no overlap.
            self.assertEqual(starts_during_save, [0])
            # Once the turn finished, the queued request starts once.
            self.assertEqual(worker.return_value.start.call_count, 1)
            ui._dispatch("next request")
            self.assertEqual(worker.return_value.start.call_count, 1)

    def test_completed_edit_does_not_sleep(self):
        ui = self.make_ui()
        ui.console = Console(file=ui.output, force_terminal=True, width=80)
        ui.agent.run_turn = lambda text, token, reason, call, update, *a, **kw: (
            update(ToolEvent("write_file", {"path": "demo.py", "content": "a\nb\nc\n"},
                             result="OK: created demo.py", status="done")) or Turn(text))
        with patch("fullagent.tui.time.sleep") as sleep:
            ui._run_turn_thread("hello")
        sleep.assert_not_called()

    def test_failed_save_releases_turn(self):
        ui = self.make_ui()
        ui.agent.run_turn = lambda text, *args, **kwargs: Turn(text)
        def save():
            raise OSError("disk full")
        ui.agent.save_session = save
        with self.assertRaises(OSError):
            ui._run_turn_thread("hello")
        with patch("fullagent.tui.threading.Thread") as worker:
            ui._dispatch("next request")
            worker.return_value.start.assert_called_once()

    def test_1000_token_burst_throttles_renders_without_losing_text(self):
        """1000 rapid tokens must not cause ~1000 renders. With a frozen
        clock the 10fps throttle emits ~1 status frame, absorbs the rest,
        the final flush surfaces the complete tail, and every token's text
        still reaches the printed output (complete lines print immediately,
        never throttled)."""
        ui = self.make_ui()
        ui._approve_request = None
        ui._spinner_i = 0
        statuses = []
        real_set_status = UI._set_status

        def spy(text):
            statuses.append(text)
            real_set_status(ui, text)

        ui._set_status = spy
        renders = {"n": 0}
        ui._invalidate = lambda: renders.__setitem__("n", renders["n"] + 1)
        N = 1000
        words = [f"tok{i}" for i in range(N)]

        def run(text, on_token, *args, **kwargs):
            # every 7th token completes a line (must print immediately);
            # the rest accumulate as partial tail shown in the preview
            for i, w in enumerate(words):
                on_token(w + ("\n" if i % 7 == 6 else " "))
            return Turn(text, assistant_text=" ".join(words))

        ui.agent.run_turn = run
        with patch("fullagent.tui.time.monotonic", return_value=100.0):
            ui._run_turn_thread("hello")
        out = ui.output.getvalue()
        # no text lost: every token appears in the printed output
        for w in words:
            self.assertIn(w, out)
        th = ui._stream_throttle
        self.assertIsInstance(th, _StreamThrottle)
        # renders collapsed to turn-lifecycle only (thinking…, first frame,
        # final flush, teardown "", + 2 unconditional invalidates) —
        # versus ~1 invalidation per token before throttling
        self.assertLessEqual(renders["n"], 6,
                             f"expected <=6 renders for {N} rapid tokens, "
                             f"got {renders['n']}")
        self.assertEqual(th.frames, 2,
                         "only 2 stream frames should emit: first + final flush")
        self.assertGreaterEqual(th.suppressed, N - 2,
                                "throttle must absorb nearly the whole burst")
        # the final flush surfaced the complete tail as a status frame
        self.assertTrue(any(words[-1] in s for s in statuses),
                        "final flush must surface the complete tail")
        print(f"\n  [throttle] {N} tokens -> {renders['n']} renders "
              f"({th.suppressed} absorbed, {th.frames} frames emitted)")

    def test_cancel_before_worker_start_is_preserved(self):
        ui = self.make_ui()
        observed = []
        def run(text, *args, should_cancel, **kwargs):
            observed.append(should_cancel())
            return Turn(text, error="cancelled")
        ui.agent.run_turn = run
        with patch("fullagent.tui.threading.Thread"):
            ui._dispatch("hello")
        ui._cancel_flag.set()
        ui._run_turn_thread("hello")
        self.assertEqual(observed, [True])

    @staticmethod
    def _wait_idle(ui, timeout=5.0):
        deadline = time.monotonic() + timeout
        while ui._busy and time.monotonic() < deadline:
            time.sleep(0.01)
        return not ui._busy

    def test_busy_input_is_queued_and_runs_after_current_turn(self):
        ui = self.make_ui()
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        calls = []

        def run(text, *args, **kwargs):
            calls.append(text)
            if text == "first":
                first_started.set()
                release_first.wait(5)
            elif text == "second":
                second_started.set()
            return Turn(text)

        ui.agent.run_turn = run
        ui._dispatch("first")
        self.assertTrue(first_started.wait(3), "first turn never started")

        # Enter is accepted while busy, but must not start a second loop.
        ui._dispatch("second")
        self.assertEqual(ui._pending_count(), 1)
        self.assertFalse(second_started.is_set(),
                         "queued turn overlapped the running turn")
        self.assertEqual(calls, ["first"])

        release_first.set()
        self.assertTrue(second_started.wait(5),
                        "queued message never started")
        self.assertTrue(self._wait_idle(ui))
        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(ui._pending_count(), 0)

    def test_queued_messages_drain_fifo(self):
        ui = self.make_ui()
        first_started = threading.Event()
        release_first = threading.Event()
        all_done = threading.Event()
        calls = []

        def run(text, *args, **kwargs):
            calls.append(text)
            if text == "first":
                first_started.set()
                release_first.wait(5)
            if len(calls) == 3:
                all_done.set()
            return Turn(text)

        ui.agent.run_turn = run
        ui._dispatch("first")
        self.assertTrue(first_started.wait(3))
        ui._dispatch("second")
        ui._dispatch("third")
        self.assertEqual(ui._pending_count(), 2)

        release_first.set()
        self.assertTrue(all_done.wait(5), "queued turns did not drain")
        self.assertTrue(self._wait_idle(ui))
        self.assertEqual(calls, ["first", "second", "third"])
        self.assertEqual(ui._pending_count(), 0)

    def test_dispatch_waits_for_a_force_cleared_thread(self):
        """The cancel watchdog may clear _busy while the old worker is
        still alive; a new run_turn must not start until it returns."""
        ui = self.make_ui()
        ui._busy = False
        ui._turn_alive = True

        with patch("fullagent.tui.threading.Thread") as worker:
            ui._dispatch("while old thread shutting down")
            worker.return_value.start.assert_not_called()
        self.assertEqual(ui._pending_count(), 1)

        # Once the old worker finally returns, the queued text starts.
        ui._turn_alive = False
        with patch("fullagent.tui.threading.Thread") as worker:
            ui._advance_queue()
            worker.return_value.start.assert_called_once()

    def test_cancel_drops_queued_followups(self):
        ui = self.make_ui()
        first_started = threading.Event()
        release_first = threading.Event()
        calls = []

        def run(text, *args, **kwargs):
            calls.append(text)
            if text == "first":
                first_started.set()
                release_first.wait(5)
                return Turn(text, error="cancelled")
            return Turn(text)

        ui.agent.run_turn = run
        ui._dispatch("first")
        self.assertTrue(first_started.wait(3))
        ui._dispatch("follow-up")
        self.assertEqual(ui._pending_count(), 1)

        cleared = ui._request_cancel()
        self.assertEqual(cleared, 1)
        self.assertEqual(ui._pending_count(), 0)

        release_first.set()
        self.assertTrue(self._wait_idle(ui))
        self.assertEqual(calls, ["first"])
        self.assertEqual(ui._pending_count(), 0)

    def test_message_queued_after_cancel_runs_next(self):
        """Ctrl+C clears old follow-ups, not a message the user types
        while the cancelled turn is still unwinding."""
        ui = self.make_ui()
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        calls = []

        def run(text, *args, **kwargs):
            calls.append(text)
            if text == "first":
                first_started.set()
                release_first.wait(5)
                return Turn(text, error="cancelled")
            if text == "after-cancel":
                second_started.set()
            return Turn(text)

        ui.agent.run_turn = run
        ui._dispatch("first")
        self.assertTrue(first_started.wait(3))
        ui._request_cancel()
        ui._dispatch("after-cancel")
        self.assertEqual(ui._pending_count(), 1)

        release_first.set()
        self.assertTrue(second_started.wait(5),
                        "post-cancel follow-up was dropped")
        self.assertTrue(self._wait_idle(ui))
        self.assertEqual(calls, ["first", "after-cancel"])

    def test_live_diff_streaming_on_tool_args_and_diff_command(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            target = f"{td}/demo_live.py"
            with open(target, "w", encoding="utf-8") as fh:
                fh.write("def greet():\n    return 'old'\n")
            ui = self.make_ui()
            ui._diff_history = []
            ui.cfg.save = lambda: None

            def run(text, on_token, on_reasoning, on_tool_call, on_tool_update,
                    on_status, approve, **kwargs):
                on_tool_args = kwargs.get("on_tool_args")
                chunks = [
                    f'{{"path": "{target}", ',
                    '"old_string": "    return \'old\'\\n", ',
                    '"new_string": "    return \'new_live_diff\'\\n"}',
                ]
                for ch in chunks:
                    on_tool_args("edit_file", ch)
                ev = ToolEvent(
                    "edit_file",
                    {
                        "path": target,
                        "old_string": "    return 'old'\n",
                        "new_string": "    return 'new_live_diff'\n",
                    },
                    result="OK: edited demo_live.py",
                    status="done",
                )
                on_tool_call(ev)
                on_tool_update(ev)
                return Turn(text, assistant_text="updated")

            ui.agent.run_turn = run
            ui._run_turn_thread("edit demo_live.py")
            out = ui.output.getvalue()
            self.assertIn("LIVE DIFF", out)
            self.assertIn("return 'old'", out)
            self.assertIn("return 'new_live_diff'", out)
            self.assertEqual(len(ui._diff_history), 1)

            # Verify /diff command renders the recorded diff panel
            ui._cmd_diff("last")
            self.assertIn("Live Diff", ui.output.getvalue())

            # Verify /theme command switches palette
            ui._cmd_theme("dracula")
            self.assertEqual(ui._theme_name, "dracula")


if __name__ == "__main__":
    unittest.main()
