"""Qoder-style chrome: welcome card, HUD, footer and live agent board.

Drives real code only — ``fullagent.cliui`` layout helpers and the real
``fullagent.tui.UI`` fragment builders / key actions against stub objects.
Headless: rich writes to a StringIO console, no TTY needed.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import io
import threading
import unittest
from types import SimpleNamespace

from rich.console import Console

from fullagent import cliui
from fullagent.agent import ToolEvent
from fullagent.config import DEFAULT_MODEL_ID
from fullagent.tui import UI


def _width(frags):
    return sum(cliui.display_width(text) for _, text in frags)


def _snapshot(states, role="researcher"):
    counts = {}
    for state in states:
        counts[state] = counts.get(state, 0) + 1
    return {"workers": [
        {"id": f"ag-{i}", "state": state, "role": role,
         "task": f"Inspect src/file{i}.py and report",
         "current_action": f"Read src/file{i}.py",
         "summary": "done" if state == "done" else "",
         "elapsed_ms": i * 1000}
        for i, state in enumerate(states)
    ], "counts": counts,
        "active": sum(1 for s in states if s in cliui.ACTIVE_STATES)}


class _Mgr:
    def __init__(self, states):
        self._snap = _snapshot(states)

    def snapshot(self):
        return self._snap


def _make_ui(states=None, skills=3, autonomy=3):
    ui = UI.__new__(UI)
    ui.cfg = SimpleNamespace(extra={}, model_id=DEFAULT_MODEL_ID,
                             effort="high", show_reasoning=False)
    ui._busy = False
    ui._cancel_flag = threading.Event()
    ui._focus_remaining = 0
    ui._status_text = ""
    ui._approve_request = None
    ui._width = lambda: 120
    ui._invalidate = lambda: None
    ui._start_spinner = lambda: None
    ui._stop_spinner = lambda: None
    ui._emit_user = lambda text: None

    flashes = []
    ui._set_flash = lambda text, color=None: flashes.append(text)
    ui.flashes = flashes

    ui.output = io.StringIO()
    ui.console = Console(file=ui.output, width=100, color_system=None)

    calls = []

    def set_autonomy(level):
        calls.append(level)
        ui.agent.autonomy = max(0, min(5, int(level)))
        return "ok"

    ui.agent = SimpleNamespace(
        autonomy=autonomy, set_autonomy=set_autonomy,
        subagents=_Mgr(states or []),
        skill_forge=SimpleNamespace(registry={f"s{i}": None
                                              for i in range(skills)}),
        session_id="abcd1234",
        messages=[],
        model=SimpleNamespace(id=DEFAULT_MODEL_ID, context_window=262144,
                              label="Test Model"),
        autonomy_calls=calls,
    )
    ui._auto_prev = 3
    ui._detail_expanded = False
    ui._last_detail_event = None
    return ui



class CliuiLayoutTests(unittest.TestCase):
    def test_align_pads_to_exact_width(self):
        for width in (24, 60, 100, 137):
            line = cliui.align([("", "left")], [("", "right")], width)
            self.assertEqual(_width(line), width)

    def test_align_keeps_right_badge_on_narrow_terminal(self):
        line = cliui.align([("", "x" * 200)], [("", "tail")], 30)
        self.assertEqual(_width(line), 30)
        self.assertIn("tail", "".join(t for _, t in line))

    def test_align_survives_oversized_right_badge(self):
        line = cliui.align([("", "left")], [("", "x" * 80)], 20)
        self.assertLessEqual(_width(line), 20)
        self.assertGreaterEqual(_width(line), 19)

    def test_board_and_footer_split_queued_from_running(self):
        snap = _snapshot(["running"] * 4 + ["queued"] * 4)
        board = cliui.agent_board(snap, width=100, colors={})
        head = "".join(t for _, t in board[0])
        self.assertIn("8 Agents", head)
        self.assertIn("4 running", head)
        self.assertIn("4 queued", head)
        self.assertNotIn("8 running", head)

        foot = "".join(t for _, t in cliui.status_fragments(
            width=140, colors={}, model_label="M", effort_label="high",
            context_window=1000, used_pct=1, cwd="~", branch="main",
            agents_running=4, agents_queued=4))
        self.assertIn("8 agents | 4 running", foot)
        self.assertIn("4 queued", foot)

    def test_footer_without_live_workers_has_no_agent_badge(self):
        foot = "".join(t for _, t in cliui.status_fragments(
            width=140, colors={}, model_label="M", effort_label="high",
            context_window=1000, used_pct=1, cwd="~", branch="main",
            agents_running=0, agents_queued=0))
        self.assertNotIn("agents", foot)
        self.assertIn("main", foot)

    def test_manager_progress_events_refresh_the_board_immediately(self):
        class Mgr:
            def __init__(self):
                self.cb = None

            def on_progress(self, cb):
                self.cb = cb

            def snapshot(self):
                return _snapshot(["running"])

        ui = _make_ui([])
        mgr = Mgr()
        ui.agent.subagents = mgr
        ui._agents_cache = None
        ui._agents_snapshot()  # first read wires the real progress hook
        self.assertIsNotNone(mgr.cb, "progress hook not registered")

        ui._agents_cache = _snapshot(["running"])
        ui._agents_cache_ts = 1e18  # pretend a fresh cache
        redraws = []
        ui._invalidate = lambda: redraws.append(1)
        mgr.cb(None, "subagent.running")
        self.assertIsNone(ui._agents_cache,
                          "worker event must drop the stale snapshot")
        self.assertTrue(redraws, "worker event must request a redraw")

    def test_board_shows_task_while_worker_waits_on_model(self):
        snap = {"workers": [{
            "id": "ag-1", "state": "running", "role": "researcher",
            "current_action": "model step 2/20",
            "task": "Read fullagent/tui.py and report the layout",
            "summary": "", "elapsed_ms": 1200,
        }], "counts": {"running": 1}, "active": 1}
        board = "".join(t for _, t in cliui.agent_board(
            snap, width=100, colors={})[1])
        self.assertIn("Read fullagent/tui.py", board)
        self.assertNotIn("model step", board)

    def test_retained_workers_never_inflate_live_counts(self):
        # 9 finished workers from older batches + 1 genuinely running
        snap = _snapshot(["done"] * 9 + ["running"])
        board = cliui.agent_board(snap, width=100, colors={})
        head = "".join(t for _, t in board[0])
        self.assertIn("1 Agent", head)
        self.assertNotIn("10", head)
        self.assertIn("1 running", head)

        foot = "".join(t for _, t in cliui.status_fragments(
            width=140, colors={}, model_label="M", effort_label="high",
            context_window=262144, used_pct=1, cwd="~", branch="main",
            agents_running=1, agents_queued=0))
        self.assertIn("1 agent | 1 running", foot)
        self.assertNotIn("10", foot)

    def test_agent_chip_names(self):
        self.assertEqual(cliui.short_role("researcher"), "Explore")
        self.assertEqual(cliui.short_role("coder"), "Code")
        self.assertEqual(cliui.short_role("mystery_role"), "Mystery Role")

    def test_agent_board_only_while_active_and_bounded(self):
        self.assertEqual(cliui.agent_board({"workers": []}, width=80,
                                           colors={}), [])
        idle = cliui.agent_board(_snapshot(["done"]), width=80, colors={})
        self.assertEqual(idle, [])
        board = cliui.agent_board(_snapshot(["running"] * 9), width=80,
                                  colors={})
        self.assertEqual(len(board), 1 + cliui.PANEL_LIMIT + 1)
        self.assertIn("9 Agents", "".join(t for _, t in board[0]))
        self.assertIn("Explore", "".join(t for _, t in board[1]))
        self.assertIn("+3 more", "".join(t for _, t in board[-1]))
        for row in board:
            self.assertLessEqual(_width(row), 80)


class UiChromeTests(unittest.TestCase):
    def test_agents_board_visible_only_with_live_workers(self):
        live = _make_ui(["running", "queued"])
        self.assertTrue(live._agents_visible())
        text = "".join(t for _, t in live._agents_fragments())
        self.assertIn("2 Agents", text)
        self.assertIn("Explore", text)

        idle = _make_ui(["done", "done"])
        self.assertFalse(idle._agents_visible())
        self.assertEqual(idle._agents_fragments(), [])

    def test_hud_shows_mode_and_skill_count(self):
        auto = _make_ui(["running"], skills=21, autonomy=3)
        text = "".join(t for _, t in auto._hud_fragments())
        self.assertIn("COLLAB", text)
        self.assertIn("21 skills", text)
        self.assertIn("⇧Tab", text)
        self.assertIn("1 running", text)

        pilot = _make_ui([], skills=0, autonomy=4)
        self.assertIn("PILOT",
                      "".join(t for _, t in pilot._hud_fragments()))
        top = _make_ui([], skills=0, autonomy=5)
        self.assertIn("AUTONOMOUS",
                      "".join(t for _, t in top._hud_fragments()))

    def test_footer_shows_model_context_cwd_and_agents(self):
        ui = _make_ui(["running"] * 8)
        text = "".join(t for _, t in ui._meta_fragments())
        self.assertIn(ui._model().label, text)
        self.assertIn("ctx", text)
        self.assertIn("context", text)
        self.assertIn("8 agents | 8 running", text)
        self.assertEqual(_width(ui._meta_fragments()), 120)

    def test_agent_badge_disappears_when_nothing_is_running(self):
        # retained (finished) workers must never leave a fake "8 running"
        ui = _make_ui(["done"] * 8)
        text = "".join(t for _, t in ui._meta_fragments())
        self.assertNotIn("agents", text)
        self.assertNotIn("running", text)
        self.assertIn(ui._model().label, text)

    def test_footer_branch_is_the_real_git_branch(self):
        ui = _make_ui([])
        branch = ui._branch_label()
        self.assertTrue(branch)
        self.assertNotIn("\n", branch)
        # cached value is stable and identical on the next frame
        self.assertEqual(branch, ui._branch_label())

        import subprocess
        try:
            expected = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            expected = ""
        if expected and expected != "HEAD":
            self.assertEqual(branch, expected)
        else:
            self.assertTrue(branch.startswith("session ") or
                            branch == "no git")

    def test_signed_in_label_is_honest(self):
        ui = _make_ui([])
        label = ui._signed_in_label()
        self.assertNotIn("Signed in", label)
        self.assertTrue(label.startswith("API key ·") or
                        label.startswith("No API key") or
                        label.startswith("No provider"), label)

    def test_shift_tab_toggles_auto_mode_both_ways(self):
        ui = _make_ui([], autonomy=3)
        ui._toggle_auto_mode()
        self.assertEqual(ui.agent.autonomy, 4)
        ui._toggle_auto_mode()
        self.assertEqual(ui.agent.autonomy, 3)
        self.assertEqual(ui.agent.autonomy_calls, [4, 3])

    def test_banner_renders_logo_version_and_tips(self):
        ui = _make_ui([])
        ui.print_banner()
        out = ui.output.getvalue()
        self.assertIn("Tips for getting started", out)
        self.assertIn("FullAgent CLI", out)
        self.assertIn(ui._model().label, out)
        self.assertIn("abcd1234", out)
        for idx in range(1, 5):
            self.assertIn(f"{idx}.", out)


class DetailToggleTests(unittest.TestCase):
    def _shell_event(self, lines=60):
        stdout = "\n".join(f"line {i}" for i in range(lines))
        result = f"exit code: 0\n--- stdout ---\n{stdout}"
        return ToolEvent(name="run_command", args={"command": "seq"},
                         result=result, status="done")

    def test_ctrl_o_expands_last_block_and_collapses(self):
        ui = _make_ui([])
        ev = self._shell_event(60)

        compact = ui._live_rows(ev, 80)
        self.assertTrue(any("Ctrl+O" in seg for row in compact
                            for seg, _ in row))
        self.assertFalse(any("line 40" == seg for row in compact
                             for seg, _ in row))

        ui._last_detail_event = (ev, "live")
        ui._toggle_detail_view()
        self.assertTrue(ui._detail_expanded)
        out = ui.output.getvalue()
        self.assertIn("detail ·", out)
        self.assertIn("line 40", out)
        self.assertIn("line 59", out)
        self.assertTrue(ui.flashes and "expanded" in ui.flashes[-1])

        expanded = ui._live_rows(ev, 80)
        self.assertFalse(any("Ctrl+O" in seg for row in expanded
                             for seg, _ in row))

        ui._toggle_detail_view()
        self.assertFalse(ui._detail_expanded)

    def test_ctrl_o_without_history_only_flashes(self):
        ui = _make_ui([])
        ui._toggle_detail_view()
        self.assertTrue(ui._detail_expanded)
        self.assertEqual(ui.output.getvalue(), "")
        self.assertTrue(ui.flashes)


class ThinkingBlockTests(unittest.TestCase):
    def test_thinking_block_collapses_and_ctrl_o_expands(self):
        ui = _make_ui([])
        text = "\n".join(f"thought {i}" for i in range(15))
        ui._print_thinking_block(text)
        out = ui.output.getvalue()
        self.assertIn("Thinking", out)
        self.assertIn("thought 0", out)
        self.assertNotIn("thought 12", out)
        self.assertIn("+7 rows (Ctrl+O to expand)", out)

        ui._toggle_detail_view()
        out2 = ui.output.getvalue()
        self.assertIn("thought 14", out2)
        self.assertTrue(any("thinking expanded" in f for f in ui.flashes))


if __name__ == "__main__":
    unittest.main()
