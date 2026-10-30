"""Prompt-box UX regressions: Enter/completion, frame shrink, no I/O in render.

Every test drives the REAL ``fullagent.tui.UI`` key bindings and render
callbacks against stub objects — headless, no TTY needed.
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
from unittest.mock import patch

from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from rich.console import Console

from fullagent.config import DEFAULT_MODEL_ID
from fullagent.tui import UI, OverlayList


def _make_ui(**agent_attrs):
    ui = UI.__new__(UI)
    ui.cfg = SimpleNamespace(extra={}, model_id=DEFAULT_MODEL_ID,
                             effort="high", show_reasoning=False)
    ui._busy = False
    ui._cancel_flag = threading.Event()
    ui._focus_remaining = 0
    ui._status_text = ""
    ui._approve_request = None
    ui._width = lambda: 100
    ui._invalidate = lambda: None
    ui._set_flash = lambda *a, **k: None
    ui._emit_user = lambda text: None
    ui.output = io.StringIO()
    ui.console = Console(file=ui.output, width=100, color_system=None)
    ui.buffer = Buffer(multiline=True)
    attrs = dict(session_id="abcd1234", autonomy=3, messages=[],
                 subagents=None, skill_forge=SimpleNamespace(registry={}),
                 model=SimpleNamespace(id=DEFAULT_MODEL_ID,
                                       context_window=262144,
                                       label="Test Model"))
    attrs.update(agent_attrs)
    ui.agent = SimpleNamespace(**attrs)
    return ui


def _binding(ui, name):
    """The real key-binding handler registered as ``name`` (a closure)."""
    for binding in ui._build_key_bindings().bindings:
        if getattr(binding.handler, "__name__", "") == name:
            return binding
    raise AssertionError(f"no key binding named {name!r}")


class _FakeRenderer:
    def __init__(self):
        self.erased = 0

    def erase(self, *_a, **_k):
        self.erased += 1


class EnterCompletionTests(unittest.TestCase):
    """Enter with the completion menu open must APPLY the selection."""

    def _ui_with_menu(self):
        ui = _make_ui()
        submitted = []
        ui.buffer = Buffer(multiline=True, accept_handler=(
            lambda buff: submitted.append(buff.text) or False))
        ui.submitted = submitted
        ui.buffer.insert_text("/he")
        ui.buffer.complete_state = CompletionState(
            original_document=Document("/he", 3),
            completions=[Completion("/help", start_position=-3,
                                    display_meta="show help")],
            complete_index=0,
        )
        return ui

    def test_enter_applies_the_highlighted_completion(self):
        ui = self._ui_with_menu()
        _binding(ui, "_enter").handler(SimpleNamespace(app=None))
        self.assertEqual(ui.buffer.text, "/help",
                         "Enter dropped the highlighted completion")
        self.assertIsNone(ui.buffer.complete_state)
        self.assertEqual(ui.submitted, [],
                         "the completion fill must not submit the turn")

    def test_second_enter_submits_the_completed_text(self):
        ui = self._ui_with_menu()
        handle = _binding(ui, "_enter").handler
        handle(SimpleNamespace(app=None))
        handle(SimpleNamespace(app=None))
        self.assertEqual(ui.submitted, ["/help"])

    def test_enter_without_a_menu_still_submits(self):
        ui = _make_ui()
        submitted = []
        ui.buffer = Buffer(multiline=True, accept_handler=(
            lambda buff: submitted.append(buff.text) or False))
        ui.buffer.insert_text("hello")
        _binding(ui, "_enter").handler(SimpleNamespace(app=None))
        self.assertEqual(submitted, ["hello"])

    def test_empty_enter_submits_nothing(self):
        ui = _make_ui()
        submitted = []
        ui.buffer = Buffer(multiline=True, accept_handler=(
            lambda buff: submitted.append(buff.text) or False))
        _binding(ui, "_enter").handler(SimpleNamespace(app=None))
        self.assertEqual(submitted, [])

    def _ui_with_unhighlighted_menu(self, text, completions, cursor=None):
        """complete-while-typing state: menu open, NOTHING highlighted."""
        ui = _make_ui()
        submitted = []
        ui.buffer = Buffer(multiline=True, accept_handler=(
            lambda buff: submitted.append(buff.text) or False))
        ui.submitted = submitted
        ui.buffer.insert_text(text)
        doc = Document(text, cursor if cursor is not None else len(text))
        ui.buffer.complete_state = CompletionState(
            original_document=doc, completions=completions,
            complete_index=None)
        return ui

    def test_enter_fills_the_first_completion_when_none_is_highlighted(self):
        ui = self._ui_with_unhighlighted_menu(
            "/he", [Completion("/help", start_position=-3),
                    Completion("/health", start_position=-3)])
        _binding(ui, "_enter").handler(SimpleNamespace(app=None))
        self.assertEqual(ui.buffer.text, "/help")
        self.assertEqual(ui.submitted, [])

    def test_enter_runs_a_fully_typed_command_instead_of_eating_it(self):
        ui = self._ui_with_unhighlighted_menu(
            "/help", [Completion("/help", start_position=-5)])
        _binding(ui, "_enter").handler(SimpleNamespace(app=None))
        self.assertEqual(ui.submitted, ["/help"],
                         "a complete-looking command must run on Enter")

    def test_completion_fill_check_handles_the_replaced_span(self):
        from fullagent.tui import _completion_fills_text

        # typing "/he" -> "/help" changes the text (fill, don't submit)
        self.assertTrue(_completion_fills_text(
            Completion("/help", start_position=-3), Document("/he", 3)))
        # already fully typed -> nothing to insert (run it on Enter)
        self.assertFalse(_completion_fills_text(
            Completion("/help", start_position=-5), Document("/help", 5)))
        # mid-line prefix -> only that span is replaced
        self.assertTrue(_completion_fills_text(
            Completion("/help", start_position=-3),
            Document("cmd /he", 7)))
        # a start_position that would reach before the buffer never crashes
        self.assertTrue(_completion_fills_text(
            Completion("/help", start_position=-99), Document("/he", 3)))


class FrameShrinkTests(unittest.TestCase):
    """Closing a picker / clearing the box must re-anchor the frame height.

    prompt_toolkit renders a non-full-screen frame at
    ``max(last_height, preferred_height)``, so a shrinking layout used to
    leave the prompt box tall with a hollow interior (the freed rows were
    handed to the flexible input window).
    """

    def _ui_with_picker(self):
        ui = _make_ui()
        ui.app = SimpleNamespace(renderer=_FakeRenderer(),
                                 invalidate=lambda: None)
        ui.picked = []
        ui.overlay = OverlayList("SELECT MODEL",
                                 [("one", "1"), ("two", "2")], 0,
                                 on_select=ui.picked.append)
        ui.overlay.open()
        return ui

    def test_escape_closes_the_picker_and_resets_the_frame(self):
        ui = self._ui_with_picker()
        _binding(ui, "_ov_close").handler(SimpleNamespace(app=ui.app))
        self.assertFalse(ui.overlay.visible)
        self.assertEqual(ui.app.renderer.erased, 1,
                         "closed picker left the frame at its old height")

    def test_selecting_in_the_picker_resets_the_frame(self):
        ui = self._ui_with_picker()
        _binding(ui, "_ov_enter").handler(SimpleNamespace(app=ui.app))
        self.assertFalse(ui.overlay.visible)
        self.assertEqual(ui.picked, [0])
        self.assertEqual(ui.app.renderer.erased, 1)

    def test_multiline_submit_resets_the_frame(self):
        order = []
        ui = _make_ui()
        ui.app = SimpleNamespace(renderer=_FakeRenderer(),
                                 invalidate=lambda: None)
        ui.app.renderer.erase = lambda *a, **k: order.append("erase")

        def dispatch(text):
            order.append("dispatch")

        ui._dispatch = dispatch
        buff = Buffer(multiline=True)
        buff.insert_text("line one\nline two")
        self.assertFalse(ui._accept_handler(buff))
        self.assertEqual(order, ["erase", "dispatch"],
                         "the frame must be re-anchored before the turn "
                         "thread prints its first line")

    def test_single_line_submit_does_not_reset_the_frame(self):
        ui = _make_ui()
        ui.app = SimpleNamespace(renderer=_FakeRenderer(),
                                 invalidate=lambda: None)
        ui._dispatch = lambda text: None
        buff = Buffer(multiline=True)
        buff.insert_text("one line")
        ui._accept_handler(buff)
        self.assertEqual(ui.app.renderer.erased, 0)

    def test_reset_frame_survives_a_broken_renderer(self):
        def boom(*_a, **_k):
            raise RuntimeError("boom")

        ui = _make_ui()
        ui.app = SimpleNamespace(renderer=SimpleNamespace(erase=boom),
                                 invalidate=lambda: None)
        ui._reset_frame()  # must not raise into the key handler


class RenderCallbackIOTests(unittest.TestCase):
    """Render callbacks must never block on subprocess or disk I/O."""

    def test_slow_git_never_blocks_the_footer(self):
        ui = _make_ui()
        calls = []

        def slow_run(*args, **kwargs):
            calls.append(args)
            time.sleep(0.4)
            return SimpleNamespace(returncode=0, stdout="main\n")

        with patch("fullagent.tui.shutil.which", return_value="/usr/bin/git"), \
                patch("subprocess.run", side_effect=slow_run):
            start = time.perf_counter()
            ui._meta_fragments()
            frame_ms = (time.perf_counter() - start) * 1000.0
            self.assertLess(frame_ms, 50.0,
                            f"footer blocked {frame_ms:.0f}ms on git")
            start = time.perf_counter()
            ui._meta_fragments()
            again_ms = (time.perf_counter() - start) * 1000.0
            self.assertLess(again_ms, 50.0,
                            f"second footer blocked {again_ms:.0f}ms")
            deadline = time.time() + 10.0
            while time.time() < deadline and ui._branch_probe_running:
                time.sleep(0.01)
        self.assertTrue(calls, "the branch probe never ran")
        self.assertEqual(ui._branch_cache, "main",
                         "background probe did not resolve the real branch")

    def test_branch_fallback_when_there_is_no_repo(self):
        ui = _make_ui()
        with patch("fullagent.tui.shutil.which", return_value=None):
            label = ui._branch_label()
            self.assertEqual(label, "session abcd1234")
            deadline = time.time() + 5.0
            while time.time() < deadline and ui._branch_probe_running:
                time.sleep(0.01)
        self.assertEqual(ui._branch_cache, "session abcd1234")

    def test_skill_md_scan_never_blocks_the_hud(self):
        ui = _make_ui(skill_forge=SimpleNamespace(registry={"a": None,
                                                            "b": None}))

        class SlowManager:
            calls = 0

            def list_skills(self):
                SlowManager.calls += 1
                time.sleep(0.4)
                return [{"name": "disk-skill", "description": "",
                         "path": "x"}]

        with patch("fullagent.skills.SkillMDManager", SlowManager):
            start = time.perf_counter()
            hud = "".join(t for _s, t in ui._hud_fragments())
            hud_ms = (time.perf_counter() - start) * 1000.0
            self.assertLess(hud_ms, 50.0,
                            f"HUD blocked {hud_ms:.0f}ms on the skills scan")
            self.assertIn("2 skills", hud,
                          "the in-memory forge registry must count inline")
            deadline = time.time() + 10.0
            while time.time() < deadline and ui._skills_scan_running:
                time.sleep(0.01)
        self.assertEqual(SlowManager.calls, 1, "scan must be single-flight")
        self.assertEqual(ui._skills_count(), 3,
                         "background scan results were dropped")

    def test_skills_scan_failure_keeps_the_cheap_count(self):
        ui = _make_ui(skill_forge=SimpleNamespace(registry={"a": None}))

        class Broken:
            def list_skills(self):
                raise OSError("permission denied")

        with patch("fullagent.skills.SkillMDManager", Broken):
            self.assertEqual(ui._skills_count(), 1)
            deadline = time.time() + 5.0
            while time.time() < deadline and ui._skills_scan_running:
                time.sleep(0.01)
        self.assertEqual(ui._skills_count(), 1)


if __name__ == "__main__":
    unittest.main()
