"""Worker 5 proof tests: Esc/Ctrl+C must stop running subagents (rewrite
regression), the /agents dashboard renders the real subagents snapshot,
and /agents spawn accepts any-case role names.

Drives REAL code: fullagent.tui.UI._request_cancel,
UI._print_agents_dashboard, UI._cmd_agents against a REAL
fullagent.subagents.ParallelSubagents manager with REAL Subagent rows
(offline chat stub, no network, no mocks of code under test).
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

from fullagent.subagents import ACTIVE_STATES, ParallelSubagents
from fullagent.tui import UI


class _Log:
    """Minimal event log satisfying ParallelSubagents' log contract."""

    def __init__(self):
        self.events = []

    def append(self, kind, data, **kwargs):
        event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
        self.events.append((kind, data, kwargs))
        return event


def _offline_chat(*_args, **_kwargs):
    return SimpleNamespace(content="offline result", tool_calls=[],
                           usage={"prompt_tokens": 1, "completion_tokens": 1})


def _make_manager(chat=_offline_chat, max_workers=2, max_active=2):
    provider = SimpleNamespace(id="offline")
    model = SimpleNamespace(id="offline-model", supports_tools=True)
    effort = SimpleNamespace(key="low")
    manager = ParallelSubagents(
        _Log(), {},
        lambda _model_id: (provider, model, effort),
        max_workers=max_workers, max_active=max_active, chat=chat)
    return manager


def _make_ui(agent_ns):
    ui = UI.__new__(UI)
    ui.cfg = SimpleNamespace(extra={}, show_reasoning=False)
    ui._busy = True
    ui._cancel_flag = threading.Event()
    ui.output = io.StringIO()
    ui.console = Console(file=ui.output, width=100, color_system=None)
    ui.agent = agent_ns
    return ui


class EscStopsSubagentsTests(unittest.TestCase):
    def test_request_cancel_stops_running_subagent(self):
        """Esc path must stop a live worker and leave the manager usable."""
        release = threading.Event()

        def blocking_chat(*_a, **_k):
            release.wait(10)  # bounded: never hangs the suite
            return _offline_chat()

        manager = _make_manager(chat=blocking_chat)
        try:
            worker = manager.spawn("long investigation", role="researcher")
            deadline = time.monotonic() + 5
            while worker.state not in ACTIVE_STATES and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIn(worker.state, ACTIVE_STATES,
                          f"worker never started: {worker.state}")

            agent_ns = SimpleNamespace(
                _cancel_flag=threading.Event())
            agent_ns.__dict__["subagents"] = manager
            ui = _make_ui(agent_ns)

            UI._request_cancel(ui)  # real TUI cancel path (Esc/Ctrl+C)

            self.assertTrue(ui._cancel_flag.is_set())
            self.assertTrue(agent_ns._cancel_flag.is_set())
            self.assertTrue(worker.stop_event.is_set(),
                            "stop_event not set — worker keeps running")
            self.assertNotIn(worker.state, {"running", "queued"},
                             f"worker still active after cancel: {worker.state}")

            # manager must NOT be destroyed by the cancel: spawn again
            release.set()
            manager.wait([worker.id], timeout=5)
            worker2 = manager.spawn("follow-up task", role="researcher")
            manager.wait([worker2.id], timeout=5)
            self.assertEqual(worker2.state, "done")
        finally:
            release.set()
            manager.shutdown(wait=True)

    def test_request_cancel_without_manager_never_raises(self):
        """Cancel on an agent with no subagents runtime must be a no-op."""
        agent_ns = SimpleNamespace(_cancel_flag=threading.Event())
        ui = _make_ui(agent_ns)
        UI._request_cancel(ui)  # must not raise
        self.assertTrue(ui._cancel_flag.is_set())
        self.assertTrue(agent_ns._cancel_flag.is_set())

    def test_request_cancel_with_broken_manager_never_raises(self):
        """A manager whose snapshot()/stop() throws must not break Esc."""

        class BadManager:
            def snapshot(self):
                raise RuntimeError("boom")

        agent_ns = SimpleNamespace(_cancel_flag=threading.Event())
        agent_ns.__dict__["subagents"] = BadManager()
        ui = _make_ui(agent_ns)
        UI._request_cancel(ui)  # must not raise
        self.assertTrue(ui._cancel_flag.is_set())


class AgentsDashboardTests(unittest.TestCase):
    def test_dashboard_renders_real_snapshot(self):
        manager = _make_manager()
        try:
            w1 = manager.spawn("first task", role="researcher")
            w2 = manager.spawn("second task", role="analyst")
            states = manager.wait([w1.id, w2.id], timeout=10)
            self.assertEqual(states[w1.id], "done")
            self.assertEqual(states[w2.id], "done")

            ui = _make_ui(SimpleNamespace(_cancel_flag=threading.Event()))
            UI._print_agents_dashboard(ui, manager)  # real render

            out = ui.output.getvalue()
            self.assertIn("Parallel crew", out)
            self.assertIn(w1.id, out)
            self.assertIn(w2.id, out)
            self.assertIn("2 done", out)
            self.assertIn("DONE", out)
            self.assertIn("Codebase Researcher", out)
        finally:
            manager.shutdown(wait=True)

    def test_dashboard_empty_manager(self):
        manager = _make_manager()
        try:
            ui = _make_ui(SimpleNamespace(_cancel_flag=threading.Event()))
            UI._print_agents_dashboard(ui, manager)
            out = ui.output.getvalue()
            self.assertIn("Parallel crew", out)
            self.assertIn("No workers yet", out)
        finally:
            manager.shutdown(wait=True)


class AgentsSpawnTests(unittest.TestCase):
    def _ui_with_manager(self, manager):
        agent_ns = SimpleNamespace(
            _cancel_flag=threading.Event(),
            _ensure_session=lambda: None,
            subagents=manager)
        return _make_ui(agent_ns)

    def test_spawn_role_is_case_insensitive(self):
        manager = _make_manager()
        try:
            ui = self._ui_with_manager(manager)
            UI._cmd_agents(ui, "spawn RESEARCHER investigate the auth flow")
            worker = manager.list()[-1]
            self.assertEqual(worker.role, "researcher")
            self.assertEqual(worker.task, "investigate the auth flow")
            manager.wait([worker.id], timeout=10)
        finally:
            manager.shutdown(wait=True)

    def test_spawn_unknown_first_word_stays_task_text(self):
        manager = _make_manager()
        try:
            ui = self._ui_with_manager(manager)
            UI._cmd_agents(ui, "spawn frobnicate the build system")
            worker = manager.list()[-1]
            self.assertEqual(worker.role, "researcher")
            self.assertEqual(worker.task, "frobnicate the build system")
            manager.wait([worker.id], timeout=10)
        finally:
            manager.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
