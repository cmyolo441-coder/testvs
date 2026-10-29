"""Cancellation-path regression tests for fullagent.subagents (worker 11/20).

Uses REAL ParallelSubagents/Subagent classes with injectable chats —
no mocks of the code under test.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import threading
import time
from types import SimpleNamespace

try:
    import pytest
except ImportError:
    import importlib.util as _ilu
    _compat_spec = _ilu.spec_from_file_location(
        "pytest", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
    )
    pytest = _ilu.module_from_spec(_compat_spec)
    _sys.modules["pytest"] = pytest
    _sys.modules["tests._pytest_compat"] = pytest
    _compat_spec.loader.exec_module(pytest)


from fullagent.cancelguard import TurnCancelled
from fullagent.subagents import DRAIN_TIMEOUT, ParallelSubagents


class MemoryLog:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def append(self, kind, data, **kwargs):
        with self.lock:
            event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
            self.events.append((kind, data, kwargs, event.id))
            return event


def _result(content=""):
    return SimpleNamespace(
        content=content, tool_calls=[],
        usage={"prompt_tokens": 1, "completion_tokens": 1}, reasoning="")


def _manager(tmp_path, chat, **kwargs):
    model = SimpleNamespace(id="test-model", supports_tools=True)
    provider = SimpleNamespace(id="test")
    effort = SimpleNamespace(key="low")
    return ParallelSubagents(
        MemoryLog(), {}, lambda _mid: (provider, model, effort),
        workspace=tmp_path, max_workers=kwargs.get("max_workers", 4),
        max_active=kwargs.get("max_active", 8),
        max_steps=kwargs.get("max_steps", 4), chat=chat)


def _slow_cancellable_chat(*_args, should_cancel=None, **_kwargs):
    for _ in range(80):  # ~4s of work, polling cancellation every 50ms
        if should_cancel and should_cancel():
            raise TurnCancelled()
        time.sleep(0.05)
    return _result("slow done")


def _pool_threads():
    return [t for t in threading.enumerate()
            if t.is_alive() and "fullagent:subagent" in (t.name or "")]


def test_midrun_stop_is_prompt_and_leaks_no_threads(tmp_path):
    manager = _manager(tmp_path, _slow_cancellable_chat)
    try:
        workers = manager.spawn_many([f"task {i}" for i in range(4)])
        time.sleep(0.3)  # let every worker enter its model call
        started = time.monotonic()
        for worker in workers:
            manager.stop(worker.id)
        states = manager.wait([w.id for w in workers], timeout=10)
        elapsed = time.monotonic() - started
        assert elapsed < DRAIN_TIMEOUT, f"cancel took {elapsed:.1f}s"
        assert set(states.values()) == {"cancelled"}, states
    finally:
        manager.shutdown(wait=True)
    deadline = time.monotonic() + 5
    while _pool_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pool_threads(), "pool threads leaked after cancel+shutdown"


def test_wait_cancel_stops_the_waited_on_workers(tmp_path):
    manager = _manager(tmp_path, _slow_cancellable_chat, max_workers=2)
    try:
        workers = manager.spawn_many(["alpha", "beta"])
        time.sleep(0.2)
        flag = {"cancel": False}

        def trigger():
            time.sleep(0.8)
            flag["cancel"] = True

        threading.Thread(target=trigger, daemon=True).start()
        started = time.monotonic()
        with pytest.raises(TurnCancelled):
            manager.wait([w.id for w in workers], timeout=30,
                         should_cancel=lambda: flag["cancel"])
        elapsed = time.monotonic() - started
        assert elapsed < 5, f"cancel not prompt: {elapsed:.1f}s"
        # wait() stopped exactly what it was waiting on: no worker keeps
        # burning model calls after the turn dies.
        states = manager.wait([w.id for w in workers], timeout=10)
        assert set(states.values()) == {"cancelled"}, states
    finally:
        manager.shutdown(wait=True)


def test_reset_session_returns_promptly_with_uninterruptible_worker(tmp_path):
    release = threading.Event()
    calls = {"n": 0}

    def stuck_once_chat(*_args, **_kwargs):
        # Simulates a provider call that ignores should_cancel entirely.
        calls["n"] += 1
        if calls["n"] == 1:
            release.wait(timeout=60)
        return _result("done")

    manager = _manager(tmp_path, stuck_once_chat, max_workers=2)
    try:
        manager.spawn("stuck task")
        time.sleep(0.3)  # worker is now wedged inside the chat call
        done = threading.Event()
        outcome = {}

        def do_reset():
            try:
                manager.reset_session("session-2")
            except Exception as exc:  # noqa: BLE001 - record, don't lose
                outcome["error"] = exc
            finally:
                done.set()

        started = time.monotonic()
        thread = threading.Thread(target=do_reset, daemon=True)
        thread.start()
        assert done.wait(timeout=DRAIN_TIMEOUT + 5), \
            "reset_session hung on an uninterruptible worker"
        elapsed = time.monotonic() - started
        assert "error" not in outcome, outcome["error"]
        assert elapsed < DRAIN_TIMEOUT + 5, f"reset took {elapsed:.1f}s"
        thread.join(timeout=5)
        # The manager is still usable for the new session.
        fresh = manager.spawn("fresh task")
        states = manager.wait([fresh.id], timeout=15)
        assert states[fresh.id] == "done", states
    finally:
        release.set()  # let the wedged worker finish on its own
        manager.shutdown(wait=True)


def test_shutdown_wait_returns_promptly_with_uninterruptible_worker(tmp_path):
    release = threading.Event()

    def stuck_chat(*_args, **_kwargs):
        release.wait(timeout=60)  # ignores should_cancel entirely
        return _result("stuck done")

    manager = _manager(tmp_path, stuck_chat, max_workers=2)
    manager.spawn("stuck task")
    time.sleep(0.3)
    started = time.monotonic()
    manager.shutdown(wait=True)
    elapsed = time.monotonic() - started
    release.set()  # let the wedged worker finish so nothing leaks
    assert elapsed < DRAIN_TIMEOUT + 5, f"shutdown hung {elapsed:.1f}s"
    deadline = time.monotonic() + 10
    while _pool_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _pool_threads(), "pool thread leaked after shutdown"


if __name__ == "__main__":
    import importlib.util as _ilu
    if "tests._pytest_compat" not in _sys.modules:
        _cs = _ilu.spec_from_file_location(
            "tests._pytest_compat", str(_Path(__file__).resolve().parent / "_pytest_compat.py")
        )
        _cm = _ilu.module_from_spec(_cs)
        _sys.modules["tests._pytest_compat"] = _cm
        _cs.loader.exec_module(_cm)
    raise SystemExit(_sys.modules["tests._pytest_compat"].run_module_tests(_sys.modules[__name__]))
