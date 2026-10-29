"""Worker 1/20 proof tests: subagents.py spawn/wait/timeout regressions.

Each test uses the real ParallelSubagents class with an injected fake chat
callable (never mocks of the code under test). They FAIL on the buggy code
and PASS after the fix.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import inspect
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
from fullagent.subagents import (ACTIVE_STATES, TERMINAL_STATES,
                                 ParallelSubagents, SubagentError)


class MemoryLog:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def append(self, kind, data, **kwargs):
        with self.lock:
            event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
            self.events.append((kind, data, kwargs, event.id))
            return event


def result(content="", tool_calls=()):
    return SimpleNamespace(content=content, tool_calls=list(tool_calls),
                           usage={"prompt_tokens": 5, "completion_tokens": 3},
                           reasoning="")


def make_manager(tmp_path, chat, **kwargs):
    model = SimpleNamespace(id="test-model", provider="test",
                            supports_tools=True)
    provider = SimpleNamespace(id="test")
    effort = SimpleNamespace(key="low")
    log = MemoryLog()
    manager = ParallelSubagents(
        log, {}, lambda _model_id: (provider, model, effort),
        workspace=tmp_path, max_workers=4, max_active=8,
        max_steps=4, chat=chat, **kwargs)
    return manager, log


def cancellable_chat(entered):
    """A chat callable that blocks until cancelled, like a long model call."""
    def chat(*_args, should_cancel=None, **_kwargs):
        entered.set()
        while True:
            if should_cancel is not None and should_cancel():
                break
            time.sleep(0.01)
        return result("wound down")
    return chat


def test_wait_cancel_stops_awaited_workers(tmp_path):
    """REGRESSION vs crew.py: old wait() called force_stop() on Esc; the new
    wait() raises TurnCancelled but leaves the workers running, burning
    model calls invisibly after the turn dies."""
    entered = threading.Event()
    manager, _log = make_manager(tmp_path, cancellable_chat(entered))
    try:
        worker = manager.spawn("do a slow thing")
        assert entered.wait(2)
        cancel_at = time.monotonic() + 0.3
        with pytest.raises(TurnCancelled):
            manager.wait([worker.id], timeout=10,
                         should_cancel=lambda: time.monotonic() >= cancel_at)
        # The awaited worker must have been asked to stop...
        assert worker.stop_event.is_set(), \
            "BUG: wait() raised TurnCancelled but never stopped the worker"
        # ...and it must actually wind down instead of running forever.
        states = manager.wait([worker.id], timeout=5)
        assert states[worker.id] == "cancelled"
    finally:
        manager.shutdown(wait=True)


def test_wait_cancel_leaves_unawaited_workers_alone(tmp_path):
    """Stopping on Esc must be scoped to the ids being waited on."""
    entered = threading.Event()
    manager, _log = make_manager(tmp_path, cancellable_chat(entered))
    try:
        waited = manager.spawn("waited task")
        other = manager.spawn("background task")
        assert entered.wait(2)
        cancel_at = time.monotonic() + 0.3
        with pytest.raises(TurnCancelled):
            manager.wait([waited.id], timeout=10,
                         should_cancel=lambda: time.monotonic() >= cancel_at)
        assert waited.stop_event.is_set()
        assert not other.stop_event.is_set()
        assert manager.wait([waited.id], timeout=5)[waited.id] == "cancelled"
        assert manager.get(other.id).state in ACTIVE_STATES
    finally:
        manager.stop(other.id)
        manager.shutdown(wait=True)


def test_spawn_many_submit_failure_terminalizes_every_worker(tmp_path):
    """If pool.submit raises mid-batch (shutdown race), workers after the
    failure must still reach a terminal state + subagent.done — otherwise
    they sit in 'queued' forever, leak max_active capacity, and can never
    be pruned."""
    manager, log = make_manager(tmp_path, lambda *a, **k: result("ok"))
    # Simulate the race: the pool is shut down but the manager's _closed
    # flag is still False when spawn_many performs its check.
    manager._pool.shutdown(wait=True)
    try:
        with pytest.raises(SubagentError):
            manager.spawn_many(["one", "two", "three"])
        workers = manager.list()
        assert len(workers) == 3
        assert all(w.state in TERMINAL_STATES for w in workers), \
            "BUG: unsubmitted workers stranded in %s" % \
            [w.state for w in workers]
        assert all(w.finished_at > 0 for w in workers)
        done_events = [e for e in log.events if e[0] == "subagent.done"]
        assert len(done_events) == 3, \
            f"BUG: only {len(done_events)}/3 workers emitted subagent.done"
        leaked = [w.id for w in workers if w.state in ACTIVE_STATES]
        assert not leaked, f"BUG: capacity leaked by {leaked}"
    finally:
        manager.shutdown(wait=True)


def test_shutdown_default_waits_for_inflight_workers():
    """REGRESSION vs crew.py: shutdown(wait=True) was the old default;
    the rewrite flipped it to wait=False, so abandoned non-daemon pool
    threads can keep running (and the process alive) after shutdown."""
    sig = inspect.signature(ParallelSubagents.shutdown)
    assert sig.parameters["wait"].default is True, \
        "BUG: shutdown() default flipped from wait=True to wait=False"


def test_shutdown_waits_by_default(tmp_path):
    entered = threading.Event()
    manager, _log = make_manager(tmp_path, cancellable_chat(entered))
    worker = manager.spawn("slow job")
    assert entered.wait(2)
    manager.shutdown()  # default wait behavior
    assert worker.state in TERMINAL_STATES, \
        f"BUG: shutdown() returned while worker was {worker.state}"
    assert worker.state == "cancelled"


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
