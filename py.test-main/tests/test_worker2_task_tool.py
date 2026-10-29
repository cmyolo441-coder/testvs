"""Worker 2/20 — deep bug hunt: Task tool integration (agent.py <-> subagents.py).

Drives the REAL Task/TaskOutput/TaskCollect/TaskList/TaskStop handlers
registered by Agent._register_subagent_tools, against the REAL
ParallelSubagents manager. The only injected seam is the ``chat`` callable
(a constructor-level dependency, exactly like tests/test_subagents.py does);
nothing under test is mocked.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import inspect
import re
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
from fullagent.subagents import ParallelSubagents
from fullagent.tools import Tool


class MemoryLog:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def append(self, kind, data, **kwargs):
        with self.lock:
            event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
            self.events.append((kind, data, kwargs, event.id))
            return event


def _result(content="", tool_calls=(), usage=None):
    return SimpleNamespace(
        content=content, tool_calls=list(tool_calls),
        usage=usage or {"prompt_tokens": 5, "completion_tokens": 3},
        reasoning="")


def _make_agent(monkeypatch, tmp_path, chat):
    """Build a real Agent far enough to register the Task tool family."""
    from fullagent.agent import Agent
    model = SimpleNamespace(id="test-model", provider="test",
                            supports_tools=True)
    provider = SimpleNamespace(id="test")
    effort = SimpleNamespace(key="low")
    agent = Agent.__new__(Agent)
    agent.log = MemoryLog()
    agent.tools = {}
    agent._subagent_runtime = lambda _model_id="": (provider, model, effort)
    agent._snapshot_subagent_write = lambda *_a: None
    agent._after_subagent_write = lambda *_a: ""
    monkeypatch.chdir(tmp_path)
    agent._register_subagent_tools()
    agent.subagents.chat = chat  # injected dependency, not code under test
    return agent


def _spawn_ids(start_message):
    m = re.search(r"Started batch (\S+): (\S+) ", start_message)
    assert m, f"unexpected Task reply: {start_message!r}"
    return m.group(1), m.group(2)


@pytest.fixture()
def agent(monkeypatch, tmp_path):
    ag = _make_agent(monkeypatch, tmp_path,
                     lambda *_a, **_k: _result("evidence-backed report"))
    try:
        yield ag
    finally:
        ag.subagents.shutdown(wait=True)


# -- full path: LLM calls Task -> subagent spawns -> TaskOutput returns report

def test_task_tool_full_round_trip(agent):
    task = agent.tools["Task"].handler
    msg = task(task="trace how retries work in client.py")
    assert msg.startswith("Started batch batch-")
    batch_id, worker_id = _spawn_ids(msg)

    report = agent.tools["TaskOutput"].handler(
        task_id=worker_id, wait_seconds=10)
    assert "evidence-backed report" in report
    assert "done" in report
    assert worker_id in report

    collected = agent.tools["TaskCollect"].handler(
        batch_id=batch_id, wait_seconds=10)
    assert "COLLECTED 1 SUBAGENT REPORT(S)" in collected
    assert "evidence-backed report" in collected

    status = agent.tools["TaskList"].handler()
    assert worker_id in status
    assert agent.tools["TaskStop"].handler(task_id=worker_id) is not None


def test_task_group_round_trip_reports_every_worker(agent):
    msg = agent.tools["TaskGroup"].handler(
        tasks=[{"task": "review auth", "role": "security"},
               {"task": "review retries", "role": "reviewer"}])
    assert "Launched batch batch-" in msg
    batch_id = re.search(r"Launched batch (\S+)", msg).group(1)
    collected = agent.tools["TaskCollect"].handler(
        batch_id=batch_id, wait_seconds=10)
    assert "COLLECTED 2 SUBAGENT REPORT(S)" in collected
    assert collected.count("evidence-backed report") == 2


# -- BUG 1 (real): wait() did not strip ids, so TaskOutput on a
#    whitespace-padded id skipped the wait and returned a premature report
#    (get() strips, wait() did not -> inconsistent).

def test_manager_wait_strips_worker_ids(tmp_path):
    release = threading.Event()

    def chat(*_a, **_k):
        release.wait(timeout=10)
        return _result("slow report")

    model = SimpleNamespace(id="m", provider="p", supports_tools=True)
    mgr = ParallelSubagents(
        MemoryLog(), {}, lambda _m: (SimpleNamespace(id="p"), model,
                                     SimpleNamespace(key="low")),
        workspace=tmp_path, chat=chat)
    try:
        worker = mgr.spawn("slow task")
        threading.Timer(0.6, release.set).start()
        t0 = time.monotonic()
        states = mgr.wait(["  " + worker.id + " "], timeout=10)
        dt = time.monotonic() - t0
        # the padded id must resolve to the real worker and the call must
        # actually wait for it instead of answering "missing" instantly.
        assert states == {worker.id: "done"}, states
        assert dt >= 0.5, f"wait returned after {dt:.2f}s without waiting"
    finally:
        release.set()
        mgr.shutdown(wait=True)


def test_task_output_waits_despite_whitespace_padded_id(
        monkeypatch, tmp_path):
    release = threading.Event()

    def chat(*_a, **_k):
        release.wait(timeout=10)
        return _result("patient report")

    agent = _make_agent(monkeypatch, tmp_path, chat)
    try:
        _batch, worker_id = _spawn_ids(
            agent.tools["Task"].handler(task="take your time"))
        handler = agent.tools["TaskOutput"].handler
        holder = {}
        thread = threading.Thread(
            target=lambda: holder.setdefault(
                "report", handler(task_id=f"  {worker_id}\t",
                                  wait_seconds=10)))
        thread.start()
        time.sleep(0.5)
        # the premature-report bug answered instantly with a live state;
        # the fixed code is still waiting here.
        assert thread.is_alive(), (
            "TaskOutput returned before the worker finished: "
            + repr(holder.get("report")))
        release.set()
        thread.join(timeout=10)
        report = holder["report"]
        assert "patient report" in report
        assert "done" in report
    finally:
        release.set()
        agent.subagents.shutdown(wait=True)


# -- adversarial inputs: every one must become an "ERROR:" string, never a
#    traceback escaping the handler.

def test_task_rejects_adversarial_inputs(agent):
    task = agent.tools["Task"].handler
    assert task(task="x", subagent_type="no-such-role").startswith("ERROR:")
    assert task(task="   ").startswith("ERROR:")
    assert task(task=None).startswith("ERROR:")
    assert task(task="x" * 12_001).startswith("ERROR:")
    assert task(task="x", model_id=None).startswith("ERROR:")
    assert task(task="x", context=None).startswith("ERROR:")
    group = agent.tools["TaskGroup"].handler
    assert group(tasks={"task": "not-a-list"}).startswith("ERROR:")
    assert group(tasks=[]).startswith("ERROR:"), \
        "empty TaskGroup must fail fast, not fabricate a 'batch none' launch"
    assert group(tasks=[{"task": "  "}]).startswith("ERROR:")
    wgroup = agent.tools["TaskWriteGroup"].handler
    assert wgroup(tasks=[]).startswith("ERROR:")
    collect = agent.tools["TaskCollect"].handler
    assert collect(task_ids=[]).startswith("ERROR:")
    assert collect(task_ids=["sa-1", "sa-1"]).startswith("ERROR:")
    assert collect(task_ids=["sa-nope"], wait_seconds=1).startswith("ERROR:")
    assert collect(batch_id="batch-9999", wait_seconds=1).startswith("ERROR:")
    assert collect(batch_id="batch-1", task_ids=["sa-1"]).startswith("ERROR:")


def test_task_output_unknown_id_is_error_string_not_exception(agent):
    out = agent.tools["TaskOutput"].handler(task_id="sa-4242",
                                            wait_seconds=1)
    assert out == "ERROR: unknown subagent: sa-4242"


def test_task_stop_unknown_id_is_error_string_not_exception(agent):
    out = agent.tools["TaskStop"].handler(task_id="sa-4242")
    assert out.startswith("ERROR: unknown subagent")


def test_task_collect_with_no_batches_yet(agent):
    out = agent.tools["TaskCollect"].handler(wait_seconds=1)
    assert out == "No retained subagents in that batch."


def test_task_list_with_no_subagents(agent):
    out = agent.tools["TaskList"].handler()
    assert "No subagents yet" in out


# -- Esc during a TaskOutput wait must abort the turn (TurnCancelled
#    propagates), not be swallowed into an "ERROR:" tool result.

def test_esc_during_task_output_wait_raises_turn_cancelled(
        monkeypatch, tmp_path):
    release = threading.Event()

    def chat(*_a, **_k):
        release.wait(timeout=15)
        return _result("too late")

    agent = _make_agent(monkeypatch, tmp_path, chat)
    try:
        _batch, worker_id = _spawn_ids(
            agent.tools["Task"].handler(task="hang on"))
        cancel_at = time.monotonic() + 0.4
        with pytest.raises(TurnCancelled):
            agent.tools["TaskOutput"].handler(
                task_id=worker_id, wait_seconds=10,
                should_cancel=lambda: time.monotonic() >= cancel_at)
    finally:
        release.set()
        agent.subagents.shutdown(wait=True)


# -- a crashing worker must surface state=error with the real error text.

def test_worker_crash_surfaces_error_in_task_output(
        monkeypatch, tmp_path):
    def chat(*_a, **_k):
        raise RuntimeError("provider exploded")

    agent = _make_agent(monkeypatch, tmp_path, chat)
    try:
        _batch, worker_id = _spawn_ids(
            agent.tools["Task"].handler(task="doomed"))
        report = agent.tools["TaskOutput"].handler(
            task_id=worker_id, wait_seconds=10)
        assert "error" in report
        assert "provider exploded" in report
        assert "RuntimeError" in report
    finally:
        agent.subagents.shutdown(wait=True)


# -- large results must be bounded, not dumped raw into the turn.

def test_large_worker_summary_is_bounded_in_reports(
        monkeypatch, tmp_path):
    agent = _make_agent(
        monkeypatch, tmp_path,
        lambda *_a, **_k: _result("Z" * 30_000))
    try:
        _batch, worker_id = _spawn_ids(
            agent.tools["Task"].handler(task="be verbose"))
        report = agent.tools["TaskOutput"].handler(
            task_id=worker_id, wait_seconds=10)
        assert "Z" * 30_000 not in report
        assert len(report) < 12_000, len(report)
    finally:
        agent.subagents.shutdown(wait=True)


# -- schema <-> handler signature agreement for the whole Task family.

def test_task_tool_schemas_match_handler_signatures(agent):
    for name in ("Task", "TaskGroup", "TaskWriteGroup", "TaskOutput",
                 "TaskCollect", "TaskList", "TaskStop"):
        tool = agent.tools[name]
        schema = tool.parameters
        props = set(schema.get("properties", {}))
        sig = inspect.signature(tool.handler)
        params = set(sig.parameters)
        assert props <= params, (name, props - params)
        for req in schema.get("required", []):
            assert req in props, (name, req)
        # every declared property must bind as a keyword argument
        sig.bind(**{p: None for p in props},
                 **({"should_cancel": None}
                    if "should_cancel" in params else {}))


# -- defense in depth: a write-capable role via the read-only Task tool
#    must not receive write tools.

def test_task_with_coder_role_stays_read_only(agent):
    agent.tools["read_file"] = Tool("read_file", "read",
                                    {"type": "object"},
                                    lambda path: "contents")
    agent.tools["write_file"] = Tool("write_file", "write",
                                     {"type": "object"},
                                     lambda path, content: "written")
    _batch, worker_id = _spawn_ids(
        agent.tools["Task"].handler(task="plan the change",
                                    subagent_type="coder"))
    worker = agent.subagents.get(worker_id)
    assert "write_file" not in worker.tools
    report = agent.tools["TaskOutput"].handler(
        task_id=worker_id, wait_seconds=10)
    assert "done" in report


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
