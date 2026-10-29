
import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)

import json
import threading
import time
from pathlib import Path
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
from fullagent.subagents import ParallelSubagents, SubagentError
from fullagent.tools import RISK_CONFIRM, Tool


class MemoryLog:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def append(self, kind, data, **kwargs):
        with self.lock:
            event = SimpleNamespace(id=f"event-{len(self.events) + 1}")
            self.events.append((kind, data, kwargs, event.id))
            return event


def result(content="", tool_calls=(), usage=None):
    return SimpleNamespace(content=content, tool_calls=list(tool_calls),
                           usage=usage or {"prompt_tokens": 5,
                                           "completion_tokens": 3},
                           reasoning="")


def call(name, args, call_id="call-1"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}


def make_manager(tmp_path, chat, *, tools=None, before_write=None,
                 after_write=None, max_workers=4, max_active=8):
    model = SimpleNamespace(id="test-model", provider="test",
                            supports_tools=True)
    provider = SimpleNamespace(id="test")
    effort = SimpleNamespace(key="low")
    log = MemoryLog()
    manager = ParallelSubagents(
        log, tools or {}, lambda _model_id: (provider, model, effort),
        before_write=before_write, after_write=after_write,
        workspace=tmp_path, max_workers=max_workers, max_active=max_active,
        max_steps=4, chat=chat)
    return manager, log


def test_parallel_batch_really_runs_concurrently(tmp_path):
    rendezvous = threading.Barrier(2)

    def chat(*_args, **_kwargs):
        rendezvous.wait(timeout=2)
        return result("independent result")

    manager, _log = make_manager(tmp_path, chat, max_workers=2)
    try:
        workers = manager.spawn_many(["inspect module A", "inspect module B"])
        assert len(workers) == 2
        assert manager.active_count() == 2
        assert manager.wait([w.id for w in workers], timeout=3) == {
            workers[0].id: "done", workers[1].id: "done"}
        assert manager.active_count() == 0
    finally:
        manager.shutdown(wait=True)


def test_named_batch_collects_every_report_in_one_call(tmp_path):
    manager, _log = make_manager(
        tmp_path, lambda *_args, **_kwargs: result("evidence-backed report"))
    try:
        workers = manager.spawn_many([
            {"task": "inspect module alpha", "role": "reviewer"},
            {"task": "inspect module beta", "role": "security"},
        ])
        batch_id = workers[0].batch_id
        assert batch_id.startswith("batch-")
        assert {worker.batch_id for worker in workers} == {batch_id}

        report = manager.collect(batch_id=batch_id, timeout=3)

        assert "COLLECTED 2 SUBAGENT REPORT(S)" in report
        assert all(worker.id in report for worker in workers)
        assert "evidence-backed report" in report
        snapshot = manager.snapshot()
        assert snapshot["latest_batch_id"] == batch_id
        assert snapshot["counts"]["done"] == 2
        assert batch_id in manager.format_status()
    finally:
        manager.shutdown(wait=True)


def test_batch_collect_validates_selector_and_worker_bounds(tmp_path):
    manager, _log = make_manager(tmp_path, lambda *_a, **_k: result("ok"))
    try:
        worker = manager.spawn("inspect one thing")
        with pytest.raises(SubagentError, match="batch_id or worker_ids"):
            manager.collect(worker.batch_id, [worker.id], timeout=0)
        with pytest.raises(SubagentError, match="duplicates"):
            manager.collect(worker_ids=[worker.id, worker.id], timeout=0)
        with pytest.raises(SubagentError, match="unknown subagent"):
            manager.collect(worker_ids=["sa-missing"], timeout=0)
    finally:
        manager.shutdown(wait=True)


def test_read_only_coder_role_does_not_get_write_or_shell_tools(tmp_path):
    def chat(*_args, **_kwargs):
        return result("analysis complete")

    shell_calls = []
    registry = {
        "read_file": Tool("read_file", "read", {"type": "object"},
                           lambda path: "read"),
        "write_file": Tool("write_file", "write", {"type": "object"},
                            lambda path, content: "written"),
        "run_command": Tool("run_command", "shell", {"type": "object"},
                             lambda command: shell_calls.append(command)),
    }
    manager, _log = make_manager(tmp_path, chat, tools=registry)
    try:
        worker = manager.spawn("inspect implementation", role="coder")
        assert "read_file" in worker.tools
        assert "write_file" not in worker.tools
        assert "run_command" not in worker.tools
        assert manager.wait([worker.id], 2)[worker.id] == "done"
        assert shell_calls == []
    finally:
        manager.shutdown(wait=True)


def test_security_specialist_is_read_only_and_gets_focus_specific_prompt(tmp_path):
    from fullagent.systemprompt import subagent_prompt

    registry = {
        "read_file": Tool("read_file", "read", {"type": "object"},
                           lambda path: "read"),
        "write_file": Tool("write_file", "write", {"type": "object"},
                            lambda path, content: "written"),
    }
    manager, _log = make_manager(
        tmp_path, lambda *_args, **_kwargs: result("security review complete"),
        tools=registry)
    try:
        worker = manager.spawn("review trust boundaries", role="security")
        assert set(worker.tools) == {"read_file"}
        assert "trust boundaries" in subagent_prompt("security")
        assert "report only evidence-backed risks" in subagent_prompt("security")
        assert manager.wait([worker.id], timeout=2)[worker.id] == "done"
    finally:
        manager.shutdown(wait=True)


def test_batch_validation_is_atomic(tmp_path):
    manager, log = make_manager(tmp_path, lambda *_a, **_k: result("ok"))
    try:
        with pytest.raises(SubagentError, match="non-empty task"):
            manager.spawn_many(["valid task", {"task": "   "}])
        assert manager.list() == []
        assert not log.events
    finally:
        manager.shutdown(wait=True)


def test_writer_cannot_escape_its_confirmed_path_scope(tmp_path):
    writes = []
    registry = {
        "write_file": Tool("write_file", "write", {"type": "object"},
                            lambda path, content: writes.append(path) or "OK"),
    }
    responses = 0

    def chat(_provider, _model, _effort, messages, _schemas, **_kwargs):
        nonlocal responses
        responses += 1
        if not any(m.get("role") == "tool" for m in messages):
            return result(tool_calls=[call(
                "write_file", {"path": "outside.py", "content": "bad"})])
        return result("finished")

    manager, log = make_manager(tmp_path, chat, tools=registry)
    try:
        worker = manager.spawn("make a scoped edit", role="coder",
                               write_scope=["allowed.py"], allow_writes=True)
        assert manager.wait([worker.id], 2)[worker.id] == "done"
        assert writes == []
        tool_event = next(e for e in log.events if e[0] == "subagent.tool")
        assert tool_event[1]["status"] == "error"
        assert "outside this worker's write_scope" in tool_event[1]["preview"]
    finally:
        manager.shutdown(wait=True)


def test_valid_scoped_write_requires_snapshot_and_runs_postcheck(tmp_path):
    target = tmp_path / "allowed.py"
    target.write_text("before\n", encoding="utf-8")
    snapshots = []
    postchecks = []
    registry = {
        "write_file": Tool("write_file", "write", {"type": "object"},
                            lambda path, content: (Path(path).write_text(
                                content, encoding="utf-8") and "OK")),
    }
    responses = 0

    def before(worker, tool, args):
        snapshots.append((worker.id, tool.name, target.read_text()))
        return None

    def after(worker, tool, args):
        postchecks.append((worker.id, tool.name, target.read_text()))
        return "post-write check passed"

    def chat(_provider, _model, _effort, messages, _schemas, **_kwargs):
        nonlocal responses
        responses += 1
        if not any(m.get("role") == "tool" for m in messages):
            return result(tool_calls=[call(
                "write_file", {"path": str(target), "content": "after\n"})])
        return result("edit complete")

    manager, _log = make_manager(tmp_path, chat, tools=registry,
                                 before_write=before, after_write=after)
    try:
        worker = manager.spawn("edit the scoped file", role="coder",
                               write_scope=[str(target)], allow_writes=True)
        assert manager.wait([worker.id], 2)[worker.id] == "done"
        assert snapshots == [(worker.id, "write_file", "before\n")]
        assert postchecks == [(worker.id, "write_file", "after\n")]
        assert worker.to_dict()["files_touched"] == [str(target)]
        tool_message = next(m for m in worker.messages if m.get("role") == "tool")
        assert "post-write check passed" in tool_message["content"]
    finally:
        manager.shutdown(wait=True)


def test_snapshot_failure_blocks_the_actual_write(tmp_path):
    writes = []
    registry = {
        "write_file": Tool("write_file", "write", {"type": "object"},
                            lambda path, content: writes.append(path) or "OK"),
    }

    def chat(_provider, _model, _effort, messages, _schemas, **_kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            return result(tool_calls=[call(
                "write_file", {"path": "allowed.py", "content": "x"})])
        return result("done")

    manager, log = make_manager(
        tmp_path, chat, tools=registry,
        before_write=lambda *_: "ERROR: simulated snapshot failure")
    try:
        worker = manager.spawn("write scoped file", role="coder",
                               write_scope=["allowed.py"], allow_writes=True)
        manager.wait([worker.id], 2)
        assert writes == []
        tool_event = next(e for e in log.events if e[0] == "subagent.tool")
        assert tool_event[1]["status"] == "error"
        assert "snapshot failure" in tool_event[1]["preview"]
    finally:
        manager.shutdown(wait=True)


def test_overlapping_parallel_writer_scopes_rejected_before_spawn(tmp_path):
    manager, log = make_manager(tmp_path, lambda *_a, **_k: result("ok"))
    try:
        with pytest.raises(SubagentError, match="overlapping"):
            manager.spawn_many([
                {"task": "write one", "write_scope": ["src/"]},
                {"task": "write two", "write_scope": ["src/module.py"]},
            ], role="coder", allow_writes=True)
        assert manager.list() == []
        assert not log.events
    finally:
        manager.shutdown(wait=True)


def test_scope_rejects_traversal_and_paths_outside_workspace(tmp_path):
    manager, _log = make_manager(tmp_path, lambda *_a, **_k: result("ok"))
    try:
        with pytest.raises(SubagentError, match="cannot contain"):
            manager.spawn("bad", role="coder", write_scope=["../outside.py"],
                          allow_writes=True)
        with pytest.raises(SubagentError, match="escapes the workspace"):
            manager.spawn("bad", role="coder",
                          write_scope=[str(tmp_path.parent / "outside.py")],
                          allow_writes=True)
        assert manager.list() == []
    finally:
        manager.shutdown(wait=True)


def test_stop_cancels_an_inflight_model_call(tmp_path):
    entered = threading.Event()

    def chat(*_args, should_cancel=None, **_kwargs):
        entered.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if should_cancel and should_cancel():
                raise TurnCancelled()
            time.sleep(0.01)
        return result("unexpected")

    manager, _log = make_manager(tmp_path, chat)
    try:
        worker = manager.spawn("wait until stopped")
        assert entered.wait(1)
        manager.stop(worker.id)
        assert manager.wait([worker.id], 2)[worker.id] == "cancelled"
    finally:
        manager.shutdown(wait=True)


def test_stop_is_forwarded_into_cancellable_worker_tools(tmp_path):
    entered = threading.Event()

    def web_fetch(url, should_cancel=None):
        entered.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if should_cancel and should_cancel():
                raise TurnCancelled()
            time.sleep(0.01)
        return "unexpected network response"

    registry = {
        "web_fetch": Tool("web_fetch", "fetch", {"type": "object"},
                           web_fetch),
    }

    def chat(_provider, _model, _effort, messages, _schemas, **_kwargs):
        if not any(m.get("role") == "tool" for m in messages):
            return result(tool_calls=[call("web_fetch", {"url": "https://example.com"})])
        return result("unexpected")

    manager, _log = make_manager(tmp_path, chat, tools=registry)
    try:
        worker = manager.spawn("fetch and summarize a page")
        assert entered.wait(1)
        manager.stop(worker.id)
        assert manager.wait([worker.id], 2)[worker.id] == "cancelled"
    finally:
        manager.shutdown(wait=True)


def test_subagent_tools_are_registered_and_writer_tool_is_confirmation_gated(
        tmp_path, monkeypatch):
    from fullagent.agent import Agent

    class RegistrationLog(MemoryLog):
        pass

    model = SimpleNamespace(id="test-model", provider="test",
                            supports_tools=True)
    provider = SimpleNamespace(id="test")
    effort = SimpleNamespace(key="low")
    agent = Agent.__new__(Agent)
    agent.log = RegistrationLog()
    agent.tools = {}
    agent._subagent_runtime = lambda _model_id="": (provider, model, effort)
    agent._snapshot_subagent_write = lambda *_args: None
    agent._after_subagent_write = lambda *_args: ""
    monkeypatch.chdir(tmp_path)

    agent._register_subagent_tools()
    try:
        assert {"Task", "TaskGroup", "TaskWriteGroup", "TaskOutput",
                "TaskCollect", "TaskList", "TaskStop"} <= set(agent.tools)
        assert agent.tools["TaskWriteGroup"].risk == RISK_CONFIRM
        assert agent.subagents.workspace == tmp_path.resolve()
    finally:
        agent.subagents.shutdown(wait=True)


def test_tool_filter_surfaces_parallel_agent_tools_only_for_relevant_turns():
    from fullagent.toolfilter import select_tools

    registry = {name: object() for name in (
        "read_file", "Task", "TaskGroup", "TaskWriteGroup", "TaskOutput",
        "TaskCollect", "TaskList", "TaskStop")}
    plain = select_tools(registry, "show me the current status")
    parallel = select_tools(
        registry, "run a team of specialists in parallel for review")
    assert "TaskGroup" not in plain
    assert "TaskCollect" not in plain
    assert {"TaskGroup", "TaskWriteGroup", "TaskOutput", "TaskList",
            "TaskCollect", "TaskStop", "Task"} <= set(parallel)



def test_dag_wave_dependency_execution_and_context_piping(tmp_path):
    order = []
    lock = threading.Lock()

    def chat(_prov, _mod, _eff, messages, *_a, **_kw):
        user_msg = messages[-1]["content"]
        with lock:
            order.append(user_msg)
        if "TASK: step-alpha" in user_msg:
            return result("alpha-finding-42")
        if "TASK: step-beta" in user_msg:
            assert "alpha-finding-42" in user_msg
            return result("beta-synthesized-alpha")
        return result("done")

    manager, _log = make_manager(tmp_path, chat, max_workers=2)
    try:
        workers = manager.spawn_many([
            {"name": "alpha", "task": "step-alpha"},
            {"name": "beta", "task": "step-beta", "depends_on": ["alpha"]},
        ])
        states = manager.wait([w.id for w in workers], timeout=5)
        assert states == {workers[0].id: "done", workers[1].id: "done"}
        assert workers[0].wave == 0
        assert workers[1].wave == 1
        assert workers[1].summary == "beta-synthesized-alpha"
        synth = manager.synthesize_batch(workers[0].batch_id)
        assert synth["waves"] == 2
        assert synth["all_settled"] is True
    finally:
        manager.shutdown(wait=True)


def test_dag_cycle_detection_rejected_atomically(tmp_path):
    manager, log = make_manager(tmp_path, lambda *_a, **_k: result("ok"))
    try:
        with pytest.raises(SubagentError, match="cycle"):
            manager.spawn_many([
                {"name": "a", "task": "first", "depends_on": ["b"]},
                {"name": "b", "task": "second", "depends_on": ["a"]},
            ])
        assert manager.list() == []
        assert not log.events
    finally:
        manager.shutdown(wait=True)


def test_upstream_dependency_failure_blocks_downstream_worker(tmp_path):
    def chat(_prov, _mod, _eff, messages, *_a, **_kw):
        if "TASK: fail-upstream" in messages[-1]["content"]:
            raise RuntimeError("upstream boom")
        return result("should not run")

    manager, _log = make_manager(tmp_path, chat, max_workers=2)
    try:
        workers = manager.spawn_many([
            {"name": "up", "task": "fail-upstream"},
            {"name": "down", "task": "needs-upstream", "depends_on": ["up"]},
        ])
        states = manager.wait([w.id for w in workers], timeout=5)
        assert states[workers[0].id] == "error"
        assert states[workers[1].id] == "blocked"
        assert "Blocked because upstream dependency" in workers[1].summary
    finally:
        manager.shutdown(wait=True)


def test_intra_step_parallel_read_tools_speedup_and_ordering(tmp_path):
    def slow_read(path):
        time.sleep(0.15)
        return f"contents-of-{path}"

    registry = {
        "read_file": Tool("read_file", "read", {"type": "object"}, slow_read),
    }

    def chat(_prov, _mod, _eff, messages, *_a, **_kw):
        if not any(m.get("role") == "tool" for m in messages):
            return result(tool_calls=[
                call("read_file", {"path": "a.py"}, "c1"),
                call("read_file", {"path": "b.py"}, "c2"),
                call("read_file", {"path": "c.py"}, "c3"),
            ])
        tool_msgs = [m["content"] for m in messages if m.get("role") == "tool"]
        return result(" | ".join(tool_msgs))

    manager, _log = make_manager(tmp_path, chat, tools=registry)
    try:
        t0 = time.monotonic()
        worker = manager.spawn("read three files in one step", role="researcher")
        assert manager.wait([worker.id], timeout=5)[worker.id] == "done"
        elapsed = time.monotonic() - t0
        assert elapsed < 0.40, f"expected parallel read speedup, took {elapsed:.2f}s"
        assert worker.summary == "contents-of-a.py | contents-of-b.py | contents-of-c.py"
        assert worker.tool_breakdown == {"read_file": 3}
    finally:
        manager.shutdown(wait=True)


def test_per_worker_transient_retry_with_backoff(tmp_path):
    attempts = {"n": 0}

    def flaky_chat(*_a, **_kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient gateway 502")
        return result("recovered on attempt 3")

    manager, _log = make_manager(tmp_path, flaky_chat)
    try:
        worker = manager.spawn("flaky task", max_retries=2)
        assert manager.wait([worker.id], timeout=5)[worker.id] == "done"
        assert worker.retries == 2
        assert worker.summary == "recovered on attempt 3"
    finally:
        manager.shutdown(wait=True)



def test_subagent_write_captures_live_diff_and_updates_blackboard(tmp_path):
    target = tmp_path / "feature.py"
    target.write_text("x = 1\n", encoding="utf-8")
    registry = {
        "write_file": Tool(
            "write_file",
            "write",
            {"type": "object"},
            lambda path, content: Path(path).write_text(content, encoding="utf-8") and "OK",
        ),
    }

    def chat(_prov, _mod, _eff, messages, *_a, **_kw):
        if not any(m.get("role") == "tool" for m in messages):
            return result(
                tool_calls=[
                    call("write_file", {"path": str(target), "content": "x = 2\ny = 3\n"})
                ]
            )
        return result("updated feature.py with x=2, y=3")

    manager, _log = make_manager(tmp_path, chat, tools=registry)
    try:
        worker = manager.spawn(
            "update feature.py",
            role="coder",
            name="coder-alpha",
            write_scope=[str(target)],
            allow_writes=True,
        )
        assert manager.wait([worker.id], timeout=5)[worker.id] == "done"
        assert worker.lines_added == 2
        assert worker.lines_removed == 1
        assert len(worker.diffs) == 1
        assert "+x = 2" in worker.diffs[0]["diff"]
        assert "-x = 1" in worker.diffs[0]["diff"]
        # Blackboard automatically received worker summary
        bb_items = manager.blackboard.read("summary:coder-alpha")
        assert len(bb_items) == 1
        assert "updated feature.py" in bb_items[0]["value"]
        dag_str = manager.format_dag(worker.batch_id)
        assert "Wave 0" in dag_str and "coder-alpha" in dag_str
    finally:
        manager.shutdown(wait=True)


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
