"""Regression tests for the recent hardening fixes.

Kept deterministic and offline: temp files only, no network, no Playwright.
"""

import sys as _sys
from pathlib import Path as _Path
_ROOT_DIR = str(_Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in _sys.path:
    _sys.path.insert(0, _ROOT_DIR)


import ast
import json
import sqlite3
import sys
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


from fullagent import dbtools, diffpreview, emailops, hooks, mcp, sandbox
from fullagent import cronsched, editfallback, jsonout, notify, progress
from fullagent import skills
from fullagent import taint, tools
from fullagent.snapshots import SnapshotStore
from fullagent.browserauto import _BrowserSession
from fullagent.cancelguard import run_cancellable
from fullagent.debate import DebateTournament
from fullagent.kernel import EventLog, replay
from fullagent.report import format_forecast
from fullagent.tools import RISK_CONFIRM
from fullagent.cancelguard import TurnCancelled
from fullagent.workflows import Workflow, WorkflowEngine, WorkflowStep


def test_eventlog_deep_rewind_survives_close_and_reload(tmp_path):
    path = tmp_path / "deep-rewind.jsonl"
    log = EventLog(path, max_events=5)
    for i in range(40):
        log.append("user.message", {"text": "msg%d" % i})

    marker_seq = log.rewind(2)
    assert marker_seq == 40
    assert [e.seq for e in replay(log)] == [2, 40]
    log.close()

    reloaded = EventLog(path, max_events=5)
    try:
        events = replay(reloaded)
        assert [e.seq for e in events] == [2, 40]
        assert events[0].data["text"] == "msg2"
        assert events[1].type == "kernel.rewind"
        assert events[1].data["to"] == 2
        ok, detail = reloaded.verify()
        assert ok, detail
    finally:
        reloaded.close()


def test_diffpreview_non_tty_without_ui_fails_closed(monkeypatch, tmp_path,
                                                     capsys):
    class NonTtyStdin:
        def isatty(self):
            return False

        def read(self, *args):
            return ""

    monkeypatch.setattr(sys, "stdin", NonTtyStdin())
    target = tmp_path / "notes.txt"
    assert diffpreview._ask_yes_no("Apply?", None) is False
    assert diffpreview.preview_and_confirm(
        "write_file", {"path": str(target), "content": "hello\n"}, None
    ) is False


def test_browser_open_url_rejects_non_http_scheme_before_launch(monkeypatch):
    session = _BrowserSession()
    touched = []

    def fail_if_launched():
        touched.append(True)
        raise AssertionError("Playwright must not be touched")

    monkeypatch.setattr(session, "_ensure", fail_if_launched)
    for url in ("file:///etc/passwd", "javascript:alert(1)", "ftp://host/x"):
        result = session.open_url(url)
        assert result.startswith("ERROR:")
        assert "http://" in result and "https://" in result
    assert touched == []


def test_emailops_rejects_crlf_recipient_and_subject():
    assert emailops._valid_email("user@example.com\r\nBcc: evil@example.com") \
        is False
    assert emailops._valid_email("user@example.com\n") is False

    cfg = {
        "host": "127.0.0.1",
        "port": 1,
        "from_addr": "sender@example.com",
        "use_tls": False,
    }
    subject_result = emailops.send_email(
        "victim@example.com", "hi\r\nBcc: evil@example.com", "body", config=cfg)
    assert subject_result.startswith("ERROR:")
    assert "newline" in subject_result

    recipient_result = emailops.send_email(
        ["victim@example.com\r\nBcc: evil@example.com"], "hi", "body",
        config=cfg)
    assert recipient_result.startswith("ERROR:")
    assert "no valid recipient" in recipient_result


def test_dbtools_handles_uri_metacharacters_and_stays_read_only(tmp_path):
    db_path = tmp_path / "odd # 100% name.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany("INSERT INTO users(name) VALUES (?)",
                         [("alice",), ("bob",)])
        conn.commit()
    finally:
        conn.close()

    out = dbtools.db_tables(str(db_path))
    assert "users" in out and "2" in out
    assert not out.startswith("ERROR:")

    refused = dbtools.db_query(
        str(db_path), "INSERT INTO users(name) VALUES ('mallory')")
    assert refused.startswith("ERROR:")
    assert "confirm=True" in refused

    readonly = dbtools._connect_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            readonly.execute("INSERT INTO users(name) VALUES ('mallory')")
    finally:
        readonly.close()


def test_sandbox_parses_string_network_and_rejects_invalid_timeout(
        monkeypatch):
    real_run = sandbox.run_sandboxed
    captured = {}

    def fake_run(command, network=False, timeout=30.0, **kwargs):
        captured.update(command=command, network=network, timeout=timeout)
        return {
            "sandbox": "rlimit", "command": command, "network": network,
            "returncode": 0, "stdout": "", "stderr": "", "timed_out": False,
            "elapsed": 0.0, "error": None,
        }

    monkeypatch.setattr(sandbox, "run_sandboxed", fake_run)
    sandbox._handle_sandboxed_run("echo hi", network="false")
    assert captured["network"] is False
    sandbox._handle_sandboxed_run("echo hi", network="TRUE")
    assert captured["network"] is True

    invalid = real_run("echo hi", timeout="not-a-number")
    assert isinstance(invalid, dict)
    assert invalid["error"].startswith("invalid timeout")


def test_report_format_forecast_tolerates_missing_cost_per_turn():
    forecast = {
        "turns": 3,
        "tokens_in": 100,
        "tokens_out": 50,
        "tokens_in_per_turn": 33,
        "tokens_out_per_turn": 17,
        "cost_per_turn_usd": None,
        "goal_distance": 0.5,
        "est_ticks_remaining": None,
    }
    out = format_forecast(forecast)
    assert "per turn" in out
    assert "$" not in out
    assert "None" not in out


def test_cancelguard_propagates_base_exception_without_hanging():
    class FatalExit(BaseException):
        pass

    def boom():
        raise FatalExit("worker died")

    seen = {}

    def caller():
        try:
            run_cancellable(boom, lambda: False, poll=0.05)
        except BaseException as exc:  # noqa: BLE001 - captured for assertion
            seen["exc"] = exc

    worker = threading.Thread(target=caller, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "run_cancellable hung on BaseException"
    assert isinstance(seen.get("exc"), FatalExit)


def test_debate_refute_accepts_model_id_list_clusters(tmp_path):
    log = EventLog(tmp_path / "debate.jsonl")
    try:
        tournament = DebateTournament(
            log, lambda model, prompt: "", ["modelA", "modelB", "modelC"])
        tournament._clusters = [["modelB", "modelC"], ["modelA"]]
        tournament._champion_cluster = {"modelA"}

        trust = tournament.refute("modelB")

        assert trust["modelB"] == trust["modelC"]
        assert trust["modelB"] > 0.5
        assert trust["modelA"] < 0.5
    finally:
        log.close()


def test_hooks_non_string_match_skips_only_that_entry():
    cfg = {
        "PreToolUse": [
            {"match": None, "command": "echo broken"},
            {"match": 42, "command": "echo broken"},
            {"match": "(unclosed", "command": "echo broken"},
            {"match": "run_command", "command": "echo ok"},
        ]
    }
    matched = hooks._iter_matching(cfg, "PreToolUse", "run_command")
    assert [m["command"] for m in matched] == ["echo ok"]


def test_skills_reject_pathlib_constructor_mutation():
    tree = ast.parse(
        "from pathlib import Path\n"
        "def evil(path):\n"
        "    Path(path).write_text('pwned')\n"
        "    return Path(path)\n"
    )
    problem = skills._validate_safety(tree)
    assert problem is not None
    assert "forbidden attribute" in problem


def test_taint_flags_bare_environment_subscript_source():
    findings = taint.TaintAnalyzer().analyze(
        'x = os.environ["SECRET"]\neval(x)\n')
    assert findings, "environment read must taint eval()"
    assert findings[0].sink == "eval"
    assert "os.environ" in findings[0].source


def test_mcp_tools_require_confirmation():
    tool = mcp._make_tool(None, "srv", {"name": "do", "inputSchema": {}})
    assert tool.risk == RISK_CONFIRM
    assert tool.handler() == "ERROR: MCP server 'srv' unavailable"


def test_eventlog_append_does_not_glue_onto_torn_line(tmp_path):
    path = tmp_path / "torn.jsonl"
    log = EventLog(path, max_events=10)
    log.append("user.message", {"text": "before"})
    log.close()
    with path.open("ab") as fh:
        fh.write(b'{"seq": 999, "torn')  # crash mid-write, no newline

    log = EventLog(path, max_events=10)
    try:
        last = log.append("user.message", {"text": "after-crash"})
    finally:
        log.close()

    reloaded = EventLog(path, max_events=10)
    try:
        events = replay(reloaded)
        assert events[-1].seq == last.seq
        assert events[-1].data["text"] == "after-crash"
        assert [e.data["text"] for e in events[:-1]] == ["before"]
    finally:
        reloaded.close()


def test_snapshot_gc_keeps_archived_references(tmp_path):
    store = SnapshotStore(tmp_path / "snap")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("archived snapshot content\n", encoding="utf-8")
    taken = store.take([str(tracked)])
    blob = taken["paths"][str(tracked.resolve())]

    log = EventLog(tmp_path / "snap.jsonl", max_events=1)
    try:
        log.append("snapshot.taken", taken)
        # push the snapshot event out of the in-RAM window
        log.append("user.message", {"text": "after"})
        assert store.reachable_hashes(log) >= {taken["tree"], blob}
        doomed = store.put_blob(b"unreferenced")
        store.gc(log)
    finally:
        log.close()

    assert store.get_blob(blob) == b"archived snapshot content\n"
    assert store.get_blob(taken["tree"]) is not None
    assert store.get_blob(doomed) is None


def test_progress_rejects_infinity_without_raising():
    tracker = progress.ProgressTracker()
    pid = tracker.start("work", float("inf"))
    assert tracker.get(pid)["total"] == 0
    tracker.increment(pid, float("inf"))
    tracker.update(pid, float("-inf"))
    assert tracker.get(pid)["current"] == 0


def test_jsonout_coerces_non_serializable_turn_fields():
    turn = SimpleNamespace(
        session_id=object(), model_id=object(), user_text="hi",
        assistant_text="hello", tools=[], usage=None,
        error=object(), duration=0.0, cost_usd=None,
    )
    out = jsonout.turn_to_json(turn, None)
    json.dumps(out)  # must never raise
    assert isinstance(out["session_id"], str)
    assert isinstance(out["error"], str)


def test_notify_never_raises_on_unserializable_payload():
    assert notify.notify_event(
        "test", {"bad": object()}, url="http://127.0.0.1:1",
        timeout=1) is False


def test_web_fetch_blocks_private_and_metadata_urls():
    for url in (
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/admin",
        "http://metadata.google.internal/computeMetadata/v1/",
    ):
        out = tools.web_fetch(url)
        assert out.startswith("ERROR:"), out
        assert "blocked" in out.lower(), out


def test_editfallback_preserves_file_mode(tmp_path):
    target = tmp_path / "secret.py"
    target.write_text("value = 1\n", encoding="utf-8")
    target.chmod(0o600)
    result = editfallback.apply_edit_with_fallbacks(
        str(target), "value = 1", "value = 2")
    assert result.get("ok"), result
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert target.stat().st_mode & 0o777 == 0o600


def test_cron_timeout_kills_grandchildren(tmp_path, monkeypatch):
    monkeypatch.setattr(cronsched, "RUN_TIMEOUT", 0.3)
    marker = tmp_path / "grandchild-survived"
    sched = cronsched.CronScheduler(tmp_path / "cron.db")
    try:
        sched.stop()  # drive run_due() deterministically
        cron_id, _ = sched.add(
            "* * * * *", f"(sleep 1; echo alive > {marker}) & wait")
        with sched._lock:
            sched._conn.execute(
                "UPDATE crons SET next_run = ? WHERE id = ?",
                (time.time() - 1, cron_id))
            sched._conn.commit()
        sched.run_due()
        time.sleep(1.2)
        assert not marker.exists(), "grandchild survived cron timeout"
        assert sched.runs(cron_id)[0]["exit_code"] == 124
    finally:
        sched.close()


def test_workflow_cancel_seals_abandoned_state(tmp_path):
    log = EventLog(tmp_path / "wf.jsonl")
    try:
        def cancel(step):
            raise TurnCancelled("user pressed esc")

        engine = WorkflowEngine(log, tmp_path / "defs", executor=cancel)
        engine.save(Workflow(name="demo",
                             steps=[WorkflowStep(task="one")]))
        with pytest.raises(TurnCancelled):
            engine.run("demo")
        done = [e for e in replay(log) if e.type == "workflow.done"]
        assert done and done[-1].data["state"] == "ABANDONED"
    finally:
        log.close()


def test_bgsh_output_buffer_is_capped_and_offsets_stay_correct(monkeypatch):
    from fullagent import bgsh

    monkeypatch.setattr(bgsh, "MAX_OUTPUT_CHARS", 10)
    task = bgsh._BgTask("t1", "cap", 30.0)
    for chunk in ("aaaa", "bbbb", "cccc"):
        with task._lock:
            bgsh._append_capped(task, "out", chunk)
    assert "".join(task._stdout) == "bbbbcccc"
    assert task._stdout_dropped == 4
    got, _ = bgsh._read_new(task)
    assert got == "bbbbcccc"
    with task._lock:
        bgsh._append_capped(task, "out", "dddd")
    got, _ = bgsh._read_new(task)
    assert got == "dddd"
    with bgsh._TASKS_LOCK:
        bgsh._TASKS["t1"] = task
    try:
        snap = bgsh.poll("t1")
    finally:
        with bgsh._TASKS_LOCK:
            bgsh._TASKS.pop("t1", None)
    assert "truncated" in snap["output_so_far"]


def test_bgsh_finished_task_registry_is_bounded(monkeypatch):
    from fullagent import bgsh

    monkeypatch.setattr(bgsh, "MAX_FINISHED_TASKS", 2)
    with bgsh._TASKS_LOCK:
        bgsh._TASKS.clear()
        for i in range(6):
            t = bgsh._BgTask(f"done{i}", "d", 30.0)
            t.started_at = float(i)
            t.status = bgsh.STATUS_DONE
            bgsh._TASKS[t.id] = t
        running = bgsh._BgTask("run", "r", 30.0)
        running.started_at = 0.5
        bgsh._TASKS[running.id] = running
        bgsh._evict_finished_locked()
        ids = set(bgsh._TASKS)
        bgsh._TASKS.clear()
    assert "run" in ids
    assert {"done4", "done5"} <= ids
    assert "done0" not in ids and "done1" not in ids


def test_debate_malformed_trust_does_not_crash(tmp_path):
    log = EventLog(tmp_path / "trust.jsonl")
    try:
        log.append("debate.calibration", {"model": "a", "trust": "bogus"})
        log.append("debate.calibration", {"model": "b", "trust": None})
        log.append("debate.calibration", {"model": "c", "trust": float("nan")})
        log.append("debate.calibration", {"model": "d", "trust": 2.5})
        tour = DebateTournament(log, lambda m, p: "", ["a", "b", "c", "d"])
        assert tour.calibration("a") == 0.5
        assert tour.calibration("b") == 0.5
        assert tour.calibration("c") == 0.5
        assert tour.calibration("d") == 0.95
    finally:
        log.close()


def test_watch_debounce_map_is_pruned():
    from fullagent.watch import Watch, WatchManager

    w = Watch(id="w", path="/tmp", pattern="*", action="true",
              added_at=0.0)
    now = 1000.0
    for i in range(1100):
        w._fired[f"w|modified|/tmp/f{i}"] = now
    WatchManager._prune_fired(w, now, "w|modified|/tmp/f1099")
    assert len(w._fired) <= 1024
    assert "w|modified|/tmp/f1099" in w._fired


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
