from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from omh.installer import OmhError
from omh.workflows.browser_workflow_learning import BrowserTraceError, JsonObject, MAX_TRACE_BYTES
from omh.workflows.browser_workflow_learning_store import approve_browser_workflow_trace, browser_workflow_trace_status, read_browser_workflow_trace, replay_stored_browser_workflow_trace, write_browser_workflow_trace

from .common import _print_json


def cmd_browser_trace_record(args: argparse.Namespace) -> int:
    return _run(lambda: write_browser_workflow_trace(_read_json(args.input), args.project_root))


def cmd_browser_trace_inspect(args: argparse.Namespace) -> int:
    return _run(lambda: read_browser_workflow_trace(args.project_root, args.trace_id))


def cmd_browser_trace_status(args: argparse.Namespace) -> int:
    return _run(lambda: browser_workflow_trace_status(args.project_root, args.trace_id))


def cmd_browser_trace_approve(args: argparse.Namespace) -> int:
    return _run(lambda: approve_browser_workflow_trace(args.project_root, args.trace_id, args.digest))


def cmd_browser_trace_replay(args: argparse.Namespace) -> int:
    return _run(lambda: replay_stored_browser_workflow_trace(args.project_root, args.trace_id, _read_json(args.observation)))


def _run(action: Callable[[], JsonObject]) -> int:
    try:
        _print_json(action())
    except BrowserTraceError as exc:
        raise OmhError(str(exc)) from exc
    return 0


def add_browser_trace_commands(parent: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    trace = parent.add_parser("trace", help="Record and inspect redacted offline browser workflow traces.")
    commands = trace.add_subparsers(dest="browser_trace_command", required=True)
    record = commands.add_parser("record", help="Parse and store a selected terminal-success trace.")
    record.add_argument("--input", required=True)
    _project_root(record)
    record.set_defaults(func=cmd_browser_trace_record)
    inspect = commands.add_parser("inspect", help="Inspect a stored browser trace.")
    inspect.add_argument("--trace-id", required=True)
    _project_root(inspect)
    inspect.set_defaults(func=cmd_browser_trace_inspect)
    status = commands.add_parser("status", help="Show trace lifecycle status.")
    status.add_argument("--trace-id", required=True)
    _project_root(status)
    status.set_defaults(func=cmd_browser_trace_status)
    approve = commands.add_parser("approve", help="Approve exactly one trace digest.")
    approve.add_argument("--trace-id", required=True)
    approve.add_argument("--digest", required=True)
    _project_root(approve)
    approve.set_defaults(func=cmd_browser_trace_approve)
    replay = commands.add_parser("replay", help="Replay one bounded fixture observation offline.")
    replay.add_argument("--trace-id", required=True)
    replay.add_argument("--observation", required=True)
    _project_root(replay)
    replay.set_defaults(func=cmd_browser_trace_replay)


def _project_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default=".", help="Project Git root or a path below it.")


def _read_json(path: str) -> dict[str, object]:
    try:
        candidate = Path(path)
        if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > MAX_TRACE_BYTES:
            raise BrowserTraceError("JSON input exceeds bounds or is unsafe")
        raw = candidate.read_bytes()
        if len(raw) > MAX_TRACE_BYTES:
            raise BrowserTraceError("JSON input exceeds bounds")
        value = json.loads(raw.decode("utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except BrowserTraceError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise BrowserTraceError("JSON input could not be read") from exc
    if not isinstance(value, dict):
        raise BrowserTraceError("JSON input must be an object")
    return value
