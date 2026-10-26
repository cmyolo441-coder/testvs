"""Attachable CLI commands for managed host-owned web-QA observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Callable

from omh.installer import OmhError
from omh.workflows.browser_workflow_learning_store import resolved_browser_workflow_trace_reference
from omh.workflows.web_qa_comparison import compare_web_qa_observations
from omh.workflows.web_qa_motion_capture_store import (
    MotionCaptureImportError,
    import_motion_capture,
    motion_capture_gate,
    read_motion_capture,
)
from omh.workflows.web_qa_observation_store import (
    WebQaObservationStoreError,
    import_web_qa_observation,
    observation_envelope,
    prepare_web_qa_observation,
    read_web_qa_observation,
)

from .common import _print_json


_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_RUN_ID = re.compile(r"^web-qa-[a-f0-9]{24}$")
_CAPTURE_ID = re.compile(r"^motion-[a-f0-9]{24}$")
_CELL_ID = re.compile(r"^cell-[a-f0-9]{24}$")
MAX_CLI_JSON_BYTES = 262_144


def cmd_web_qa_observation_plan(args: argparse.Namespace) -> int:
    try:
        result = prepare_web_qa_observation(
            _read_json_input(args.plan_json),
            args.project_root,
            trusted_trace_resolver=_trace_resolver(args.project_root),
        )
    except (OSError, ValueError, WebQaObservationStoreError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_web_qa_observation_import(args: argparse.Namespace) -> int:
    try:
        result = import_web_qa_observation(
            args.project_root,
            _read_json_input(args.plan_json),
            _read_json_input(args.receipt_json),
            _capture_arguments(args.capture),
            trusted_trace_resolver=_trace_resolver(args.project_root),
        )
    except (OSError, ValueError, WebQaObservationStoreError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_web_qa_observation_show(args: argparse.Namespace) -> int:
    try:
        result = read_web_qa_observation(args.project_root, _valid_run_id(args.run_id), trusted_trace_resolver=_trace_resolver(args.project_root))
    except (OSError, ValueError, WebQaObservationStoreError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_web_qa_observation_compare(args: argparse.Namespace) -> int:
    try:
        deployment_resolver = _deployment_resolver(args.deployment_observation_json)
        result = compare_web_qa_observations(
            observation_envelope(
                args.project_root,
                _valid_run_id(args.baseline_run_id),
                trusted_trace_resolver=_trace_resolver(args.project_root),
            ),
            observation_envelope(
                args.project_root,
                _valid_run_id(args.candidate_run_id),
                trusted_trace_resolver=_trace_resolver(args.project_root),
            ),
            trusted_deployment_resolver=deployment_resolver,
            trusted_trace_resolver=_trace_resolver(args.project_root),
        )
    except (OSError, ValueError, WebQaObservationStoreError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_web_qa_motion_import(args: argparse.Namespace) -> int:
    try:
        result = import_motion_capture(args.project_root, _read_json_input(args.plan_json), _read_json_input(args.receipt_json))
    except (OSError, ValueError, MotionCaptureImportError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_web_qa_motion_show(args: argparse.Namespace) -> int:
    try:
        result = read_motion_capture(args.project_root, _valid_run_id(args.run_id), _valid_capture_id(args.capture_id))
    except (OSError, ValueError, MotionCaptureImportError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def cmd_web_qa_motion_gate(args: argparse.Namespace) -> int:
    try:
        result = motion_capture_gate(
            args.project_root,
            _valid_run_id(args.run_id),
            [_valid_cell_id(value) for value in args.motion_cell],
            trusted_trace_resolver=_trace_resolver(args.project_root),
        )
    except (OSError, ValueError, MotionCaptureImportError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def add_web_qa_observation_commands(parent_subparsers) -> None:
    """Add ``observation plan/import/show/compare`` below an existing web-QA parser."""
    observation = parent_subparsers.add_parser(
        "observation",
        help="Prepare, import, inspect, and compare managed host-owned web-QA observations.",
    )
    sub = observation.add_subparsers(dest="web_qa_observation_command", required=True)

    plan = sub.add_parser("plan", help="Normalize a plan and report whether its run identity is already completed.")
    _project_root(plan)
    plan.add_argument("--plan-json", required=True, metavar="PATH", help="Bounded plan request JSON file, or - for stdin.")
    plan.set_defaults(func=cmd_web_qa_observation_plan)

    import_command = sub.add_parser("import", help="Validate local captures once and atomically persist host-observed evidence.")
    _project_root(import_command)
    import_command.add_argument("--plan-json", required=True, metavar="PATH")
    import_command.add_argument("--receipt-json", required=True, metavar="PATH")
    import_command.add_argument("--capture", action="append", default=[], metavar="SHA256=PATH", help="Observed screenshot digest and local PNG/JPEG/WebP file; repeat once per screenshot digest.")
    import_command.set_defaults(func=cmd_web_qa_observation_import)

    show = sub.add_parser("show", help="Re-admit and project managed evidence for one completed run.")
    _project_root(show)
    show.add_argument("--run-id", required=True, type=_valid_run_id)
    show.set_defaults(func=cmd_web_qa_observation_show)

    compare = sub.add_parser("compare", help="Compare two re-admitted stored envelopes; never deploys or rolls back.")
    _project_root(compare)
    compare.add_argument("--baseline-run-id", required=True, type=_valid_run_id)
    compare.add_argument("--candidate-run-id", required=True, type=_valid_run_id)
    compare.add_argument("--deployment-observation-json", metavar="PATH", default="", help="Explicit host-observed closed deployment record for a canary candidate; never a success flag.")
    compare.set_defaults(func=cmd_web_qa_observation_compare)

    motion = sub.add_parser(
        "motion",
        help="Agent/operator: import and gate metadata-only motion_interaction_capture/v1 records for one planned cell.",
    )
    motion_sub = motion.add_subparsers(dest="web_qa_motion_command", required=True)
    motion_import = motion_sub.add_parser("import", help="Admit one host motion-capture receipt against a plan cell and the checkout revision; stores no media.")
    _project_root(motion_import)
    motion_import.add_argument("--plan-json", required=True, metavar="PATH")
    motion_import.add_argument("--receipt-json", required=True, metavar="PATH", help="host_motion_capture_receipt/v1 JSON file, or - for stdin.")
    motion_import.set_defaults(func=cmd_web_qa_motion_import)
    motion_show = motion_sub.add_parser("show", help="Re-admit one stored motion capture record.")
    _project_root(motion_show)
    motion_show.add_argument("--run-id", required=True, type=_valid_run_id)
    motion_show.add_argument("--capture-id", required=True, type=_valid_capture_id)
    motion_show.set_defaults(func=cmd_web_qa_motion_show)
    motion_gate = motion_sub.add_parser("gate", help="Project stored motion records onto a stored run's verdict for the named cells only; never raises it to PASS.")
    _project_root(motion_gate)
    motion_gate.add_argument("--run-id", required=True, type=_valid_run_id)
    motion_gate.add_argument("--motion-cell", action="append", required=True, type=_valid_cell_id, metavar="CELL_ID", help="Planned matrix cell whose motion evidence is in scope; repeat per cell.")
    motion_gate.set_defaults(func=cmd_web_qa_motion_gate)


def _project_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", required=True, metavar="PATH", help="Existing local Git project root containing managed .omh evidence.")


def _read_json_input(value: str) -> object:
    raw = _bounded_stdin_json() if value == "-" else _bounded_file_json(Path(value).expanduser())
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("JSON input is malformed") from exc


def _bounded_stdin_json() -> bytes:
    raw = sys.stdin.buffer.read(MAX_CLI_JSON_BYTES + 1)
    if len(raw) > MAX_CLI_JSON_BYTES:
        raise ValueError("JSON input exceeds the bounded observation input size")
    return raw


def _bounded_file_json(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("JSON input symlink is refused")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError("JSON input must be a readable local regular file") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CLI_JSON_BYTES:
            raise ValueError("JSON input exceeds bounds or is not a regular file")
        chunks: list[bytes] = []
        remaining = MAX_CLI_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) > MAX_CLI_JSON_BYTES:
        raise ValueError("JSON input exceeds the bounded observation input size")
    return raw


def _capture_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        digest, separator, raw_path = value.partition("=")
        if not separator or not _DIGEST.fullmatch(digest) or not raw_path or digest in result:
            raise ValueError("--capture must be one unique SHA256=PATH value")
        result[digest] = Path(raw_path).expanduser()
    return result


def _trace_resolver(project_root: str) -> Callable[[str, str, str, tuple[str, ...]], bool]:
    def resolve(trace_id: str, digest: str, project_identity: str, origins: tuple[str, ...]) -> bool:
        try:
            reference = resolved_browser_workflow_trace_reference(project_root, trace_id)
        except (OSError, ValueError):
            return False
        return (
            reference.get("digest") == digest
            and reference.get("project_identity") == project_identity
            and tuple(reference.get("origins", [])) == origins
        )
    return resolve


def _deployment_resolver(path: str) -> Callable[[str], object] | None:
    if not path:
        return None
    observed = _read_json_input(path)
    if type(observed) is not dict:
        raise ValueError("deployment observation must be an object")

    def resolve(deployment_ref: str) -> object:
        # The comparison module validates the complete closed host record. This
        # seam only binds the explicitly supplied observation to its named ref.
        return observed if observed.get("deployment_ref") == deployment_ref else {}
    return resolve


def _valid_run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise argparse.ArgumentTypeError("run_id must be a bounded web-QA run identity")
    return value


def _valid_capture_id(value: str) -> str:
    if not _CAPTURE_ID.fullmatch(value):
        raise argparse.ArgumentTypeError("capture_id must be a motion capture identity")
    return value


def _valid_cell_id(value: str) -> str:
    if not _CELL_ID.fullmatch(value):
        raise argparse.ArgumentTypeError("cell_id must be a planned matrix cell identity")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "add_web_qa_observation_commands",
    "cmd_web_qa_observation_compare",
    "cmd_web_qa_observation_import",
    "cmd_web_qa_observation_plan",
    "cmd_web_qa_observation_show",
    "cmd_web_qa_motion_gate",
    "cmd_web_qa_motion_import",
    "cmd_web_qa_motion_show",
]
