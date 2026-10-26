"""Bounded runtime CLI boundary for implemented workflow artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Final

from ..installer import OmhError
from ..workflows.workflow_artifact_operations import (
    WORKFLOW_ARTIFACT_OPERATIONS,
    WorkflowArtifactOperationError,
    run_workflow_artifact_operation,
)
from .common import _paths, _print_json


_MAX_INPUT_BYTES: Final = 262_144


def cmd_runtime_workflow_artifact(args: argparse.Namespace) -> int:
    """Run one closed, metadata-only workflow artifact operation."""
    try:
        payload = _input_object(args.input)
        result = run_workflow_artifact_operation(_paths(args), args.workflow, args.operation, payload)
    except (OSError, json.JSONDecodeError, WorkflowArtifactOperationError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json(result)
    return 0


def _input_object(source: str) -> dict[str, Any]:
    raw = _read_input(source)
    value = json.loads(raw)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkflowArtifactOperationError("workflow artifact input must be a JSON object")
    return value


def _read_input(source: str) -> str:
    try:
        if source == "-":
            try:
                raw = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
            except AttributeError:
                raw = sys.stdin.read(_MAX_INPUT_BYTES + 1).encode("utf-8")
        else:
            with Path(source).expanduser().open("rb") as input_file:
                raw = input_file.read(_MAX_INPUT_BYTES + 1)
        if len(raw) > _MAX_INPUT_BYTES:
            raise WorkflowArtifactOperationError("workflow artifact input exceeds 262144 bytes")
        return raw.decode("utf-8")
    except UnicodeError as exc:
        raise WorkflowArtifactOperationError("workflow artifact input must be valid UTF-8") from exc


def add_runtime_workflow_artifact_commands(
    runtime_sub: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the agent/operator workflow-artifact command group."""
    artifact = runtime_sub.add_parser(
        "workflow-artifact",
        help="Agent/operator surface for bounded metadata-only workflow artifacts; it never executes or schedules work.",
    )
    workflows = artifact.add_subparsers(dest="workflow", required=True)
    for workflow, operations in WORKFLOW_ARTIFACT_OPERATIONS.items():
        workflow_parser = workflows.add_parser(workflow)
        operation_sub = workflow_parser.add_subparsers(dest="operation", required=True)
        for operation in operations:
            parser = operation_sub.add_parser(operation)
            parser.add_argument("--input", required=True, help="JSON object file, or '-' for bounded stdin.")
            parser.set_defaults(func=cmd_runtime_workflow_artifact)
