"""Validated command inputs for Hermes-child operations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..core.errors import OmhError
from ..system.metadata_safety import require_opaque_metadata_ref


def validate_metadata_args(args: argparse.Namespace) -> None:
    for field in ("model", "provider", "reasoning", "parent_run_id", "run_id"):
        try:
            require_opaque_metadata_ref(getattr(args, field), field=field)
        except ValueError as exc:
            raise OmhError(str(exc)) from exc
    validate_run_id(args.run_id)


def validate_run_id(run_id: str) -> None:
    try:
        safe_run_id = require_opaque_metadata_ref(run_id, field="run_id")
    except ValueError as exc:
        raise OmhError(str(exc)) from exc
    if safe_run_id in {".", ".."} or "/" in safe_run_id or "\\" in safe_run_id:
        raise OmhError("run_id must be a single safe opaque metadata reference")


def read_prompt(source: str) -> str:
    try:
        prompt = sys.stdin.read() if source == "-" else Path(source).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        raise OmhError(f"could not read prompt file: {exc}") from exc
    if not prompt.strip():
        raise OmhError("a non-empty prompt is required via stdin or --prompt-file")
    return prompt
