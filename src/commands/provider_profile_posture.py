from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..installer import OmhError
from ..workflows.provider_profile_posture import (
    build_provider_profile_posture,
    parse_provider_profile_posture_input,
    write_provider_profile_posture,
)
from .common import _paths, _print_json


def cmd_ops_provider_profile_posture(args: argparse.Namespace) -> int:
    try:
        raw = json.loads(Path(args.input).expanduser().read_text(encoding="utf-8"))
        posture = build_provider_profile_posture(parse_provider_profile_posture_input(raw))
        artifact = write_provider_profile_posture(_paths(args), posture) if args.write else {"written": False}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _print_json({**posture, "artifact": artifact})
    return 0


def add_ops_provider_profile_posture_command(ops_sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    posture = ops_sub.add_parser(
        "provider-profile-posture",
        help="Prepare provider/profile readiness metadata without reading secrets or calling providers.",
    )
    posture.add_argument("--input", required=True, help="Path to a provider_profile_posture_input/v1 JSON file.")
    posture.add_argument("--write", action="store_true", help="Write the deterministic posture under the OMH operations store.")
    posture.set_defaults(func=cmd_ops_provider_profile_posture)
