"""Confirmed Hermes skill-load probe and authenticated status commands."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import TypeVar

from ..coding.hermes_child_receipts import (
    ReceiptVerificationError,
    canonical_observation,
    hermes_child_run_dir,
    load_or_create_observation_key,
)
from ..coding.skill_load_observation import (
    SkillLoadProbeRequest,
    probe_skill_load,
    skill_load_observation_is_fresh,
    validate_skill_load_observation,
)
from ..core.errors import OmhError
from ..system.approval_tier import TIER_AUTO_ALLOWED, resolve_approval_tier
from ..system.local_store import atomic_write_json, read_json_object
from ..system.security_posture import resolve_security_posture
from .common import _paths, _print_json, _wants_json

_AUDIENCE = "agent/maintainer"
_ObservationValue = TypeVar("_ObservationValue")


def cmd_hermes_child_skill_load_probe(args: argparse.Namespace) -> int:
    decision = resolve_approval_tier(
        "hermes_child_dispatch", confirmed=bool(args.confirm_dispatch), posture=resolve_security_posture()
    )
    if decision.tier != TIER_AUTO_ALLOWED:
        raise OmhError("Hermes skill-load probe requires --confirm-dispatch")
    run_dir = _run_dir(args)
    run_dir.mkdir(mode=0o700, exist_ok=True)
    reservation_path = run_dir / "skill-load-probe.reserved"
    try:
        reservation = os.open(
            reservation_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise OmhError(f"Hermes skill-load probe run already exists: {args.run_id}") from exc
    os.close(reservation)
    try:
        observation = probe_skill_load(
            SkillLoadProbeRequest(
                expected_skills=tuple(args.expected_skill),
                hermes=args.hermes,
                timeout_seconds=args.timeout,
                termination_grace_seconds=args.termination_grace,
            ),
            confirmed=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise OmhError(str(exc)) from exc
    _write_observation(args, observation)
    _emit(args, observation)
    return 1 if observation["probe_status"] == "probe_error" else 0


def cmd_hermes_child_skill_load_status(args: argparse.Namespace) -> int:
    run_dir = _run_dir(args)
    try:
        observation = read_json_object(run_dir / "skill-load-observation.json")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise OmhError(f"Hermes skill-load observation is unreadable: {exc}") from exc
    if observation is None:
        raise OmhError(f"Hermes skill-load probe not found: {args.run_id}")
    if validate_skill_load_observation(observation) or not _signature_valid(
        run_dir,
        args.run_id,
        observation,
    ):
        raise OmhError("Hermes skill-load observation is invalid")
    if not skill_load_observation_is_fresh(observation):
        raise OmhError("Hermes skill-load observation is expired")
    _emit(args, observation)
    return 0


def _run_dir(args: argparse.Namespace) -> Path:
    try:
        return hermes_child_run_dir(_paths(args).omh_home, args.run_id, create_root=True)
    except ReceiptVerificationError as exc:
        raise OmhError(str(exc)) from exc


def _write_observation(
    args: argparse.Namespace,
    observation: dict[str, _ObservationValue],
) -> None:
    run_dir = _run_dir(args)
    atomic_write_json(run_dir / "skill-load-observation.json", observation, private=True)
    signature = hmac.new(
        load_or_create_observation_key(run_dir.parent),
        _signature_message(args.run_id, observation),
        hashlib.sha256,
    ).hexdigest()
    atomic_write_json(
        run_dir / "skill-load-observation.signature.json",
        {
            "schema_version": "skill_load_observation_signature/v2",
            "run_id": args.run_id,
            "hmac_sha256": signature,
        },
        private=True,
    )


def _signature_valid(
    run_dir: Path,
    run_id: str,
    observation: dict[str, _ObservationValue],
) -> bool:
    try:
        signature = read_json_object(run_dir / "skill-load-observation.signature.json")
        key = load_or_create_observation_key(run_dir.parent)
    except (OSError, json.JSONDecodeError, ValueError, ReceiptVerificationError):
        return False
    if signature is None or set(signature) != {
        "schema_version",
        "run_id",
        "hmac_sha256",
    }:
        return False
    if (
        signature.get("schema_version") != "skill_load_observation_signature/v2"
        or signature.get("run_id") != run_id
    ):
        return False
    observed = signature.get("hmac_sha256")
    if not isinstance(observed, str):
        return False
    expected = hmac.new(
        key,
        _signature_message(run_id, observation),
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(observed, expected)


def _signature_message(
    run_id: str,
    observation: dict[str, _ObservationValue],
) -> bytes:
    return (
        b"omh-skill-load-observation-signature/v2\0"
        + run_id.encode("utf-8")
        + b"\0"
        + canonical_observation(observation)
    )


def _emit(args: argparse.Namespace, observation: dict[str, _ObservationValue]) -> None:
    if _wants_json(args):
        _print_json(observation)
        return
    print(f"AUDIENCE {_AUDIENCE}")
    print(f"PROBE {observation['probe_status']}")
    print(f"REASON {observation['reason_code']}")
    if "load_state" in observation:
        print(f"LOAD {observation['load_state']}")
