"""`omh update-check` -- configure and inspect the startup update-availability check.

The check itself runs from the `omh`/`hermes` launch path in `main.py`; this
module is only the read/write surface over its policy
(`update_check.mode`, `update_check.interval_hours`, persisted in
setup-profile.json) and a read-only view of its last cached outcome. `status`
never makes a network call -- it reports the policy and whatever the launch
path last recorded, exactly like the check itself does within one interval.
"""

from __future__ import annotations

import argparse
import sys

from ..maintenance.update_check import (
    UPDATE_CHECK_MODES,
    accept_update_check_gap,
    read_update_check_cache,
    read_update_check_policy,
    update_check_cache_path,
    write_update_check_policy,
)
from .common import _paths, _print_json, _wants_json


def _status_payload(paths) -> dict[str, object]:
    return {
        "schema_version": "omh_update_check_status/v1",
        "policy": read_update_check_policy(paths),
        "last_check": read_update_check_cache(paths),
        "cache_path": str(update_check_cache_path(paths)),
    }


def _print_status(payload: dict[str, object]) -> None:
    policy = payload.get("policy", {})
    cache = payload.get("last_check", {})
    mode = policy.get("mode") if isinstance(policy, dict) else "off"
    interval = policy.get("interval_hours") if isinstance(policy, dict) else ""
    print(f"Update-check mode: {mode}")
    print(f"Interval: every {interval} hour(s)")
    if isinstance(cache, dict) and cache:
        print(f"Last checked: {cache.get('last_checked_at', 'never')}")
        print(f"Last outcome: {cache.get('outcome', 'unknown')}")
        remote = cache.get("remote_commit")
        if remote:
            print(f"Remote main: {remote}")
        ancestry = cache.get("ancestry")
        if ancestry:
            print(f"Ancestry: {ancestry}")
        gap = cache.get("gap")
        if isinstance(gap, dict) and gap.get("status") in ("open", "accepted"):
            since = gap.get("since") or "unknown"
            print(f"Coverage gap: {gap.get('status')} since {since}")
            if gap.get("status") == "open":
                print("Review with `omh update-check status --json`; accept with `omh update-check accept-gap`.")
    else:
        print("Last checked: never")
    print("Change with `omh update-check set --mode off|notify|auto [--interval-hours N]`.")


def cmd_update_check_status(args: argparse.Namespace) -> int:
    payload = _status_payload(_paths(args))
    if _wants_json(args):
        _print_json(payload)
    else:
        _print_status(payload)
    return 0


def cmd_update_check_set(args: argparse.Namespace) -> int:
    mode = getattr(args, "mode", None)
    interval_hours = getattr(args, "interval_hours", None)
    if mode is None and interval_hours is None:
        print("omh: update-check set needs --mode and/or --interval-hours", file=sys.stderr)
        return 2
    try:
        policy = write_update_check_policy(_paths(args), mode=mode, interval_hours=interval_hours)
    except ValueError as exc:
        print(f"omh: {exc}", file=sys.stderr)
        return 2
    if _wants_json(args):
        _print_json({"schema_version": "omh_update_check_status/v1", "policy": policy})
        return 0
    print(f"Update-check mode: {policy['mode']}")
    print(f"Interval: every {policy['interval_hours']} hour(s)")
    if policy["mode"] == "off":
        print("Off: no network attempts are made at Hermes launch (shipped default).")
    elif policy["mode"] == "notify":
        print("Notify: a one-line notice prints at launch when main is ahead; run `omh update` to apply it.")
    else:
        print("Auto: `omh update` runs automatically at launch when main is ahead.")
    return 0


def cmd_update_check_accept_gap(args: argparse.Namespace) -> int:
    """Maintainer acceptance of an open coverage gap (issue #1282).

    Acceptance is explicit: it records that a human reviewed the rewritten
    history and the uncertain interval, and permits the next `omh update` to
    re-anchor the cursor. It never claims the unreachable commits were
    recovered.
    """
    paths = _paths(args)
    before = read_update_check_cache(paths)
    before_gap = before.get("gap") if isinstance(before, dict) else None
    was_open = isinstance(before_gap, dict) and before_gap.get("status") == "open"
    cache = accept_update_check_gap(paths)
    gap = cache.get("gap") if isinstance(cache, dict) else None
    if _wants_json(args):
        _print_json(
            {
                "schema_version": "omh_update_check_status/v1",
                "accepted": was_open,
                "gap": gap if isinstance(gap, dict) else {},
                "cache_path": str(update_check_cache_path(paths)),
            }
        )
        return 0
    if was_open:
        print("Coverage gap accepted; the next `omh update` may re-anchor the recorded cursor.")
    else:
        print("No open coverage gap to accept.")
    return 0


def _add_update_check_commands(sub) -> None:
    group = sub.add_parser(
        "update-check",
        help="Configure the opt-in startup check that compares this install against origin/main.",
    )
    group_sub = group.add_subparsers(dest="update_check_command", required=True)

    status = group_sub.add_parser("status", help="Show the update-check mode, interval, and last known outcome.")
    status.add_argument("--json", action="store_true", help="Print the machine-readable status payload.")
    status.set_defaults(func=cmd_update_check_status)

    set_cmd = group_sub.add_parser("set", help="Change the update-check mode and/or interval.")
    set_cmd.add_argument("--mode", choices=UPDATE_CHECK_MODES, default=None, help="off (shipped default), notify, or auto.")
    set_cmd.add_argument(
        "--interval-hours",
        type=float,
        default=None,
        help="Minimum hours between network checks (default 24).",
    )
    set_cmd.add_argument("--json", action="store_true", help="Print the machine-readable status payload.")
    set_cmd.set_defaults(func=cmd_update_check_set)

    accept_gap = group_sub.add_parser(
        "accept-gap",
        help="Accept an open coverage gap after reviewing rewritten upstream history.",
    )
    accept_gap.add_argument("--json", action="store_true", help="Print the machine-readable status payload.")
    accept_gap.set_defaults(func=cmd_update_check_accept_gap)
