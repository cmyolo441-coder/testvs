"""Deterministic demo of the issue #1282 rewritten-history update-watch recovery.

The recovery engine lives in `omh.maintenance.update_check` (the network-safety
allowlist names that module as the only curl door); this facade replays the
classification scenarios against a throwaway OMH home with injected fake curl
runners -- zero network, zero real processes, fixed timestamps, byte-stable
output.

Scenario reel: an ordinary fast-forward, a rewritten `main` whose recorded
cursor is unreachable (coverage gap opens with old/new refs and the uncertain
interval), a repeated probe that dedupes to one ledger entry per event key,
an incompletely enumerated recovery window that keeps the gap open even after
the cursor converges, and a completed recovery that closes the gap and
advances the branch generation.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

if __package__ and __package__ != "runtime":
    from ..local_store import atomic_write_json
    from ..maintenance.update_check import (
        cursor_advance_allowed,
        evaluate_update_check,
        read_update_check_cache,
        write_update_check_policy,
    )
    from ..paths import OmhPaths
else:  # top-level `runtime.*` import (e.g. PYTHONPATH=src)
    from omh.local_store import atomic_write_json
    from omh.maintenance.update_check import (
        cursor_advance_allowed,
        evaluate_update_check,
        read_update_check_cache,
        write_update_check_policy,
    )
    from omh.paths import OmhPaths

UPDATE_WATCH_RECOVERY_DEMO_SCHEMA_VERSION = "omh_update_watch_recovery_demo/v1"

_T0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_CURSOR = "a" * 40
_HEAD_FAST_FORWARD = "b" * 40
_HEAD_REWRITTEN = "c" * 40


def _http(status: int, body: object = None, *, etag: str = "") -> str:
    lines = [f"HTTP/2 {status}"]
    if etag:
        lines.append(f"etag: {etag}")
    payload = json.dumps(body) if body is not None else ""
    return "\n".join(lines) + "\n\n" + payload


def _demo_runner(*, head_sha: str, head_time: str, compare_status: int = 200, tags_status: int = 200):
    """Fake `subprocess.run` for the watch's curl probes. Never a real process."""

    def runner(argv, timeout=None):
        urls = [arg for arg in argv if isinstance(arg, str) and arg.startswith("http")]
        write_out = argv[argv.index("-w") + 1] if "-w" in argv else ""
        if any("/compare/" in url for url in urls):
            body = {"status": "ahead", "ahead_by": 1, "behind_by": 0} if compare_status == 200 else None
            return subprocess.CompletedProcess(argv, 0, stdout=_http(compare_status, body), stderr="")
        if any("/tags" in url for url in urls):
            body = [{"name": "v0.2.0"}] if tags_status == 200 else None
            return subprocess.CompletedProcess(argv, 0, stdout=_http(tags_status, body), stderr="")
        if any("/releases" in url for url in urls):
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=_http(200, [{"tag_name": "v0.2.0"}]),
                stderr="",
            )
        head = _http(200, {"sha": head_sha, "commit": {"committer": {"date": head_time}}}, etag='"demo"')
        metadata = _http(200, {"default_branch": "main"})
        return subprocess.CompletedProcess(argv, 0, stdout=head + write_out + metadata + write_out, stderr="")

    return runner


def demo_rewrite_recovery() -> dict[str, Any]:
    """Replay the rewritten-history recovery scenarios. Deterministic output."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        paths = OmhPaths(root / ".omh", root / ".hermes")
        write_update_check_policy(paths, mode="notify")
        paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(paths.runtime_state_path, {"release_source_commit": _CURSOR})

        # 1. Ordinary incremental collection: a verified fast-forward.
        fast_forward = evaluate_update_check(
            paths,
            now=_T0,
            runner=_demo_runner(head_sha=_HEAD_FAST_FORWARD, head_time="2026-01-01T01:00:00Z"),
        )

        # 2. History rewritten: the head moved, the recorded cursor is
        # unreachable (compare 404), and the tags recovery source is failing.
        # Releases are enumerated separately and cannot compensate for it.
        rewritten_runner = _demo_runner(
            head_sha=_HEAD_REWRITTEN,
            head_time="2026-01-02T01:00:00Z",
            compare_status=404,
            tags_status=500,
        )
        rewritten = evaluate_update_check(paths, now=_T0 + timedelta(hours=1), force=True, runner=rewritten_runner)

        # 3. The same rewritten head probed again: one ledger entry per event key.
        repeat = evaluate_update_check(paths, now=_T0 + timedelta(hours=2), force=True, runner=rewritten_runner)
        cache = read_update_check_cache(paths)
        event_key = f"head_moved:main:{_HEAD_REWRITTEN}"
        deduped = [
            entry
            for entry in cache.get("recovery_attempts", [])
            if event_key in (entry.get("candidate_event_keys") or [])
        ]

        # 4. The cursor converges on the new head (as after `omh update`), but
        # the tags source still has not succeeded: coverage stays incomplete.
        atomic_write_json(paths.runtime_state_path, {"release_source_commit": _HEAD_REWRITTEN})
        converged = evaluate_update_check(paths, now=_T0 + timedelta(hours=3), force=True, runner=rewritten_runner)
        blocked_while_incomplete = not cursor_advance_allowed(paths)

        # 5. The failed recovery source succeeds: the gap closes and the
        # branch generation advances; only then may the cursor advance again.
        recovered_runner = _demo_runner(
            head_sha=_HEAD_REWRITTEN,
            head_time="2026-01-02T01:00:00Z",
            compare_status=404,
            tags_status=200,
        )
        recovered = evaluate_update_check(paths, now=_T0 + timedelta(hours=4), force=True, runner=recovered_runner)
        allowed_after_recovery = cursor_advance_allowed(paths)
        final_cache = read_update_check_cache(paths)

    return {
        "schema_version": UPDATE_WATCH_RECOVERY_DEMO_SCHEMA_VERSION,
        "watched_branch": "main",
        "default_branch": "main",
        "scenarios": [
            {
                "name": "fast_forward",
                "ancestry": fast_forward["ancestry"],
                "outcome": fast_forward["outcome"],
                "gap_status": fast_forward["gap"]["status"],
            },
            {
                "name": "cursor_unreachable",
                "ancestry": rewritten["ancestry"],
                "outcome": rewritten["outcome"],
                "gap_status": rewritten["gap"]["status"],
                "refs": {"old": _CURSOR, "new": _HEAD_REWRITTEN},
                "uncertain_interval": {
                    "since": rewritten["gap"]["since"],
                    "until": rewritten["gap"]["until"],
                },
            },
            {
                "name": "dedupe_repeat",
                "checked": repeat["checked"],
                "event_key": event_key,
                "ledger_entries": len(deduped),
                "latest_attempted_at": deduped[-1]["attempted_at"] if deduped else "",
            },
            {
                "name": "incomplete_coverage",
                "outcome": converged["outcome"],
                "gap_status": converged["gap"]["status"],
                "cursor_advance_allowed": not blocked_while_incomplete,
            },
            {
                "name": "recovered",
                "ancestry": recovered["ancestry"],
                "gap_status": recovered["gap"]["status"],
                "branch_generation": recovered["branch_generation"],
                "cursor_advance_allowed": allowed_after_recovery,
            },
        ],
        "dedupe": {"event_key": event_key, "ledger_entries": len(deduped), "probes": 2},
        "coverage": {"incomplete": converged["gap"]["status"], "recovered": recovered["gap"]["status"]},
        "advance_policy": {
            "blocked_while_recovery_incomplete": blocked_while_incomplete,
            "allowed_after_recovery": allowed_after_recovery,
        },
        "ledger_sources": sorted(
            {str(entry.get("source", "")) for entry in final_cache.get("recovery_attempts", [])}
        ),
    }
