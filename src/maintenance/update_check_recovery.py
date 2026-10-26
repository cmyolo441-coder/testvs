"""Ancestry classification, recovery ledger, and update-watch notices."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..paths import OmhPaths
from .update_check_probe import UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS, fetch_cursor_ancestry, fetch_recovery_releases, fetch_recovery_tags, fetch_watched_branch_state
from .update_check_state import UPDATE_CHECK_RESULT_SCHEMA_VERSION, WATCHED_BRANCH, interval_elapsed, local_installed_channel, local_installed_commit, read_update_check_cache, read_update_check_policy, write_update_check_cache

UPDATE_CHECK_ANCESTRY_CLASSES = ("fast_forward", "rewound", "rewritten", "branch_recreated", "cursor_unreachable", "default_branch_changed", "unknown")
RECOVERY_LEDGER_LIMIT = 20


def empty_gap() -> dict[str, str]:
    return {"status": "none", "since": "", "until": "", "reason": "", "note": ""}


def normalized_gap(cache: dict[str, Any]) -> dict[str, str]:
    gap, raw = empty_gap(), cache.get("gap")
    if isinstance(raw, dict):
        if str(raw.get("status", "") or "") in ("open", "accepted"):
            gap["status"] = str(raw["status"])
        for key in ("since", "until", "reason", "note"):
            gap[key] = str(raw.get(key, "") or "")
    return gap


def open_gap(gap: dict[str, str], *, since: str, until: str, reason: str, note: str) -> dict[str, str]:
    if gap["status"] == "open" and gap["since"]:
        since, reason, note = min(gap["since"], since), gap["reason"] or reason, gap["note"] or note
    return {"status": "open", "since": since, "until": until, "reason": reason, "note": note}


def cache_generation(cache: dict[str, Any]) -> int:
    value = cache.get("branch_generation")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def append_attempt(ledger: list[dict[str, Any]], *, attempted_at: str, source: str, result: str, old_ref: str = "", new_ref: str = "", event_keys: list[str] | None = None, uncertain_interval: dict[str, str] | None = None, note: str = "") -> list[dict[str, Any]]:
    keys = list(event_keys or [])
    entry: dict[str, Any] = {"attempted_at": attempted_at, "source": source, "result": result, "old_ref": old_ref, "new_ref": new_ref, "candidate_event_keys": keys, "uncertain_interval": uncertain_interval or {"since": "", "until": ""}}
    if note:
        entry["note"] = note
    wanted = set(keys)
    for existing in ledger:
        if wanted and wanted & set(existing.get("candidate_event_keys") or []):
            existing.update(entry)
            return ledger
    ledger.append(entry)
    del ledger[:-RECOVERY_LEDGER_LIMIT]
    return ledger


def recovery_covered(ledger: list[dict[str, Any]], since: str, *, head_ok_now: bool) -> bool:
    for source in ("branch_head", "compare", "tags", "releases"):
        if source == "branch_head" and head_ok_now:
            continue
        if not any(entry.get("source") == source and entry.get("result") in ("ok", "not_found") and str(entry.get("attempted_at", "")) >= since for entry in ledger):
            return False
    return True


def record_recovery_sources(
    ledger: list[dict[str, Any]],
    *,
    attempted_at: str,
    old_ref: str,
    new_ref: str,
    since: str,
    runner: object,
    timeout: float,
) -> list[dict[str, Any]]:
    tags = fetch_recovery_tags(timeout=timeout, runner=runner)
    ledger = append_attempt(ledger, attempted_at=attempted_at, source="tags", result="ok" if tags.ok else "error", old_ref=old_ref, new_ref=new_ref, event_keys=[f"tags:{WATCHED_BRANCH}:{new_ref}"], uncertain_interval={"since": since, "until": attempted_at}, note=tags.error or "")
    releases = fetch_recovery_releases(timeout=timeout, runner=runner)
    return append_attempt(ledger, attempted_at=attempted_at, source="releases", result="ok" if releases.ok else "error", old_ref=old_ref, new_ref=new_ref, event_keys=[f"releases:{WATCHED_BRANCH}:{new_ref}"], uncertain_interval={"since": since, "until": attempted_at}, note=releases.error or "")


def cursor_advance_allowed(paths: OmhPaths) -> bool:
    cache = read_update_check_cache(paths)
    if not cache:
        return True
    gap = normalized_gap(cache)
    if gap["status"] == "open":
        return False
    if gap["status"] == "accepted":
        return True
    ancestry = cache.get("ancestry")
    return ancestry is None or str(ancestry) == "fast_forward"


def accept_update_check_gap(paths: OmhPaths, *, now: datetime | None = None) -> dict[str, Any]:
    cache = read_update_check_cache(paths)
    gap = normalized_gap(read_update_check_cache(paths))
    if gap["status"] != "open":
        return cache
    now_iso = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    gap["status"], gap["until"] = "accepted", gap["until"] or now_iso
    ledger = [dict(entry) for entry in cache.get("recovery_attempts") or [] if isinstance(entry, dict)]
    ledger = append_attempt(ledger, attempted_at=now_iso, source="maintainer", result="ok", event_keys=[f"gap_accepted:{WATCHED_BRANCH}:{cache_generation(cache)}:{gap['since']}"], uncertain_interval={"since": gap["since"], "until": now_iso}, note="maintainer accepted the unresolved coverage gap")
    return write_update_check_cache(paths, {"gap": gap, "recovery_attempts": ledger})


def result_payload(*, mode: str, outcome: str, checked: bool, local_commit: str, remote_commit: str, channel: str, ancestry: str, default_branch: str, generation: int, gap: dict[str, str]) -> dict[str, Any]:
    return {"schema_version": UPDATE_CHECK_RESULT_SCHEMA_VERSION, "mode": mode, "outcome": outcome, "checked": checked, "should_auto_update": mode == "auto" and checked and outcome == "behind" and ancestry == "fast_forward", "local_commit": local_commit, "remote_commit": remote_commit, "channel": channel, "ancestry": ancestry, "watched_branch": WATCHED_BRANCH, "default_branch": default_branch, "branch_generation": generation, "gap": gap}


def evaluate_update_check(paths: OmhPaths, *, now: datetime | None = None, force: bool = False, runner=None, timeout: float = UPDATE_CHECK_NETWORK_TIMEOUT_SECONDS) -> dict[str, Any]:
    policy = read_update_check_policy(paths)
    cache = read_update_check_cache(paths)
    mode = policy["mode"]
    local_commit = local_installed_commit(paths)
    channel = local_installed_channel(paths)
    if mode == "off":
        return result_payload(mode=mode, outcome="skipped_off", checked=False, local_commit="", remote_commit="", channel="", ancestry="unknown", default_branch="", generation=0, gap=empty_gap())
    now_iso = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    if not force and not interval_elapsed(cache, float(policy["interval_hours"]), datetime.fromisoformat(now_iso)):
        return result_payload(mode=mode, outcome=str(cache.get("outcome") or "inconclusive"), checked=False, local_commit=local_commit, remote_commit=str(cache.get("remote_commit", "")), channel=channel, ancestry=str(cache.get("ancestry") or "unknown"), default_branch=str(cache.get("default_branch") or ""), generation=cache_generation(cache), gap=normalized_gap(cache))
    probe = fetch_watched_branch_state(etag=str(cache.get("remote_etag", "")), timeout=timeout, runner=runner)
    ledger = [dict(entry) for entry in cache.get("recovery_attempts") or [] if isinstance(entry, dict)]
    gap = normalized_gap(cache)
    generation = cache_generation(cache)
    cutoff = str(cache.get("last_successful_cutoff") or "")
    persisted_default = str(cache.get("default_branch") or "") or None
    recorded_time = str(cache.get("remote_head_time") or "")
    if not probe.head.ok and not probe.visibility_suspect:
        ledger = append_attempt(ledger, attempted_at=now_iso, source="branch_head", result="error", old_ref=local_commit, event_keys=[f"head_probe_error:{WATCHED_BRANCH}"], uncertain_interval={"since": cutoff, "until": now_iso}, note=probe.head.error or "probe failure")
        write_update_check_cache(paths, {"last_checked_at": now_iso, "recovery_attempts": ledger})
        return result_payload(mode=mode, outcome=str(cache.get("outcome") or "inconclusive"), checked=True, local_commit=local_commit, remote_commit=str(cache.get("remote_commit", "")), channel=channel, ancestry=str(cache.get("ancestry") or "unknown"), default_branch=str(cache.get("default_branch") or ""), generation=generation, gap=gap)
    remote = probe.head.sha if not probe.head.not_modified else str(cache.get("remote_commit", ""))
    etag = probe.head.etag or str(cache.get("remote_etag", ""))
    head_time = probe.head_time or recorded_time
    if probe.metadata_partial:
        ledger = append_attempt(ledger, attempted_at=now_iso, source="repo_metadata", result="error", event_keys=[f"metadata_partial:{WATCHED_BRANCH}"], uncertain_interval={"since": cutoff, "until": now_iso}, note="partial or missing repository-metadata read")
    ancestry, outcome = "unknown", str(cache.get("outcome") or "inconclusive")
    if probe.visibility_suspect:
        outcome = "inconclusive"
        note = "shallow or delayed remote visibility"
        gap = open_gap(gap, since=cutoff or now_iso, until=now_iso, reason="unknown", note=note)
        ledger = append_attempt(ledger, attempted_at=now_iso, source="branch_head", result="error", old_ref=local_commit, new_ref=remote, event_keys=[f"visibility_suspect:{WATCHED_BRANCH}"], uncertain_interval={"since": gap["since"], "until": now_iso}, note=probe.head.error or note)
    elif probe.default_branch and persisted_default and probe.default_branch != persisted_default:
        ancestry, outcome = "default_branch_changed", "inconclusive"
        gap = open_gap(gap, since=cutoff or now_iso, until=now_iso, reason=ancestry, note=f"default branch changed: {persisted_default} -> {probe.default_branch}")
        ledger = append_attempt(ledger, attempted_at=now_iso, source="repo_metadata", result="ok", old_ref=persisted_default, new_ref=probe.default_branch, event_keys=[f"default_branch_changed:{WATCHED_BRANCH}:{probe.default_branch}"], uncertain_interval={"since": gap["since"], "until": now_iso})
    elif not local_commit or not remote:
        outcome = "inconclusive"
    elif local_commit == remote:
        ancestry, outcome = "fast_forward", "up_to_date"
        if gap["status"] == "open" and gap["reason"] != "default_branch_changed":
            if not recovery_covered(ledger, gap["since"], head_ok_now=True):
                ledger = record_recovery_sources(ledger, attempted_at=now_iso, old_ref=local_commit, new_ref=remote, since=gap["since"], runner=runner, timeout=timeout)
            if recovery_covered(ledger, gap["since"], head_ok_now=True):
                generation += 1
                ledger = append_attempt(ledger, attempted_at=now_iso, source="branch_head", result="ok", old_ref=local_commit, new_ref=remote, event_keys=[f"gap_closed:{WATCHED_BRANCH}:{generation}:{gap['since']}"], uncertain_interval={"since": gap["since"], "until": now_iso}, note="recovery window fully enumerated; gap closed")
                gap = empty_gap()
    else:
        compare = fetch_cursor_ancestry(local_commit, remote, timeout=timeout, runner=runner)
        ledger = append_attempt(ledger, attempted_at=now_iso, source="compare", result="not_found" if compare.classification == "cursor_unreachable" else "ok" if compare.classification != "unknown" else "error", old_ref=local_commit, new_ref=remote, event_keys=[f"head_moved:{WATCHED_BRANCH}:{remote}"], uncertain_interval={"since": cutoff, "until": now_iso}, note=compare.error or "")
        ancestry = compare.classification
        outcome = "behind" if ancestry == "fast_forward" else str(cache.get("outcome") or "inconclusive") if ancestry == "unknown" else "inconclusive"
        if recorded_time and probe.head_time and probe.head_time < recorded_time:
            if ancestry == "cursor_unreachable":
                ancestry = "branch_recreated"
            elif ancestry != "unknown":
                ancestry, outcome = "unknown", "inconclusive"
                note = "remote head time regressed (shallow or delayed visibility)"
                gap = open_gap(gap, since=cutoff or now_iso, until=now_iso, reason="unknown", note=note)
                ledger = append_attempt(ledger, attempted_at=now_iso, source="branch_head", result="error", old_ref=local_commit, new_ref=remote, event_keys=[f"visibility_suspect:{WATCHED_BRANCH}"], uncertain_interval={"since": gap["since"], "until": now_iso}, note=note)
        if ancestry in ("rewritten", "cursor_unreachable", "branch_recreated"):
            note = f"cursor {local_commit[:7] or 'unknown'} is not reachable from the watched head"
            if ancestry == "branch_recreated":
                note += "; branch recreation heuristic matched (remote head time regressed)"
            gap = open_gap(gap, since=cutoff or now_iso, until=now_iso, reason=ancestry, note=note)
            ledger = record_recovery_sources(ledger, attempted_at=now_iso, old_ref=local_commit, new_ref=remote, since=gap["since"], runner=runner, timeout=timeout)
    resolved_default = probe.default_branch or persisted_default
    patch: dict[str, Any] = {"last_checked_at": now_iso, "remote_commit": remote, "remote_etag": etag, "outcome": outcome, "ancestry": ancestry, "watched_branch": WATCHED_BRANCH, "branch_generation": generation, "recovery_attempts": ledger, "gap": gap}
    if resolved_default:
        patch["default_branch"] = resolved_default
    if head_time:
        patch["remote_head_time"] = max(recorded_time, head_time) if recorded_time else head_time
    if probe.metadata_ok and ancestry == "fast_forward" and gap["status"] != "open":
        patch["last_successful_cutoff"] = now_iso
    write_update_check_cache(paths, patch)
    return result_payload(mode=mode, outcome=outcome, checked=True, local_commit=local_commit, remote_commit=remote, channel=channel, ancestry=ancestry, default_branch=str(resolved_default or ""), generation=generation, gap=gap)


def refresh_cache_after_auto_update(paths: OmhPaths) -> dict[str, Any]:
    local = local_installed_commit(paths)
    remote = str(read_update_check_cache(paths).get("remote_commit", ""))
    return write_update_check_cache(paths, {"outcome": "up_to_date" if local and local == remote else "inconclusive"})


def format_notice_line(result: dict[str, Any]) -> str:
    gap, outcome = result.get("gap"), result.get("outcome")
    if isinstance(gap, dict) and gap.get("status") == "open":
        ancestry, since = str(result.get("ancestry") or ""), str(gap.get("since") or "") or "unknown"
        if ancestry == "default_branch_changed":
            return f"OMH update check: {str(gap.get('note') or 'the repository default branch changed')}; the recorded cursor is pinned and a coverage gap since {since} is open; run `omh update-check status` for details."
        if ancestry in ("rewound", "rewritten", "branch_recreated", "cursor_unreachable"):
            return f"OMH update check: origin/main history rewritten (ancestry: {ancestry}); coverage gap since {since} is recorded; run `omh update` to re-anchor."
        return f"OMH update check: origin/main ancestry could not be verified; coverage gap since {since} is recorded; run `omh update` to re-anchor."
    if outcome == "behind":
        return f"OMH update available: {str(result.get('local_commit', ''))[:7] or 'unknown'} -> {str(result.get('remote_commit', ''))[:7] or 'unknown'}; run `omh update`."
    channel = str(result.get("channel", ""))
    if outcome == "inconclusive" and channel and channel != "preview":
        return f"OMH update check: not applicable on the {channel} channel (only preview tracks main)."
    return "OMH update check inconclusive -- run `omh update`." if outcome == "inconclusive" else ""
