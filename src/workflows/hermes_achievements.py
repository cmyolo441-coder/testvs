from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from typing import Any

from ..system.local_store import read_json_object_result, utc_now
from ..system.paths import OmhPaths

ACHIEVEMENTS_OBSERVATION_SCHEMA_VERSION = "hermes_achievements_observation/v1"
ACHIEVEMENTS_AGENT_PROFILE_SCHEMA_VERSION = "hermes_achievements_agent_profile/v1"
ACHIEVEMENTS_EVIDENCE_BOUNDARY = (
    "Achievements are observed from local hermes-achievements plugin artifacts only. "
    "OMH does not rescan Hermes session history and does not claim unlocks it did not read."
)
BADGE_STATES = ("unlocked", "discovered", "secret")
TIER_ORDER = ("copper", "silver", "gold", "diamond", "olympian")
# The plugin's `SNAPSHOT_TTL_SECONDS`: it treats a snapshot older than this as
# stale and rescans when its dashboard tab next asks for badges. OMH applies the
# same rule to the file it read and never calls the dashboard API.
PLUGIN_SNAPSHOT_TTL_SECONDS = 120
SNAPSHOT_STALENESS_RULE = (
    "stale when generated_at is missing or more than ttl_seconds old, the plugin's own rescan rule"
)

# The plugin artifact layout is upstream-internal; only badge ids are stable.
# Every extractor below tolerates missing keys and unknown shapes by degrading
# to fewer observed fields instead of raising.
_SNAPSHOT_BADGE_KEYS = ("achievements", "badges", "catalog", "items")
_STATE_UNLOCK_KEYS = ("unlocked", "unlocks", "unlocked_badges", "history")
_SCAN_TIMESTAMP_KEYS = ("completed_at", "scanned_at", "generated_at", "last_scan_at", "updated_at")


def observe_achievements(
    paths: OmhPaths,
    *,
    recent_limit: int = 5,
    now: float | None = None,
) -> dict[str, Any]:
    snapshot_path, snapshot_location = _plugin_file(paths, "scan_snapshot.json")
    state_path, state_location = _plugin_file(paths, "state.json")
    errors: list[str] = []
    # `utf-8-sig`, as current Hermes (0.21.6 and later) reads them: a file
    # saved with a BOM still parses, and one without is unaffected.
    snapshot, snapshot_error = read_json_object_result(snapshot_path, encoding="utf-8-sig")
    if snapshot_error:
        errors.append(f"scan_snapshot.json: {snapshot_error}")
    state, state_error = read_json_object_result(state_path, encoding="utf-8-sig")
    if state_error:
        errors.append(f"state.json: {state_error}")

    badges = _badges_from_snapshot(snapshot or {})
    _apply_state_unlocks(badges, state or {})
    badge_list = sorted(badges.values(), key=lambda badge: (badge["category"], badge["name"], badge["badge_id"]))

    observed = snapshot is not None or state is not None
    summary = _summary(badge_list, snapshot or {})
    return {
        "schema_version": ACHIEVEMENTS_OBSERVATION_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "status": "observed" if observed else "not_observed",
        "observed": observed,
        "source": {
            "data_dir": str(paths.hermes_achievements_data_dir),
            "legacy_dir": str(paths.hermes_achievements_legacy_dir),
            "snapshot_path": str(snapshot_path),
            "snapshot_location": snapshot_location,
            "snapshot_observed": snapshot is not None,
            "state_path": str(state_path),
            "state_location": state_location,
            "state_observed": state is not None,
        },
        "snapshot_freshness": _snapshot_freshness(snapshot, now=time.time() if now is None else now),
        "errors": errors,
        "summary": summary,
        "badges": badge_list,
        "recent_unlocks": _recent_unlocks(badge_list, limit=recent_limit),
        "agent_profile": _agent_profile(summary, observed=observed),
        "evidence_boundary": ACHIEVEMENTS_EVIDENCE_BOUNDARY,
        "privacy": "metadata_only",
    }


def _plugin_file(paths: OmhPaths, name: str) -> tuple[Path, str]:
    # The plugin's own per-file choice: the plugin-data copy whenever it exists,
    # readable or not, and the legacy copy only when plugin-data has none.
    # `os.path.exists` because it never raises, so the read below stays the one
    # place a failure is reported.
    current = paths.hermes_achievements_data_dir / name
    if os.path.exists(current):
        return current, "plugin_data"
    legacy = paths.hermes_achievements_legacy_dir / name
    if os.path.exists(legacy):
        return legacy, "legacy_plugins_dir"
    return current, "not_found"


def _snapshot_freshness(snapshot: dict[str, Any] | None, *, now: float) -> dict[str, Any]:
    value = (snapshot or {}).get("generated_at")
    number = _finite_float(value)
    generated_at = int(value) if number is not None and number > 0 else None
    age_seconds = int(now) - generated_at if generated_at is not None else None
    return {
        "generated_at": generated_at,
        "age_seconds": age_seconds,
        "ttl_seconds": PLUGIN_SNAPSHOT_TTL_SECONDS,
        "stale": age_seconds is None or age_seconds > PLUGIN_SNAPSHOT_TTL_SECONDS,
        "rule": SNAPSHOT_STALENESS_RULE,
    }


def filter_badges(
    badges: list[dict[str, Any]],
    *,
    category: str = "",
    state: str = "",
    limit: int = 0,
) -> list[dict[str, Any]]:
    selected = list(badges)
    if category:
        wanted_category = category.strip().lower()
        selected = [badge for badge in selected if badge["category"].lower() == wanted_category]
    if state:
        wanted_state = state.strip().lower()
        selected = [badge for badge in selected if badge["state"] == wanted_state]
    if limit > 0:
        selected = selected[:limit]
    return selected


def find_badge(badges: list[dict[str, Any]], badge_id: str) -> dict[str, Any] | None:
    wanted = badge_id.strip()
    for badge in badges:
        if badge["badge_id"] == wanted:
            return badge
    lowered = wanted.lower()
    for badge in badges:
        if badge["badge_id"].lower() == lowered or badge["name"].lower() == lowered:
            return badge
    return None


def render_achievements_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    lines = [
        "# Hermes Achievements (OMH observation)",
        "",
        f"- Status: {payload.get('status', 'not_observed')}",
        f"- Generated: {payload.get('generated_at', '')}",
        f"- Unlocked: {summary.get('unlocked_count', 0)} / {summary.get('total_count', 0)}",
    ]
    if summary.get("top_tier"):
        lines.append(f"- Top tier: {summary['top_tier']}")
    if summary.get("last_scan_at"):
        lines.append(f"- Last plugin scan: {display_timestamp(summary['last_scan_at'])}")
    freshness_line = snapshot_freshness_text(payload)
    if freshness_line:
        lines.append(f"- {freshness_line}")
    badges = payload.get("badges", [])
    if badges:
        current_category = None
        for badge in badges:
            if badge["category"] != current_category:
                current_category = badge["category"]
                lines.extend(
                    [
                        "",
                        f"## {current_category}",
                        "",
                        "| Badge | Tier | State | Progress | Unlocked at |",
                        "| --- | --- | --- | --- | --- |",
                    ]
                )
            lines.append(
                f"| {badge['name']} (`{badge['badge_id']}`) | {badge.get('tier') or ''} "
                f"| {badge['state']} | {_progress_text(badge.get('progress_percent'))} "
                f"| {display_timestamp(badge.get('unlocked_at'))} |"
            )
    else:
        lines.extend(["", "No local hermes-achievements badge artifacts were observed."])
    lines.extend(["", f"Boundary: {payload.get('evidence_boundary', ACHIEVEMENTS_EVIDENCE_BOUNDARY)}", ""])
    return "\n".join(lines)


def snapshot_freshness_text(payload: dict[str, Any]) -> str:
    freshness = payload.get("snapshot_freshness")
    if not isinstance(freshness, dict) or freshness.get("generated_at") is None:
        return ""
    verdict = "stale" if freshness.get("stale") else "fresh"
    return (
        f"Snapshot: {verdict} by the plugin's {freshness.get('ttl_seconds')}s rescan rule "
        f"(generated {display_timestamp(freshness['generated_at'])}, age {freshness.get('age_seconds')}s)"
    )


def display_timestamp(value: Any) -> str:
    """A plugin timestamp for people to read; the JSON payload keeps it raw.

    The plugin writes epoch seconds, shown here as UTC ISO-8601 to the second.
    A value that is not a positive epoch number, such as an ISO string, is
    shown as written.
    """
    text = _clean_text(value)
    try:
        epoch = float(text)
    except ValueError:
        return text
    if not math.isfinite(epoch) or epoch <= 0:
        return text
    try:
        moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return text
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _progress_text(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return f"{float(value):g}%"


def _clean_text(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return ""


def _badge_entries(container: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(container, dict):
        return [(_clean_text(key), value) for key, value in container.items() if isinstance(value, dict)]
    if isinstance(container, list):
        return [("", value) for value in container if isinstance(value, dict)]
    return []


def _badges_from_snapshot(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    badges: dict[str, dict[str, Any]] = {}
    for key in _SNAPSHOT_BADGE_KEYS:
        entries = _badge_entries(snapshot.get(key))
        if not entries:
            continue
        for fallback_id, entry in entries:
            badge = _normalize_badge(entry, fallback_id=fallback_id)
            if badge:
                badges[badge["badge_id"]] = badge
        break
    return badges


def _normalize_badge(entry: dict[str, Any], *, fallback_id: str = "") -> dict[str, Any] | None:
    badge_id = _clean_text(entry.get("id") or entry.get("badge_id") or entry.get("key")) or fallback_id
    if not badge_id:
        return None
    return {
        "badge_id": badge_id,
        "name": _clean_text(entry.get("name") or entry.get("title")) or badge_id,
        "tier": _normalize_tier(entry.get("tier")),
        "category": _clean_text(entry.get("category") or entry.get("group")) or "uncategorized",
        "state": _normalize_state(entry),
        "progress_percent": _entry_progress_percent(entry),
        "unlocked_at": _clean_text(entry.get("unlocked_at") or entry.get("unlockedAt")),
    }


def _entry_progress_percent(entry: dict[str, Any]) -> float | None:
    # The bundled plugin's `progress` is the raw metric count for a tiered
    # badge (the averaged percent for a multi-condition one), so a count of 1
    # or 450 must not be read as a fraction or a percent. Its `progress_pct` is
    # the percent toward the next tier, or toward completion, for both kinds.
    # When `progress_pct` is there but unusable, `progress` is no substitute.
    if "progress_pct" in entry:
        percent = _finite_float(entry["progress_pct"])
        return round(min(max(percent, 0.0), 100.0), 1) if percent is not None else None
    return _progress_percent(entry.get("progress"))


def _finite_float(value: Any) -> float | None:
    # A JSON integer can be too large for a float (OverflowError) and JSON as
    # Python reads it allows NaN and Infinity; none of those is a usable number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _normalize_tier(value: Any) -> str:
    tier = _clean_text(value).lower()
    return tier if tier in TIER_ORDER else ""


def _normalize_state(entry: dict[str, Any]) -> str:
    state = _clean_text(entry.get("state") or entry.get("status")).lower()
    if state in BADGE_STATES:
        return state
    if entry.get("unlocked") is True:
        return "unlocked"
    if entry.get("secret") is True or entry.get("hidden") is True:
        return "secret"
    return "discovered"


def _progress_percent(value: Any) -> float | None:
    number = _finite_float(value)
    if number is not None:
        percent = number * 100.0 if 0.0 <= number <= 1.0 else number
        return round(min(max(percent, 0.0), 100.0), 1)
    if isinstance(value, dict):
        current = _finite_float(value.get("current"))
        target = _finite_float(value.get("target") or value.get("total"))
        if current is not None and target is not None and target > 0:
            return round(min(max(current / target * 100.0, 0.0), 100.0), 1)
        percent = _finite_float(value.get("percent"))
        if percent is not None:
            return round(min(max(percent, 0.0), 100.0), 1)
    return None


def _unlock_entries(state: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    for key in _STATE_UNLOCK_KEYS:
        container = state.get(key)
        if isinstance(container, dict):
            return [
                (_clean_text(badge_id), meta if isinstance(meta, dict) else {})
                for badge_id, meta in container.items()
            ]
        if isinstance(container, list):
            entries: list[tuple[str, dict[str, Any]]] = []
            for item in container:
                if isinstance(item, dict):
                    badge_id = _clean_text(item.get("id") or item.get("badge_id") or item.get("key"))
                    if badge_id:
                        entries.append((badge_id, item))
                else:
                    badge_id = _clean_text(item)
                    if badge_id:
                        entries.append((badge_id, {}))
            return entries
    return []


def _apply_state_unlocks(badges: dict[str, dict[str, Any]], state: dict[str, Any]) -> None:
    for badge_id, meta in _unlock_entries(state):
        if not badge_id:
            continue
        badge = badges.get(badge_id)
        # state.json is the plugin's unlock record. Current Hermes (0.21.4 and
        # later) treats it as a sticky floor: an unlock there stands even when
        # the snapshot shows the badge locked. Like that floor, this fills a
        # missing tier from `first_tier`, never lowers a tier, and leaves the
        # next-tier progress the snapshot recorded.
        first_tier = _normalize_tier(meta.get("first_tier"))
        if badge is None:
            badges[badge_id] = {
                "badge_id": badge_id,
                "name": _clean_text(meta.get("name") or meta.get("title")) or badge_id,
                "tier": _normalize_tier(meta.get("tier")) or first_tier,
                "category": _clean_text(meta.get("category") or meta.get("group")) or "uncategorized",
                "state": "unlocked",
                "progress_percent": 100.0,
                "unlocked_at": _clean_text(meta.get("unlocked_at") or meta.get("unlockedAt")),
            }
            continue
        badge["state"] = "unlocked"
        if badge["progress_percent"] is None:
            badge["progress_percent"] = 100.0
        # A `tier` key may only raise the tier. No bundled plugin version writes
        # one; it is kept from the original generic reader.
        tier = _normalize_tier(meta.get("tier"))
        if tier and (not badge["tier"] or TIER_ORDER.index(tier) > TIER_ORDER.index(badge["tier"])):
            badge["tier"] = tier
        elif not badge["tier"] and first_tier:
            badge["tier"] = first_tier
        unlocked_at = _clean_text(meta.get("unlocked_at") or meta.get("unlockedAt"))
        if unlocked_at:
            badge["unlocked_at"] = unlocked_at


def _summary(badges: list[dict[str, Any]], snapshot: dict[str, Any]) -> dict[str, Any]:
    unlocked = [badge for badge in badges if badge["state"] == "unlocked"]
    categories: dict[str, dict[str, int]] = {}
    for badge in badges:
        bucket = categories.setdefault(badge["category"], {"total": 0, "unlocked": 0})
        bucket["total"] += 1
        if badge["state"] == "unlocked":
            bucket["unlocked"] += 1
    return {
        "total_count": len(badges),
        "unlocked_count": len(unlocked),
        "discovered_count": sum(1 for badge in badges if badge["state"] == "discovered"),
        "secret_count": sum(1 for badge in badges if badge["state"] == "secret"),
        "top_tier": _top_tier(unlocked),
        "categories": [
            {"category": name, "total": bucket["total"], "unlocked": bucket["unlocked"]}
            for name, bucket in sorted(categories.items())
        ],
        "last_scan_at": _last_scan_at(snapshot),
    }


def _top_tier(unlocked: list[dict[str, Any]]) -> str:
    best = -1
    for badge in unlocked:
        tier = badge.get("tier", "")
        if tier in TIER_ORDER:
            best = max(best, TIER_ORDER.index(tier))
    return TIER_ORDER[best] if best >= 0 else ""


def _last_scan_at(snapshot: dict[str, Any]) -> str:
    for key in _SCAN_TIMESTAMP_KEYS:
        value = _clean_text(snapshot.get(key))
        if value:
            return value
    return ""


def _agent_profile(summary: dict[str, Any], *, observed: bool) -> dict[str, Any]:
    # Derived from the observed badges only: the bundled plugin writes no
    # agent summary file of its own, so there is no upstream profile to read.
    profile: dict[str, Any] = {
        "schema_version": ACHIEVEMENTS_AGENT_PROFILE_SCHEMA_VERSION,
        "observed": observed,
        "derivation": "none",
        "strengths": [],
        "gaps": [],
        "top_tier": summary.get("top_tier", ""),
        "unlocked_count": summary.get("unlocked_count", 0),
        "total_count": summary.get("total_count", 0),
        "routing_rule": (
            "Use strengths and gaps only as advisory workflow-suggestion context; "
            "never present them as productivity scores or execution evidence."
        ),
    }
    if observed:
        categories = summary.get("categories", [])
        if isinstance(categories, list):
            unlocked = [entry for entry in categories if isinstance(entry, dict) and entry.get("unlocked", 0) > 0]
            locked = [entry for entry in categories if isinstance(entry, dict) and entry.get("unlocked", 0) == 0]
            unlocked.sort(key=lambda entry: (-int(entry.get("unlocked", 0)), str(entry.get("category", ""))))
            profile["derivation"] = "derived_from_observed_badges"
            profile["strengths"] = [str(entry.get("category", "")) for entry in unlocked[:5]]
            profile["gaps"] = [str(entry.get("category", "")) for entry in locked[:5]]
    return profile


def _recent_unlocks(badges: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    unlocked = [badge for badge in badges if badge["state"] == "unlocked"]
    dated = sorted(
        (badge for badge in unlocked if badge["unlocked_at"]),
        key=lambda badge: badge["unlocked_at"],
        reverse=True,
    )
    undated = [badge for badge in unlocked if not badge["unlocked_at"]]
    safe_limit = max(0, int(limit))
    return [dict(badge) for badge in (dated + undated)[:safe_limit]]
