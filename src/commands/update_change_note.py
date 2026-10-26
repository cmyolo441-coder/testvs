"""What an `omh update` changed for this user, as a metadata-only record.

Most users never read the changelog, so a new skill, a request that now
reaches a different skill, or a model chain that moved stays invisible unless
the update says so itself. This module compares the generation that was
installed before the update with the one installed now and writes the result
to `<omh_home>/runtime/update-change-note.json`, where `omh update` prints it
and the `omh_status` plugin tool returns it when someone asks what changed.

Every input is OMH's own output, never the user's configuration:

* skills come from the managed manifest's per-skill checksums, which record
  what the installer wrote; a local edit is refused (or overwritten with
  `--force`), so it never reaches the comparison;
* request routing comes from each skill's shipped trigger phrases, recorded
  in `<omh_home>/runtime/update-baseline.json` at the end of every update;
* model chains compare the SHIPPED chains of both generations. The user's
  chain overrides and provider answers are read once and applied to both
  sides, so an edit there cancels out and a category the user overrides is
  reported as "your override still applies" rather than as a change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..local_store import atomic_write_json, read_json_object, utc_now
from ..plugin_bundle.omh.hermes_delegation import (
    HERMES_MIXTURE_CATEGORY_CHAINS,
    effective_provider_entitlements,
    entitlement_shaped_chain,
    load_mixture_chain_overrides,
    load_model_provider_routes,
)
from ..skills.catalog import omh_skill_display_name, routable_definitions
from ..version import __version__
from .language import tr

BASELINE_SCHEMA_VERSION = "update_baseline/v1"
NOTE_SCHEMA_VERSION = "update_change_note/v1"
BASELINE_FILE_NAME = "update-baseline.json"
NOTE_FILE_NAME = "update-change-note.json"

# How much of a list the record and the printed note carry. The full answer is
# one `omh list` / `omh model-chains show` away; the note only has to make the
# change visible.
PHRASE_LIMIT = 3
ITEM_LIMIT = 8
SUMMARY_CHAR_LIMIT = 90


def _baseline_snapshot() -> dict[str, Any]:
    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "version": __version__,
        "triggers": {
            definition.name: sorted(set(definition.triggers)) for definition in routable_definitions()
        },
        "model_chains": {
            category: [list(entry) for entry in chain]
            for category, chain in HERMES_MIXTURE_CATEGORY_CHAINS.items()
        },
    }


def _skill_hashes(manifest: dict[str, Any] | None) -> dict[str, str]:
    skills = (manifest or {}).get("skills")
    if not isinstance(skills, list):
        return {}
    return {
        str(row["name"]): str(row.get("sha256", ""))
        for row in skills
        if isinstance(row, dict) and row.get("name")
    }


def _summary(description: str) -> str:
    text = description.removeprefix("[omh]").strip()
    _, separator, rest = text.partition(" - ")
    text = (rest if separator else text).strip()
    return text if len(text) <= SUMMARY_CHAR_LIMIT else text[: SUMMARY_CHAR_LIMIT - 3].rstrip() + "..."


def _skill_changes(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    before = _skill_hashes(previous)
    after = _skill_hashes(current)
    descriptions = {definition.name: definition.description for definition in routable_definitions()}

    def row(name: str, *, with_summary: bool = False) -> dict[str, str]:
        entry = {"name": name, "label": omh_skill_display_name(name)}
        if with_summary and descriptions.get(name):
            entry["summary"] = _summary(descriptions[name])
        return entry

    return {
        "added": [row(name, with_summary=True) for name in sorted(after.keys() - before.keys())],
        "removed": [row(name) for name in sorted(before.keys() - after.keys())],
        "changed": [row(name) for name in sorted(before.keys() & after.keys()) if before[name] != after[name]],
    }


def _routing_changes(previous: dict[str, Any], current: dict[str, Any], installed: set[str]) -> list[dict[str, Any]]:
    before = previous.get("triggers") if isinstance(previous.get("triggers"), dict) else {}
    after = current["triggers"]
    rows: list[dict[str, Any]] = []
    # A skill that only appeared or disappeared is already named under skills;
    # routing reports the ones that stayed and now answer different requests.
    for name in sorted((before.keys() & after.keys()) & installed):
        old = set(before[name]) if isinstance(before[name], list) else set()
        new = set(after[name])
        added, removed = sorted(new - old), sorted(old - new)
        if not added and not removed:
            continue
        rows.append(
            {
                "name": name,
                "label": omh_skill_display_name(name),
                "added": added[:PHRASE_LIMIT],
                "added_count": len(added),
                "removed": removed[:PHRASE_LIMIT],
                "removed_count": len(removed),
            }
        )
    return rows


def _model_chain_changes(previous: dict[str, Any], current: dict[str, Any], paths: Any) -> list[dict[str, Any]]:
    before = previous.get("model_chains") if isinstance(previous.get("model_chains"), dict) else {}
    after = current["model_chains"]
    overrides, _status = load_mixture_chain_overrides(paths.omh_home)
    entitlements, _source, _providers = effective_provider_entitlements(paths.omh_home, paths.hermes_home)
    routes, _ = load_model_provider_routes(paths.omh_home)

    def shaped(chain: object) -> list[list[str]]:
        entries = tuple(
            (str(item[0]), str(item[1]))
            for item in (chain if isinstance(chain, list) else [])
            if isinstance(item, list) and len(item) == 2
        )
        if entitlements is not None:
            entries = entitlement_shaped_chain(entries, entitlements, routes)
        return [list(entry) for entry in entries]

    rows: list[dict[str, Any]] = []
    for category in sorted(before.keys() | after.keys()):
        old, new = shaped(before.get(category)), shaped(after.get(category))
        if old == new:
            continue
        rows.append({"category": category, "before": old, "after": new, "override_applies": category in overrides})
    return rows


def record_update_change_note(
    paths: Any,
    *,
    previous_manifest: dict[str, Any] | None,
    current_manifest: dict[str, Any],
    previous_version: str,
) -> dict[str, Any]:
    """Compare the previous generation with the one just installed; persist both.

    Returns the note. A note that found nothing to report (`no_change`) is not
    written, so the stored record keeps naming the last update that did change
    something.
    """
    runtime_dir = Path(paths.runtime_dir)
    baseline = read_json_object(runtime_dir / BASELINE_FILE_NAME) or {}
    if baseline.get("schema_version") != BASELINE_SCHEMA_VERSION:
        baseline = {}
    current = _baseline_snapshot()
    note: dict[str, Any] = {
        "schema_version": NOTE_SCHEMA_VERSION,
        "recorded_at": utc_now(),
        "from_version": str(baseline.get("version") or previous_version or ""),
        "to_version": __version__,
        "privacy": "metadata_only",
    }
    if not previous_manifest:
        note["status"] = "first_install"
    else:
        skills = _skill_changes(previous_manifest, current_manifest)
        installed = set(_skill_hashes(current_manifest))
        note["skills"] = skills
        if baseline:
            note["routing"] = _routing_changes(baseline, current, installed)
            note["model_chains"] = _model_chain_changes(baseline, current, paths)
        else:
            # The generation before this one predates the baseline record, so
            # only the skill checksums (which every manifest carries) compare.
            note["unrecorded"] = ["routing", "model_chains"]
        moved = any(skills.values()) or note.get("routing") or note.get("model_chains")
        note["status"] = "changed" if moved else "no_change"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if note["status"] != "no_change":
        atomic_write_json(runtime_dir / NOTE_FILE_NAME, note)
    atomic_write_json(runtime_dir / BASELINE_FILE_NAME, current)
    return note


def _names(rows: list[dict[str, Any]], *, with_summary: bool = False) -> str:
    shown = rows[:ITEM_LIMIT]
    parts = [
        f"{row['label']} ({row['summary']})" if with_summary and row.get("summary") else str(row["label"])
        for row in shown
    ]
    text = ", ".join(parts)
    if len(rows) > len(shown):
        text += f" (+{len(rows) - len(shown)})"
    return text


def _phrases(values: list[str], total: int) -> str:
    text = ", ".join(f'"{value}"' for value in values)
    return text + (f" (+{total - len(values)})" if total > len(values) else "")


def _chain(entries: list[list[str]]) -> str:
    return " > ".join(f"{model}:{effort}" for model, effort in entries) or "-"


def update_change_note_lines(note: dict[str, Any], *, language: str) -> list[str]:
    """Plain-language lines for a note; empty when there is nothing to say."""
    status = note.get("status")
    if status == "first_install":
        return [tr(language, "change_note_title"), f"  {tr(language, 'change_note_first_install')}"]
    if status != "changed" and not note.get("unrecorded"):
        return []
    lines = [tr(language, "change_note_title")]
    skills = note.get("skills") or {}
    if skills.get("added"):
        lines.append(f"  {tr(language, 'change_note_skills_added', names=_names(skills['added'], with_summary=True))}")
    if skills.get("removed"):
        lines.append(f"  {tr(language, 'change_note_skills_removed', names=_names(skills['removed']))}")
    if skills.get("changed"):
        lines.append(f"  {tr(language, 'change_note_skills_changed', names=_names(skills['changed']))}")
    routing = note.get("routing") or []
    if routing:
        lines.append(f"  {tr(language, 'change_note_routing')}")
        for row in routing[:ITEM_LIMIT]:
            if row["added_count"]:
                lines.append(f"    - {tr(language, 'change_note_routing_added', skill=row['label'], phrases=_phrases(row['added'], row['added_count']))}")
            if row["removed_count"]:
                lines.append(f"    - {tr(language, 'change_note_routing_removed', skill=row['label'], phrases=_phrases(row['removed'], row['removed_count']))}")
        if len(routing) > ITEM_LIMIT:
            lines.append(f"    - (+{len(routing) - ITEM_LIMIT})")
    chains = note.get("model_chains") or []
    if chains:
        lines.append(f"  {tr(language, 'change_note_model_chains')}")
        for row in chains:
            if row["override_applies"]:
                lines.append(f"    - {tr(language, 'change_note_chain_overridden', category=row['category'])}")
            else:
                lines.append(f"    - {row['category']}: {_chain(row['before'])} -> {_chain(row['after'])}")
    if note.get("unrecorded"):
        lines.append(f"  {tr(language, 'change_note_unrecorded')}")
    if len(lines) == 1:
        return []
    lines.append(f"  {tr(language, 'change_note_ask')}")
    return lines
