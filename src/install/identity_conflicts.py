"""Every OMH-facing local name that a second visible source also claims.

Doctor already scanned foreign configured skill directories for names matching
OMH's core skills. Three things were missing, and #815 needs all three.

Coverage. The scan saw skill directory names and nothing else. The two other
surfaces carrying an OMH-facing name -- the tools the managed bridge registers
and the lifecycle hooks it subscribes to -- were invisible, so "why did a
familiar command trigger the wrong workflow" had no answer at all when the
competing source was a second local plugin rather than a second skill directory.

Attribution. A collision was one line of text naming the foreign directory. It
never said which side of the contest OMH owns, so an operator could not tell
"OMH installed this twice" from "something else took the name", and those need
opposite repairs. Ownership here is read from the install manifests --
`manifest.json` for skills, `.omh-plugin-manifest.json` for the bridge -- never
inferred from what a path looks like. A directory named `omh` that OMH did not
install is precisely the case the attribution exists to catch, and a path-shaped
guess would call it OMH's own.

Precedence. A contested skill name is resolved from Hermes' documented tier
order (`agent/skill_utils.py` `get_skill_search_roots`: trusted project >
local profile > `skills.create_dir` > `skills.external_dirs`), which Hermes
applies from 0.21.6 (f525ecdb9c); an older host lists only the higher-tier
copy and refuses to load the name as ambiguous, and the wording says so. OMH's managed
directory is always an external one, so a same-name copy in a higher tier
hides the OMH copy, and a copy in another external directory shares its tier,
where Hermes refuses the bare name rather than guess. A foreign skill matches
by its directory name or by the `name` its frontmatter declares, the two
aliases Hermes resolves a bare name by. A foreign skill under only OMH's
canonical name hides nothing: Hermes resolves the two names separately.
Every managed name is covered, under both spellings, and every tier OMH can
see from here is scanned except trusted project directories, which depend on
a session's working directory. A contested tool name is still `unknown`:
Hermes' plugin load order is not readable from here.
`uncontested` is reserved for a scan that actually enumerated every source and
found no contest; an unreadable directory or a Hermes config that could not be
read keeps the answer `unknown`, because "nothing was found" and "nothing was
looked at" are different statements. A shared lifecycle hook also leaves the
report at `unknown` -- the field says whether the scan came back clean, and it
did not -- but no winner is missing there: Hermes lets any number of plugins
subscribe to one event, so the wording for a hook (the next action, the doctor
line, the claim boundary) never speaks of a winner or tells anyone to rename or
remove a subscriber (#1902).

Nothing in this module writes, moves, renames, or deletes. It reads directory
names, `plugin.yaml` declarations, and two manifests, and returns a report. The
repair it names is always the operator's to perform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..plugin_bundle.omh.metadata import PROVIDED_HOOKS, PROVIDED_TOOLS
from ..plugin_bundle.omh.skill_ownership import declared_skill_name, skill_directories
from .plugin_pack import PLUGIN_NAME, PLUGIN_SCHEMA_VERSION, read_plugin_manifest

IDENTITY_CONFLICT_REPORT_SCHEMA_VERSION = "identity_conflict_report/v2"

# The three local surfaces that carry a user-visible OMH name. `command` is the
# tool name Hermes exposes to the model; the issue's vocabulary calls it a
# command and so does this report.
SOURCE_KINDS = ("command", "hook", "skill")
# Two values only. There is no `unknown` ownership: every source counted here
# either appears in an OMH install manifest or does not, and treating "no
# manifest record" as unknown would let an unattributed source quietly borrow
# OMH's side of a contest.
OWNERSHIPS = ("omh_managed", "user_owned")
# `resolved`: every contest found is a skill name, and each one's outcome
# follows from the tier order. `unknown`: a tool or hook contest, or a scan
# that did not finish.
PRECEDENCE_STATES = ("resolved", "uncontested", "unknown")
# Hermes' skill tiers, highest precedence first. `plugin` is not a tier; it
# marks the source of a tool or hook name.
SKILL_TIERS = ("project", "local", "create_dir", "external")
SOURCE_TIERS = (*SKILL_TIERS, "plugin")
# What Hermes does with a contested skill name, per the tier order:
#   shadowed  -- a copy in a higher tier answers the name; the OMH copy is hidden.
#   ambiguous -- a copy in another external directory shares OMH's tier, and
#                Hermes refuses the bare name rather than guess.
#   separate  -- the other skill holds only OMH's canonical name, which Hermes
#                resolves apart from the installed one; nothing is hidden.
#   unknown   -- a tool or hook name, whose resolution is not read here.
RESOLUTIONS = ("ambiguous", "separate", "shadowed", "unknown")
CONFLICT_SEVERITIES = ("blocking", "warning")
REPORT_SEVERITIES = ("blocking", "ok", "warning")

# Whether an unknown precedence is a warning or a blocker is decided by one
# question, and it is not ownership: after the contest, is the OMH side still
# reachable under some name?
#
#   blocking -- `command`. Hermes exposes one tool per name to the model, so a
#     second local plugin declaring `omh_status` means at most one of the two
#     registrations answers and OMH cannot say which. The user-visible result is
#     a capability that is installed and absent at the same time; no phrasing in
#     chat recovers it, and every other doctor check still reports green.
#
#   warning -- `skill` and `hook`. A contested skill name still answers: a
#     higher-tier copy serves it, or within one tier each copy stays loadable
#     by its path, and the report names which outcome applies. A shadowed OMH
#     copy is out of reach by name until the other copy moves, which is the
#     repair the warning names, but a skill still answers the request. Hermes'
#     hook names are lifecycle events that every plugin subscribes to, so an
#     overlap there is a second subscriber rather than a stolen slot. Worth
#     reporting -- it is the honest answer to "why did that trigger the wrong
#     workflow" -- and not worth failing an install over. The repair differs by
#     kind too, which is why `_next_action` composes one sentence per kind
#     present: a skill directory can be renamed, but a hook name is the event
#     itself, and the other subscriber is usually working as intended.
#
# Ownership deliberately does not decide this. AC3 forbids OMH from touching a
# user-owned asset, so keying severity to ownership would paint the ordinary
# case permanently red while telling the operator nothing they can act on.
EXCLUSIVE_NAME_KINDS = frozenset({"command"})

IDENTITY_CONFLICT_CLAIM_BOUNDARY = (
    "This is a read-only scan of locally visible sources: the Hermes home's own skills directory, "
    "skills.create_dir, configured skill directories, installed Hermes plugin declarations, and OMH's "
    "own install manifests. Hermes runtime precedence is not observed. A contested skill name is resolved "
    "from Hermes' documented tier order (project > local > create_dir > external_dirs), which Hermes "
    "applies from 0.21.6; older hosts list only the higher-tier copy and refuse to load the name as "
    "ambiguous. Trusted project directories depend on a session's working directory and are not scanned. "
    "A contested tool name has no resolved winner here. "
    "A shared lifecycle hook has no winner to resolve, since Hermes lets every plugin subscribe "
    "to the same event; the order the subscribers run in is not observed either. Nothing is "
    "renamed, rewritten, or removed by this diagnosis."
)

IDENTITY_CONFLICT_REPORT_KEYS = (
    "claim_boundary",
    "conflicts",
    "next_action",
    "precedence",
    "scanned",
    "schema_version",
    "severity",
    "unreadable",
)
IDENTITY_CONFLICT_KEYS = ("kind", "name", "resolution", "severity", "sources")
IDENTITY_CONFLICT_SOURCE_KEYS = ("location", "ownership", "tier")
IDENTITY_CONFLICT_SCANNED_KEYS = ("plugin_dirs", "skill_dirs")

_OMH_PLUGIN_MANIFEST_PACKAGE = "oh-my-hermes"


def build_identity_conflict_report(
    *,
    skills_dir: Path,
    manifest: dict[str, Any] | None,
    plugins_dir: Path,
    configured_skill_dirs: list[str] | None,
    local_skills_dir: Path | None = None,
    create_dir: Path | None = None,
) -> dict[str, Any]:
    """Report every contested OMH-facing name with all of its visible sources.

    `configured_skill_dirs` is `None` when the Hermes config could not be read
    at all. That is not the same as an empty list: an empty list means Hermes
    was asked and registers no foreign directory, while `None` means the
    question went unanswered, and only the first of those may end in
    `uncontested`. `local_skills_dir` is the Hermes home's own `skills`
    directory and `create_dir` its `skills.create_dir`; either one may be
    absent, which is an ordinary home rather than a source left unread.
    """
    unreadable: list[str] = []
    if configured_skill_dirs is None:
        unreadable.append(
            "Hermes config was not read, so configured skill directories could not be enumerated"
        )
    foreign_dirs = _foreign_skill_dirs(
        skills_dir, configured_skill_dirs or [], unreadable, local_skills_dir=local_skills_dir, create_dir=create_dir
    )
    conflicts = _skill_conflicts(skills_dir, manifest, foreign_dirs, unreadable)
    plugin_sources = _plugin_sources(plugins_dir, unreadable)
    conflicts.extend(_plugin_conflicts(plugin_sources))
    conflicts.sort(key=lambda conflict: (conflict["kind"], conflict["name"]))
    unreadable.sort()
    severity = _severity(conflicts, unreadable)
    return {
        "schema_version": IDENTITY_CONFLICT_REPORT_SCHEMA_VERSION,
        "precedence": _precedence(conflicts, unreadable),
        "severity": severity,
        "conflicts": conflicts,
        "unreadable": unreadable,
        "scanned": {"plugin_dirs": len(plugin_sources), "skill_dirs": len(foreign_dirs)},
        "next_action": _next_action(conflicts, unreadable),
        "claim_boundary": IDENTITY_CONFLICT_CLAIM_BOUNDARY,
    }


def validate_identity_conflict_report(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["identity_conflict_report must be an object"]
    extra = sorted(set(payload) - set(IDENTITY_CONFLICT_REPORT_KEYS))
    if extra:
        errors.append(f"identity_conflict_report has unsupported keys: {extra}")
    missing = sorted(set(IDENTITY_CONFLICT_REPORT_KEYS) - set(payload))
    if missing:
        errors.append(f"identity_conflict_report is missing keys: {missing}")
    if payload.get("schema_version") != IDENTITY_CONFLICT_REPORT_SCHEMA_VERSION:
        errors.append(
            f"identity_conflict_report schema_version must be {IDENTITY_CONFLICT_REPORT_SCHEMA_VERSION}"
        )
    if payload.get("precedence") not in PRECEDENCE_STATES:
        errors.append(f"identity_conflict_report precedence must be one of {list(PRECEDENCE_STATES)}")
    if payload.get("severity") not in REPORT_SEVERITIES:
        errors.append(f"identity_conflict_report severity must be one of {list(REPORT_SEVERITIES)}")
    for key in ("claim_boundary", "next_action"):
        if not isinstance(payload.get(key), str) or not payload.get(key):
            errors.append(f"identity_conflict_report {key} must be a non-empty string")
    errors.extend(_validate_unreadable(payload.get("unreadable")))
    errors.extend(_validate_scanned(payload.get("scanned")))
    errors.extend(_validate_conflicts(payload.get("conflicts")))
    errors.extend(_validate_consistency(payload))
    return errors


def _validate_unreadable(unreadable: Any) -> list[str]:
    if not isinstance(unreadable, list) or not all(isinstance(item, str) for item in unreadable):
        return ["identity_conflict_report unreadable must be a list of strings"]
    if list(unreadable) != sorted(unreadable):
        return ["identity_conflict_report unreadable must be sorted"]
    return []


def _validate_scanned(scanned: Any) -> list[str]:
    if not isinstance(scanned, dict):
        return ["identity_conflict_report scanned must be an object"]
    if sorted(scanned) != sorted(IDENTITY_CONFLICT_SCANNED_KEYS):
        return [f"identity_conflict_report scanned keys must be {list(IDENTITY_CONFLICT_SCANNED_KEYS)}"]
    errors: list[str] = []
    for key in IDENTITY_CONFLICT_SCANNED_KEYS:
        value = scanned.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"identity_conflict_report scanned {key} must be a non-negative integer")
    return errors


def _validate_conflicts(conflicts: Any) -> list[str]:
    if not isinstance(conflicts, list):
        return ["identity_conflict_report conflicts must be a list"]
    errors: list[str] = []
    order: list[tuple[str, str]] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            errors.append("identity_conflict_report conflict must be an object")
            continue
        if sorted(conflict) != sorted(IDENTITY_CONFLICT_KEYS):
            errors.append(f"identity_conflict_report conflict keys must be {list(IDENTITY_CONFLICT_KEYS)}")
            continue
        kind = conflict.get("kind")
        if kind not in SOURCE_KINDS:
            errors.append(f"identity_conflict_report conflict kind must be one of {list(SOURCE_KINDS)}")
        if not isinstance(conflict.get("name"), str) or not conflict.get("name"):
            errors.append("identity_conflict_report conflict name must be a non-empty string")
        resolution = conflict.get("resolution")
        if resolution not in RESOLUTIONS:
            errors.append(f"identity_conflict_report conflict resolution must be one of {list(RESOLUTIONS)}")
        elif (resolution == "unknown") != (kind != "skill"):
            errors.append(f"identity_conflict_report conflict resolution {resolution!r} does not fit kind {kind!r}")
        if conflict.get("severity") not in CONFLICT_SEVERITIES:
            errors.append(
                f"identity_conflict_report conflict severity must be one of {list(CONFLICT_SEVERITIES)}"
            )
        elif conflict.get("severity") != ("blocking" if kind in EXCLUSIVE_NAME_KINDS else "warning"):
            errors.append(f"identity_conflict_report conflict severity does not match the rule for kind {kind!r}")
        errors.extend(_validate_sources(conflict.get("sources")))
        order.append((str(kind), str(conflict.get("name", ""))))
    if order != sorted(order):
        errors.append("identity_conflict_report conflicts must be sorted by (kind, name)")
    return errors


def _validate_sources(sources: Any) -> list[str]:
    if not isinstance(sources, list) or len(sources) < 2:
        # One source is not a contest. A conflict entry naming a single source
        # would claim a competitor the scan never found.
        return ["identity_conflict_report conflict sources must list at least two sources"]
    errors: list[str] = []
    ownerships: set[str] = set()
    locations: list[str] = []
    for source in sources:
        if not isinstance(source, dict) or sorted(source) != sorted(IDENTITY_CONFLICT_SOURCE_KEYS):
            errors.append(f"identity_conflict_report source keys must be {list(IDENTITY_CONFLICT_SOURCE_KEYS)}")
            continue
        if source.get("ownership") not in OWNERSHIPS:
            errors.append(f"identity_conflict_report source ownership must be one of {list(OWNERSHIPS)}")
        if source.get("tier") not in SOURCE_TIERS:
            errors.append(f"identity_conflict_report source tier must be one of {list(SOURCE_TIERS)}")
        if not isinstance(source.get("location"), str) or not source.get("location"):
            errors.append("identity_conflict_report source location must be a non-empty string")
        ownerships.add(str(source.get("ownership")))
        locations.append(str(source.get("location")))
    if locations != sorted(locations):
        errors.append("identity_conflict_report conflict sources must be sorted by location")
    if len(set(locations)) != len(locations):
        errors.append("identity_conflict_report conflict sources must not repeat a location")
    if not errors and ownerships != set(OWNERSHIPS):
        # A managed asset colliding with itself is not a conflict, and neither
        # is a contest OMH takes no part in.
        errors.append("identity_conflict_report conflict must name one omh_managed and one user_owned source")
    return errors


def _validate_consistency(payload: dict[str, Any]) -> list[str]:
    conflicts = payload.get("conflicts")
    unreadable = payload.get("unreadable")
    if not isinstance(conflicts, list) or not isinstance(unreadable, list):
        return []
    errors: list[str] = []
    expected_severity = _severity(
        [conflict for conflict in conflicts if isinstance(conflict, dict)],
        [str(item) for item in unreadable],
    )
    if payload.get("severity") != expected_severity:
        errors.append(f"identity_conflict_report severity must be {expected_severity} for these conflicts")
    expected_precedence = _precedence(
        [conflict for conflict in conflicts if isinstance(conflict, dict)],
        [str(item) for item in unreadable],
    )
    if payload.get("precedence") != expected_precedence:
        errors.append(f"identity_conflict_report precedence must be {expected_precedence} for these conflicts")
    return errors


def _precedence(conflicts: list[dict[str, Any]], unreadable: list[str]) -> str:
    if unreadable or any(conflict.get("resolution") == "unknown" for conflict in conflicts):
        return "unknown"
    return "resolved" if conflicts else "uncontested"


def _severity(conflicts: list[dict[str, Any]], unreadable: list[str]) -> str:
    if any(conflict.get("severity") == "blocking" for conflict in conflicts):
        return "blocking"
    if conflicts or unreadable:
        return "warning"
    return "ok"


_NEXT_ACTION_BY_KIND = {
    "command": (
        "Rename or uninstall the competing tool name in the local plugin OMH did not install, then rerun "
        "`omh doctor`; OMH will not edit or remove a plugin it does not own."
    ),
    "skill": (
        "Rename or unregister whichever competing skill directory you own, then rerun `omh doctor`; OMH does "
        "not rename or delete assets it did not install. A copy in the Hermes home's own skills directory, "
        "such as one a hub install put there, never receives OMH's updates; on Hermes 0.21.6 and later it "
        "outranks OMH's copy, and older hosts refuse the name as ambiguous."
    ),
    # No rename, no removal. A hook name is the lifecycle event itself, and the
    # other plugin subscribing to it is ordinary Hermes behaviour (#1902).
    "hook": (
        "Hermes lets any number of plugins subscribe to the same lifecycle event, so a shared hook is a "
        "second subscriber and nothing needs to change. Look at the other plugin's callback only if its "
        "effect clashes with OMH's, for example when both rewrite the same tool call or both inject context, "
        "and keep a working subscriber in place rather than removing it to clear this warning."
    ),
}


def _next_action(conflicts: list[dict[str, Any]], unreadable: list[str]) -> str:
    # One sentence per kind present, in `SOURCE_KINDS` order, so the blocking
    # `command` repair always leads and a hook never inherits a repair written
    # for an exclusive name.
    kinds = {conflict.get("kind") for conflict in conflicts}
    actions = [_NEXT_ACTION_BY_KIND[kind] for kind in SOURCE_KINDS if kind in kinds]
    if actions:
        return " ".join(actions)
    if unreadable:
        return "Repair or re-read the reported source, then rerun `omh doctor` for a complete scan."
    return (
        "No conflict repair needed. Hermes runtime precedence stays unobserved until a host observation "
        "records it."
    )


def _foreign_skill_dirs(
    skills_dir: Path,
    configured_dirs: list[str],
    unreadable: list[str],
    *,
    local_skills_dir: Path | None = None,
    create_dir: Path | None = None,
) -> list[tuple[str, Path]]:
    """`(tier, directory)` for every skill root other than OMH's own, highest tier first.

    A directory named by two tiers counts once, in the higher one, the way
    Hermes de-duplicates its roots. The local and create_dir roots are optional
    on an ordinary home, so a missing one is skipped; a configured external
    directory that is missing stays reported, as it always was.
    """
    try:
        managed = skills_dir.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as error:
        managed = None
        unreadable.append(f"managed skill directory {skills_dir}: {error}")
    candidates: list[tuple[str, str, bool]] = []
    if local_skills_dir is not None:
        candidates.append(("local", str(local_skills_dir), False))
    if create_dir is not None:
        candidates.append(("create_dir", str(create_dir), False))
    candidates.extend(("external", raw_dir, True) for raw_dir in configured_dirs)
    seen: set[Path] = set()
    foreign: list[tuple[str, Path]] = []
    for tier, raw_dir, required in candidates:
        try:
            candidate = Path(raw_dir).expanduser().resolve(strict=False)
        except (OSError, RuntimeError) as error:
            unreadable.append(f"{raw_dir}: {error}")
            continue
        if candidate == managed or candidate in seen:
            continue
        if not required and not candidate.is_dir():
            continue
        seen.add(candidate)
        foreign.append((tier, candidate))
    return foreign


def _managed_skill_locations(skills_dir: Path, manifest: dict[str, Any] | None) -> dict[str, tuple[str, bool]]:
    """Contested skill name -> (the installed location OMH owns, whether the name is its label).

    Every skill the manifest records is covered, under both its installed
    directory name -- the name Hermes resolves, which is also the skill's
    frontmatter `name` -- and its canonical catalog name, which the router and
    the operator say. The label is recorded first so a canonical name that
    happens to equal another skill's label never displaces it.

    Rule: only a label match is a contest inside Hermes' resolution, so only
    a label match can be `shadowed` or `ambiguous`. Hermes resolves a skill by
    its directory name, its frontmatter name, or its path under a root, and
    for an OMH skill the first two are the label; a foreign skill holding the
    canonical name (`doctor` beside `omh-doctor`) answers a different name and
    hides nothing. That match is still reported, as `separate`, because the
    router and the operator say the canonical name, but calling it shadowing
    would claim an outcome the tier order does not produce.

    A managed location is recorded only when the manifest accounts for it. A
    file sitting under `skills_dir` that no record explains is not something OMH
    installed, and letting it stand in for OMH's side would fabricate a
    competitor.
    """
    records = [
        (str(record.get("name", "")), Path(str(record.get("path", ""))).parent)
        for record in (manifest or {}).get("skills", [])
        if isinstance(record, dict) and record.get("path")
    ]
    locations: dict[str, tuple[str, bool]] = {}
    for _name, directory in records:
        if directory.name:
            locations.setdefault(directory.name, (str(skills_dir / directory), True))
    for name, directory in records:
        if name:
            locations.setdefault(name, (str(skills_dir / directory), False))
    return locations


def _skill_conflicts(
    skills_dir: Path,
    manifest: dict[str, Any] | None,
    foreign_dirs: list[tuple[str, Path]],
    unreadable: list[str],
) -> list[dict[str, Any]]:
    managed = _managed_skill_locations(skills_dir, manifest)
    managed_names = set(managed)
    claims: dict[str, list[tuple[str, str]]] = {}
    for tier, foreign_dir in foreign_dirs:
        if not foreign_dir.is_dir():
            unreadable.append(f"{foreign_dir}: not a readable directory")
            continue
        # A directory counts only when it holds a SKILL.md, the way Hermes
        # finds a skill; symlinked directories are not followed, so an alias
        # of one already scanned does not report the same contest twice.
        for directory in skill_directories(foreign_dir, unreadable):
            for alias in sorted({directory.name, declared_skill_name(directory)} & managed_names):
                claims.setdefault(alias, []).append((str(directory), tier))
    conflicts: list[dict[str, Any]] = []
    for name in sorted(claims):
        location, is_label = managed[name]
        tiers = [tier for _location, tier in claims[name]]
        if not is_label:
            resolution = "separate"
        elif any(tier != "external" for tier in tiers):
            resolution = "shadowed"
        else:
            resolution = "ambiguous"
        sources = [(location, "omh_managed", "external")]
        sources.extend((foreign, "user_owned", tier) for foreign, tier in claims[name])
        conflicts.append(_conflict("skill", name, sources, resolution))
    return conflicts


def skill_conflict_outcome(conflict: dict[str, Any]) -> str:
    """One sentence for what Hermes does with a contested skill name, from the tier order."""
    sources = conflict["sources"]
    omh = next(source for source in sources if source["ownership"] == "omh_managed")
    others = [source for source in sources if source["ownership"] != "omh_managed"]
    if conflict["resolution"] == "shadowed":
        top = min(SKILL_TIERS.index(source["tier"]) for source in others)
        winners = [source for source in others if SKILL_TIERS.index(source["tier"]) == top]
        copies = "copy" if len(winners) == 1 else "copies"
        listed = ", ".join(source["location"] for source in winners)
        return f"OMH copy at {omh['location']} shadowed by {SKILL_TIERS[top]} {copies} at {listed}"
    listed = ", ".join(f"{source['tier']} copy at {source['location']}" for source in others)
    if conflict["resolution"] == "ambiguous":
        return f"OMH copy at {omh['location']} shares the external tier with the {listed}, so Hermes refuses the bare name"
    return (
        f"OMH's catalog name for the skill at {omh['location']} also names the {listed}; "
        "Hermes resolves the two names separately"
    )


def _plugin_sources(plugins_dir: Path, unreadable: list[str]) -> list[dict[str, Any]]:
    """Every locally installed Hermes plugin that declares tools or hooks."""
    try:
        children = sorted(plugins_dir.iterdir(), key=lambda item: item.name)
    except FileNotFoundError:
        # No plugins directory is the ordinary pre-install state, not a source
        # the scan failed to read.
        return []
    except OSError as error:
        unreadable.append(f"{plugins_dir}: {error}")
        return []
    sources: list[dict[str, Any]] = []
    for child in children:
        if child.is_symlink() or not child.is_dir():
            continue
        descriptor = child / "plugin.yaml"
        if not descriptor.is_file():
            continue
        try:
            text = descriptor.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            unreadable.append(f"{descriptor}: {error}")
            continue
        sources.append(
            {
                "location": str(child),
                "ownership": _plugin_ownership(child),
                "command": _declared_names(text, "provides_tools"),
                "hook": _declared_names(text, "provides_hooks"),
            }
        )
    return sources


def _plugin_ownership(plugin_dir: Path) -> str:
    """Whether OMH's installer wrote this bundle, according to its install manifest.

    Never inferred from the directory name. `~/.hermes/plugins/omh` created by
    anyone else is a user-owned source claiming OMH's names -- exactly the
    conflict this report exists to surface -- and a name-shaped guess would
    silently classify it as OMH's own and report nothing.
    """
    manifest = read_plugin_manifest(plugin_dir)
    if not isinstance(manifest, dict):
        return "user_owned"
    if str(manifest.get("schema_version", "")) != PLUGIN_SCHEMA_VERSION:
        return "user_owned"
    if str(manifest.get("plugin_name", "")) != PLUGIN_NAME:
        return "user_owned"
    if str(manifest.get("package", "")) != _OMH_PLUGIN_MANIFEST_PACKAGE:
        return "user_owned"
    return "omh_managed"


def _plugin_conflicts(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for kind, vocabulary in (("command", PROVIDED_TOOLS), ("hook", PROVIDED_HOOKS)):
        for name in sorted(frozenset(vocabulary)):
            claiming = [source for source in sources if name in source[kind]]
            ownerships = {str(source["ownership"]) for source in claiming}
            # Both sides are required. OMH's own bridge declaring a name it owns
            # is not a conflict with itself, and a contest between two foreign
            # plugins over a name OMH does not currently register is not OMH's
            # to report.
            if ownerships != set(OWNERSHIPS):
                continue
            conflicts.append(
                _conflict(
                    kind,
                    name,
                    [(str(source["location"]), str(source["ownership"]), "plugin") for source in claiming],
                    "unknown",
                )
            )
    return conflicts


def _conflict(kind: str, name: str, sources: list[tuple[str, str, str]], resolution: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "name": name,
        "resolution": resolution,
        "severity": "blocking" if kind in EXCLUSIVE_NAME_KINDS else "warning",
        "sources": [
            {"location": location, "ownership": ownership, "tier": tier}
            for location, ownership, tier in sorted(sources)
        ],
    }


def _declared_names(text: str, key: str) -> list[str]:
    """Values under a top-level `key:` list in a Hermes `plugin.yaml`.

    A hand-rolled reader for the same reason `install.config_adapter` has one:
    the core carries no YAML dependency. It accepts the two shapes a plugin
    descriptor uses in practice -- a block list and an inline list -- and reads
    nothing else, so an unfamiliar shape yields no names rather than a guess.
    """
    names: list[str] = []
    collecting = False
    for line in text.splitlines():
        stripped = line.strip()
        if not line.startswith(" ") and stripped:
            head, separator, rest = stripped.partition(":")
            if not separator or head.strip() != key:
                collecting = False
                continue
            inline = rest.strip()
            collecting = not inline
            if inline:
                names.extend(_inline_list(inline))
            continue
        if not collecting or not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            names.append(stripped[2:].strip().strip("\"'"))
        else:
            collecting = False
    return [name for name in names if name]


def _inline_list(value: str) -> list[str]:
    if not (value.startswith("[") and value.endswith("]")):
        return []
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [item.strip().strip("\"'") for item in inner.split(",")]
