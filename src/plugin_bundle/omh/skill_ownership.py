"""Refuse a Hermes skill change that would land on an OMH-managed skill.

Hermes' background self-improvement review may patch, edit, or write files
into any skill it loaded, skills under `skills.external_dirs` included. Upstream
guards a delete or archive only inside that review fork
(`tools/skill_manager_guards.py::_background_review_delete_guard`); a
foreground `skill_manage` delete is not guarded at all. An OMH skill is
rendered by the installer and recorded in `<omh_home>/manifest.json`, so a
change made in place either holds that skill back from every later OMH update
or stays behind in a retired generation Hermes no longer loads, and a delete
lasts only until the next update renders the skill again or reports it
missing.

`pre_tool_call` therefore blocks `skill_manage` edit, patch, write_file,
remove_file and delete when the named skill resolves to one the manifest
records, and the message says where a lesson goes instead: a separately named
companion skill, created where `skill_manage` creates skills, or OMH's
workflow-learning lane. Upstream rejected read-only external directories
because a bare refusal breeds silent local duplicates; that pointer is what
keeps this refusal from doing the same.

The hook reaches the background-review fork too. `hermes_cli/plugins.py`
`_get_pre_tool_call_directive_details` checks the fork's thread whitelist and
then invokes `pre_tool_call` for every tool the whitelist admits, and
`skill_manage` is one of them. The arguments arrive parsed from the model's
JSON but not yet coerced to the tool schema (`agent/tool_executor.py`
`_parse_tool_call`), so an `operations` array the model sent as a JSON string
is decoded here.

Resolution follows `tools/skill_manager_tool.py::_find_skill`, which serves a
bare name from the profile's own skills directory, then `skills.create_dir`,
then the external directories. A name either of the first two also holds
edits that copy, not OMH's, so it passes through here.

Rule: only the installed directory name is a target, never the canonical
catalog name and never a prefix. The installed name is also every OMH
skill's frontmatter `name`, and it is the only one of OMH's names that
`_find_skill` resolves to the managed directory; `skill_manage` on a
canonical name such as `doctor` edits some other skill or nothing, and a
user skill called `omh-notes` is not OMH's. A categorized `category/name`
only resolves inside the local directory, so it never reaches an OMH skill.

Anything this cannot read -- no manifest, an unreadable one, a `config.yaml`
that exists but cannot be read, a `skills:` block in a shape this scan does
not parse, or a filesystem check that raises -- lets the call through: the
installer and `omh doctor` still report the change, and a refusal aimed at a
copy OMH does not own would be the worse error.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path, PurePosixPath
from typing import Final

SKILL_MANAGE_TOOL: Final = "skill_manage"
# The `skill_manage` actions that change or remove an existing skill. `create`
# refuses a name that already resolves, so it never reaches an OMH skill.
GUARDED_ACTIONS: Final = frozenset({"edit", "patch", "write_file", "remove_file", "delete"})
MANIFEST_NAME: Final = "manifest.json"
WORKFLOW_LEARNING_SKILL: Final = "workflow-learning"
# Directory names Hermes' skill scanners never descend into
# (`agent/skill_utils.py` `EXCLUDED_SKILL_DIRS`), and the support directories
# inside a skill package, which hold files rather than skills.
SKILL_SCAN_EXCLUDED_DIRS: Final = frozenset(
    {
        ".git", ".github", ".hub", ".archive", ".curator_backups", ".locks",
        ".venv", "venv", "node_modules", "site-packages", "__pycache__",
        ".tox", ".nox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    }
)
SKILL_SUPPORT_DIRS: Final = frozenset({"references", "templates", "assets", "scripts"})
_YAML_NULLS: Final = frozenset({"", "~", "null", "Null", "NULL"})
# How much of a SKILL.md is read for its frontmatter, as Hermes' own
# `_read_frontmatter_name` does.
_FRONTMATTER_READ_CHARS: Final = 4000


def skill_directories(root: Path, errors: list[str] | None = None) -> list[Path]:
    """Every directory under `root` holding a SKILL.md, the way Hermes finds skills.

    Symlinked directories are not followed, so an alias of a directory already
    walked is not counted twice. A directory that cannot be listed, or an entry
    that cannot be checked, is skipped and, when `errors` is given, named
    there: "nothing found" and "nothing looked at" are different answers.
    """
    found: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except FileNotFoundError:
            continue
        except OSError as error:
            if errors is not None:
                errors.append(f"{directory}: {error}")
            continue
        try:
            holds_skill = directory != root and any(
                child.name == "SKILL.md" and child.is_file() for child in children
            )
        except OSError as error:
            if errors is not None:
                errors.append(f"{directory}: {error}")
            continue
        if holds_skill:
            found.append(directory)
        for child in reversed(children):
            if child.name in SKILL_SCAN_EXCLUDED_DIRS or (holds_skill and child.name in SKILL_SUPPORT_DIRS):
                continue
            try:
                descend = not child.is_symlink() and child.is_dir()
            except OSError as error:
                if errors is not None:
                    errors.append(f"{child}: {error}")
                continue
            if descend:
                pending.append(child)
    return sorted(found)


def yaml_scalar(value: str) -> str | None:
    """The text of one YAML scalar written after `key:`, or None when this reader cannot say.

    A quoted value runs to its matching close quote, so a comment after it is
    not part of it (`"/x"  # shared` is `/x`); `''` inside single quotes is one
    quote. Inside double quotes only `\\\\` and `\\"` are read; any other escape
    (`\\n`, `\\u00e9`, a Windows path written with single backslashes) answers
    None rather than a mangled string, as does a quote that never closes or
    one followed by anything but a comment. An
    unquoted value stops at `#`. Null spellings are returned as written; see
    `yaml_null` for whether they mean null.
    """
    stripped = value.strip()
    if stripped[:1] not in {"'", '"'}:
        return stripped.split("#", 1)[0].strip()
    quote = stripped[0]
    chars: list[str] = []
    index = 1
    while index < len(stripped):
        char = stripped[index]
        if quote == '"' and char == "\\":
            escaped = stripped[index + 1 : index + 2]
            if escaped not in {"\\", '"'}:
                return None
            chars.append(escaped)
            index += 2
            continue
        if char == quote:
            if quote == "'" and stripped[index + 1 : index + 2] == "'":
                chars.append("'")
                index += 2
                continue
            # Only a comment may follow the closing quote.
            trailing = stripped[index + 1 :]
            if trailing.strip() and not (trailing[:1].isspace() and trailing.lstrip().startswith("#")):
                return None
            return "".join(chars)
        chars.append(char)
        index += 1
    return None


def yaml_null(value: str) -> bool:
    """Whether the text after `key:` is a YAML null: an unquoted `~`, `null`, or nothing.

    A quoted `"~"` is a string, not a null.
    """
    stripped = value.strip()
    return stripped[:1] not in {"'", '"'} and stripped.split("#", 1)[0].strip() in _YAML_NULLS


def configured_create_dir(hermes_home: Path) -> tuple[Path | None, bool]:
    """`skills.create_dir` as Hermes resolves it, and whether the config could be read.

    A text scan in the spirit of the bundle's other `config.yaml` readers: only
    a direct child of the top-level `skills:` block counts, the last one wins
    as in YAML, and a null or empty value is unset. Like
    `agent/skill_utils.py::get_skill_create_dir`, `~` and `${VAR}` expand and a
    relative path is anchored at the Hermes home. A missing `config.yaml` is
    an ordinary home with nothing configured; one that exists but cannot be
    read, an inline `skills:` value, or a quote that never closes answers
    "not readable". A leading byte-order mark is not part of the first key.
    """
    try:
        text = (hermes_home / "config.yaml").read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None, True
    except (OSError, UnicodeDecodeError):
        return None, False
    raw: str | None = None
    in_skills = False
    child_indent: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line[0].isspace():
            key, _separator, rest = stripped.partition(":")
            in_skills = key.strip() == "skills"
            child_indent = None
            if in_skills and not (yaml_null(rest) or rest.split("#", 1)[0].strip() == "{}"):
                return None, False
            continue
        if not in_skills:
            continue
        indent = len(line) - len(line.lstrip())
        if child_indent is None:
            child_indent = indent
        key, separator, rest = stripped.partition(":")
        if indent == child_indent and separator and key.strip() == "create_dir":
            raw = rest
    if raw is None or yaml_null(raw):
        return None, True
    value = yaml_scalar(raw)
    if value is None:
        return None, False
    if not value.strip():
        return None, True
    path = Path(os.path.expanduser(os.path.expandvars(value.strip())))
    return (path if path.is_absolute() else hermes_home / path), True


def declared_skill_name(skill_dir: Path) -> str:
    """The frontmatter `name` a SKILL.md declares, or "" when it declares none or cannot be read."""
    try:
        with (skill_dir / "SKILL.md").open(encoding="utf-8-sig", errors="replace") as handle:
            text = handle.read(_FRONTMATTER_READ_CHARS)
    except OSError:
        return ""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, separator, rest = line.partition(":")
        if separator and not line[:1].isspace() and key.strip() == "name":
            return (yaml_scalar(rest) or "").strip()
    return ""


def managed_skill_files(omh_home: str | Path) -> dict[str, tuple[str, Path]]:
    """Installed directory name -> (canonical name, SKILL.md path) for each skill the manifest records."""
    try:
        payload = json.loads((Path(omh_home) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or not isinstance(payload.get("skills"), list):
        return {}
    recorded_dir = payload.get("skills_dir")
    # A manifest without a recorded directory predates generations and can
    # only describe the one pack directory that existed then.
    skills_dir = Path(recorded_dir) if isinstance(recorded_dir, str) and recorded_dir else Path(omh_home) / "skills"
    files: dict[str, tuple[str, Path]] = {}
    for record in payload["skills"]:
        if not isinstance(record, dict):
            continue
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            continue
        label = PurePosixPath(relative).parent.name
        if not label:
            continue
        name = record.get("name")
        files[label] = (name if isinstance(name, str) and name else label, skills_dir / relative)
    return files


def skill_manage_targets(tool_input: object) -> list[tuple[str, str]]:
    """(action, name) for every operation one `skill_manage` call carries.

    The advertised shape is an `operations` array whose ops name their own
    target, falling back to a top-level `name`; the older flat shape (one
    top-level `action` and `name`) is still accepted by the tool, so it is
    read too. An `operations` value sent as a JSON string is decoded, because
    the hook sees the arguments before Hermes coerces them to the schema.
    """
    if not isinstance(tool_input, Mapping):
        return []
    default_name = tool_input.get("name")
    operations = tool_input.get("operations")
    if isinstance(operations, str) and operations.strip()[:1] in {"[", "{"}:
        try:
            operations = json.loads(operations)
        except ValueError:
            return []
    if operations is None:
        operations = [tool_input]
    elif isinstance(operations, Mapping):
        operations = [operations]
    elif not isinstance(operations, list):
        return []
    targets: list[tuple[str, str]] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        action = operation.get("action")
        name = operation.get("name") or default_name
        if isinstance(action, str) and isinstance(name, str) and name.strip():
            targets.append((action.strip(), name.strip()))
    return targets


def managed_skill_edit_directive(
    *, tool_name: object, tool_input: object, omh_home: str, hermes_home: str
) -> dict[str, str] | None:
    """A `block` naming where a lesson goes, when this call changes an OMH-managed skill; else None."""
    if str(tool_name or "") != SKILL_MANAGE_TOOL:
        return None
    targets = [(action, name) for action, name in skill_manage_targets(tool_input) if action in GUARDED_ACTIONS]
    if not targets:
        return None
    managed = managed_skill_files(omh_home)
    if not managed:
        return None
    try:
        return _managed_target_directive(targets, managed, Path(hermes_home))
    except OSError:
        # A stat that raises (EACCES on a directory the walk reached) fails
        # open here rather than escaping the hook, which would also skip every
        # OMH guard `pre_tool_call` runs after this one.
        return None


def _managed_target_directive(
    targets: list[tuple[str, str]], managed: dict[str, tuple[str, Path]], hermes_home: Path
) -> dict[str, str] | None:
    higher_tier_names: set[str] | None = None
    create_dir: Path | None = None
    for action, name in targets:
        entry = managed.get(name)
        if entry is None or not entry[1].is_file():
            continue
        if higher_tier_names is None:
            # The two tiers `_find_skill` searches before the external ones.
            create_dir, readable = configured_create_dir(hermes_home)
            if not readable:
                return None
            roots = [hermes_home / "skills", *([create_dir] if create_dir is not None else [])]
            higher_tier_names = {directory.name for root in roots for directory in skill_directories(root)}
        if name in higher_tier_names:
            continue
        workflow_learning = next(
            (label for label, (canonical, _path) in managed.items() if canonical == WORKFLOW_LEARNING_SKILL), ""
        )
        create_target = str(create_dir) if create_dir is not None else "this profile's own skills directory"
        return {
            "action": "block",
            "message": managed_skill_edit_message(name, action, workflow_learning, create_target=create_target),
        }
    return None


def managed_skill_edit_message(
    label: str,
    action: str,
    workflow_learning: str = "",
    *,
    create_target: str = "this profile's own skills directory",
) -> str:
    """What the model reads instead of the change: why it was refused, and where a lesson goes."""
    lane = "OMH's workflow-learning lane"
    if workflow_learning:
        lane += f" (skill_view name='{workflow_learning}')"
    # Every label OMH renders starts with one of these, so the suggested
    # companion leaves it off: a name wearing it reads as OMH's to anyone who
    # goes by the name.
    companion = f"{label.removeprefix('omh-').removeprefix('ulw-')}-notes"
    if action == "delete":
        consequence = "so deleting it here does not last: the next update renders it again or reports it missing"
    else:
        article = "an" if action.startswith(("a", "e", "i", "o", "u")) else "a"
        consequence = (
            f"so {article} {action} here would hold it back from later updates or be left behind in a retired install"
        )
    return (
        f"Refused: '{label}' is an oh-my-hermes skill that OMH re-renders from its catalog on every update, "
        f"{consequence}. Keep any lesson in a skill OMH does not manage: skill_manage action=create with a "
        f"different name, such as '{companion}', holding a short companion that names '{label}' and the case it "
        f"adds rather than a copy of it (it lands in {create_target}), or record it through {lane}. "
        f"Do not retry this {action} on '{label}'."
    )


__all__ = [
    "GUARDED_ACTIONS",
    "SKILL_MANAGE_TOOL",
    "configured_create_dir",
    "declared_skill_name",
    "managed_skill_edit_directive",
    "managed_skill_edit_message",
    "managed_skill_files",
    "skill_directories",
    "skill_manage_targets",
    "yaml_null",
    "yaml_scalar",
]
