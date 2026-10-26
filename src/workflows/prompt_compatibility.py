from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Literal, TypeAlias, assert_never

from ..system.metadata_safety import is_sensitive_metadata_text
from .prompt_source_io import read_regular_prompt_source


PROMPT_COMPATIBILITY_AUDIT_SCHEMA_VERSION = "prompt_compatibility_audit/v1"
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
PromptFormat: TypeAlias = Literal["yaml_frontmatter", "yaml", "toml", "raw_text", "unsupported"]
MAX_PROMPT_AUDIT_SOURCES = 20
MAX_PROMPT_AUDIT_SOURCE_BYTES = 131_072
_TEXT_SUFFIXES = frozenset({"", ".md", ".markdown", ".prompt", ".txt"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_COMMAND_NAME = re.compile(r"[^a-z0-9]+")
_YAML_NAME = re.compile(r"^name\s*:\s*[\"']?([A-Za-z0-9 _.-]{1,80})[\"']?\s*$", re.MULTILINE)
_TOML_NAME = re.compile(r"^name\s*=\s*[\"']([A-Za-z0-9 _.-]{1,80})[\"']\s*$", re.MULTILINE)
_PROMPT_INJECTION = re.compile(r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior)\s+instructions\b", re.IGNORECASE)


class PromptCompatibilityAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _PromptSourceAudit:
    path: str
    sha256: str
    byte_count: int
    format_name: PromptFormat
    proposed_command: str
    argument_syntax: tuple[str, ...]
    trust_reasons: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "format": self.format_name,
            "proposed_command": self.proposed_command,
            "argument_syntax": list(self.argument_syntax),
            "trust_review": {"required": True, "reasons": list(self.trust_reasons)},
            "status": "unsupported" if self.format_name == "unsupported" else "review_required",
        }


def audit_prompt_compatibility(
    prompt_paths: tuple[Path, ...], *, existing_commands: tuple[str, ...] = ()
) -> JsonObject:
    paths = _validated_paths(prompt_paths)
    normalized_existing = _normalized_existing_commands(existing_commands)
    sources = tuple(_audit_source(path) for path in paths)
    collisions = _collisions(sources, normalized_existing)
    syntax_counts = Counter(syntax for source in sources for syntax in source.argument_syntax)
    compatible_count = sum(source.format_name != "unsupported" for source in sources)
    return {
        "schema_version": PROMPT_COMPATIBILITY_AUDIT_SCHEMA_VERSION,
        "sources": {
            "provided_count": len(sources),
            "auto_discovery": False,
            "files": [source.to_dict() for source in sources],
        },
        "collisions": collisions,
        "summary": {
            "file_count": len(sources),
            "compatible_file_count": compatible_count,
            "unsupported_file_count": len(sources) - compatible_count,
            "candidate_command_count": len({source.proposed_command for source in sources if source.proposed_command}),
            "collision_count": len(collisions),
            "argument_syntax_counts": dict(sorted(syntax_counts.items())),
        },
        "not_observed": {
            "source_directory_discovery": {"status": "not_observed"},
            "slash_command_registration": {"status": "not_observed"},
            "prompt_mutation": {"status": "not_observed"},
            "command_activation": {"status": "not_observed"},
            "prompt_execution": {"status": "not_observed"},
        },
        "claim_boundary": (
            "The audit reads only explicitly named regular local files and returns bounded compatibility metadata. "
            "It is not source-directory discovery, prompt import, prompt mutation, slash command registration, "
            "command activation, prompt execution, or imported prompt trust evidence."
        ),
    }


def _validated_paths(prompt_paths: tuple[Path, ...]) -> tuple[Path, ...]:
    if not 1 <= len(prompt_paths) <= MAX_PROMPT_AUDIT_SOURCES:
        raise PromptCompatibilityAuditError(f"prompt compatibility audit requires 1 to {MAX_PROMPT_AUDIT_SOURCES} paths")
    paths: list[Path] = []
    for path in prompt_paths:
        absolute = Path(os.path.abspath(path.expanduser()))
        if is_sensitive_metadata_text(str(absolute)):
            raise PromptCompatibilityAuditError("prompt audit source path must be safe metadata")
        if absolute.is_symlink():
            raise PromptCompatibilityAuditError("prompt audit source must not be a symlink")
        if any(component.is_symlink() for component in absolute.parents):
            raise PromptCompatibilityAuditError("prompt audit source path must not contain a symlink")
        if not absolute.is_file():
            raise PromptCompatibilityAuditError("prompt audit source must be a regular file")
        if absolute in paths:
            raise PromptCompatibilityAuditError("prompt audit source paths must be unique")
        paths.append(absolute)
    return tuple(sorted(paths))


def _normalized_existing_commands(values: tuple[str, ...]) -> frozenset[str]:
    if len(values) > 128:
        raise PromptCompatibilityAuditError("prompt compatibility audit accepts at most 128 existing command names")
    commands: set[str] = set()
    for value in values:
        if is_sensitive_metadata_text(value):
            raise PromptCompatibilityAuditError("existing command name must be safe metadata")
        command = _command_name(value)
        if not command:
            raise PromptCompatibilityAuditError("existing command name must contain letters or digits")
        commands.add(command)
    return frozenset(commands)


def _audit_source(path: Path) -> _PromptSourceAudit:
    content = read_regular_prompt_source(path, maximum_bytes=MAX_PROMPT_AUDIT_SOURCE_BYTES)
    digest = hashlib.sha256(content).hexdigest()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = ""
    format_name = _format_name(path, text)
    command_seed = _frontmatter_name(text, format_name) or path.stem
    if is_sensitive_metadata_text(command_seed):
        raise PromptCompatibilityAuditError("prompt audit candidate command must be safe metadata")
    proposed_command = _command_name(command_seed) if format_name != "unsupported" else ""
    syntax = _argument_syntax(text) if format_name != "unsupported" else ()
    return _PromptSourceAudit(
        str(path),
        digest,
        len(content),
        format_name,
        proposed_command,
        syntax,
        _trust_reasons(text, format_name),
    )


def _format_name(path: Path, text: str) -> PromptFormat:
    suffix = path.suffix.lower()
    if not text:
        return "unsupported"
    if suffix == ".toml":
        return "toml"
    if _has_yaml_frontmatter(text):
        return "yaml_frontmatter"
    if suffix in _YAML_SUFFIXES:
        return "yaml"
    if suffix in _TEXT_SUFFIXES:
        return "raw_text"
    return "unsupported"


def _has_yaml_frontmatter(text: str) -> bool:
    lines = text.splitlines()
    return len(lines) > 1 and lines[0] == "---" and "---" in lines[1:64]


def _frontmatter_name(text: str, format_name: PromptFormat) -> str:
    head = "\n".join(text.splitlines()[:64])
    match format_name:
        case "yaml_frontmatter" | "yaml":
            name = _YAML_NAME.search(head)
        case "toml":
            name = _TOML_NAME.search(head)
        case "raw_text" | "unsupported":
            name = None
        case unreachable:
            assert_never(unreachable)
    return name.group(1).strip() if name else ""


def _command_name(value: str) -> str:
    normalized = _COMMAND_NAME.sub("-", value.lower()).strip("-")
    return normalized[:64].strip("-")


def _argument_syntax(text: str) -> tuple[str, ...]:
    syntax: list[str] = []
    if re.search(r"\$ARGUMENTS\b", text, flags=re.IGNORECASE):
        syntax.append("dollar_arguments")
    if re.search(r"\$[1-9]\b", text):
        syntax.append("positional_dollar")
    if re.search(r"\{\{\s*args\s*\}\}", text, flags=re.IGNORECASE):
        syntax.append("brace_args")
    named = re.findall(r"\{\{\s*([A-Za-z_][A-Za-z0-9_-]*)\s*\}\}", text)
    if any(value.lower() != "args" for value in named):
        syntax.append("named_brace")
    return tuple(syntax)


def _trust_reasons(text: str, format_name: PromptFormat) -> tuple[str, ...]:
    reasons = ["explicit_local_source"]
    if format_name == "unsupported":
        reasons.append("unsupported_format")
    if text and is_sensitive_metadata_text(text):
        reasons.append("secret_like_text")
    if _PROMPT_INJECTION.search(text):
        reasons.append("prompt_injection_indicator")
    return tuple(reasons)


def _collisions(sources: tuple[_PromptSourceAudit, ...], existing_commands: frozenset[str]) -> list[JsonObject]:
    sources_by_command: dict[str, list[str]] = {}
    for source in sources:
        if source.proposed_command:
            sources_by_command.setdefault(source.proposed_command, []).append(source.path)
    collisions: list[JsonObject] = []
    for command_name, paths in sorted(sources_by_command.items()):
        if len(paths) > 1:
            collisions.append({"kind": "duplicate_candidate", "command_name": command_name, "source_count": len(paths)})
        if command_name in existing_commands:
            collisions.append({"kind": "existing_command", "command_name": command_name, "source_count": len(paths)})
    return collisions
