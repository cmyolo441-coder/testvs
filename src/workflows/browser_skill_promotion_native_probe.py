#!/usr/bin/env python3
"""Fixed read-only Hermes-native preflight subprocess for browser-skill promotion.

The parent invokes this shipped script only through the installed Hermes venv
with an explicit Hermes source cwd.  The script never imports OMH and never
executes project-skill bytes.  Native config loading runs against a temporary
copy of stable target config bytes, because Hermes' malformed-config recovery
can create a ``.corrupt`` backup even from ``load_config_readonly``.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict

_PREFIX = "OMH_BROWSER_SKILL_PROMOTION_NATIVE_PREFLIGHT="
_SCHEMA = "browser_skill_promotion_native_preflight/v1"
_MAX_REQUEST_BYTES = 512 * 1024
_MAX_CONFIG_BYTES = 262144


class NativeProbeRequest(TypedDict):
    project_root: str
    package: dict[str, str]


def main() -> None:
    try:
        request = _request(sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1))
        project_root = Path(request["project_root"])
        package = request["package"]
        skill_name = _skill_name(package["SKILL.md"])
        preflight = _inspect(project_root, package, skill_name)
    except Exception as exc:
        _emit({"error": "native preflight unavailable", "detail": type(exc).__name__})
        return
    _emit(preflight)


def _request(raw: bytes) -> NativeProbeRequest:
    if len(raw) > _MAX_REQUEST_BYTES:
        raise ValueError("request exceeds bound")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or set(value) != {"schema_version", "project_root", "package"}:
        raise ValueError("unsupported request")
    project_root = value.get("project_root")
    if value.get("schema_version") != _SCHEMA or not isinstance(project_root, str):
        raise ValueError("unsupported request")
    package = value.get("package")
    if not isinstance(package, dict) or not package:
        raise ValueError("invalid package")
    parsed_package: dict[str, str] = {}
    for name, text in package.items():
        if not isinstance(name, str) or not isinstance(text, str):
            raise ValueError("invalid package")
        parsed_package[name] = text
    return {"project_root": project_root, "package": parsed_package}


def _inspect(project_root: Path, package: Mapping[str, str], skill_name: str) -> dict[str, object]:
    # Snapshot/evaluate target policy before importing any tool module that could
    # transitively consult Hermes config.  Invalid target YAML must stop here,
    # before a host recovery path can create a target-profile backup.
    trusted, policy = _native_trust_and_write_policy(project_root)
    from tools.skill_linter import ERROR, lint_skill
    from tools.skill_manager_tool import _validate_frontmatter
    from tools.skills_guard import scan_skill

    with tempfile.TemporaryDirectory(prefix="omh-browser-promotion-") as raw:
        stage = Path(raw) / skill_name
        for relative, text in package.items():
            _stage_file(stage, relative, text)
        entry = (stage / "SKILL.md").read_text(encoding="utf-8")
        structure_error = _validate_frontmatter(entry, new_skill=True)
        lint_errors = [
            finding.message
            for finding in lint_skill(stage / "SKILL.md")
            if finding.severity == ERROR
        ]
        security_verdict = scan_skill(stage, source="project-local").verdict
    return {
        "schema_version": _SCHEMA,
        "project_root": str(project_root),
        "package_digest": _package_digest(package),
        "trusted": trusted,
        "structure_error": structure_error,
        "lint_errors": lint_errors,
        "security_verdict": security_verdict,
        "policy": policy,
    }


def _skill_name(entry: str) -> str:
    for line in entry.splitlines():
        if line.startswith("name: "):
            name = line.removeprefix("name: ").strip()
            if name and all(ch.isascii() and (ch.islower() or ch.isdigit() or ch == "-") for ch in name):
                return name
    raise ValueError("invalid skill name")


def _stage_file(root: Path, relative: str, text: str) -> None:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {".", ".."} for part in parts):
        raise ValueError("unsafe package path")
    destination = root.joinpath(*parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8", newline="")


def _native_trust_and_write_policy(project_root: Path) -> tuple[bool, dict[str, object]]:
    # Importing hermes_cli.config itself initializes profile state and may recover
    # malformed config, so defer that import until HERMES_HOME names our copy.
    from hermes_cli.managed_scope import get_managed_dir
    from hermes_constants import get_config_path
    from utils import fast_safe_load

    managed_dir = get_managed_dir()
    target_paths = (Path(get_config_path()), None if managed_dir is None else managed_dir / "config.yaml")
    before = _config_inputs(target_paths, fast_safe_load)
    if before is None:
        raise ValueError("native policy unavailable")
    # Do not give Hermes' recovery loader a real profile path.  It gets exact bytes
    # only after strict parse + fd/path identity checks, then its policy API evaluates
    # that private snapshot under its real interpreter/dependencies.
    with tempfile.TemporaryDirectory(prefix="omh-browser-policy-") as raw:
        temporary = Path(raw)
        home, managed = temporary / "home", temporary / "managed"
        home.mkdir(); managed.mkdir()
        _copy_snapshot(before[0], home / "config.yaml")
        _copy_snapshot(before[1], managed / "config.yaml")
        with _temporary_config_environment(home, managed):
            from hermes_cli.config import cfg_get, get_active_config_parse_failure, load_config_readonly
            trusted = _native_trust(project_root)
            effective = cfg_get(load_config_readonly(), "skills", "write_approval", default=None)
            parse_failure = get_active_config_parse_failure()
    after = _config_inputs(target_paths, fast_safe_load)
    if after is None or _revisions(before) != _revisions(after) or parse_failure is not None or not isinstance(effective, bool):
        raise ValueError("native policy unavailable")
    revision = _digest(_canonical({"files": _revisions(before), "effective": effective}))
    if effective:
        return trusted, {"requirement": "required", "approval": "not_obtained", "support": "unsupported", "revision": revision}
    return trusted, {"requirement": "not_required", "approval": "not_applicable", "support": "available", "revision": revision}


def _native_trust(project_root: Path) -> bool:
    from agent.skill_utils import is_project_root_trusted
    return bool(is_project_root_trusted(project_root))


@contextmanager
def _temporary_config_environment(home: Path, managed: Path) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in ("HERMES_HOME", "HERMES_MANAGED_DIR")}
    os.environ["HERMES_HOME"] = str(home)
    os.environ["HERMES_MANAGED_DIR"] = str(managed)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _config_inputs(paths: tuple[Path | None, ...], loader: Callable[[str], object]) -> list[tuple[tuple[str, int, int, str] | None, bytes | None]] | None:
    result: list[tuple[tuple[str, int, int, str] | None, bytes | None]] = []
    for path in paths:
        if path is None or not os.path.lexists(path):
            result.append((None, None))
            continue
        snapshot = _config_snapshot(path, loader)
        if snapshot is None:
            return None
        result.append(snapshot)
    return result


def _config_snapshot(path: Path, loader: Callable[[str], object]) -> tuple[tuple[str, int, int, str], bytes] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_CONFIG_BYTES:
                return None
            chunks: list[bytes] = []
            remaining = _MAX_CONFIG_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining == 0:
                return None
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        parsed = loader(raw.decode("utf-8"))
        current = os.stat(path, follow_symlinks=False)
    except Exception:
        return None
    if (
        not stat.S_ISREG(current.st_mode)
        or (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)
        != (current.st_dev, current.st_ino, current.st_mtime_ns, current.st_size)
        or (parsed is not None and not isinstance(parsed, dict))
    ):
        return None
    return ((str(path.resolve()), info.st_mtime_ns, info.st_size, _digest(raw)), raw)


def _copy_snapshot(snapshot: tuple[tuple[str, int, int, str] | None, bytes | None], destination: Path) -> None:
    revision, raw = snapshot
    if revision is None:
        return
    if raw is None or _digest(raw) != revision[-1]:
        raise ValueError("invalid snapshot")
    destination.write_bytes(raw)


def _revisions(inputs: list[tuple[tuple[str, int, int, str] | None, bytes | None]]) -> list[tuple[str, int, int, str] | None]:
    return [revision for revision, _ in inputs]


def _package_digest(package: Mapping[str, str]) -> str:
    return _digest(b"".join(name.encode("utf-8") + b"\0" + package[name].encode("utf-8") for name in sorted(package)))


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _emit(value: Mapping[str, object]) -> None:
    print(_PREFIX + json.dumps(value, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
