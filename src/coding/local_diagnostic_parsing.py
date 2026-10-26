"""Normalize bounded provider JSON without retaining diagnostic prose."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath


def parse_local_diagnostics(
    provider_id: str,
    payload: bytes,
    snapshot: Path,
    files: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    """Parse only severity, code, relative position, and provider identity."""
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("local diagnostic output was not valid UTF-8 JSON") from exc
    if provider_id == "ruff":
        if not isinstance(document, list):
            raise ValueError("ruff diagnostic output must be a list")
        rows = document
    else:
        if not isinstance(document, dict):
            raise ValueError("pyright diagnostic output must be an object")
        rows = document.get("generalDiagnostics")
        if not isinstance(rows, list):
            raise ValueError("pyright diagnostics must be a list")
    normalized = tuple(
        _diagnostic_item(provider_id, row, snapshot, files)
        for row in rows
    )
    by_identity = {
        _diagnostic_sort_key(item): item
        for item in normalized
    }
    return tuple(by_identity[key] for key in sorted(by_identity))


def _diagnostic_item(
    provider_id: str,
    row: object,
    snapshot: Path,
    files: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(row, dict):
        raise ValueError("local diagnostic item must be an object")
    if provider_id == "ruff":
        location = row.get("location")
        if not isinstance(location, dict):
            raise ValueError("ruff diagnostic location must be an object")
        line, character = location.get("row"), location.get("column")
        filename, code = row.get("filename"), row.get("code")
        severity = _ruff_severity(code)
    else:
        range_value = row.get("range")
        if not isinstance(range_value, dict):
            raise ValueError("pyright diagnostic range must be an object")
        start = range_value.get("start")
        if not isinstance(start, dict):
            raise ValueError("pyright diagnostic start must be an object")
        raw_line, raw_character = start.get("line"), start.get("character")
        line = raw_line + 1 if isinstance(raw_line, int) else raw_line
        character = (
            raw_character + 1
            if isinstance(raw_character, int)
            else raw_character
        )
        filename, code = row.get("file"), row.get("rule")
        severity = row.get("severity")
    if (
        not isinstance(filename, str)
        or not isinstance(code, str)
        or not code
        or not isinstance(line, int)
        or isinstance(line, bool)
        or not isinstance(character, int)
        or isinstance(character, bool)
        or severity not in ("error", "warning", "information", "hint")
    ):
        raise ValueError("local diagnostic item has invalid metadata")
    path = _relative_provider_path(filename, snapshot)
    if path not in files:
        raise ValueError("local diagnostic item escaped the requested file scope")
    return {
        "severity": severity,
        "code": code,
        "path": path,
        "line": line,
        "character": character,
        "source": provider_id,
    }


def _relative_provider_path(filename: str, snapshot: Path) -> str:
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = snapshot / candidate
    try:
        relative = candidate.resolve().relative_to(snapshot.resolve())
    except ValueError as exc:
        raise ValueError(
            "local diagnostic path escaped its revision worktree"
        ) from exc
    posix = PurePosixPath(relative.as_posix())
    if ".." in posix.parts or ".git" in posix.parts:
        raise ValueError("local diagnostic path is unsafe")
    return posix.as_posix()


def _ruff_severity(code: object) -> str:
    if isinstance(code, str) and code.startswith(("E", "F")):
        return "error"
    if isinstance(code, str) and code.startswith("W"):
        return "warning"
    return "information"


def _diagnostic_sort_key(
    item: dict[str, object],
) -> tuple[str, int, int, str]:
    line = item["line"]
    character = item["character"]
    if (
        not isinstance(line, int)
        or isinstance(line, bool)
        or not isinstance(character, int)
        or isinstance(character, bool)
    ):
        raise ValueError("local diagnostic position was not normalized")
    return (
        str(item["path"]),
        line,
        character,
        str(item["code"]),
    )
