"""Search tools: glob for filenames, grep for content."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, ToolSpec

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".next",
    "dist",
    "build",
    "target",
    "coverage",
    ".tox",
}
MAX_FILE_BYTES = 2_000_000


def _walk(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames if name not in SKIP_DIRS and not name.startswith(".")
        )
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _glob(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    pattern = str(arguments["pattern"]).strip()
    if not pattern:
        raise ToolError("pattern must not be empty")
    if pattern.startswith("/") or ".." in Path(pattern).parts:
        raise ToolError("pattern must stay inside the workspace")
    root = context.resolve(arguments.get("path", "."))
    if not root.is_dir():
        raise ToolError(f"{_relative(root, context.workspace)} is not a directory")
    max_results = min(1000, max(1, int(arguments.get("max_results", 200) or 200)))
    matches = []
    try:
        for candidate in root.glob(pattern):
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            if candidate.is_file():
                matches.append(_relative(candidate, context.workspace))
            if len(matches) >= max_results:
                break
    except (ValueError, OSError) as exc:
        raise ToolError(f"bad glob pattern {pattern!r}: {exc}") from exc
    matches.sort()
    if not matches:
        return ToolResult.success(f"no files match {pattern!r} under {_relative(root, context.workspace)}")
    note = f"\n… capped at {max_results}" if len(matches) >= max_results else ""
    return ToolResult.success(
        f"{len(matches)} match{'es' if len(matches) != 1 else ''} for {pattern!r}\n"
        + "\n".join(matches)
        + note
    )


def _grep(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    pattern = str(arguments["pattern"])
    if not pattern:
        raise ToolError("pattern must not be empty")
    flags = re.IGNORECASE if arguments.get("ignore_case") else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        raise ToolError(f"invalid regex {pattern!r}: {exc}") from exc
    root = context.resolve(arguments.get("path", "."))
    if not root.exists():
        raise ToolError(f"path not found: {_relative(root, context.workspace)}")
    file_glob = str(arguments.get("glob") or "").strip()
    limit = min(500, max(1, int(arguments.get("max_results", 100) or 100)))
    results: List[str] = []
    files_scanned = 0
    truncated = False

    candidates: Iterator[Path]
    if root.is_file():
        candidates = iter([root])
    else:
        candidates = _walk(root)

    for path in candidates:
        relative = _relative(path, context.workspace)
        if file_glob and not fnmatch.fnmatch(relative, file_glob) and not fnmatch.fnmatch(path.name, file_glob):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        files_scanned += 1
        text = raw.decode("utf-8", "replace")
        for number, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                snippet = line.strip()
                if len(snippet) > 240:
                    snippet = snippet[:237] + "…"
                results.append(f"{relative}:{number}: {snippet}")
                if len(results) >= limit:
                    truncated = True
                    break
        if truncated:
            break

    if not results:
        return ToolResult.success(
            f"no matches for {pattern!r} ({files_scanned} files scanned)"
        )
    header = f"{len(results)} match{'es' if len(results) != 1 else ''} for {pattern!r} ({files_scanned} files scanned)"
    if truncated:
        header += f" — capped at {limit}"
    return ToolResult.success(header + "\n" + "\n".join(results), matches=len(results))


def _preview_search(arguments: Dict[str, Any]) -> str:
    if "pattern" in arguments:
        return f"{arguments['pattern']}  in {arguments.get('path', '.')}"
    return str(arguments.get("path", "."))


def search_tools() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="glob",
            description=(
                "Find files by glob pattern, e.g. `**/*.py` or `tests/test_*.py`. "
                "Skips .git, node_modules, virtualenvs and build output."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory to search from."},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            run=_glob,
            preview=lambda arguments: str(arguments.get("pattern") or ""),
        ),
        ToolSpec(
            name="grep",
            description=(
                "Search file contents with a Python regular expression. Returns "
                "`path:line: text`. Filter with glob, invert case with ignore_case."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "e.g. `*.py` or `src/**`"},
                    "ignore_case": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            run=_grep,
            preview=_preview_search,
        ),
    ]

