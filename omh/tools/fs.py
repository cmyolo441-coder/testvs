"""Filesystem tools scoped to the workspace."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, ToolSpec

MAX_READ_BYTES = 400_000
MAX_READ_LINES = 4_000


def _display(path: Path, context: ToolContext) -> str:
    try:
        return str(path.relative_to(context.workspace)) or "."
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolError(f"cannot read {path}: {exc}") from exc
    if b"\x00" in raw[:4096]:
        raise ToolError(f"{path} looks binary; this tool only reads text")
    if len(raw) > MAX_READ_BYTES:
        raw = raw[:MAX_READ_BYTES]
    return raw.decode("utf-8", "replace")


def _read_file(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    path = context.resolve(arguments["path"])
    if not path.exists():
        raise ToolError(f"file not found: {_display(path, context)}")
    if path.is_dir():
        raise ToolError(f"{_display(path, context)} is a directory; use list_dir")
    lines = _read_text(path).splitlines()
    total = len(lines)
    start = max(1, int(arguments.get("start_line", 1) or 1))
    end = int(arguments.get("end_line", 0) or 0) or total
    end = min(total, max(start, end), start + MAX_READ_LINES - 1)
    if start > total:
        return ToolResult.success(
            f"{_display(path, context)} has {total} lines; start_line {start} is past the end"
        )
    numbered = "\n".join(
        f"{number:>6}  {lines[number - 1]}" for number in range(start, end + 1)
    )
    header = f"{_display(path, context)}  (lines {start}-{end} of {total})"
    return ToolResult.success(f"{header}\n{numbered}", path=str(path), lines=end - start + 1)


def _write_file(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    path = context.resolve(arguments["path"])
    content = str(arguments["content"])
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot write {_display(path, context)}: {exc}") from exc
    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    action = "updated" if existed else "created"
    return ToolResult.success(
        f"{action} {_display(path, context)} ({line_count} lines, {len(content)} bytes)",
        path=str(path),
    )


def _edit_file(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    path = context.resolve(arguments["path"])
    if not path.exists():
        raise ToolError(f"file not found: {_display(path, context)}")
    old = str(arguments["old_string"])
    new = str(arguments["new_string"])
    if old == new:
        raise ToolError("old_string and new_string are identical")
    if not old:
        raise ToolError("old_string must not be empty; use write_file for new files")
    text = _read_text(path)
    occurrences = text.count(old)
    if occurrences == 0:
        raise ToolError(
            "old_string was not found",
            hint="read the file again and copy the exact text, including indentation",
        )
    replace_all = bool(arguments.get("replace_all", False))
    if occurrences > 1 and not replace_all:
        raise ToolError(
            f"old_string appears {occurrences} times",
            hint="include more surrounding lines, or pass replace_all=true",
        )
    updated = text.replace(old, new, -1 if replace_all else 1)
    try:
        path.write_text(updated, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"cannot write {_display(path, context)}: {exc}") from exc

    diff = list(
        difflib.unified_diff(
            text.splitlines(),
            updated.splitlines(),
            fromfile=f"a/{_display(path, context)}",
            tofile=f"b/{_display(path, context)}",
            n=2,
            lineterm="",
        )
    )
    diff_text = "\n".join(diff[:120])
    if len(diff) > 120:
        diff_text += f"\n… {len(diff) - 120} more diff lines"
    label = _display(path, context)
    count = occurrences if replace_all else 1
    return ToolResult.success(
        f"edited {label} ({count} replacement{'s' if count > 1 else ''})\n{diff_text}",
        path=str(path),
        diff=diff_text,
    )


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{size}B"


def _list_dir(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    path = context.resolve(arguments.get("path", "."))
    if not path.exists():
        raise ToolError(f"directory not found: {_display(path, context)}")
    if not path.is_dir():
        raise ToolError(f"{_display(path, context)} is not a directory")
    show_all = bool(arguments.get("all", False))
    entries: List[str] = []
    try:
        children = sorted(
            path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())
        )
    except OSError as exc:
        raise ToolError(f"cannot list {_display(path, context)}: {exc}") from exc
    for child in children:
        if not show_all and child.name.startswith("."):
            continue
        if child.is_dir():
            entries.append(f"{child.name}/")
        else:
            try:
                size = _human_size(child.stat().st_size)
            except OSError:
                size = "?"
            entries.append(f"{child.name}  ({size})")
        if len(entries) >= 400:
            entries.append("… more entries truncated")
            break
    header = f"{_display(path, context)}/  ({len(entries)} entries)"
    return ToolResult.success(header + "\n" + "\n".join(entries))


def _preview_path(arguments: Dict[str, Any]) -> str:
    return str(arguments.get("path") or ".")


def fs_tools() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="read_file",
            description=(
                "Read a UTF-8 text file from the workspace. Output is numbered. "
                "Use start_line/end_line for very large files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the workspace.",
                    },
                    "start_line": {"type": "integer", "description": "First line (1-based)."},
                    "end_line": {"type": "integer", "description": "Last line, inclusive."},
                },
                "required": ["path"],
            },
            run=_read_file,
            preview=_preview_path,
        ),
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file with exact content. Parent directories "
                "are created. Prefer edit_file for existing files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            run=_write_file,
            requires_approval=True,
            preview=_preview_path,
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Replace an exact string in a file. old_string must match once "
                "(include surrounding lines) unless replace_all is true. Returns a diff."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
            run=_edit_file,
            requires_approval=True,
            preview=_preview_path,
        ),
        ToolSpec(
            name="list_dir",
            description=(
                "List a directory: folders first, then files with sizes. Hidden "
                "entries are skipped unless all=true."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "all": {"type": "boolean"},
                },
            },
            run=_list_dir,
            preview=_preview_path,
        ),
    ]


