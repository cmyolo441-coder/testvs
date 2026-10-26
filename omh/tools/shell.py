"""Shell tool: one command, captured output, hard timeout."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List

from ..errors import ToolError
from .base import ToolContext, ToolResult, ToolSpec


def _run_shell(arguments: Dict[str, Any], context: ToolContext) -> ToolResult:
    command = str(arguments["command"]).strip()
    if not command:
        raise ToolError("command must not be empty")
    cwd = context.resolve(arguments.get("cwd", "."))
    if not cwd.is_dir():
        raise ToolError(f"cwd is not a directory: {cwd}")
    timeout = min(600, max(1, int(arguments.get("timeout", 120) or 120)))
    env = {
        **os.environ,
        "TERM": "dumb",
        "PAGER": "cat",
        "GIT_PAGER": "cat",
        "PYTHONUNBUFFERED": "1",
    }
    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        partial = ""
        for stream in (exc.stdout, exc.stderr):
            if stream:
                partial += stream if isinstance(stream, str) else stream.decode("utf-8", "replace")
        note = f"timed out after {timeout}s"
        return ToolResult.error(f"{note}\n{partial}".strip(), exit_code=None)

    output = completed.stdout or ""
    if completed.stderr:
        if output and not output.endswith("\n"):
            output += "\n"
        output += completed.stderr
    output = output.rstrip() or "(no output)"
    header = f"$ {command}\nexit {completed.returncode}"
    return ToolResult(
        ok=completed.returncode == 0,
        output=f"{header}\n{output}",
        meta={"exit_code": completed.returncode, "command": command},
    )


def _preview_shell(arguments: Dict[str, Any]) -> str:
    command = str(arguments.get("command") or "").replace("\n", " ; ")
    return command[:100] + ("…" if len(command) > 100 else "")


def shell_tools() -> List[ToolSpec]:
    return [
        ToolSpec(
            name="run_shell",
            description=(
                "Run a shell command in the workspace and return its combined "
                "stdout/stderr plus the exit code. Use it to run tests, builds and "
                "git; commands are killed after the timeout."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "description": "Working directory."},
                    "timeout": {
                        "type": "integer",
                        "description": "Seconds before the command is killed (max 600).",
                    },
                },
                "required": ["command"],
            },
            run=_run_shell,
            requires_approval=True,
            preview=_preview_shell,
        )
    ]
