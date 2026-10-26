"""Tool registry factory."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .base import ToolContext, ToolRegistry, ToolResult, ToolSpec
from .fs import fs_tools
from .search import search_tools
from .shell import shell_tools
from .web import web_tools

__all__ = [
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "default_tools",
    "default_registry",
]


def default_tools() -> List[ToolSpec]:
    return [*fs_tools(), *search_tools(), *shell_tools(), *web_tools()]


def default_registry(
    workspace: str | Path,
    *,
    max_output: int = 20_000,
    approval_mode: str = "ask",
    tools: Optional[List[ToolSpec]] = None,
) -> ToolRegistry:
    context = ToolContext(
        workspace=Path(workspace).resolve(),
        max_output=max_output,
        approval_mode=approval_mode,
    )
    return ToolRegistry(tools if tools is not None else default_tools(), context)
