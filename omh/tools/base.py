"""Tool primitives: schemas, validation, approval and execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from ..errors import ApprovalDenied, OmhError, ToolError
from ..providers.base import ToolCall

Approver = Callable[["ToolSpec", Dict[str, Any], str], bool]


@dataclass
class ToolContext:
    workspace: Path
    max_output: int = 20_000
    approval_mode: str = "ask"  # ask | auto | deny

    def resolve(self, raw: str) -> Path:
        """Resolve a tool path inside the workspace sandbox."""
        candidate = Path(str(raw or ".")).expanduser()
        if not candidate.is_absolute():
            candidate = self.workspace / candidate
        resolved = candidate.resolve()
        workspace = self.workspace.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ToolError(
                f"path escapes the workspace: {raw}",
                hint=f"workspace is {workspace}",
            )
        return resolved


@dataclass
class ToolResult:
    ok: bool
    output: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(cls, output: str, **meta: Any) -> "ToolResult":
        return cls(True, output, meta)

    @classmethod
    def error(cls, output: str, **meta: Any) -> "ToolResult":
        return cls(False, output, meta)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[[Dict[str, Any], ToolContext], ToolResult]
    requires_approval: bool = False
    preview: Optional[Callable[[Dict[str, Any]], str]] = None

    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def summary(self, arguments: Dict[str, Any]) -> str:
        if self.preview is None:
            return ""
        try:
            return self.preview(arguments)
        except Exception:  # pragma: no cover - previews must never break a run
            return ""


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def validate_arguments(spec: ToolSpec, arguments: Dict[str, Any]) -> None:
    schema = spec.parameters or {}
    properties = schema.get("properties") or {}
    for key in schema.get("required") or []:
        if key not in arguments:
            raise ToolError(f"{spec.name}: missing required argument `{key}`")
    for key, value in arguments.items():
        if key not in properties:
            allowed = ", ".join(sorted(properties))
            raise ToolError(
                f"{spec.name}: unknown argument `{key}`",
                hint=f"allowed: {allowed}" if allowed else "",
            )
        expected = (properties[key] or {}).get("type")
        if expected and not _type_ok(value, expected):
            raise ToolError(
                f"{spec.name}: `{key}` must be {expected}, got {type(value).__name__}"
            )


def truncate_output(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: limit - 200]
    tail = text[-150:]
    hidden = len(text) - len(head) - len(tail)
    return f"{head}\n\n… [{hidden:,} characters truncated] …\n\n{tail}"


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec], context: ToolContext) -> None:
        self.context = context
        self.specs: Dict[str, ToolSpec] = {spec.name: spec for spec in specs}

    def schema(self) -> List[Dict[str, Any]]:
        return [spec.schema() for spec in self.specs.values()]

    def get(self, name: str) -> Optional[ToolSpec]:
        return self.specs.get(name)

    def names(self) -> List[str]:
        return sorted(self.specs)

    def execute(
        self,
        call: ToolCall,
        approver: Optional[Approver] = None,
        *,
        approved: bool = False,
    ) -> ToolResult:
        spec = self.specs.get(call.name)
        if spec is None:
            return ToolResult.error(
                f"unknown tool `{call.name}`",
                hint="available: " + ", ".join(self.names()),
            )
        try:
            arguments = call.parsed()
            validate_arguments(spec, arguments)
        except (ValueError, ToolError) as exc:
            return ToolResult.error(str(exc))

        summary = spec.summary(arguments)
        try:
            if (
                spec.requires_approval
                and not approved
                and self.context.approval_mode != "auto"
            ):
                if self.context.approval_mode == "deny":
                    raise ApprovalDenied(
                        f"{spec.name} is disabled by approval_mode=deny"
                    )
                if approver is None or not approver(spec, arguments, summary):
                    raise ApprovalDenied(f"{spec.name} was not approved")
            result = spec.run(arguments, self.context)
        except ApprovalDenied as exc:
            return ToolResult.error(str(exc))
        except OmhError as exc:
            hint = f"\n{exc.hint}" if exc.hint else ""
            return ToolResult.error(f"{exc.message}{hint}")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # a tool bug is reported, not fatal
            return ToolResult.error(f"{spec.name} failed: {type(exc).__name__}: {exc}")

        result.output = truncate_output(result.output or "", self.context.max_output)
        return result


def arguments_json(arguments: Dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

