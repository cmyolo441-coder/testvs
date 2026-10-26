"""Provider-agnostic message and streaming types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"

    def parsed(self) -> Dict[str, Any]:
        """Arguments as a dict; malformed JSON becomes a readable tool error."""
        if not self.arguments.strip():
            return {}
        try:
            value = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON arguments for {self.name}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"arguments for {self.name} must be a JSON object")
        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ToolCall":
        fn = raw.get("function") or {}
        return cls(
            id=str(raw.get("id") or ""),
            name=str(fn.get("name") or raw.get("name") or ""),
            arguments=str(fn.get("arguments") or raw.get("arguments") or "{}"),
        )


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: Optional[List[ToolCall]] = None) -> "ChatMessage":
        return cls("assistant", content, list(tool_calls or []))

    @classmethod
    def tool(cls, tool_call_id: str, content: str, name: str = "") -> "ChatMessage":
        return cls("tool", content, tool_call_id=tool_call_id, name=name)

    def to_dict(self) -> Dict[str, Any]:
        if self.role == "tool":
            return {"role": "tool", "tool_call_id": self.tool_call_id, "content": self.content}
        if self.role == "assistant":
            message: Dict[str, Any] = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                message["tool_calls"] = [call.to_dict() for call in self.tool_calls]
            return message
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ChatMessage":
        return cls(
            role=str(raw.get("role") or "user"),
            content=str(raw.get("content") or ""),
            tool_calls=[ToolCall.from_dict(call) for call in raw.get("tool_calls") or []],
            tool_call_id=str(raw.get("tool_call_id") or ""),
            name=str(raw.get("name") or ""),
        )


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: Optional["Usage"]) -> None:
        if other is None:
            return
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens


@dataclass(slots=True)
class Chunk:
    """One streamed piece: text, a final tool-call list, or the finish marker."""

    text: str = ""
    tool_calls: Optional[List[ToolCall]] = None
    finish: Optional[str] = None
    usage: Optional[Usage] = None
    model: Optional[str] = None


class Provider:
    """Interface every backend implements."""

    kind = "base"

    def __init__(self, name: str) -> None:
        self.name = name

    def available(self) -> bool:  # pragma: no cover - overridden
        return True

    def stream(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[Chunk]:  # pragma: no cover - overridden
        raise NotImplementedError
