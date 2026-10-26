"""Anthropic Messages API provider (streaming, tool use)."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..config import ProviderConfig
from ..errors import ProviderError
from .base import ChatMessage, Chunk, Provider, ToolCall, Usage
from .http import post_sse, sse_json


def convert_messages(messages: List[ChatMessage]) -> tuple[str, List[Dict[str, Any]]]:
    """Split OMH messages into Anthropic's ``system`` + alternating turns."""
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    turns: List[Dict[str, Any]] = []

    for message in messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content or "(no output)",
            }
            turns.append({"role": "user", "content": [block]})
            continue
        if message.role == "assistant":
            blocks: List[Dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            for call in message.tool_calls:
                try:
                    args = call.parsed()
                except ValueError:
                    args = {}
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": args}
                )
            if not blocks:
                blocks.append({"type": "text", "text": "(empty)"})
            turns.append({"role": "assistant", "content": blocks})
            continue
        turns.append(
            {"role": "user", "content": [{"type": "text", "text": message.content}]}
        )

    merged: List[Dict[str, Any]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"].extend(turn["content"])
        else:
            merged.append({"role": turn["role"], "content": list(turn["content"])})
    return "\n\n".join(system_parts), merged


def convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    converted = []
    for tool in tools or []:
        function = tool.get("function") or {}
        converted.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object"},
            }
        )
    return converted


STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}


class AnthropicProvider(Provider):
    kind = "anthropic"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config.name)
        self.config = config

    def available(self) -> bool:
        return self.config.available()

    def _url(self) -> str:
        base = self.config.base_url
        if not base:
            raise ProviderError(
                f"{self.name} has no base_url; add it to config.toml",
                retryable=False,
            )
        base = base.rstrip("/")
        if base.endswith("/messages"):
            return base
        if base.endswith("/v1"):
            return base + "/messages"
        return base + "/v1/messages"

    def _headers(self) -> Dict[str, str]:
        headers = dict(self.config.headers)
        headers.setdefault("anthropic-version", "2023-06-01")
        key = self.config.resolved_key()
        if key:
            headers.setdefault("x-api-key", key)
        return headers

    def stream(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[Chunk]:
        system, turns = convert_messages(messages)
        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": max(0.0, min(1.0, temperature)),
            "messages": turns,
            "stream": True,
        }
        if system:
            payload["system"] = system
        converted_tools = convert_tools(tools)
        if converted_tools:
            payload["tools"] = converted_tools

        events = post_sse(
            self._url(),
            payload,
            self._headers(),
            timeout=self.config.timeout,
            provider=self.name,
        )
        blocks: Dict[int, Dict[str, Any]] = {}
        usage = Usage()
        finish: Optional[str] = None
        seen_model = model

        for event, data in sse_json(events, self.name):
            if data.get("type") == "message_start":
                message = data.get("message") or {}
                seen_model = str(message.get("model") or seen_model)
                raw_usage = message.get("usage") or {}
                usage.prompt_tokens = int(raw_usage.get("input_tokens", 0) or 0)
                usage.completion_tokens = int(raw_usage.get("output_tokens", 0) or 0)
            elif data.get("type") == "content_block_start":
                index = int(data.get("index", 0) or 0)
                block = data.get("content_block") or {}
                if block.get("type") == "tool_use":
                    blocks[index] = {
                        "id": str(block.get("id") or ""),
                        "name": str(block.get("name") or ""),
                        "args": [],
                    }
            elif data.get("type") == "content_block_delta":
                index = int(data.get("index", 0) or 0)
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = str(delta.get("text") or "")
                    if text:
                        yield Chunk(text=text, model=seen_model)
                elif delta.get("type") == "input_json_delta":
                    if index in blocks:
                        blocks[index]["args"].append(str(delta.get("partial_json") or ""))
            elif data.get("type") == "message_delta":
                delta = data.get("delta") or {}
                if delta.get("stop_reason"):
                    finish = STOP_REASONS.get(str(delta["stop_reason"]), "stop")
                raw_usage = data.get("usage") or {}
                if raw_usage.get("output_tokens"):
                    usage.completion_tokens = int(raw_usage["output_tokens"])
            elif data.get("type") == "error":
                error = data.get("error") or {}
                raise ProviderError(
                    f"{self.name} error: {error.get('message', 'stream error')}",
                    retryable=False,
                )
            elif data.get("type") == "message_stop":
                break

        calls: List[ToolCall] = []
        for index in sorted(blocks):
            block = blocks[index]
            calls.append(
                ToolCall(
                    id=block["id"] or f"call_{index}",
                    name=block["name"],
                    arguments="".join(block["args"]) or "{}",
                )
            )
        if calls:
            yield Chunk(
                tool_calls=calls,
                finish="tool_calls",
                usage=usage if usage.total else None,
                model=seen_model,
            )
        else:
            yield Chunk(
                finish=finish or "stop",
                usage=usage if usage.total else None,
                model=seen_model,
            )

