"""OpenAI-compatible ``/chat/completions`` streaming.

Works with OpenAI, Groq, OpenRouter, DeepSeek, Together, vLLM, LM Studio,
Ollama's OpenAI endpoint and anything else that speaks the same dialect.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..config import ProviderConfig
from ..errors import ProviderError
from .base import ChatMessage, Chunk, Provider, ToolCall, Usage
from .http import post_sse, sse_json


class _Accumulator:
    """Collects streamed tool-call fragments by index."""

    def __init__(self) -> None:
        self.slots: Dict[int, Dict[str, Any]] = {}

    def add(self, delta: Dict[str, Any]) -> None:
        index = int(delta.get("index", 0) or 0)
        slot = self.slots.setdefault(index, {"id": "", "name": "", "args": []})
        if delta.get("id"):
            slot["id"] = str(delta["id"])
        function = delta.get("function") or {}
        if function.get("name"):
            slot["name"] = str(function["name"])
        if function.get("arguments"):
            slot["args"].append(str(function["arguments"]))

    def calls(self) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for index in sorted(self.slots):
            slot = self.slots[index]
            if not slot["name"]:
                continue
            calls.append(
                ToolCall(
                    id=slot["id"] or f"call_{index}",
                    name=slot["name"],
                    arguments="".join(slot["args"]) or "{}",
                )
            )
        return calls


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _usage_of(raw: Optional[Dict[str, Any]]) -> Optional[Usage]:
    if not raw:
        return None
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
    )


class OpenAIProvider(Provider):
    kind = "openai"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config.name)
        self.config = config

    def available(self) -> bool:
        return self.config.available()

    def _url(self) -> str:
        base = self.config.base_url or "https://api.openai.com/v1"
        if base.endswith("/chat/completions"):
            return base
        return base.rstrip("/") + "/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = dict(self.config.headers)
        key = self.config.resolved_key()
        if key:
            headers.setdefault("Authorization", f"Bearer {key}")
        return headers

    def _payload(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        include_usage: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [message.to_dict() for message in messages],
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if include_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def stream(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[Chunk]:
        try:
            yield from self._stream_once(
                model, messages, tools, temperature, max_tokens, include_usage=True
            )
        except ProviderError as exc:
            # Older/leaner OpenAI-compatible servers reject stream_options.
            if exc.status == 400 and "stream_options" in exc.message:
                yield from self._stream_once(
                    model, messages, tools, temperature, max_tokens, include_usage=False
                )
            else:
                raise

    def _stream_once(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[Dict[str, Any]]],
        temperature: float,
        max_tokens: int,
        *,
        include_usage: bool,
    ) -> Iterator[Chunk]:
        payload = self._payload(
            model, messages, tools, temperature, max_tokens, include_usage
        )
        events = post_sse(
            self._url(),
            payload,
            self._headers(),
            timeout=self.config.timeout,
            provider=self.name,
        )
        accumulator = _Accumulator()
        usage: Optional[Usage] = None
        finish: Optional[str] = None
        seen_model = model

        for _, data in sse_json(events, self.name):
            if data.get("model"):
                seen_model = str(data["model"])
            choices = data.get("choices") or []
            if choices:
                choice = choices[0] or {}
                delta = choice.get("delta") or {}
                text = _text_of(delta.get("content"))
                if text:
                    yield Chunk(text=text, model=seen_model)
                for call_delta in delta.get("tool_calls") or []:
                    if isinstance(call_delta, dict):
                        accumulator.add(call_delta)
                if choice.get("finish_reason"):
                    finish = str(choice["finish_reason"])
            if data.get("usage"):
                usage = _usage_of(data.get("usage"))

        calls = accumulator.calls()
        if calls:
            yield Chunk(
                tool_calls=calls,
                finish="tool_calls",
                usage=usage,
                model=seen_model,
            )
        else:
            yield Chunk(finish=finish or "stop", usage=usage, model=seen_model)

