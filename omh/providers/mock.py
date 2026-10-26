"""Offline provider used by ``omh --offline``, demos and smoke tests."""

from __future__ import annotations

import time
from typing import Iterator, List, Optional

from .base import ChatMessage, Chunk, Provider, ToolCall, Usage

WELCOME = """\
## OMH is running offline

No API key is needed for this demo: the router is serving the built-in \
`mock` model, and the whole harness — streaming, tool calls, sessions — is \
live. Ask me to **list the files** and I will call `list_dir` for real.

To talk to a real model, set one of these and restart:

- `OPENAI_API_KEY` → `gpt-4o-mini`, `gpt-4o`
- `ANTHROPIC_API_KEY` → `sonnet-4`, `haiku-3.5`
- `GROQ_API_KEY` / `DEEPSEEK_API_KEY` / `OPENROUTER_API_KEY`
- run Ollama locally, or point `OMH_BASE_URL` at any OpenAI-compatible server

`/models` shows the pool, `/balance` shows live routing stats.
"""


class MockProvider(Provider):
    kind = "mock"

    def __init__(self, name: str = "mock", *, delay: float = 0.008) -> None:
        super().__init__(name)
        self.delay = delay

    def available(self) -> bool:
        return True

    def _reply(self, messages: List[ChatMessage]) -> str:
        last_user = ""
        tool_results = [m for m in messages if m.role == "tool"]
        for message in reversed(messages):
            if message.role == "user" and message.content.strip():
                last_user = message.content.strip()
                break
        if tool_results:
            output = tool_results[-1].content.strip()
            preview = "\n".join(output.splitlines()[:8])
            return (
                "### Tool result, as the agent saw it\n\n"
                "```text\n"
                f"{preview}\n"
                "```\n\n"
                "The loop is complete: the model asked for a tool, OMH ran it in "
                "parallel-safe mode, and the output came back as a tool message."
            )
        return (
            f"{WELCOME}\n"
            "---\n"
            f"You said: `{last_user or '(nothing)'}`\n\n"
            "Try `/tools`, `/balance`, or `!ls -la`. Press `ctrl-c` to cancel a run."
        )

    def stream(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        tools: Optional[List[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> Iterator[Chunk]:
        prompt_chars = sum(len(m.content) for m in messages)
        usage = Usage(prompt_tokens=max(1, prompt_chars // 4))
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        already_listed = any(m.role == "tool" for m in messages)

        if tools and not already_listed and "list" in last_user.lower():
            yield Chunk(text="Let me look at the workspace first.\n\n", model=model)
            if self.delay:
                time.sleep(self.delay)
            usage.completion_tokens = 12
            yield Chunk(
                tool_calls=[ToolCall("call_mock_list", "list_dir", '{"path": "."}')],
                finish="tool_calls",
                usage=usage,
                model=model,
            )
            return

        text = self._reply(messages)
        # Stream in small word groups so the renderer gets a real workout.
        buffer = ""
        for token in text.split(" "):
            buffer += token + " "
            if len(buffer) >= 24:
                yield Chunk(text=buffer, model=model)
                usage.completion_tokens += max(1, len(buffer) // 4)
                buffer = ""
                if self.delay:
                    time.sleep(self.delay)
        if buffer:
            yield Chunk(text=buffer, model=model)
            usage.completion_tokens += max(1, len(buffer) // 4)
        yield Chunk(finish="stop", usage=usage, model=model)
