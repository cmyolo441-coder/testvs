"""Provider factory."""

from __future__ import annotations

from typing import Dict

from ..config import Config, ProviderConfig
from .base import ChatMessage, Chunk, Provider, ToolCall, Usage
from .anthropic import AnthropicProvider
from .mock import MockProvider
from .openai import OpenAIProvider

__all__ = [
    "ChatMessage",
    "Chunk",
    "Provider",
    "ToolCall",
    "Usage",
    "AnthropicProvider",
    "MockProvider",
    "OpenAIProvider",
    "build_provider",
    "build_providers",
]


def build_provider(config: ProviderConfig) -> Provider:
    if config.kind == "anthropic":
        return AnthropicProvider(config)
    if config.kind == "mock":
        return MockProvider(config.name)
    return OpenAIProvider(config)


def build_providers(config: Config) -> Dict[str, Provider]:
    return {
        name: build_provider(provider)
        for name, provider in config.providers.items()
        if provider.available()
    }
