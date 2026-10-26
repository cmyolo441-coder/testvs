"""Shared fixtures for the OMH test suite (stdlib unittest only)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from omh.config import Config, ModelProfile, ProviderConfig
from omh.providers.base import Chunk, Provider


def make_config(
    workspace: str | Path,
    *,
    models: Optional[List[ModelProfile]] = None,
    providers: Optional[Dict[str, ProviderConfig]] = None,
    **overrides: Any,
) -> Config:
    providers = providers or {
        "a": ProviderConfig(name="a", api_key="test-key"),
        "b": ProviderConfig(name="b", api_key="test-key"),
    }
    models = models or [
        ModelProfile("model-a", "a", label="a", weight=2.0),
        ModelProfile("model-b", "b", label="b", weight=1.0),
    ]
    defaults: Dict[str, Any] = {
        "workspace": str(workspace),
        "session_dir": str(Path(workspace) / ".sessions"),
        "history_file": str(Path(workspace) / ".history"),
        "providers": providers,
        "models": models,
        "approval_mode": "auto",
        "theme": "hermes",
        "max_steps": 6,
    }
    defaults.update(overrides)
    return Config(**defaults)


class ScriptedProvider(Provider):
    """Replays a script of chunks, callables, exceptions or delays per call."""

    def __init__(self, name: str, script: Iterable[Any]) -> None:
        super().__init__(name)
        self.script = list(script)
        self.calls = 0
        self.seen: List[Dict[str, Any]] = []

    def available(self) -> bool:
        return True

    def stream(self, **kwargs: Any):
        self.calls += 1
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = list(snapshot["messages"])
        self.seen.append(snapshot)
        index = min(self.calls - 1, len(self.script) - 1)
        step = self.script[index]
        if isinstance(step, BaseException):
            raise step
        if callable(step) and not isinstance(step, (list, tuple)):
            yield from step(**kwargs)
            return
        for item in step:
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                time.sleep(item)
                continue
            yield item


def responder(
    *texts: str,
    model: str = "model",
    delay: float = 0.0,
    finish: str = "stop",
) -> Callable[..., Any]:
    """A script step that emits text (optionally with a leading delay)."""

    def run(**_: Any):
        if delay:
            time.sleep(delay)
        for text in texts:
            yield Chunk(text=text, model=model)
        yield Chunk(finish=finish, model=model)

    return run

