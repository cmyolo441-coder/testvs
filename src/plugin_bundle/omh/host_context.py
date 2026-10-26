from __future__ import annotations

from contextvars import ContextVar


_ACTIVE_MAIN_AGENT_MODEL: ContextVar[str] = ContextVar(
    "omh_active_main_agent_model",
    default="",
)


def record_active_main_agent_model(model: object) -> None:
    normalized = " ".join(str(model or "").split())
    _ = _ACTIVE_MAIN_AGENT_MODEL.set(normalized[:180])


def active_main_agent_model() -> str:
    return _ACTIVE_MAIN_AGENT_MODEL.get()


__all__ = ["active_main_agent_model", "record_active_main_agent_model"]
