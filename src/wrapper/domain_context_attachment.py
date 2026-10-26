from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from ..routing.domain_context_eligibility import classify_domain_context_eligibility
from ..skills.expert_question_rendering import domain_expert_question_body
from ..workflows.domain_project_context import HostProjectBinding, bind_session_project
from ..workflows.domain_routing_context import (
    MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS,
    DomainRoutingResolution,
)
from .localized_copy import detect_copy_locale


HostProjectBindingFactory = Callable[[], HostProjectBinding | None]


class DomainContextResolver(Protocol):
    def __call__(
        self,
        binding: HostProjectBinding,
        message: str,
        *,
        locale: str,
    ) -> DomainRoutingResolution: ...


def rendered_domain_context_fragment(
    resolved_context: dict[str, object] | None,
    chat_response: dict[str, object],
) -> dict[str, object] | None:
    """Keep context only when its catalog question is the rendered body."""
    if not isinstance(resolved_context, dict):
        return None
    context = resolved_context.get("domain_routing_context")
    if not isinstance(context, dict):
        return None
    expected_body = domain_expert_question_body(context)
    if expected_body is None or chat_response.get("body") != expected_body:
        return None
    return {"domain_routing_context": context}


def resolve_attached_domain_context(
    route_payload: dict[str, object],
    message: str,
    *,
    host_project_binding_factory: HostProjectBindingFactory | None,
    resolver: DomainContextResolver,
) -> dict[str, object] | None:
    """Resolve advisory context only after the ordinary route is eligible."""
    eligibility = classify_domain_context_eligibility(route_payload, message)
    if not eligibility.eligible:
        return None
    if len(message) > MAX_DOMAIN_CONTEXT_INPUT_CODE_POINTS:
        return None

    binding = (
        host_project_binding_factory()
        if host_project_binding_factory is not None
        else None
    )
    try:
        if binding is None:
            return None
        resolution = resolver(binding, message, locale=detect_copy_locale(message))
        if resolution.status != "applied" or resolution.reason != "applied":
            return None
        return resolution.context
    finally:
        if binding is not None:
            binding.close()


def build_session_project_binding_factory(
    host_project_binding_factory: HostProjectBindingFactory | None,
) -> HostProjectBindingFactory:
    """Build a factory that reopens project scope for one wrapper turn."""

    def session_project_binding() -> HostProjectBinding | None:
        host_binding = (
            host_project_binding_factory()
            if host_project_binding_factory is not None
            else None
        )
        try:
            return bind_session_project(host_binding)
        finally:
            if host_binding is not None:
                host_binding.close()

    return session_project_binding
