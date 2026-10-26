"""Stateful, serial provider selection with fallback.

One check is active at a time: `begin_check` refuses while a ticket is
outstanding, and `end_check` only accepts the outstanding ticket. A provider
whose last check timed out or crashed is skipped by the next selection -- the
fallback -- until it is marked available again or completes a check itself.
The scheduler never runs the provider; it only decides which allowlisted one
a check belongs to and records what the caller reported about it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .diagnostic_provider_config import DiagnosticProviderConfig, _files_in_scope
from .diagnostic_provider_models import DIAGNOSTIC_PROVIDER_IDS, DiagnosticProviderError
from .diagnostic_provider_outcomes import DiagnosticCheckOutcome, build_diagnostic_check_outcome
from .diagnostic_provider_scope import DiagnosticCheckRequest, DiagnosticCheckTicket


class DiagnosticProviderScheduler:
    """Stateful, serial provider selection with fallback."""

    def __init__(self, config: DiagnosticProviderConfig | None = None) -> None:
        self._config = config if config is not None else DiagnosticProviderConfig()
        self._active_ticket: DiagnosticCheckTicket | None = None
        self._last_provider_id = ""
        self._last_outcome = ""
        self._completed_checks = 0
        self._unavailable_providers: tuple[str, ...] = ()

    @property
    def config(self) -> DiagnosticProviderConfig:
        return self._config

    @property
    def active_provider_id(self) -> str:
        return self._active_ticket.provider_id if self._active_ticket else ""

    @property
    def last_provider_id(self) -> str:
        return self._last_provider_id

    @property
    def last_outcome(self) -> str:
        return self._last_outcome

    @property
    def completed_checks(self) -> int:
        return self._completed_checks

    def begin_check(self, request: DiagnosticCheckRequest) -> DiagnosticCheckTicket:
        if self._active_ticket is not None:
            raise DiagnosticProviderError(
                "diagnostic_providers scheduler is serial: end the active check before beginning another"
            )
        capability = self._config.select_provider(request.changed_files, self._unavailable_providers)
        check_number = self._completed_checks + 1
        if capability is None:
            ticket = DiagnosticCheckTicket(
                request.workspace_id,
                request.revision,
                "",
                (),
                request.changed_files,
                self._config.config_identity(),
                check_number,
            )
        else:
            ticket = DiagnosticCheckTicket(
                request.workspace_id,
                request.revision,
                capability.provider_id,
                _files_in_scope(request.changed_files, capability),
                tuple(
                    path
                    for path in request.changed_files
                    if not path.endswith(capability.file_suffixes)
                ),
                self._config.config_identity(),
                check_number,
            )
        self._active_ticket = ticket
        return ticket

    def end_check(
        self,
        ticket: DiagnosticCheckTicket,
        *,
        terminal_state: str = "completed",
        diagnostics_revision: str = "",
        diagnosed_files: Iterable[str] = (),
        diagnostics: Iterable[Mapping[str, object]] = (),
    ) -> DiagnosticCheckOutcome:
        active = self._active_ticket
        if active is None:
            raise DiagnosticProviderError(
                "diagnostic_providers scheduler has no active check to end"
            )
        if ticket != active:
            raise DiagnosticProviderError(
                "diagnostic_providers scheduler can only end its own active check"
            )
        supplied = tuple(diagnostics)
        capability = self._config.capability_for(ticket.provider_id) if ticket.provider_id else None
        if capability is not None and len(supplied) > capability.max_diagnostics_per_check:
            raise DiagnosticProviderError(
                f"diagnostic_providers {ticket.provider_id} accepts at most "
                f"{capability.max_diagnostics_per_check} diagnostics per check"
            )
        outcome = build_diagnostic_check_outcome(
            workspace_id=ticket.workspace_id,
            revision=ticket.revision,
            diagnostics_revision=diagnostics_revision,
            provider_id=ticket.provider_id,
            terminal_state=terminal_state,
            compatibility="provider_selected" if ticket.provider_id else "provider_disabled",
            in_scope_files=ticket.in_scope_files,
            out_of_scope_files=ticket.out_of_scope_files,
            diagnosed_files=diagnosed_files,
            diagnostics=supplied,
            config_identity=ticket.config_identity,
        )
        self._active_ticket = None
        self._last_provider_id = ticket.provider_id
        self._last_outcome = outcome.outcome
        self._completed_checks += 1
        if ticket.provider_id:
            if outcome.outcome in ("timeout", "crashed"):
                if ticket.provider_id not in self._unavailable_providers:
                    self._unavailable_providers = (*self._unavailable_providers, ticket.provider_id)
            elif outcome.outcome in ("ok", "partial"):
                self._unavailable_providers = tuple(
                    provider_id
                    for provider_id in self._unavailable_providers
                    if provider_id != ticket.provider_id
                )
        return outcome

    def mark_provider_available(self, provider_id: str) -> None:
        """Clear a provider's fallback skip, once its owner says it recovered."""
        if provider_id not in DIAGNOSTIC_PROVIDER_IDS:
            raise DiagnosticProviderError(
                f"diagnostic_providers provider_id is not allowlisted: {provider_id!r}"
            )
        self._unavailable_providers = tuple(
            provider for provider in self._unavailable_providers if provider != provider_id
        )
