"""The scope of one serial diagnostic check.

This is the request/ticket half of the `diagnostic_providers/v1` contract:
the changed-file scope a check is asked over, and the ticket that carries the
selected provider, the in-scope/out-of-scope partition of that scope, and
the config identity the check runs under.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diagnostic_provider_models import GLOBAL_MAX_FILES_PER_CHECK, _checked_ref
from .diagnostic_provider_parse import _normalized_files


@dataclass(frozen=True)
class DiagnosticCheckRequest:
    """A serial check over a changed-file scope, at one revision.

    The scope is the changed files only: a diagnostic check exists to answer
    what an edit introduced, and the request refuses paths that escape the
    workspace for the same reason the v1 record does.
    """

    workspace_id: str
    revision: str
    changed_files: tuple[str, ...]

    def __post_init__(self) -> None:
        workspace_id = _checked_ref(self.workspace_id, field="diagnostic_providers workspace_id", required=True)
        revision = _checked_ref(self.revision, field="diagnostic_providers revision", required=True)
        changed_files = _normalized_files(self.changed_files, field="changed_files", bound=GLOBAL_MAX_FILES_PER_CHECK)
        object.__setattr__(self, "workspace_id", workspace_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "changed_files", changed_files)


@dataclass(frozen=True)
class DiagnosticCheckTicket:
    """The one active check: who runs it, over which files, under which config."""

    workspace_id: str
    revision: str
    provider_id: str
    in_scope_files: tuple[str, ...]
    out_of_scope_files: tuple[str, ...]
    config_identity: str
    check_number: int
