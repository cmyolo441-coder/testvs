"""Error types used across OMH.

Everything the user can see derives from :class:`OmhError` so the CLI can
render one clean line instead of a traceback.
"""

from __future__ import annotations

from typing import Optional


class OmhError(Exception):
    """Base class for expected, user-facing failures."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class ConfigError(OmhError):
    """The configuration file or an environment override is invalid."""


class ProviderError(OmhError):
    """A model provider rejected or failed a request."""

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
        hint: str = "",
        retryable: bool = True,
    ) -> None:
        super().__init__(message, hint=hint)
        self.status = status
        self.retry_after = retry_after
        self.retryable = retryable


class AuthError(ProviderError):
    """Credentials are missing, invalid or forbidden."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message, status=401, hint=hint, retryable=False)


class RateLimitError(ProviderError):
    """The provider asked us to slow down."""

    def __init__(self, message: str, *, retry_after: Optional[float] = None) -> None:
        super().__init__(message, status=429, retry_after=retry_after)


class StreamInterrupted(ProviderError):
    """The winning stream failed after it had already emitted content."""


class ToolError(OmhError):
    """A tool could not run; the message is fed back to the model."""


class ApprovalDenied(ToolError):
    """The user declined to run a tool."""
