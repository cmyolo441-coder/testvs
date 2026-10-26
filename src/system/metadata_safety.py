from __future__ import annotations

import re


_OPAQUE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,159}$")
_SENSITIVE_MARKERS = (
    "secret",
    "token",
    "password",
    "private-key",
    "api_key",
    "apikey",
    "credential",
    "authorization",
    "bearer",
)
_COMMON_SECRET_PREFIX = re.compile(
    r"(?:^|[^a-z0-9])(?:sk|pk|rk|xox[a-z]?)[_-][a-z0-9_-]{8,}",
    re.IGNORECASE,
)
_PLATFORM_SECRET_PREFIX = re.compile(
    r"(?:^|[^a-z0-9])(?:gh[oprs]|github_pat|npm|whsec|glpat)[_-][a-z0-9_-]{16,}",
    re.IGNORECASE,
)
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_GOOGLE_API_KEY = re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b", re.IGNORECASE)
# Bounded metadata is one short line. A line break, a tab, a NUL, or a fence
# marker means the value is carrying a body -- a code block, a raw prompt, a
# pasted log -- and a rule evaluation must never receive one of those.
_BODY_SHAPED = re.compile(r"[\r\n\t\f\v\x00]|```")
# Raw PII: a phone number or an email address is a person, not an opaque
# platform reference. E.164-ish international, and national shapes with
# separators, plus the classic local@domain email shape.
_PHONE_SHAPED = re.compile(
    r"^\+?\d[\d ().-]{6,}\d$"
)
_EMAIL_SHAPED = re.compile(
    r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
)


def is_secret_value_shaped(value: str) -> bool:
    """True when the text looks like a credential *value* rather than a name.

    The value half of `is_sensitive_metadata_text`, split out because the two
    halves answer different questions. The marker words below flag anything
    *named* like a secret -- `GITHUB_TOKEN`, `token_store.py` -- which is right
    for a free-form opaque reference and wrong for a field whose whole content
    is an environment variable *name*, since such a name legitimately contains
    them. These four patterns flag the issued material itself, and a name can
    never match one, so a field that must accept `GITHUB_TOKEN` while refusing
    an issued `AKIA`-prefixed access key id reads this predicate instead of
    the wider one.

    No new pattern is introduced here: these are exactly the detectors this
    module already owned, which also bounds what the screen can catch. A
    credential value that none of them recognises and that also matches the
    caller's own name shape is not caught by either predicate.
    """
    return (
        bool(_COMMON_SECRET_PREFIX.search(value))
        or bool(_PLATFORM_SECRET_PREFIX.search(value))
        or bool(_AWS_ACCESS_KEY.search(value))
        or bool(_GOOGLE_API_KEY.search(value))
    )


def is_sensitive_metadata_text(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _SENSITIVE_MARKERS) or is_secret_value_shaped(value)


def redact_metadata_text(value: str, *, limit: int) -> str:
    if is_sensitive_metadata_text(value):
        return "[redacted]"
    return value[:limit]


def is_body_shaped_metadata_text(value: str, *, limit: int) -> bool:
    """True when the text is a body rather than one bounded metadata line."""
    return len(value) > limit or _BODY_SHAPED.search(value) is not None


def is_raw_pii_shaped(value: str) -> bool:
    """True when the text is a raw phone number or email address.

    A platform reference must be an opaque identifier the platform issued,
    never a raw person identifier typed in by hand. These two detectors flag
    the shapes a hand-typed identifier takes -- digits with phone punctuation,
    or local@domain -- so callers can reject them before the value leaves the
    process.
    """
    return bool(_PHONE_SHAPED.fullmatch(value)) or bool(_EMAIL_SHAPED.fullmatch(value))


def require_opaque_metadata_ref(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _OPAQUE_REFERENCE.fullmatch(value) or is_sensitive_metadata_text(value):
        raise ValueError(f"{field} must be a safe opaque metadata reference")
    return value
