"""Verification tiering: sensitive-path escalation.

Absorbs "verification tiering with path escalation": a change that touches a
security-sensitive surface -- authentication, secrets/config, schema or
database migrations, or payment/crypto code -- must force the thorough
verification lane regardless of how small the diff is. Change size is a poor
proxy for risk on these surfaces: a one-line edit to `src/auth/jwt.py` or
`.env.production` can be more consequential than a hundred-line refactor of an
unrelated module, so `_verification` in `coding_delegation.py` calls
`sensitive_path_escalation` on every declared target path rather than sizing
the checklist off the request alone.

Matching is by path component and filename pattern (`fnmatch`, case folded so
the decision does not depend on which case convention the caller used), never
substring search over the whole path string. `src/blog/author.py` must never
escalate on "auth" the way `src/auth/login.py` does, and `docs/environment.md`
must never escalate on "env" the way `.env.production` does -- so directory
patterns are matched against whole path parts and filename patterns are
matched against the basename only, and neither ever runs as a raw substring
check over the full path string.
"""

from __future__ import annotations

from collections.abc import Sequence
import fnmatch
from typing import Final

SENSITIVE_PATH_ESCALATION_SCHEMA_VERSION: Final = "omh_sensitive_path_escalation/v1"

# (category, human label, whole-path-part patterns, basename patterns). A path
# escalates when any of its path parts fnmatch-matches a dir pattern, or its
# basename fnmatch-matches a basename pattern. Every pattern is matched as a
# whole string against one path part or the basename -- never as a substring
# scan over the joined path -- so a part or basename merely containing the
# letters never matches a pattern naming the whole part.
_SENSITIVE_PATH_RULES: Final = (
    (
        "auth",
        "authentication/authorization surface",
        ("auth", "auth-*", "*-auth"),
        ("*jwt*",),
    ),
    (
        "secrets_config",
        "secrets or environment configuration",
        ("secrets", "credentials"),
        (".env", ".env.*", "*credential*"),
    ),
    (
        "schema_migrations",
        "schema or database migration",
        ("migrations", "migration"),
        ("schema.*",),
    ),
    (
        "payment_crypto",
        "payment or cryptographic surface",
        ("payment", "payments", "billing", "crypto"),
        ("*wallet*",),
    ),
)


def sensitive_path_escalation(paths: Sequence[str]) -> dict[str, object] | None:
    """The escalation verdict for a set of changed paths, or None.

    Rules are tried in `_SENSITIVE_PATH_RULES` order, and within a rule paths
    are tried in `paths` order, so the same input always names the same
    category and matched path. Each path is read as a plain string; a
    non-string or blank entry is skipped rather than raising, since this is a
    pure classifier over caller-supplied metadata and not a validator.
    """
    for category, label, dir_patterns, basename_patterns in _SENSITIVE_PATH_RULES:
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                continue
            parts = _path_parts(path)
            if not parts:
                continue
            basename = parts[-1]
            if _matches_any(parts, dir_patterns) or _matches_any((basename,), basename_patterns):
                return {
                    "schema_version": SENSITIVE_PATH_ESCALATION_SCHEMA_VERSION,
                    "category": category,
                    "matched_path": path,
                    "reason": (
                        f"{path} is in the {label}; verification escalates to the thorough "
                        "lane regardless of change size."
                    ),
                }
    return None


def _matches_any(values: Sequence[str], patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(value.lower(), pattern.lower()) for value in values for pattern in patterns)


def _path_parts(path: str) -> tuple[str, ...]:
    """A path as comparable, case-preserved segments, on every host.

    Both separators are folded and empty and `.` segments dropped, matching
    `safety_preflight._path_parts` and `action_gate`'s path handling, so a
    path handed to this classifier reads the same way it reads everywhere else
    in the coding-delegation lane.
    """
    return tuple(part for part in path.replace("\\", "/").split("/") if part not in ("", "."))
