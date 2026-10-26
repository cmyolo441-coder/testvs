"""PATH-only detection of the optional ast-grep structural search tool.

OMH never executes ast-grep: presence on PATH is the whole observation,
matching the ``probe_buzz`` claim boundary and ``EXECUTOR_VERSION_POLICY``
("OMH pins no executor CLI version."). Doctor and probe consume this
inspection; absence is the normal case and never degrades either surface.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable

STRUCTURAL_SEARCH_TOOL = "ast-grep"
STRUCTURAL_SEARCH_CLAIM_BOUNDARY = (
    "This inspection resolves the binary on PATH only. It does not execute "
    "ast-grep, read its version, or prove it can parse this repository."
)


def inspect_structural_search(*, which: Callable[[str], str | None] | None = None) -> dict[str, object]:
    """Resolve ast-grep on PATH without executing it.

    The default resolves ``shutil.which`` at CALL time (a def-time default
    would freeze the binding and make the probe untestable/unpatchable).
    """
    resolved_which = shutil.which if which is None else which
    path = resolved_which(STRUCTURAL_SEARCH_TOOL)
    return {
        "tool": STRUCTURAL_SEARCH_TOOL,
        "found": path is not None,
        "path": path,
        "observed": True,
        "read_only": True,
        "reason_code": "ast_grep_present_not_executed" if path is not None else "ast_grep_missing",
        "claim_boundary": STRUCTURAL_SEARCH_CLAIM_BOUNDARY,
    }
