"""Pure, versioned preflight checks for every OMH boundary that spawns a subprocess.

Backlog item G ("Preflight resolution before any subprocess spawn"): everything
that CAN be resolved before a subprocess is spawned MUST be, so a doomed spawn
never happens. Each check here answers one narrow, local question --- does an
executable resolve, does a working directory exist, is a declared environment
prerequisite set, does a declared credential file exist --- and never reads
credential *content*, never makes a network call, and never spawns anything
itself. A caller composes the checks relevant to its own boundary and passes
them to `run_spawn_preflight`, which returns one versioned verdict payload a
caller can cite in a refusal instead of attempting a spawn that was always
going to fail.

Every failed check carries its own remedy: what a person or wrapper would do
to make the check pass. A verdict with `ready=False` is a structured refusal,
not an exception dressed up after the fact --- the whole point is that nothing
was spawned to produce it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

SPAWN_PREFLIGHT_SCHEMA_VERSION: Final = "omh_spawn_preflight_verdict/v1"
SPAWN_PREFLIGHT_CLAIM_BOUNDARY: Final = (
    "A spawn preflight verdict says only whether the checked prerequisites were observed present "
    "before a subprocess was spawned. It is not dispatch, execution, verification, review, CI, or "
    "merge evidence, and a ready verdict is not proof the spawned process will succeed."
)

CHECK_EXECUTABLE_PRESENCE: Final = "executable_presence"
CHECK_WORKING_DIRECTORY: Final = "working_directory"
CHECK_ENV_PREREQUISITE: Final = "env_prerequisite"
CHECK_CREDENTIAL_FILE_PRESENCE: Final = "credential_file_presence"


def _check(name: str, passed: bool, detail: str, remedy: str) -> dict[str, object]:
    return {"check": name, "passed": passed, "detail": detail, "remedy": "" if passed else remedy}


def check_executable_present(
    command: str,
    *,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, object]:
    """Whether `command` resolves to a runnable file before it is ever spawned.

    An absolute path or one carrying a path separator is checked directly
    (exists, is a file, is executable); a bare name is resolved on PATH the
    same way the eventual spawn would resolve it, so this reports the
    identical binary a spawn would use rather than a different PATH entry.
    """
    if not command:
        return _check(
            CHECK_EXECUTABLE_PRESENCE, False, "no executable was named", "name the executable to spawn"
        )
    has_separator = "/" in command or "\\" in command
    if has_separator or Path(command).is_absolute():
        candidate = Path(command).expanduser()
        if not candidate.is_file():
            return _check(
                CHECK_EXECUTABLE_PRESENCE,
                False,
                f"`{command}` does not exist",
                f"install the executable, or point at a valid path instead of `{command}`",
            )
        if os.name != "nt" and not os.access(candidate, os.X_OK):
            return _check(
                CHECK_EXECUTABLE_PRESENCE,
                False,
                f"`{command}` exists but is not executable",
                f"grant execute permission on `{command}` (chmod +x)",
            )
        return _check(CHECK_EXECUTABLE_PRESENCE, True, f"`{command}` is a runnable file", "")
    resolved = which(command)
    if not resolved:
        return _check(
            CHECK_EXECUTABLE_PRESENCE,
            False,
            f"`{command}` was not found on PATH",
            f"install `{command}` or add it to PATH",
        )
    return _check(CHECK_EXECUTABLE_PRESENCE, True, f"`{command}` resolves to `{resolved}`", "")


def check_working_directory(path: str | Path | None) -> dict[str, object]:
    """Whether the requested spawn working directory exists and is a directory.

    `None` passes: a spawn with no explicit cwd override inherits the parent
    process's own working directory, which always exists, so there is nothing
    to check.
    """
    if path is None:
        return _check(CHECK_WORKING_DIRECTORY, True, "no working directory override was requested", "")
    candidate = Path(path)
    if not candidate.exists():
        return _check(
            CHECK_WORKING_DIRECTORY,
            False,
            f"working directory `{candidate}` does not exist",
            f"create `{candidate}`, or point the working directory at one that exists",
        )
    if not candidate.is_dir():
        return _check(
            CHECK_WORKING_DIRECTORY,
            False,
            f"`{candidate}` exists but is not a directory",
            f"point the working directory at a directory, not `{candidate}`",
        )
    return _check(CHECK_WORKING_DIRECTORY, True, f"`{candidate}` is a directory", "")


def check_env_any_present(names: Sequence[str], *, env: Mapping[str, str]) -> dict[str, object]:
    """Whether at least one of `names` is set to a non-empty value in `env`.

    An empty `names` sequence means the boundary declared no environment
    prerequisite for this spawn, and passes -- there is nothing to require.
    Checking existence only: the value itself is never logged or returned.
    """
    if not names:
        return _check(CHECK_ENV_PREREQUISITE, True, "no environment prerequisite was declared", "")
    present = [name for name in names if env.get(name)]
    if present:
        return _check(CHECK_ENV_PREREQUISITE, True, f"{present[0]} is set", "")
    joined = ", ".join(names)
    return _check(
        CHECK_ENV_PREREQUISITE,
        False,
        f"none of the required environment variables are set: {joined}",
        f"set one of {joined} before dispatching",
    )


def check_credential_file_present(path: str | Path | None) -> dict[str, object]:
    """Whether a named credential file exists. Existence only -- content is never read.

    `None` or an empty path passes: no credential file was declared for this
    boundary.
    """
    if not path:
        return _check(CHECK_CREDENTIAL_FILE_PRESENCE, True, "no credential file was declared", "")
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        return _check(
            CHECK_CREDENTIAL_FILE_PRESENCE,
            False,
            f"credential file `{candidate}` does not exist",
            f"place the credential file at `{candidate}` before dispatching",
        )
    return _check(CHECK_CREDENTIAL_FILE_PRESENCE, True, f"`{candidate}` exists", "")


def run_spawn_preflight(checks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Combine individual check results into one versioned, citable verdict.

    Every check that ran is carried in `checks` (passed and failed alike), so
    a caller can show the whole picture; `failed_checks` is the subset a
    refusal cites. `ready` is the single boolean a spawn boundary gates on.
    """
    materialized = [dict(item) for item in checks]
    failed = [item for item in materialized if not item.get("passed")]
    return {
        "schema_version": SPAWN_PREFLIGHT_SCHEMA_VERSION,
        "status": "blocked" if failed else "ready",
        "ready": not failed,
        "checks": materialized,
        "failed_checks": failed,
        "claim_boundary": SPAWN_PREFLIGHT_CLAIM_BOUNDARY,
    }
