"""Project instruction files (``AGENTS.md``) — real loading, real precedence.

FullAgent reads the same kind of project guide other terminal agents use,
so a repo can describe its own conventions once and have the model honour
them. Loading is deliberately boring and observable:

* ``~/.fullagent/AGENTS.md``       — user-wide instructions (weakest)
* ``<ancestor>/AGENTS.md``         — from the filesystem root down to cwd
* ``<cwd>/AGENTS.md``              — nearest project file (strongest)

Every existing file is included, in that order, each with its real path as
a heading so the model can cite where a rule came from. Missing or
unreadable files are skipped silently (a broken checkout must never take
the agent down). Output is capped so a giant file cannot crowd out the
actual conversation.

``collect()`` caches by (path, mtime_ns, size) — the file is only re-read
when it actually changes.

``python -m fullagent.instructions`` runs the built-in self-test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

FILE_NAME = "AGENTS.md"
MAX_CHARS = 12_000

# (path, mtime_ns, size) -> body for that file
_CACHE: dict[tuple[str, int, int], str] = {}
_CACHE_LIMIT = 64


def _read_cached(path: Path) -> str:
    """Read *path* through the mtime/size cache ("" when unreadable)."""
    try:
        st = path.stat()
    except OSError:
        return ""
    key = (str(path), int(st.st_mtime_ns), int(st.st_size))
    if key in _CACHE:
        return _CACHE[key]
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    text = text.strip()
    if len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.clear()
    _CACHE[key] = text
    return text


def candidate_files(
    cwd: str | os.PathLike | None = None,
    extra_files: Sequence[str | os.PathLike] = (),
    user_dir: str | os.PathLike | None = None,
) -> list[Path]:
    """Instruction files for *cwd*, weakest first (global → root → cwd)."""
    base = Path(cwd or os.getcwd()).expanduser().resolve()
    found: list[Path] = []

    if user_dir is None:
        home = os.environ.get("FULLAGENT_HOME") or (Path.home() / ".fullagent")
        user_dir = Path(home).expanduser()
    try:
        found.append(Path(user_dir) / FILE_NAME)
    except OSError:
        pass

    ancestors: list[Path] = []
    for parent in [base, *base.parents]:
        candidate = parent / FILE_NAME
        try:
            if candidate.is_file():
                ancestors.append(candidate)
        except OSError:
            continue
    found.extend(reversed(ancestors))  # root first, cwd last

    for extra in extra_files:
        try:
            found.append(Path(extra).expanduser())
        except (OSError, TypeError):
            continue
    return found


def collect(
    cwd: str | os.PathLike | None = None,
    extra_files: Sequence[str | os.PathLike] = (),
    user_dir: str | os.PathLike | None = None,
    limit: int = MAX_CHARS,
) -> str:
    """Combined project instructions, capped at *limit* characters.

    Returns "" when no instruction file exists. Each included file is
    introduced by its real path so the model can name its source.
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for path in candidate_files(cwd, extra_files, user_dir):
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path)
        if resolved in seen:
            continue
        body = _read_cached(path)
        if not body:
            continue
        seen.add(resolved)
        blocks.append(f"### {resolved}\n{body}")
    if not blocks:
        return ""
    text = "\n\n".join(blocks)
    if limit and len(text) > limit:
        text = text[:max(0, limit - 20)].rstrip() + "\n… (truncated)"
    return text


def describe(cwd: str | os.PathLike | None = None,
             extra_files: Sequence[str | os.PathLike] = (),
             user_dir: str | os.PathLike | None = None) -> list[str]:
    """Real paths that currently carry instructions (for /help-style views)."""
    out: list[str] = []
    for path in candidate_files(cwd, extra_files, user_dir):
        if _read_cached(path):
            try:
                out.append(str(path.resolve()))
            except OSError:
                out.append(str(path))
    return out


# ---------------------------------------------------------------------------
# Module self-test: real files, real precedence, real cache invalidation
# ---------------------------------------------------------------------------

def _selftest() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="fullagent-instructions-") as d:
        root = Path(d)
        project = root / "project"
        nested = project / "src" / "pkg"
        nested.mkdir(parents=True)
        user = root / "home"
        user.mkdir()

        (user / FILE_NAME).write_text("user-wide rule\n", encoding="utf-8")
        (project / FILE_NAME).write_text("project rule: run tests\n",
                                         encoding="utf-8")
        (nested / FILE_NAME).write_text("nearest rule: use tabs\n",
                                        encoding="utf-8")

        files = [str(p) for p in candidate_files(nested, user_dir=user)]
        assert files[-1].endswith("pkg/AGENTS.md"), files
        assert files[-2].endswith("project/AGENTS.md"), files
        assert any(f.endswith("home/AGENTS.md") for f in files), files

        text = collect(nested, user_dir=user)
        assert "user-wide rule" in text
        assert "project rule: run tests" in text
        assert "nearest rule: use tabs" in text
        assert text.index("user-wide rule") < text.index("project rule")
        assert text.index("project rule") < text.index("nearest rule")

        # instruction-free directory yields nothing
        bare = root / "bare"
        bare.mkdir()
        assert collect(bare, user_dir=root / "empty-home") == ""

        # edits are picked up (mtime/size cache)
        target = nested / FILE_NAME
        target.write_text("nearest rule: spaces\n", encoding="utf-8")
        assert "spaces" in collect(nested, user_dir=user)

        # the cap is real
        target.write_text("x" * 5000, encoding="utf-8")
        capped = collect(nested, user_dir=user, limit=1200)
        assert len(capped) <= 1200, len(capped)
        assert capped.endswith("(truncated)")

        assert describe(nested, user_dir=user)

    print("instructions self-test PASS: precedence, cache, cap")


if __name__ == "__main__":
    _selftest()
