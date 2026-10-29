"""REVIEW — heuristic code-change reviewer, no LLM needed.

Two entry points over a unified diff:

    review_diff(diff_text) -> list[dict]
        Parse a unified diff (stdlib only) and return per-file findings of
        the shape {file, line, severity, message, suggestion}. Severities are
        "critical" | "high" | "medium" | "low".

    review_git(repo=".", staged=False) -> str
        Run ``git diff`` (optionally ``--staged``) and format the findings as
        a markdown report. Prints "No changes to review" when the diff is
        empty.

    review_files(paths) -> str
        ``git diff -- <paths>`` for the given paths; when the current
        directory is not a git repo, each file is read directly and the whole
        content is reviewed as if it were all added.

Heuristics are deliberately cheap, static, and deterministic:

    secret literals (api_key / password / token / secret assignments) -> critical
    bare ``except:``                                            -> high
    SQL string concatenation (execute(f"...") / +" SELECT")     -> high
    ``shell=True``                                              -> high
    ``open(...)`` with no error handling in sight               -> medium
    very long hunks (>80 added lines in one hunk)                -> medium
    TODO / FIXME markers left in new code                       -> low
    print-debug leftovers (print, pprint, console.log, pdb...)  -> low

The agent-facing tool is ``ReviewChanges`` (args: ``path="."``,
``staged=false``); the TUI slash command is ``/review [--staged] [paths…]``.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._foundation import get_logger

_log = get_logger("review")

# ---------------------------------------------------------------------------
# Heuristic patterns (precompiled once — review_diff runs per added line)
# ---------------------------------------------------------------------------

#: Secret-looking assignments: api_key="...", password = '...', token: "..."
_SECRET_RX = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?secret|secret[_-]?key|password|passwd|pwd|"
    r"auth[_-]?token|access[_-]?token|private[_-]?key)\b"
    r"\s*[:=]\s*(['\"])((?!\2).{3,})\2"
)

_BARE_EXCEPT_RX = re.compile(r"^\s*except\s*:")

#: execute(f"SELECT ...") / cursor.execute("..." + var) style string SQL
_SQL_CONCAT_RX = re.compile(
    r"(?i)\bexecute\s*\(\s*(f['\"]|['\"][^'\"]*(?:\+|\{))"
)

_SHELL_TRUE_RX = re.compile(r"\bshell\s*=\s*True\b")

#: crude open() detector; the missing-error-handling check is contextual
_OPEN_RX = re.compile(r"(?i)\bopen\s*\(")

_TODO_RX = re.compile(r"(?i)\b(TODO|FIXME|XXX|HACK)\b")

_DEBUG_RX = re.compile(
    r"^\s*(print\s*\(|pprint\s*\(|console\.log\s*\(|debugger;?|"
    r"import\s+pdb|pdb\.set_trace\s*\()"
)

# A hunk adding more than this many lines gets flagged as "too long".
LONG_HUNK_LINES = 80

# ---------------------------------------------------------------------------
# Unified-diff parsing (stdlib only)
# ---------------------------------------------------------------------------

_HUNK_RX = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NEWFILE_RX = re.compile(r"^\+\+\+ b/(.+)$")


@dataclass
class _AddedLine:
    file: str
    line: int
    text: str


def _parse_diff(diff_text: str) -> tuple[list[_AddedLine], list[tuple[str, int]]]:
    """Split a unified diff into added lines and (file, added-count) hunks.

    Returns (added_lines, hunks) where hunks entries are (file, added_in_hunk).
    """
    added: list[_AddedLine] = []
    hunks: list[tuple[str, int]] = []
    cur_file = ""
    new_line = 0
    hunk_added = 0
    hunk_file = ""
    in_hunk = False
    old_left = new_left = 0

    for raw in (diff_text or "").splitlines():
        if raw.startswith("diff --git "):
            in_hunk = False
            continue
        # ---/+++ are file headers only BETWEEN hunks; inside a hunk a
        # data line whose text starts with "++ " / "-- " produces the
        # same prefix and must not be mistaken for a header.
        if not in_hunk and raw.startswith("+++ "):
            m = _NEWFILE_RX.match(raw)
            cur_file = m.group(1).strip() if m else raw[4:].strip()
            continue
        if not in_hunk and raw.startswith("--- "):
            continue
        m = _HUNK_RX.match(raw)
        if m:
            if hunk_file:
                hunks.append((hunk_file, hunk_added))
            new_line = int(m.group(3))
            old_left = int(m.group(2)) if m.group(2) is not None else 1
            new_left = int(m.group(4)) if m.group(4) is not None else 1
            in_hunk = old_left > 0 or new_left > 0
            hunk_added = 0
            hunk_file = cur_file
            continue
        if not cur_file:
            continue
        if raw.startswith("+"):
            added.append(_AddedLine(cur_file, new_line, raw[1:]))
            new_line += 1
            hunk_added += 1
            if in_hunk and new_left > 0:
                new_left -= 1
        elif raw.startswith("-"):
            # removed line: new-file line numbers don't advance
            if in_hunk and old_left > 0:
                old_left -= 1
        elif raw.startswith("\\"):
            pass  # "\ No newline at end of file" is not a context line
        else:
            new_line += 1  # context line
            if in_hunk:
                old_left = max(0, old_left - 1)
                new_left = max(0, new_left - 1)
        if in_hunk and old_left <= 0 and new_left <= 0:
            in_hunk = False
    if hunk_file:
        hunks.append((hunk_file, hunk_added))
    return added, hunks


def _finding(file: str, line: int, severity: str,
             message: str, suggestion: str) -> dict:
    return {"file": file, "line": line, "severity": severity,
            "message": message, "suggestion": suggestion}


def review_diff(diff_text: str) -> list[dict]:
    """Review a unified diff, returning per-line findings.

    Each finding: {file, line, severity, message, suggestion}.
    """
    added, hunks = _parse_diff(diff_text)
    findings: list[dict] = []

    for a in added:
        f, n, t = a.file, a.line, a.text

        if _SECRET_RX.search(t):
            findings.append(_finding(
                f, n, "critical", "Possible hardcoded secret",
                "Move the secret to an environment variable or a secret "
                "manager; never commit credentials."))
            continue  # one flag per line is enough

        if _BARE_EXCEPT_RX.match(t):
            findings.append(_finding(
                f, n, "high", "Bare `except:` swallows every exception",
                "Catch specific exceptions (e.g. `except OSError:`) so real "
                "bugs surface instead of being silenced."))

        if _SQL_CONCAT_RX.search(t):
            findings.append(_finding(
                f, n, "high", "SQL built via string interpolation/concat",
                "Use parameterized queries (`?` / `%s` placeholders) to "
                "avoid SQL injection."))

        if _SHELL_TRUE_RX.search(t):
            findings.append(_finding(
                f, n, "high", "`shell=True` in subprocess call",
                "Pass argv as a list without `shell=True`; if a shell is "
                "truly needed, sanitize/quote all inputs with shlex.quote."))

        if _OPEN_RX.search(t):
            findings.append(_finding(
                f, n, "medium", "`open()` with no visible error handling",
                "Wrap I/O in try/except (OSError) or let a helper raise a "
                "clear error; handle missing/permission failures."))

        if _TODO_RX.search(t):
            findings.append(_finding(
                f, n, "low", "TODO/FIXME marker left in new code",
                "Resolve it now or file a tracked task; stale TODOs rot."))

        if _DEBUG_RX.search(t):
            findings.append(_finding(
                f, n, "low", "Debug print / debugger leftover",
                "Remove before committing; use the logging framework for "
                "permanent output."))

    for hfile, count in hunks:
        if count > LONG_HUNK_LINES:
            findings.append(_finding(
                hfile, 0, "medium",
                f"Very long hunk: {count} added lines in one hunk",
                "Split the change into smaller, reviewable pieces (one "
                "concern per commit)."))

    # Deterministic order: file, then line.
    findings.sort(key=lambda d: (d["file"], d["line"]))
    return findings


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_SEV_EMOJI = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "⚪"}


def format_report(findings: list[dict], title: str = "Code review") -> str:
    """Render findings as a markdown report with severity summary counts."""
    lines = [f"# {title}", ""]
    if not findings:
        lines.append("No changes to review.")
        return "\n".join(lines) + "\n"

    counts: dict[str, int] = {}
    for d in findings:
        counts[d["severity"]] = counts.get(d["severity"], 0) + 1
    summary = ", ".join(
        f"{_SEV_EMOJI[s]} {s}: {counts[s]}" for s in _SEV_ORDER
        if s in counts)
    lines.append(f"**{len(findings)} finding(s)** — {summary}")
    lines.append("")

    ordered = sorted(findings,
                     key=lambda d: (_SEV_ORDER.get(d["severity"], 9),
                                    d["file"], d["line"]))
    cur_sev = None
    for d in ordered:
        if d["severity"] != cur_sev:
            cur_sev = d["severity"]
            lines.append(f"## {_SEV_EMOJI[cur_sev]} {cur_sev.upper()}")
        loc = f"{d['file']}:{d['line']}" if d["line"] else d["file"]
        lines.append(f"- `{loc}` — {d['message']}")
        lines.append(f"  - Suggestion: {d['suggestion']}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# git-backed entry points
# ---------------------------------------------------------------------------

def _git_diff(args: list[str], repo: str) -> str | None:
    """Run git diff; return None when repo isn't a git working tree."""
    try:
        p = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        _log.warning("review: git diff failed: %s", e)
        return None
    if p.returncode != 0:
        _log.warning("review: git diff rc=%s: %s", p.returncode,
                     p.stderr.strip()[:200])
        return None
    return p.stdout


def review_git(repo: str = ".", staged: bool = False) -> str:
    """Review uncommitted changes in *repo* (`--staged` for the index)."""
    args = ["diff", "--no-color", "--no-ext-diff"]
    if staged:
        args.append("--staged")
    diff = _git_diff(args, repo)
    if diff is None:
        return "error: not a git repository (or git failed)"
    if not diff.strip():
        return "No changes to review."
    title = "Code review (staged changes)" if staged else "Code review"
    return format_report(review_diff(diff), title)


def review_files(paths: list[str]) -> str:
    """Review specific paths: `git diff -- <paths>`, else file contents.

    When the cwd is not a git repo, each file is read directly and its
    whole content is reviewed as newly-added lines.
    """
    paths = [str(p) for p in paths if str(p).strip()]
    if not paths:
        return review_git(".")

    diff = _git_diff(["diff", "--no-color", "--no-ext-diff", "--", *paths],
                     ".")
    if diff is not None:
        if not diff.strip():
            return "No changes to review."
        return format_report(review_diff(diff), "Code review")

    # Not a git repo: review raw file contents directly.
    findings: list[dict] = []
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            findings.append(_finding(p, 0, "low", "Path is not a file",
                                     "Pass existing file paths."))
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            findings.append(_finding(p, 0, "low", f"Could not read file: {e}",
                                     "Check permissions and retry."))
            continue
        lines = text.splitlines()
        fake_diff = (f"--- /dev/null\n+++ b/{p}\n"
                     + f"@@ -0,0 +1,{len(lines)} @@\n"
                     + "".join(f"+{ln}\n" for ln in lines))
        findings.extend(review_diff(fake_diff))
    return format_report(findings, "Code review")


# ---------------------------------------------------------------------------
# Agent tool registration (duck-typed — never import .agent)
# ---------------------------------------------------------------------------

def _handle_review_changes(path: str = ".", staged: bool = False) -> str:
    """ReviewChanges tool handler: path=".", staged=false."""
    if path and path != ".":
        return review_files([path])
    return review_git(".", staged=bool(staged))


def register(agent: Any) -> None:
    """Register the ReviewChanges tool on *agent* (expects .tools dict)."""
    from .tools import Tool  # local import: avoids any import-time cycles

    agent.tools["ReviewChanges"] = Tool(
        "ReviewChanges",
        "Review code changes with heuristic severity findings "
        "(critical/high/medium/low): hardcoded secrets, bare except, SQL "
        "concat, shell=True, I/O without error handling, long hunks, "
        "TODO/FIXME, debug prints. Args: path (default '.' = all "
        "uncommitted changes), staged (default false).",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "staged": {"type": "boolean"}},
         "required": []},
        _handle_review_changes,
    )


# ---------------------------------------------------------------------------
# /review TUI command (duck-typed ui — never import .tui)
# ---------------------------------------------------------------------------

def run_review_command(ui: Any, arg: str) -> None:
    """Handle the `/review [--staged] [paths...]` slash command.

    *ui* must provide print_info / print_error. Flags and paths may appear
    in any order; `--staged` reviews the index, anything else is a path.
    """
    staged = False
    paths: list[str] = []
    for tok in (arg or "").split():
        if tok == "--staged":
            staged = True
        else:
            paths.append(tok)
    try:
        if paths:
            report = review_files(paths)
        else:
            report = review_git(".", staged=staged)
    except Exception as e:  # never crash the TUI on a review
        ui.print_error(f"/review failed: {e}")
        return
    ui.print_info(report, "cyan")


# ---------------------------------------------------------------------------
# Self-test: `python3 -m fullagent.review` → PASS
# ---------------------------------------------------------------------------

def _selftest() -> None:
    # Crafted diff: a hardcoded secret + a bare except + an 85-line hunk.
    long_body = "\n".join(f"+    x{i} = {i}" for i in range(85))
    diff = (
        "diff --git a/app/db.py b/app/db.py\n"
        "index 1111111..2222222 100644\n"
        "--- a/app/db.py\n"
        "+++ b/app/db.py\n"
        "@@ -1,3 +1,6 @@\n"
        " import os\n"
        "+api_key = \"sk-live-abc123xyz\"\n"
        " config = load()\n"
        "@@ -10,4 +13,7 @@\n"
        " def fetch():\n"
        "+    try:\n"
        "+        return query()\n"
        "+    except:\n"
        "+        pass\n"
        "     return None\n"
        "diff --git a/app/big.py b/app/big.py\n"
        "index 3333333..4444444 100644\n"
        "--- a/app/big.py\n"
        "+++ b/app/big.py\n"
        "@@ -1,2 +1,87 @@\n"
        " import sys\n"
        f"{long_body}\n"
    )

    findings = review_diff(diff)
    by_sev = {f["severity"] for f in findings}
    assert "critical" in by_sev, f"expected critical secret finding: {findings}"
    assert "high" in by_sev, f"expected high bare-except finding: {findings}"
    assert "medium" in by_sev, f"expected medium long-hunk finding: {findings}"

    secret = next(f for f in findings if f["severity"] == "critical")
    assert secret["file"] == "app/db.py" and secret["line"] == 2, secret
    assert all(set(f) == {"file", "line", "severity", "message", "suggestion"}
               for f in findings)

    bare = next(f for f in findings if "Bare" in f["message"])
    assert bare["severity"] == "high" and bare["line"] == 16, bare

    long_hunk = next(f for f in findings if "Very long hunk" in f["message"])
    assert long_hunk["severity"] == "medium"
    assert long_hunk["file"] == "app/big.py", long_hunk

    # Empty diff → no findings; report handles "no changes".
    assert review_diff("") == []
    assert "No changes" in format_report([])

    # ReviewChanges tool wiring via duck-typed agent.
    class FakeAgent:
        def __init__(self):
            self.tools = {}
    a = FakeAgent()
    register(a)
    tool = a.tools["ReviewChanges"]
    assert tool.name == "ReviewChanges"
    out = tool.handler(path=".", staged=False)
    assert isinstance(out, str) and out  # real repo: git diff or error string

    # review_files on a non-git temp dir reviews raw contents directly.
    import tempfile, os
    tmp = tempfile.mkdtemp(prefix="review-selftest-")
    fp = os.path.join(tmp, "s.py")
    with open(fp, "w") as fh:
        fh.write('password = "hunter2"\nprint("debug")\n')
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        rep = review_files([fp])
    finally:
        os.chdir(cwd)
    assert "CRITICAL" in rep and "LOW" in rep, rep

    # /review TUI command prints via duck-typed ui.
    class FakeUI:
        def __init__(self):
            self.infos = []
            self.errors = []
        def print_info(self, msg, color=None):
            self.infos.append(msg)
        def print_error(self, msg):
            self.errors.append(msg)
    ui = FakeUI()
    run_review_command(ui, "")
    assert ui.infos and not ui.errors, (ui.infos, ui.errors)
    ui2 = FakeUI()
    run_review_command(ui2, "--staged")
    assert ui2.infos and not ui2.errors

    print("PASS")


if __name__ == "__main__":
    _selftest()
