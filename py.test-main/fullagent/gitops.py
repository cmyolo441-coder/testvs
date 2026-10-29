"""Git automation: real git operations via the git CLI (subprocess).

No GitPython dependency — everything shells out to ``git``. Six tools,
all running in the agent's current working directory:

* ``GitStatus`` — real ``git status --porcelain`` plus current branch.
* ``GitCommit`` — ``git add -A`` then ``git commit -m <message>``; returns
  the real commit hash. Refuses outright if there is nothing to commit.
* ``GitBranch`` — with a ``name`` arg: ``git checkout -b <name>``.
  Without an arg: lists branches (current one marked ``*``).
* ``GitPush`` — ``git push``; captures real stdout/stderr, including
  errors (e.g. no upstream, auth failure).
* ``GitDiff`` — ``git diff --stat``; with a ``file`` arg: full unified
  diff for that one file.
* ``GitLog`` — last ``n`` commits, one per line.

Every tool first runs ``git rev-parse --git-dir`` in the working
directory and returns a clear error if this is not a git repository.
"""

from __future__ import annotations

import subprocess
from typing import Any

from .tools import (
    RISK_CONFIRM,
    RISK_SAFE,
    Tool,
    _coerce_int,
)

_GIT_TIMEOUT = 30.0


def _run_git(args: list[str], cwd: str | None = None
             ) -> tuple[int, str, str]:
    """Run git and return (returncode, stdout, stderr), stripped.

    Never raises: a missing git binary or an invalid/inaccessible ``cwd``
    used to propagate FileNotFoundError/NotADirectoryError through every
    git tool; a hung git used to raise TimeoutExpired after the tool layer
    had already reported progress. Both now surface as ordinary errors.
    """
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            cwd=cwd or None,
        )
    except subprocess.TimeoutExpired:
        return 124, "", f"git timed out after {_GIT_TIMEOUT:g}s"
    except (OSError, ValueError) as e:
        return 127, "", f"could not run git: {e}"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _check_repo(cwd: str | None = None) -> str | None:
    """Return an ERROR string if cwd is not a git repo, else None."""
    rc, _, err = _run_git(["rev-parse", "--git-dir"], cwd=cwd)
    if rc == 124:
        return f"ERROR: {err[:200]}"
    if rc == 127:
        return f"ERROR: {err[:200]}"
    if rc != 0:
        return ("ERROR: not a git repository (git rev-parse failed). "
                f"{err[:200] if err else 'Run this from inside a git repo.'}")
    return None


def _cwd_or(cwd: Any) -> str | None:
    cwd = (cwd or "").strip() if isinstance(cwd, str) else ""
    return cwd or None


# ---------------------------------------------------------------------------
# GitStatus
# ---------------------------------------------------------------------------

_GIT_STATUS_DESC = (
    "Show git status in the current working directory: current branch "
    "plus porcelain-format changed files. Fails with a clear error when "
    "not inside a git repository."
)
_GIT_STATUS_PARAMS = {
    "type": "object",
    "properties": {},
}


def git_status(cwd: str = "") -> str:
    cwdd = _cwd_or(cwd)
    err = _check_repo(cwdd)
    if err:
        return err
    rc, branch, _ = _run_git(["branch", "--show-current"], cwd=cwdd)
    branch = branch if rc == 0 and branch else "(detached HEAD)"
    rc, porcelain, _ = _run_git(["status", "--porcelain"], cwd=cwdd)
    if rc != 0:
        return "ERROR: git status failed."
    if not porcelain:
        return f"branch: {branch}\n(clean working tree)"
    return f"branch: {branch}\n{porcelain}"


# ---------------------------------------------------------------------------
# GitCommit
# ---------------------------------------------------------------------------

_GIT_COMMIT_DESC = (
    "Stage everything (git add -A) and commit with the given message. "
    "Returns the real commit hash. Refuses if there is nothing to commit."
)
_GIT_COMMIT_PARAMS = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
            "description": "Commit message (required, non-empty).",
        },
        "cwd": {
            "type": "string",
            "description": "Directory to run in (default: current working directory).",
        },
    },
    "required": ["message"],
}


def git_commit(message: str, cwd: str = "") -> str:
    cwdd = _cwd_or(cwd)
    err = _check_repo(cwdd)
    if err:
        return err
    message = (message or "").strip()
    if not message:
        return "ERROR: commit message is required and cannot be empty."
    # Refuse if nothing to commit — check porcelain before staging.
    rc, porcelain, _ = _run_git(["status", "--porcelain"], cwdd)
    if rc != 0:
        return "ERROR: git status failed; cannot determine changes."
    if not porcelain:
        return ("ERROR: nothing to commit — working tree is clean. "
                "Make changes first.")
    rc, _, err_out = _run_git(["add", "-A"], cwdd)
    if rc != 0:
        return f"ERROR: git add -A failed: {err_out[:300]}"
    rc, _, err_out = _run_git(["commit", "-m", message], cwdd)
    if rc != 0:
        return f"ERROR: git commit failed: {err_out[:500]}"
    rc, commit_hash, _ = _run_git(["rev-parse", "HEAD"], cwdd)
    commit_hash = commit_hash if rc == 0 else "(hash unavailable)"
    rc2, oneline, _ = _run_git(["log", "--oneline", "-1"], cwdd)
    return f"committed: {commit_hash}\n{oneline if rc2 == 0 else ''}".rstrip()


# ---------------------------------------------------------------------------
# GitBranch
# ---------------------------------------------------------------------------

_GIT_BRANCH_DESC = (
    "With a 'name' argument: create and switch to a new branch "
    "(git checkout -b <name>). Without an argument: list all branches, "
    "marking the current one with '*'."
)
_GIT_BRANCH_PARAMS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "New branch name. Omit to list branches instead.",
        },
        "cwd": {
            "type": "string",
            "description": "Directory to run in (default: current working directory).",
        },
    },
}


def git_branch(name: str = "", cwd: str = "") -> str:
    cwdd = _cwd_or(cwd)
    err = _check_repo(cwdd)
    if err:
        return err
    name = (name or "").strip() if isinstance(name, str) else ""
    if not name:
        rc, out, err_out = _run_git(["branch"], cwdd)
        if rc != 0:
            return f"ERROR: git branch failed: {err_out[:300]}"
        return out or "(no branches)"
    rc, _, err_out = _run_git(["checkout", "-b", name], cwdd)
    if rc != 0:
        return f"ERROR: could not create branch '{name}': {err_out[:300]}"
    rc, cur, _ = _run_git(["branch", "--show-current"], cwdd)
    return f"switched to new branch '{cur if rc == 0 else name}'"


# ---------------------------------------------------------------------------
# GitPush
# ---------------------------------------------------------------------------

_GIT_PUSH_DESC = (
    "Run 'git push' in the current working directory. Captures the real "
    "output and errors (no upstream, auth failures, etc.) — this tool "
    "never pretends a push succeeded."
)
_GIT_PUSH_PARAMS = {
    "type": "object",
    "properties": {
        "cwd": {
            "type": "string",
            "description": "Directory to run in (default: current working directory).",
        },
    },
}


def git_push(cwd: str = "") -> str:
    cwdd = _cwd_or(cwd)
    err = _check_repo(cwdd)
    if err:
        return err
    rc, out, err_out = _run_git(["push"], cwdd)
    combined = "\n".join(p for p in (out, err_out) if p).strip()
    if rc != 0:
        return f"ERROR: git push failed (exit {rc}):\n{combined[:800]}"
    return combined or "pushed (no output)"


# ---------------------------------------------------------------------------
# GitDiff
# ---------------------------------------------------------------------------

_GIT_DIFF_DESC = (
    "Show 'git diff --stat' (files changed, insertions, deletions). "
    "With a 'file' argument: show the full unified diff for that file only."
)
_GIT_DIFF_PARAMS = {
    "type": "object",
    "properties": {
        "file": {
            "type": "string",
            "description": "Optional single file path to diff fully.",
        },
        "cwd": {
            "type": "string",
            "description": "Directory to run in (default: current working directory).",
        },
    },
}


def git_diff(file: str = "", cwd: str = "") -> str:
    cwdd = _cwd_or(cwd)
    err = _check_repo(cwdd)
    if err:
        return err
    f = (file or "").strip() if isinstance(file, str) else ""
    if f:
        rc, out, err_out = _run_git(["diff", "--", f], cwdd)
    else:
        rc, out, err_out = _run_git(["diff", "--stat"], cwdd)
    if rc != 0:
        return f"ERROR: git diff failed: {err_out[:300]}"
    return out or "(no changes)"


# ---------------------------------------------------------------------------
# GitLog
# ---------------------------------------------------------------------------

_GIT_LOG_DESC = (
    "Show the last n commits (one line each: hash + subject). "
    "Default n = 10."
)
_GIT_LOG_PARAMS = {
    "type": "object",
    "properties": {
        "n": {
            "type": "integer",
            "description": "Number of commits to show (1-50, default 10).",
        },
        "cwd": {
            "type": "string",
            "description": "Directory to run in (default: current working directory).",
        },
    },
}


def git_log(n: int = 10, cwd: str = "") -> str:
    cwdd = _cwd_or(cwd)
    err = _check_repo(cwdd)
    if err:
        return err
    n, coer = _coerce_int(n, "n", 10, 1, 50)
    if coer:
        return coer
    rc, out, err_out = _run_git(["log", "--oneline", f"-{n}"], cwdd)
    if rc != 0:
        return f"ERROR: git log failed: {err_out[:300]}"
    return out or "(no commits yet)"


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------

def register(agent) -> None:
    """Register the GitStatus/GitCommit/GitBranch/GitPush/GitDiff/GitLog
    tools on an agent."""
    agent.tools["GitStatus"] = Tool("GitStatus", _GIT_STATUS_DESC,
                                    _GIT_STATUS_PARAMS, git_status,
                                    risk=RISK_SAFE)
    agent.tools["GitCommit"] = Tool("GitCommit", _GIT_COMMIT_DESC,
                                    _GIT_COMMIT_PARAMS, git_commit,
                                    risk=RISK_CONFIRM)
    agent.tools["GitBranch"] = Tool("GitBranch", _GIT_BRANCH_DESC,
                                    _GIT_BRANCH_PARAMS, git_branch,
                                    risk=RISK_CONFIRM)
    agent.tools["GitPush"] = Tool("GitPush", _GIT_PUSH_DESC,
                                  _GIT_PUSH_PARAMS, git_push,
                                  risk=RISK_CONFIRM)
    agent.tools["GitDiff"] = Tool("GitDiff", _GIT_DIFF_DESC,
                                  _GIT_DIFF_PARAMS, git_diff,
                                  risk=RISK_SAFE)
    agent.tools["GitLog"] = Tool("GitLog", _GIT_LOG_DESC,
                                 _GIT_LOG_PARAMS, git_log,
                                 risk=RISK_SAFE)


if __name__ == "__main__":
    import os
    import tempfile
    from pathlib import Path

    def check(name: str, cond: bool) -> None:
        print(("PASS" if cond else "FAIL"), "-", name)
        if not cond:
            raise SystemExit(f"self-test failed: {name}")

    tmpdir = tempfile.mkdtemp(prefix="gitops-test-")
    os.chdir(tmpdir)

    # Not-a-repo guard: all tools must refuse cleanly before git init.
    check("not-a-repo GitStatus refuses",
          git_status().startswith("ERROR: not a git repository"))
    check("not-a-repo GitCommit refuses",
          git_commit("x").startswith("ERROR: not a git repository"))
    check("not-a-repo GitPush refuses",
          git_push().startswith("ERROR: not a git repository"))
    check("not-a-repo GitLog refuses",
          git_log(3).startswith("ERROR: not a git repository"))

    # git init a REAL repo.
    rc, out, err = _run_git(["init"])
    check("git init works", rc == 0)
    _run_git(["config", "user.email", "test@example.com"])
    _run_git(["config", "user.name", "Test"])

    # GitStatus: see untracked file.
    Path("hello.txt").write_text("hello\n")
    st = git_status(cwd=tmpdir)
    check("GitStatus shows untracked file", "?? hello.txt" in st)

    # GitDiff on an untracked file shows no diff (real git behaviour —
    # untracked files aren't in any diff); verify against a tracked file.
    d = git_diff(cwd=tmpdir)
    check("GitDiff --stat empty for untracked-only change",
          d == "(no changes)")

    # GitCommit: returns a REAL hash, and git log shows it.
    commit_out = git_commit("initial commit", cwd=tmpdir)
    check("GitCommit returns hash",
          commit_out.startswith("committed: ")
          and len(commit_out.split("\n")[0].split(": ")[1]) == 40)
    real_hash = commit_out.split("\n")[0].split(": ")[1]
    rc, log_out, _ = _run_git(["log", "--format=%H", "-1"])
    check("git log shows the real hash", log_out == real_hash)
    gl = git_log(5, cwd=tmpdir)
    check("GitLog shows the commit", "initial commit" in gl)

    # GitCommit refuses when there is nothing to commit.
    check("GitCommit refuses on clean tree",
          git_commit("nothing", cwd=tmpdir).startswith(
              "ERROR: nothing to commit"))

    # GitStatus after commit: clean tree.
    st2 = git_status(cwd=tmpdir)
    check("GitStatus clean after commit", "(clean working tree)" in st2)

    # GitBranch: create and switch, verify branch exists.
    b = git_branch("feature/x", cwd=tmpdir)
    check("GitBranch creates branch", "feature/x" in b)
    rc, cur, _ = _run_git(["branch", "--show-current"])
    check("branch really switched", cur == "feature/x")

    # GitBranch without arg: lists branches with '*' on current.
    bl = git_branch(cwd=tmpdir)
    check("GitBranch lists branches", "* feature/x" in bl)

    # GitDiff on a modified file: stat + per-file diff.
    Path("hello.txt").write_text("hello world\n")
    dstat = git_diff(cwd=tmpdir)
    check("GitDiff --stat after edit", "hello.txt" in dstat)
    dfile = git_diff(file="hello.txt", cwd=tmpdir)
    check("GitDiff per-file shows diff",
          "+hello world" in dfile and "-hello" in dfile)

    # GitPush with no remote: real captured error, no fake success.
    p = git_push(cwd=tmpdir)
    check("GitPush captures no-remote error",
          p.startswith("ERROR: git push failed") and "fatal" in p)

    # Second commit via the tool on the new branch, then GitLog(n).
    git_commit("second commit", cwd=tmpdir)
    gl2 = git_log(2, cwd=tmpdir)
    check("GitLog(2) shows two commits",
          "second commit" in gl2 and "initial commit" in gl2
          and len(gl2.splitlines()) == 2)

    print("gitops self-test: ALL PASS")
